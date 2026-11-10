"""Tests of the clear-ice statistics, the contract and the convergence study."""

from __future__ import annotations

import numpy as np
import pytest

from tensordem.clear_ice.contract import (
    CONTRACT_VERSION,
    ClearIceResult,
    FloePopulation,
    population_from_breakup,
    population_hash,
    validate_population,
)
from tensordem.clear_ice.field import ClearIceField, FieldConfig
from tensordem.clear_ice.statistics import (
    areal_porosity,
    coordination_number,
    force_distribution,
    region_masks,
    voronoi_free_volume,
)


# ---------------------------------------------------------------- statistics
def test_force_distribution_recovers_exponential_sample():
    """A synthetic exponential tail must be recovered with R^2 ~ 1."""
    rng = np.random.default_rng(3)
    f = rng.exponential(scale=1.0, size=200000)      # P(f) = exp(-f)
    stats = force_distribution(f, tail_window=(1.0, 4.0), min_count=100)
    assert stats.r2_exponential > 0.995
    assert stats.f0 == pytest.approx(1.0, rel=0.05)


def test_force_distribution_rejects_degenerate_input():
    with pytest.raises(ValueError):
        force_distribution(np.zeros(5))


def test_coordination_number_counts_both_directions():
    i = np.array([0, 1, 2])
    j = np.array([1, 2, 0])
    fn = np.array([1.0, 1.0, 1.0])
    z = coordination_number(i, j, 3, fn)
    assert z["Z_all"] == pytest.approx(2.0)
    assert z["Z_non_rattler"] == pytest.approx(2.0)
    assert z["rattler_fraction"] == pytest.approx(0.0)
    # a zero-force contact is filtered out by the load threshold
    z2 = coordination_number(np.array([0]), np.array([1]), 3, np.array([0.0]))
    assert z2["Z_all"] == pytest.approx(0.0)


def test_porosity_estimators_on_a_lattice():
    """Hexagonal lattice: areal porosity 1 - phi, Voronoi free volume matches."""
    n = 10
    spacing = 1.0
    pts = []
    for row in range(n):
        for col in range(n):
            pts.append([col * spacing + 0.5 * spacing * (row % 2), row * spacing * np.sqrt(3) / 2])
    pts = np.array(pts)
    r = np.full(len(pts), 0.5 * spacing)
    box = (n * spacing, n * spacing * np.sqrt(3) / 2)
    phi = np.sum(np.pi * r ** 2) / (box[0] * box[1])
    assert areal_porosity(pts, r, box[0] * box[1]) == pytest.approx(1 - phi, rel=1e-12)
    vor = voronoi_free_volume(pts, r, box)
    # the Voronoi free volume of a triangular lattice is exact up to the
    # periodic image correction; 5 % is the discretisation/edge allowance
    assert vor == pytest.approx(1 - phi, rel=0.05, abs=0.02)


def test_region_masks_are_strips_with_exact_areas():
    cfg = FieldConfig()
    x = np.stack([np.linspace(0, cfg.domain_x, 50),
                  np.full(50, 0.5 * cfg.domain_y)], axis=1)
    masks, areas = region_masks(x, cfg)
    for name in ("feed", "jam", "wake"):
        assert name in masks
    # the feed strip must be strictly upstream of the gate
    assert np.all(x[masks["feed"], 0] < cfg.gate_centre[0])
    assert np.all(x[masks["wake"], 0] >= cfg.gate_centre[0])
    assert areas["jam"] > 0.0


# ---------------------------------------------------------------- contract
def test_population_from_breakup_conserves_area():
    areas = np.array([3.0, 1.0, 0.5, 2.0])
    centroids = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    pop = population_from_breakup(areas, centroids, 0.3)
    assert pop.area == pytest.approx(areas.sum(), rel=1e-12)
    assert np.allclose(np.pi * pop.radius ** 2, areas)
    assert pop.parent_area == pytest.approx(areas.sum())
    audit = validate_population(pop, domain=(10.0, 10.0))
    assert audit["pass"]
    assert audit["checks"]["area_ledger"]["relative_error"] < 1e-12


def test_validate_population_detects_area_leak_and_overlap():
    areas = np.array([3.0, 3.0])
    centroids = np.array([[0.0, 0.0], [1.0, 0.0]])       # overlapping
    pop = population_from_breakup(areas, centroids, 0.3)
    audit = validate_population(pop, domain=(10.0, 10.0))
    assert audit["checks"]["non_overlap"]["pass"] is False
    assert audit["pass"] is False

    pop2 = population_from_breakup(areas, np.array([[0.0, 0.0], [5.0, 0.0]]), 0.3,
                                   parent_area=7.0)      # 6 vs 7 -> leak
    audit2 = validate_population(pop2, domain=(10.0, 10.0))
    assert audit2["checks"]["area_ledger"]["pass"] is False


def test_population_hash_is_deterministic_and_roundtrips():
    pop = population_from_breakup([1.0, 1.0], [[0.0, 0.0], [3.0, 0.0]], 0.3)
    assert pop.hash == population_hash(pop.x, pop.radius, pop.thickness)
    again = FloePopulation.from_dict(pop.as_dict())
    assert again.hash == pop.hash
    assert again.contract_version == CONTRACT_VERSION
    with pytest.raises(ValueError):
        FloePopulation.from_dict({**pop.as_dict(), "contract_version": "clear-ice-contract/2.0"})


def test_result_payload_requires_the_five_blocks():
    blocks = dict(floe_kinematics={}, force_chain_stats={}, porosity={},
                  convergence_report={})
    payload = ClearIceResult(**blocks).as_dict()
    for key in ("floe_kinematics", "force_chain_stats", "porosity",
                "convergence_report", "uncertainty_quantification", "provenance"):
        assert key in payload
    with pytest.raises(ValueError):
        ClearIceResult.validate_payload({"floe_kinematics": {}})


# ---------------------------------------------------------------- field / C1
def test_population_handover_reproduces_the_field():
    """The contract payload alone must rebuild the same initial state."""
    cfg = FieldConfig(n_floes=24, duration=0.3, warmup=0.15, sample_every=0.05)
    from tensordem.clear_ice.study import build_population
    pop = build_population(cfg)
    f1 = ClearIceField(cfg, population=(pop.x, pop.radius, pop.thickness))
    f2 = ClearIceField(cfg, population=(pop.x, pop.radius, pop.thickness))
    assert np.allclose(f1.x, f2.x)
    assert np.allclose(f1.mass, f2.mass)
    # no floe is placed inside the hull face
    assert np.all(f1.x[:, 0] <= cfg.gate_centre[0])


def test_trajectory_convergence_order_at_least_one():
    """C1: halving the step halves the error (order >= 1)."""
    from tensordem.clear_ice.study import trajectory_convergence
    res = trajectory_convergence(dt_levels=(2e-3, 1e-3, 5e-4), duration=0.3)
    assert res["observed_order_fit"] >= 1.0
    assert res["pass"] is True
    assert res["l2_error_m"][0] > res["l2_error_m"][-1]


def test_kinematics_oscillation_index_flags_data_quality():
    """The resolution metric separates a smooth record from a one-step zigzag."""
    from tensordem.clear_ice.kinematics import oscillation_index
    n_samples, n_floes = 200, 5
    t = np.arange(n_samples)
    smooth = np.zeros((n_samples, n_floes, 2))
    smooth[:, :, 0] = (0.8 + 0.05 * np.sin(2 * np.pi * t / n_samples))[:, None]
    q_smooth, hf_smooth = oscillation_index(smooth, 1e-3)
    zigzag = smooth.copy()
    zigzag[:, :, 0] += 0.2 * (-1.0) ** t[:, None]
    q_zig, hf_zig = oscillation_index(zigzag, 1e-3)
    assert q_zig > 10 * q_smooth
    assert hf_zig > hf_smooth
    assert q_smooth < 0.05