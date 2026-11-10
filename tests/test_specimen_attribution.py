"""Gates for bonded-DEM specimen strength tests (module + one engine anchor).

The module under test is ``tensordem.specimen_attribution``; the empirics come
from the real 4x4x3 ``TriaxialRunner`` specimen used by
``docs/ice_bond_specimen_rate_dependence.md`` (mission m6464), whose reported
strength was an unmeasured post-failure peak above a load-path bound that
undercounted the gauge section.

The engine anchor is deliberately cheap: the fastest campaign rate
(``edot = 4 1/s``) needs ~240 steps, and the lattice is built without stepping
for the geometry assertions.
"""
from __future__ import annotations

import math

import pytest

from tensordem.specimen_attribution import (
    BondPathCapacity,
    bond_path_capacity,
    classify_peak,
    inertia_numbers,
    load_path_integrity_ratio,
    particle_mass,
)

from scripts.verify_ice_bond_specimen_attribution import (
    BREAKING_STRAIN,
    BOND_STIFFNESS,
    GROSS_AREA,
    NX,
    NY,
    RADIUS,
    arm_configs,
    capacity,
    crossing_bonds,
    naive_m6464_bound_Pa,
)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------
def test_particle_mass_matches_sphere_formula():
    r, rho = 0.025, 917.0
    expected = rho * 4.0 / 3.0 * math.pi * r ** 3
    assert particle_mass(r, rho) == pytest.approx(expected, rel=1e-15)
    assert particle_mass(r, rho) == pytest.approx(0.060, abs=5e-4)
    for bad in ((0.0, 1.0), (-1.0, 1.0), (1.0, 0.0)):
        with pytest.raises(ValueError):
            particle_mass(*bad)


def test_bond_path_capacity_is_the_cleavage_force_times_dz():
    """Axial force a bond carries at its own cleavage strain is k_u eps_b |dz|."""
    k, eps, area = 2000.0, 0.015, 0.04
    cap = bond_path_capacity([0.05, 0.05, -0.05], bond_stiffness_N_m=k,
                             breaking_strain=eps, gross_area_m2=area)
    assert cap.n_bonds == 3
    assert cap.axial_force_per_bond_N == pytest.approx((1.5, 1.5, 1.5))
    assert cap.capacity_force_N == pytest.approx(4.5)
    assert cap.capacity_stress_Pa == pytest.approx(4.5 / area)


def test_bond_path_capacity_takes_absolute_projection_and_validates():
    cap = bond_path_capacity([-0.05], bond_stiffness_N_m=2000.0,
                             breaking_strain=0.015, gross_area_m2=0.04)
    assert cap.capacity_force_N == pytest.approx(1.5)
    with pytest.raises(ValueError):
        bond_path_capacity([], bond_stiffness_N_m=2000.0,
                           breaking_strain=0.015, gross_area_m2=0.04)
    with pytest.raises(ValueError):
        bond_path_capacity([0.05], bond_stiffness_N_m=0.0,
                           breaking_strain=0.015, gross_area_m2=0.04)
    with pytest.raises(ValueError):
        BondPathCapacity(n_bonds=2, axial_force_per_bond_N=(1.0,),
                         bond_stiffness_N_m=1.0, breaking_strain=0.01,
                         gross_area_m2=1.0)


def test_inertia_numbers_reproduce_the_two_conflicting_criteria():
    """The local platen-speed ratio and the specimen-scale transit ratio differ."""
    nums = inertia_numbers(
        strain_rate=4.0, breaking_strain=0.015, specimen_height_m=0.15,
        rest_length_m=0.05, bond_stiffness_N_m=2000.0, radius_m=0.025,
        density=917.0)
    m = particle_mass(0.025, 917.0)
    omega = math.sqrt(2000.0 / m)
    c = 0.05 * omega
    assert nums.bond_wave_speed_m_s == pytest.approx(c)
    assert nums.bond_period_s == pytest.approx(1.0 / omega)
    assert nums.platen_speed_ratio == pytest.approx(0.5 * 4.0 * 0.15 / c)
    assert nums.loading_time_ratio == pytest.approx((0.015 / 4.0) * omega)
    assert nums.transit_ratio == pytest.approx((0.15 / c) / (0.015 / 4.0))
    # 4 1/s is 3 % of the lattice wave speed (the number the m6464 report quoted)
    # yet the specimen needs more than one wave transit to equilibrate.
    assert nums.platen_speed_ratio == pytest.approx(0.033, abs=2e-3)
    assert nums.transit_ratio > 1.0
    assert nums.quasi_static is False
    slow = inertia_numbers(
        strain_rate=0.25, breaking_strain=0.015, specimen_height_m=0.15,
        rest_length_m=0.05, bond_stiffness_N_m=2000.0, radius_m=0.025,
        density=917.0)
    assert slow.quasi_static is True
    assert slow.transit_ratio == pytest.approx(nums.transit_ratio / 16.0)
    with pytest.raises(ValueError):
        inertia_numbers(strain_rate=0.0, breaking_strain=0.015,
                        specimen_height_m=0.15, rest_length_m=0.05,
                        bond_stiffness_N_m=2000.0, radius_m=0.025,
                        density=917.0)


def test_load_path_integrity_ratio_uses_magnitudes():
    assert load_path_integrity_ratio(-121.0, 120.0) == pytest.approx(121.0 / 120.0)
    with pytest.raises(ValueError):
        load_path_integrity_ratio(1.0, 0.0)


def _history(peak_after_break: bool):
    rows = [
        {"axial_engineering_strain": 0.001, "axial_stress_Pa": 100.0, "broken_bonds": 0},
        {"axial_engineering_strain": 0.002, "axial_stress_Pa": 900.0, "broken_bonds": 0},
    ]
    if peak_after_break:
        rows += [
            {"axial_engineering_strain": 0.003,
             "axial_stress_Pa": 950.0, "broken_bonds": 8},
            {"axial_engineering_strain": 0.004,
             "axial_stress_Pa": 4000.0, "broken_bonds": 200},
        ]
    else:
        rows.append({"axial_engineering_strain": 0.003,
                     "axial_stress_Pa": 950.0, "broken_bonds": 0})
    return rows


def test_classify_peak_separates_service_strength_from_jamming_peak():
    post = classify_peak(_history(True), capacity_stress_Pa=1000.0)
    assert post["peak_is_prefailure"] is False
    assert post["broken_bonds_at_peak"] == 200
    assert post["peak_over_capacity"] == pytest.approx(4.0)
    assert post["prefailure_stress_Pa"] == pytest.approx(900.0)
    assert post["prefailure_over_capacity"] == pytest.approx(0.9)
    assert post["valid_strength_measurement"] is True

    pre = classify_peak(_history(False), capacity_stress_Pa=1000.0)
    assert pre["peak_is_prefailure"] is True
    assert pre["prefailure_stress_Pa"] == pytest.approx(950.0)

    over = classify_peak(_history(True), capacity_stress_Pa=500.0)
    assert over["valid_strength_measurement"] is False

    with pytest.raises(ValueError):
        classify_peak([], capacity_stress_Pa=1000.0)
    with pytest.raises(ValueError):
        classify_peak(_history(True), capacity_stress_Pa=0.0)


# ---------------------------------------------------------------------------
# Engine anchor (cheap: 4 1/s needs ~240 steps)
# ---------------------------------------------------------------------------
def _build(rate: float = 4.0):
    from tensordem.dem3d_triaxial import TriaxialRunner
    cfg = arm_configs("linear", rate, dt_hint=3.4e-5)
    return cfg, TriaxialRunner(cfg)


def test_gauge_section_is_carried_by_100_bonds_not_16():
    """The m6464 bound counted nx*ny vertical bonds; the lattice has diagonals."""
    _, runner = _build()
    mask, dz = crossing_bonds(runner.sim)
    assert int(mask.sum()) == 100
    # every crossing bond has the same axial projection L0 = 2R
    assert float(dz[mask].abs().max()) == pytest.approx(2.0 * RADIUS)
    assert float(dz[mask].abs().min()) == pytest.approx(2.0 * RADIUS)
    cap = capacity(runner.sim)
    assert cap.n_bonds == 100
    assert cap.capacity_stress_Pa == pytest.approx(
        BOND_STIFFNESS * BREAKING_STRAIN * 2.0 * RADIUS * 100 / GROSS_AREA)
    assert cap.capacity_stress_Pa == pytest.approx(3750.0)
    assert cap.capacity_stress_Pa / naive_m6464_bound_Pa() == pytest.approx(
        100.0 / (NX * NY))


def test_platen_reaction_is_carried_by_the_bonds_while_intact():
    """Load-path integrity: no second load path, so the ratio is 1.

    Run in the quasi-static window (0.25 1/s, wave transit ratio 0.27): at
    4 1/s the specimen needs 4.4 wave transits, so single-sample ratios swing
    with the inertia and only their average would be meaningful.
    """
    from tensordem.dem3d_triaxial import TriaxialRunner
    cfg = arm_configs("linear", 0.25, dt_hint=3.4e-5)
    cfg.max_axial_steps = 1400          # bond strain reaches ~0.008, quiescent
    runner = TriaxialRunner(cfg)
    sim = runner.sim
    import torch
    i, j = sim.pairs.T
    mask, _ = crossing_bonds(sim)
    ratios = []
    for step in range(cfg.max_axial_steps):
        sim.step()
        if step < 200 or step % 40 or not bool(sim.alive.all()):
            continue
        d = sim.positions[j] - sim.positions[i]
        length = torch.linalg.vector_norm(d, dim=1)
        cos = (d[:, 2] / length).abs()
        axial = float((BOND_STIFFNESS * (length - sim.rest_lengths) * cos)[mask].sum())
        if axial:
            ratios.append(load_path_integrity_ratio(
                float(sim.last_top_platen_reaction_z), axial))
    assert len(ratios) >= 5
    mean = sum(ratios) / len(ratios)
    assert mean == pytest.approx(1.0, abs=0.08)
    assert min(ratios) > 0.85
    assert max(ratios) < 1.15


def test_specimen_peak_is_postfailure_and_prefailure_stress_respects_capacity():
    """Regression: the m6464 dependent variable is a debris-compaction peak."""
    from tensordem.specimen_attribution import classify_peak as cp
    _, runner = _build()
    record = runner.run()
    rows = [r for r in record["rows"] if r["phase"] == "axial"] or record["rows"]
    history = [{"axial_engineering_strain": float(r["axial_engineering_strain"]),
                "axial_stress_Pa": float(r["axial_stress_Pa"]),
                "broken_bonds": int(r["broken_bonds"])} for r in rows]
    cap = capacity(runner.sim)
    verdict = cp(history, cap.capacity_stress_Pa)
    assert verdict["peak_is_prefailure"] is False
    assert verdict["broken_bonds_at_peak"] > 0
    assert verdict["prefailure_over_capacity"] is not None
    assert verdict["prefailure_over_capacity"] <= 1.5