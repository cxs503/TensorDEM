"""Tests for the IN383-DEM01 scale-ceiling module (TensorDEM).

Everything here is deliberately small: the tests verify *correctness* of the
contact search, of the contact law and of the scaling estimators, not the
campaign timings (those live in ``scripts/scale_ceiling_dem.py``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordem.scale_ceiling import (
    FloeBed,
    FloeBedConfig,
    amdahl_serial_fraction,
    brute_force_pairs,
    cell_list_pairs,
    fit_loglog,
    memory_linearity,
    strong_scaling,
    weak_scaling,
)
from tensordem.scale_ceiling.controls import run_negative_controls
from tensordem.scale_ceiling.observables import (
    crush_resistance,
    settle,
    slab_config,
    slump_repose_angle,
)


def _pair_set(i, j):
    return set(zip(np.asarray(i).tolist(), np.asarray(j).tolist()))


# ------------------------------------------------------------------ search
def test_cell_list_matches_brute_force_exactly():
    rng = np.random.default_rng(0)
    n = 240
    lengths = np.array([8.0, 8.0, 8.0])
    pos = rng.random((n, 3)) * lengths
    radii = np.full(n, 0.5)
    i, j, _ = cell_list_pairs(pos, lengths, radii, skin=0.2, periodic=(True, True, True))
    bi, bj = brute_force_pairs(pos, lengths, radii, skin=0.2, periodic=(True, True, True))
    assert _pair_set(i, j) == _pair_set(bi, bj)


def test_cell_list_pairs_are_unique_and_ordered():
    rng = np.random.default_rng(1)
    n = 200
    lengths = np.array([6.0, 6.0, 6.0])
    pos = rng.random((n, 3)) * lengths
    radii = np.full(n, 0.5)
    i, j, stats = cell_list_pairs(pos, lengths, radii, skin=0.2, periodic=(False, False, False))
    assert np.all(i < j)
    keys = i.astype(np.int64) * n + j.astype(np.int64)
    assert np.unique(keys).size == keys.size
    assert stats["n_cells"] >= 1


def test_truncated_cell_list_misses_pairs():
    """A too small search radius must under-count: the detector must notice."""
    rng = np.random.default_rng(2)
    n = 200
    lengths = np.array([6.0, 6.0, 6.0])
    pos = rng.random((n, 3)) * lengths
    radii = np.full(n, 0.5)
    good_i, good_j, _ = cell_list_pairs(pos, lengths, radii, skin=0.2, periodic=(True,) * 3)
    bad_i, bad_j, _ = cell_list_pairs(pos, lengths, radii * 0.5, skin=0.0, periodic=(True,) * 3)
    truth = _pair_set(good_i, good_j)
    broken = _pair_set(bad_i, bad_j)
    assert len(broken - truth) == 0          # never invents pairs ...
    assert len(truth - broken) > 0           # ... but does miss contacts


def test_pair_count_is_linear_in_n():
    rng = np.random.default_rng(3)
    counts = {}
    for n in (500, 2000):
        lengths = np.array([(n / 0.5) ** (1 / 3)] * 3) * 1.0
        pos = rng.random((n, 3)) * lengths
        radii = np.full(n, 0.5)
        i, _, _ = cell_list_pairs(pos, lengths, radii, skin=0.2, periodic=(True,) * 3)
        counts[n] = i.size
    ratio = (counts[2000] / 2000.0) / (counts[500] / 500.0)
    assert 0.85 < ratio < 1.15


# ------------------------------------------------------------------ kernel
def test_config_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        FloeBedConfig(n=0)
    with pytest.raises(ValueError):
        FloeBedConfig(n=10, restitution=1.5)
    with pytest.raises(ValueError):
        FloeBedConfig(n=10, skin=-0.1)


def test_initial_configuration_carries_no_spurious_energy():
    """The deposition lattice may only touch: no measurable elastic energy."""
    bed = FloeBed(FloeBedConfig(n=120, seed=1))
    assert bed.kinetic_energy() == 0.0
    assert bed.dt > 0.0
    weight_scale = float(bed.mass.sum()) * bed.config.gravity * float(bed.box[2])
    assert bed.energy_ledger()["contact"] < 1e-3 * weight_scale
    assert np.allclose(bed.box, np.array([bed.config.n_x, bed.config.n_y, bed.config.n_z])
                       * bed.config.deposition_spacing)


def test_hertz_normal_force_matches_analytic_law():
    """Static overlap of one pair must give (4/3) E* sqrt(R*) delta^(3/2)."""
    bed = FloeBed(FloeBedConfig(n=2, layers_y=2, height_layers=1, floor=False,
                                left_wall=False, right_wall=False, gravity=0.0,
                                restitution=1.0, mu_grain=0.0, seed=0))
    delta = 0.01
    bed.pos = torch.tensor([[2.0, 1.0, 1.0], [2.0 + 1.0 - delta, 1.0, 1.0]], dtype=torch.float64)
    bed.vel = torch.zeros((2, 3), dtype=torch.float64)
    bed.build_neighbours(keep_state=False)
    f = bed.forces()
    cfg = bed.config
    r_eff = 0.5 * 0.5
    expected = (4.0 / 3.0) * cfg.effective_modulus * math.sqrt(r_eff) * delta**1.5
    assert f.abs().max().item() == pytest.approx(expected, rel=1e-12)


def test_restitution_of_a_single_collision():
    bed = FloeBed(FloeBedConfig(n=2, layers_y=2, height_layers=1, floor=False,
                                left_wall=False, right_wall=False, gravity=0.0,
                                restitution=0.5, mu_grain=0.0, seed=0))
    bed.pos = torch.tensor([[2.0, 1.0, 1.0], [4.0, 1.0, 1.0]], dtype=torch.float64)
    bed.vel = torch.tensor([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], dtype=torch.float64)
    bed.build_neighbours(keep_state=False)
    e0 = bed.kinetic_energy()
    for _ in range(60):
        bed.step(50)
        if bed.vel[0, 0] < 0:
            break
    e_eff = abs(bed.vel[0, 0].item()) / 2.0
    assert 0.35 < e_eff < 0.75           # discretisation broadens the target 0.5
    assert bed.kinetic_energy() < e0     # a dissipative contact never gains energy


def test_elastic_contact_energy_is_conserved_within_discretisation():
    bed = FloeBed(FloeBedConfig(n=64, layers_y=4, height_layers=4, seed=5, restitution=1.0,
                                mu_grain=0.0, mu_wall=0.0, gravity=0.0, floor=False,
                                left_wall=False, right_wall=False))
    e0 = bed.energy_ledger()["total"]
    bed.step(80)
    rel = abs(bed.energy_ledger()["total"] - e0) / abs(e0)
    assert rel < 2e-2


def test_free_flight_conserves_momentum_and_energy_exactly():
    bed = FloeBed(FloeBedConfig(n=54, gravity=0.0, floor=False, left_wall=False,
                                right_wall=False, restitution=1.0, mu_grain=0.0,
                                mu_wall=0.0, seed=6))
    bed.pos = bed.pos * 4.0
    bed.box = bed.box * 4.0
    bed.build_neighbours(keep_state=False)
    bed.vel = torch.from_numpy(np.random.default_rng(6).normal(0, 0.3, (bed.config.n, 3)))
    p0 = (bed.mass.unsqueeze(1) * bed.vel).sum(dim=0)
    e0 = bed.kinetic_energy()
    bed.step(200)
    p1 = (bed.mass.unsqueeze(1) * bed.vel).sum(dim=0)
    assert torch.allclose(p0, p1, atol=1e-12)
    assert abs(bed.kinetic_energy() - e0) / e0 < 1e-14


def test_frictionless_wall_has_no_tangential_reaction():
    bed = FloeBed(FloeBedConfig(n=120, seed=2, mu_wall=0.0, mu_grain=0.0))
    bed.step(150)
    for _ in range(20):
        bed.step(1)
        assert bed.wall_tangent_force("left") == 0.0
        assert bed.wall_tangent_force("floor") == 0.0


def test_frictional_floor_carries_tangential_load():
    bed = FloeBed(FloeBedConfig(n=120, seed=2, mu_wall=0.6))
    bed.vel = torch.from_numpy(np.full((bed.config.n, 3), 0.05))
    bed.step(40)
    assert bed.wall_tangent_force("floor") > 0.0


def test_settled_bed_reaches_realistic_coordination_and_floor_load():
    bed = FloeBed(FloeBedConfig(n=240, layers_y=6, height_layers=8, seed=3))
    settle(bed, 600)
    assert bed.coordination_number() > 3.0
    weight = float(bed.mass.sum()) * bed.config.gravity
    assert 0.4 * weight < bed.wall_force("floor") < 1.6 * weight


def test_digest_is_reproducible_and_seed_dependent():
    def run(seed):
        bed = FloeBed(FloeBedConfig(n=120, seed=seed))
        bed.step(60)
        return bed.digest()

    assert run(4) == run(4)
    assert run(4) != run(5)


# ------------------------------------------------------------------ scaling
def test_fit_loglog_recovers_linear_and_quadratic_exponents():
    n = np.array([1000.0, 4000.0, 16000.0, 64000.0])
    lin = fit_loglog(n, 3e-6 * n)
    assert lin.slope == pytest.approx(1.0, abs=1e-9)
    assert lin.r2 == pytest.approx(1.0, abs=1e-12)
    quad = fit_loglog(n, 2e-10 * n**2)
    assert quad.slope == pytest.approx(2.0, abs=1e-9)


def test_strong_scaling_and_amdahl_on_ideal_data():
    threads = [8, 16, 32, 64]
    times = [1.0, 0.5, 0.25, 0.125]
    sc = strong_scaling(threads, times, base_threads=8)
    assert sc["efficiency_vs_base"][-1] == pytest.approx(1.0)
    amd = amdahl_serial_fraction(threads, times)
    assert amd["serial_fraction"] == pytest.approx(0.0, abs=1e-9)


def test_amdahl_detects_a_serial_bottleneck():
    threads = [8.0, 16.0, 32.0, 64.0]
    times = [0.25 + 0.75 / p for p in threads]
    amd = amdahl_serial_fraction(threads, times)
    assert amd["serial_fraction"] == pytest.approx(0.25, rel=1e-6)   # T(1) = 1.0
    assert amd["speedup_cap"] == pytest.approx(1.0 / amd["serial_fraction"], rel=1e-6)


def test_weak_scaling_ratio_flags_growth():
    good = weak_scaling([8000, 16000, 32000], [1.0, 1.05, 1.10])
    assert good["max_ratio"] == pytest.approx(1.10)
    assert good["efficiency"] == pytest.approx(1.0 / 1.10)
    bad = weak_scaling([8000, 16000, 32000], [1.0, 1.4, 2.0])
    assert bad["max_ratio"] > 1.15


def test_memory_linearity_fit_accepts_affine_and_rejects_superlinear():
    n = [2000, 8000, 32000, 128000]
    ok = memory_linearity(n, [400.0 + 1e-3 * x for x in n])
    assert ok["max_relative_deviation"] < 1e-9
    assert ok["fit_r2"] == pytest.approx(1.0, abs=1e-12)
    bad = memory_linearity(n, [400.0 + 1e-3 * x + 5e-8 * x**2 for x in n])
    assert bad["max_relative_deviation"] > 0.05


# --------------------------------------------------------------- controls
def test_negative_controls_have_no_false_negatives():
    res = run_negative_controls()
    assert res["false_negatives"] == 0
    assert res["false_positives"] == 0
    names = {c["name"] for c in res["controls"]}
    assert "sidewall_friction_drift" in names
    assert "search_truncation" in names


# ------------------------------------------------------------- observables
def test_crush_protocol_produces_positive_resistance():
    cfg = slab_config(240, n_x=8, height_layers=6, seed=1)
    bed = FloeBed(cfg)
    res = crush_resistance(bed, steps=40, settle_steps=300)
    assert res.indenter_travel > 0.0
    assert res.mean_force > 0.0
    assert res.mean_force_per_width > 0.0
    assert len(res.force_series) == 8


def test_slab_geometry_is_scale_invariant():
    """Growing N must only grow the periodic y-extent, never the local geometry."""
    small = slab_config(1000, n_x=8, height_layers=6, seed=0)
    big = slab_config(100000, n_x=8, height_layers=6, seed=0)
    assert small.n_x == big.n_x
    assert small.n_z == big.n_z
    assert big.layers_y > small.layers_y
    assert small.deposition_spacing == big.deposition_spacing


def test_repose_angle_of_a_slump_is_physical():
    cfg = slab_config(240, n_x=6, height_layers=14, seed=1)
    bed = FloeBed(cfg)
    out = slump_repose_angle(bed, settle_steps=200, collapse_steps=400)
    assert math.isfinite(out["angle_deg"])
    assert 0.0 < out["angle_deg"] < 60.0