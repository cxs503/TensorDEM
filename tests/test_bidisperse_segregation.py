"""Regression tests for the bidisperse segregation verification face (P70-N03).

Fast (≈30 s) unit tests on a 700-particle bed: they pin the kernel bookkeeping
(mass ledger, per-step interpenetration, no non-finite state), the equal-size
label control, the segregation-index definitions and the closed-form stability
mapping audit.  The physics campaign itself lives in
``runs/p70n03/`` + ``artifacts/p70n03/`` (see docs/bidisperse_segregation.md).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.bidisperse_segregation import (
    BedConfig, BidisperseBed, build_packing, coefficient_of_variation,
    damping_ratio, fit_size_exponent, sphere_volume,
)
from tensordem.segregation_stability import (
    LAMBDA_GRAIN, free_surface_inertia, partition_table,
    transmission_from_partitions,
)

QUICK = dict(n_particles=700, d_small=0.01, width=0.09, density=2500.0)


@pytest.fixture(scope="module")
def bed():
    cfg = BedConfig(size_ratio=2.0, large_volume_fraction=0.2, seed=3, **QUICK)
    b = BidisperseBed(cfg)
    b.settle(0.01)
    return b


def test_counts_match_volume_fraction():
    cfg = BedConfig(size_ratio=2.0, large_volume_fraction=0.2, n_particles=600)
    n_l, n_s = cfg.counts
    assert n_l + n_s == 600
    vol_l = n_l * float(sphere_volume(cfg.radius_large))
    vol_s = n_s * float(sphere_volume(cfg.radius_small))
    assert vol_l / (vol_l + vol_s) == pytest.approx(0.2, abs=2e-3)


def test_packing_is_overlap_free_and_mixed():
    cfg = BedConfig(size_ratio=3.0, large_volume_fraction=0.4, seed=5, **QUICK).resolved()
    pos, rad, coarse, Lx, Ly = build_packing(cfg)
    assert len(pos) == cfg.n_particles
    assert int(coarse.sum()) == cfg.counts[0]
    # no initial interpenetration beyond 2 % of the small radius
    from scipy.spatial import cKDTree
    pairs = cKDTree(pos).query_pairs(2.0 * rad.max() * 1.02, output_type="ndarray")
    d = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
    overlap = rad[pairs[:, 0]] + rad[pairs[:, 1]] - d
    assert overlap.max() < 0.1 * rad.min()
    # coarse particles must not be stratified vertically at t=0
    zc = pos[coarse, 2].mean()
    assert abs(zc - pos[:, 2].mean()) < 0.25 * (pos[:, 2].max() - pos[:, 2].min())


def test_kernel_mass_ledger_and_finite_state(bed):
    m0 = bed.total_mass
    bed.floor_offset(0.0)
    z0 = float(np.sum(bed.mass * bed.pos[:, 2]) / m0)
    for _ in range(60):
        bed.step()
    assert np.isfinite(bed.pos).all()
    assert np.isfinite(bed.vel).all()
    assert bed.last_max_overlap < 0.2          # interpenetration / r_small
    assert bed.total_mass == pytest.approx(m0, rel=1e-15)
    z1 = float(np.sum(bed.mass * bed.pos[:, 2]) / m0)
    assert np.isfinite(z1) and abs(z1 - z0) < 0.05     # no COM explosion


def test_shaking_amplitude_defines_gamma(bed):
    amp, freq = 1.5e-3, 25.0
    res = bed.run_shaking(0.004, amp, freq, sample_dt=1e-3)
    assert res["gamma"] == pytest.approx(amp * (2 * math.pi * freq) ** 2 / 9.81, rel=1e-12)


def test_equal_size_control_labels_have_equal_radii():
    cfg = BedConfig(size_ratio=1.0, large_volume_fraction=0.2, seed=7, **QUICK)
    b = BidisperseBed(cfg)
    coarse = b.cls == 1
    assert coarse.sum() == cfg.counts[0]
    assert b.r[coarse].max() == pytest.approx(b.r[~coarse].max(), rel=1e-12)
    b.settle(0.006)
    m = b.measure()
    # equal-size labels are physically identical: the index must stay a
    # statistical fluctuation, not a systematic offset
    assert abs(m["S"]) < 0.35
    assert m["S_fine"] == pytest.approx(-0.2 / 0.8 * m["S"], rel=1e-9)


def test_partition_audit_closed_form_vs_direct(bed):
    p = partition_table(bed, 6, 3)
    assert p["kg_closed_form"] == pytest.approx(p["kg_direct"], rel=1e-9)
    assert p["lcg_closed_form"] == pytest.approx(p["lcg_direct"], rel=1e-9)
    assert p["mass_bin_sum"] == pytest.approx(p["mass_direct"], rel=1e-12)
    # bin masses add up to the particle total
    assert float(np.sum(np.asarray(p["mass"]))) == pytest.approx(p["mass_direct"], rel=1e-12)


def test_free_surface_inertia_level_surface():
    widths = [0.5] * 6 + [0.5] * 6
    fs = free_surface_inertia(widths, x_width=1.0, breadth=4.0)
    # level surface: area = 12 strips x 1 m x 4 m, I = A B^2 / 12
    assert fs["free_surface_area"] == pytest.approx(12.0 * 1.0 * 4.0, rel=1e-12)
    assert fs["free_surface_inertia"] == pytest.approx(48.0 * 16.0 / 12.0, rel=1e-12)
    assert fs["max_slope"] == pytest.approx(0.0, abs=1e-12)


def test_transmission_channels_add_up(bed):
    p0 = partition_table(bed, 6, 3)
    for _ in range(30):
        bed.step()
    p1 = partition_table(bed, 6, 3)
    out = transmission_from_partitions(p0, p1, hold_length=10.0, hold_breadth=5.0,
                                       displacement=5000.0)
    assert out["dGM_total"] == pytest.approx(
        out["dGM_kg_channel"] + out["dGM_freesurface_channel"], rel=1e-12)
    assert out["audit"]["kg_rel_err"] <= 1e-6
    assert out["audit"]["lcg_rel_err"] <= 1e-6
    assert out["lambda_grain"] == LAMBDA_GRAIN
    # symmetric partition => no longitudinal shift
    assert abs(out["dLCG_cargo"]) < 1e-12


def test_damping_ratio_and_fit_helpers():
    assert damping_ratio(1.0) < 1e-6
    assert 0.2 < damping_ratio(0.3) < 0.5
    rows = [{"size_ratio": s, "v_up": 2.0e-3 * (s - 1) ** 1.1} for s in (1.5, 2.0, 3.0, 4.0)]
    fit = fit_size_exponent(rows)
    assert fit["p"] == pytest.approx(1.1, abs=0.02)
    assert coefficient_of_variation([1.0, 1.0, 1.0]) < 1e-12
    assert math.isnan(coefficient_of_variation([0.0]))