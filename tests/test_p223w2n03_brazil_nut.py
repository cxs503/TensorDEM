"""Verification tests for node p223w2-n03, face B: Brazil-nut / reverse Brazil-nut.

Cheap but real: the bed packing is checked for non-overlap and composition, the
timescale estimator is checked against a synthetic exponential with a known
decay time, the segregation profile against a hand-built segregated bed, and one
short shaken run is checked for determinism and for the absence of segregation
without shaking (the negative control).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from tensordem.granular_convection import (
    BNE_ANCHORS,
    RBNE_BOUNDARY_ANCHOR,
    BedConfig,
    BedSystem,
    ShakeCase,
    anchor_errors,
    interpolated_boundary,
    run_shake,
    segregate_direction,
    segregation_timescale,
    timescale_from_history,
)
from tensordem.granular_convection.bed import build_bed, packing_lattice
from tensordem.granular_convection.brazil_nut import segregation_profile


# ---------------------------------------------------------------------------
# Bed construction
# ---------------------------------------------------------------------------

def test_packing_is_non_overlapping_and_inside_the_cell():
    cfg = BedConfig()
    pos, rad = packing_lattice(cfg)
    assert len(pos) > 60
    tree = cKDTree(pos)
    pairs = tree.query_pairs(float(2 * rad.max()) + 1e-9, output_type="ndarray")
    for i, j in pairs:
        assert math.dist(pos[i], pos[j]) >= rad[i] + rad[j] - 1e-12
    assert (pos[:, 0] - rad >= -1e-12).all()
    assert (pos[:, 0] + rad <= cfg.width + 1e-12).all()
    assert (pos[:, 1] - rad >= -1e-12).all()


def test_packing_carries_both_species_near_the_requested_area_fraction():
    cfg = BedConfig()
    pos, rad = packing_lattice(cfg)
    thr = 0.25 * (cfg.d_small + cfg.d_large)
    is_large = rad > thr
    assert is_large.sum() >= 5
    assert (~is_large).sum() >= 20
    frac = (np.pi * rad[is_large] ** 2).sum() / (np.pi * rad**2).sum()
    assert 0.3 < frac < 0.8          # target 0.4, blocked neighbours widen it


def test_config_validation_rejects_impossible_parameters():
    with pytest.raises(ValueError):
        BedConfig(d_large=0.004, d_small=0.006)
    with pytest.raises(ValueError):
        BedConfig(mu=-0.1)
    with pytest.raises(ValueError):
        BedConfig(solid_fraction=1.5)


def test_shake_case_amplitude_follows_gamma():
    case = ShakeCase(gamma=3.0, frequency=25.0)
    amp = case.amplitude(9.81)
    assert amp * case.omega**2 / 9.81 == pytest.approx(3.0, rel=1e-12)


def test_settled_bed_comes_to_rest():
    cfg = BedConfig()
    system, _ = build_bed(cfg, settle_steps=1500)
    assert system.kinetic_energy() == pytest.approx(0.0, abs=1e-12)
    # and it is a real bed: the top of the bed is several particle diameters deep
    depth = float(np.max(system.pos[:, 1] + system.radii))
    assert depth > 5.0 * cfg.d_small


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

def test_segregate_direction_signs():
    assert segregate_direction(0.02, 0.01) == 1
    assert segregate_direction(0.01, 0.02) == -1
    assert segregate_direction(0.01, 0.01) == 0


def test_profile_detects_a_hand_built_segregated_bed():
    """Large disks on top: the free-surface enrichment S must be positive."""
    cfg = BedConfig(d_large=0.012, d_small=0.006, width=0.08, height=0.06,
                    bed_height=0.05, max_particles=60)
    n = 40
    rs, rl = 0.003, 0.006
    pos = np.zeros((n, 2))
    rad = np.zeros(n)
    # bottom half: small disks; top half: large disks
    for k in range(n):
        if k < n // 2:
            pos[k] = (rs + (k % 5) * 2.2 * rs, rs + (k // 5) * 2.2 * rs)
            rad[k] = rs
        else:
            j = k - n // 2
            pos[k] = (rl + (j % 4) * 2.2 * rl, 0.03 + rl + (j // 4) * 2.2 * rl)
            rad[k] = rl
    system = BedSystem(pos, rad, np.full(n, 2500.0), cfg)
    prof = segregation_profile(system, rad > 0.25 * (cfg.d_small + cfg.d_large), 6)
    assert prof["S"] > 0.2
    assert prof["phi_0"] == pytest.approx((np.pi * rad**2)[rad > 0.0045].sum()
                                          / (np.pi * rad**2).sum())


def test_profile_of_a_well_mixed_bed_is_near_zero():
    cfg = BedConfig(width=0.08, height=0.06, bed_height=0.05, max_particles=80)
    pos, rad = packing_lattice(cfg)
    system = BedSystem(pos, rad, np.where(rad > 0.25 * (cfg.d_small + cfg.d_large),
                                          2500.0, 2500.0), cfg)
    prof = segregation_profile(system, rad > 0.25 * (cfg.d_small + cfg.d_large), 8)
    assert abs(prof["S"]) < 0.6       # small bed: the noise floor is large


# ---------------------------------------------------------------------------
# Timescale estimator against a synthetic exponential
# ---------------------------------------------------------------------------

def test_timescale_recovers_a_known_exponential():
    tau_true = 12.0
    cycles = np.linspace(0.0, 60.0, 241)
    dz = 0.010 * (1.0 - np.exp(-cycles / tau_true))
    fit = timescale_from_history(cycles, dz)
    assert fit["crossed"] is True
    # The estimator averages the asymptote over the last quarter of the record,
    # which is biased low for a still-rising signal: the measured bias is -2.3 %
    # for a pure exponential.  5 % is the documented accuracy of the estimator.
    assert fit["tau_cycles"] == pytest.approx(tau_true, rel=0.05)
    assert fit["dz_inf"] == pytest.approx(0.010, rel=0.02)


def test_timescale_scales_with_the_decay_time():
    cycles = np.linspace(0.0, 120.0, 481)
    slow = timescale_from_history(cycles, 0.01 * (1.0 - np.exp(-cycles / 40.0)))
    fast = timescale_from_history(cycles, 0.01 * (1.0 - np.exp(-cycles / 10.0)))
    assert slow["tau_cycles"] > fast["tau_cycles"]


def test_timescale_is_infinite_when_the_signal_never_moves():
    cycles = np.linspace(0.0, 40.0, 50)
    fit = timescale_from_history(cycles, np.zeros_like(cycles))
    assert fit["crossed"] is False
    assert math.isinf(fit["tau_cycles"])


def test_timescale_rejects_short_records():
    with pytest.raises(ValueError):
        timescale_from_history(np.array([0.0, 1.0]), np.array([0.0, 1.0]))


def test_timescale_is_not_quantised_by_the_sampling_interval():
    """The 63 % crossing is interpolated: a coarse record still lands on tau."""
    cycles = np.linspace(0.0, 80.0, 9)          # only 9 samples
    dz = 0.01 * (1.0 - np.exp(-cycles / 20.0))
    fit = timescale_from_history(cycles, dz)
    assert abs(fit["tau_cycles"] - 20.0) < 0.35 * 20.0


# ---------------------------------------------------------------------------
# Short shaken runs (determinism and the negative control)
# ---------------------------------------------------------------------------

def _short_cfg() -> BedConfig:
    return BedConfig(width=0.08, height=0.10, bed_height=0.05, max_particles=70)


def test_shake_is_deterministic():
    cfg = _short_cfg()
    case = ShakeCase(gamma=3.0, frequency=20.0, n_cycles=3, sample_every=400)
    a = run_shake(case, cfg, settle_steps=600)
    b = run_shake(case, cfg, settle_steps=600)
    assert [h["dz"] for h in a["history"]] == [h["dz"] for h in b["history"]]


def test_no_vibration_means_no_segregation():
    cfg = _short_cfg()
    case = ShakeCase(gamma=1e-9, frequency=20.0, n_cycles=3, sample_every=400,
                     label="control")
    res = run_shake(case, cfg, settle_steps=600)
    # < 2 mm drift on a 50 mm bed, i.e. far below the segregation signal of a
    # shaken run; the *sign* of such a drift is noise and is not asserted.
    assert abs(res["dz_from_initial"]) < 2e-3


def test_segregation_timescale_wrapper_reports_the_gamma():
    cfg = _short_cfg()
    out = segregation_timescale(ShakeCase(gamma=2.0, frequency=20.0, n_cycles=3,
                                          sample_every=400), cfg, settle_steps=600)
    assert out["gamma"] == 2.0
    assert "tau_cycles" in out


# ---------------------------------------------------------------------------
# Anchor bookkeeping
# ---------------------------------------------------------------------------

def test_anchor_errors_require_both_range_and_tolerance():
    good = anchor_errors({"segregation_timescale_cycles": 30.0}, tolerance=0.10)
    assert good["anchors"]["segregation_timescale_cycles"]["pass"] is True
    bad = anchor_errors({"segregation_timescale_cycles": 90.0}, tolerance=0.10)
    assert bad["anchors"]["segregation_timescale_cycles"]["pass"] is False
    assert bad["all_pass"] is False


def test_anchor_errors_rejects_unknown_anchors():
    with pytest.raises(KeyError):
        anchor_errors({"not_an_anchor": 1.0}, tolerance=0.1)


def test_interpolated_boundary_finds_the_crossing():
    pts = [{"gamma": 6.0, "density_ratio": 1.0, "dz": 1.0},
           {"gamma": 6.0, "density_ratio": 2.0, "dz": 0.2},
           {"gamma": 6.0, "density_ratio": 3.0, "dz": -0.6}]
    out = interpolated_boundary(pts)
    assert out[0]["gamma"] == 6.0
    assert out[0]["rho_boundary"] == pytest.approx(2.25, rel=1e-9)


def test_interpolated_boundary_is_none_without_a_crossing():
    pts = [{"gamma": 3.0, "density_ratio": 1.0, "dz": 1.0},
           {"gamma": 3.0, "density_ratio": 2.0, "dz": 0.5}]
    assert interpolated_boundary(pts)[0]["rho_boundary"] is None


def test_phase_anchors_are_ordered():
    lo, hi = RBNE_BOUNDARY_ANCHOR["range"]
    assert lo < RBNE_BOUNDARY_ANCHOR["value"] < hi
    ts = BNE_ANCHORS["segregation_timescale_cycles"]
    assert ts["range"][0] < ts["value"] < ts["range"][1]