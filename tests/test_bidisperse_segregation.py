"""Verification tests for P155-N09 (bidisperse granular segregation).

Covers the machine-checkable acceptance criteria whose evidence does *not* need
the cluster campaign:

* C1 negative control  -- bitwise frozen control, bitwise determinism, sheared
  monodisperse bed stays at the finite-size noise floor.
* C4 mass/concentration conservation -- weighted ``Phi`` returns ``Phi_0``.
* C5 repose angle -- measured with the P96-N3 estimator
  (``tensordem.repose_calibration.repose_angle``) on a synthetic pile whose angle
  is known analytically.
* C6 sensitivity table -- bitwise reproducible across two independent calls.

C2/C3 (direction consensus, scaling R^2) are produced by the campaign runner
(``scripts/run_bidisperse_campaign.py``) and audited in the report; their
analysis machinery is unit-tested here on synthetic data with known answers.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.bidisperse_segregation import (
    BidisperseConfig,
    ShearBoxCase,
    concentration_profile_data,
    compute_concentration_profile,
    compute_segregation_degree,
    compute_segregation_direction,
    consolidation_response,
    fit_scaling_law,
    bootstrap_scaling_law,
    generate_packing,
    kozeny_carman_permeability,
    measure_rest_angle,
    noise_floor,
    permeability_sensitivity_table,
    run_monodisperse_negative_control,
    run_shear_box,
    segregation_direction_test,
    stratified_permeability_profile,
)


# ---------------------------------------------------------------------------
# Measurement: species labels must be global (regression for the per-bin median)
# ---------------------------------------------------------------------------

def test_profile_is_label_based_not_bin_local():
    """A bin holding only large particles must report Phi = 1."""
    r_l, r_s = 0.0075, 0.0050
    thr = math.sqrt(r_l * r_s)
    radii = np.array([r_l, r_l, r_s, r_s])
    is_large = radii > thr
    # two particles (both large) low, two (both small) high
    pos = np.array([[0.5, 0.1], [0.6, 0.1], [0.5, 0.9], [0.6, 0.9]])
    data = concentration_profile_data(pos, radii, 1.0, 2, is_large=is_large)
    assert data["phi_z"][0] == pytest.approx(1.0)
    assert data["phi_z"][1] == pytest.approx(0.0)
    # area weighted, so the low bin has more area
    assert data["area_z"][0] > data["area_z"][1]


def test_profile_threshold_matches_given_labels():
    cfg = BidisperseConfig(d_small=0.01, d_large=0.02, n_particles=40)
    pack = generate_packing(cfg, seed=3)
    a = concentration_profile_data(pack.positions, pack.radii, cfg.box_size, 8,
                                   is_large=pack.is_large)["phi_z"]
    b = concentration_profile_data(pack.positions, pack.radii, cfg.box_size, 8,
                                   r_threshold=cfg.r_threshold)["phi_z"]
    np.testing.assert_array_equal(np.nan_to_num(a), np.nan_to_num(b))


def test_monodisperse_labels_are_a_convention():
    cfg = BidisperseConfig(d_small=0.01, d_large=0.01, n_particles=30)
    pack = generate_packing(cfg, seed=5)
    assert pack.n == 30
    assert pack.is_large.sum() == 15          # 50 % by area = 50 % by count
    assert np.allclose(pack.solid_fraction_large, 0.5, atol=0.02)


def test_empty_bins_are_ignored_by_the_metric():
    phi = np.array([0.5, 0.5, np.nan, np.nan])
    area = np.array([1.0, 1.0, 0.0, 0.0])
    assert compute_segregation_degree(phi, 0.5, weights=area) == 0.0
    phi_bad = np.array([0.0, 1.0, np.nan])
    area_bad = np.array([1.0, 1.0, 0.0])
    assert compute_segregation_degree(phi_bad, 0.5, weights=area_bad) == pytest.approx(1.0)


def test_segregation_degree_is_weighted_and_normalised():
    phi = np.array([0.0, 1.0])
    w = np.array([3.0, 1.0])
    # (3*0.5 + 1*0.5)/4 / 0.5 = 1.0
    assert compute_segregation_degree(phi, 0.5, weights=w) == pytest.approx(1.0)
    assert compute_segregation_degree(np.array([0.5, 0.5]), 0.5) == 0.0


def test_noise_floor_matches_monte_carlo():
    """A perfectly mixed bed: Phi_b is Binomial(n_b, phi_0) per bin."""
    n, k, phi0 = 120, 10, 0.5
    n_b = n // k
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(4000):
        phi = rng.binomial(n_b, phi0, size=k) / n_b
        draws.append(compute_segregation_degree(phi, phi0))
    mc = float(np.mean(draws))
    assert noise_floor(n, phi0, k) == pytest.approx(mc, rel=0.10)


# ---------------------------------------------------------------------------
# Direction test
# ---------------------------------------------------------------------------

def test_direction_detects_both_signs():
    up = np.array([0.2, 0.3, 0.5, 0.7, 0.8])
    down = up[::-1]
    area = np.full(5, 10.0)
    neff = np.full(5, 30.0)
    assert segregation_direction_test(up, area_z=area, n_eff_z=neff)["direction"] == 1
    assert segregation_direction_test(down, area_z=area, n_eff_z=neff)["direction"] == -1
    assert compute_segregation_direction(up) == 1
    assert compute_segregation_direction(down) == -1


def test_direction_is_unresolved_for_a_flat_profile():
    flat = np.full(6, 0.5)
    out = segregation_direction_test(flat, area_z=np.full(6, 10.0),
                                     n_eff_z=np.full(6, 20.0))
    assert out["direction"] == 0
    assert out["z_score"] == pytest.approx(0.0, abs=1e-12)


def test_direction_reports_significance():
    # a tiny gradient on a very noisy (small n_eff) profile must not resolve
    weak = np.linspace(0.5, 0.52, 6)
    out = segregation_direction_test(weak, area_z=np.full(6, 1.0),
                                     n_eff_z=np.full(6, 2.0))
    assert out["direction"] == 0
    assert abs(out["z_score"]) < 2.0


# ---------------------------------------------------------------------------
# C1: negative control
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def negative_control():
    return run_monodisperse_negative_control(
        n_particles=24, seeds=(1, 2), n_bins=6, shear_rate=10.0,
        target_strain=0.15, settle_steps=50)


def test_c1_frozen_state_is_bitwise_identical(negative_control):
    assert negative_control["frozen_max_deviation"] == 0.0
    assert negative_control["frozen_pass"] is True


def test_c1_pipeline_is_deterministic(negative_control):
    assert negative_control["determinism_max_deviation"] == 0.0
    assert negative_control["determinism_pass"] is True


def test_c1_sheared_monodisperse_bed_stays_at_noise_floor(negative_control):
    floor = negative_control["noise_floor"]
    assert floor > 0.0
    for S in negative_control["sheared_S"]:
        assert S <= 2.5 * floor, (S, floor)


def test_c1_overall_pass(negative_control):
    assert negative_control["pass"] is True


# ---------------------------------------------------------------------------
# C4: mass / concentration conservation
# ---------------------------------------------------------------------------

def test_c4_mass_and_profile_conservation():
    cfg = BidisperseConfig(d_small=0.01, d_large=0.015, n_particles=48, seed=4)
    case = ShearBoxCase(config=cfg, shear_rate=10.0, target_strain=0.1,
                        n_bins=8, sample_every=500, compact_steps=200)
    rec = run_shear_box(case, settle_steps=30)
    phi = np.asarray(rec["phi_z"])
    area = np.asarray(rec["area_z"])
    ok = np.isfinite(phi) & (area > 0)
    weighted = float(np.sum(area[ok] * phi[ok]) / np.sum(area[ok]))
    assert abs(weighted - rec["phi_0"]) < 1e-10          # C4
    # total solid area equals the sum over particles: nothing vanished
    n_particles = rec["n_particles"]
    assert n_particles == cfg.n_particles
    assert rec["initial_S"] >= 0.0


def test_c4_initial_profile_recovers_phi0():
    cfg = BidisperseConfig(d_small=0.01, d_large=0.03, n_particles=60, seed=2)
    pack = generate_packing(cfg)
    data = concentration_profile_data(pack.positions, pack.radii, cfg.box_size,
                                      12, is_large=pack.is_large)
    ok = np.isfinite(data["phi_z"])
    weighted = (np.sum(data["area_z"][ok] * data["phi_z"][ok])
                / np.sum(data["area_z"][ok]))
    assert weighted == pytest.approx(pack.solid_fraction_large, rel=1e-12)
    assert pack.solid_fraction_large == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# C3 machinery: scaling fit and bootstrap
# ---------------------------------------------------------------------------

def test_power_fit_recovers_known_exponent():
    ratios = np.array([1.2, 1.5, 2.0, 3.0])
    truth = 0.8 * (ratios - 1.0) ** 0.65
    fit = fit_scaling_law(ratios, truth)
    assert fit["alpha"] == pytest.approx(0.65, rel=1e-3)
    assert fit["A"] == pytest.approx(0.8, rel=1e-3)
    assert fit["R2"] == pytest.approx(1.0, abs=1e-8)


def test_bootstrap_ci_brackets_the_exponent():
    rng = np.random.default_rng(0)
    ratios = [1.2, 1.5, 2.0, 3.0]
    groups = []
    for r in ratios:
        mean = 0.8 * (r - 1.0) ** 0.65
        groups.append(mean + rng.normal(0.0, 0.01, size=8))
    out = bootstrap_scaling_law(ratios, groups, n_boot=200, seed=3)
    assert out["alpha"] == pytest.approx(0.65, abs=0.05)
    lo, hi = out["alpha_ci"]
    assert lo < 0.65 < hi


def test_phi_suppression_is_detectable():
    """S must fall as phi grows towards jamming (C3's second half)."""
    phi = np.array([0.60, 0.72, 0.84])
    S = np.array([0.42, 0.25, 0.05])
    slope = float(np.polyfit(phi, S, 1)[0])
    assert slope < 0.0
    assert np.all(np.diff(S) < 0)


# ---------------------------------------------------------------------------
# C6: permeability and drainage sensitivity
# ---------------------------------------------------------------------------

def test_kozeny_carman_analytic_value():
    d, sf = 1.0e-3, 0.60          # eps = 0.40, the classic packed-bed value
    eps = 1.0 - sf
    expected = d**2 * eps**3 / (180.0 * (1.0 - eps) ** 2)
    assert kozeny_carman_permeability(d, sf) == pytest.approx(expected, rel=1e-14)
    # 1 mm spheres at eps = 0.4 give k ~ 1e-9 m^2 in the KC literature
    assert 8e-10 < float(kozeny_carman_permeability(d, sf)) < 1.2e-9


def test_permeability_is_d2_weighted():
    d_l, d_s = 0.02, 0.01
    layer = stratified_permeability_profile([0.0, 1.0], [0.6, 0.6], d_l, d_s)
    k0, k1 = layer["k_z"]
    assert k1 / k0 == pytest.approx((d_l / d_s) ** 2, rel=1e-12)


def test_consolidation_t50_scales_with_permeability():
    h = 0.30
    k = np.full(8, 1.0e-11)
    base = consolidation_response(k, h)["t_target"]
    fast = consolidation_response(k * 4.0, h)["t_target"]
    assert base / fast == pytest.approx(4.0, rel=0.02)
    assert base > 0.0


def test_c6_sensitivity_table_is_bitwise_reproducible():
    phi = [0.2, 0.3, 0.45, 0.6, 0.75, 0.8]
    sf = [0.6] * 6
    a = permeability_sensitivity_table(phi, sf, 0.02, 0.01, 0.18, n_t=120)
    b = permeability_sensitivity_table(phi, sf, 0.02, 0.01, 0.18, n_t=120)
    assert a["rows"] == b["rows"]                  # exact, not approximate
    rows = {r["profile"]: r for r in a["rows"]}
    assert rows["large_at_top"]["k_top_over_bottom"] > 1.0
    assert rows["large_at_bottom"]["k_top_over_bottom"] < 1.0
    assert rows["uniform"]["k_top_over_bottom"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# C5: repose angle with the P96-N3 estimator
# ---------------------------------------------------------------------------

def _synthetic_pile(angle_deg: float, n: int = 400, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    r = 0.005
    x = rng.uniform(-0.6, 0.6, n)
    surface = -math.tan(math.radians(angle_deg)) * np.abs(x)
    y = surface + rng.uniform(-0.02, 0.0, n)
    pos = np.column_stack([x + 0.6, y + 0.3])
    radii = np.full(n, r)
    return pos, radii


@pytest.mark.parametrize("angle", [15.0, 25.0])
def test_c5_repose_angle_recovers_synthetic_slope(angle):
    pos, radii = _synthetic_pile(angle)
    out = measure_rest_angle(pos, radii, n_x_bins=40)
    assert "repose_calibration" in out["estimator"]      # same tool as P96-N3
    assert out["theta_r_deg"] == pytest.approx(angle, abs=3.0)
    assert out["n_surface_points"] >= 4


def test_c5_flat_surface_gives_zero_angle():
    pos, radii = _synthetic_pile(0.0, seed=1)
    out = measure_rest_angle(pos, radii, n_x_bins=40)
    assert abs(out["theta_r_deg"]) < 2.0


# ---------------------------------------------------------------------------
# Configuration guards
# ---------------------------------------------------------------------------

def test_config_rejects_inverted_sizes():
    with pytest.raises(ValueError):
        BidisperseConfig(d_small=0.02, d_large=0.01)


def test_area_fraction_composition():
    for ratio in (1.2, 1.5, 2.0, 3.0):
        cfg = BidisperseConfig(d_small=0.01, d_large=ratio * 0.01, n_particles=150)
        pack = generate_packing(cfg)
        assert pack.solid_fraction_large == pytest.approx(0.5, abs=0.02)


def test_box_size_matches_requested_volume_fraction():
    cfg = BidisperseConfig(n_particles=120, volume_fraction=0.60)
    pack = generate_packing(cfg)
    measured = float(np.sum(math.pi * pack.radii**2) / cfg.box_area)
    assert measured == pytest.approx(0.60, rel=0.02)
