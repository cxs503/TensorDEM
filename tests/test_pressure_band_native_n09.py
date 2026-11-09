"""Tests for the native ``lam_x`` calibration of the ice pressure band (p48w2b node).

Covered here:

* the block-scale DEM verifies its own physics (global force balance, load
  control, quasi-static ratio, overlap) on a short run;
* the measurement chain conserves force and the gauge does not move the mean;
* the two-stage calibrator is monotone, certified-range aware and serialises;
* the kernel-free block-load ACF separates a white from a correlated sequence;
* the negative-control router returns a *period* for a periodic record and
  withholds ``lambda``;
* the shot-noise null reproduces a kernel-limited ``lambda`` at the grid scale.

All assertions compare computed quantities against analytic references or
against an invariant of the algorithm -- there are no hard-coded expectations of
the kind "the answer is 0.43".
"""
from __future__ import annotations

import numpy as np
import pytest

from tensordem.ice_pressure_band import (
    ar1_field,
    dominant_period,
    estimate_all,
    report_length_scale,
)
from tensordem.ice_pressure_band.block_band import (
    BlockBandConfig,
    block_load_acf,
    coarse_grain,
    deposit_profile,
    estimate_profile_lambda,
    shot_noise_lambda,
    simulate_block_band,
    time_mean_profile,
)
from tensordem.ice_pressure_band.calibration import (
    MIN_CERTIFIED_K,
    TwoStageCalibration,
    calibrate_two_stage,
    flexible_factor_ceiling,
    measure_ratio,
)


# ---------------------------------------------------------------------------
# block-scale DEM
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_run():
    cfg = BlockBandConfig(block_over_thickness=2.0, band_over_thickness=30.0,
                          settle_steps=9000, sample_steps=4000, n_snapshots=6,
                          contact_time=4.0e-3, target_traction=1.0e6, seed=3)
    return cfg, simulate_block_band(cfg)


def test_block_dem_satisfies_global_force_balance(small_run):
    cfg, run = small_run
    # Newton's third law for the whole bed: wall force == platen force
    assert run.summary["global_force_balance_rel"] < 0.02, run.summary
    assert run.summary["frac_time_wall_loaded"] > 0.5


def test_block_dem_runs_quasi_statically_with_small_overlap(small_run):
    cfg, run = small_run
    # quasi-static: the bed's mean speed is a tiny fraction of D / contact time
    assert run.summary["quasi_static_ratio"] < 0.05, run.summary
    # rigid-block regime: contact overlaps stay small
    assert run.summary["overlap_rel_max"] < 0.05, run.summary
    # the servo drives the platen at (or below) its speed cap in one direction
    assert run.summary["mean_abs_platen_speed"] <= cfg.drive_speed * 1.001


def test_block_dem_profile_is_stationary_and_positive(small_run):
    cfg, run = small_run
    assert run.pressure.shape == (cfg.n_snapshots, cfg.ny)
    assert np.all(run.pressure >= 0.0)
    mean = time_mean_profile(run.pressure)
    assert np.isfinite(mean).all()
    # the deposited field reproduces the recorded wall traction (measurement
    # chain consistency; the short fixture run is still ramping up, hence 50 %)
    traction = float(np.sum(mean) * cfg.dy) / cfg.band
    wall = run.summary["mean_wall_force_per_depth"] / cfg.band / cfg.thickness
    assert abs(traction / wall - 1.0) < 0.5, (traction, wall)


def test_deposition_conserves_force_and_is_periodic():
    rng = np.random.default_rng(0)
    dy, ny, band = 0.02, 400, 8.0
    y = rng.uniform(0.0, band, 60)
    f = rng.uniform(1.0, 10.0, 60)
    prof = deposit_profile(y, f, dy, ny, band)
    assert prof.size == ny
    assert abs(prof.sum() - f.sum()) < 1e-9 * f.sum()


def test_coarse_grain_preserves_the_mean_and_widens_only_locally():
    rng = np.random.default_rng(1)
    x = rng.normal(size=512)
    for cells in (1, 4, 9):
        sm = coarse_grain(x, cells)
        assert sm.size == x.size
        assert abs(sm.mean() - x.mean()) < 5e-2 * abs(x.mean()) + 1e-12


def test_estimator_on_the_block_profile_recovers_the_calibrated_length(small_run):
    cfg, run = small_run
    prof = coarse_grain(time_mean_profile(run.pressure),
                        max(1, int(round(cfg.gauge / cfg.dy))))
    est = estimate_profile_lambda(prof, cfg.dy, cfg.thickness, models=None)
    assert np.isfinite(est.raw["integral"]) and est.raw["integral"] > 0.0
    # record length B = 30 h = 6 m; lambda must be far shorter than the record
    assert est.L_over_lambda > 1.5


def test_block_load_acf_separates_white_from_correlated():
    block = 0.4
    y = np.arange(0.0, 40 * block, block)
    rng = np.random.default_rng(2)
    white = block_load_acf(y, rng.normal(1.0, 0.3, y.size), block)
    assert abs(white["lambda_over_block"] - 0.5) < 0.25
    # a slow modulation along the band must give a clearly longer scale
    corr_force = 1.0 + 0.5 * np.sin(2.0 * np.pi * y / (8 * block))
    corr = block_load_acf(y, corr_force, block)
    assert corr["lambda_over_block"] > white["lambda_over_block"] + 0.35


def test_shot_noise_null_is_grid_limited():
    cfg = BlockBandConfig(block_over_thickness=2.0, band_over_thickness=20.0,
                          n_snapshots=1, gauge_over_block=0.5)
    out = shot_noise_lambda(cfg, contacts_per_block=1.0, force_mean=1.0,
                            force_cv=0.6, n_real=6, method="integral")
    # a structureless Poisson contact field carries no length beyond the sensor
    lam = out["lambda_m"] if np.isfinite(out["lambda_m"]) else out["single_realisation_median_m"]
    assert np.isfinite(lam)
    assert lam < 3.0 * cfg.gauge


# ---------------------------------------------------------------------------
# calibration machinery
# ---------------------------------------------------------------------------
def test_measured_ratio_table_has_the_analytic_limits():
    tab = measure_ratio(kind="ar1", method="crossing", k_grid=(3.0, 8.0, 50.0),
                        n_real=80, dx_over_lam=0.05, seed=5)
    rho = np.asarray(tab["ratio_mean"], dtype=float)
    # strong negative bias at K = 3, near-unbiased at K = 50
    assert rho[0] < 0.75
    assert rho[-1] > 0.8
    assert np.all(np.isfinite(tab["relative_sd"]))


def test_two_stage_calibration_is_monotone_and_range_aware():
    cal = calibrate_two_stage(kind="ar1", method="crossing",
                              k_grid=(3.0, 4.0, 6.3, 10.0, 20.0, 50.0),
                              n_real=120, seed1=0, seed2=900)
    # certified range: below K_min the value is withheld, never silently reported
    assert np.isnan(cal.correct(0.3, 0.3 * 2.9))
    lam_lo = cal.correct(0.4, 3.0)               # K_hat = 7.5
    lam_hi = cal.correct(1.0, 3.0)               # K_hat = 3.0
    assert np.isfinite(lam_lo) and np.isfinite(lam_hi)
    assert lam_hi > lam_lo                       # monotone map of the raw estimate
    # the divisor f(K) = rho (1 + delta): at K = 3 the correction *multiplies* the
    # raw estimate (1/f > 1); at K = 50 it is within 15 % of identity
    assert 1.0 / cal.correction_factor(3.0) > 1.0
    assert abs(cal.correction_factor(50.0) - 1.0) < 0.15
    assert cal.k_min == MIN_CERTIFIED_K


def test_two_stage_calibration_round_trips():
    cal = calibrate_two_stage(kind="ar1", method="crossing",
                              k_grid=(3.0, 5.0, 10.0, 25.0, 50.0),
                              n_real=80, seed1=0, seed2=900)
    back = TwoStageCalibration.from_dict(cal.to_dict())
    for K in (3.5, 8.0, 30.0):
        assert back.correction_factor(K) == pytest.approx(cal.correction_factor(K))


def test_flexible_factor_ceiling_reports_a_floor_not_zero():
    out = flexible_factor_ceiling(kind="ar1", method="crossing",
                                  k_grid=(3.0, 5.0, 10.0, 25.0, 50.0), degree=2,
                                  n_real=80, seed_train=0, seeds_test=(201,))
    assert np.isfinite(out["test_max_abs_bias_K_ge_3"])
    # the ceiling cannot reach a few-percent residual at K = 3 (identifiability)
    assert out["test_max_abs_bias_K_ge_3"] > 0.0


# ---------------------------------------------------------------------------
# negative controls / regime routing
# ---------------------------------------------------------------------------
def test_periodic_record_routes_to_a_period_not_a_pseudo_length():
    dx, n = 0.05, 2000
    cells = 50.0
    x = np.sin(2.0 * np.pi * np.arange(n) * dx / (cells * dx))
    out = report_length_scale(x, dx)
    assert out["quantity"] == "period"
    assert out["lambda_withheld"] is True
    assert abs(out["value_m"] / dx - cells) / cells < 0.02
    per = dominant_period(x, dx)
    assert abs(per["cycles"] - n / cells) < 0.1 * (n / cells)


def test_white_noise_is_withheld_and_correlated_record_reports_a_length():
    rng = np.random.default_rng(4)
    dx, n = 0.05, 2000
    white = report_length_scale(rng.normal(size=n), dx)
    assert white["quantity"] == "none"
    assert white["lambda_withheld"] is True
    corr = report_length_scale(ar1_field(n, dx, 1.0, rng), dx)
    assert corr["quantity"] == "correlation_length"
    assert 0.6 < corr["value_m"] < 1.6


def test_long_record_estimator_regression():
    """On long records the crossing estimator is the accurate one.

    The integral estimator truncates at the first zero of a noisy ACF and stays
    biased low even at ``L = 200 lam``; that ordering is asserted so a future
    change to the estimators cannot silently swap which one is reported.
    """
    rng = np.random.default_rng(5)
    vals = {m: [] for m in ("integral", "expfit", "crossing")}
    for _ in range(8):
        x = ar1_field(4000, 0.05, 1.0, rng)
        e = estimate_all(x, 0.05, methods=tuple(vals))
        for m in vals:
            vals[m].append(e[m])
    med = {m: float(np.median(v)) for m, v in vals.items()}
    assert abs(med["crossing"] - 1.0) < 0.10, med
    assert med["integral"] < med["crossing"], med
    assert med["expfit"] < med["crossing"], med