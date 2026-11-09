"""Maturation checks for the ``lam_x`` estimator chain.

These tests pin the properties this node claims:

* the level-crossing / local estimators carry the asymptotic
  ``Var(lam_hat/lam) ~ lam/L`` scaling that the first-zero-truncated integral
  scale does not (campaign criterion A3);
* the fixed-record-length variance scan really holds ``L`` fixed (the earlier
  release held ``L/lam`` fixed, which forces a zero slope by construction and
  cannot test a scaling law at all);
* the reported bootstrap interval contains the reported value;
* the level-crossing estimator does not depend on the sampling step.
"""
from __future__ import annotations

import numpy as np

from tensordem.ice_pressure_band import (
    METHODS,
    PRIMARY_METHODS,
    BandConfig,
    bootstrap_lambda,
    dem_band_cell,
    lambda_crossing,
    lambda_integral,
    lambda_lagslope,
    loglog_fit,
    run_variance_family,
    snapshot_profiles,
    variance_scaling_grid,
    windowed_field,
)


def test_crossing_estimator_recovers_known_lambda_at_two_steps():
    """Level-crossing length is a physical length: the step must not matter."""
    rng = np.random.default_rng(7)
    lam = 1.0
    for dx in (0.05, 0.25):
        n = int(round(64 * lam / dx))
        vals = [lambda_crossing(windowed_field(n, dx, lam, rng, kind="ar1"), dx)
                for _ in range(25)]
        med = float(np.nanmedian(vals))
        assert 0.88 < med / lam < 1.12, (dx, med)


def test_lagslope_bias_depends_on_sampling_step_so_it_is_not_a_reporting_estimator():
    """The leading-lag inversion is a *variance* demonstrator, not the reported value.

    Its sensitivity to an error in the lag-1 autocorrelation is
    ``1 / (R(dx) ln(1/R(dx)))``, which is ~20 at ``dx = 0.05 lam`` and ~3 at
    ``dx = 0.5 lam``, and it also inherits the window-mean removal
    (``~ Var(mean)/Var``) divided by the same small denominator.  The measured
    median therefore moves by more than 10 % when only the sampling step changes,
    which is exactly why the reported ``lam_x`` uses the level-crossing /
    integral estimators and the leading-lag one is quoted as a cross-check.
    """
    rng = np.random.default_rng(11)
    est = {}
    for dx in (0.05, 0.5):
        n = int(round(64 / dx))
        est[dx] = float(np.nanmedian(
            [lambda_lagslope(windowed_field(n, dx, 1.0, rng, kind="ar1"), dx)
             for _ in range(25)]))
    assert abs(est[0.5] - est[0.05]) > 0.05, est
    assert 0.7 < est[0.05] < 1.15, est
    assert 0.7 < est[0.5] < 1.15, est


def test_variance_follows_lambda_over_record_length_for_local_estimators():
    """``Var(lam_hat/lam) ~ lam/L``: slope ~1 on a 2-D grid, R**2 > 0.9."""
    res = variance_scaling_grid(
        K_values=(32.0, 64.0, 128.0, 256.0), lam_values=(1.0,),
        n_real=120, kind="ar1", seed=3, methods=("integral", "lagslope", "crossing"),
    )
    fits = {m: res["per_method"][m]["relvar_vs_lam_over_L"] for m in
            ("integral", "lagslope", "crossing")}
    for m in ("lagslope", "crossing"):
        assert 0.7 <= fits[m]["slope"] <= 1.3, (m, fits[m])
        assert fits[m]["r2"] > 0.9, (m, fits[m])
    # the first-zero-truncated integral scale does *not* obey the law: it is the
    # defect this node repairs by changing the estimator, not the fit
    assert fits["integral"]["r2"] < fits["crossing"]["r2"], fits


def test_fixed_length_variance_scan_holds_length_constant():
    """Regression test: the scan must vary ``lam`` at fixed ``L``, not fixed ``L/lam``."""
    lam = (0.5, 1.0, 2.0)
    res = run_variance_family(lam_values=lam, record_length=64.0, n_real=6,
                              kind="ar1", methods=("crossing",))
    assert res["record_length"] == 64.0
    assert np.allclose(res["L_over_lam"], [64.0 / v for v in lam])
    entry = res["crossing"]
    assert len(entry["rel_sd"]) == len(lam)
    assert np.isfinite(entry["rel_sd_vs_lam_slope"]) or np.isnan(entry["rel_sd_vs_lam_slope"])
    assert "rel_sd_vs_lam_r2" in entry


def test_basic_bootstrap_interval_contains_the_reported_value():
    """An interval that excludes the point estimate is not an uncertainty statement."""
    rng = np.random.default_rng(5)
    ny, T = 200, 60
    y = np.arange(ny) * 0.0125
    base = np.exp(-y / 0.20)
    snaps = np.array([base * (1.0 + 0.2 * rng.standard_normal(ny)) for _ in range(T)])
    point = float(lambda_integral(snaps.mean(axis=0), 0.0125))
    assert np.isfinite(point) and 0.1 < point < 0.4
    out = bootstrap_lambda(snaps, 0.0125, n_boot=80, block=6, seed=0, point_m=point)
    assert out["basic_contains_point"] is True
    assert out["basic_ci_low_m"] <= point <= out["basic_ci_high_m"]
    assert out["basic_ci_low_m"] < out["basic_ci_high_m"]


def test_loglog_fit_reports_r2_and_slope():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    y = 3.0 * x ** 0.5
    fit = loglog_fit(x, y)
    assert abs(fit["slope"] - 0.5) < 1e-9
    assert abs(fit["r2"] - 1.0) < 1e-9
    assert fit["n"] == 4


def test_primary_methods_are_registered_and_distinct():
    assert "crossing" in METHODS and "lagslope" in METHODS
    assert set(PRIMARY_METHODS).issubset(set(METHODS))
    assert "spectral" not in PRIMARY_METHODS  # not calibratable at K < 50


def test_dem_band_cell_reports_interval_and_scales():
    """A short band run must return the full auditable record (structure only)."""
    cfg = dict(width_over_thickness=10.0, d50_ratio=0.25, seed=0,
               n_max=600, n_boot=40, ladder_factors=(1, 2))
    cell = dem_band_cell(**cfg, methods=("integral", "crossing"))
    for key in ("lambda_m", "lambda_over_h", "lambda_over_d50", "bootstrap",
                "resolution_ladder", "shuffle_control", "snapshot_scale", "regime"):
        assert key in cell, key
    assert cell["bootstrap"]["basic_contains_point"] in (True, False)
    assert cell["estimate"]["n_samples"] == BandConfig(
        width_over_thickness=10.0, d50_ratio=0.25, n_max=600).ny
    snap = cell["snapshot_scale"]["crossing"]
    # a single snapshot of a grating-limited band may carry no measurable
    # crossing at all; the record must then say so instead of inventing a number
    assert snap["n_snapshots"] > 0 or np.isnan(snap["median_m"])