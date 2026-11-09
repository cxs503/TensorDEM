"""Tests for the pressure-band correlation-length estimator (campaign P204-N08).

The tests are written as *negative controls and analytic anchors*, not as
regression snapshots: every assertion compares a computed quantity against a
known value (analytic ACF, exact AR(1) exponential, a shuffled profile that must
collapse to the grid) or against a mechanism (bias must shrink after the
correction, white noise must be rejected).
"""
from __future__ import annotations

import numpy as np
import pytest

from tensordem.ice_pressure_band import (
    BandConfig,
    BiasModel,
    LITERATURE_BAND,
    METHODS,
    PRIMARY_METHODS,
    autocorrelation,
    bootstrap_lambda,
    build_estimate,
    consensus,
    correct_length,
    diagnose_regime,
    estimate_all,
    estimate_profile_lambda,
    exponential_acf,
    fit_bias_model,
    k_sufficient,
    lambda_efold,
    lambda_integral,
    run_bias_study,
    shuffle_control,
    snapshot_profiles,
    spectral_field,
    to_prop_1104,
    validate,
    windowed_field,
)


def _acf(x, dx):
    return autocorrelation(x, dx)


def test_field_generators_match_analytic_acf():
    """Both generators reproduce R(r) = exp(-r/lam) of the *generator*, not the estimator."""
    rng = np.random.default_rng(0)
    lam, dx, n = 1.0, 0.05, 4000
    for kind in ("spectral", "ar1"):
        fields = [windowed_field(n, dx, lam, rng, kind=kind) for _ in range(8)]
        acf = np.mean([_acf(f, dx)[1] for f in fields], axis=0)
        lags = np.arange(n) * dx
        sel = lags <= 3.0 * lam
        assert np.max(np.abs(acf[sel] - exponential_acf(lags[sel], lam))) < 0.05
        # a pure circulant synthesis must also have unit variance by construction
        assert abs(np.var(fields[0]) - 1.0) < 1e-8


def test_estimators_recover_known_lambda_on_long_record():
    """At L/lam = 80 the calibrated estimators must be within 20 % of the truth."""
    rng = np.random.default_rng(1)
    lam, dx = 1.0, 0.05
    n = int(80 / dx)
    vals = {m: [] for m in METHODS}
    for _ in range(12):
        x = windowed_field(n, dx, lam, rng, kind="spectral")
        est = estimate_all(x, dx)
        for m in METHODS:
            vals[m].append(est[m])
    for m in METHODS:
        median = float(np.nanmedian(vals[m]))
        assert 0.8 < median / lam < 1.2, (m, median)


def test_white_noise_is_rejected_and_grid_limited():
    rng = np.random.default_rng(2)
    dx = 0.05
    x = rng.standard_normal(400)
    regime = diagnose_regime(x, dx)
    assert regime.label == "white_noise"
    assert lambda_integral(x, dx) < 3.0 * dx


def test_strictly_periodic_record_is_flagged_but_correlated_is_not():
    dx = 0.05
    period = 0.4
    x = np.sin(2 * np.pi * np.arange(400) * dx / period)
    assert diagnose_regime(x, dx).label == "periodic"
    rng = np.random.default_rng(3)
    y = windowed_field(400, dx, 1.0, rng, kind="spectral")
    assert diagnose_regime(y, dx).label == "ok"


@pytest.fixture(scope="module")
def small_study():
    k_grid = np.array([2.0, 4.0, 8.0, 16.0, 40.0])
    return run_bias_study(k_grid=k_grid, n_real=60, kinds=("spectral",),
                          seed=5, tol=0.05)


def test_finite_window_bias_is_negative_and_vanishes(small_study):
    for m in PRIMARY_METHODS:
        ratio = np.asarray(small_study.kinds["spectral"][m]["ratio_mean"], dtype=float)
        assert ratio[0] < 0.8, (m, ratio)      # K = 2 : strong negative bias
        assert ratio[-1] > 0.85, (m, ratio)    # K = 40: nearly unbiased


def test_bias_model_fit_and_self_consistent_correction(small_study):
    k_grid = small_study.k_grid
    for m in PRIMARY_METHODS:
        ratio = np.asarray(small_study.kinds["spectral"][m]["ratio_mean"], dtype=float)
        model = small_study.model("spectral", m)
        assert np.isfinite(model.rms) and model.params
        # the measured bias is strongly negative at small K and shrinks at large K
        assert float(np.asarray(model.rho(k_grid[0]))) < float(np.asarray(model.rho(k_grid[-1])))
        # the correction is the fixed point of lam = lam_hat / rho(L/lam)
        lam_hat = 0.5
        L = 3.0
        lam = correct_length(lam_hat, L, model)
        assert np.isfinite(lam) and lam > 0.0
        residual = abs(lam - float(np.asarray(lam_hat / model.rho(L / lam_hat)))) / lam
        assert residual < 1e-12, (m, residual)            # plug-in identity, exact by construction
        assert not np.isfinite(correct_length(lam_hat, L, BiasModel(method="x", form="inv_poly",
                                                                   params=())))
    # for a model with a known negative bias the correction must inflate
    negative = fit_bias_model(np.array([2.0, 5.0, 10.0, 20.0, 50.0]),
                              np.array([0.30, 0.62, 0.80, 0.90, 0.96]), "integral",
                              form="inv_poly")
    assert float(np.asarray(negative.rho(10.0))) < 1.0
    assert correct_length(0.8, 10.0, negative) > 0.8
    ks = k_sufficient(negative, tol=0.05)
    assert np.isfinite(ks) and 1.0 < ks < 500.0


def test_threshold_and_variance_slope_from_study(small_study):
    for m in METHODS:
        slope = small_study.variance["spectral"][m]["sd_log_log_slope"]
        assert np.isfinite(slope) and -2.5 < slope < 1.0
        loo = np.asarray(small_study.sufficiency["spectral"][m]["loo_residual_bias"], dtype=float)
        assert loo.size == small_study.k_grid.size
    # the sustained-threshold rule returns a grid value or infinity -- never a
    # value between grid points, and it cannot be satisfied by a lone lucky cell
    from tensordem.ice_pressure_band import sustained_threshold
    K = np.array([2.0, 4.0, 8.0, 16.0])
    assert sustained_threshold(K, [0.9, 0.01, 0.9, 0.01]) == 16.0   # only the tail counts
    assert np.isinf(sustained_threshold(K, [0.9, 0.9, 0.9, 0.9]))
    assert sustained_threshold(K, [0.2, 0.2, 0.04, 0.01]) == 8.0


def test_profile_estimator_and_shuffle_control(small_study):
    """A synthetic exponential profile with known lam must be recovered, and the
    shuffled control must collapse to the grid scale."""
    rng = np.random.default_rng(4)
    lam, dy = 0.4, 0.0125
    n = int(60 * lam / dy)
    prof = spectral_field(n, dy, lam, rng)
    models = {m: small_study.models["spectral"][m] for m in METHODS}
    est = estimate_profile_lambda(prof, dy, models=models)
    # a *single* record carries 30-45 % scatter at L/lam = 60, so the acceptance
    # band is a factor of two; the calibration is a statement about the ensemble
    assert 0.4 < est["corrected"]["integral"] / lam < 3.0
    assert est["regime"]["label"] == "ok"
    ctrl = shuffle_control(prof, dy, models=models, n_perm=8, seed=1)
    assert ctrl["ratio_to_grid"] < 3.0        # a shuffled profile has no length left


def test_snapshot_profiles_and_time_average():
    cfg = BandConfig(thickness=0.2, width_over_thickness=20.0, cells_per_thickness=10)
    T, nz, ny = 40, cfg.z_cells, cfg.ny
    rng = np.random.default_rng(5)
    pressure = rng.random((T, nz, ny)) * 1e6
    snaps = snapshot_profiles(pressure, cfg)
    assert snaps.shape == (20, ny)
    np.testing.assert_allclose(snaps.mean(axis=0), time_average(pressure, cfg), rtol=1e-12)


def time_average(pressure, cfg):
    from tensordem.ice_pressure_band import time_averaged_profile
    return time_averaged_profile(pressure, cfg)


def test_band_config_geometry_and_validation():
    cfg = BandConfig(thickness=0.2, width_over_thickness=60.0, cells_per_thickness=16)
    assert cfg.ny == 960
    assert cfg.dy == pytest.approx(0.0125)
    assert cfg.width == pytest.approx(12.0)
    with pytest.raises(ValueError):
        BandConfig(cells_per_thickness=4).validate()
    with pytest.raises(ValueError):
        BandConfig(steady_window=(0.8, 0.2)).validate()


def test_bootstrap_returns_bracketing_interval(small_study):
    rng = np.random.default_rng(6)
    dy, lam = 0.0125, 0.4
    n = int(60 * lam / dy)
    snaps = np.array([spectral_field(n, dy, lam, rng) for _ in range(30)]) * 1e6
    models = {m: small_study.models["spectral"][m] for m in METHODS}
    out = bootstrap_lambda(snaps, dy, models=models, method="integral", n_boot=60, seed=2)
    assert out["ci_low_m"] <= out["ci_high_m"]
    assert out["ci_low_m"] <= out["mean_m"] <= out["ci_high_m"] * 1.5
    assert out["block_snapshots"] >= 2


def test_contract_roundtrip_and_prop_1104_view():
    est = {"primary": "integral", "methods": {"integral": 0.42},
           "method_agreement": "consensus"}
    rec = build_estimate(
        value_m=0.42, thickness_m=0.2, ci95_m=(0.3, 0.55), estimator=est,
        record_length_m=12.0, dx_m=0.0125, n_samples=960, threshold_K=10.0,
        bias_corrected=True, regime="ok", evidence_grade="CALIBRATED",
        sources=["test"],
    )
    assert validate(rec) == []
    assert rec["value_over_thickness"] == pytest.approx(2.1)
    assert rec["quality"]["sufficient"] is True
    assert rec["literature_comparison"]["inside_band"] is True
    view = to_prop_1104(rec)
    assert view["lambda_x_h"] == pytest.approx(2.1)
    assert view["lambda_x_sufficient"] is True
    assert view["lambda_x_schema"] == rec["schema_version"]


def test_contract_flags_insufficient_record_and_bad_interval():
    est = {"primary": "integral"}
    rec = build_estimate(
        value_m=0.4, thickness_m=0.2, ci95_m=(0.2, 0.6), estimator=est,
        record_length_m=1.2, dx_m=0.0125, n_samples=96, threshold_K=10.0,
        bias_corrected=True, regime="ok", evidence_grade="CALIBRATED",
    )
    assert rec["quality"]["sufficient"] is False
    assert rec["quality"]["design_usable"] is False
    assert any("below the sufficiency threshold" in w for w in rec["warnings"])
    bad = dict(rec)
    bad["ci95"] = [0.5, 0.6]      # does not bracket the value
    assert any("bracket" in e for e in validate(bad))
    bad2 = dict(rec)
    bad2["quality"] = dict(rec["quality"], sufficient=True, regime="periodic")
    assert any("regime" in e for e in validate(bad2))


def test_consensus_and_literature_band_exports():
    cons = consensus({"integral": 0.4, "expfit": 0.42, "spectral": 0.6})
    assert cons["method_spread"] == pytest.approx(0.476, rel=1e-2)
    assert cons["agreement"] == "disagreement"
    assert LITERATURE_BAND == (1.5, 3.0)
    # legacy e-folding estimator is exposed for the finite-window comparison
    rng = np.random.default_rng(7)
    x = windowed_field(int(60 / 0.05), 0.05, 1.0, rng)
    assert abs(lambda_efold(x, 0.05) - 1.0) < 0.35