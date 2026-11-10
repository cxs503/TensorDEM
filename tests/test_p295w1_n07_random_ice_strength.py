"""Acceptance tests for campaign node P295W1-N07 (random ice-plate strength field).

Every criterion of the node is asserted here on the *library* functions, and the
end-to-end acceptance study is re-run at reduced resolution
(:func:`tensordem.random_ice_strength.study.run_study`) so the whole verdict
chain is covered by one test.

Criteria covered
----------------
(a) Weibull marginal: MLE recovers the generating parameters, QQ-plot ``R^2``
    >= 0.95, KS test does not reject, and a lognormal sample *does* fail the
    same QQ test (the test has discriminating power).
(b) Gaussian random field: relative ``L2`` error of the realisation correlation
    function <= 10 %, measured ``1/e`` correlation length within 10 % of the
    prescribed one, and the copula's rank correlation matches the analytic
    ``(6/pi) asin(rho/2)`` target.
(c) Breaking-force ensemble: ``mu_F`` converged to 5 %, ``sigma_F/mu_F`` in
    ``[0.1, 0.3]``, ``F_5% < mu_F``, and ``F_5%/B`` monotonically decreasing for
    ``B = 10 -> 100 m``.

Plus the structural checks that make the above more than bookkeeping: the KL
decomposition reproduces its own covariance exactly, the minimum over a nested
window is pointwise monotone, the ``db -> 0`` limit is insensitive to the strip
resolution, and the two load-sharing idealisations predict size effects of
opposite sign.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.random_ice_strength import (
    CRITERIA,
    IceSheetStrengthModel,
    WeibullMarginal,
    breaking_force_ensemble,
    correlation_kernel,
    correlation_l2_error,
    correlation_length_1e,
    correlated_weibull_field,
    empirical_correlation,
    empirical_rank_correlation,
    ensemble_statistics,
    fit_weibull_mle,
    gaussian_field_1d,
    ks_distance,
    monotone_decreasing,
    negative_control_lognormal,
    qq_plot_r2,
    running_mean_convergence,
    size_effect_sweep,
    spearman_from_gaussian_correlation,
)
from tensordem.random_ice_strength.study import run_study

MEAN_PA = 0.54e6
SEED = 20260101


def _grid(width, db=1.0):
    return np.linspace(0.0, width, int(round(width / db)) + 1)


# --------------------------------------------------------------------- (a)
def test_weibull_moments_match_analytic_integrals():
    m = WeibullMarginal.from_mean_modulus(MEAN_PA, 5.0)
    # integrate the density numerically, independently of the Gamma identities
    x = np.linspace(0.0, 6.0 * m.scale, 400001)
    pdf = (m.modulus / m.scale) * (x / m.scale) ** (m.modulus - 1.0) * np.exp(
        -(x / m.scale) ** m.modulus)
    dx = x[1] - x[0]
    mu = float(np.sum(x * pdf) * dx)
    var = float(np.sum((x - mu) ** 2 * pdf) * dx)
    assert mu == pytest.approx(m.mean, rel=2e-6)
    assert math.sqrt(var) / mu == pytest.approx(m.cov, rel=2e-5)


def test_from_mean_cov_inverts_the_variation_coefficient():
    for target in (0.12, 0.229, 0.428):
        m = WeibullMarginal.from_mean_cov(MEAN_PA, target)
        assert m.cov == pytest.approx(target, rel=1e-9)
        assert m.mean == pytest.approx(MEAN_PA, rel=1e-12)


def test_mle_recovers_generating_parameters():
    m = WeibullMarginal.from_mean_modulus(MEAN_PA, 5.0)
    sample = m.sample(20000, rng=np.random.default_rng(SEED))
    fit = fit_weibull_mle(sample)
    assert fit["k_hat"] == pytest.approx(5.0, rel=0.05)
    assert fit["scale_hat"] == pytest.approx(m.scale, rel=0.02)
    assert fit["ks_p_value"] > 0.01
    assert fit["qq_r2"] >= 0.95


def test_mle_agrees_with_the_package_regression_estimator():
    """Independent code path: log-log regression in ``bpm_calibration``."""
    from tensordem.bpm_calibration import weibull_mle as regression_mle

    m = WeibullMarginal.from_mean_modulus(MEAN_PA, 4.0)
    sample = m.sample(20000, rng=np.random.default_rng(SEED + 1))
    ours = fit_weibull_mle(sample, quantile_plot=False)
    theirs_k, their_sigma = regression_mle(sample)
    assert ours["k_hat"] == pytest.approx(theirs_k, rel=0.05)
    assert ours["scale_hat"] == pytest.approx(their_sigma, rel=0.05)


def test_qq_r2_rejects_lognormal_negative_control():
    control = negative_control_lognormal(n=20000, sigma_ln=1.0, seed=SEED + 7)
    assert control["qq_r2"] < CRITERIA["qq_r2_min"]
    assert control["ks_p_value"] < 1e-6


def test_qq_quantiles_use_median_rank_plotting_positions():
    m = WeibullMarginal.from_mean_modulus(MEAN_PA, 5.0)
    sample = m.sample(400, rng=np.random.default_rng(SEED + 2))
    from tensordem.random_ice_strength.distributions import qq_quantiles

    x, theoretical, p = qq_quantiles(sample, m)
    assert np.all(np.diff(x) >= 0.0)
    assert p[0] == pytest.approx(0.7 / (400 + 0.4))
    assert theoretical[-1] > theoretical[0]
    assert qq_plot_r2(sample, m)["r2"] >= 0.95


# --------------------------------------------------------------------- (b)
def test_kl_decomposition_reproduces_its_own_covariance():
    """The retained modes must rebuild the discretised kernel exactly."""
    kl = gaussian_field_1d(_grid(60.0), 10.0)
    rebuilt = kl.modes.T @ kl.modes
    target = correlation_kernel(kl.x[None, :] - kl.x[:, None], 10.0)
    assert np.max(np.abs(rebuilt - target)) < 1e-10
    assert kl.energy_fraction == pytest.approx(1.0, abs=1e-10)


def test_realisation_correlation_matches_kernel_within_10_percent():
    grid = _grid(100.0)
    for length in (5.0, 10.0, 20.0):
        kl = gaussian_field_1d(grid, length)
        z = kl.sample(2000, seed=SEED)
        lags, rho = empirical_correlation(z, kl.x, n_lags=kl.n_points // 2)
        target = correlation_kernel(lags, length)
        err = correlation_l2_error(rho, target, lags=lags, max_lag=50.0)
        assert err["l2_relative"] <= 0.10, (length, err)
        assert correlation_length_1e(rho, lags) == pytest.approx(length, rel=0.10)


def test_copula_rank_correlation_matches_analytic_target():
    kl = gaussian_field_1d(_grid(100.0), 10.0)
    z = kl.sample(1500, seed=SEED + 3)
    lags, rho_rank = empirical_rank_correlation(z, kl.x, n_lags=25)
    target = spearman_from_gaussian_correlation(correlation_kernel(lags, 10.0))
    err = correlation_l2_error(rho_rank, target, lags=lags, max_lag=25.0)
    assert err["l2_relative"] <= 0.10, err


def test_copula_field_keeps_the_weibull_marginal_exactly():
    m = WeibullMarginal.from_mean_modulus(MEAN_PA, 5.0)
    kl = gaussian_field_1d(_grid(60.0), 10.0)
    field = correlated_weibull_field(m, kl, 2000, seed=SEED + 4)
    pooled = fit_weibull_mle(field.ravel())
    assert pooled["k_hat"] == pytest.approx(5.0, rel=0.10)
    assert pooled["scale_hat"] == pytest.approx(m.scale, rel=0.05)
    assert field.mean() == pytest.approx(m.mean, rel=0.02)


# --------------------------------------------------------------------- (c)
def test_weakest_link_minimum_is_pointwise_monotone_in_width():
    m = WeibullMarginal.from_mean_modulus(MEAN_PA, 5.0)
    kl = gaussian_field_1d(_grid(100.0), 10.0)
    field = correlated_weibull_field(m, kl, 200, seed=SEED + 5)
    mins = [field[:, :n].min(axis=1) for n in (10, 30, 60, 100)]
    for narrow, wide in zip(mins, mins[1:]):
        assert np.all(wide <= narrow + 1e-12)


def test_ensemble_statistics_meet_the_criteria_at_reference_width():
    m = WeibullMarginal.from_mean_modulus(MEAN_PA, 6.0)
    sheet = IceSheetStrengthModel(marginal=m)
    kl = gaussian_field_1d(_grid(50.0), 10.0)
    ens = breaking_force_ensemble(sheet, m, kl, 50, seed=SEED + 6)
    force = np.asarray(ens["forces"])
    stats = ensemble_statistics(force)
    conv = running_mean_convergence(force)
    assert 0.10 <= stats["cov"] <= 0.30
    assert stats["se_relative"] <= 0.05
    # the running-mean drift is a *diagnostic* whose own dispersion at N=50 is of
    # the order of the standard error, so it is bounded loosely here; the study
    # reports its distribution over 20 seeds
    assert conv["max_relative_drift"] <= 0.10
    assert stats["q05"] < stats["mean"]
    # the deterministic formula applied to the mean strength is NOT the answer
    assert stats["mean"] < ens["deterministic_force"]
    assert ens["deterministic_force"] == pytest.approx(
        sheet.deterministic_per_width_n_per_m() * ens["width_m"], rel=1e-12)


def test_size_effect_is_monotone_and_has_the_weibull_exponent():
    m = WeibullMarginal.from_mean_modulus(MEAN_PA, 5.0)
    sheet = IceSheetStrengthModel(marginal=m)
    kl = gaussian_field_1d(_grid(100.0), 10.0)
    widths = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    nested = size_effect_sweep(sheet, m, kl, widths, n_real=200, db_m=1.0,
                               nested=True, seed=SEED + 7)
    curve = [row["per_width_q05"] for row in nested]
    assert monotone_decreasing(curve, rtol=1e-9)["monotone_decreasing"]
    assert curve[-1] / curve[0] < 0.95
    from tensordem.random_ice_strength import fit_power_law

    assert fit_power_law(widths, curve)["exponent"] == pytest.approx(1.0 / 6.0, rel=0.35)


def test_bundle_model_predicts_the_opposite_sign_of_size_effect():
    """Documenting the contrast: averaging concentrates, weakest link degrades."""
    m = WeibullMarginal.from_mean_modulus(MEAN_PA, 6.0)
    sheet = IceSheetStrengthModel(marginal=m)
    kl = gaussian_field_1d(_grid(100.0), 10.0)
    widths = [10.0, 50.0, 100.0]
    bundle = size_effect_sweep(sheet, m, kl, widths, n_real=400, db_m=1.0,
                               failure_model="bundle", nested=True, seed=SEED + 8)
    curve = [row["per_width_q05"] for row in bundle]
    assert curve[0] < curve[-1]           # rises towards the mean
    with pytest.raises(ValueError):
        sheet.strip_forces(np.ones((2, 4)), failure_model="softening")


def test_strip_resolution_does_not_change_the_answer():
    """``db`` is a resolution parameter: halving it must not move F_5%/B."""
    m = WeibullMarginal.from_mean_modulus(MEAN_PA, 6.0)
    sheet = IceSheetStrengthModel(marginal=m)
    out = []
    for db in (1.0, 0.5, 0.25):
        kl = gaussian_field_1d(_grid(50.0, db), 10.0)
        ens = breaking_force_ensemble(sheet, m, kl, 2000, db_m=db, seed=SEED + 9)
        out.append(float(np.quantile(np.asarray(ens["per_width"]), 0.05)))
    # a finer grid can only find a lower minimum (the discrete grid caps it from
    # above), so the coarse-grid value must not be below the fine-grid one
    assert out[0] >= out[-1]
    assert out[0] == pytest.approx(out[-1], rel=0.05), out


def test_uniform_strength_recovers_the_deterministic_formula():
    m = WeibullMarginal.from_mean_modulus(MEAN_PA, 5.0)
    sheet = IceSheetStrengthModel(marginal=m)
    uniform = np.full((3, 50), m.mean)
    force = sheet.strip_forces(uniform, db_m=1.0, failure_model="weakest_link")
    assert np.allclose(force, sheet.deterministic_force(50.0), rtol=1e-12)
    force_bundle = sheet.strip_forces(uniform, db_m=1.0, failure_model="bundle")
    assert np.allclose(force_bundle, sheet.deterministic_force(50.0), rtol=1e-12)


def test_deterministic_reference_delegates_to_the_graded_slab_module():
    from tensordem.ice_channel3d.slab import (
        bending_capacity_per_width_n_per_m,
        characteristic_length_m,
    )

    m = WeibullMarginal.from_mean_modulus(MEAN_PA, 5.0)
    sheet = IceSheetStrengthModel(marginal=m, thickness_m=1.2, elastic_modulus_pa=4.0e9)
    lc = characteristic_length_m(1.2, 4.0e9)
    assert sheet.char_length_m == pytest.approx(lc, rel=1e-12)
    assert sheet.deterministic_per_width_n_per_m() == pytest.approx(
        bending_capacity_per_width_n_per_m(m.mean, 1.2, lc), rel=1e-12)


# --------------------------------------------------------------- end to end
def test_acceptance_study_all_criteria_pass():
    out = run_study(sensitivity=False, write=False,
                    calibration={"n_realisations_field": 800,
                                 "n_samples_marginal": 5000},
                    criteria={"sweep_n_real_high": 400})
    failed = {k: v for k, v in out["checks"].items() if not v["pass"]}
    assert not failed, failed
    assert out["all_pass"] is True
    # the criterion numbers must be present in the report payload
    for key in ("criterion_a_qq_r2", "criterion_b_correlation_l2",
                "criterion_c_cov_band", "criterion_c_size_effect_monotone"):
        assert key in out["checks"]
        assert "value" in out["checks"][key]
    assert out["artifacts"]["study_json"].endswith("study.json")