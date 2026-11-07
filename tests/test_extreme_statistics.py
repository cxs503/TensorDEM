"""Tests of the extreme-value / sampling-error statistics of wear fields.

Every test is a closed-form or Monte-Carlo-controlled check of an estimator that
the ice-abrasion certification relies on (``docs/extreme_statistics_acceptance_protocol.md``).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import tensordem.extreme_statistics as es


def _sample(params: es.GEVParams, n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return es.gev_ppf(params, rng.random(n))


# --- distribution and maximum likelihood -----------------------------------


@pytest.mark.parametrize("xi", [-0.25, 0.0, 0.25])
def test_gev_ppf_cdf_roundtrip(xi: float) -> None:
    params = es.GEVParams(mu=1.5, sigma=0.7, xi=xi)
    p = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    x = es.gev_ppf(params, p)
    assert np.allclose(es.gev_cdf(params, x), p, rtol=1e-10, atol=1e-12)


def test_gumbel_limit_is_continuous_in_shape() -> None:
    p = 0.95
    a = es.GEVParams(0.0, 1.0, 1e-9).quantile(p)
    b = es.GEVParams(0.0, 1.0, 0.0).quantile(p)
    c = es.GEVParams(0.0, 1.0, 1e-9).quantile(p)
    assert abs(a - b) < 1e-6 and abs(c - b) < 1e-6


def test_gumbel_fit_recovers_parent() -> None:
    parent = es.GEVParams(1.0, 0.5, 0.0)
    x = _sample(parent, 20000, seed=11)
    fit = es.gumbel_fit(x)
    assert abs(fit.mu - parent.mu) / abs(parent.mu) < 0.03
    assert abs(fit.sigma - parent.sigma) / parent.sigma < 0.03
    assert abs(fit.quantile(0.95) - parent.quantile(0.95)) / parent.quantile(0.95) < 0.01


def test_gev_fit_recovers_shape_and_quantile() -> None:
    parent = es.GEVParams(1.0, 1.0, 0.25)
    x = _sample(parent, 20000, seed=13)
    fit = es.gev_fit(x)
    assert abs(fit.xi - parent.xi) < 0.06
    assert abs(fit.quantile(0.95) - parent.quantile(0.95)) / parent.quantile(0.95) < 0.03


def test_loglik_gradient_matches_finite_difference() -> None:
    x = _sample(es.GEVParams(1.0, 1.0, 0.2), 200, seed=17)
    params = es.GEVParams(1.05, 0.95, 0.22)
    ll, grad = es.gev_loglik(params, x, gradient=True)
    steps = np.array([1e-6, 1e-6, 1e-6])

    def f(p):
        return es.gev_loglik(es.GEVParams(*p), x)

    base = np.array([params.mu, params.sigma, params.xi])
    num = np.array([(f(base + steps * np.eye(3)[i]) - f(base - steps * np.eye(3)[i]))
                    / (2 * steps[i]) for i in range(3)])
    assert np.allclose(grad, num, rtol=1e-4, atol=1e-4)


# --- confidence intervals ---------------------------------------------------


def test_gumbel_profile_interval_contains_truth_and_scales_with_n() -> None:
    parent = es.GEVParams(1.0, 0.5, 0.0)
    truth = parent.quantile(0.95)
    wide = es.gumbel_profile_ci(_sample(parent, 200, seed=19), 0.95)
    tight = es.gumbel_profile_ci(_sample(parent, 20000, seed=19), 0.95)
    assert wide["ci_lo"] < truth < wide["ci_hi"]
    # sqrt(n) narrowing: 100x the sample must be ~10x tighter (single draws can
    # miss by a few percent, hence the Monte Carlo coverage arm below)
    assert tight["ci_rel_halfwidth_pct"] < 0.4 * wide["ci_rel_halfwidth_pct"]
    cov = es.coverage_study(parent, n=2000, n_rep=80, methods=("profile",))
    assert cov["methods"]["profile"]["coverage"] >= 0.85


def test_gev_profile_interval_is_asymmetric_and_brackets_truth() -> None:
    parent = es.GEVParams(1.0, 1.0, 0.25)
    truth = parent.quantile(0.95)
    ci = es.gev_profile_ci(_sample(parent, 400, seed=23), 0.95)
    assert ci["ci_lo"] < truth < ci["ci_hi"]
    up = ci["ci_hi"] - ci["estimate"]
    down = ci["estimate"] - ci["ci_lo"]
    assert up > 0 and down > 0
    assert abs(up - down) / max(up, down) > 1e-3       # genuinely asymmetric


def test_gumbel_profile_coverage_quick() -> None:
    cov = es.coverage_study(es.GEVParams(0.0, 1.0, 0.0), n=150, n_rep=60,
                            methods=("profile", "order"))
    assert cov["methods"]["profile"]["coverage"] >= 0.85
    assert cov["methods"]["order"]["coverage"] >= 0.85
    assert cov["methods"]["profile"]["mean_rel_halfwidth_pct"] > 0.0


def test_order_statistic_interval_is_ordered() -> None:
    x = _sample(es.GEVParams(1.0, 1.0, 0.1), 500, seed=29)
    ci = es.order_statistic_ci(x, 0.95)
    assert ci["ci_lo"] <= ci["estimate"] <= ci["ci_hi"]
    assert ci["rank_lo"] < ci["rank_hi"] <= ci["n"]


def test_delta_method_interval_positive_width() -> None:
    x = _sample(es.GEVParams(1.0, 1.0, 0.2), 2000, seed=31)
    ci = es.delta_method_ci(x, 0.95)
    assert ci["ci_hi"] > ci["ci_lo"] > 0
    assert math.isfinite(ci["ci_rel_halfwidth_pct"])


# --- extreme-value index ----------------------------------------------------


def test_hill_estimator_frechet() -> None:
    x = _sample(es.GEVParams(1.0, 1.0, 0.25), 20000, seed=37)
    hill = es.hill_estimator(x, 500)
    assert abs(hill["xi"] - 0.25) < 0.06
    assert hill["se"] > 0.0


def test_block_maxima_shapes_and_values() -> None:
    x = np.arange(10.0)
    assert np.allclose(es.block_maxima(x, 3), [2.0, 5.0, 8.0])
    with pytest.raises(ValueError):
        es.block_maxima(x, 1)


def test_evi_recovery_smoke() -> None:
    res = es.evi_recovery_study(n=4000, blocks=(20, 100), n_rep=2)
    for row in res["parents"]["frechet"]["rows"]:
        assert np.isfinite(row["xi_mean"])
        assert row["xi_rel_error_pct"] < 60.0


# --- extremal index ---------------------------------------------------------


def test_extremal_index_iid_is_one() -> None:
    x = es.max_ar1_series(20000, 0.0, seed=41)
    res = es.extremal_index_runs(x, u_quantile=0.95, run_length=1)
    assert abs(res["theta"] - 1.0) < 0.12
    blocks = es.extremal_index_blocks(x, 100)
    assert abs(blocks["theta"] - 1.0) < 0.25


@pytest.mark.parametrize("a,theta", [(0.5, 0.5), (0.7, 0.3)])
def test_extremal_index_max_ar1_recovers_one_minus_a(a: float, theta: float) -> None:
    x = es.max_ar1_series(20000, a, seed=43)
    stab = es.extremal_index_stability(x, run_lengths=(1, 2, 3, 5, 8))
    assert abs(stab["theta"] - theta) < 0.08
    assert stab["plateau_size"] >= 3


def test_extremal_index_stability_scan_is_monotone_in_run_length() -> None:
    x = es.max_ar1_series(20000, 0.5, seed=47)
    scan = [es.extremal_index_runs(x, run_length=r)["theta"] for r in (1, 2, 4, 8)]
    assert scan[0] >= scan[-1]            # declustering removes exceedances


# --- shot noise, cell averaging and the resolution law ----------------------


def test_shot_noise_quantile_limits() -> None:
    shape = np.exp(np.random.default_rng(3).normal(0.0, 0.5, 5000))
    low = es.shot_noise_quantile(0.5, shape, 0.95, n_mc=50000)
    high = es.shot_noise_quantile(500.0, shape, 0.95, n_mc=50000)
    assert low["normalised_quantile"] > high["normalised_quantile"]
    assert abs(high["normalised_quantile"] - 1.0) < 0.1       # -> deterministic
    assert high["zero_probability"] < low["zero_probability"]


def test_shot_rate_from_zero_fraction_roundtrip() -> None:
    profile = np.linspace(0.2, 2.0, 500)
    c_true = 3.7
    zero = float(np.mean(np.exp(-c_true * profile / profile.mean())))
    assert abs(es.shot_rate_from_zero_fraction(profile, zero) - c_true) < 1e-6


def test_band_field_p95_converges_to_mean_for_dense_pits() -> None:
    profile = np.ones((4, 20))
    shape = np.exp(np.random.default_rng(5).normal(0.0, 0.2, 2000))
    dense = es.band_field_p95(profile, 1.0e-5, 4000.0, shape, n_realisations=20)
    assert abs(dense["p95_mean_m"] - 1.0e-5) / 1.0e-5 < 0.05


def test_activity_profile_p95_increases_with_refinement() -> None:
    import tensordem.ice_abrasion as ab

    hull = ab.HullParticulars()
    act = ab.ActivityDistribution()
    coarse = es.waterline_activity_p95(act, hull, dx_m=1.0, dz_m=1.0)
    fine = es.waterline_activity_p95(act, hull, dx_m=0.125, dz_m=0.125)
    iso_mid = es.waterline_activity_p95(act, hull, dx_m=0.25, dz_m=0.25)
    assert fine["p95_relative"] > iso_mid["p95_relative"] > coarse["p95_relative"]
    assert fine["n_cells"] > iso_mid["n_cells"] > coarse["n_cells"]
    # the cell average preserves the mean of the underlying activity profile
    assert abs(fine["mean_relative"] - 1.0) < 1e-6


def test_resolution_scaling_law_recovers_known_power_law() -> None:
    rng = np.random.default_rng(53)
    cells = [1.0, 0.5, 0.25, 0.125]
    q_inf, b_true, beta_true = 5.0e-5, 3.0e-5, 0.5
    values = {h: q_inf - b_true * h ** beta_true * (1.0 + 0.002 * rng.standard_normal(12))
              for h in cells}
    law = es.resolution_scaling_law(values, n_boot=200)
    assert abs(law["beta"] - beta_true) < 0.05
    assert abs(law["continuum_value_m"] - q_inf) / q_inf < 0.03
    assert law["finest_shortfall_pct"] > 0.0
    assert len(law["local_exponents"]) == len(cells) - 1


def test_local_scaling_exponent_constant_for_pure_power() -> None:
    rows = es.local_scaling_exponent([1.0, 0.5, 0.25], [1.0, 2.0, 4.0])
    assert all(abs(r["beta"] - 1.0) < 1e-12 for r in rows)


# --- field certificate, adjudication and protocol ---------------------------


def _synthetic_field_set(seed: int = 61):
    """Replicate gamma fields on two grid levels (for the certificate bookkeeping)."""
    rng = np.random.default_rng(seed)
    records, fields = [], {}
    for cell, shape in ((0.5, (2, 40)), (0.25, (4, 80)), (0.125, (8, 160))):
        for rep in range(12):
            d = rng.gamma(shape=0.8, scale=1.0e-5, size=shape)
            d[rng.random(shape) < 0.02] = 0.0
            fields[(cell, rep)] = d
            records.append({"cell_m": cell, "rep": rep, "p95_m": float(np.quantile(d, 0.95)),
                            "mean_m": float(d.mean()), "n_cells": int(d.size),
                            "n_nonzero": int((d > 0).sum())})
    return records, fields


def test_wear_field_p95_certificate_structure() -> None:
    records, fields = _synthetic_field_set()
    cert = es.wear_field_p95_certificate(records, fields, n_boot=200,
                                         block_shape=(1, 4),
                                         shape_sample=np.exp(np.random.default_rng(2)
                                                             .normal(0, 0.4, 500)))
    assert set(cert["per_level"]) == {"0.5", "0.25", "0.125"}
    lvl = cert["per_level"]["0.125"]
    assert lvl["p95_ci_normal_m"][0] < lvl["p95_mean_m"] < lvl["p95_ci_normal_m"][1]
    assert "block" in lvl["bootstrap"] and "iid" in lvl["bootstrap"]
    assert cert["resolution_law"]["beta"] > 0
    assert "0.5_vs_0.125" in cert["paired_differences_pct"]


def test_block_bootstrap_is_wider_than_iid_for_correlated_field() -> None:
    rng = np.random.default_rng(67)
    base = rng.gamma(shape=0.6, scale=1.0e-5, size=(6, 120))
    field = base + np.repeat(base.mean(axis=0, keepdims=True), 6, axis=0) * 0.0
    iid = es.bootstrap_quantile_ci(field.ravel(), 0.95, n_boot=400)
    blk = es.bootstrap_quantile_ci(field, 0.95, n_boot=400, block=(1, 12))
    assert blk["ci_rel_halfwidth_pct"] > iid["ci_rel_halfwidth_pct"]


def test_replicate_quantile_ci_intervals() -> None:
    vals = np.array([1.0, 1.02, 0.98, 1.01, 0.99, 1.03])
    ci = es.replicate_quantile_ci(vals)
    assert ci["ci_normal"][0] < ci["mean"] < ci["ci_normal"][1]
    assert ci["ci_percentile"][0] <= ci["ci_percentile"][1]
    assert ci["cv_pct"] > 0


@pytest.mark.parametrize("ci,model,verdict", [
    ([-1.0, 2.5], 19.8, "sampling-noise"),
    ([7.0, 12.0], 19.8, "measure-dependence"),
    ([7.0, 12.0], 60.0, "unexplained"),
])
def test_adjudicate_gci_vs_ci_branches(ci, model, verdict) -> None:
    out = es.adjudicate_gci_vs_ci(gci_pct=19.8, paired_diff_ci_pct=ci,
                                  sampling_halfwidth_pct=0.5,
                                  model_prediction_pct=model)
    assert out["verdict"] == verdict


def test_acceptance_protocol_evaluate_flags_missing_evidence() -> None:
    records, fields = _synthetic_field_set()
    cert = es.wear_field_p95_certificate(records, fields, n_boot=100,
                                         shape_sample=np.exp(np.random.default_rng(4)
                                                             .normal(0, 0.4, 500)),
                                         gci_pct=19.8)
    bare = es.acceptance_protocol_evaluate(cert)
    assert bare["all_pass"] is False
    assert bare["checks"]["P2_profile_coverage"]["pass"] is False
    full = es.acceptance_protocol_evaluate(
        cert, coverage={"level": 0.95, "n_rep": 300,
                        "methods": {"profile": {"coverage": 0.95}}},
        theta={"theta": 0.4, "plateau_size": 5})
    assert set(full["checks"]) == {"P1_three_way_ci_agreement", "P2_profile_coverage",
                                   "P3_ci_quoted_everywhere", "P4_resolution_status_declared",
                                   "P5_theta_plateau"}
    assert full["checks"]["P2_profile_coverage"]["pass"] is True
    assert full["checks"]["P5_theta_plateau"]["pass"] is True


def test_anisotropic_band_spec_keeps_geometry_audit() -> None:
    import tensordem.ice_abrasion as ab

    hull = ab.HullParticulars()
    iso = ab.build_waterline_grid(hull, ab.WaterlineBandSpec(cell_size_m=0.25))
    ani = ab.build_waterline_grid(
        hull, ab.WaterlineBandSpec(cell_size_m=0.25, cell_size_z_m=1.0))
    assert ani.shape == (2, 740) and iso.shape == (9, 740)
    assert ani.audit["geometry_audit_passed"] is True
    assert abs(ani.audit["block_coefficient_mesh"]
               - iso.audit["block_coefficient_mesh"]) < 1e-12
    assert ab.WaterlineBandSpec(cell_size_m=0.5).dz_m == 0.5
