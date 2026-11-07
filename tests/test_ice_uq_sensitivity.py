"""Tests for the DEM ice-resistance uncertainty-quantification toolkit.

Fast analytic checks cover the estimators (Ishigami references, linear-Gaussian
posterior), the design generators and the identifiability diagnostics.  The DEM
chain itself is exercised through a single guarded smoke test because one wedge
case costs tens of seconds; enable it with ``TENSORDEM_RUN_DEM_UQ=1``.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from tensordem.sensitivity import identifiability as idf
from tensordem.sensitivity import propagation as prop
from tensordem.sensitivity.design import Parameter, ParameterSpace
from tensordem.sensitivity.ice_chain import ICE_SPACE, _lattice, wedge_case
from tensordem.sensitivity.sobol import (
    bootstrap_indices, ishigami, ishigami_exact_indices, saltelli_indices,
)
from tensordem.sensitivity.surrogate import GaussianProcess, PolynomialChaos, r2_score


def _ishigami_space() -> ParameterSpace:
    return ParameterSpace([Parameter(f"x{i}", 0.0, 1.0) for i in range(3)])


def _unit_to_ishigami(U):
    return 2.0 * math.pi * (np.atleast_2d(U) - 0.5)


# ------------------------------------------------------------------ design --
def test_parameter_round_trip_linear_and_log():
    linear = Parameter("a", 2.0, 8.0)
    logp = Parameter("b", 1.0e-3, 1.0, log=True)
    values = np.array([0.0, 0.25, 0.5, 1.0])
    assert np.allclose(linear.to_unit(linear.to_physical(values)), values)
    assert np.allclose(logp.to_unit(logp.to_physical(values)), values)
    assert math.isclose(logp.to_physical(0.5), math.sqrt(1.0e-3), rel_tol=1e-12)


def test_parameter_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        Parameter("bad", 2.0, 1.0)
    with pytest.raises(ValueError):
        Parameter("bad_log", 0.0, 1.0, log=True)


def test_latin_hypercube_is_stratified():
    space = _ishigami_space()
    U = space.latin_hypercube(16, seed=3, candidates=2)
    assert U.shape == (16, 3)
    for dim in range(3):
        strata = np.floor(U[:, dim] * 16).astype(int)
        assert len(set(strata.tolist())) == 16          # one point per stratum
    assert np.array_equal(U, space.latin_hypercube(16, seed=3, candidates=2))


def test_saltelli_design_structure():
    space = _ishigami_space()
    A, B, AB = space.saltelli_design(64, seed=5)
    assert A.shape == B.shape == (64, 3)
    assert AB.shape == (3, 64, 3)
    for i in range(3):
        # AB_i keeps every column of A except the i-th, taken from B.
        others = [j for j in range(3) if j != i]
        assert np.array_equal(AB[i][:, others], A[:, others])
        assert np.array_equal(AB[i][:, i], B[:, i])


# ------------------------------------------------------------------- sobol --
def test_saltelli_matches_ishigami_reference():
    space = _ishigami_space()
    A, B, AB = space.saltelli_design(2**14, seed=11)
    result = saltelli_indices(lambda U: ishigami(_unit_to_ishigami(U)), A, B, AB)
    exact = ishigami_exact_indices()
    assert np.max(np.abs(result["first"] - exact["first"])) < 0.03
    assert np.max(np.abs(result["total"] - exact["total"])) < 0.03
    assert result["first"][2] < 1e-3                       # x3 has no main effect


def test_saltelli_rejects_constant_response():
    space = _ishigami_space()
    A, B, AB = space.saltelli_design(32, seed=2)
    with pytest.raises(ValueError):
        saltelli_indices(lambda U: np.ones(len(U)), A, B, AB)


def test_bootstrap_indices_bracket_point_estimate():
    space = _ishigami_space()
    A, B, AB = space.saltelli_design(2**11, seed=13)
    evaluate = lambda U: ishigami(_unit_to_ishigami(U))  # noqa: E731
    point = saltelli_indices(evaluate, A, B, AB)
    ci = bootstrap_indices(evaluate, A, B, AB, replicates=32, seed=17)
    for i in range(3):
        assert ci["first_ci"][0][i] <= point["first"][i] <= ci["first_ci"][1][i]
        assert ci["total_ci"][0][i] <= point["total"][i] <= ci["total_ci"][1][i]


# --------------------------------------------------------------- surrogate --
def test_pce_reproduces_ishigami_indices():
    space = _ishigami_space()
    U = space.sobol_design(1024, seed=21)
    y = ishigami(_unit_to_ishigami(U))
    chaos = PolynomialChaos(3, degree=8).fit(U, y)
    indices = chaos.sobol_indices()
    exact = ishigami_exact_indices()
    assert np.max(np.abs(indices["first"] - exact["first"])) < 5e-3
    assert np.max(np.abs(indices["total"] - exact["total"])) < 5e-3
    assert r2_score(y, chaos.predict(U)) > 0.999


def test_pce_loo_error_reports_overfit_basis():
    space = _ishigami_space()
    U = space.sobol_design(64, seed=23)
    rng = np.random.default_rng(23)
    y = rng.standard_normal(len(U))
    chaos = PolynomialChaos(3, degree=3).fit(U, y)
    # White noise cannot be fitted: the leave-one-out score must go negative.
    assert chaos.loo_error()["loo_r2"] < 0.2


def test_gp_predicts_held_out_ishigami():
    space = _ishigami_space()
    U = space.sobol_design(512, seed=29)
    y = ishigami(_unit_to_ishigami(U))
    gp = GaussianProcess(3).fit(U[:384], y[:384])
    mean, std = gp.predict(U[384:], return_std=True)
    assert r2_score(y[384:], mean) > 0.99
    assert np.all(std > 0.0)


def test_surrogate_rejects_short_training_set():
    with pytest.raises(ValueError):
        PolynomialChaos(3, degree=4).fit(np.random.default_rng(0).random((5, 3)),
                                         np.zeros(5))


# ---------------------------------------------------------- identifiability --
def test_finite_difference_jacobian_matches_linear_map():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(12, 4))
    jac = idf.finite_difference_jacobian(lambda u: A @ np.asarray(u).ravel(),
                                          np.full(4, 0.5))
    assert np.max(np.abs(jac - A)) < 1e-7


def test_fisher_information_and_correlation_are_scale_free():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(20, 4))
    fim = idf.fisher_information(A, np.full(20, 0.1))
    assert np.isclose(fim, A.T @ A / 0.01).all()
    corr = idf.correlation_matrix(fim)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)
    # Scaling a column by 10 leaves the (unit-free) correlation unchanged.
    scaled = A.copy()
    scaled[:, 2] *= 10.0
    corr_scaled = idf.correlation_matrix(idf.fisher_information(scaled, np.full(20, 0.1)))
    assert np.allclose(corr, corr_scaled, atol=1e-8)


def test_collinear_pair_is_detected_and_condition_number_drops():
    # Two parameters act only through their sum: the pair is unidentifiable.
    def model(u):
        u = np.asarray(u, dtype=float).ravel()
        s = u[0] + u[1]
        return np.array([s, s**2, u[2], np.sin(3 * u[2])])

    u0 = np.array([0.4, 0.6, 0.5])
    jac = idf.finite_difference_jacobian(model, u0)
    fim = idf.fisher_information(jac, np.full(4, 0.01))
    corr = idf.correlation_matrix(fim)
    pairs = idf.collinear_pairs(corr, ["a", "b", "c"], threshold=0.9)
    assert pairs and pairs[0]["pair"] == ["a", "b"]
    assert abs(pairs[0]["correlation"]) > 0.95
    subset = idf.best_identifiable_subset(fim, ["a", "b", "c"], sizes=(2, 3))
    two = next(e for e in subset["per_size"] if e["size"] == 2)
    assert two["condition_number"] < subset["full_condition_number"]
    assert two["condition_gain_vs_full"] > 10.0


def test_profile_likelihood_flags_inert_parameter():
    def model(u):
        u = np.asarray(u, dtype=float).ravel()
        # The second parameter has exactly zero sensitivity: the curve cannot
        # constrain it at all, so its profile must run into the box face.
        return np.array([2.0 * u[0], 0.5 + 0.0 * u[1]])

    truth = np.array([0.4, 0.6])
    y = model(truth)
    profiles = idf.profile_likelihood(model, y, np.full(2, 1e-3), np.array([0.5, 0.5]),
                                      names=["active", "inert"])
    active, inert = profiles
    assert active["width"] < 0.2
    assert not active["bounded_by_prior"]
    # The second parameter is untested by this curve: its interval hits the box.
    assert inert["width"] > 0.5 or inert["bounded_by_prior"]


# ------------------------------------------------------------- propagation --
def test_laplace_posterior_matches_analytic_gaussian_update():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(6, 2))
    sigma = 0.2
    prior_sd = 0.3
    u0 = np.array([0.4, 0.6])
    residual = np.array([0.1, -0.2, 0.05, 0.0, 0.03, -0.01])
    post = prop.laplace_posterior(A, residual, np.full(6, sigma), u0, prior_sd=prior_sd)
    expected_cov = np.linalg.inv(A.T @ A / sigma**2 + np.eye(2) / prior_sd**2)
    prior_pull = (u0 - 0.5) / prior_sd**2          # prior mean of the unit cube
    expect_mean = u0 + expected_cov @ (A.T @ residual / sigma**2 - prior_pull)
    assert np.allclose(post["covariance"], expected_cov)
    assert np.allclose(post["mean"], expect_mean)


def test_sample_gaussian_respects_the_box():
    draws = prop.sample_gaussian(np.array([0.02, 0.98]),
                                 np.diag([0.2, 0.2]) ** 2, 500, seed=3)
    assert draws.shape == (500, 2)
    assert draws.min() >= 0.0 and draws.max() <= 1.0


def test_coverage_test_counts_misses():
    truth = np.array([1.0, 2.0, 3.0, 40.0])
    result = prop.coverage_test(truth, np.array([0.5, 1.5, 2.5, 3.5]),
                                np.array([1.5, 2.5, 3.5, 4.5]))
    assert result["covered"] == 3
    assert result["coverage"] == pytest.approx(0.75)
    assert result["misses_high"] == 1
    lo, hi = result["wilson95"]
    assert lo <= 0.75 <= hi


def test_posterior_predictive_widens_with_observation_noise():
    samples = np.linspace(0.0, 1.0, 200)[:, None]
    model = lambda s: s.ravel()  # noqa: E731
    tight = prop.posterior_predictive(model, samples, sigma_obs=0.01)
    loose = prop.posterior_predictive(model, samples, sigma_obs=0.5)
    assert loose["u95_half_width"] > tight["u95_half_width"]
    assert tight["mean"] == pytest.approx(0.5, abs=1e-9)


# ----------------------------------------------------------------- the chain --
def test_lattice_mapping_is_integer_layers():
    nz, radius = _lattice(4.4)
    assert nz == 4
    assert math.isclose(2 * nz * radius, 0.16)
    with pytest.raises(ValueError):
        _lattice(2.0)


def test_ice_space_matches_the_expected_parameter_count():
    assert len(ICE_SPACE) == 8
    labels = list(ICE_SPACE.names)
    assert labels[:5] == ["mu", "e", "k_n", "k_t_ratio", "eps_b"]


@pytest.mark.skipif(os.environ.get("TENSORDEM_RUN_DEM_UQ") != "1",
                    reason="DEM wedge case costs tens of seconds; set "
                           "TENSORDEM_RUN_DEM_UQ=1 to enable")
def test_wedge_case_returns_finite_resistance_and_trace():
    metrics = wedge_case(np.full(8, 0.5))
    assert metrics["R_window_mean_N"] > 0.0
    assert len(metrics["trace_N"]) == 12
    assert all(value > 0.0 for value in metrics["trace_N"])
    assert metrics["particles"] > 0
    with pytest.raises(ValueError):
        wedge_case(np.full(8, 0.5), dt_factor=2.0)