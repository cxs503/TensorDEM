"""Tests for the nonlocal granular fluidity (NGF) closure and its calibration."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.nonlocal_rheology.calibration import (RheologySample, fit_amplitude,
                                                     fit_local_law, ngf_predict,
                                                     rate_floor, sample_from_profile,
                                                     score)
from tensordem.nonlocal_rheology.fluidity import (JopLaw, NGFError, NGFParams,
                                                  RadialGrid, cg_jacobi,
                                                  cartesian_2d_operator,
                                                  local_fluidity, solve_ngf)
import scipy.sparse as sp


# ---------------------------------------------------------------- local law
def test_jop_inversion_roundtrip():
    law = JopLaw(mu_s=0.31, mu_2=0.72, i_0=0.28)
    I = np.array([1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0])
    back = law.inertia(law.mu(I))
    assert np.allclose(back, I, rtol=1e-10)


def test_local_fluidity_zero_below_yield_and_positive_above():
    law = JopLaw(mu_s=0.3, mu_2=0.7, i_0=0.3)
    mu = np.array([0.2, 0.3, 0.4, 0.6])
    g = local_fluidity(mu, np.full(4, 1000.0), 2e-3, 2500.0, law)
    assert g[0] == 0.0 and g[1] == 0.0
    assert np.all(g[2:] > 0.0)
    assert np.all(np.diff(g[2:]) > 0.0)


def test_local_fluidity_rejects_bad_input():
    law = JopLaw()
    with pytest.raises(NGFError):
        local_fluidity(np.array([0.4]), np.array([-1.0]), 2e-3, 2500.0, law)
    with pytest.raises(NGFError):
        local_fluidity(np.array([0.4]), np.array([1.0]), 0.0, 2500.0, law)


# ---------------------------------------------------------------- CG solver
def test_cg_jacobi_matches_dense_solve():
    rng = np.random.default_rng(3)
    n = 40
    R = rng.normal(size=(n, n))
    A = sp.csr_matrix(R @ R.T + n * np.eye(n))
    b = rng.normal(size=n)
    x, rep = cg_jacobi(A, b, rtol=1e-12, maxiter=5000)
    assert rep.converged
    assert rep.residual <= 1e-11
    assert np.allclose(A @ x, b, atol=1e-9 * max(1.0, np.abs(b).max()))


def test_cg_rejects_indefinite_operator():
    A = sp.csr_matrix(np.diag([1.0, -1.0]))
    with pytest.raises(NGFError):
        cg_jacobi(A, np.ones(2))


def test_radial_operator_is_symmetric_positive_definite():
    grid = RadialGrid(0.01, 0.045, 25)
    K = grid.operator(np.full(25, 1e-5)).toarray()
    M = sp.diags(grid.measure).toarray()
    A = M + K
    assert np.max(np.abs(A - A.T)) == 0.0
    assert np.linalg.eigvalsh(A).min() > 0.0


# ---------------------------------------------------------------- NGF solve
def test_ngf_uniform_field_returns_local_fluidity():
    law = JopLaw(mu_s=0.3, mu_2=0.7, i_0=0.3)
    grid = RadialGrid(0.01, 0.04, 20)
    mu = np.full(20, 0.5)
    P = np.full(20, 1000.0)
    res = solve_ngf(mu, P, 2e-3, 2500.0, NGFParams(law=law, amplitude=0.5), grid)
    assert res.converged
    assert np.allclose(res.g, res.g_loc, rtol=1e-10)


def test_ngf_zero_amplitude_is_local():
    law = JopLaw(mu_s=0.3, mu_2=0.7, i_0=0.3)
    grid = RadialGrid(0.01, 0.04, 30)
    mu = np.linspace(0.6, 0.32, 30)
    P = np.full(30, 800.0)
    res = solve_ngf(mu, P, 2e-3, 2500.0, NGFParams(law=law, amplitude=0.0), grid)
    assert np.allclose(res.g, res.g_loc, rtol=1e-12)


def test_ngf_amplification_grows_with_amplitude_and_reaches_plug():
    law = JopLaw(mu_s=0.3, mu_2=0.7, i_0=0.3)
    grid = RadialGrid(0.01, 0.03, 40)
    # flowing inner half, plug (mu < mu_s) outer half
    mu = np.concatenate([np.full(20, 0.55), np.full(20, 0.22)])
    P = np.full(40, 700.0)
    amps = []
    for A in (0.05, 0.3, 0.8):
        res = solve_ngf(mu, P, 2e-3, 2500.0, NGFParams(law=law, amplitude=A), grid)
        amps.append(res.g[-1])
    assert amps[0] < amps[1] < amps[2]
    assert amps[-1] > 0.0                      # creep reaches into the plug


def test_ngf_creep_profile_decays_with_distance():
    law = JopLaw(mu_s=0.3, mu_2=0.7, i_0=0.3)
    grid = RadialGrid(0.01, 0.03, 80)
    mu = np.concatenate([np.full(20, 0.6), np.full(60, 0.2)])
    P = np.full(80, 700.0)
    res = solve_ngf(mu, P, 2e-3, 2500.0, NGFParams(law=law, amplitude=0.35), grid)
    tail = res.g[20:]
    assert np.all(np.diff(tail) <= 1e-12 * max(tail.max(), 1e-30))   # monotone decay
    assert tail[-1] < tail[0]


def test_ngf_manufactured_solution_second_order():
    law = JopLaw()
    errors = []
    for n in (21, 41):
        h = 1.0 / (n - 1)
        K, measure, index = cartesian_2d_operator(n, n, h)
        X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
        exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
        xi = 0.05
        src = (1.0 + 2.0 * np.pi ** 2 * xi ** 2) * exact
        A = (sp.diags(measure) + xi ** 2 * K).tocsr()
        active = np.ones((n, n), dtype=bool)
        active[0, :] = active[-1, :] = active[:, 0] = active[:, -1] = False
        free = index[active]
        fixed = np.ones(n * n, dtype=bool)
        fixed[free] = False
        rhs = measure[free] * src.ravel()[free] - A[free][:, fixed] @ exact.ravel()[fixed]
        sol, rep = cg_jacobi(A[free][:, free], rhs, rtol=1e-13)
        assert rep.converged
        errors.append(float(np.max(np.abs(sol - exact.ravel()[free]))))
    ratio = errors[0] / errors[1]
    assert 3.4 < ratio < 4.6, ratio      # 2nd-order convergence under h -> h/2


def test_ngf_rejects_bad_wall_condition():
    law = JopLaw()
    grid = RadialGrid(0.01, 0.03, 10)
    with pytest.raises(Exception):
        solve_ngf(np.full(10, 0.5), np.full(10, 100.0), 2e-3, 2500.0,
                  NGFParams(law=law), grid, inner="clamped")


# ---------------------------------------------------------------- calibration
def _synthetic_sample(a_true: float, law: JopLaw, *, gap_d: float, n: int = 18,
                      seed: int = 0) -> RheologySample:
    """A profile generated exactly by the NGF model (recovery test)."""
    r = np.linspace(0.011, 0.030, n)
    mu = np.linspace(0.62, 0.25, n)
    P = np.full(n, 600.0)
    d, rho = 2e-3, 2500.0
    grid = RadialGrid(float(r[0] - 0.5 * (r[1] - r[0])),
                      float(r[-1] + 0.5 * (r[1] - r[0])), n)
    res = solve_ngf(mu, P, d, rho, NGFParams(law=law, amplitude=a_true), grid)
    rate = res.g * mu
    return RheologySample(run="synthetic", gap_d=gap_d, omega=10.0, seed=seed, r=r,
                          phi=np.full(n, 0.8), pressure=P, mu=mu, gamma_dot=rate,
                          tau=mu * P, tau_torque=mu * P, d50=d, rho=rho)


def test_fit_local_law_recovers_declared_law():
    truth = JopLaw(mu_s=0.25, mu_2=0.85, i_0=0.35)
    d, rho = 2e-3, 2500.0
    rates, mus, pressures = [], [], []
    for I in np.linspace(0.01, 0.4, 40):
        mu = float(truth.mu(I))
        P = 1000.0
        rates.append(I * math.sqrt(P / rho) / d)
        mus.append(mu)
        pressures.append(P)
    n = len(rates)
    sample = RheologySample(run="syn", gap_d=20.0, omega=1.0, seed=0,
                            r=np.linspace(0.011, 0.03, n), phi=np.full(n, 0.8),
                            pressure=np.array(pressures), mu=np.array(mus),
                            gamma_dot=np.array(rates), tau=np.array(mus) * 1000.0,
                            tau_torque=np.array(mus) * 1000.0, d50=d, rho=rho)
    fit = fit_local_law([sample], i_fraction=0.9)
    assert fit["r2"] > 0.999
    assert abs(fit["law"].mu_s - 0.25) < 0.02
    assert abs(fit["law"].i_0 - 0.35) < 0.05


def test_fit_amplitude_recovers_generating_amplitude():
    law = JopLaw(mu_s=0.3, mu_2=0.7, i_0=0.3)
    a_true = 0.45
    thin = [_synthetic_sample(a_true, law, gap_d=6.0, seed=s) for s in (0, 1)]
    floor = rate_floor(thin)
    fit = fit_amplitude(thin, law, floor, bounds=(0.02, 2.0))
    assert abs(fit["amplitude"] - a_true) < 0.03


def test_score_improves_with_nonlocal_correction():
    law = JopLaw(mu_s=0.3, mu_2=0.7, i_0=0.3)
    samples = [_synthetic_sample(0.45, law, gap_d=6.0),
               _synthetic_sample(0.45, law, gap_d=20.0)]
    floor = rate_floor(samples)
    params = NGFParams(law=law, amplitude=0.45)
    scored = score(samples, law, params, floor=floor)
    assert scored["r2_log_ngf"] > scored["r2_log_local"]
    assert scored["n_bins"] == sum(int(s.valid().sum()) for s in samples)


def test_ngf_predict_returns_finite_amplification():
    law = JopLaw()
    sample = _synthetic_sample(0.3, law, gap_d=6.0)
    pred = ngf_predict(sample, law, NGFParams(law=law, amplitude=0.3))
    assert pred["converged"]
    assert np.all(np.isfinite(pred["rate_nl"]))
    assert np.all(pred["rate_nl"] >= 0.0)


# ---------------------------------------------------------------- Couette kernel
def test_couette_kernel_short_run_is_physical():
    from tensordem.nonlocal_rheology.couette import CouetteCase, simulate_couette
    case = CouetteCase(r_inner=0.010, r_outer=0.022, omega=50.0, seed=3,
                       settle_strain=1.0, shear_strain=2.0, average_strain=1.0,
                       n_bins=8)
    out = simulate_couette(case)
    assert out["n_particles"] > 50
    assert np.all(out["phi"] <= 1.0 + 1e-9)          # clipped-area porosity
    assert np.all(np.isfinite(out["mu"])) and np.all(out["mu"] >= 0.0)
    assert out["pressure_mean"] > 0.0
    assert out["r_wall"] >= case.r_inner + 2.5 * case.d50
    valid = np.isfinite(out["gamma_dot"]) & (out["phi"] > 0.5)
    assert valid.sum() >= 3
    assert np.all(out["tau_torque"][valid] > 0.0)


def test_couette_case_rejects_bad_geometry():
    from tensordem.nonlocal_rheology.couette import CouetteCase, CouetteError
    with pytest.raises(CouetteError):
        CouetteCase(r_inner=0.02, r_outer=0.021)
    with pytest.raises(CouetteError):
        CouetteCase(fill_fraction=1.2)