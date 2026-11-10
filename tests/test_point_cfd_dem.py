"""Tests for the point-particle (unresolved) two-way CFD-DEM coupling.

Three layers:

1. closure-level identities (pure functions, exact);
2. discretisation-level identities of the spectral fluid and the
   interpolation/spreading adjoint pair (round-off);
3. the closed-form verification gates (short, reduced-size configurations) --
   the full-size runs live in ``benchmarks/point_particle_cfd_dem/run.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.point_cfd_dem import closures, verification
from tensordem.point_cfd_dem.coupling import PointParticleCFDDEM
from tensordem.point_cfd_dem.fluid import SpectralPeriodicFluid, trilinear_weights
from tensordem.point_cfd_dem.particles import (ParticleSet, PointParticleModel,
                                               evaluate_forces)


# ---------------------------------------------------------------------------
# 1. closures
# ---------------------------------------------------------------------------
def test_stokes_terminal_velocity_and_relaxation_time():
    v_t = closures.terminal_velocity_stokes(2.0, 1.0, 0.5, 1.0, 0.5)
    assert v_t == pytest.approx(1.0 * 0.5 * 1.0 ** 2 / (18 * 0.5), rel=1e-14)
    # tau_p = m_p |g|_eff / (3 pi mu d v_t) must equal rho_p d^2/(18 mu)
    tau = closures.relaxation_time_stokes(2.0, 1.0, 0.5)
    assert tau == pytest.approx(2.0 / (18 * 0.5), rel=1e-14)
    m_p = closures.sphere_mass(1.0, 2.0)
    f_drag = closures.stokes_drag_coefficient(0.5, 1.0) * v_t
    assert tau == pytest.approx(m_p * v_t / f_drag, rel=1e-12)


def test_schiller_naumann_tends_to_stokes():
    assert closures.drag_correction(0.0) == pytest.approx(1.0, abs=1e-15)
    assert closures.drag_correction(1e-6) == pytest.approx(1.0, abs=1e-4)
    assert closures.drag_correction(100.0) > 1.0
    assert closures.drag_correction(100.0, "stokes") == pytest.approx(1.0)


def test_richardson_zaki_exponent_branches():
    assert closures.richardson_zaki_exponent(0.1) == pytest.approx(4.65)
    assert closures.richardson_zaki_exponent(0.5) == pytest.approx(
        4.35 * 0.5 ** -0.03)
    assert closures.richardson_zaki_exponent(100.0) == pytest.approx(
        4.45 * 100.0 ** -0.1)
    assert closures.richardson_zaki_exponent(1000.0) == pytest.approx(2.39)


def test_richardson_zaki_hindrance_is_consistent_with_settling_law():
    """The drag multiplier must reproduce ``u_s = u_t eps**n`` in the container frame."""
    eps, n = 0.93, 4.65
    h = closures.richardson_zaki_hindrance(eps, n)
    # container frame: net flux zero -> u_rel = u_s/eps ; drag balance u_rel = u_t/H
    u_s = eps * (1.0 / h)
    assert u_s == pytest.approx(eps ** n, rel=1e-14)


# ---------------------------------------------------------------------------
# 2. discretisation identities
# ---------------------------------------------------------------------------
def test_trilinear_weights_partition_of_unity():
    fl = SpectralPeriodicFluid((8, 8, 8), 0.5)
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, 1.0, size=(20, 3)) * fl.extent
    idx, w = trilinear_weights(pos, fl.shape, fl.h)
    assert np.allclose(np.sum(w, axis=0), 1.0, atol=1e-14)
    assert idx.shape == (8, 20, 3)
    assert np.all(idx >= 0) and np.all(idx < 8)


def test_spread_is_exactly_the_transpose_of_interpolate():
    fl = SpectralPeriodicFluid((8, 8, 8), 0.5)
    # well separated particles: stencils do not overlap, so the interpolated
    # value at particle i is the self-coupling of its own spread force
    pos = np.array([[0.2, 0.2, 0.2], [2.1, 0.3, 0.4], [0.4, 2.2, 0.3],
                    [1.6, 1.7, 1.8], [3.1, 3.2, 0.9]])
    force = np.arange(15, dtype=float).reshape(3, 5) / 10.0
    field = fl.spread(force, pos)
    # total is conserved (partition of unity)
    assert np.allclose(field.sum(axis=(1, 2, 3)), force.sum(axis=1), rtol=0,
                       atol=1e-13)
    # self-coupling is the analytic weight sum
    _, w = trilinear_weights(pos, fl.shape, fl.h)
    self_sum = np.sum(w ** 2, axis=0)
    assert np.allclose(fl.interpolate(field, pos), force * self_sum,
                       rtol=0, atol=1e-13)
    assert np.allclose(fl.self_weight_sum(pos), self_sum, rtol=0, atol=1e-15)


def test_projection_is_divergence_free_and_preserves_the_mean():
    fl = SpectralPeriodicFluid((8, 8, 8), 0.5)
    rng = np.random.default_rng(2)
    u = rng.normal(size=(3, 8, 8, 8))
    mean_before = u.mean(axis=(1, 2, 3))
    up = fl.project(u)
    assert np.allclose(up.mean(axis=(1, 2, 3)), mean_before, rtol=0, atol=1e-15)
    assert fl.max_divergence(up) < 1e-10
    assert fl.max_divergence(u) > 1e-3


def test_spectral_gradient_is_exact_for_a_resolved_mode():
    fl = SpectralPeriodicFluid((24, 24, 24), 0.5)
    L = fl.extent[2]
    k = 2 * math.pi / L
    z = np.arange(fl.shape[2]) * fl.h
    p = np.broadcast_to(np.sin(k * z)[None, None, :], fl.shape).copy()
    grad = fl.gradient(p)
    analytic = np.broadcast_to((k * np.cos(k * z))[None, None, :], fl.shape)
    assert np.max(np.abs(grad[2] - analytic)) / k < 1e-12


def test_self_response_is_positive_and_grows_in_time():
    fl = SpectralPeriodicFluid((12, 12, 12), 1.0, rho=1.0, mu=0.5)
    pos = np.array([[3.0, 3.0, 3.0], [4.2, 5.1, 2.7]])
    r0 = fl.self_response(pos, 0.01, elapsed=0.01)
    r1 = fl.self_response(pos, 0.01, elapsed=5.0)
    r_inf = fl.self_response(pos, 0.01)
    assert np.all(r0 > 0)
    assert np.all(r1 > r0)
    assert np.all(r_inf >= r1)
    # inviscid fluid has no steady response at all
    assert np.allclose(SpectralPeriodicFluid((8, 8, 8), 1.0, mu=0.0)
                       .self_response(pos, 0.01), 0.0)


# ---------------------------------------------------------------------------
# 3. force budget
# ---------------------------------------------------------------------------
def test_neutrally_buoyant_particle_feels_no_net_force():
    part = ParticleSet.create([[4.0, 4.0, 4.0]], [[0.0, 0.0, 0.0]],
                              radius=0.5, density=1.0)
    model = PointParticleModel(rho_f=1.0, mu=0.5,
                               gravity=np.array([0.0, 0.0, -0.5]))
    f = evaluate_forces(part, model, np.zeros((3, 1)), grad_p=np.zeros((3, 1)))
    assert np.allclose(f["total"], 0.0, atol=1e-15)
    assert f["effective_mass"][0] == pytest.approx(
        closures.sphere_mass(1.0, 1.0) + 0.5 * closures.sphere_volume(1.0))


def test_drag_balance_gives_the_terminal_velocity():
    d, rho_p, rho_f, mu, g = 1.0, 2.0, 1.0, 0.5, 0.5
    v_t = closures.terminal_velocity_stokes(rho_p, rho_f, g, d, mu)
    part = ParticleSet.create([[1.0, 1.0, 1.0]], [[0.0, 0.0, -v_t]],
                              radius=d / 2, density=rho_p)
    model = PointParticleModel(rho_f=rho_f, mu=mu, added_mass=0.0,
                               drag_scheme="stokes",
                               gravity=np.array([0.0, 0.0, -g]))
    f = evaluate_forces(part, model, np.zeros((3, 1)))
    assert np.allclose(f["total"], 0.0, atol=1e-12 * abs(f["weight"]).max())


def test_added_mass_is_implicit_in_the_acceleration():
    """a = F/(m_p + C_A m_a) -- the factor that makes the oscillator exact."""
    d = 1.0
    part = ParticleSet.create([[1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0]],
                              radius=d / 2, density=2.0)
    model = PointParticleModel(rho_f=1.0, mu=0.0, added_mass=0.5,
                               gravity=np.array([0.0, 0.0, 0.0]))
    f = evaluate_forces(part, model, np.zeros((3, 1)),
                        external_force=np.array([[0.0, 0.0, 1.0]]))
    a = f["total"][0, 2] / f["effective_mass"][0]
    m_a = 0.5 * closures.sphere_volume(d)
    assert a == pytest.approx(1.0 / (closures.sphere_mass(d, 2.0) + m_a),
                              rel=1e-13)


# ---------------------------------------------------------------------------
# 4. coupling invariants
# ---------------------------------------------------------------------------
def test_interphase_momentum_is_conserved_to_roundoff():
    r = verification.momentum_exchange_case(n_particles=8, cells=8, dt=0.01,
                                            n_steps=20)
    assert r["residual_adjoint"] < 1e-12
    # and the gate detects a deliberate 2 % exchange bias
    assert r["residual_exchange_bias_2pct"] > 1e-3
    assert r["gate"]["passed"] is True


def test_fluid_momentum_equals_the_exchange_impulse():
    fl = SpectralPeriodicFluid((8, 8, 8), 1.0, rho=1.0, mu=0.2)
    part = ParticleSet.create([[4.0, 4.0, 4.0]], [[0.2, 0.0, 0.0]],
                              radius=0.3, density=5.0)
    model = PointParticleModel(rho_f=1.0, mu=0.2, added_mass=0.0,
                               gravity=np.array([0.0, 0.0, 0.0]))
    sim = PointParticleCFDDEM(fl, part, model)
    p0 = fl.momentum(sim.u)
    sim.step(0.01)
    dp_f = fl.momentum(sim.u) - p0
    dp_p = part.momentum() - np.array([5.0 * closures.sphere_volume(0.6) * 0.2,
                                       0.0, 0.0])
    assert np.allclose(dp_f + dp_p, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# 5. verification gates (reduced size)
# ---------------------------------------------------------------------------
def test_gate_stokes_terminal_velocity():
    r = verification.stokes_terminal_velocity_case(cells=16, dt=0.01,
                                                   t_end=2.0, window=0.4)
    assert r["rel_error"] < 0.02, r["gate"]
    assert r["rel_error_uncorrected"] > 0.05     # the gate discriminates


def test_gate_added_mass_oscillation():
    r = verification.added_mass_oscillation_case(cells=6, h=1.0, dt=0.02,
                                                 periods=2.5)
    assert r["rel_error"] < 0.05, r["gate"]
    assert r["counterfactual_no_added_mass_error"] > 0.05


def test_gate_pressure_gradient():
    r = verification.pressure_gradient_case(cells=16)
    assert r["operator_error"] < 1e-10
    assert r["force_error"] < 0.02


def test_gate_richardson_zaki():
    r = verification.richardson_zaki_case()
    assert r["rel_error"] < 0.10, r["gate"]


def test_run_all_quick_reports_gate_table():
    res = verification.run_all(quick=True)
    assert res["n_gates"] == 5
    assert len(res["gates"]) == 5
    assert all(set(g) >= {"gate", "criterion", "measured", "threshold",
                          "passed", "evidence"} for g in res["gates"])