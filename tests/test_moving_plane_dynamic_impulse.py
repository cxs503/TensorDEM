"""Dynamic ``tool_impulse`` of the moving-plane boundary against a closed form.

Companion of ``docs/moving_plane_dynamic_impulse.md`` and
``scripts/verify_moving_plane_dynamic_impulse.py``.  The frozen-network limit was
already covered by ``tests/test_moving_plane_ledger_verification.py``; what is
checked here is the *elastic, dynamic* regime, which had no reference at all.

Every reference below is built from a different route than the engine:
modal closed form, DOP853 ODE integration, and a momentum identity that uses
only the engine's end velocity.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from tensordem.moving_plane_boundary import MovingPlaneNetwork, smooth_plane
from tensordem.moving_plane_impulse import (
    assemble_linear_model,
    momentum_identity_impulse,
    plane_derivatives,
    plane_integral,
    poly_cos_moment,
    quasi_static_tool_impulse,
    run_engine,
)

DEPTH = 1.0e-3


@pytest.fixture(scope="module")
def model():
    return assemble_linear_model(MovingPlaneNetwork(8, 2))


# --------------------------------------------------------------------------- #
# reference self-checks: the closed form must be trustworthy before it judges
# --------------------------------------------------------------------------- #
def test_plane_derivatives_match_finite_differences():
    ramp = 0.5
    for t in (0.05, 0.125, 0.25, 0.49):
        h = 1e-6
        # the second difference needs a coarser step: p itself is O(1) while
        # p'' is O(1e-2), so h = 1e-6 loses every significant digit of the
        # difference to cancellation.
        h2 = 1e-3
        _, pd, pdd = plane_derivatives(t, ramp, DEPTH)
        fd1 = (smooth_plane(t + h, ramp=ramp, depth=DEPTH)
               - smooth_plane(t - h, ramp=ramp, depth=DEPTH)) / (2 * h)
        fd2 = (smooth_plane(t + h2, ramp=ramp, depth=DEPTH)
               - 2 * smooth_plane(t, ramp=ramp, depth=DEPTH)
               + smooth_plane(t - h2, ramp=ramp, depth=DEPTH)) / h2**2
        # scale-based tolerances: p' and p'' both vanish at the C2 control points
        # (e.g. p''(s = 1/2) = 0 exactly), so a relative test is meaningless there.
        scale_d1 = 1.875 * DEPTH / ramp          # max |p'| = 1.875 delta / ramp
        scale_d2 = 120.0 * DEPTH / ramp**2       # max |p''|
        assert abs(pd - fd1) <= 1e-7 * scale_d1
        assert abs(pdd - fd2) <= 5e-3 * scale_d2


def test_plane_motion_is_c2_and_endpoint_exact():
    ramp = 1.0
    assert plane_derivatives(0.0, ramp, DEPTH) == (1.0, 0.0, 0.0)
    assert plane_derivatives(ramp, ramp, DEPTH)[0] == pytest.approx(1.0 - DEPTH, abs=0)
    assert abs(plane_derivatives(ramp, ramp, DEPTH)[1]) < 1e-15
    assert abs(plane_derivatives(ramp - 1e-12, ramp, DEPTH)[2]) < 1e-6


def test_plane_integral_matches_quadrature():
    for ramp, T in ((0.5, 0.3), (0.5, 1.0), (1.0, 1.0), (2.0, 3.0)):
        ref = quad(lambda s: smooth_plane(s, ramp=ramp, depth=DEPTH), 0.0, T,
                   epsabs=1e-16, epsrel=1e-14, limit=400)[0]
        assert plane_integral(ramp, DEPTH, T) == pytest.approx(ref, rel=1e-13)


def test_poly_cos_moment_matches_quadrature():
    ramp = 0.7
    coeffs = (0.0, -60.0 * DEPTH / ramp**3, 180.0 * DEPTH / ramp**4,
              -120.0 * DEPTH / ramp**5)
    poly = lambda s: sum(c * s**n for n, c in enumerate(coeffs))
    for omega in (34.0, 326.0, 1200.0):
        ref = quad(lambda s: poly(s) * math.cos(omega * (ramp - s)), 0.0, ramp,
                   epsabs=1e-20, epsrel=1e-13, limit=800)[0]
        assert poly_cos_moment(coeffs, ramp, omega) == pytest.approx(ref, rel=1e-10)


# --------------------------------------------------------------------------- #
# the model itself: linear operator reproduced from the engine's own arrays
# --------------------------------------------------------------------------- #
def test_reference_model_reproduces_engine_forces():
    net = MovingPlaneNetwork(8, 2)
    m = assemble_linear_model(net)
    rng = np.random.default_rng(3)
    u = rng.normal(scale=1e-5, size=len(m.free))
    plane = 1.0 - 4e-4
    net.u.zero_()
    flat = np.zeros(2 * len(net.x))
    flat[m.free] = u
    net.u.copy_(__import__("torch").tensor(flat.reshape(-1, 2)))
    _, ext, _, u_pen = net.forces(plane=plane)
    engine_force = -float(ext.sum())
    assert m.plane_force(u, plane) == pytest.approx(engine_force, abs=1e-9)
    gap = (net.x[net.right, 0] + net.u[net.right, 0] - plane).clamp_min(0)
    ref_pen = float(0.5 * (net.penalty * net.areas * gap.square()).sum())
    assert ref_pen == pytest.approx(u_pen, rel=1e-13)


def test_contact_stays_active_over_the_whole_ramp():
    """The closed form assumes a frozen active set; verify, do not assume it."""
    run = run_engine(8, 2, 0.25, DEPTH, 0.25, substeps=1)
    assert run["gap_min"] > 0.0


# --------------------------------------------------------------------------- #
# closed form vs an independent ODE integration
# --------------------------------------------------------------------------- #
def test_closed_form_matches_ode_integration(model):
    ramp = 1.0
    nf = len(model.free)

    def rhs(t, y):
        u, v = y[:nf], y[nf:]
        p = plane_derivatives(t, ramp, DEPTH)[0]
        return np.concatenate([v, (-model.A @ u + model.q * p - model.q * model.xf) / model.mass])

    for T in (0.05, 0.5):
        sol = solve_ivp(rhs, (0.0, T), np.zeros(2 * nf), method="DOP853",
                        rtol=1e-12, atol=1e-15, dense_output=True)
        ts = np.linspace(0.0, T, 6001)
        Y = sol.sol(ts)
        planes = np.array([plane_derivatives(t, ramp, DEPTH)[0] for t in ts])
        forces = np.array([model.q @ (model.xf + Y[:nf, i] - planes[i]) for i in range(len(ts))])
        ode = float(np.trapezoid(forces, ts))
        closed = model.exact_impulse(ramp, DEPTH, T)
        assert abs(closed - ode) <= 1e-5 * abs(ode)


def test_modal_force_history_matches_ode_history(model):
    ramp = 0.5
    nf = len(model.free)

    def rhs(t, y):
        u, v = y[:nf], y[nf:]
        p = plane_derivatives(t, ramp, DEPTH)[0]
        return np.concatenate([v, (-model.A @ u + model.q * p - model.q * model.xf) / model.mass])

    sol = solve_ivp(rhs, (0.0, ramp), np.zeros(2 * nf), method="DOP853",
                    rtol=1e-12, atol=1e-15, dense_output=True)
    ts = np.linspace(0.0, ramp, 401)
    Y = sol.sol(ts)
    planes = np.array([plane_derivatives(t, ramp, DEPTH)[0] for t in ts])
    ode = np.array([model.q @ (model.xf + Y[:nf, i] - planes[i]) for i in range(len(ts))])
    hist = model.exact_history(ramp, DEPTH, ts)
    assert np.max(np.abs(hist - ode)) <= 1e-8 * np.max(np.abs(ode))


# --------------------------------------------------------------------------- #
# the engine ledger
# --------------------------------------------------------------------------- #
def test_engine_impulse_matches_closed_form(model):
    ramp = 0.5
    run = run_engine(8, 2, ramp, DEPTH, ramp, substeps=1)
    exact = model.exact_impulse(ramp, DEPTH, ramp)
    assert abs(run["tool_impulse"] - exact) <= 1e-4 * abs(exact)


def test_engine_impulse_obeys_the_momentum_identity(model):
    """Ledger vs the analytic identity that uses only the engine's end velocity."""
    ramp = 0.5
    run = run_engine(8, 2, ramp, DEPTH, ramp, substeps=1)
    v = run["net"].v.detach().cpu().numpy().reshape(-1)[model.free]
    ident = momentum_identity_impulse(run["net"], ramp, DEPTH, ramp, v)
    exact = model.exact_impulse(ramp, DEPTH, ramp)
    assert abs(ident - run["tool_impulse"]) <= 1e-6 * abs(ident)
    assert abs(ident - exact) <= 1e-4 * abs(exact)


def test_impulse_converges_second_order_in_time_step(model):
    ramp = 0.5
    exact = model.exact_impulse(ramp, DEPTH, ramp)
    errs = []
    for sub in (1, 2):
        run = run_engine(8, 2, ramp, DEPTH, ramp, substeps=sub)
        errs.append(abs(run["tool_impulse"] - exact))
    assert errs[1] < errs[0]
    assert 1.2 < math.log2(errs[0] / errs[1]) < 6.0


def test_quasi_static_limit_is_approached_from_below(model):
    """Long ramp: the dynamic impulse tends to the rate-free integral."""
    short = model.exact_impulse(0.5, DEPTH, 0.5) - quasi_static_tool_impulse(
        MovingPlaneNetwork(8, 2), 0.5, DEPTH, 0.5)
    long = model.exact_impulse(8.0, DEPTH, 8.0) - quasi_static_tool_impulse(
        MovingPlaneNetwork(8, 2), 8.0, DEPTH, 8.0)
    assert abs(short) > abs(long)


def test_zero_depth_gives_zero_impulse():
    run = run_engine(8, 2, 0.5, 0.0, 0.5, substeps=1)
    assert abs(run["tool_impulse"]) < 1e-14
    assert abs(assemble_linear_model(MovingPlaneNetwork(8, 2)).exact_impulse(0.5, 0.0, 0.5)) < 1e-14