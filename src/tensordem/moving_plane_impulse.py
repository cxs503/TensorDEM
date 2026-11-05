"""Closed-form reference for the *dynamic* ``tool_impulse`` of the moving-plane boundary.

Context
-------
:mod:`tensordem.moving_plane_boundary` exposes two ledgers, ``tool_work`` and
``tool_impulse``.  ``docs/dem3d_moving_plane_ledger_verification.md`` closed
``tool_work`` against a closed-form static solution (``W = 0.5 F delta``) but had
to record the **dynamic** ``tool_impulse`` as ``WITHHELD``: the only reference was
the frozen-network limit, and no independent ``integral F dt`` comparison existed.
This module supplies that reference.

Why a closed form exists
------------------------
``BoundaryNetwork`` is a *linear central-spring* network with **frozen initial
normals** and its moving plane is a quadratic (linear-in-gap) penalty contact:

    f_i = -(K u)_i - Q (x_i + u_i - p(t))     for the reference (right) nodes

Every node's y degree of freedom is fixed and the left column is clamped, so the
free degrees of freedom are the x components of the interior/right nodes.  The
contact is active for all ``t > 0`` in the regimes used here (verified, not
assumed: :func:`linear_model` / ``scripts/verify_moving_plane_dynamic_impulse.py``
monitor the gap), so the active set never changes and the free-DOF motion obeys

    M u'' + A u = b(t),   A = K_ff + Q_ff,   b(t) = Q_ff (p(t) e - x_f)

with ``M`` diagonal (lumped tributary masses), ``A`` symmetric positive definite
(the springs anchor the network to the clamped left column, so there is no rigid
mode) and ``p(t)`` the prescribed quintic (C2) plane motion of ``smooth_plane``.
Consequently ``p''(t)`` is a *cubic polynomial* on ``0 <= t <= ramp`` and zero
afterwards, which makes the impulse integrable in closed form.

The reference
-------------
Split ``u = u_s(p(t)) + z``, where ``u_s(p) = A^-1 (p q - r)`` is the instant
static solution (``q = Q_ff e``, ``r = Q_ff x_f``).  Then

    M z'' + A z = -M A^-1 q p''(t),   z(0) = z'(0) = 0

and, with ``B = M^-1/2 A M^-1/2 = V diag(w_k^2) V^T`` and ``c = M^-1/2 q``,

    z(t) = M^-1/2 V Z(t),   Z_k(t) = -(V^T c)_k / w_k * int_0^t sin(w_k (t-s)) p''(s) ds

The reported impulse is ``F(t) = q^T (x_f + u_f - e p(t))`` (the penalty load the
plane applies to the network, summed over the reference nodes), hence

    J(T) = int_0^T F dt = q^T x_f T + q^T int_0^T u_f dt - (sum q) int_0^T p dt

    q^T int_0^T u_f dt = q^T A^-1 (q int p - r T) + q^T int_0^T z dt

    int_0^T Z_k dt = -(V^T c)_k / w_k^2 * [ int_0^T p'' ds - int_0^T p''(s) cos(w_k (T-s)) ds ]

Both remaining integrals are elementary: the first is ``p'(T) - p'(0)`` and the
second is a polynomial-times-cosine moment, evaluated exactly in
:func:`poly_cos_moment`.  There is no fitting, no time stepping and no reuse of
the engine's quadrature, so the comparison is independent.

Units are SI throughout (N, m, s, kg); results carry ``N s``.
"""
from __future__ import annotations

import math

import numpy as np
import torch

__all__ = [
    "plane_derivatives",
    "plane_integral",
    "assemble_linear_model",
    "SceneModel",
    "exact_tool_impulse",
    "exact_tool_impulse_history",
    "quasi_static_tool_impulse",
    "momentum_identity_impulse",
    "run_engine",
]

#: coefficients of the quintic smoothstep ``Q(s) = s^3 (10 - 15 s + 6 s^2)``,
#: ``Q(0) = 0``, ``Q(1) = 1``, ``Q'(0) = Q'(1) = Q''(0) = Q''(1) = 0``.
_QUINTIC = (0.0, 0.0, 0.0, 10.0, -15.0, 6.0)


def _clip_unit(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def plane_derivatives(time: float, ramp: float, depth: float) -> tuple[float, float, float]:
    """``p(t)``, ``p'(t)``, ``p''(t)`` for ``smooth_plane`` at ``time``.

    ``smooth_plane(t, ramp, depth) = 1 - depth * Q(clip(t/ramp, 0, 1))``; the
    derivatives below are the exact analytic ones (not finite differences).
    Valid for any real ``time >= 0``; the hold phase gives ``p' = p'' = 0``.
    """
    if not (math.isfinite(time) and math.isfinite(ramp) and math.isfinite(depth)):
        raise ValueError("time, ramp and depth must be finite")
    if ramp <= 0.0:
        raise ValueError("ramp must be positive")
    s = _clip_unit(time / ramp)
    inside = 0.0 <= time <= ramp
    p = 1.0 - depth * (s**3 * (10.0 + s * (-15.0 + 6.0 * s)))
    if not inside:
        return p, 0.0, 0.0
    dp = -depth * (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / ramp
    ddp = -depth * (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / (ramp * ramp)
    return p, dp, ddp


def plane_integral(ramp: float, depth: float, horizon: float) -> float:
    """``int_0^T p(t) dt`` in closed form (``int_0^1 Q ds = 1/2``)."""
    if horizon <= 0.0:
        return 0.0
    if horizon <= ramp:
        s = horizon / ramp
        return horizon - depth * ramp * (2.5 * s**4 - 3.0 * s**5 + s**6)
    return ramp - 0.5 * depth * ramp + (horizon - ramp) * (1.0 - depth)


def poly_cos_moment(coeffs, horizon: float, omega: float) -> float:
    """``int_0^H (sum_n c_n s^n) cos(omega (T - s)) ds`` exactly (``omega > 0``)."""
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    # int_0^H s^n exp(-i omega s) ds, closed form of the exponential-moment recursion.
    def moment(n: int) -> complex:
        a = -1j * omega
        total = 0j
        for k in range(n + 1):
            coef = (-1) ** k * math.factorial(n) / math.factorial(n - k)
            total += coef * horizon ** (n - k) / a ** (k + 1)
        return total * complex(math.cos(-omega * horizon), math.sin(-omega * horizon)) - (
            (-1) ** n * math.factorial(n) / a ** (n + 1)
        )

    phase = complex(math.cos(omega * horizon), math.sin(omega * horizon))
    value = 0.0
    for n, c in enumerate(coeffs):
        if c:
            value += c * (phase * moment(n)).real
    return value


class SceneModel:
    """Linear free-DOF model of one ``MovingPlaneNetwork`` configuration.

    ``A = K_ff + Q_ff`` (symmetric positive definite), ``M`` lumped-diagonal,
    ``q`` the penalty load pattern, ``x_f`` the initial free-DOF coordinates.
    """

    def __init__(self, free: np.ndarray, mass: np.ndarray, A: np.ndarray,
                 q: np.ndarray, xf: np.ndarray, right_dof: np.ndarray):
        self.free = free
        self.mass = mass
        self.A = A
        self.q = q
        self.xf = xf
        self.right_dof = right_dof
        self.P_total = float(q.sum())
        sqrt_m = np.sqrt(mass)
        self.sqrt_m = sqrt_m
        self.B = (A / sqrt_m[:, None]) / sqrt_m[None, :]
        lam, V = np.linalg.eigh(self.B)
        if lam.min() <= 0.0:
            raise ValueError("free-DOF operator is not positive definite (rigid mode)")
        self.omegas = np.sqrt(lam)
        self.V = V
        # forcing of the mass-weighted system: w'' + B w = -(M^1/2 A^-1 q) p''
        self.c = sqrt_m * np.linalg.solve(A, q)

    # ------------------------------------------------------------------ #
    def static_displacement(self, plane: float) -> np.ndarray:
        """``u_s(p) = A^-1 (p q - Q x_f)``."""
        return np.linalg.solve(self.A, plane * self.q - self.q * self.xf)

    def plane_force(self, u_free: np.ndarray, plane: float) -> float:
        """``F = q^T (x_f + u_f - e p)`` = penalty load the plane applies."""
        return float(self.q @ (self.xf + u_free - plane))

    def gaps(self, u_free: np.ndarray, plane: float) -> np.ndarray:
        """Penetration at the reference nodes (must stay ``>= 0`` for the model)."""
        idx = np.searchsorted(self.free, self.right_dof)
        return self.xf[idx] + u_free[idx] - plane

    # ------------------------------------------------------------------ #
    def exact_impulse(self, ramp: float, depth: float, horizon: float) -> float:
        """Closed-form ``int_0^T F dt`` (dynamic, elastic network)."""
        if horizon <= 0.0:
            return 0.0
        int_p = plane_integral(ramp, depth, horizon)
        # static part
        u_int = np.linalg.solve(self.A, self.q * int_p - (self.q * self.xf) * horizon)
        # grouped to avoid cancelling O(P*L*T) terms before the subtraction
        total = float(np.dot(self.q, self.xf * horizon - int_p)) + float(self.q @ u_int)
        # dynamic (vibration) part
        if ramp > 0.0 and depth != 0.0 and horizon > 0.0:
            R = min(horizon, ramp)
            coeffs = (0.0, -60.0 * depth / ramp**3, 180.0 * depth / ramp**4,
                      -120.0 * depth / ramp**5)
            accel_integral = plane_derivatives(R, ramp, depth)[1]  # p'(R) - p'(0)
            proj = self.V.T @ self.c
            per_mode = np.empty(len(self.omegas))
            for k, omega in enumerate(self.omegas):
                cos_part = poly_cos_moment(coeffs, R, float(omega))
                per_mode[k] = -(proj[k] / omega**2) * (accel_integral - cos_part)
            z_int = (self.V @ per_mode) / self.sqrt_m
            total += self.q @ z_int
        return float(total)

    # ------------------------------------------------------------------ #
    def exact_history(self, ramp: float, depth: float, times: np.ndarray) -> np.ndarray:
        """``F(t)`` from the modal solution, vectorised over ``times``.

        Uses the closed form of the sine convolution of ``p''``, so the curve is
        analytic at every sample (no time stepping, no interpolation).
        """
        if ramp <= 0.0:
            raise ValueError("ramp must be positive")
        times = np.asarray(times, dtype=float)
        coeffs = (0.0, -60.0 * depth / ramp**3, 180.0 * depth / ramp**4,
                  -120.0 * depth / ramp**5)
        proj = self.V.T @ self.c
        plane = np.array([plane_derivatives(t, ramp, depth)[0] for t in times])
        rhs = np.outer(self.q, plane) - np.outer(self.q * self.xf, np.ones_like(plane))
        u = np.linalg.solve(self.A, rhs).T
        z = np.zeros_like(u)
        ramp_mask = times < ramp
        if depth != 0.0 and np.any(times > 0.0):
            for k, omega in enumerate(self.omegas):
                contrib = np.zeros(len(times), dtype=complex)
                if np.any(ramp_mask):
                    r = times[ramp_mask]
                    contrib[ramp_mask] = _poly_exp_moment(coeffs, r, float(omega))
                hold = ~ramp_mask & (times > 0.0)
                if np.any(hold):
                    contrib[hold] = _poly_exp_moment(coeffs, np.array([ramp]), float(omega))[0]
                phase = np.exp(1j * omega * times)
                z += (-(proj[k] / omega) * np.imag(phase * contrib))[:, None] * self.V[:, k][None, :]
        u = u + z / self.sqrt_m
        return np.array([self.q @ (self.xf + u[i] - plane[i]) for i in range(len(times))])


def _poly_exp_moment(coeffs, horizon, omega: float) -> np.ndarray:
    """``int_0^H (sum_n c_n s^n) exp(-i omega s) ds`` exactly, vectorised in ``H``."""
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    H = np.atleast_1d(np.asarray(horizon, dtype=float))
    a = -1j * omega
    out = np.zeros(H.shape, dtype=complex)
    for n, c in enumerate(coeffs):
        if not c:
            continue
        term = np.zeros(H.shape, dtype=complex)
        for k in range(n + 1):
            coef = (-1) ** k * math.factorial(n) / math.factorial(n - k)
            term += coef * H ** (n - k) / a ** (k + 1)
        term *= np.exp(a * H)
        term -= (-1) ** n * math.factorial(n) / a ** (n + 1)
        out += c * term
    return out


def assemble_linear_model(net) -> SceneModel:
    """Build the free-DOF linear model from the network's own arrays.

    ``K`` (spring stiffness, frozen normals), the penalty diagonal ``Q`` and the
    lumped masses all come from ``net`` itself, so the reference is the *same
    discrete model* the engine integrates -- only the solution method differs.
    """
    x = net.x.detach().cpu().numpy()
    fixed = net.fixed.detach().cpu().numpy()
    edges = net.edges.detach().cpu().numpy()
    k = net.k.detach().cpu().numpy()
    nvec = net.n.detach().cpu().numpy()
    mass_node = net.mass.detach().cpu().numpy()
    right = net.right.detach().cpu().numpy()
    pen = float(net.penalty) * net.areas.detach().cpu().numpy()

    dof_x = 2 * np.arange(len(x))
    free_mask = np.zeros(2 * len(x), dtype=bool)
    free_mask[dof_x[~fixed[:, 0]]] = True
    free = np.flatnonzero(free_mask)

    K = np.zeros((2 * len(x), 2 * len(x)))
    for (a, b), kk, v in zip(edges, k, nvec):
        blk = kk * np.outer(v, v)
        ia = slice(2 * a, 2 * a + 2)
        ib = slice(2 * b, 2 * b + 2)
        K[ia, ia] += blk
        K[ib, ib] += blk
        K[ia, ib] -= blk
        K[ib, ia] -= blk

    q_full = np.zeros(2 * len(x))
    x_full = x.reshape(-1)
    right_dof = dof_x[right]
    q_full[right_dof] = pen

    M_full = np.zeros(2 * len(x))
    M_full[dof_x] = mass_node
    M_full[2 * np.arange(len(x)) + 1] = mass_node

    A_ff = (K + np.diag(q_full))[np.ix_(free, free)]
    return SceneModel(free=free, mass=M_full[free], A=A_ff, q=q_full[free],
                      xf=x_full[free], right_dof=right_dof)


def exact_tool_impulse(net, ramp: float, depth: float, horizon: float) -> float:
    """Convenience wrapper: assemble then return the closed-form impulse."""
    return assemble_linear_model(net).exact_impulse(ramp, depth, horizon)


def exact_tool_impulse_history(net, ramp: float, depth: float, times) -> np.ndarray:
    model = assemble_linear_model(net)
    return model.exact_history(ramp, depth, np.asarray(times, dtype=float))


def quasi_static_tool_impulse(net, ramp: float, depth: float, horizon: float,
                              samples: int = 4001) -> float:
    """``int_0^T F_static(p(t)) dt`` -- the rate-free limit (trapezoid on ``F_s``).

    ``u_s(p) = A^-1 (p q - q*x_f)`` is affine in ``p``, so two solves give the
    whole curve; no time stepping is involved.
    """
    model = assemble_linear_model(net)
    ts = np.linspace(0.0, horizon, samples)
    planes = np.array([plane_derivatives(t, ramp, depth)[0] for t in ts])
    u1 = np.linalg.solve(model.A, model.q)
    u0 = np.linalg.solve(model.A, model.q * model.xf)
    forces = (np.outer(planes, u1) - u0[None, :]) @ model.q + model.q @ model.xf - model.P_total * planes
    return float(np.trapezoid(forces, ts))


def momentum_identity_impulse(net, ramp: float, depth: float, horizon: float,
                              u_dot_free) -> float:
    """Impulse implied by the *end velocity* alone (no integral, no modes).

    Integrating ``M u'' + A u = b`` once gives ``A int_0^T u dt = int_0^T b dt -
    M u'(T)``, hence

        J = q^T (x_f T - 1 int p) + q^T A^-1 (q int p - q*x_f T - M u'(T))

    which uses only ``u'(T)`` -- the state the engine reports -- and closed-form
    linear algebra.  Independent of both the modal expansion and any quadrature.
    """
    model = assemble_linear_model(net)
    u_dot = np.asarray(u_dot_free, dtype=float)
    if u_dot.shape != model.mass.shape:
        raise ValueError("end velocity must have one entry per free dof")
    int_p = plane_integral(ramp, depth, horizon)
    rhs = model.q * int_p - model.q * model.xf * horizon - model.mass * u_dot
    return float(np.dot(model.q, model.xf * horizon - int_p) + model.q @ np.linalg.solve(model.A, rhs))


def run_engine(nx: int, ny: int, ramp: float, depth: float, horizon: float,
               substeps: int = 1):
    """March ``MovingPlaneNetwork`` by ``dt/substeps`` and return the ledgers.

    ``substeps`` sub-divides the module's own stable ``dt`` so the same physical
    horizon can be run at successive time steps (quadrature-order study).
    """
    from .moving_plane_boundary import MovingPlaneNetwork, smooth_plane

    net = MovingPlaneNetwork(nx, ny)
    dt = net.dt / substeps
    steps = int(math.ceil(horizon / dt))
    gap_min = math.inf
    history = []
    for s in range(steps):
        t0 = s * dt
        t1 = min(t0 + dt, horizon)
        h = t1 - t0
        net.moving_step(h, smooth_plane(t0, ramp=ramp, depth=depth),
                        smooth_plane(t1, ramp=ramp, depth=depth))
        gap = (net.x[net.right, 0] + net.u[net.right, 0]
               - smooth_plane(t1, ramp=ramp, depth=depth))
        gap_min = min(gap_min, float(gap.min()))
        history.append((t1, float((net.penalty * net.areas * gap).sum())))
    return dict(dt=dt, substeps=substeps, steps=steps, tool_impulse=float(net.tool_impulse),
                tool_work=float(net.tool_work), gap_min=gap_min,
                time=np.array([h[0] for h in history]),
                force=np.array([h[1] for h in history]),
                net=net)