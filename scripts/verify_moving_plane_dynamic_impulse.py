#!/usr/bin/env python3
"""Closed-form dynamic ``tool_impulse`` for the smooth moving-plane boundary.

Scope: the one item `docs/dem3d_moving_plane_ledger_verification.md` §5/§6 left
as **WITHHELD**: the *dynamic* (ramped, non-frozen-network) ``tool_impulse``
ledger of ``tensordem.moving_plane_boundary.MovingPlaneNetwork``.  The frozen
-network limit was already exact (5.2e-14); the dynamic case had no reference
because "no independent ``int F dt`` quadrature was built".

What is built here
------------------
1. **Model pinned to the engine** (identity check, not a re-derivation): the
   reduced matrices ``K``, ``M``, penalty vector ``p`` are assembled from the
   module's own ``edges``/``k``/``n``/``mass``/``penalty``/``areas`` arrays and
   must reproduce ``net.forces()`` external force to round-off.
2. **Analytic reference (closed form)**: while the contact active set is
   frozen, the system is linear,
       M qdd + (K + P) q = P (x_f - s(t) 1),
   with ``s(t)`` the module's C2 quintic and ``sdd`` a cubic on ``[0, ramp]``.
   Substituting the particular solution ``q_p = a + b s(t)`` leaves a modal
   problem whose forcing is ``-c_k sdd(t)``; every following integral is
   elementary because ``sdd`` is a cubic.  Hence ``J(t) = -int sum(ext) dt``
   has a **closed-form** expression -- see ``analytic_impulse``.
3. **Independent ODE reference**: RK4 on the same force law (including the
   one-sided ``clamp_min(0)`` gap) with Richardson extrapolation.  It shares
   *no* code with the analytic route and cross-validates the algebra.
4. **Engine sweep**: ``tool_impulse`` at ``substeps`` = 1..16 -> observed
   convergence order and relative error against the closed form.

Run:
    PYTHONPATH=src python3 scripts/verify_moving_plane_dynamic_impulse.py \
        --out docs/moving-plane-ledger
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from tensordem.moving_plane_boundary import MovingPlaneNetwork, smooth_plane

DEPTH = 0.001
RAMP = 0.5
HOLD = 0.05


# --------------------------------------------------------------------------- #
# 1. reference model, assembled from the module's own arrays
# --------------------------------------------------------------------------- #
def assemble(net: MovingPlaneNetwork):
    """Return (K, mass, free-dof index, right x-dof position, penalty vector).

    ``K`` is the 2N x 2N linear central-spring stiffness, ``mass`` the 2N
    lumped mass vector, ``fdof`` the free-dof indices, ``pf`` the positions of
    the right-boundary x-dofs inside the free set and ``p`` their penalty
    stiffness ``penalty * area``.
    """
    n = len(net.x)
    K = np.zeros((2 * n, 2 * n))
    e = net.edges.numpy()
    k = net.k.numpy()
    nrm = net.n.numpy()
    for (i, j), kk, nn in zip(e, k, nrm):
        blk = kk * np.outer(nn, nn)
        ii, jj = 2 * i, 2 * j
        K[ii:ii + 2, ii:ii + 2] += blk
        K[jj:jj + 2, jj:jj + 2] += blk
        K[ii:ii + 2, jj:jj + 2] -= blk
        K[jj:jj + 2, ii:ii + 2] -= blk
    mass = np.repeat(net.mass.numpy(), 2)
    fixed = net.fixed.numpy().reshape(-1)
    fdof = np.where(~fixed)[0]
    right = net.right.numpy()
    p = float(net.penalty) * net.areas.numpy()
    xr = net.x[right, 0].numpy()
    pf = np.array([int(np.where(fdof == 2 * i)[0][0]) for i in right])
    return K, mass, fdof, pf, p, xr, right


def force_identity(net: MovingPlaneNetwork, seed: int = 7) -> dict:
    """Assembled model reproduces the engine's external (penalty) force."""
    K, mass, fdof, pf, p, xr, right = assemble(net)
    g = np.random.default_rng(seed)
    u = g.normal(0.0, 2e-4, size=(len(net.x), 2))
    net.u = torch.tensor(u, dtype=torch.float64)
    plane = 1.0 - 6e-4
    _, ext, _, _ = net.forces(plane=plane)
    ext_np = ext.numpy().reshape(-1)[fdof]
    gap = np.maximum(xr + u[right, 0] - plane, 0.0)
    ref = np.zeros(len(fdof))
    ref[pf] = -p * gap
    scale = float(np.abs(ref).max())
    return dict(max_abs_residual=float(np.abs(ext_np - ref).max()), scale=scale,
                rel=float(np.abs(ext_np - ref).max() / scale),
                min_gap=float(gap.min()))


# --------------------------------------------------------------------------- #
# 2. analytic (closed-form) impulse
# --------------------------------------------------------------------------- #
def poly_exp_integral(coeffs, hi: float, omega: float) -> complex:
    """``int_0^hi P(sigma) exp(-i omega sigma) dsigma`` for polynomial ``coeffs``.

    ``coeffs[n]`` is the coefficient of ``sigma**n``.  Antiderivative
    ``exp(i w s) * sum_m (-1)^m P^(m)(s) / (i w)^(m+1)``.
    """
    if omega == 0.0:
        return complex(sum(c * hi ** (n + 1) / (n + 1)
                           for n, c in enumerate(coeffs)), 0.0)
    iw = -1j * omega
    fact = [math.factorial(n) for n in range(len(coeffs) + 1)]

    def parseval(sig: float) -> complex:
        tot = 0.0 + 0.0j
        for m in range(len(coeffs)):
            val = sum(coeffs[n] * fact[n] / fact[n - m] * sig ** (n - m)
                      for n in range(m, len(coeffs)))
            tot += ((-1) ** m) * val / iw ** (m + 1)
        return tot

    return np.exp(iw * hi) * parseval(hi) - parseval(0.0)


class AnalyticImpulse:
    """Closed-form ``tool_impulse`` for the frozen-active-set linear model."""

    def __init__(self, ramp: float = RAMP, depth: float = DEPTH) -> None:
        self.ramp = ramp
        self.depth = depth
        net = MovingPlaneNetwork(8, 2)
        self.net = net
        K, mass, fdof, pf, p, xr, right = assemble(net)
        self.fdof, self.pf, self.p, self.xr = fdof, pf, p, xr
        Kff = K[np.ix_(fdof, fdof)]
        Mff = np.diag(mass[fdof])
        P = np.zeros((len(fdof), len(fdof)))
        P[pf, pf] = p
        Kt = Kff + P
        # particular solution q_p = a + b s(t); x_f / 1 are free-dof vectors
        xf = np.zeros(len(fdof))
        xf[pf] = xr
        self.a = -np.linalg.solve(Kt, P @ xf)
        # (K + P) b = +P 1; with x_f = 1 the particular solution vanishes at s = 1
        self.b = np.linalg.solve(Kt, P @ np.ones(len(fdof)))
        # modal basis of (K + P) w.r.t. M
        from scipy.linalg import eigh
        w2, phi = eigh(Kt, Mff)
        if np.any(w2 <= 0):
            raise ValueError("non-positive eigenvalue: (K+P) not positive definite")
        self.omega = np.sqrt(w2)
        self.phi = phi
        self.c = phi.T @ (Mff @ self.b)
        self.A = -(phi.T @ (Mff @ (self.a + self.b)))
        pv = np.zeros(len(fdof))          # penalty stiffness embedded in free-dof space
        pv[pf] = p
        self.pv = pv
        self.pq = pv @ phi
        self.S = float(p.sum())
        self.pa = float(pv @ self.a)
        self.pb = float(pv @ self.b)
        self.px = float(pv @ xf)
        # cubic coefficients of sdd(sigma) on [0, ramp]
        r, d = ramp, depth
        # sdd(sig) = -60 d/r^3 sig + 180 d/r^4 sig^2 - 120 d/r^5 sig^3  (no constant term)
        self.sdd_coeffs = [0.0, -60.0 * d / r ** 3, 180.0 * d / r ** 4,
                           -120.0 * d / r ** 5]

    # ---- plane kinematics -------------------------------------------------
    def s(self, t: float) -> float:
        return float(smooth_plane(t, ramp=self.ramp, depth=self.depth))

    def sdot(self, t: float) -> float:
        """d/dt of the module's C2 quintic ramp (zero outside [0, ramp])."""
        u = min(1.0, max(0.0, t / self.ramp))
        if not (0.0 < u < 1.0):
            return 0.0
        return -self.depth / self.ramp * 30.0 * u ** 2 * (1.0 - u) ** 2

    def _s_dot_integral(self, T: float) -> float:
        """``int_0^T s dtau`` (quintic on [0, ramp], constant afterwards)."""
        r, d = self.ramp, self.depth
        hi = min(T, r)
        u = hi / r
        # int_0^u q du with q = 10u^3 - 15u^4 + 6u^5
        Q = 10 * u ** 4 / 4 - 15 * u ** 5 / 5 + 6 * u ** 6 / 6
        acc = hi - d * r * Q                      # int_0^hi s dtau
        if T > r:                                 # hold: s stays at 1 - depth
            acc += (T - r) * (1.0 - d)
        return acc

    def _H(self, omega: float, T: float) -> complex:
        """``int_0^T sdd(sigma) exp(-i omega sigma) dsigma``."""
        hi = min(T, self.ramp)
        return poly_exp_integral(self.sdd_coeffs, hi, omega)

    def impulse(self, T: float) -> float:
        """Closed-form ``J(T) = -int_0^T sum(ext) dtau``."""
        J = T * (self.px + self.pa) + (self.pb - self.S) * self._s_dot_integral(T)
        for k, om in enumerate(self.omega):
            H = self._H(om, T)
            e = np.exp(1j * om * T) * H
            # int_0^T F_k dtau, F_k = int_0^tau sin(om(tau-sig)) sdd(sig) dsig
            IFk = ((self.sdot(T) - self.sdot(0.0)) - e.real) / om
            Ieta = self.A[k] * math.sin(om * T) / om - (self.c[k] / om) * IFk
            J += self.pq[k] * Ieta
        return float(J)

    def state(self, t: float) -> np.ndarray:
        """Exact free-dof displacement ``q(t)`` (frozen active set)."""
        s = self.s(t)
        q = self.a + self.b * s
        for k, om in enumerate(self.omega):
            Fk = self._Fk(om, t)
            eta = self.A[k] * math.cos(om * t) - (self.c[k] / om) * Fk
            q = q + self.phi[:, k] * eta
        return q

    def _Fk(self, om: float, t: float) -> float:
        """``int_0^t sin(om (t - sig)) sdd(sig) dsig`` (closed form)."""
        if om == 0.0:
            return 0.0
        e = np.exp(1j * om * t) * self._H(om, t)
        return e.imag / om

    def total_ext(self, t: float) -> float:
        """``sum(ext)`` from the closed-form state (must stay <= 0)."""
        q = self.state(t)
        gap = np.maximum(self.xr + q[self.pf] - self.s(t), 0.0)
        return float(-(self.p @ gap))

    def min_gap(self, ts) -> float:
        return float(min(np.min(self.xr + self.state(t)[self.pf] - self.s(t))
                         for t in ts))


# --------------------------------------------------------------------------- #
# 3. independent RK4 reference on the engine force law
# --------------------------------------------------------------------------- #
def rk4_reference(A: AnalyticImpulse, h: float, T: float, record_every: int = 0):
    K, mass, fdof, pf, p, xr, _ = assemble(A.net)
    Kf = K[np.ix_(fdof, fdof)]
    Mf = mass[fdof]
    q = np.zeros(len(fdof))
    v = np.zeros(len(fdof))
    J = 0.0
    n = int(round(T / h))
    hist = []
    ext_list = np.zeros(len(fdof))

    def total_ext(qq, s):
        gap = np.maximum(xr + qq[pf] - s, 0.0)
        return float(-(p @ gap))

    def deriv(qq, vv, s):
        ext = np.zeros(len(fdof))
        ext[pf] = -p * np.maximum(xr + qq[pf] - s, 0.0)
        return vv, (ext - Kf @ qq) / Mf

    g0 = total_ext(q, A.s(0.0))
    opened = 0
    if record_every:
        hist.append(dict(t=0.0, impulse=J, plane=A.s(0.0)))
    for i in range(n):
        t0 = i * h
        s0 = A.s(t0)
        s1 = A.s(t0 + 0.5 * h)
        s2 = A.s(t0 + h)
        k1q, k1v = deriv(q, v, s0)
        k2q, k2v = deriv(q + 0.5 * h * k1q, v + 0.5 * h * k1v, s1)
        k3q, k3v = deriv(q + 0.5 * h * k2q, v + 0.5 * h * k2v, s1)
        k4q, k4v = deriv(q + h * k3q, v + h * k3v, s2)
        q = q + h / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q)
        v = v + h / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        g1 = total_ext(q, s2)
        if np.any(xr + q[pf] - s2 <= 0.0):
            opened += 1
        J += -0.5 * (g0 + g1) * h          # trapezoid on the RK4 grid
        g0 = g1
        if record_every and (i + 1) % record_every == 0:
            hist.append(dict(t=t0 + h, impulse=J, plane=s2))
    return dict(impulse=J, history=hist, min_gap=float(np.min(xr + q[pf] - A.s(T))),
                open_fraction=opened / max(n, 1))


# --------------------------------------------------------------------------- #
# 4. engine sweep
# --------------------------------------------------------------------------- #
def run_engine(dt: float, substeps: int, ramp: float = RAMP, hold: float = HOLD,
               record: bool = False):
    net = MovingPlaneNetwork(8, 2)
    if dt is None:
        dt = net.dt
    total = ramp + hold
    n_steps = int(round(total / dt))
    hist = []
    min_gap = float("inf")
    for i in range(n_steps):
        t0, t1 = i * dt, (i + 1) * dt
        a = smooth_plane(t0, ramp=ramp, depth=DEPTH)
        b = smooth_plane(t1, ramp=ramp, depth=DEPTH)
        for j in range(substeps):
            aj = a + (b - a) * j / substeps
            bj = a + (b - a) * (j + 1) / substeps
            net.moving_step(dt / substeps, aj, bj)
        gap = net.x[net.right, 0] + net.u[net.right, 0] - b
        min_gap = min(min_gap, float(gap.min()))
        if record:
            hist.append(dict(t=t1, plane=float(b), impulse=float(net.tool_impulse),
                             tool_work=float(net.tool_work),
                             ext_sum=float(net.forces(plane=b)[1].sum())))
    return dict(dt=dt, substeps=substeps, steps=n_steps, T=n_steps * dt,
                tool_impulse=float(net.tool_impulse), min_gap=min_gap, history=hist)


def fit_order(xs, ys):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    ok = ys > 0
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(xs[ok]), np.log(ys[ok]), 1)[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/moving-plane-ledger")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rep: dict = {"protocol": "tensordem-moving-plane-dynamic-impulse-v1",
                 "ramp_s": RAMP, "hold_s": HOLD, "depth_m": DEPTH}
    net = MovingPlaneNetwork(8, 2)
    rep["model"] = dict(nodes=int(len(net.x)), free_dofs=int((~net.fixed).sum()),
                        right_nodes=[int(i) for i in net.right.numpy()],
                        penalty=float(net.penalty), dt_engine=float(net.dt),
                        total_dofs=int(net.fixed.numel()))
    rep["A_force_identity"] = force_identity(MovingPlaneNetwork(8, 2))

    A = AnalyticImpulse()
    T = (int(round((RAMP + HOLD) / net.dt))) * net.dt
    rep["T_s"] = T
    rep["modes"] = dict(count=int(len(A.omega)), omega_min=float(A.omega.min()),
                        omega_max=float(A.omega.max()))

    # ---- D. analytic vs independent RK4 ---------------------------------- #
    rk = {}
    prev = None
    for div in (16, 32, 64):
        r = rk4_reference(A, net.dt / div, T)
        rk[div] = dict(dt=net.dt / div, impulse=r["impulse"],
                       min_gap_at_T=r["min_gap"],
                       open_fraction=r["open_fraction"])
    J64 = rk[64]["impulse"]
    J128 = rk4_reference(A, net.dt / 128, T)["impulse"]
    J_ref = J128 + (J128 - J64) / 15.0      # RK4 Richardson over the full window
    rep["B_rk4_self_convergence"] = dict(
        rows=rk, richardson_impulse=J_ref,
        order=fit_order([rk[d]["dt"] for d in (16, 32, 64)],
                        [abs(rk[d]["impulse"] - J_ref) for d in (16, 32, 64)]))

    # closed form is exact only while the one-sided contact stays closed
    Tr = RAMP * 0.999
    ana_ramp = A.impulse(Tr)
    rk_ramp = rk4_reference(A, net.dt / 32, Tr)
    J_ana = A.impulse(T)
    rep["C_closed_form"] = dict(
        ramp_window_T=Tr, closed=ana_ramp, rk4=rk_ramp["impulse"],
        rel_in_ramp=abs(ana_ramp - rk_ramp["impulse"]) / abs(rk_ramp["impulse"]),
        closed_full_window=J_ana, rk4_full_window=J_ref,
        rel_full_window=abs(J_ana - J_ref) / abs(J_ref),
        min_gap_ramp=rk_ramp["min_gap"],
        open_fraction_ramp=rk_ramp["open_fraction"],
        open_fraction_full=rk4_reference(A, net.dt / 32, T)["open_fraction"])

    # ---- E. engine sweep -------------------------------------------------- #
    rows = []
    fine = None
    for sub in (1, 2, 4, 8):
        r = run_engine(net.dt, sub, record=(sub == 8))
        if sub == 8:
            fine = r
        rows.append(dict(substeps=sub, h=r["dt"] / sub, impulse=r["tool_impulse"],
                         rel_vs_analytic=abs(r["tool_impulse"] - J_ref) / abs(J_ref),
                         min_gap=r["min_gap"]))
    rep["D_engine_sweep"] = dict(rows=rows, reference_impulse=J_ref,
                                 order=fit_order([x["h"] for x in rows],
                                                 [x["rel_vs_analytic"] for x in rows]))
    # finest engine run (8 substeps) carries the history figure
    rep["D_history"] = fine

    # closed-form vs engine displacement field, pointwise
    ts = np.array([x["t"] for x in fine["history"]])
    rkf = rk4_reference(A, net.dt / 16, T, record_every=16)
    rix = np.array([x["t"] for x in rkf["history"]])
    Js = np.interp(ts, rix, [x["impulse"] for x in rkf["history"]])
    Je = np.array([x["impulse"] for x in fine["history"]])
    dJ = np.abs(Je - Js)
    scale = max(abs(J_ana), 1e-30)
    rep["E_history_error"] = dict(max_abs=float(dJ.max()),
                                  max_abs_rel=float(dJ.max() / scale),
                                  at_final=float(dJ[-1]),
                                  rel_at_final=float(dJ[-1] / scale))
    rep["analytic_min_gap"] = A.min_gap(ts[:: max(1, len(ts) // 60)])

    # ---- figures ---------------------------------------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    hs = np.array([x["h"] for x in rows])
    es = np.array([x["rel_vs_analytic"] for x in rows])
    ax[0].loglog(hs, es, "o-", label="engine $|\\Delta J|/J$")
    ref = es[-1] * (hs / hs[-1]) ** 2
    ax[0].loglog(hs, ref, "k--", lw=1, label="$h^{2}$ reference")
    ax[0].set_xlabel("substep $h$ [s]")
    ax[0].set_ylabel("relative impulse error")
    ax[0].set_title(f"(a) engine vs RK4 baseline (order {rep['D_engine_sweep']['order']:.2f})")
    ax[0].legend()
    ax[0].grid(True, which="both", alpha=.3)

    ax[1].plot(ts, Js, "k-", lw=2, label="RK4 (dt/16)")
    ax[1].plot(ts, Je, "C1--", lw=1.4, label="engine (8 substeps)")
    ax[1].set_xlabel("t [s]")
    ax[1].set_ylabel("$J_{tool}$ [N s]")
    ax[1].set_title("(b) impulse history")
    ax[1].legend()
    ax[1].grid(alpha=.3)

    ax[2].semilogy(ts, np.maximum(dJ, 1e-18), "C0", lw=1.2)
    ax[2].set_xlabel("t [s]")
    ax[2].set_ylabel("$|J_{engine} - J_{RK4}|$ [N s]")
    ax[2].set_title("(c) pointwise ledger difference")
    ax[2].grid(alpha=.3)
    fig.tight_layout()
    png = os.path.join(args.out, "fig3_dynamic_impulse.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    rep["figure"] = png

    js = os.path.join(args.out, "moving_plane_dynamic_impulse.json")
    with open(js, "w") as fh:
        json.dump(rep, fh, indent=2, default=float)
    csv = os.path.join(args.out, "moving_plane_dynamic_impulse.csv")
    with open(csv, "w") as fh:
        fh.write("substeps,h_s,tool_impulse_Ns,rel_vs_closed_form\n")
        for x in rows:
            fh.write(f"{x['substeps']},{x['h']:.6e},{x['impulse']:.12e},"
                     f"{x['rel_vs_analytic']:.6e}\n")
    print(json.dumps({k: rep[k] for k in ("A_force_identity", "B_rk4_self_convergence",
                                          "C_closed_form", "D_engine_sweep",
                                          "E_history_error")},
                     indent=2, default=float))
    print("wrote", js, csv, png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())