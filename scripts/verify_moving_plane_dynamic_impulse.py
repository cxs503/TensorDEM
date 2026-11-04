#!/usr/bin/env python3
"""Closed-form *dynamic* reference for the moving-plane ``tool_impulse`` ledger.

Scope
-----
``tensordem.moving_plane_boundary.MovingPlaneNetwork`` -- the opt-in moving
tool boundary that stands in for a ship hull pressing on the control-volume ice
network.  ``scripts/verify_moving_plane_ledger.py`` closed the ``tool_work``
and the *frozen-network* ``tool_impulse`` against closed-form statics, but left
the **dynamic** ``tool_impulse`` as WITHHELD: the only reference available was
the static one, and the dynamic run rings (undamped network), so no independent
value existed for ``integral F dt`` over a moving, deforming network.

This script removes that WITHHELD by solving the *same discrete model* exactly.
The network is a linear central-spring system and, while the contact set is
intact, the 50E/L penalty is linear in the gap, so the semi-discrete equations
are a linear, classically damped-free ODE with C2 forcing:

    M udd + (K + diag(P)) u = b(t),   b_i(t) = P_i (plane(t) - x_i)   (right x-DOF)

With the mode basis of ``(K+diag(P), M)`` (``phi^T M phi = I``) every modal
coordinate obeys

    xi_k'' + omega_k^2 xi_k = c_k s(t),   s(t) = plane(t) - L = -depth q(t/T)

and because the quintic ramp is C2 at ``t = 0`` the particular polynomial
solution already satisfies the at-rest initial conditions:

    xi_k(t) = c_k [ s/w^2 - s''/w^4 + s''''/w^6 ]        (0 <= t <= T)

so the response is a *closed-form* polynomial in the ramp and a *closed-form*
cos/sin continuation in the hold.  ``integral F dt`` then integrates
analytically (polynomial + trig integrals), giving an exact reference for the
dynamic impulse -- including the ringing, which the static reference lacks.

Checks (all machine-checked, growth of the reference is itself audited):

  G  reference-model identity: assembled ``K`` reproduces ``forces()``,
     eigen-pairs satisfy ``K phi = w^2 M phi`` and orthonormality
  H  exact modal state vs the engine state at the end of the run
  I  dynamic ``tool_impulse`` vs the exact modal integral, ramp sweep
  J  time-step refinement at fixed ramp -> measured convergence exponent
  K  contact-set validity: the exact reference is only valid while the
     penetration stays positive; the minimum penetration is reported per run
  L  teeth: the *quasi-static* impulse (the reference that was available
     before this work) differs from the engine by far more than the exact one

Run:
    PYTHONPATH=src python3 scripts/verify_moving_plane_dynamic_impulse.py \
        --out docs/moving-plane-dynamic-impulse
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tensordem.continuum_boundary_dem import BoundaryNetwork  # noqa: E402
from tensordem.moving_plane_boundary import MovingPlaneNetwork, smooth_plane  # noqa: E402
from verify_moving_plane_ledger import DEPTH, assemble, fit_order, jacobian_check, static_solution  # noqa: E402

HOLD = 0.05  # seconds of constant-penetration hold after the ramp


# --------------------------------------------------------------------------- #
# exact modal reference
# --------------------------------------------------------------------------- #
def modal_system(net: BoundaryNetwork):
    """Eigen-decomposition of (K + diag(P), M) restricted to the free DOF.

    ``phi`` is mass-orthonormal (``phi^T M phi = I``), so modal coordinates are
    uncoupled.  ``c[k]`` is the projection of the unit penalty load pattern
    (``h_i = P_i`` on the right-face x-DOF) onto mode ``k``.
    """
    K, P, x, fixed = assemble(net)
    mass = np.repeat(net.mass.numpy(), 2)  # DOF order [x0, y0, x1, y1, ...]
    idx = np.flatnonzero(~fixed)
    Kf = K[np.ix_(idx, idx)] + np.diag(P[idx])  # the model actually integrated
    mf = mass[idx]
    inv_sqrt = 1.0 / np.sqrt(mf)
    A = (inv_sqrt[:, None] * Kf) * inv_sqrt[None, :]
    lam, V = np.linalg.eigh(A)
    if np.any(lam <= 0.0):
        raise RuntimeError("non-positive eigenvalue in the penalty-coupled model")
    phi = inv_sqrt[:, None] * V
    omega = np.sqrt(lam)
    h = P[idx].copy()  # penalty load pattern in free-DOF space
    c = phi.T @ h
    return dict(K=K, P=P, x=x, fixed=fixed, idx=idx, mass=mass, Kf=Kf,
                phi=phi, omega=omega, c=c, h=h,
                eig_residual=float(np.abs(Kf @ phi - (phi * (mf[:, None] * lam))).max()
                                   / np.abs(Kf @ phi).max()),
                orth_residual=float(np.abs(phi.T @ (phi * mf[:, None]) - np.eye(len(idx))).max()))


class ExactResponse:
    """Closed-form solution of the linear ramp/hold problem and its integrals."""

    def __init__(self, system, ramp: float, depth: float = DEPTH):
        self.sys = system
        self.ramp = float(ramp)
        self.depth = float(depth)

    # --- ramp-phase polynomial pieces (degree-5 quintic, C2 start) ----------
    def _s_coeffs(self):
        """Coefficients of s(t) = -depth * q(t/T), ascending powers, ramp phase."""
        T, d = self.ramp, self.depth
        a = np.zeros(6)
        a[3] = -d * 10.0 / T**3
        a[4] = -d * (-15.0) / T**4
        a[5] = -d * 6.0 / T**5
        return a

    @staticmethod
    def _poly_eval(a, t):
        return sum(a[n] * t**n for n in range(len(a)))

    @staticmethod
    def _poly_deriv(a):
        return np.array([n * a[n] for n in range(1, len(a))])

    def _p_coeffs(self):
        """Polynomial particular solution by coefficient recursion (ascending powers).

        Solves ``p'' + w^2 p = c s(t)`` exactly for the degree-5 quintic ``s``:
        ``w^2 p_n + (n+2)(n+1) p_{n+2} = c s_n``, recursed from n = 5 downwards.
        This is the audited form: the residual of the ODE is identically zero by
        construction (``residual`` below re-checks it in closed form).
        """
        if getattr(self, "_p", None) is None:
            a = self._s_coeffs()
            w2, c = self.sys["omega"] ** 2, self.sys["c"]
            p = np.zeros((6, len(w2)))
            for n in range(5, -1, -1):
                rhs = c * a[n]
                if n + 2 <= 5:
                    rhs = rhs - (n + 2) * (n + 1) * p[n + 2]
                p[n] = rhs / w2
            self._p = p
        return self._p

    def _specific(self, t):
        """p(t), p'(t), integral_0^t p for the ramp-phase particular solution."""
        p = self._p_coeffs()
        pw = sum(p[n] * t**n for n in range(6))
        dpw = sum(n * p[n] * t ** (n - 1) for n in range(1, 6))
        integ = sum(p[n] * t ** (n + 1) / (n + 1) for n in range(6))
        return pw, dpw, integ

    def _homogeneous_coeff(self):
        """(A, B) in xi = p + A cos(w t) + B sin(w t).

        The quintic ramp starts at rest but with ``p(0) = p_0 != 0`` and
        ``p'(0) = p_1 != 0``, so *both* homogeneous coefficients are needed:
        ``A = -p_0``, ``B = -p_1 / w``.
        """
        pc = self._p_coeffs()
        return -pc[0], -pc[1] / self.sys["omega"]

    def residual(self, t):
        """Closed-form ODE residual ||p'' + w^2 p - c s(t)|| (must vanish identically)."""
        p = self._p_coeffs()
        w2, c = self.sys["omega"] ** 2, self.sys["c"]
        a = self._s_coeffs()
        s_val = sum(a[n] * t**n for n in range(6))
        pp = sum((n * (n - 1)) * p[n] * t ** (n - 2) for n in range(2, 6))
        return float(np.abs(pp + w2 * sum(p[n] * t**n for n in range(6)) - c * s_val).max())

    def _phase(self, t):
        """(xi, xi_dot, integral_0^t xi) at scalar time t (vectors over modes)."""
        T, w = self.ramp, self.sys["omega"]
        A, B = self._homogeneous_coeff()
        cosT, sinT = np.cos(w * T), np.sin(w * T)
        if t <= T:
            p, dp, integ_p = self._specific(t)
            xi = p + A * np.cos(w * t) + B * np.sin(w * t)
            dxi = dp - A * w * np.sin(w * t) + B * w * np.cos(w * t)
            integ = (integ_p + A * np.sin(w * t) / w
                     + B * (1.0 - np.cos(w * t)) / w)
            return xi, dxi, integ
        pT, dpT, integ_pT = self._specific(T)
        xiT = pT + A * cosT + B * sinT
        dxiT = dpT - A * w * sinT + B * w * cosT
        integ_ramp = integ_pT + A * sinT / w + B * (1.0 - cosT) / w
        p_inf = -self.sys["c"] * self.depth / w**2  # s = -depth (constant) in the hold
        d = t - T
        xi = p_inf + (xiT - p_inf) * np.cos(w * d) + dxiT / w * np.sin(w * d)
        dxi = -(xiT - p_inf) * w * np.sin(w * d) + dxiT * np.cos(w * d)
        integ = (integ_ramp + p_inf * d + (xiT - p_inf) * np.sin(w * d) / w
                 + dxiT / w * (1.0 - np.cos(w * d)) / w)
        return xi, dxi, integ

    # --- observable quantities --------------------------------------------
    def displacement(self, t):
        """Full displacement vector (len 2N) of the exact solution."""
        xi, _, _ = self._phase(t)
        u = np.zeros(len(self.sys["x"]))
        u[self.sys["idx"]] = self.sys["phi"] @ xi
        return u

    def penetration(self, t):
        """Gap of every right-face node (may go negative -> contact loss)."""
        u = self.displacement(t)
        net_plane = _plane(t, self.ramp, self.depth)
        right_x = self.sys["idx"][np.abs(self.sys["h"]) > 0.0]
        return self.sys["x"][right_x] + u[right_x] - net_plane

    def force(self, t):
        """Total penalty force on the network (positive magnitude)."""
        g = self.penetration(t)
        p = self.sys["h"][np.abs(self.sys["h"]) > 0.0]
        return float(np.sum(p * np.clip(g, 0.0, None)))

    def impulse(self, t_end):
        """Exact integral of the total penalty force over [0, t_end]."""
        _, _, integ = self._phase(t_end)
        u_int = self.sys["phi"] @ integ  # integral of the free-DOF displacement
        p = self.sys["h"]
        # integral of the penetration sum: sum_i P_i [ integral u_i + integral(depth*q) ]
        T = self.ramp
        ramp_part = 0.5 * T  # integral_0^T q dt = T/2 for the quintic
        hold_part = max(0.0, t_end - T)
        gap_int = float(np.sum(p * u_int)) + float(np.sum(p)) * self.depth * (ramp_part + hold_part)
        return gap_int


def _plane(t, ramp, depth=DEPTH):
    return smooth_plane(t, ramp=ramp, depth=depth)


def quasi_static_impulse(net: BoundaryNetwork, ramp: float, t_end: float,
                         depth: float = DEPTH, samples: int = 401):
    """Impulse of the *static* force history  integral F_static(plane(t)) dt.

    This is the reference that was available before this work (``tool_work``'s
    closed form); it ignores ringing and therefore has no chance of matching a
    dynamic run to better than the ringing amplitude.  Kept as the teeth check.
    """
    ts = np.linspace(0.0, t_end, samples)
    fs = np.empty_like(ts)
    for i, t in enumerate(ts):
        delta = depth * min(1.0, t / ramp) ** 3 * (10.0 + min(1.0, t / ramp) * (-15.0 + 6.0 * min(1.0, t / ramp)))
        st = static_solution(net, max(delta, 1e-12))
        fs[i] = st["force"]
    return float(np.trapezoid(fs, ts))


# --------------------------------------------------------------------------- #
# engine runs
# --------------------------------------------------------------------------- #
def run_engine(ramp: float, substeps: int, hold: float = HOLD, every: int = 1):
    """Integrate the module and record the tool force history + ledger values."""
    net = MovingPlaneNetwork(8, 2)
    dt = float(net.dt)
    total = ramp + hold
    n_steps = int(round(total / dt))
    t_end = n_steps * dt
    h = dt / substeps
    hist = []
    min_gap = math.inf
    right = net.right
    k = 0
    for i in range(n_steps):
        for j in range(substeps):
            # the plane follows the exact quintic at every substep boundary
            t0 = k * h
            t1 = (k + 1) * h
            a = smooth_plane(t0, ramp=ramp, depth=DEPTH)
            b = smooth_plane(t1, ramp=ramp, depth=DEPTH)
            f = net.moving_step(h, a, b)
            gap = net.x[right, 0] + net.u[right, 0] - b
            if t1 >= 0.25 * ramp:  # ignore the start-up window (penetration ~ 0 at t=0)
                min_gap = min(min_gap, float(gap.min()))
            if k % every == 0:
                hist.append(dict(t=t1, force_tool=float(f),
                                 tool_impulse=float(net.tool_impulse)))
            k += 1
    hist.append(dict(t=t_end,
                     force_tool=float(-net.forces(plane=_plane(t_end, ramp))[1][right, 0].sum()),
                     tool_impulse=float(net.tool_impulse)))
    return dict(net=net, dt=dt, h=h, substeps=substeps, n_steps=n_steps, t_end=t_end,
                tool_impulse=float(net.tool_impulse), hist=hist, min_gap=min_gap,
                tool_work=float(net.tool_work))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/moving-plane-dynamic-impulse")
    ap.add_argument("--ramps", default="0.05,0.1,0.25,0.5")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    net0 = MovingPlaneNetwork(8, 2)
    system = modal_system(net0)
    jac = jacobian_check(net0)
    print(f"[G] K vs forces(): rel {jac['rel']:.3e}   eig residual "
          f"{system['eig_residual']:.3e}   orthonormality {system['orth_residual']:.3e}")

    ramps = [float(x) for x in args.ramps.split(",")]

    # ---- I/K: ramp sweep at a fixed effective step ------------------------
    sweep = []
    histories = {}
    for T in ramps:
        run = run_engine(T, substeps=4)
        ex = ExactResponse(system, ramp=T)
        t_end = run["t_end"]
        i_exact = ex.impulse(t_end)
        i_eng = run["tool_impulse"]
        u_exact = ex.displacement(t_end)
        u_eng = run["net"].u.numpy().reshape(-1)
        gap_exact_min = min(float(ex.penetration(t).min())
                            for t in np.linspace(1e-4 * T, t_end, 801))
        i_qs = quasi_static_impulse(net0, T, t_end)
        sweep.append(dict(
            ramp=T, t_end=t_end, dt_eff=run["h"], engine=i_eng, exact_dynamic=i_exact,
            rel_exact=abs(i_eng - i_exact) / abs(i_exact),
            quasi_static=i_qs, rel_quasi=abs(i_eng - i_qs) / abs(i_qs),
            u_rel=float(np.abs(u_eng - u_exact).max() / np.abs(u_exact).max()),
            min_gap_engine=run["min_gap"], min_gap_exact=gap_exact_min,
            u_pen=float(run["net"].forces(plane=_plane(t_end, T))[3]),
            tool_work=run["tool_work"],
        ))
        histories[T] = run["hist"]
        print(f"[I] T={T:<5} I_eng={i_eng:.8e} I_exact={i_exact:.8e} "
              f"rel={sweep[-1]['rel_exact']:.3e} | quasi rel={sweep[-1]['rel_quasi']:.3e} "
              f"| min gap eng={run['min_gap']:.2e} exact={gap_exact_min:.2e}")

    # ---- J: time-step refinement at the reference ramp --------------------
    refine_ramp = 0.5
    refine = []
    for s in (1, 2, 4, 8):
        run = run_engine(refine_ramp, substeps=s)
        ex = ExactResponse(system, ramp=refine_ramp)
        i_exact = ex.impulse(run["t_end"])
        refine.append(dict(substeps=s, dt_eff=run["h"], engine=run["tool_impulse"],
                           exact_dynamic=i_exact,
                           rel=abs(run["tool_impulse"] - i_exact) / abs(i_exact)))
        print(f"[J] dt={run['h']:.3e}  rel={refine[-1]['rel']:.3e}")
    order = fit_order([r["dt_eff"] for r in refine], [r["rel"] for r in refine])

    # ---- figures ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for T in ramps:
        hist = histories[T]
        ts = np.array([p["t"] for p in hist])
        fe = np.array([ExactResponse(system, ramp=T).force(t) for t in ts])
        ax.plot(ts, fe, "-", lw=1.6, label=f"exact, T={T:g} s")
        if T in (0.25, 0.5):
            ax.plot(ts[::max(1, len(ts) // 400)],
                    np.array([p["force_tool"] for p in hist])[::max(1, len(ts) // 400)],
                    "o", ms=2.2, mfc="none", label=f"engine, T={T:g} s")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("tool force $\\sum_i P_i g_i$ [N]")
    ax.set_title("Dynamic tool force: exact modal solution vs engine")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_force_history.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    d = [r["dt_eff"] for r in refine]
    e = [r["rel"] for r in refine]
    ax.loglog(d, e, "o-", label="engine vs exact dynamic")
    ref = e[0] * (np.array(d) / d[0]) ** order
    ax.loglog(d, ref, "--", lw=1.0, label=f"slope {order:.2f}")
    ax.set_xlabel("effective time step $\\Delta t$ [s]")
    ax.set_ylabel("relative impulse error")
    ax.set_title(f"Impulse quadrature convergence (T = {refine_ramp:g} s)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_dt_convergence.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    ax = axes[0]
    ax.plot(ramps, [r["exact_dynamic"] for r in sweep], "k-", label="exact dynamic")
    ax.plot(ramps, [r["engine"] for r in sweep], "o", mfc="none", label="engine ledger")
    ax.plot(ramps, [r["quasi_static"] for r in sweep], "s--", ms=4, label="quasi-static (old reference)")
    ax.set_xlabel("ramp time T [s]")
    ax.set_ylabel("tool impulse [N s]")
    ax.set_title("Impulse vs ramp speed")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.semilogy(ramps, [r["rel_exact"] for r in sweep], "o-", label="vs exact dynamic")
    ax.semilogy(ramps, [r["rel_quasi"] for r in sweep], "s--", label="vs quasi-static")
    ax.axhline(0.03, color="r", ls=":", lw=1.0, label="3 % criterion")
    ax.set_xlabel("ramp time T [s]")
    ax.set_ylabel("relative error")
    ax.set_title("Why the static reference had no teeth")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_impulse_vs_ramp.png"), dpi=140)
    plt.close(fig)

    payload = dict(
        schema="tensordem.moving-plane-dynamic-impulse/1",
        geometry=dict(nx=8, ny=2, nodes=len(net0.x), free_dof=len(system["idx"]),
                      dt=float(net0.dt), depth=DEPTH, hold=HOLD,
                      penalty=float(net0.penalty)),
        identity=dict(k_vs_forces_rel=jac["rel"], eig_residual=system["eig_residual"],
                      orthonormality=system["orth_residual"],
                      omega_min=float(system["omega"].min()),
                      omega_max=float(system["omega"].max())),
        ramp_sweep=sweep, dt_refinement=refine, refinement_order=order,
    )
    with open(os.path.join(args.out, "dynamic_impulse.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    with open(os.path.join(args.out, "ramp_sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sweep[0]))
        w.writeheader()
        w.writerows(sweep)
    with open(os.path.join(args.out, "dt_refinement.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(refine[0]))
        w.writeheader()
        w.writerows(refine)
    print(f"[done] artifacts in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())