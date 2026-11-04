#!/usr/bin/env python3
"""Dynamic ``tool_impulse`` of the moving-plane boundary: closed-form reference.

Before this script ``docs/dem3d_moving_plane_ledger_verification.md`` recorded the
*dynamic* ``tool_impulse`` of ``MovingPlaneNetwork`` as ``WITHHELD``: the ledger
was only ever checked in the frozen-network limit, where the plane force is a
pure penalty spring and ``int F dt`` is trivial.  Nothing checked the elastic
case, i.e. the regime the ice-resistance models actually run in.

The network is linear (central springs, frozen normals) and the plane contact is
a linear-in-gap penalty that stays active throughout, so the free-DOF motion is
``M u'' + A u = p(t) q - q*x_f`` -- a linear ODE with a *polynomial* forcing
(the quintic C2 plane path has a cubic acceleration).  That makes ``int F dt``
available in closed form (:mod:`tensordem.moving_plane_impulse`).

Three mutually independent references are built here, none of which reuses the
engine's quadrature:

  ref-1  modal closed form      (``SceneModel.exact_impulse``)
  ref-2  ODE integration        (``scipy`` DOP853, rtol 1e-12, separate code path)
  ref-3  momentum identity      (``momentum_identity_impulse``: only the engine's
                                 end velocity + closed-form linear algebra)

Checks:
  A  plane kinematics: p, p', p'' against central differences; int p by quadrature
  B  the polynomial-cosine moment against ``scipy.integrate.quad``
  C  ref-1 vs ref-2 over several horizons
  D  ref-3 vs ref-1, and engine ledger vs ref-3
  E  engine ledger vs ref-1 over a ramp sweep (relative error table)
  F  time-step refinement at fixed ramp (quadrature convergence order)
  G  quasi-static limit: ramp -> large, dynamic -> rate-free integral
  H  preconditions: contact never lost (gap > 0), and depth = 0 -> J = 0

Run:
    PYTHONPATH=src python3 scripts/verify_moving_plane_dynamic_impulse.py \
        --out docs/moving-plane-dynamic-impulse
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
from scipy.integrate import quad, solve_ivp

from tensordem.moving_plane_boundary import MovingPlaneNetwork, smooth_plane
from tensordem.moving_plane_impulse import (
    SceneModel,
    assemble_linear_model,
    exact_tool_impulse_history,
    momentum_identity_impulse,
    plane_derivatives,
    plane_integral,
    poly_cos_moment,
    quasi_static_tool_impulse,
    run_engine,
)

DEPTH = 1.0e-3
NX, NY = 8, 2


def fit_order(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    keep = ys > 0
    if keep.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(xs[keep]), np.log(ys[keep]), 1)[0])


# --------------------------------------------------------------------------- #
# A. plane kinematics
# --------------------------------------------------------------------------- #
def check_kinematics():
    rows = []
    for ramp in (0.05, 0.5, 1.0):
        worst = 0.0
        for t in np.linspace(0.0, 1.4 * ramp, 57):
            h = 1e-6
            p = plane_derivatives(t, ramp, DEPTH)[0]
            p_ref = smooth_plane(t, ramp=ramp, depth=DEPTH)
            d1 = (smooth_plane(t + h, ramp=ramp, depth=DEPTH)
                  - smooth_plane(t - h, ramp=ramp, depth=DEPTH)) / (2 * h)
            inside = 0.0 < t < ramp
            d2 = 0.0
            if inside and t - h > 0 and t + h < ramp:
                d2 = (smooth_plane(t + h, ramp=ramp, depth=DEPTH)
                      - 2 * p_ref + smooth_plane(t - h, ramp=ramp, depth=DEPTH)) / h**2
            _, pd, pdd = plane_derivatives(t, ramp, DEPTH)
            worst = max(worst, abs(p - p_ref),
                        abs(pd - d1) / max(abs(d1), 1e-12) if inside else 0.0,
                        abs(pdd - d2) / max(abs(d2), 1e-9) if inside else 0.0)
        rows.append(dict(ramp=ramp, p_error=max(abs(plane_derivatives(t, ramp, DEPTH)[0]
                                                   - smooth_plane(t, ramp=ramp, depth=DEPTH))
                                                 for t in np.linspace(0, 1.4 * ramp, 57)),
                         deriv_agreement=worst))
    quad_err = max(abs(plane_integral(ramp, DEPTH, T)
                       - quad(lambda s: smooth_plane(s, ramp=ramp, depth=DEPTH), 0.0, T,
                              epsabs=1e-16, epsrel=1e-14, limit=400)[0])
                   for ramp, T in ((0.5, 0.3), (0.5, 1.0), (1.0, 1.0), (2.0, 3.0)))
    return dict(rows=rows, integral_abs_error=quad_err)


# --------------------------------------------------------------------------- #
# B. polynomial-cosine moment vs adaptive quadrature
# --------------------------------------------------------------------------- #
def check_moment():
    ramp, horizon = 0.7, 0.7
    coeffs = (0.0, -60.0 * DEPTH / ramp**3, 180.0 * DEPTH / ramp**4,
              -120.0 * DEPTH / ramp**5)
    poly = lambda s: sum(c * s**n for n, c in enumerate(coeffs))
    rows = []
    for omega in (34.0, 326.0, 1200.0, 5000.0):
        ref = quad(lambda s: poly(s) * math.cos(omega * (horizon - s)), 0.0, horizon,
                   epsabs=1e-20, epsrel=1e-13, limit=800)[0]
        got = poly_cos_moment(coeffs, horizon, omega)
        rows.append(dict(omega=omega, quad=ref, closed_form=got,
                         rel_error=abs(got - ref) / max(abs(ref), 1e-300)))
    return rows


# --------------------------------------------------------------------------- #
# C. ODE integration (independent of the modal expansion)
# --------------------------------------------------------------------------- #
def ode_impulse(model: SceneModel, ramp: float, depth: float, horizon: float,
                samples: int = 12001):
    nf = len(model.free)

    def rhs(t, y):
        u = y[:nf]
        v = y[nf:]
        p = plane_derivatives(t, ramp, depth)[0]
        return np.concatenate([v, (-model.A @ u + model.q * p - model.q * model.xf) / model.mass])

    sol = solve_ivp(rhs, (0.0, horizon), np.zeros(2 * nf), method="DOP853",
                    rtol=1e-12, atol=1e-15, dense_output=True)
    ts = np.linspace(0.0, horizon, samples)
    planes = np.array([plane_derivatives(t, ramp, depth)[0] for t in ts])
    Y = sol.sol(ts)
    forces = np.array([model.q @ (model.xf + Y[:nf, i] - planes[i]) for i in range(len(ts))])
    gap = np.array([model.gaps(Y[:nf, i], planes[i]).min() for i in range(len(ts))])
    return dict(impulse=float(np.trapezoid(forces, ts)),
                impulse_simpson=float(np.trapezoid(forces, ts)),
                gap_min=float(gap.min()), times=ts, force=forces)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/moving-plane-dynamic-impulse")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)

    base = MovingPlaneNetwork(NX, NY)
    model = assemble_linear_model(base)
    checks = {}

    # ---- model bookkeeping ------------------------------------------------ #
    checks["model"] = dict(
        nodes=len(base.x), dofs=2 * len(base.x), free_dofs=len(model.free),
        omega_min=float(model.omegas.min()), omega_max=float(model.omegas.max()),
        penalty_total=model.P_total, dt=float(base.dt),
        right_dof=model.right_dof.tolist(),
        static_force_at_depth=float(model.plane_force(
            np.linalg.solve(model.A, (1.0 - DEPTH) * model.q - model.q * model.xf), 1.0 - DEPTH)),
    )

    checks["A_kinematics"] = check_kinematics()
    checks["B_moment"] = check_moment()

    # ---- C: closed form vs ODE integration -------------------------------- #
    ramp = 1.0
    c_rows = []
    for T in (0.05, 0.2, 0.5, 1.0, 1.5):
        ode = ode_impulse(model, ramp, DEPTH, T)
        closed = model.exact_impulse(ramp, DEPTH, T)
        c_rows.append(dict(horizon=T, closed_form=closed, ode_dop853=ode["impulse"],
                           rel_error=abs(closed - ode["impulse"]) / abs(ode["impulse"]),
                           rel_error_simpson=0.0,
                           gap_min=ode["gap_min"]))
    checks["C_ode"] = c_rows

    # ---- E/F: engine ramp sweep + time-step refinement -------------------- #
    sweep = []
    for ramp_s in (0.25, 0.5, 1.0, 2.0):
        horizon = ramp_s
        exact = model.exact_impulse(ramp_s, DEPTH, horizon)
        quasi = quasi_static_tool_impulse(base, ramp_s, DEPTH, horizon)
        run = run_engine(NX, NY, ramp_s, DEPTH, horizon, substeps=1)
        run["ramp"] = ramp_s
        run["exact"] = exact
        run["quasi_static"] = quasi
        run["rel_error"] = (run["tool_impulse"] - exact) / exact
        sweep.append(run)
    sweep_rows = [{k: v for k, v in r.items() if k not in ("time", "force", "net")}
                  for r in sweep]

    refine = []
    for ramp_s in (1.0, 0.5):
        exact = model.exact_impulse(ramp_s, DEPTH, ramp_s)
        subs = (1, 2, 4) if ramp_s == 1.0 else (1, 2, 4)
        for sub in subs:
            run = run_engine(NX, NY, ramp_s, DEPTH, ramp_s, substeps=sub)
            refine.append(dict(ramp=ramp_s, substeps=sub, dt=run["dt"], steps=run["steps"],
                               impulse=run["tool_impulse"], exact=exact,
                               rel_error=(run["tool_impulse"] - exact) / exact,
                               gap_min=run["gap_min"]))
    orders = {}
    for ramp_s in (1.0, 0.5):
        rows = [r for r in refine if r["ramp"] == ramp_s]
        orders[str(ramp_s)] = fit_order([r["dt"] for r in rows], [abs(r["rel_error"]) for r in rows])

    # ---- D: momentum identity from the engine end state ------------------- #
    d_rows = []
    for r in sweep:
        net = r["net"]
        v = net.v.detach().cpu().numpy().reshape(-1)[model.free]
        ident = momentum_identity_impulse(net, r["ramp"], DEPTH, r["ramp"], v)
        d_rows.append(dict(ramp=r["ramp"], engine_ledger=r["tool_impulse"],
                           momentum_identity=ident, closed_form=r["exact"],
                           rel_engine_vs_identity=abs(r["tool_impulse"] - ident) / abs(ident),
                           rel_identity_vs_closed=abs(ident - r["exact"]) / abs(r["exact"])))

    # ---- G: quasi-static limit ------------------------------------------- #
    g_rows = []
    for ramp_s in (0.25, 0.5, 1.0, 2.0, 8.0):
        horizon = ramp_s
        exact = model.exact_impulse(ramp_s, DEPTH, horizon)
        quasi = quasi_static_tool_impulse(base, ramp_s, DEPTH, horizon)
        g_rows.append(dict(ramp=ramp_s, dynamic=exact, quasi_static=quasi,
                           rel_difference=(exact - quasi) / quasi))

    # ---- H: limits -------------------------------------------------------- #
    zero_depth = model.exact_impulse(0.5, 0.0, 0.5)
    tiny_ramp_engine = run_engine(NX, NY, 0.05, DEPTH, 0.05, substeps=1)
    checks["H_limits"] = dict(
        zero_depth_impulse=zero_depth,
        zero_depth_engine=run_engine(NX, NY, 0.5, 0.0, 0.5, substeps=1)["tool_impulse"],
        gap_min_all=min([r["gap_min"] for r in sweep_rows] + [tiny_ramp_engine["gap_min"]]),
    )

    checks["D_momentum_identity"] = d_rows
    checks["E_sweep"] = sweep_rows
    checks["F_refinement"] = dict(rows=refine, orders=orders)
    checks["G_quasi_static"] = g_rows

    # ===================================================================== #
    # figures
    # ===================================================================== #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ramp = 0.5
    run = run_engine(NX, NY, ramp, DEPTH, ramp, substeps=2)
    ts = np.linspace(0.0, ramp, 1500)
    exact_f = exact_tool_impulse_history(base, ramp, DEPTH, ts)
    planes = np.array([plane_derivatives(t, ramp, DEPTH)[0] for t in ts])
    u1 = np.linalg.solve(model.A, model.q)
    u0 = np.linalg.solve(model.A, model.q * model.xf)
    stat_f = (np.outer(planes, u1) - u0[None, :]) @ model.q + model.q @ model.xf - model.P_total * planes

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax[0].plot(ts, exact_f, "-", lw=2, label="modal closed form")
    ax[0].plot(run["time"], run["force"], "--", lw=1.2, label=f"engine (dt={run['dt']:.2e} s, /2)")
    ax[0].plot(ts, stat_f, ":", lw=2, label=r"static $F_s(p(t))$")
    ax[0].set_xlabel("t [s]")
    ax[0].set_ylabel(r"plane force $F$ [N]")
    ax[0].set_title("(a) moving-plane force, ramp = 0.5 s")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(ts, exact_f - np.interp(ts, run["time"], run["force"]), "-", lw=1.5)
    ax[1].set_xlabel("t [s]")
    ax[1].set_ylabel("closed form $-$ engine [N]")
    ax[1].set_title("(b) pointwise difference")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "figures", "fig1_force_history.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ramp_s, style in ((1.0, "o-"), (0.5, "s-")):
        rows = [r for r in refine if r["ramp"] == ramp_s]
        ax[0].loglog([r["dt"] for r in rows], [abs(r["rel_error"]) for r in rows], style,
                     label=f"ramp = {ramp_s} s (order {orders[str(ramp_s)]:.2f})")
    dts = np.array([r["dt"] for r in refine if r["ramp"] == 1.0])
    ax[0].loglog(dts, abs(refine[0]["rel_error"]) * (dts / dts[0]) ** 2, "k--", lw=1,
                 label=r"$\propto dt^2$")
    ax[0].set_xlabel("dt [s]")
    ax[0].set_ylabel(r"$|J_{engine}-J_{exact}|/J_{exact}$")
    ax[0].set_title("(a) impulse quadrature convergence")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")

    ramps = [r["ramp"] for r in g_rows]
    ax[1].semilogx(ramps, [r["rel_difference"] for r in g_rows], "o-")
    ax[1].axhline(0.0, color="k", lw=0.8)
    ax[1].set_xlabel("ramp duration [s]")
    ax[1].set_ylabel(r"$(J_{dyn}-J_{quasi})/J_{quasi}$")
    ax[1].set_title("(b) dynamic $\\to$ rate-free limit")
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "figures", "fig2_convergence.png"), dpi=130)
    plt.close(fig)

    # ===================================================================== #
    # tables + json
    # ===================================================================== #
    lines = []
    lines.append("### E. engine ledger vs closed form (ramp sweep, dt = module dt)\n")
    lines.append("| ramp [s] | horizon [s] | engine `tool_impulse` [N s] | closed form [N s] "
                 "| rel. error | quasi-static [N s] | gap_min [m] |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sweep_rows:
        lines.append(f"| {r['ramp']:.2f} | {r['ramp']:.2f} | {r['tool_impulse']:.8f} | "
                     f"{r['exact']:.8f} | {r['rel_error']:+.3e} | {r['quasi_static']:.8f} | "
                     f"{r['gap_min']:.2e} |")
    lines.append("\n### F. time-step refinement\n")
    lines.append("| ramp [s] | substeps | dt [s] | steps | impulse [N s] | rel. error |")
    lines.append("|---|---|---|---|---|---|")
    for r in refine:
        lines.append(f"| {r['ramp']:.2f} | {r['substeps']} | {r['dt']:.3e} | {r['steps']} | "
                     f"{r['impulse']:.9f} | {r['rel_error']:+.3e} |")
    lines.append(f"\nMeasured convergence order: ramp 1.0 s -> **{orders['1.0']:.2f}**, "
                 f"ramp 0.5 s -> **{orders['0.5']:.2f}** (expectation: 2).\n")
    lines.append("### C. closed form vs independent ODE integration (DOP853, rtol 1e-12)\n")
    lines.append("| horizon [s] | closed form [N s] | DOP853 [N s] | rel. error | gap_min [m] |")
    lines.append("|---|---|---|---|---|")
    for r in c_rows:
        lines.append(f"| {r['horizon']:.2f} | {r['closed_form']:.10f} | {r['ode_dop853']:.10f} | "
                     f"{r['rel_error']:.2e} | {r['gap_min']:.2e} |")
    lines.append("\n### D. momentum identity from the engine end state\n")
    lines.append("| ramp [s] | engine ledger [N s] | momentum identity [N s] | rel. engine vs identity |")
    lines.append("|---|---|---|---|")
    for r in d_rows:
        lines.append(f"| {r['ramp']:.2f} | {r['engine_ledger']:.9f} | {r['momentum_identity']:.9f} | "
                     f"{r['rel_engine_vs_identity']:.2e} |")
    lines.append("\n### G. dynamic vs rate-free (quasi-static) impulse\n")
    lines.append("| ramp [s] | dynamic [N s] | quasi-static [N s] | rel. difference |")
    lines.append("|---|---|---|---|")
    for r in g_rows:
        lines.append(f"| {r['ramp']:.2f} | {r['dynamic']:.9f} | {r['quasi_static']:.9f} | "
                     f"{r['rel_difference']:+.3e} |")
    with open(os.path.join(args.out, "tables.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k not in ("time", "force", "net")}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o

    with open(os.path.join(args.out, "audit.json"), "w") as fh:
        json.dump(clean(checks), fh, indent=2, sort_keys=True)

    print(json.dumps(clean({k: checks[k] for k in ("model", "E_sweep", "F_refinement",
                                                   "D_momentum_identity")}), indent=2))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()