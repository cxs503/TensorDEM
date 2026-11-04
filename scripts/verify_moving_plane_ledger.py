#!/usr/bin/env python3
"""Closed-form verification of the smooth moving-plane tool ledgers.

Scope: ``tensordem.moving_plane_boundary.MovingPlaneNetwork`` -- the opt-in
moving tool boundary that stands in for a ship hull pressing on the
control-volume ice network.  Its bookkeeping has two reported quantities,
``tool_work`` and ``tool_impulse``, plus the 50E/L penalty contact on the
reference boundary nodes.  Before this script the only evidence was a
self-consistency test (``tests/test_moving_plane.py``: sign of ``tool_work``,
restart equality, and a 1 % closure of ``energy + U_pen - tool_work``), i.e.
no reference value at all.

The network is a *linear central-spring* system (``e = (u_j - u_i)·n`` with
frozen initial normals) and the penalty contact is quadratic in the gap, so for
a frozen active contact set the total potential is a quadratic form.  That
gives an exact reference in closed form:

    (K + P) u = -P (x - plane)                      -> u affine in plane
    F_plane(plane) = sum_i P_i (x_i + u_i - plane)   -> affine in plane
    W_static(delta) = 0.5 * F_plane(delta) * delta   -> exact, and equal to
                                                       the stored energy U(delta)

Energy balance for the non-autonomous system (plane position is an explicit
time parameter; forces at fixed supports do no work):

    W_tool(t) = dKE + dU_spring + dU_penalty

Checks (machine-checked in ``tests/test_moving_plane_ledger_verification.py``):

  A  reference-model identity: assembled K reproduces ``forces()`` to 1e-16
  B  tributary-area / penalty-law / penalty-energy identities
  C  static plane force exactly affine in penetration (closed-form work),
     and dU/d(delta) == F (thermodynamic consistency of the reference itself)
  D  dynamic ``tool_work`` vs the closed-form work over a ramp-time sweep
     (quasi-static convergence with measured exponent)
  E  time-step refinement at fixed ramp (quadrature convergence)
  F  frozen-network limit: work *and* impulse quadrature are exact there

Run:
    PYTHONPATH=src python3 scripts/verify_moving_plane_ledger.py \
        --out docs/moving-plane-ledger
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from tensordem.continuum_boundary_dem import BoundaryNetwork
from tensordem.moving_plane_boundary import MovingPlaneNetwork, smooth_plane

DEPTH = 0.001


# --------------------------------------------------------------------------- #
# reference model: assemble the quadratic potential the engine actually uses
# --------------------------------------------------------------------------- #
def assemble(net: BoundaryNetwork):
    """Stiffness matrix K (2N x 2N), penalty diagonal P, coordinates, fixed mask.

    Built from the module's own ``edges``/``k``/``n``/``penalty``/``areas``
    arrays so the reference is the same discrete model, not a re-derivation.
    DOF order is [x0, y0, x1, y1, ...] (``reshape(-1)`` of the node arrays).
    """
    n = len(net.x)
    K = np.zeros((2 * n, 2 * n))
    e = net.edges.numpy()
    k = net.k.numpy()
    nn = net.n.numpy()
    for (a, b), kk, v in zip(e, k, nn):
        blk = kk * np.outer(v, v)
        ia = slice(2 * a, 2 * a + 2)
        ib = slice(2 * b, 2 * b + 2)
        K[ia, ia] += blk
        K[ib, ib] += blk
        K[ia, ib] -= blk
        K[ib, ia] -= blk
    P = np.zeros(2 * n)
    P[2 * net.right.numpy()] = net.penalty * net.areas.numpy()
    x = net.x.numpy().reshape(-1)
    fixed = net.fixed.numpy().reshape(-1)
    return K, P, x, fixed


def static_solution(net: BoundaryNetwork, delta: float, max_iter: int = 40):
    """Exact static equilibrium for ``plane = length - delta`` (frozen active set)."""
    K, P, x, fixed = assemble(net)
    n = len(net.x)
    right = net.right.numpy()
    plane = net.config["length"] - delta
    u = np.zeros(2 * n)
    active = np.zeros(n, dtype=bool)
    idx = np.flatnonzero(~fixed)
    for it in range(max_iter):
        gap = x[2 * right] + u[2 * right] - plane
        new_active = gap > 0.0
        if it > 0 and np.array_equal(new_active, active):
            break
        active = new_active
        Pd = np.zeros(2 * n)
        Pd[2 * right[active]] = P[2 * right[active]]
        A = K + np.diag(Pd)
        rhs = np.zeros(2 * n)
        rhs[2 * right[active]] = -Pd[2 * right[active]] * (x[2 * right[active]] - plane)
        u = np.zeros(2 * n)
        u[idx] = np.linalg.solve(A[np.ix_(idx, idx)], rhs[idx])
    gap = np.clip(x[2 * right] + u[2 * right] - plane, 0.0, None)
    force = float(np.sum(P[2 * right] * gap))  # plane pushes the network in -x
    u_pen = float(0.5 * np.sum(P[2 * right] * gap**2))
    u_spring = float(0.5 * u @ K @ u)
    return dict(delta=delta, plane=plane, u=u, gap=gap, force=force,
                u_pen=u_pen, u_spring=u_spring, energy=u_spring + u_pen)


def jacobian_check(net: BoundaryNetwork, seed: int = 7):
    """Internal spring force from ``forces()`` vs the assembled ``-K u``."""
    K, _, _, _ = assemble(net)
    g = np.random.default_rng(seed)
    u = g.normal(0.0, 1e-5, size=(len(net.x), 2))
    net.u = torch.tensor(u, dtype=torch.float64)
    f, ext, _, _ = net.forces(traction=0.0, plane=None)
    f_int = f.numpy().reshape(-1) - ext.numpy().reshape(-1)
    ref = -(K @ u.reshape(-1))
    scale = float(np.abs(ref).max())
    return dict(max_abs_residual=float(np.abs(f_int - ref).max()),
                scale=scale, rel=float(np.abs(f_int - ref).max() / scale))


# --------------------------------------------------------------------------- #
# dynamic sweeps
# --------------------------------------------------------------------------- #
def run_ramp(nx: int, ny: int, ramp: float, dt: float, substeps: int = 1,
             record: bool = False, hold_seconds: float = 0.05):
    """Deterministic C2 plane ramp from 1.0 down to 1 - DEPTH, then a hold."""
    net = MovingPlaneNetwork(nx, ny)
    total = ramp + hold_seconds if hold_seconds > 0 else ramp
    n_steps = int(round(total / dt))
    K, _, _, _ = assemble(net)
    hist = []
    every = 1 if record else max(1, n_steps)
    for i in range(n_steps):
        t0, t1 = i * dt, (i + 1) * dt
        a = smooth_plane(t0, ramp=ramp, depth=DEPTH)
        b = smooth_plane(t1, ramp=ramp, depth=DEPTH)
        for j in range(substeps):
            aj = a + (b - a) * j / substeps
            bj = a + (b - a) * (j + 1) / substeps
            net.moving_step(dt / substeps, aj, bj)
        if record and i % every == 0:
            f, _, _, u_pen = net.forces(plane=b)
            u = net.u.numpy().reshape(-1)
            u_spring = float(0.5 * u @ K @ u)
            hist.append(dict(t=t1, plane=b, delta=1.0 - b,
                             tool_work=net.tool_work,
                             tool_impulse=float(net.tool_impulse),
                             ke=float(net.energy()) - u_spring,
                             u_spring=u_spring, u_pen=u_pen,
                             energy_total=float(net.energy()) + u_pen))
    plane_end = float(smooth_plane(total, ramp=ramp, depth=DEPTH))
    _, _, _, u_pen = net.forces(plane=plane_end)
    u = net.u.numpy().reshape(-1)
    u_spring = float(0.5 * u @ K @ u)
    ke = float(net.energy()) - u_spring
    return dict(nx=nx, ny=ny, ramp=ramp, dt=dt, substeps=substeps, steps=n_steps,
                tool_work=float(net.tool_work),
                tool_impulse=float(net.tool_impulse),
                ke=ke, u_spring=u_spring, u_pen=float(u_pen),
                energy_total=float(ke + u_spring + u_pen), history=hist)


def fit_order(xs, ys):
    """Least-squares slope of log|y| against log|x| (convergence exponent)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    ok = ys > 0
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(xs[ok]), np.log(ys[ok]), 1)[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/moving-plane-ledger")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    checks = {}
    net = MovingPlaneNetwork(8, 2)
    base = net

    # ---- A. reference-model identity ------------------------------------- #
    checks["A_jacobian"] = jacobian_check(net)

    # ---- B. tributary areas / penalty law / penalty energy ---------------- #
    h_t = float(net.config["height"] * net.config["thickness"])
    sum_a = float(net.areas.sum())
    probe = MovingPlaneNetwork(8, 2)
    plane = probe.config["length"] - 3.0e-4
    f, ext, _, u_pen = probe.forces(plane=plane)
    gap = (probe.x[probe.right, 0] + probe.u[probe.right, 0] - plane).clamp_min(0)
    pred = -(probe.penalty * probe.areas * gap)
    off_col = [i for i in range(len(probe.x)) if i not in probe.right.tolist()]
    checks["B_area_identity"] = dict(sum_areas=sum_a, height_times_thickness=h_t,
                                     rel=float(abs(sum_a - h_t) / h_t))
    checks["B_penalty_law"] = dict(
        max_rel=float(((ext[probe.right, 0] - pred).abs()
                       / pred.abs().clamp_min(1e-300)).max()),
        active_nodes=int((gap > 0).sum()), expected_active=int(len(probe.right)),
        u_pen_rel=float(abs(u_pen - float(0.5 * (probe.penalty * probe.areas
                                                 * gap**2).sum()))
                        / float(0.5 * (probe.penalty * probe.areas * gap**2).sum())),
        off_axis_max=float(ext[:, 1].abs().max()),
        off_column_max=float(ext[torch.tensor(off_col), 0].abs().max()))

    # ---- C. static force affine in penetration; dU/ddelta == F ------------ #
    deltas = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
    statics = [static_solution(net, d) for d in deltas]
    slopes = [s["force"] / s["delta"] for s in statics]
    delta_max = DEPTH
    s_max = static_solution(net, delta_max)
    w_exact = 0.5 * s_max["force"] * delta_max
    # thermodynamic consistency: dU/ddelta = F, finite difference on the statics
    du_check = []
    for lo, hi in zip(statics[:-1], statics[1:]):
        dd = hi["delta"] - lo["delta"]
        du_check.append(abs((hi["energy"] - lo["energy"]) / dd
                            - 0.5 * (hi["force"] + lo["force"]))
                        / (0.5 * (hi["force"] + lo["force"])))
    checks["C_affine"] = dict(
        deltas=deltas, forces=[s["force"] for s in statics],
        stiffness=max(slopes), stiffness_min=min(slopes),
        rel_spread=float((max(slopes) - min(slopes)) / max(slopes)),
        force_at_zero=float(static_solution(net, 0.0)["force"]),
        dU_ddelta_max_rel=float(max(du_check)),
        delta_max=dict(force=s_max["force"], u_spring=s_max["u_spring"],
                       u_pen=s_max["u_pen"], energy=s_max["energy"]))
    checks["C_work_exact"] = dict(
        delta=delta_max, w_exact=w_exact, energy_exact=s_max["energy"],
        rel_work_energy=float(abs(w_exact - s_max["energy"]) / s_max["energy"]),
        ke_ref_rel=float(max(abs(np.abs(s_max["u"])).max(), 1e-300)))

    # penalty / network stiffness share (the 50E/L law is E-scaled by design,
    # so there is no E-only rigid-network limit -- documented, not tested)
    col = int(np.argmax(net.x[net.right, 0].numpy()))
    checks["C_stiffness_share"] = dict(
        penalty_x_dof=float(net.penalty * float(net.areas[col])),
        plane_stiffness=float(max(slopes)),
        penalty_share=float(net.penalty * float(net.areas[col])
                            / (net.penalty * float(net.areas[col]) + max(slopes))))

    # ---- D. ramp-time sweep (quasi-static convergence) -------------------- #
    ramp_sweep = []
    dt = 0.5 * base.dt
    for ramp in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        r = run_ramp(8, 2, ramp, dt)
        r["w_error"] = (r["tool_work"] - w_exact) / w_exact
        r["balance"] = (r["tool_work"] - r["energy_total"]) / w_exact
        r["ke_fraction"] = r["ke"] / w_exact
        ramp_sweep.append(r)
    checks["D_ramp"] = dict(
        w_exact=w_exact,
        rows=[{k: r[k] for k in ("ramp", "dt", "steps", "tool_work", "w_error",
                                 "balance", "ke", "ke_fraction", "u_spring",
                                 "u_pen", "energy_total")} for r in ramp_sweep],
        order=fit_order([r["ramp"] for r in ramp_sweep[2:]],
                        [abs(r["w_error"]) for r in ramp_sweep[2:]]),
        order_all=fit_order([r["ramp"] for r in ramp_sweep],
                            [abs(r["w_error"]) for r in ramp_sweep]),
    )

    # ---- E. time-step refinement at fixed ramp ---------------------------- #
    dt_sweep = []
    for sub in (1, 2, 4, 8):
        r = run_ramp(8, 2, 0.5, base.dt, substeps=sub)
        r["w_error"] = (r["tool_work"] - w_exact) / w_exact
        r["balance"] = (r["tool_work"] - r["energy_total"]) / w_exact
        dt_sweep.append(r)
    checks["E_dt"] = dict(
        w_exact=w_exact,
        rows=[{k: r[k] for k in ("dt", "substeps", "steps", "tool_work",
                                 "w_error", "balance", "ke")} for r in dt_sweep],
        order_balance=fit_order([r["dt"] for r in dt_sweep],
                                [abs(r["balance"]) for r in dt_sweep]))

    # ---- F. frozen network: work and impulse quadrature exact ------------- #
    frozen = MovingPlaneNetwork(8, 2)
    frozen.fixed[:] = True
    frozen.v.zero_()
    plane0 = frozen.config["length"] - 2.0e-4
    steps, dpl = 200, 1.0e-7
    for i in range(steps):
        frozen.moving_step(frozen.dt, plane0 - i * dpl, plane0 - (i + 1) * dpl)
    plane1 = plane0 - steps * dpl
    a_tot = frozen.penalty * float(frozen.areas.sum())
    f0 = a_tot * (frozen.config["length"] - plane0)
    f1 = a_tot * (frozen.config["length"] - plane1)
    w_frozen = 0.5 * (f0 + f1) * (plane0 - plane1)
    i_frozen = 0.5 * (f0 + f1) * steps * frozen.dt
    checks["F_frozen"] = dict(
        tool_work=float(frozen.tool_work), work_exact=float(w_frozen),
        work_rel=float(abs(frozen.tool_work - w_frozen) / w_frozen),
        tool_impulse=float(frozen.tool_impulse), impulse_exact=float(i_frozen),
        impulse_rel=float(abs(frozen.tool_impulse - i_frozen) / abs(i_frozen)))

    # ---- history + figures ------------------------------------------------ #
    hist = run_ramp(8, 2, 0.5, base.dt, substeps=2, record=True,
                    hold_seconds=0.25)["history"]
    figures = make_figures(args.out, hist, ramp_sweep, dt_sweep, statics,
                           checks, w_exact)

    summary = dict(
        schema="tensordem.moving-plane-ledger-verification/1",
        network=dict(nodes=int(len(net.x)), free=int((~net.fixed).sum()),
                     dt=float(net.dt), penalty=float(net.penalty),
                     E=net.config["E"], density=net.config["density"],
                     depth=DEPTH),
        exact_static_work=w_exact,
        figures=figures,
        checks=checks,
    )
    with open(os.path.join(args.out, "moving_plane_ledger.json"), "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    with open(os.path.join(args.out, "moving_plane_history.json"), "w") as fh:
        json.dump(hist, fh, indent=1)

    print(json.dumps({k: checks[k] for k in
                      ("A_jacobian", "B_area_identity", "B_penalty_law")}, indent=1))
    print("affine:", json.dumps({k: checks["C_affine"][k] for k in
                                 ("rel_spread", "force_at_zero",
                                  "dU_ddelta_max_rel")}))
    print("W_exact:", w_exact, "rel(work,energy):",
          checks["C_work_exact"]["rel_work_energy"])
    print("ramp sweep (T, W_err, balance, KE/W):")
    for r in ramp_sweep:
        print(f"  T={r['ramp']:6.2f} {r['w_error']:+.3e} {r['balance']:+.3e} "
              f"{r['ke_fraction']:.3e}")
    print("ramp orders:", checks["D_ramp"]["order"], checks["D_ramp"]["order_all"])
    print("dt sweep (dt, W_err, balance):")
    for r in dt_sweep:
        print(f"  dt={r['dt']:.3e} {r['w_error']:+.3e} {r['balance']:+.3e}")
    print("balance order in dt:", checks["E_dt"]["order_balance"])
    print("frozen:", json.dumps(checks["F_frozen"]))
    print("figures:", figures)
    return 0


def make_figures(out, hist, ramp_sweep, dt_sweep, statics, checks, w_exact):
    """Two closed-form comparison figures (no animation, no decoration)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"skipped": str(exc)}
    files = {}

    fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.8))
    d = np.array([s["delta"] for s in statics])
    f = np.array([s["force"] for s in statics])
    ax[0].plot(d, f, "o", ms=4, label="exact static solve (K+P)")
    ax[0].plot(d, f[0] / d[0] * d, "-", lw=1,
               label=f"linear law, k={f[0] / d[0]:.4g} N/m")
    ax[0].set_xlabel("plane penetration $\\delta$ [m]")
    ax[0].set_ylabel("plane force $F$ [N]")
    ax[0].set_title("(a) penalty contact law: static reference")
    ax[0].legend(fontsize=7)
    resid = (f - f[0] / d[0] * d) / f
    ax[1].semilogy(d, np.abs(resid), "s-", ms=4)
    ax[1].set_xlabel("plane penetration $\\delta$ [m]")
    ax[1].set_ylabel("|linear-fit residual|")
    ax[1].set_title(f"(b) affinity residual (max {np.abs(resid).max():.1e})")
    fig.tight_layout()
    name = "fig1_static_law.png"
    fig.savefig(os.path.join(out, name), dpi=140)
    plt.close(fig)
    files["fig1_static_law"] = name

    fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.8))
    T = [r["ramp"] for r in ramp_sweep]
    err = [abs(r["w_error"]) for r in ramp_sweep]
    ax[0].loglog(T, err, "o-", label="|W_tool - W_exact| / W_exact")
    ax[0].loglog(T, [err[0] * (t / T[0]) ** -2 for t in T], "--", lw=1,
                 label="slope -2 reference")
    ax[0].set_xlabel("ramp time $T$ [s]")
    ax[0].set_ylabel("relative work error")
    ax[0].set_title(f"(c) quasi-static convergence, fitted slope "
                    f"{checks['D_ramp']['order']:.2f}")
    ax[0].legend(fontsize=7)
    t = np.array([h["t"] for h in hist])
    ax[1].plot(t, np.array([h["tool_work"] for h in hist]) / w_exact, label="$W_{tool}/W_{exact}$")
    ax[1].plot(t, np.array([h["u_spring"] for h in hist]) / w_exact, label="$U_{spring}/W_{exact}$")
    ax[1].plot(t, np.array([h["ke"] for h in hist]) / w_exact, label="$K/W_{exact}$")
    ax[1].plot(t, np.array([h["u_pen"] for h in hist]) / w_exact, label="$U_{pen}/W_{exact}$")
    ax[1].axhline(1.0, color="k", lw=.6, ls=":")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("energy / exact static work")
    ax[1].set_title("(d) ledger energy partition during the ramp")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    name = "fig2_convergence.png"
    fig.savefig(os.path.join(out, name), dpi=140)
    plt.close(fig)
    files["fig2_convergence"] = name
    return files


if __name__ == "__main__":
    raise SystemExit(main())