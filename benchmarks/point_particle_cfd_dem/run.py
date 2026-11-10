"""Benchmark driver for the point-particle CFD-DEM verification faces.

Produces (all inside this directory unless stated otherwise):

``result.json``    every measured number of the five closed-form gates
``gates.json``     the gate table (criterion / measured / threshold / pass)
``figures/*.png``  the five report figures
``docs/point-particle-cfd-dem.md``  the report, with every number injected
                   programmatically from ``result.json`` (no hand copying)

Usage::

    PYTHONPATH=src python benchmarks/point_particle_cfd_dem/run.py [--quick]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.point_cfd_dem import verification as V  # noqa: E402
from tensordem.point_cfd_dem.fluid import SpectralPeriodicFluid  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
def sensitivity_studies(quick=False):
    """Model-knob sensitivity (documented limitation of the closure)."""
    out = {}
    # 1) drag filter width in the Richardson-Zaki case (default grid)
    widths = [None, 0.5, 1.0] if not quick else [None, 0.5]
    rz_scan = []
    for w in widths:
        r = V.richardson_zaki_case(drag_filter_width=w)
        rz_scan.append({"filter_width": ("none" if w is None else str(w)),
                        "rel_error": r["rel_error"],
                        "u_s": r["settling_velocity_measured"]})
    out["richardson_zaki_filter_scan"] = rz_scan

    # 2) grid resolution of the Richardson-Zaki case (h = d vs 1.5 d)
    res_scan = []
    for cells, h, lat in ((12, 1.0, (6, 6, 5)), (12, 1.5, (8, 8, 8))):
        r = V.richardson_zaki_case(cells=cells, h=h, lattice=lat)
        res_scan.append({"h_over_d": h, "u_s": r["settling_velocity_measured"],
                         "rel_error": r["rel_error"],
                         "solids_fraction": r["solids_fraction"]})
    out["richardson_zaki_resolution"] = res_scan

    # 2) self-response correction on/off in the single-particle case
    r_off = V.stokes_terminal_velocity_case(self_response=None,
                                            drag_filter_width=None,
                                            cells=12 if quick else 16)
    r_on = V.stokes_terminal_velocity_case(self_response="spectral",
                                           drag_filter_width=None,
                                           cells=12 if quick else 16)
    out["stokes_self_response"] = {
        "off_rel_error": r_off["rel_error"], "off_u": r_off["v_terminal_measured"],
        "on_rel_error": r_on["rel_error"], "on_u": r_on["v_terminal_measured"],
        "analytic": r_on["v_terminal_analytic"],
    }
    return out


# ---------------------------------------------------------------------------
def make_figures(res, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    figs = []

    # ---- (1) Stokes terminal velocity -------------------------------------
    st = res["cases"]["stokes_terminal"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(st["time"], -np.array(st["vz"]), "C0-", lw=1.6,
            label="two-way coupled (self-response corrected)")
    ax.plot(st["time"], -np.array(st["vz_uncorrected"]), "C1--", lw=1.4,
            label="uncorrected point particle")
    ax.axhline(st["v_terminal_analytic"], color="k", ls=":", lw=1.4,
               label=r"$v_t=(\rho_p-\rho_f)gd^2/18\mu$")
    ax.set_xlabel("t")
    ax.set_ylabel("settling velocity  $-u_{p,z}$")
    ax.set_title(f"Stokes terminal velocity: error {100*st['rel_error']:.2f} % "
                 f"(gate 2 %)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    figs.append(("stokes_terminal_velocity", fig))

    # ---- (2) added-mass oscillation --------------------------------------
    am = res["cases"]["added_mass"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    t = np.array(am["time"])
    ax.plot(t, am["displacement"], "C0-", lw=1.2, label=r"$C_A=0.5$ (coupled)")
    ax.plot(np.array(am["time_no_added_mass"]), am["displacement_no_added_mass"],
            "C1--", lw=1.0, label=r"counterfactual $C_A=0$")
    ax.set_xlabel("t")
    ax.set_ylabel("vertical displacement  $z-z_0$")
    ax.set_title(r"Added-mass oscillator: $\omega_{num}$ error "
                 f"{100*am['rel_error']:.2f} % "
                 rf"($\omega_{{th}}={am['omega_theory']:.4f}$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    figs.append(("added_mass_oscillation", fig))

    # ---- (3) momentum budget ---------------------------------------------
    me = res["cases"]["momentum_exchange"]
    hist = me.get("history")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if hist:
        tt = np.array(hist["time"])
        dp = np.array(hist["momentum_particle"]) - np.array(
            hist["momentum_particle"][0])
        df = np.array(hist["momentum_fluid"]) - np.array(hist["momentum_fluid"][0])
        ax.plot(tt, dp[:, 0], "C0-", lw=1.5, label=r"$\Delta p_{particles}$")
        ax.plot(tt, df[:, 0], "C1-", lw=1.5, label=r"$\Delta p_{fluid}$")
        ax.plot(tt, dp[:, 0] + df[:, 0], "k-", lw=1.8,
                label=r"$\Delta p_{part}+\Delta p_{fluid}$")
        ax2 = ax.twinx()
        ax2.semilogy(tt[1:], np.abs(dp[1:, 0] + df[1:, 0]) + 1e-30, "g:",
                     label="residual")
        ax2.set_ylabel("|residual|")
    ax.set_xlabel("t")
    ax.set_ylabel("x-momentum")
    ax.set_title("Interphase momentum exchange: "
                 f"residual {me['residual_adjoint']:.2e} (gate 1e-6)")
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.3)
    figs.append(("momentum_exchange", fig))

    # ---- (4) Richardson-Zaki ---------------------------------------------
    rz = res["cases"]["richardson_zaki"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(rz["time"], -np.array(rz["vz_mean"]), "C0-", lw=1.5,
            label=r"DEM mean  $-u_{s}$")
    ax.plot(rz["time"], -np.array(rz["vz_fluid"]), "C2-", lw=1.0,
            label=r"fluid mean  $-u_{f,z}$ (counter-flow)")
    ax.axhline(rz["settling_velocity_rz"], color="k", ls=":", lw=1.5,
               label=rf"$u_t\varepsilon^n$ $= {rz['settling_velocity_rz']:.5f}$")
    ax.axhline(rz["v_terminal_single"], color="C3", ls="--", lw=1.0,
               label=rf"$u_t = {rz['v_terminal_single']:.5f}$ (single particle)")
    ax.set_xlabel("t")
    ax.set_ylabel("velocity")
    ax.set_title(f"Hindered settling: error {100*rz['rel_error']:.2f} % "
                 rf"($\varepsilon={rz['voidage']:.3f}$, $n={rz['rz_exponent']:.2f}$)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    figs.append(("richardson_zaki", fig))

    # ---- (5) pressure-gradient operator ----------------------------------
    pg = res["cases"]["pressure_gradient"]
    fl = SpectralPeriodicFluid((24,) * 3, 0.5)
    L = fl.extent[2]
    k = 2 * np.pi / L
    z = np.arange(fl.shape[2]) * fl.h
    p = np.broadcast_to(pg["amplitude"] * np.sin(k * z)[None, None, :],
                        fl.shape).copy()
    grad = fl.gradient(p)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(z, grad[2, 0, 0, :], "C0-", lw=2.0, label="spectral $\\partial p/\\partial z$")
    ax.plot(z, pg["amplitude"] * k * np.cos(k * z), "k:", lw=1.4,
            label="analytic")
    ax.set_xlabel("z")
    ax.set_ylabel(r"$\partial p/\partial z$")
    ax.set_title(f"Pressure-gradient operator: error {pg['operator_error']:.1e}; "
                 f"particle force error {100*pg['force_error']:.2f} %")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    figs.append(("pressure_gradient", fig))

    written = []
    for name, fig in figs:
        path = outdir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
def write_report_md(res, sens, path):
    c = res["cases"]
    st, am, me, rz, pg = (c["stokes_terminal"], c["added_mass"],
                          c["momentum_exchange"], c["richardson_zaki"],
                          c["pressure_gradient"])
    rows = "\n".join(
        f"| {g['gate']} | {g['criterion']} | {g['measured']:.3e} | "
        f"{g['threshold']:.0e} | {'PASS' if g['passed'] else 'FAIL'} |"
        for g in res["gates"])
    scan = "\n".join(
        f"| {r['filter_width']} | {r['u_s']:.5f} | "
        f"{100 * r['rel_error']:.2f} % |"
        for r in sens["richardson_zaki_filter_scan"])
    res_rows = "\n".join(
        f"| {r['h_over_d']:.2f} d | {r['u_s']:.5f} | {100 * r['rel_error']:.2f} % |"
        for r in sens["richardson_zaki_resolution"])
    md = f"""# Point-particle (unresolved) two-way CFD-DEM coupling in TensorDEM

*node* `p158w1-n03-cfd-dem` · *module* `src/tensordem/point_cfd_dem` ·
*benchmark* `benchmarks/point_particle_cfd_dem` · *commit* see `git log`

**All numbers below are injected programmatically from `result.json`
(`benchmarks/point_particle_cfd_dem/run.py`); none is hand-copied.**

## 1. Case description (physics / geometry / parameters)

Point (unresolved) particles are smaller than a fluid cell: the particle surface
is not resolved, the hydrodynamic force is supplied by analytic closures and the
reaction is returned to the Eulerian fluid as a momentum source.  This is a
*different* capability from `tensordem.lbmdem` (resolved immersed-boundary
coupling on a lattice-Boltzmann fluid) and does not reuse any verified case.

Governing equations

```
m_p du_p/dt = F_drag + F_pg + F_am + (rho_p - rho_f) V_p g + F_ext
F_drag      = 3 pi mu d (u_f - u_p) f(Re) H(eps)          f = 1 + 0.15 Re^0.687
F_pg        = - V_p grad p
F_am        = - C_A rho_f V_p (du_p/dt - Du_f/Dt)          C_A = 0.5 (sphere)
rho_f Du_f/Dt = -grad p + mu lap u_f + f_p->f /(1-eps)     div u_f = 0
f_p->f      = -(sum_p F_drag)  (adjoint trilinear spread)
```

Fluid: triply periodic pseudo-spectral, exact viscous integrating factor,
IFRK2 advection (2/3 dealiased), Leray projection.  Particles: vectorised
Lagrangian set with the added-mass term integrated implicitly
(`(m_p + C_A m_a) a = F`).

| parameter | symbol | value |
|---|---|---|
| particle diameter | $d$ | {st['h']/2:.3g} (gate 1), {rz['h']:.3g} (gate 4) |
| density ratio | $\\rho_p/\\rho_f$ | 2 |
| dynamic viscosity | $\\mu$ | 0.5 |
| gravity | $g$ | 0.5 |
| gate-1 grid / box | $h$, $L$ | {st['h']:.3g} = $d$, $L$ = {st['box_length_in_diameters']:.0f}$d$ |
| gate-4 grid / box | $h$, $L$ | {rz['h']:.3g} = 1.5$d$, {rz['grid'][0]} cells |
| gate-4 solids fraction | $\\phi$ | {rz['solids_fraction']:.4f} ($N$ = {rz['particle_count']}) |
| gate-4 RZ exponent | $n(Re_t)$ | {rz['rz_exponent']:.3f} at $Re_t$ = {rz['reynolds_terminal']:.4f} |
| time step | $\\Delta t$ | {st['dt']:.3g} (gate 1), {rz['dt']:.3g} (gate 4) |

## 2. How it was done

1. `closures.py` holds every coefficient as a pure function (Stokes drag,
   Schiller-Naumann, terminal velocity, Richardson-Zaki exponent and the
   **derived** hindrance factor `H = eps^(1-n)`: from `u_rel = u_t eps^(n-1)`
   the container-frame settling velocity becomes `u_s = eps u_rel = u_t eps^n`).
2. `fluid.py` provides the spectral solver and the two transfer operators.
   Interpolation `I` and spreading `S = I^T` are exact transposes, which makes
   the interphase momentum transfer exact to round-off.
3. `coupling.py` runs the two-way loop, applies the container-frame
   (zero net volumetric flux) Galilean shift and keeps the momentum budget.
4. `verification.py` contains the five closed-form faces; `run.py` executes
   them, writes `result.json`/`gates.json`, the figures and this report.

Two model corrections are needed because an unresolved particle deposits
momentum on a grid of the order of its own size:

* **self-response removal** -- the flow induced by a particle's *own* source is
  `R(t)·F`, with `R(t)` obtained from the same discretisation
  (`R = (2/3) sum_k |W(k)|^2 (1-exp(-nu k^2 t)) / (mu k^2 V)`).  The drag
  relation is solved implicitly: `C_eff = C/(1 - C R)`.  At steady state this
  gives `u_p = v_t` **exactly, for any R** (algebra in `fluid.self_response`).
* **drag-velocity filtering** -- a Gaussian low-pass of width `h/4` applied only
  to the velocity used by the drag closure, removing the sub-cell part of the
  collective source smearing in dense suspensions.

## 3. Results

| gate | criterion | measured | threshold | verdict |
|---|---|---|---|---|
{rows}

| quantity | value |
|---|---|
| $v_t$ (analytic, gate 1) | {st['v_terminal_analytic']:.6f} |
| $u_p$ (measured, gate 1) | {st['v_terminal_measured']:.6f} |
| relative error | {100*st['rel_error']:.3f} % |
| uncorrected point-particle baseline | {100*st['rel_error_uncorrected']:.2f} % (FAIL) |
| $\\omega$ theory (gate 2) | {am['omega_theory']:.6f} |
| $\\omega$ measured | {am['omega_measured']:.6f} |
| $\\omega$ error | {100*am['rel_error']:.3f} % |
| counterfactual $C_A=0$ error | {100*am['counterfactual_no_added_mass_error']:.2f} % (FAIL) |
| momentum residual (gate 3) | {me['residual_adjoint']:.3e} |
| residual with +2 % exchange bias | {me['residual_exchange_bias_2pct']:.3e} (FAIL) |
| $u_s$ measured (gate 4) | {rz['settling_velocity_measured']:.6f} |
| $u_t\\varepsilon^n$ (gate 4) | {rz['settling_velocity_rz']:.6f} |
| gate-4 error | {100*rz['rel_error']:.3f} % |
| gate-4 error, uncorrected | {100*rz['rel_error_uncorrected']:.2f} % |
| pressure-gradient operator error | {pg['operator_error']:.2e} |
| pressure-gradient force error | {100*pg['force_error']:.3f} % |

### Figures

`figures/stokes_terminal_velocity.png`,
`figures/added_mass_oscillation.png`,
`figures/momentum_exchange.png`,
`figures/richardson_zaki.png`,
`figures/pressure_gradient.png`.

## 4. Points of attention (traps, limitations, range of validity)

1. **Periodic box ≠ unbounded fluid.**  A point force in a periodic cell
   accumulates a large-scale back-flow with the viscous time `L^2/nu`.  Ignoring
   it costs {100*st['rel_error_uncorrected']:.0f} % on the terminal velocity; the
   time-dependent self-response `R(t)` removes it and is what makes gate 1 pass.
2. **The self-response correction is a *steady/creeping* construct**: it assumes
   the force history is roughly constant over the viscous development time.  It
   also becomes ill conditioned (`C R -> 1`) when `h < d`, which excludes it from
   the dense-suspension configuration of gate 4; a drag-velocity filter is used
   there instead.
3. **Gate 4 is sensitive to the drag-velocity filter width** (table below),
   i.e. the *collective* part of the smeared interphase source is the dominant
   model uncertainty of unresolved coupling, not the drag law itself:

| filter width (in $h$) | $u_s$ | error vs $u_t\\varepsilon^n$ |
|---|---|---|
{scan}

4. **No particle-particle contacts** are modelled here (point particles without
   collision detection); the module is a verification of the *fluid coupling*,
   not a complete dense-suspension solver.
5. **Model B** (`div u = 0`, voidage only in the closures) is used; no
   volume-averaged (Model A) momentum equation with `d(rho eps u)/dt`.
6. The pressure-gradient force is taken from the *periodic* part of the
   pressure; a mean pressure gradient cannot exist in a periodic box and is
   represented by the container-frame constraint instead.
7. Velocity field is collocated and spectral: no walls, so wall-bounded
   point-particle coupling (e.g. a ship hull boundary layer) is out of scope.

## 5. Error assessment versus the requirement

| key quantity | reference | error | < 3 % ? |
|---|---|---|---|
| single-particle terminal velocity | Stokes $v_t$ | {100*st['rel_error']:.3f} % | {'YES' if st['rel_error'] < 0.03 else 'NO'} |
| added-mass oscillation frequency | $\\sqrt{{k/(m_p+C_A m_a)}}$ | {100*am['rel_error']:.3f} % | {'YES' if am['rel_error'] < 0.03 else 'NO'} |
| interphase momentum conservation | exact (round-off) | {me['residual_adjoint']:.3e} | {'YES' if me['residual_adjoint'] < 0.03 else 'NO'} |
| hindered settling velocity | Richardson-Zaki | {100*rz['rel_error']:.3f} % | {'NO (>3 %, within the 10 % gate)' if rz['rel_error'] > 0.03 else 'YES'} |
| pressure-gradient force | analytic $-V_p$ grad $p$ | {100*pg['force_error']:.3f} % | {'YES' if pg['force_error'] < 0.03 else 'NO'} |

The 3 % question is answered explicitly per quantity: three of the four
physical gates are inside 3 %; the hindered-settling velocity is
{100*rz['rel_error']:.2f} % (inside its own 10 % gate but outside 3 %) because of
point 3 above — the collective source smearing of unresolved particles.

## 6. Reproduction

```
PYTHONPATH=src python benchmarks/point_particle_cfd_dem/run.py
PYTHONPATH=src pytest tests/test_point_cfd_dem.py -q
```
"""
    path.write_text(md)
    return path


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--no-sensitivity", action="store_true")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    res = V.run_all(quick=args.quick)
    # per-step momentum history for the budget figure
    me = V.momentum_exchange_case(**({"n_particles": 10, "cells": 10,
                                      "dt": 0.004, "n_steps": 40}
                                     if args.quick else {}))
    res["cases"]["momentum_exchange"]["history"] = {
        "time": me.get("history_time", []),
        "momentum_particle": me.get("history_momentum_particle", []),
        "momentum_fluid": me.get("history_momentum_fluid", []),
    }

    sens = {} if args.no_sensitivity else sensitivity_studies(args.quick)
    res["sensitivity"] = sens

    payload = {
        "node": "p158w1-n03-cfd-dem",
        "module": "tensordem.point_cfd_dem",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - t0, 1),
        "quick": bool(args.quick),
        **res,
    }
    (outdir / "result.json").write_text(json.dumps(payload, indent=2))
    (outdir / "gates.json").write_text(json.dumps(
        {"node": payload["node"], "all_passed": res["all_passed"],
         "n_passed": res["n_passed"], "n_gates": res["n_gates"],
         "gates": res["gates"]}, indent=2))

    figs = make_figures(res, outdir / "figures")
    if not args.quick and sens:
        md = write_report_md(res, sens, REPO / "docs" / "point-particle-cfd-dem.md")
        print(f"report -> {md}")

    print(json.dumps({"gates": [(g["gate"], g["measured"], g["passed"])
                                for g in res["gates"]],
                      "all_passed": res["all_passed"],
                      "figures": [str(p) for p in figs],
                      "runtime": payload["runtime_seconds"]}, indent=2))
    return 0 if res["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())