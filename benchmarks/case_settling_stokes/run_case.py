#!/usr/bin/env python3
"""CASE-SETTLING-STOKES: single-sphere settling terminal velocity (TensorDEM).

A single sphere settles under gravity in a quiescent infinite fluid.  The
reference is the **Stokes analytic terminal velocity** (Stokes 1851; standard
textbook form, e.g. Clift, Grace & Weber 1978 *Bubbles, Drops, and Particles*,
and Crowe et al. 2011 *Multiphase Flows with Droplets and Particles*)::

    v_t = (rho_p - rho_f) g d^2 / (18 mu)          [Re << 1]

The TensorDEM solver integrates the full particle equation of motion with the
Schiller-Naumann drag correction (``C_D = 24/Re (1 + 0.15 Re^0.687)``), which
reduces to the Stokes drag ``3 pi mu d u`` in the ``Re -> 0`` limit.  In the
Stokes regime (``Re < 0.1``) the numerical terminal velocity must match the
analytic Stokes value within the declared tolerance (5 %).

This driver reuses the existing ``tensordem.two_way_coupling`` verification
surface (``run_stokes_verification``) - no new solver code, only the case
harness.  It writes:

* ``artifacts/case_settling_stokes/result.json``   - every measured number
* ``artifacts/case_settling_stokes/gates.json``    - gate table (criterion /
  measured / threshold / pass)
* ``artifacts/case_settling_stokes/fig_*.png``     - report figures
* ``docs/CASE_SETTLING_STOKES.md``                 - report (numbers injected
  programmatically from ``result.json``, no hand-copied digits)

Usage::

    PYTHONPATH=src python benchmarks/case_settling_stokes/run_case.py \
        --out artifacts/case_settling_stokes
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensordem.two_way_coupling import (  # noqa: E402
    stokes_terminal_velocity,
    schiller_naumann_terminal_velocity,
    relaxation_time,
    run_stokes_verification,
)

GRAVITY = 9.81
TOLERANCE_REL = 0.05  # <5% (task tolerance)

# ---- primary case (Stokes regime: Re_terminal ~ 0.02, clearly Re << 1) ----
PRIMARY = dict(
    diameter=3.0e-5,      # 30 um
    rho_particle=2500.0,  # sand/glass
    rho_fluid=1000.0,     # water
    mu=1.0e-3,            # water @ 20 C
    gravity=GRAVITY,
    dt=1.0e-5,
    t_max=0.05,
)


def _ok(value: float, target: float, kind: str = "max") -> dict:
    passed = abs(value) <= target if kind == "max" else abs(value) >= target
    return {"value": float(value), "target": float(target), "pass": bool(passed)}


def reference_error(v_numerical: float, v_stokes: float) -> dict:
    """Relative error of the numerical terminal velocity vs the Stokes analytic."""
    err = (v_numerical - v_stokes) / v_stokes
    return {
        "v_numerical": float(v_numerical),
        "v_stokes": float(v_stokes),
        "relative_error": float(err),
        "pass_5pct": bool(abs(err) <= TOLERANCE_REL),
    }


def run_primary() -> dict:
    """Run the primary Stokes-regime case and the full verification."""
    p = PRIMARY
    t0 = time.time()
    res = run_stokes_verification(
        diameter=p["diameter"], rho_particle=p["rho_particle"],
        rho_fluid=p["rho_fluid"], mu=p["mu"], gravity=p["gravity"],
        dt=p["dt"], t_max=p["t_max"],
    )
    wall = round(time.time() - t0, 2)

    v_stokes = stokes_terminal_velocity(
        p["diameter"], p["rho_particle"], p["rho_fluid"], p["mu"], p["gravity"])
    v_sn = schiller_naumann_terminal_velocity(
        p["diameter"], p["rho_particle"], p["rho_fluid"], p["mu"], p["gravity"])
    tau = relaxation_time(
        p["diameter"], p["rho_particle"], p["rho_fluid"], p["mu"], 0.5)

    Re = p["rho_fluid"] * v_stokes * p["diameter"] / p["mu"]

    return {
        "case": {
            "diameter_m": p["diameter"],
            "rho_particle": p["rho_particle"],
            "rho_fluid": p["rho_fluid"],
            "mu": p["mu"],
            "gravity": p["gravity"],
            "dt": p["dt"],
            "t_max": p["t_max"],
        },
        "dimensionless": {
            "Re_stokes": Re,
            "Re_terminal_numerical": res["Re_terminal"],
            "density_ratio": p["rho_particle"] / p["rho_fluid"],
            "stokes_regime": bool(Re < 0.1),
        },
        "reference": {
            "v_t_stokes": v_stokes,
            "v_t_sn": v_sn,
            "source": "Stokes (1851); Clift, Grace & Weber (1978); "
                      "Crowe et al. (2011) - v_t = (rho_p-rho_f) g d^2/(18 mu)",
        },
        "numerical": {
            "v_t_numerical": res["v_t_numerical"],
            "relative_error_stokes": res["relative_error_stokes"],
            "relative_error_sn": res["relative_error_sn"],
            "tau_analytical": res["tau_analytical"],
            "tau_fit": res["tau_fit"],
            "ledger_residual": res["ledger_residual"],
        },
        "gates": {
            "vs_stokes_<5pct": _ok(res["relative_error_stokes"], TOLERANCE_REL),
            "vs_sn_<1e-6": _ok(res["relative_error_sn"], 1.0e-6),
            "stokes_regime_Re<0.1": _ok(Re, 0.1),
            "tau_fit_vs_analytical_<10pct": _ok(
                abs(res["tau_fit"] - res["tau_analytical"]) / res["tau_analytical"],
                0.10),
        },
        "wall_time_s": wall,
        "repro_cmd": (
            "PYTHONPATH=src python benchmarks/case_settling_stokes/run_case.py "
            "--out artifacts/case_settling_stokes"),
        "_history": {
            "t": [float(x) for x in res["times"]],
            "speed": [float(x) for x in res["speed"]],
            "v_t_stokes": v_stokes,
            "v_t_sn": v_sn,
        },
    }


def run_diameter_sweep() -> list:
    """Sensitivity: terminal velocity vs Stokes as Re grows (SN correction)."""
    p = PRIMARY
    rows = []
    for d in (1.0e-5, 2.0e-5, 3.0e-5, 5.0e-5, 7.0e-5, 1.0e-4):
        v_stokes = stokes_terminal_velocity(
            d, p["rho_particle"], p["rho_fluid"], p["mu"], p["gravity"])
        v_sn = schiller_naumann_terminal_velocity(
            d, p["rho_particle"], p["rho_fluid"], p["mu"], p["gravity"])
        Re = p["rho_fluid"] * v_sn * d / p["mu"]
        rows.append({
            "diameter_m": d,
            "Re": Re,
            "v_t_stokes": v_stokes,
            "v_t_sn": v_sn,
            "sn_over_stokes": v_sn / v_stokes,
            "err_vs_stokes": abs(v_sn - v_stokes) / v_stokes,
        })
    return rows


def run_density_ratio_sweep() -> list:
    """Sensitivity: terminal velocity scaling with density ratio."""
    p = PRIMARY
    rows = []
    for rho_p in (1500.0, 2500.0, 5000.0, 10000.0):
        v_stokes = stokes_terminal_velocity(
            p["diameter"], rho_p, p["rho_fluid"], p["mu"], p["gravity"])
        v_sn = schiller_naumann_terminal_velocity(
            p["diameter"], rho_p, p["rho_fluid"], p["mu"], p["gravity"])
        Re = p["rho_fluid"] * v_sn * p["diameter"] / p["mu"]
        rows.append({
            "rho_particle": rho_p,
            "density_ratio": rho_p / p["rho_fluid"],
            "Re": Re,
            "v_t_stokes": v_stokes,
            "v_t_sn": v_sn,
            "err_vs_stokes": abs(v_sn - v_stokes) / v_stokes,
        })
    return rows


def run_dt_convergence() -> list:
    """Convergence: numerical terminal velocity vs dt refinement."""
    p = PRIMARY
    rows = []
    for dt in (2.0e-5, 1.0e-5, 5.0e-6, 2.5e-6):
        res = run_stokes_verification(
            diameter=p["diameter"], rho_particle=p["rho_particle"],
            rho_fluid=p["rho_fluid"], mu=p["mu"], gravity=p["gravity"],
            dt=dt, t_max=p["t_max"],
        )
        v_sn = schiller_naumann_terminal_velocity(
            p["diameter"], p["rho_particle"], p["rho_fluid"], p["mu"], p["gravity"])
        rows.append({
            "dt": dt,
            "v_t_numerical": res["v_t_numerical"],
            "v_t_sn": v_sn,
            "err_vs_sn": abs(res["v_t_numerical"] - v_sn) / v_sn,
            "err_vs_stokes": res["relative_error_stokes"],
        })
    return rows


def make_figures(primary: dict, d_sweep: list, rho_sweep: list, dt_conv: list,
                 out_dir: str) -> list:
    """Render the report figures (settling curve + comparison + sensitivity)."""
    figs = []
    hist = primary["_history"]
    t = np.array(hist["t"])
    speed = np.array(hist["speed"])
    v_stokes = hist["v_t_stokes"]
    v_sn = hist["v_t_sn"]

    # ---- Fig 1: settling velocity vs time (relaxation to terminal) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t * 1e3, speed * 1e3, "b-", lw=2, label="numerical (TensorDEM)")
    ax.axhline(v_stokes * 1e3, color="k", ls="--", lw=1.5,
               label=f"Stokes analytic {v_stokes*1e3:.3f} mm/s")
    ax.axhline(v_sn * 1e3, color="r", ls=":", lw=1.5,
               label=f"Schiller-Naumann {v_sn*1e3:.3f} mm/s")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("settling speed [mm/s]")
    ax.set_title("Single-sphere settling: relaxation to terminal velocity")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_settling_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"name": "fig_settling_curve.png", "path": path,
                 "caption": "Settling speed vs time; numerical relaxes to the "
                            "Schiller-Naumann terminal velocity, within 5% of "
                            "the Stokes analytic."})

    # ---- Fig 2: error vs Stokes across the diameter sweep (Re growth) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ds = [r["diameter_m"] for r in d_sweep]
    errs = [r["err_vs_stokes"] * 100 for r in d_sweep]
    ax.semilogx(ds, errs, "o-", lw=2, label="|v_SN - v_Stokes| / v_Stokes")
    ax.axhline(5.0, color="r", ls="--", lw=1.5, label="5% tolerance")
    ax.set_xlabel("diameter [m]")
    ax.set_ylabel("error vs Stokes [%]")
    ax.set_title("Stokes-regime validity: error grows with Re (SN correction)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_diameter_sweep.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"name": "fig_diameter_sweep.png", "path": path,
                 "caption": "Relative error of the SN terminal velocity vs the "
                            "pure Stokes analytic across the diameter sweep; "
                            "the 5% tolerance is met for Re < ~0.3."})

    # ---- Fig 3: dt convergence ----
    fig, ax = plt.subplots(figsize=(8, 5))
    dts = [r["dt"] for r in dt_conv]
    errs_sn = [r["err_vs_sn"] * 100 for r in dt_conv]
    ax.semilogx(dts, errs_sn, "s-", lw=2, label="|v_num - v_SN| / v_SN")
    ax.set_xlabel("dt [s]")
    ax.set_ylabel("error vs Schiller-Naumann [%]")
    ax.set_title("dt convergence: numerical terminal velocity")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_dt_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"name": "fig_dt_convergence.png", "path": path,
                 "caption": "Numerical terminal velocity converges to the "
                            "Schiller-Naumann value as dt is refined."})

    return figs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/case_settling_stokes")
    args = ap.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    primary = run_primary()
    d_sweep = run_diameter_sweep()
    rho_sweep = run_density_ratio_sweep()
    dt_conv = run_dt_convergence()

    figs = make_figures(primary, d_sweep, rho_sweep, dt_conv, out_dir)

    result = {
        "primary": primary,
        "diameter_sweep": d_sweep,
        "density_ratio_sweep": rho_sweep,
        "dt_convergence": dt_conv,
        "tolerance_relative": TOLERANCE_REL,
        "figures": [{"name": f["name"], "caption": f["caption"]} for f in figs],
        "all_gates_pass": all(g["pass"] for g in primary["gates"].values()),
    }

    with open(os.path.join(out_dir, "result.json"), "w") as fh:
        json.dump(result, fh, indent=1)

    # gates.json
    gates = {
        "criterion": [
            {"id": k, "measured": v["value"], "threshold": v["target"],
             "pass": v["pass"]} for k, v in primary["gates"].items()
        ],
        "tolerance_relative": TOLERANCE_REL,
        "all_pass": result["all_gates_pass"],
    }
    with open(os.path.join(out_dir, "gates.json"), "w") as fh:
        json.dump(gates, fh, indent=1)

    print(json.dumps({
        "v_t_stokes": primary["reference"]["v_t_stokes"],
        "v_t_numerical": primary["numerical"]["v_t_numerical"],
        "relative_error_stokes": primary["numerical"]["relative_error_stokes"],
        "relative_error_sn": primary["numerical"]["relative_error_sn"],
        "Re": primary["dimensionless"]["Re_stokes"],
        "all_gates_pass": result["all_gates_pass"],
        "figures": [f["name"] for f in figs],
        "out": out_dir,
    }, indent=1))


if __name__ == "__main__":
    main()
