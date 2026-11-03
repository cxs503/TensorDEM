"""Figures for the finite-amplitude floe-heave verification (m2843).

Reads ``artifacts/floe_heave/heave_report.json`` (produced by
``benchmarks/floe_heave_finite_amplitude.py``) and writes three PNGs next to it.
Nothing is transcribed by hand: every curve comes from the report or from
re-evaluating the closed-form quadrature on a dense amplitude grid (cheap).

    python benchmarks/floe_heave_figures.py [--dir artifacts/floe_heave]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import floe_heave_finite_amplitude as H   # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="artifacts/floe_heave")
    args = parser.parse_args()
    out = Path(args.dir)
    report = json.loads((out / "heave_report.json").read_text())

    y_eq = report["equilibrium"]["y_eq_module"]
    t0 = report["stiffness"]["period_linear"]

    # --- Figure 1: restoring force and local stiffness -------------------------
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ys = [row["y"] for row in report["force_curve"]]
    ax[0].plot(ys, [row["F_reference"] for row in report["force_curve"]],
               "-", lw=2.2, label="closed form  $\\rho_w g t A_{sub}(y) - mg$")
    ax[0].plot(ys, [row["F_code"] for row in report["force_curve"]], "o", ms=3.2,
               mfc="none", label="repository law")
    ax[0].axhline(0.0, color="0.6", lw=0.8)
    ax[0].axvline(y_eq, color="0.6", ls="--", lw=0.8)
    ax[0].set_xlabel("floe centre height $y$ [m]")
    ax[0].set_ylabel("net vertical force [N]")
    ax[0].set_title("(a) restoring force: asymmetric about $y_{eq}$")
    ax[0].legend(fontsize=7, loc="upper right")
    ax[1].plot(ys, [row["k_local"] for row in report["force_curve"]], "-", color="C3")
    ax[1].axvline(y_eq, color="0.6", ls="--", lw=0.8, label="$y_{eq}$")
    ax[1].set_xlabel("floe centre height $y$ [m]")
    ax[1].set_ylabel("$k(y)=\\rho_w g t \\, w(y)$ [N/m]")
    ax[1].set_title("(b) instantaneous heave stiffness varies by ~3x")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig1_restoring_force.png", dpi=140)
    plt.close(fig)

    # --- Figure 2: period vs amplitude, quadrature vs simulation ---------------
    ratios = [r for r in (i / 200.0 for i in range(1, 81))]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for sign, style, name in ((1.0, "-", "released upward"), (-1.0, "--", "released downward")):
        grid = [sign * r for r in ratios]
        periods = [H.period_and_mean_quadrature(y_eq + r * H.RADIUS, y_eq)["period"] for r in grid]
        ax.plot([abs(r) for r in grid], [p / t0 - 1.0 for p in periods], style, lw=1.8,
                label=f"quadrature, {name}")
    for row in report["rows"]:
        sign = 1.0 if row["amplitude_ratio"] > 0 else -1.0
        ax.plot([abs(row["amplitude_ratio"])], [row["period_sim_refined"] / t0 - 1.0],
                "o" if sign > 0 else "s", ms=5, mfc="none", color="C0" if sign > 0 else "C1")
    ax.plot([], [], "o", mfc="none", color="C0", label="simulation, released upward")
    ax.plot([], [], "s", mfc="none", color="C1", label="simulation, released downward")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("$|A|/R$")
    ax.set_ylabel("$T/T_0 - 1$")
    ax.set_title("finite-amplitude heave period ($T_0=%.4f$ s)" % t0)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "fig2_period_vs_amplitude.png", dpi=140)
    plt.close(fig)

    # --- Figure 3: time series and rectified mean draft -----------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for key, series in report["series"].items():
        ax.plot(np.asarray(series["t"]), np.asarray(series["y"]) - series["y_eq"],
                lw=1.1, label=f"A/R = {key}")
        ax.axhline(series["mean_quadrature"] - series["y_eq"], lw=0.9, ls=":",
                   color=ax.lines[-1].get_color())
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("$y - y_{eq}$ [m]")
    ax.set_title("heave orbits; dotted = quadrature time-mean (rides lower)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig3_rectified_mean_draft.png", dpi=140)
    plt.close(fig)

    print("wrote fig1_restoring_force.png fig2_period_vs_amplitude.png "
          "fig3_rectified_mean_draft.png to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())