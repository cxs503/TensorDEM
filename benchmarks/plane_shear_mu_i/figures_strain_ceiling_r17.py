"""Round 17 (m21066) figures: the strain-budget defect and the ceiling fix.

Reads ``runs/round17/strain_ceiling_r17.json`` (written by
``run_strain_ceiling_r17.py``) so no number is hand-copied into a plot.

Usage:  PYTHONPATH=src python benchmarks/plane_shear_mu_i/figures_strain_ceiling_r17.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROUND17 = Path(__file__).resolve().parent / "runs" / "round17"
TOL = 0.05


def main() -> int:
    data = json.loads((ROUND17 / "strain_ceiling_r17.json").read_text())
    table = data["worst_case_table"]
    runs = {r["label"]: r for r in data["runs"]}

    # ---- figure 1: the defect and the trajectory -------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6))

    Is = [r["I_target"] for r in table]
    worst = [r["worst_case_overrun"] for r in table]
    steps = [r["shear_steps"] for r in table]
    ax1.plot(Is, worst, "o-", color="#c0392b", lw=1.8, ms=6,
             label="worst-case overrun\n$\\gamma_{max}/\\gamma_{req}$  (analytic)")
    ax1.axhline(1.0 + TOL, color="#2c3e50", ls="--", lw=1.2,
                label="acceptance band  1.05x")
    ax1.annotate("round 16 observed 49.5x\n(recovery cap exhausted)",
                 xy=(0.3, 53.0), xytext=(0.012, 30.0), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="0.35", lw=0.9))
    for I, w, st in zip(Is, worst, steps):
        ax1.annotate(f"{st}", xy=(I, w), xytext=(0, 7), fontsize=6.5,
                     ha="center", color="0.35", textcoords="offset points")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("inertial number  I")
    ax1.set_ylabel("worst-case strain overrun  [x requested]")
    ax1.set_title("(a) a strain request was a floor:\n"
                  "the overrun grows with I (step floor shown per point)")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=7.5, loc="upper left")

    colours = {"historical": "#c0392b",
               "ceiling_only": "#e67e22",
               "ceiling_and_strain_only_budget": "#27ae60"}
    for label, r in runs.items():
        tr = r["trace"]
        x = [p["step"] for p in tr]
        y = [p["strain"] for p in tr]
        ax2.plot(x, y, lw=1.7, color=colours[label],
                 label=f"{label.replace('_', ' ')}\n"
                       f"delivered {r['shear_strain_achieved']:.3f} "
                       f"({r['strain_overrun_ratio']:.2f}x)")
    req = runs["historical"]["shear_strain_requested"]
    ax2.axhline(req, color="#2c3e50", ls="--", lw=1.2, label="requested strain")
    ax2.axhspan(req, req * (1 + TOL), color="#2c3e50", alpha=0.12)
    ax2.set_yscale("log")
    ax2.set_xlabel("shear step")
    ax2.set_ylabel("accumulated strain  $\\gamma$")
    ax2.set_title("(b) strain trajectory: the recovery gate\n"
                  "used to shear on until the servo settled")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=7, loc="lower right")
    fig.suptitle("Round 17 — CASE-DEM-PLANE-SHEAR-MUIDI: making a strain request a ceiling",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p1 = ROUND17 / "fig_strain_budget_defect.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # ---- figure 2: verdict per arm ---------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    labels = list(runs)
    ys = [runs[k]["strain_overrun_ratio"] for k in labels]
    bars = ax.bar(range(len(labels)), ys,
                  color=[colours[k] for k in labels], width=0.6)
    ax.axhline(1.0, color="#2c3e50", lw=1.0)
    ax.axhspan(1.0, 1.0 + TOL, color="#2c3e50", alpha=0.15)
    ax.text(len(labels) - 0.5, 1.0 + TOL / 2, "5 % band", fontsize=7,
            ha="right", va="center", color="#2c3e50")
    for b, k in zip(bars, labels):
        r = runs[k]
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.08,
                f"{r['strain_overrun_ratio']:.2f}x\n"
                f"{r['strain_budget_status']}",
                ha="center", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=7.5)
    ax.set_ylabel("delivered / requested strain")
    ax.set_title("Round 17 — delivered strain against the request\n"
                 "(same configuration, I = 0.3, request = 0.5)")
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    p2 = ROUND17 / "fig_ceiling_verdict.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    print(p1)
    print(p2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())