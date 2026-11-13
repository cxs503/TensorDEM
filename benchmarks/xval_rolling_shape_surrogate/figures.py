"""Figures for the rolling-friction shape-surrogate cross-validation.

Every number is read from ``artifacts/xval_rolling_shape_surrogate/result.json``;
nothing is retyped.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("artifacts/xval_rolling_shape_surrogate")


def main() -> int:
    res = json.loads((OUT / "result.json").read_text())
    surf = res["surface"]
    mu = np.array([p["mu_r"] for p in surf], dtype=float)
    th = np.array([p["theta_deg"] for p in surf], dtype=float)
    err = np.array([p["theta_stderr_deg"] for p in surf], dtype=float)
    lit = res["literature"]
    lit_mu = np.array(lit["mu_r"], dtype=float)
    lit_th = np.array(lit["theta_deg"], dtype=float)
    fit = res["saturation_fit"]
    grid = np.linspace(0.0, max(0.5, float(mu.max())), 200)
    fit_curve = fit["theta_inf_deg"] - fit["amplitude_deg"] * np.exp(-grid / fit["mu_c"])

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8))

    ax = axes[0]
    ax.errorbar(mu, th, yerr=err, fmt="o", color="C0", capsize=3,
                label="DEM sandpile, reference condition")
    ax.plot(grid, fit_curve, "-", color="C0", lw=1.2, label="saturating fit")
    ax.plot(lit_mu, lit_th, "s--", color="C3",
            label="Wensrich & Katterfeld (2012) correspondence")
    off = res["off_reference_points"]
    if off:
        ax.plot([p["mu_r"] for p in off], [p["theta_deg"] for p in off], "D",
                mfc="none", color="C2", label="held-out conditions (N/seed/dt)")
    ax.set_xlabel(r"rolling-friction coefficient $\mu_r$")
    ax.set_ylabel("angle of repose [deg]")
    ax.set_title("calibration surface vs literature")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    rows = res["literature_comparison"]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["theta_ref_deg"] for r in rows], 0.4, label="literature", color="C3")
    ax.bar(x + 0.2, [r["theta_got_deg"] for r in rows], 0.4, label="DEM (this work)", color="C0")
    for i, r in enumerate(rows):
        ax.text(i, max(r["theta_ref_deg"], r["theta_got_deg"]) + 0.6,
                f"{100 * r['rel_error']:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x, [f"mu_r={r['mu_r']:g}" for r in rows])
    ax.set_ylabel("angle of repose [deg]")
    ax.set_title("literature anchor points (gate 15 %)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    ax.plot(mu, th, "o", color="C7", label="measured surface")
    for i, row in enumerate(res["inversions"]):
        if row.get("status") == "unattainable":
            ax.axhline(row["reference_theta_deg"], color=f"C{i}", ls=":", lw=1.0)
            ax.text(mu.max(), row["reference_theta_deg"], " unattainable", fontsize=8,
                    color=f"C{i}", va="bottom", ha="right")
            continue
        ax.axvline(row["mu_r_star"], color=f"C{i}", lw=1.2,
                   label=(r"$\mu_r^*$=%.3f for %g deg"
                          % (row["mu_r_star"], row["reference_theta_deg"])))
    ax.set_xlabel(r"$\mu_r$")
    ax.set_ylabel("angle of repose [deg]")
    ax.set_title("surrogate inversion")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("XVAL-DEM-ROLLING-FRICTION-AS-SHAPE-SURROGATE — sandpile calibration surface",
                 fontsize=11)
    fig.tight_layout()
    out = OUT / "fig_shape_surrogate_calibration.png"
    fig.savefig(out, dpi=140)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())