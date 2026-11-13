#!/usr/bin/env python3
"""Generate figures for the friction-dependent RCP analysis.

Produces:
- figures/fig_rcp_vs_friction.png: RCP reference curve + measured values
- figures/fig_error_comparison.png: Old vs new reference errors

Usage: PYTHONPATH=src python docs/gen_rcp_friction_figures.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.rcp_references import get_reference, reference_table

FIGDIR = os.path.join(ROOT, "benchmarks", "case_rcp_deposition", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Measured values from existing runs
# ---------------------------------------------------------------------------
MEASURED = {
    "production": [
        {"mu": 0.0, "phi": 0.64237, "std": 0.00022, "label": "mu=0 production (N=10000)"},
        {"mu": 0.1, "phi": 0.62628, "std": 0.00040, "label": "mu=0.1 production (N=10000)"},
    ],
    "hard_sphere": [
        {"mu": 0.0, "phi": 0.63644, "label": "mu=0 hard-sphere limit"},
        {"mu": 0.1, "phi": 0.61705, "label": "mu=0.1 hard-sphere limit"},
    ],
    "trend": [
        {"mu": 0.3, "phi": 0.60454, "label": "mu=0.3 trend (N=2000)"},
        {"mu": 0.5, "phi": 0.59310, "label": "mu=0.5 trend (N=2000)"},
    ],
}

# ---------------------------------------------------------------------------
# Figure 1: RCP vs Friction
# ---------------------------------------------------------------------------
def fig_rcp_vs_friction():
    tbl = reference_table()
    # Get unique mu values (there are two mu=0 entries)
    mu_unique = sorted({r["mu"] for r in tbl})
    # For plotting, use the first entry for each mu
    ref_by_mu = {}
    for r in tbl:
        if r["mu"] not in ref_by_mu:
            ref_by_mu[r["mu"]] = r

    mus_ref = [ref_by_mu[mu]["mu"] for mu in mu_unique]
    phis_ref = [ref_by_mu[mu]["phi_rcp"] for mu in mu_unique]
    bands_lo = [ref_by_mu[mu]["band_2sigma"][0] for mu in mu_unique]
    bands_hi = [ref_by_mu[mu]["band_2sigma"][1] for mu in mu_unique]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Reference curve (interpolated)
    mus_fine = np.linspace(0, 1.0, 100)
    phis_fine = [get_reference(mu, method="interpolate").phi_rcp for mu in mus_fine]
    bands_lo_fine = [get_reference(mu, method="interpolate").band[0] for mu in mus_fine]
    bands_hi_fine = [get_reference(mu, method="interpolate").band[1] for mu in mus_fine]

    ax.fill_between(mus_fine, bands_lo_fine, bands_hi_fine, alpha=0.15, color="blue",
                    label="2σ uncertainty band (interpolated)")
    ax.plot(mus_fine, phis_fine, "b-", linewidth=2, label="RCP reference (interpolated)")

    # Reference points
    ax.errorbar(mus_ref, phis_ref,
                yerr=[[(phis_ref[i] - bands_lo[i]) for i in range(len(mus_ref))],
                      [(bands_hi[i] - phis_ref[i]) for i in range(len(mus_ref))]],
                fmt="bs", markersize=8, capsize=4, label="Reference points (literature)")

    # Measured values
    colors = {"production": "green", "hard_sphere": "red", "trend": "orange"}
    markers = {"production": "o", "hard_sphere": "^", "trend": "x"}
    for category, points in MEASURED.items():
        for p in points:
            yerr = p.get("std", 0)
            ax.errorbar(p["mu"], p["phi"], yerr=yerr, fmt=markers[category],
                       color=colors[category], markersize=10, capsize=3,
                       label=p["label"])

    # Ledger band
    ax.axhspan(0.62, 0.66, alpha=0.1, color="gray", label="Ledger band [0.62, 0.66]")

    ax.set_xlabel("Inter-particle friction coefficient μ", fontsize=12)
    ax.set_ylabel("Volume fraction φ", fontsize=12)
    ax.set_title("RCP Volume Fraction vs Friction Coefficient\n"
                 "TensorDEM deposition results vs literature references", fontsize=13)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(0.54, 0.67)
    ax.grid(True, alpha=0.3)

    path = os.path.join(FIGDIR, "fig_rcp_vs_friction.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 2: Error Comparison (Old vs New Reference)
# ---------------------------------------------------------------------------
def fig_error_comparison():
    categories = ["μ=0 production", "μ=0.1 production",
                  "μ=0 hard-sphere", "μ=0.1 hard-sphere",
                  "μ=0.3 trend", "μ=0.5 trend"]

    measured_vals = [0.64237, 0.62628, 0.63644, 0.61705, 0.60454, 0.59310]
    old_refs = [0.6366, 0.6366, 0.6366, 0.6366, 0.6366, 0.6366]
    new_refs = [0.6366, 0.6250, 0.6366, 0.6250, 0.6000, 0.5850]

    old_errors = [(m / r - 1) * 100 for m, r in zip(measured_vals, old_refs)]
    new_errors = [(m / r - 1) * 100 for m, r in zip(measured_vals, new_refs)]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, old_errors, width, label="Old (single ref φ=0.6366)",
                   color="salmon", edgecolor="black")
    bars2 = ax.bar(x + width/2, new_errors, width, label="New (friction-dependent ref)",
                   color="steelblue", edgecolor="black")

    # 3% threshold lines
    ax.axhline(y=3, color="red", linestyle="--", alpha=0.5, label="±3% threshold")
    ax.axhline(y=-3, color="red", linestyle="--", alpha=0.5)
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax.set_ylabel("Relative error (%)", fontsize=12)
    ax.set_title("Error Comparison: Single Reference vs Friction-Dependent Reference\n"
                 "Negative = measured below reference", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars1, old_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:+.2f}%", ha="center", va="bottom", fontsize=8,
                color="red" if abs(val) > 3 else "black")
    for bar, val in zip(bars2, new_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:+.2f}%", ha="center", va="bottom", fontsize=8,
                color="red" if abs(val) > 3 else "black")

    path = os.path.join(FIGDIR, "fig_rcp_error_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


if __name__ == "__main__":
    fig_rcp_vs_friction()
    fig_error_comparison()
    print("Done.")
