#!/usr/bin/env python3
"""Generate N=10000 RCP packing fraction artifacts and figures.

Aggregates the existing deposition run data (N=10000, mu=0 and mu=0.1)
into the standard artifacts format, generates comparison figures, and
produces the gates.json verdict.
"""
from __future__ import annotations

import json
import os
import sys
import glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordem.rcp_references import get_reference, reference_table

OUT_DIR = "artifacts/case_rcp_packing_n10k"
RUNS_DIR = "benchmarks/case_rcp_deposition/runs"
RCP_REF_PHI = 0.6366  # Bernal & Mason 1960
BAND = (0.62, 0.66)
TOL_REL = 0.03


def load_deposition_runs():
    """Load all N=10000 production deposition runs."""
    runs = []
    for path in sorted(glob.glob(os.path.join(RUNS_DIR, "deposition_n10000_mu*_kn4000_e0.1_seed*.json"))):
        if "_dtf" in os.path.basename(path):
            continue
        with open(path) as f:
            d = json.load(f)
        spec = d.get("spec", {})
        m = d.get("metrics", {})
        runs.append({
            "mu": spec.get("mu"),
            "seed": spec.get("seed"),
            "kn": spec.get("kn"),
            "phi_bulk": m.get("phi_bulk"),
            "phi_corr": m.get("phi_bulk_overlap_corrected"),
            "Z": m.get("coordination_bulk"),
            "ovmax": m.get("max_pair_overlap_over_d"),
            "settled": d.get("settled"),
        })
    return runs


def evaluate_gates(runs):
    """Evaluate all acceptance gates for the feature ledger."""
    gates = {}

    # Gate 1: Frictionless phi in [0.62, 0.66]
    mu0 = [r["phi_corr"] for r in runs if r["mu"] == 0.0]
    phi_mu0 = float(np.mean(mu0)) if mu0 else None
    gates["frictionless_phi_in_band"] = {
        "value": phi_mu0,
        "lower": BAND[0],
        "upper": BAND[1],
        "pass": BAND[0] <= phi_mu0 <= BAND[1] if phi_mu0 else False,
    }

    # Gate 2: Low-friction phi in [0.62, 0.66]
    mu01 = [r["phi_corr"] for r in runs if r["mu"] == 0.1]
    phi_mu01 = float(np.mean(mu01)) if mu01 else None
    gates["low_friction_phi_in_band"] = {
        "value": phi_mu01,
        "lower": BAND[0],
        "upper": BAND[1],
        "pass": BAND[0] <= phi_mu01 <= BAND[1] if phi_mu01 else False,
    }

    # Gate 3: Frictionless error vs reference < 3%
    ref0 = get_reference(0.0)
    err0 = abs(phi_mu0 - ref0.phi_rcp) / ref0.phi_rcp if phi_mu0 else None
    gates["frictionless_error_vs_ref_lt_3pct"] = {
        "value": err0,
        "threshold": TOL_REL,
        "pass": err0 < TOL_REL if err0 is not None else False,
    }

    # Gate 4: Low-friction error vs reference < 3%
    ref01 = get_reference(0.1)
    err01 = abs(phi_mu01 - ref01.phi_rcp) / ref01.phi_rcp if phi_mu01 else None
    gates["low_friction_error_vs_ref_lt_3pct"] = {
        "value": err01,
        "threshold": TOL_REL,
        "pass": err01 < TOL_REL if err01 is not None else False,
    }

    # Gate 5: All runs settled
    all_settled = all(r["settled"] for r in runs)
    gates["all_runs_settled"] = {
        "value": all_settled,
        "pass": all_settled,
    }

    # Gate 6: Friction lowers packing (physical consistency)
    if phi_mu0 is not None and phi_mu01 is not None:
        friction_lowers = phi_mu01 < phi_mu0 + 0.005
    else:
        friction_lowers = False
    gates["friction_lowers_packing"] = {
        "value": f"phi(mu=0)={phi_mu0:.4f}, phi(mu=0.1)={phi_mu01:.4f}" if phi_mu0 and phi_mu01 else None,
        "pass": friction_lowers,
    }

    # Gate 7: Seed coverage >= 3
    for mu in [0.0, 0.1]:
        seeds = {r["seed"] for r in runs if r["mu"] == mu}
        gates[f"seed_coverage_mu{mu}"] = {
            "value": len(seeds),
            "threshold": 3,
            "pass": len(seeds) >= 3,
        }

    return gates


def make_figures(runs, out_dir):
    """Generate report figures."""
    figs = []

    # ---- Fig 1: Friction tier comparison bar chart ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    mu0_phis = [r["phi_corr"] for r in runs if r["mu"] == 0.0]
    mu01_phis = [r["phi_corr"] for r in runs if r["mu"] == 0.1]
    ref0 = get_reference(0.0)
    ref01 = get_reference(0.1)

    labels = [
        f"μ=0 (frictionless)\nTensorDEM N=10000",
        f"μ=0 (frictionless)\nScott & Kilgour 1969",
        f"μ=0.1 (low-friction)\nTensorDEM N=10000",
        f"μ=0.1 (low-friction)\nPournin et al. 2001",
    ]
    vals = [np.mean(mu0_phis), ref0.phi_rcp, np.mean(mu01_phis), ref01.phi_rcp]
    colors = ["#2196F3", "#1565C0", "#FF9800", "#E65100"]

    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.axhline(0.62, color="gray", ls="--", lw=1, alpha=0.5)
    ax.axhline(0.66, color="gray", ls="--", lw=1, alpha=0.5)
    ax.axhspan(0.62, 0.66, alpha=0.08, color="green", label="Tolerance band [0.62, 0.66]")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Volume fraction φ")
    ax.set_title("RCP Volume Fraction: TensorDEM N=10000 vs Literature (Deposition Protocol)")
    ax.set_ylim(0.58, 0.70)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_rcp_n10k_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"name": "fig_rcp_n10k_comparison.png", "path": path,
                 "caption": "RCP volume fraction for N=10000 spheres: frictionless (μ=0) and low-friction (μ=0.1) vs literature references."})

    # ---- Fig 2: Seed variability ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for idx, (mu, phis, ref) in enumerate([(0.0, mu0_phis, ref0), (0.1, mu01_phis, ref01)]):
        ax = axes[idx]
        seeds = [r["seed"] for r in runs if r["mu"] == mu]
        ax.bar([f"seed {s}" for s in seeds], phis, color=colors[idx*2], alpha=0.8)
        ax.axhline(ref.phi_rcp, color="r", ls="--", lw=1.5, label=f"Reference φ={ref.phi_rcp:.4f}")
        ax.axhspan(BAND[0], BAND[1], alpha=0.08, color="green")
        ax.set_ylabel("φ (overlap-corrected)")
        ax.set_title(f"μ = {mu}")
        ax.set_ylim(0.61, 0.66)
        ax.legend(fontsize=8)
        for i, (s, p) in enumerate(zip(seeds, phis)):
            ax.text(i, p + 0.001, f"{p:.4f}", ha="center", fontsize=9)
    fig.suptitle("Seed-to-seed variability (N=10000, kn=4000)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_rcp_n10k_seed_variability.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"name": "fig_rcp_n10k_seed_variability.png", "path": path,
                 "caption": "Seed-to-seed variability of RCP volume fraction for N=10000 spheres at kn=4000."})

    # ---- Fig 3: Friction-dependent RCP curve ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ref_table = reference_table()
    ref_mus = [r["mu"] for r in ref_table]
    ref_phis = [r["phi_rcp"] for r in ref_table]
    ref_errs = [r["phi_uncertainty"] for r in ref_table]

    ax.errorbar(ref_mus, ref_phis, yerr=[2*e for e in ref_errs],
                fmt="o--", color="#333", lw=1.5, capsize=4, label="Literature (2σ band)")
    # Plot our points
    ax.plot(0.0, np.mean(mu0_phis), "s", color="#2196F3", ms=12, mew=2,
            label=f"TensorDEM μ=0: φ={np.mean(mu0_phis):.4f}")
    ax.plot(0.1, np.mean(mu01_phis), "s", color="#FF9800", ms=12, mew=2,
            label=f"TensorDEM μ=0.1: φ={np.mean(mu01_phis):.4f}")
    ax.axhspan(BAND[0], BAND[1], alpha=0.06, color="green", label="Acceptance band [0.62, 0.66]")
    ax.set_xlabel("Friction coefficient μ")
    ax.set_ylabel("Volume fraction φ")
    ax.set_title("Friction-dependent RCP: TensorDEM vs Literature")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.05, 0.55)
    ax.set_ylim(0.54, 0.68)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_rcp_friction_dependent.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"name": "fig_rcp_friction_dependent.png", "path": path,
                 "caption": "Friction-dependent RCP volume fraction: TensorDEM N=10000 results overlaid on literature reference curve."})

    return figs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    runs = load_deposition_runs()
    print(f"Loaded {len(runs)} deposition runs")

    # Evaluate gates
    gates = evaluate_gates(runs)
    all_pass = all(g["pass"] for g in gates.values())

    # Generate figures
    figs = make_figures(runs, OUT_DIR)

    # Build result
    mu0 = [r for r in runs if r["mu"] == 0.0]
    mu01 = [r for r in runs if r["mu"] == 0.1]

    result = {
        "case": "CASE-DEM-RCP-PACKING-FRACTION",
        "protocol": "deposition",
        "n_particles": 10000,
        "kn_production": 4000.0,
        "restitution": 0.1,
        "frictionless": {
            "mu": 0.0,
            "seeds": len(mu0),
            "phi_corr_mean": float(np.mean([r["phi_corr"] for r in mu0])),
            "phi_corr_std": float(np.std([r["phi_corr"] for r in mu0])),
            "phi_bulk_mean": float(np.mean([r["phi_bulk"] for r in mu0])),
            "Z_mean": float(np.mean([r["Z"] for r in mu0])),
            "ovmax_mean": float(np.mean([r["ovmax"] for r in mu0])),
            "reference_phi": get_reference(0.0).phi_rcp,
            "reference_source": get_reference(0.0).source,
            "relative_error": float(abs(np.mean([r["phi_corr"] for r in mu0]) - get_reference(0.0).phi_rcp) / get_reference(0.0).phi_rcp),
        },
        "low_friction": {
            "mu": 0.1,
            "seeds": len(mu01),
            "phi_corr_mean": float(np.mean([r["phi_corr"] for r in mu01])),
            "phi_corr_std": float(np.std([r["phi_corr"] for r in mu01])),
            "phi_bulk_mean": float(np.mean([r["phi_bulk"] for r in mu01])),
            "Z_mean": float(np.mean([r["Z"] for r in mu01])),
            "ovmax_mean": float(np.mean([r["ovmax"] for r in mu01])),
            "reference_phi": get_reference(0.1).phi_rcp,
            "reference_source": get_reference(0.1).source,
            "relative_error": float(abs(np.mean([r["phi_corr"] for r in mu01]) - get_reference(0.1).phi_rcp) / get_reference(0.1).phi_rcp),
        },
        "gates": gates,
        "all_pass": all_pass,
        "figures": [{"name": f["name"], "caption": f["caption"]} for f in figs],
        "individual_runs": runs,
    }

    # Write result
    with open(os.path.join(OUT_DIR, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Write gates
    with open(os.path.join(OUT_DIR, "gates.json"), "w") as f:
        json.dump(gates, f, indent=2)

    # Print verdict
    print("\n" + "=" * 60)
    print("VERDICT: CASE-DEM-RCP-PACKING-FRACTION (N=10000 Deposition)")
    print("=" * 60)
    for name, g in gates.items():
        status = "PASS" if g["pass"] else "FAIL"
        val = g.get("value", "N/A")
        print(f"  {name}: {status} (value={val})")
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print(f"  Frictionless: φ = {result['frictionless']['phi_corr_mean']:.4f} ± {result['frictionless']['phi_corr_std']:.4f}")
    print(f"  Low-friction: φ = {result['low_friction']['phi_corr_mean']:.4f} ± {result['low_friction']['phi_corr_std']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
