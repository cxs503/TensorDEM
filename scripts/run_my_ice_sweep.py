#!/usr/bin/env python3
"""Run the MY ice parameter sweep and generate report figures.

Sweep parameters:
- MY fraction: 0-100% (step 10%)
- Ice thickness: 0.5-3.0 m (step 0.25 m)
- Ship speed: 2-10 kn (step 1 kn)

Outputs:
- runs/my_ice_sweep_results.json: full sweep data
- artifacts/my_ice_sweep_summary.csv: summary table
- artifacts/fig_my_resistance_vs_thickness.png
- artifacts/fig_my_resistance_vs_speed.png
- artifacts/fig_my_resistance_vs_my_fraction.png
- artifacts/fig_my_component_split.png
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensordem.level_ice import formula as F
from tensordem.level_ice.multiyear_ice import (
    FY_DEFAULT, MY_DEFAULT, IceAgeProperties, MixedIceField,
    lindqvist_my_components, lindqvist_my_sweep, my_correction_factor,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def run_full_sweep():
    my_fractions = [i / 10.0 for i in range(0, 11)]
    thicknesses = [0.5 + 0.25 * i for i in range(11)]
    speeds_kn = list(range(2, 11))
    return lindqvist_my_sweep(
        F.ICEBREAKER, my_fractions=my_fractions,
        thicknesses_m=thicknesses, speeds_kn=speeds_kn,
    )


def save_results(results):
    json_path = RUNS_DIR / "my_ice_sweep_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {json_path}")

    csv_path = OUTPUT_DIR / "my_ice_sweep_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "my_fraction", "thickness_m", "speed_kn", "speed_m_s",
            "k_MY", "k_effective",
            "crushing_N", "bending_corrected_N", "submersion_corrected_N",
            "added_drag_N", "total_N",
            "crushing_fraction", "bending_fraction", "submersion_fraction", "drag_fraction",
        ])
        for r in results:
            sp = r["sweep_params"]
            writer.writerow([
                sp["my_fraction"], sp["thickness_m"], sp["speed_kn"], sp["speed_m_s"],
                r["k_MY"], r["k_effective"],
                r["crushing_N"], r["bending_corrected_N"], r["submersion_corrected_N"],
                r["added_drag_N"], r["total_N"],
                r["crushing_fraction"], r["bending_fraction"], r["submersion_fraction"], r["drag_fraction"],
            ])
    print(f"Saved summary to {csv_path}")


def generate_figures(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    data = {}
    for r in results:
        sp = r["sweep_params"]
        key = (sp["my_fraction"], sp["thickness_m"], sp["speed_kn"])
        data[key] = r

    # Fig 1: Resistance vs thickness
    fig, ax = plt.subplots(figsize=(10, 6))
    speed_kn = 4.0
    for my_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        thicknesses = sorted(set(t for (mf, t, sk) in data if mf == my_frac and sk == speed_kn))
        totals = [data[(my_frac, t, speed_kn)]["total_N"] / 1e3 for t in thicknesses]
        ax.plot(thicknesses, totals, "o-", label=f"MY={my_frac:.0%}", linewidth=2)
    ax.set_xlabel("Ice Thickness [m]", fontsize=12)
    ax.set_ylabel("Total Resistance [kN]", fontsize=12)
    ax.set_title(f"Lindqvist MY-Extended: Resistance vs Ice Thickness (V={speed_kn} kn)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path1 = OUTPUT_DIR / "fig_my_resistance_vs_thickness.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    figures.append({
        "name": "resistance_vs_thickness",
        "caption": f"Total resistance vs ice thickness at V={speed_kn} kn for varying MY ice fractions. "
                   "MY ice (sigma_c=7.5 MPa) produces 2-3x higher resistance than FY ice (sigma_c=1.5 MPa).",
        "png_path": str(path1),
    })

    # Fig 2: Resistance vs speed
    fig, ax = plt.subplots(figsize=(10, 6))
    thickness = 1.5
    for my_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        speeds = sorted(set(sk for (mf, t, sk) in data if mf == my_frac and t == thickness))
        totals = [data[(my_frac, thickness, sk)]["total_N"] / 1e3 for sk in speeds]
        ax.plot(speeds, totals, "o-", label=f"MY={my_frac:.0%}", linewidth=2)
    ax.set_xlabel("Ship Speed [kn]", fontsize=12)
    ax.set_ylabel("Total Resistance [kN]", fontsize=12)
    ax.set_title(f"Lindqvist MY-Extended: Resistance vs Speed (h={thickness} m)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path2 = OUTPUT_DIR / "fig_my_resistance_vs_speed.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    figures.append({
        "name": "resistance_vs_speed",
        "caption": f"Total resistance vs ship speed at h={thickness} m. Speed dependence is quadratic (drag term).",
        "png_path": str(path2),
    })

    # Fig 3: Resistance vs MY fraction
    fig, ax = plt.subplots(figsize=(10, 6))
    thickness = 1.5
    for speed_kn in [2, 4, 6, 8, 10]:
        my_fracs = sorted(set(mf for (mf, t, sk) in data if t == thickness and sk == speed_kn))
        totals = [data[(mf, thickness, speed_kn)]["total_N"] / 1e3 for mf in my_fracs]
        ax.plot(my_fracs, totals, "o-", label=f"V={speed_kn} kn", linewidth=2)
    ax.set_xlabel("MY Ice Fraction", fontsize=12)
    ax.set_ylabel("Total Resistance [kN]", fontsize=12)
    ax.set_title(f"Lindqvist MY-Extended: Resistance vs MY Fraction (h={thickness} m)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path3 = OUTPUT_DIR / "fig_my_resistance_vs_my_fraction.png"
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    figures.append({
        "name": "resistance_vs_my_fraction",
        "caption": f"Total resistance vs MY ice fraction at h={thickness} m. "
                   "Resistance increases monotonically with MY fraction due to higher ice strength.",
        "png_path": str(path3),
    })

    # Fig 4: Component split
    fig, ax = plt.subplots(figsize=(10, 6))
    thickness = 1.5
    speed_kn = 4.0
    my_fracs = sorted(set(mf for (mf, t, sk) in data if t == thickness and sk == speed_kn))
    crushing = [data[(mf, thickness, speed_kn)]["crushing_N"] / 1e3 for mf in my_fracs]
    bending = [data[(mf, thickness, speed_kn)]["bending_corrected_N"] / 1e3 for mf in my_fracs]
    submersion = [data[(mf, thickness, speed_kn)]["submersion_corrected_N"] / 1e3 for mf in my_fracs]
    drag = [data[(mf, thickness, speed_kn)]["added_drag_N"] / 1e3 for mf in my_fracs]

    x = np.arange(len(my_fracs))
    width = 0.2
    ax.bar(x - 1.5*width, crushing, width, label="Crushing", color="#e74c3c")
    ax.bar(x - 0.5*width, bending, width, label="Bending (corrected)", color="#3498db")
    ax.bar(x + 0.5*width, submersion, width, label="Submersion (corrected)", color="#2ecc71")
    ax.bar(x + 1.5*width, drag, width, label="Added Drag", color="#f39c12")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{mf:.0%}" for mf in my_fracs])
    ax.set_xlabel("MY Ice Fraction", fontsize=12)
    ax.set_ylabel("Resistance Component [kN]", fontsize=12)
    ax.set_title(f"Component Breakdown (h={thickness} m, V={speed_kn} kn)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path4 = OUTPUT_DIR / "fig_my_component_split.png"
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    figures.append({
        "name": "component_split",
        "caption": f"Resistance component breakdown at h={thickness} m, V={speed_kn} kn. "
                   "Bending and submersion terms are scaled by k_MY; crushing is strength-driven.",
        "png_path": str(path4),
    })

    return figures


def main():
    print("Running MY ice parameter sweep...")
    results = run_full_sweep()
    print(f"Generated {len(results)} sweep cases")

    save_results(results)
    figures = generate_figures(results)
    print(f"Generated {len(figures)} figures")

    # Print key statistics
    fy_only = [r for r in results if r["sweep_params"]["my_fraction"] == 0.0]
    my_only = [r for r in results if r["sweep_params"]["my_fraction"] == 1.0]

    fy_avg = np.mean([r["total_N"] for r in fy_only])
    my_avg = np.mean([r["total_N"] for r in my_only])
    print(f"\nKey Statistics:")
    print(f"  FY-only average resistance: {fy_avg/1e3:.1f} kN")
    print(f"  MY-only average resistance: {my_avg/1e3:.1f} kN")
    print(f"  MY/FY ratio: {my_avg/fy_avg:.2f}")

    # k_MY range
    k_mys = [r["k_MY"] for r in results]
    print(f"  k_MY range: [{min(k_mys):.3f}, {max(k_mys):.3f}]")

    return results, figures


if __name__ == "__main__":
    main()
