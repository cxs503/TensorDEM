"""Generate comprehensive report for DEM ice resistance vs Lindqvist validation."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordem.ice_straight_ahead_final import DEMStraightAheadConfig, run_dem_straight_ahead


def main():
    """Generate full report with figures."""
    print("=" * 80)
    print("DEM Ice Resistance vs Lindqvist (1989) - Comprehensive Report")
    print("=" * 80)

    # Run campaign
    thicknesses_m = [0.3, 0.5, 0.8]
    concentrations = [0.6, 0.8]
    seeds = [42, 43, 44]

    all_results = []
    for h in thicknesses_m:
        for C in concentrations:
            for seed in seeds:
                config = DEMStraightAheadConfig(
                    ice_thickness_m=h,
                    concentration=C,
                    seed=seed,
                    domain_length_m=100.0,
                    domain_width_m=40.0,
                    travel_distance_m=100.0,
                    dt_s=0.02,
                )
                result = run_dem_straight_ahead(config)
                all_results.append(result)

    # Save results
    output_dir = Path(__file__).parent.parent / "artifacts" / "ice_straight_ahead_final"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dict = {
        "campaign": {
            "thicknesses_m": thicknesses_m,
            "concentrations": concentrations,
            "seeds": seeds,
            "n_cases": len(all_results),
        },
        "results": [r.as_dict() for r in all_results],
    }

    json_path = output_dir / "campaign_results.json"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Generate figures
    print("\nGenerating figures...")

    # Figure 1: DEM vs Lindqvist bar chart
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("DEM Ice Resistance vs Lindqvist (1989)", fontsize=16, fontweight="bold")

    for idx, h in enumerate(thicknesses_m):
        for jdx, C in enumerate(concentrations):
            ax = axes[jdx, idx]
            case_results = [
                r for r in all_results
                if r.config.ice_thickness_m == h and r.config.concentration == C
            ]
            
            seeds_list = range(len(case_results))
            dem_resistances = [r.mean_resistance_N / 1e3 for r in case_results]
            lindqvist = case_results[0].lindqvist_total_N / 1e3

            ax.bar(seeds_list, dem_resistances, label="DEM", alpha=0.7, color="blue")
            ax.axhline(lindqvist, color="red", linestyle="--", label="Lindqvist", linewidth=2)
            ax.set_xlabel("Seed")
            ax.set_ylabel("Resistance [kN]")
            ax.set_title(f"h={h}m, C={C}")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1_path = output_dir / "fig1_dem_vs_lindqvist.png"
    plt.savefig(fig1_path, dpi=150)
    print(f"Figure 1 saved to: {fig1_path}")

    # Figure 2: Error distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    all_ratios = [r.ratio_dem_to_lindqvist for r in all_results]
    ax.hist(all_ratios, bins=15, alpha=0.7, color="blue", edgecolor="black")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Perfect match")
    ax.axvline(0.85, color="orange", linestyle=":", linewidth=2, label="±15% bounds")
    ax.axvline(1.15, color="orange", linestyle=":", linewidth=2)
    ax.set_xlabel("DEM / Lindqvist Ratio")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of DEM/Lindqvist Ratios (18 cases)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig2_path = output_dir / "fig2_ratio_distribution.png"
    plt.savefig(fig2_path, dpi=150)
    print(f"Figure 2 saved to: {fig2_path}")

    # Figure 3: Error vs thickness and concentration
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Relative Error: DEM vs Lindqvist", fontsize=14, fontweight="bold")

    # By thickness
    ax = axes[0]
    for C in concentrations:
        errors_by_h = []
        for h in thicknesses_m:
            case_results = [
                r for r in all_results
                if r.config.ice_thickness_m == h and r.config.concentration == C
            ]
            errors_by_h.append(np.mean([r.error_percent for r in case_results]))
        ax.plot(thicknesses_m, errors_by_h, "o-", label=f"C={C}")
    ax.set_xlabel("Ice thickness [m]")
    ax.set_ylabel("Error [%]")
    ax.set_title("Error vs Ice Thickness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(15, color="red", linestyle="--", label="15% target", alpha=0.5)

    # By concentration
    ax = axes[1]
    for h in thicknesses_m:
        errors_by_C = []
        for C in concentrations:
            case_results = [
                r for r in all_results
                if r.config.ice_thickness_m == h and r.config.concentration == C
            ]
            errors_by_C.append(np.mean([r.error_percent for r in case_results]))
        ax.plot(concentrations, errors_by_C, "o-", label=f"h={h}m")
    ax.set_xlabel("Concentration [-]")
    ax.set_ylabel("Error [%]")
    ax.set_title("Error vs Concentration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(15, color="red", linestyle="--", label="15% target", alpha=0.5)

    plt.tight_layout()
    fig3_path = output_dir / "fig3_error_analysis.png"
    plt.savefig(fig3_path, dpi=150)
    print(f"Figure 3 saved to: {fig3_path}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    all_errors = [r.error_percent for r in all_results]
    print(f"Total cases: {len(all_results)}")
    print(f"Mean error: {np.mean(all_errors):.1f}%")
    print(f"Std error: {np.std(all_errors):.1f}%")
    print(f"Min error: {np.min(all_errors):.1f}%")
    print(f"Max error: {np.max(all_errors):.1f}%")
    print(f"Cases within 15%: {sum(1 for e in all_errors if e <= 15)}/{len(all_results)}")

    # Detailed table
    print("\nDetailed results by case:")
    for h in thicknesses_m:
        for C in concentrations:
            case_results = [
                r for r in all_results
                if r.config.ice_thickness_m == h and r.config.concentration == C
            ]
            errors = [r.error_percent for r in case_results]
            ratios = [r.ratio_dem_to_lindqvist for r in case_results]
            dem_mean = np.mean([r.mean_resistance_N for r in case_results])
            lind = case_results[0].lindqvist_total_N
            print(f"  h={h}m, C={C}: DEM={dem_mean/1e3:.1f}kN, Lind={lind/1e3:.1f}kN, "
                  f"ratio={np.mean(ratios):.3f}, error={np.mean(errors):.1f}%")

    print("\n" + "=" * 80)
    print("Report generation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
