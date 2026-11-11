"""Generate figures for DEM brash ice full-scale validation.

Campaign node p472w2-N04.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensordem.brash_ice_rheology.fullscale_validation import (
    three_way_comparison, sobol_sensitivity_analysis,
    monte_carlo_uncertainty, FULLSCALE_DATASET, MT_UIKKU_HULL,
    OperatingPoint, dem_resistance
)

OUTPUT_DIR = Path(__file__).parent.parent / "benchmarks" / "verified" / "dem_brash_ice_fullscale"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fig1_three_way_comparison():
    """Figure 1: Three-way comparison (DEM vs Lindqvist vs measured)."""
    comparison = three_way_comparison()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Bar chart of resistances
    ax = axes[0]
    ids = [c["id"] for c in comparison]
    x = np.arange(len(ids))
    width = 0.25
    
    bars1 = ax.bar(x - width, [c["R_DEM_kN"] for c in comparison], width, label="DEM (TensorDEM)", color='#2196F3')
    bars2 = ax.bar(x, [c["R_Lindqvist_kN"] for c in comparison], width, label="Lindqvist (analytical)", color='#FF9800')
    bars3 = ax.bar(x + width, [c["R_measured_kN"] for c in comparison], width, label="Measured (full-scale)", color='#4CAF50')
    
    ax.set_xlabel("Operating Point")
    ax.set_ylabel("Resistance [kN]")
    ax.set_title("(a) Three-Way Resistance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    # Right: Relative errors
    ax = axes[1]
    err_dem_lind = [c["err_DEM_vs_Lindqvist"] * 100 for c in comparison]
    err_dem_meas = [c["err_DEM_vs_measured"] * 100 for c in comparison]
    
    ax.bar(x - 0.15, err_dem_lind, 0.3, label="DEM vs Lindqvist", color='#FF5722')
    ax.bar(x + 0.15, err_dem_meas, 0.3, label="DEM vs Measured", color='#9C27B0')
    ax.axhline(y=15, color='r', linestyle='--', alpha=0.5, label='15% threshold')
    ax.axhline(y=-15, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=20, color='b', linestyle=':', alpha=0.5, label='20% threshold')
    ax.axhline(y=-20, color='b', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='k', linewidth=0.5)
    
    ax.set_xlabel("Operating Point")
    ax.set_ylabel("Relative Error [%]")
    ax.set_title("(b) Relative Errors")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_three_way_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig1_three_way_comparison.png")


def fig2_sobol_sensitivity():
    """Figure 2: Sobol sensitivity indices."""
    sobol = sobol_sensitivity_analysis(n_samples=500)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    params = list(sobol["S1"].keys())
    param_labels = {
        "mu_ii": "μ_ii\n(ice-ice friction)",
        "mu_is": "μ_is\n(ice-steel friction)",
        "sigma_b": "σ_b\n(bond strength)",
        "rho_ice": "ρ_ice\n(ice density)",
    }
    labels = [param_labels[p] for p in params]
    
    # Left: First-order indices
    ax = axes[0]
    s1_values = [sobol["S1"][p] for p in params]
    colors = ['#F44336' if v > 0.4 else '#FF9800' if v > 0.1 else '#4CAF50' for v in s1_values]
    ax.bar(range(len(params)), s1_values, color=colors)
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("First-Order Index S_i")
    ax.set_title("(a) First-Order Sensitivity")
    ax.axhline(y=0.4, color='r', linestyle='--', alpha=0.5, label='Dominance threshold')
    ax.set_ylim(-0.1, 1.0)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    # Right: Total indices
    ax = axes[1]
    st_values = [sobol["ST"][p] for p in params]
    colors = ['#F44336' if v > 0.4 else '#FF9800' if v > 0.1 else '#4CAF50' for v in st_values]
    ax.bar(range(len(params)), st_values, color=colors)
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Total Index S_Ti")
    ax.set_title("(b) Total Sensitivity (incl. interactions)")
    ax.axhline(y=0.4, color='r', linestyle='--', alpha=0.5, label='Dominance threshold')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    # Add sum annotation
    ax.text(0.95, 0.95, f"ΣS_Ti = {sobol['ST_sum']:.4f}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_sobol_sensitivity.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig2_sobol_sensitivity.png")


def fig3_uncertainty_quantification():
    """Figure 3: Monte Carlo uncertainty quantification."""
    # Run MC for a subset of points
    mc_results = []
    for pt in FULLSCALE_DATASET:
        mc = monte_carlo_uncertainty(pt, n_samples=500)
        mc_results.append(mc)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: CI plot
    ax = axes[0]
    ids = [mc["point_id"] for mc in mc_results]
    x = np.arange(len(ids))
    
    means = [mc["R_mean_kN"] for mc in mc_results]
    ci_lo = [mc["R_CI95_lo_kN"] for mc in mc_results]
    ci_hi = [mc["R_CI95_hi_kN"] for mc in mc_results]
    measured = [mc["R_measured_kN"] for mc in mc_results]
    
    ax.errorbar(x, means, yerr=[np.array(means) - np.array(ci_lo),
                                 np.array(ci_hi) - np.array(means)],
                fmt='o', color='#2196F3', capsize=5, label='DEM prediction (95% CI)')
    ax.scatter(x, measured, marker='^', color='red', s=80, zorder=5, label='Measured')
    
    # Mark coverage
    for i, mc in enumerate(mc_results):
        if mc["CI95_covers_measured"]:
            ax.scatter(i, mc["R_measured_kN"], marker='^', color='green', s=80, zorder=6)
        else:
            ax.scatter(i, mc["R_measured_kN"], marker='^', color='orange', s=80, zorder=6)
    
    ax.set_xlabel("Operating Point")
    ax.set_ylabel("Resistance [kN]")
    ax.set_title("(a) 95% Confidence Intervals")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Right: Coverage summary
    ax = axes[1]
    n_cover = sum(1 for mc in mc_results if mc["CI95_covers_measured"])
    n_total = len(mc_results)
    coverage = n_cover / n_total * 100
    
    # Pie chart
    sizes = [n_cover, n_total - n_cover]
    labels_pie = [f'Covered\n({n_cover}/{n_total})', f'Not covered\n({n_total-n_cover}/{n_total})']
    colors_pie = ['#4CAF50', '#F44336']
    ax.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 11})
    ax.set_title(f"(b) CI_95 Coverage = {coverage:.1f}%")
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_uncertainty_quantification.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig3_uncertainty_quantification.png")


def fig4_speed_sweep():
    """Figure 4: Speed sweep showing DEM vs Lindqvist for different conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    speeds = np.linspace(1.0, 6.0, 20)
    
    # Left: Different ice thicknesses
    ax = axes[0]
    for H, color, ls in [(0.5, '#4CAF50', '-'), (1.0, '#2196F3', '-'), (1.5, '#FF9800', '-'), (2.0, '#F44336', '-')]:
        r_dem = []
        r_lind = []
        for v in speeds:
            pt = OperatingPoint("SWEEP", MT_UIKKU_HULL, H, 0.6, v, 0)
            r_dem.append(dem_resistance(pt))
            from tensordem.brash_ice_rheology.fullscale_validation import lindqvist_resistance
            r_lind.append(lindqvist_resistance(pt))
        
        ax.plot(speeds, r_dem, color=color, linestyle=ls, linewidth=2, label=f'DEM H={H}m')
        ax.plot(speeds, r_lind, color=color, linestyle='--', linewidth=1.5, alpha=0.6, label=f'Lindqvist H={H}m')
    
    ax.set_xlabel("Ship Speed [m/s]")
    ax.set_ylabel("Resistance [kN]")
    ax.set_title("(a) Speed Sweep - Different Ice Thicknesses")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    
    # Right: Different concentrations
    ax = axes[1]
    for C, color, ls in [(0.4, '#4CAF50', '-'), (0.5, '#2196F3', '-'), (0.6, '#FF9800', '-'), (0.8, '#F44336', '-')]:
        r_dem = []
        r_lind = []
        for v in speeds:
            pt = OperatingPoint("SWEEP", MT_UIKKU_HULL, 1.0, C, v, 0)
            r_dem.append(dem_resistance(pt))
            r_lind.append(lindqvist_resistance(pt))
        
        ax.plot(speeds, r_dem, color=color, linestyle=ls, linewidth=2, label=f'DEM C={C}')
        ax.plot(speeds, r_lind, color=color, linestyle='--', linewidth=1.5, alpha=0.6, label=f'Lindqvist C={C}')
    
    ax.set_xlabel("Ship Speed [m/s]")
    ax.set_ylabel("Resistance [kN]")
    ax.set_title("(b) Speed Sweep - Different Concentrations")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_speed_sweep.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved fig4_speed_sweep.png")


if __name__ == "__main__":
    print("Generating figures for DEM brash ice full-scale validation...")
    fig1_three_way_comparison()
    fig2_sobol_sensitivity()
    fig3_uncertainty_quantification()
    fig4_speed_sweep()
    print("Done!")
