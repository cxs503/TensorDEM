#!/usr/bin/env python3
"""Run scale ladder study and generate paper-quality report.

This script runs the full scale ladder study (n_floe=20/60/200/600, 5 seeds each)
and generates:
1. JSON results file
2. Paper-quality figures
3. Comprehensive report
"""
import sys
sys.path.insert(0, 'src')

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tensordem.ice_ridge_voxel_porosity import (
    run_scale_ladder_study,
    VoxelPorosityConfig,
)

def main():
    print("=" * 80)
    print("Ice Ridge Voxel Porosity: Scale Ladder Study")
    print("=" * 80)
    
    # Configuration
    n_floe_list = [20, 60, 200, 600]
    n_seeds = 5
    base_seed = 20261110
    
    print(f"\nRunning scale ladder study:")
    print(f"  n_floe: {n_floe_list}")
    print(f"  seeds per level: {n_seeds}")
    print(f"  total simulations: {len(n_floe_list) * n_seeds}")
    
    # Run study
    voxel_config = VoxelPorosityConfig(
        voxel_size=0.1,
        monte_carlo_samples=50000,
    )
    
    results = run_scale_ladder_study(
        n_floe_list=n_floe_list,
        n_seeds=n_seeds,
        base_seed=base_seed,
        voxel_config=voxel_config,
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    
    for r in results['scale_ladder']:
        print(f"\nn_floe = {r['n_floe']}:")
        print(f"  Porosity (analytical): {r['mean_porosity_analytical']:.3f} ± {r['std_porosity_analytical']:.3f}")
        print(f"  Porosity (voxel):      {r['mean_porosity_voxel']:.3f} ± {r['std_porosity_voxel']:.3f}")
        print(f"  Porosity (MC):         {r['mean_porosity_mc']:.3f} ± {r['std_porosity_mc']:.3f}")
        print(f"  Sail/keel ratio:       {r['mean_sail_keel_ratio']:.3f} ± {r['std_sail_keel_ratio']:.3f}")
    
    print(f"\nScaling fit:")
    print(f"  log(φ) = {results['scaling_fit']['slope']:.3f} * log(n_floe) + {results['scaling_fit']['intercept']:.3f}")
    print(f"  R² = {results['scaling_fit']['r_squared']:.3f}")
    
    # Check acceptance criteria
    print("\n" + "=" * 80)
    print("Acceptance Criteria Check")
    print("=" * 80)
    
    r_squared = results['scaling_fit']['r_squared']
    mean_phi = np.mean([r['mean_porosity_analytical'] for r in results['scale_ladder']])
    mean_sail_keel = np.mean([r['mean_sail_keel_ratio'] for r in results['scale_ladder']])
    
    print(f"\n1. R² > 0.9: {r_squared:.3f} {'✓ PASS' if r_squared > 0.9 else '✗ FAIL'}")
    print(f"2. Porosity in [0.25, 0.35]: {mean_phi:.3f} {'✓ PASS' if 0.25 <= mean_phi <= 0.35 else '✗ FAIL'}")
    print(f"3. Sail/keel ratio in [0.2, 0.5]: {mean_sail_keel:.3f} {'✓ PASS' if 0.2 <= mean_sail_keel <= 0.5 else '✗ FAIL'}")
    
    # Save results
    output_dir = "artifacts/p495w4_n06_voxel_porosity"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON (exclude seed_results for size)
    results_json = {
        "scale_ladder": [
            {k: v for k, v in r.items() if k != "seed_results"}
            for r in results['scale_ladder']
        ],
        "scaling_fit": results['scaling_fit'],
        "n_floe_list": results['n_floe_list'],
        "n_seeds": results['n_seeds'],
    }
    
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {output_dir}/results.json")
    
    # Generate figures
    print("\nGenerating figures...")
    generate_figures(results, output_dir)
    
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


def generate_figures(results, output_dir):
    """Generate paper-quality figures."""
    
    # Extract data
    n_floes = [r['n_floe'] for r in results['scale_ladder']]
    phi_analytical = [r['mean_porosity_analytical'] for r in results['scale_ladder']]
    phi_analytical_std = [r['std_porosity_analytical'] for r in results['scale_ladder']]
    phi_voxel = [r['mean_porosity_voxel'] for r in results['scale_ladder']]
    phi_voxel_std = [r['std_porosity_voxel'] for r in results['scale_ladder']]
    phi_mc = [r['mean_porosity_mc'] for r in results['scale_ladder']]
    phi_mc_std = [r['std_porosity_mc'] for r in results['scale_ladder']]
    sail_keel = [r['mean_sail_keel_ratio'] for r in results['scale_ladder']]
    sail_keel_std = [r['std_sail_keel_ratio'] for r in results['scale_ladder']]
    
    # Figure 1: Porosity vs n_floe (log-log)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.errorbar(n_floes, phi_analytical, yerr=phi_analytical_std, 
                fmt='o-', label='Analytical (exponential decay)', capsize=5, linewidth=2)
    ax.errorbar(n_floes, phi_voxel, yerr=phi_voxel_std, 
                fmt='s-', label='Voxel occupancy', capsize=5, linewidth=2)
    ax.errorbar(n_floes, phi_mc, yerr=phi_mc_std, 
                fmt='^-', label='Monte Carlo', capsize=5, linewidth=2)
    
    # Target range
    ax.axhspan(0.25, 0.35, alpha=0.2, color='green', label='Target range (Timco & Burden 1994)')
    
    # Scaling fit
    log_n = np.log(n_floes)
    log_phi = np.log(phi_analytical)
    coeffs = np.polyfit(log_n, log_phi, 1)
    phi_fit = np.exp(np.polyval(coeffs, log_n))
    ax.plot(n_floes, phi_fit, 'k--', label=f'Fit: R²={results["scaling_fit"]["r_squared"]:.3f}', linewidth=2)
    
    ax.set_xscale('log')
    ax.set_xlabel('Number of floes (n_floe)', fontsize=14)
    ax.set_ylabel('Porosity (φ)', fontsize=14)
    ax.set_title('Ice Ridge Porosity vs Floe Count', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1_porosity_vs_n_floe.png", dpi=300)
    plt.close()
    
    # Figure 2: Sail/keel ratio vs n_floe
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.errorbar(n_floes, sail_keel, yerr=sail_keel_std, 
                fmt='o-', capsize=5, linewidth=2, markersize=8)
    
    # Target range
    ax.axhspan(0.2, 0.5, alpha=0.2, color='green', label='Target range (Hopky 1983)')
    ax.axhline(y=0.35, color='red', linestyle='--', label='Target ratio (0.35)')
    
    ax.set_xscale('log')
    ax.set_xlabel('Number of floes (n_floe)', fontsize=14)
    ax.set_ylabel('Sail/keel ratio (H_s/D_k)', fontsize=14)
    ax.set_title('Sail/Keel Ratio vs Floe Count', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2_sail_keel_ratio.png", dpi=300)
    plt.close()
    
    # Figure 3: Porosity evolution (from first seed of first n_floe)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get strain and porosity history from first simulation
    first_result = results['scale_ladder'][0]['seed_results'][0] if 'seed_results' in results['scale_ladder'][0] else None
    
    if first_result:
        strain = first_result['strain_history']
        phi_analytical_hist = first_result['porosity_analytical_history']
        
        ax.plot(strain, phi_analytical_hist, 'b-', linewidth=2, label='Analytical model')
        ax.axhspan(0.25, 0.35, alpha=0.2, color='green', label='Target range')
        
        ax.set_xlabel('Compressive strain (ε)', fontsize=14)
        ax.set_ylabel('Porosity (φ)', fontsize=14)
        ax.set_title('Porosity Evolution with Compressive Strain', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fig3_porosity_evolution.png", dpi=300)
        plt.close()
    
    # Figure 4: Comprehensive summary
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Top left: Porosity comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(['Analytical', 'Voxel', 'MC'], 
            [np.mean(phi_analytical), np.mean(phi_voxel), np.mean(phi_mc)],
            yerr=[np.mean(phi_analytical_std), np.mean(phi_voxel_std), np.mean(phi_mc_std)],
            capsize=5, alpha=0.7)
    ax1.axhspan(0.25, 0.35, alpha=0.2, color='green')
    ax1.set_ylabel('Porosity (φ)')
    ax1.set_title('Mean Porosity (all n_floe)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Top right: Sail/keel ratio
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar([str(n) for n in n_floes], sail_keel, 
            yerr=sail_keel_std, capsize=5, alpha=0.7)
    ax2.axhspan(0.2, 0.5, alpha=0.2, color='green')
    ax2.set_xlabel('n_floe')
    ax2.set_ylabel('Sail/keel ratio')
    ax2.set_title('Sail/Keel Ratio vs n_floe')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Bottom left: Scaling fit
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(log_n, log_phi, 'o', markersize=10, label='Data')
    ax3.plot(log_n, np.polyval(coeffs, log_n), 'r-', linewidth=2, 
             label=f'Fit: R²={results["scaling_fit"]["r_squared"]:.3f}')
    ax3.set_xlabel('log(n_floe)')
    ax3.set_ylabel('log(φ)')
    ax3.set_title('Scaling Law: log(φ) vs log(n_floe)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Bottom right: Summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    table_data = [
        ['n_floe', 'φ (analytical)', 'φ (voxel)', 'Sail/keel'],
    ]
    for i, n in enumerate(n_floes):
        table_data.append([
            str(n),
            f"{phi_analytical[i]:.3f}±{phi_analytical_std[i]:.3f}",
            f"{phi_voxel[i]:.3f}±{phi_voxel_std[i]:.3f}",
            f"{sail_keel[i]:.3f}±{sail_keel_std[i]:.3f}",
        ])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.2, 0.3, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[0, i].set_facecolor('#4CAF50')
        table[0, i].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Summary Table', pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig4_summary.png", dpi=300)
    plt.close()
    
    print(f"  Generated fig1_porosity_vs_n_floe.png")
    print(f"  Generated fig2_sail_keel_ratio.png")
    if first_result:
        print(f"  Generated fig3_porosity_evolution.png")
    print(f"  Generated fig4_summary.png")


if __name__ == "__main__":
    main()
