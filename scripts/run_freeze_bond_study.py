#!/usr/bin/env python3
"""Freeze-bond evolution study: generate results and figures.

This script runs a comprehensive study of ice ridge keel freeze-bond strength
time evolution and generates publication-quality figures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tensordem.ice_ridge_freeze_bond_evolution import (
    run_freeze_bond_study,
    FREEZING_POINT_K,
)


def create_bond_strength_figure(study_result: dict, output_path: Path) -> None:
    """Create figure showing bond strength evolution over time."""
    bond_evo = study_result['bond_evolution']
    params = study_result['parameters']
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=150)
    
    # Top panel: tensile and shear strength
    ax1 = axes[0]
    ages_hours = bond_evo['ages_s'] / 3600.0
    tensile_MPa = bond_evo['tensile_strength_Pa'] / 1e6
    shear_MPa = bond_evo['shear_strength_Pa'] / 1e6
    
    ax1.plot(ages_hours, tensile_MPa, 'b-', linewidth=2, label='Tensile strength')
    ax1.plot(ages_hours, shear_MPa, 'r-', linewidth=2, label='Shear strength')
    
    # Add asymptotic lines
    ax1.axhline(params['asymptotic_tensile_Pa'] / 1e6, color='b', linestyle='--', 
                alpha=0.5, label=f'Asymptotic tensile ({params["asymptotic_tensile_Pa"]/1e6:.1f} MPa)')
    ax1.axhline(params['asymptotic_shear_Pa'] / 1e6, color='r', linestyle='--',
                alpha=0.5, label=f'Asymptotic shear ({params["asymptotic_shear_Pa"]/1e6:.1f} MPa)')
    
    ax1.set_xlabel('Bond age [hours]', fontsize=12)
    ax1.set_ylabel('Bond strength [MPa]', fontsize=12)
    ax1.set_title('Freeze-Bond Strength Evolution', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, params['max_age_hours']])
    
    # Bottom panel: strength ratio
    ax2 = axes[1]
    strength_ratio = bond_evo['strength_ratio'] * 100
    
    ax2.plot(ages_hours, strength_ratio, 'k-', linewidth=2)
    ax2.axhline(90, color='g', linestyle=':', alpha=0.7, label='90% saturation')
    ax2.axhline(99, color='r', linestyle=':', alpha=0.7, label='99% saturation')
    
    # Mark characteristic time
    tau_hours = params['growth_time_hours']
    ax2.axvline(tau_hours, color='orange', linestyle='--', alpha=0.7, 
                label=f'τ = {tau_hours:.0f} h')
    ax2.plot(tau_hours, (1 - np.exp(-1)) * 100, 'ko', markersize=8)
    
    ax2.set_xlabel('Bond age [hours]', fontsize=12)
    ax2.set_ylabel('Strength ratio [%]', fontsize=12)
    ax2.set_title('Normalized Strength Growth', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, params['max_age_hours']])
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved bond strength figure: {output_path}")
    plt.close()


def create_resistance_evolution_figure(study_result: dict, output_path: Path) -> None:
    """Create figure showing keel resistance evolution over time."""
    resistance_evo = study_result['resistance_evolution']
    params = study_result['parameters']
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=150)
    
    # Top panel: total resistance
    ax1 = axes[0]
    ages_hours = resistance_evo['ages_hours']
    total_MN_per_m = resistance_evo['total_resistance_N_per_m'] / 1e6
    
    ax1.plot(ages_hours, total_MN_per_m, 'b-', linewidth=2.5, label='Total resistance')
    
    ax1.set_xlabel('Keel age [hours]', fontsize=12)
    ax1.set_ylabel('Total resistance [MN/m]', fontsize=12)
    ax1.set_title('Ice Ridge Keel Resistance vs. Age', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, params['max_age_hours']])
    
    # Add text annotation
    initial = total_MN_per_m[0]
    final = total_MN_per_m[-1]
    increase = (final - initial) / initial * 100
    ax1.text(0.05, 0.95, f'Initial: {initial:.2f} MN/m\nFinal: {final:.2f} MN/m\nIncrease: {increase:.1f}%',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Bottom panel: resistance components
    ax2 = axes[1]
    rubble_MN = resistance_evo['rubble_friction_N_per_m'] / 1e6
    bond_MN = resistance_evo['freeze_bond_N_per_m'] / 1e6
    buoyancy_MN = resistance_evo['buoyancy_N_per_m'] / 1e6
    
    ax2.stackplot(ages_hours, rubble_MN, bond_MN, buoyancy_MN,
                  labels=['Rubble friction', 'Freeze bond', 'Buoyancy'],
                  colors=['#888888', '#ff7f0e', '#1f77b4'],
                  alpha=0.7)
    
    ax2.set_xlabel('Keel age [hours]', fontsize=12)
    ax2.set_ylabel('Resistance [MN/m]', fontsize=12)
    ax2.set_title('Resistance Component Breakdown', fontsize=14, fontweight='bold')
    ax2.legend(loc='center right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, params['max_age_hours']])
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved resistance evolution figure: {output_path}")
    plt.close()


def create_parameter_sensitivity_figure(output_path: Path) -> None:
    """Create figure showing sensitivity to growth time parameter."""
    from tensordem.ice_ridge_freeze_bond_evolution import (
        FreezeBondFormationCriterion,
        ExponentialSaturationGrowth,
        FreezeBondModel,
        compute_bond_strength_evolution,
    )
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    formation = FreezeBondFormationCriterion()
    ages_s = np.linspace(0, 500 * 3600, 200)
    ages_hours = ages_s / 3600.0
    
    # Different growth times
    growth_times = [50, 100, 200, 400]  # hours
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for tau, color in zip(growth_times, colors):
        growth = ExponentialSaturationGrowth(
            asymptotic_tensile_strength_Pa=1.0e6,
            growth_time_s=tau * 3600.0,
        )
        model = FreezeBondModel(formation, growth)
        result = compute_bond_strength_evolution(model, ages_s, 263.15)
        
        tensile_MPa = result['tensile_strength_Pa'] / 1e6
        ax.plot(ages_hours, tensile_MPa, color=color, linewidth=2, 
                label=f'τ = {tau} h')
    
    ax.set_xlabel('Bond age [hours]', fontsize=12)
    ax.set_ylabel('Tensile strength [MPa]', fontsize=12)
    ax.set_title('Sensitivity to Growth Time Parameter', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 500])
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved parameter sensitivity figure: {output_path}")
    plt.close()


def main():
    """Run complete study and generate all outputs."""
    print("=" * 70)
    print("ICE RIDGE FREEZE-BOND EVOLUTION STUDY")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("artifacts/freeze_bond_study")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run baseline study
    print("\n[1/4] Running baseline study...")
    study = run_freeze_bond_study(
        max_age_hours=240.0,
        n_points=200,
        temperature_K=263.15,  # -10°C
        asymptotic_tensile_Pa=1.0e6,
        asymptotic_shear_Pa=2.0e6,
        growth_time_hours=100.0,
        keel_depth_m=10.0,
        keel_width_m=20.0,
        porosity=0.30,
        bond_area_fraction=0.5,
    )
    
    # Save study results to JSON
    results_json = output_dir / "study_results.json"
    with open(results_json, 'w') as f:
        json.dump(study, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    print(f"✓ Saved study results: {results_json}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("STUDY SUMMARY")
    print("=" * 70)
    summary = study['summary']
    print(f"Final bond strength: {summary['final_bond_strength_Pa']/1e6:.3f} MPa")
    print(f"Final strength ratio: {summary['strength_ratio_final']*100:.1f}%")
    print(f"Initial resistance: {summary['initial_resistance_N_per_m']/1e6:.3f} MN/m")
    print(f"Final resistance: {summary['final_resistance_N_per_m']/1e6:.3f} MN/m")
    print(f"Resistance increase: {summary['resistance_increase_percent']:.1f}%")
    
    # Generate figures
    print("\n[2/4] Generating bond strength evolution figure...")
    create_bond_strength_figure(study, output_dir / "bond_strength_evolution.png")
    
    print("\n[3/4] Generating resistance evolution figure...")
    create_resistance_evolution_figure(study, output_dir / "resistance_evolution.png")
    
    print("\n[4/4] Generating parameter sensitivity figure...")
    create_parameter_sensitivity_figure(output_dir / "parameter_sensitivity.png")
    
    print("\n" + "=" * 70)
    print("STUDY COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
