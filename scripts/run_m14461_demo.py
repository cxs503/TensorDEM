#!/usr/bin/env python
"""Demonstration and report generation for ice ridge resistance with pore fluid coupling.

Generates:
1. Sail resistance component analysis (Kujala 1994 / Riska 1997)
2. Pore fluid consolidation and strength degradation
3. Ridge resistance quantiles (P10/P50/P90)
4. Sensitivity analysis
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tensordem.ridge_sail_resistance import (
    SailResistanceConfig,
    compute_sail_resistance,
    kujala_breaking_resistance,
    riska_pushing_resistance,
    passive_earth_pressure_coefficient,
    effective_unit_weight,
)
from tensordem.ridge_pore_fluid import (
    PoreFluidConfig,
    KeelPoreFluidModel,
    degree_of_consolidation,
    hydraulic_conductivity_kozeny_carman,
)
from tensordem.ridge_resistance_quantiles import (
    RidgeResistanceQuantileConfig,
    compute_ridge_resistance_quantiles,
    sensitivity_analysis,
)

OUTPUT_DIR = Path(__file__).parent.parent / "artifacts" / "m14461"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fig_sail_resistance_components():
    """Figure 1: Sail resistance components vs sail height."""
    fig, ax = plt.subplots(figsize=(10, 6))

    h_s_values = np.linspace(0.5, 5.0, 50)
    B = 20.0
    sigma_f = 500e3
    n = 0.35
    phi = 40.0

    F_sb_list = []
    F_sp_list = []
    F_sail_list = []

    for h_s in h_s_values:
        cfg = SailResistanceConfig(h_s=h_s, B=B, sigma_f=sigma_f, n=n, phi=phi)
        result = compute_sail_resistance(cfg)
        F_sb_list.append(result.F_sb / 1e6)  # MN
        F_sp_list.append(result.F_sp / 1e6)
        F_sail_list.append(result.F_sail / 1e6)

    ax.plot(h_s_values, F_sb_list, 'b-', linewidth=2, label='Breaking (Kujala 1994)')
    ax.plot(h_s_values, F_sp_list, 'r--', linewidth=2, label='Pushing (Riska 1997)')
    ax.plot(h_s_values, F_sail_list, 'k-', linewidth=2.5, label='Total')

    ax.set_xlabel('Sail height $h_s$ (m)', fontsize=12)
    ax.set_ylabel('Resistance (MN)', fontsize=12)
    ax.set_title('Sail Resistance Components vs Sail Height\n'
                 r'$B=20\,$m, $\sigma_f=0.5\,$MPa, $n=0.35$, $\phi=40°$', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "fig01_sail_resistance_components.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return path


def fig_pore_pressure_dissipation():
    """Figure 2: Excess pore pressure dissipation over time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Degree of consolidation vs T_v
    ax1 = axes[0]
    T_v_values = np.logspace(-3, 2, 100)
    U_values = [degree_of_consolidation(Tv) for Tv in T_v_values]
    ax1.semilogx(T_v_values, U_values, 'b-', linewidth=2)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='U=0.5')
    ax1.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='U=0.9')
    ax1.set_xlabel('Time factor $T_v$', fontsize=12)
    ax1.set_ylabel('Degree of consolidation $U$', fontsize=12)
    ax1.set_title('Terzaghi 1D Consolidation Curve', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1e-3, 100)
    ax1.set_ylim(0, 1.05)

    # Right: Strength reduction vs time for different c_v
    ax2 = axes[1]
    times = np.linspace(0, 86400 * 30, 100)  # 30 days
    c_v_values = [1e-6, 1e-5, 1e-4]
    colors = ['b', 'g', 'r']
    labels = [r'$c_v=10^{-6}$ m²/s', r'$c_v=10^{-5}$ m²/s', r'$c_v=10^{-4}$ m²/s']

    for c_v, color, label in zip(c_v_values, colors, labels):
        cfg = PoreFluidConfig(n_layers=10, keel_depth=10.0, c_v=c_v)
        model = KeelPoreFluidModel(config=cfg)
        model.apply_surface_load(100e3)

        factors = []
        for t in times:
            model_copy = KeelPoreFluidModel(config=cfg)
            model_copy.apply_surface_load(100e3)
            # Simulate consolidation
            n_steps = max(1, int(t / 3600))
            for _ in range(min(n_steps, 100)):
                model_copy.consolidate(dt=3600.0)
            factors.append(np.mean(model_copy.get_strength_reduction_factor()))

        ax2.plot(times / 86400, factors, color=color, linewidth=2, label=label)

    ax2.set_xlabel('Time (days)', fontsize=12)
    ax2.set_ylabel('Mean strength reduction factor', fontsize=12)
    ax2.set_title('Shear Strength Degradation due to Excess Pore Pressure\n'
                  'Keel depth = 10 m, applied load = 100 kPa', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    path = OUTPUT_DIR / "fig02_pore_pressure_dissipation.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return path


def fig_resistance_quantiles():
    """Figure 3: Ridge resistance quantile distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    cfg = RidgeResistanceQuantileConfig(n_samples=500, seed=42)
    result = compute_ridge_resistance_quantiles(cfg)

    # F_sail histogram
    ax1 = axes[0, 0]
    ax1.hist(result.F_sail_samples / 1e6, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    for pct, val in result.quantiles["F_sail"].items():
        ax1.axvline(val / 1e6, color='r', linestyle='--', label=f'{pct}={val/1e6:.1f} MN')
    ax1.set_xlabel('Sail resistance (MN)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Sail Resistance Distribution', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # F_keel histogram
    ax2 = axes[0, 1]
    ax2.hist(result.F_keel_samples / 1e6, bins=30, edgecolor='black', alpha=0.7, color='coral')
    for pct, val in result.quantiles["F_keel"].items():
        ax2.axvline(val / 1e6, color='r', linestyle='--', label=f'{pct}={val/1e6:.1f} MN')
    ax2.set_xlabel('Keel resistance (MN)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Keel Resistance Distribution\n(with pore fluid effect)', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # F_total histogram
    ax3 = axes[1, 0]
    ax3.hist(result.F_total_samples / 1e6, bins=30, edgecolor='black', alpha=0.7, color='darkgreen')
    for pct, val in result.quantiles["F_total"].items():
        ax3.axvline(val / 1e6, color='r', linestyle='--', label=f'{pct}={val/1e6:.1f} MN')
    ax3.set_xlabel('Total resistance (MN)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Total Ridge Resistance Distribution', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Grounding depth histogram
    ax4 = axes[1, 1]
    ax4.hist(result.grounding_depth_samples, bins=30, edgecolor='black', alpha=0.7, color='purple')
    for pct, val in result.quantiles["grounding_depth"].items():
        ax4.axvline(val, color='r', linestyle='--', label=f'{pct}={val:.1f} m')
    ax4.set_xlabel('Grounding depth (m)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Grounding Depth Distribution', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "fig03_resistance_quantiles.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return path


def fig_sensitivity():
    """Figure 4: Sensitivity analysis tornado chart."""
    cfg = RidgeResistanceQuantileConfig(n_samples=200, seed=42)
    sens = sensitivity_analysis(cfg, n_samples=200)

    fig, ax = plt.subplots(figsize=(10, 6))

    params = list(sens.keys())
    indices = [sens[p]["sensitivity_index"] for p in params]
    labels_short = ['Sail height', 'Keel depth', 'Porosity', 'Ice strength', 'Friction angle']

    y_pos = np.arange(len(params))
    colors = ['red' if idx > 0 else 'blue' for idx in indices]
    ax.barh(y_pos, indices, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_short, fontsize=11)
    ax.set_xlabel('Sensitivity index (ΔP50 / P50 per ±10% parameter change)', fontsize=11)
    ax.set_title('Sensitivity of Total Ridge Resistance (P50)\n'
                 'to Input Parameter Variations', fontsize=13)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')

    # Add values
    for i, (idx, p) in enumerate(zip(indices, params)):
        ax.text(idx + 0.01 * (1 if idx > 0 else -1), i,
                f'{idx:.3f}', va='center', fontsize=10,
                ha='left' if idx > 0 else 'right')

    plt.tight_layout()
    path = OUTPUT_DIR / "fig04_sensitivity.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return path


def generate_data_table():
    """Generate numerical results table."""
    # Sail resistance for typical parameters
    cfg = SailResistanceConfig(h_s=2.0, B=20.0, sigma_f=500e3, n=0.35, phi=40.0)
    sail_result = compute_sail_resistance(cfg)

    # Pore fluid model
    pore_cfg = PoreFluidConfig(n_layers=10, keel_depth=10.0, c_v=1e-5)
    pore_model = KeelPoreFluidModel(config=pore_cfg)
    pore_model.apply_surface_load(100e3)
    pore_model.consolidate(dt=3600.0)
    pore_summary = pore_model.summary()

    # Quantiles
    q_cfg = RidgeResistanceQuantileConfig(n_samples=500, seed=42)
    q_result = compute_ridge_resistance_quantiles(q_cfg)

    table = {
        "sail_resistance": {
            "F_sb_MN": sail_result.F_sb / 1e6,
            "F_sp_MN": sail_result.F_sp / 1e6,
            "F_sail_MN": sail_result.F_sail / 1e6,
            "gamma_eff_N_m3": sail_result.gamma_eff,
            "K_p": sail_result.K_p,
        },
        "pore_fluid": pore_summary,
        "quantiles": q_result.quantiles,
        "statistics": q_result.statistics,
    }

    path = OUTPUT_DIR / "results_table.json"
    with open(path, 'w') as f:
        json.dump(table, f, indent=2, default=str)
    print(f"Saved: {path}")
    return table


def main():
    print("=" * 60)
    print("Ice Ridge Resistance with Pore Fluid Coupling")
    print("Campaign P226W3-N01 | Mission m14461")
    print("=" * 60)
    print()

    print("Generating figures...")
    fig_sail_resistance_components()
    fig_pore_pressure_dissipation()
    fig_resistance_quantiles()
    fig_sensitivity()

    print("\nGenerating data table...")
    table = generate_data_table()

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"\nSail Resistance (h_s=2m, B=20m, σ_f=0.5MPa):")
    print(f"  Breaking (Kujala):  {table['sail_resistance']['F_sb_MN']:.2f} MN")
    print(f"  Pushing (Riska):    {table['sail_resistance']['F_sp_MN']:.2f} MN")
    print(f"  Total:              {table['sail_resistance']['F_sail_MN']:.2f} MN")

    print(f"\nPore Fluid (keel depth=10m, c_v=1e-5 m²/s):")
    print(f"  Max excess pore pressure: {table['pore_fluid']['max_excess_pore_pressure_Pa']/1e3:.1f} kPa")
    print(f"  Mean effective stress:    {table['pore_fluid']['mean_effective_stress_Pa']/1e3:.1f} kPa")

    print(f"\nTotal Resistance Quantiles (MN):")
    for pct, val in table['quantiles']['F_total'].items():
        print(f"  {pct}: {val/1e6:.2f} MN")

    print(f"\nGrounding Depth Quantiles (m):")
    for pct, val in table['quantiles']['grounding_depth'].items():
        print(f"  {pct}: {val:.2f} m")

    print("\nAll artifacts saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
