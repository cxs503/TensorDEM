#!/usr/bin/env python3
"""Generate figures for DEM ice-contact mechanics calibration benchmark.

Produces:
1. Hertz force-displacement curve (D1)
2. Mindlin tangential force-displacement curve (D2)
3. Friction calibration comparison (D3)
4. Ice resistance comparison with Lindqvist (D4)

All figures saved as PNG in benchmarks/verified/dem_ice_contact/figures/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tensordem.dem3d import (
    hertz_reduced_modulus,
    hertz_effective_radius,
    hertz_normal_force,
)
from tensordem import hertz_mindlin
from tensordem.ice_steel_friction import (
    fit_ice_steel_friction,
    CALIBRATION_ANCHORS,
    VALIDATION_ANCHORS,
)
from tensordem.ice_dem_contact import (
    IceMaterial,
    IcePlateConfig,
    ShipWedgeConfig,
    run_single_case,
    lindqvist_anchor_for_config,
)


def plot_hertz_force_displacement():
    """D1: Hertz force-displacement curve."""
    print("Generating D1: Hertz force-displacement curve...")
    
    E = 1.0e9
    nu = 0.3
    R = 0.01
    
    e_star = hertz_reduced_modulus(E, nu)
    r_eff = hertz_effective_radius(R)
    
    # Sweep overlaps
    deltas = np.logspace(-6, -3, 50)
    f_analytical = (4.0 / 3.0) * e_star * np.sqrt(r_eff) * deltas**1.5
    f_impl = [float(hertz_normal_force(d, e_star, r_eff)) for d in deltas]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(deltas * 1e6, f_analytical, 'b-', linewidth=2, label='Analytical')
    ax.loglog(deltas * 1e6, f_impl, 'ro', markersize=4, label='Implementation')
    ax.set_xlabel('Overlap δ [μm]', fontsize=12)
    ax.set_ylabel('Normal force F_n [N]', fontsize=12)
    ax.set_title('D1: Hertz Normal Contact\nF_n = (4/3) E* √R* δ^(3/2)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([0.5, 1500])
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "figures" / "d1_hertz_force_displacement.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_mindlin_tangential_force():
    """D2: Mindlin tangential force-displacement curve."""
    print("Generating D2: Mindlin tangential force-displacement curve...")
    
    E = 1.0e9
    nu = 0.3
    R = 0.01
    delta = 1.0e-4
    
    e_star = hertz_reduced_modulus(E, nu)
    g_star = hertz_mindlin.hertz_reduced_shear_modulus(E, nu)
    r_eff = hertz_effective_radius(R)
    
    k_t = float(hertz_mindlin.hertz_mindlin_tangential_stiffness(
        delta, e_star, g_star, r_eff))
    
    # Sweep tangential displacements
    delta_t = np.logspace(-8, -5, 50)
    f_analytical = k_t * delta_t
    f_impl = []
    
    for dt in delta_t:
        tang_disp = torch.tensor([[dt, 0.0, 0.0]], dtype=torch.float64)
        f_n_mag = torch.tensor([100.0], dtype=torch.float64)
        mu = 0.5
        
        f_t, _ = hertz_mindlin.hertz_mindlin_tangential_force(
            delta, tang_disp, e_star, g_star, r_eff, mu, f_n_mag)
        
        f_impl.append(float(f_t.norm(dim=1).item()))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(delta_t * 1e6, f_analytical, 'b-', linewidth=2, label='Analytical (k_t · δ_t)')
    ax.loglog(delta_t * 1e6, f_impl, 'ro', markersize=4, label='Implementation')
    ax.set_xlabel('Tangential displacement δ_t [μm]', fontsize=12)
    ax.set_ylabel('Tangential force F_t [N]', fontsize=12)
    ax.set_title('D2: Mindlin Tangential Contact\nF_t = k_t · δ_t (stick regime)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "figures" / "d2_mindlin_tangential_force.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_friction_calibration():
    """D3: Friction calibration comparison."""
    print("Generating D3: Friction calibration comparison...")
    
    model = fit_ice_steel_friction(CALIBRATION_ANCHORS)
    
    # Test on validation anchors
    validation = VALIDATION_ANCHORS[:5]
    
    v_exp = [a.mu for a in validation]
    v_pred = [model.mu(a.v_m_per_s, a.temperature_c, a.pressure_mpa) for a in validation]
    labels = [f"v={a.v_m_per_s:.3f}\nT={a.temperature_c:.0f}°C\np={a.pressure_mpa:.1f}MPa" 
              for a in validation]
    
    x = np.arange(len(validation))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, v_exp, width, label='Experimental', color='steelblue')
    bars2 = ax.bar(x + width/2, v_pred, width, label='Model prediction', color='coral')
    
    ax.set_xlabel('Validation condition', fontsize=12)
    ax.set_ylabel('Friction coefficient μ', fontsize=12)
    ax.set_title('D3: Ice-Steel Friction Calibration\nLoglinear model vs experimental data', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add error bars
    for i, (exp, pred) in enumerate(zip(v_exp, v_pred)):
        error = abs(pred - exp) / exp * 100
        ax.text(i, max(exp, pred) + 0.005, f'{error:.1f}%', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "figures" / "d3_friction_calibration.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_ice_resistance_comparison():
    """D4: Ice resistance comparison with Lindqvist."""
    print("Generating D4: Ice resistance comparison...")
    
    ice_material = IceMaterial()
    ship = ShipWedgeConfig(speed_knots=2.0)
    
    # Run DEM at different ice thicknesses
    thicknesses = [0.5, 1.0, 1.5]
    r_dem = []
    r_lindqvist = []
    
    for h in thicknesses:
        ice_plate = IcePlateConfig(thickness_m=h)
        
        # DEM
        result = run_single_case(ice_material, ice_plate, ship, steps=60)
        r_dem.append(result["mean_ice_resistance_N"])
        
        # Lindqvist
        lindqvist = lindqvist_anchor_for_config(ice_material, ice_plate, ship)
        r_lindqvist.append(lindqvist["crushing_N"] + lindqvist["bending_N"])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thicknesses, r_dem, 'ro-', linewidth=2, markersize=8, label='DEM simulation')
    ax.plot(thicknesses, r_lindqvist, 'bs-', linewidth=2, markersize=8, label='Lindqvist (R_c + R_b)')
    ax.set_xlabel('Ice thickness h_i [m]', fontsize=12)
    ax.set_ylabel('Ice resistance R [N]', fontsize=12)
    ax.set_title('D4: Ice Resistance vs Ice Thickness\nDEM vs Lindqvist empirical formula', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Add annotations
    for i, (h, rd, rl) in enumerate(zip(thicknesses, r_dem, r_lindqvist)):
        ratio = rd / rl if rl > 0 else 0
        ax.annotate(f'Ratio: {ratio:.2e}', 
                   xy=(h, rd), xytext=(10, 10),
                   textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "figures" / "d4_ice_resistance_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    """Generate all figures."""
    print("=" * 70)
    print("DEM Ice-Contact Mechanics Calibration - Figure Generation")
    print("=" * 70)
    print()
    
    plot_hertz_force_displacement()
    print()
    
    plot_mindlin_tangential_force()
    print()
    
    plot_friction_calibration()
    print()
    
    plot_ice_resistance_comparison()
    print()
    
    print("=" * 70)
    print("All figures generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
