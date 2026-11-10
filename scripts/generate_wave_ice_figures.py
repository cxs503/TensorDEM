"""Generate verification report figures for wave-ice coupling and ice-steel friction.

This script generates all figures for the paper-quality report:
1. Single floe RAO (heave, pitch, roll)
2. Transmission coefficient vs wave period
3. Wave-induced breaking threshold
4. Ice-steel friction μ(V) curve
5. Energy conservation check
"""
from __future__ import annotations

import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from tensordem.wave_ice import (
    FloeRAOConfig,
    FloeArrayConfig,
    IceBreakingConfig,
    compute_floe_rao_analytic,
    compute_floe_rao_dem,
    compute_transmission_analytic_meylan,
    compute_transmission_coefficient_dem,
    compute_critical_wave_height_theory,
    compute_critical_wave_height,
)
from tensordem.bowden_hughes_thermal import (
    BowdenHughesConfig,
    compute_bowden_hughes_equilibrium,
    compute_mu_velocity_curve,
)

OUT_DIR = Path("artifacts/wave_ice_verification")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fig_rao_heave():
    """Figure 1: Single floe heave RAO - analytic vs DEM."""
    config = FloeRAOConfig(
        floe_radius=5.0,
        floe_thickness=0.5,
        wave_periods=list(np.linspace(3.0, 15.0, 30))
    )
    
    analytic = compute_floe_rao_analytic(config)
    dem = compute_floe_rao_dem(config, n_cycles=10, dt=0.02)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(analytic.wave_periods, analytic.rao_heave, 'b-', lw=2, label='Analytic (potential flow)')
    ax.plot(dem.wave_periods, dem.rao_heave, 'ro', ms=5, label='DEM (time-domain)')
    ax.axvline(analytic.peak_period_heave, color='b', ls='--', alpha=0.5, 
               label=f'Analytic peak: T={analytic.peak_period_heave:.1f}s')
    ax.set_xlabel('Wave period [s]', fontsize=12)
    ax.set_ylabel('Heave RAO [m/m]', fontsize=12)
    ax.set_title('Single Floe Heave RAO (R=5m, h=0.5m)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig1_rao_heave.png', dpi=150)
    plt.close(fig)
    print(f"Saved fig1_rao_heave.png")


def fig_rao_three_sizes():
    """Figure 2: Heave RAO for three floe sizes."""
    radii = [3.0, 5.0, 10.0]
    colors = ['green', 'blue', 'red']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for R, c in zip(radii, colors):
        config = FloeRAOConfig(floe_radius=R, floe_thickness=0.5)
        result = compute_floe_rao_analytic(config)
        ax.plot(result.wave_periods, result.rao_heave, '-', color=c, lw=2,
                label=f'R={R}m (peak T={result.peak_period_heave:.1f}s)')
    
    ax.set_xlabel('Wave period [s]', fontsize=12)
    ax.set_ylabel('Heave RAO [m/m]', fontsize=12)
    ax.set_title('Heave RAO for Different Floe Sizes (h=0.5m)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig2_rao_three_sizes.png', dpi=150)
    plt.close(fig)
    print(f"Saved fig2_rao_three_sizes.png")


def fig_transmission():
    """Figure 3: Transmission coefficient vs wave period."""
    coverages = [0.2, 0.4, 0.6]
    colors = ['green', 'blue', 'red']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for c, color in zip(coverages, colors):
        config = FloeArrayConfig(coverage=c)
        result = compute_transmission_analytic_meylan(config)
        ax.plot(result.wave_periods, result.transmission_coefficient, '-', 
                color=color, lw=2, label=f'Coverage={c}')
    
    ax.set_xlabel('Wave period [s]', fontsize=12)
    ax.set_ylabel('Transmission coefficient T [-]', fontsize=12)
    ax.set_title('Wave Transmission Through Floe Array', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig3_transmission.png', dpi=150)
    plt.close(fig)
    print(f"Saved fig3_transmission.png")


def fig_breaking():
    """Figure 4: Critical wave height vs ice thickness."""
    config = IceBreakingConfig(wave_period=10.0)
    
    theory = compute_critical_wave_height_theory(config)
    dem = compute_critical_wave_height(config, n_steps=100, H_max=5.0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(theory.ice_thicknesses, theory.critical_wave_heights, 'b-', lw=2, 
            label='Theory (elastic plate)')
    ax.plot(dem.ice_thicknesses, dem.critical_wave_heights, 'ro', ms=6, 
            label='DEM (stress threshold)')
    ax.set_xlabel('Ice thickness [m]', fontsize=12)
    ax.set_ylabel('Critical wave height H_crit [m]', fontsize=12)
    ax.set_title('Wave-Induced Ice Breaking Threshold (T=10s)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig4_breaking.png', dpi=150)
    plt.close(fig)
    print(f"Saved fig4_breaking.png")


def fig_friction_mu_v():
    """Figure 5: Ice-steel friction μ(V) curve - non-monotonic behavior."""
    config = BowdenHughesConfig(
        bulk_ice_temperature=-5.0,
        asperity_friction=0.15
    )
    
    velocities = np.logspace(-5, 0, 50)
    stresses = [0.1e6, 0.5e6, 2.0e6]
    colors = ['green', 'blue', 'red']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for sigma, color in zip(stresses, colors):
        mu_array, _ = compute_mu_velocity_curve(sigma, velocities, config)
        ax.semilogx(velocities, mu_array, '-', color=color, lw=2,
                    label=f'σ={sigma/1e6:.1f} MPa')
    
    ax.set_xlabel('Sliding velocity V [m/s]', fontsize=12)
    ax.set_ylabel('Friction coefficient μ [-]', fontsize=12)
    ax.set_title('Ice-Steel Friction: μ(V) Non-Monotonic Curve (T_bulk=-5°C)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig5_friction_mu_v.png', dpi=150)
    plt.close(fig)
    print(f"Saved fig5_friction_mu_v.png")


def fig_friction_temperature():
    """Figure 6: Friction coefficient vs temperature."""
    temperatures = np.linspace(-20.0, -0.5, 30)
    v = 0.1  # m/s
    sigma = 1e6  # Pa
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    mu_values = []
    for T in temperatures:
        config = BowdenHughesConfig(bulk_ice_temperature=T)
        result = compute_bowden_hughes_equilibrium(v, sigma, config)
        mu_values.append(result.friction_coefficient)
    
    ax.plot(temperatures, mu_values, 'b-', lw=2)
    ax.set_xlabel('Bulk ice temperature [°C]', fontsize=12)
    ax.set_ylabel('Friction coefficient μ [-]', fontsize=12)
    ax.set_title(f'Ice-Steel Friction vs Temperature (V={v} m/s, σ={sigma/1e6:.1f} MPa)', fontsize=14)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig6_friction_temperature.png', dpi=150)
    plt.close(fig)
    print(f"Saved fig6_friction_temperature.png")


def fig_energy_conservation():
    """Figure 7: Energy conservation check."""
    config = BowdenHughesConfig()
    velocities = np.logspace(-4, 0, 30)
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    residuals = []
    for v in velocities:
        result = compute_bowden_hughes_equilibrium(v, 1e6, config)
        q_total = result.heat_flux_total
        q_sum = result.heat_flux_ice + result.heat_flux_steel + result.heat_flux_melt
        if q_total > 0:
            residual = abs(q_total - q_sum) / q_total * 100
        else:
            residual = 0.0
        residuals.append(residual)
    
    axes[0].semilogx(velocities, residuals, 'b-', lw=2)
    axes[0].axhline(5.0, color='r', ls='--', label='5% tolerance')
    axes[0].set_ylabel('Energy residual [%]', fontsize=12)
    axes[0].set_title('Energy Conservation: q_total = q_ice + q_steel + q_melt', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3, which='both')
    
    # Heat partition
    q_ice_list = []
    q_steel_list = []
    q_melt_list = []
    for v in velocities:
        result = compute_bowden_hughes_equilibrium(v, 1e6, config)
        q_ice_list.append(result.heat_flux_ice)
        q_steel_list.append(result.heat_flux_steel)
        q_melt_list.append(result.heat_flux_melt)
    
    axes[1].loglog(velocities, q_ice_list, 'b-', lw=2, label='q_ice')
    axes[1].loglog(velocities, q_steel_list, 'r-', lw=2, label='q_steel')
    axes[1].loglog(velocities, q_melt_list, 'g-', lw=2, label='q_melt')
    axes[1].set_xlabel('Sliding velocity V [m/s]', fontsize=12)
    axes[1].set_ylabel('Heat flux [W/m²]', fontsize=12)
    axes[1].set_title('Heat Partition vs Velocity', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3, which='both')
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig7_energy_conservation.png', dpi=150)
    plt.close(fig)
    print(f"Saved fig7_energy_conservation.png")


if __name__ == "__main__":
    print("Generating verification report figures...")
    fig_rao_heave()
    fig_rao_three_sizes()
    fig_transmission()
    fig_breaking()
    fig_friction_mu_v()
    fig_friction_temperature()
    fig_energy_conservation()
    print(f"\nAll figures saved to {OUT_DIR}")
