"""Generate figures for campaign p40w2-n06 verification report.

Produces:
  - thermal_resistance.png: Hertz patch R_th vs analytic
  - chain_temperature.png: Particle chain T(x) steady + transient
  - runout_scaling.png: Runout scaling law fit
  - mu_i_rheology.png: mu(I) rheology law
  - energy_ledger.png: Energy conservation ledger
"""
from __future__ import annotations

import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "src"))

import math
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordem.interface_thermal.contact_chain import (
    hertz_patch_radius, pair_contact_resistance, HalfSpacePatch,
    grain_area, chain_steady_analytic, chain_lumped_steady,
    chain_transient_lumped, chain_transient_fvm, rod_temperature_fvm,
    ice_conductivity, ice_bridge_resistance, ice_bond_gate,
)
from tensordem.granular2d.column_collapse import (
    LUBE_TWO_THIRDS, ColumnSpec, run_collapse, fit_runout_scaling,
    fit_mu_inertial, fit_phi_inertial, runout_scan,
)

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(OUTDIR, exist_ok=True)

SMALL = dict(width=0.060, radius=0.006, kn=1.0e4, thickness=0.02,
             mu=0.5, mu_rolling=0.5, max_time=0.3, snapshot_interval=0.05)


def fig_thermal_resistance():
    """Figure 1: Hertz contact patch thermal resistance vs analytic."""
    forces = np.logspace(0, 2, 8)
    R = 0.005
    E = 9.0e9
    nu = 0.33
    k_ice = 2.2

    r_analytic = []
    r_mesh = []
    for f in forces:
        a = hertz_patch_radius(f, R, R, E, E, nu, nu)
        r_analytic.append(1.0 / (4.0 * k_ice * a))
        mesh = HalfSpacePatch(k_ice, a, cells_per_radius=12)
        r_mesh.append(mesh.resistance_k_w)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(forces, r_analytic, 'ko-', label='Analytic 1/(4ka)', markersize=4)
    ax.loglog(forces, r_mesh, 'rs--', label='Axisymmetric FV mesh', markersize=4)
    ax.set_xlabel('Normal force F (N)')
    ax.set_ylabel('Thermal resistance R_th (K/W)')
    ax.set_title('Hertz Contact Thermal Resistance')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    # Error inset
    errors = [abs(m - a) / a for m, a in zip(r_mesh, r_analytic)]
    ax2 = ax.inset_axes([0.55, 0.55, 0.35, 0.35])
    ax2.bar(range(len(errors)), [e * 100 for e in errors], color='steelblue')
    ax2.axhline(10, color='red', linestyle='--', label='10 % threshold')
    ax2.set_xticks(range(len(errors)))
    ax2.set_xticklabels([f'{f:.0f}' for f in forces], fontsize=6)
    ax2.set_ylabel('Error (%)', fontsize=7)
    ax2.set_title('Relative Error', fontsize=7)
    ax2.legend(fontsize=6)

    plt.tight_layout()
    path = os.path.join(OUTDIR, "thermal_resistance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
    return {"max_error_pct": max(errors) * 100}


def fig_chain_temperature():
    """Figure 2: Particle chain temperature field (steady + transient)."""
    RADIUS = 0.005
    K_ICE = 2.2
    RHO = 917.0
    CP = 2100.0
    N = 20
    area = grain_area(RADIUS)
    grain_len = 2.0 * RADIUS / (K_ICE * area)
    Rc = 3.0 * grain_len

    # Steady state
    lumped = chain_lumped_steady(N, RADIUS, K_ICE, RHO, CP, Rc)
    fvm = rod_temperature_fvm(N, RADIUS, K_ICE, RHO, CP, Rc,
                              cells_per_particle=24, time_steps=200, total_time=1.0e9)
    centres = (np.arange(N) + 0.5) * 2.0 * RADIUS
    idx = [int(np.argmin(np.abs(fvm["x"] - x))) for x in centres]

    # Transient
    diffusion_time = 0.5 * (2.0 * RADIUS * N) ** 2 / (K_ICE / (RHO * CP))
    fvm_t = chain_transient_fvm(N, RADIUS, K_ICE, RHO, CP, Rc,
                                total_time=diffusion_time, steps=200)
    lumped_t = chain_transient_lumped(N, RADIUS, K_ICE, RHO, CP, Rc,
                                      total_time=diffusion_time, steps=200)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Steady
    ax1.plot(centres, lumped, 'ko-', label='Lumped DEM network', markersize=3)
    ax1.plot(fvm["x"], fvm["final"], 'b-', label='FVM rod reference', linewidth=1.5)
    ax1.plot(fvm["x"][idx], fvm["final"][idx], 'rs', label='FVM at particle centres', markersize=4)
    ax1.set_xlabel('Position x (m)')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title('Steady-State Temperature Field')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Transient
    t_steps = [0, 50, 100, 150, 199]
    for ts in t_steps:
        ax2.plot(centres, lumped_t["node_history"][ts], '-',
                 label=f't={ts}/{len(lumped_t["node_history"])-1}', alpha=0.7)
    ax2.set_xlabel('Particle index')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_title('Transient Temperature Evolution')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTDIR, "chain_temperature.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    steady_error = float(np.max(np.abs(lumped - fvm["final"][idx])))
    transient_error = float(np.max(np.abs(fvm_t["centre_history"] - lumped_t["node_history"])))
    return {"steady_error": steady_error, "transient_error": transient_error}


def fig_runout_scaling():
    """Figure 3: Runout scaling law fit."""
    aspects = [1.0, 1.5, 2.5, 4.0]
    results = runout_scan(aspects, base=ColumnSpec(**SMALL))
    fit = fit_runout_scaling(results, aspect_min=1.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    a_vals = np.array(fit["aspect_ratios"])
    r_vals = np.array(fit["runouts"])
    ax.loglog(a_vals, r_vals, 'ko', markersize=6, label='DEM runs')

    a_fit = np.linspace(0.8, 5.0, 100)
    r_fit = fit["prefactor"] * a_fit ** fit["exponent"]
    ax.loglog(a_fit, r_fit, 'r-', label=f'Fit: c·a^n, n={fit["exponent"]:.3f}')

    r_ref = fit["prefactor"] * a_fit ** LUBE_TWO_THIRDS
    ax.loglog(a_fit, r_ref, 'b--', label=f'Lube ref: n={LUBE_TWO_THIRDS:.3f}')

    ax.set_xlabel('Aspect ratio a_0')
    ax.set_ylabel('Dimensionless runout ΔL*')
    ax.set_title('Granular Column Collapse: Runout Scaling')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTDIR, "runout_scaling.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
    return fit


def fig_mu_i_rheology():
    """Figure 4: mu(I) rheology law."""
    mu_s, mu_2, I_0 = 0.35, 0.70, 0.30
    I_vals = np.logspace(-3, 1, 100)
    mu_vals = [mu_s + (mu_2 - mu_s) / (I_0 / I + 1) for I in I_vals]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(I_vals, mu_vals, 'b-', linewidth=2, label='μ(I) = μ_s + (μ_2-μ_s)/(I_0/I+1)')
    ax.axhline(mu_s, color='gray', linestyle='--', alpha=0.5, label=f'μ_s = {mu_s}')
    ax.axhline(mu_2, color='gray', linestyle='--', alpha=0.5, label=f'μ_2 = {mu_2}')
    ax.axvline(I_0, color='red', linestyle=':', alpha=0.5, label=f'I_0 = {I_0}')
    ax.set_xlabel('Inertial number I')
    ax.set_ylabel('Effective friction μ(I)')
    ax.set_title('μ(I) Rheology Law (Jop et al. 2006)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(1e-3, 10)
    ax.set_ylim(0.3, 0.8)

    plt.tight_layout()
    path = os.path.join(OUTDIR, "mu_i_rheology.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
    return {"mu_s": mu_s, "mu_2": mu_2, "I_0": I_0}


def fig_energy_ledger():
    """Figure 5: Energy ledger for column collapse."""
    result = run_collapse(ColumnSpec(aspect_ratio=1.5, **SMALL), sample_flow=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ['E_p0', 'E_p1', 'E_k1', 'E_el1', 'E_diss']
    values = [
        result.energy_potential_initial,
        result.energy_potential_final,
        result.energy_kinetic_final,
        result.energy_elastic_final,
        result.energy_dissipated,
    ]
    colors = ['steelblue', 'lightblue', 'orange', 'green', 'red']
    bars = ax.bar(categories, values, color=colors, edgecolor='black')

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f'{val:.4g}', ha='center', va='bottom', fontsize=8)

    # Conservation check
    residual = abs(result.energy_residual_relative)
    ax.axhline(result.energy_potential_initial, color='gray', linestyle='--', alpha=0.5,
               label=f'E_p0 = {result.energy_potential_initial:.4g}')
    ax.set_ylabel('Energy (J)')
    ax.set_title(f'Energy Ledger (residual = {residual*100:.2f} %)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(OUTDIR, "energy_ledger.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
    return {
        "E_p0": result.energy_potential_initial,
        "E_diss": result.energy_dissipated,
        "residual_pct": residual * 100,
    }


def main():
    print("Generating figures for campaign p40w2-n06...")
    results = {}
    results["thermal_resistance"] = fig_thermal_resistance()
    results["chain_temperature"] = fig_chain_temperature()
    results["runout_scaling"] = fig_runout_scaling()
    results["mu_i_rheology"] = fig_mu_i_rheology()
    results["energy_ledger"] = fig_energy_ledger()

    # Save summary
    summary_path = os.path.join(OUTDIR, "verification_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
