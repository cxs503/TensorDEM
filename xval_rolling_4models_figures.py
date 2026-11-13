#!/usr/bin/env python
"""Generate figures for XVAL-DEM-ROLLING-MODEL-ASSESSMENT-4-MODELS."""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from tensordem.rolling_resistance import (
    RollingResistanceConfig,
    SphereInclineConfig,
    incline_rolling_acceleration,
    simulate_sphere_on_incline,
    fit_acceleration,
)

# Parameters (must match xval_rolling_4models.py)
MU_R = 0.15
RADIUS = 0.01
MASS = 4.0/3.0 * math.pi * RADIUS**3 * 2500.0
INERTIA = 0.4 * MASS * RADIUS**2
INERTIA_FACTOR = 0.4
G = 9.81
K_N = 2.0e4
MU_S = 0.5
DT = 1.0e-4
DURATION = 0.2
SETTLE = 0.05
OMEGA_REF = 1.0
K_R = 2.25 * MU_R * K_N * RADIUS**2
THETA_DEG = 15.0

OUTDIR = os.path.join(os.path.dirname(__file__), "artifacts", "xval_rolling_4models", "figures")
os.makedirs(OUTDIR, exist_ok=True)


def fig1_torque_comparison():
    """Torque magnitude vs |omega| for all 4 models."""
    f_n_val = 5.0
    r_eff_val = RADIUS
    limit = MU_R * f_n_val * r_eff_val
    
    omegas = np.logspace(-2, 2, 200)
    
    # Model 1: CDT
    tau_cdt = np.full_like(omegas, limit)
    
    # Model 2: Viscous (linear)
    tau_visc = limit * omegas / OMEGA_REF
    
    # Model 3: EPSD (saturated)
    tau_epsd = np.full_like(omegas, limit)
    
    # Model 4: Regularized
    tau_regv = limit * omegas / (omegas + OMEGA_REF)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(omegas, tau_cdt, 'b-', linewidth=2, label='CDT (Model 1)')
    ax.loglog(omegas, tau_visc, 'g--', linewidth=2, label='Viscous (Model 2)')
    ax.loglog(omegas, tau_epsd, 'r-', linewidth=2, label='EPSD (Model 3)')
    ax.loglog(omegas, tau_regv, 'm-.', linewidth=2, label='Regularized (Model 4)')
    ax.axhline(limit, color='k', linestyle=':', alpha=0.5, label=f'Coulomb limit = {limit:.4e} N·m')
    ax.axvline(OMEGA_REF, color='k', linestyle=':', alpha=0.3, label=f'ω_ref = {OMEGA_REF} rad/s')
    ax.set_xlabel('|ω_roll| [rad/s]', fontsize=12)
    ax.set_ylabel('|M_r| [N·m]', fontsize=12)
    ax.set_title('Rolling Resistance Torque vs Angular Velocity\n(Ai et al. 2011, 4 models)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(1e-5, 1e1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig1_torque_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig1_torque_comparison.png")


def fig2_incline_velocity():
    """Velocity vs time for all 4 models on incline."""
    a_ref = incline_rolling_acceleration(THETA_DEG, MU_R, gravity=G, inertia_factor=INERTIA_FACTOR)
    harness = SphereInclineConfig(k_n=K_N, dt=DT, mu=MU_S)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Model 1: CDT
    cfg = RollingResistanceConfig(model="cdt", mu_r=MU_R)
    res = simulate_sphere_on_incline([THETA_DEG], cfg, config=harness, duration=DURATION, settle_time=SETTLE)
    t = res["time"].numpy()
    v = res["v"][:, 0].numpy()
    ax.plot(t, v, 'b-', linewidth=2, label=f'CDT (a={float(fit_acceleration(res)[0][0]):.4f} m/s²)')
    
    # Model 2: Viscous
    cfg = RollingResistanceConfig(model="viscous", mu_r=MU_R, omega_ref=OMEGA_REF)
    res = simulate_sphere_on_incline([THETA_DEG], cfg, config=harness, duration=DURATION, settle_time=SETTLE)
    v = res["v"][:, 0].numpy()
    ax.plot(t, v, 'g--', linewidth=2, label=f'Viscous (a≈{float(fit_acceleration(res)[0][0]):.4f} m/s²)')
    
    # Model 3: EPSD
    cfg = RollingResistanceConfig(model="epsd", mu_r=MU_R, k_r=K_R)
    res = simulate_sphere_on_incline([THETA_DEG], cfg, config=harness, duration=DURATION, settle_time=SETTLE)
    v = res["v"][:, 0].numpy()
    ax.plot(t, v, 'r-', linewidth=2, label=f'EPSD (a={float(fit_acceleration(res)[0][0]):.4f} m/s²)')
    
    # Analytical reference line
    ax.plot(t, a_ref * t, 'k:', linewidth=1.5, alpha=0.7, label=f'Analytical (CDT): a={a_ref:.4f} m/s²')
    
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Down-slope velocity [m/s]', fontsize=12)
    ax.set_title(f'Sphere on Incline (θ={THETA_DEG}°, μ_r={MU_R})\nVelocity vs Time — 4 Rolling Models', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig2_incline_velocity.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig2_incline_velocity.png")


def fig3_dt_convergence():
    """dt convergence study."""
    a_ref = incline_rolling_acceleration(THETA_DEG, MU_R, gravity=G, inertia_factor=INERTIA_FACTOR)
    
    dt_values = [5e-4, 2e-4, 1e-4, 5e-5, 2e-5]
    errors = []
    
    for dt in dt_values:
        cfg = RollingResistanceConfig(model="cdt", mu_r=MU_R)
        h = SphereInclineConfig(k_n=K_N, dt=dt, mu=MU_S)
        res = simulate_sphere_on_incline([THETA_DEG], cfg, config=h, duration=DURATION, settle_time=SETTLE)
        a_fit, _ = fit_acceleration(res)
        a_got = float(a_fit[0])
        rel_err = abs(a_got - a_ref) / abs(a_ref) * 100  # percent
        errors.append(rel_err)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(dt_values, errors, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Time step dt [s]', fontsize=12)
    ax.set_ylabel('Relative error in acceleration [%]', fontsize=12)
    ax.set_title('dt-Convergence Study (CDT model, θ=15°, μ_r=0.15)\nReference: a = g(sin θ - μ_r cos θ)/(1 + I/mR²)', fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='1% threshold')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig3_dt_convergence.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig3_dt_convergence.png")


def fig4_mu_r_sweep():
    """Acceleration vs mu_r for CDT model."""
    a_ref_no_rr = incline_rolling_acceleration(THETA_DEG, 0.0, gravity=G, inertia_factor=INERTIA_FACTOR)
    
    mu_r_values = np.linspace(0, 0.35, 50)
    a_got_list = []
    a_ref_list = []
    
    for mu_r in mu_r_values:
        a_ref_local = incline_rolling_acceleration(THETA_DEG, mu_r, gravity=G, inertia_factor=INERTIA_FACTOR)
        a_ref_list.append(a_ref_local)
        
        cfg = RollingResistanceConfig(model="cdt", mu_r=mu_r)
        h = SphereInclineConfig(k_n=K_N, dt=DT, mu=MU_S)
        res = simulate_sphere_on_incline([THETA_DEG], cfg, config=h, duration=DURATION, settle_time=SETTLE)
        a_fit, _ = fit_acceleration(res)
        a_got_list.append(float(a_fit[0]))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mu_r_values, a_ref_list, 'k-', linewidth=2, label='Analytical (CDT)')
    ax.plot(mu_r_values, a_got_list, 'bo', markersize=4, label='TensorDEM simulation')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    theta_c_max = math.degrees(math.atan(0.35))
    ax.axvline(math.degrees(math.atan(MU_R)), color='r', linestyle='--', alpha=0.5, 
               label=f'θ_c(μ_r=0.15) = {math.degrees(math.atan(0.15)):.1f}°')
    ax.set_xlabel('Rolling resistance coefficient μ_r', fontsize=12)
    ax.set_ylabel('Down-slope acceleration [m/s²]', fontsize=12)
    ax.set_title(f'Acceleration vs μ_r (θ={THETA_DEG}°)\nCritical angle θ_c = arctan(μ_r)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig4_mu_r_sweep.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig4_mu_r_sweep.png")


def fig5_energy_dissipation():
    """Energy dissipation rate vs omega for all 4 models."""
    f_n_val = 5.0
    r_eff_val = RADIUS
    limit = MU_R * f_n_val * r_eff_val
    
    omegas = np.logspace(-2, 2, 200)
    
    # Power = M . omega = -|M| * |omega| (since M opposes omega)
    P_cdt = -limit * omegas
    P_visc = -limit * omegas**2 / OMEGA_REF
    P_epsd = -limit * omegas  # saturated
    P_regv = -limit * omegas**2 / (omegas + OMEGA_REF)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(omegas, -P_cdt, 'b-', linewidth=2, label='CDT (Model 1)')
    ax.loglog(omegas, -P_visc, 'g--', linewidth=2, label='Viscous (Model 2)')
    ax.loglog(omegas, -P_epsd, 'r-', linewidth=2, label='EPSD (Model 3)')
    ax.loglog(omegas, -P_regv, 'm-.', linewidth=2, label='Regularized (Model 4)')
    ax.set_xlabel('|ω_roll| [rad/s]', fontsize=12)
    ax.set_ylabel('Dissipation rate |M·ω| [W]', fontsize=12)
    ax.set_title('Energy Dissipation Rate vs Angular Velocity\n(All models dissipative: M·ω ≤ 0)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig5_energy_dissipation.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig5_energy_dissipation.png")


if __name__ == "__main__":
    print("Generating figures for XVAL-DEM-ROLLING-MODEL-ASSESSMENT-4-MODELS...")
    fig1_torque_comparison()
    fig2_incline_velocity()
    fig3_dt_convergence()
    fig4_mu_r_sweep()
    fig5_energy_dissipation()
    print(f"All figures saved to {OUTDIR}/")
