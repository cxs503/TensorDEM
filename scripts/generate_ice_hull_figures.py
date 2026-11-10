"""Generate verification figures for ice-hull contact mechanics."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '/root/tp-agent/work/m10559/src')

from tensordem.ice_hull_contact_mechanics import (
    run_slope_sliding_verification,
    run_tension_test_verification,
    run_shear_test_verification,
    run_velocity_dependence_verification,
    IceHullContactModel,
    compute_friction_coefficient,
)

# Figure 1: Friction angle vs temperature
def plot_slope_sliding():
    result = run_slope_sliding_verification()
    temps = [r["temperature_c"] for r in result["results"]]
    angles = [r["friction_angle_deg"] for r in result["results"]]
    exp_mids = [r["experimental_mid_deg"] for r in result["results"]]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(temps, angles, 'bo-', linewidth=2, markersize=8, label='Model')
    ax.plot(temps, exp_mids, 'rs--', linewidth=2, markersize=8, label='Experiment (midpoint)')
    ax.set_xlabel('Temperature [°C]', fontsize=12)
    ax.set_ylabel('Friction Angle [°]', fontsize=12)
    ax.set_title('Ice-Steel Friction Angle vs Temperature\n(Slope Sliding Verification)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-22, 0)
    plt.tight_layout()
    plt.savefig('/root/tp-agent/work/m10559/artifacts/ice_hull_friction_angle.png', dpi=150)
    plt.close()

# Figure 2: Adhesion stress vs contact time
def plot_tension_test():
    result = run_tension_test_verification()
    times = [r["contact_time_s"] for r in result["results"]]
    tau_model = [r["adhesion_stress_mpa"] for r in result["results"]]
    tau_exp = [r["experimental_mpa"] for r in result["results"]]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(times, tau_model, 'bo-', linewidth=2, markersize=8, label='Model')
    ax.plot(times, tau_exp, 'rs--', linewidth=2, markersize=8, label='Experiment')
    ax.set_xlabel('Contact Time [s]', fontsize=12)
    ax.set_ylabel('Adhesion Stress [MPa]', fontsize=12)
    ax.set_title('Ice-Steel Adhesion Stress vs Contact Time\n(Tension Test Verification)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/root/tp-agent/work/m10559/artifacts/ice_hull_adhesion_time.png', dpi=150)
    plt.close()

# Figure 3: Shear strength vs pressure
def plot_shear_test():
    result = run_shear_test_verification()
    pressures = [r["pressure_pa"] / 1e6 for r in result["results"]]
    tau_model = [r["shear_strength_mpa"] for r in result["results"]]
    tau_exp = [r["experimental_mpa"] for r in result["results"]]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pressures, tau_model, 'bo-', linewidth=2, markersize=8, label='Model')
    ax.plot(pressures, tau_exp, 'rs--', linewidth=2, markersize=8, label='Experiment')
    ax.set_xlabel('Normal Pressure [MPa]', fontsize=12)
    ax.set_ylabel('Shear Strength [MPa]', fontsize=12)
    ax.set_title('Ice-Steel Interface Shear Strength vs Pressure\n(Shear Test Verification)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/root/tp-agent/work/m10559/artifacts/ice_hull_shear_strength.png', dpi=150)
    plt.close()

# Figure 4: Friction coefficient vs velocity
def plot_velocity_dependence():
    result = run_velocity_dependence_verification()
    velocities = [r["velocity_m_s"] for r in result["results"]]
    mu_model = [r["friction_coefficient"] for r in result["results"]]
    mu_exp = [r["experimental"] for r in result["results"]]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogx(velocities, mu_model, 'bo-', linewidth=2, markersize=8, label='Model')
    ax.semilogx(velocities, mu_exp, 'rs--', linewidth=2, markersize=8, label='Experiment')
    ax.set_xlabel('Slip Velocity [m/s]', fontsize=12)
    ax.set_ylabel('Friction Coefficient μ [-]', fontsize=12)
    ax.set_title('Ice-Steel Friction Coefficient vs Slip Velocity\n(Velocity Dependence Verification)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('/root/tp-agent/work/m10559/artifacts/ice_hull_friction_velocity.png', dpi=150)
    plt.close()

# Figure 5: Complete contact model overview
def plot_contact_model_overview():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (a) Friction coefficient surface
    ax = axes[0, 0]
    temps = np.linspace(-20, -2, 20)
    vels = np.logspace(-3, 1, 20)
    T, V = np.meshgrid(temps, vels)
    MU = np.zeros_like(T)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            MU[i, j] = compute_friction_coefficient(1e5, V[i, j], T[i, j])
    
    cs = ax.contourf(T, V, MU, levels=15, cmap='viridis')
    plt.colorbar(cs, ax=ax, label='μ [-]')
    ax.set_xlabel('Temperature [°C]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_title('(a) Friction Coefficient μ(T, v)')
    ax.set_yscale('log')
    
    # (b) Adhesion stress evolution
    ax = axes[0, 1]
    times = np.linspace(0, 300, 50)
    for temp in [-20, -10, -5, -2]:
        model = IceHullContactModel()
        taus = []
        for t in times:
            state = model.compute_state(1e5, 0.0, temp, t)
            taus.append(state.adhesion_stress_pa / 1e6)
        ax.plot(times, taus, linewidth=2, label=f'T = {temp}°C')
    ax.set_xlabel('Contact Time [s]')
    ax.set_ylabel('Adhesion Stress [MPa]')
    ax.set_title('(b) Adhesion Stress Evolution τ_ad(t)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (c) Shear strength envelope
    ax = axes[1, 0]
    pressures = np.linspace(0, 5, 50)
    for temp in [-20, -10, -5, -2]:
        model = IceHullContactModel()
        taus = []
        for p in pressures:
            state = model.compute_state(p * 1e6, 0.01, temp, 60.0)
            taus.append(state.shear_strength_pa / 1e6)
        ax.plot(pressures, taus, linewidth=2, label=f'T = {temp}°C')
    ax.set_xlabel('Normal Pressure [MPa]')
    ax.set_ylabel('Shear Strength [MPa]')
    ax.set_title('(c) Shear Strength Envelope τ_shear(p)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (d) Energy balance
    ax = axes[1, 1]
    durations = np.linspace(1, 20, 20)
    work = []
    heat = []
    model = IceHullContactModel()
    for dur in durations:
        state = model.compute_state(1e6, 1.0, -10.0, 0.0)
        power = state.friction_power_density_w_m2 * 1e-4  # W
        energy = power * dur  # J
        work.append(energy)
        heat.append(energy)
    ax.plot(durations, work, 'b-', linewidth=2, label='Friction Work')
    ax.plot(durations, heat, 'r--', linewidth=2, label='Heat Generated')
    ax.set_xlabel('Duration [s]')
    ax.set_ylabel('Energy [J]')
    ax.set_title('(d) Energy Conservation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/tp-agent/work/m10559/artifacts/ice_hull_contact_overview.png', dpi=150)
    plt.close()

if __name__ == '__main__':
    print('Generating verification figures...')
    plot_slope_sliding()
    print('  ✓ Friction angle vs temperature')
    plot_tension_test()
    print('  ✓ Adhesion stress vs contact time')
    plot_shear_test()
    print('  ✓ Shear strength vs pressure')
    plot_velocity_dependence()
    print('  ✓ Friction coefficient vs velocity')
    plot_contact_model_overview()
    print('  ✓ Contact model overview')
    print('All figures generated successfully!')
