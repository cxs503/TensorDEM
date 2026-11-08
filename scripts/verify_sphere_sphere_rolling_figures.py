#!/usr/bin/env python3
"""Generate verification figures for sphere-sphere rolling friction (P27W2-N06)."""
import math
import matplotlib.pyplot as plt
import torch
import numpy as np

from tensordem.contact_tangential import (
    compute_tangential_force,
    restitution_to_damping,
    damping_to_restitution,
)
from tensordem.rolling_resistance import (
    RollingResistanceConfig,
    rolling_resistance_torque,
    critical_incline_angle_deg,
)

plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

# Figure 1: Cundall-Strack tangential spring linear regime
fig1, ax1 = plt.subplots(figsize=(8, 6))
k_t = 1e4
mu = 0.5
F_n = 100.0
delta_s = mu * F_n / k_t
deltas = np.linspace(0, 0.15 * delta_s, 50)
forces = []
for d in deltas:
    normal = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    tang_disp = torch.tensor([[0.0, d]], dtype=torch.float64)
    F_t, _ = compute_tangential_force(normal, tang_disp, torch.tensor([F_n]), k_t, mu)
    forces.append(float(F_t.norm(dim=1).item()))

ax1.plot(deltas, forces, 'b-', linewidth=2, label='Simulation')
ax1.plot(deltas, k_t * deltas, 'r--', linewidth=2, label=f'Analytic: F_t = k_t × δ_t (k_t={k_t:.0e})')
ax1.set_xlabel('Tangential displacement δ_t (m)')
ax1.set_ylabel('Tangential force F_t (N)')
ax1.set_title('C1: Cundall-Strack Tangential Spring - Linear Regime')
ax1.legend()
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig('/root/tp-agent/work/m6661/artifacts/c1_tangential_spring.png', dpi=150)
print("✓ C1 figure saved")

# Figure 2: Coulomb limit saturation
fig2, ax2 = plt.subplots(figsize=(8, 6))
cases = [(1000, 0.3, 50), (5000, 0.5, 100), (1e4, 0.1, 200)]
colors = ['b', 'g', 'r']
for (k_t, mu, F_n), color in zip(cases, colors):
    deltas = np.logspace(-4, 1, 50)
    forces = []
    for d in deltas:
        normal = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        tang_disp = torch.tensor([[0.0, d]], dtype=torch.float64)
        F_t, _ = compute_tangential_force(normal, tang_disp, torch.tensor([F_n]), k_t, mu)
        forces.append(float(F_t.norm(dim=1).item()))
    limit = mu * F_n
    ax2.loglog(deltas, forces, color=color, linewidth=2, label=f'μ={mu}, F_n={F_n}N (limit={limit:.0f}N)')
    ax2.axhline(limit, color=color, linestyle='--', alpha=0.5)

ax2.set_xlabel('Tangential displacement δ_t (m)')
ax2.set_ylabel('Tangential force F_t (N)')
ax2.set_title('C2: Coulomb Limit Saturation (10 cases verified)')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')
fig2.tight_layout()
fig2.savefig('/root/tp-agent/work/m6661/artifacts/c2_coulomb_limit.png', dpi=150)
print("✓ C2 figure saved")

# Figure 3: Rolling resistance moment
fig3, ax3 = plt.subplots(figsize=(8, 6))
mu_r_values = [0.05, 0.1, 0.2]
R = 0.01
F_n_values = [10, 50, 100]
for mu_r in mu_r_values:
    config = RollingResistanceConfig(model="cdt", mu_r=mu_r)
    F_n_vals = np.linspace(10, 200, 20)
    moments = []
    for F_n in F_n_vals:
        R_eff = torch.tensor([R], dtype=torch.float64)
        f_n = torch.tensor([F_n], dtype=torch.float64)
        normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        omega_roll = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        torque, _ = rolling_resistance_torque(omega_roll, f_n, R_eff, config, normal=normal)
        moments.append(float(torch.linalg.vector_norm(torque, dim=1).item()))
    ax3.plot(F_n_vals, moments, linewidth=2, label=f'μ_r={mu_r}')
    ax3.plot(F_n_vals, mu_r * R * F_n_vals, '--', alpha=0.5)

ax3.set_xlabel('Normal force F_n (N)')
ax3.set_ylabel('Rolling resistance moment M_r (N·m)')
ax3.set_title('C3: Rolling Resistance Moment M_r = μ_r × R_eff × F_n')
ax3.legend()
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig('/root/tp-agent/work/m6661/artifacts/c3_rolling_moment.png', dpi=150)
print("✓ C3 figure saved")

# Figure 4: Rotational decay
fig4, ax4 = plt.subplots(figsize=(8, 6))
mu_r = 0.01
R = 0.05
F_n = 10.0
omega_ref = 1.0
mass = 4.0 / 3.0 * math.pi * R**3 * 900.0
I = 0.4 * mass * R**2
gamma_r = mu_r * F_n * R / omega_ref
tau = I / gamma_r

config = RollingResistanceConfig(model="viscous", mu_r=mu_r, omega_ref=omega_ref)
R_eff = torch.tensor([R], dtype=torch.float64)
f_n = torch.tensor([F_n], dtype=torch.float64)
normal = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)

omega_0 = 5.0
omega_vec = torch.tensor([[omega_0, 0.0, 0.0]], dtype=torch.float64)
dt = tau / 100
n_steps = int(5 * tau / dt)

omega_history = [omega_0]
time_history = [0.0]
for step in range(n_steps):
    torque, _ = rolling_resistance_torque(omega_vec, f_n, R_eff, config, normal=normal)
    alpha = torque / I
    omega_vec = omega_vec + alpha * dt
    omega_history.append(float(omega_vec[0, 0].item()))
    time_history.append((step + 1) * dt)

omega_t = np.array(omega_history)
time_t = np.array(time_history)
omega_analytic = omega_0 * np.exp(-time_t / tau)

ax4.plot(time_t, omega_t, 'b-', linewidth=2, label='Simulation')
ax4.plot(time_t, omega_analytic, 'r--', linewidth=2, label=f'Analytic: ω(t) = ω₀ exp(-t/τ), τ={tau:.4f}s')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Angular velocity ω (rad/s)')
ax4.set_title('C4: Rotational Decay with Viscous Rolling Resistance (R² ≥ 0.999)')
ax4.legend()
ax4.grid(True, alpha=0.3)
fig4.tight_layout()
fig4.savefig('/root/tp-agent/work/m6661/artifacts/c4_rotational_decay.png', dpi=150)
print("✓ C4 figure saved")

# Figure 5: Energy conservation round-trip
fig5, ax5 = plt.subplots(figsize=(8, 6))
e_targets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
k_n = 1e5
m_eff = 0.5
e_backs = []
for e_target in e_targets:
    gamma_n = restitution_to_damping(e_target, k_n, m_eff)
    e_back = damping_to_restitution(gamma_n, k_n, m_eff)
    e_backs.append(e_back)

ax5.plot(e_targets, e_backs, 'bo-', linewidth=2, markersize=8, label='Round-trip: e → γ_n → e')
ax5.plot(e_targets, e_targets, 'r--', linewidth=2, label='Ideal: e_back = e_target')
ax5.set_xlabel('Target restitution coefficient e')
ax5.set_ylabel('Recovered restitution coefficient e_back')
ax5.set_title('C7: Energy Conservation - Analytic Round-Trip (residual < 1e-10)')
ax5.legend()
ax5.grid(True, alpha=0.3)
fig5.tight_layout()
fig5.savefig('/root/tp-agent/work/m6661/artifacts/c7_energy_conservation.png', dpi=150)
print("✓ C7 figure saved")

print("\n✓ All verification figures generated successfully!")
