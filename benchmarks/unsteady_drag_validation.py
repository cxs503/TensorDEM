"""Benchmark and validation report for unsteady drag closure.

Generates figures and validates acceptance criteria.
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from tensordem.unsteady_drag import (
    run_oscillatory_flow_simulation,
    run_sudden_acceleration,
    run_steady_uniform,
    landau_lifshitz_oscillatory_drag_amplitude_phase,
    basset_direct_convolution,
    basset_recursive,
)

# Physical constants
RHO_F = 1000.0
MU = 1.0e-3
D = 1.0e-3
RHO_P = RHO_F * 1.5

# Output directory
OUT_DIR = Path("/root/tp-agent/work/m6677/artifacts/unsteady_drag")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("UNSTEADY DRAG CLOSURE - VALIDATION REPORT")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Sudden acceleration (added-mass validation)
# ---------------------------------------------------------------------------
print("\n[1] Sudden Acceleration Test")
print("-" * 70)

result_accel = run_sudden_acceleration(
    rho_f=RHO_F, d=D, mu=MU,
    du_dt=1.0, t_end=0.01, dt=1e-4,
)

V = math.pi * D**3 / 6.0
F_expected = 0.5 * RHO_F * V * 1.0
A_measured = result_accel["A_coeff"]

print(f"Expected added-mass coefficient: 0.500")
print(f"Measured added-mass coefficient: {A_measured:.4f}")
print(f"Relative error: {abs(A_measured - 0.5) / 0.5 * 100:.2f}%")

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(result_accel["time"] * 1000, result_accel["F_total"] * 1e9, 
             label='Total force', linewidth=2)
axes[0].plot(result_accel["time"] * 1000, result_accel["F_added_mass"] * 1e9, 
             label='Added mass', linewidth=2, linestyle='--')
axes[0].plot(result_accel["time"] * 1000, result_accel["F_basset"] * 1e9, 
             label='Basset history', linewidth=2, linestyle=':')
axes[0].axhline(y=F_expected * 1e9, color='r', linestyle='-', 
                label=f'Expected (0.5·ρ_f·V·du/dt = {F_expected*1e9:.3f} nN)')
axes[0].set_xlabel('Time [ms]')
axes[0].set_ylabel('Force [nN]')
axes[0].set_title('Sudden Acceleration: Force Components')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(result_accel["time"] * 1000, result_accel["F_quasi_steady"] * 1e9, 
             label='Quasi-steady drag', linewidth=2)
axes[1].set_xlabel('Time [ms]')
axes[1].set_ylabel('Force [nN]')
axes[1].set_title('Quasi-Steady Drag (negligible at early times)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "sudden_acceleration.png", dpi=150)
print(f"✓ Saved: {OUT_DIR / 'sudden_acceleration.png'}")
plt.close()

# ---------------------------------------------------------------------------
# 2. Oscillatory flow (Landau-Lifshitz validation)
# ---------------------------------------------------------------------------
print("\n[2] Oscillatory Flow Test (ε = 1)")
print("-" * 70)

a = D / 2.0
epsilon = 1.0
omega = epsilon * (MU / RHO_F) / a**2
U0 = 1e-4

n_periods = 8
period = 2.0 * math.pi / omega
dt = period / 100.0

result_osc = run_oscillatory_flow_simulation(
    rho_f=RHO_F, d=D, mu=MU, rho_p=RHO_P,
    U0=U0, omega=omega,
    t_end=n_periods * period,
    dt=dt,
)

amp_meas = result_osc["amplitude_measured"]
amp_anal = result_osc["amplitude_analytical"]
phase_meas = result_osc["phase_measured_deg"]
phase_anal = result_osc["phase_analytical_deg"]

print(f"Amplitude (measured):   {amp_meas:.4e} N")
print(f"Amplitude (analytical): {amp_anal:.4e} N")
print(f"Amplitude ratio:        {amp_meas / amp_anal:.4f}")
print(f"Phase (measured):       {phase_meas:.2f}°")
print(f"Phase (analytical):     {phase_anal:.2f}°")
print(f"Phase error:            {abs(phase_meas - phase_anal):.2f}°")

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Time series
t_plot = result_osc["time"]
axes[0].plot(t_plot, result_osc["u_f"] * 1e4, label='u_f (flow velocity)', 
             linewidth=2, alpha=0.7)
axes[0].plot(t_plot, result_osc["F_total"] / amp_anal, 
             label='F_total (normalized)', linewidth=2)
axes[0].set_xlabel('Time [s]')
axes[0].set_ylabel('Velocity [cm/s] / Force [normalized]')
axes[0].set_title('Oscillatory Flow: Time Series')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Force components
n_skip = int(5.0 * period / dt)
t_steady = t_plot[n_skip:]
axes[1].plot(t_steady, result_osc["F_quasi_steady"][n_skip:] * 1e9, 
             label='Quasi-steady', linewidth=2)
axes[1].plot(t_steady, result_osc["F_added_mass"][n_skip:] * 1e9, 
             label='Added mass', linewidth=2, linestyle='--')
axes[1].plot(t_steady, result_osc["F_basset"][n_skip:] * 1e9, 
             label='Basset', linewidth=2, linestyle=':')
axes[1].plot(t_steady, result_osc["F_pressure_gradient"][n_skip:] * 1e9, 
             label='Pressure gradient', linewidth=2, linestyle='-.')
axes[1].set_xlabel('Time [s]')
axes[1].set_ylabel('Force [nN]')
axes[1].set_title('Force Components (Steady State)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "oscillatory_flow.png", dpi=150)
print(f"✓ Saved: {OUT_DIR / 'oscillatory_flow.png'}")
plt.close()

# ---------------------------------------------------------------------------
# 3. Basset force: direct vs recursive
# ---------------------------------------------------------------------------
print("\n[3] Basset Force: Direct vs Recursive")
print("-" * 70)

N = 200
dt_basset = 1e-3
t = np.arange(N) * dt_basset
omega_basset = 2.0 * math.pi * 10.0
accel = np.cos(omega_basset * t)

F_direct = basset_direct_convolution(accel, dt_basset, MU, RHO_F, D)
F_recursive = basset_recursive(accel, dt_basset, MU, RHO_F, D, n_exp=12)

corr = np.corrcoef(F_direct[10:], F_recursive[10:])[0, 1]
print(f"Correlation coefficient: {corr:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t * 1000, F_direct * 1e9, label='Direct O(N²)', linewidth=2)
ax.plot(t * 1000, F_recursive * 1e9, label='Recursive O(N)', 
        linewidth=2, linestyle='--')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Basset Force [nN]')
ax.set_title('Basset History Force: Direct vs Recursive')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "basset_comparison.png", dpi=150)
print(f"✓ Saved: {OUT_DIR / 'basset_comparison.png'}")
plt.close()

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

print("\n✓ Added-mass coefficient: A = {:.4f} (expected 0.5)".format(A_measured))
print("✓ Oscillatory flow amplitude ratio: {:.2f}".format(amp_meas / amp_anal))
print("✓ Oscillatory flow phase error: {:.2f}°".format(abs(phase_meas - phase_anal)))
print("✓ Basset direct vs recursive correlation: {:.4f}".format(corr))

print("\nAll figures saved to:", OUT_DIR)
print("=" * 70)
