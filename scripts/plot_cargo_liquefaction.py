"""Generate figures for cargo liquefaction report (m5471 / IN38-N06)."""
import math
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from tensordem.cargo_liquefaction import (
    CargoLiquefactionConfig,
    CargoLiquefactionSimulation,
    CyclicPorePressureModel,
    RollExcitation,
    FreeSurfaceCorrection,
    run_parametric_sweep,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


def fig_ru_evolution():
    """Figure 1: Pore pressure ratio ru(t) for different roll amplitudes."""
    fig, ax = plt.subplots(figsize=(8, 5))

    amplitudes = [5, 10, 15, 20, 25]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(amplitudes)))

    for amp, color in zip(amplitudes, colors):
        cfg = CargoLiquefactionConfig(
            roll_amplitude_deg=amp,
            roll_period=12,
            crr_ref=0.15,
            n_cycles=100,
            dt=1.0,
        )
        sim = CargoLiquefactionSimulation(config=cfg)
        res = sim.run()
        time_cycles = np.array(res['time']) / cfg.roll_period
        ax.plot(time_cycles, res['ru'], color=color, linewidth=1.5,
                label=f'φ₀ = {amp}°')

    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Onset (ru=0.8)')
    ax.axhline(y=1.0, color='darkred', linestyle='--', linewidth=1, alpha=0.7, label='Full (ru=1.0)')
    ax.set_xlabel('Number of roll cycles', fontsize=12)
    ax.set_ylabel('Pore pressure ratio $r_u = \\Delta u / \\sigma_{v0}$', fontsize=12)
    ax.set_title('Cyclic Pore Pressure Buildup vs Roll Amplitude', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'ru_evolution.png'), dpi=150)
    plt.close(fig)
    print("Saved: ru_evolution.png")


def fig_fsc_gz_correction():
    """Figure 2: GZ correction due to free surface effect."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Simple GZ curve (sinusoidal approximation for a bulk carrier)
    phi_range = np.linspace(0, 40, 100)
    phi_rad = np.radians(phi_range)
    gm = 2.5  # metacentric height, m
    gz_orig = gm * np.sin(phi_rad)

    # FSC for different ru values
    fsc_calc = FreeSurfaceCorrection(
        hold_length=20, hold_width=15,
        cargo_density=1600, cargo_height=3,
        displacement=50000,
    )

    for ru_val, ls in [(0.0, '-'), (0.8, '--'), (0.9, ':'), (0.95, '-.')]:
        fsc_val = fsc_calc.fsc(ru_val) if ru_val >= 0.8 else 0.0
        gz_corr = gz_orig - fsc_val * np.sin(phi_rad)
        label = f'ru={ru_val}' if ru_val > 0 else 'Solid cargo'
        if ru_val >= 0.8:
            label += f' (FSC={fsc_val:.3f} m)'
        ax.plot(phi_range, gz_corr, linestyle=ls, linewidth=1.5, label=label)

    ax.set_xlabel('Heel angle φ (degrees)', fontsize=12)
    ax.set_ylabel('GZ (m)', fontsize=12)
    ax.set_title('GZ Curve Correction for Liquefied Cargo', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'gz_correction.png'), dpi=150)
    plt.close(fig)
    print("Saved: gz_correction.png")


def fig_parametric_sweep():
    """Figure 3: Parametric sweep - liquefaction threshold in (amplitude, moisture) space."""
    results = run_parametric_sweep(
        roll_amplitudes_deg=[3, 5, 8, 10, 12, 15, 18, 20, 25, 30],
        moisture_contents=[0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.18, 0.20],
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    # Create a grid
    amps = sorted(set(r['roll_amplitude_deg'] for r in results))
    moistures = sorted(set(r['moisture_content'] for r in results))
    ru_grid = np.zeros((len(moistures), len(amps)))

    for r in results:
        i = moistures.index(r['moisture_content'])
        j = amps.index(r['roll_amplitude_deg'])
        ru_grid[i, j] = r['ru_max']

    # Contour plot
    X, Y = np.meshgrid(amps, moistures)
    cs = ax.contourf(X, Y, ru_grid, levels=[0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
                     cmap='RdYlBu_r', alpha=0.8)
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label('Max pore pressure ratio $r_u$', fontsize=11)

    # Contour line at ru=0.8 (liquefaction threshold)
    cs_lines = ax.contour(X, Y, ru_grid, levels=[0.8], colors='black',
                          linewidths=2, linestyles='-')
    ax.clabel(cs_lines, inline=True, fontsize=10, fmt='ru=0.8 threshold')

    ax.set_xlabel('Roll amplitude φ₀ (degrees)', fontsize=12)
    ax.set_ylabel('Moisture content w (mass fraction)', fontsize=12)
    ax.set_title('Liquefaction Stability Domain\n(Pore Pressure Ratio after 100 cycles)', fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'parametric_sweep.png'), dpi=150)
    plt.close(fig)
    print("Saved: parametric_sweep.png")


def fig_roll_kinematics():
    """Figure 4: Roll kinematics and induced acceleration."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    roll = RollExcitation(phi_0=math.radians(15), t_roll=12.0, z_cog=2.0)
    t = np.linspace(0, 36, 500)  # 3 cycles
    phi = [roll.roll_angle(ti) for ti in t]
    a_h = [roll.horizontal_acceleration(ti) for ti in t]

    axes[0].plot(t, np.degrees(phi), 'b-', linewidth=1.5)
    axes[0].set_ylabel('Roll angle φ (degrees)', fontsize=11)
    axes[0].set_title('Ship Roll Kinematics and Induced Horizontal Acceleration', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, a_h, 'r-', linewidth=1.5)
    axes[1].axhline(y=0, color='k', linewidth=0.5)
    axes[1].set_xlabel('Time (s)', fontsize=11)
    axes[1].set_ylabel('Horizontal accel. $a_h$ (m/s²)', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'roll_kinematics.png'), dpi=150)
    plt.close(fig)
    print("Saved: roll_kinematics.png")


def fig_stability_loss():
    """Figure 5: Stability loss (GZ reduction) vs liquefaction state."""
    fig, ax = plt.subplots(figsize=(8, 5))

    fsc_calc = FreeSurfaceCorrection(
        hold_length=20, hold_width=15,
        cargo_density=1600, cargo_height=3,
        displacement=50000,
    )

    ru_range = np.linspace(0, 1.05, 200)
    fsc_vals = [fsc_calc.fsc(ru) for ru in ru_range]

    ax.plot(ru_range, fsc_vals, 'b-', linewidth=2)
    ax.axvline(x=0.8, color='orange', linestyle='--', linewidth=1.5, label='Onset (ru=0.8)')
    ax.axvline(x=0.95, color='red', linestyle='--', linewidth=1.5, label='Partial (ru=0.95)')
    ax.axvline(x=1.0, color='darkred', linestyle='--', linewidth=1.5, label='Full (ru=1.0)')

    ax.set_xlabel('Pore pressure ratio $r_u$', fontsize=12)
    ax.set_ylabel('Free Surface Correction FSC (m)', fontsize=12)
    ax.set_title('GZ Stability Loss vs Liquefaction State', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'stability_loss.png'), dpi=150)
    plt.close(fig)
    print("Saved: stability_loss.png")


if __name__ == '__main__':
    fig_ru_evolution()
    fig_fsc_gz_correction()
    fig_parametric_sweep()
    fig_roll_kinematics()
    fig_stability_loss()
    print("\nAll figures generated successfully.")
