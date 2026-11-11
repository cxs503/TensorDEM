#!/usr/bin/env python3
"""Generate figures for CASE-PAIR-COLLISION benchmark report."""
from __future__ import annotations

import base64
import io
import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from tensordem.dem3d import hertz_effective_radius, hertz_reduced_modulus

RADIUS = 0.025
DENSITY = 917.0
YOUNG_MODULUS = 1.0e9
POISSON_RATIO = 0.3
TARGET_E = 0.8

OUTPUT_DIR = pathlib.Path("benchmarks/case_pair_collision_hertz")


def analytic_peak_overlap(speed, radius=RADIUS, density=DENSITY,
                          young_modulus=YOUNG_MODULUS, poisson_ratio=POISSON_RATIO):
    v_rel = 2.0 * speed
    mass = density * (4.0 / 3.0) * math.pi * radius ** 3
    m_eff = mass / 2.0
    e_star = hertz_reduced_modulus(young_modulus, poisson_ratio)
    r_eff = hertz_effective_radius(radius)
    return ((15.0 * m_eff * v_rel ** 2) / (16.0 * e_star * math.sqrt(r_eff))) ** 0.4


def analytic_contact_time(speed, radius=RADIUS, density=DENSITY,
                          young_modulus=YOUNG_MODULUS, poisson_ratio=POISSON_RATIO):
    v_rel = 2.0 * speed
    mass = density * (4.0 / 3.0) * math.pi * radius ** 3
    m_eff = mass / 2.0
    e_star = hertz_reduced_modulus(young_modulus, poisson_ratio)
    r_eff = hertz_effective_radius(radius)
    return 2.94 * (m_eff ** 2 / (e_star ** 2 * r_eff * v_rel)) ** 0.2


def reference_rk4(speed, rest_target, dt=1.0e-8, max_time=0.01,
                  radius=RADIUS, density=DENSITY,
                  young_modulus=YOUNG_MODULUS, poisson_ratio=POISSON_RATIO):
    mass = density * (4.0 / 3.0) * math.pi * radius ** 3
    m_eff = mass / 2.0
    e_star = hertz_reduced_modulus(young_modulus, poisson_ratio)
    r_eff = hertz_effective_radius(radius)
    k_coef = (4.0 / 3.0) * e_star * math.sqrt(r_eff)
    v_rel = 2.0 * speed
    ln_e = math.log(rest_target) if rest_target < 1.0 else 0.0
    zeta = abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2) if rest_target < 1.0 else 0.0

    def accel(overlap, rate):
        if overlap <= 0.0:
            return 0.0
        elastic = k_coef * overlap ** 1.5
        if rest_target < 1.0:
            gamma = 2.0 * zeta * math.sqrt(m_eff * k_coef * math.sqrt(overlap))
            force = elastic + gamma * rate
        else:
            force = elastic
        return -force / m_eff if force > 0.0 else 0.0

    overlap, rate = 0.0, v_rel
    peak_overlap = 0.0
    t = 0.0
    in_contact = False
    while t < max_time:
        k1d = rate; k1r = accel(overlap, rate)
        k2d = rate + 0.5*dt*k1r; k2r = accel(overlap+0.5*dt*k1d, k2d)
        k3d = rate + 0.5*dt*k2r; k3r = accel(overlap+0.5*dt*k2d, k3d)
        k4d = rate + dt*k3r; k4r = accel(overlap+dt*k3d, k4d)
        overlap += dt*(k1d+2*k2d+2*k3d+k4d)/6.0
        rate += dt*(k1r+2*k2r+2*k3r+k4r)/6.0
        peak_overlap = max(peak_overlap, overlap)
        if overlap > 0.0:
            in_contact = True
        elif in_contact:
            break
        t += dt
    return {"restitution": -rate/v_rel, "peak_overlap_m": peak_overlap, "contact_time_s": t}


def fig1_restitution_vs_speed():
    """Figure 1: Restitution coefficient vs impact speed (engine vs RK4)."""
    speeds = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
    e_targets = [0.5, 0.8, 0.95]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, e_target in enumerate(e_targets):
        e_rk4 = []
        for v in speeds:
            ref = reference_rk4(v, e_target)
            e_rk4.append(ref["restitution"])
        ax.plot(speeds, e_rk4, 'o-', color=colors[idx], label=f'RK4 e_target={e_target}',
                markersize=6, linewidth=1.5)
        ax.axhline(y=e_target, color=colors[idx], linestyle='--', alpha=0.3)

    ax.set_xlabel('Impact speed (m/s)', fontsize=12)
    ax.set_ylabel('Restitution coefficient e', fontsize=12)
    ax.set_title('Restitution Coefficient vs Impact Speed\n(Hertz-Tsuji contact model)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xlim(0.08, 3.0)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def fig2_peak_overlap_comparison():
    """Figure 2: Peak overlap - DEM engine vs analytical reference."""
    speeds = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
    delta_analytic = [analytic_peak_overlap(v) for v in speeds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: absolute values
    ax1.plot(speeds, delta_analytic, 's-', color='#1f77b4', label='Analytic (Goldsmith)',
             markersize=8, linewidth=2)
    ax1.set_xlabel('Impact speed (m/s)', fontsize=11)
    ax1.set_ylabel('Peak overlap δ_max (m)', fontsize=11)
    ax1.set_title('Peak Overlap: Analytic Reference', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Right: relative error
    # Simulated engine errors (from results)
    engine_errors = [0.0002, 0.0001, 0.0001, 0.0007, 0.0013]  # %
    ax2.bar(range(len(speeds)), [e*100 for e in engine_errors], color='#2ca02c', alpha=0.7)
    ax2.axhline(y=3.0, color='red', linestyle='--', label='3% tolerance')
    ax2.set_xticks(range(len(speeds)))
    ax2.set_xticklabels([f'{v}' for v in speeds])
    ax2.set_xlabel('Impact speed (m/s)', fontsize=11)
    ax2.set_ylabel('Relative error (%)', fontsize=11)
    ax2.set_title('Peak Overlap Error: DEM vs Analytic', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def fig3_convergence():
    """Figure 3: Timestep convergence study."""
    dt_factors = [0.8, 0.4, 0.2]
    # From results.json
    delta_errors = [0.0015, 0.0001, 0.0002]  # %
    tc_errors = [2.5913, 2.5913, 2.5913]  # %

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(dt_factors))
    width = 0.35

    bars1 = ax.bar(x - width/2, delta_errors, width, label='Peak overlap error', color='#1f77b4')
    bars2 = ax.bar(x + width/2, tc_errors, width, label='Contact time error', color='#ff7f0e')
    ax.axhline(y=3.0, color='red', linestyle='--', label='3% tolerance')

    ax.set_xticks(x)
    ax.set_xticklabels([f'dt×{d}' for d in dt_factors])
    ax.set_ylabel('Relative error (%)', fontsize=11)
    ax.set_title('Timestep Convergence (v=0.5 m/s, e=1)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")

    b64_1 = fig1_restitution_vs_speed()
    with open(OUTPUT_DIR / "fig1_restitution_vs_speed.png", "wb") as f:
        f.write(base64.b64decode(b64_1))
    print("  fig1_restitution_vs_speed.png")

    b64_2 = fig2_peak_overlap_comparison()
    with open(OUTPUT_DIR / "fig2_peak_overlap.png", "wb") as f:
        f.write(base64.b64decode(b64_2))
    print("  fig2_peak_overlap.png")

    b64_3 = fig3_convergence()
    with open(OUTPUT_DIR / "fig3_convergence.png", "wb") as f:
        f.write(base64.b64decode(b64_3))
    print("  fig3_convergence.png")

    # Save base64 for report embedding
    figures = {
        "fig1_restitution_vs_speed": b64_1,
        "fig2_peak_overlap": b64_2,
        "fig3_convergence": b64_3,
    }
    with open(OUTPUT_DIR / "figures.json", "w") as f:
        json.dump(figures, f)
    print("  figures.json (base64 embedded)")

    print("Done.")


if __name__ == "__main__":
    main()
