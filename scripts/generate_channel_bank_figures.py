#!/usr/bin/env python3
"""Generate figures for the channel bank collapse report."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import json
import base64
import io

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tensordem.channel_bank_collapse import (
    BankwallConfig, BankwallNetwork, IceFieldConfig, IceField,
    ChannelBankCollapseConfig, ChannelBankCollapseSim,
    rankine_k0, rankine_ka, rankine_kp, rankine_lateral_pressure,
    mohr_coulomb_shear_strength,
)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_rankine_coefficients():
    """Figure 1: Rankine earth pressure coefficients vs friction angle."""
    fig, ax = plt.subplots(figsize=(8, 5))
    phi = np.linspace(0, 50, 100)
    k0 = [rankine_k0(p) for p in phi]
    ka = [rankine_ka(p) for p in phi]
    kp = [rankine_kp(p) for p in phi]

    ax.plot(phi, k0, 'b-', label='K₀ (at-rest)', linewidth=2)
    ax.plot(phi, ka, 'r--', label='Kₐ (active)', linewidth=2)
    ax.plot(phi, kp, 'g-.', label='Kₚ (passive)', linewidth=2)
    ax.axvline(x=30, color='gray', linestyle=':', alpha=0.5)
    ax.text(31, 0.55, 'φ=30°', fontsize=10, color='gray')

    ax.set_xlabel('Friction angle φ (degrees)', fontsize=12)
    ax.set_ylabel('Earth pressure coefficient K', fontsize=12)
    ax.set_title('Rankine Earth Pressure Coefficients', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 5)

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_lateral_pressure_profile():
    """Figure 2: Lateral earth pressure vs depth."""
    fig, ax = plt.subplots(figsize=(8, 5))
    gamma = 20000.0  # N/m³
    phi = 30.0
    z = np.linspace(0, 10, 100)

    p_at_rest = [rankine_lateral_pressure(zi, gamma, phi, 'at_rest') for zi in z]
    p_active = [rankine_lateral_pressure(zi, gamma, phi, 'active') for zi in z]
    p_passive = [rankine_lateral_pressure(zi, gamma, phi, 'passive') for zi in z]

    ax.plot(p_at_rest, z, 'b-', label='At-rest (K₀)', linewidth=2)
    ax.plot(p_active, z, 'r--', label='Active (Kₐ)', linewidth=2)
    ax.plot(p_passive, z, 'g-.', label='Passive (Kₚ)', linewidth=2)

    ax.set_xlabel('Lateral pressure σₕ (Pa)', fontsize=12)
    ax.set_ylabel('Depth z (m)', fontsize=12)
    ax.set_title('Rankine Lateral Earth Pressure Profile (φ=30°, γ=20 kN/m³)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_mohr_coulomb_envelope():
    """Figure 3: Mohr-Coulomb failure envelope."""
    fig, ax = plt.subplots(figsize=(8, 5))
    c = 5000.0
    phi = 30.0
    sigma_n = np.linspace(0, 30000, 100)
    tau = np.array([mohr_coulomb_shear_strength(s, c, phi) for s in sigma_n])

    ax.plot(sigma_n / 1000, tau / 1000, 'b-', linewidth=2, label='τ = c + σ·tan(φ)')
    ax.fill_between(sigma_n / 1000, 0, tau / 1000, alpha=0.2, color='blue')

    # Mark a point
    sigma_mark = 15000
    tau_mark = mohr_coulomb_shear_strength(sigma_mark, c, phi)
    ax.plot(sigma_mark / 1000, tau_mark / 1000, 'ro', markersize=8)
    ax.annotate(f'({sigma_mark/1000:.0f}, {tau_mark/1000:.0f}) kPa',
                xy=(sigma_mark / 1000, tau_mark / 1000),
                xytext=(sigma_mark / 1000 + 2, tau_mark / 1000 + 1),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_xlabel('Normal stress σₙ (kPa)', fontsize=12)
    ax.set_ylabel('Shear strength τ (kPa)', fontsize=12)
    ax.set_title(f'Mohr–Coulomb Failure Envelope (c={c/1000:.0f} kPa, φ={phi}°)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_channel_width_evolution():
    """Figure 4: Channel width evolution over time."""
    cfg = BankwallConfig(
        nx=10, ny=8, block_size=0.5, density=2000.0,
        E=1e7, thickness=1.0, cohesion=2000.0,
        friction_angle=30.0, channel_width=50.0, water_depth=5.0,
    )
    bw = BankwallNetwork(cfg)

    for s in range(300):
        pressure = torch.full((len(bw.right),), 5000.0 + s * 50.0, dtype=torch.float64)
        bw.step(pressure)

    fig, ax = plt.subplots(figsize=(8, 5))
    times = [t for t, _ in bw.channel_width_history]
    widths = [w for _, w in bw.channel_width_history]

    ax.plot(times, widths, 'b-', linewidth=2, marker='o', markersize=3, markevery=5)
    ax.axhline(y=cfg.channel_width, color='gray', linestyle='--', alpha=0.5, label='Initial width')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Channel width (m)', fontsize=12)
    ax.set_title('Channel Width Evolution During Bankwall Collapse', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_bond_breaking_history():
    """Figure 5: Cumulative bond breaking over time."""
    cfg = BankwallConfig(
        nx=10, ny=8, block_size=0.5, density=2000.0,
        E=1e7, thickness=1.0, cohesion=2000.0,
        friction_angle=30.0, channel_width=50.0, water_depth=5.0,
    )
    bw = BankwallNetwork(cfg)

    history = []
    for s in range(300):
        pressure = torch.full((len(bw.right),), 5000.0 + s * 50.0, dtype=torch.float64)
        info = bw.step(pressure)
        history.append(info)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    times = [h['time'] for h in history]
    n_broken = [h['n_broken'] for h in history]
    n_newly = [h['n_newly_broken'] for h in history]

    ax1.plot(times, n_broken, 'r-', linewidth=2)
    ax1.set_ylabel('Cumulative broken bonds', fontsize=11)
    ax1.set_title('Bond Breaking History', fontsize=13)
    ax1.grid(True, alpha=0.3)

    ax2.bar(times[::5], n_newly[::5], width=0.02, color='orange', alpha=0.7)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Newly broken bonds/step', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_energy_balance():
    """Figure 6: Energy balance over time."""
    cfg = BankwallConfig(
        nx=8, ny=6, block_size=0.5, density=2000.0,
        E=1e8, thickness=1.0, cohesion=1000.0,
        friction_angle=30.0, channel_width=50.0, water_depth=5.0,
    )
    bw = BankwallNetwork(cfg)

    history = []
    for s in range(150):
        pressure = torch.full((len(bw.right),), 10000.0, dtype=torch.float64)
        info = bw.step(pressure)
        history.append({
            'time': bw.time,
            'kinetic_energy': bw.kinetic_energy,
            'strain_energy': bw.strain_energy,
            'external_work': bw.total_external_work,
            'fracture_energy': bw.total_fracture_energy,
        })

    fig, ax = plt.subplots(figsize=(8, 5))
    times = [h['time'] for h in history]

    ax.plot(times, [h['external_work'] for h in history], 'b-', linewidth=2, label='External work')
    ax.plot(times, [h['strain_energy'] for h in history], 'g--', linewidth=2, label='Strain energy')
    ax.plot(times, [h['kinetic_energy'] for h in history], 'r:', linewidth=2, label='Kinetic energy')
    ax.plot(times, [h['fracture_energy'] for h in history], 'm-.', linewidth=2, label='Fracture energy')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Energy (J)', fontsize=12)
    ax.set_title('Energy Balance During Bankwall Loading', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def main():
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
    os.makedirs(output_dir, exist_ok=True)

    figures = {
        'rankine_coefficients': {
            'name': 'Rankine Earth Pressure Coefficients',
            'caption': 'Rankine earth pressure coefficients K₀, Kₐ, Kₚ as functions of internal friction angle φ. For φ=30°: K₀=0.5, Kₐ=1/3, Kₚ=3.0.',
            'png_base64': generate_rankine_coefficients(),
        },
        'lateral_pressure_profile': {
            'name': 'Lateral Earth Pressure Profile',
            'caption': 'Lateral earth pressure σₕ = K·γ·z vs depth for φ=30°, γ=20 kN/m³. At-rest, active, and passive states shown.',
            'png_base64': generate_lateral_pressure_profile(),
        },
        'mohr_coulomb_envelope': {
            'name': 'Mohr-Coulomb Failure Envelope',
            'caption': 'Mohr-Coulomb shear failure envelope τ = c + σₙ·tan(φ) for c=5 kPa, φ=30°. The shaded region represents stable stress states.',
            'png_base64': generate_mohr_coulomb_envelope(),
        },
        'channel_width_evolution': {
            'name': 'Channel Width Evolution',
            'caption': 'Channel width W(t) decreases monotonically as bankwall blocks detach under increasing ice pressure. Initial width = 50 m.',
            'png_base64': generate_channel_width_evolution(),
        },
        'bond_breaking_history': {
            'name': 'Bond Breaking History',
            'caption': 'Cumulative broken bonds (top) and newly broken bonds per step (bottom) during bankwall collapse simulation.',
            'png_base64': generate_bond_breaking_history(),
        },
        'energy_balance': {
            'name': 'Energy Balance',
            'caption': 'Energy components during bankwall loading: external work, strain energy, kinetic energy, and fracture energy release.',
            'png_base64': generate_energy_balance(),
        },
    }

    # Save figures as individual PNG files
    for name, fig_data in figures.items():
        png_path = os.path.join(output_dir, f'{name}.png')
        with open(png_path, 'wb') as f:
            f.write(base64.b64decode(fig_data['png_base64']))
        print(f"Saved {png_path}")

    # Save figures JSON for report
    figures_json_path = os.path.join(output_dir, 'channel_bank_collapse_figures.json')
    with open(figures_json_path, 'w') as f:
        json.dump(figures, f)
    print(f"Saved {figures_json_path}")

    return figures


if __name__ == '__main__':
    figures = main()
    print(f"Generated {len(figures)} figures")
