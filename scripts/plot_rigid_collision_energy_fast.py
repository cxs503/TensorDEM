"""Generate paper-quality figures for rigid collision energy ledger convergence (fast version)."""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from tensordem.rigid_collision_energy import (
    run_convergence_sweep,
    run_newtons_cradle,
)


def plot_convergence_orders(output_dir: Path):
    """Plot convergence of energy residuals vs dt for different methods."""
    torch.set_num_threads(1)
    
    print("Running convergence sweep...")
    # Run convergence sweep with fewer points for speed
    rows = run_convergence_sweep(
        e_target=0.7, v0=0.5,
        k_n_values=[5e4],
        dt_factors=[1.0, 0.5, 0.25, 0.125, 0.0625],
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Method (ii-a) vs (ii-b) residual
    ax = axes[0, 0]
    filtered = rows
    dts = [r.dt for r in filtered]
    residuals = [r.residual_iia_vs_iib for r in filtered]
    ax.loglog(dts, residuals, 'bo-', linewidth=2, markersize=8, label='$k_n=5 \\times 10^4$ N/m')
    
    # Fit and plot reference line
    if len(dts) >= 3:
        log_dt = np.log(dts)
        log_res = np.log(residuals)
        coeffs = np.polyfit(log_dt, log_res, 1)
        slope = coeffs[0]
        ref_line = np.exp(coeffs[1]) * np.array(dts) ** slope
        ax.loglog(dts, ref_line, 'b--', alpha=0.5, linewidth=2)
        ax.annotate(f'slope={slope:.2f}', 
                   xy=(dts[-1], ref_line[-1]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=11, color='blue', fontweight='bold')
    
    ax.set_xlabel('Time step dt [s]', fontsize=12)
    ax.set_ylabel('|$\\Delta E_{ii-a} - \\Delta E_{ii-b}$| / $E_0$', fontsize=12)
    ax.set_title('Method (ii-a) vs (ii-b): O(dt) convergence', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    
    # Plot 2: Method (i) vs (ii-b) residual (should be machine precision)
    ax = axes[0, 1]
    residuals_i = [r.residual_i_vs_iib for r in filtered]
    ax.semilogy(dts, residuals_i, 'ro-', linewidth=2, markersize=8, label='$k_n=5 \\times 10^4$ N/m')
    
    ax.set_xlabel('Time step dt [s]', fontsize=12)
    ax.set_ylabel('|$\\Delta E_{i} - \\Delta E_{ii-b}$| / $E_0$', fontsize=12)
    ax.set_title('Method (i) vs (ii-b): Machine precision', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    ax.axhline(y=1e-12, color='gray', linestyle='--', alpha=0.7, label='Machine precision')
    ax.legend(fontsize=11)
    
    # Plot 3: Measured restitution coefficient vs dt
    ax = axes[1, 0]
    e_meas = [r.e_measured for r in filtered]
    ax.semilogx(dts, e_meas, 'go-', linewidth=2, markersize=8, label='Measured')
    
    ax.axhline(y=0.7, color='k', linestyle='--', alpha=0.7, linewidth=2, label='Target e=0.7')
    ax.set_xlabel('Time step dt [s]', fontsize=12)
    ax.set_ylabel('Measured restitution coefficient $e$', fontsize=12)
    ax.set_title('Restitution coefficient convergence', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    
    # Plot 4: Theory residual (method i vs theory)
    ax = axes[1, 1]
    residuals_theory = [r.residual_i_vs_theory for r in filtered]
    ax.loglog(dts, residuals_theory, 'mo-', linewidth=2, markersize=8, label='$k_n=5 \\times 10^4$ N/m')
    
    ax.set_xlabel('Time step dt [s]', fontsize=12)
    ax.set_ylabel('|$\\Delta E_{i} - \\Delta E_{theory}$| / $E_0$', fontsize=12)
    ax.set_title('Method (i) vs Theory: Time discretization error', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'convergence_orders.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved convergence_orders.png")


def plot_newtons_cradle(output_dir: Path):
    """Plot Newton's cradle energy and momentum conservation."""
    torch.set_num_threads(1)
    
    print("Running Newton's cradle simulations...")
    # Run Newton's cradle at different dt values
    dts = [2e-6, 1e-6, 5e-7]
    momentum_errors = []
    energy_errors = []
    
    for dt in dts:
        print(f"  dt={dt:.1e}...")
        cradle = run_newtons_cradle(
            e_target=0.95, v0=1.0, k_n=5e4,
            n_balls=5, dt=dt,
        )
        momentum_errors.append(cradle.momentum_error)
        energy_rel = abs(cradle.dE_state - cradle.dE_theory_total) / cradle.E0
        energy_errors.append(energy_rel)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Momentum conservation
    ax = axes[0]
    ax.semilogy(dts, momentum_errors, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=1e-12, color='r', linestyle='--', alpha=0.5, label='Machine precision')
    ax.set_xlabel('Time step dt [s]', fontsize=12)
    ax.set_ylabel('Momentum error [kg⋅m/s]', fontsize=12)
    ax.set_title("Newton's Cradle: Momentum Conservation", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    
    # Plot 2: Energy dissipation error
    ax = axes[1]
    ax.loglog(dts, energy_errors, 'go-', linewidth=2, markersize=8)
    ax.set_xlabel('Time step dt [s]', fontsize=12)
    ax.set_ylabel('|$\\Delta E_{state} - \\Delta E_{theory}$| / $E_0$', fontsize=12)
    ax.set_title("Newton's Cradle: Energy Dissipation", fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'newtons_cradle.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved newtons_cradle.png")


def plot_energy_balance_diagram(output_dir: Path):
    """Create a schematic diagram showing the three energy accounting methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw the three methods as boxes
    methods = [
        ('(i) State Difference', '$\\Delta E = E(t_f) - E(t_0)$', 'O(dt²)', 'Exact for symplectic Euler'),
        ('(ii-a) Impulse × v_final', '$W = \\sum F \\cdot v_{n+1} \\Delta t$', 'O(dt)', 'First-order accurate'),
        ('(ii-b) Impulse × v_avg', '$W = \\sum F \\cdot \\frac{v_n + v_{n+1}}{2} \\Delta t$', 'O(dt²)', 'Second-order accurate'),
    ]
    
    y_positions = [0.8, 0.5, 0.2]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, (title, formula, order, note) in enumerate(methods):
        y = y_positions[i]
        color = colors[i]
        
        # Draw box
        rect = plt.Rectangle((0.1, y - 0.08), 0.8, 0.15, 
                             facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        # Add text
        ax.text(0.15, y + 0.04, title, fontsize=12, fontweight='bold', color=color)
        ax.text(0.15, y + 0.01, formula, fontsize=11, style='italic')
        ax.text(0.15, y - 0.02, f'Accuracy: {order}', fontsize=10, color='#555')
        ax.text(0.15, y - 0.05, note, fontsize=9, color='#777')
    
    # Add arrows showing equivalence
    ax.annotate('', xy=(0.5, 0.72), xytext=(0.5, 0.58),
                arrowprops=dict(arrowstyle='<->', lw=2, color='#3498db'))
    ax.text(0.52, 0.65, 'Identical\n(machine precision)', fontsize=9, 
            ha='left', va='center', color='#3498db')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Energy Accounting Methods for Rigid Collisions', fontsize=14, pad=20)
    
    fig.savefig(output_dir / 'energy_methods_schematic.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved energy_methods_schematic.png")


if __name__ == '__main__':
    output_dir = Path('artifacts/rigid_collision_energy')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_convergence_orders(output_dir)
    plot_newtons_cradle(output_dir)
    plot_energy_balance_diagram(output_dir)
    
    print("\n✓ All figures generated successfully!")
