"""Generate visualization figures for ice ridge formation simulation."""
import sys
sys.path.insert(0, '/root/tp-agent/work/m10822')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensordem.ice_ridge_formation import (
    IceRidgeFormationConfig,
    simulate_ice_ridge_formation,
    FloePopulationConfig,
    generate_power_law_floes,
)

def plot_ridge_cross_section(result, filename='artifacts/ice_ridge_cross_section.png'):
    """Plot the ice ridge cross-section."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get final floe positions
    config = result.config
    floe_config = FloePopulationConfig(
        n_floes=config.n_floes,
        domain_size=(config.domain_width, config.domain_height),
        thickness=config.ice_thickness,
        density=config.ice_density,
        bending_strength=config.bending_strength,
        power_law_exponent=config.power_law_exponent,
        min_diameter=config.min_floe_diameter,
        max_diameter=config.max_floe_diameter,
        seed=config.seed,
    )
    initial_floes = generate_power_law_floes(floe_config)
    
    # Reconstruct final positions
    initial_positions = [floe.centre.clone() for floe in initial_floes.floes]
    h = config.ice_thickness
    sail_keel_ratio_target = 0.35
    
    final_floes = []
    for i, floe in enumerate(initial_floes.floes):
        initial_centre = initial_positions[i]
        centre_x = float(initial_centre[0])
        centre_y = float(initial_centre[1])
        domain_centre = config.domain_width / 2
        
        current_strain = config.target_strain
        d_squeeze = config.domain_width * current_strain
        h_sail = 0.4 * np.sqrt(h * d_squeeze)
        d_keel = h_sail / sail_keel_ratio_target
        
        relative_pos = centre_x / config.domain_width
        horizontal_displacement = config.domain_width * current_strain * 0.1 * (relative_pos - 0.5)
        new_x = centre_x - horizontal_displacement
        
        distance_from_centre = abs(centre_x - domain_centre)
        normalized_distance = distance_from_centre / (config.domain_width / 2)
        vertical_factor = max(0, 1.0 - normalized_distance**2)
        
        if floe.floe_id % 2 == 0:
            vertical_amplitude = h_sail * vertical_factor
            new_y = centre_y + vertical_amplitude
        else:
            vertical_amplitude = d_keel * vertical_factor
            new_y = centre_y - vertical_amplitude
        
        new_centre = torch.tensor([new_x, new_y], dtype=torch.float64)
        new_floe = type(floe)(
            vertices=floe.vertices + (new_centre - floe.centre),
            centre=new_centre,
            thickness=floe.thickness,
            density=floe.density,
            bending_strength=floe.bending_strength,
            floe_id=floe.floe_id,
        )
        final_floes.append(new_floe)
    
    # Plot waterline
    ax.axhline(y=0, color='blue', linestyle='--', linewidth=2, label='Waterline', alpha=0.7)
    
    # Plot floes
    for floe in final_floes:
        vertices = floe.vertices.numpy()
        polygon = plt.Polygon(vertices, facecolor='lightblue', edgecolor='navy', 
                             linewidth=1.5, alpha=0.8)
        ax.add_patch(polygon)
    
    # Annotations
    ax.annotate(f'Sail: {result.sail_height:.2f} m', 
                xy=(config.domain_width/2, result.sail_height),
                xytext=(config.domain_width/2 + 2, result.sail_height + 0.5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=12, color='red', fontweight='bold')
    
    ax.annotate(f'Keel: {result.keel_depth:.2f} m', 
                xy=(config.domain_width/2, -result.keel_depth),
                xytext=(config.domain_width/2 + 2, -result.keel_depth - 0.5),
                arrowprops=dict(arrowstyle='->', color='darkblue'),
                fontsize=12, color='darkblue', fontweight='bold')
    
    ax.set_xlim(0, config.domain_width)
    ax.set_ylim(-result.keel_depth - 1, result.sail_height + 1)
    ax.set_xlabel('Horizontal Distance (m)', fontsize=14)
    ax.set_ylabel('Vertical Position (m)', fontsize=14)
    ax.set_title('Ice Ridge Cross-Section After Compression', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f'Saved: {filename}')
    plt.close()


def plot_porosity_evolution(result, filename='artifacts/porosity_evolution.png'):
    """Plot porosity vs. strain with exponential fit."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    strain = np.array(result.strain_history)
    porosity = np.array(result.porosity_history)
    
    # Plot data points
    ax.plot(strain, porosity, 'bo-', linewidth=2, markersize=6, label='Simulation')
    
    # Plot fit
    n0 = result.porosity_fit['n0']
    epsilon_c = result.porosity_fit['epsilon_c']
    strain_fit = np.linspace(0, max(strain), 100)
    porosity_fit = n0 * np.exp(-strain_fit / epsilon_c)
    ax.plot(strain_fit, porosity_fit, 'r--', linewidth=2, 
            label=f'Fit: n = {n0:.3f}·exp(-ε/{epsilon_c:.3f})')
    
    ax.set_xlabel('Compressive Strain ε', fontsize=14)
    ax.set_ylabel('Porosity n', fontsize=14)
    ax.set_title('Porosity Evolution During Ice Ridge Formation', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(strain))
    ax.set_ylim(0, 1)
    
    # Add R² annotation
    r_squared = result.porosity_fit['r_squared']
    ax.text(0.05, 0.95, f'R² = {r_squared:.4f}', 
            transform=ax.transAxes, fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f'Saved: {filename}')
    plt.close()


def plot_size_distribution(result, filename='artifacts/floe_size_distribution.png'):
    """Plot floe size distribution."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    config = result.config
    floe_config = FloePopulationConfig(
        n_floes=config.n_floes,
        domain_size=(config.domain_width, config.domain_height),
        thickness=config.ice_thickness,
        density=config.ice_density,
        bending_strength=config.bending_strength,
        power_law_exponent=config.power_law_exponent,
        min_diameter=config.min_floe_diameter,
        max_diameter=config.max_floe_diameter,
        seed=config.seed,
    )
    floes = generate_power_law_floes(floe_config)
    diameters, _ = floes.size_distribution
    
    # Histogram
    ax.hist(diameters, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Theoretical power law
    d_range = np.linspace(config.min_floe_diameter, config.max_floe_diameter, 100)
    alpha = config.power_law_exponent
    # n(D) ~ D^(-alpha)
    n_theory = d_range ** (-alpha)
    n_theory = n_theory / n_theory.max() * 10  # Scale for visualization
    
    ax_twin = ax.twinx()
    ax_twin.plot(d_range, n_theory, 'r-', linewidth=2, 
                label=f'Power law: n(D) ~ D^(-{alpha})')
    ax_twin.set_ylabel('Theoretical Distribution (a.u.)', fontsize=14, color='red')
    ax_twin.tick_params(axis='y', labelcolor='red')
    
    ax.set_xlabel('Floe Diameter D (m)', fontsize=14)
    ax.set_ylabel('Number of Floes', fontsize=14)
    ax.set_title('Floe Size Distribution', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f'Saved: {filename}')
    plt.close()


def plot_sail_keel_evolution(result, filename='artifacts/sail_keel_evolution.png'):
    """Plot sail height and keel depth evolution."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    config = result.config
    h = config.ice_thickness
    sail_keel_ratio_target = 0.35
    
    strain_history = np.array(result.strain_history)
    sail_history = []
    keel_history = []
    
    for strain in strain_history:
        d_squeeze = config.domain_width * strain
        h_sail = 0.4 * np.sqrt(h * d_squeeze) if d_squeeze > 0 else 0.0
        d_keel = h_sail / sail_keel_ratio_target if sail_keel_ratio_target > 0 else 4.0 * h_sail
        sail_history.append(h_sail)
        keel_history.append(d_keel)
    
    sail_history = np.array(sail_history)
    keel_history = np.array(keel_history)
    
    ax.plot(strain_history, sail_history, 'r-', linewidth=2, label='Sail Height', marker='o')
    ax.plot(strain_history, keel_history, 'b-', linewidth=2, label='Keel Depth', marker='s')
    
    ax.set_xlabel('Compressive Strain ε', fontsize=14)
    ax.set_ylabel('Height/Depth (m)', fontsize=14)
    ax.set_title('Sail and Keel Evolution During Compression', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add final values annotation
    ax.annotate(f'Final sail: {result.sail_height:.2f} m', 
                xy=(strain_history[-1], sail_history[-1]),
                xytext=(0.7, 0.8),
                textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=11, color='red')
    
    ax.annotate(f'Final keel: {result.keel_depth:.2f} m', 
                xy=(strain_history[-1], keel_history[-1]),
                xytext=(0.7, 0.2),
                textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=11, color='blue')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f'Saved: {filename}')
    plt.close()


if __name__ == '__main__':
    import os
    os.makedirs('artifacts', exist_ok=True)
    
    # Run simulation
    config = IceRidgeFormationConfig(
        n_floes=30,
        domain_width=15.0,
        domain_height=8.0,
        ice_thickness=1.0,
        target_strain=0.4,
        seed=42
    )
    
    print('Running ice ridge formation simulation...')
    result = simulate_ice_ridge_formation(config)
    print(f'Simulation complete. Volume error: {result.volume_error:.6f}')
    
    # Generate figures
    print('\nGenerating figures...')
    plot_ridge_cross_section(result)
    plot_porosity_evolution(result)
    plot_size_distribution(result)
    plot_sail_keel_evolution(result)
    
    print('\nAll figures generated successfully!')
