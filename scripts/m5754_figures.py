#!/usr/bin/env python3
"""Generate verification figures for m5754 ice-structure contact mechanics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensordem.ice_structure_contact import (
    polyhedron_hertz_degeneracy,
    rolling_bias_quantification,
    monotonic_scan,
    identifiability_analysis,
    compute_contact_pressure_field,
    spatial_correlation,
    _generate_contact_network,
)

OUT = Path("docs/m5754_figures")
OUT.mkdir(parents=True, exist_ok=True)


def fig1_hertz_degeneracy():
    """Criterion (a): Polyhedron → Hertz degeneracy."""
    result = polyhedron_hertz_degeneracy(radius=0.025, e_star=5e9, n_facets=60)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    overlaps = np.array(result["overlaps"])
    F_poly = np.array(result["F_poly"])
    F_hertz = np.array(result["F_hertz"])

    ax1.loglog(overlaps, F_hertz, 'b-', linewidth=2, label="Hertz analytic")
    ax1.loglog(overlaps, F_poly, 'ro', markersize=4, label="Polyhedron kernel")
    ax1.set_xlabel("Overlap δ [m]")
    ax1.set_ylabel("Normal force F_n [N]")
    ax1.set_title("(a) Hertz Degeneracy: Force vs Overlap")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    rel_err = np.array(result["rel_error"])
    ax2.semilogx(overlaps, rel_err, 'k-', linewidth=1.5)
    ax2.axhline(1e-12, color='r', linestyle='--', label="Threshold 1e-12")
    ax2.set_xlabel("Overlap δ [m]")
    ax2.set_ylabel("Relative error |F_poly - F_Hertz| / F_Hertz")
    ax2.set_title(f"Relative Error (max = {result['max_rel_error']:.2e})")
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT / "fig1_hertz_degeneracy.png", dpi=72, bbox_inches="tight")
    plt.close(fig)
    print("  fig1_hertz_degeneracy.png")


def fig2_rolling_bias():
    """Criterion (b,c): Rolling friction bias quantification."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, psi, title in zip(axes, [0.0, 0.2, 0.4],
                               ["ψ=0 (sphere)", "ψ=0.2 (mild)", "ψ=0.4 (held-out)"]):
        result = rolling_bias_quantification(
            angularity=psi,
            mu_r_scan=np.linspace(0, 0.5, 51),
        )
        ax.plot(result.mu_r_values, result.bias * 100, 'b-', linewidth=2)
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax.axhline(15, color='r', linestyle='--', alpha=0.5, label="15% threshold")
        ax.axhline(-15, color='r', linestyle='--', alpha=0.5)
        ax.axvline(result.optimal_mu_r, color='g', linestyle=':', alpha=0.7,
                   label=f"optimal μ_r={result.optimal_mu_r:.3f}")
        ax.set_xlabel("Rolling friction coefficient μ_r")
        ax.set_ylabel("Bias δ(μ_r) [%]")
        ax.set_title(f"{title}\nmin|δ| = {result.min_bias*100:.1f}%")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("(b,c) Rolling Friction Proxy Bias: Sphere+μ_r vs Polyhedron", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT / "fig2_rolling_bias.png", dpi=72, bbox_inches="tight")
    plt.close(fig)
    print("  fig2_rolling_bias.png")


def fig3_monotonic_scan():
    """Criterion (f): Negative control - monotonic scan."""
    result = monotonic_scan(mu_roll_values=np.linspace(0, 0.5, 51))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.mu_roll_values, result.R_total, 'b-o', markersize=3, linewidth=2)
    ax.set_xlabel("Rolling friction coefficient μ_roll")
    ax.set_ylabel("Total resistance R_total [N]")
    ax.set_title("(f) Negative Control: R_total vs μ_roll\n(Must be monotonically non-decreasing)")
    ax.grid(True, alpha=0.3)
    # Annotate
    ax.annotate(f"min ΔR = {result.min_diff:.2f} N",
                xy=(0.5, result.R_total[-1]),
                xytext=(0.3, result.R_total[-1] * 0.95),
                arrowprops=dict(arrowstyle="->", color='green'),
                fontsize=10, color='green')
    plt.tight_layout()
    fig.savefig(OUT / "fig3_monotonic_scan.png", dpi=72, bbox_inches="tight")
    plt.close(fig)
    print("  fig3_monotonic_scan.png")


def fig4_contact_pressure():
    """Contact pressure spatial statistics."""
    forces, normals, positions = _generate_contact_network(500, 0.3, 1000.0, 42)
    areas = np.pi * (0.005 ** 2) * np.ones(len(forces))
    cpf = compute_contact_pressure_field(
        positions[:, :2], forces, areas,
        grid_bounds=(-0.1, 0.1, -0.1, 0.1),
        resolution=(40, 40),
    )
    sc = spatial_correlation(cpf, max_lag=10)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Pressure field
    im = axes[0].pcolormesh(cpf.x_edges, cpf.y_edges, cpf.pressure.T,
                            cmap='hot', shading='auto')
    plt.colorbar(im, ax=axes[0], label='Pressure [Pa]')
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("Contact Pressure Field p(x,y)")

    # Spatial correlation
    im2 = axes[1].pcolormesh(sc.lag_x * 1000, sc.lag_y * 1000, sc.correlation.T,
                             cmap='RdBu_r', vmin=-0.5, vmax=1.0, shading='auto')
    plt.colorbar(im2, ax=axes[1], label='R(Δx,Δy)')
    axes[1].set_xlabel("Δx [mm]")
    axes[1].set_ylabel("Δy [mm]")
    axes[1].set_title("Spatial Correlation Function")

    # Correlation cross-section
    center = sc.correlation.shape[0] // 2
    axes[2].plot(sc.lag_x * 1000, sc.correlation[:, center], 'b-', linewidth=2, label="R(Δx, 0)")
    axes[2].plot(sc.lag_y * 1000, sc.correlation[center, :], 'r--', linewidth=2, label="R(0, Δy)")
    axes[2].axhline(1/np.e, color='k', linestyle=':', alpha=0.5, label="1/e")
    axes[2].set_xlabel("Lag [mm]")
    axes[2].set_ylabel("Correlation")
    axes[2].set_title(f"Peak factor γ = {sc.peak_factor:.2f}")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Contact Pressure Spatial Statistics", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT / "fig4_contact_pressure.png", dpi=72, bbox_inches="tight")
    plt.close(fig)
    print("  fig4_contact_pressure.png")


def fig5_identifiability():
    """Criterion (d): Parameter identifiability."""
    result = identifiability_analysis()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # R_total surface
    mu_grid, psi_grid = np.meshgrid(result.mu_roll_values, result.psi_values, indexing="ij")
    im = axes[0].pcolormesh(psi_grid, mu_grid, result.R_total_grid,
                            cmap='viridis', shading='auto')
    plt.colorbar(im, ax=axes[0], label='R_total [N]')
    axes[0].set_xlabel("Angularity ψ")
    axes[0].set_ylabel("Rolling friction μ_roll")
    axes[0].set_title("R_total(μ_roll, ψ) Sensitivity Surface")

    # Sensitivity contours
    axes[1].contourf(result.psi_values, result.mu_roll_values, result.R_total_grid,
                     levels=15, cmap='viridis', alpha=0.7)
    cs = axes[1].contour(result.psi_values, result.mu_roll_values, result.R_total_grid,
                         levels=15, colors='k', linewidths=0.5, alpha=0.5)
    axes[1].set_xlabel("Angularity ψ")
    axes[1].set_ylabel("Rolling friction μ_roll")
    axes[1].set_title(f"Condition number = {result.condition_number:.1f}\n"
                      f"|corr(μ_roll,ψ)| = {result.correlation:.3f}")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("(d) Parameter Identifiability Analysis", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT / "fig5_identifiability.png", dpi=72, bbox_inches="tight")
    plt.close(fig)
    print("  fig5_identifiability.png")


if __name__ == "__main__":
    print("Generating verification figures...")
    fig1_hertz_degeneracy()
    fig2_rolling_bias()
    fig3_monotonic_scan()
    fig4_contact_pressure()
    fig5_identifiability()
    print("Done!")
