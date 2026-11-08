"""Generate figures for the DEM contact mechanics verification report."""
import math
import pathlib
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensordem.dem_contact_verification import (
    cundall_strack_loading_unloading,
    rolling_resistance_convergence,
    rotational_damping_decay,
    superquadric_plane_contact_verification,
    clump_surface_error_vs_N,
)

FIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "dem" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def fig_c1_tangential_spring():
    """C1: Cundall-Strack tangential spring loading-unloading hysteresis."""
    k_t = 1000.0
    mu_s = 0.3
    F_n = 100.0
    result = cundall_strack_loading_unloading(k_t, mu_s, F_n, delta_max=0.08, n_steps=500)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result["deltas"], result["forces"], 'b-', linewidth=2, label='Cundall-Strack')
    ax.axhline(y=mu_s * F_n, color='r', linestyle='--', alpha=0.5, label=f'Coulomb limit = {mu_s*F_n:.0f} N')
    ax.axhline(y=-mu_s * F_n, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Tangential displacement δ_t (m)', fontsize=12)
    ax.set_ylabel('Tangential force F_t (N)', fontsize=12)
    ax.set_title('C1: Cundall-Strack Tangential Spring — Loading/Unloading Hysteresis', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.09, 0.09)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "c1_tangential_spring.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'c1_tangential_spring.png'}")


def fig_c2_rolling_resistance():
    """C2: Rolling resistance moment convergence to analytic limit."""
    mu_r = 0.1
    F_n = 50.0
    R_eff = 0.005
    k_r = 1.0
    omega = 10.0
    dt = 1e-4
    result = rolling_resistance_convergence(mu_r, F_n, R_eff, k_r, omega, dt, n_steps=5000)

    fig, ax = plt.subplots(figsize=(8, 5))
    steps = np.arange(len(result["torque_history"]))
    ax.plot(steps * dt * 1000, result["torque_history"], 'b-', linewidth=1.5, label='EPSD torque')
    ax.axhline(y=result["analytic_limit"], color='r', linestyle='--', linewidth=2,
               label=f'Analytic limit = {result["analytic_limit"]:.4f} N·m')
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Rolling resistance moment |M_r| (N·m)', fontsize=12)
    ax.set_title('C2: EPSD Rolling Spring — Convergence to Coulomb Rolling Limit', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "c2_rolling_resistance.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'c2_rolling_resistance.png'}")


def fig_c3_rotational_damping():
    """C3: Rotational damping exponential decay."""
    gamma_r = 0.01
    I = 1e-5
    omega0 = 100.0
    dt = 1e-4
    n_steps = 5000
    result = rotational_damping_decay(gamma_r, I, omega0, dt, n_steps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Linear scale
    ax1.plot(result["times"] * 1000, result["omegas"], 'b-', linewidth=2, label='Simulation (Euler)')
    ax1.plot(result["times"] * 1000, result["analytic"], 'r--', linewidth=2, label='Analytic exp(-t/τ)')
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Angular velocity ω (rad/s)', fontsize=12)
    ax1.set_title(f'Rotational Damping Decay (R² = {result["R_squared"]:.4f})', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Log scale
    valid = result["omegas"] > 1e-10
    ax2.semilogy(result["times"][valid] * 1000, result["omegas"][valid], 'b-', linewidth=2, label='Simulation')
    ax2.semilogy(result["times"][valid] * 1000, result["analytic"][valid], 'r--', linewidth=2, label='Analytic')
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Angular velocity ω (rad/s)', fontsize=12)
    ax2.set_title('Log-scale: Exponential Decay Verification', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "c3_rotational_damping.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'c3_rotational_damping.png'}")


def fig_c4_hertz_contact():
    """C4: Superquadric-plane Hertz contact stiffness."""
    R = 0.01
    E_star = 1e7
    deltas = np.logspace(-6, -3, 50)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Sphere limit
    result_sphere = superquadric_plane_contact_verification(R, R, R, 2.0, 2.0, E_star)
    F_sphere = np.array([result_sphere["k_n"] * 0.5 * d**0.5 * d for d in deltas])  # Approximate
    from tensordem.dem_contact_verification import hertz_contact_stiffness
    F_hertz = np.array([hertz_contact_stiffness(E_star, R, d) for d in deltas])

    ax1.loglog(deltas, F_hertz, 'b-', linewidth=2, label=f'Hertz (R_eff={R:.4f} m)')
    ax1.set_xlabel('Overlap δ (m)', fontsize=12)
    ax1.set_ylabel('Normal force F_n (N)', fontsize=12)
    ax1.set_title('C4: Hertz Contact Law — Sphere vs Plane', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')

    # Curvature comparison for different shapes
    shapes = [(R, R, R, 2.0, 2.0, 'Sphere'),
              (0.015, 0.01, 0.012, 2.0, 2.0, 'Ellipsoid'),
              (R, R, R, 3.0, 3.0, 'SQ n=3'),
              (R, R, R, 5.0, 5.0, 'SQ n=5')]
    colors = ['blue', 'green', 'orange', 'red']
    for (a, b, c, n1, n2, label), color in zip(shapes, colors):
        res = superquadric_plane_contact_verification(a, b, c, n1, n2, E_star)
        ax2.bar(label, res["R_eff"] * 1000, color=color, alpha=0.7)
    ax2.axhline(y=R * 1000, color='blue', linestyle='--', alpha=0.5, label=f'Sphere R={R*1000:.0f} mm')
    ax2.set_ylabel('Effective contact radius R_eff (mm)', fontsize=12)
    ax2.set_title('C4: Effective Contact Radius by Shape', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(FIG_DIR / "c4_hertz_contact.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'c4_hertz_contact.png'}")


def fig_c5_clump_convergence():
    """C5: Clump surface error vs number of spheres."""
    result = clump_surface_error_vs_N(
        a=1.0, b=1.0, c=1.0, n1=3.0, n2=3.0,
        N_values=[2, 4, 8, 16, 32])

    N_vals = [r["N"] for r in result["results"]]
    max_errs = [r["max_surface_error"] for r in result["results"]]
    mean_errs = [r["mean_surface_error"] for r in result["results"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(N_vals, max_errs, 'bo-', linewidth=2, markersize=8, label='Max surface error')
    ax.semilogy(N_vals, mean_errs, 'rs--', linewidth=2, markersize=8, label='Mean surface error')
    ax.set_xlabel('Number of clump spheres N', fontsize=12)
    ax.set_ylabel('Surface deviation (m)', fontsize=12)
    ax.set_title('C5: Clump Geometric Convergence — Superquadric (n₁=n₂=3)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(FIG_DIR / "c5_clump_convergence.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'c5_clump_convergence.png'}")


if __name__ == "__main__":
    fig_c1_tangential_spring()
    fig_c2_rolling_resistance()
    fig_c3_rotational_damping()
    fig_c4_hertz_contact()
    fig_c5_clump_convergence()
    print("\nAll figures generated successfully.")
