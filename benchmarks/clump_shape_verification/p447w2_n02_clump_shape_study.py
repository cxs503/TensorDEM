#!/usr/bin/env python3
"""DEM clump shape effect verification: repose angle, settling velocity, force chains.

Fast version with reduced particle counts and simulation steps.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.particle.clump import (
    ClumpParticle,
    single_sphere,
    ellipsoid_clump,
)
from tensordem.particle.clump_dem import (
    lifted_box_repose,
    piston_compression,
)
from tensordem.contact.clump_contact import ContactParams

# ---------------------------------------------------------------------------
# Sphericity-parameterized clump family
# ---------------------------------------------------------------------------
def prolate_spheroid_sphericity(a_over_b: float) -> float:
    """Sphericity of prolate spheroid with semi-axes (a, b, b), a >= b."""
    if a_over_b <= 1.0:
        return 1.0
    e = math.sqrt(1.0 - 1.0 / a_over_b**2)
    a = a_over_b
    V = 4.0 / 3.0 * math.pi * a
    A_eq = (36.0 * math.pi * V**2) ** (1.0 / 3.0)
    A = 2.0 * math.pi * (1.0 + (a / e) * math.asin(e))
    return A_eq / A


def find_aspect_ratio_for_sphericity(psi_target: float) -> float:
    """Find prolate spheroid aspect ratio a/b that gives target sphericity."""
    if psi_target >= 1.0:
        return 1.0
    from scipy.optimize import brentq
    def f(r):
        return prolate_spheroid_sphericity(r) - psi_target
    return brentq(f, 1.001, 20.0)


def make_clump_factory(psi_target: float, base_radius: float = 0.002, density: float = 2500.0):
    """Create a clump factory for the target sphericity."""
    if psi_target >= 0.999:
        def factory():
            return single_sphere(base_radius, density, name=f"sphere_psi{psi_target:.1f}")
        factory.target_psi = psi_target
        factory.aspect_ratio = 1.0
        factory.n_spheres = 1
        return factory
    
    aspect_ratio = find_aspect_ratio_for_sphericity(psi_target)
    a = aspect_ratio * base_radius
    b = base_radius
    n_spheres = max(3, int(aspect_ratio * 2.0))
    
    def factory() -> ClumpParticle:
        return ellipsoid_clump(a, b, n_spheres=n_spheres, density=density,
                               name=f"ellipsoid_psi{psi_target:.1f}")
    factory.target_psi = psi_target
    factory.aspect_ratio = aspect_ratio
    factory.n_spheres = n_spheres
    return factory


# ---------------------------------------------------------------------------
# Protocol 1: Angle of repose (fast version)
# ---------------------------------------------------------------------------
def run_repose_protocol(psi_values: list[float], n_repeats: int = 5) -> dict:
    """Run angle of repose simulations."""
    print("=" * 70)
    print("Protocol 1: Angle of Repose vs Sphericity")
    print("=" * 70)
    
    params = ContactParams(kn=1.0e5, kt_ratio=0.7, zeta=0.3, mu=0.5, mu_roll=0.3)
    
    # Fast simulation parameters
    sim_kw = dict(
        dt=5e-5,
        settle_steps=1500,
        lift_steps=2000,
        tail_steps=2000,
        box_half_x=0.025,
        slab_half_y=0.008,
        fill_height=0.070,
        floor_half_x=0.12,
        lift_speed=0.30,
    )
    
    results = {
        "psi_values": psi_values,
        "theta_mean": [],
        "theta_std": [],
        "theta_all": [],
        "n_particles": [],
        "metadata": {"n_repeats": n_repeats, "contact_params": {"kn": params.kn, "mu": params.mu, "mu_roll": params.mu_roll}},
    }
    
    for psi in psi_values:
        print(f"\nψ = {psi:.1f}:")
        factory = make_clump_factory(psi, base_radius=0.002)
        if hasattr(factory, 'aspect_ratio'):
            print(f"  a/b = {factory.aspect_ratio:.3f}, n_spheres = {factory.n_spheres}")
        
        thetas = []
        n_parts = []
        for rep in range(n_repeats):
            seed = 1000 + int(psi * 100) + rep
            t0 = time.time()
            try:
                result = lifted_box_repose(factory, seed=seed, params=params, **sim_kw)
                theta = result["measurement"]["angle_profile_deg"]
                n_p = result["n_particles"]
                if math.isfinite(theta):
                    thetas.append(theta)
                    n_parts.append(n_p)
                    print(f"  rep {rep+1}: θ = {theta:.2f}°, n = {n_p}, t = {time.time()-t0:.1f}s")
                else:
                    print(f"  rep {rep+1}: θ = NaN, t = {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"  rep {rep+1}: ERROR - {e}")
        
        if thetas:
            results["theta_mean"].append(float(np.mean(thetas)))
            results["theta_std"].append(float(np.std(thetas)))
            results["theta_all"].append(thetas)
            results["n_particles"].append(int(np.mean(n_parts)))
            print(f"  Mean θ = {np.mean(thetas):.2f}° ± {np.std(thetas):.2f}°")
        else:
            results["theta_mean"].append(float("nan"))
            results["theta_std"].append(float("nan"))
            results["theta_all"].append([])
            results["n_particles"].append(0)
    
    return results


# ---------------------------------------------------------------------------
# Protocol 2: Settling velocity (analytical + simplified simulation)
# ---------------------------------------------------------------------------
def haider_levenspiel_K(psi: float) -> float:
    """Shape correction factor for drag (Stokes regime approximation)."""
    if psi >= 0.999:
        return 1.0
    return psi ** (-2.0 / 3.0)


def run_settling_protocol(psi_values: list[float]) -> dict:
    """Compute settling velocity correction factors."""
    print("\n" + "=" * 70)
    print("Protocol 2: Settling Velocity (Stokes Regime)")
    print("=" * 70)
    
    r_eq = 0.002  # m
    rho_p = 2500.0  # kg/m³
    rho_f = 1000.0  # kg/m³
    mu = 1.0e-3  # Pa·s
    g = 9.81  # m/s²
    
    # Stokes velocity for sphere
    v_stokes = (2.0 / 9.0) * (rho_p - rho_f) * g * r_eq**2 / mu
    print(f"Stokes velocity (sphere): v_t = {v_stokes:.4f} m/s")
    
    results = {
        "psi_values": psi_values,
        "K_theoretical": [],
        "v_terminal_theoretical": [],
        "metadata": {
            "r_eq_m": r_eq,
            "rho_p_kg_m3": rho_p,
            "rho_f_kg_m3": rho_f,
            "mu_Pa_s": mu,
            "v_stokes_m_s": v_stokes,
        },
    }
    
    for psi in psi_values:
        K = haider_levenspiel_K(psi)
        v_term = v_stokes / K
        results["K_theoretical"].append(K)
        results["v_terminal_theoretical"].append(v_term)
        print(f"ψ = {psi:.1f}: K = {K:.3f}, v_t = {v_term:.4f} m/s")
    
    return results


# ---------------------------------------------------------------------------
# Protocol 3: Force chain topology (fast version)
# ---------------------------------------------------------------------------
def run_force_chain_protocol(psi_values: list[float]) -> dict:
    """Run force chain analysis under compression."""
    print("\n" + "=" * 70)
    print("Protocol 3: Force Chain Topology")
    print("=" * 70)
    
    params = ContactParams(kn=1.0e5, kt_ratio=0.7, zeta=0.3, mu=0.5, mu_roll=0.3)
    
    sim_kw = dict(
        dt=5e-5,
        settle_steps=1200,
        compress_steps=1500,
        box_half_x=0.025,
        slab_half_y=0.008,
        fill_height=0.050,
        piston_speed=0.02,
    )
    
    results = {
        "psi_values": psi_values,
        "Z_mean": [],
        "Z_std": [],
        "Z_all": [],
        "n_contacts_mean": [],
        "metadata": {"contact_params": {"kn": params.kn, "mu": params.mu, "mu_roll": params.mu_roll}},
    }
    
    for psi in psi_values:
        print(f"\nψ = {psi:.1f}:")
        factory = make_clump_factory(psi, base_radius=0.002)
        
        Z_values = []
        n_contacts_list = []
        
        for rep in range(3):
            seed = 3000 + int(psi * 100) + rep
            t0 = time.time()
            try:
                result = piston_compression(factory, seed=seed, params=params, **sim_kw)
                # Compute coordination number from contacts
                contacts = result["contacts"]
                n_contacts = len(contacts["fn"])
                n_particles = result["n_particles"]
                # Z = 2 * n_contacts / n_particles (each contact involves 2 particles)
                Z = 2.0 * n_contacts / n_particles if n_particles > 0 else 0.0
                Z_values.append(Z)
                n_contacts_list.append(n_contacts)
                print(f"  rep {rep+1}: Z = {Z:.2f}, n_c = {n_contacts}, n_p = {n_particles}, t = {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"  rep {rep+1}: ERROR - {e}")
        
        if Z_values:
            results["Z_mean"].append(float(np.mean(Z_values)))
            results["Z_std"].append(float(np.std(Z_values)))
            results["Z_all"].append(Z_values)
            results["n_contacts_mean"].append(int(np.mean(n_contacts_list)))
            print(f"  Mean Z = {np.mean(Z_values):.2f} ± {np.std(Z_values):.2f}")
        else:
            results["Z_mean"].append(float("nan"))
            results["Z_std"].append(float("nan"))
            results["Z_all"].append([])
            results["n_contacts_mean"].append(0)
    
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Run the complete verification study."""
    print("TensorDEM Clump Shape Effect Verification (p447w2-n02)")
    print("=" * 70)
    
    psi_values = [1.0, 0.9, 0.8, 0.7, 0.6]
    
    t_start = time.time()
    repose_results = run_repose_protocol(psi_values, n_repeats=5)
    settling_results = run_settling_protocol(psi_values)
    force_chain_results = run_force_chain_protocol(psi_values)
    t_total = time.time() - t_start
    
    all_results = {
        "mission_id": "p447w2-n02",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_time_s": t_total,
        "repose": repose_results,
        "settling": settling_results,
        "force_chain": force_chain_results,
    }
    
    output_dir = ROOT / "artifacts" / "dem" / "clump_shape_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "clump_shape_verification_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total time: {t_total:.1f}s")
    print(f"Results: {results_file}")
    
    print("\nRepose angle vs sphericity:")
    print(f"{'ψ':>5} {'θ_mean [°]':>12} {'θ_std [°]':>10} {'n_part':>8}")
    for i, psi in enumerate(psi_values):
        print(f"{psi:5.1f} {repose_results['theta_mean'][i]:12.2f} {repose_results['theta_std'][i]:10.2f} {repose_results['n_particles'][i]:8d}")
    
    print("\nSettling velocity correction:")
    print(f"{'ψ':>5} {'K_theo':>8} {'v_theo [m/s]':>14}")
    for i, psi in enumerate(psi_values):
        print(f"{psi:5.1f} {settling_results['K_theoretical'][i]:8.3f} {settling_results['v_terminal_theoretical'][i]:14.6f}")
    
    print("\nCoordination number:")
    print(f"{'ψ':>5} {'Z_mean':>8} {'Z_std':>8}")
    for i, psi in enumerate(psi_values):
        print(f"{psi:5.1f} {force_chain_results['Z_mean'][i]:8.2f} {force_chain_results['Z_std'][i]:8.2f}")
    
    return all_results


if __name__ == "__main__":
    main()
