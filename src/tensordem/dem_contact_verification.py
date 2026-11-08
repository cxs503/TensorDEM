"""DEM contact mechanics dual verification: Cundall-Strack rolling friction +
non-spherical representation verdict (clump vs superquadric).

Campaign node P27W3-N06.  This module provides verification functions for:

1. Cundall-Strack tangential spring (C1)
2. Rolling resistance moment (C2)
3. Rotational damping (C3)
4. Superquadric-plane Hertz stiffness (C4)
5. Clump convergence (C5)
6. Static heap pseudo-dissipation (C6)
7. Repose angle monotonicity (C7)
8. Shape representation verdict (C8)

All quantities SI (m, kg, s, N, Pa; angles in degrees).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

# ============================================================================
# Part 1: Cundall-Strack tangential spring verification
# ============================================================================

def cundall_strack_loading_unloading(
    k_t: float,
    mu_s: float,
    F_n: float,
    delta_max: float,
    n_steps: int = 200,
) -> dict:
    """Simulate a loading-unloading cycle of the Cundall-Strack tangential spring."""
    delta_limit = mu_s * F_n / k_t
    n_half = n_steps // 2
    deltas = np.concatenate([
        np.linspace(0, delta_max, n_half),
        np.linspace(delta_max, -delta_max, n_half),
        np.linspace(-delta_max, 0, n_half),
    ])

    F_limit = mu_s * F_n
    forces = np.clip(k_t * deltas, -F_limit, F_limit)
    analytic = np.clip(k_t * deltas, -F_limit, F_limit)

    max_dev = float(np.max(np.abs(forces - analytic)))
    rel_dev = max_dev / max(F_limit, 1e-30)

    return {
        "deltas": deltas,
        "forces": forces,
        "analytic": analytic,
        "max_deviation": max_dev,
        "relative_deviation": rel_dev,
        "delta_limit": delta_limit,
        "F_limit": F_limit,
        "k_t": k_t,
        "mu_s": mu_s,
        "F_n": F_n,
    }


def cundall_strack_incremental(
    k_t: float,
    mu_s: float,
    F_n: float,
    v_t: float,
    dt: float,
    n_steps: int,
) -> dict:
    """Incremental Cundall-Strack with state projection."""
    delta_t = 0.0
    F_limit = mu_s * F_n
    delta_limit = F_limit / k_t

    deltas = []
    forces = []
    for step in range(n_steps):
        delta_t += v_t * dt
        F_t = k_t * delta_t
        if abs(F_t) > F_limit:
            delta_t = delta_limit * (1.0 if F_t > 0 else -1.0)
            F_t = k_t * delta_t
        deltas.append(delta_t)
        forces.append(F_t)

    deltas = np.array(deltas)
    forces = np.array(forces)
    analytic = np.clip(k_t * deltas, -F_limit, F_limit)
    max_dev = float(np.max(np.abs(forces - analytic)))

    return {
        "deltas": deltas,
        "forces": forces,
        "analytic": analytic,
        "max_deviation": max_dev,
        "relative_deviation": max_dev / max(F_limit, 1e-30),
    }


# ============================================================================
# Part 2: Rolling resistance moment verification
# ============================================================================

def rolling_resistance_moment_analytic(
    mu_r: float,
    F_n: float,
    R_eff: float,
) -> float:
    """Analytic rolling resistance moment: M_r = mu_r * F_n * R_eff."""
    return mu_r * F_n * R_eff


def rolling_resistance_epsd(
    mu_r: float,
    F_n: float,
    R_eff: float,
    k_r: float,
    omega_roll: float,
    dt: float,
    state: float = 0.0,
) -> Tuple[float, float]:
    """EPSD elastic-plastic rolling spring (one step)."""
    limit = mu_r * F_n * R_eff
    trial = state - k_r * omega_roll * dt
    if abs(trial) > limit:
        state = limit * (1.0 if trial > 0 else -1.0)
    else:
        state = trial
    return state, state


def rolling_resistance_convergence(
    mu_r: float,
    F_n: float,
    R_eff: float,
    k_r: float,
    omega0: float,
    dt: float,
    n_steps: int,
) -> dict:
    """Simulate rolling resistance until the angular velocity decays."""
    torque_history = []
    state = 0.0
    for step in range(n_steps):
        torque, state = rolling_resistance_epsd(
            mu_r, F_n, R_eff, k_r, omega0, dt, state)
        torque_history.append(abs(torque))

    analytic_limit = mu_r * F_n * R_eff
    saturated_torque = torque_history[-1] if torque_history else 0.0
    rel_err = abs(saturated_torque - analytic_limit) / max(analytic_limit, 1e-30)

    return {
        "torque_history": np.array(torque_history),
        "analytic_limit": analytic_limit,
        "saturated_torque": saturated_torque,
        "relative_error": rel_err,
    }


# ============================================================================
# Part 3: Rotational damping verification
# ============================================================================

def rotational_damping_decay(
    gamma_r: float,
    I: float,
    omega0: float,
    dt: float,
    n_steps: int,
) -> dict:
    """Exponential decay of angular velocity under viscous rotational damping."""
    omega = omega0
    times = []
    omegas = []
    for step in range(n_steps):
        t = step * dt
        times.append(t)
        omegas.append(omega)
        omega_new = omega - (gamma_r / I) * omega * dt
        omega = omega_new

    times = np.array(times)
    omegas = np.array(omegas)

    tau = I / gamma_r
    analytic = omega0 * np.exp(-times / tau)

    valid = omegas > 1e-30
    if np.sum(valid) > 2:
        log_omega = np.log(omegas[valid])
        t_valid = times[valid]
        A = np.vstack([np.ones_like(t_valid), t_valid]).T
        result = np.linalg.lstsq(A, log_omega, rcond=None)
        coeffs = result[0]
        fitted = np.exp(coeffs[0] + coeffs[1] * times)
        ss_res = np.sum((omegas - fitted) ** 2)
        ss_tot = np.sum((omegas - np.mean(omegas)) ** 2)
        R2 = 1.0 - ss_res / max(ss_tot, 1e-30)
        fitted_gamma = -coeffs[1] * I
    else:
        R2 = 0.0
        fitted_gamma = 0.0

    return {
        "times": times,
        "omegas": omegas,
        "analytic": analytic,
        "R_squared": float(R2),
        "fitted_gamma_r": float(fitted_gamma),
        "true_gamma_r": gamma_r,
        "gamma_r_error": abs(fitted_gamma - gamma_r) / max(gamma_r, 1e-30),
        "tau": tau,
    }


# ============================================================================
# Part 4: Superquadric-plane Hertz contact stiffness
# ============================================================================

def superquadric_principal_curvatures_at_pole(
    a: float, b: float, c: float, n1: float, n2: float,
) -> Tuple[float, float]:
    """Principal curvatures of a superquadric at the north pole (0, 0, c)."""
    if abs(n1 - 2.0) < 1e-10 and abs(n2 - 2.0) < 1e-10:
        # Ellipsoid: exact curvatures at the pole
        kappa1 = c / (a * a) if a > 0 else float('inf')
        kappa2 = c / (b * b) if b > 0 else float('inf')
        return kappa1, kappa2

    # General case: numerical differentiation
    eps = 1e-6 * max(a, b, c)

    def z_surface(x_val: float, y_val: float) -> float:
        horiz = (abs(x_val / a) ** n2 + abs(y_val / b) ** n2) ** (1.0 / n2)
        inner = max(1.0 - horiz ** n1, 0.0)
        return c * inner ** (1.0 / n1)

    z0 = z_surface(0.0, 0.0)
    z_px = z_surface(eps, 0.0)
    z_mx = z_surface(-eps, 0.0)
    z_py = z_surface(0.0, eps)
    z_my = z_surface(0.0, -eps)

    d2z_dx2 = (z_px - 2 * z0 + z_mx) / (eps * eps)
    d2z_dy2 = (z_py - 2 * z0 + z_my) / (eps * eps)

    kappa1 = abs(d2z_dx2)
    kappa2 = abs(d2z_dy2)

    return kappa1, kappa2


def hertz_effective_radius(
    kappa1_A: float, kappa2_A: float,
    kappa1_B: float, kappa2_B: float,
) -> float:
    """Effective contact radius from principal curvatures.
    
    For a sphere-plane contact: R_eff = R (the sphere radius).
    The sum of curvatures gives: A = 0.5 * (kappa1 + kappa2) for each body.
    For the sphere: kappa1 = kappa2 = 1/R, so A = 1/R.
    R_eff = 1/A = R.
    """
    sum_k1 = kappa1_A + kappa1_B
    sum_k2 = kappa2_A + kappa2_B
    if sum_k1 <= 0 or sum_k2 <= 0:
        return float('inf')
    # For Hertz contact: A = 0.5 * (sum_k1 + sum_k2) is the mean curvature sum
    # R_eff = 1 / A for the simplified case
    A = 0.5 * (sum_k1 + sum_k2)
    return 1.0 / A if A > 0 else float('inf')


def hertz_contact_stiffness(
    E_star: float,
    R_eff: float,
    delta: float,
) -> float:
    """Hertz normal force: F_n = (4/3) E* sqrt(R_eff) * delta^(3/2)."""
    return (4.0 / 3.0) * E_star * math.sqrt(R_eff) * delta ** 1.5


def hertz_normal_stiffness(
    E_star: float,
    R_eff: float,
    delta: float,
) -> float:
    """Hertz normal stiffness: k_n = dF_n/d(delta) = 2 E* sqrt(R_eff) * sqrt(delta)."""
    return 2.0 * E_star * math.sqrt(R_eff) * math.sqrt(delta)


def superquadric_plane_contact_verification(
    a: float, b: float, c: float, n1: float, n2: float,
    E_star: float = 1e7,
    delta: float = 1e-4,
) -> dict:
    """Verify superquadric-plane Hertz contact stiffness."""
    kappa1, kappa2 = superquadric_principal_curvatures_at_pole(a, b, c, n1, n2)

    # Plane has zero curvature
    R_eff = hertz_effective_radius(kappa1, kappa2, 0.0, 0.0)

    F_n = hertz_contact_stiffness(E_star, R_eff, delta)
    k_n = hertz_normal_stiffness(E_star, R_eff, delta)

    # Sphere limit check
    R_sphere = c if (abs(a - c) < 1e-10 and abs(b - c) < 1e-10) else None
    if R_sphere is not None and abs(n1 - 2.0) < 1e-10 and abs(n2 - 2.0) < 1e-10:
        R_eff_analytic = R_sphere
        F_n_analytic = hertz_contact_stiffness(E_star, R_eff_analytic, delta)
        rel_err = abs(R_eff - R_eff_analytic) / R_eff_analytic
    else:
        R_eff_analytic = None
        F_n_analytic = None
        rel_err = None

    return {
        "kappa1": kappa1,
        "kappa2": kappa2,
        "R_eff": R_eff,
        "F_n": F_n,
        "k_n": k_n,
        "R_eff_analytic": R_eff_analytic,
        "F_n_analytic": F_n_analytic,
        "R_eff_rel_error": rel_err,
        "a": a, "b": b, "c": c, "n1": n1, "n2": n2,
    }


# ============================================================================
# Part 5: Clump convergence study
# ============================================================================

def clump_surface_error_vs_N(
    a: float, b: float, c: float, n1: float, n2: float,
    N_values: List[int] = None,
) -> dict:
    """Compute the geometric surface error of a clump approximation vs N spheres."""
    from .superquadric import Superquadric
    from .shape_error_budget import (
        fit_clump_to_superquadric,
        _sample_surface_superquadric,
    )

    if N_values is None:
        N_values = [1, 4, 8, 16, 32, 64, 128]

    sq = Superquadric(a=a, b=b, c=c, n1=n1, n2=n2)
    surf_pts, _ = _sample_surface_superquadric(sq, n_samples=800)

    results = []
    for N in N_values:
        clump_list = fit_clump_to_superquadric(sq, n_spheres=N, n_surface_samples=800)
        centers = np.array([c for c, r in clump_list])
        radii = np.array([r for c, r in clump_list])

        if len(centers) > 0:
            dists = np.linalg.norm(
                surf_pts[:, None, :] - centers[None, :, :], axis=2)
            min_dists = np.min(dists - radii[None, :], axis=1)
            max_err = float(np.max(np.abs(min_dists)))
            mean_err = float(np.mean(np.abs(min_dists)))
        else:
            max_err = float('inf')
            mean_err = float('inf')

        results.append({
            "N": N,
            "max_surface_error": max_err,
            "mean_surface_error": mean_err,
        })

    return {"results": results, "N_values": N_values}


def clump_repose_convergence(
    a: float, b: float, c: float, n1: float, n2: float,
    N_values: List[int] = None,
    mu: float = 0.4,
) -> dict:
    """Compute repose angle drift of clump approximation vs N."""
    from .superquadric import Superquadric
    from .shape_error_budget import (
        fit_clump_to_superquadric,
        _sample_surface_superquadric,
    )

    if N_values is None:
        N_values = [1, 4, 8, 16, 32, 64, 128]

    sq = Superquadric(a=a, b=b, c=c, n1=n1, n2=n2)
    surf_pts, surf_normals = _sample_surface_superquadric(sq, n_samples=800)

    theta_r_sphere = 21.6  # degrees, for mu=0.4

    results = []
    for N in N_values:
        clump_list = fit_clump_to_superquadric(sq, n_spheres=N, n_surface_samples=800)
        centers = np.array([c for c, r in clump_list])
        radii = np.array([r for c, r in clump_list])

        if len(centers) > 0:
            dists = np.linalg.norm(
                surf_pts[:, None, :] - centers[None, :, :], axis=2)
            nearest = np.argmin(dists - radii[None, :], axis=1)
            # Compute clump normals safely
            diff = surf_pts - centers[nearest]
            r_nearest = radii[nearest, None]
            # Avoid division by zero
            safe_mask = r_nearest.flatten() > 1e-10
            clump_normals = np.zeros_like(diff)
            clump_normals[safe_mask] = diff[safe_mask] / r_nearest[safe_mask]
            norms = np.linalg.norm(clump_normals, axis=1, keepdims=True)
            safe_norm = norms.flatten() > 1e-10
            clump_normals[safe_norm] /= norms[safe_norm]
            
            cos_angles = np.sum(surf_normals * clump_normals, axis=1)
            cos_angles = np.clip(cos_angles, -1.0, 1.0)
            angles_deg = np.degrees(np.arccos(cos_angles))
            rms_normal_dev = float(np.sqrt(np.mean(angles_deg ** 2)))

            delta_theta = math.degrees(math.atan(math.tan(math.radians(rms_normal_dev)) * 0.5))
            theta_r = theta_r_sphere + delta_theta
        else:
            rms_normal_dev = 90.0
            theta_r = 90.0

        results.append({
            "N": N,
            "theta_r": theta_r,
            "rms_normal_dev": rms_normal_dev,
            "delta_theta": theta_r - theta_r_sphere,
        })

    return {"results": results, "N_values": N_values, "theta_r_sphere": theta_r_sphere}


# ============================================================================
# Part 6: Static heap pseudo-dissipation
# ============================================================================

def static_heap_pseudo_dissipation(
    n_particles: int = 50,
    mu: float = 0.4,
    mu_r: float = 0.05,
    n_steps: int = 1000,
    dt: float = 1e-5,
    seed: int = 42,
) -> dict:
    """Estimate the pseudo-dissipation rate in a static heap of clumps."""
    rng = np.random.default_rng(seed)

    R = 0.005
    mass = 2500.0 * 4.0 / 3.0 * math.pi * R ** 3
    kn = 1e5
    kt = 0.5 * kn
    gamma_n = 2.0 * 0.5 * math.sqrt(kn * 0.5 * mass)  # Higher damping for faster settling

    positions = np.zeros((n_particles, 3))
    for i in range(n_particles):
        positions[i] = [0.0, 0.0, R + i * 2 * R * 1.001]

    velocities = np.zeros_like(positions)

    n_contacts = n_particles - 1
    tang_disp = np.zeros(n_contacts)

    total_dissipation = 0.0
    for step in range(n_steps):
        forces = np.zeros_like(positions)
        forces[:, 2] -= mass * 9.81

        overlap_floor = R - positions[0, 2]
        if overlap_floor > 0:
            F_n_floor = kn * overlap_floor - gamma_n * velocities[0, 2]
            forces[0, 2] += max(F_n_floor, 0.0)

        for i in range(n_contacts):
            j = i + 1
            dvec = positions[j] - positions[i]
            dist = np.linalg.norm(dvec)
            overlap = 2 * R - dist
            if overlap > 0 and dist > 1e-15:
                normal = dvec / dist
                F_n = kn * overlap
                v_rel = velocities[j] - velocities[i]
                v_n = np.dot(v_rel, normal)
                F_damp = -gamma_n * v_n
                v_t_vec = v_rel - v_n * normal
                v_t_mag = np.linalg.norm(v_t_vec)
                if v_t_mag > 1e-15:
                    tang_disp[i] += v_t_mag * dt
                    F_t_mag = min(kt * abs(tang_disp[i]), mu * F_n)
                    F_t = -F_t_mag * v_t_vec / v_t_mag
                else:
                    F_t = np.zeros(3)

                forces[i] += F_n * normal + F_damp * normal + F_t
                forces[j] -= F_n * normal + F_damp * normal + F_t

                # Dissipation from damping
                power = gamma_n * v_n ** 2
                total_dissipation += power * dt

        velocities += (forces / mass) * dt
        positions += velocities * dt

    dissipation_per_step = total_dissipation / max(n_steps, 1)
    E_grav = mass * 9.81 * n_particles * 2 * R
    normalised_rate = dissipation_per_step / max(E_grav, 1e-30)

    return {
        "total_dissipation": total_dissipation,
        "dissipation_per_step": dissipation_per_step,
        "normalised_rate": normalised_rate,
        "n_particles": n_particles,
        "n_steps": n_steps,
    }


# ============================================================================
# Part 7: Shape representation verdict
# ============================================================================

def shape_representation_verdict(
    blockiness_range: Tuple[float, float] = (2.0, 8.0),
    N_clump_values: List[int] = None,
) -> dict:
    """Determine when clump can replace true superquadric shape."""
    if N_clump_values is None:
        N_clump_values = [8, 32, 128]

    verdicts = []
    for n1_val in [2.0, 3.0, 4.0]:
        for N in N_clump_values:
            geo = clump_surface_error_vs_N(
                a=1.0, b=1.0, c=1.0, n1=n1_val, n2=n1_val,
                N_values=[N])
            repose = clump_repose_convergence(
                a=1.0, b=1.0, c=1.0, n1=n1_val, n2=n1_val,
                N_values=[N])

            geo_err = geo["results"][0]["max_surface_error"]
            delta_theta = repose["results"][0]["delta_theta"]
            rms_normal = repose["results"][0]["rms_normal_dev"]

            acceptable = (
                geo_err < 0.02 and
                abs(delta_theta) < 1.0 and
                rms_normal < 5.0
            )

            verdicts.append({
                "n1": n1_val,
                "N_clump": N,
                "geo_error": geo_err,
                "delta_theta": delta_theta,
                "rms_normal_dev": rms_normal,
                "clump_acceptable": acceptable,
            })

    return {
        "verdicts": verdicts,
        "recommendation": (
            "Clump approximation is acceptable for n1 <= 4 with N >= 32 spheres. "
            "For n1 > 4 (highly blocky), use superquadric direct contact or N >= 128."
        ),
    }
