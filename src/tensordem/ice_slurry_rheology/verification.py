"""Acceptance faces for ice-water slurry rheology (campaign node p387w2-n09).

Six machine-verifiable criteria:
(a) Krieger-Dougherty mutual-adjudication error <= 30 %
(b) Eilers formula error <= 25 %
(c) Shear-thinning / shear-thickening transition detected
(d) Shape effect: angular / spherical viscosity ratio >= 1.2
(e) Mass conservation <= 1e-10
(f) Two negative controls
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

from .closures import (
    IceSlurrySpec,
    krieger_dougherty,
    eilers,
    shape_factor_angularity,
    viscosity_temperature,
    cross_adjudicate,
)
from .dem_slurry import IceSlurryCouette

__all__ = [
    "face_a_kd",
    "face_b_eilers",
    "face_c_shear_transition",
    "face_d_shape_effect",
    "face_e_mass_conservation",
    "face_f_negative_controls",
    "run_all",
]


# ---------------------------------------------------------------------------
# Face (a): Krieger-Dougherty mutual adjudication
# ---------------------------------------------------------------------------

def face_a_kd(
    phi_values: Sequence[float] = None,
    phi_max: float = 0.60,
    intrinsic_viscosity: float = 2.5,
) -> Dict[str, object]:
    """Criterion (a): Krieger-Dougherty vs Eilers mutual adjudication <= 30 %.

    The KD law is cross-checked against the Eilers law over a range of
    concentrations. The maximum relative deviation must be <= 30 %.
    """
    if phi_values is None:
        phi_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    result = cross_adjudicate(phi_values, phi_max, intrinsic_viscosity)

    # Also compute KD self-consistency: at low phi, should approach Einstein
    phi_low = 0.01
    eta_kd_low = krieger_dougherty(phi_low, phi_max, intrinsic_viscosity)
    eta_einstein = 1.0 + intrinsic_viscosity * phi_low
    einstein_relerr = abs(eta_kd_low - eta_einstein) / eta_einstein

    result["einstein_check"] = {
        "phi": phi_low,
        "eta_kd": eta_kd_low,
        "eta_einstein": eta_einstein,
        "relerr": einstein_relerr,
        "pass": bool(einstein_relerr < 0.05),
    }

    return {
        "criterion": "Krieger-Dougherty mutual adjudication",
        "threshold": 0.30,
        "max_relative_deviation": result["max_relative_deviation"],
        "pass": result["pass_kd"],
        "n_points": result["n_points"],
        "phi_values": list(phi_values),
        "eta_kd_values": [r["eta_kd"] for r in result["rows"]],
        "eta_eilers_values": [r["eta_eilers"] for r in result["rows"]],
        "einstein_check": result["einstein_check"],
    }


# ---------------------------------------------------------------------------
# Face (b): Eilers formula error
# ---------------------------------------------------------------------------

def face_b_eilers(
    phi_values: Sequence[float] = None,
    phi_max: float = 0.60,
    intrinsic_viscosity: float = 2.5,
) -> Dict[str, object]:
    """Criterion (b): Eilers formula error <= 25 % vs KD reference.

    The Eilers formula is compared against the Krieger-Dougherty law
    (taken as the reference). The maximum relative error must be <= 25 %.
    """
    if phi_values is None:
        phi_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    rows = []
    for phi in phi_values:
        eta_kd = krieger_dougherty(phi, phi_max, intrinsic_viscosity)
        eta_ei = eilers(phi, phi_max, intrinsic_viscosity)
        # Error relative to KD (reference)
        rel_err = abs(eta_ei - eta_kd) / eta_kd if eta_kd > 0 else 0.0
        rows.append({
            "phi": phi,
            "eta_kd": eta_kd,
            "eta_eilers": eta_ei,
            "rel_err_vs_kd": rel_err,
        })

    max_err = max(r["rel_err_vs_kd"] for r in rows) if rows else 0.0

    return {
        "criterion": "Eilers formula error vs Krieger-Dougherty",
        "threshold": 0.25,
        "max_relative_error": max_err,
        "pass": bool(max_err <= 0.25),
        "n_points": len(rows),
        "phi_values": list(phi_values),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Face (c): Shear-thinning / shear-thickening transition
# ---------------------------------------------------------------------------

def face_c_shear_transition(
    phi: float = 0.20,
    d: float = 0.005,
    mu_f: float = 1.787e-3,
    mu_contact: float = 0.3,
    shear_rates: Sequence[float] = None,
    duration: float = 0.01,
    sample_start: float = 0.005,
    seed: int = 42,
) -> Dict[str, object]:
    """Criterion (c): Detect shear-thinning to shear-thickening transition.

    Run DEM Couette at multiple shear rates and check that the effective
    viscosity shows a non-monotonic trend (decreasing then increasing),
    indicating a transition from shear-thinning to shear-thickening behavior.

    The transition is detected by fitting a quadratic to log(η) vs log(γ̇)
    and checking that the curvature is positive (U-shaped).
    """
    if shear_rates is None:
        shear_rates = [5.0, 10.0, 20.0, 40.0, 80.0]

    results = []
    for gamma_dot in shear_rates:
        H = 0.05
        cell = IceSlurryCouette(
            H=H, L=0.10, d=d, phi=phi,
            mu_f=mu_f, mu_contact=mu_contact,
            wall_velocity=gamma_dot * H,
            dt=5e-6, seed=seed,
        )
        res = cell.run(duration, sample_start)
        results.append({
            "gamma_dot": gamma_dot,
            "tau_Pa": res["tau_Pa"],
            "eta_eff_Pa_s": res["eta_eff_Pa_s"],
        })

    # Fit quadratic to log(eta) vs log(gamma_dot)
    log_gd = np.array([math.log10(r["gamma_dot"]) for r in results])
    log_eta = np.array([math.log10(max(r["eta_eff_Pa_s"], 1e-12)) for r in results])

    # Quadratic fit: log_eta = a + b*log_gd + c*log_gd^2
    A = np.vstack([np.ones_like(log_gd), log_gd, log_gd ** 2]).T
    coeffs, *_ = np.linalg.lstsq(A, log_eta, rcond=None)
    a, b, c = coeffs

    # Positive curvature (c > 0) indicates U-shape = thinning -> thickening
    curvature = float(c)

    # Also check for actual non-monotonicity in the data
    eta_values = [r["eta_eff_Pa_s"] for r in results]
    # Find minimum
    min_idx = int(np.argmin(eta_values))
    is_nonmonotonic = (
        min_idx > 0 and min_idx < len(eta_values) - 1 and
        eta_values[0] > eta_values[min_idx] and
        eta_values[-1] > eta_values[min_idx]
    )

    # Transition point (shear rate at minimum viscosity)
    transition_gamma_dot = float(results[min_idx]["gamma_dot"]) if is_nonmonotonic else None
    eta_min = float(eta_values[min_idx]) if is_nonmonotonic else None
    eta_ratio = float(eta_values[0] / eta_values[-1]) if eta_values[-1] > 0 else None

    return {
        "criterion": "Shear-thinning to shear-thickening transition",
        "curvature_log_log": curvature,
        "curvature_positive": bool(curvature > 0),
        "is_nonmonotonic": is_nonmonotonic,
        "transition_gamma_dot_1_s": transition_gamma_dot,
        "eta_min_Pa_s": eta_min,
        "eta_at_low_shear_Pa_s": float(eta_values[0]) if eta_values else None,
        "eta_at_high_shear_Pa_s": float(eta_values[-1]) if eta_values else None,
        "eta_ratio_low_to_high": eta_ratio,
        "pass": bool(is_nonmonotonic or curvature > 0),
        "n_shear_rates": len(results),
        "shear_rates": list(shear_rates),
        "eta_values": eta_values,
        "quadratic_coeffs": {"a": float(a), "b": float(b), "c": float(c)},
    }


# ---------------------------------------------------------------------------
# Face (d): Shape effect
# ---------------------------------------------------------------------------

def face_d_shape_effect(
    phi: float = 0.20,
    phi_max: float = 0.60,
    intrinsic_viscosity: float = 2.5,
) -> Dict[str, object]:
    """Criterion (d): Angular / spherical viscosity ratio >= 1.2.

    Compare the relative viscosity of angular particles vs spherical
    particles at the same concentration using the shape-factor-corrected
    Krieger-Dougherty law.
    """
    # Spherical: shape_factor = 1.0
    sf_spherical = 1.0
    eta_spherical = krieger_dougherty(phi, phi_max, intrinsic_viscosity * sf_spherical)

    # Angular: shape_factor = 1.35 (from literature)
    sf_angular, desc = shape_factor_angularity("angular")
    eta_angular = krieger_dougherty(phi, phi_max, intrinsic_viscosity * sf_angular)

    ratio = eta_angular / eta_spherical if eta_spherical > 0 else 0.0

    # Also test at multiple concentrations
    phi_test = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    ratios = []
    for p in phi_test:
        eta_s = krieger_dougherty(p, phi_max, intrinsic_viscosity)
        eta_a = krieger_dougherty(p, phi_max, intrinsic_viscosity * sf_angular)
        ratios.append(eta_a / eta_s if eta_s > 0 else 0.0)

    min_ratio = min(ratios) if ratios else 0.0

    return {
        "criterion": "Shape effect: angular / spherical viscosity ratio",
        "threshold": 1.2,
        "phi_test": phi,
        "eta_spherical": eta_spherical,
        "eta_angular": eta_angular,
        "shape_factor_spherical": sf_spherical,
        "shape_factor_angular": sf_angular,
        "shape_factor_description": desc,
        "ratio_at_phi": ratio,
        "ratios_at_multiple_phi": dict(zip(phi_test, ratios)),
        "min_ratio": min_ratio,
        "pass": bool(ratio >= 1.2),
    }


# ---------------------------------------------------------------------------
# Face (e): Mass conservation
# ---------------------------------------------------------------------------

def face_e_mass_conservation(
    phi: float = 0.20,
    d: float = 0.005,
    duration: float = 0.01,
    seed: int = 42,
) -> Dict[str, object]:
    """Criterion (e): Mass conservation <= 1e-10.

    Verify that the total mass of ice particles is conserved throughout
    the DEM simulation (no particle creation/destruction).
    """
    cell = IceSlurryCouette(
        H=0.05, L=0.10, d=d, phi=phi,
        wall_velocity=1.0,
        dt=5e-6, seed=seed,
    )
    cell.initialize()
    mass_initial = cell._mass_total
    n_initial = cell.n_particles

    # Run simulation
    n_steps = int(duration / cell.dt)
    for _ in range(n_steps):
        cell.step()

    mass_final = float(cell.n_particles) * cell.rho_p * math.pi * cell.r ** 2
    mass_error = abs(mass_final - mass_initial)
    mass_rel_error = mass_error / mass_initial if mass_initial > 0 else 0.0

    return {
        "criterion": "Mass conservation",
        "threshold": 1e-10,
        "mass_initial_kg": mass_initial,
        "mass_final_kg": mass_final,
        "mass_error_kg": mass_error,
        "mass_relative_error": mass_rel_error,
        "n_particles_initial": n_initial,
        "n_particles_final": cell.n_particles,
        "pass": bool(mass_rel_error <= 1e-10),
    }


# ---------------------------------------------------------------------------
# Face (f): Negative controls
# ---------------------------------------------------------------------------

def face_f_negative_controls(
    phi_max: float = 0.60,
    intrinsic_viscosity: float = 2.5,
) -> Dict[str, object]:
    """Criterion (f): Two negative controls.

    NC1: phi = 0 (pure water) -> eta_r = 1.0 exactly.
    NC2: phi -> phi_max -> eta_r diverges (raises ValueError).
    """
    # NC1: pure water
    eta_nc1_kd = krieger_dougherty(0.0, phi_max, intrinsic_viscosity)
    eta_nc1_ei = eilers(0.0, phi_max, intrinsic_viscosity)
    nc1_pass = bool(eta_nc1_kd == 1.0 and eta_nc1_ei == 1.0)

    # NC2: approach phi_max
    phi_near_max = phi_max * 0.999
    try:
        eta_nc2_kd = krieger_dougherty(phi_near_max, phi_max, intrinsic_viscosity)
        eta_nc2_ei = eilers(phi_near_max, phi_max, intrinsic_viscosity)
        # Both should be very large
        nc2_diverges = bool(eta_nc2_kd > 100 and eta_nc2_ei > 100)
        nc2_error = None
    except ValueError as e:
        nc2_diverges = True
        nc2_error = str(e)

    # Also test that phi >= phi_max raises
    try:
        krieger_dougherty(phi_max, phi_max, intrinsic_viscosity)
        nc2_raises = False
    except ValueError:
        nc2_raises = True

    return {
        "criterion": "Negative controls",
        "nc1": {
            "description": "phi=0 (pure water) -> eta_r = 1.0",
            "eta_kd": eta_nc1_kd,
            "eta_eilers": eta_nc1_ei,
            "pass": nc1_pass,
        },
        "nc2": {
            "description": "phi -> phi_max -> eta_r diverges",
            "phi_test": phi_near_max,
            "eta_kd_at_near_max": eta_nc2_kd if not nc2_error else None,
            "eta_eilers_at_near_max": eta_nc2_ei if not nc2_error else None,
            "diverges": nc2_diverges,
            "raises_at_phi_max": nc2_raises,
            "pass": bool(nc2_diverges and nc2_raises),
        },
        "pass": bool(nc1_pass and nc2_diverges and nc2_raises),
    }


# ---------------------------------------------------------------------------
# Run all faces
# ---------------------------------------------------------------------------

def run_all() -> Dict[str, object]:
    """Run all six acceptance faces and return the audit table."""
    return {
        "face_a_kd": face_a_kd(),
        "face_b_eilers": face_b_eilers(),
        "face_c_shear_transition": face_c_shear_transition(),
        "face_d_shape_effect": face_d_shape_effect(),
        "face_e_mass_conservation": face_e_mass_conservation(),
        "face_f_negative_controls": face_f_negative_controls(),
    }
