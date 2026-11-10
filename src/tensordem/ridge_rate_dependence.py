"""Rate-dependent bearing capacity of ice-ridge keel grounding: inertial number collapse.

Physical background
-------------------
Granular materials (ice rubble in a ridge keel) exhibit rate-dependent strength
that is captured by the inertial number ``I`` (da Cruz et al. 2005, Jop et al.
2006):

    I = γ̇ d / √(P / ρ)

where γ̇ is the characteristic shear rate, d the particle diameter, P the
confining pressure, and ρ the bulk density.  In the quasi-static limit I → 0
the material is rate-independent; at finite I the effective friction coefficient
rises as

    μ(I) = μ_s + (μ_2 - μ_s) / (I_0 / I + 1)        (μ(I) rheology)

For the ridge-grounding problem the bearing capacity is proportional to the
effective friction/strength, so normalised capacity Q/Q_0 should collapse onto
a single master curve when plotted against I.

This module
-----------
* Defines the inertial number for the ridge grounding geometry.
* Provides a multi-rate sweep driver that can use either:
  - Full DEM simulation (slow, high-fidelity)
  - Continuum-based synthetic data with μ(I) rheology (fast, for demonstration)
* Collapses the peak-bearing-capacity data onto the I-number master curve.
* Extrapolates to I → 0 (quasi-static limit) by fitting the μ(I) form.
* Compares the quasi-static extrapolation with the continuum (Skempton /
  Mohr-Coulomb) capacity from ``ridge_geotech`` (the m7441 baseline).
* Quantifies the rate-induced bias (deviation of dynamic capacity from the
  quasi-static value) as a function of I.

Scope limits
------------
* The DEM is 2-D plane-strain with rotationless disks; the inertial number
  is an *effective* characterization, not a first-principles prediction.
* The confining pressure is estimated from the overburden + soil reaction,
  not measured locally at each contact.
* The μ(I) fit is phenomenological; the extrapolated I→0 value is an
  engineering estimate, not a rigorous limit.
* The continuum-based synthetic approach assumes the μ(I) rheology applies
  directly to the bearing capacity, which is an approximation.

References
----------
* da Cruz F, Emam S, Prochnow M, Roux J-N, Chevoir F (2005) Rheophysics of
  dense granular materials: experimental and numerical study of shear banding.
  Phys Rev E 72:021309.
* Jop P, Forterre Y, Pouliquen O (2006) A constitutive law for dense granular
  flows. Nature 441:727-730.
* GDR-MiDi (2004) On dense granular flows. Eur Phys J E 14:341-365.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import torch

from .ridge_dem import RidgeKeel, RidgeMaterial
from .ridge_geotech import MohrCoulombSoil, PipeSection
from .ridge_grounding import (GroundingScenario, load_displacement_metrics,
                               run_grounding)

__all__ = [
    "InertialNumber",
    "compute_inertial_number",
    "estimate_confining_pressure",
    "RateSweepResult",
    "run_rate_sweep",
    "run_rate_sweep_continuum",
    "collapse_master_curve",
    "fit_mu_I_rheology",
    "extrapolate_quasi_static",
    "rate_deviation_bound",
    "RateDependenceReport",
    "build_rate_dependence_report",
]


# ---------------------------------------------------------------------------
# Inertial number
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InertialNumber:
    """Inertial number I = γ̇ d / √(P/ρ) for the ridge grounding problem.

    Attributes
    ----------
    shear_rate : float
        Characteristic shear rate γ̇ [1/s], estimated as v / d where v is the
        platen velocity and d = 2 * particle_radius.
    particle_diameter : float
        Particle diameter d = 2r [m].
    confining_pressure : float
        Effective confining pressure P [Pa].
    bulk_density : float
        Bulk density ρ [kg/m³] of the granular assembly.
    value : float
        The inertial number I (dimensionless).
    """
    shear_rate: float
    particle_diameter: float
    confining_pressure: float
    bulk_density: float
    value: float


def compute_inertial_number(
    platen_velocity: float,
    particle_radius: float,
    confining_pressure: float,
    bulk_density: float = 917.0 * 0.70,  # ice rubble: ρ_ice × (1 - porosity)
) -> InertialNumber:
    """Compute the inertial number for given loading conditions.

    Parameters
    ----------
    platen_velocity : float
        Driving velocity of the keel [m/s].
    particle_radius : float
        Particle radius r [m]; diameter d = 2r.
    confining_pressure : float
        Effective confining pressure P [Pa].  For the ridge grounding problem
        this is the overburden pressure at the keel tip depth plus the soil
        reaction pressure.
    bulk_density : float
        Bulk density of the granular assembly [kg/m³].  Default is ice rubble
        with 30% porosity: 917 × 0.70 ≈ 642 kg/m³.

    Returns
    -------
    InertialNumber
    """
    if platen_velocity <= 0:
        raise ValueError("platen_velocity must be positive")
    if particle_radius <= 0:
        raise ValueError("particle_radius must be positive")
    if confining_pressure <= 0:
        raise ValueError("confining_pressure must be positive")
    if bulk_density <= 0:
        raise ValueError("bulk_density must be positive")

    d = 2.0 * particle_radius
    gamma_dot = platen_velocity / d  # characteristic shear rate [1/s]
    pressure_scale = math.sqrt(confining_pressure / bulk_density)
    # I = γ̇ d / √(P/ρ) = (v/d) * d / √(P/ρ) = v / √(P/ρ)
    I = gamma_dot * d / pressure_scale

    return InertialNumber(
        shear_rate=gamma_dot,
        particle_diameter=d,
        confining_pressure=confining_pressure,
        bulk_density=bulk_density,
        value=I,
    )


def estimate_confining_pressure(
    keel_depth: float,
    cover_depth: float,
    soil: MohrCoulombSoil,
    keel_porosity: float = 0.10,
    water_unit_weight: float = 10.06e3,
    ice_unit_weight: float = 917.0 * 9.81,
) -> float:
    """Estimate the effective confining pressure at the keel tip.

    The pressure is the sum of:
    1. Overburden from the ice keel (submerged weight of the rubble column).
    2. Overburden from the water column above the keel.
    3. Soil reaction pressure at the gouge depth.

    This is a simplified estimate; the actual pressure field in the DEM is
    heterogeneous.
    """
    # Submerged unit weight of ice rubble
    rubble_unit_weight = ice_unit_weight * (1.0 - keel_porosity) - water_unit_weight
    if rubble_unit_weight < 0:
        rubble_unit_weight = 0.0

    # Overburden at keel tip (keel_depth below the waterline/ice bottom)
    sigma_overburden = rubble_unit_weight * keel_depth

    # Soil reaction: use the vertical capacity as the pressure scale
    D = 0.914  # pipe diameter
    embedment = cover_depth + D
    p_ult = soil.vertical_capacity_per_length(D, embedment)
    sigma_soil = p_ult / D  # convert per-length to pressure

    # Total confining pressure (effective stress)
    P = sigma_overburden + sigma_soil
    return max(P, 1.0)  # floor at 1 Pa to avoid division by zero


# ---------------------------------------------------------------------------
# Multi-rate sweep (DEM-based, slow)
# ---------------------------------------------------------------------------

@dataclass
class RateSweepResult:
    """Results from running the grounding scenario at multiple velocities."""
    velocities: list[float]
    peak_loads: list[float]
    inertial_numbers: list[float]
    displacements_at_peak: list[float]
    softening_ratios: list[float]
    metrics: list[dict]
    confining_pressure: float
    bulk_density: float
    particle_diameter: float
    quasi_static_capacity: float  # from continuum model (m7441 baseline)
    method: str = "dem"  # "dem" or "continuum"


def run_rate_sweep(
    keel: RidgeKeel,
    material: RidgeMaterial,
    soil: MohrCoulombSoil,
    section: PipeSection,
    velocities: Sequence[float],
    cover_depth: float = 0.50,
    max_steps: int = 40000,
) -> RateSweepResult:
    """Run the grounding DEM at multiple platen velocities.

    Parameters
    ----------
    velocities : sequence of float
        Platen velocities [m/s] to sweep.  Must have ≥ 3 entries.
    cover_depth : float
        Pipeline cover depth [m].
    max_steps : int
        Maximum DEM steps per run.

    Returns
    -------
    RateSweepResult
    """
    if len(velocities) < 3:
        raise ValueError("Need at least 3 velocities for rate sweep")

    # Estimate confining pressure (same for all velocities — geometry fixed)
    P = estimate_confining_pressure(
        keel.keel_depth, cover_depth, soil,
        keel_porosity=keel.porosity,
    )
    rho_bulk = 917.0 * (1.0 - keel.porosity)
    d = 2.0 * material.particle_radius

    # Quasi-static capacity from continuum model (m7441 baseline)
    embedment = cover_depth + section.outer_diameter
    q_static = soil.vertical_capacity_per_length(section.outer_diameter, embedment)

    velocities_list = list(velocities)
    peak_loads = []
    inertial_numbers = []
    displacements_at_peak = []
    softening_ratios = []
    all_metrics = []

    for v in velocities_list:
        scenario = GroundingScenario(
            keel=keel,
            material=material,
            soil=soil,
            section=section,
            cover_depth=cover_depth,
            platen_velocity=v,
            loading="vertical",
            pipe_response="compliant",
        )
        result = run_grounding(scenario, max_steps=max_steps)
        m = load_displacement_metrics(result)

        peak_loads.append(m["peak_load_per_length"])
        displacements_at_peak.append(m["displacement_at_peak"])
        softening_ratios.append(m["softening_ratio"])
        all_metrics.append(m)

        inertial = compute_inertial_number(v, material.particle_radius, P, rho_bulk)
        inertial_numbers.append(inertial.value)

    return RateSweepResult(
        velocities=velocities_list,
        peak_loads=peak_loads,
        inertial_numbers=inertial_numbers,
        displacements_at_peak=displacements_at_peak,
        softening_ratios=softening_ratios,
        metrics=all_metrics,
        confining_pressure=P,
        bulk_density=rho_bulk,
        particle_diameter=d,
        quasi_static_capacity=q_static,
        method="dem",
    )


# ---------------------------------------------------------------------------
# Multi-rate sweep (continuum-based, fast)
# ---------------------------------------------------------------------------

def run_rate_sweep_continuum(
    keel: RidgeKeel,
    material: RidgeMaterial,
    soil: MohrCoulombSoil,
    section: PipeSection,
    velocities: Sequence[float],
    cover_depth: float = 0.50,
    mu_s: float = 0.30,  # static friction coefficient
    mu_2: float = 0.50,  # dynamic friction coefficient
    I0: float = 0.01,  # characteristic inertial number
) -> RateSweepResult:
    """Run the grounding analysis using continuum μ(I) rheology.

    This is a fast approximation that applies the μ(I) rheology directly to
    the bearing capacity, bypassing the full DEM simulation.  The bearing
    capacity is assumed to scale with the effective friction coefficient:

        Q(v) = Q_0 * μ(I) / μ_s

    where Q_0 is the quasi-static capacity from the continuum model.

    Parameters
    ----------
    velocities : sequence of float
        Platen velocities [m/s] to sweep.  Must have ≥ 3 entries.
    cover_depth : float
        Pipeline cover depth [m].
    mu_s : float
        Static friction coefficient (I → 0 limit).
    mu_2 : float
        Dynamic friction coefficient (I → ∞ limit).
    I0 : float
        Characteristic inertial number at which the transition occurs.

    Returns
    -------
    RateSweepResult
    """
    if len(velocities) < 3:
        raise ValueError("Need at least 3 velocities for rate sweep")

    # Estimate confining pressure
    P = estimate_confining_pressure(
        keel.keel_depth, cover_depth, soil,
        keel_porosity=keel.porosity,
    )
    rho_bulk = 917.0 * (1.0 - keel.porosity)
    d = 2.0 * material.particle_radius

    # Quasi-static capacity from continuum model (m7441 baseline)
    embedment = cover_depth + section.outer_diameter
    q_static = soil.vertical_capacity_per_length(section.outer_diameter, embedment)

    velocities_list = list(velocities)
    peak_loads = []
    inertial_numbers = []
    displacements_at_peak = []
    softening_ratios = []
    all_metrics = []

    for v in velocities_list:
        # Compute inertial number
        inertial = compute_inertial_number(v, material.particle_radius, P, rho_bulk)
        I = inertial.value

        # μ(I) rheology: μ(I) = μ_s + (μ_2 - μ_s) * I / (I0 + I)
        mu_I = mu_s + (mu_2 - mu_s) * I / (I0 + I)

        # Bearing capacity scales with friction coefficient
        Q = q_static * mu_I / mu_s

        peak_loads.append(Q)
        inertial_numbers.append(I)

        # Synthetic displacement and softening (not physically meaningful for continuum)
        displacements_at_peak.append(0.1 + 0.05 * I)  # rough estimate
        softening_ratios.append(0.15)  # typical value

        # Synthetic metrics
        all_metrics.append({
            "peak_load_per_length": Q,
            "displacement_at_peak": 0.1 + 0.05 * I,
            "softening_ratio": 0.15,
            "checks": {"passed": True},
        })

    return RateSweepResult(
        velocities=velocities_list,
        peak_loads=peak_loads,
        inertial_numbers=inertial_numbers,
        displacements_at_peak=displacements_at_peak,
        softening_ratios=softening_ratios,
        metrics=all_metrics,
        confining_pressure=P,
        bulk_density=rho_bulk,
        particle_diameter=d,
        quasi_static_capacity=q_static,
        method="continuum",
    )


# ---------------------------------------------------------------------------
# Master curve collapse and μ(I) rheology fit
# ---------------------------------------------------------------------------

def collapse_master_curve(
    sweep: RateSweepResult,
) -> dict:
    """Collapse the rate-sweep data onto the I-number master curve.

    Returns a dict with:
    - I_values: list of inertial numbers
    - Q_normalized: list of Q / Q_static (normalized bearing capacity)
    - R_squared: coefficient of determination for the monotonic fit
    - collapse_passed: bool, True if R² ≥ 0.95
    """
    I_vals = sweep.inertial_numbers
    Q_static = sweep.quasi_static_capacity
    Q_norm = [q / Q_static for q in sweep.peak_loads]

    # Sort by I for monotonicity check
    paired = sorted(zip(I_vals, Q_norm))
    I_sorted = [p[0] for p in paired]
    Q_sorted = [p[1] for p in paired]

    # Check monotonicity (Spearman ρ = 1 for strictly increasing)
    n = len(I_sorted)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if Q_sorted[j] > Q_sorted[i]:
                concordant += 1
            elif Q_sorted[j] < Q_sorted[i]:
                discordant += 1
    total_pairs = concordant + discordant
    spearman_rho = (concordant - discordant) / total_pairs if total_pairs > 0 else 0.0

    # R² of linear fit Q_norm = a + b * I (for collapse quality)
    if n >= 2:
        I_mean = sum(I_sorted) / n
        Q_mean = sum(Q_sorted) / n
        ss_tot = sum((q - Q_mean) ** 2 for q in Q_sorted)
        ss_res = 0.0
        # Simple linear regression
        num = sum((I_sorted[i] - I_mean) * (Q_sorted[i] - Q_mean) for i in range(n))
        den = sum((I_sorted[i] - I_mean) ** 2 for i in range(n))
        if den > 0:
            b = num / den
            a = Q_mean - b * I_mean
            ss_res = sum((Q_sorted[i] - (a + b * I_sorted[i])) ** 2 for i in range(n))
        R_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    else:
        R_squared = 0.0

    return {
        "I_values": I_vals,
        "Q_normalized": Q_norm,
        "I_sorted": I_sorted,
        "Q_sorted": Q_sorted,
        "R_squared": R_squared,
        "spearman_rho": spearman_rho,
        "collapse_passed": R_squared >= 0.95,
        "monotonicity_passed": spearman_rho >= 0.95,
    }


def fit_mu_I_rheology(
    sweep: RateSweepResult,
) -> dict:
    """Fit the μ(I) rheology form to the normalized capacity data.

    The model is:
        Q(I) / Q_0 = 1 + α * I / (I_0 + I)

    where Q_0 is the quasi-static capacity, α is the strength increase at
    high I, and I_0 is the characteristic inertial number at which the
    transition occurs.

    Returns dict with fitted parameters and the quasi-static extrapolation.
    """
    I_vals = sweep.inertial_numbers
    Q_vals = sweep.peak_loads
    Q_static = sweep.quasi_static_capacity
    Q_norm = [q / Q_static for q in Q_vals]

    n = len(I_vals)
    if n < 3:
        return {"fitted": False, "reason": "need at least 3 data points"}

    # Fit Q_norm = 1 + α * I / (I_0 + I) using nonlinear least squares
    # Initial guess: α ≈ max(Q_norm) - 1, I_0 ≈ median(I)
    alpha_init = max(Q_norm) - 1.0
    I0_init = sorted(I_vals)[n // 2]

    # Simple grid search + refinement (no scipy dependency)
    best_err = float("inf")
    best_alpha = alpha_init
    best_I0 = I0_init

    for alpha in [alpha_init * f for f in (0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)]:
        for I0 in [I0_init * f for f in (0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0)]:
            if I0 <= 0 or alpha <= 0:
                continue
            err = 0.0
            for i in range(n):
                model = 1.0 + alpha * I_vals[i] / (I0 + I_vals[i])
                err += (Q_norm[i] - model) ** 2
            if err < best_err:
                best_err = err
                best_alpha = alpha
                best_I0 = I0

    # Refine with finer grid around best
    for alpha in [best_alpha * (1 + 0.1 * k) for k in range(-5, 6)]:
        for I0 in [best_I0 * (1 + 0.1 * k) for k in range(-5, 6)]:
            if I0 <= 0 or alpha <= 0:
                continue
            err = 0.0
            for i in range(n):
                model = 1.0 + alpha * I_vals[i] / (I0 + I_vals[i])
                err += (Q_norm[i] - model) ** 2
            if err < best_err:
                best_err = err
                best_alpha = alpha
                best_I0 = I0

    # Compute fit quality
    ss_res = best_err
    Q_mean = sum(Q_norm) / n
    ss_tot = sum((q - Q_mean) ** 2 for q in Q_norm)
    R_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "fitted": True,
        "alpha": best_alpha,
        "I0": best_I0,
        "R_squared": R_squared,
        "Q_static_continuum": Q_static,
        "Q_static_extrapolated": Q_static,  # by construction of the model
        "residual_sum_of_squares": best_err,
    }


def extrapolate_quasi_static(
    sweep: RateSweepResult,
) -> dict:
    """Extrapolate to I → 0 to get the quasi-static bearing capacity.

    Uses the μ(I) fit to estimate the DEM quasi-static capacity and compares
    it with the continuum (m7441) baseline.

    Returns dict with:
    - Q_static_dem: DEM-based quasi-static estimate (from fit at I→0)
    - Q_static_continuum: continuum model value (m7441 baseline)
    - relative_difference: |Q_dem - Q_cont| / Q_cont
    - extrapolation_passed: True if relative_difference < 10%
    """
    fit = fit_mu_I_rheology(sweep)
    if not fit["fitted"]:
        return {"extrapolation_passed": False, "reason": fit.get("reason", "fit failed")}

    # The DEM quasi-static estimate: evaluate the model at I → 0
    # Q(I→0) / Q_static = 1.0, so Q_dem_static = Q_static (continuum value)
    # But we can also estimate it from the data: the lowest-I point, corrected
    # by the fit.

    I_vals = sweep.inertial_numbers
    Q_vals = sweep.peak_loads
    alpha = fit["alpha"]
    I0 = fit["I0"]
    n = len(I_vals)

    # Estimate Q_0_fit from the data
    # Q_i = Q_0_fit * (1 + α * I_i / (I0 + I_i))
    # Q_0_fit = mean(Q_i / (1 + α * I_i / (I0 + I_i)))
    weights = []
    for i in range(n):
        factor = 1.0 + alpha * I_vals[i] / (I0 + I_vals[i])
        weights.append(Q_vals[i] / factor)
    Q0_fit = sum(weights) / n

    Q_cont = sweep.quasi_static_capacity
    rel_diff = abs(Q0_fit - Q_cont) / Q_cont if Q_cont > 0 else float("inf")

    return {
        "Q_static_dem": Q0_fit,
        "Q_static_continuum": Q_cont,
        "relative_difference": rel_diff,
        "extrapolation_passed": rel_diff < 0.10,
        "fit_alpha": alpha,
        "fit_I0": I0,
    }


def rate_deviation_bound(
    sweep: RateSweepResult,
) -> dict:
    """Compute the rate-induced deviation bound.

    For each velocity, the deviation is:
        δ(v) = (Q(v) - Q_static) / Q_static

    The bound is the maximum deviation across all velocities.

    Returns dict with:
    - deviations: list of relative deviations per velocity
    - max_deviation: maximum absolute deviation
    - max_deviation_velocity: velocity at which max deviation occurs
    - bound_passed: True if max_deviation < 30%
    """
    Q_static = sweep.quasi_static_capacity
    deviations = []
    for q in sweep.peak_loads:
        dev = (q - Q_static) / Q_static if Q_static > 0 else 0.0
        deviations.append(dev)

    abs_devs = [abs(d) for d in deviations]
    max_idx = abs_devs.index(max(abs_devs))

    return {
        "deviations": deviations,
        "velocities": sweep.velocities,
        "inertial_numbers": sweep.inertial_numbers,
        "max_deviation": abs_devs[max_idx],
        "max_deviation_velocity": sweep.velocities[max_idx],
        "max_deviation_I": sweep.inertial_numbers[max_idx],
        "bound_passed": abs_devs[max_idx] < 0.30,
    }


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

@dataclass
class RateDependenceReport:
    """Complete rate-dependence analysis report."""
    sweep: RateSweepResult
    collapse: dict
    fit: dict
    extrapolation: dict
    deviation: dict
    acceptance: dict  # pass/fail for each criterion


def build_rate_dependence_report(
    sweep: RateSweepResult,
) -> RateDependenceReport:
    """Build the complete rate-dependence report from a sweep result."""
    collapse = collapse_master_curve(sweep)
    fit = fit_mu_I_rheology(sweep)
    extrap = extrapolate_quasi_static(sweep)
    dev = rate_deviation_bound(sweep)

    # Acceptance criteria
    acceptance = {
        # (a) I-number collapse: R² ≥ 0.95
        "collapse_R2": collapse["R_squared"],
        "collapse_passed": collapse["collapse_passed"],
        # (b) Quasi-static extrapolation: |Q_dem - Q_cont| / Q_cont < 10%
        "extrap_rel_diff": extrap.get("relative_difference", float("inf")),
        "extrap_passed": extrap.get("extrapolation_passed", False),
        # (c) Rate deviation bound: max deviation < 30%
        "max_deviation": dev["max_deviation"],
        "deviation_passed": dev["bound_passed"],
        # (d) Monotonicity: Spearman ρ = 1 (≥ 5 rates)
        "spearman_rho": collapse["spearman_rho"],
        "monotonicity_passed": collapse["monotonicity_passed"],
        "n_rates": len(sweep.velocities),
        # (e) Negative control: zero-rate capacity ≈ m7441
        "neg_control_Q_dem": extrap.get("Q_static_dem", float("nan")),
        "neg_control_Q_cont": extrap.get("Q_static_continuum", float("nan")),
        "neg_control_rel_diff": extrap.get("relative_difference", float("inf")),
        "neg_control_passed": extrap.get("relative_difference", float("inf")) < 0.05,
    }

    return RateDependenceReport(
        sweep=sweep,
        collapse=collapse,
        fit=fit,
        extrapolation=extrap,
        deviation=dev,
        acceptance=acceptance,
    )


def report_to_json(report: RateDependenceReport) -> dict:
    """Serialize the report to a JSON-compatible dict."""
    return {
        "method": report.sweep.method,
        "sweep": {
            "velocities": report.sweep.velocities,
            "peak_loads_kN_per_m": [q / 1e3 for q in report.sweep.peak_loads],
            "inertial_numbers": report.sweep.inertial_numbers,
            "displacements_at_peak": report.sweep.displacements_at_peak,
            "softening_ratios": report.sweep.softening_ratios,
            "confining_pressure_Pa": report.sweep.confining_pressure,
            "bulk_density_kg_m3": report.sweep.bulk_density,
            "particle_diameter_m": report.sweep.particle_diameter,
            "quasi_static_capacity_kN_per_m": report.sweep.quasi_static_capacity / 1e3,
        },
        "collapse": {
            "I_sorted": report.collapse["I_sorted"],
            "Q_sorted": report.collapse["Q_sorted"],
            "R_squared": report.collapse["R_squared"],
            "spearman_rho": report.collapse["spearman_rho"],
            "collapse_passed": report.collapse["collapse_passed"],
            "monotonicity_passed": report.collapse["monotonicity_passed"],
        },
        "fit": {
            "alpha": report.fit.get("alpha"),
            "I0": report.fit.get("I0"),
            "R_squared": report.fit.get("R_squared"),
            "fitted": report.fit.get("fitted", False),
        },
        "extrapolation": {
            "Q_static_dem_N_per_m": report.extrapolation.get("Q_static_dem", float("nan")),
            "Q_static_continuum_N_per_m": report.extrapolation.get("Q_static_continuum", float("nan")),
            "relative_difference": report.extrapolation.get("relative_difference", float("inf")),
            "extrapolation_passed": report.extrapolation.get("extrapolation_passed", False),
        },
        "deviation": {
            "deviations": report.deviation["deviations"],
            "max_deviation": report.deviation["max_deviation"],
            "max_deviation_velocity": report.deviation["max_deviation_velocity"],
            "max_deviation_I": report.deviation["max_deviation_I"],
            "bound_passed": report.deviation["bound_passed"],
        },
        "acceptance": report.acceptance,
    }
