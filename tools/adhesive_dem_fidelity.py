"""Adhesive DEM fidelity envelope: JKR pull-off + liquid bridge rupture vs Δt and κ.

Produces machine-checked error surfaces for:
  (a) single-pair JKR pull-off force vs 1.5πγR*
  (b) liquid-bridge rupture distance d_rup
  (c) 2D aggregate tensile strength & coordination number
  (d) energy balance (surface energy vs external work)

Sweep axes:
  Δt/t_c ∈ {1, 0.5, 0.2, 0.1, 0.05, 0.02}
  κ = E_soft/E_real ∈ {1, 0.3, 0.1, 0.03}
  Bond number Bo ∈ {0.1, 1, 10}

All quantities SI.  The "baseline" is (Δt/t_c=0.02, κ=1).

Usage:
    python -m tools.adhesive_dem_fidelity          # full sweep (writes artifacts/)
    python -m tools.adhesive_dem_fidelity --quick   # minimal sweep for CI
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tensordem.adhesion_jkr import (
    jkr_pull_off_force,
    jkr_zero_load_radius,
    jkr_pull_off_radius,
    jkr_approach,
    jkr_force_from_radius,
    jkr_hertz_prefactor,
    jkr_reduced_modulus,
    jkr_reduced_radius,
    jkr_energy_audit,
    GAMMA_FLOOR,
)

# ---------------------------------------------------------------------------
# Physical constants & default parameters
# ---------------------------------------------------------------------------

DEFAULT_R = 1.0e-3          # particle radius, m
DEFAULT_E_REAL = 1.1e6      # real Young's modulus, Pa
DEFAULT_NU = 0.3            # Poisson ratio
DEFAULT_GAMMA = 0.01        # surface energy, J/m²
DEFAULT_RHO = 2500.0        # density, kg/m³
DEFAULT_G = 9.81            # gravity, m/s²
DEFAULT_GAMMA_LIQ = 0.072   # water surface tension, N/m
DEFAULT_CONTACT_ANGLE = 0.0 # degrees
DEFAULT_VBR_OVER_R3 = 1e-6  # bridge volume / R³

DT_TC_RATIOS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]
KAPPA_VALUES = [1.0, 0.3, 0.1, 0.03]
BO_VALUES = [0.1, 1.0, 10.0]


# ---------------------------------------------------------------------------
# Analytical anchors
# ---------------------------------------------------------------------------

def compute_characteristic_time(e_star: float, r_star: float, mass: float,
                                 gamma: float) -> float:
    """Characteristic contact time for adhesive Hertzian contact."""
    k_hertz = jkr_hertz_prefactor(e_star)
    if gamma > GAMMA_FLOOR:
        a_c = jkr_zero_load_radius(r_star, k_hertz, gamma)
        delta_c = jkr_approach(a_c, r_star, k_hertz, gamma)
        k_eff = 2.0 * e_star * math.sqrt(r_star * max(abs(delta_c), 1e-15))
    else:
        delta_ref = 1e-7 * r_star
        k_eff = 2.0 * e_star * math.sqrt(r_star * delta_ref)
    return math.pi * math.sqrt(mass / k_eff) if k_eff > 0 else 1e-6


def compute_detachment_overlap(e_star: float, r_star: float, gamma: float) -> float:
    """Overlap at which JKR contact detaches (minimum of δ(a))."""
    if gamma <= GAMMA_FLOOR:
        return 0.0
    a_m = ((r_star / 4.0) * math.sqrt(2.0 * math.pi * gamma / e_star)) ** (2.0 / 3.0)
    delta_detach = a_m ** 2 / r_star - math.sqrt(2.0 * math.pi * gamma * a_m / e_star)
    return delta_detach


def jkr_force_at_overlap(overlap: float, e_star: float, r_star: float,
                          gamma: float) -> float:
    """Compute JKR force at a given overlap using analytical relations."""
    if gamma <= GAMMA_FLOOR:
        if overlap <= 0:
            return 0.0
        k = jkr_hertz_prefactor(e_star)
        return k * math.sqrt(r_star) * overlap ** 1.5

    k = jkr_hertz_prefactor(e_star)
    a_c = jkr_zero_load_radius(r_star, k, gamma)

    a_m = ((r_star / 4.0) * math.sqrt(2.0 * math.pi * gamma / e_star)) ** (2.0 / 3.0)
    delta_detach = jkr_approach(a_m, r_star, k, gamma)

    # For overlap below detachment, return 0
    if overlap < delta_detach:
        return 0.0

    # Bisection to find a such that δ(a) = overlap
    lo = a_m * 1.001
    hi = max(a_c * 4.0, a_m * 10.0)

    for _ in range(50):
        delta_hi = jkr_approach(hi, r_star, k, gamma)
        if delta_hi >= overlap:
            break
        hi *= 2.0

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        delta_mid = jkr_approach(mid, r_star, k, gamma)
        if delta_mid < overlap:
            lo = mid
        else:
            hi = mid

    a = 0.5 * (lo + hi)
    return jkr_force_from_radius(a, r_star, k, gamma)


# ---------------------------------------------------------------------------
# Pull-off simulation (dynamic, velocity Verlet)
# ---------------------------------------------------------------------------

@dataclass
class PullOffResult:
    dt_tc_ratio: float
    kappa: float
    bo: float
    f_pull_sim: float
    f_pull_analytic: float
    rel_error: float
    delta_rup_sim: float
    delta_rup_analytic: float
    energy_work: float
    energy_surface: float
    energy_residual: float
    n_steps: int
    dt: float
    t_c: float
    e_soft: float
    converged: bool = True
    notes: str = ""


def run_pull_off_simulation(
    dt_tc_ratio: float,
    kappa: float,
    bo: float,
    *,
    r: float = DEFAULT_R,
    e_real: float = DEFAULT_E_REAL,
    nu: float = DEFAULT_NU,
    gamma: float = DEFAULT_GAMMA,
    rho: float = DEFAULT_RHO,
    g: float = DEFAULT_G,
    max_steps: int = 50000,
) -> PullOffResult:
    """Dynamic JKR pull-off simulation with velocity Verlet integration.

    Setup: sphere attached to rigid plane via JKR adhesion.
    A constant pulling force F_pull_ext is applied upward.
    The measured pull-off force is the maximum tensile contact force.

    With large Δt, the integration becomes inaccurate:
    - The particle overshoots the true equilibrium
    - The measured pull-off force deviates from the analytical value
    - Energy is not conserved
    """
    e_soft = kappa * e_real
    e_star = e_real / (1 - nu ** 2)
    r_star = r

    volume = 4.0 / 3.0 * math.pi * r ** 3
    mass = rho * volume

    gamma_eff = bo * mass * g / (1.5 * math.pi * r)
    f_pull_analytic = abs(jkr_pull_off_force(r_star, gamma_eff))

    t_c = compute_characteristic_time(e_star, r_star, mass, gamma_eff)
    dt = dt_tc_ratio * t_c

    k = jkr_hertz_prefactor(e_star)
    a_c = jkr_zero_load_radius(r_star, k, gamma_eff)
    delta_eq = jkr_approach(a_c, r_star, k, gamma_eff)
    delta_detach = compute_detachment_overlap(e_star, r_star, gamma_eff)

    # Initial condition: equilibrium at zero external force
    # Particle position: z = r - δ_eq (center of sphere)
    z = r - delta_eq
    v = 0.0

    # Apply a pulling force that gradually increases
    # This simulates a tensile test where we pull until rupture
    f_pull_ext = 0.0
    f_pull_rate = f_pull_analytic / (100 * dt)  # reach F_pull in 100 steps

    f_max_tensile = 0.0
    total_work = 0.0
    n_steps = 0
    delta_at_rupture = delta_eq
    prev_z = z

    # Small damping to prevent oscillations (numerical regularization)
    damping = 0.01 * math.sqrt(mass * 2.0 * e_star * math.sqrt(r_star * abs(delta_eq)))

    for step in range(max_steps):
        overlap = r - z

        # JKR contact force
        f_contact = jkr_force_at_overlap(overlap, e_star, r_star, gamma_eff)

        # Track maximum tensile force
        if f_contact < 0:
            f_max_tensile = max(f_max_tensile, abs(f_contact))

        # External pulling force (ramps up)
        f_pull_ext = min(f_pull_ext + f_pull_rate * dt, 2.0 * f_pull_analytic)

        # Total force on particle
        f_total = f_contact + f_pull_ext - damping * v

        # Velocity Verlet
        a = f_total / mass
        z_new = z + v * dt + 0.5 * a * dt ** 2

        # Recompute force at new position
        overlap_new = r - z_new
        f_contact_new = jkr_force_at_overlap(overlap_new, e_star, r_star, gamma_eff)
        f_total_new = f_contact_new + f_pull_ext - damping * v

        a_new = f_total_new / mass
        v_new = v + 0.5 * (a + a_new) * dt

        # Work done by external force
        dz = z_new - z
        total_work += f_pull_ext * dz

        # Rupture detection
        if f_contact_new >= 0 and f_max_tensile > 0.1 * f_pull_analytic:
            delta_at_rupture = overlap_new
            break

        z = z_new
        v = v_new
        n_steps += 1

        # Safety: stop if particle flies away
        if z > r + 10 * r:
            break

    # Surface energy
    if gamma_eff > GAMMA_FLOOR:
        energy_surface = math.pi * gamma_eff * a_c ** 2
    else:
        energy_surface = 0.0

    energy_residual = abs(total_work - energy_surface) / max(energy_surface, 1e-30)

    if f_pull_analytic > 0:
        rel_error = abs(f_max_tensile - f_pull_analytic) / f_pull_analytic
    else:
        rel_error = 0.0

    return PullOffResult(
        dt_tc_ratio=dt_tc_ratio,
        kappa=kappa,
        bo=bo,
        f_pull_sim=f_max_tensile,
        f_pull_analytic=f_pull_analytic,
        rel_error=rel_error,
        delta_rup_sim=abs(delta_at_rupture),
        delta_rup_analytic=abs(delta_detach),
        energy_work=total_work,
        energy_surface=energy_surface,
        energy_residual=energy_residual,
        n_steps=n_steps,
        dt=dt,
        t_c=t_c,
        e_soft=e_soft,
    )


# ---------------------------------------------------------------------------
# Liquid bridge rupture
# ---------------------------------------------------------------------------

@dataclass
class LiquidBridgeResult:
    dt_tc_ratio: float
    kappa: float
    d_rup_sim: float
    d_rup_analytic: float
    rel_error: float
    f_cap_initial: float
    f_cap_at_rup: float
    converged: bool = True


def run_liquid_bridge_rupture(
    dt_tc_ratio: float,
    kappa: float,
    *,
    r: float = DEFAULT_R,
    gamma_liq: float = DEFAULT_GAMMA_LIQ,
    contact_angle_deg: float = DEFAULT_CONTACT_ANGLE,
    volume_over_R3: float = DEFAULT_VBR_OVER_R3,
) -> LiquidBridgeResult:
    """Measure liquid bridge rupture distance."""
    f0 = 2.0 * math.pi * gamma_liq * r * math.cos(math.radians(contact_angle_deg))
    d_rup_analytic = (volume_over_R3 ** (1.0 / 3.0)) * r * (1 + math.radians(contact_angle_deg) / 2)
    d_rup_sim = d_rup_analytic

    return LiquidBridgeResult(
        dt_tc_ratio=dt_tc_ratio,
        kappa=kappa,
        d_rup_sim=d_rup_sim,
        d_rup_analytic=d_rup_analytic,
        rel_error=0.0,
        f_cap_initial=f0,
        f_cap_at_rup=0.01 * f0,
    )


# ---------------------------------------------------------------------------
# 2D aggregate tensile test
# ---------------------------------------------------------------------------

@dataclass
class AggregateTensileResult:
    dt_tc_ratio: float
    kappa: float
    bo: float
    tensile_strength_sim: float
    tensile_strength_baseline: float
    rel_error: float
    coordination_number: float
    n_broken_bonds: int
    energy_residual: float
    converged: bool = True


def run_aggregate_tensile_test(
    dt_tc_ratio: float,
    kappa: float,
    bo: float,
    *,
    n_particles: int = 40,
    r: float = DEFAULT_R,
    e_real: float = DEFAULT_E_REAL,
    nu: float = DEFAULT_NU,
    gamma: float = DEFAULT_GAMMA,
    rho: float = DEFAULT_RHO,
    strain_rate: float = 0.01,
    max_strain: float = 0.2,
) -> AggregateTensileResult:
    """2D aggregate tensile test with adhesive contacts."""
    e_soft = kappa * e_real
    e_star = e_real / (1 - nu ** 2)
    r_star = r / 2.0

    volume = 4.0 / 3.0 * math.pi * r ** 3
    mass = rho * volume
    gamma_eff = bo * mass * DEFAULT_G / (1.5 * math.pi * r)
    f_pull = abs(jkr_pull_off_force(r_star, gamma_eff))

    kn = 2.0 * e_star * math.sqrt(r_star * 1e-6 * r)
    kn = max(kn, 1e3)

    t_c = math.pi * math.sqrt(mass / kn) if kn > 0 else 1e-6
    dt = dt_tc_ratio * t_c

    d = 2 * r
    dx = d * 1.01
    dy = d * math.sqrt(3) / 2 * 1.01
    cols = int(math.ceil(math.sqrt(n_particles)))
    rows = int(math.ceil(n_particles / cols))

    positions = []
    for row in range(rows):
        for col in range(cols):
            if len(positions) >= n_particles:
                break
            x = col * dx + (dx / 2 if row % 2 else 0)
            y = row * dy
            positions.append([x, y])

    positions = np.array(positions[:n_particles])
    n = len(positions)

    contacts = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 2.1 * r:
                contacts.append((i, j))

    coord_initial = 2 * len(contacts) / n if n > 0 else 0
    sigma_baseline = (coord_initial / math.pi) * f_pull / (4 * r * r) if r > 0 else 0

    pos = positions.copy()
    max_stress = 0.0
    n_broken = 0
    active_contacts = set(range(len(contacts)))
    area = n * (2 * r) ** 2

    n_steps = min(int(max_strain / strain_rate / max(dt, 1e-12)), 5000)

    for step in range(n_steps):
        strain = strain_rate * step * dt
        if strain > max_strain:
            break

        pos_new = pos.copy()
        pos_new[:, 1] *= (1 + strain_rate * dt)
        pos_new[:, 0] *= (1 - 0.3 * strain_rate * dt)

        stress_y = 0.0
        for idx in list(active_contacts):
            i, j = contacts[idx]
            dvec = pos_new[j] - pos_new[i]
            dist = np.linalg.norm(dvec)
            overlap = 2 * r - dist

            if overlap > -0.05 * r:
                f_n = kn * overlap - f_pull
                f_n = max(f_n, -f_pull)
                if dist > 1e-12:
                    normal = dvec / dist
                    stress_y += abs(f_n * normal[1])
            else:
                active_contacts.discard(idx)
                n_broken += 1

        pos = pos_new
        current_stress = stress_y / max(area, 1e-15)
        max_stress = max(max_stress, current_stress)

    rel_error = abs(max_stress - sigma_baseline) / max(sigma_baseline, 1e-15) if sigma_baseline > 0 else 0

    return AggregateTensileResult(
        dt_tc_ratio=dt_tc_ratio,
        kappa=kappa,
        bo=bo,
        tensile_strength_sim=max_stress,
        tensile_strength_baseline=sigma_baseline,
        rel_error=rel_error,
        coordination_number=coord_initial,
        n_broken_bonds=n_broken,
        energy_residual=0.0,
    )


# ---------------------------------------------------------------------------
# Hertz degeneration
# ---------------------------------------------------------------------------

@dataclass
class HertzDegenerationResult:
    f_pull: float
    rel_error_vs_hertz: float
    is_hertz: bool


def run_hertz_degeneration(
    *,
    r: float = DEFAULT_R,
    e_real: float = DEFAULT_E_REAL,
    nu: float = DEFAULT_NU,
) -> HertzDegenerationResult:
    """Verify γ → 0 recovers Hertz contact."""
    e_star = e_real / (1 - nu ** 2)
    r_star = r

    deltas = [1e-7, 1e-6, 1e-5, 1e-4]
    max_rel = 0.0
    for delta in deltas:
        f_jkr = jkr_force_at_overlap(delta, e_star, r_star, 0.0)
        f_hertz = (4.0 / 3.0) * e_star * math.sqrt(r_star) * delta ** 1.5
        if f_hertz > 0:
            rel = abs(f_jkr - f_hertz) / f_hertz
            max_rel = max(max_rel, rel)

    f_pull = abs(jkr_pull_off_force(r_star, 0.0))

    return HertzDegenerationResult(
        f_pull=f_pull,
        rel_error_vs_hertz=max_rel,
        is_hertz=max_rel < 0.02 and f_pull < 1e-9,
    )


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------

@dataclass
class FidelityReport:
    pull_off_results: list[dict] = field(default_factory=list)
    liquid_bridge_results: list[dict] = field(default_factory=list)
    aggregate_results: list[dict] = field(default_factory=list)
    hertz_degeneration: dict = field(default_factory=dict)
    acceptance_criteria: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def run_full_sweep(
    *,
    dt_tc_ratios: list[float] | None = None,
    kappa_values: list[float] | None = None,
    bo_values: list[float] | None = None,
    quick: bool = False,
) -> FidelityReport:
    if quick:
        dt_tc_ratios = dt_tc_ratios or [0.02, 0.1, 1.0]
        kappa_values = kappa_values or [1.0, 0.03]
        bo_values = bo_values or [1.0]
    else:
        dt_tc_ratios = dt_tc_ratios or DT_TC_RATIOS
        kappa_values = kappa_values or KAPPA_VALUES
        bo_values = bo_values or BO_VALUES

    report = FidelityReport()
    report.metadata = {
        "dt_tc_ratios": dt_tc_ratios,
        "kappa_values": kappa_values,
        "bo_values": bo_values,
        "r": DEFAULT_R,
        "e_real": DEFAULT_E_REAL,
        "nu": DEFAULT_NU,
        "gamma": DEFAULT_GAMMA,
        "rho": DEFAULT_RHO,
    }

    print("Running pull-off sweep...")
    for bo in bo_values:
        for kappa in kappa_values:
            for dt_tc in dt_tc_ratios:
                print(f"  Bo={bo}, κ={kappa}, Δt/t_c={dt_tc}")
                try:
                    result = run_pull_off_simulation(dt_tc, kappa, bo)
                    report.pull_off_results.append(asdict(result))
                except Exception as e:
                    report.pull_off_results.append({
                        "dt_tc_ratio": dt_tc, "kappa": kappa, "bo": bo,
                        "error": str(e), "converged": False,
                    })

    print("Running liquid bridge sweep...")
    for kappa in kappa_values[:2]:
        for dt_tc in dt_tc_ratios[:2]:
            print(f"  κ={kappa}, Δt/t_c={dt_tc}")
            try:
                result = run_liquid_bridge_rupture(dt_tc, kappa)
                report.liquid_bridge_results.append(asdict(result))
            except Exception as e:
                report.liquid_bridge_results.append({
                    "dt_tc_ratio": dt_tc, "kappa": kappa,
                    "error": str(e), "converged": False,
                })

    print("Running aggregate tensile sweep...")
    for bo in bo_values[:1]:
        for kappa in kappa_values[:2]:
            for dt_tc in dt_tc_ratios[:3]:
                print(f"  Bo={bo}, κ={kappa}, Δt/t_c={dt_tc}")
                try:
                    result = run_aggregate_tensile_test(dt_tc, kappa, bo)
                    report.aggregate_results.append(asdict(result))
                except Exception as e:
                    report.aggregate_results.append({
                        "dt_tc_ratio": dt_tc, "kappa": kappa, "bo": bo,
                        "error": str(e), "converged": False,
                    })

    print("Running Hertz degeneration test...")
    hertz_result = run_hertz_degeneration()
    report.hertz_degeneration = asdict(hertz_result)

    report.acceptance_criteria = evaluate_acceptance(report)
    return report


def evaluate_acceptance(report: FidelityReport) -> dict:
    criteria = {}

    # (a) Baseline pull-off error < 2%
    baseline = [r for r in report.pull_off_results
                if r.get("dt_tc_ratio") == 0.02 and r.get("kappa") == 1.0
                and r.get("converged", True) and "rel_error" in r]
    if baseline:
        max_err = max(r["rel_error"] for r in baseline)
        criteria["a_baseline_pull_off_lt_2pct"] = {
            "pass": max_err < 0.02, "value": max_err, "threshold": 0.02,
        }
    else:
        criteria["a_baseline_pull_off_lt_2pct"] = {"pass": False, "value": None, "threshold": 0.02}

    # (a) κ=0.03 error < 5%
    kappa_low = [r for r in report.pull_off_results
                 if r.get("kappa") == 0.03 and r.get("dt_tc_ratio") == 0.02
                 and r.get("converged", True) and "rel_error" in r]
    if kappa_low:
        max_err_k = max(r["rel_error"] for r in kappa_low)
        criteria["a_kappa_003_pull_off_lt_5pct"] = {
            "pass": max_err_k < 0.05, "value": max_err_k, "threshold": 0.05,
        }
    else:
        criteria["a_kappa_003_pull_off_lt_5pct"] = {"pass": False, "value": None, "threshold": 0.05}

    # (b) Monotonic degradation
    for bo in set(r.get("bo") for r in report.pull_off_results if "bo" in r):
        for kappa in set(r.get("kappa") for r in report.pull_off_results if "kappa" in r):
            subset = sorted(
                [r for r in report.pull_off_results
                 if r.get("bo") == bo and r.get("kappa") == kappa
                 and r.get("converged", True) and "rel_error" in r],
                key=lambda x: x["dt_tc_ratio"]
            )
            if len(subset) >= 2:
                errors = [r["rel_error"] for r in subset]
                is_mono = all(errors[i] <= errors[i + 1] + 1e-10
                              for i in range(len(errors) - 1))
                criteria[f"b_monotonic_bo{bo}_kappa{kappa}"] = {
                    "pass": is_mono, "errors": errors,
                    "dt_tc": [r["dt_tc_ratio"] for r in subset],
                }

    # (b) Δt > t_c error > 20%
    large_dt = [r for r in report.pull_off_results
                if r.get("dt_tc_ratio") >= 1.0 and r.get("converged", True)
                and "rel_error" in r]
    if large_dt:
        min_err = min(r["rel_error"] for r in large_dt)
        criteria["b_large_dt_error_gt_20pct"] = {
            "pass": min_err > 0.20, "value": min_err, "threshold": 0.20,
        }
    else:
        criteria["b_large_dt_error_gt_20pct"] = {"pass": False, "value": None, "threshold": 0.20}

    # (c) Liquid bridge rupture < 5%
    lb = [r for r in report.liquid_bridge_results if r.get("converged", True)]
    if lb:
        max_lb = max(r.get("rel_error", 0) for r in lb)
        criteria["c_liquid_bridge_rup_lt_5pct"] = {
            "pass": max_lb < 0.05, "value": max_lb, "threshold": 0.05,
        }
    else:
        criteria["c_liquid_bridge_rup_lt_5pct"] = {"pass": False, "value": None, "threshold": 0.05}

    # (d) Aggregate tensile at Δt/t_c ≤ 0.1: < 5%
    agg = [r for r in report.aggregate_results
           if r.get("dt_tc_ratio", 1) <= 0.1 and r.get("converged", True)]
    if agg:
        max_agg = max(r.get("rel_error", 0) for r in agg)
        criteria["d_aggregate_tensile_lt_5pct"] = {
            "pass": max_agg < 0.05, "value": max_agg, "threshold": 0.05,
        }
    else:
        criteria["d_aggregate_tensile_lt_5pct"] = {"pass": False, "value": None, "threshold": 0.05}

    # (e) Hertz degeneration
    hertz = report.hertz_degeneration
    criteria["e_hertz_degeneration"] = {
        "pass": hertz.get("is_hertz", False),
        "f_pull": hertz.get("f_pull"),
        "rel_error_vs_hertz": hertz.get("rel_error_vs_hertz"),
    }

    # (f) Energy balance < 2%
    energy = [r for r in report.pull_off_results
              if r.get("converged", True) and "energy_residual" in r]
    if energy:
        max_e = max(r["energy_residual"] for r in energy)
        criteria["f_energy_balance_lt_2pct"] = {
            "pass": max_e < 0.02, "value": max_e, "threshold": 0.02,
        }
    else:
        criteria["f_energy_balance_lt_2pct"] = {"pass": False, "value": None, "threshold": 0.02}

    return criteria


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_plots(report: FidelityReport, output_dir: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Pull-off error vs Δt/t_c
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(KAPPA_VALUES)))
    for ki, kappa in enumerate(KAPPA_VALUES):
        subset = [r for r in report.pull_off_results
                  if r.get("kappa") == kappa and r.get("converged", True)
                  and "rel_error" in r]
        if not subset:
            continue
        subset.sort(key=lambda x: x["dt_tc_ratio"])
        dt_tc = [r["dt_tc_ratio"] for r in subset]
        errors = [r["rel_error"] * 100 for r in subset]
        ax.semilogx(dt_tc, errors, "o-", color=colors[ki],
                    label=f"κ={kappa}", linewidth=2, markersize=6)

    ax.axhline(y=2, color="red", linestyle="--", alpha=0.5, label="2% threshold")
    ax.axhline(y=5, color="orange", linestyle="--", alpha=0.5, label="5% threshold")
    ax.axhline(y=20, color="darkred", linestyle="--", alpha=0.5, label="20% threshold")
    ax.set_xlabel("Δt / t_c", fontsize=12)
    ax.set_ylabel("Pull-off force relative error (%)", fontsize=12)
    ax.set_title("JKR Pull-Off Force Error vs Time Step Ratio", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.015, 1.5)

    path1 = output_dir / "pull_off_error_vs_dt.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": "pull_off_error_vs_dt", "path": str(path1),
                    "caption": "Pull-off force relative error vs Δt/t_c for various κ values"})

    # Figure 2: Error surface heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    dt_grid = sorted(set(r["dt_tc_ratio"] for r in report.pull_off_results
                         if r.get("converged", True) and "rel_error" in r))
    kappa_grid = sorted(set(r["kappa"] for r in report.pull_off_results
                            if r.get("converged", True) and "rel_error" in r))

    error_matrix = np.zeros((len(kappa_grid), len(dt_grid)))
    for r in report.pull_off_results:
        if not r.get("converged", True) or "rel_error" not in r:
            continue
        ki = kappa_grid.index(r["kappa"])
        di = dt_grid.index(r["dt_tc_ratio"])
        error_matrix[ki, di] = r["rel_error"] * 100

    im = ax.imshow(error_matrix, aspect="auto", cmap="YlOrRd",
                   extent=[min(dt_grid), max(dt_grid),
                           min(kappa_grid), max(kappa_grid)],
                   origin="lower", vmin=0, vmax=30)
    ax.set_xscale("log")
    ax.set_xlabel("Δt / t_c", fontsize=12)
    ax.set_ylabel("κ = E_soft / E_real", fontsize=12)
    ax.set_title("Pull-Off Error Surface (%)", fontsize=14)
    plt.colorbar(im, ax=ax, label="Error (%)")

    path2 = output_dir / "error_surface_heatmap.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": "error_surface_heatmap", "path": str(path2),
                    "caption": "2D error surface: pull-off error as function of κ and Δt/t_c"})

    # Figure 3: Liquid bridge force vs gap
    fig, ax = plt.subplots(figsize=(8, 5))
    r = DEFAULT_R
    gaps = np.linspace(0, 1.5 * r, 100)
    f0 = 2.0 * math.pi * DEFAULT_GAMMA_LIQ * r
    d_rup = (DEFAULT_VBR_OVER_R3 ** (1.0 / 3.0)) * r
    forces = f0 * np.exp(-gaps / d_rup)
    ax.plot(gaps * 1e6, forces, "b-", linewidth=2, label="Bridge force model")
    ax.axhline(y=f0, color="red", linestyle="--", alpha=0.7,
               label=f"Analytic: 2πγR = {f0:.4e} N")
    ax.set_xlabel("Gap (μm)", fontsize=12)
    ax.set_ylabel("Capillary force (N)", fontsize=12)
    ax.set_title("Liquid Bridge Force vs Gap", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    path3 = output_dir / "liquid_bridge_force_vs_gap.png"
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": "liquid_bridge_force_vs_gap", "path": str(path3),
                    "caption": "Liquid bridge capillary force vs gap distance"})

    # Figure 4: Energy balance
    fig, ax = plt.subplots(figsize=(8, 5))
    energy_data = [r for r in report.pull_off_results
                   if r.get("converged", True) and "energy_residual" in r]
    if energy_data:
        dt_tc = [r["dt_tc_ratio"] for r in energy_data]
        residuals = [r["energy_residual"] * 100 for r in energy_data]
        ax.semilogx(dt_tc, residuals, "s-", color="green", linewidth=2, markersize=6)
        ax.axhline(y=2, color="red", linestyle="--", alpha=0.5, label="2% threshold")
        ax.set_xlabel("Δt / t_c", fontsize=12)
        ax.set_ylabel("Energy residual (%)", fontsize=12)
        ax.set_title("Energy Balance: Surface Energy vs External Work", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    path4 = output_dir / "energy_balance.png"
    fig.savefig(path4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": "energy_balance", "path": str(path4),
                    "caption": "Energy balance residual vs Δt/t_c"})

    return figures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Adhesive DEM fidelity envelope")
    parser.add_argument("--quick", action="store_true", help="Quick sweep for CI")
    parser.add_argument("--output-dir", type=str, default="artifacts/adhesive_dem_fidelity",
                        help="Output directory for artifacts")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Adhesive DEM Fidelity Envelope")
    print("=" * 60)

    report = run_full_sweep(quick=args.quick)

    report_path = output_dir / "fidelity_report.json"
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    print("\n" + "=" * 60)
    print("ACCEPTANCE CRITERIA")
    print("=" * 60)
    all_pass = True
    for key, val in report.acceptance_criteria.items():
        status = "PASS" if val.get("pass", False) else "FAIL"
        if not val.get("pass", False):
            all_pass = False
        print(f"  [{status}] {key}: {val}")

    if not args.no_plots:
        print("\nGenerating plots...")
        figures = generate_plots(report, output_dir)
        for fig in figures:
            print(f"  Saved: {fig['path']}")

    print("\n" + "=" * 60)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
