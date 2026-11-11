#!/usr/bin/env python3
"""CASE-PAIR-COLLISION: Two-sphere Hertz impact restitution coefficient benchmark.

Physical problem
================
Two identical elastic spheres approach head-on with equal and opposite velocities.
The contact follows the classical Hertz law:

    F_n = (4/3) E* sqrt(R*) delta^(3/2)

with a Tsuji-type nonlinear dashpot for dissipation:

    gamma_n(delta) = 2 zeta sqrt(m_eff k_s(delta))

where k_s(delta) = (4/3) E* sqrt(R*) sqrt(delta) is the secant stiffness.

Analytical references
=====================
1. Elastic limit (e=1): peak overlap
   delta_max = (15 m_eff v_rel^2 / (16 E* sqrt(R*)))^(2/5)
   [Goldsmith, "Impact", 1960, Edward Arnold, Eq. 5.6, p.152]

2. Elastic limit: contact duration
   t_c = 2.94 * (m_eff^2 / (E*^2 R* v_rel))^(1/5)
   [Goldsmith, "Impact", 1960, Eq. 5.8, p.153]

3. Damped case: restitution coefficient
   For the Tsuji damper, e should be velocity-independent.
   The mapping zeta(e) uses linear spring-dashpot theory:
   zeta = |ln e| / sqrt(pi^2 + ln^2 e)
   [Tsuji, Tanaka & Ishida, Powder Technology 71 (1992) 239-246]

Usage:
    PYTHONPATH=src python benchmarks/case_pair_collision_hertz/run.py [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from tensordem.dem3d import (
    DEM3DConfig,
    IceDEM3D,
    hertz_effective_radius,
    hertz_reduced_modulus,
)

# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------
RADIUS = 0.025          # m, sphere radius
DENSITY = 917.0         # kg/m^3, ice density
YOUNG_MODULUS = 1.0e9   # Pa, ice Young's modulus
POISSON_RATIO = 0.3     # dimensionless
PARK = 1.0              # m, parking distance for unused particles

PRIMARY_SPEED = 0.5     # m/s, primary benchmark speed
SPEEDS_SWEEP = [0.25, 0.5, 1.0]  # m/s, for restitution velocity-independence check
TARGET_E = 0.8          # target restitution coefficient
DT_FACTOR = 0.4         # timestep as fraction of recommended_dt


# ---------------------------------------------------------------------------
# Analytical reference functions (independent of solver)
# ---------------------------------------------------------------------------
def analytic_peak_overlap(speed: float, radius: float, density: float,
                          young_modulus: float, poisson_ratio: float) -> float:
    """Peak overlap for elastic equal-sphere Hertz impact.

    Reference: Goldsmith, "Impact: The Theory and Physical Behaviour of Colliding Solids",
    1960, Edward Arnold, Eq. 5.6, p.152.
    """
    v_rel = 2.0 * speed
    mass = density * (4.0 / 3.0) * math.pi * radius ** 3
    m_eff = mass / 2.0
    e_star = hertz_reduced_modulus(young_modulus, poisson_ratio)
    r_eff = hertz_effective_radius(radius)
    return ((15.0 * m_eff * v_rel ** 2) / (16.0 * e_star * math.sqrt(r_eff))) ** 0.4


def analytic_contact_time(speed: float, radius: float, density: float,
                          young_modulus: float, poisson_ratio: float) -> float:
    """Contact duration for elastic equal-sphere Hertz impact.

    Reference: Goldsmith, "Impact", 1960, Eq. 5.8, p.153.
    """
    v_rel = 2.0 * speed
    mass = density * (4.0 / 3.0) * math.pi * radius ** 3
    m_eff = mass / 2.0
    e_star = hertz_reduced_modulus(young_modulus, poisson_ratio)
    r_eff = hertz_effective_radius(radius)
    return 2.94 * (m_eff ** 2 / (e_star ** 2 * r_eff * v_rel)) ** 0.2


def reference_rk4(speed: float, rest_target: float, dt: float = 1.0e-8,
                  max_time: float = 0.01,
                  radius: float = RADIUS, density: float = DENSITY,
                  young_modulus: float = YOUNG_MODULUS,
                  poisson_ratio: float = POISSON_RATIO) -> dict:
    """Independent RK4 integration of the relative-coordinate Hertz contact law.

    Uses the correct Hertz force law: F = (4/3) E* sqrt(R*) delta^(3/2).
    Initial closing speed = 2 * speed (two spheres approaching).
    """
    mass = density * (4.0 / 3.0) * math.pi * radius ** 3
    m_eff = mass / 2.0
    e_star = hertz_reduced_modulus(young_modulus, poisson_ratio)
    r_eff = hertz_effective_radius(radius)
    k_coef = (4.0 / 3.0) * e_star * math.sqrt(r_eff)

    v_rel = 2.0 * speed
    ln_e = math.log(rest_target) if rest_target < 1.0 else 0.0
    zeta = abs(ln_e) / math.sqrt(math.pi ** 2 + ln_e ** 2) if rest_target < 1.0 else 0.0

    def accel(overlap: float, rate: float) -> float:
        if overlap <= 0.0:
            return 0.0
        elastic = k_coef * overlap ** 1.5
        if rest_target < 1.0:
            gamma = 2.0 * zeta * math.sqrt(m_eff * k_coef * math.sqrt(overlap))
            force = elastic + gamma * rate
        else:
            force = elastic
        return -force / m_eff if force > 0.0 else 0.0

    overlap, rate = 0.0, v_rel
    peak_overlap = 0.0
    t = 0.0
    in_contact = False
    while t < max_time:
        k1d = rate
        k1r = accel(overlap, rate)
        k2d = rate + 0.5 * dt * k1r
        k2r = accel(overlap + 0.5 * dt * k1d, k2d)
        k3d = rate + 0.5 * dt * k2r
        k3r = accel(overlap + 0.5 * dt * k2d, k3d)
        k4d = rate + dt * k3r
        k4r = accel(overlap + dt * k3d, k4d)
        overlap += dt * (k1d + 2 * k2d + 2 * k3d + k4d) / 6.0
        rate += dt * (k1r + 2 * k2r + 2 * k3r + k4r) / 6.0
        peak_overlap = max(peak_overlap, overlap)
        if overlap > 0.0:
            in_contact = True
        elif in_contact:
            break
        t += dt
    return {
        "restitution": -rate / v_rel,
        "peak_overlap_m": peak_overlap,
        "contact_time_s": t,
        "zeta": zeta,
    }


# ---------------------------------------------------------------------------
# DEM engine measurement
# ---------------------------------------------------------------------------
def pair_config(dt_factor: float = DT_FACTOR, **kwargs) -> DEM3DConfig:
    """2x2x2 lattice container for one free contact pair."""
    params = dict(
        nx=2, ny=2, nz=2,
        radius=RADIUS, density=DENSITY,
        contact_model="hertz", young_modulus=YOUNG_MODULUS, poisson_ratio=POISSON_RATIO,
        contact_damping=0.0, drag=0.0,
        fix_bottom=False, fix_x_edges=False,
        tool_speed=1.0, tool_gap=10.0,
    )
    params.update(kwargs)
    base = DEM3DConfig(**params)
    return DEM3DConfig(**{**base.__dict__, "dt": dt_factor * base.recommended_dt})


def build_pair(speed: float, cfg: DEM3DConfig, gap_fraction: float = 0.05) -> IceDEM3D:
    """Two free spheres approaching head-on along z."""
    sim = IceDEM3D(cfg)
    sim.alive[:] = False
    sim.positions[0] = torch.tensor([0.0, 0.0, 0.0], dtype=sim.dtype)
    sim.positions[1] = torch.tensor([0.0, 0.0, 2.0 * RADIUS * (1.0 + gap_fraction)],
                                    dtype=sim.dtype)
    for k in range(2, len(sim.positions)):
        sim.positions[k] = torch.tensor([PARK * (k - 1), PARK * (k - 1), 3.0 * PARK],
                                        dtype=sim.dtype)
    sim.velocities.zero_()
    sim.velocities[0, 2] = +speed
    sim.velocities[1, 2] = -speed
    return sim


def _pair_state(sim: IceDEM3D) -> tuple:
    """(separation, closing velocity) of the pair (0, 1)."""
    r = sim.positions[1] - sim.positions[0]
    separation = float(torch.linalg.vector_norm(r))
    normal = r / separation
    closing = -float((sim.velocities[1] - sim.velocities[0]) @ normal)
    return separation, closing


def measure_engine(speed: float, dt_factor: float = DT_FACTOR,
                   max_steps: int = 500_000, **kwargs) -> dict:
    """Integrate one impact; return measured restitution and diagnostics."""
    cfg = pair_config(dt_factor=dt_factor, **kwargs)
    sim = build_pair(speed, cfg)
    separation0, approach = _pair_state(sim)
    min_separation = separation0
    in_contact = False
    contact_start_step = None
    rebounded = None
    rebound_step = None
    steps = 0
    t0 = time.perf_counter()
    while steps < max_steps:
        sim.step()
        steps += 1
        separation, closing = _pair_state(sim)
        min_separation = min(min_separation, separation)
        if separation < 2.0 * RADIUS:
            if not in_contact:
                contact_start_step = steps
            in_contact = True
        elif in_contact:
            rebounded = abs(closing)
            rebound_step = steps
            break
    elapsed = time.perf_counter() - t0
    if rebounded is None:
        raise RuntimeError(f"impact did not separate within {max_steps} steps at speed={speed}")

    contact_steps = (rebound_step - contact_start_step) if contact_start_step else steps
    contact_time = contact_steps * float(cfg.dt)

    return {
        "speed_in_m_per_s": approach,
        "speed_out_m_per_s": rebounded,
        "restitution": rebounded / approach,
        "peak_overlap_m": 2.0 * RADIUS - min_separation,
        "contact_time_s": contact_time,
        "contact_start_step": contact_start_step,
        "rebound_step": rebound_step,
        "steps": steps,
        "dt_s": float(cfg.dt),
        "wall_time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_primary_elastic() -> dict:
    """Primary elastic case (e=1, v=0.5 m/s): compare vs analytic."""
    speed = PRIMARY_SPEED
    got = measure_engine(speed, restitution_coefficient=1.0)
    ref_delta = analytic_peak_overlap(speed, RADIUS, DENSITY, YOUNG_MODULUS, POISSON_RATIO)
    ref_tc = analytic_contact_time(speed, RADIUS, DENSITY, YOUNG_MODULUS, POISSON_RATIO)
    err_delta = abs(got["peak_overlap_m"] - ref_delta) / ref_delta
    err_tc = abs(got["contact_time_s"] - ref_tc) / ref_tc
    return {
        "speed_m_per_s": speed,
        "peak_overlap_m": got["peak_overlap_m"],
        "peak_overlap_ref_m": ref_delta,
        "peak_overlap_rel_error": err_delta,
        "contact_time_s": got["contact_time_s"],
        "contact_time_ref_s": ref_tc,
        "contact_time_rel_error": err_tc,
        "restitution": got["restitution"],
        "steps": got["steps"],
        "dt_s": got["dt_s"],
        "wall_time_s": got["wall_time_s"],
    }


def run_primary_damped() -> dict:
    """Primary damped case (e_target=0.8, v=0.5 m/s): compare engine vs RK4."""
    speed = PRIMARY_SPEED
    got = measure_engine(speed, restitution_coefficient=TARGET_E)
    ref = reference_rk4(speed, TARGET_E)
    err_e = abs(got["restitution"] - ref["restitution"]) / ref["restitution"]
    err_delta = abs(got["peak_overlap_m"] - ref["peak_overlap_m"]) / ref["peak_overlap_m"]
    return {
        "speed_m_per_s": speed,
        "restitution_engine": got["restitution"],
        "restitution_rk4": ref["restitution"],
        "restitution_rel_error": err_e,
        "peak_overlap_engine_m": got["peak_overlap_m"],
        "peak_overlap_rk4_m": ref["peak_overlap_m"],
        "peak_overlap_rel_error": err_delta,
        "contact_time_engine_s": got["contact_time_s"],
        "contact_time_rk4_s": ref["contact_time_s"],
        "steps": got["steps"],
        "dt_s": got["dt_s"],
        "wall_time_s": got["wall_time_s"],
    }


def run_velocity_sweep() -> list[dict]:
    """Restitution velocity-independence check (e_target=0.8)."""
    rows = []
    for speed in SPEEDS_SWEEP:
        got = measure_engine(speed, restitution_coefficient=TARGET_E)
        ref = reference_rk4(speed, TARGET_E)
        err_e = abs(got["restitution"] - ref["restitution"]) / ref["restitution"]
        rows.append({
            "speed_m_per_s": speed,
            "restitution_engine": got["restitution"],
            "restitution_rk4": ref["restitution"],
            "restitution_rel_error": err_e,
            "steps": got["steps"],
            "wall_time_s": got["wall_time_s"],
        })
    return rows


def run_convergence_study() -> list[dict]:
    """Timestep convergence: dt_factor = 0.8, 0.4, 0.2 at v=0.5 m/s, e=1."""
    speed = PRIMARY_SPEED
    rows = []
    for dt_factor in [0.8, 0.4, 0.2]:
        got = measure_engine(speed, dt_factor=dt_factor, restitution_coefficient=1.0)
        ref_delta = analytic_peak_overlap(speed, RADIUS, DENSITY, YOUNG_MODULUS, POISSON_RATIO)
        ref_tc = analytic_contact_time(speed, RADIUS, DENSITY, YOUNG_MODULUS, POISSON_RATIO)
        err_delta = abs(got["peak_overlap_m"] - ref_delta) / ref_delta
        err_tc = abs(got["contact_time_s"] - ref_tc) / ref_tc
        rows.append({
            "dt_factor": dt_factor,
            "dt_s": got["dt_s"],
            "steps": got["steps"],
            "peak_overlap_m": got["peak_overlap_m"],
            "peak_overlap_ref_m": ref_delta,
            "peak_overlap_rel_error": err_delta,
            "contact_time_s": got["contact_time_s"],
            "contact_time_ref_s": ref_tc,
            "contact_time_rel_error": err_tc,
            "restitution": got["restitution"],
            "wall_time_s": got["wall_time_s"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="CASE-PAIR-COLLISION: Hertz two-sphere impact")
    parser.add_argument("--output-dir", type=str, default="benchmarks/case_pair_collision_hertz")
    args = parser.parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CASE-PAIR-COLLISION: Two-sphere Hertz impact benchmark")
    print("=" * 70)
    print(f"Radius={RADIUS} m, Density={DENSITY} kg/m^3")
    print(f"E={YOUNG_MODULUS} Pa, nu={POISSON_RATIO}")
    print(f"Primary speed: {PRIMARY_SPEED} m/s")
    print(f"Target restitution: {TARGET_E}")
    print()

    # 1. Primary elastic case
    print("--- Primary elastic case (e=1, v=0.5 m/s) ---")
    elastic = run_primary_elastic()
    print(f"  peak_overlap: {elastic['peak_overlap_m']:.6e} m (ref: {elastic['peak_overlap_ref_m']:.6e} m)")
    print(f"  peak_overlap error: {elastic['peak_overlap_rel_error']:.4%}")
    print(f"  contact_time: {elastic['contact_time_s']:.6e} s (ref: {elastic['contact_time_ref_s']:.6e} s)")
    print(f"  contact_time error: {elastic['contact_time_rel_error']:.4%}")
    print(f"  restitution: {elastic['restitution']:.6f}")
    print(f"  wall_time: {elastic['wall_time_s']:.2f} s")

    # 2. Primary damped case
    print("\n--- Primary damped case (e_target=0.8, v=0.5 m/s) ---")
    damped = run_primary_damped()
    print(f"  e_engine: {damped['restitution_engine']:.6f}")
    print(f"  e_rk4: {damped['restitution_rk4']:.6f}")
    print(f"  restitution error: {damped['restitution_rel_error']:.4%}")
    print(f"  peak_overlap error: {damped['peak_overlap_rel_error']:.4%}")
    print(f"  wall_time: {damped['wall_time_s']:.2f} s")

    # 3. Velocity sweep
    print("\n--- Velocity sweep (e_target=0.8) ---")
    sweep = run_velocity_sweep()
    for r in sweep:
        print(f"  v={r['speed_m_per_s']} m/s: e_engine={r['restitution_engine']:.4f}, "
              f"e_rk4={r['restitution_rk4']:.4f}, err={r['restitution_rel_error']:.4%}")

    # 4. Convergence study
    print("\n--- Timestep convergence (v=0.5 m/s, e=1) ---")
    conv = run_convergence_study()
    for r in conv:
        print(f"  dt_factor={r['dt_factor']}: dt={r['dt_s']:.2e}, steps={r['steps']}, "
              f"delta_err={r['peak_overlap_rel_error']:.4%}, tc_err={r['contact_time_rel_error']:.4%}")

    # Write results
    results = {
        "case": "case_pair_collision_hertz",
        "description": "Two-sphere Hertz impact: restitution coefficient benchmark",
        "parameters": {
            "radius_m": RADIUS,
            "density_kg_m3": DENSITY,
            "young_modulus_Pa": YOUNG_MODULUS,
            "poisson_ratio": POISSON_RATIO,
            "primary_speed_m_per_s": PRIMARY_SPEED,
            "speeds_sweep_m_per_s": SPEEDS_SWEEP,
            "target_restitution": TARGET_E,
            "dt_factor": DT_FACTOR,
        },
        "dimensionless_groups": {
            "Hertz_dimensionless_overlap": "delta/R ~ (rho v^2 / E)^(2/5)",
            "damping_ratio_zeta": "zeta = |ln e| / sqrt(pi^2 + ln^2 e)",
        },
        "primary_elastic": elastic,
        "primary_damped": damped,
        "velocity_sweep": sweep,
        "convergence_study": conv,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {results_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    delta_err = elastic["peak_overlap_rel_error"]
    tc_err = elastic["contact_time_rel_error"]
    rest_err = damped["restitution_rel_error"]
    max_sweep_err = max(r["restitution_rel_error"] for r in sweep)
    print(f"Peak overlap error (elastic): {delta_err:.4%}")
    print(f"Contact time error (elastic): {tc_err:.4%}")
    print(f"Restitution error (damped, vs RK4): {rest_err:.4%}")
    print(f"Max restitution error (velocity sweep): {max_sweep_err:.4%}")
    print(f"\nPass (<3%): peak_overlap={'PASS' if delta_err < 0.03 else 'FAIL'}, "
          f"contact_time={'PASS' if tc_err < 0.03 else 'FAIL'}, "
          f"restitution={'PASS' if rest_err < 0.03 else 'FAIL'}")

    return results


if __name__ == "__main__":
    main()
