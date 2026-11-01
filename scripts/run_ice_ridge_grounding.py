"""Beaufort-sea ice-ridge grounding on a buried pipeline: full artifact runner.

Runs the coupled DEM / subgrade / beam workflow documented in
``docs/ice_ridge_pipeline_grounding.md`` and writes machine-readable artifacts
to ``artifacts/ridge_grounding/``:

* ``main_case.json``      -- load-displacement record, metrics, DNV utilisation
* ``rate_study.json``     -- loading-rate sensitivity of the peak load
* ``sensitivity.json``    -- keel depth / keel width / soil strength trends
* ``calibration.json``    -- DEM micro -> macro strength calibration attempt
* ``beam_verification.json`` -- Winkler beam vs closed-form Hetenyi
* ``ledger.json``         -- machine-checkable criterion ledger

Everything is programmatic: no number in the report is copied by hand.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.ridge_dem import (RidgeKeel, RidgeMaterial,  # noqa: E402
                                calibrate_rubble_strength)
from tensordem.ridge_geotech import (MohrCoulombSoil, PipeSection,  # noqa: E402
                                     SofteningVerticalSpring,
                                     hetenyi_verification)
from tensordem.ridge_grounding import (GroundingScenario,  # noqa: E402
                                       capacity_envelope, continuum_limit_capacity,
                                       load_displacement_metrics, pipe_stress_profile,
                                       run_grounding)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "artifacts", "ridge_grounding")

# --------------------------------------------------------------------------- deck
BEAUFORT_SOIL = MohrCoulombSoil(
    friction_angle_deg=0.0, cohesion=30.0e3, saturated_unit_weight=18.0e3,
    young_modulus=10.0e6, poisson_ratio=0.3, water_unit_weight=10.06e3, undrained=True)
BEAUFORT_SECTION = PipeSection(outer_diameter=0.914, wall_thickness=0.0159,
                               young_modulus=207.0e9, yield_strength=448.0e6)
BEAUFORT_MATERIAL = RidgeMaterial(
    particle_radius=0.5, drag=2.0e4, freeze_bond_fraction=1.0,
    bond_tensile_strength=100.0e3, bond_shear_strength=200.0e3)


def _scenario(keel_depth=12.0, keel_width=24.0, *, velocity=0.15, stroke=2.0,
              soil=None, material=None, cover_depth=0.5, tip_clearance=0.15,
              response="compliant") -> GroundingScenario:
    return GroundingScenario(
        keel=RidgeKeel(keel_depth, keel_width, 0.10),
        material=material or BEAUFORT_MATERIAL,
        soil=soil or BEAUFORT_SOIL,
        section=BEAUFORT_SECTION,
        cover_depth=cover_depth, tip_clearance=tip_clearance, stroke=stroke,
        platen_velocity=velocity, pipe_response=response, contact_length=24.0)


def _dump(name: str, payload) -> None:
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(f"wrote {path}", flush=True)


def _history_csv(result, name: str) -> None:
    keys = ["time", "stroke", "keel_penetration", "pipe_load", "pipe_load_smoothed",
            "pipe_load_soil", "pipe_displacement", "drive_force", "soil_force",
            "kinetic_energy", "contacts"]
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        rows = min(len(result.history[key]) for key in keys)
        for index in range(rows):
            handle.write(",".join(str(float(result.history[key][index])) for key in keys) + "\n")
    print(f"wrote {path}", flush=True)


def stage_main() -> dict:
    result = run_grounding(_scenario())
    metrics = load_displacement_metrics(result)
    profile = pipe_stress_profile(result.capacity * result.scenario.contact_length, result.scenario)
    spring = result.info.get("soil_spring", {})
    payload = {
        "scenario": {
            "keel": {"depth": result.scenario.keel.keel_depth,
                     "width": result.scenario.keel.keel_width,
                     "porosity": result.scenario.keel.porosity,
                     "flank_angle_deg": result.scenario.keel.angle_deg},
            "material": {key: getattr(result.scenario.material, key) for key in
                         ("particle_radius", "ice_density", "water_density", "contact_stiffness",
                          "contact_damping", "friction_coefficient", "bond_stiffness",
                          "bond_shear_stiffness", "bond_tensile_strength",
                          "bond_shear_strength", "freeze_bond_fraction", "drag")},
            "bond_failure_strain": result.scenario.material.bond_failure_strain,
            "soil": {key: getattr(result.scenario.soil, key) for key in
                     ("friction_angle_deg", "cohesion", "saturated_unit_weight",
                      "young_modulus", "poisson_ratio", "water_unit_weight",
                      "pore_pressure_ratio", "undrained")},
            "section": {"outer_diameter": BEAUFORT_SECTION.outer_diameter,
                        "wall_thickness": BEAUFORT_SECTION.wall_thickness,
                        "young_modulus": BEAUFORT_SECTION.young_modulus,
                        "yield_strength": BEAUFORT_SECTION.yield_strength,
                        "inertia": BEAUFORT_SECTION.inertia,
                        "plastic_moment": BEAUFORT_SECTION.plastic_moment},
            "cover_depth": result.scenario.cover_depth,
            "tip_clearance": result.scenario.tip_clearance,
            "stroke": result.scenario.stroke,
            "platen_velocity": result.scenario.platen_velocity,
            "contact_length": result.scenario.contact_length,
            "pipe_line_mass": result.scenario.pipe_line_mass,
            "pipe_response": result.scenario.pipe_response,
        },
        "numerics": {"dt": result.dt, "steps": result.steps,
                     "particles": result.info.get("particles"),
                     "bonds_initial": result.info.get("bonds"),
                     "bonds_final": result.info.get("bonds_final"),
                     "broken_bonds": result.info.get("broken_bonds"),
                     "smoothing_window": result.info.get("smoothing_window"),
                     "dt_rule": "0.2*min(sqrt(m/k_eff), m/(sum(c)+drag))"},
        "soil_spring": spring,
        "capacity_per_length": result.capacity,
        "metrics": metrics,
        "history": {key: [float(v) for v in result.history[key]] for key in
                    ("time", "stroke", "keel_penetration", "pipe_load",
                     "pipe_load_smoothed", "pipe_load_soil", "pipe_displacement",
                     "drive_force", "soil_force", "kinetic_energy", "contacts")},
        "pipe_profile": {key: profile[key] for key in
                         ("peak_moment", "peak_moment_reference", "moment_relative_error",
                          "peak_deflection", "peak_deflection_reference",
                          "deflection_relative_error", "peak_stress", "peak_utilization",
                          "beta", "subgrade_stiffness", "element_length", "converged")},
        "pipe_curves": {key: profile[key] for key in ("x", "moment", "stress", "deflection")},
        "quasi_static": {
            "kinetic_energy_max": max(result.history["kinetic_energy"]),
            "platen_work_estimate": abs(max(result.history["drive_force"], key=abs))
            * result.scenario.platen_velocity * result.history["time"][-1],
            "note": "ratio of kinetic to external work quantifies the inertial error",
        },
    }
    total = abs(payload["quasi_static"]["platen_work_estimate"])
    payload["quasi_static"]["kinetic_to_work_ratio"] = (
        payload["quasi_static"]["kinetic_energy_max"] / total if total else math.inf)
    _dump("main_case.json", payload)
    _history_csv(result, "main_case_history.csv")
    return payload


def stage_rate() -> dict:
    rows = []
    for velocity in (0.60, 0.30, 0.15):
        result = run_grounding(_scenario(velocity=velocity, stroke=1.4))
        metrics = load_displacement_metrics(result)
        rows.append({"platen_velocity": velocity,
                     "capacity_per_length": result.capacity,
                     "peak_from_metrics": metrics["peak_load_per_length"],
                     "steps": result.steps,
                     "broken_bonds": result.info.get("broken_bonds"),
                     "kinetic_to_work_ratio": payload_ratio(result)})
        print(f"rate {velocity}: {result.capacity:.0f} N/m", flush=True)
    payload = {"rows": rows}
    _dump("rate_study.json", payload)
    return payload


def payload_ratio(result) -> float:
    work = abs(max(result.history["drive_force"], key=abs)) \
        * result.scenario.platen_velocity * result.history["time"][-1]
    return max(result.history["kinetic_energy"]) / work if work else math.inf


def stage_sensitivity() -> dict:
    depths = [8.0, 12.0, 16.0]
    widths = [16.0, 24.0, 32.0]
    depth_rows = []
    for depth in depths:
        result = run_grounding(_scenario(keel_depth=depth, keel_width=24.0,
                                        velocity=0.40, stroke=1.2))
        depth_rows.append({"keel_depth": depth, "capacity_per_length": result.capacity,
                           "particles": result.info.get("particles")})
        print(f"depth {depth}: {result.capacity:.0f}", flush=True)
    width_rows = []
    for width in widths:
        result = run_grounding(_scenario(keel_depth=12.0, keel_width=width,
                                        velocity=0.40, stroke=1.2))
        width_rows.append({"keel_width": width, "capacity_per_length": result.capacity,
                           "particles": result.info.get("particles")})
        print(f"width {width}: {result.capacity:.0f}", flush=True)
    soil_rows = []
    for strength in (10.0e3, 30.0e3, 60.0e3):
        soil = MohrCoulombSoil(friction_angle_deg=0.0, cohesion=strength,
                               saturated_unit_weight=18.0e3, young_modulus=10.0e6,
                               water_unit_weight=10.06e3, undrained=True)
        spring = SofteningVerticalSpring(soil, BEAUFORT_SECTION.outer_diameter, 0.5)
        soil_rows.append({"undrained_shear_strength": strength,
                          "soil_peak_capacity": spring.peak,
                          "soil_ultimate_capacity": spring.ultimate,
                          "soil_residual_capacity": spring.residual})
    payload = {"keel_depth": depth_rows, "keel_width": width_rows, "soil_strength": soil_rows}
    _dump("sensitivity.json", payload)
    return payload


def stage_calibration() -> dict:
    result = calibrate_rubble_strength(BEAUFORT_MATERIAL, columns=8, rows=8,
                                       loading_rate=0.2, strain_target=0.08,
                                       max_steps=4000)
    _dump("calibration.json", result)
    return result


def stage_beam() -> dict:
    payload = {"point_load": hetenyi_verification(),
               "patch_load": hetenyi_verification(patch_half_length=4.0, load=2.0e5)}
    _dump("beam_verification.json", payload)
    return payload


def stage_ledger(main, rate, sensitivity, calibration, beam) -> dict:
    metrics = main["metrics"]
    soil_peak = main["soil_spring"]["peak_reaction"]
    soil_residual = main["soil_spring"]["residual_reaction"]
    drive_pressure = max(main["history"]["drive_force"]) / main["scenario"]["contact_length"]
    diameter = main["scenario"]["section"]["outer_diameter"]
    brackets = []
    for phi in (20.0, 30.0, 40.0):
        for cohesion in (20.0e3, 60.0e3, 100.0e3):
            brackets.append({
                "friction_angle_deg": phi, "cohesion": cohesion,
                "limit_capacity_per_length": continuum_limit_capacity(
                    cohesion, phi, surcharge=drive_pressure, diameter=diameter)})
    low = min(row["limit_capacity_per_length"] for row in brackets)
    high = max(row["limit_capacity_per_length"] for row in brackets)
    dem_ice_capacity = main["capacity_per_length"]
    inside = low <= dem_ice_capacity <= high
    mid = 0.5 * (low + high)
    deviation = abs(dem_ice_capacity - mid) / mid
    ledger = {
        "case": "Beaufort-like grounded ridge keel over a buried 36 in pipeline",
        "criteria": {
            "a_curve_shape": {
                "evidence": "REAL",
                "metrics": metrics,
                "passed": bool(metrics["passed"]),
                "note": "peak-then-softening signature of the coupled load-displacement "
                        "record; softening mechanism is rubble crushing plus the "
                        "peak-to-residual subgrade branch, not a pure ice-indentation test",
            },
            "b_capacity_vs_limit_analysis": {
                "evidence": "DERIVED (Vesic limit equilibrium) + WITHHELD (Cross/DNV "
                            "empirical reference text not available in the worktree)",
                "dem_capacity_per_length": dem_ice_capacity,
                "drive_pressure": drive_pressure,
                "bracket_low": low, "bracket_high": high, "bracket_mid": mid,
                "inside_bracket": inside,
                "relative_deviation_from_mid": deviation,
                "deviation_below_20_percent": deviation < 0.20,
                "calibrated_cohesion": calibration.get("cohesion"),
                "calibrated_friction_angle_deg": calibration.get("friction_angle_deg"),
                "calibration_usable": bool(
                    max(s["confining_relative_error"] for s in calibration["specimens"]) < 0.10),
            },
            "c_pipe_stress_vs_winkler": {
                "evidence": "REAL",
                "beam_point_load": beam["point_load"],
                "beam_patch_load": beam["patch_load"],
                "pipeline_moment_relative_error": main["pipe_profile"]["moment_relative_error"],
                "pipeline_deflection_relative_error":
                    main["pipe_profile"]["deflection_relative_error"],
                "deviation_below_15_percent": bool(
                    main["pipe_profile"]["moment_relative_error"] < 0.15
                    and beam["point_load"]["moment_relative_error"] < 0.15),
            },
            "d_sensitivity": {
                "evidence": "REAL",
                "keel_depth_trend": [row["capacity_per_length"] for row in
                                     sensitivity["keel_depth"]],
                "keel_width_trend": [row["capacity_per_length"] for row in
                                     sensitivity["keel_width"]],
                "soil_strength_trend": [row["soil_ultimate_capacity"] for row in
                                        sensitivity["soil_strength"]],
                "expected": "capacity increases with keel depth and width; the soil-limited "
                            "capacity scales linearly with undrained shear strength",
            },
            "e_pytest": {"evidence": "REAL", "suite": "tests/test_ridge_grounding.py"},
            "f_full_case": {
                "evidence": "REAL",
                "artifacts": ["artifacts/ridge_grounding/main_case.json",
                              "artifacts/ridge_grounding/main_case_history.csv",
                              "artifacts/ridge_grounding/figures/load_displacement.png",
                              "artifacts/ridge_grounding/figures/pipe_stress.png",
                              "artifacts/ridge_grounding/figures/dem_field.png"],
            },
        },
        "rate_sensitivity": rate["rows"],
        "rate_note": "capacity decreases with decreasing platen velocity; the slowest "
                     "affordable rate is the reported value and the report states the "
                     "linear extrapolation to zero rate",
        "soil_limits": {"peak": soil_peak, "residual": soil_residual,
                        "ultimate": main["soil_spring"]["ultimate_bearing_capacity"]},
    }
    _dump("ledger.json", ledger)
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", default="beam,main,rate,sensitivity,calibration",
                        help="comma separated subset of beam,main,rate,sensitivity,calibration")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(1)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    beam = stage_beam() if "beam" in stages else {}
    main_payload = stage_main() if "main" in stages else {}
    rate = stage_rate() if "rate" in stages else {"rows": []}
    sensitivity = stage_sensitivity() if "sensitivity" in stages else {
        "keel_depth": [], "keel_width": [], "soil_strength": []}
    calibration = stage_calibration() if "calibration" in stages else {}
    if main_payload and sensitivity["keel_depth"]:
        stage_ledger(main_payload, rate, sensitivity, calibration, beam)
    envelope = capacity_envelope(
        [row["keel_depth"] for row in sensitivity["keel_depth"]],
        [row["capacity_per_length"] for row in sensitivity["keel_depth"]],
        soil=BEAUFORT_SOIL, section=BEAUFORT_SECTION, cover_depth=0.5,
        contact_length=24.0)
    _dump("envelope.json", envelope)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
