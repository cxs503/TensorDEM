"""Ridge-keel grounding vs the ISO 19906 form: DEM case runner.

Each case is one coupled DEM run and is addressed by an index, so a study can
be sharded over several machines::

    python scripts/run_ridge_iso_grounding.py list
    python scripts/run_ridge_iso_grounding.py run --index 3
    python scripts/run_ridge_iso_grounding.py merge

``merge`` collects the per-case JSON files into the study tables declared in
``docs/ice_ridge_iso19906_grounding.md`` and applies the campaign criteria
mechanically (no hand-copied numbers anywhere).

Studies
-------
``cover``   cover-depth sweep d/D in {1, 2, 4, 8}: compliant pipe (DEM capacity,
            settlement) and rigid pipe (ice-side demand) at each cover.
``lateral`` lateral loading mode with a lateral pipe degree of freedom.
``grid``    mesh independence: particle radius halved (particle count doubled).
``energy``  energy-ledger closure against the time step.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from tensordem.ridge_dem import RidgeKeel, RidgeMaterial  # noqa: E402
from tensordem.ridge_geotech import MohrCoulombSoil, PipeSection  # noqa: E402
from tensordem.ridge_grounding import (GroundingScenario,  # noqa: E402
                                       load_displacement_metrics, run_grounding)
from tensordem.ridge_iso19906 import (compare_with_dem, cover_envelope,  # noqa: E402
                                      critical_cover_depth,
                                      energy_convergence_table,
                                      grid_convergence_table,
                                      instability_limits, iso19906_lateral_load,
                                      iso19906_vertical_load)

ARTIFACTS = os.path.join(HERE, "..", "artifacts", "ridge_iso_grounding")
CASES = os.path.join(ARTIFACTS, "cases")

KEEL = RidgeKeel(keel_depth=12.0, keel_width=24.0, porosity=0.10)
SECTION = PipeSection(outer_diameter=0.914, wall_thickness=0.0159,
                      young_modulus=207.0e9, yield_strength=448.0e6)
BASE_SOIL = MohrCoulombSoil(friction_angle_deg=0.0, cohesion=30.0e3,
                            saturated_unit_weight=18.0e3, young_modulus=10.0e6,
                            poisson_ratio=0.3, water_unit_weight=10.06e3,
                            undrained=True)
DRAINED_SOIL = MohrCoulombSoil(friction_angle_deg=30.0, cohesion=5.0e3,
                               saturated_unit_weight=20.0e3, young_modulus=30.0e6,
                               poisson_ratio=0.3, water_unit_weight=10.06e3,
                               undrained=False)


def material(radius: float = 0.5) -> RidgeMaterial:
    return RidgeMaterial(particle_radius=radius, drag=2.0e4,
                         freeze_bond_fraction=1.0, bond_tensile_strength=100.0e3,
                         bond_shear_strength=200.0e3)


def tip_depth(cover_over_d: float) -> float:
    """Gouge depth that puts the keel tip 0.3 m below the pipe crown."""
    return cover_over_d * SECTION.outer_diameter + 0.3


def case_table() -> list[dict]:
    """Declarative list of every DEM case in the node."""
    cases: list[dict] = []
    coverage = (1.0, 2.0, 4.0, 8.0)
    for index, ratio in enumerate(coverage):
        cover = ratio * SECTION.outer_diameter
        for response in ("compliant", "rigid"):
            cases.append({
                "study": "cover", "index": len(cases), "cover_over_d": ratio,
                "cover_depth": cover, "pipe_response": response,
                "loading": "vertical", "radius": 0.5, "velocity": 0.3,
                "soil": "undrained", "track_energy": False,
                "name": f"cover_d{ratio:g}_{response}",
            })
    for ratio in (1.0, 2.0):
        cover = ratio * SECTION.outer_diameter
        for response in ("compliant", "rigid"):
            cases.append({
                "study": "cover", "index": len(cases), "cover_over_d": ratio,
                "cover_depth": cover, "pipe_response": response,
                "loading": "vertical", "radius": 0.5, "velocity": 0.3,
                "soil": "drained", "track_energy": False,
                "name": f"cover_d{ratio:g}_{response}_drained",
            })
    for ratio in (1.0, 2.0, 4.0):
        cover = ratio * SECTION.outer_diameter
        for soil_name in ("undrained", "drained"):
            for response, compliance in (("rigid", False), ("compliant", True)):
                cases.append({
                    "study": "lateral", "index": len(cases),
                    "cover_over_d": ratio, "cover_depth": cover,
                    "pipe_response": "rigid", "loading": "lateral",
                    "radius": 0.5, "velocity": 0.30, "soil": soil_name,
                    "track_energy": False,
                    "lateral_compliance": compliance,
                    "penetration": tip_depth(ratio),
                    "name": f"lateral_d{ratio:g}_{soil_name}_{response}",
                })
    # 0.354 is the radius at which the particle count doubles (2-D: area ~ r^2).
    # 0.25 (4x the particles) is documented as unresolved: the run aborts with a
    # double-free inside the torch index_add path after ~1e5 steps.
    for radius in (0.5, 0.354):
        cases.append({
            "study": "grid", "index": len(cases), "cover_over_d": 1.0,
            "cover_depth": SECTION.outer_diameter, "pipe_response": "compliant",
            "loading": "vertical", "radius": radius, "velocity": 0.15,
            "soil": "undrained", "track_energy": False,
            "name": f"grid_r{radius:g}",
        })
    for factor in (1.0, 0.5, 0.25):
        cases.append({
            "study": "energy", "index": len(cases), "cover_over_d": 0.5,
            "cover_depth": 0.5 * SECTION.outer_diameter,
            "pipe_response": "compliant", "loading": "vertical", "radius": 0.5,
            "velocity": 0.15, "soil": "undrained", "track_energy": True,
            "dt_factor": factor, "name": f"energy_dt{factor:g}",
        })
    return cases


def _soil(name: str) -> MohrCoulombSoil:
    return BASE_SOIL if name == "undrained" else DRAINED_SOIL


def build_scenario(case: dict) -> GroundingScenario:
    soil = _soil(case["soil"])
    cover = case["cover_depth"]
    diameter = SECTION.outer_diameter
    tip_clearance = 0.10
    # lateral cases start with the keel already grounded: a ridge that has not
    # gouged down to the pipe crown cannot load it sideways
    penetration = case.get("penetration", 0.0)
    stroke = tip_clearance - penetration + cover + 1.2 * diameter
    mat = material(case["radius"])
    dt = mat.recommended_dt * case.get("dt_factor", 1.0)
    return GroundingScenario(
        keel=KEEL, material=mat, soil=soil, section=SECTION,
        cover_depth=cover, tip_clearance=tip_clearance, stroke=stroke,
        platen_velocity=case["velocity"], loading=case["loading"],
        pipe_response=case["pipe_response"],
        pipe_lateral_compliance=bool(case.get("lateral_compliance", False)),
        initial_penetration=penetration,
        dt=dt, sample_every=20, track_energy=bool(case["track_energy"]),
        contact_length=30.0)


def run_case(case: dict) -> dict:
    scenario = build_scenario(case)
    started = time.time()
    result = run_grounding(scenario, max_steps=400000)
    elapsed = time.time() - started
    metrics = load_displacement_metrics(result)
    soil = scenario.soil
    diameter = SECTION.outer_diameter
    iso_vertical = iso19906_vertical_load(soil, diameter,
                                          cover_depth=case["cover_depth"])
    iso_keel = iso19906_vertical_load(soil, diameter,
                                      cover_depth=case["cover_depth"],
                                      keel_depth=KEEL.keel_depth,
                                      surcharge_mode="keel")
    iso_lateral = iso19906_lateral_load(soil, diameter,
                                       cover_depth=case["cover_depth"])
    record = {
        "name": case["name"], "study": case["study"], "case": case,
        "elapsed_s": elapsed, "steps": result.steps, "dt": result.dt,
        "capacity_n_per_m": result.capacity,
        "peak_raw_n_per_m": result.info["pipe_load_peak_raw"],
        "particles": result.info["particles"],
        "broken_bonds": result.info["broken_bonds"],
        "final_pipe_displacement_m": result.final_pipe_displacement,
        "lateral": result.info["lateral"],
        "metrics": metrics,
        "iso": {"vertical": iso_vertical.as_dict(),
                "vertical_total_n_per_m": iso_vertical.total,
                "vertical_keel_mode_n_per_m": iso_keel.total,
                "lateral": iso_lateral},
        "loads": {"pipe_load": result.history["pipe_load"],
                  "pipe_load_horizontal": result.history["pipe_load_horizontal"],
                  "pipe_displacement": result.history["pipe_displacement"],
                  "pipe_displacement_lateral":
                      result.history["pipe_displacement_lateral"],
                  "stroke": result.history["stroke"],
                  "kinetic_energy": result.history["kinetic_energy"],
                  "broken_bonds": result.history["broken_bonds"]},
        "energy": result.info.get("energy"),
    }
    if result.capacity:
        record["comparison"] = compare_with_dem(result.capacity,
                                                iso_vertical.total)
    return record


def _path(name: str) -> str:
    return os.path.join(CASES, f"{name}.json")


def merge() -> dict:
    cases = case_table()
    records = {}
    for case in cases:
        path = _path(case["name"])
        if os.path.exists(path):
            with open(path) as handle:
                records[case["name"]] = json.load(handle)
    limits = instability_limits(SECTION)
    diameter = SECTION.outer_diameter

    cover_rows = []
    for ratio in (1.0, 2.0, 4.0, 8.0):
        compliant = records.get(f"cover_d{ratio:g}_compliant")
        rigid = records.get(f"cover_d{ratio:g}_rigid")
        if not compliant or not rigid:
            continue
        lateral = records.get(f"lateral_d{ratio:g}_undrained_compliant")
        cover_rows.append({
            "cover_depth": ratio * diameter,
            "dem_capacity": compliant["capacity_n_per_m"],
            "iso_vertical": compliant["iso"]["vertical_total_n_per_m"],
            "dem_demand": rigid["capacity_n_per_m"],
            "dem_displacement": compliant["final_pipe_displacement_m"],
            "iso_lateral": compliant["iso"]["lateral"]["total_n_per_m"],
            "dem_lateral": (lateral["lateral"]["peak_load"] if lateral else 0.0),
            "dem_lateral_displacement": (abs(lateral["lateral"]["final_displacement"])
                                         if lateral else 0.0),
        })
    envelope = cover_envelope(cover_rows, diameter=diameter,
                              vertical_limit=limits["vertical"],
                              lateral_limit=limits["lateral"])
    demand_reference = max((row["dem_demand"] for row in cover_rows), default=0.0)
    cover_critical = (critical_cover_depth(BASE_SOIL, diameter, demand_reference)
                      if demand_reference else 0.0)

    lateral_rows = []
    for ratio in (1.0, 2.0, 4.0):
        for soil_name in ("undrained", "drained"):
            rigid = records.get(f"lateral_d{ratio:g}_{soil_name}_rigid")
            compliant = records.get(f"lateral_d{ratio:g}_{soil_name}_compliant")
            if not rigid or not compliant:
                continue
            iso = rigid["iso"]["lateral"]["total_n_per_m"]
            peak = rigid["lateral"]["peak_load"]
            sustained = rigid["lateral"]["sustained_load"]
            lateral_rows.append({
                "cover_over_d": ratio, "soil": soil_name,
                "dem_lateral_peak_n_per_m": peak,
                "dem_lateral_sustained_n_per_m": sustained,
                "iso_lateral_n_per_m": iso,
                "peak_ratio": peak / iso if iso else float("nan"),
                "peak_deviation": (peak - iso) / iso if iso else float("nan"),
                "sustained_deviation": (sustained - iso) / iso if iso else float("nan"),
                "final_displacement_m": compliant["lateral"]["final_displacement"],
                "regime": compliant["lateral"]["regime"],
            })

    grid_rows = []
    for radius in (0.5, 0.354):
        record = records.get(f"grid_r{radius:g}")
        if record:
            grid_rows.append({"particle_radius": radius,
                              "particles": record["particles"],
                              "capacity": record["capacity_n_per_m"]})
    grid = grid_convergence_table(grid_rows)

    energy_rows = []
    for factor in (1.0, 0.5, 0.25):
        record = records.get(f"energy_dt{factor:g}")
        if record and record.get("energy"):
            energy = record["energy"]
            energy_rows.append({"dt": record["dt"],
                                "steps": record["steps"],
                                "closure_error": energy["closure_error"],
                                "residual": energy["residual"],
                                "sources": energy["sources"]})
    energy = energy_convergence_table(energy_rows)

    # drained cases exercise the N_q overburden term of the campaign form (the
    # undrained branch is cohesion-only); both land in criterion (a).
    drained_rows = []
    for ratio in (1.0, 2.0):
        record = records.get(f"cover_d{ratio:g}_compliant_drained")
        if not record:
            continue
        cover = ratio * diameter
        form = iso19906_vertical_load(DRAINED_SOIL, diameter, cover_depth=cover)
        full = iso19906_vertical_load(DRAINED_SOIL, diameter, cover_depth=cover,
                                      include_gamma_term=True)
        drained_rows.append({
            "cover_over_d": ratio,
            "dem_capacity_n_per_m": record["capacity_n_per_m"],
            "iso_vertical_n_per_m": form.total,
            "iso_vertical_with_gamma_n_per_m": full.total,
            "capacity_deviation": (record["capacity_n_per_m"] / form.total - 1.0)
            if form.total else None,
            "capacity_deviation_vs_full": (record["capacity_n_per_m"] / full.total - 1.0)
            if full.total else None,
            "n_q": form.n_q, "n_c": form.n_c,
        })
    engaged = [row for row in envelope["rows"]
               if row["dem_capacity_n_per_m"] > 0.0]
    for row in drained_rows:
        if row["dem_capacity_n_per_m"] > 0.0 and row["capacity_deviation"] is not None:
            engaged.append({"capacity_deviation": row["capacity_deviation"]})
    max_deviation = (max(abs(row["capacity_deviation"]) for row in engaged)
                     if engaged else None)
    criteria = {
        "a_vertical_vs_iso": {
            "tolerance": 0.20,
            "engaged_cases": len(engaged),
            "non_contact_cases": len(envelope["rows"]) - len(engaged),
            "max_abs_deviation": max_deviation,
            "satisfied": bool(max_deviation is not None and max_deviation < 0.20),
            "dem_below_iso_all": all(row["dem_capacity_n_per_m"]
                                     <= row["iso_vertical_n_per_m"]
                                     for row in envelope["rows"]),
        },
        "b_lateral_vs_iso": {
            "tolerance": 0.25,
            "peak_deviations": [row["peak_deviation"] for row in lateral_rows],
            "sustained_deviations": [row["sustained_deviation"]
                                     for row in lateral_rows],
            # the sustained (quasi-static) drag is the load the pipe carries; the
            # peak is the dynamic rubble impact, compared with the same capacity
            # only to answer "can the cover resist it".
            "satisfied": bool(lateral_rows and all(
                abs(row["sustained_deviation"]) < 0.25 for row in lateral_rows)),
        },
        "c_cover_instability": {
            "vertical_limit_m": limits["vertical"],
            "lateral_limit_m": limits["lateral"],
            "consistent_predictions": envelope["consistent_predictions"],
            "cases": envelope["cases"],
            "satisfied": envelope["verdict"] == "consistent",
            "critical_cover_depth_m": cover_critical,
            "critical_cover_over_d": cover_critical / diameter,
        },
        "d_energy_closure": {
            "tolerance": 0.05,
            "finest_closure": energy["finest_closure"],
            "refinement_ratios": energy["refinement_ratios"],
            "satisfied": energy["satisfied"],
        },
        "e_grid_independence": {
            "tolerance": 0.05,
            "particle_ratio": grid["particle_ratio"],
            "capacity_change": grid["capacity_change"],
            "satisfied": grid["satisfied"],
        },
    }
    merged = {
        "cases": len(records),
        "section": {"outer_diameter_m": SECTION.outer_diameter,
                    "wall_thickness_m": SECTION.wall_thickness,
                    "young_modulus_pa": SECTION.young_modulus,
                    "yield_strength_pa": SECTION.yield_strength},
        "keel": {"keel_depth_m": KEEL.keel_depth, "keel_width_m": KEEL.keel_width,
                 "porosity": KEEL.porosity},
        "soil": {"undrained": {"cohesion_pa": BASE_SOIL.cohesion,
                               "friction_angle_deg": 0.0,
                               "saturated_unit_weight": BASE_SOIL.saturated_unit_weight,
                               "young_modulus_pa": BASE_SOIL.young_modulus},
                 "drained": {"cohesion_pa": DRAINED_SOIL.cohesion,
                             "friction_angle_deg": DRAINED_SOIL.friction_angle_deg,
                             "saturated_unit_weight": DRAINED_SOIL.saturated_unit_weight,
                             "young_modulus_pa": DRAINED_SOIL.young_modulus}},
        "cover_envelope": envelope,
        "drained_rows": drained_rows,
        "lateral": {"rows": lateral_rows},
        "grid": grid,
        "energy": energy,
        "criteria": criteria,
        "records_present": sorted(records),
    }
    with open(os.path.join(ARTIFACTS, "ledger.json"), "w") as handle:
        json.dump(merged, handle, indent=2)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "run", "merge", "all"))
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--study", type=str, default=None)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(CASES, exist_ok=True)
    cases = case_table()
    if args.action == "list":
        for case in cases:
            print(case["index"], case["name"], case["study"])
        return
    if args.action == "merge":
        merged = merge()
        print(json.dumps(merged["criteria"], indent=2))
        return
    selected = cases
    if args.index is not None:
        selected = [cases[args.index]]
    if args.study is not None:
        selected = [case for case in selected if case["study"] == args.study]
    for case in selected:
        record = run_case(case)
        with open(_path(case["name"]), "w") as handle:
            json.dump(record, handle)
        print(f"[{case['index']:2d}] {case['name']:28s} "
              f"capacity={record['capacity_n_per_m']/1e3:8.2f} kN/m "
              f"disp={record['final_pipe_displacement_m']:.3f} m "
              f"steps={record['steps']:6d} {record['elapsed_s']:6.1f}s",
              flush=True)
    if args.action == "all":
        merge()


if __name__ == "__main__":
    main()