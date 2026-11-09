#!/usr/bin/env python
"""Multi-year ice ridge keel interaction campaign (TensorDEM).

Closed chain
------------
1. ridge keel geometry statistics (:mod:`tensordem.myi_ridge.geometry`):
   keel-depth distribution, porosity, block-size distribution, plane-strain
   keel section reconstruction;
2. plane-strain DEM crushing of the bonded rubble by the hull ice belt
   (:mod:`tensordem.myi_ridge.crushing`) with a closed energy ledger;
3. extreme-load scaling law ``F_max = C sigma_ice^a h_keel^b`` and the load
   envelope (:mod:`tensordem.myi_ridge.scaling`);
4. ice-belt plate-panel structural check of the DEM pressure
   (:mod:`tensordem.myi_ridge.panel`).

Usage
-----
    python scripts/run_myi_ridge_campaign.py --case 3        # one sweep case
    python scripts/run_myi_ridge_campaign.py --all --jobs 8  # whole campaign
    python scripts/run_myi_ridge_campaign.py --aggregate     # fit + report data

Outputs land in ``runs/myi_ridge/`` (per-case JSON), ``artifacts/myi_ridge/``
(results JSON) and ``docs/myi-ridge-keel-interaction/`` (report + figures).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from tensordem.myi_ridge import crushing, geometry, panel, scaling  # noqa: E402

RUNS = ROOT / "runs" / "myi_ridge"
ARTIFACTS = ROOT / "artifacts" / "myi_ridge"
DOCS = ROOT / "docs" / "myi-ridge-keel-interaction"

BASE = {
    "sigma_ice": 7.5e6,
    "h_keel": 20.0,
    "hull_velocity": 2.0,
    "porosity": 0.30,
}

#: one-factor-at-a-time sweep around the base point (controlled comparison:
#: every non-varying parameter, the packing seed and the numerical settings are
#: held fixed so a difference between cases is attributable to the swept factor)
SWEEP = []
for sigma in (5.0e6, 7.5e6, 10.0e6):
    SWEEP.append({**BASE, "sigma_ice": sigma, "label": f"sigma_{sigma/1e6:.2f}"})
for depth in (10.0, 30.0):
    SWEEP.append({**BASE, "h_keel": depth, "label": f"depth_{depth:.0f}"})
for speed in (1.0, 2.5, 5.0):
    SWEEP.append({**BASE, "hull_velocity": speed, "label": f"speed_{speed:.2f}"})
SWEEP.append({**BASE, "porosity": 0.40, "label": "porosity_0.40"})

LOADED_LENGTH = 1.5     # loaded length of hull in ridge contact [m]
ADVANCE = 0.8           # plate advance per run [m]
LENGTH_RATIO = 1.2      # specimen length / height
SEED = 20261108
RADIUS = 0.5            # DEM disc radius [m]
BOND_FRACTION = 0.10    # freeze-bond cohesion / sigma_ice (0.75 MPa at 7.5 MPa)
SAIL_RATIO = 0.22


def case_path(label: str) -> Path:
    return RUNS / f"{label}.json"


def run_case(index: int) -> dict:
    case = SWEEP[index]
    label = case["label"]
    height = float(case["h_keel"])
    specimen = crushing.RidgeSpecimen(height=height, length=LENGTH_RATIO * height,
                                     porosity=float(case["porosity"]),
                                     particle_radius=RADIUS, seed=SEED)
    params = crushing.CrushParameters(ice_strength=float(case["sigma_ice"]),
                                      bond_strength_fraction=BOND_FRACTION,
                                      hull_velocity=float(case["hull_velocity"]),
                                      advance=ADVANCE,
                                      sail_height=SAIL_RATIO * height)
    dem = crushing.RidgeCrushDem(specimen, params)
    run = dem.run()
    trace = np.asarray(run["force_trace"]["mean_force"], dtype=float)
    record = {
        "label": label,
        "sigma_ice": float(case["sigma_ice"]),
        "h_keel": height,
        "hull_velocity": float(case["hull_velocity"]),
        "porosity": float(case["porosity"]),
        "seed": SEED,
        "particles": run["particles"],
        "bonds": run["bonds_initial"],
        "porosity_measured": run["porosity_measured"],
        "steps": run["steps"],
        "dt": run["dt"],
        "advance": run["advance"],
        "force_peak": run["force_peak"],
        "force_steady": run["force_steady"],
        "pressure_peak": run["pressure_peak"],
        "pressure_steady": run["pressure_steady"],
        "pressure_ratio_peak": run["pressure_peak"] / float(case["sigma_ice"]),
        "pressure_ratio_steady": run["pressure_steady"] / float(case["sigma_ice"]),
        "ledger": run["ledger"],
        "force_trace_advance": run["force_trace"]["advance"],
        "force_trace_force": [float(v) for v in trace],
        "F_max_peak": scaling.extreme_load(run["pressure_peak"], height, LOADED_LENGTH),
        "F_max_steady": scaling.extreme_load(run["pressure_steady"], height, LOADED_LENGTH),
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    case_path(label).write_text(json.dumps(record, indent=2))
    return record


def _case_worker(index: int) -> str:
    record = run_case(index)
    return (f"{record['label']}: p_peak={record['pressure_peak']/1e6:.2f} MPa "
            f"p_steady={record['pressure_steady']/1e6:.2f} MPa "
            f"closure={record['ledger']['closure_of_exchanged']:.4f}")


def run_all(jobs: int) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    pending = [index for index in range(len(SWEEP))
               if not case_path(SWEEP[index]["label"]).exists()]
    if not pending:
        print("all cases already present", flush=True)
        return
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for message in pool.map(_case_worker, pending):
            print(message, flush=True)


def load_cases() -> list[dict]:
    records = []
    for index in range(len(SWEEP)):
        path = case_path(SWEEP[index]["label"])
        if not path.exists():
            raise SystemExit(f"missing case {path}; run --all first")
        records.append(json.loads(path.read_text()))
    return records


def aggregate() -> dict:
    records = load_cases()
    for record in records:
        record["closure_of_exchanged"] = record["ledger"]["closure_of_exchanged"]
    envelope_peak = scaling.load_envelope(
        [{**record, "pressure_steady": record["pressure_peak"]} for record in records],
        loaded_length=LOADED_LENGTH)
    envelope_steady = scaling.load_envelope(
        [{**record, "pressure_steady": record["pressure_steady"]} for record in records],
        loaded_length=LOADED_LENGTH)

    # geometry end of the chain
    stats = geometry.MyiRidgeStatistics()
    ensemble = geometry.ensemble_statistics(stats, 400, seed=SEED)
    sections = []
    for depth in (10.0, 20.0, 30.0):
        for porosity in (0.20, 0.30, 0.40):
            section = geometry.build_keel_section(depth, porosity, stats, seed=SEED)
            sections.append(geometry.section_statistics(section, stats))
    geometry_summary = {
        "ensemble": ensemble,
        "sections": sections,
        "max_porosity_error": max(abs(s["porosity_error"]) for s in sections),
        "max_depth_error": max(abs(s["keel_depth_error"]) for s in sections),
        "mean_keel_depth_error": abs(ensemble["keel_depth_mean_error"]),
        "mean_block_size_error": abs(ensemble["block_diameter_mean_error"]),
    }

    # structural end of the chain: design pressure -> ice belt panel
    design_pressure = max(record["pressure_steady"] for record in records)
    frame_spacings = [2.4, 3.0, 3.6]
    panel_checks = {}
    for spacing in frame_spacings:
        check = panel.check_ice_belt(design_pressure, span_x=spacing, span_y=spacing,
                                    thickness=0.030, elements_x=12, elements_y=12)
        panel_checks[f"{spacing:.1f}m"] = check
    required = panel.required_thickness(2.4, 2.4, design_pressure,
                                       allowable_stress=panel.STEEL_EH36["yield_strength"] / 1.15,
                                       reference_thickness=0.030, elements_x=12, elements_y=12)
    verification = {}
    for thickness in (0.030, 0.040):
        reference = panel.solve_panel(1.0, 1.0, thickness, 1.0e4, elements_x=12,
                                      elements_y=12, boundary="clamped")
        verification[f"clamped_{thickness*1000:.0f}mm"] = reference.as_dict()
        reference = panel.solve_panel(1.0, 1.0, thickness, 1.0e4, elements_x=12,
                                      elements_y=12, boundary="simply_supported")
        verification[f"simple_{thickness*1000:.0f}mm"] = reference.as_dict()

    result = {
        "cases": records,
        "base": BASE,
        "loaded_length": LOADED_LENGTH,
        "advance": ADVANCE,
        "bond_strength_fraction": BOND_FRACTION,
        "envelope_peak": envelope_peak,
        "envelope_steady": envelope_steady,
        "geometry": geometry_summary,
        "design_pressure": design_pressure,
        "panel_checks": panel_checks,
        "required_thickness_2p4m": required,
        "panel_verification": verification,
        "energy_closure_max": max(r["ledger"]["closure_of_exchanged"] for r in records),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "myi_ridge_results.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 4))
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.case is not None:
        print(_case_worker(args.case))
    if args.all:
        run_all(args.jobs)
    if args.aggregate:
        result = aggregate()
        print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2)[:4000])


if __name__ == "__main__":
    main()