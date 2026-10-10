"""Run repeatable speed, wedge-angle and timestep sensitivity campaigns for bow resistance."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch
from scripts.benchmark_dem3d_moving_wedge_bow import _run_once


def _first_fracture(rows: list[dict[str, Any]]) -> float | None:
    for row in rows:
        if int(row["broken_bonds"]) > 0:
            return float(row["time_s"])
    return None


def _relative_delta(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-30)


def run_sensitivity_campaign(
    output: Path,
    *,
    base_steps: int = 300,
    speeds_m_s: tuple[float, ...] = (0.05, 0.1, 0.2),
    angles_deg: tuple[float, ...] = (30.0, 45.0, 60.0),
    dt_factors: tuple[float, ...] = (1.0, 0.5, 0.25),
) -> dict[str, Any]:
    if isinstance(base_steps, bool) or not isinstance(base_steps, int) or base_steps < 1:
        raise ValueError("base_steps must be a positive integer")
    for name, values, lower, upper in (
        ("speeds_m_s", speeds_m_s, 0.0, math.inf),
        ("angles_deg", angles_deg, 5.0, 85.0),
        ("dt_factors", dt_factors, 0.0, 1.0),
    ):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be nonempty and unique")
        if any(not math.isfinite(v) or v <= lower or v > upper for v in values):
            raise ValueError(f"{name} contains an out-of-range value")
    if any(v >= 1.0 for v in dt_factors):
        raise ValueError("dt_factors must be in (0, 1]; use a factor of 1.0 for the nominal timestep")
    torch.set_num_threads(1)

    definitions: list[dict[str, Any]] = []
    for angle in angles_deg:
        definitions.append({"sweep": "wedge_angle", "value": float(angle), "bow_speed_m_s": 0.1,
                            "wedge_angle_deg": float(angle), "dt_factor": 1.0, "steps": base_steps})
    for speed in speeds_m_s:
        definitions.append({"sweep": "bow_speed", "value": float(speed), "bow_speed_m_s": float(speed),
                            "wedge_angle_deg": 45.0, "dt_factor": 1.0, "steps": base_steps})
    for factor in dt_factors:
        steps = int(math.ceil(base_steps / factor))
        definitions.append({"sweep": "timestep", "value": float(factor), "bow_speed_m_s": 0.1,
                            "wedge_angle_deg": 45.0, "dt_factor": float(factor), "steps": steps})

    records: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for index, spec in enumerate(definitions):
        rows, signature, metrics = _run_once(
            int(spec["steps"]), bow_speed=float(spec["bow_speed_m_s"]),
            wedge_angle_deg=float(spec["wedge_angle_deg"]), dt_factor=float(spec["dt_factor"]),
        )
        repeat_rows, repeat_signature, repeat_metrics = _run_once(
            int(spec["steps"]), bow_speed=float(spec["bow_speed_m_s"]),
            wedge_angle_deg=float(spec["wedge_angle_deg"]), dt_factor=float(spec["dt_factor"]),
        )
        finite = all(math.isfinite(float(value)) for row in rows for value in row.values())
        first_fracture = _first_fracture(rows)
        repeatable = signature == repeat_signature and metrics == repeat_metrics
        record = {
            "case_id": f"{spec['sweep']}_{index + 1:02d}",
            **spec,
            "dt_s": float(rows[1]["time_s"] - rows[0]["time_s"]),
            "simulated_duration_s": float(rows[-1]["time_s"]),
            "peak_ice_resistance_N": float(metrics["peak_ice_resistance_N"]),
            "mean_ice_resistance_N": float(metrics["mean_ice_resistance_N"]),
            "integrated_resistance_work_J": float(metrics["integrated_resistance_work_J"]),
            "final_broken_bonds": int(metrics["final_broken_bonds"]),
            "broken_bond_fraction": float(metrics["broken_bond_fraction"]),
            "first_fracture_time_s": first_fracture,
            "bow_travel_m": float(metrics["bow_travel_m"]),
            "finite": finite,
            "repeatable": repeatable,
            "signature_sha256": signature,
        }
        records.append(record)
        for row in rows:
            history.append({"case_id": record["case_id"], "sweep": spec["sweep"],
                            "parameter_value": spec["value"], **row})

    # Compare each one-factor sweep with its named reference, not across
    # physically different experiments.
    for record in records:
        if record["sweep"] == "wedge_angle":
            reference = next(r for r in records if r["sweep"] == "wedge_angle" and r["value"] == 45.0)
        elif record["sweep"] == "bow_speed":
            reference = next(r for r in records if r["sweep"] == "bow_speed" and r["value"] == 0.1)
        else:
            reference = next(r for r in records if r["sweep"] == "timestep" and r["value"] == min(dt_factors))
        record["reference_value"] = reference["value"]
        record["peak_resistance_relative_delta_vs_reference"] = _relative_delta(
            record["peak_ice_resistance_N"], reference["peak_ice_resistance_N"]
        )
        record["work_relative_delta_vs_reference"] = _relative_delta(
            record["integrated_resistance_work_J"], reference["integrated_resistance_work_J"]
        )

    checks = {
        "all_histories_finite": all(r["finite"] for r in records),
        "all_cases_repeatable": all(r["repeatable"] for r in records),
        "all_bows_advance": all(float(r["bow_travel_m"]) > 0.0 for r in records),
        "nonnegative_resistance_work": all(float(r["integrated_resistance_work_J"]) >= 0.0 for r in records),
    }
    report = {
        "protocol": "tensordem-dem3d-wedge-resistance-sensitivity-v1",
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "base_steps": base_steps,
        "case_count": len(records),
        "sweep_design": {
            "wedge_angle_deg": list(angles_deg),
            "bow_speed_m_s": list(speeds_m_s),
            "dt_factors": list(dt_factors),
            "duration_policy": "Timestep cases scale step count by inverse dt factor to approximately preserve physical duration.",
        },
        "checks": checks,
        "cases": records,
        "interpretation": (
            "This is a numerical sensitivity screen for an idealized dry wedge/contact model. "
            "Relative deltas quantify sensitivity and are not accuracy errors against experiments. "
            "Timestep cases approximately preserve physical duration; speed and angle cases use the same step count. "
            "No fluid, hydrostatic, buoyancy, finite-width hull, or calibrated material response is included."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "wedge_resistance_sensitivity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "wedge_resistance_sensitivity_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output / "wedge_resistance_sensitivity_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-wedge-sensitivity"))
    parser.add_argument("--base-steps", type=int, default=300)
    args = parser.parse_args(argv)
    report = run_sensitivity_campaign(args.output, base_steps=args.base_steps)
    print(json.dumps({k: report[k] for k in ("protocol", "verdict", "checks", "cases")}, indent=2))
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
