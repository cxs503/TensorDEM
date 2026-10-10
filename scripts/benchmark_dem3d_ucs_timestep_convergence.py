"""Quantify timestep sensitivity of DEM3D UCS load and fracture observables.

This is a numerical convergence diagnostic, not an ice-material calibration.
"""
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
from scripts.benchmark_dem3d_ucs import _run_once


def _first_fracture_time(rows: list[dict[str, Any]]) -> float | None:
    for row in rows:
        if int(row["broken_bonds"]) > 0:
            return float(row["time_s"])
    return None


def _relative_delta(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1.0e-30)


def run_convergence_campaign(
    output: Path,
    *,
    base_steps: int = 250,
    factors: tuple[float, ...] = (1.0, 0.5, 0.25),
    relative_tolerance: float = 0.10,
) -> dict[str, Any]:
    if isinstance(base_steps, bool) or not isinstance(base_steps, int) or base_steps < 1:
        raise ValueError("base_steps must be a positive integer")
    if not factors or any(not math.isfinite(f) or f <= 0.0 or f > 1.0 for f in factors):
        raise ValueError("dt factors must be finite and in (0, 1]")
    if len(set(factors)) != len(factors):
        raise ValueError("dt factors must be unique")
    if not math.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and nonnegative")
    factors = tuple(sorted((float(f) for f in factors), reverse=True))
    torch.set_num_threads(1)
    manifest = json.loads((ROOT / "benchmarks" / "dem3d" / "ucs_cases.json").read_text(encoding="utf-8"))
    case = next((item for item in manifest["cases"] if item["id"] == "ucs_dynamic_smoke"), None)
    if case is None:
        raise ValueError("manifest is missing ucs_dynamic_smoke")

    records: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    for factor in factors:
        steps = int(math.ceil(base_steps / factor))
        rows, signature, metrics = _run_once(case, steps, dt_factor=factor)
        repeat_rows, repeat_signature, repeat_metrics = _run_once(case, steps, dt_factor=factor)
        finite = all(math.isfinite(float(v)) for row in rows for v in row.values())
        first_fracture = _first_fracture_time(rows)
        repeatable = (
            signature == repeat_signature
            and metrics["peak_engineering_stress_Pa"] == repeat_metrics["peak_engineering_stress_Pa"]
            and metrics["final_broken_bonds"] == repeat_metrics["final_broken_bonds"]
            and first_fracture == _first_fracture_time(repeat_rows)
        )
        records.append({
            "dt_factor": factor,
            "steps": steps,
            "simulated_duration_s": float(rows[-1]["time_s"]),
            "peak_load_N": float(metrics["peak_load_N"]),
            "peak_stress_Pa": float(metrics["peak_engineering_stress_Pa"]),
            "strain_at_peak": float(metrics["strain_at_peak_stress"]),
            "final_broken_bonds": int(metrics["final_broken_bonds"]),
            "first_fracture_time_s": first_fracture,
            "finite": finite,
            "repeatable": repeatable,
            "signature_sha256": signature,
        })
        for row in rows:
            histories.append({"dt_factor": factor, **row})

    fine = records[-1]
    for row in records:
        row["peak_load_relative_delta_vs_fine"] = _relative_delta(row["peak_load_N"], fine["peak_load_N"])
        row["peak_stress_relative_delta_vs_fine"] = _relative_delta(row["peak_stress_Pa"], fine["peak_stress_Pa"])
        row["strain_at_peak_absolute_delta_vs_fine"] = abs(row["strain_at_peak"] - fine["strain_at_peak"])
        t, tf = row["first_fracture_time_s"], fine["first_fracture_time_s"]
        row["fracture_time_relative_delta_vs_fine"] = (
            None if t is None or tf is None else _relative_delta(t, tf)
        )
        row["within_peak_stress_tolerance"] = row["peak_stress_relative_delta_vs_fine"] <= relative_tolerance
        row["convergence_status"] = (
            "WARN" if not row["within_peak_stress_tolerance"] else "PASS"
        )

    hard_pass = all(row["finite"] and row["repeatable"] for row in records)
    report = {
        "protocol": "tensordem-dem3d-ucs-timestep-convergence-v1",
        "verdict": "PASS" if hard_pass else "FAIL",
        "convergence_status": "WARN" if any(r["convergence_status"] == "WARN" for r in records) else "PASS",
        "base_steps_at_dt_factor_1": base_steps,
        "relative_tolerance": relative_tolerance,
        "reference": "finest timestep in this campaign",
        "acceptance": [
            "all history values are finite",
            "each timestep case is deterministic on repeated CPU runs",
        ],
        "interpretation": (
            "Peak-stress deltas are compared with the finest timestep in this campaign. "
            "The tolerance is a user-configurable screening threshold, not a universal physical standard. "
            "WARN means the selected numerical tolerance was not met; it does not imply solver failure. "
            "Fracture-time deltas are undefined when either run has no fracture event."
        ),
        "cases": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "ucs_timestep_convergence_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "ucs_timestep_convergence_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output / "ucs_timestep_convergence_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dt_factor", *histories[0].keys()][1:])
        writer.writeheader()
        writer.writerows(histories)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-ucs-timestep"))
    parser.add_argument("--base-steps", type=int, default=250)
    parser.add_argument("--relative-tolerance", type=float, default=0.10)
    args = parser.parse_args(argv)
    report = run_convergence_campaign(
        args.output, base_steps=args.base_steps, relative_tolerance=args.relative_tolerance
    )
    print(json.dumps({k: report[k] for k in ("protocol", "verdict", "convergence_status", "cases")}, indent=2))
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
