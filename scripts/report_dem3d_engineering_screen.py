"""Combine DEM3D numerical sensitivity reports into an auditable screening verdict.

This is a numerical QA gate, not experimental validation or certification.
Thresholds are explicit CLI inputs and are copied into the output report.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"report not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return payload


def _number(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field}: expected a numeric value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field}: value must be finite")
    if nonnegative and result < 0:
        raise ValueError(f"{field}: value must be nonnegative")
    return result


def _levels(report: dict[str, Any], field: str, path: Path) -> list[dict[str, Any]]:
    rows = report.get("levels")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError(f"{path}: '{field}' must contain at least two levels")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: each level must be a JSON object")
    return rows


def build_acceptance_report(
    timestep_path: Path,
    resolution_path: Path,
    cross_resolution_path: Path,
    *,
    max_timestep_peak_force_delta: float = 0.05,
    max_resolution_peak_force_delta: float = 0.10,
    max_cross_resolution_force_nrmse: float = 0.10,
    max_energy_residual_ratio: float = 0.05,
    final_time_relative_tolerance: float = 1e-8,
) -> dict[str, Any]:
    limits = {
        "max_timestep_peak_force_delta": max_timestep_peak_force_delta,
        "max_resolution_peak_force_delta": max_resolution_peak_force_delta,
        "max_cross_resolution_force_nrmse": max_cross_resolution_force_nrmse,
        "max_energy_residual_ratio": max_energy_residual_ratio,
        "final_time_relative_tolerance": final_time_relative_tolerance,
    }
    for name, value in limits.items():
        limits[name] = _number(value, name, nonnegative=True)
    for name in (
        "max_timestep_peak_force_delta", "max_resolution_peak_force_delta",
        "max_cross_resolution_force_nrmse",
    ):
        if limits[name] > 1:
            raise ValueError(f"{name} must be <= 1")
    if limits["final_time_relative_tolerance"] > 0.1:
        raise ValueError("final_time_relative_tolerance must be <= 0.1")

    timestep = _read_json(timestep_path)
    resolution = _read_json(resolution_path)
    cross = _read_json(cross_resolution_path)
    ts_levels = _levels(timestep, "levels", timestep_path)
    rs_levels = _levels(resolution, "levels", resolution_path)
    cs_levels = _levels(cross, "levels", cross_resolution_path)

    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, observed: Any, threshold: Any, detail: str) -> None:
        checks.append({
            "check": name, "status": status, "observed": observed,
            "threshold": threshold, "detail": detail,
        })

    def compare_final_time(report: dict[str, Any], source: str) -> None:
        target = _number(report.get("target_final_time_s"), f"{source}.target_final_time_s", nonnegative=True)
        finals = [
            _number(row.get("final_time_s"), f"{source}.levels.final_time_s", nonnegative=True)
            for row in report["levels"]
        ]
        rel = max(abs(value - target) / max(abs(target), 1e-30) for value in finals)
        add(
            f"{source}_common_final_time",
            "PASS" if rel <= limits["final_time_relative_tolerance"] else "FAIL",
            rel, limits["final_time_relative_tolerance"],
            "Maximum relative deviation of any level from requested physical end time.",
        )

    compare_final_time(timestep, "timestep")
    compare_final_time(resolution, "resolution")

    def peak_delta(rows: list[dict[str, Any]], field: str, threshold: float, name: str) -> None:
        values = [
            _number(row.get(field), f"{name}.{field}", nonnegative=True)
            for row in rows
        ]
        # The finest level is expected to be last after the producer sorts factors.
        worst = max(values)
        add(
            name, "PASS" if worst <= threshold else "WARN",
            worst, threshold,
            "Largest reported peak-force relative difference versus the finest level. "
            "WARN signals sensitivity; it does not invalidate the simulation by itself.",
        )

    peak_delta(
        ts_levels, "peak_force_relative_delta_vs_finest",
        limits["max_timestep_peak_force_delta"], "timestep_peak_force_sensitivity",
    )
    peak_delta(
        rs_levels, "peak_force_relative_delta_vs_finest",
        limits["max_resolution_peak_force_delta"], "resolution_peak_force_sensitivity",
    )

    cross_values = [
        _number(row.get("force_normalized_rmse_vs_finest"),
                "cross_resolution.force_normalized_rmse_vs_finest", nonnegative=True)
        for row in cs_levels
    ]
    cross_worst = max(cross_values)
    add(
        "cross_resolution_force_history_sensitivity",
        "PASS" if cross_worst <= limits["max_cross_resolution_force_nrmse"] else "WARN",
        cross_worst, limits["max_cross_resolution_force_nrmse"],
        "Largest normalized force-history RMSE versus the finest-resolution reference.",
    )

    residual_values: list[float] = []
    energy_values: list[float] = []
    for source, rows in (("timestep", ts_levels), ("resolution", rs_levels)):
        for index, row in enumerate(rows):
            residual_key = "max_abs_energy_balance_residual_J"
            if residual_key not in row:
                residual_key = "max_abs_energy_balance_residual_J"
            residual_values.append(_number(
                row.get(residual_key), f"{source}.levels[{index}].{residual_key}", nonnegative=True
            ))
            energy_values.append(abs(_number(
                row.get("final_mechanical_energy_J"),
                f"{source}.levels[{index}].final_mechanical_energy_J"
            )))
    # Normalize each run's residual by its own final energy scale, with a tiny
    # numerical floor to keep a zero-energy case defined.
    ratios = [
        residual / max(energy, 1e-30)
        for residual, energy in zip(residual_values, energy_values)
    ]
    worst_ratio = max(ratios)
    add(
        "energy_balance_residual_ratio",
        "PASS" if worst_ratio <= limits["max_energy_residual_ratio"] else "FAIL",
        worst_ratio, limits["max_energy_residual_ratio"],
        "Maximum absolute energy-balance residual divided by absolute final mechanical energy; "
        "inspect the raw joule values when the energy scale is very small.",
    )

    statuses = {row["status"] for row in checks}
    verdict = "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"
    return {
        "protocol": "tensordem-dem3d-engineering-screen-v1",
        "verdict": verdict,
        "meaning": (
            "PASS/WARN/FAIL applies only to the configured numerical screening gates. "
            "It is not proof of material calibration, experimental validation, "
            "asymptotic convergence, or certification for full-scale icebreaking."
        ),
        "inputs": {
            "timestep_report": str(timestep_path),
            "resolution_report": str(resolution_path),
            "cross_resolution_report": str(cross_resolution_path),
        },
        "thresholds": limits,
        "check_count": len(checks),
        "pass_count": sum(row["status"] == "PASS" for row in checks),
        "warn_count": sum(row["status"] == "WARN" for row in checks),
        "fail_count": sum(row["status"] == "FAIL" for row in checks),
        "checks": checks,
        "recommended_follow_up": (
            "Investigate all FAIL checks before interpreting results; review WARN checks, "
            "inspect full force/damage histories, and calibrate against experimental data."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DEM3D engineering numerical screening report")
    parser.add_argument("--timestep", type=Path, required=True,
                        help="JSON from validate_dem3d_convergence.py")
    parser.add_argument("--resolution", type=Path, required=True,
                        help="JSON from validate_dem3d_resolution.py")
    parser.add_argument("--cross-resolution", type=Path, required=True,
                        help="JSON from compare_dem3d_resolutions.py")
    parser.add_argument("--max-timestep-peak-force-delta", type=float, default=0.05)
    parser.add_argument("--max-resolution-peak-force-delta", type=float, default=0.10)
    parser.add_argument("--max-cross-resolution-force-nrmse", type=float, default=0.10)
    parser.add_argument("--max-energy-residual-ratio", type=float, default=0.05)
    parser.add_argument("--final-time-relative-tolerance", type=float, default=1e-8)
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-acceptance"))
    args = parser.parse_args(argv)
    try:
        report = build_acceptance_report(
            args.timestep, args.resolution, args.cross_resolution,
            max_timestep_peak_force_delta=args.max_timestep_peak_force_delta,
            max_resolution_peak_force_delta=args.max_resolution_peak_force_delta,
            max_cross_resolution_force_nrmse=args.max_cross_resolution_force_nrmse,
            max_energy_residual_ratio=args.max_energy_residual_ratio,
            final_time_relative_tolerance=args.final_time_relative_tolerance,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "engineering_screen_3d.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with (args.output / "engineering_screen_3d.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            import csv
            writer = csv.DictWriter(
                stream, fieldnames=["check", "status", "observed", "threshold", "detail"]
            )
            writer.writeheader()
            writer.writerows(report["checks"])
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "protocol": report["protocol"], "verdict": report["verdict"],
        "check_count": report["check_count"], "pass_count": report["pass_count"],
        "warn_count": report["warn_count"], "fail_count": report["fail_count"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
