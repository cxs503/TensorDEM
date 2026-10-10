"""Cross-resolution DEM3D force, damage, and energy validation.

Consumes resolution_history_3d.csv emitted by validate_dem3d_resolution.py.
Force histories are linearly interpolated onto the finest-level time grid over
the common time window; cumulative broken-bond counts use zero-order hold.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence


REQUIRED = (
    "resolution_factor", "step", "time_s", "reaction_z_N",
    "broken_bonds", "mechanical_energy_J", "energy_balance_residual_J",
)


def _finite(row: dict[str, str], key: str, line: int) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"line {line}: missing/invalid {key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"line {line}: {key} must be finite")
    return value


def read_resolution_history(path: Path) -> dict[int, list[dict[str, float | int]]]:
    if not path.is_file():
        raise FileNotFoundError(f"history file not found: {path}")
    by_factor: dict[int, list[dict[str, float | int]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or any(key not in reader.fieldnames for key in REQUIRED):
            raise ValueError(f"{path}: expected columns {', '.join(REQUIRED)}")
        for line, raw in enumerate(reader, start=2):
            factor_value = _finite(raw, "resolution_factor", line)
            if factor_value < 1 or not factor_value.is_integer():
                raise ValueError(f"line {line}: resolution_factor must be a positive integer")
            factor = int(factor_value)
            time_s = _finite(raw, "time_s", line)
            step_value = _finite(raw, "step", line)
            broken_value = _finite(raw, "broken_bonds", line)
            if time_s < 0 or step_value < 0 or not step_value.is_integer():
                raise ValueError(f"line {line}: time and step must be nonnegative")
            if broken_value < 0 or not broken_value.is_integer():
                raise ValueError(f"line {line}: broken_bonds must be a nonnegative integer")
            row = {
                "resolution_factor": factor,
                "step": int(step_value),
                "time_s": time_s,
                "reaction_z_N": _finite(raw, "reaction_z_N", line),
                "broken_bonds": int(broken_value),
                "mechanical_energy_J": _finite(raw, "mechanical_energy_J", line),
                "energy_balance_residual_J": _finite(raw, "energy_balance_residual_J", line),
            }
            by_factor.setdefault(factor, []).append(row)
    if len(by_factor) < 2:
        raise ValueError("history must contain at least two resolution factors")
    for factor, rows in by_factor.items():
        if len(rows) < 2:
            raise ValueError(f"factor {factor}: at least two history samples are required")
        rows.sort(key=lambda row: float(row["time_s"]))
        for previous, current in zip(rows, rows[1:]):
            if float(current["time_s"]) <= float(previous["time_s"]):
                raise ValueError(f"factor {factor}: sample times must be strictly increasing")
            if int(current["broken_bonds"]) < int(previous["broken_bonds"]):
                raise ValueError(f"factor {factor}: cumulative broken_bonds decreased")
    return dict(sorted(by_factor.items()))


def _linear(rows: Sequence[dict[str, float | int]], time_s: float, key: str) -> float:
    if time_s < float(rows[0]["time_s"]) or time_s > float(rows[-1]["time_s"]):
        raise ValueError("interpolation requested outside history window")
    lo, hi = 0, len(rows) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if float(rows[mid]["time_s"]) <= time_s:
            lo = mid
        else:
            hi = mid
    t0, t1 = float(rows[lo]["time_s"]), float(rows[hi]["time_s"])
    y0, y1 = float(rows[lo][key]), float(rows[hi][key])
    if math.isclose(time_s, t0, rel_tol=0.0, abs_tol=1e-14):
        return y0
    if math.isclose(time_s, t1, rel_tol=0.0, abs_tol=1e-14):
        return y1
    return y0 + (y1 - y0) * (time_s - t0) / (t1 - t0)


def _hold(rows: Sequence[dict[str, float | int]], time_s: float, key: str) -> float:
    value = float(rows[0][key])
    for row in rows:
        if float(row["time_s"]) > time_s:
            break
        value = float(row[key])
    return value


def _trapz(times: Sequence[float], values: Sequence[float]) -> float:
    return sum(
        (values[i] + values[i + 1]) * (times[i + 1] - times[i]) / 2.0
        for i in range(len(times) - 1)
    )


def compare_resolution_histories(
    histories: dict[int, list[dict[str, float | int]]],
) -> dict[str, Any]:
    factors = sorted(histories)
    reference_factor = factors[-1]
    start = max(float(histories[f][0]["time_s"]) for f in factors)
    end = min(float(histories[f][-1]["time_s"]) for f in factors)
    if end <= start:
        raise ValueError("resolution histories have no common time interval")
    reference = histories[reference_factor]
    times = [float(row["time_s"]) for row in reference if start <= float(row["time_s"]) <= end]
    if not times or not math.isclose(times[0], start, rel_tol=1e-10, abs_tol=1e-14):
        times.insert(0, start)
    if not math.isclose(times[-1], end, rel_tol=1e-10, abs_tol=1e-14):
        times.append(end)
    if len(times) < 2:
        raise ValueError("common time interval has fewer than two comparison points")

    reference_force = [_linear(reference, t, "reaction_z_N") for t in times]
    ref_rms = math.sqrt(sum(v * v for v in reference_force) / len(reference_force))
    ref_peak_idx = max(range(len(reference_force)), key=lambda i: abs(reference_force[i]))
    ref_impulse = _trapz(times, reference_force)
    aligned: list[dict[str, Any]] = []
    summary_levels: list[dict[str, Any]] = []
    ref_damage = [_hold(reference, t, "broken_bonds") for t in times]

    for factor in factors:
        rows = histories[factor]
        forces = [_linear(rows, t, "reaction_z_N") for t in times]
        damage = [_hold(rows, t, "broken_bonds") for t in times]
        differences = [a - b for a, b in zip(forces, reference_force)]
        rmse = math.sqrt(sum(v * v for v in differences) / len(differences))
        force_rms = math.sqrt(sum(v * v for v in forces) / len(forces))
        normalized_rmse = rmse / max(ref_rms, 1e-30)
        peak_idx = max(range(len(forces)), key=lambda i: abs(forces[i]))
        fracture_rows = [row for row in rows if int(row["broken_bonds"]) > 0]
        first_fracture = float(fracture_rows[0]["time_s"]) if fracture_rows else None
        final = rows[-1]
        aligned.extend({
            "time_s": t,
            "resolution_factor": factor,
            "reaction_z_N": force,
            "reference_reaction_z_N": ref_force,
            "reaction_delta_vs_finest_N": force - ref_force,
            "broken_bonds": int(damage_value),
            "reference_broken_bonds": int(ref_damage_value),
            "broken_bond_delta_vs_finest": int(damage_value - ref_damage_value),
        } for t, force, ref_force, damage_value, ref_damage_value in zip(
            times, forces, reference_force, damage, ref_damage
        ))
        summary_levels.append({
            "resolution_factor": factor,
            "sample_count_original": len(rows),
            "common_window_start_s": start,
            "common_window_end_s": end,
            "common_window_duration_s": end - start,
            "peak_abs_reaction_z_N": max(abs(v) for v in forces),
            "peak_reaction_time_s": times[peak_idx],
            "signed_reaction_impulse_Ns": _trapz(times, forces),
            "force_rms_N": force_rms,
            "force_rmse_vs_finest_N": rmse,
            "force_normalized_rmse_vs_finest": normalized_rmse,
            "final_broken_bonds_at_common_end": int(damage[-1]),
            "final_broken_bonds_vs_finest_delta": int(damage[-1] - ref_damage[-1]),
            "damage_curve_rmse_bonds": math.sqrt(
                sum((a - b) ** 2 for a, b in zip(damage, ref_damage)) / len(damage)
            ),
            "first_fracture_time_s_original_samples": first_fracture,
            "final_mechanical_energy_J": float(final["mechanical_energy_J"]),
            "max_abs_energy_balance_residual_J": max(
                abs(float(row["energy_balance_residual_J"])) for row in rows
            ),
            "is_reference_finest": factor == reference_factor,
        })
    return {
        "protocol": "tensordem-dem3d-cross-resolution-validation-v1",
        "reference_resolution_factor": reference_factor,
        "comparison_time_grid_points": len(times),
        "common_window_start_s": start,
        "common_window_end_s": end,
        "reference_peak_abs_reaction_z_N": abs(reference_force[ref_peak_idx]),
        "reference_peak_reaction_time_s": times[ref_peak_idx],
        "reference_signed_impulse_Ns": ref_impulse,
        "interpolation": {
            "force": "linear interpolation on the finest-resolution time grid",
            "cumulative_damage": "zero-order hold",
        },
        "interpretation": (
            "Cross-resolution sensitivity metrics only. The finest level is a numerical "
            "reference, not ground truth. Different bond topology and calibration can "
            "change fracture response; these metrics do not establish convergence or "
            "experimental validity."
        ),
        "levels": summary_levels,
        "aligned_history": aligned,
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in report.items() if key != "aligned_history"}
    (output / "cross_resolution_summary_3d.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for filename, key in (
        ("cross_resolution_levels_3d.csv", "levels"),
        ("cross_resolution_history_3d.csv", "aligned_history"),
    ):
        rows = report[key]
        if rows:
            with (output / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare DEM3D force/damage histories across resolutions")
    parser.add_argument("--history", type=Path, action="append", required=True,
                        help="resolution_history_3d.csv path; may be supplied once per campaign")
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-cross-resolution"))
    args = parser.parse_args(argv)
    try:
        combined: dict[int, list[dict[str, float | int]]] = {}
        for path in args.history:
            for factor, rows in read_resolution_history(path).items():
                if factor in combined:
                    raise ValueError(f"duplicate resolution factor {factor} across input files")
                combined[factor] = rows
        if len(combined) < 2:
            raise ValueError("at least two distinct resolution factors are required")
        report = compare_resolution_histories(dict(sorted(combined.items())))
        write_report(report, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    printable = {key: value for key, value in report.items() if key != "aligned_history"}
    printable["aligned_history_rows_written"] = len(report["aligned_history"])
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
