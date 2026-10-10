"""Correlate sampled DEM3D indenter reactions with per-bond fracture events.

This is descriptive post-processing: temporal association is not proof that force
changes were caused by fracture, and interval sampling can hide short force peaks.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence


def _read_csv(path: Path, *, allow_empty: bool = False) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"input file not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
    if not rows and not allow_empty:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows


def _number(row: dict[str, str], key: str, source: str, index: int) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source} row {index}: invalid or missing {key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{source} row {index}: {key} must be finite")
    return value


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    sx = sum(v * v for v in dx)
    sy = sum(v * v for v in dy)
    if sx <= 0 or sy <= 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(sx * sy)


def analyze_fracture_force(
    history_path: Path,
    events_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate event/history consistency and write interval and summary reports."""
    raw_history = _read_csv(history_path)
    # A valid no-fracture run writes a header-only event CSV by design.\n    raw_events = _read_csv(events_path, allow_empty=True)
    history: list[dict[str, float | int]] = []
    for idx, row in enumerate(raw_history, start=2):
        t = _number(row, "time", "history", idx)
        force = _number(row, "reaction_z", "history", idx)
        broken_value = _number(row, "broken_bonds", "history", idx)
        if broken_value < 0 or not broken_value.is_integer():
            raise ValueError(f"history row {idx}: broken_bonds must be a nonnegative integer")
        step_value = _number(row, "step", "history", idx)
        if step_value < 0 or not step_value.is_integer():
            raise ValueError(f"history row {idx}: step must be a nonnegative integer")
        history.append({"time": t, "reaction_z": force,
                        "broken_bonds": int(broken_value), "step": int(step_value)})
    if len(history) < 2:
        raise ValueError("history must contain at least two sampled times")
    for prev, cur in zip(history, history[1:]):
        if cur["time"] <= prev["time"] or cur["step"] <= prev["step"]:
            raise ValueError("history time and step must be strictly increasing")
        if cur["broken_bonds"] < prev["broken_bonds"]:
            raise ValueError("cumulative broken_bonds must never decrease")

    events: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_events, start=2):
        t = _number(row, "failure_time_s", "events", idx)
        if t < 0:
            raise ValueError(f"events row {idx}: broken bond has invalid failure_time_s")
        try:
            bond_id = int(row["bond_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"events row {idx}: invalid or missing bond_id") from exc
        if bond_id < 0:
            raise ValueError(f"events row {idx}: bond_id must be nonnegative")
        events.append({"failure_time_s": t, "bond_id": bond_id,
                       "failure_mode_label": row.get("failure_mode_label", "unknown")})
    event_ids = [event["bond_id"] for event in events]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("fracture event bond_id values must be unique")
    events.sort(key=lambda event: (event["failure_time_s"], event["bond_id"]))

    final_broken = int(history[-1]["broken_bonds"])
    if len(events) != final_broken:
        raise ValueError(
            f"event/history mismatch: {len(events)} fracture rows but final history "
            f"reports {final_broken} broken bonds"
        )
    start_time, end_time = float(history[0]["time"]), float(history[-1]["time"])
    tol = max(1e-12, abs(end_time - start_time) * 1e-10)
    if any(event["failure_time_s"] < start_time - tol or
           event["failure_time_s"] > end_time + tol for event in events):
        raise ValueError("fracture event time lies outside the sampled history window")

    intervals: list[dict[str, Any]] = []
    for idx, (left, right) in enumerate(zip(history, history[1:])):
        t0, t1 = float(left["time"]), float(right["time"])
        # Half-open intervals (t0, t1] avoid double-counting boundary events.
        # The first interval includes events at the first sampled time.
        interval_events = [
            event for event in events
            if (event["failure_time_s"] > t0 + tol or
                (idx == 0 and event["failure_time_s"] >= t0 - tol))
            and event["failure_time_s"] <= t1 + tol
        ]
        damage_delta = int(right["broken_bonds"]) - int(left["broken_bonds"])
        if len(interval_events) != damage_delta:
            raise ValueError(
                f"interval {idx} [{t0:g}, {t1:g}]: {len(interval_events)} events "
                f"but cumulative damage increased by {damage_delta}; event times "
                "and sampled history are inconsistent"
            )
        dt = t1 - t0
        f0, f1 = float(left["reaction_z"]), float(right["reaction_z"])
        abs0, abs1 = abs(f0), abs(f1)
        rate = damage_delta / dt
        intervals.append({
            "interval_index": idx,
            "start_time_s": t0,
            "end_time_s": t1,
            "duration_s": dt,
            "start_step": int(left["step"]),
            "end_step": int(right["step"]),
            "reaction_z_start_N": f0,
            "reaction_z_end_N": f1,
            "reaction_z_mean_trapezoid_N": (f0 + f1) / 2,
            "reaction_z_abs_mean_trapezoid_N": (abs0 + abs1) / 2,
            "reaction_z_peak_sample_abs_N": max(abs0, abs1),
            "signed_reaction_impulse_Ns": (f0 + f1) * dt / 2,
            "absolute_reaction_impulse_Ns": (abs0 + abs1) * dt / 2,
            "new_fractures": damage_delta,
            "fracture_rate_per_s": rate,
            "event_bond_ids": ";".join(str(event["bond_id"]) for event in interval_events),
            "event_failure_modes": ";".join(str(event["failure_mode_label"])
                                             for event in interval_events),
        })

    rates = [float(row["fracture_rate_per_s"]) for row in intervals]
    abs_forces = [float(row["reaction_z_abs_mean_trapezoid_N"]) for row in intervals]
    summary: dict[str, Any] = {
        "protocol": "tensordem-dem3d-fracture-force-analysis-v1",
        "interpretation": (
            "Descriptive temporal association only. Force values are sampled at history "
            "times; interval peak is the maximum endpoint sample, not a guaranteed true "
            "continuous-time peak. Correlation does not establish causation."
        ),
        "history_file": history_path.name,
        "fracture_events_file": events_path.name,
        "sample_count": len(history),
        "interval_count": len(intervals),
        "start_time_s": start_time,
        "end_time_s": end_time,
        "duration_s": end_time - start_time,
        "total_broken_bonds": final_broken,
        "peak_sample_abs_reaction_z_N": max(abs(float(row["reaction_z"])) for row in history),
        "maximum_interval_fracture_rate_per_s": max(rates, default=0.0),
        "total_signed_reaction_impulse_Ns": sum(
            float(row["signed_reaction_impulse_Ns"]) for row in intervals
        ),
        "total_absolute_reaction_impulse_Ns": sum(
            float(row["absolute_reaction_impulse_Ns"]) for row in intervals
        ),
        "pearson_fracture_rate_vs_abs_reaction": _pearson(rates, abs_forces),
        "interval_damage_matches_event_timestamps": True,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    interval_path = output_dir / "fracture_force_intervals_3d.csv"
    with interval_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(intervals[0]))
        writer.writeheader()
        writer.writerows(intervals)
    summary_path = output_dir / "fracture_force_summary_3d.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Correlate DEM3D sampled indenter force with bond-fracture events"
    )
    parser.add_argument("--history", type=Path, required=True,
                        help="history_3d.csv from benchmark_dem3d.py")
    parser.add_argument("--events", type=Path, required=True,
                        help="fracture_events_3d.csv from benchmark_dem3d.py")
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-fracture-analysis"))
    args = parser.parse_args(argv)
    try:
        result = analyze_fracture_force(args.history, args.events, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
