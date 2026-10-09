"""Reproducible DEM verification campaigns and force-history analysis.

This module provides engineering bookkeeping around the existing IceDEM solver.
It does not change the constitutive law or claim that a converged response is
physically validated. All stored dimensional quantities use SI units.

The public functions deliberately operate on plain Python dictionaries/lists so
reports can be inspected, versioned, compared in code review, and loaded without
NumPy or pandas.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .dem import DEMConfig, IceDEM

REPORT_SCHEMA = "tensordem.verification-campaign/1"
HISTORY_SCHEMA = "tensordem.history/1"
FORCE_CHANNELS = ("reaction_x", "reaction_y")
DEFAULT_HISTORY_CHANNELS = (
    "reaction_x",
    "reaction_y",
    "boundary_reaction_x",
    "boundary_reaction_y",
    "kinetic_energy",
    "mechanical_energy",
    "bond_energy",
    "pair_contact_energy",
    "tool_contact_energy",
    "broken_bonds",
)


class CampaignValidationError(ValueError):
    """Raised when campaign inputs or report structures are inconsistent."""


@dataclass(frozen=True)
class CampaignSpec:
    """A single controlled-displacement indentation run.

    Width is kept fixed by choosing particle radius from nx. The number of rows
    is computed from the target height and then the realized geometry is
    recorded; the height is therefore approximate for coarse resolutions.
    """

    nx: int = 9
    target_width_m: float = 0.3
    target_height_m: float = 0.2
    thickness_m: float = 0.2
    duration_s: float = 0.02
    speed_m_s: float = 0.2
    save_every: int = 1
    density_kg_m3: float = 917.0
    bond_stiffness_N_m: float = 2_000.0
    contact_stiffness_N_m: float = 2_000.0
    breaking_strain: float = 0.015
    shear_breaking_strain: float = 0.03
    contact_damping_N_s_m: float = 5.0
    drag_N_s_m: float = 2.0
    tool_radius_m: float = 0.1
    tool_gap_m: float = 0.01
    dt_s: float | None = None
    device: str = "cpu"
    fix_edges: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.nx, bool) or not isinstance(self.nx, int) or self.nx < 3:
            raise CampaignValidationError("nx must be an integer >= 3")
        if isinstance(self.save_every, bool) or not isinstance(self.save_every, int) or self.save_every < 1:
            raise CampaignValidationError("save_every must be a positive integer")
        positive = (
            "target_width_m",
            "target_height_m",
            "thickness_m",
            "duration_s",
            "speed_m_s",
            "density_kg_m3",
            "bond_stiffness_N_m",
            "contact_stiffness_N_m",
            "breaking_strain",
            "shear_breaking_strain",
            "tool_radius_m",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CampaignValidationError(f"{name} must be numeric")
            if not math.isfinite(value) or value <= 0:
                raise CampaignValidationError(f"{name} must be finite and positive")
        for name in ("contact_damping_N_s_m", "drag_N_s_m", "tool_gap_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CampaignValidationError(f"{name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise CampaignValidationError(f"{name} must be finite and nonnegative")
        if self.dt_s is not None:
            if (
                isinstance(self.dt_s, bool)
                or not isinstance(self.dt_s, (int, float))
                or not math.isfinite(self.dt_s)
                or self.dt_s <= 0
            ):
                raise CampaignValidationError("dt_s must be finite and positive")
        if self.device not in ("cpu", "cuda"):
            raise CampaignValidationError("device must be cpu or cuda")
        if type(self.fix_edges) is not bool:
            raise CampaignValidationError("fix_edges must be bool")

    def to_dem_config(self) -> DEMConfig:
        """Convert this case to the production solver configuration."""
        radius = self.target_width_m / (2 * (self.nx - 1))
        ny = max(2, round(self.target_height_m / (2 * radius)) + 1)
        return DEMConfig(
            nx=self.nx,
            ny=ny,
            radius=radius,
            thickness=self.thickness_m,
            density=self.density_kg_m3,
            bond_stiffness=self.bond_stiffness_N_m,
            contact_stiffness=self.contact_stiffness_N_m,
            breaking_strain=self.breaking_strain,
            shear_breaking_strain=self.shear_breaking_strain,
            contact_damping=self.contact_damping_N_s_m,
            drag=self.drag_N_s_m,
            tool_radius=self.tool_radius_m,
            tool_speed=self.speed_m_s,
            tool_gap=self.tool_gap_m,
            dt=self.dt_s,
            device=self.device,
            fix_edges=self.fix_edges,
        )

    def realized_geometry(self) -> dict[str, float | int]:
        config = self.to_dem_config()
        return {
            "nx": config.nx,
            "ny": config.ny,
            "particle_count": config.nx * config.ny,
            "particle_radius_m": config.radius,
            "target_width_m": self.target_width_m,
            "actual_width_m": 2 * config.radius * (config.nx - 1),
            "target_height_m": self.target_height_m,
            "actual_height_m": 2 * config.radius * (config.ny - 1),
            "thickness_m": config.thickness,
            "disk_mass_kg": config.mass,
            "total_disk_mass_kg": config.mass * config.nx * config.ny,
        }


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CampaignValidationError(f"{name} must be finite")
    return result


def validate_history(
    history: Sequence[Mapping[str, Any]],
    *,
    required_channels: Sequence[str] = ("time", "reaction_x", "reaction_y"),
    strict: bool = True,
) -> list[dict[str, float]]:
    """Validate and normalize a sampled scalar history.

    Time must be strictly increasing. Every row must contain every requested
    channel. With strict=True, all numeric entries in all rows must be finite;
    optional string metadata should be stored outside the numeric history.
    """
    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        raise CampaignValidationError("history must be a sequence of mapping rows")
    if len(history) < 2:
        raise CampaignValidationError("history must contain at least two samples")
    normalized: list[dict[str, float]] = []
    for row_index, row in enumerate(history):
        if not isinstance(row, Mapping):
            raise CampaignValidationError(f"history row {row_index} must be a mapping")
        missing = [key for key in required_channels if key not in row]
        if missing:
            raise CampaignValidationError(
                f"history row {row_index} is missing channels: {', '.join(missing)}"
            )
        converted: dict[str, float] = {}
        for key, value in row.items():
            if not isinstance(key, str):
                raise CampaignValidationError("history channel names must be strings")
            if strict or key in required_channels:
                converted[key] = _finite_number(value, f"row {row_index} channel {key}")
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                converted[key] = float(value)
        normalized.append(converted)
    times = [row["time"] for row in normalized]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise CampaignValidationError("history times must be strictly increasing")
    return normalized


def _time_values(history: Sequence[Mapping[str, float]], channel: str) -> tuple[list[float], list[float]]:
    if not channel:
        raise CampaignValidationError("channel name cannot be empty")
    rows = validate_history(history, required_channels=("time", channel))
    return [row["time"] for row in rows], [row[channel] for row in rows]


def interpolate_series(
    times: Sequence[float], values: Sequence[float], target_times: Sequence[float]
) -> list[float]:
    """Linearly interpolate scalar samples, rejecting extrapolation."""
    if len(times) != len(values) or len(times) < 2:
        raise CampaignValidationError("times and values must have equal length >= 2")
    tx = [_finite_number(v, "source time") for v in times]
    vy = [_finite_number(v, "source value") for v in values]
    if any(b <= a for a, b in zip(tx, tx[1:])):
        raise CampaignValidationError("source times must be strictly increasing")
    tolerance = 1e-12 * max(1.0, abs(tx[0]), abs(tx[-1]))
    result = []
    for raw_target in target_times:
        target = _finite_number(raw_target, "target time")
        if target < tx[0] - tolerance or target > tx[-1] + tolerance:
            raise CampaignValidationError("interpolation target lies outside source interval")
        if target <= tx[0]:
            result.append(vy[0])
            continue
        if target >= tx[-1]:
            result.append(vy[-1])
            continue
        low, high = 0, len(tx) - 1
        while high - low > 1:
            middle = (low + high) // 2
            if tx[middle] <= target:
                low = middle
            else:
                high = middle
        fraction = (target - tx[low]) / (tx[high] - tx[low])
        result.append(vy[low] + fraction * (vy[high] - vy[low]))
    return result


def trapezoidal_integral(times: Sequence[float], values: Sequence[float]) -> float:
    """Integrate a scalar signal with the trapezoidal rule."""
    if len(times) != len(values) or len(times) < 2:
        raise CampaignValidationError("integration requires equal arrays of length >= 2")
    tx = [_finite_number(v, "time") for v in times]
    vy = [_finite_number(v, "value") for v in values]
    if any(b <= a for a, b in zip(tx, tx[1:])):
        raise CampaignValidationError("integration times must be strictly increasing")
    return sum(
        0.5 * (vy[index] + vy[index + 1]) * (tx[index + 1] - tx[index])
        for index in range(len(tx) - 1)
    )


def _rms(values: Sequence[float]) -> float:
    if not values:
        raise CampaignValidationError("RMS requires at least one value")
    return math.sqrt(sum(value * value for value in values) / len(values))


def summarize_history(
    history: Sequence[Mapping[str, Any]], *, force_channel: str = "reaction_y"
) -> dict[str, float | int]:
    """Compute force peaks, signed impulse, and energy/bond end-state metrics."""
    rows = validate_history(history, required_channels=("time", force_channel))
    times = [row["time"] for row in rows]
    force = [row[force_channel] for row in rows]
    result: dict[str, float | int] = {
        "sample_count": len(rows),
        "start_time_s": times[0],
        "end_time_s": times[-1],
        "duration_s": times[-1] - times[0],
        "peak_positive_force_N": max(0.0, max(force)),
        "peak_negative_force_N": min(0.0, min(force)),
        "peak_absolute_force_N": max(abs(value) for value in force),
        "signed_impulse_Ns": trapezoidal_integral(times, force),
        "absolute_force_impulse_Ns": trapezoidal_integral(times, [abs(v) for v in force]),
        "mean_force_N": trapezoidal_integral(times, force) / (times[-1] - times[0]),
    }
    for channel, key in (
        ("broken_bonds", "final_broken_bonds"),
        ("kinetic_energy", "final_kinetic_energy_J"),
        ("mechanical_energy", "final_mechanical_energy_J"),
        ("bond_energy", "final_bond_energy_J"),
        ("pair_contact_energy", "final_pair_contact_energy_J"),
        ("tool_contact_energy", "final_tool_contact_energy_J"),
    ):
        if channel in rows[-1]:
            result[key] = rows[-1][channel]
    if "broken_bonds" in rows[-1]:
        result["maximum_broken_bonds"] = max(row["broken_bonds"] for row in rows)
    return result


def compare_histories(
    candidate: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    *,
    force_channel: str = "reaction_y",
) -> dict[str, float | int]:
    """Compare force signals on their shared interval, interpolating both grids.

    Sample points are the sorted union of candidate and reference samples
    within the overlap, so differences in sampling interval do not silently
    change the comparison to a nearest-neighbour metric.
    """
    candidate_rows = validate_history(candidate, required_channels=("time", force_channel))
    reference_rows = validate_history(reference, required_channels=("time", force_channel))
    ct = [row["time"] for row in candidate_rows]
    rt = [row["time"] for row in reference_rows]
    start, end = max(ct[0], rt[0]), min(ct[-1], rt[-1])
    if end <= start:
        raise CampaignValidationError("histories must have a positive overlapping interval")
    sample_times = sorted({
        start,
        end,
        *(time for time in ct if start <= time <= end),
        *(time for time in rt if start <= time <= end),
    })
    cv = interpolate_series(ct, [row[force_channel] for row in candidate_rows], sample_times)
    rv = interpolate_series(rt, [row[force_channel] for row in reference_rows], sample_times)
    difference = [left - right for left, right in zip(cv, rv)]
    reference_rms = _rms(rv)
    difference_rms = _rms(difference)
    candidate_summary = summarize_history(candidate_rows, force_channel=force_channel)
    reference_summary = summarize_history(reference_rows, force_channel=force_channel)
    # Peaks and impulses are computed on the same overlap used by the waveform
    # comparison, not on unmatched tails from different physical horizons.
    ref_peak = max(abs(value) for value in rv)
    cand_peak = max(abs(value) for value in cv)
    candidate_impulse = trapezoidal_integral(sample_times, cv)
    reference_impulse = trapezoidal_integral(sample_times, rv)
    bias = sum(difference) / len(difference)
    variance_c = sum((value - sum(cv) / len(cv)) ** 2 for value in cv)
    variance_r = sum((value - sum(rv) / len(rv)) ** 2 for value in rv)
    covariance = sum(
        (left - sum(cv) / len(cv)) * (right - sum(rv) / len(rv))
        for left, right in zip(cv, rv)
    )
    correlation = covariance / math.sqrt(variance_c * variance_r) if variance_c > 0 and variance_r > 0 else 0.0
    return {
        "overlap_start_s": start,
        "overlap_end_s": end,
        "overlap_duration_s": end - start,
        "sample_count": len(sample_times),
        "force_difference_rms_N": difference_rms,
        "force_relative_rms_difference": difference_rms / max(reference_rms, 1e-15),
        "force_mean_bias_N": bias,
        "force_peak_absolute_relative_difference": abs(cand_peak - ref_peak) / max(ref_peak, 1e-15),
        "force_correlation": correlation,
        "candidate_signed_impulse_Ns": candidate_impulse,
        "reference_signed_impulse_Ns": reference_impulse,
        "signed_impulse_relative_difference": abs(candidate_impulse - reference_impulse)
        / max(abs(reference_impulse), 1e-15),
    }


def _sample(simulation: IceDEM) -> dict[str, float | int]:
    diagnostics = simulation.diagnostics()
    energy = simulation.mechanical_energy()
    return {
        **diagnostics,
        "mechanical_energy": energy["mechanical_J"],
        "bond_energy": energy["bond_J"],
        "pair_contact_energy": energy["pair_contact_J"],
        "tool_contact_energy": energy["tool_contact_J"],
        "step_count": simulation.step_count,
    }


def run_case(spec: CampaignSpec, *, max_steps: int = 2_000_000) -> dict[str, Any]:
    """Run one deterministic indentation case and return a self-describing record.

    The solver's own stable timestep check remains authoritative. The step cap
    prevents an accidental very small timestep from producing an unbounded job.
    The final sample is always written even when save_every does not divide the
    number of steps.
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise CampaignValidationError("max_steps must be a positive integer")
    config = spec.to_dem_config()
    simulation = IceDEM(config)
    steps = max(1, math.ceil(spec.duration_s / simulation.dt))
    if steps > max_steps:
        raise CampaignValidationError(
            f"case requires {steps} steps, above max_steps={max_steps}; reduce duration or resolution"
        )
    history: list[dict[str, float | int]] = [_sample(simulation)]
    for step_index in range(1, steps + 1):
        simulation.step()
        if step_index % spec.save_every == 0 or step_index == steps:
            history.append(_sample(simulation))
    normalized = validate_history(history, required_channels=("time", "reaction_x", "reaction_y"))
    summary = summarize_history(normalized)
    geometry = spec.realized_geometry()
    return {
        "spec": asdict(spec),
        "solver": {
            "name": "IceDEM",
            "integrator": "semi-implicit Euler",
            "requested_duration_s": spec.duration_s,
            "actual_duration_s": simulation.time,
            "dt_s": simulation.dt,
            "step_count": simulation.step_count,
            "time_samples": len(normalized),
            "device": str(simulation.device),
            "model_scope": "2-D central-force bonded disks; no particle rotations or explicit bond moments",
        },
        "geometry": geometry,
        "summary": summary,
        "history": normalized,
        "history_sha256": history_digest(normalized),
    }


def history_digest(history: Sequence[Mapping[str, Any]]) -> str:
    """Stable SHA-256 for a validated numeric history, independent of dict order."""
    rows = validate_history(history)
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_resolution_campaign(
    specs: Sequence[CampaignSpec],
    *,
    reference_index: int | None = None,
    max_steps: int = 2_000_000,
) -> dict[str, Any]:
    """Run cases serially and compare every force history to a chosen reference."""
    if not specs:
        raise CampaignValidationError("at least one CampaignSpec is required")
    if any(not isinstance(spec, CampaignSpec) for spec in specs):
        raise CampaignValidationError("all campaign cases must be CampaignSpec instances")
    if reference_index is None:
        reference_index = max(range(len(specs)), key=lambda index: specs[index].nx)
    if isinstance(reference_index, bool) or not isinstance(reference_index, int):
        raise CampaignValidationError("reference_index must be an integer")
    if reference_index < 0 or reference_index >= len(specs):
        raise CampaignValidationError("reference_index is outside the case list")
    cases = [run_case(spec, max_steps=max_steps) for spec in specs]
    reference = cases[reference_index]
    comparisons = []
    for index, case in enumerate(cases):
        metrics = compare_histories(case["history"], reference["history"])
        comparisons.append({
            "case_index": index,
            "reference_index": reference_index,
            "case_nx": case["spec"]["nx"],
            "reference_nx": reference["spec"]["nx"],
            **metrics,
            "is_reference": index == reference_index,
        })
    return {
        "schema": REPORT_SCHEMA,
        "units": {"length": "m", "time": "s", "force": "N", "energy": "J", "mass": "kg"},
        "methodology": {
            "purpose": "numerical sensitivity and reproducibility evidence",
            "physical_validation": False,
            "comparison": "linear interpolation on the union of samples over shared time",
            "reference_selection": "user-selected index or maximum nx",
        },
        "reference_index": reference_index,
        "case_count": len(cases),
        "cases": cases,
        "comparisons": comparisons,
    }


def parameter_grid(
    base: CampaignSpec,
    parameters: Mapping[str, Sequence[Any]],
) -> list[CampaignSpec]:
    """Create a deterministic Cartesian product of selected CampaignSpec fields."""
    if not isinstance(base, CampaignSpec):
        raise CampaignValidationError("base must be a CampaignSpec")
    if not isinstance(parameters, Mapping) or not parameters:
        raise CampaignValidationError("parameters must be a non-empty mapping")
    allowed = set(CampaignSpec.__dataclass_fields__)
    keys = list(parameters)
    for key in keys:
        if key not in allowed:
            raise CampaignValidationError(f"unknown CampaignSpec field: {key}")
        values = parameters[key]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
            raise CampaignValidationError(f"parameter {key} must have a non-empty sequence")
    base_values = asdict(base)
    cases = []
    for values in itertools.product(*(parameters[key] for key in keys)):
        candidate = dict(base_values)
        candidate.update(zip(keys, values))
        cases.append(CampaignSpec(**candidate))
    return cases


def validate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate report schema and internal case/comparison references."""
    if not isinstance(report, Mapping):
        raise CampaignValidationError("report must be a mapping")
    if report.get("schema") != REPORT_SCHEMA:
        raise CampaignValidationError("unsupported campaign report schema")
    cases = report.get("cases")
    comparisons = report.get("comparisons")
    if not isinstance(cases, list) or not cases:
        raise CampaignValidationError("report cases must be a non-empty list")
    if not isinstance(comparisons, list) or len(comparisons) != len(cases):
        raise CampaignValidationError("report must contain one comparison per case")
    reference_index = report.get("reference_index")
    if isinstance(reference_index, bool) or not isinstance(reference_index, int):
        raise CampaignValidationError("invalid reference_index")
    if not 0 <= reference_index < len(cases):
        raise CampaignValidationError("reference_index is out of range")
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping) or not isinstance(case.get("spec"), Mapping):
            raise CampaignValidationError(f"case {index} has no spec")
        validate_history(case.get("history", []))
        expected_digest = history_digest(case["history"])
        if case.get("history_sha256") != expected_digest:
            raise CampaignValidationError(f"case {index} history digest mismatch")
        if not isinstance(case.get("summary"), Mapping):
            raise CampaignValidationError(f"case {index} has no summary")
    for index, comparison in enumerate(comparisons):
        if not isinstance(comparison, Mapping) or comparison.get("case_index") != index:
            raise CampaignValidationError(f"comparison {index} has inconsistent case_index")
        if comparison.get("reference_index") != reference_index:
            raise CampaignValidationError(f"comparison {index} has inconsistent reference_index")
    # JSON round-trip also rejects unsupported objects and non-finite floats.
    try:
        json.dumps(report, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CampaignValidationError(f"report is not strict JSON data: {exc}") from exc
    return dict(report)


def write_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically write a strict JSON report, avoiding partially written files."""
    normalized = validate_report(report)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except OSError:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise
    return target


def read_report(path: str | Path) -> dict[str, Any]:
    """Read and fully validate a JSON report."""
    target = Path(path)
    try:
        with target.open("r", encoding="utf-8") as stream:
            report = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignValidationError(f"cannot read campaign report {target}: {exc}") from exc
    return validate_report(report)


def write_summary_csv(report: Mapping[str, Any], path: str | Path) -> Path:
    """Export case-level scalar metrics to CSV for spreadsheet plotting."""
    normalized = validate_report(report)
    rows = []
    for index, case in enumerate(normalized["cases"]):
        row: dict[str, Any] = {"case_index": index}
        row.update({f"spec_{key}": value for key, value in case["spec"].items()})
        row.update({f"geometry_{key}": value for key, value in case["geometry"].items()})
        row.update({f"summary_{key}": value for key, value in case["summary"].items()})
        comparison = normalized["comparisons"][index]
        row.update({f"comparison_{key}": value for key, value in comparison.items() if key not in ("case_index",)})
        rows.append(row)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_name = stream.name
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except OSError:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise
    return target


def report_metadata(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact provenance fields suitable for logs and CI summaries."""
    normalized = validate_report(report)
    case_digests = [case["history_sha256"] for case in normalized["cases"]]
    joined = "|".join(case_digests).encode("ascii")
    return {
        "schema": normalized["schema"],
        "case_count": len(normalized["cases"]),
        "reference_index": normalized["reference_index"],
        "campaign_sha256": hashlib.sha256(joined).hexdigest(),
        "physical_validation": normalized["methodology"]["physical_validation"],
    }


def make_resolution_specs(
    levels: Iterable[int],
    *,
    target_width_m: float = 0.3,
    target_height_m: float = 0.2,
    duration_s: float = 0.02,
    speed_m_s: float = 0.2,
    **overrides: Any,
) -> list[CampaignSpec]:
    """Convenience builder for a resolution series with common physical targets."""
    level_list = list(levels)
    if not level_list:
        raise CampaignValidationError("levels must not be empty")
    if len(set(level_list)) != len(level_list):
        raise CampaignValidationError("resolution levels must be unique")
    return [
        CampaignSpec(
            nx=level,
            target_width_m=target_width_m,
            target_height_m=target_height_m,
            duration_s=duration_s,
            speed_m_s=speed_m_s,
            **overrides,
        )
        for level in level_list
    ]
