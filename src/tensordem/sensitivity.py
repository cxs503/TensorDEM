"""Resolution-sensitivity sweep for the 2-D indentation prototype.

This is a numerical sensitivity tool, not a material-calibration or validation
claim. The ice width and target height are held approximately fixed while the
particle spacing changes; the actual discretized height is recorded per case.
"""

import argparse
import csv
import math
from pathlib import Path

import torch

from .dem import DEMConfig, IceDEM


def build_config(
    nx: int,
    width: float = 0.3,
    thickness: float = 0.2,
    speed: float = 0.2,
    device: str = "cpu",
) -> DEMConfig:
    """Build a lattice with approximately fixed physical width and thickness."""
    if isinstance(nx, bool) or not isinstance(nx, int) or nx < 3:
        raise ValueError("each nx level must be an integer >= 3")
    for name, value in (("width", width), ("thickness", thickness), ("speed", speed)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    radius = width / (2 * (nx - 1))
    ny = max(2, round(thickness / (2 * radius)) + 1)
    return DEMConfig(nx=nx, ny=ny, radius=radius, tool_speed=speed, device=device)


def simulate_case(config: DEMConfig, duration: float) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """Simulate to a comparable physical horizon and summarize the reaction history."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    sim = IceDEM(config)
    if sim.device.type == "cpu":
        torch.set_num_threads(1)
    steps = max(1, math.ceil(duration / sim.dt))
    history: list[dict[str, float | int]] = [sim.diagnostics()]
    for _ in range(steps):
        sim.step()
        history.append(sim.diagnostics())

    times = [float(row["time"]) for row in history]
    reactions = [float(row["reaction_y"]) for row in history]
    impulse = sum(
        0.5 * (reactions[i] + reactions[i + 1]) * (times[i + 1] - times[i])
        for i in range(len(times) - 1)
    )
    summary: dict[str, float | int] = {
        "nx": config.nx,
        "ny": config.ny,
        "particles": config.nx * config.ny,
        "radius_m": config.radius,
        "actual_width_m": 2 * config.radius * (config.nx - 1),
        "actual_height_m": 2 * config.radius * (config.ny - 1),
        "dt_s": sim.dt,
        "steps": steps,
        "simulated_time_s": sim.time,
        "peak_abs_reaction_y_N": max(abs(value) for value in reactions),
        "peak_positive_reaction_y_N": max(0.0, max(reactions)),
        "signed_reaction_impulse_y_Ns": impulse,
        "final_broken_bonds": sim.broken_bonds,
        "total_bonds": int(sim.bonded.sum().item()),
        "peak_kinetic_energy_J": max(float(row["kinetic_energy"]) for row in history),
    }
    return summary, history



def compare_histories(
    candidate: list[dict[str, float | int]],
    reference: list[dict[str, float | int]],
) -> dict[str, float | int]:
    """Compare reaction histories on a shared time interval.

    The reference is typically the finest grid. Candidate samples are linearly
    interpolated against the reference history; this is a sensitivity metric,
    not a formal convergence-order estimate.
    """
    if len(candidate) < 2 or len(reference) < 2:
        raise ValueError("each history must contain at least two samples")
    candidate_times = [float(row["time"]) for row in candidate]
    reference_times = [float(row["time"]) for row in reference]
    if any(b <= a for a, b in zip(candidate_times, candidate_times[1:])):
        raise ValueError("candidate times must be strictly increasing")
    if any(b <= a for a, b in zip(reference_times, reference_times[1:])):
        raise ValueError("reference times must be strictly increasing")
    start = max(candidate_times[0], reference_times[0])
    end = min(candidate_times[-1], reference_times[-1])
    if end <= start:
        raise ValueError("histories must have a non-zero overlapping time interval")

    sample_times = [t for t in candidate_times if start <= t <= end]
    if not sample_times or sample_times[-1] < end:
        sample_times.append(end)

    def interpolate(times: list[float], values: list[float], target: float) -> float:
        import bisect

        right = bisect.bisect_left(times, target)
        if right == 0:
            return values[0]
        if right == len(times):
            return values[-1]
        if times[right] == target:
            return values[right]
        left = right - 1
        fraction = (target - times[left]) / (times[right] - times[left])
        return values[left] + fraction * (values[right] - values[left])

    candidate_force = [float(row["reaction_y"]) for row in candidate]
    reference_force = [float(row["reaction_y"]) for row in reference]
    differences = []
    reference_samples = []
    for t in sample_times:
        candidate_value = interpolate(candidate_times, candidate_force, t)
        reference_value = interpolate(reference_times, reference_force, t)
        differences.append(candidate_value - reference_value)
        reference_samples.append(reference_value)
    diff_rms = math.sqrt(sum(value * value for value in differences) / len(differences))
    ref_rms = math.sqrt(sum(value * value for value in reference_samples) / len(reference_samples))
    relative_rms = diff_rms / max(ref_rms, 1e-12)
    candidate_peak = max(abs(value) for value in candidate_force)
    reference_peak = max(abs(value) for value in reference_force)
    peak_relative = abs(candidate_peak - reference_peak) / max(reference_peak, 1e-12)
    return {
        "overlap_duration_s": end - start,
        "sample_count": len(sample_times),
        "reaction_rms_difference_N": diff_rms,
        "reaction_relative_rms_difference": relative_rms,
        "peak_abs_reaction_relative_difference": peak_relative,
    }

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="二维 DEM 网格/分辨率敏感性扫描")
    parser.add_argument("--nx-levels", default="7,11,15", help="逗号分隔的列数，例如 7,11,15")
    parser.add_argument("--width", type=float, default=0.3, help="目标冰层宽度 (m)")
    parser.add_argument("--thickness", type=float, default=0.2, help="目标冰层高度 (m)")
    parser.add_argument("--duration", type=float, default=0.12, help="目标模拟时长 (s)")
    parser.add_argument("--speed", type=float, default=0.2, help="压头速度 (m/s)")
    parser.add_argument("--device", default="cpu", help="cpu 或 cuda")
    parser.add_argument("--output", type=Path, default=Path("sensitivity-results"))
    args = parser.parse_args(argv)
    try:
        levels = [int(item.strip()) for item in args.nx_levels.split(",")]
        if not levels or len(set(levels)) != len(levels):
            raise ValueError("nx levels must be non-empty and unique")
        if args.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        configs = [
            build_config(nx, args.width, args.thickness, args.speed, args.device)
            for nx in levels
        ]
        if not math.isfinite(args.duration) or args.duration <= 0:
            raise ValueError("duration must be finite and positive")
    except ValueError as exc:
        parser.error(str(exc))

    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    histories = []
    for config in configs:
        summary, history = simulate_case(config, args.duration)
        summaries.append(summary)
        histories.append(history)
        history_path = args.output / f"history_nx_{config.nx}.csv"
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(
            f"nx={config.nx}, ny={config.ny}, particles={summary['particles']}, "
            f"peak |reaction_y|={summary['peak_abs_reaction_y_N']:.6g} N, "
            f"broken={summary['final_broken_bonds']}/{summary['total_bonds']}"
        )

    summary_path = args.output / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    reference_index = max(range(len(configs)), key=lambda index: configs[index].nx)
    reference_nx = configs[reference_index].nx
    comparison_rows = []
    for index, (config, history) in enumerate(zip(configs, histories)):
        metrics = compare_histories(history, histories[reference_index])
        comparison_rows.append({"nx": config.nx, "reference_nx": reference_nx, **metrics})
    comparison_path = args.output / "comparison_to_finest.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"扫描完成。汇总: {summary_path}")
    print(f"分辨率对比: {comparison_path}")


if __name__ == "__main__":
    main()
