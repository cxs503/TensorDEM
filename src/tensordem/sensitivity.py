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
    for config in configs:
        summary, history = simulate_case(config, args.duration)
        summaries.append(summary)
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
    print(f"扫描完成。汇总: {summary_path}")


if __name__ == "__main__":
    main()
