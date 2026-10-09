"""Run the reproducible ice indentation demo and export its trajectory."""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import torch

from .dem import DEMConfig, IceDEM


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PyTorch 二维 DEM 破冰模拟")
    parser.add_argument("--nx", type=int, default=21, help="冰层列数")
    parser.add_argument("--ny", type=int, default=5, help="冰层行数")
    parser.add_argument("--steps", type=int, default=2000, help="积分步数")
    parser.add_argument("--dt", type=float, default=None, help="时间步长 (s)，默认自动")
    parser.add_argument("--speed", type=float, default=1.0, help="破冰头速度 (m/s)")
    parser.add_argument("--breaking-strain", type=float, default=0.015)
    parser.add_argument("--shear-breaking-strain", type=float, default=0.03)
    parser.add_argument("--device", default="cpu", help="cpu 或 cuda")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--save-every", type=int, default=20, help="轨迹采样间隔")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.save_every < 1:
        parser.error("--steps and --save-every must be positive")
    try:
        simulation = IceDEM(DEMConfig(
            nx=args.nx, ny=args.ny, dt=args.dt, tool_speed=args.speed,
            breaking_strain=args.breaking_strain,
            shear_breaking_strain=args.shear_breaking_strain, device=args.device,
        ))
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    # Small all-pairs workloads avoid CPU thread-pool overhead.
    if simulation.device.type == "cpu":
        torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    frames, velocities, bonds, tool_positions, times = [], [], [], [], []
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        row = simulation.diagnostics()
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        for step in range(args.steps + 1):
            if step:
                simulation.step()
            if step % args.save_every == 0 or step == args.steps:
                row = simulation.diagnostics()
                writer.writerow(row)
                frames.append(simulation.positions.cpu().clone())
                velocities.append(simulation.velocities.cpu().clone())
                bonds.append(simulation.alive[simulation.bonded].cpu().clone())
                tool_positions.append(simulation.tool_position.cpu().clone())
                times.append(simulation.time)
    torch.save({
        "config": asdict(simulation.config),
        "dt": simulation.dt,
        "times": torch.tensor(times, dtype=torch.float64),
        "positions": torch.stack(frames),
        "velocities": torch.stack(velocities),
        "bond_pairs": simulation.pairs[simulation.bonded].cpu(),
        "bond_alive": torch.stack(bonds),
        "fixed": simulation.fixed.cpu(),
        "tool_positions": torch.stack(tool_positions),
    }, args.output / "trajectory.pt")
    print(
        f"完成: t={simulation.time:.6f} s, dt={simulation.dt:.6g} s, "
        f"断键={simulation.broken_bonds}/{int(simulation.bonded.sum().item())}; "
        f"结果目录: {args.output}"
    )
