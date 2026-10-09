"""Run the reproducible 2-D DEM demo and export its trajectory/history."""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import torch

from .dem import DEMConfig, IceDEM


def read_hull_profile(path: Path) -> torch.Tensor:
    """Read local hull vertices from a CSV with numeric x,y columns (metres)."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not {"x", "y"}.issubset(reader.fieldnames):
                raise ValueError("hull profile CSV must contain x,y column headers")
            points = []
            for line_number, row in enumerate(reader, start=2):
                try:
                    points.append((float(row["x"]), float(row["y"])))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid x,y value in hull profile CSV line {line_number}"
                    ) from exc
    except OSError as exc:
        raise ValueError(f"cannot read hull profile CSV: {exc}") from exc
    if len(points) < 2:
        raise ValueError("hull profile CSV must contain at least two vertices")
    profile = torch.tensor(points, dtype=torch.float64)
    if not bool(torch.isfinite(profile).all()):
        raise ValueError("hull profile coordinates must be finite")
    if bool((torch.linalg.vector_norm(profile[1:] - profile[:-1], dim=1) <= 1e-12).any()):
        raise ValueError("hull profile must not contain zero-length segments")
    return profile


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PyTorch 二维 DEM 破冰模拟")
    parser.add_argument("--nx", type=int, default=21, help="冰层列数")
    parser.add_argument("--ny", type=int, default=5, help="冰层行数")
    parser.add_argument("--steps", type=int, default=2000, help="积分步数")
    parser.add_argument("--dt", type=float, default=None, help="时间步长 (s)，默认自动")
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="圆形压头向下速度；船体模式下作为时间步长速度下限 (m/s)",
    )
    parser.add_argument("--hull-profile", type=Path, default=None,
                        help="船体局部折线 CSV，要求 x,y 两列，单位 m")
    parser.add_argument("--hull-start-x", type=float, default=0.0,
                        help="船体局部坐标原点初始全局 x (m)")
    parser.add_argument("--hull-start-y", type=float, default=0.0,
                        help="船体局部坐标原点初始全局 y (m)")
    parser.add_argument("--hull-velocity-x", type=float, default=0.0,
                        help="船体规定平移速度 x 分量 (m/s)")
    parser.add_argument("--hull-velocity-y", type=float, default=0.0,
                        help="船体规定平移速度 y 分量 (m/s)")
    parser.add_argument("--breaking-strain", type=float, default=0.015)
    parser.add_argument("--shear-breaking-strain", type=float, default=0.03)
    parser.add_argument("--device", default="cpu", help="cpu 或 cuda")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--save-every", type=int, default=20, help="轨迹采样间隔")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.save_every < 1:
        parser.error("--steps and --save-every must be positive")
    try:
        hull_profile = read_hull_profile(args.hull_profile) if args.hull_profile else None
        hull_speed = (args.hull_velocity_x**2 + args.hull_velocity_y**2) ** 0.5
        speed_bound = max(args.speed, hull_speed, 1e-12)
        config = DEMConfig(
            nx=args.nx, ny=args.ny, dt=args.dt, tool_speed=speed_bound,
            breaking_strain=args.breaking_strain,
            shear_breaking_strain=args.shear_breaking_strain, device=args.device,
        )
        if hull_profile is None:
            simulation = IceDEM(config)
        else:
            simulation = IceDEM(
                config, hull_profile=hull_profile,
                hull_start=(args.hull_start_x, args.hull_start_y),
                hull_velocity=(args.hull_velocity_x, args.hull_velocity_y),
            )
    except (ValueError, RuntimeError, TypeError) as exc:
        parser.error(str(exc))
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
        "hull_profile": None if simulation.hull_profile is None else simulation.hull_profile.cpu(),
        "hull_velocity": simulation.hull_velocity.cpu(),
    }, args.output / "trajectory.pt")
    print(
        f"完成: t={simulation.time:.6f} s, dt={simulation.dt:.6g} s, "
        f"断键={simulation.broken_bonds}/{int(simulation.bonded.sum().item())}; "
        f"结果目录: {args.output}"
    )
