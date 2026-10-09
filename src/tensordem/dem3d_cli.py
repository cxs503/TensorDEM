"""Command-line driver for the 3-D DEM prototype."""
import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import torch
from .dem3d import DEM3DConfig, IceDEM3D


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PyTorch 三维黏结颗粒 DEM 破冰原型")
    parser.add_argument("--nx", type=int, default=9, help="x 方向颗粒数")
    parser.add_argument("--ny", type=int, default=5, help="y 方向颗粒数")
    parser.add_argument("--nz", type=int, default=3, help="z 方向颗粒数")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=None, help="时间步长 (s)，默认自动")
    parser.add_argument("--speed", type=float, default=0.5, help="压头向下速度 (m/s)")
    parser.add_argument("--radius", type=float, default=0.025, help="颗粒半径 (m)")
    parser.add_argument("--tool-radius", type=float, default=0.1, help="球形压头半径 (m)")
    parser.add_argument("--breaking-strain", type=float, default=0.015)
    parser.add_argument("--shear-breaking-strain", type=float, default=0.03)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results-3d"))
    parser.add_argument("--restart", type=Path, default=None,
                        help="从之前保存的 checkpoint_3d.pt 续算；物理参数以检查点为准")
    parser.add_argument("--fix-bottom", action="store_true", help="固定底部颗粒层")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.save_every < 1:
        parser.error("--steps and --save-every must be positive")
    try:
        if args.restart is not None:
            if not args.restart.is_file():
                parser.error(f"restart checkpoint not found: {args.restart}")
            checkpoint = torch.load(args.restart, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, dict) or "state" not in checkpoint:
                parser.error("restart file must be a TensorDEM 3-D checkpoint with a 'state' entry")
            saved_config = checkpoint["state"].get("config")
            if not isinstance(saved_config, dict):
                parser.error("restart checkpoint has no valid saved configuration")
            config = DEM3DConfig(**{**saved_config, "device": args.device})
            sim = IceDEM3D(config)
            sim.load_state_dict(checkpoint["state"])
        else:
            config = DEM3DConfig(nx=args.nx, ny=args.ny, nz=args.nz, dt=args.dt,
                tool_speed=args.speed, radius=args.radius, tool_radius=args.tool_radius,
                breaking_strain=args.breaking_strain,
                shear_breaking_strain=args.shear_breaking_strain,
                device=args.device, fix_bottom=args.fix_bottom)
            sim = IceDEM3D(config)
    except (ValueError, RuntimeError, TypeError, OSError, KeyError) as exc:
        parser.error(str(exc))
    if sim.device.type == "cpu":
        torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    frames, velocities, alive_frames, times, tool_positions = [], [], [], [], []
    history = args.output / "history.csv"
    append_history = args.restart is not None and history.exists()
    with history.open("a" if append_history else "w", newline="", encoding="utf-8") as stream:
        writer = None
        for step in range(args.steps + 1):
            if step:
                sim.step()
            if (step % args.save_every == 0 or step == args.steps) and not (append_history and step == 0):
                row = sim.diagnostics()
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    if not append_history:
                        writer.writeheader()
                writer.writerow(row)
                frames.append(sim.positions.cpu().clone())
                velocities.append(sim.velocities.cpu().clone())
                alive_frames.append(sim.alive[sim.bonded].cpu().clone())
                times.append(sim.time)
                tool_positions.append(sim.tool_position.cpu().clone())
                torch.save({"format": "tensordem-dem3d-checkpoint-v1", "state": sim.state_dict()},
                           args.output / "checkpoint_3d.pt")
    torch.save({
        "dimension": 3, "config": asdict(config), "dt": sim.dt,
        "times": torch.tensor(times, dtype=torch.float64),
        "positions": torch.stack(frames), "velocities": torch.stack(velocities),
        "bond_pairs": sim.pairs[sim.bonded].cpu(),
        "bond_alive": torch.stack(alive_frames), "fixed": sim.fixed.cpu(),
        "tool_positions": torch.stack(tool_positions),
        "failure_mode": sim.failure_mode.cpu(),
        "failure_time_s": sim.failure_time_s.cpu(),
        "failure_extension_m": sim.failure_extension_m.cpu(),
        "failure_energy_J": sim.failure_energy_J.cpu(),
    }, args.output / "trajectory_3d.pt")
    print(f"3-D 完成: t={sim.time:.6g} s, dt={sim.dt:.4g} s, "
          f"particles={len(sim.positions)}, broken={sim.broken_bonds}, output={args.output}")


if __name__ == "__main__":
    main()
