"""CLI entry points for TensorDEM: 2-D ice-breaking DEM and verification tools."""

import argparse
import csv
import json
import sys
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


def _cmd_dem_2d(args: argparse.Namespace) -> None:
    """Run the reproducible 2-D DEM ice-breaking simulation."""
    if args.steps < 1 or args.save_every < 1:
        sys.exit("--steps and --save-every must be positive")
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
        sys.exit(str(exc))
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
    accounting_data = dict(simulation.accounting)
    accounting_data["total_bonds"] = int(simulation.bonded.sum().item())
    accounting_data["broken_bonds"] = simulation.broken_bonds
    accounting_data["failure_energy_sum_J"] = float(simulation.failure_energy_J.sum().item())
    accounting_data["total_mass_kg"] = float(simulation.config.mass * len(simulation.positions))
    accounting_data["dt"] = simulation.dt
    accounting_data["steps"] = simulation.step_count
    with (args.output / "accounting.json").open("w", encoding="utf-8") as f:
        f.write(json.dumps(accounting_data, indent=2, sort_keys=True, allow_nan=False))
    print(
        f"完成: t={simulation.time:.6f} s, dt={simulation.dt:.6g} s, "
        f"断键={simulation.broken_bonds}/{int(simulation.bonded.sum().item())}; "
        f"结果目录: {args.output}"
    )


def _cmd_verify_friction_polyhedron(args: argparse.Namespace) -> None:
    """Run friction+polyhedron DEM verification and export machine-checkable report."""
    from .friction_polyhedron_real_verification import run_full_verification

    args.output.mkdir(parents=True, exist_ok=True)
    report = run_full_verification(include_packing=args.packing)

    # Write full JSON report
    report_path = args.output / "friction_polyhedron_verification.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # Write summary for quick inspection
    summary_path = args.output / "friction_polyhedron_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Friction + Polyhedron DEM Verification Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Campaign node: {report['campaign_node']}\n")
        f.write(f"Corrected on: {report['corrected_on']}\n")
        f.write(f"Overall: {report['overall']}\n\n")
        for criterion, status in report['summary'].items():
            f.write(f"  {criterion}: {status}\n")
        f.write("\nKnown limitations:\n")
        for key, val in report.get('known_limitations', {}).items():
            f.write(f"  {key}: {val}\n")

    # Print summary to stdout
    print(f"Verification complete: overall={report['overall']}")
    for criterion, status in report['summary'].items():
        print(f"  {criterion}: {status}")
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")


def _cmd_h1_broken_ice(args: argparse.Namespace) -> None:
    """Run the H1-BROKEN-ICE-DEM cases (buoyancy / repose / channel)."""
    from .polyhedron_floe3d.h1_cases import validate

    cases = [c for c in args.cases.split(",") if c]
    kwargs = {}
    if args.quick:
        kwargs = {}
    else:
        kwargs = {"repose": {"n_steps": args.repose_steps, "n_bodies": args.repose_bodies},
                  "channel": {"n_steps": args.channel_steps}}
    result = validate(cases=cases, quick=args.quick, **kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, default=float))
    print("cases:", result["passed"])
    print("report:", args.output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="TensorDEM: Discrete Element Method solver and verification tools",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- dem-2d subcommand ---
    dem_parser = subparsers.add_parser("dem-2d", help="2-D ice-breaking DEM simulation")
    dem_parser.add_argument("--nx", type=int, default=21, help="冰层列数")
    dem_parser.add_argument("--ny", type=int, default=5, help="冰层行数")
    dem_parser.add_argument("--steps", type=int, default=2000, help="积分步数")
    dem_parser.add_argument("--dt", type=float, default=None, help="时间步长 (s)，默认自动")
    dem_parser.add_argument(
        "--speed", type=float, default=1.0,
        help="圆形压头向下速度；船体模式下作为时间步长速度下限 (m/s)",
    )
    dem_parser.add_argument("--hull-profile", type=Path, default=None,
                            help="船体局部折线 CSV，要求 x,y 两列，单位 m")
    dem_parser.add_argument("--hull-start-x", type=float, default=0.0,
                            help="船体局部坐标原点初始全局 x (m)")
    dem_parser.add_argument("--hull-start-y", type=float, default=0.0,
                            help="船体局部坐标原点初始全局 y (m)")
    dem_parser.add_argument("--hull-velocity-x", type=float, default=0.0,
                            help="船体规定平移速度 x 分量 (m/s)")
    dem_parser.add_argument("--hull-velocity-y", type=float, default=0.0,
                            help="船体规定平移速度 y 分量 (m/s)")
    dem_parser.add_argument("--breaking-strain", type=float, default=0.015)
    dem_parser.add_argument("--shear-breaking-strain", type=float, default=0.03)
    dem_parser.add_argument("--device", default="cpu", help="cpu 或 cuda")
    dem_parser.add_argument("--output", type=Path, default=Path("results"))
    dem_parser.add_argument("--save-every", type=int, default=20, help="轨迹采样间隔")
    dem_parser.set_defaults(func=_cmd_dem_2d)

    # --- verify-friction-polyhedron subcommand ---
    verify_parser = subparsers.add_parser(
        "verify-friction-polyhedron",
        help="Run friction+polyhedron DEM verification (A1-A4 anchors)",
    )
    verify_parser.add_argument(
        "--packing", action="store_true",
        help="Include A4 packing-fraction measurement (slow, needs many particles)",
    )
    verify_parser.add_argument(
        "--output", type=Path, default=Path("runs/friction_polyhedron"),
        help="Output directory for verification report",
    )
    verify_parser.set_defaults(func=_cmd_verify_friction_polyhedron)

    # --- h1-broken-ice subcommand (in-repo caller of the H1 case builders) ---
    h1_parser = subparsers.add_parser(
        "h1-broken-ice",
        help="Broken-ice DEM cases: floating equilibrium, rubble repose, channel clearing",
    )
    h1_parser.add_argument("--cases", default="buoyancy,repose,channel",
                           help="comma separated subset of buoyancy,repose,channel")
    h1_parser.add_argument("--quick", action="store_true",
                           help="reduced case sizes (how the pytest gate runs it)")
    h1_parser.add_argument("--repose-steps", type=int, default=6000)
    h1_parser.add_argument("--repose-bodies", type=int, default=45)
    h1_parser.add_argument("--channel-steps", type=int, default=900)
    h1_parser.add_argument("--output", type=Path,
                           default=Path("artifacts/h1_broken_ice_dem/cli_report.json"))
    h1_parser.set_defaults(func=_cmd_h1_broken_ice)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)
