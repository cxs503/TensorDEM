"""Brash-ice channel campaign runner (W4-N07).

Sub-commands
------------
``matrix``       one DEM channel-resistance case of the (concentration, thickness) sweep
``convoy``       one two-hull convoy case (icebreaker + escort at a fixed spacing)
``convergence``  one particle-resolution case (50k / 100k pieces, same physics)

Every case writes a self-describing JSON record under ``--case-dir``; the
aggregation (``scripts/channel_aggregate.py``) never re-runs a simulation and
never hand-copies a number.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem.brash_channel import BrashChannelConfig, BrashChannelDEM, HullSpec  # noqa: E402

# ---------------------------------------------------------------- shared set-up
CHANNEL_WIDTH_M = 4.6
BEAM_M = 4.0
BOW_HALF_ANGLE_DEG = 60.0
TIP_RADIUS_M = 0.12
BODY_LENGTH_M = 1.2
SPEED_MS = 0.5
MEAN_RADIUS_M = 0.13
MASS_SCALE = 100.0
DRAG_COEFFICIENT = 40.0
NORMAL_STIFFNESS = 1.0e6
BOW_LENGTH_M = (0.5 * BEAM_M) / math.tan(math.radians(BOW_HALF_ANGLE_DEG))
SHIP_LENGTH_M = BOW_LENGTH_M + BODY_LENGTH_M


def base_config(**overrides: Any) -> BrashChannelConfig:
    params: dict[str, Any] = dict(
        length_m=5.0,
        width_m=CHANNEL_WIDTH_M,
        mean_radius_m=MEAN_RADIUS_M,
        radius_spread=0.2,
        mass_scale=MASS_SCALE,
        normal_stiffness=NORMAL_STIFFNESS,
        drag_coefficient=DRAG_COEFFICIENT,
        floor_depth_m=3.0,
        seed=20261031,
    )
    params.update(overrides)
    return BrashChannelConfig(**params)


def make_hull(x_apex: float = -0.35, name: str = "icebreaker") -> HullSpec:
    return HullSpec(
        beam_m=BEAM_M,
        bow_half_angle_deg=BOW_HALF_ANGLE_DEG,
        bow_length_m=BOW_LENGTH_M,
        body_length_m=BODY_LENGTH_M,
        tip_radius_m=TIP_RADIUS_M,
        y_center_m=0.5 * CHANNEL_WIDTH_M,
        x_apex_m=x_apex,
        speed_ms=SPEED_MS,
        name=name,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def summarise(out: dict[str, Any], wall: float, extra: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "wall_time_s": wall,
        "concentration": out["concentration"],
        "thickness_m": out["thickness_m"],
        "particle_count": out["particle_count"],
        "mean_radius_m": out["mean_radius_m"],
        "dt_s": out["dt_s"],
        "steps_measured": out["steps_measured"],
        "steps_window": out["steps_window"],
        "samples": out["samples"],
        "mean_resistance_n": out["mean_resistance_n"],
        "std_resistance_n": out["std_resistance_n"],
        "relative_std": out["std_resistance_n"] / max(1e-9, out["mean_resistance_n"]),
        "hull_resistances_n": out["hull_resistances_n"],
        "hull_resistance_std_n": out["hull_resistance_std_n"],
        "mean_ice_speed_m_s": out["mean_ice_speed_m_s"],
        "max_ice_speed_m_s": out["max_ice_speed_m_s"],
        "max_overlap_ratio": out["max_overlap_ratio"],
        "clamped_contacts": out["clamped_contacts"],
        "energy_ledger": out["energy_ledger"],
        "hull_stats": out["hull_stats"],
        "beam_m": BEAM_M,
        "velocity_ms": SPEED_MS,
        "ship_length_m": SHIP_LENGTH_M,
        "channel_width_m": CHANNEL_WIDTH_M,
        "mass_scale": out.get("mass_scale"),
    }
    record.update({k: v for k, v in extra.items() if k not in record})
    return record


# -------------------------------------------------------------------- commands
def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    config = base_config(
        length_m=args.length_m,
        concentration=args.concentration,
        thickness_m=args.thickness_m,
        mass_scale=args.mass_scale,
    )
    dem = BrashChannelDEM(config, [make_hull()])
    start = time.time()
    out = dem.run(
        settle_steps=args.settle_steps,
        spinup_travel_m=args.spinup_travel_m,
        measure_travel_m=args.measure_travel_m,
        sample_interval=args.sample_interval,
    )
    case_id = f"matrix_C{args.concentration:.2f}_h{args.thickness_m:.2f}"
    if args.tag:
        case_id += f"_{args.tag}"
    return summarise(
        out,
        time.time() - start,
        {
            "case_id": case_id,
            "kind": "matrix",
            "length_m": args.length_m,
            "measure_travel_m": args.measure_travel_m,
            "mass_scale": args.mass_scale,
        },
    )


def run_convoy(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    spacing = args.spacing_over_length * SHIP_LENGTH_M
    config = base_config(
        length_m=args.length_m,
        concentration=args.concentration,
        thickness_m=args.thickness_m,
        mass_scale=args.mass_scale,
    )
    leader = make_hull(x_apex=-0.35, name="icebreaker")
    follower = make_hull(x_apex=-0.35 - spacing, name="escort")
    dem = BrashChannelDEM(config, [leader, follower])
    start = time.time()
    out = dem.run(
        settle_steps=args.settle_steps,
        lead_in_m=spacing + args.spinup_travel_m,
        spinup_travel_m=0.0,
        measure_travel_m=args.measure_travel_m,
        sample_interval=args.sample_interval,
        sample_hull=1,
    )
    case_id = f"convoy_s{args.spacing_over_length:.2f}L"
    return summarise(
        out,
        time.time() - start,
        {
            "case_id": case_id,
            "kind": "convoy",
            "spacing_m": spacing,
            "spacing_over_length": args.spacing_over_length,
            "follower_resistance_n": out["mean_resistance_n"],
            "follower_resistance_std_n": out["std_resistance_n"],
            "leader_resistance_n": out["hull_resistances_n"][0],
            "length_m": args.length_m,
            "mass_scale": args.mass_scale,
        },
    )


def run_convergence(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    count = args.particle_count
    ice_volume = args.concentration * args.length_m * args.width_m * args.thickness_m
    mean_volume = ice_volume / count
    radius = (3.0 * mean_volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    config = base_config(
        length_m=args.length_m,
        width_m=args.width_m,
        concentration=args.concentration,
        thickness_m=args.thickness_m,
        mean_radius_m=radius,
        mass_scale=args.mass_scale,
    )
    dem = BrashChannelDEM(config, [make_hull()])
    start = time.time()
    out = dem.run(
        settle_steps=args.settle_steps,
        spinup_travel_m=args.spinup_travel_m,
        measure_travel_m=args.measure_travel_m,
        sample_interval=args.sample_interval,
    )
    return summarise(
        out,
        time.time() - start,
        {
            "case_id": f"convergence_N{count}",
            "kind": "convergence",
            "target_particle_count": count,
            "mean_radius_target_m": radius,
            "length_m": args.length_m,
            "mass_scale": args.mass_scale,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--case-dir", type=Path, default=Path("runs/channel_cases"))
        p.add_argument("--threads", type=int, default=2)
        p.add_argument("--length-m", type=float, default=5.0)
        p.add_argument("--width-m", type=float, default=CHANNEL_WIDTH_M)
        p.add_argument("--concentration", type=float, default=0.6)
        p.add_argument("--thickness-m", type=float, default=1.0)
        p.add_argument("--mass-scale", type=float, default=MASS_SCALE)
        p.add_argument("--settle-steps", type=int, default=600)
        p.add_argument("--spinup-travel-m", type=float, default=0.35)
        p.add_argument("--measure-travel-m", type=float, default=0.5)
        p.add_argument("--sample-interval", type=int, default=10)

    p_matrix = sub.add_parser("matrix")
    add_common(p_matrix)
    p_matrix.add_argument("--tag", type=str, default="")
    p_matrix.set_defaults(func=run_matrix)

    p_convoy = sub.add_parser("convoy")
    add_common(p_convoy)
    p_convoy.add_argument("--spacing-over-length", type=float, required=True)
    p_convoy.set_defaults(func=run_convoy)

    p_conv = sub.add_parser("convergence")
    add_common(p_conv)
    p_conv.add_argument("--particle-count", type=int, required=True)
    p_conv.set_defaults(func=run_convergence)

    args = parser.parse_args(argv)
    record = args.func(args)
    out_path = args.case_dir / f"{record['case_id']}.json"
    write_json(out_path, record)
    print(json.dumps({k: record[k] for k in ("case_id", "mean_resistance_n", "wall_time_s")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())