"""Command-line interface for the formation ice-channel framework.

Example
-------
    tensordem-formation --ships 3 --spacing 2.0 --thickness 1.0 --speed 2.0 \
        --out runs/formation_case.json

The command writes the full result bundle (per-ship resistance time histories,
interference matrix, channel evolution) as JSON and prints a one-line summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .formation import FormationScenario, simulate_formation, write_result
from .formation_resistance import IceResistanceModel, IceSheet, IcebreakerShip


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ships", type=int, default=3, help="number of ships in the formation")
    parser.add_argument("--length", type=float, default=100.0, help="ship length (m)")
    parser.add_argument("--beam", type=float, default=20.0, help="ship beam (m)")
    parser.add_argument("--draft", type=float, default=8.0, help="ship draft (m)")
    parser.add_argument("--spacing", type=float, default=2.0,
                        help="centre-to-centre spacing in ship lengths")
    parser.add_argument("--lateral", type=float, default=0.0,
                        help="lateral offset in beams (staggered layout only)")
    parser.add_argument("--layout", choices=("column", "staggered"), default="column")
    parser.add_argument("--speed", type=float, default=2.0, help="formation speed (m/s)")
    parser.add_argument("--thickness", type=float, default=1.0, help="parent ice thickness (m)")
    parser.add_argument("--air-temperature", type=float, default=-20.0,
                        help="air temperature (C)")
    parser.add_argument("--scale-factor", type=float, default=1.0,
                        help="resistance-law scale factor")
    parser.add_argument("--passes", type=int, default=1, help="number of passes over the track")
    parser.add_argument("--pass-interval", type=float, default=3.6e3,
                        help="seconds between passes")
    parser.add_argument("--reuse-channel", action="store_true",
                        help="follow a channel created by an earlier convoy")
    parser.add_argument("--initial-channel-age", type=float, default=0.0,
                        help="age (s) of the reused channel at mission start")
    parser.add_argument("--transit-length", type=float, default=2.0e4, help="track length (m)")
    parser.add_argument("--out", default="formation_result.json", help="output JSON path")
    return parser


def scenario_from_args(args: argparse.Namespace) -> FormationScenario:
    ships = [IcebreakerShip(name=f"P{index + 1}", length=args.length, beam=args.beam,
                            draft=args.draft) for index in range(args.ships)]
    return FormationScenario(
        ships=ships, speed=args.speed, parent_thickness=args.thickness,
        ice=IceSheet(), resistance_model=IceResistanceModel(scale_factor=args.scale_factor),
        spacing_factor=args.spacing, lateral_factor=args.lateral, layout=args.layout,
        passes=args.passes, pass_interval_s=args.pass_interval,
        reuse_previous_channel=args.reuse_channel,
        initial_channel_age_s=args.initial_channel_age,
        transit_length=args.transit_length)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenario = scenario_from_args(args)
    result = simulate_formation(scenario)
    path = write_result(result, Path(args.out))
    summary = result["summary"]
    print(json.dumps({
        "output": str(path),
        "ships": summary["ships"],
        "spacings_m": summary["spacings_m"],
        "leader_ice_resistance_kN": summary["leader_ice_resistance_N"] / 1000.0,
        "mean_follower_resistance_kN": summary["mean_follower_resistance_N"] / 1000.0,
        "follower_saving_vs_level_ice": summary["follower_saving_vs_level_ice"],
        "total_energy_GJ": summary["total_fuel_energy_GJ"],
        "mission_duration_h": summary["mission_duration_h"],
        "interference_neumann_residual": result["interference"]["neumann_residual"],
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())