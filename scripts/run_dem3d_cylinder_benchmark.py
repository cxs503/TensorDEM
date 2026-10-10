"""Reproducible 3-D circular-cylinder diametral-compression benchmark.

Run from a source checkout after installing TensorDEM:
    python scripts/run_dem3d_cylinder_benchmark.py --output results-cylinder
The solver writes the raw history/checkpoint/trajectory; this driver adds
force-displacement data and a machine-readable summary without replacing raw data.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from tensordem.dem3d_cli import main as dem3d_main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results-cylinder-compression"))
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--nx", type=int, default=9)
    parser.add_argument("--ny", type=int, default=3, help="cylinder thickness in particle layers")
    parser.add_argument("--nz", type=int, default=9)
    parser.add_argument("--radius", type=float, default=0.025)
    parser.add_argument("--platen-speed", type=float, default=0.02)
    parser.add_argument("--breaking-strain", type=float, default=0.015)
    parser.add_argument("--shear-breaking-strain", type=float, default=0.03)
    args = parser.parse_args()
    if args.steps < 1 or args.save_every < 1:
        parser.error("--steps and --save-every must be positive")
    if args.platen_speed <= 0 or not math.isfinite(args.platen_speed):
        parser.error("--platen-speed must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=True)
    dem3d_main([
        "--geometry", "cylinder_xz", "--nx", str(args.nx), "--ny", str(args.ny),
        "--nz", str(args.nz), "--radius", str(args.radius),
        "--breaking-strain", str(args.breaking_strain),
        "--shear-breaking-strain", str(args.shear_breaking_strain),
        "--steps", str(args.steps), "--save-every", str(args.save_every),
        "--top-platen", "--bottom-platen",
        "--top-platen-velocity", str(-args.platen_speed),
        "--bottom-platen-velocity", str(args.platen_speed),
        "--output", str(args.output),
    ])
    history_path = args.output / "history.csv"
    with history_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"solver produced an empty history: {history_path}")
    required = ("time_s", "top_platen_reaction_z_N", "bottom_platen_reaction_z_N",
                "broken_bonds")
    missing = [key for key in required if key not in rows[0]]
    if missing:
        raise RuntimeError(f"history is missing required diagnostics: {missing}")
    curve_path = args.output / "load_displacement.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["time_s", "compression_displacement_m", "top_load_N",
                  "bottom_load_N", "broken_bonds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            t = float(row["time_s"])
            writer.writerow({
                "time_s": t,
                "compression_displacement_m": 2.0 * args.platen_speed * t,
                "top_load_N": abs(float(row["top_platen_reaction_z_N"])),
                "bottom_load_N": abs(float(row["bottom_platen_reaction_z_N"])),
                "broken_bonds": int(float(row["broken_bonds"])),
            })
    top_loads = [abs(float(row["top_platen_reaction_z_N"])) for row in rows]
    bottom_loads = [abs(float(row["bottom_platen_reaction_z_N"])) for row in rows]
    broken = [int(float(row["broken_bonds"])) for row in rows]
    summary = {
        "benchmark": "dem3d_cylinder_xz_diametral_compression",
        "model": "3-D bonded-sphere DEM; circular x-z cross-section; finite y thickness",
        "units": {"length": "m", "time": "s", "force": "N", "energy": "J"},
        "configuration": {
            "nx": args.nx, "ny": args.ny, "nz": args.nz, "radius_m": args.radius,
            "platen_speed_m_s_each": args.platen_speed,
            "breaking_strain": args.breaking_strain,
            "shear_breaking_strain": args.shear_breaking_strain,
            "steps": args.steps, "save_every": args.save_every,
        },
        "samples": len(rows),
        "final_time_s": float(rows[-1]["time_s"]),
        "peak_top_load_N": max(top_loads),
        "peak_bottom_load_N": max(bottom_loads),
        "final_broken_bonds": broken[-1],
        "peak_broken_bonds": max(broken),
        "files": ["history.csv", "load_displacement.csv", "checkpoint_3d.pt",
                  "trajectory_3d.pt"],
        "validation_note": (
            "Reproducibility/software benchmark, not experimental validation. "
            "Material stiffness and bond thresholds require calibration; "
            "compare load-displacement and crack paths with a cited test before "
            "engineering predictions."
        ),
    }
    (args.output / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
