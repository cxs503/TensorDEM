"""Scan the macroscopic repose / shear anchors used by the shape error budget.

Runs one (or a list of) macroscopic DEM experiment(s) and writes the raw result
as JSON so the artifact builder can assemble ``artifacts/dem/shape_error_table.json``
without re-running the simulations.

Examples
--------
    python scripts/shape_macro_scan.py --mode drum --mu 0.40 --seed 3 \
        --duration 6.0 --out runs/macro_drum_mu040_s3.json
    python scripts/shape_macro_scan.py --mode shear --mu 0.40 --out runs/macro_shear.json
    python scripts/shape_macro_scan.py --mode collapse --mu 0.40 --radius 0.007
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.shape_macro import (  # noqa: E402
    DiskConfig,
    collapse_column_repose,
    drum_repose_angle,
    shear_box_friction_angle,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("drum", "shear", "collapse"), required=True)
    ap.add_argument("--mu", type=float, default=0.40)
    ap.add_argument("--mu-rolling", type=float, default=0.05)
    ap.add_argument("--mu-wall", type=float, default=None)
    ap.add_argument("--kn", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n-disks", type=int, default=320)
    ap.add_argument("--angular-velocity", type=float, default=0.6)
    ap.add_argument("--settle-time", type=float, default=1.5)
    ap.add_argument("--measure-time", type=float, default=4.0)
    ap.add_argument("--shear-velocity", type=float, default=2.0e-3)
    ap.add_argument("--radius", type=float, default=0.007, help="collapse column particle radius")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    started = time.time()
    if args.mode == "drum":
        cfg = DiskConfig(kn=args.kn, mu=args.mu, mu_rolling=args.mu_rolling,
                         mu_wall=args.mu_wall)
        result = drum_repose_angle(cfg, n_disks=args.n_disks, seed=args.seed,
                                   angular_velocity=args.angular_velocity,
                                   settle_time=args.settle_time,
                                   measure_time=args.measure_time)
    elif args.mode == "shear":
        cfg = DiskConfig(kn=args.kn, mu=args.mu, mu_rolling=args.mu_rolling,
                         mu_wall=args.mu_wall)
        result = shear_box_friction_angle(cfg, n_disks=args.n_disks, seed=args.seed,
                                          shear_velocity=args.shear_velocity)
    else:
        result = collapse_column_repose(args.mu, args.mu_rolling, args.radius,
                                        kn=args.kn)
    result["wall_clock_s"] = round(time.time() - started, 1)
    result["argv"] = sys.argv[1:]
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())