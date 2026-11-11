#!/usr/bin/env python
"""Entry point: run the breakable-floe ridge DEM and write a run artifact.

Usage
-----
    python scripts/run_ice_ridge_rubbling.py --out runs/p318w3_n08 [--steps 20000]

The script is the reproducible driver for the mission's machine-checkable
numbers: it runs the DEM, computes the volume audit, the porosity-versus-strain
fit, the sail/keel geometry and the isostasy prediction, and writes
``ridge_rubbling_run.json`` (history + diagnostics) plus ``volume_audit.json``.
"""
from __future__ import annotations

import argparse
import json
import os

from tensordem.ice_ridge_formation.rubbling import RidgeDEMConfig, run_rubbling_ridge


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/p318w3_n08")
    ap.add_argument("--floes", type=int, default=6)
    ap.add_argument("--thickness", type=float, default=0.5)
    ap.add_argument("--min-length", type=float, default=2.0)
    ap.add_argument("--max-length", type=float, default=6.0)
    ap.add_argument("--servo-force", type=float, default=3.0e4)
    ap.add_argument("--servo-velocity", type=float, default=0.05)
    ap.add_argument("--mass-scale", type=float, default=8.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--samples", type=int, default=2000)
    args = ap.parse_args()

    cfg = RidgeDEMConfig(
        n_floes=args.floes, ice_thickness=args.thickness,
        min_floe_length=args.min_length, max_floe_length=args.max_length,
        servo_force=args.servo_force, servo_velocity=args.servo_velocity,
        mass_scale=args.mass_scale, max_steps=args.steps,
        sample_every=args.samples)
    res = run_rubbling_ridge(cfg)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ridge_rubbling_run.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    audit = {
        "criterion": "initial ice volume == final ice volume, loss < 1%",
        "initial_volume_m3_per_m": res["volume"]["initial_volume"],
        "final_volume_m3_per_m": res["volume"]["final_volume"],
        "relative_error": res["volume"]["rel_error"],
        "pass": bool(res["volume"]["rel_error"] < 0.01),
        "n_fractures": res["n_fractures"],
        "n_blocks": res["final"]["n_blocks"],
        "strain": res["final"]["strain"],
    }
    with open(os.path.join(args.out, "volume_audit.json"), "w") as fh:
        json.dump(audit, fh, indent=2)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
