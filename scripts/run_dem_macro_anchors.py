#!/usr/bin/env python
"""Run one DEM macro-anchor case (frozen sandpile protocol) and dump JSON.

The script is deliberately single-case so the campaign scale scan can be sharded
over the cluster:

    python scripts/run_dem_macro_anchors.py --label N1e4 --n 10000 \
        --pour-time 1.0 --settle 0.2 --out artifacts/macro/cases/N1e4.json

Cases (label -> arguments) used by the campaign are listed in
``docs/dem_macro_anchor_closure.md``; the aggregation step is
``scripts/aggregate_dem_macro_anchors.py``.

All contact parameters are the frozen literature priors of
``tensordem.protocols.sandpile.PRIORS``; a case can only deviate through a
whitelisted *negative control* (``--control``), which is tagged in the record.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from tensordem.protocols.sandpile import (
    PRIORS,
    SandpileConfig,
    SandpileRun,
    SandpileProtocolError,
)


def build_config(args: argparse.Namespace) -> SandpileConfig:
    pour_rate = args.rate if args.rate is not None else args.n / args.pour_time
    kwargs = dict(
        n_particles=int(args.n),
        radius=args.radius,
        density=args.density,
        restitution=args.restitution,
        pour_rate=pour_rate,
        drop_height=args.drop_height,
        nozzle_radius=args.nozzle_radius,
        settle_time=args.settle,
        sector_count=args.sectors,
        control=args.control,
        seed=args.seed,
    )
    if args.control == "undamped":
        kwargs["restitution"] = 1.0
    return SandpileConfig(**kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--pour-time", type=float, default=1.0,
                    help="pour duration, s (pour_rate = n / pour_time)")
    ap.add_argument("--rate", type=float, default=None,
                    help="explicit pour rate, grains/s (overrides --pour-time)")
    ap.add_argument("--settle", type=float, default=0.20)
    ap.add_argument("--radius", type=float, default=0.005)
    ap.add_argument("--density", type=float, default=2500.0)
    ap.add_argument("--restitution", type=float, default=0.3)
    ap.add_argument("--drop-height", type=float, default=0.04)
    ap.add_argument("--nozzle-radius", type=float, default=None)
    ap.add_argument("--sectors", type=int, default=16)
    ap.add_argument("--control", default="baseline")
    ap.add_argument("--seed", type=int, default=6932)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.nozzle_radius is None:
        # The nozzle scales with the deposit: a fixed 3 cm nozzle cannot carry
        # 1e5 grains/s, and a nozzle wider than the pile would spread the stream
        # over the whole heap.
        volume = args.n * (4.0 / 3.0 * math.pi * args.radius ** 3) / 0.6
        heap_radius = (3.0 * volume / (math.pi * math.tan(math.radians(30.0)))) ** (1.0 / 3.0)
        args.nozzle_radius = max(0.03, 0.28 * heap_radius)
    cfg = build_config(args)
    run = SandpileRun(cfg)
    t0 = time.time()
    record = run.run()
    record["label"] = args.label
    record["wall_time_s"] = time.time() - t0
    record["priors_used"] = {"mu_s": float(cfg.mu_s), "mu_r": float(cfg.mu_r)}
    record["priors_match_file"] = (
        abs(float(cfg.mu_s) - PRIORS["mu_s"]) == 0.0
        and abs(float(cfg.mu_r) - PRIORS["mu_r"]) == 0.0
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(record, fh, indent=1)
    print(json.dumps({
        "label": record["label"],
        "control": record["control"],
        "n_particles": record["n_particles"],
        "poured": record["poured"],
        "repose_deg": record["repose_deg"],
        "repose_r2": record["repose_fit"]["r2"],
        "theta_stderr_deg": record["repose_fit"]["theta_stderr_deg"],
        "packing_probe": record["packing"]["packing_fraction"],
        "packing_bulk": record["packing"]["bulk"]["packing_fraction"],
        "coordination": record["fabric"]["coordination_number"],
        "wall_time_s": record["wall_time_s"],
        "steps": record["steps"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())