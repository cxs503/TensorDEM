"""Run ONE friction shard of the H1 rubble-repose case and write its JSON.

The C2 criterion of the `H1-BROKEN-ICE-DEM` anchor (heap angle of repose of a
polyhedral ice-rubble pile) is compute bound: a settled pile needs ~6 s of
simulated time at ``dt = 2e-4`` s, i.e. 30 000 steps.  This runner isolates a
single ``mu`` so the three shards can go to three processes / three cluster
jobs, and writes its evidence under ``runs/h1_repose/`` (``artifacts/`` is
filtered by the merge gate, so committed evidence never lives there).

    PYTHONPATH=src python scripts/run_h1_repose.py --shard 1
    PYTHONPATH=src python scripts/run_h1_repose.py --shard 3 --steps 30000

Exit code: 0 always (this is an evidence producer, not a CI gate); the verdict
is in the JSON (``settled``, ``repose_angle_deg``, ``inside_literature_band``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.polyhedron_floe3d.h1_cases import repose_case  # noqa: E402

#: friction sweep fixed in ``docs/H1_BROKEN_ICE_DEM_ROUND2.md`` section 4
FRICTION_GRID = (0.2, 0.4, 0.6)

OUT_DIR = os.path.join(ROOT, "runs", "h1_repose")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True, choices=range(1, len(FRICTION_GRID) + 1))
    ap.add_argument("--bodies", type=int, default=45)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--dt", type=float, default=2.0e-4)
    ap.add_argument("--kn", type=float, default=5.0e4)
    ap.add_argument("--zeta", type=float, default=0.2)
    ap.add_argument("--settle-tol", type=float, default=0.01)
    ap.add_argument("--settle-window", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mu = FRICTION_GRID[args.shard - 1]
    t0 = time.time()
    result = repose_case(n_bodies=args.bodies, friction=(mu,), kn=args.kn,
                         zeta=args.zeta, dt=args.dt, n_steps=args.steps,
                         settle_tol=args.settle_tol,
                         settle_window=args.settle_window,
                         shard_of=FRICTION_GRID, shard_index=args.shard)
    result["wall_clock_s"] = time.time() - t0
    os.makedirs(OUT_DIR, exist_ok=True)
    out = args.out or "repose_mu%s.json" % ("%g" % mu)
    path = os.path.join(OUT_DIR, out)
    with open(path, "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    run = result["runs"][0]
    print(json.dumps({
        "shard": args.shard, "mu": mu, "written": path,
        "settled": run["settled"], "valid": run["valid"],
        "steps": run["steps"], "sim_time_s": run["sim_time_s"],
        "repose_angle_deg": run["repose_angle_deg"],
        "n_contacts": run["n_contacts"],
        "energy_residual": run["energy_residual"],
        "inside_band": result["inside_literature_band"][0],
        "wall_clock_s": result["wall_clock_s"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())