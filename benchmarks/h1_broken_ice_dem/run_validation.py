"""H1-BROKEN-ICE-DEM runner: build the three cases, write ``results.json``.

    PYTHONPATH=src python benchmarks/h1_broken_ice_dem/run_validation.py --quick
    PYTHONPATH=src python benchmarks/h1_broken_ice_dem/run_validation.py \
        --cases buoyancy,channel --out results.json
    # one friction per shard, so the three rubble runs can go to three cluster jobs
    PYTHONPATH=src python benchmarks/h1_broken_ice_dem/run_validation.py \
        --cases repose --repose-shard 2 --repose-bodies 45 --repose-steps 30000 \
        --repose-dt 2e-4 --out repose_shard_2.json

Every number in the JSON is produced by the run; nothing is copied by hand.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.polyhedron_floe3d.h1_cases import validate  # noqa: E402

OUT_DIR = os.path.join(ROOT, "artifacts", "h1_broken_ice_dem")

#: friction sweep declared in ``docs/H1_BROKEN_ICE_DEM_ROUND2.md`` section 4
FRICTION_GRID = (0.2, 0.4, 0.6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cases", default="buoyancy,repose,channel")
    ap.add_argument("--repose-bodies", type=int, default=None)
    ap.add_argument("--repose-steps", type=int, default=None)
    ap.add_argument("--repose-dt", type=float, default=None)
    ap.add_argument("--repose-kn", type=float, default=None,
                    help="normal stiffness; must be co-varied with --repose-dt so that "
                         "the step resolves the contact period (see the stability table "
                         "in docs/H1_BROKEN_ICE_DEM_ROUND3.md)")
    ap.add_argument("--repose-settle-tol", type=float, default=None)
    ap.add_argument("--repose-shard", type=int, default=None,
                    help="1-based index into FRICTION_GRID %s; runs that mu only"
                         % (FRICTION_GRID,))
    ap.add_argument("--channel-floes", type=int, default=None)
    ap.add_argument("--channel-steps", type=int, default=None)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    kwargs = {}
    rk = {}
    if args.repose_bodies:
        rk["n_bodies"] = args.repose_bodies
    if args.repose_steps:
        rk["n_steps"] = args.repose_steps
    if args.repose_dt:
        rk["dt"] = args.repose_dt
    if args.repose_kn:
        rk["kn"] = args.repose_kn
    if args.repose_settle_tol is not None:
        rk["settle_tol"] = args.repose_settle_tol
    if args.repose_shard is not None:
        if not 1 <= args.repose_shard <= len(FRICTION_GRID):
            raise SystemExit("--repose-shard out of range: %s" % (FRICTION_GRID,))
        rk["friction"] = (FRICTION_GRID[args.repose_shard - 1],)
        rk["shard_of"] = list(FRICTION_GRID)
        rk["shard_index"] = args.repose_shard
    if rk:
        kwargs["repose"] = rk
    ck = {}
    if args.channel_floes:
        ck["n_floes"] = args.channel_floes
    if args.channel_steps:
        ck["n_steps"] = args.channel_steps
    if ck:
        kwargs["channel"] = ck

    result = validate(cases=[c for c in args.cases.split(",") if c], quick=args.quick,
                      **kwargs)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    print(json.dumps({"passed": result["passed"], "runtime_s": result["runtime_s"],
                      "written": path}, indent=1))
    # CI gate: any requested case that did not pass must fail the run (rc != 0),
    # so a regression cannot be silently swallowed by a 0 exit code.
    # (audit m15192: "run_validation.py 恒返回 rc=0, case passed=False 也返回 0
    #  -> 不能当 CI 闸门")
    failed = [name for name, ok in result["passed"].items() if not ok]
    if failed:
        print("FAILED cases: %s" % (", ".join(failed),), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())