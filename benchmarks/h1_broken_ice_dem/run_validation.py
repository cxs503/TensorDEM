"""H1-BROKEN-ICE-DEM round-2 driver: three acceptance cases, raw JSON + figures.

Usage::

    PYTHONPATH=src python benchmarks/h1_broken_ice_dem/run_validation.py --quick
    PYTHONPATH=src python benchmarks/h1_broken_ice_dem/run_validation.py \
        --repose-bodies 27 --repose-steps 3000

Every number in the markdown report is read from the JSON written here; nothing is
transcribed by hand.  The figures are generated from the same JSON.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cases", default="buoyancy,repose,channel")
    ap.add_argument("--repose-bodies", type=int, default=None)
    ap.add_argument("--repose-steps", type=int, default=None)
    ap.add_argument("--channel-floes", type=int, default=None)
    ap.add_argument("--channel-steps", type=int, default=None)
    args = ap.parse_args()

    kwargs = {}
    if args.repose_bodies or args.repose_steps:
        rk = {}
        if args.repose_bodies:
            rk["n_bodies"] = args.repose_bodies
        if args.repose_steps:
            rk["n_steps"] = args.repose_steps
        kwargs["repose"] = rk
    if args.channel_floes or args.channel_steps:
        ck = {}
        if args.channel_floes:
            ck["n_floes"] = args.channel_floes
        if args.channel_steps:
            ck["n_steps"] = args.channel_steps
        kwargs["channel"] = ck

    result = validate(cases=[c for c in args.cases.split(",") if c], quick=args.quick,
                      **kwargs)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "results.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    print(json.dumps({"passed": result["passed"], "runtime_s": result["runtime_s"],
                      "written": path}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
