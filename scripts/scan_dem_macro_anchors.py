#!/usr/bin/env python
"""Shard driver for the DEM macro-anchor scale/rate/control scan.

    python scripts/scan_dem_macro_anchors.py --shard-index 3 --shard-count 13

Runs the cases of ``scripts/dem_macro_cases.json`` whose index is congruent to
``--shard-index`` modulo ``--shard-count`` and prints one JSON line per case
(so a cluster shard log is directly parseable).  Artifacts are written to
``artifacts/macro/cases/<label>.json`` inside the checkout.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--table", default=os.path.join(HERE, "dem_macro_cases.json"))
    args = ap.parse_args()
    with open(args.table) as fh:
        table = json.load(fh)
    cases = table["cases"]
    for idx, case in enumerate(cases):
        if idx % args.shard_count != args.shard_index:
            continue
        out = os.path.join(ROOT, "artifacts", "macro", "cases", f"{case['label']}.json")
        cmd = [sys.executable, os.path.join(HERE, "run_dem_macro_anchors.py"),
               "--label", case["label"], "--n", str(case["n"]),
               "--pour-time", str(case["pour_time"]),
               "--settle", str(case["settle"]),
               "--control", case["control"], "--out", out]
        env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            print(json.dumps({"label": case["label"], "status": "error",
                              "stderr": proc.stderr[-2000:]}), flush=True)
            continue
        summary = json.loads(proc.stdout)
        summary["status"] = "ok"
        print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
