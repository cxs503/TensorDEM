#!/usr/bin/env python
"""Cheap factorial screen: which knob drives the 12 deg spread in phi_peak?

Motivation
----------
The anchor record (24x48, eps_dot=0.05, cons=4000) reports phi_wall = 41.41 deg.
The stress-chain audit (12x24, eps_dot=0.02, cons=2000, eps_a=0.06) reports
28.85 deg.  Same material, same nominal case, 12 deg apart.  Before spending
full-size cluster time on a convergence study we screen the candidate drivers
(specimen size, strain rate, consolidation length) at sizes that run in ~1 min.

Each cell writes runs/m21262/screen/<tag>.json so the analysis is offline.
"""
from __future__ import annotations

import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tensordem.biaxial_friction_case import run_biaxial_friction_case  # noqa: E402

OUT = os.path.join(REPO, "runs", "m21262", "screen")

# tag -> overrides on top of ANCHOR_CONFIG
CELLS: dict[str, dict] = {
    # finite-size sweep at the anchor rate / consolidation length
    "s08x16": dict(columns=8, rows=16, strain_rate=0.05, consolidation_steps=4000),
    "s12x24": dict(columns=12, rows=24, strain_rate=0.05, consolidation_steps=4000),
    "s16x32": dict(columns=16, rows=32, strain_rate=0.05, consolidation_steps=4000),
    "s24x48": dict(columns=24, rows=48, strain_rate=0.05, consolidation_steps=4000),
}

STRAIN_TARGET = 0.30  # anchor value: the peak moves to eps_a ~ 0.084 at 4000 cons steps,
                      # so a 0.10 window would truncate it


def run_cell(tag: str, over: dict) -> dict:
    t0 = time.time()
    res = run_biaxial_friction_case(strain_target=STRAIN_TARGET, **over)
    rec = {
        "tag": tag,
        "overrides": over,
        "strain_target": STRAIN_TARGET,
        "phi_wall_deg": res["friction_angle"]["phi_peak_wall_deg"],
        "phi_internal_deg": res["friction_angle"]["phi_peak_internal_deg"],
        "peak": res["peak"],
        "specimen": res["specimen"],
        "loading": res["loading"],
        "elapsed_s": round(time.time() - t0, 2),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{tag}.json"), "w") as f:
        json.dump(rec, f, indent=2, default=str)
    return rec


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args(argv[1:])

    tags = list(CELLS)
    if args.nshards > 1:
        tags = [t for i, t in enumerate(tags) if i % args.nshards == args.shard]
    recs = []
    for tag in tags:
        rec = run_cell(tag, CELLS[tag])
        n = rec["specimen"].get("n_particles", "?")
        print(f"{tag:14s} n={n!s:5s} phi_wall={rec['phi_wall_deg']:7.2f}  "
              f"phi_int={rec['phi_internal_deg']:7.2f}  "
              f"eps_a_peak={rec['peak']['axial_strain']:.4f}  "
              f"{rec['elapsed_s']:6.1f}s", flush=True)
        recs.append(rec)
    if recs:
        print("\nspread phi_wall: %.2f .. %.2f deg" % (
            min(r["phi_wall_deg"] for r in recs),
            max(r["phi_wall_deg"] for r in recs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))