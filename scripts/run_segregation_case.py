"""Run one segregation campaign case and print a compact JSON record on stdout.

The full record (positions, full statistics) stays in the process; what is
printed is everything the analysis needs, downsampled so that a few dozen shard
logs can be collected over stdout without a file transfer:

* ``metrics`` (segregation indices, rise velocities, audits, profiles),
* ``base_load`` (zone floor pressure),
* ``stats`` (the zone/z-bin accumulators the transfer mapping consumes),
* a position ``snapshot`` subsample for the figures.

Usage (one case)::

    PYTHONPATH=src python scripts/run_segregation_case.py --spec specs.json --name vib_R2_G3_s0

Usage (one shard of a campaign)::

    PYTHONPATH=src python scripts/run_segregation_case.py --spec specs.json --shard 3/16
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from tensordem.segregation.campaign import CaseSpec, run_case

#: keep the printed record small enough for a shard log
SERIES_STRIDE = 20
ZBIN_STRIDE = 2
MAX_SNAPSHOT = 2500


def compact(record: dict) -> dict:
    out = {k: v for k, v in record.items() if k != "snapshot"}
    series = out.get("series", {})
    out["series"] = {
        k: (v[::SERIES_STRIDE] if isinstance(v, list) else v) for k, v in series.items()
    }
    stats = out.get("stats", {})
    for key in ("z_area", "z_large", "z_moment"):
        if key in stats:
            arr = stats[key]
            stats[key] = arr[::ZBIN_STRIDE]
            if key == "z_area":
                stats["z_edges"] = stats["z_edges"][::ZBIN_STRIDE]
    snap = record.get("snapshot")
    if snap:
        pos = np.asarray(snap["pos"], dtype=np.float64)
        if len(pos) > MAX_SNAPSHOT:
            keep = np.linspace(0, len(pos) - 1, MAX_SNAPSHOT).astype(int)
        else:
            keep = np.arange(len(pos))
        out["snapshot"] = {
            "pos": pos[keep].round(6).tolist(),
            "radii": np.asarray(snap["radii"])[keep].round(7).tolist(),
            "is_large": np.asarray(snap["is_large"])[keep].tolist(),
            "width_m": snap["width_m"],
            "settled_bed_height_m": snap["settled_bed_height_m"],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="JSON list of case specs")
    ap.add_argument("--name", default=None, help="run this single case")
    ap.add_argument("--shard", default="0/1", help="i/n shard selector")
    ap.add_argument("--max-name", default=None, help="comma separated selection")
    args = ap.parse_args(argv)

    with open(args.spec) as fh:
        specs = [CaseSpec(**s) for s in json.load(fh)]
    specs = sorted(specs, key=lambda s: s.name)
    if args.name:
        specs = [s for s in specs if s.name == args.name]
        if not specs:
            raise SystemExit(f"unknown case {args.name}")
    else:
        i, n = (int(v) for v in args.shard.split("/"))
        if args.max_name:
            keep = set(args.max_name.split(","))
            specs = [s for s in specs if s.name in keep]
        specs = [s for k, s in enumerate(specs) if k % n == i]
    for spec in specs:
        record = run_case(spec)
        payload = compact(record)
        print("CASE_RECORD " + json.dumps(payload), flush=True)
        print(
            f"[done] {spec.name} wall={record['audit']['wall_clock_s']:.0f}s "
            f"S_bn={record['metrics']['segregation']['S_bn']:+.4f}",
            file=sys.stderr, flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())