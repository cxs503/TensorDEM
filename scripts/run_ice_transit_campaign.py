#!/usr/bin/env python
"""Run the steady-state broken-ice transit campaign (p156w2-n01).

Single case:      python scripts/run_ice_transit_campaign.py --name standard
Whole set:        python scripts/run_ice_transit_campaign.py --workers 24
Cluster shard:    python scripts/run_ice_transit_campaign.py --shard 3 --shards 24

Each case writes ``artifacts/ice_transit/cases/<name>.json`` (payload including
the resistance statistics, the energy ledger and the anchors); the report is
built from those files by ``--report`` (post-processing only, no simulation).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.ice_transit.campaign import (CASE_SPECS, case_spec,  # noqa: E402
                                            run_campaign, run_case)

CASE_DIR = ROOT / "artifacts" / "ice_transit" / "cases"


def _write(name: str, payload: dict, out: Path, max_series: int = 600) -> Path:
    """Write one case payload, decimating the recorded series.

    The full time history is ~2800 rows per case (~0.6 MB); the report only needs
    the envelope, so the series is decimated to <= ``max_series`` rows and the
    stride is recorded with it.  (The host filesystem is shared and was full
    during the first pass, which is why this is now explicit.)
    """
    out.mkdir(parents=True, exist_ok=True)
    series = payload.get("series", [])
    stride = max(1, len(series) // max_series)
    payload = dict(payload)
    payload["series"] = series[::stride]
    payload["series_stride"] = stride
    path = out / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", nargs="*", default=None,
                        help="case name(s); default: the whole set")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--shard", type=int, default=None,
                        help="shard index (0-based) for a cluster scan")
    parser.add_argument("--shards", type=int, default=None)
    parser.add_argument("--out", type=Path, default=CASE_DIR)
    parser.add_argument("--report", action="store_true",
                        help="rebuild the report from the existing case files")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)

    if args.report:
        from tensordem.ice_transit.report import build_report
        build_report(case_dir=args.out, skip_tests=args.skip_tests)
        return 0

    names = list(CASE_SPECS) if args.name is None else list(args.name)
    if args.shard is not None:
        if not args.shards:
            parser.error("--shard requires --shards")
        names = [n for i, n in enumerate(names) if i % args.shards == args.shard]
    if not names:
        print("nothing to do for this shard")
        return 0

    if args.workers and args.workers > 1 and args.shard is None:
        results = run_campaign(names, workers=args.workers)
        for name, payload in results.items():
            path = _write(name, payload, args.out)
            print(f"{name}: R = {payload['mean_kN']:.2f} kN -> {path}")
        return 0

    for name in names:
        payload = run_case(case_spec(name))
        path = _write(name, payload, args.out)
        print(f"{name}: R = {payload['mean_kN']:.2f} kN "
              f"(ledger residual {payload['ledger_residual_rel']:.3%}) -> {path}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())