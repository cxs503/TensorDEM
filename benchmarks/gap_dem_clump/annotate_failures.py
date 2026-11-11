#!/usr/bin/env python3
"""Attach the round-1 honest-FAIL bookkeeping to a finished results.json.

The failure record is *derived from the numbers already in the JSON* (no re-run,
no new numbers), using the same function the campaign runner uses:
``benchmarks.gap_dem_clump.run_gap_dem_clump.derive_failure_reason``.

    python benchmarks/gap_dem_clump/annotate_failures.py [--results PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "gap_dem_clump"))

from run_gap_dem_clump import derive_failure_reason  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "artifacts" / "GAP-DEM-CLUMP" / "results.json"))
    args = ap.parse_args()
    path = Path(args.results)
    d = json.loads(path.read_text())
    failures = derive_failure_reason(d["C1_analytic"], d["C2_inertia_scan"], d["C3_repose_scan"])
    d["failures"] = failures
    d["honest_fail"] = bool(failures)
    d["all_pass"] = all(d["criteria"].values()) and not failures
    path.write_text(json.dumps(d, indent=2, default=float))
    print(json.dumps({"failures": sorted(failures), "all_pass": d["all_pass"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())