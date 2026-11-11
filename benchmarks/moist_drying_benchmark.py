#!/usr/bin/env python3
"""Benchmark CLI: moist-particle drum drying reference-vs-computed table.

Mission p380w2-n05 (m14592).  Runs the acceptance suite of
:mod:`tensordem.rotary_dryer.moist_drying_case` and writes

* ``artifacts/moist_drying/benchmark.json``   -- reference-vs-computed table
  (analytic anchors, literature correlations, independent RK4 solver),
* ``artifacts/moist_drying/figures/*.png``   -- report figures.

Reproduce::

    python -m tensordem.rotary_dryer.moist_drying_case --profile full \
        --out runs/m14592_moist_drying/acceptance.json \
        --figdir artifacts/moist_drying/figures
    # or, equivalently:
    python benchmarks/moist_drying_benchmark.py --profile full

Exit code 0 iff every acceptance criterion passes.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.rotary_dryer.moist_drying_case import PROFILES, run_acceptance_suite  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=sorted(PROFILES))
    ap.add_argument("--outdir", default=str(ROOT / "artifacts" / "moist_drying"))
    args = ap.parse_args()
    outdir = pathlib.Path(args.outdir)
    bundle = run_acceptance_suite(profile=args.profile,
                                  figdir=str(outdir / "figures"))
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "benchmark.json").write_text(json.dumps(
        {"profile": args.profile, "benchmark": bundle["benchmark"],
         "criterion_pass": bundle["criterion_pass"],
         "benchmark_all_passed": bundle["benchmark_all_passed"],
         "benchmark_validation_failures": bundle["benchmark_validation_failures"]},
        indent=2))
    print(json.dumps({"benchmark_all_passed": bundle["benchmark_all_passed"],
                      "validation_failures": bundle["benchmark_validation_failures"],
                      "criterion_pass": bundle["criterion_pass"]}, indent=2))
    return 0 if bundle["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
