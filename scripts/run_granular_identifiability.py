#!/usr/bin/env python
"""Run the granular parameter-identifiability acceptance study.

Usage::

    python scripts/run_granular_identifiability.py --workers 8

Writes ``artifacts/granular_identifiability/study.json`` and the figures next to
it, and prints the acceptance verdicts.
"""
from __future__ import annotations

import argparse
import json
import sys

from tensordem.granular_identifiability.study import run_study


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="artifacts/granular_identifiability")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--json", action="store_true", help="print the whole report")
    args = parser.parse_args(argv)

    report = run_study(
        out_dir=args.out_dir, workers=args.workers, write_figures=not args.no_figures
    )
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0
    print(f"case: {report['case']}")
    print(f"reference parameters: {report['reference_params']}")
    print("observables at the reference point:")
    for name, value in report["nominal"]["targets"].items():
        sigma = report["measurement_sigma"][name]
        print(f"  {name:22s} = {value: .6g}  (sigma = {sigma:.3g}, "
              f"structural sd = {report['structural_jitter'][name]['relative_sd'] * 100:.2f} %)")
    print(f"cost: {report['cost']['wall_seconds']:.1f} s, "
          f"{report['cost']['forward_evaluations']} forward evaluations "
          f"({report['workers']} workers)")
    print("verdicts:")
    for key, verdict in report["verdicts"].items():
        value = verdict.get("value")
        extra = "" if value is None else f" value={value:.4g} threshold={verdict.get('threshold')}"
        print(f"  [{'PASS' if verdict['pass'] else 'FAIL'}] {key}{extra}")
    print(f"all criteria pass: {report['all_criteria_pass']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())