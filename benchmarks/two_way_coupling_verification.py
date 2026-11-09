#!/usr/bin/env python3
"""Benchmark runner for two-way coupling verification.

Runs the four verification surfaces and generates a paper-level report
with figures and machine-checkable criteria.

Usage:
    python benchmarks/two_way_coupling_verification.py [--quick]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.two_way_coupling.study import (
    run_acceptance_study,
    write_results,
    StudyOptions,
    NODE_ID,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Reduced test case for CI")
    parser.add_argument("--outdir", default=str(ROOT / "artifacts" / "two_way_coupling"))
    args = parser.parse_args()

    print(f"Running two-way coupling verification (node: {NODE_ID})")
    print("=" * 70)

    if args.quick:
        opts = StudyOptions(
            stokes_t_max=0.02,
            osc_n_periods=10,
            osc_periods_warmup=8,
            ledger_n_steps=200,
            ledger_n_particles=20,
            ergun_n_phi=15,
            ergun_n_vel=2,
        )
    else:
        opts = StudyOptions()

    results = run_acceptance_study(opts)
    written = write_results(results, args.outdir, figures=True)

    print(f"\nResults written to: {args.outdir}")
    for path in written:
        print(f"  - {path}")

    print("\n" + "=" * 70)
    print("CRITERIA SUMMARY")
    print("=" * 70)

    n_pass = 0
    n_total = len(results["criteria"])
    for name, crit in results["criteria"].items():
        status = "✓ PASS" if crit["passed"] else "✗ FAIL"
        if crit["passed"]:
            n_pass += 1
        print(f"{status}  {name}")
        print(f"         value = {crit['value']:.3e}, threshold = {crit['threshold']:.3e}")

    print("\n" + "=" * 70)
    print(f"RESULT: {n_pass}/{n_total} criteria passed")
    if results["all_criteria_passed"]:
        print("STATUS: ALL CRITERIA PASSED ✓")
    else:
        print("STATUS: SOME CRITERIA FAILED ✗")
    print("=" * 70)

    return 0 if results["all_criteria_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
