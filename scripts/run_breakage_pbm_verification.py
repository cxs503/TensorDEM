"""Run the DEM-PBM breakage coupling verification surface and post the report.

Usage
-----
    PYTHONPATH=src python scripts/run_breakage_pbm_verification.py \
        --crush-samples 10000 --n-particles 10000 --t-end 8 \
        --out artifacts/breakage_pbm

The script writes ``summary.json`` and ``breakage_pbm_summary.png`` into the
output directory and prints the criterion table.  It performs no hand
transcription: every number in the printed table (and in the report built by
``scripts/post_breakage_pbm_report.py``) comes from the returned dictionary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.breakage_pbm.study import run_verification  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crush-samples", type=int, default=1000)
    ap.add_argument("--crush-batch", type=int, default=250)
    ap.add_argument("--n-particles", type=int, default=2000)
    ap.add_argument("--diameter", type=float, default=0.01)
    ap.add_argument("--t-end", type=float, default=8.0)
    ap.add_argument("--stress-max", type=float, default=4.5e7)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--out", default="artifacts/breakage_pbm")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    result = run_verification(
        crush_samples=args.crush_samples, crush_batch_size=args.crush_batch,
        n_particles=args.n_particles, assembly_diameter=args.diameter,
        t_end=args.t_end, record_times=(1.0, 2.0, 4.0, 8.0),
        stress_max=args.stress_max, seed=args.seed, out_dir=args.out,
        make_figures=not args.no_figures, verbose=True,
    )
    print("\n=== criteria ===")
    for key, crit in result["criteria"].items():
        print(f"{key:28s} value={crit['value']:.6g} "
              f"threshold={crit['threshold']:.6g} passed={crit['passed']}")
    print("passed_all =", result["passed_all"])
    print("artifacts ->", os.path.abspath(args.out))
    print(json.dumps({"passed_all": result["passed_all"],
                      "criteria": result["criteria"]}, indent=2))
    return 0 if result["passed_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())