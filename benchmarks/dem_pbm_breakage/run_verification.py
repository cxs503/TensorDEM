"""Entry point: run the DEM -> PBM breakage coupling verification surface.

This is the *reproducible entry* for the campaign node
``p176w2-n10-dem-pbm-breakage``: it runs the whole chain
(bonded-cluster crushing -> Weibull strength -> DEM daughter law -> sectional
population balance -> DEM/PBM PSD cross validation -> negative controls),
writes ``results.json`` (every number in the documentation and in the posted
report is copied from this file, zero hand transcription) and drops the summary
figure into ``docs/`` so that it survives the repository push gate
(``artifacts/`` and ``runs/`` are local record areas and are not pushed).

Usage
-----
    PYTHONPATH=src python benchmarks/dem_pbm_breakage/run_verification.py
    PYTHONPATH=src python benchmarks/dem_pbm_breakage/run_verification.py \\
        --crush-samples 800 --n-particles 150 --t-end 8 --seed 20261110

Exit code 0 iff every machine criterion (C1-C7) passes; the criteria table and
the failing rows are printed either way (an honest FAIL is a result, not an
error to hide).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.breakage_pbm.study import run_verification  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crush-samples", type=int, default=800)
    ap.add_argument("--crush-batch", type=int, default=200)
    ap.add_argument("--n-particles", type=int, default=150)
    ap.add_argument("--diameter", type=float, default=0.01)
    ap.add_argument("--t-end", type=float, default=8.0)
    ap.add_argument("--stress-max", type=float, default=4.5e7)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--out", default=HERE, help="directory for results.json")
    ap.add_argument("--figure-dir", default=os.path.join(ROOT, "docs"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    result = run_verification(
        crush_samples=args.crush_samples, crush_batch_size=args.crush_batch,
        n_particles=args.n_particles, assembly_diameter=args.diameter,
        t_end=args.t_end, record_times=(1.0, 2.0, 4.0, 8.0),
        stress_max=args.stress_max, seed=args.seed, out_dir=out_dir,
        make_figures=not args.no_figures, verbose=True,
    )

    # the summary figure must live under docs/ to be pushed (pnq gate)
    if not args.no_figures:
        src = os.path.join(out_dir, "breakage_pbm_summary.png")
        if os.path.exists(src) and os.path.abspath(args.figure_dir) != out_dir:
            os.makedirs(args.figure_dir, exist_ok=True)
            shutil.copyfile(src, os.path.join(args.figure_dir,
                                              "breakage_pbm_summary.png"))

    print("\n=== criteria ===")
    for key, crit in result["criteria"].items():
        print(f"{key:30s} value={crit['value']:.6g}  threshold={crit['threshold']:.6g}"
              f"  passed={crit['passed']}")
    print("\n=== benchmark (reference vs computed) ===")
    for row in result.get("benchmark", []):
        print(f"{row['quantity']:48s} ref={row['reference']:.6g} "
              f"got={row['computed']:.6g} rel_err={row['relative_error']:.3g} "
              f"tol={row['tolerance']:.3g} pass={row['passed']}")
    print("\npassed_all =", result["passed_all"])
    print("results ->", os.path.join(out_dir, "results.json"))
    return 0 if result["passed_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())