"""Production driver: calibrate DEM contact parameters for the material classes.

Usage
-----
    PYTHONPATH=src python3 scripts/calibrate_dem_materials.py --material sand \
        --out artifacts/dem_auto_calibration/sand.json

    # all three classes in one process
    PYTHONPATH=src python3 scripts/calibrate_dem_materials.py --all \
        --out artifacts/dem_auto_calibration/calibration.json --merge-dir artifacts/dem_auto_calibration

Sharding
--------
Each material is an independent job (``--material``), so the campaign can be
split across cluster nodes; ``--merge-dir`` re-assembles the per-material JSON
files into one campaign record and re-evaluates the campaign criteria.  Every
number in the report is read from those JSON files, never hand-copied.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from tensordem.dem_calibration.materials import MATERIALS, spec_by_name
from tensordem.dem_calibration.pipeline import calibrate_material, dump, run_all


def _print_progress(message: str) -> None:
    print(f"[calibrate] {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default=None, help="material class name")
    ap.add_argument("--all", action="store_true", help="run every material class")
    ap.add_argument("--n-init", type=int, default=8)
    ap.add_argument("--n-iter", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-posterior", type=int, default=2000)
    ap.add_argument("--out", default="artifacts/dem_auto_calibration/calibration.json")
    ap.add_argument("--merge-dir", default=None,
                    help="directory of per-material JSON files to merge into --out")
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    if args.merge_dir:
        records = []
        for spec in MATERIALS:
            path = os.path.join(args.merge_dir, f"{spec.name}.json")
            if not os.path.exists(path):
                raise SystemExit(f"missing per-material record: {path}")
            with open(path, "r", encoding="utf-8") as handle:
                records.append(json.load(handle))
        criteria = {
            "criterion_accuracy": all(r["verdicts"]["angle_within_5deg"] for r in records),
            "criterion_convergence": all(r["verdicts"]["objective_below_0p05"]
                                         and r["verdicts"]["iterations_within_50"]
                                         for r in records),
            "criterion_generalisation": all(r["verdicts"]["generalisation_within_10pct"]
                                            for r in records),
            "criterion_uncertainty": all(r["verdicts"]["ci_covers_all"] for r in records),
            "criterion_reproducibility": all(r["verdicts"]["reproducible"] for r in records),
        }
        criteria["all_pass"] = all(criteria.values())
        payload = {"records": records, "criteria": criteria}
        dump(payload, args.out)
        print(json.dumps(criteria, indent=2), flush=True)
        return 0

    if args.all:
        result = run_all(MATERIALS, n_init=args.n_init, n_iter=args.n_iter,
                         seed=args.seed, progress=_print_progress)
        dump(result, args.out)
        print(json.dumps(result["criteria"], indent=2), flush=True)
        return 0

    if not args.material:
        raise SystemExit("pass --material NAME or --all")
    spec = spec_by_name(args.material)
    _print_progress(f"starting {spec.name}")
    record = calibrate_material(spec, n_init=args.n_init, n_iter=args.n_iter,
                                seed=args.seed, n_posterior=args.n_posterior)
    dump(record, args.out)
    print(json.dumps(record["verdicts"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())