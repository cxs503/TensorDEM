"""Driver: joint three-benchmark DEM calibration (node P284W3-N08).

Examples
--------
    python scripts/run_joint_calibration.py --material sand \
        --out artifacts/dem_joint_calibration/sand.json
    python scripts/run_joint_calibration.py --all --jobs 3        # sand + glass + gravel
    python scripts/run_joint_calibration.py --replicates 3        # twin noise floor

Every number that ends up in the report is written to the JSON artefact here,
so the paper tables can be generated programmatically (zero hand-copying).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.dem_calibration.contact_engine import ContactParams  # noqa: E402
from tensordem.dem_calibration import joint_calibration as jc  # noqa: E402
from tensordem.dem_calibration.joint_benchmarks import (  # noqa: E402
    ReposeCase, ShearCase, UniaxialCase, run_repose, run_shear, run_uniaxial,
)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def _write(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_jsonable(payload), fh, indent=2, sort_keys=True)


def run_one(name: str, out: str, n_init: int, n_iter: int, seed: int,
            n_posterior: int) -> dict:
    t0 = time.time()
    result = jc.calibrate_material(name, n_init=n_init, n_iter=n_iter, seed=seed,
                                   n_posterior=n_posterior)
    result["node"] = "P284W3-N08"
    result["seed"] = seed
    result["wall_seconds_total"] = time.time() - t0
    _write(out, result)
    v = result["verdicts"]
    print(f"[{name}] J {result['best_objective']:.4g} drop {result['objective_drop']:.3f} "
          f"| (a) {v['a_repose']['value']:.2f} deg {'PASS' if v['a_repose']['pass'] else 'FAIL'}"
          f" | (b) {v['b_shear_rmse']['value']:.3f} {'PASS' if v['b_shear_rmse']['pass'] else 'FAIL'}"
          f" | (c) {v['c_peak_stress']['value']:.3f} {'PASS' if v['c_peak_stress']['pass'] else 'FAIL'}"
          f" | (d) {'PASS' if v['d_objective_drop']['pass'] else 'FAIL'}"
          f" | (e) {v['e_coverage']['value']:.3f} {'PASS' if v['e_coverage']['pass'] else 'FAIL'}"
          f" | {result['wall_seconds_total']:.0f}s", flush=True)
    return {k: result[k] for k in ("material", "best_params", "best_objective",
                                   "objective_drop", "verdicts", "all_pass")}


def replicate_study(name: str, n_rep: int, out: str) -> dict:
    """Seed-to-seed scatter of the three responses at the declared truth.

    This is the noise floor of any acceptance criterion: a twin-experiment
    calibration cannot be more accurate than the run-to-run scatter of the
    forward model.
    """
    truth = jc.TRUTH[name]
    spec = jc.ExperimentSpec(name=name)
    feats = []
    for k in range(n_rep):
        s = spec
        if k:
            from dataclasses import replace as _replace
            s = jc.ExperimentSpec(
                name=name, material=spec.material,
                repose=_replace(spec.repose, seed=spec.repose.seed + 17 * k),
                shear=_replace(spec.shear, seed=spec.shear.seed + 17 * k),
                uniaxial=_replace(spec.uniaxial, seed=spec.uniaxial.seed + 17 * k))
        raw = jc.forward_features(s, truth)
        feats.append(jc.feature_vector(raw))
    keys = list(feats[0])
    summary = {}
    for key in keys:
        vals = np.array([f[key] for f in feats], dtype=float)
        mean = float(vals.mean())
        summary[key] = {"mean": mean, "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                        "rel_std": float(vals.std(ddof=1) / abs(mean)) if len(vals) > 1 and mean else 0.0}
    payload = {"material": name, "n_replicates": n_rep, "features": summary,
               "repose_theta_deg": [f["repose.theta"] for f in feats],
               "uniaxial_peak": [f["uniaxial.peak"] for f in feats]}
    _write(out, payload)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="sand")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--outdir", default="artifacts/dem_joint_calibration")
    ap.add_argument("--n-init", type=int, default=12)
    ap.add_argument("--n-iter", type=int, default=38)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-posterior", type=int, default=3000)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--replicates", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    names = list(jc.TRUTH) if args.all else [args.material]

    if args.replicates:
        for name in names:
            payload = replicate_study(name, args.replicates,
                                      os.path.join(args.outdir, f"{name}_replicates.json"))
            print(f"[{name}] theta replicates {np.round(payload['repose_theta_deg'], 2)} "
                  f"peak {np.round(payload['uniaxial_peak'], 1)}", flush=True)
        return 0

    if len(names) == 1:
        out = args.out or os.path.join(args.outdir, f"{names[0]}.json")
        run_one(names[0], out, args.n_init, args.n_iter, args.seed, args.n_posterior)
        return 0

    rc = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_one, name, os.path.join(args.outdir, f"{name}.json"),
                        args.n_init, args.n_iter, args.seed, args.n_posterior): name
            for name in names
        }
        for fut, name in futures.items():
            res = fut.result()
            rc |= 0 if res["all_pass"] else 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())