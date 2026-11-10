"""Benchmark driver for the 3-D polyhedral floe DEM with Weibull weakest-link fracture.

Runs the three validation cases of :mod:`tensordem.polyhedron_floe3d.study`, writes
the raw numbers to ``results.json``, the figures next to them and a paper-style
markdown report.  Every number in the report is read from the JSON that this
script produced -- nothing is transcribed by hand.

Usage::

    PYTHONPATH=src python benchmarks/polyhedron_floe3d/run.py [--quick]

``--quick`` shrinks the ensembles (a smoke run, not the reported numbers).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.polyhedron_floe3d.study import (  # noqa: E402
    calibrate_strength,
    case1_impact,
    case2_channel,
    case3_repose,
    conservation_case,
    evaluate_criteria,
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def run(quick: bool = False, out_dir: str = HERE) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    results: Dict[str, object] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    t0 = time.time()
    results["conservation"] = conservation_case()
    print(f"[conservation] {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    results["calibration"] = calibrate_strength()
    print(f"[calibration] {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    results["case1"] = case1_impact(realizations=40 if quick else 120)
    print(f"[case1] {time.time() - t0:.1f}s", flush=True)

    case2_kwargs = dict(ensemble_seeds=(0,) if quick else (0, 1),
                        n_steps=250 if quick else 600,
                        thickness_ratios=(1.0, 0.25) if quick else (1.0, 0.5, 0.25))
    t0 = time.time()
    results["case2"] = case2_channel(**case2_kwargs)
    print(f"[case2] {time.time() - t0:.1f}s", flush=True)

    case3_kwargs = dict(n_bodies=12 if quick else 18, n_steps=400 if quick else 1800,
                        friction=(0.2, 0.4, 0.6) if not quick else (0.2, 0.6))
    t0 = time.time()
    results["case3"] = case3_repose(**case3_kwargs)
    print(f"[case3] {time.time() - t0:.1f}s", flush=True)

    results["verdict"] = evaluate_criteria(results)
    results["runtime_s"] = time.time() - t0
    results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload = _jsonable(results)
    with open(os.path.join(out_dir, "results.json"), "w") as handle:
        json.dump(payload, handle, indent=1)
    print("[json] written", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=HERE)
    args = parser.parse_args()
    payload = run(quick=args.quick, out_dir=args.out)
    print(json.dumps(payload["verdict"], indent=1))


if __name__ == "__main__":
    main()