#!/usr/bin/env python
"""LBM--DEM drag-closure calibration benchmark (campaign P55-N08 / m5430).

Runs the case matrix that closes the two-way drag closure against analytic
anchors, then writes ``result.json`` and ``gates.json`` next to this file.

Usage
-----
    python benchmarks/verified/lbm_dem_drag_calibration/run.py --list
    python benchmarks/verified/lbm_dem_drag_calibration/run.py --all --jobs 4
    python benchmarks/verified/lbm_dem_drag_calibration/run.py --case stokes_d8
    python benchmarks/verified/lbm_dem_drag_calibration/run.py --merge

Every case writes ``parts/<label>.json``; ``--merge`` assembles the parts,
recomputes the gates from the stored numbers (no hand-edited values) and writes
``result.json`` + ``gates.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensordem.lbmdem import FixedSphereSpec, SettlingSpec, evaluate_gates  # noqa: E402
from tensordem.lbmdem.calibration import run_fixed_sphere, run_settling  # noqa: E402

CASE_SPECS: dict[str, FixedSphereSpec | SettlingSpec] = {
    # -- Stokes anchor + resolution study (d = 4, 6, 8 at L/d = 8) ------
    "stokes_d4": FixedSphereSpec(label="stokes_d4", diameter=4.0, domain_ratio=8.0,
                                 re_target=0.05, scheme="mem", dtype="float32",
                                 max_steps=1600, min_steps=400, sample_every=50, rtol=1e-4),
    "stokes_d6": FixedSphereSpec(label="stokes_d6", diameter=6.0, domain_ratio=8.0,
                                 re_target=0.05, scheme="mem", dtype="float32",
                                 max_steps=1600, min_steps=400, sample_every=50, rtol=1e-4),
    "stokes_d8": FixedSphereSpec(label="stokes_d8", diameter=8.0, domain_ratio=8.0,
                                 re_target=0.05, scheme="mem", dtype="float32",
                                 max_steps=1400, min_steps=400, sample_every=50, rtol=1e-4),
    # -- Smagorinsky sensitivity at the d = 6 Stokes case ----------------
    "stokes_cs0.5": FixedSphereSpec(label="stokes_cs0.5", diameter=6.0, domain_ratio=8.0,
                                    re_target=0.05, scheme="mem", smagorinsky_cs=0.5,
                                    dtype="float32", max_steps=1600, min_steps=400,
                                    sample_every=50, rtol=1e-4),
    "stokes_cs1": FixedSphereSpec(label="stokes_cs1", diameter=6.0, domain_ratio=8.0,
                                  re_target=0.05, scheme="mem", smagorinsky_cs=1.0,
                                  dtype="float32", max_steps=1600, min_steps=400,
                                  sample_every=50, rtol=1e-4),
    "stokes_cs2": FixedSphereSpec(label="stokes_cs2", diameter=6.0, domain_ratio=8.0,
                                  re_target=0.05, scheme="mem", smagorinsky_cs=2.0,
                                  dtype="float32", max_steps=1600, min_steps=400,
                                  sample_every=50, rtol=1e-4),
    # -- domain sensitivity at fixed resolution (d = 4) ------------------
    "stokes_ld8": FixedSphereSpec(label="stokes_ld8", diameter=4.0, domain_ratio=8.0,
                                  re_target=0.05, scheme="mem", dtype="float32",
                                  max_steps=1600, min_steps=400, sample_every=50, rtol=1e-4),
    "stokes_ld16": FixedSphereSpec(label="stokes_ld16", diameter=4.0, domain_ratio=16.0,
                                   re_target=0.05, scheme="mem", dtype="float32",
                                   max_steps=1400, min_steps=400, sample_every=50, rtol=1e-4),
    # -- Schiller-Naumann band -------------------------------------------
    "reich_re5": FixedSphereSpec(label="reich_re5", diameter=6.0, domain_ratio=8.0,
                                 re_target=5.0, scheme="mem", dtype="float32",
                                 max_steps=1600, min_steps=400, sample_every=50, rtol=1e-4),
    "reich_re20": FixedSphereSpec(label="reich_re20", diameter=6.0, domain_ratio=8.0,
                                  re_target=20.0, scheme="mem", dtype="float32",
                                  max_steps=1600, min_steps=400, sample_every=50, rtol=1e-4),
    "reich_re50": FixedSphereSpec(label="reich_re50", diameter=6.0, domain_ratio=8.0,
                                  re_target=50.0, scheme="mem", smagorinsky_cs=0.16,
                                  dtype="float32", max_steps=1600, min_steps=400,
                                  sample_every=50, rtol=1e-4),
    # -- higher-resolution Re points (the d = 6 points show the resolution
    #    limit of the coarse sphere; these run at d = 8, L/d = 8) ----------
    "reich_re5_d8": FixedSphereSpec(label="reich_re5_d8", diameter=8.0, domain_ratio=8.0,
                                    re_target=5.0, scheme="mem", tau0=0.644, dtype="float32",
                                    max_steps=1400, min_steps=400, sample_every=50, rtol=1e-4),
    "reich_re20_d8": FixedSphereSpec(label="reich_re20_d8", diameter=8.0, domain_ratio=8.0,
                                     re_target=20.0, scheme="mem", tau0=0.536, dtype="float32",
                                     max_steps=1400, min_steps=400, sample_every=50, rtol=1e-4),
    # -- IBM scheme agreement (same case, two independent force paths) ---
    "scheme_direct_re0.05": FixedSphereSpec(label="scheme_direct_re0.05", diameter=6.0,
                                            domain_ratio=8.0, re_target=0.05, scheme="direct",
                                            marker_density=2.0, dtype="float32",
                                            max_steps=1600, min_steps=400, sample_every=50,
                                            rtol=1e-4),
    "scheme_direct_re20": FixedSphereSpec(label="scheme_direct_re20", diameter=6.0,
                                          domain_ratio=8.0, re_target=20.0, scheme="direct",
                                          marker_density=2.0, dtype="float32",
                                          max_steps=1600, min_steps=400, sample_every=50,
                                          rtol=1e-4),
    # -- two-sphere tandem (Stimson-Jeffery / DKT-family anchor) ---------
    "pair_r8_d6": FixedSphereSpec(label="pair_r8_d6", diameter=6.0, domain_ratio=10.6666667,
                                  re_target=0.05, scheme="mem", separation_radius=8.0,
                                  dtype="float32", max_steps=1400, min_steps=400,
                                  sample_every=50, rtol=1e-4),
    "single_d6_l64": FixedSphereSpec(label="single_d6_l64", diameter=6.0,
                                     domain_ratio=10.6666667, re_target=0.05, scheme="mem",
                                     dtype="float32", max_steps=1400, min_steps=400,
                                     sample_every=50, rtol=1e-4),
    # -- free settling: terminal velocity + momentum audit ---------------
    "settling_d6": SettlingSpec(label="settling_d6", diameter=6.0, domain_ratio=8.0,
                                density_ratio=2.0, re_target=1.0, scheme="mem", steps=900,
                                sample_every=5, window=100, dtype="float64"),
}


def run_one(label: str) -> dict:
    """Run a single case and persist its part file."""
    torch_threads = int(os.environ.get("TORCH_THREADS", "2"))
    try:
        import torch

        torch.set_num_threads(torch_threads)
    except Exception:  # pragma: no cover - torch is a hard dependency
        pass
    spec = CASE_SPECS[label]
    started = time.time()
    if isinstance(spec, SettlingSpec):
        result = run_settling(spec)
    else:
        result = run_fixed_sphere(spec)
    result["wall_seconds"] = round(time.time() - started, 3)
    result["dtype"] = spec.dtype
    out = HERE / "parts" / f"{label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def merge() -> dict:
    """Assemble all parts into result.json + gates.json."""
    parts = sorted((HERE / "parts").glob("*.json"))
    if not parts:
        raise SystemExit("no part files found; run --all first")
    cases = [json.loads(part.read_text()) for part in parts]
    cases.sort(key=lambda case: case["label"])
    gates = evaluate_gates(cases)
    result = {
        "campaign": "P55-N08",
        "mission_id": 5430,
        "title": "Two-way LBM-DEM drag closure calibration (Stokes / Schiller-Naumann / "
                 "tandem pair / settling / momentum audit)",
        "generated_by": "benchmarks/verified/lbm_dem_drag_calibration/run.py --merge",
        "n_cases": len(cases),
        "case_labels": [case["label"] for case in cases],
        "gates": gates,
        "cases": cases,
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (HERE / "gates.json").write_text(
        json.dumps(
            {
                "campaign": "P55-N08",
                "mission_id": 5430,
                "generated_by": result["generated_by"],
                "all_recomputed_pass": gates["all_recomputed_pass"],
                "n_gates": gates["n_gates"],
                "n_passed": gates["n_passed"],
                "gates": gates["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list case labels")
    parser.add_argument("--case", action="append", default=[], help="run one case label")
    parser.add_argument("--all", action="store_true", help="run every case")
    parser.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    parser.add_argument("--merge", action="store_true", help="merge parts into result.json")
    args = parser.parse_args(argv)

    if args.list:
        for label in CASE_SPECS:
            print(label)
        return 0

    labels = list(CASE_SPECS) if args.all else args.case
    unknown = [label for label in labels if label not in CASE_SPECS]
    if unknown:
        raise SystemExit(f"unknown case(s): {unknown}")

    if labels:
        if args.jobs <= 1:
            for label in labels:
                payload = run_one(label)
                print(
                    f"{label}: force={payload.get('force_total', payload.get('drag_terminal')):.6g} "
                    f"wall={payload['wall_seconds']}s",
                    flush=True,
                )
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futures = {pool.submit(run_one, label): label for label in labels}
                for future in as_completed(futures):
                    label = futures[future]
                    payload = future.result()
                    print(
                        f"{label}: force={payload.get('force_total', payload.get('drag_terminal')):.6g} "
                        f"wall={payload['wall_seconds']}s",
                        flush=True,
                    )

    if args.merge or (labels and not args.case):
        result = merge()
        print(
            f"gates: {result['gates']['n_passed']}/{result['gates']['n_gates']} passed, "
            f"all_recomputed_pass={result['gates']['all_recomputed_pass']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())