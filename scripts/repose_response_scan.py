"""Response-surface scan: theta_r(mu), theta_r(e), theta_r(k_n/k_s) per material.

Produces the physical-validation evidence of the report: monotonicity of the
repose angle in the sliding friction, the sensitivity to restitution and to the
stiffness ratio, and the *seed repeatability* (two independent packings at the
same parameters) which sets the noise floor of the inverse problem.

    PYTHONPATH=src python3 scripts/repose_response_scan.py --material sand \
        --out artifacts/dem_auto_calibration/response_sand.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

from tensordem.dem_calibration.heap import HeapError, run_heap
from tensordem.dem_calibration.materials import MATERIALS, case_for, spec_by_name


def _responses(spec, condition: str, params: dict, seed_offset: int = 0) -> dict:
    case = case_for(spec, condition, params)
    if seed_offset:
        from dataclasses import replace
        case = replace(case, seed=case.seed + seed_offset)
    out = run_heap(case)
    return {"theta_r_deg": out["theta_r_deg"], "phi_solid": out["phi_solid"],
            "r2": out["r2"], "ke_ratio": out["ke_ratio"],
            "digest": out["digest"], "n_particles": out["n_particles"],
            "n_steps": out["n_steps"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="sand")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args(argv)

    spec = spec_by_name(args.material)
    base = dict(spec.truth)
    rows = []

    def record(label: str, params: dict) -> None:
        thetas = []
        for offset in range(args.seeds):
            t0 = time.time()
            try:
                resp = _responses(spec, "calibration", params, seed_offset=offset)
            except HeapError as exc:
                rows.append({"label": label, "params": params, "seed_offset": offset,
                             "error": f"{type(exc).__name__}: {exc}"})
                continue
            resp.update({"label": label, "params": params, "seed_offset": offset,
                         "wall_s": round(time.time() - t0, 2)})
            rows.append(resp)
            thetas.append(resp["theta_r_deg"])
        if thetas:
            print("SCAN " + json.dumps({
                "material": spec.name, "label": label, "params": params,
                "theta_mean": statistics.fmean(thetas),
                "theta_span": max(thetas) - min(thetas), "n": len(thetas)}), flush=True)

    for mu in (0.15, 0.30, 0.45, 0.60, 0.80):
        record("mu_scan", {**base, "mu": mu})
    for e in (0.10, 0.50, 0.90):
        record("restitution_scan", {**base, "restitution": e})
    for sr in (1.0, 3.0):
        record("stiffness_scan", {**base, "stiffness_ratio": sr})
    record("validation_truth", {**base})

    payload = {"material": spec.name, "truth": base,
               "experiment": case_for(spec, "calibration", base).signature(),
               "rows": rows}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())