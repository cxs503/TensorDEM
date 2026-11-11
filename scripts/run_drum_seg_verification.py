#!/usr/bin/env python
"""Run the p176w2-n08 rotating-drum segregation case matrix.

One case per process (``multiprocessing``), one JSON record per case.  The
records are the only input of the report generator -- no number in the report
is typed by hand.

Usage
-----
    PYTHONPATH=src python scripts/run_drum_seg_verification.py \
        --out runs/p176w2_n08/cases --jobs 16

    # pilot (short, small) run used only to check that segregation develops
    PYTHONPATH=src python scripts/run_drum_seg_verification.py \
        --out runs/p176w2_n08/pilot --jobs 8 --groups ratio mono --revs 2 --d-small 0.012
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
import traceback
from dataclasses import asdict
from multiprocessing import Pool

import numpy as np

from tensordem.drum_segregation import metrics, sweeps
from tensordem.drum_segregation.kernel2d import DrumConfig, simulate


def _cfg_kwargs(case):
    kw = dict(case["cfg"])
    return kw


def _averages(series, frac=0.25):
    out = {}
    for key, name in (("core_shell_excess", "excess_avg"),
                      ("size_pos_corr", "corr_avg"),
                      ("segregation", "S_avg")):
        arr = np.asarray(series.get(key, []), float)
        if arr.size:
            k = max(1, int(round(frac * arr.size)))
            out[name] = float(np.mean(arr[-k:]))
            out[name + "_std"] = float(np.std(arr[-k:]))
    return out


def run_case(case):
    name = case["name"]
    cfg = DrumConfig(**_cfg_kwargs(case))
    t0 = time.time()
    try:
        run = simulate(cfg)
    except Exception:                                  # pragma: no cover
        return {"name": name, "group": case["group"], "error": traceback.format_exc()}
    wall = time.time() - t0

    r_norm = np.linalg.norm(run.pos, axis=1) / cfg.radius
    null = metrics.permutation_null(run.pos, run.radii, run.is_large,
                                   cfg.radius, cfg.n_bins, n_perm=200,
                                   seed=cfg.seed)
    prof = null.pop("profile")
    mode = metrics.classify_mode(null["excess_observed"], null["excess_z"],
                                null["corr_observed"], null["corr_z"])

    rec = {
        "name": name,
        "group": case["group"],
        "seed": cfg.seed,
        "cfg": {k: v for k, v in asdict(cfg).items()},
        "actual_fill": run.actual_fill,
        "actual_large_area_fraction": run.actual_large_area_fraction,
        "n_large": run.n_large,
        "n_small": run.n_small,
        "n_particles": run.n_large + run.n_small,
        "froude": run.diagnostics["froude"],
        "max_overlap_ratio": run.diagnostics["max_overlap_ratio"],
        "physical_time": run.diagnostics["physical_time"],
        "steps": run.diagnostics["steps"],
        "wall_time_s": wall,
        "final": {
            "mean_r_large": run.diagnostics["mean_radius_large_final"],
            "mean_r_small": run.diagnostics["mean_radius_small_final"],
            "sep": run.diagnostics["mean_radius_large_final"]
                   - run.diagnostics["mean_radius_small_final"],
            # time-averaged amplitudes over the last quarter of the run: the
            # single final snapshot of a 581-particle bed carries O(1/sqrt(N))
            # sampling noise, the time average over the last 4 revolutions is the
            # physically meaningful (quasi-steady) amplitude
            **_averages(run.series),
        },
        "null": null,
        "profile": {"centres": prof["centres"].tolist(),
                    "edges": prof["edges"].tolist(),
                    "phi": [None if not np.isfinite(v) else float(v) for v in prof["phi"]],
                    "count": prof["count"].tolist(),
                    "phi_0": prof["phi_0"]},
        "series": {k: np.asarray(v, float).tolist() for k, v in run.series.items()},
        "mode": mode,
        # final configuration: needed for the species maps in the report
        "final_positions": np.round(run.pos, 6).tolist(),
        "final_radii": np.round(run.radii, 6).tolist(),
        "final_is_large": run.is_large.astype(int).tolist(),
        "settle": {"steps_used": run.diagnostics["settle_steps_used"],
                   "quench_rounds": run.diagnostics["quench_rounds"],
                   "rms_speed": run.diagnostics["settle_rms_speed"],
                   "initial_ke": run.diagnostics["initial_ke"]},
        "flow": {"surface_slope_deg": run.diagnostics["surface_slope_deg"],
                 "mean_azimuthal_speed": run.diagnostics["mean_azimuthal_speed"]},
        "env": {"python": platform.python_version(), "machine": platform.machine(),
                "processor": platform.processor()},
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--revs", type=float, default=None)
    ap.add_argument("--d-small", type=float, default=None)
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cases = sweeps.build_cases(revs=args.revs, d_small=args.d_small,
                               radius=args.radius, dt=args.dt)
    if args.groups:
        cases = [c for c in cases if c["group"] in args.groups]
    if args.only:
        cases = [c for c in cases if c["name"] in args.only]
    print(f"[runner] {len(cases)} cases -> {args.out}", flush=True)

    t0 = time.time()
    with Pool(processes=min(args.jobs, len(cases))) as pool:
        for rec in pool.imap_unordered(run_case, cases):
            path = os.path.join(args.out, f"{rec['name']}.json")
            with open(path, "w") as fh:
                json.dump(rec, fh, indent=1)
            if "error" in rec:
                print(f"[runner] FAILED {rec['name']}: {rec['error'].splitlines()[-1]}",
                      flush=True)
            else:
                n = rec["null"]
                print(f"[runner] {rec['name']:>18s} mode={rec['mode']:<9s} "
                      f"excess={n['excess_observed']:+.4f} z={n['excess_z']:+.1f} "
                      f"corr={n['corr_observed']:+.4f} S={n['S_observed']:.4f} "
                      f"N={rec['n_particles']} t={rec['wall_time_s']:.0f}s", flush=True)
    print(f"[runner] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()