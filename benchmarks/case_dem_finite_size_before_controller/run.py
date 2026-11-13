#!/usr/bin/env python
"""XVAL-DEM-FINITE-SIZE-BEFORE-CONTROLLER: cross-validation of *where* the
periodic-box (finite-size) drag enhancement is applied relative to the flow
controller / settling drive.

Two independent shapes of the same question are measured with the repo's own
LBM--DEM kernels (no new solver):

A. free settling -- the drive (effective weight) fixes the operating point, so a
   correction applied *after* the run leaves the reached Re biased while a
   correction applied *before* the drive (and not repeated post hoc) lands on the
   target Re.  Three placements: M0 post-hoc only (current path), M1 drive and
   post-hoc (double counting), M2 drive only (the practice under test).

B. body-force-driven fixed sphere with the constant-flow-rate pump -- here the
   controller pins the mean velocity, so the placement can only change the
   transient; this isolates the "convergence-budget" part of the practice.

C. a phi-scan at fixed resolution plus a resolution scan at fixed L/d, which
   separates the analytic finite-size ratio from the immersed-boundary error.

Every number in the result file comes from ``run_fixed_sphere`` / ``run_settling``;
analysis code only derives ratios from those dictionaries.

    python benchmarks/case_dem_finite_size_before_controller/run.py --workers 6
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = HERE / "runs"

# ---------------------------------------------------------------------------
# case list (pre-declared in prereg.json, not edited after inspecting results)
# ---------------------------------------------------------------------------
# The viscous relaxation time of the box is ~L^2/nu (5760 steps at L=24), so the
# settling budget is 12000 steps: this was fixed after a 4000-step pilot showed an
# 18.9 % window-to-window drift (a budget-limited, not steady, measurement).
SETTLE_BASE = dict(diameter=6.0, density_ratio=2.0, re_target=1.0, tau0=0.8,
                   scheme="mem", steps=12000, sample_every=5, window=400,
                   dtype="float32")

FIXED_BASE = dict(diameter=6.0, re_target=0.05, scheme="mem", tau0=0.8,
                  dtype="float32", max_steps=3000, min_steps=500,
                  sample_every=50, rtol=1.0e-4, flow_control=True)


def case_list() -> list[dict]:
    cases: list[dict] = []
    # -- A: settling, correction placement at two box sizes -------------------
    placements = {
        "M0_posthoc": dict(finite_size_in_drive=False, apply_array_correction_posthoc=True),
        "M1_drive_and_posthoc": dict(finite_size_in_drive=True, apply_array_correction_posthoc=True),
        "M2_drive_only": dict(finite_size_in_drive=True, apply_array_correction_posthoc=False),
    }
    # L/d=8 is run at d=4 (L=32) to keep the 12000-step budget affordable; the
    # immersed-boundary error is then a constant offset shared by M0 and M2, so
    # the placement contrast at that box size is not confounded by it.
    settle_boxes = {4.0: dict(diameter=6.0), 8.0: dict(diameter=4.0)}
    for ld, override in settle_boxes.items():
        names = ("M0_posthoc", "M2_drive_only") if ld == 8.0 else tuple(placements)
        for name in names:
            flags = placements[name]
            cases.append({
                "label": f"settle_ld{ld:g}_{name}",
                "kind": "settling",
                "spec": {**SETTLE_BASE, **override, "domain_ratio": ld, **flags,
                         "label": f"settle_ld{ld:g}_{name}"},
            })
    # -- B: fixed sphere, pump-controller seed placement ---------------------
    for gain in (0.005, 0.02):
        for timing in ("before", "after"):
            cases.append({
                "label": f"fixed_ld4_g{gain:g}_{timing}",
                "kind": "fixed",
                "spec": {**FIXED_BASE, "domain_ratio": 4.0, "controller_gain": gain,
                         "correction_timing": timing,
                         "label": f"fixed_ld4_g{gain:g}_{timing}"},
            })
    # -- C: phi scan at fixed resolution (d=6) -------------------------------
    for ld in (6.0, 8.0):
        cases.append({
            "label": f"fixed_phi_ld{ld:g}_d6",
            "kind": "fixed",
            "spec": {**FIXED_BASE, "domain_ratio": ld, "correction_timing": "before",
                     "label": f"fixed_phi_ld{ld:g}_d6"},
        })
    # fixed_ld4_g0.02_before above doubles as the phi=0.0654 d=6 point.
    # -- D: resolution scan at fixed L/d=8, and a fixed-resolution phi pair ---
    cases.append({
        "label": "fixed_phi_ld8_d4",
        "kind": "fixed",
        "spec": {**FIXED_BASE, "diameter": 4.0, "domain_ratio": 8.0,
                 "correction_timing": "before", "label": "fixed_phi_ld8_d4"},
    })
    cases.append({
        "label": "fixed_phi_ld16_d4",
        "kind": "fixed",
        "spec": {**FIXED_BASE, "diameter": 4.0, "domain_ratio": 16.0,
                 "max_steps": 3000, "correction_timing": "before",
                 "label": "fixed_phi_ld16_d4"},
    })
    cases.append({
        "label": "fixed_phi_ld8_d8",
        "kind": "fixed",
        "spec": {**FIXED_BASE, "diameter": 8.0, "domain_ratio": 8.0,
                 "correction_timing": "before", "label": "fixed_phi_ld8_d8"},
    })
    return cases


# ---------------------------------------------------------------------------
# derivatives
# ---------------------------------------------------------------------------
def _steps_to_within(history: list[dict], key: str, target: float, rel: float) -> int | None:
    scale = abs(target) if target else 1.0
    for row in history:
        if abs(row[key] - target) <= rel * scale:
            return int(row["step"])
    return None


def _value_at(history: list[dict], key: str, step: int) -> float | None:
    for row in history:
        if int(row["step"]) >= step:
            return float(row[key])
    return None


def derive(case: dict, result: dict) -> dict:
    """Add the pre-declared metrics on top of the raw case dictionary."""
    history = result.get("history") or []
    out = dict(result)
    if case["kind"] == "fixed":
        final_force = history[-1]["force"] if history else result["force_total"]
        out["steps_to_within_2pct"] = _steps_to_within(history, "force", final_force, 0.02)
        out["converged_flag"] = bool(result["converged"])
        # ratio evaluated at a fixed early budget (600 steps) and at the end
        for budget in (600, 1200):
            f = _value_at(history, "force", budget)
            u = _value_at(history, "u_mean", budget)
            if f is not None and u:
                import math as _m
                out[f"stokes_ratio_corrected_at_{budget}"] = (
                    f / (result["n_spheres"] * 3.0 * _m.pi * result["nu"]
                         * result["diameter"] * u) / result["array_correction"]
                )
        out["stokes_ratio_unbounded_model"] = result["array_correction"]
        out["xval_error_vs_hasimoto"] = result["stokes_ratio"] / result["array_correction"] - 1.0
    else:
        hist = [row for row in history if "velocity" in row]
        if hist:
            w = result.get("window", 200)
            tail = [row["velocity"] for row in hist[-w:]]
            prev = [row["velocity"] for row in hist[-2 * w:-w]]
            u_tail = sum(tail) / len(tail)
            u_prev = sum(prev) / len(prev) if prev else u_tail
            out["steadiness"] = abs(u_tail - u_prev) / abs(u_tail) if u_tail else None
            out["steps_to_within_2pct_velocity"] = _steps_to_within(
                hist, "velocity", u_tail, 0.02)
            out["u_terminal_tail"] = u_tail
        out["predicted_re_vs_target"] = (
            1.0 / result["array_correction"]
            if not result["finite_size_in_drive"]
            else 1.0
        ) - 1.0
        posthoc = result["array_correction"] if result["apply_array_correction_posthoc"] else 1.0
        drive = result["array_correction"] if result["finite_size_in_drive"] else 1.0
        out["u_reported_vs_target"] = (1.0 + result["u_relative_vs_target"]) * posthoc - 1.0
        out["predicted_u_reported_vs_target"] = (
            (1.0 / result["array_correction"]
             if not result["finite_size_in_drive"] else 1.0)
            * (result["array_correction"] if result["apply_array_correction_posthoc"] else 1.0)
        ) - 1.0
    # keep the JSON small: thin the history for the artifact
    keep = history[::max(1, len(history) // 80)] if history else []
    out["history"] = keep
    return out


def _run_one(case: dict) -> tuple[dict, dict]:
    import torch

    torch.set_num_threads(int(os.environ.get("TD_THREADS", "16")))
    sys.path.insert(0, str(REPO / "src"))
    from tensordem.lbmdem.calibration import (
        FixedSphereSpec, SettlingSpec, run_fixed_sphere, run_settling)

    t0 = time.time()
    if case["kind"] == "fixed":
        raw = run_fixed_sphere(FixedSphereSpec(**case["spec"]))
    else:
        raw = run_settling(SettlingSpec(**case["spec"]))
    wall = time.time() - t0
    row = derive(case, raw)
    row["wall_seconds"] = wall
    return case, row


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default=None, help="comma separated label substrings")
    args = ap.parse_args()

    cases = case_list()
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        cases = [c for c in cases if any(k in c["label"] for k in keys)]
    RUNS.mkdir(parents=True, exist_ok=True)
    casedir = RUNS / "cases"
    casedir.mkdir(exist_ok=True)

    # Each finished case is persisted immediately, so a long sweep can be
    # resumed (and a crash cannot lose finished work).  This is why a partial
    # result file is trustworthy: only fully returned dictionaries are written.
    done: dict[str, dict] = {}
    for case in cases:
        path = casedir / f"{case['label']}.json"
        if path.exists():
            done[case["label"]] = json.loads(path.read_text())
    todo = [c for c in cases if c["label"] not in done]
    print(f"{len(cases)} cases ({len(done)} resumed) on {args.workers} workers", flush=True)

    results: list[dict] = list(done.values())
    if todo:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            for i, (case, row) in enumerate(pool.imap_unordered(_run_one, todo), 1):
                (casedir / f"{row['label']}.json").write_text(json.dumps(row, indent=1))
                results.append(row)
                print(f"[{i}/{len(todo)}] {row['label']}: wall={row['wall_seconds']:.1f}s "
                      f"corr={row['array_correction']:.4f}", flush=True)

    results.sort(key=lambda r: r["label"])
    payload = {
        "mission_id": 17417,
        "anchor": "XVAL-DEM-FINITE-SIZE-BEFORE-CONTROLLER",
        "prereg": json.loads((HERE / "prereg.json").read_text()),
        "environment": {
            "git_sha": _git_sha(),
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
            "host": platform.node(),
            "nproc": os.cpu_count(),
            "workers": args.workers,
            "td_threads": int(os.environ.get("TD_THREADS", "16")),
            "cluster_queue_depth_at_submit": 517,
        },
        "cases": results,
    }
    (RUNS / "result.json").write_text(json.dumps(payload, indent=1))
    total = sum(r["wall_seconds"] for r in results)
    print(f"wrote {RUNS / 'result.json'}  ({len(results)} cases, "
          f"{total:.0f} s CPU, makespan {max(r['wall_seconds'] for r in results):.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())