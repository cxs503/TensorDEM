"""CASE-COLUMN-COLLAPSE — arrival-time sampling-stride convergence (m21945).

Pre-registration: ``docs/PREREG_COLUMN_COLLAPSE_ARRIVAL_SAMPLING_m21945.md``
(written before any run; thresholds frozen there).

The debt this pays
------------------
``docs/AUDIT_COLUMN_COLLAPSE_SHALLOW_CONVERGENCE_m17917.md`` §"给下一次自找活的
建议线索" (2) and ``docs/CASE_COLUMN_COLLAPSE_SYNTHESIS_m20978.md`` §3.2 leave one
honest failure open: the emplacement-time law does **not** converge at
``a = 0.5`` (``t90`` error ``+39.5 % -> +12.6 % -> +19.8 %`` non-monotone in
resolution).  Every archived case of this card stored the front history every
5-20 integration steps, i.e. at 2-8 ms, which is the same order as the arrival
module's own 10 ms smoothing window.  Nobody has ever checked whether the
non-monotone scatter is a *sampling* artefact.

This benchmark re-integrates the five pre-registered points with
``sample_stride = 1`` (dense history) and **nothing else changed**, then
evaluates all 13 pre-registered arrival definitions at a ladder of offline
strides from the *same* trajectory -- so a stride difference is pure
post-processing, not new physics.

Usage
-----
::

    python benchmarks/case_column_collapse_arrival_sampling_m21945.py --list
    python benchmarks/case_column_collapse_arrival_sampling_m21945.py --case AS_a0.5_g8
    python benchmarks/case_column_collapse_arrival_sampling_m21945.py --shard 0/5
    python benchmarks/case_column_collapse_arrival_sampling_m21945.py --collect
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

from tensordem.granular_column.arrival import arrival_report  # noqa: E402
from tensordem.granular_column.arrival_sampling import (  # noqa: E402
    DEFAULT_STRIDES, DEFINITION_BAND_TOL, PRIMARY_KEY, SAMPLING_TOL, adjudicate,
    arrival_grid, definition_band, emplacement_errors, sampling_bound_violations,
    stride_scan)
from tensordem.granular_column.axisym import simulate_axisymmetric_column  # noqa: E402


def _load_base():
    """Import the frozen m18261 protocol module (read-only reuse)."""
    path = os.path.join(_HERE, "case_column_collapse_time_m18261.py")
    spec = importlib.util.spec_from_file_location("_cc_base_m18261", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE_MOD = _load_base()
ROUND = "m21945"
ANCHOR = "CASE-COLUMN-COLLAPSE"
TOLERANCE = 0.15
OUT_DIR = os.path.join("docs", "artifacts", "column_arrival_sampling_m21945")
CASE_DIR = os.path.join(OUT_DIR, "cases")

#: Pre-registered cases (5).  ``sample_stride = 1`` is the only protocol change.
NEW_CASES = (
    {"case_id": "AS_a0.5_g8", "axis": "S", "aspect_ratio": 0.5,
     "grains_per_radius": 8.0, "sample_stride": 1},
    {"case_id": "AS_a0.5_g12", "axis": "S", "aspect_ratio": 0.5,
     "grains_per_radius": 12.0, "sample_stride": 1},
    {"case_id": "AS_a0.5_g16", "axis": "S", "aspect_ratio": 0.5,
     "grains_per_radius": 16.0, "sample_stride": 1},
    {"case_id": "AS_a1_g8", "axis": "S", "aspect_ratio": 1.0,
     "grains_per_radius": 8.0, "sample_stride": 1},
    {"case_id": "AS_a3_g8", "axis": "S", "aspect_ratio": 3.0,
     "grains_per_radius": 8.0, "sample_stride": 1},
)
CASE_BY_ID = {c["case_id"]: c for c in NEW_CASES}


def case_path(cid: str) -> str:
    return os.path.join(CASE_DIR, f"{cid}.json")


def _error(measured: float, reference: float) -> float:
    if not (math.isfinite(measured) and math.isfinite(reference)) or reference == 0.0:
        return float("nan")
    return (measured - reference) / reference


# ---------------------------------------------------------------------------
# DEM run (cache-first, deterministic)
# ---------------------------------------------------------------------------
def run_case(cid: str, progress: bool = False) -> dict:
    if cid not in CASE_BY_ID:
        raise SystemExit(f"unknown case {cid!r}; known: {sorted(CASE_BY_ID)}")
    spec = CASE_BY_ID[cid]
    path = case_path(cid)
    if os.path.exists(path):
        with open(path) as fh:
            res = json.load(fh)
        res["cached"] = True
        return res
    case = BASE_MOD.make_case(spec)
    sig = BASE_MOD.protocol_signature(spec)
    sig["sample_stride"] = int(case.sample_stride)
    if int(case.sample_stride) != 1:
        raise SystemExit("this round requires sample_stride=1 (dense history)")
    t0 = time.perf_counter()
    res = simulate_axisymmetric_column(case, return_positions=False,
                                       progress=progress)
    wall = time.perf_counter() - t0
    hist = res["history"]
    time_hist = np.asarray(hist["time"], dtype=float)
    front_hist = np.asarray(hist["front"], dtype=float)
    radius = float(res["column_radius_measured"])
    grain_diameter = 2.0 * case.radius
    dense = arrival_report(time_hist, front_hist, radius=radius,
                           grain_diameter=grain_diameter)
    scan = stride_scan(time_hist, front_hist, radius=radius,
                       grain_diameter=grain_diameter, strides=DEFAULT_STRIDES)
    refs = BASE_MOD.reference_values(spec, res["aspect_measured"], radius,
                                     case.gravity)
    grid_dense = arrival_grid(time_hist, front_hist, radius=radius,
                              grain_diameter=grain_diameter, stride=1)
    out = {
        "case_id": cid,
        "axis": spec["axis"],
        "protocol": sig,
        "protocol_sha256": BASE_MOD.signature_sha256(sig),
        "n_particles": res["n_particles"],
        "dt": res["dt"],
        "n_steps": res["n_steps"],
        "time_end": res["time_end"],
        "stop_reason": res["stop_reason"],
        "aspect_nominal": float(spec["aspect_ratio"]),
        "aspect_measured": res["aspect_measured"],
        "column_radius_measured": radius,
        "column_height_measured": res["column_height_measured"],
        "solid_fraction_initial": res["solid_fraction_initial"],
        "runout_ratio_p995": res["runout_ratio_p995"],
        "ballistic_time": res["ballistic_time"],
        "reference": refs,
        "arrival_dense": dense,
        "stride_scan": scan,
        "definition_band_dense": definition_band(grid_dense),
        "energy": res["energy"],
        "history": {"time": time_hist.tolist(), "front": front_hist.tolist(),
                    "kinetic": hist["kinetic"]},
        "wall_clock_s": wall,
    }
    os.makedirs(CASE_DIR, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh)
    return out


def run_shard(index: int, total: int, progress: bool = False) -> list[dict]:
    if total < 1 or not (0 <= index < total):
        raise SystemExit("shard must satisfy 0 <= index < total")
    done = []
    for i, spec in enumerate(NEW_CASES):
        if i % total != index:
            continue
        done.append(run_case(spec["case_id"], progress=progress))
    return done


# ---------------------------------------------------------------------------
# reduction (pure; unit-tested without running DEM)
# ---------------------------------------------------------------------------
def case_row(case: dict) -> dict:
    """Flatten one stored case JSON into the headline row (no hand-typed values)."""
    refs = case["reference"]["measured"]
    scan = case["stride_scan"]
    t90 = scan["definitions"][PRIMARY_KEY]["times"]["1"]
    return {
        "case_id": case["case_id"],
        "aspect_nominal": float(case["aspect_nominal"]),
        "aspect_measured": float(case["aspect_measured"]),
        "grains_per_radius": float(case["protocol"]["grains_per_radius"]),
        "n_particles": int(case["n_particles"]),
        "dt": float(case["dt"]),
        "n_steps": int(case["n_steps"]),
        "n_samples_dense": int(scan["n_samples"]["1"]),
        "sample_dt_dense": float(scan["dense_sample_dt"]),
        "wall_clock_s": float(case.get("wall_clock_s", float("nan"))),
        "stop_reason": case["stop_reason"],
        "runout_ratio": float(case["runout_ratio_p995"]),
        "ref_runout_ratio": float(refs["runout_ratio"]),
        "err_runout": _error(float(case["runout_ratio_p995"]),
                             float(refs["runout_ratio"])),
        "t90_dense": float(t90),
        "ref_time": float(refs["emplacement_time"]),
        "err_t90_dense": _error(float(t90), float(refs["emplacement_time"])),
        "err_t90_stride20": _error(
            float(scan["definitions"][PRIMARY_KEY]["times"]["20"]),
            float(refs["emplacement_time"])),
        "sampling_bias_t90": float(
            scan["definitions"][PRIMARY_KEY]["bias_rel"]["20"]),
        "band_rel_dense": float(case["definition_band_dense"]["spread_rel"]),
        "band_min_definition": case["definition_band_dense"]["min_definition"],
        "band_max_definition": case["definition_band_dense"]["max_definition"],
    }


def judge(rows: list[dict], cases: dict | None = None) -> dict:
    """Apply the pre-registered H1..H3 rules (thresholds frozen in the prereg)."""
    missing = [c["case_id"] for c in NEW_CASES
               if not any(r["case_id"] == c["case_id"] for r in rows)]
    if missing:
        return {"round": ROUND, "anchor": ANCHOR, "verdict": "incomplete",
                "missing": missing,
                "reason": "pre-registered cases not all collected"}
    h1 = {r["case_id"]: {
        "sampling_bias_t90": r["sampling_bias_t90"],
        "tolerance": SAMPLING_TOL,
        "verdict": "pass" if abs(r["sampling_bias_t90"]) < SAMPLING_TOL
                   else "fail"}
        for r in rows}
    h1_verdict = ("pass" if all(v["verdict"] == "pass" for v in h1.values())
                  else "fail")
    h2 = {r["case_id"]: {
        "band_rel": r["band_rel_dense"],
        "min_definition": r["band_min_definition"],
        "max_definition": r["band_max_definition"],
        "threshold": DEFINITION_BAND_TOL,
        "verdict": "pass" if r["band_rel_dense"] > DEFINITION_BAND_TOL
                   else "fail"}
        for r in rows}
    h2_verdict = ("pass" if all(v["verdict"] == "pass" for v in h2.values())
                  else "fail")
    shallow = sorted((r for r in rows if abs(r["aspect_nominal"] - 0.5) < 1e-9),
                     key=lambda r: r["grains_per_radius"])
    if len(shallow) < 3:
        h3 = {"verdict": "incomplete",
              "reason": "need the three a=0.5 resolution rungs"}
    else:
        e = [r["err_t90_dense"] for r in shallow]
        mid = e[1]
        monotone = (e[0] <= mid <= e[2]) or (e[0] >= mid >= e[2])
        h3 = {"errors_by_gpr": {f"{r['grains_per_radius']:g}": r["err_t90_dense"]
                                for r in shallow},
              "spread_pp": 100.0 * (max(e) - min(e)),
              "monotone": bool(monotone),
              "verdict": "fail" if monotone else "pass",
              "reason": ("monotone in resolution => the shallow-end scatter is a "
                         "sampling artefact" if monotone else
                         "still non-monotone => the shallow-end non-convergence is "
                         "not a sampling artefact")}
    if cases:
        for r in rows:
            case = cases.get(r["case_id"])
            if case is None:
                continue
            grid = arrival_grid(np.asarray(case["history"]["time"]),
                                np.asarray(case["history"]["front"]),
                                radius=float(case["column_radius_measured"]),
                                grain_diameter=2.0 * float(case["protocol"]["radius"]),
                                stride=1)
            r["adjudication"] = adjudicate(
                emplacement_errors(grid, r["ref_time"]), tolerance=TOLERANCE)
    verdicts = [h1_verdict, h2_verdict, h3["verdict"]]
    overall = "pass" if all(v == "pass" for v in verdicts) else "fail"
    return {
        "round": ROUND, "anchor": ANCHOR, "tolerance": TOLERANCE,
        "sampling_tol": SAMPLING_TOL, "definition_band_tol": DEFINITION_BAND_TOL,
        "primary_definition": PRIMARY_KEY,
        "H1_sampling_artefact": {"verdict": h1_verdict, "per_case": h1},
        "H2_definition_dominated": {"verdict": h2_verdict, "per_case": h2},
        "H3_shallow_non_monotone": h3,
        "verdict": overall,
        "verdicts": verdicts,
    }


def collect() -> dict:
    rows, cases = [], {}
    for spec in NEW_CASES:
        path = case_path(spec["case_id"])
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            case = json.load(fh)
        cases[spec["case_id"]] = case
        rows.append(case_row(case))
    summary = judge(rows, cases)
    summary["cases"] = rows
    summary["sampling_bound_violations"] = {
        cid: sampling_bound_violations(case["stride_scan"])
        for cid, case in cases.items()}
    summary["total_bound_violations"] = sum(
        len(v) for v in summary["sampling_bound_violations"].values())
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    if rows:
        keys = [k for k in rows[0] if k != "adjudication"]
        with open(os.path.join(OUT_DIR, "cases.csv"), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", "--list-cases", action="store_true", dest="list")
    ap.add_argument("--case")
    ap.add_argument("--shard", help="i/n index of this worker")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        for spec in NEW_CASES:
            print(f"{spec['case_id']:<14} a={spec['aspect_ratio']:<4g} "
                  f"gpr={spec['grains_per_radius']:<4g} "
                  f"stride={spec['sample_stride']}")
        return 0
    if args.collect:
        summary = collect()
        print(json.dumps({k: v for k, v in summary.items()
                          if k not in ("cases", "sampling_bound_violations")},
                         indent=2))
        return 0
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        done = run_shard(i, n, progress=args.progress)
        print(json.dumps([{"case_id": d["case_id"],
                           "n_particles": d["n_particles"],
                           "wall_clock_s": d["wall_clock_s"],
                           "stop_reason": d["stop_reason"]} for d in done],
                         indent=2))
        return 0
    if args.case:
        res = run_case(args.case, progress=args.progress)
        print(json.dumps({"case_id": res["case_id"],
                          "n_particles": res["n_particles"],
                          "n_steps": res["n_steps"],
                          "n_samples": res["stride_scan"]["n_samples"]["1"],
                          "runout_ratio_p995": res["runout_ratio_p995"],
                          "wall_clock_s": res["wall_clock_s"],
                          "stop_reason": res["stop_reason"]}, indent=2))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())