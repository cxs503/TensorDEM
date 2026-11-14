"""CASE-COLUMN-COLLAPSE — square-root branch (a=3) third refinement rung.

算例池锚点 CASE-COLUMN-COLLAPSE 的验证欠账（m20931，预登记见
``docs/PREREG_COLUMN_COLLAPSE_SQRT_BRANCH_m20931.md``）。

m20873 的 a=3 阶梯只有两档（gpr 8 → +8.09 %, gpr 12 → +15.40 %，超 15 % 判据
0.40 pp），两档不足以区分「离散化未收敛」与「平方根支系统性偏差」。本脚本补
第三档 ``SB_a3_g16``（gpr 16, N≈4.7 万），**不改** m18261/m20873 的冻结协议，
只读引用它们的逐算例 JSON 作阶梯前两档，然后按跑前声明的 H1/H2/H3 分档裁决。

Reference law (evaluated at run time, never transcribed) — Lube, Huppert,
Sparks & Freundt (2004), JFM **508**:175-199, doi:10.1017/S0022112004009036::

    r_inf/ri = 1 + 1.6 a^(1/2)   (a > 1.7)
    t_inf    = 3 (ri a / g)^(1/2)

Usage
-----
::

    python benchmarks/case_column_collapse_sqrt_branch_m20931.py --case
    python benchmarks/case_column_collapse_sqrt_branch_m20931.py --collect
"""
from __future__ import annotations

import argparse
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


def _load_base():
    """Import the frozen m18261 benchmark module (read-only reuse)."""
    path = os.path.join(_HERE, "case_column_collapse_time_m18261.py")
    spec = importlib.util.spec_from_file_location("_cc_time_m18261", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE_MOD = _load_base()
from tensordem.granular_column.arrival import arrival_report  # noqa: E402

ROUND = "m20931"
ANCHOR = "CASE-COLUMN-COLLAPSE"
TOLERANCE = 0.15
OUT_DIR = os.path.join("docs", "artifacts", "column_collapse_sqrt_branch_m20931")
CASE_DIR = os.path.join(OUT_DIR, "cases")
#: Archived provenance of the two rungs that already exist (m20873, read-only).
ARCHIVE_DIR = os.path.join("docs", "artifacts",
                           "column_collapse_deep_refinement_m20873", "cases")
#: The single pre-registered case added by this round.
CASE = {"case_id": "SB_a3_g16", "axis": "R", "aspect_ratio": 3.0,
        "grains_per_radius": 16.0}
#: Pre-registered ladder (gpr ascending) and its archived case ids.
LADDER = ((8.0, "R_a3_g8"), (12.0, "R_a3_g12"), (16.0, CASE["case_id"]))
#: Primary arrival definitions frozen by m20873 (not redefined here).
PRIMARY = ("fraction_0.9", "velocity_0.01", "length_1d")


def case_path(cid: str) -> str:
    return os.path.join(CASE_DIR, f"{cid}.json")


def archive_path(cid: str) -> str:
    return os.path.join(ARCHIVE_DIR, f"{cid}.json")


def _error(measured: float, reference: float) -> float:
    if not (math.isfinite(measured) and math.isfinite(reference)) or reference == 0.0:
        return float("nan")
    return (measured - reference) / reference


def run_case(progress: bool = False) -> dict:
    """Integrate ``SB_a3_g16`` (cache-first, deterministic)."""
    path = case_path(CASE["case_id"])
    if os.path.exists(path):
        with open(path) as fh:
            res = json.load(fh)
        res["cached"] = True
        return res
    case = BASE_MOD.make_case(CASE)
    sig = BASE_MOD.protocol_signature(CASE)
    sig["round"] = ROUND
    t0 = time.perf_counter()
    res = BASE_MOD.simulate_axisymmetric_column(case, return_positions=False,
                                               progress=progress)
    wall = time.perf_counter() - t0
    hist = res["history"]
    arrivals = arrival_report(np.asarray(hist["time"], dtype=float),
                              np.asarray(hist["front"], dtype=float),
                              radius=res["column_radius_measured"],
                              grain_diameter=2.0 * case.radius)
    refs = BASE_MOD.reference_values(CASE, res["aspect_measured"],
                                     res["column_radius_measured"],
                                     case.gravity)
    out = {
        "case_id": CASE["case_id"],
        "axis": CASE["axis"],
        "protocol": sig,
        "protocol_sha256": BASE_MOD.signature_sha256(sig),
        "n_particles": res["n_particles"],
        "dt": res["dt"],
        "n_steps": res["n_steps"],
        "time_end": res["time_end"],
        "stop_reason": res["stop_reason"],
        "aspect_nominal": float(CASE["aspect_ratio"]),
        "aspect_measured": res["aspect_measured"],
        "column_radius_measured": res["column_radius_measured"],
        "column_height_measured": res["column_height_measured"],
        "solid_fraction_initial": res["solid_fraction_initial"],
        "runout_ratio_p995": res["runout_ratio_p995"],
        "ballistic_time": res["ballistic_time"],
        "reference": refs,
        "arrival": arrivals,
        "energy": res["energy"],
        "inertial": res["inertial"],
        "coordination_max": res["coordination_max"],
        "wall_clock_s": wall,
    }
    for conv in ("nominal", "measured"):
        out[f"runout_error_{conv}"] = _error(res["runout_ratio_p995"],
                                            refs[conv]["runout_ratio"])
        out[f"t_error_{conv}"] = {
            key: _error(value, refs[conv]["emplacement_time"])
            for key, value in arrivals["primary"].items()}
    out["within_tolerance_runout"] = (abs(out["runout_error_nominal"]) <= TOLERANCE)
    os.makedirs(CASE_DIR, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh)
    return out


# ---------------------------------------------------------------------------
# analysis (pure functions: unit-tested without running DEM)
# ---------------------------------------------------------------------------
def ladder_row(path: str, gpr: float) -> dict:
    """Reduce one stored case JSON to the ladder row (no hand-typed numbers)."""
    with open(path) as fh:
        case = json.load(fh)
    refs = case["reference"]
    ref = float(refs["measured"]["runout_ratio"])
    runout = float(case["runout_ratio_p995"])
    return {
        "case_id": case["case_id"],
        "grains_per_radius": float(gpr),
        "source": "this mission (m20931)" if case["case_id"] == CASE["case_id"]
                  else f"archive m20873 ({path})",
        "n_particles": int(case["n_particles"]),
        "dt": float(case["dt"]),
        "n_steps": int(case["n_steps"]),
        "wall_clock_s": float(case.get("wall_clock_s", float("nan"))),
        "stop_reason": case["stop_reason"],
        "aspect_nominal": float(case["aspect_nominal"]),
        "aspect_measured": float(case["aspect_measured"]),
        "runout_ratio": runout,
        "ref_runout_ratio": ref,
        "err_runout": _error(runout, ref),
        "err_runout_nominal": _error(runout, float(refs["nominal"]["runout_ratio"])),
        "t90": float(case["arrival"]["primary"]["fraction_0.9"]),
        "ref_time": float(refs["measured"]["emplacement_time"]),
        "err_t90": _error(float(case["arrival"]["primary"]["fraction_0.9"]),
                          float(refs["measured"]["emplacement_time"])),
        "energy_residual": float(case["energy"]["residual_fraction_of_potential"]),
    }


def richardson_bracket(rows: list[dict], orders=(1.0, 2.0)) -> dict:
    """Indicative extrapolation from the two finest rungs (assumed order p).

    ``f_inf ~ f3 + (f3 - f2) / (r^p - 1)`` with ``r = gpr3 / gpr2``.
    Indicative only: the ladder has three points, so the observed order cannot
    be measured independently of the extrapolation.
    """
    if len(rows) < 2:
        return {"available": False}
    f2, f3 = rows[-2]["runout_ratio"], rows[-1]["runout_ratio"]
    g2, g3 = rows[-2]["grains_per_radius"], rows[-1]["grains_per_radius"]
    ratio = g3 / g2
    out = {"available": True, "gpr_pair": [g2, g3], "ratio": ratio,
           "f_finest": f3, "ref": rows[-1]["ref_runout_ratio"]}
    for p in orders:
        ext = f3 + (f3 - f2) / (ratio ** p - 1.0)
        out[f"f_inf_p{p:g}"] = ext
        out[f"err_inf_p{p:g}"] = _error(ext, rows[-1]["ref_runout_ratio"])
    return out


def difference_ratio(rows: list[dict]) -> float:
    """``|Δ23| / |Δ12|`` in the runout *ratio* (not the error); <1 → converging."""
    if len(rows) < 3:
        return float("nan")
    d12 = rows[1]["runout_ratio"] - rows[0]["runout_ratio"]
    d23 = rows[2]["runout_ratio"] - rows[1]["runout_ratio"]
    if d12 == 0.0:
        return float("nan")
    return abs(d23) / abs(d12)


def judge(rows: list[dict]) -> dict:
    """Apply the pre-registered H1/H2/H3 rule to the a=3 ladder."""
    if len(rows) < 3:
        return {"hypothesis": "incomplete", "verdict": "incomplete",
                "tolerance": TOLERANCE, "reason": "ladder incomplete",
                "rows_present": [r["case_id"] for r in rows],
                "abs_delta_ratio": None}
    e16 = rows[-1]["err_runout"]
    d12 = rows[1]["runout_ratio"] - rows[0]["runout_ratio"]
    d23 = rows[2]["runout_ratio"] - rows[1]["runout_ratio"]
    contracting = abs(d23) < abs(d12)
    in_band = abs(e16) <= TOLERANCE
    rich = richardson_bracket(rows)
    if in_band and contracting:
        hyp, verdict = "H1", "pass"
        reading = ("in-band at gpr=16 with contracting increments: the earlier "
                   "15.40 % overshoot was discretisation-limited")
    elif (not in_band) and (not contracting):
        hyp, verdict = "H2", "fail"
        reading = ("out of band AND increments non-contracting: the square-root "
                   "branch overshoot is systematic, not resolution-limited")
    else:
        hyp, verdict = "H3", "incomplete"
        reading = ("contracting but still out of band: converging, not converged; "
                   "reported as incomplete (no extrapolated pass)")
    return {
        "hypothesis": hyp, "verdict": verdict, "tolerance": TOLERANCE,
        "err_gpr8": rows[0]["err_runout"],
        "err_gpr12": rows[1]["err_runout"],
        "err_gpr16": e16,
        "delta_12": d12, "delta_23": d23,
        "abs_delta_ratio": difference_ratio(rows),
        "contracting": contracting, "in_band_at_finest": in_band,
        "richardson": rich, "reading": reading,
        "energy_residual": [r["energy_residual"] for r in rows],
        #: Informational only: the ledger closure is *reported*, never gated on a
        #: post-hoc threshold (archived a=3 rungs already sit at 11-12 %).
        "energy_flagged": [r["case_id"] for r in rows
                           if abs(r["energy_residual"]) > 0.10],
        "stop_reasons": [r["stop_reason"] for r in rows],
    }


def collect() -> dict:
    rows, missing = [], []
    for gpr, cid in LADDER:
        path = case_path(cid) if cid == CASE["case_id"] else archive_path(cid)
        if not os.path.exists(path):
            missing.append(cid)
            continue
        rows.append(ladder_row(path, gpr))
    payload = {
        "round": ROUND, "anchor": ANCHOR, "tolerance": TOLERANCE,
        "branch": "square-root (a = 3 > 1.7)", "aspect_nominal": 3.0,
        "rows": rows, "missing": missing, "verdict": judge(rows),
        "prereg": "docs/PREREG_COLUMN_COLLAPSE_SQRT_BRANCH_m20931.md",
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "verdict.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    header = ["case_id", "grains_per_radius", "n_particles", "dt", "n_steps",
              "stop_reason", "aspect_nominal", "aspect_measured",
              "runout_ratio", "ref_runout_ratio", "err_runout",
              "err_runout_nominal", "t90", "ref_time", "err_t90",
              "wall_clock_s"]
    with open(os.path.join(OUT_DIR, "cases_sqrt_branch.csv"), "w") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k])
                              for k in header) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", action="store_true",
                    help=f"integrate {CASE['case_id']}")
    ap.add_argument("--collect", action="store_true",
                    help="reduce stored cases into verdict.json + csv")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args(argv)
    if args.case:
        res = run_case(progress=args.progress)
        print(f"{res['case_id']}: N={res['n_particles']} dt={res['dt']:.3e} "
              f"steps={res['n_steps']} runout={res['runout_ratio_p995']:.4f} "
              f"err={res['runout_error_measured']:+.2%} "
              f"stop={res['stop_reason']} wall={res['wall_clock_s']:.0f}s",
              flush=True)
    if args.collect:
        payload = collect()
        print(json.dumps(payload["verdict"], indent=1))
    if not (args.case or args.collect):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())