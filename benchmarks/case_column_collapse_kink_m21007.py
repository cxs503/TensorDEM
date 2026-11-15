"""CASE-COLUMN-COLLAPSE — 断点邻域扫描（kink neighbourhood, a ~ 1.7）.

算例池锚点 ``CASE-COLUMN-COLLAPSE`` 的验证欠账（m21007）。预登记见
``docs/PREREG_COLUMN_COLLAPSE_KINK_m21007.md``（跑前写定）。

待补的欠账（``docs/CASE_COLUMN_COLLAPSE_SYNTHESIS_m20978.md`` §⑤ 未做项之三）:
Lube et al. (2004) 的跑距律是**分段**的::

    r_inf/ri = 1 + 1.24 a         (a < 1.7)
    r_inf/ri = 1 + 1.6  a^(1/2)   (a > 1.7)

两支斜率在断点处相差 2.02 倍。已测点在断点两侧符号相反（a=1.5 -> -5.7 %，
a=2.0 -> +4.6 %），误差在 (1.5, 2.0) 内穿零，但 1.5 与 2.0 之间**一个点都没有**。
本单补四个 gpr=12 点（a = 1.5 / 1.6 / 1.7 / 1.8）加一个断点处分辨率对照
（a = 1.7, gpr = 8），检验断点邻域的入带性与「先陡后平」形状。

不改 m18261/m20873/m20931 的任何冻结协议，只**只读复用**它们的
``make_case``/``reference_values``/``protocol_signature`` 与归档逐算例 JSON。

Usage
-----
::

    python benchmarks/case_column_collapse_kink_m21007.py --list
    python benchmarks/case_column_collapse_kink_m21007.py --case KN_a1.7_g12
    python benchmarks/case_column_collapse_kink_m21007.py --shard 0/5
    python benchmarks/case_column_collapse_kink_m21007.py --collect
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

from tensordem.granular_column.arrival import arrival_report  # noqa: E402
from tensordem.granular_column.axisym import (  # noqa: E402
    LUBE2004_INERTIAL_PREFACTOR, LUBE2004_LINEAR_PREFACTOR,
    LUBE2004_REGIME_BREAK, lube_2004_runout_ratio,
    simulate_axisymmetric_column)


def _load_base():
    """Import the frozen m18261 protocol module (read-only reuse)."""
    path = os.path.join(_HERE, "case_column_collapse_time_m18261.py")
    spec = importlib.util.spec_from_file_location("_cc_base_m18261", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE_MOD = _load_base()
ROUND = "m21007"
ANCHOR = "CASE-COLUMN-COLLAPSE"
TOLERANCE = 0.15
OUT_DIR = os.path.join("docs", "artifacts", "column_collapse_kink_m21007")
CASE_DIR = os.path.join(OUT_DIR, "cases")
#: Archived gpr=12 anchor on the linear branch (read-only, m20873).
ARCHIVE_DIR = os.path.join("docs", "artifacts", "column_collapse_deep_refinement_m20873",
                           "cases")
ARCHIVE_CASES = ((1.0, "R_a1_g12"),)

#: Pre-registered new cases (5).  ``gpr`` is the only protocol knob that differs.
NEW_CASES = (
    {"case_id": "KN_a1.5_g12", "axis": "K", "aspect_ratio": 1.5,
     "grains_per_radius": 12.0},
    {"case_id": "KN_a1.6_g12", "axis": "K", "aspect_ratio": 1.6,
     "grains_per_radius": 12.0},
    {"case_id": "KN_a1.7_g12", "axis": "K", "aspect_ratio": 1.7,
     "grains_per_radius": 12.0},
    {"case_id": "KN_a1.8_g12", "axis": "K", "aspect_ratio": 1.8,
     "grains_per_radius": 12.0},
    {"case_id": "KN_a1.7_g8", "axis": "K", "aspect_ratio": 1.7,
     "grains_per_radius": 8.0},
)
CASE_BY_ID = {c["case_id"]: c for c in NEW_CASES}
#: Pre-registered H2/H3/H4 thresholds (pp = percentage points).
MAX_ERROR_JUMP_PP = 5.0
MIN_SLOPE_RATIO = 1.0


def case_path(cid: str) -> str:
    return os.path.join(CASE_DIR, f"{cid}.json")


def archive_path(cid: str) -> str:
    return os.path.join(ARCHIVE_DIR, f"{cid}.json")


def _error(measured: float, reference: float) -> float:
    if not (math.isfinite(measured) and math.isfinite(reference)) or reference == 0.0:
        return float("nan")
    return (measured - reference) / reference


# ---------------------------------------------------------------------------
# analytic reference-law facts (no DEM needed)
# ---------------------------------------------------------------------------
def crossover_aspect() -> float:
    """Aspect ratio where the two Lube (2004) branches intersect."""
    return (LUBE2004_INERTIAL_PREFACTOR / LUBE2004_LINEAR_PREFACTOR) ** 2


def law_discontinuity(aspect: float = LUBE2004_REGIME_BREAK) -> dict:
    """Relative jump of the *reference law* at the pre-registered break.

    The card states the break at ``a = 1.7``, the two branches actually cross at
    ``a* = (1.6/1.24)^2 = 1.665``.  Their values at the card's break differ
    slightly; that difference is the law's own discontinuity and must be
    quantified before any DEM error jump is blamed on the model.
    """
    a = float(aspect)
    linear = 1.0 + LUBE2004_LINEAR_PREFACTOR * a
    root = 1.0 + LUBE2004_INERTIAL_PREFACTOR * math.sqrt(a)
    return {
        "aspect": a,
        "value_linear_branch": linear,
        "value_sqrt_branch": root,
        "absolute_jump": linear - root,
        "relative_jump": (linear - root) / root,
        "crossover_aspect": crossover_aspect(),
        "break_minus_crossover": a - crossover_aspect(),
        "linear_slope_at_break": LUBE2004_LINEAR_PREFACTOR,
        "sqrt_slope_at_break": 0.5 * LUBE2004_INERTIAL_PREFACTOR / math.sqrt(a),
        "slope_ratio_across_break": (LUBE2004_LINEAR_PREFACTOR
                                     / (0.5 * LUBE2004_INERTIAL_PREFACTOR
                                        / math.sqrt(a))),
    }


def local_slope(a_lo: float, r_lo: float, a_hi: float, r_hi: float) -> float:
    """Finite-difference slope ``d(r_inf/r_i)/da`` between two measured points."""
    if a_hi == a_lo:
        return float("nan")
    return (float(r_hi) - float(r_lo)) / (float(a_hi) - float(a_lo))


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
    sig["round"] = ROUND
    t0 = time.perf_counter()
    res = simulate_axisymmetric_column(case, return_positions=False,
                                       progress=progress)
    wall = time.perf_counter() - t0
    hist = res["history"]
    arrivals = arrival_report(np.asarray(hist["time"], dtype=float),
                              np.asarray(hist["front"], dtype=float),
                              radius=res["column_radius_measured"],
                              grain_diameter=2.0 * case.radius)
    refs = BASE_MOD.reference_values(spec, res["aspect_measured"],
                                     res["column_radius_measured"],
                                     case.gravity)
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
    os.makedirs(CASE_DIR, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh)
    return out


# ---------------------------------------------------------------------------
# analysis (pure functions; unit-tested without running DEM)
# ---------------------------------------------------------------------------
def point_row(path: str, source: str) -> dict:
    """Reduce one stored case JSON to a scan row (no hand-typed numbers)."""
    with open(path) as fh:
        case = json.load(fh)
    refs = case["reference"]
    runout = float(case["runout_ratio_p995"])
    ref_meas = float(refs["measured"]["runout_ratio"])
    t90 = float(case["arrival"]["primary"]["fraction_0.9"])
    return {
        "case_id": case["case_id"],
        "source": source,
        "aspect_nominal": float(case["aspect_nominal"]),
        "aspect_measured": float(case["aspect_measured"]),
        "grains_per_radius": float(case["protocol"]["grains_per_radius"]),
        "n_particles": int(case["n_particles"]),
        "dt": float(case["dt"]),
        "n_steps": int(case["n_steps"]),
        "wall_clock_s": float(case.get("wall_clock_s", float("nan"))),
        "stop_reason": case["stop_reason"],
        "runout_ratio": runout,
        "ref_runout_ratio": ref_meas,
        "err_runout": _error(runout, ref_meas),
        "err_runout_nominal": _error(runout, float(refs["nominal"]["runout_ratio"])),
        "t90": t90,
        "ref_time": float(refs["measured"]["emplacement_time"]),
        "err_t90": _error(t90, float(refs["measured"]["emplacement_time"])),
        "energy_residual": float(case["energy"]["residual_fraction_of_potential"]),
    }


def _find(rows: list[dict], case_id: str) -> dict | None:
    for row in rows:
        if row["case_id"] == case_id:
            return row
    return None


def judge(rows: list[dict]) -> dict:
    """Apply the pre-registered H1..H4 rules (thresholds fixed in the prereg)."""
    g12 = [_find(rows, c) for c in ("KN_a1.5_g12", "KN_a1.6_g12",
                                    "KN_a1.7_g12", "KN_a1.8_g12")]
    missing = [c for c, r in zip(("KN_a1.5_g12", "KN_a1.6_g12", "KN_a1.7_g12",
                                  "KN_a1.8_g12"), g12) if r is None]
    r17_8 = _find(rows, "KN_a1.7_g8")
    if missing:
        return {"verdict": "incomplete", "tolerance": TOLERANCE,
                "missing": missing,
                "reason": "pre-registered gpr=12 points not all collected"}
    a15, a16, a17, a18 = g12
    errs = [r["err_runout"] for r in g12]
    in_band = [abs(e) <= TOLERANCE for e in errs]
    h1 = {"all_in_band": all(in_band), "verdict": "pass" if all(in_band) else "fail",
          "errors": errs, "in_band": in_band,
          "worst": max(errs, key=abs)}

    jump_lo = abs(a17["err_runout"] - a16["err_runout"])
    jump_hi = abs(a18["err_runout"] - a17["err_runout"])
    h2 = {"jump_below_pp": 100.0 * jump_lo, "jump_above_pp": 100.0 * jump_hi,
          "threshold_pp": MAX_ERROR_JUMP_PP,
          "verdict": "pass" if max(jump_lo, jump_hi) <= MAX_ERROR_JUMP_PP / 100.0
                     else "fail"}

    # H3: shape of the measured curve across the break (low power, 2-point diffs).
    s_short_below = local_slope(a15["aspect_measured"], a15["runout_ratio"],
                                a16["aspect_measured"], a16["runout_ratio"])
    s_short_above = local_slope(a17["aspect_measured"], a17["runout_ratio"],
                                a18["aspect_measured"], a18["runout_ratio"])
    long_rows = [_find(rows, "R_a1_g12"), a15]
    if long_rows[0] is None:
        s_long_below = float("nan")
    else:
        s_long_below = local_slope(long_rows[0]["aspect_measured"],
                                   long_rows[0]["runout_ratio"],
                                   a15["aspect_measured"], a15["runout_ratio"])
    ratio_short = (s_short_below / s_short_above
                   if s_short_above not in (0.0,) and math.isfinite(s_short_above)
                   else float("nan"))
    ratio_long = (s_long_below / s_short_above
                  if s_short_above not in (0.0,) and math.isfinite(s_short_above)
                  else float("nan"))
    law = law_discontinuity()
    if not (math.isfinite(ratio_short) and math.isfinite(ratio_long)):
        h3 = {"verdict": "incomplete", "reason": "slope undefined"}
    elif (ratio_short > MIN_SLOPE_RATIO) and (ratio_long > MIN_SLOPE_RATIO):
        h3 = {"verdict": "pass"}
    elif (ratio_short <= MIN_SLOPE_RATIO) and (ratio_long <= MIN_SLOPE_RATIO):
        h3 = {"verdict": "fail"}
    else:
        h3 = {"verdict": "incomplete",
              "reason": "short and long baselines disagree (low power)"}
    h3.update({
        "slope_below_short": s_short_below, "slope_below_long": s_long_below,
        "slope_above": s_short_above,
        "slope_ratio_short": ratio_short, "slope_ratio_long": ratio_long,
        "slope_ratio_threshold": MIN_SLOPE_RATIO,
        "law_slope_below": law["linear_slope_at_break"],
        "law_slope_above": law["sqrt_slope_at_break"],
        "law_slope_ratio": law["slope_ratio_across_break"],
    })

    if r17_8 is None:
        h4 = {"verdict": "incomplete", "reason": "gpr=8 control missing"}
    else:
        d = abs(a17["err_runout"] - r17_8["err_runout"])
        h4 = {"err_gpr12": a17["err_runout"], "err_gpr8": r17_8["err_runout"],
              "delta_pp": 100.0 * d, "threshold_pp": MAX_ERROR_JUMP_PP,
              "verdict": "pass" if d <= MAX_ERROR_JUMP_PP / 100.0 else "fail"}

    verdicts = [h1["verdict"], h2["verdict"], h3["verdict"], h4["verdict"]]
    overall = "pass" if all(v == "pass" for v in verdicts) else (
        "fail" if "fail" in verdicts else "incomplete")
    return {
        "round": ROUND, "anchor": ANCHOR, "tolerance": TOLERANCE,
        "H1_band": h1, "H2_no_branch_artefact": h2,
        "H3_slope_shape": h3, "H4_resolution": h4,
        "law_discontinuity": law,
        "overall": overall,
        "energy_flagged": [r["case_id"] for r in rows
                           if abs(r["energy_residual"]) > 0.10],
        "stop_reasons": {r["case_id"]: r["stop_reason"] for r in rows},
    }


def collect() -> dict:
    rows, missing = [], []
    for spec in NEW_CASES:
        path = case_path(spec["case_id"])
        if not os.path.exists(path):
            missing.append(spec["case_id"])
            continue
        rows.append(point_row(path, f"this mission ({ROUND})"))
    for aspect, cid in ARCHIVE_CASES:
        path = archive_path(cid)
        if not os.path.exists(path):
            missing.append(cid)
            continue
        rows.append(point_row(path, f"archive m20873 ({path})"))
    rows.sort(key=lambda r: (r["grains_per_radius"], r["aspect_nominal"]))
    payload = {
        "round": ROUND, "anchor": ANCHOR, "tolerance": TOLERANCE,
        "prereg": "docs/PREREG_COLUMN_COLLAPSE_KINK_m21007.md",
        "rows": rows, "missing": missing, "verdict": judge(rows),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "verdict.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    header = ["case_id", "source", "grains_per_radius", "aspect_nominal",
              "aspect_measured", "n_particles", "dt", "n_steps", "stop_reason",
              "runout_ratio", "ref_runout_ratio", "err_runout",
              "err_runout_nominal", "t90", "ref_time", "err_t90",
              "energy_residual", "wall_clock_s"]
    with open(os.path.join(OUT_DIR, "cases_kink.csv"), "w") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(
                f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k])
                for k in header) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--case", metavar="CASE_ID", help="integrate one case")
    ap.add_argument("--shard", metavar="I/N", help="integrate shard I of N")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        for c in NEW_CASES:
            print(json.dumps(c))
    if args.shard:
        idx, tot = (int(x) for x in args.shard.split("/"))
        todo = [c for k, c in enumerate(NEW_CASES) if k % tot == idx]
        for spec in todo:
            res = run_case(spec["case_id"], progress=args.progress)
            print(f"{res['case_id']}: N={res['n_particles']} "
                  f"steps={res['n_steps']} runout={res['runout_ratio_p995']:.4f} "
                  f"err={res['runout_error_measured']:+.2%} "
                  f"stop={res['stop_reason']} wall={res.get('wall_clock_s', 0):.0f}s",
                  flush=True)
    if args.case:
        res = run_case(args.case, progress=args.progress)
        print(f"{res['case_id']}: N={res['n_particles']} "
              f"steps={res['n_steps']} runout={res['runout_ratio_p995']:.4f} "
              f"err={res['runout_error_measured']:+.2%} "
              f"stop={res['stop_reason']} wall={res.get('wall_clock_s', 0):.0f}s",
              flush=True)
    if args.collect:
        payload = collect()
        print(json.dumps(payload["verdict"], indent=1))
    if not (args.list or args.case or args.shard or args.collect):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())