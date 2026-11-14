"""Reducer + pre-registered verdicts for the two refinement gaps (mission m20873).

Anchor ``CASE-COLUMN-COLLAPSE`` (算例池·颗粒柱坍塌; reference Lube, Huppert,
Sparks & Freundt 2004 JFM **508**:175-199, doi:10.1017/S0022112004009036;
tolerance **15 %**).

Gap being closed
----------------
``docs/CASE_COLUMN_COLLAPSE_REFINEMENT_m18261.md`` §5-5 declared two cases that
were never integrated (``R_a1_g16`` third resolution rung at ``a=1``; ``R_a3_g12``
deep-end refinement at ``a=3``) and therefore recorded the refinement verdict
``incomplete``.  ``docs/AUDIT_COLUMN_COLLAPSE_REFINEMENT_FETCH_m20559.md`` tried
to close them on the cluster and got ``rc=0`` but could not retrieve the numbers.
This mission re-runs **the whole five-case ladder** (a=1: 8/12/16 grains per
column radius; a=3: 8/12) so that the a=1 and a=3 ladders are machine-readable
end to end with zero transcription, and judges the pre-registered criteria in
``docs/prereg/PREREG_CASE_COLUMN_COLLAPSE_DEEP_REFINEMENT_m20873.md``.

Discipline
----------
* Reference values are **evaluated** through
  :func:`tensordem.granular_column.axisym.lube_2004_runout_ratio` /
  ``lube_2004_emplacement_time`` at the **measured** aspect ratio (the paper's
  ``a`` is the released column's ``h_i/r_i``, cf. m20533), never transcribed.
* Nothing in the frozen protocol of m18261 is retuned; the cases are the ones
  already listed in ``benchmarks/case_column_collapse_time_m18261.py``.
* A missing / truncated / non-finite case is ``insufficient``, never ``pass``.

Thresholds are frozen here and asserted against the pre-registration document by
``tests/test_column_collapse_deep_refinement_m20873.py``.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from tensordem.granular_column.axisym import (  # noqa: E402
    lube_2004_emplacement_time, lube_2004_runout_ratio)

ROUND = "m20873"
TOLERANCE = 0.15
CASE_DIR = os.path.join("runs", "case_column_collapse_time_m18261", "cases")
OUT_DIR = os.path.join("docs", "artifacts", "column_collapse_deep_refinement_m20873")
#: Durable, committed copy of the same case JSONs (fetch-proof: the node worktree
#: is not returned by the compute gate, so the numbers are archived in-repo too).
CASE_DIR_FALLBACKS = (
    CASE_DIR,
    os.path.join(OUT_DIR, "cases"),
)
M20337_CSV = os.path.join("docs", "artifacts",
                          "column_collapse_resolution_ladder_m20337",
                          "cases_resolution_ladder.csv")

# --- frozen pre-registered thresholds (mirror the prereg document) -----------
P1_BAND = TOLERANCE           # |err_runout(a_meas)| at a=1, gpr=16
P2_DRIFT_PP = 5.0             # |err16 - err12| at a=1  [percentage points]
P3_BAND = TOLERANCE           # |err_runout(a_meas)| at a=3, gpr=12
P4_DRIFT_PP = 15.0            # |err12 - err8| at a=3   [percentage points]
P5_BAND = TOLERANCE           # |err_t95(a_meas)| at a=3, gpr=12
P6_BAND = TOLERANCE           # all four sqrt-branch points inside the band
PRIMARY_T = "fraction_0.9"    # t95 primary definition frozen by m20337 / m17770

#: The five cases this mission integrates itself (ids as declared by m18261).
LADDER = ("R_a1_g8", "R_a1_g12", "R_a1_g16", "R_a3_g8", "R_a3_g12")
#: Deep-branch points taken from the archived m20337 ladder (a=2.0, 8/12).
M20337_IDS = {"w5b_confirm_wall0.8_a2": 2.0, "w5e_gpr12_a2": 2.0}


class Insufficient(Exception):
    """Raised when a case cannot support a verdict."""


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _err(value, reference) -> float:
    if not _finite(value) or not _finite(reference) or float(reference) == 0.0:
        return float("nan")
    return (float(value) - float(reference)) / float(reference)


def find_case(cid: str) -> str:
    for base in CASE_DIR_FALLBACKS:
        path = os.path.join(base, f"{cid}.json")
        if os.path.exists(path):
            return path
    raise Insufficient(f"case {cid} has no JSON in {CASE_DIR_FALLBACKS}")


def load_case(cid: str) -> dict:
    path = find_case(cid)
    with open(path) as fh:
        return json.load(fh)


def case_row(cid: str) -> dict:
    """Recompute every judged quantity for one freshly integrated case."""
    case = load_case(cid)
    a_meas = case["aspect_measured"]
    r_i = case["column_radius_measured"]
    gravity = case["protocol"]["gravity"]
    runout = case["runout_ratio_p995"]
    t95 = case["arrival"]["primary"][PRIMARY_T]
    ref_runout = lube_2004_runout_ratio(a_meas)
    ref_time = lube_2004_emplacement_time(a_meas, r_i, gravity)
    return {
        "case_id": cid,
        "aspect_nominal": case["aspect_nominal"],
        "aspect_measured": a_meas,
        "grains_per_radius": case["protocol"]["grains_per_radius"],
        "grain_diameter": 2.0 * case["protocol"]["radius"],
        "kn": case["protocol"]["kn"],
        "n_particles": case["n_particles"],
        "dt": case["dt"],
        "n_steps": case["n_steps"],
        "stop_reason": case["stop_reason"],
        "runout_ratio": runout,
        "ref_runout_ratio": ref_runout,
        "err_runout": _err(runout, ref_runout),
        "t95": t95,
        "ref_time": ref_time,
        "err_t95": _err(t95, ref_time),
        "err_runout_nominal": _err(runout, lube_2004_runout_ratio(
            case["aspect_nominal"])),
        "overlap_pairs": case.get("overlap_pairs"),
        "wall_clock_s": case["wall_clock_s"],
        "protocol_sha256": case["protocol_sha256"],
        "source": f"this mission ({ROUND}), freshly integrated",
    }


def m20337_rows() -> list[dict]:
    """Deep-branch points from the archived m20337 ladder (machine-read CSV)."""
    if not os.path.exists(M20337_CSV):
        raise Insufficient(f"archived ladder CSV missing: {M20337_CSV}")
    rows = []
    with open(M20337_CSV) as fh:
        for row in csv.DictReader(fh):
            if row["case_id"] not in M20337_IDS:
                continue
            a_nom = float(row["aspect_ratio"])
            a_meas = float(row["aspect_measured"])
            runout = float(row["runout_ratio"])
            # The m20337 archive evaluated the law at the **nominal** aspect
            # (the m20533 audit's "nominal convention").  Verify that claim on
            # the archive itself before re-evaluating at the measured aspect.
            if abs(lube_2004_runout_ratio(a_nom)
                   - float(row["ref_runout_ratio"])) > 1e-9:
                raise Insufficient(f"archived reference mismatch on {row['case_id']}")
            ref = lube_2004_runout_ratio(a_meas)
            rows.append({
                "case_id": row["case_id"],
                "aspect_nominal": float(row["aspect_ratio"]),
                "aspect_measured": a_meas,
                "grains_per_radius": float(row["grains_per_radius"]),
                "grain_diameter": float("nan"),
                "kn": float("nan"),
                "n_particles": int(float(row["n_particles"])),
                "dt": float(row["dt"]),
                "n_steps": int(float(row["n_steps"])),
                "stop_reason": row["stop_reason"],
                "runout_ratio": runout,
                "ref_runout_ratio": ref,
                "err_runout": _err(runout, ref),
                "t95": float(row["t95"]),
                "ref_time": lube_2004_emplacement_time(a_meas, 0.48, 9.81),
                "err_t95": _err(float(row["t95"]),
                                lube_2004_emplacement_time(a_meas, 0.48, 9.81)),
                "archive_convention": "nominal aspect (re-evaluated here at measured)",
                "overlap_pairs": None,
                "wall_clock_s": float(row["wall_clock_s"]),
                "protocol_sha256": row.get("sha256", ""),
                "source": "archived m20337 ladder CSV (machine-read)",
            })
    if len(rows) != len(M20337_IDS):
        raise Insufficient("archived ladder is missing the a=2 deep-branch points")
    return rows


def _pp(err_a, err_b) -> float:
    return abs(100.0 * (err_a - err_b))


def judge(rows: dict[str, dict], deep: list[dict]) -> dict:
    """Machine verdicts for P1…P7. Missing data -> ``insufficient``."""
    def need(cid):
        if cid not in rows:
            raise Insufficient(f"{cid} missing")
        return rows[cid]

    verdicts = {}
    try:
        a1_8, a1_12, a1_16 = need("R_a1_g8"), need("R_a1_g12"), need("R_a1_g16")
        a3_8, a3_12 = need("R_a3_g8"), need("R_a3_g12")
        v = abs(a1_16["err_runout"])
        verdicts["P1_A1_G16_BAND"] = {
            "value": v, "threshold": P1_BAND,
            "result": "pass" if (math.isfinite(v) and v <= P1_BAND) else "fail"}
        d = _pp(a1_16["err_runout"], a1_12["err_runout"])
        verdicts["P2_A1_DRIFT"] = {
            "value_pp": d, "threshold_pp": P2_DRIFT_PP,
            "result": "pass" if d <= P2_DRIFT_PP else "fail"}
        v = abs(a3_12["err_runout"])
        verdicts["P3_A3_G12_BAND"] = {
            "value": v, "threshold": P3_BAND,
            "result": "pass" if (math.isfinite(v) and v <= P3_BAND) else "fail"}
        d = _pp(a3_12["err_runout"], a3_8["err_runout"])
        verdicts["P4_A3_DRIFT"] = {
            "value_pp": d, "threshold_pp": P4_DRIFT_PP,
            "result": "pass" if d <= P4_DRIFT_PP else "fail"}
        v = abs(a3_12["err_t95"])
        verdicts["P5_TIME_A3_BAND"] = {
            "value": v, "threshold": P5_BAND,
            "result": "pass" if (math.isfinite(v) and v <= P5_BAND) else "fail"}
        family = [a3_8, a3_12] + deep
        worst = max(abs(r["err_runout"]) for r in family)
        verdicts["P6_DEEP_FAMILY"] = {
            "n_points": len(family), "worst": worst, "threshold": P6_BAND,
            "result": "pass" if (math.isfinite(worst)
                                 and worst <= P6_BAND) else "fail"}
        integrity = {}
        for cid in LADDER:
            r = rows[cid]
            ok_stop = r["stop_reason"] == "at_rest"
            integrity[cid] = {"stop_reason": r["stop_reason"],
                              "overlap_pairs": r["overlap_pairs"],
                              "ok_stop": bool(ok_stop)}
        # m18261's protocol does not archive particle overlaps (return_positions
        # = False), so the overlap clause of P7 is NOT RUN.  It is reported as
        # such instead of being silently satisfied (honest-FAIL discipline).
        verdicts["P7_INTEGRITY"] = {
            "cases": integrity,
            "overlap_clause": "not_run (protocol archives no overlap count)",
            "result": "pass" if all(c["ok_stop"] for c in integrity.values())
            else "fail"}
    except Insufficient as exc:
        verdicts["insufficient"] = str(exc)
    return verdicts


def reduce_all() -> dict:
    rows, missing = {}, []
    for cid in LADDER:
        try:
            rows[cid] = case_row(cid)
        except Insufficient as exc:
            missing.append(str(exc))
    try:
        deep = m20337_rows()
    except Insufficient as exc:
        deep, missing = [], missing + [str(exc)]
    verdicts = judge(rows, deep) if not missing else {"insufficient": missing}
    ordered = [rows[c] for c in LADDER if c in rows] + deep
    return {"round": ROUND, "anchor": "CASE-COLUMN-COLLAPSE",
            "tolerance": TOLERANCE, "rows": ordered, "missing": missing,
            "verdicts": verdicts}


def write_outputs(payload: dict) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    header = ["case_id", "source", "aspect_nominal", "aspect_measured",
              "grains_per_radius", "n_particles", "dt", "n_steps",
              "stop_reason", "runout_ratio", "ref_runout_ratio", "err_runout",
              "err_runout_nominal", "t95", "ref_time", "err_t95",
              "overlap_pairs", "wall_clock_s",
              "protocol_sha256"]
    with open(os.path.join(OUT_DIR, "cases_deep_ladder.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in payload["rows"]:
            w.writerow(["" if r.get(k) is None else
                        (f"{r.get(k):.6g}" if isinstance(r.get(k), float) else r.get(k))
                        for k in header])
    with open(os.path.join(OUT_DIR, "verdict.json"), "w") as fh:
        json.dump(payload, fh, indent=1)


def main(argv: list[str]) -> int:
    payload = reduce_all()
    write_outputs(payload)
    for r in payload["rows"]:
        print(f"{r['case_id']:<24} a_meas={r['aspect_measured']:.4f} "
              f"gpr={r['grains_per_radius']:>4g} N={r['n_particles']:>6} "
              f"runout={r['runout_ratio']:.4f} err={100 * r['err_runout']:+.2f}% "
              f"t95err={100 * r['err_t95']:+.2f}% stop={r['stop_reason']}")
    print()
    for key, val in payload["verdicts"].items():
        if key == "P7_INTEGRITY":
            print(f"{key:<18} {val['result']:<11} "
                  f"overlap_clause={val['overlap_clause']}")
        elif key == "insufficient":
            print(f"{key:<18} {val}")
        else:
            shown = val.get("value", val.get("value_pp", val.get("worst")))
            thr = val.get("threshold", val.get("threshold_pp"))
            print(f"{key:<18} {val['result']:<11} value={shown:.4f} "
                  f"threshold={thr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))