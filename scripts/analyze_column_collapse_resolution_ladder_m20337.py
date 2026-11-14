#!/usr/bin/env python3
"""Resolution-ladder closure of the CASE-COLUMN-COLLAPSE anchor's judged laws (m20337).

Anchor
------
``CASE-COLUMN-COLLAPSE`` (算例池·颗粒柱坍塌), reference Lube et al. (2004) *JFM*
508:175-199, doi:10.1017/S0022112004009036, acceptance tolerance < 15 %.

Why this reducer exists
-----------------------
Rounds 1-6 validated the two Lube laws with a *single* discretisation per aspect
ratio (8 grains per column radius, ``dt_scale = 1``).  Later audits added
one-off resolution controls, but the record has three holes, all named by the
archive itself (``docs/AUDIT_COLUMN_COLLAPSE_SHALLOW_CONVERGENCE_m17917.md``
§5-2/§5-3):

* the shallow branch (a = 0.5) has no third rung: 8 -> 12 grains/radius moved the
  runout error from -2.9 % to -11.9 %, i.e. towards the 15 % band edge, with no
  evidence about where the ladder stops;
* the *combined* control (12 grains/radius AND dt/2 at a = 0.5) and the
  transitional time-step control (dt/2 at a = 1.5) were declared but never run;
* the resolution axis was never controlled at a = 2.0, although a = 2.0 is a
  validated point of the closing arm.

This script adjudicates the wave-5e/wave-5d cases that fill those holes.  All
reference values are **re-evaluated from Lube's published formulas** at run time
(``tensordem.granular_column.axisym``), never copied from a stored target.

Pre-registered hypotheses (constants below; fixed before the first run)
----------------------------------------------------------------------
``R_BAND16``  at a = 0.5, the 16 grains/radius runout error stays inside 15 %
              (the anchor's runout verdict survives the third rung).
``R_DRIFT``   the runout error moves < 15 percentage points from 12 -> 16
              grains/radius at a = 0.5 (no resolution runaway).
``T_TREND``   the a = 0.5 emplacement-time error magnitude decreases
              monotonically over 8 -> 12 -> 16 grains/radius.
``T_BAND16``  at a = 0.5, the 16 grains/radius t95 error is inside 15 %
              (the shallow "late arrival" is a discretisation artefact, as
              m17917's falsified H_RES suggests).
``DT_TRANS``  at a = 1.5, dt -> dt/2 changes t95 by < 5 % relative and the
              runout ratio by < 2 % relative.
``DEEP_RES``  at a = 2.0, the 12 grains/radius runout error is inside 15 % and
              moves < 15 pp from the 8 grains/radius baseline.
``COMBINED``  at a = 0.5, the combined control (12 grains/radius + dt/2) keeps
              runout inside 15 % and its t95 error within 5 pp of the
              12 grains/radius-only rung (the two axes do not interact).

Adjudication is three-state (pass / fail / insufficient).  Missing or unreadable
cases are ``insufficient`` -- never ``pass``.  A hypothesis is ``fail`` if any of
its members fails; ``fail`` dominates ``insufficient`` dominates ``pass``.
Thresholds are frozen below and are not retuned after seeing a number.

CLI
---
``python3 scripts/analyze_column_collapse_resolution_ladder_m20337.py
[--runs runs/case_column_collapse_lube2004/cases] [--out docs/artifacts/...]``
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

from tensordem.granular_column.axisym import (  # noqa: E402
    lube_2004_emplacement_time,
    lube_2004_runout_ratio,
)

TOLERANCE = 0.15          # anchor acceptance band (< 15 %)
BAND_TOL = 0.15           # pre-registered band hypothesis
DRIFT_PP = 0.15           # pre-registered cross-rung drift cap
DT_T_REL = 0.05           # dt -> dt/2 changes t95 by < 5 % relative
DT_RUNOUT_REL = 0.02      # dt -> dt/2 changes runout by < 2 % relative
COMBINED_T_PP = 0.05      # combined vs resolution-only t95 gap
ARRIVAL_KEY = "0.95"      # primary arrival definition, declared in the row

# Case ids (frozen).  ``None`` marks a member that must exist for the hypothesis
# to be adjudicable; if its JSON is absent the verdict becomes ``insufficient``.
SHALLOW_LADDER = ["w5b_confirm_wall0.8_a0.5", "w5d_gpr12_a0.5", "w5e_gpr16_a0.5"]
SHALLOW_DT = ["w5b_confirm_wall0.8_a0.5", "w5d_dt0.5_a0.5"]
SHALLOW_COMBINED = ["w5d_gpr12_a0.5", "w5d_fine_a0.5"]
TRANSITIONAL = ["w5b_confirm_wall0.8_a1.5", "w5d_gpr12_a1.5", "w5d_dt0.5_a1.5"]
DEEP = ["w5b_confirm_wall0.8_a2", "w5e_gpr12_a2"]
ALL_IDS = sorted(set(SHALLOW_LADDER + SHALLOW_DT + SHALLOW_COMBINED + TRANSITIONAL + DEEP))


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reduce_case(path: str) -> dict:
    """One per-case JSON -> the scalars this study judges (reference recomputed)."""
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)
    case = res.get("case", {})
    a_nominal = float(case.get("aspect_ratio", res.get("aspect_measured", float("nan"))))
    r_i = float(res.get("column_radius_measured", case.get("column_radius", float("nan"))))
    g = float(case.get("gravity", 9.81))
    a_eval = a_nominal if math.isfinite(a_nominal) else float(res["aspect_measured"])
    # references evaluated from Lube's formulas -- nothing is copied
    ref_runout = lube_2004_runout_ratio(a_eval)
    ref_time = lube_2004_emplacement_time(a_eval, r_i, g)
    runout = float(res["runout_ratio_p995"])
    t_by_frac = res.get("t_reach_by_fraction", {})
    t95 = float(t_by_frac.get(ARRIVAL_KEY, res.get("t_reach", float("nan"))))
    return {
        "case": res.get("case", {}).get("label", os.path.basename(path)[:-5]),
        "case_id": os.path.basename(path)[:-5],
        "phase": res.get("phase", ""),
        "aspect_ratio": a_nominal,
        "aspect_measured": float(res.get("aspect_measured", float("nan"))),
        "grains_per_radius": float(case.get("grains_per_radius", float("nan"))),
        "dt_scale": float(case.get("dt_scale", 1.0)),
        "dt": float(res.get("dt", float("nan"))),
        "n_steps": int(res.get("n_steps", 0)),
        "n_particles": int(res.get("n_particles", 0)),
        "mu": float(case.get("mu", float("nan"))),
        "mu_rolling": float(case.get("mu_rolling", float("nan"))),
        "mu_wall": float(case.get("mu_wall", float("nan"))),
        "stop_reason": res.get("stop_reason", ""),
        "runout_ratio": runout,
        "ref_runout_ratio": ref_runout,
        "runout_err": runout / ref_runout - 1.0,
        "t95": t95,
        "ref_time": ref_time,
        "t95_err": t95 / ref_time - 1.0,
        "wall_clock_s": float(res.get("wall_clock_s", float("nan"))),
        "sha256": _sha256(path),
        "path": os.path.abspath(path),
    }


def load(runs_dir: str, ids: list[str]) -> tuple[dict, list[str]]:
    rows, missing = {}, []
    for cid in ids:
        path = os.path.join(runs_dir, f"{cid}.json")
        if not os.path.exists(path):
            missing.append(cid)
            continue
        try:
            rows[cid] = reduce_case(path)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            missing.append(cid)
    return rows, missing


# --------------------------------------------------------------------------- #
# adjudication
# --------------------------------------------------------------------------- #
def _verdict(ok: bool | None) -> str:
    return "insufficient" if ok is None else ("pass" if ok else "fail")


def _dominate(votes: list[str]) -> str:
    if "fail" in votes:
        return "fail"
    if "insufficient" in votes:
        return "insufficient"
    return "pass"


def adjudicate(rows: dict, missing: list[str]) -> dict:
    """Machine verdicts for the seven frozen hypotheses.  NaN/absent -> insufficient."""
    def err(cid: str, key: str) -> float | None:
        r = rows.get(cid)
        if r is None:
            return None
        v = r.get(key)
        if v is None or not math.isfinite(float(v)):
            return None
        return float(v)

    out: dict[str, dict] = {}

    def add(name: str, members: list[str], checks: list[tuple[str, float | None, str]]):
        votes, detail = [], []
        for label, value, rule in checks:
            if value is None:
                votes.append("insufficient")
                detail.append({"check": label, "value": None, "rule": rule,
                               "verdict": "insufficient"})
                continue
            ok = eval(rule, {"abs": abs, "value": value, "BAND_TOL": BAND_TOL,
                             "DRIFT_PP": DRIFT_PP, "DT_T_REL": DT_T_REL,
                             "DT_RUNOUT_REL": DT_RUNOUT_REL,
                             "COMBINED_T_PP": COMBINED_T_PP})  # frozen constants only
            v = _verdict(bool(ok))
            votes.append(v)
            detail.append({"check": label, "value": value, "rule": rule, "verdict": v})
        out[name] = {"verdict": _dominate(votes) if votes else "insufficient",
                     "members": members,
                     "missing_members": [m for m in members if m in missing],
                     "checks": detail}

    # --- shallow resolution ladder -----------------------------------------
    e16 = err("w5e_gpr16_a0.5", "runout_err")
    e12 = err("w5d_gpr12_a0.5", "runout_err")
    add("R_BAND16", ["w5e_gpr16_a0.5"],
        [("|runout err| at 16 grains/radius, a=0.5", None if e16 is None else abs(e16),
          "value <= BAND_TOL")])
    add("R_DRIFT", ["w5d_gpr12_a0.5", "w5e_gpr16_a0.5"],
        [("|runout err(16) - runout err(12)|", None if (e16 is None or e12 is None)
          else abs(e16 - e12), "value <= DRIFT_PP")])

    # --- shallow emplacement-time trend ------------------------------------
    t8 = err("w5b_confirm_wall0.8_a0.5", "t95_err")
    t12 = err("w5d_gpr12_a0.5", "t95_err")
    t16 = err("w5e_gpr16_a0.5", "t95_err")
    if None in (t8, t12, t16):
        trend: float | None = None
    else:
        # 1.0 when the magnitudes strictly decrease over the ladder, else 0.0
        trend = 1.0 if (abs(t12) < abs(t8) and abs(t16) < abs(t12)) else 0.0
    add("T_TREND", SHALLOW_LADDER,
        [("|t95 err| strictly decreasing 8->12->16", trend, "value == 1.0")])
    add("T_BAND16", ["w5e_gpr16_a0.5"],
        [("|t95 err| at 16 grains/radius, a=0.5",
          None if t16 is None else abs(t16), "value <= BAND_TOL")])

    # --- transitional time-step control ------------------------------------
    r8_15 = err("w5b_confirm_wall0.8_a1.5", "runout_ratio")
    rdt_15 = err("w5d_dt0.5_a1.5", "runout_ratio")
    t8_15 = err("w5b_confirm_wall0.8_a1.5", "t95")
    tdt_15 = err("w5d_dt0.5_a1.5", "t95")
    add("DT_TRANS", ["w5b_confirm_wall0.8_a1.5", "w5d_dt0.5_a1.5"], [
        ("|dt95 / t95| at a=1.5",
         None if (t8_15 in (None, 0.0) or tdt_15 is None) else abs(tdt_15 / t8_15 - 1.0),
         "value <= DT_T_REL"),
        ("|drunout / runout| at a=1.5",
         None if (r8_15 in (None, 0.0) or rdt_15 is None) else abs(rdt_15 / r8_15 - 1.0),
         "value <= DT_RUNOUT_REL"),
    ])

    # --- deep branch -------------------------------------------------------
    e2_12 = err("w5e_gpr12_a2", "runout_err")
    e2_8 = err("w5b_confirm_wall0.8_a2", "runout_err")
    add("DEEP_RES", DEEP, [
        ("|runout err| at 12 grains/radius, a=2.0",
         None if e2_12 is None else abs(e2_12), "value <= BAND_TOL"),
        ("|runout err(12) - runout err(8)| at a=2.0",
         None if (e2_12 is None or e2_8 is None) else abs(e2_12 - e2_8),
         "value <= DRIFT_PP"),
    ])

    # --- combined control --------------------------------------------------
    ec = err("w5d_fine_a0.5", "runout_err")
    tc = err("w5d_fine_a0.5", "t95_err")
    add("COMBINED", SHALLOW_COMBINED, [
        ("|runout err| combined (12 gpr + dt/2), a=0.5",
         None if ec is None else abs(ec), "value <= BAND_TOL"),
        ("|t95 err(combined) - t95 err(12 gpr)|, a=0.5",
         None if (tc is None or t12 is None) else abs(tc - t12),
         "value <= COMBINED_T_PP"),
    ])
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(rows: dict, out_dir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    ladder = [r for r in (rows.get(c) for c in SHALLOW_LADDER) if r]
    ladder.sort(key=lambda r: r["grains_per_radius"])
    if ladder:
        gpr = [r["grains_per_radius"] for r in ladder]
        e_r = [100 * r["runout_err"] for r in ladder]
        e_t = [100 * r["t95_err"] for r in ladder]
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
        for a, ys, title, ylab in ((ax[0], e_r, "runout law  $r_\\infty/r_i$", "error [%]"),
                                   (ax[1], e_t, "emplacement time  $t_{95}$",
                                    "error [%]")):
            a.axhspan(-15, 15, color="0.85", label="±15 % anchor band")
            a.axhline(0, color="k", lw=0.6)
            a.plot(gpr, ys, "o-", color="C0", label="a = 0.5 (closing arm)")
            for x, y in zip(gpr, ys):
                a.annotate(f"{y:+.1f}%", (x, y), textcoords="offset points",
                           xytext=(0, 7), ha="center", fontsize=8)
            a.set_xlabel("grains per column radius")
            a.set_title(title)
            a.set_ylabel(ylab)
            a.set_xticks(gpr)
            a.legend(fontsize=8)
        fig.suptitle("CASE-COLUMN-COLLAPSE · resolution ladder at a = 0.5 (m20337)")
        fig.tight_layout()
        p = os.path.join(out_dir, "fig_resolution_ladder.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)

    deep = [r for r in (rows.get(c) for c in DEEP) if r]
    trans = [r for r in (rows.get(c) for c in TRANSITIONAL) if r]
    if deep or trans:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.axhspan(-15, 15, color="0.85", label="±15 % anchor band")
        ax.axhline(0, color="k", lw=0.6)
        for group, label, marker in ((trans, "a = 1.5", "s"), (deep, "a = 2.0", "o")):
            if not group:
                continue
            group = sorted(group, key=lambda r: (r["grains_per_radius"], r["dt_scale"]))
            xs = [f"{r['grains_per_radius']:g} gpr" + ("" if r["dt_scale"] == 1.0 else
                                                        f", dt×{r['dt_scale']:g}")
                  for r in group]
            ys = [100 * r["runout_err"] for r in group]
            ax.plot(xs, ys, marker + "-", label=label)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:+.1f}%", (x, y), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=8)
        ax.set_ylabel("runout-law error [%]")
        ax.set_title("transitional / deep branch discretisation controls")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(out_dir, "fig_deep_and_transitional.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)
    return made


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(root, "runs",
                                                   "case_column_collapse_lube2004", "cases"))
    ap.add_argument("--out", default=os.path.join(root, "docs", "artifacts",
                                                  "column_collapse_resolution_ladder_m20337"))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    rows, missing = load(args.runs, ALL_IDS)
    verdicts = adjudicate(rows, missing)

    csv_path = os.path.join(args.out, "cases_resolution_ladder.csv")
    cols = ["case_id", "phase", "aspect_ratio", "aspect_measured", "grains_per_radius",
            "dt_scale", "dt", "n_steps", "n_particles", "mu", "mu_rolling", "mu_wall",
            "stop_reason", "runout_ratio", "ref_runout_ratio", "runout_err", "t95",
            "ref_time", "t95_err", "wall_clock_s", "sha256"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for cid in ALL_IDS:
            if cid in rows:
                w.writerow(rows[cid])

    figs = make_figures(rows, args.out)
    manifest = {
        "study": "CASE-COLUMN-COLLAPSE resolution-ladder closure (m20337)",
        "reference": "Lube et al. 2004 JFM 508:175-199, doi:10.1017/S0022112004009036",
        "tolerance": TOLERANCE,
        "thresholds": {"BAND_TOL": BAND_TOL, "DRIFT_PP": DRIFT_PP,
                       "DT_T_REL": DT_T_REL, "DT_RUNOUT_REL": DT_RUNOUT_REL,
                       "COMBINED_T_PP": COMBINED_T_PP, "ARRIVAL_KEY": ARRIVAL_KEY},
        "cases_present": [c for c in ALL_IDS if c in rows],
        "cases_missing": missing,
        "verdicts": verdicts,
        "figures": [os.path.basename(f) for f in figs],
        "csv": os.path.basename(csv_path),
        "csv_sha256": _sha256(csv_path),
        "files": {},
    }
    for p in [csv_path] + figs:
        manifest["files"][os.path.basename(p)] = _sha256(p)
    for cid, r in rows.items():
        manifest["files"][cid + ".json"] = r["sha256"]
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    with open(os.path.join(args.out, "manifest_sha256.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest["files"], fh, indent=1, sort_keys=True)

    print(json.dumps({"missing": missing,
                      "verdicts": {k: v["verdict"] for k, v in verdicts.items()}},
                     indent=1))
    for cid in ALL_IDS:
        r = rows.get(cid)
        if r is None:
            print(f"{cid:32s} MISSING")
            continue
        print(f"{cid:32s} a={r['aspect_ratio']:4.1f} gpr={r['grains_per_radius']:5.1f} "
              f"dt×{r['dt_scale']:4.2f} n={r['n_particles']:6d} "
              f"runout={r['runout_ratio']:.3f} err={100*r['runout_err']:+6.2f}% "
              f"t95err={100*r['t95_err']:+6.2f}% {r['stop_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())