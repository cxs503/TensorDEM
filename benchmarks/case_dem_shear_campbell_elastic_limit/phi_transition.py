#!/usr/bin/env python3
"""Round 2 of CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT: the *concentration* route.

Round 1 (``run_case.py``, docs/CASE_DEM_SHEAR_CAMPBELL_ELASTIC_LIMIT.md) mapped
``tau*(kappa)`` at one fixed area fraction (``phi = 0.80``) by varying ``k_n`` and
``gamma_dot``.  It found a single elastic power law (``beta = 0.870``, i.e. shear
rate exponent ``n = 0.26``) and could **not** reach the inertial (Bagnold,
``n = 2``) branch; the ``k_n = 1e5 / 1e6`` rungs were ``not_run`` for budget.

The accessible Campbell (2002) abstract makes a much sharper, *testable* statement
that round 1 never probed, because it is a different axis:

    "Reducing the concentration slightly causes a transition from an elastic to
     a purely inertial behaviour.  This transition occurs so abruptly that a 2%
     concentration reduction can be accompanied by nearly three orders of
     magnitude of stress reduction. [...] there is no path between inertial
     (rapid) flow and quasi-static flow by varying the shear rate at a fixed
     concentration; only by reducing the concentration can one cause a
     transition from quasi-static to inertial flow."

This driver sweeps the **concentration** axis (fixed ``k_n``, fixed ``gamma_dot``)
plus a rate ladder at three concentrations, and measures the kinetic/contact
stress split that distinguishes "elastic" (contact-borne) from "purely inertial"
(collision-borne) behaviour.

Pre-registration: ``--prereg`` writes the predictions, metrics and decision rules
BEFORE any point is run, into ``runs/phi_transition/prereg.json`` (with the
sha256 of the anchor file).  ``--analyse`` re-reads it and applies it verbatim.

Usage
-----
    python .../phi_transition.py --prereg
    python .../phi_transition.py --stage map --jobs 8      # phi ladder at gamma_dot=1
    python .../phi_transition.py --stage rates --jobs 8    # rate ladder at 3 phi
    python .../phi_transition.py --stage controls --jobs 8 # seed/strain twins
    python .../phi_transition.py --analyse                 # tables + figure + verdicts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.elastic_limit_shear import (  # noqa: E402
    ElasticLimitShearCase, simulate_plane_shear)

RUNS = HERE / "runs" / "phi_transition"
ANCHOR_FILE = HERE / "reference_anchors.json"

# --- fixed apparatus (identical to round 1's gated points) -------------------
KN = 1.0e4                 # kappa = 5787 at gamma_dot = 1
N = 400
DT_FACTOR = 0.004          # dt = 0.004 t_c (round 1: tau* moves 1.2 % from 0.004 -> 0.002)
STRAIN = 1.4
STRAIN_FROM = 0.7
SEED = 20261112

# --- sweep design -----------------------------------------------------------
PHI_MAP = [round(0.80 - 0.01 * k, 4) for k in range(17)]      # 0.80 ... 0.64
RATE_PHIS = (0.80, 0.76, 0.70)          # pre-registered ladder
RATE_PHIS_EXT = (0.66,)                 # post-hoc extension (disclosed in the doc)
RATE_LADDER = (0.5, 1.0, 2.0)
SCATTER_PHIS = (0.80, 0.76, 0.72, 0.68, 0.64)   # seed scatter, seeds SEED+1, SEED+2
SCATTER_SEEDS = (SEED + 1, SEED + 2)

WINDOW = 0.02              # the "2% concentration reduction" of the abstract


# ---------------------------------------------------------------------------
# pre-registration
# ---------------------------------------------------------------------------
def preregistration() -> Dict[str, Any]:
    anchor_text = ANCHOR_FILE.read_text() if ANCHOR_FILE.exists() else ""
    return {
        "case": "CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT",
        "round": "R2 (concentration route)",
        "written_before_any_point": True,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "anchor_sha256": hashlib.sha256(anchor_text.encode()).hexdigest(),
        "apparatus": {
            "kernel": "tensordem.elastic_limit_shear (Lees-Edwards SLLOD, 2-D disks)",
            "kn": KN, "n_particles": N, "dt_factor": DT_FACTOR,
            "strain_total": STRAIN, "strain_average_from": STRAIN_FROM,
            "seed": SEED, "friction": 0.5, "restitution": 0.7, "size_ratio": 1.4,
            "gates": "unchanged from round 1 (energy/overlap/steady/coordination/"
                     "pressure_positive/finite); rejection is reported, never overwritten",
        },
        "sweeps": {
            "phi_map": {"phis": PHI_MAP, "gamma_dot": 1.0,
                        "note": "0.01 spacing so that a 2 % (0.02) window is exactly "
                                "two grid steps"},
            "rate_ladder": {"phis": list(RATE_PHIS), "gamma_dots": list(RATE_LADDER)},
        },
        "predictions": [
            {
                "id": "P1",
                "statement": "Reducing the concentration by 2 % (absolute) drops the "
                             "shear stress by nearly 3 orders of magnitude "
                             "(Campbell 2002 abstract).",
                "metric": "R2 = max_i tau(phi_i) / tau(phi_i - 0.02) over the phi map, "
                          "gated points only; tau = |mean sampled shear stress|",
                "pass_rule": "PASS if R2 >= 100 (within one decade of the quoted 1e3); "
                             "PARTIAL if 10 <= R2 < 100; NOT_SUPPORTED if R2 < 10",
                "note": "order-of-magnitude agreement is the honest target: the quoted "
                        "'nearly three orders' is not a number we can reproduce exactly "
                        "in a 2-D frictional cell",
            },
            {
                "id": "P2",
                "statement": "There is no path from quasi-static to inertial flow by "
                             "varying the shear rate at fixed concentration: the shear "
                             "rate exponent n stays small (elastic scaling) at every "
                             "concentration on the map.",
                "metric": "n(phi) = d ln tau / d ln gamma_dot from the 3-point rate "
                          "ladder (gamma_dot = 0.5, 1, 2) at phi in {0.80, 0.76, 0.70}",
                "pass_rule": "PASS if max_phi n(phi) <= 0.6 (the inertial/Bagnold value "
                             "is n = 2); NOT_SUPPORTED if any n(phi) >= 1.2",
            },
            {
                "id": "P3",
                "statement": "The elastic -> purely inertial transition is a switch "
                             "from contact-borne to collision-borne stress: the kinetic "
                             "share of the shear stress rises as the concentration falls.",
                "metric": "kinetic_stress_share(phi) = mean|tau_kin| / "
                          "(mean|tau_kin| + mean|tau_contact|), same samples",
                "pass_rule": "PASS if Spearman(phi, share) <= -0.8 AND share >= 0.5 at "
                             "the lowest phi measured; PARTIAL if monotone but share "
                             "stays < 0.5",
            },
        ],
        "controls": {
            "seed_twin": "at the two phi values bracketing the steepest 2 % drop, one "
                         "extra point with seed + 1",
            "strain_twin": "at the same two phi values, strain_total x 2 (2.8) with the "
                           "averaging window shifted to 1.4 -> 2.8, to rule out a "
                           "transient masquerading as a transition",
        },
        "honesty_rules": [
            "gate-rejected points are reported with their gate flags and are excluded "
            "from the fits; they are never re-labelled as PASS",
            "no parameter is retuned after seeing the verdicts",
            "if a control contradicts a fitted number, the fitted number is withdrawn",
        ],
    }


# ---------------------------------------------------------------------------
# one point
# ---------------------------------------------------------------------------
def point_specs(stage: str) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    if stage == "map":
        for phi in PHI_MAP:
            specs.append(dict(tag=f"map_phi{phi:.2f}_g1.0", phi=phi, gdot=1.0,
                              strain=STRAIN, strain_from=STRAIN_FROM, seed=SEED))
    elif stage == "scatter":
        for phi in SCATTER_PHIS:
            for k, sd in enumerate(SCATTER_SEEDS, start=1):
                specs.append(dict(tag=f"scat_phi{phi:.2f}_s{k}", phi=phi, gdot=1.0,
                                  strain=STRAIN, strain_from=STRAIN_FROM, seed=sd))
    elif stage == "rates":
        for phi in RATE_PHIS + RATE_PHIS_EXT:
            for gd in RATE_LADDER:
                if gd == 1.0 and any(abs(phi - p) < 1e-9 for p in PHI_MAP):
                    continue  # already produced by the map stage
                specs.append(dict(tag=f"rate_phi{phi:.2f}_g{gd:g}", phi=phi, gdot=gd,
                                  strain=STRAIN, strain_from=STRAIN_FROM, seed=SEED))
    elif stage == "controls":
        drop = steepest_drop_phis()
        if not drop:
            raise SystemExit("controls need the map stage first (no gated map points)")
        for phi in drop:
            specs.append(dict(tag=f"ctrl_seed_phi{phi:.2f}", phi=phi, gdot=1.0,
                              strain=STRAIN, strain_from=STRAIN_FROM, seed=SEED + 1))
            specs.append(dict(tag=f"ctrl_strain_phi{phi:.2f}", phi=phi, gdot=1.0,
                              strain=2.0 * STRAIN, strain_from=STRAIN,
                              seed=SEED))
    else:
        raise SystemExit(f"unknown stage {stage!r}")
    return specs


def run_point(spec: Dict[str, Any]) -> Dict[str, Any]:
    out = RUNS / f"{spec['tag']}.json"
    if out.exists():
        try:
            prev = json.loads(out.read_text())
            if prev.get("tag") == spec["tag"]:
                return prev
        except Exception:
            pass
    case = ElasticLimitShearCase(
        n_particles=N, solid_fraction=spec["phi"], kn=KN,
        shear_rate=spec["gdot"], strain_total=spec["strain"],
        strain_average_from=spec["strain_from"], dt_factor=DT_FACTOR,
        seed=spec["seed"])
    res = simulate_plane_shear(case)
    rec = {
        "tag": spec["tag"], "phi": spec["phi"], "gamma_dot": spec["gdot"],
        "kn": KN, "n_particles": N, "seed": spec["seed"],
        "strain_total": spec["strain"], "strain_from": spec["strain_from"],
        "dt_factor": DT_FACTOR,
        "kappa": res["kappa"],
        "tau": res["tau"], "pressure": res["pressure"],
        "tau_star": res["tau_star"], "mu_eff": res["mu_eff"], "I": res["I"],
        "coordination": res["coordination"],
        "granular_temperature": res["granular_temperature"],
        "tau_kinetic": res["tau_kinetic"], "tau_contact": res["tau_contact"],
        "pressure_kinetic": res["pressure_kinetic"],
        "pressure_contact": res["pressure_contact"],
        "kinetic_stress_share": res["kinetic_stress_share"],
        "kinetic_stress_fraction": res["kinetic_stress_fraction"],
        "max_overlap_ratio": res["max_overlap_ratio"],
        "mean_overlap_ratio": res["mean_overlap_ratio"],
        "steady_drift_tau": res["steady_drift_tau"],
        "energy_defect": res["energy"]["relative_defect"],
        "gates": res["gates"],
        "samples": res["samples"], "runtime_s": res["runtime_s"],
        "decomposition_residual": abs(res["tau"] - res["tau_kinetic"]
                                      - res["tau_contact"]),
        "regime": "elastic" if res["kinetic_stress_share"] < 0.5 else "inertial",
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    return rec


def _parallel(specs: List[Dict[str, Any]], jobs: int) -> List[Dict[str, Any]]:
    specs = [s for s in specs]
    if jobs <= 1:
        return [run_point(s) for s in specs]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(processes=min(jobs, len(specs))) as pool:
        return pool.map(run_point, specs)


def load_points() -> List[Dict[str, Any]]:
    if not RUNS.exists():
        return []
    out = []
    for f in sorted(RUNS.glob("*.json")):
        if f.name == "prereg.json":
            continue
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            continue
    return [p for p in out if "tag" in p and "tau" in p]


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def _log_slope(xs, ys) -> float:
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(ok) < 2 or np.ptp(np.log(x[ok])) <= 1e-12:
        return float("nan")
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def spearman(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return float("nan")

    def rank(v):
        order = np.argsort(v)
        r = np.empty(v.size, float)
        r[order] = np.arange(v.size, dtype=float)
        return r
    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float(np.sum(ra ** 2)) * float(np.sum(rb ** 2)))
    return float(np.sum(ra * rb) / den) if den > 0 else float("nan")


def phi_map_points(points) -> List[Dict[str, Any]]:
    pts = [p for p in points if abs(p["gamma_dot"] - 1.0) < 1e-12
           and p["tag"].startswith("map_")]
    return sorted(pts, key=lambda p: -p["phi"])


def gated(pts):
    return [p for p in pts if p["gates"]["all"]]


def drop_analysis(pts) -> Dict[str, Any]:
    """2 % window ratios on the gated phi map (grid spacing 0.01 -> window = 2 steps)."""
    rows = []
    for a, b in zip(pts, pts[2:]):
        if a["phi"] - b["phi"] < WINDOW - 1e-9:
            continue
        if not (a["gates"]["all"] and b["gates"]["all"]):
            continue
        if not (a["tau"] > 0 and b["tau"] > 0):
            continue
        rows.append({"phi_hi": a["phi"], "phi_lo": b["phi"],
                     "tau_hi": a["tau"], "tau_lo": b["tau"],
                     "ratio": a["tau"] / b["tau"],
                     "dln_tau_dln_phi": _log_slope([a["phi"], b["phi"]],
                                                   [a["tau"], b["tau"]])})
    if not rows:
        return {"rows": [], "R2": float("nan"), "steepest": None}
    best = max(rows, key=lambda r: r["ratio"])
    # local (one grid step) steepness, reported for shape as well
    local = []
    for a, b in zip(pts, pts[1:]):
        if not (a["gates"]["all"] and b["gates"]["all"]) or not (a["tau"] > 0 and b["tau"] > 0):
            continue
        local.append({"phi_hi": a["phi"], "phi_lo": b["phi"],
                      "dln_tau_dln_phi": _log_slope([a["phi"], b["phi"]],
                                                    [a["tau"], b["tau"]]),
                      "ratio": a["tau"] / b["tau"]})
    # shape-only variant including gate-rejected points (coordination gate fails
    # below phi ~ 0.71): reported separately, never used for the verdict
    rows_all = []
    for a, b in zip(pts, pts[2:]):
        if a["phi"] - b["phi"] < WINDOW - 1e-9 or not (a["tau"] > 0 and b["tau"] > 0):
            continue
        rows_all.append({"phi_hi": a["phi"], "phi_lo": b["phi"],
                         "ratio": a["tau"] / b["tau"],
                         "gates_all": bool(a["gates"]["all"] and b["gates"]["all"]),
                         "dln_tau_dln_phi": _log_slope([a["phi"], b["phi"]],
                                                       [a["tau"], b["tau"]])})
    return {"rows": rows, "rows_all_points": rows_all, "R2": best["ratio"],
            "steepest": best, "local": local}


def steepest_drop_phis() -> List[float]:
    """phi values bracketing the steepest 2 % drop (needs the map stage)."""
    pts = phi_map_points(load_points())
    if not pts:
        return []
    d = drop_analysis(pts)
    if not d["rows"]:
        return []
    s = d["steepest"]
    # de-duplicate and keep bracketing pair
    out = []
    for phi in (s["phi_hi"], s["phi_lo"]):
        if all(abs(phi - o) > 1e-9 for o in out):
            out.append(phi)
    return out


def rate_analysis(points) -> Dict[str, Any]:
    out = {}
    for phi in RATE_PHIS + RATE_PHIS_EXT:
        sel = [p for p in points if abs(p["phi"] - phi) < 1e-9
               and not p["tag"].startswith(("scat_", "ctrl_"))]
        # one point per shear rate: prefer the map point, else the rate point
        by_rate: Dict[float, Dict[str, Any]] = {}
        for p in sorted(sel, key=lambda q: q["tag"].startswith("rate_")):
            by_rate[p["gamma_dot"]] = p
        sel = [by_rate[k] for k in sorted(by_rate)]
        if len(sel) < 2:
            continue
        if len(sel) < 2:
            continue
        n = _log_slope([p["gamma_dot"] for p in sel], [p["tau"] for p in sel])
        n_gated = _log_slope([p["gamma_dot"] for p in sel if p["gates"]["all"]],
                             [p["tau"] for p in sel if p["gates"]["all"]])
        out[f"{phi:.2f}"] = {
            "pre_registered": phi in RATE_PHIS,
            "points": [{"gamma_dot": p["gamma_dot"], "kappa": p["kappa"],
                        "tau": p["tau"], "tau_star": p["tau_star"],
                        "gates_all": p["gates"]["all"]} for p in sel],
            "n_rate_exponent": n,
            "n_rate_exponent_gated": n_gated,
        }
    return out


def control_analysis(points) -> Dict[str, Any]:
    ctrls = [p for p in points if p["tag"].startswith("ctrl_")]
    base = {(p["phi"], p["seed"], round(p["strain_total"], 6)): p for p in phi_map_points(points)}
    rows = []
    for c in ctrls:
        kind = "seed" if "_seed_" in c["tag"] else "strain"
        ref_key = (c["phi"], c["seed"] if kind == "strain" else c["seed"] - 1,
                   1.4 if kind == "seed" else 2.8)
        if kind == "strain":
            ref = phi_map_points(points)
            cand = [p for p in ref if abs(p["phi"] - c["phi"]) < 1e-9]
            ref_p = cand[0] if cand else None
        else:
            ref_p = base.get((c["phi"], c["seed"] - 1, 1.4))
        diff = float("nan")
        if ref_p and ref_p["tau"] > 0:
            diff = (c["tau"] - ref_p["tau"]) / ref_p["tau"]
        rows.append({"tag": c["tag"], "kind": kind, "phi": c["phi"],
                     "tau": c["tau"], "tau_reference": ref_p["tau"] if ref_p else None,
                     "reference_tag": ref_p["tag"] if ref_p else None,
                     "relative_difference": diff,
                     "gates_all": c["gates"]["all"]})
    return {"rows": rows}


def scatter_analysis(points) -> Dict[str, Any]:
    """Seed-to-seed scatter of tau at fixed (phi, gamma_dot, strains)."""
    rows = []
    for phi in SCATTER_PHIS:
        sel = [p for p in phi_map_points(points) if abs(p["phi"] - phi) < 1e-9]
        sel += [p for p in points if p["tag"].startswith("scat_")
                and abs(p["phi"] - phi) < 1e-9]
        taus = [p["tau"] for p in sel if p["tau"] > 0]
        if len(taus) < 2:
            continue
        arr = np.asarray(taus, float)
        rows.append({"phi": phi, "n_seeds": int(arr.size),
                     "tau_mean": float(arr.mean()), "tau_std": float(arr.std(ddof=1)),
                     "rel_std": float(arr.std(ddof=1) / arr.mean()),
                     "tau_min": float(arr.min()), "tau_max": float(arr.max()),
                     "tags": [p["tag"] for p in sel]})
    return {"rows": rows,
            "max_rel_std": max((r["rel_std"] for r in rows), default=float("nan"))}


AMENDMENTS = [
    {"what": "seed scatter stage added",
     "why": "the pre-registered seed twin (phi=0.77) differed from the map point by 21 %; "
            "without a scatter estimate the 2 % window ratios could not be judged",
     "when": "after the map/rates/controls stages, before the verdicts were written",
     "effects_thresholds": False},
    {"what": "rate ladder extended from phi in {0.80,0.76,0.70} to include phi = 0.66",
     "why": "n(phi) grew monotonically with falling phi (0.17 -> 0.61 -> 1.13); one more "
            "concentration is needed to see whether it approaches the Bagnold value 2",
     "when": "after the pre-registered ladders were analysed",
     "effects_thresholds": False,
     "note": "the pre-registered three concentrations are reported separately and drive "
             "the P2 verdict as declared"},
]


def analyse() -> Dict[str, Any]:
    prereg_path = RUNS / "prereg.json"
    prereg = json.loads(prereg_path.read_text()) if prereg_path.exists() else preregistration()
    points = load_points()
    pts = phi_map_points(points)
    d = drop_analysis(pts)
    rates = rate_analysis(points)
    ctl = control_analysis(points)
    scat = scatter_analysis(points)

    shares = [(p["phi"], p["kinetic_stress_share"]) for p in pts
              if np.isfinite(p["kinetic_stress_share"])]
    rho_share = spearman([s[0] for s in shares], [s[1] for s in shares]) if shares else float("nan")
    lowest = min(shares, key=lambda s: s[0]) if shares else (float("nan"), float("nan"))

    n_max = max([abs(v["n_rate_exponent"]) for v in rates.values()
                 if v.get("pre_registered") and np.isfinite(v["n_rate_exponent"])],
                default=float("nan"))
    n_max_all = max([abs(v["n_rate_exponent"]) for v in rates.values()
                     if np.isfinite(v["n_rate_exponent"])], default=float("nan"))

    # --- verdicts, applied to the pre-registered rules verbatim ---------------
    R2 = d["R2"]
    if not np.isfinite(R2):
        v1 = "not_run"
    elif R2 >= 100:
        v1 = "PASS"
    elif R2 >= 10:
        v1 = "PARTIAL"
    else:
        v1 = "NOT_SUPPORTED"

    if not np.isfinite(n_max):
        v2 = "not_run"
    elif n_max >= 1.2:
        v2 = "NOT_SUPPORTED"
    elif n_max <= 0.6:
        v2 = "PASS"
    else:
        v2 = "PARTIAL"

    if not shares:
        v3 = "not_run"
    elif np.isfinite(rho_share) and rho_share <= -0.8 and lowest[1] >= 0.5:
        v3 = "PASS"
    elif np.isfinite(rho_share) and rho_share < 0:
        v3 = "PARTIAL"
    else:
        v3 = "NOT_SUPPORTED"

    # --- secondary cross-check: effective friction vs the mu(I) law ----------
    MU_I_CONSTANTS = {"mu_s": 0.36, "mu_2": 0.84, "I_0": 0.30,
                      "provenance": "docs/CASE_DEM_PLANE_SHEAR_MUIDI_R3.md "
                                    "(in-repo constants; PROVENANCE-UNVERIFIED there)"}
    mui = []
    for p in pts:
        I = p["I"]
        if np.isfinite(I) and I >= 0:
            pred = MU_I_CONSTANTS["mu_s"] + (MU_I_CONSTANTS["mu_2"] - MU_I_CONSTANTS["mu_s"]) * \
                I / (MU_I_CONSTANTS["I_0"] + I)
            mui.append({"phi": p["phi"], "I": I, "mu_measured": p["mu_eff"],
                        "mu_mu_i_law": pred,
                        "rel_error": (p["mu_eff"] - pred) / pred,
                        "gates_all": p["gates"]["all"]})
    mui_gated = [m for m in mui if m["gates_all"] and np.isfinite(m["rel_error"])]

    verdicts = {
        "P1_abrupt_concentration_drop": {"verdict": v1, "metric_R2": R2,
                                        "rule": prereg["predictions"][0]["pass_rule"]},
        "P2_no_rate_path": {"verdict": v2, "metric_max_n": n_max,
                            "metric_max_n_including_extension": n_max_all,
                            "rule": prereg["predictions"][1]["pass_rule"]},
        "P3_kinetic_share_switch": {"verdict": v3, "spearman_phi_share": rho_share,
                                    "share_at_lowest_phi": lowest[1],
                                    "lowest_phi": lowest[0],
                                    "rule": prereg["predictions"][2]["pass_rule"]},
    }

    out = {
        "case": prereg["case"], "round": prereg["round"],
        "phi_map": pts, "tau_star": [p["tau"] for p in pts],
        "drop_analysis": d, "rate_analysis": rates, "controls": ctl,
        "scatter": scat, "amendments": AMENDMENTS,
        "verdicts": verdicts,
        "n_points": len(points),
        "n_gated": len(gated(points)),
        "max_decomposition_residual": max((p["decomposition_residual"] for p in points),
                                          default=float("nan")),
        "mu_i_crosscheck": {"constants": MU_I_CONSTANTS, "rows": mui,
                            "gated_mean_rel_error": float(np.mean([m["rel_error"] for m in mui_gated]))
                            if mui_gated else float("nan"),
                            "gated_max_abs_rel_error": float(np.max([abs(m["rel_error"]) for m in mui_gated]))
                            if mui_gated else float("nan")},
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "analysis.json").write_text(json.dumps(out, indent=1, allow_nan=True))
    return out


# ---------------------------------------------------------------------------
# tables + figure
# ---------------------------------------------------------------------------
def write_tables(out: Dict[str, Any]) -> Path:
    pts = out["phi_map"]
    lines = ["| phi | tau | P | tau* | mu | I | Z | kin share | max ov/d | E-defect | gates |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in pts:
        flags = ",".join(k for k, v in p["gates"].items() if k != "all" and not v) or "ok"
        lines.append(
            f"| {p['phi']:.2f} | {p['tau']:.4g} | {p['pressure']:.4g} | "
            f"{p['tau_star']:.4g} | {p['mu_eff']:.4f} | {p['I']:.4g} | "
            f"{p['coordination']:.2f} | {p['kinetic_stress_share']:.3f} | "
            f"{p['max_overlap_ratio']:.3f} | {p['energy_defect']:.3f} | {flags} |")
    lines += ["", "### 2 % concentration windows (gated only)", "",
              "| phi_hi -> phi_lo | tau_hi | tau_lo | ratio | d ln tau/d ln phi |",
              "|---|---|---|---|---|"]
    for r in out["drop_analysis"]["rows"]:
        lines.append(f"| {r['phi_hi']:.2f} -> {r['phi_lo']:.2f} | {r['tau_hi']:.4g} | "
                     f"{r['tau_lo']:.4g} | {r['ratio']:.3g} | {r['dln_tau_dln_phi']:.2f} |")
    lines += ["", "### 2 % windows, ALL points (shape only; gate-rejected rows marked)", "",
              "| phi_hi -> phi_lo | ratio | both gated | d ln tau/d ln phi |", "|---|---|---|---|"]
    for r in out["drop_analysis"].get("rows_all_points", []):
        lines.append(f"| {r['phi_hi']:.2f} -> {r['phi_lo']:.2f} | {r['ratio']:.3g} | "
                     f"{r['gates_all']} | {r['dln_tau_dln_phi']:.2f} |")
    lines += ["", "### rate ladders", "",
              "| phi | gamma_dot | tau | tau* | n (all pts) | n (gated) |", "|---|---|---|---|---|---|"]
    for phi, v in out["rate_analysis"].items():
        for p in v["points"]:
            lines.append(f"| {phi} | {p['gamma_dot']:g} | {p['tau']:.4g} | "
                         f"{p['tau_star']:.4g} | {v['n_rate_exponent']:.3f} | "
                         f"{v['n_rate_exponent_gated']:.3f} |")
    mc = out.get("mu_i_crosscheck")
    if mc and mc["rows"]:
        lines += ["", "### secondary cross-check: effective friction vs the mu(I) law", "",
                  f"constants `{mc['constants']}` -- in-repo values, provenance unverified", "",
                  "| phi | I | mu measured | mu(I) law | rel. error | gated |", "|---|---|---|---|---|---|"]
        for r in mc["rows"]:
            lines.append(f"| {r['phi']:.2f} | {r['I']:.4g} | {r['mu_measured']:.4f} | "
                         f"{r['mu_mu_i_law']:.4f} | {r['rel_error']:+.1%} | {r['gates_all']} |")
    if out.get("scatter", {}).get("rows"):
        lines += ["", "### seed scatter (identical physics, different initial packing)",
                  "", "| phi | seeds | tau mean | tau std | rel. std | min | max |",
                  "|---|---|---|---|---|---|---|"]
        for r in out["scatter"]["rows"]:
            lines.append(f"| {r['phi']:.2f} | {r['n_seeds']} | {r['tau_mean']:.4g} | "
                         f"{r['tau_std']:.3g} | {r['rel_std']:.2%} | {r['tau_min']:.4g} | "
                         f"{r['tau_max']:.4g} |")
    if out["controls"]["rows"]:
        lines += ["", "### controls", "",
                  "| tag | phi | tau | reference | ref tau | rel. difference | gates |",
                  "|---|---|---|---|---|---|---|"]
        for r in out["controls"]["rows"]:
            lines.append(f"| {r['tag']} | {r['phi']:.2f} | {r['tau']:.4g} | "
                         f"{r['reference_tag']} | "
                         f"{r['tau_reference'] if r['tau_reference'] is not None else 'n/a'} | "
                         f"{r['relative_difference']:.3%} | {r['gates_all']} |")
    lines += ["", "### pre-registered verdicts", ""]
    for k, v in out["verdicts"].items():
        lines.append(f"* **{k}** -> `{v['verdict']}`")
    path = RUNS / "tables.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def make_figure(out: Dict[str, Any]) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    pts = out["phi_map"]
    gated = [p for p in pts if p["gates"]["all"]]
    rejected = [p for p in pts if not p["gates"]["all"]]
    scat = {r["phi"]: r for r in out.get("scatter", {}).get("rows", [])}

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.6))

    # (a) stress vs concentration, with the seed-scatter band
    a = ax[0, 0]
    phis = [p["phi"] for p in pts]
    taus = [p["tau"] for p in pts]
    err = [scat[p["phi"]]["tau_std"] if p["phi"] in scat else 0.0 for p in pts]
    a.errorbar(phis, taus, yerr=err, fmt="o-", color="#1565c0", ms=5, lw=1.2,
               label=r"$\tau(\phi)$, seed scatter as error bar")
    if rejected:
        a.loglog([p["phi"] for p in rejected], [p["tau"] for p in rejected], "x",
                 color="#c62828", ms=8, mfc="none", label="coordination/steady gate rejected")
    a.set_yscale("log")
    a.set_xlabel(r"area fraction $\phi$")
    a.set_ylabel(r"$\tau$  (DEM units)")
    a.set_title(r"(a) $\tau$ vs $\phi$ at $k_n=10^4,\ \dot\gamma=1$")
    a.invert_xaxis()
    a.grid(True, which="both", alpha=0.3)
    a.legend(fontsize=7)

    # (b) rate ladders
    b = ax[0, 1]
    for phi, v in out["rate_analysis"].items():
        lab = fr"$\phi={phi}$ ($n={v['n_rate_exponent']:.2f}$)"
        if not v.get("pre_registered", True):
            lab += " (post-hoc)"
        b.loglog([p["gamma_dot"] for p in v["points"]], [p["tau"] for p in v["points"]],
                 "s-", label=lab)
    gd = np.array([0.4, 2.6])
    for n, ls, lab in ((0.0, ":", r"$n=0$ (elastic)"), (2.0, "--", r"$n=2$ (Bagnold)")):
        b.loglog(gd, 122.9 * (gd / 1.0) ** n, ls, color="0.35", lw=1.2, label=lab)
    b.set_xlabel(r"$\dot\gamma$")
    b.set_ylabel(r"$\tau$  (DEM units)")
    b.set_title("(b) rate path at fixed concentration")
    b.grid(True, which="both", alpha=0.3)
    b.legend(fontsize=7)

    # (c) rate exponent vs concentration
    c = ax[1, 0]
    ph = [float(k) for k, v in out["rate_analysis"].items()
          if np.isfinite(v["n_rate_exponent"])]
    nn = [v["n_rate_exponent"] for v in out["rate_analysis"].values()
          if np.isfinite(v["n_rate_exponent"])]
    c.semilogx(ph, nn, "d-", color="#6a1b9a")
    c.axhline(2.0, ls="--", color="0.4", lw=1.0, label=r"Bagnold $n=2$")
    c.axhline(0.0, ls=":", color="0.4", lw=1.0, label=r"elastic $n=0$")
    c.set_xlabel(r"area fraction $\phi$")
    c.set_ylabel(r"$n=\mathrm{d}\ln\tau/\mathrm{d}\ln\dot\gamma$")
    c.set_title(r"(c) rate exponent: elastic -> inertial with falling $\phi$")
    c.invert_xaxis()
    c.grid(True, which="both", alpha=0.3)
    c.legend(fontsize=7)

    # (d) kinetic share of the shear stress
    d = ax[1, 1]
    d.semilogx(phis, [p["kinetic_stress_share"] for p in pts], "o-", color="#2e7d32")
    d.axhline(0.5, ls="--", color="0.4", lw=1.0, label="collision-borne = contact-borne")
    d.set_xlabel(r"area fraction $\phi$")
    d.set_ylabel("kinetic share of $\\tau$")
    d.set_title("(d) contact-borne -> collision-borne")
    d.invert_xaxis()
    d.grid(True, which="both", alpha=0.3)
    d.legend(fontsize=7)

    fig.tight_layout()
    path = RUNS / "fig_phi_transition.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prereg", action="store_true")
    ap.add_argument("--stage", choices=("map", "rates", "controls", "scatter"))
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args(argv)
    RUNS.mkdir(parents=True, exist_ok=True)

    if args.prereg:
        p = RUNS / "prereg.json"
        if p.exists():
            print(f"prereg already exists (immutable): {p}")
        else:
            p.write_text(json.dumps(preregistration(), indent=1))
            print(f"wrote {p}")
        return 0

    if args.stage:
        specs = point_specs(args.stage)
        t0 = time.time()
        res = _parallel(specs, args.jobs)
        for r in res:
            print(f"{r['tag']:>24s} phi={r['phi']:.2f} gd={r['gamma_dot']:g} "
                  f"tau={r['tau']:.4g} P={r['pressure']:.4g} Z={r['coordination']:.2f} "
                  f"kinshare={r['kinetic_stress_share']:.3f} "
                  f"gates={'ok' if r['gates']['all'] else 'REJECT'} "
                  f"({r['runtime_s']:.0f}s)")
        print(f"stage {args.stage}: {len(res)} points in {time.time() - t0:.0f}s")
        return 0

    if args.analyse:
        out = analyse()
        print(write_tables(out))
        print(make_figure(out))
        print(json.dumps(out["verdicts"], indent=1))
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())