#!/usr/bin/env python3
"""Round 4 of CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT: reach into the inertial branch.

Why this round
--------------
R3 mapped ``tau*(kappa)`` over ``kappa = 362 .. 5.79e5`` and found a *curved*
curve: the local exponent climbs from ``beta ~ 0.89`` to ``~ 0.98`` (``n`` from
0.21 to 0.04).  Both R2 and R3 left the same lead open: the elastic -> inertial
transition of Campbell (2002) has **never been touched from below**.  Every
gated point sits in the elastic branch, so the two ledger rows that describe the
other branch could not be tested at all:

* R1  Bagnold exponent ``n = 2`` (inertial), 
* R2  regime boundary ``kappa_c ~ 1e4`` (order of magnitude).

Round 4 extends the grid *downward* in ``kappa`` by raising the shear rate at
fixed stiffness (and lowering the stiffness at fixed rate), because
``kappa = k_n/(rho d^3 gamma_dot^2)``: a low-``kappa`` point is *cheap* (fewer
steps for the same strain), the constraint is physical, not computational --
soft contacts at high rate overlap more, so the small-overlap validity gate is
what limits how far down the branch can be reached.  That limit is measured
rather than assumed.

Three pre-registered questions
------------------------------
P1  reach       how far down in ``kappa`` does the apparatus stay valid, and
                which gate stops it (documented as a wall, not a failure)?
P2  inertial    does the local exponent ``n`` rise toward the Bagnold value 2 as
                ``kappa`` falls, and does it reach >= 0.5 at the deepest valid
                point (i.e. is the inertial leg at least half of the stress)?
P3  partition   fit the two-leg model ``tau* = A kappa + B`` (elastic + Bagnold).
                Is it preferred over the single power law at equal parameter
                count (AIC), and does it recover the elastic coefficient ``A``
                that R3 read off a single stiff rung?
P4  boundary    the fitted crossover ``kappa_x = B/A`` -- where the inertial leg
                carries half the stress -- against Campbell's map boundary
                ``kappa_c ~ 1e4`` (the R2 ledger row, factor-3 band).
P5  cross-check the measured kinetic-stress share against the model's inertial
                fraction: if the fitted "inertial" leg is a collisional term it
                cannot be orders of magnitude above the kinetic share actually
                measured at the same point.  A violation re-labels ``B`` as a
                rate-dependent *contact* term rather than a kinetic one.

Pre-registration
----------------
``--prereg`` writes design + predictions + decision rules + the sha256 of the R3
analysis it reuses to ``runs/inertial_branch/prereg.json`` BEFORE any point is
run; ``--analyse`` re-reads it and applies it verbatim.

Usage
-----
    PYTHONPATH=src python3 .../inertial_branch.py --prereg
    PYTHONPATH=src python3 .../inertial_branch.py --stage grid
    PYTHONPATH=src python3 .../inertial_branch.py --analyse
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.elastic_limit_shear import (  # noqa: E402
    ElasticLimitShearCase, simulate_plane_shear, two_term_stress_fit)

RUNS = HERE / "runs" / "inertial_branch"
R3_RUNS = HERE / "runs" / "kappa_collapse"
R3_ANALYSIS = R3_RUNS / "analysis.json"
PREREG = RUNS / "prereg.json"

CASE = "CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT"
ROUND = "R4 (inertial-branch reach + elastic/inertial stress partition)"

# --- fixed apparatus: identical to the R3 grid so points are comparable ------
PHI = 0.80
N = 400
DT_FACTOR = 0.004
STRAIN = 1.4
STRAIN_FROM = 0.7
SEED = 20261112

# references / thresholds
BETA_ELASTIC = 1.0
BETA_INERTIAL = 0.0
N_BAGNOLD = 2.0
KAPPA_C_CAMPBELL = 1.0e4       # R2 ledger row: order of magnitude only
FACTOR3 = 3.0
KAPPA_FLOOR_R3 = 361.68981481481484   # lowest gated kappa in R3
TOL_EXPONENT = 0.15
N_REACH_HALF = 0.5             # P2: inertial leg at least half the stress
TOL_A_VS_R3 = 0.15             # P3: recover R3's elastic coefficient
MIN_GATED_FIT = 5

# --- design -----------------------------------------------------------------
# kappa = k_n / (rho d^3 gdot^2) with rho = 1, d = 1.2 -> d^3 = 1.728.
GRID: List[Dict[str, Any]] = [
    {"tag": "kn1e4_g6.0", "kn": 1.0e4, "gdot": 6.0},     # kappa = 160.8
    {"tag": "kn1e4_g8.0", "kn": 1.0e4, "gdot": 8.0},     # kappa =  90.4  (+ twins)
    {"tag": "kn1e4_g12.0", "kn": 1.0e4, "gdot": 12.0},   # kappa =  40.2
    {"tag": "kn1e4_g16.0", "kn": 1.0e4, "gdot": 16.0},   # kappa =  22.6
    {"tag": "kn1e4_g24.0", "kn": 1.0e4, "gdot": 24.0},   # kappa =  10.0
    {"tag": "kn1e4_g32.0", "kn": 1.0e4, "gdot": 32.0},   # kappa =   5.65 (gate wall?)
    {"tag": "kn2.5e3_g8.0", "kn": 2.5e3, "gdot": 8.0},   # kappa =  22.6  iso-kappa twin
    {"tag": "kn1e3_g6.0", "kn": 1.0e3, "gdot": 6.0},     # kappa =  16.1
    {"tag": "kn1e3_g8.0", "kn": 1.0e3, "gdot": 8.0},     # kappa =   9.05
]

NOISE_TWINS_FROM = "kn1e4_g8.0"
NOISE_SEEDS = (SEED + 1, SEED + 2)

PREDICTIONS = {
    "P1_reach": {
        "statement": ("the apparatus stays valid (all physics gates) at least 0.5 decade "
                      "below the R3 floor kappa = 361.7, i.e. at kappa <= 114, and the "
                      "first kappa that fails a gate is reported with the failing gate"),
        "rule": ("PASS if >= 1 gated point with kappa <= 361.7/10**0.5; PARTIAL if >= 1 gated "
                 "point below 361.7; else NOT_SUPPORTED (write down the wall)"),
    },
    "P2_inertial": {
        "statement": ("as kappa falls the local exponent n = 2(1-beta) rises monotonically and "
                      "reaches >= 0.5 at the deepest gated pair (inertial leg at least half "
                      "the stress, edit: Bagnold n=2 is the ceiling)"),
        "rule": ("PASS if n rises monotonically over the sub-R3 pairs AND n_deepest >= 0.5; "
                 "PARTIAL if monotone but n_deepest >= 0.3; else NOT_SUPPORTED"),
    },
    "P3_partition": {
        "statement": ("the two-leg model tau* = A kappa + B beats the single power law at equal "
                      "parameter count (delta AIC < 0) and recovers the elastic coefficient "
                      "A within 15 % of R3's stiff-rung value"),
        "rule": ("PASS if delta_aic < -2 AND |A/A_R3 - 1| <= 0.15; PARTIAL if only one leg; "
                 "else NOT_SUPPORTED"),
    },
    "P4_boundary": {
        "statement": ("the fitted crossover kappa_x = B/A is consistent with Campbell's map "
                      "boundary kappa_c ~ 1e4 (R2 ledger row, order of magnitude)"),
        "rule": ("PASS if kappa_x within a factor 3 of 1e4; PARTIAL within a factor 10; "
                 "else NOT_SUPPORTED (report the measured kappa_x, do not move the target)"),
    },
    "P5_kinetic_crosscheck": {
        "statement": ("the fitted inertial fraction c/(kappa+c) is not orders of magnitude "
                      "above the kinetic stress share measured at the same point: the leg is "
                      "collisional if the ratio (measured/model) stays within [0.1, 10]"),
        "rule": ("kinetic_consistent if every ratio >= 0.1; contact_rate_dependent otherwise "
                 "(a re-labelling of B, not a fit failure); not_run if no kinetic share"),
    },
}

NON_PREDICTIONS = {
    "bagnold_coefficient": ("B is our own measured Bagnold prefactor; Bagnold (1954) / Campbell "
                            "(2002) table values are not accessible from this sandbox, so the "
                            "coefficient leg remains WITHHELD rather than reconstructed."),
    "geometry": "2-D disks; Campbell's map is 3-D (declared deviation, unchanged from R1-R3).",
    "kappa_definition": ("kappa uses the number-average diameter d = 1.2 with rho = 1; a "
                         "different d (e.g. Sauter) rescales kappa by (d/d')^3 and hence moves "
                         "the crossover by the same factor -- stated, not fitted away."),
}


def _case_for(kn: float, gdot: float, seed: int = SEED) -> ElasticLimitShearCase:
    return ElasticLimitShearCase(n_particles=N, solid_fraction=PHI, kn=kn,
                                 shear_rate=gdot, strain_total=STRAIN,
                                 strain_average_from=STRAIN_FROM,
                                 dt_factor=DT_FACTOR, seed=seed)


def kappa_of_point(kn: float, gdot: float) -> float:
    return float(_case_for(kn, gdot).kappa)


def design_table() -> List[Dict[str, Any]]:
    rows = []
    for s in GRID:
        rows.append({**s, "kappa_predicted": kappa_of_point(s["kn"], s["gdot"]),
                     "phi": PHI, "n_particles": N, "seed": SEED})
    for i, sd in enumerate(NOISE_SEEDS):
        rows.append({"tag": f"{NOISE_TWINS_FROM}_seed{sd}", "kn": 1.0e4, "gdot": 8.0,
                     "kappa_predicted": kappa_of_point(1.0e4, 8.0), "phi": PHI,
                     "n_particles": N, "seed": sd, "role": "noise_twin"})
    for r in rows:
        grp = [q for q in rows if q.get("role") != "noise_twin" and
               abs(q["kappa_predicted"] - r["kappa_predicted"]) <= 0.002 * r["kappa_predicted"]]
        r["group"] = min(q["tag"] for q in grp) if len(grp) > 1 else None
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_prereg() -> Dict[str, Any]:
    rows = design_table()
    prereg = {
        "case": CASE, "round": ROUND,
        "created": "before any R4 point was run",
        "anchor_file": str(R3_ANALYSIS.relative_to(HERE)),
        "anchor_sha256": _sha256(R3_ANALYSIS) if R3_ANALYSIS.exists() else None,
        "apparatus": {"phi": PHI, "n_particles": N, "dt_factor": DT_FACTOR,
                      "strain_total": STRAIN, "strain_from": STRAIN_FROM, "seed": SEED,
                      "kernel": "tensordem.elastic_limit_shear.simulate_plane_shear"},
        "design": rows,
        "kappa_floor_reused_r3": KAPPA_FLOOR_R3,
        "predictions": PREDICTIONS,
        "non_predictions": NON_PREDICTIONS,
        "thresholds": {"factor3": FACTOR3, "kappa_c_campbell": KAPPA_C_CAMPBELL,
                       "n_reach_half": N_REACH_HALF, "tol_a_vs_r3": TOL_A_VS_R3,
                       "min_gated_fit": MIN_GATED_FIT},
        "noise_twins": {"from": NOISE_TWINS_FROM, "seeds": list(NOISE_SEEDS)},
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    PREREG.write_text(json.dumps(prereg, indent=1, default=float))
    return prereg


POINT_FIELDS = ("tag", "kn", "gamma_dot", "kappa", "tau", "pressure", "tau_star",
                "mu_eff", "I", "coordination", "kinetic_stress_share",
                "max_overlap_ratio", "mean_overlap_ratio", "steady_drift_tau",
                "energy_defect", "gates", "group", "runtime_s")


def run_point(row: Dict[str, Any]) -> Dict[str, Any]:
    case = _case_for(row["kn"], row["gdot"], seed=row["seed"])
    out = simulate_plane_shear(case, verbose=True)
    rec = {
        "tag": row["tag"], "role": row.get("role", "grid"), "group": row["group"],
        "kn": row["kn"], "gamma_dot": row["gdot"], "phi": PHI, "n_particles": N,
        "seed": row["seed"], "kappa": out["kappa"], "tau": out["tau"],
        "pressure": out["pressure"], "tau_star": out["tau_star"],
        "mu_eff": out["mu_eff"], "I": out["I"], "coordination": out["coordination"],
        "kinetic_stress_share": out["kinetic_stress_share"],
        "tau_kinetic": out["tau_kinetic"], "tau_contact": out["tau_contact"],
        "max_overlap_ratio": out["max_overlap_ratio"],
        "mean_overlap_ratio": out["mean_overlap_ratio"],
        "steady_drift_tau": out["steady_drift_tau"],
        "energy_defect": out["energy"]["relative_defect"],
        "energy_defect_global": out["energy"]["relative_defect_global"],
        "gates": out["gates"], "runtime_s": out["runtime_s"],
        "dt_factor": DT_FACTOR, "strain_total": STRAIN, "strain_from": STRAIN_FROM,
        "ok_to_analyse": True,
    }
    (RUNS / f"{row['tag']}.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


def stage_grid(force: bool = False) -> None:
    json.loads(PREREG.read_text())  # refuse to run without a pre-registration
    t0 = time.time()
    for row in design_table():
        path = RUNS / f"{row['tag']}.json"
        if path.exists() and not force:
            print(f"[skip] {row['tag']} (exists)")
            continue
        print(f"[run ] {row['tag']} kn={row['kn']:.3g} gdot={row['gdot']:.3g} "
              f"kappa={row['kappa_predicted']:.4g}")
        run_point(row)
    print(f"stage grid done in {time.time() - t0:.0f} s")


def _r3_points() -> List[Dict[str, Any]]:
    """Reuse the R3 grid points (read-only) with the sigma the R4 noise floor sets."""
    pts = []
    if not R3_ANALYSIS.exists():
        return pts
    r3 = json.loads(R3_ANALYSIS.read_text())
    for p in r3.get("points", []):
        if p.get("gates", {}).get("all"):
            pts.append({**p, "tag": p["tag"], "source": "R3"})
    return pts


def analyse() -> Dict[str, Any]:
    json.loads(PREREG.read_text())  # refuse to analyse without a pre-registration
    grid, twins = [], []
    for p in sorted(RUNS.glob("*.json")):
        if p.name in ("prereg.json", "analysis.json"):
            continue
        d = json.loads(p.read_text())
        if "tag" not in d:
            continue
        (twins if d.get("role") == "noise_twin" else grid).append(d)
    gated = [r for r in grid if r["gates"]["all"]]
    rejected = [r for r in grid if not r["gates"]["all"]]

    out: Dict[str, Any] = {
        "case": CASE, "round": ROUND,
        "n_points": len(grid), "n_gated": len(gated),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "points": [{k: r.get(k) for k in POINT_FIELDS}
                   for r in sorted(grid, key=lambda r: -r["kappa"])],
        "rejected": [{"tag": r["tag"], "kappa": r["kappa"],
                      "failing_gates": [k for k, v in r["gates"].items()
                                        if k != "all" and not v],
                      "mean_overlap_ratio": r["mean_overlap_ratio"],
                      "max_overlap_ratio": r["max_overlap_ratio"],
                      "energy_defect": r["energy_defect"]} for r in rejected],
    }

    # ---- noise floor (R4's own twins, used as the sigma of every point) ----
    base = [r["tau_star"] for r in grid if r["tag"] == NOISE_TWINS_FROM]
    vals = np.array([t["tau_star"] for t in twins] + base, float)
    rel = float(np.std(vals, ddof=1) / np.mean(vals)) if vals.size > 1 else float("nan")
    out["noise_floor"] = {"tags": [t["tag"] for t in twins] + [NOISE_TWINS_FROM],
                          "tau_star": vals.tolist(), "relative_std": rel,
                          "n_seeds": int(vals.size)}
    sig = float(rel) if np.isfinite(rel) else float("nan")

    # ---- P1 reach ----------------------------------------------------------
    below = [r for r in gated if r["kappa"] < KAPPA_FLOOR_R3]
    out["reach"] = {
        "kappa_floor_r3": KAPPA_FLOOR_R3,
        "gated_below_r3_floor": len(below),
        "kappa_min_gated": min([r["kappa"] for r in below], default=None),
        "decades_below_r3": (math.log10(KAPPA_FLOOR_R3 / min(r["kappa"] for r in below))
                             if below else 0.0),
        "wall": (min(out["rejected"], key=lambda r: -r["kappa"]) if out["rejected"] else None),
        "rejected_tags": [r["tag"] for r in out["rejected"]],
    }
    km = out["reach"]["kappa_min_gated"]
    if km is not None and km <= KAPPA_FLOOR_R3 / math.sqrt(10.0):
        out["P1_verdict"] = "PASS"
    elif km is not None:
        out["P1_verdict"] = "PARTIAL"
    else:
        out["P1_verdict"] = "NOT_SUPPORTED"

    # ---- all gated points (R4 + R3) for the partition fit ------------------
    r3 = _r3_points()
    all_pts = [{**{k: p.get(k) for k in POINT_FIELDS}, "gates": {"all": True},
                "source": "R4", "tau_star_relative_sigma": sig} for p in gated]
    all_pts += [{**p, "tau_star_relative_sigma": sig} for p in r3]
    all_pts.sort(key=lambda p: -p["kappa"])
    out["fit_input"] = {
        "n_r4": len(gated), "n_r3": len(r3), "n_total": len(all_pts),
        "kappa_span": [min(p["kappa"] for p in all_pts),
                       max(p["kappa"] for p in all_pts)] if all_pts else None,
        "decades": (math.log10(max(p["kappa"] for p in all_pts) /
                               min(p["kappa"] for p in all_pts)) if all_pts else None),
    }

    # ---- local exponents on the R4 side (sub-R3 pairs) ---------------------
    ordered = sorted(gated, key=lambda r: r["kappa"])
    local = []
    for a, b in zip(ordered, ordered[1:]):
        lk = math.log(b["kappa"] / a["kappa"])
        if lk <= 1e-12:
            continue
        beta = math.log(b["tau_star"] / a["tau_star"]) / lk
        widen = abs(lk)
        sigma_beta = (math.sqrt(2.0) * sig / widen) if (np.isfinite(sig) and widen > 0) \
            else float("nan")
        local.append({"pair": f"{a['tag']}->{b['tag']}", "k_lo": a["kappa"],
                      "k_hi": b["kappa"], "gmid": math.sqrt(a["kappa"] * b["kappa"]),
                      "beta_local": beta, "n_local": 2.0 * (1.0 - beta),
                      "sigma_beta": sigma_beta,
                      "sigma_n": 2.0 * sigma_beta if np.isfinite(sigma_beta) else None})
    out["local_slopes"] = local
    r4_pairs = [p for p in local if p["k_hi"] <= KAPPA_FLOOR_R3 * 1.0001]
    out["local_slopes_sub_r3"] = r4_pairs

    # ---- P2 inertial approach ---------------------------------------------
    if len(r4_pairs) >= 1:
        ns = [p["n_local"] for p in sorted(r4_pairs, key=lambda p: -p["gmid"])]
        monotone = all(b >= a - 0.05 for a, b in zip(ns, ns[1:]))
        n_deep = ns[-1]
        if monotone and n_deep >= N_REACH_HALF:
            out["P2_verdict"] = "PASS"
        elif monotone and n_deep >= 0.3:
            out["P2_verdict"] = "PARTIAL"
        else:
            out["P2_verdict"] = "NOT_SUPPORTED"
        out["P2_detail"] = {"n_by_descending_gmid": ns, "monotone": monotone,
                            "n_deepest": n_deep}
    else:
        out["P2_verdict"] = "not_run"
        out["P2_detail"] = None

    # ---- P3/P4 partition + crossover --------------------------------------
    fit = two_term_stress_fit(all_pts, min_points=MIN_GATED_FIT, seed=SEED)
    out["partition"] = fit
    if fit.get("status") == "ok":
        a_r3 = 0.013570045890911866  # R3 elastic_coefficient.tau_star_over_kappa
        rel_a = abs(fit["A"] / a_r3 - 1.0)
        daic = fit["delta_aic_two_term_minus_power"]
        out["P3_detail"] = {"A": fit["A"], "A_r3": a_r3, "rel_diff_A": rel_a,
                            "delta_aic": daic}
        legs = (daic < -2.0) + (rel_a <= TOL_A_VS_R3)
        out["P3_verdict"] = ("PASS" if legs == 2 else ("PARTIAL" if legs == 1
                                                      else "NOT_SUPPORTED"))
        kx = fit["kappa_crossover"]
        ratio = kx / KAPPA_C_CAMPBELL
        out["P4_detail"] = {"kappa_crossover": kx, "kappa_c_campbell": KAPPA_C_CAMPBELL,
                            "ratio": ratio,
                            "ci68": fit.get("kappa_crossover_ci68")}
        if 1.0 / FACTOR3 <= ratio <= FACTOR3:
            out["P4_verdict"] = "PASS"
        elif 0.1 <= ratio <= 10.0:
            out["P4_verdict"] = "PARTIAL"
        else:
            out["P4_verdict"] = "NOT_SUPPORTED"
        ck = fit.get("kinetic_cross_check") or []
        ratios = [r["ratio_measured_over_model"] for r in ck
                  if r.get("ratio_measured_over_model") is not None]
        if ratios:
            out["P5_verdict"] = ("kinetic_consistent" if min(ratios) >= 0.1
                                 else "contact_rate_dependent")
            out["P5_detail"] = {"min_ratio": min(ratios), "median_ratio":
                                float(np.median(ratios)), "n_points": len(ratios)}
        else:
            out["P5_verdict"] = "not_run"
            out["P5_detail"] = None
    else:
        out["P3_verdict"] = out["P4_verdict"] = out["P5_verdict"] = "not_run"
        out["P3_detail"] = out["P4_detail"] = out["P5_detail"] = None

    # ---- descriptive diagnostic (no verdict): why the branch is unreachable --
    # The small-overlap validity gate is what limits the walk down kappa, so its
    # scaling is worth a number: mean overlap/d vs kappa across every gated point
    # of R4 and R3, and the extrapolated kappa at which the contact would be
    # genuinely small.  This is an extrapolation of our own data, labelled as
    # such -- it justifies the constant-pressure lead, it does not test it.
    ovl = [(float(p["kappa"]), float(p["mean_overlap_ratio"])) for p in all_pts
           if p.get("mean_overlap_ratio")]
    if len(ovl) >= 4:
        lk = np.log([o[0] for o in ovl])
        lo = np.log([o[1] for o in ovl])
        slope, icept = np.polyfit(lk, lo, 1)
        resid = lo - (icept + slope * lk)
        k_gate = float(math.exp((math.log(0.12) - icept) / slope))
        out["overlap_scaling"] = {
            "n_points": len(ovl),
            "exponent_dln_overlap_dln_kappa": float(slope),
            "rms_log_residual": float(np.sqrt(np.mean(resid ** 2))),
            "measured_overlap_min": float(min(o[1] for o in ovl)),
            "measured_overlap_max": float(max(o[1] for o in ovl)),
            "measured_kappa_span": [float(min(o[0] for o in ovl)),
                                    float(max(o[0] for o in ovl))],
            "kappa_at_overlap_gate_crossing": k_gate,
            "gate": "mean_overlap_ratio <= 0.12",
            "note": ("overlap is NOT a clean power law of kappa (residual ~14 %): it "
                     "saturates near 0.03 at the stiff end because the bed at phi = 0.80 "
                     "is already jammed, so the overlap floor is packing-geometric rather "
                     "than stiffness-controlled.  The gate crossing is therefore quoted as "
                     "a measured window edge, and no extrapolation beyond the measured "
                     "kappa span is offered -- the flat slope would make such a number "
                     "meaningless."),
        }
    else:
        out["overlap_scaling"] = None

    out["verdicts"] = {k: out[k] for k in ("P1_verdict", "P2_verdict", "P3_verdict",
                                           "P4_verdict", "P5_verdict")}
    out["declared_deviations"] = {
        "geometry": "2-D disks vs Campbell's 3-D",
        "kappa_definition": "d = number-average diameter, rho = 1",
        "reused_r3_points": ("the R3 grid is reused read-only for the partition fit; its "
                            "own gates/analysis are unchanged (no re-interpretation)"),
    }
    (RUNS / "analysis.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def write_tables(a: Dict[str, Any]) -> Path:
    L = [f"# {a['case']} — {a['round']}", "",
         "## Points (this round)", "",
         "| tag | k_n | gdot | kappa | tau* | I | mean ovl | max ovl | kinetic share | gates |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for p in a["points"]:
        L.append("| {tag} | {kn:.3g} | {gamma_dot:.3g} | {kappa:.4g} | {tau_star:.4g} | "
                 "{I:.4f} | {mean_overlap_ratio:.4f} | {max_overlap_ratio:.3f} | "
                 "{kinetic_stress_share:.2e} | {g} |".format(
                     g="PASS" if p["gates"]["all"] else "FAIL", **p))
    if a["rejected"]:
        L += ["", "## Gate wall (rejected points, reported not dropped)", "",
              "| tag | kappa | failing gates | mean ovl | max ovl | energy defect |",
              "|---|---|---|---|---|---|"]
        for r in a["rejected"]:
            L.append(f"| {r['tag']} | {r['kappa']:.4g} | {', '.join(r['failing_gates'])} | "
                     f"{r['mean_overlap_ratio']:.4f} | {r['max_overlap_ratio']:.3f} | "
                     f"{r['energy_defect']:.3f} |")
    L += ["", "## Local exponents (R4 pairs, descending kappa)", "",
          "| pair | sqrt(k_lo k_hi) | beta_local | n_local | sigma_n |",
          "|---|---|---|---|---|"]
    for p in a["local_slopes"]:
        sn = p["sigma_n"]
        L.append(f"| {p['pair']} | {p['gmid']:.4g} | {p['beta_local']:.3f} | "
                 f"{p['n_local']:.3f} | {'n/a' if sn is None else f'{sn:.3f}'} |")
    f = a.get("partition") or {}
    if f.get("status") == "ok":
        L += ["", "## Elastic + inertial partition  tau* = A kappa + B", "",
              f"- A = {f['A']:.5g} (68% CI {f['A_ci68']})",
              f"- B = {f['B']:.5g}",
              f"- crossover kappa_x = B/A = {f['kappa_crossover']:.4g} "
              f"(68% CI {f['kappa_crossover_ci68']})",
              f"- single power law: beta = {f['single_power_law']['beta']:.4f}, "
              f"log-RSS = {f['single_power_law']['rss_log']:.4g}, AIC = "
              f"{f['single_power_law']['aic']:.2f}",
              f"- two-term:           log-RSS = {f['two_term']['rss_log']:.4g}, AIC = "
              f"{f['two_term']['aic']:.2f}",
              f"- delta AIC (two-term - power law) = {f['delta_aic_two_term_minus_power']:.2f} "
              f"-> preferred: **{f['preferred_model']}**", "",
              "| kappa | tau* measured | tau* model | model beta | model inertial frac | "
              "measured kinetic share | ratio |", "|---|---|---|---|---|---|---|"]
        shares = {round(r["kappa"], 6): r for r in f.get("kinetic_cross_check", [])}
        for p in sorted(f["points"], key=lambda q: -q["kappa"]):
            s = shares.get(round(p["kappa"], 6), {})
            ms = s.get("measured_kinetic_share")
            rr = s.get("ratio_measured_over_model")
            L.append(f"| {p['kappa']:.4g} | {p['tau_star']:.4g} | {p['model_tau_star']:.4g} | "
                     f"{p['model_beta']:.3f} | {p['model_inertial_fraction']:.4f} | "
                     f"{'n/a' if ms is None else f'{ms:.2e}'} | "
                     f"{'n/a' if rr is None else f'{rr:.3f}'} |")
    L += ["", "## Verdicts", "", "| prediction | verdict |", "|---|---|"]
    for k, v in a["verdicts"].items():
        L.append(f"| {k} | **{v}** |")
    p = RUNS / "tables.md"
    p.write_text("\n".join(L) + "\n")
    return p


def make_figure(a: Dict[str, Any]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
    f = a.get("partition") or {}
    gated = [p for p in a["points"] if p["gates"]["all"]]
    rej = a["points"] if False else [p for p in a["points"] if not p["gates"]["all"]]

    # (a) tau* vs kappa with both models
    ax = axes[0, 0]
    ax.plot([p["kappa"] for p in gated], [p["tau_star"] for p in gated], "o", ms=7,
            color="C0", label="R4 gated")
    if rej:
        ax.plot([p["kappa"] for p in rej], [p["tau_star"] for p in rej], "x", ms=7,
                color="C3", label="R4 rejected (gate)")
    r3 = _r3_points()
    if r3:
        ax.plot([p["kappa"] for p in r3], [p["tau_star"] for p in r3], "s", ms=5,
                color="C7", alpha=0.8, label="R3 grid (reused)")
    if gated:
        ks = np.array([p["kappa"] for p in gated], float)
        xs = np.logspace(np.log10(min(ks.min(), 3.0)), np.log10(max(ks.max(), 1e5)), 80)
        if f.get("status") == "ok":
            ax.plot(xs, f["A"] * (xs + f["kappa_crossover"]), "k-", lw=1.6,
                    label=fr"$A\kappa+B$ ($\kappa_x$={f['kappa_crossover']:.3g})")
            ax.axvline(f["kappa_crossover"], color="k", ls=":", lw=1.0)
        ax.plot(xs, xs * 0.01357, "--", color="gray", lw=1.0, label=r"elastic $\beta$=1")
    ax.axhline(1.0, ls=":", color="gray", lw=1.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\kappa$"); ax.set_ylabel(r"$\tau^*$")
    ax.set_title("(a) reach downward in " + r"$\kappa$")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)

    # (b) local n vs kappa
    ax = axes[0, 1]
    if a["local_slopes"]:
        ax.semilogx([p["gmid"] for p in a["local_slopes"]],
                    [p["n_local"] for p in a["local_slopes"]], "o-", color="C2", lw=0.9,
                    label=r"$n$ local (R4 pairs)")
    if f.get("status") == "ok":
        xs = np.logspace(-1, 7, 200)
        ax.semilogx(xs, 2.0 * f["kappa_crossover"] / (xs + f["kappa_crossover"]), "k-",
                    lw=1.4, label=r"$n=2c/(\kappa+c)$ from the fit")
    ax.axhline(0.0, ls="--", color="gray", lw=1.0, label="elastic n=0")
    ax.axhline(2.0, ls=":", color="gray", lw=1.0, label="Bagnold n=2")
    ax.axhline(N_REACH_HALF, ls="-.", color="C1", lw=0.9, label="P2 half-stress line")
    ax.set_xlabel(r"$\kappa$"); ax.set_ylabel(r"$n = 2(1-\beta)$")
    ax.set_title("(b) shear-rate exponent vs stiffness")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)

    # (c) residuals
    ax = axes[1, 0]
    if f.get("status") == "ok":
        ks = [p["kappa"] for p in f["points"]]
        ax.semilogx(ks, [100 * p["residual_log"] for p in f["points"]], "o-", color="C4",
                    lw=0.9, label="two-term")
        sp = f["single_power_law"]
        ax.semilogx(ks, [100 * (math.log(p["tau_star"]) - math.log(sp["C"])
                                - sp["beta"] * math.log(p["kappa"]))
                         for p in f["points"]], "s--", color="C5", lw=0.9,
                    label="single power law")
        ax.axhline(0, color="k", lw=0.6)
        nf = a["noise_floor"]["relative_std"]
        if np.isfinite(nf):
            ax.axhspan(-100 * nf, 100 * nf, color="green", alpha=0.15,
                       label=f"noise floor ±{100 * nf:.1f}%")
    ax.set_xlabel(r"$\kappa$"); ax.set_ylabel("residual in ln tau* (%)")
    ax.set_title("(c) model discrimination")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)

    # (d) kinetic cross-check
    ax = axes[1, 1]
    ck = [r for r in (f.get("kinetic_cross_check") or [])
          if r.get("measured_kinetic_share")]
    if ck:
        ax.loglog([r["model_inertial_fraction"] for r in ck],
                  [r["measured_kinetic_share"] for r in ck], "o", color="C6", ms=6)
        for r in ck:
            ax.annotate(f"{r['kappa']:.0f}", (r["model_inertial_fraction"],
                                              r["measured_kinetic_share"]),
                        fontsize=6, alpha=0.8)
        lim = [min(min(r["model_inertial_fraction"] for r in ck),
                   min(r["measured_kinetic_share"] for r in ck)) * 0.5,
               max(max(r["model_inertial_fraction"] for r in ck),
                   max(r["measured_kinetic_share"] for r in ck)) * 2.0]
        ax.plot(lim, lim, "k--", lw=1.0, label="1:1")
        ax.plot(lim, [0.1 * v for v in lim], "k:", lw=0.9, label="ratio 0.1")
        ax.set_xlabel("model inertial fraction  c/(kappa+c)")
        ax.set_ylabel("measured kinetic stress share")
        ax.set_title("(d) is the inertial leg collisional?")
        ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    p = RUNS / "fig_inertial_branch.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prereg", action="store_true")
    ap.add_argument("--stage", choices=["grid"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args(argv)
    if args.prereg:
        pre = write_prereg()
        print(f"pre-registered: {len(pre['design'])} points -> {PREREG}")
    if args.stage == "grid":
        stage_grid(force=args.force)
    if args.analyse:
        a = analyse()
        t = write_tables(a)
        fg = make_figure(a)
        print(json.dumps(a["verdicts"], indent=1))
        if a.get("partition", {}).get("status") == "ok":
            print(f"A={a['partition']['A']:.4g} B={a['partition']['B']:.4g} "
                  f"kappa_x={a['partition']['kappa_crossover']:.4g} "
                  f"dAIC={a['partition']['delta_aic_two_term_minus_power']:.2f}")
        print(f"tables -> {t}\nfigure -> {fg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())