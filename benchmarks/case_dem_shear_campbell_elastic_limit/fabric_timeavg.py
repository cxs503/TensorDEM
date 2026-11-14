#!/usr/bin/env python3
"""CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT -- R5: steady-window fabric statistics.

Why this round exists
---------------------
R4 closed the *coefficient* structure of Campbell's elastic limit
(``A = tau d/k_n`` saturates above kappa ~ 2.3e4 within +5.7% = PASS) but its
falsifiable closure -- "two geometric constants (a_xy, f_norm) predict the shear
stress of an independent concentration map" -- FAILED: P4 (constant a_xy) and P5
(out-of-sample prediction) both failed, with a systematic +31% overshoot at the
dense end.  R4 named its own prime suspect and left it unquantified:

    "单快照 fabric ... 是 P5 系统性正偏的首选嫌疑，本轮未消除"

i.e. the fabric constants were measured on the **final snapshot only**, while the
stress they must explain is a **time average** over the steady window.  R4's own
diagnostic shows how large that gap can be: snapshot-vs-time-average coordination
differs by up to 14.2%.

R5 removes the suspect instead of arguing about it: the simulator now accumulates
the contact-network fabric *inside the same steady window the stress is averaged
over* (``simulate_plane_shear(fabric_statistics=True)``), together with the exact
normal/tangential split of the shear virial, so ``f_norm`` becomes a measured time
average rather than a snapshot residual.

Pre-registered question: with a time-averaged fabric, does the two-constant
closure (a) become concentration independent and (b) predict the independent R2
concentration map within the ledger's 15%?

Usage
-----
    python fabric_timeavg.py --prereg     # write predictions BEFORE any fabric run
    python fabric_timeavg.py --stage fabric
    python fabric_timeavg.py --analyse
    python fabric_timeavg.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from typing import Any, Dict, List, Sequence

import numpy as np

from tensordem.elastic_limit_shear import ElasticLimitShearCase, simulate_plane_shear

from ell_coefficient import (
    CASE, PHI_MAP_MIN, RUNS as R4_RUNS, R2_RUNS, TOL, coefficient,
    contact_geometry, diameter_from_kappa, fit_saturation, load_points,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs", "fabric_timeavg")
ROUND = "R5 (steady-window fabric statistics)"

# identical apparatus to R4 leg B: only the fabric estimator changes
PHI_GRID = (0.71, 0.76, 0.80)
SEEDS = (20261112, 20261113, 20261114)
BASE = dict(kn=1.0e4, shear_rate=1.0, n_particles=400, strain_total=1.4,
            strain_average_from=0.7, dt_factor=0.004)
SNAPSHOT_NOISE = 0.0625      # R2/R4 seed noise floor, used as the "bias or not" yardstick
R4_WORST_RESIDUAL = 0.312    # R4 leg C worst out-of-sample residual (for before/after)

PREREG = {
    "case": CASE,
    "round": ROUND,
    "reference": "Campbell 2002 JFM 465:261 (abstract hashed in reference_anchors.json)",
    "motivation": (
        "R4 measured a_xy and f_norm on the final snapshot while the stress they "
        "explain is a steady-window time average; snapshot coordination deviates "
        "from the time average by up to 14.2% (R4 leg B). R4 named this the prime "
        "suspect for its systematic +31% dense-end overshoot but did not "
        "quantify it. R5 measures the same constants as time averages inside the "
        "same window as the stress."
    ),
    "what_changes": "fabric estimator (snapshot -> steady-window time average); "
                    "the apparatus, grid, seeds and kappa are unchanged from R4",
    "grid": {"phi": list(PHI_GRID), "seeds": list(SEEDS), **BASE},
    "predictions": [
        {"id": "P1",
         "statement": "with time-averaged fabric a_xy is concentration independent",
         "rule": "max/min - 1 of the per-phi mean a_xy over the 3 concentrations <= 0.15",
         "threshold": 0.15},
        {"id": "P2",
         "statement": "with the measured f_norm time average, f_norm is concentration independent",
         "rule": "max/min - 1 of the per-phi mean f_norm over the 3 concentrations <= 0.15",
         "threshold": 0.15},
        {"id": "P3",
         "statement": "two time-averaged constants predict the independent R2 map out of sample",
         "rule": "|A_pred/A_meas - 1| <= 0.15 on every gated R2 point with phi >= 0.71",
         "threshold": 0.15},
        {"id": "P4",
         "statement": "the single snapshot is a biased fabric estimator (R4's suspicion, "
                      "tested as a number rather than a narrative)",
         "rule": "|a_xy_snapshot/a_xy_timeavg - 1| >= 0.0625 (the seed-noise floor) "
                 "on at least 5 of the 9 runs",
         "threshold": 0.0625},
        {"id": "P5",
         "statement": "the factored structural identity survives with time-averaged inputs",
         "rule": "|A_pred_factored/A_normal - 1| <= 0.15 on every run, with every input "
                 "a steady-window time average",
         "threshold": 0.15},
    ],
    "honesty_rules": [
        "the R2 map is prediction-only: its points are never refitted or re-run here",
        "the apparatus is unchanged; if P1-P3 fail the factorisation is reported as "
        "FAIL and the time-average hypothesis is recorded as falsified",
        "seed noise is reported next to every spread so a spread below the noise "
        "floor is not read as a resolved physical trend",
        "one point (phi=0.80, seed 20261112) was inspected while smoke-testing the new "
        "simulator option, before this pre-registration was written; the predictions "
        "below are structural (ledger tolerance, noise floor) and were not tuned on it",
    ],
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------- stages
def stage_prereg() -> Dict[str, Any]:
    os.makedirs(RUNS, exist_ok=True)
    payload = dict(PREREG)
    payload["written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload["script_sha256"] = _sha256(__file__)
    path = os.path.join(RUNS, "prereg.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"prereg -> {path}  sha256={payload['script_sha256'][:16]}")
    return payload


def run_point(phi: float, seed: int, verbose: bool = True) -> Dict[str, Any]:
    case = ElasticLimitShearCase(solid_fraction=phi, seed=seed, **BASE)
    t0 = time.time()
    out = simulate_plane_shear(case, verbose=False, fabric_statistics=True)
    fs = out["final_state"]
    # snapshot estimator, byte-for-byte the R4 one (for the bias comparison)
    geo = contact_geometry(fs["x"], fs["y"], case.radii, case.lx, case.ly,
                           float(fs["s"]), case.kn)
    fab = out["fabric"]
    d = case.d_mean
    mean_r2 = float(np.mean(case.radii ** 2))

    a_total = out["tau_contact"] * d / case.kn      # same definition as R4
    a_norm_snap = geo["sigma_yx_normal"] * d / case.kn
    a_xy_snap = geo["overlap_weighted_fabric_xy"] / geo["mean_overlap_ratio_snapshot"]
    f_norm_snap = a_norm_snap / a_total

    a_xy_ta = fab["a_xy_mean"]
    f_norm_ta = fab["f_norm_mean"]
    ov_ta = fab["mean_overlap_ratio_mean"]
    z_ta = fab["coordination_mean"]
    a_norm_ta = fab["tau_normal_mean"] * d / case.kn
    a_norm_pred = (phi * z_ta * d ** 3 / (2.0 * math.pi * mean_r2)) * a_xy_ta * ov_ta

    point = {
        "tag": f"fab_phi{phi:.2f}_s{seed}", "role": "fabric_timeavg",
        "phi": phi, "seed": seed, "kn": case.kn, "gamma_dot": case.shear_rate,
        "d_mean": d, "kappa": case.kappa, "mean_r2": mean_r2,
        "tau": out["tau"], "tau_contact": out["tau_contact"],
        "pressure": out["pressure"], "tau_star": out["tau_star"],
        "mu_eff": out["mu_eff"], "I": out["I"],
        "coordination_run": out["coordination"],
        "mean_overlap_run": out["mean_overlap_ratio"],
        "kinetic_stress_share": out["kinetic_stress_share"],
        "steady_drift_tau": out["steady_drift_tau"],
        "energy_defect": out["energy"]["relative_defect"],
        "gates": out["gates"], "runtime_s": time.time() - t0,
        "A_total": a_total, "A_normal_timeavg": a_norm_ta,
        "fabric": fab,
        "snapshot": {
            "coordination": geo["coordination_snapshot"],
            "mean_overlap_ratio": geo["mean_overlap_ratio_snapshot"],
            "fabric_xy": geo["fabric_xy"],
            "overlap_weighted_fabric_xy": geo["overlap_weighted_fabric_xy"],
            "a_xy": a_xy_snap, "f_norm": f_norm_snap,
            "A_normal": a_norm_snap, "A_pred_factored": (
                (phi * geo["coordination_snapshot"] * d ** 3
                 / (2.0 * math.pi * geo["mean_r2"]))
                * geo["overlap_weighted_fabric_xy"]),
        },
        "timeavg": {
            "coordination": z_ta, "mean_overlap_ratio": ov_ta,
            "a_xy": a_xy_ta, "f_norm": f_norm_ta,
            "A_normal": a_norm_ta, "A_pred_factored": a_norm_pred,
            "factored_residual": (a_norm_pred / a_norm_ta - 1.0) if a_norm_ta else float("nan"),
            "virial_split_residual_max": fab["virial_split_residual_max"],
            "a_xy_std_over_samples": fab["a_xy_std"],
            "f_norm_std_over_samples": fab["f_norm_std"],
        },
        "bias": {
            "a_xy_snapshot_vs_timeavg": a_xy_snap / a_xy_ta - 1.0,
            "f_norm_snapshot_vs_timeavg": f_norm_snap / f_norm_ta - 1.0,
            "coordination_snapshot_vs_timeavg": geo["coordination_snapshot"] / z_ta - 1.0,
            "overlap_snapshot_vs_timeavg":
                geo["mean_overlap_ratio_snapshot"] / ov_ta - 1.0,
        },
    }
    if verbose:
        print("  %-22s phi=%.2f Z_t=%.3f a_xy_t=%.4f f_t=%.3f | a_xy_s=%.4f "
              "bias_a=%+.1f%% bias_Z=%+.1f%% P5res=%+.1f%% gates=%s (%.1fs)"
              % (point["tag"], phi, z_ta, a_xy_ta, f_norm_ta, a_xy_snap,
                 100 * point["bias"]["a_xy_snapshot_vs_timeavg"],
                 100 * point["bias"]["coordination_snapshot_vs_timeavg"],
                 100 * point["timeavg"]["factored_residual"],
                 point["gates"]["all"], point["runtime_s"]))
    with open(os.path.join(RUNS, point["tag"] + ".json"), "w") as fh:
        json.dump(point, fh, indent=1)
    return point


def stage_fabric() -> List[Dict[str, Any]]:
    os.makedirs(RUNS, exist_ok=True)
    print(f"time-averaged fabric grid: {len(PHI_GRID)} phi x {len(SEEDS)} seeds "
          f"({int(BASE['n_particles'])} disks, strain {BASE['strain_total']})")
    pts = []
    for phi in PHI_GRID:
        for seed in SEEDS:
            pts.append(run_point(phi, seed))
    return pts


# -------------------------------------------------------------------- analysis
def _spread(vals: Sequence[float]) -> float:
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    return float(np.max(v) / np.min(v) - 1.0) if v.size > 1 else float("nan")


def _load_fabric() -> List[Dict[str, Any]]:
    pts = []
    if not os.path.isdir(RUNS):
        return pts
    for name in sorted(os.listdir(RUNS)):
        if not name.endswith(".json") or name in ("analysis.json", "prereg.json"):
            continue
        with open(os.path.join(RUNS, name)) as fh:
            pts.append(json.load(fh))
    return pts


def analyse() -> Dict[str, Any]:
    fab_all = _load_fabric()
    if len(fab_all) < 9:
        raise SystemExit(f"fabric points missing ({len(fab_all)}/9): run --stage fabric")
    prereg_path = os.path.join(RUNS, "prereg.json")
    prereg = json.load(open(prereg_path)) if os.path.exists(prereg_path) else {}
    fab = [p for p in fab_all if p["gates"]["all"]]
    rejected = [p["tag"] for p in fab_all if not p["gates"]["all"]]

    per_phi: Dict[str, Any] = {}
    for phi in sorted({p["phi"] for p in fab}):
        sel = [p for p in fab if p["phi"] == phi]
        per_phi["%.3f" % phi] = {
            "n": len(sel),
            "a_xy_timeavg": float(np.mean([p["timeavg"]["a_xy"] for p in sel])),
            "a_xy_timeavg_seed_std": float(np.std([p["timeavg"]["a_xy"] for p in sel])),
            "a_xy_snapshot": float(np.mean([p["snapshot"]["a_xy"] for p in sel])),
            "f_norm_timeavg": float(np.mean([p["timeavg"]["f_norm"] for p in sel])),
            "f_norm_timeavg_seed_std": float(np.std([p["timeavg"]["f_norm"] for p in sel])),
            "f_norm_snapshot": float(np.mean([p["snapshot"]["f_norm"] for p in sel])),
            "coordination_timeavg": float(np.mean([p["timeavg"]["coordination"] for p in sel])),
            "a_xy_time_std_over_samples": float(np.mean(
                [p["timeavg"]["a_xy_std_over_samples"] for p in sel])),
            "factored_residual_absmax": float(np.max(
                [abs(p["timeavg"]["factored_residual"]) for p in sel])),
        }

    a_xy_v = [v["a_xy_timeavg"] for v in per_phi.values()]
    f_v = [v["f_norm_timeavg"] for v in per_phi.values()]
    a_xy_spread = _spread(a_xy_v)
    f_spread = _spread(f_v)
    a_xy_spread_snap = _spread([v["a_xy_snapshot"] for v in per_phi.values()])
    f_spread_snap = _spread([v["f_norm_snapshot"] for v in per_phi.values()])
    P1 = bool(a_xy_spread <= TOL)
    P2 = bool(f_spread <= TOL)
    P5 = bool(all(abs(p["timeavg"]["factored_residual"]) <= TOL for p in fab))

    # ---- leg C: out-of-sample prediction of the independent R2 map -------------
    r2 = load_points(R2_RUNS)
    mean_r2 = float(np.mean([p["mean_r2"] for p in fab]))
    a_xy_bar = float(np.mean(a_xy_v))
    f_bar = float(np.mean(f_v))
    predictions = []
    for p in r2:
        if not p.get("gates", {}).get("all") or p["phi"] < PHI_MAP_MIN:
            continue
        d = diameter_from_kappa(p["kn"], p["kappa"], 1.0, p["gamma_dot"])
        a_norm_pred = (p["phi"] * p["coordination"] * d ** 3
                       / (2.0 * math.pi * mean_r2)) * a_xy_bar * p["mean_overlap_ratio"]
        a_pred = a_norm_pred / f_bar
        a_meas = coefficient(p)
        predictions.append({"tag": p["tag"], "phi": p["phi"],
                            "coordination": p["coordination"],
                            "mean_overlap_ratio": p["mean_overlap_ratio"],
                            "A_meas": a_meas, "A_pred": a_pred,
                            "A_norm_pred": a_norm_pred,
                            "residual": a_pred / a_meas - 1.0})
    worst = max((abs(r["residual"]) for r in predictions), default=float("nan"))
    P3 = bool(predictions and worst <= TOL)
    n_pos = int(sum(1 for r in predictions if r["residual"] > 0))

    # ---- P4: is the snapshot a biased fabric estimator? -----------------------
    biases = [abs(p["bias"]["a_xy_snapshot_vs_timeavg"]) for p in fab]
    n_biased = int(sum(1 for b in biases if b >= SNAPSHOT_NOISE))
    P4 = bool(n_biased >= 5)
    worst_bias_a = float(np.max(biases))
    worst_bias_z = float(np.max([abs(p["bias"]["coordination_snapshot_vs_timeavg"])
                                 for p in fab]))
    worst_bias_ov = float(np.max([abs(p["bias"]["overlap_snapshot_vs_timeavg"])
                                  for p in fab]))

    verdicts = {k: ("PASS" if v else "FAIL") for k, v in
                (("P1", P1), ("P2", P2), ("P3", P3), ("P4", P4), ("P5", P5))}

    # how much of the R4 overshoot did the new estimator remove?
    r4 = {}
    r4_path = os.path.join(R4_RUNS, "analysis.json")
    if os.path.exists(r4_path):
        with open(r4_path) as fh:
            r4d = json.load(fh)
        r4 = {"leg_c_worst_residual": r4d.get("leg_c_worst_residual"),
              "leg_b_a_xy_spread": r4d.get("leg_b_a_xy_spread"),
              "leg_b_f_spread": r4d.get("leg_b_f_spread"),
              "leg_b_factored_residual_absmax": r4d.get("leg_b_factored_residual_absmax"),
              "verdicts": r4d.get("verdicts")}

    result = {
        "case": CASE, "round": ROUND,
        "prereg_script_sha256": prereg.get("script_sha256"),
        "reference": PREREG["reference"],
        "motivation": PREREG["motivation"],
        "grid": PREREG["grid"],
        "leg_b_per_phi": per_phi,
        "leg_b_a_xy_bar": a_xy_bar, "leg_b_f_norm_bar": f_bar,
        "leg_b_a_xy_spread": a_xy_spread, "leg_b_f_spread": f_spread,
        "leg_b_a_xy_spread_snapshot_r4_estimator": a_xy_spread_snap,
        "leg_b_f_spread_snapshot_r4_estimator": f_spread_snap,
        "leg_b_factored_residual_absmax": float(np.max(
            [abs(p["timeavg"]["factored_residual"]) for p in fab])),
        "leg_b_mean_r2": mean_r2,
        "leg_b_n_runs": len(fab_all), "leg_b_n_gated": len(fab),
        "leg_b_rejected": rejected,
        "virial_split_residual_absmax": float(np.max(
            [p["timeavg"]["virial_split_residual_max"] for p in fab])),
        "snapshot_bias": {
            "a_xy_absmax": worst_bias_a,
            "coordination_absmax": worst_bias_z,
            "mean_overlap_absmax": worst_bias_ov,
            "n_runs_with_a_xy_bias_above_noise": n_biased,
            "noise_floor": SNAPSHOT_NOISE,
            "per_run": [{"tag": p["tag"], **p["bias"]} for p in fab],
        },
        "leg_c_r2_predictions": predictions,
        "leg_c_worst_residual": worst,
        "leg_c_n_points": len(predictions),
        "leg_c_n_positive_bias": n_pos,
        "leg_c_r4_worst_residual": R4_WORST_RESIDUAL,
        "r4_reference": r4,
        "tolerance": TOL,
        "verdicts": verdicts,
        "coefficient_leg": {
            "status": "MEASURED, no literature comparison",
            "campbell_table": "WITHHELD - paywalled (reference_anchors.json)",
        },
    }
    result["exploratory_amendment"] = exploratory_closure(fab, r2, mean_r2)
    with open(os.path.join(RUNS, "analysis.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    write_tables(result)
    make_figure(result, fab)
    return result


def exploratory_closure(fab: List[Dict[str, Any]], r2: List[Dict[str, Any]],
                        mean_r2: float) -> Dict[str, Any]:
    """EPILOGUE - written after the pre-registered verdicts, they stand unchanged.

    R5falsified R4's suspicion that the *fabric anisotropy* snapshot estimate was
    the culprit (P4 FAIL) but showed the culprit was ``f_norm``: its spread over
    concentration collapses from 41.2 % (snapshot) to 4.3 % (time average).
    ``a_xy``, by contrast, keeps a real 41.9 % concentration dependence, which is
    exactly why the *constant* closure (P1/P3) fails.

    So: let both constants be linear in phi (the form R4 already used in its own
    amendment, here on time-averaged inputs) and test it two ways:
      in-sample  - fit on all 3 concentrations, predict the R2 map;
      LOCO       - leave one concentration out, refit on the other two, predict
                   only the R2 points inside the held-out concentration band.
    LOCO is the honest number: the held-out concentration never enters the fit.
    """
    phi = np.array([p["phi"] for p in fab])
    a_xy = np.array([p["timeavg"]["a_xy"] for p in fab])
    f_n = np.array([p["timeavg"]["f_norm"] for p in fab])
    s_a, i_a = np.polyfit(phi, a_xy, 1)
    s_f, i_f = np.polyfit(phi, f_n, 1)
    phi_grid = sorted(set(float(x) for x in phi))

    def predict(p, a_here, f_here):
        d = diameter_from_kappa(p["kn"], p["kappa"], 1.0, p["gamma_dot"])
        a_norm = (p["phi"] * p["coordination"] * d ** 3 / (2.0 * math.pi * mean_r2)) \
            * a_here * p["mean_overlap_ratio"]
        return a_norm / f_here

    r2_sel = [p for p in r2 if p.get("gates", {}).get("all") and p["phi"] >= PHI_MAP_MIN]
    in_sample = []
    for p in r2_sel:
        a_pred = predict(p, i_a + s_a * p["phi"], i_f + s_f * p["phi"])
        a_meas = coefficient(p)
        in_sample.append({"tag": p["tag"], "phi": p["phi"], "A_meas": a_meas,
                          "A_pred": a_pred, "residual": a_pred / a_meas - 1.0})
    worst_in = max((abs(r["residual"]) for r in in_sample), default=float("nan"))

    loco = []
    for hold in phi_grid:
        train = [q for q in phi_grid if q != hold]
        design = np.array([[1.0, q] for q in train], float)
        ca = np.linalg.solve(design, np.array(
            [a_xy[phi == q].mean() for q in train], float))
        cf = np.linalg.solve(design, np.array(
            [f_n[phi == q].mean() for q in train], float))
        for p in r2_sel:
            if abs(p["phi"] - hold) > 0.02:
                continue
            a_pred = predict(p, ca[0] + ca[1] * p["phi"], cf[0] + cf[1] * p["phi"])
            a_meas = coefficient(p)
            loco.append({"held_out_phi": hold, "tag": p["tag"], "phi": p["phi"],
                         "A_meas": a_meas, "A_pred": a_pred,
                         "residual": a_pred / a_meas - 1.0})
    worst_loco = max((abs(r["residual"]) for r in loco), default=float("nan"))
    return {
        "status": "EXPLORATORY - written after the pre-registered verdicts; "
                  "P1/P3 stay FAIL, nothing is refitted into them",
        "why": "a_xy keeps a real concentration dependence (P1 FAIL) while f_norm's "
               "concentration spread collapses once it is a time average "
               "(41.2% -> 4.3%), so the closure needs a_xy(phi), not a new constant",
        "a_xy_linear_fit": {"intercept": float(i_a), "slope_per_phi": float(s_a),
                            "residual_max_on_9_runs": float(np.max(np.abs(
                                a_xy / (i_a + s_a * phi) - 1.0)))},
        "f_norm_linear_fit": {"intercept": float(i_f), "slope_per_phi": float(s_f),
                              "residual_max_on_9_runs": float(np.max(np.abs(
                                  f_n / (i_f + s_f * phi) - 1.0)))},
        "in_sample": in_sample, "in_sample_worst_residual": worst_in,
        "loco": loco, "loco_worst_residual": worst_loco, "loco_n_points": len(loco),
        "tolerance": TOL,
    }


def write_tables(res: Dict[str, Any]) -> None:
    L = [f"# {CASE} — {ROUND}: tables", "",
         "generated by `fabric_timeavg.py --analyse` from analysis.json "
         "(no hand-typed number)", "",
         "## per-concentration constants", "",
         "| phi | n | a_xy (timeavg) | seed sd | a_xy (snapshot) | f_norm (timeavg) | "
         "f_norm (snapshot) | Z (timeavg) |",
         "|---|---|---|---|---|---|---|---|"]
    for phi, v in res["leg_b_per_phi"].items():
        L.append("| %s | %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.3f |"
                 % (phi, v["n"], v["a_xy_timeavg"], v["a_xy_timeavg_seed_std"],
                    v["a_xy_snapshot"], v["f_norm_timeavg"], v["f_norm_snapshot"],
                    v["coordination_timeavg"]))
    L += ["", "spread(timeavg a_xy) = %.1f%% ; spread(snapshot a_xy) = %.1f%%"
          % (100 * res["leg_b_a_xy_spread"], 100 * res["leg_b_a_xy_spread_snapshot_r4_estimator"]),
          "spread(timeavg f_norm) = %.1f%% ; spread(snapshot f_norm) = %.1f%%"
          % (100 * res["leg_b_f_spread"], 100 * res["leg_b_f_spread_snapshot_r4_estimator"]), "",
          "## out-of-sample prediction of the independent R2 map", "",
          "| tag | phi | Z | mean ov | A_meas | A_pred | residual |", "|---|---|---|---|---|---|---|"]
    for r in res["leg_c_r2_predictions"]:
        L.append("| %s | %.2f | %.3f | %.5f | %.5f | %.5f | %+.1f%% |"
                 % (r["tag"], r["phi"], r["coordination"], r["mean_overlap_ratio"],
                    r["A_meas"], r["A_pred"], 100 * r["residual"]))
    L += ["", "worst residual = %+.1f%% (R4 with the snapshot estimator: %+.1f%%)"
          % (100 * res["leg_c_worst_residual"], 100 * res["leg_c_r4_worst_residual"]), "",
          "## predictor verdicts", "", "| id | verdict |", "|---|---|"]
    for k, v in res["verdicts"].items():
        L.append(f"| {k} | **{v}** |")
    ex = res.get("exploratory_amendment")
    if ex:
        L += ["", "## EPILOGUE (exploratory, verdicts unchanged)", "",
              ex["status"], "",
              "a_xy(phi) = %+.4f %+.4f*phi (residual max on the 9 runs %.1f%%); "
              "f_norm(phi) = %+.4f %+.4f*phi (residual max %.1f%%)"
              % (ex["a_xy_linear_fit"]["intercept"], ex["a_xy_linear_fit"]["slope_per_phi"],
                 100 * ex["a_xy_linear_fit"]["residual_max_on_9_runs"],
                 ex["f_norm_linear_fit"]["intercept"], ex["f_norm_linear_fit"]["slope_per_phi"],
                 100 * ex["f_norm_linear_fit"]["residual_max_on_9_runs"]), "",
              "| test | points | worst residual | inside 15%? |", "|---|---|---|---|",
              "| constant closure (P3, pre-registered) | %d | %+.1f%% | %s |"
              % (res["leg_c_n_points"], 100 * res["leg_c_worst_residual"],
                 "yes" if res["leg_c_worst_residual"] <= TOL else "no"),
              "| phi-linear closure, in sample | %d | %+.1f%% | %s |"
              % (len(ex["in_sample"]), 100 * ex["in_sample_worst_residual"],
                 "yes" if ex["in_sample_worst_residual"] <= TOL else "no"),
              "| phi-linear closure, leave-one-concentration-out | %d | %+.1f%% | %s |"
              % (ex["loco_n_points"], 100 * ex["loco_worst_residual"],
                 "yes" if ex["loco_worst_residual"] <= TOL else "no"), "",
              "| held out | tag | phi | A_meas | A_pred | residual |", "|---|---|---|---|---|---|"]
        for r in ex["loco"]:
            L.append("| %.2f | %s | %.2f | %.5f | %.5f | %+.1f%% |"
                     % (r["held_out_phi"], r["tag"], r["phi"], r["A_meas"], r["A_pred"],
                        100 * r["residual"]))
    with open(os.path.join(RUNS, "tables.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def make_figure(res: Dict[str, Any], fab: List[Dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    ax = axes[0][0]
    phis = sorted(res["leg_b_per_phi"])
    ta = [res["leg_b_per_phi"][p]["a_xy_timeavg"] for p in phis]
    sa = [res["leg_b_per_phi"][p]["a_xy_timeavg_seed_std"] for p in phis]
    sn = [res["leg_b_per_phi"][p]["a_xy_snapshot"] for p in phis]
    ax.errorbar([float(p) for p in phis], ta, yerr=sa, fmt="o-", label="time average",
                color="C0", capsize=3)
    ax.plot([float(p) for p in phis], sn, "s--", label="final snapshot (R4)", color="C3")
    ax.set_xlabel("solid fraction phi")
    ax.set_ylabel(r"$a_{xy}=\langle ov\,n_x n_y\rangle/\langle ov\rangle$")
    ax.set_title("(a) fabric anisotropy: real phi dependence\n"
                 "spread %.1f%% (timeavg) vs %.1f%% (snapshot)"
                 % (100 * res["leg_b_a_xy_spread"],
                    100 * res["leg_b_a_xy_spread_snapshot_r4_estimator"]))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    ta = [res["leg_b_per_phi"][p]["f_norm_timeavg"] for p in phis]
    sa = [res["leg_b_per_phi"][p]["f_norm_timeavg_seed_std"] for p in phis]
    sn = [res["leg_b_per_phi"][p]["f_norm_snapshot"] for p in phis]
    ax.errorbar([float(p) for p in phis], ta, yerr=sa, fmt="o-", color="C0",
                label="time average", capsize=3)
    ax.plot([float(p) for p in phis], sn, "s--", color="C3", label="final snapshot (R4)")
    ax.set_xlabel("solid fraction phi")
    ax.set_ylabel(r"$f_{norm}$ (normal-force share of $\tau_c$)")
    ax.set_title("(b) normal-force share: snapshot bias is here\n"
                 "spread %.1f%% -> %.1f%%"
                 % (100 * res["leg_b_f_spread_snapshot_r4_estimator"],
                    100 * res["leg_b_f_spread"]))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    r = res["leg_c_r2_predictions"]
    ax.axhspan(-15, 15, color="0.85", label="+/-15% ledger band")
    ax.plot([p["phi"] for p in r], [100 * p["residual"] for p in r], "s",
            color="C3", label="constant closure (P3)")
    ex = res.get("exploratory_amendment")
    if ex:
        ax.plot([p["phi"] for p in ex["loco"]], [100 * p["residual"] for p in ex["loco"]],
                "o", mfc="none", color="C0",
                label="phi-linear, leave-one-concentration-out (EPILOGUE)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("solid fraction phi of the predicted R2 point")
    ax.set_ylabel("residual of A_pred vs A_meas [%]")
    ax.set_title("(c) out-of-sample prediction of the independent R2 map")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    tags = [p["tag"].replace("fab_", "") for p in fab]
    bias = [100 * p["bias"]["a_xy_snapshot_vs_timeavg"] for p in fab]
    bias_z = [100 * p["bias"]["coordination_snapshot_vs_timeavg"] for p in fab]
    idx = np.arange(len(tags))
    ax.bar(idx - 0.2, bias, 0.4, color="C0", label="a_xy snapshot bias")
    ax.bar(idx + 0.2, bias_z, 0.4, color="C4", label="Z snapshot bias")
    ax.axhline(100 * res["snapshot_bias"]["noise_floor"], color="k", ls=":",
               label="seed noise floor %.1f%%" % (100 * res["snapshot_bias"]["noise_floor"]))
    ax.axhline(-100 * res["snapshot_bias"]["noise_floor"], color="k", ls=":")
    ax.set_xticks(idx)
    ax.set_xticklabels(tags, rotation=60, fontsize=7)
    ax.set_ylabel("snapshot / time average - 1 [%]")
    ax.set_title("(d) how biased is one snapshot?  a_xy absmax %.1f%%, Z absmax %.1f%%"
                 % (100 * res["snapshot_bias"]["a_xy_absmax"],
                    100 * res["snapshot_bias"]["coordination_absmax"]))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT R5 - steady-window fabric "
                 "statistics (2-D Lees-Edwards, 400 disks)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(RUNS, "fig_fabric_timeavg.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("figure ->", path)


def check() -> None:
    res = json.load(open(os.path.join(RUNS, "analysis.json")))
    assert res["leg_b_n_gated"] >= 7, "fewer than 7 gated fabric runs"
    assert res["virial_split_residual_absmax"] < 1e-9, "normal/tangential virial split broken"
    assert res["leg_b_factored_residual_absmax"] < 1.0
    assert len(res["leg_c_r2_predictions"]) >= 5
    for p in res["leg_c_r2_predictions"]:
        assert math.isfinite(p["residual"])
    print("verdicts:", res["verdicts"])
    print("a_xy spread timeavg %.1f%% vs snapshot %.1f%%"
          % (100 * res["leg_b_a_xy_spread"],
             100 * res["leg_b_a_xy_spread_snapshot_r4_estimator"]))
    print("worst out-of-sample residual %.1f%% (R4 %.1f%%)"
          % (100 * res["leg_c_worst_residual"], 100 * res["leg_c_r4_worst_residual"]))
    ex = res.get("exploratory_amendment") or {}
    if ex:
        assert ex["loco_n_points"] >= 10, "LOCO cross-validation too thin"
        assert all(math.isfinite(r["residual"]) for r in ex["loco"])
        print("LOCO worst residual %.1f%% over %d held-out points"
              % (100 * ex["loco_worst_residual"], ex["loco_n_points"]))
    print("checks OK")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", action="store_true")
    ap.add_argument("--stage", choices=["fabric"])
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if args.prereg:
        stage_prereg()
    if args.stage == "fabric":
        stage_fabric()
    if args.analyse:
        analyse()
    if args.check:
        check()
    if not any((args.prereg, args.stage, args.analyse, args.check)):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())