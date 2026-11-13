"""CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT  --  R4: the elastic-limit *coefficient*.

Rounds 1-3 measured the *exponent* legs of Campbell (2002): the kappa-collapse
(R3 P1), the shear-rate exponent n = 2(1 - beta) (R3 P2/P3) and the concentration
route (R2).  The *coefficient* leg has been carried as WITHHELD since R1 because
the reference tables are paywalled (``reference_anchors.json``).  The coefficient
is not only a literature number, though: the abstract-anchored sentence
"non-inertial quasi-static regime in which the stresses scale purely elastically"
(Campbell 2002 publisher abstract, captured and hashed in
``reference_anchors.json``) is a *statement of structure*

    tau = k_n * F(packing geometry),        independent of gamma_dot
    <=>   A := tau* / kappa = tau d / k_n   -> constant in the elastic limit.

That statement is testable with no reference table at all, and it is the leg this
round measures:

Leg A (asymptote).  A(kappa) over the 3.2-decade R3 ladder: does it *saturate*
    (elastic-only scaling) or keep drifting?  Extrapolated asymptote A_inf and its
    distance to the stiffest gated point, against the ledger's 15 % tolerance.
Leg B (structure).  A is predicted, with no free rate constant, by the packing
    geometry measured on the *contact network*:

        A_norm = (phi Z d^3 / (2 pi <r^2>)) * a_xy * <overlap/d>,   A = A_norm/f_norm

    with Z the coordination number, <r^2> the mean squared radius, a_xy the
    overlap-normalised fabric anisotropy <(delta/d) n_x n_y>/<delta/d> and f_norm
    the fraction of the shear stress carried by normal forces (the tangential
    spring lever arm is excluded, its history is not part of ``final_state``).
    The identity is derived in docs section 2; the *factorisation* that makes it
    predictive (a_xy and f_norm constant) is the falsifiable part.
Leg C (out of sample).  Two geometric constants measured on 9 fabric runs
    (3 concentrations x 3 seeds) predict the total shear stress of the gated
    points of the independent R2 concentration map (different round, different
    runs, same apparatus).

Pre-registration discipline: ``--prereg`` writes the predictions, the thresholds
and the grid *before* any fabric point is run; ``--analyse`` applies the written
rules verbatim.  Nothing is tuned after looking at the results.
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

from tensordem.elastic_limit_shear import (
    ElasticLimitShearCase,
    _min_image,
    simulate_plane_shear,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs", "elastic_coefficient")
R3_RUNS = os.path.join(HERE, "runs", "kappa_collapse")
R2_RUNS = os.path.join(HERE, "runs", "phi_transition")
CASE = "CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT"
ROUND = "R4 (elastic-limit coefficient route)"

# ---------------------------------------------------------------- pre-registered
PHI_GRID = (0.71, 0.76, 0.80)
SEEDS = (20261112, 20261113, 20261114)
BASE = dict(kn=1.0e4, shear_rate=1.0, n_particles=400, strain_total=1.4,
            strain_average_from=0.7, dt_factor=0.004)
TOP_KAPPA = 2.3e4          # top 1.4 decades of the R3 ladder (3 distinct kappa)
PHI_MAP_MIN = 0.71         # R2 points used out of sample
TOL = 0.15                 # ledger tolerance, unmodified
NOISE = 0.0625             # seed noise floor at phi=0.80 (R2 P5, reproduced R3 P5)

PREREG = {
    "case": CASE,
    "round": ROUND,
    "reference": "Campbell 2002 JFM 465:261 (abstract hashed in reference_anchors.json)",
    "claim_under_test": (
        "'non-inertial quasi-static regime in which the stresses scale purely "
        "elastically' <=> A := tau*/kappa = tau d/k_n saturates to a constant"
    ),
    "grid": {"phi": list(PHI_GRID), "seeds": list(SEEDS), **BASE},
    "predictions": [
        {"id": "P1", "statement": "A saturates: spread of A over kappa >= 2.3e4 is < 15%",
         "rule": "max(A)/min(A) - 1 <= 0.15 for gated R3 points with kappa >= 2.3e4",
         "threshold": 0.15},
        {"id": "P2", "statement": "the extrapolated A_inf lies within 15% of the stiffest gated point",
         "rule": "|A_inf - A(kappa_max)|/A(kappa_max) <= 0.15, fit A = A_inf + c kappa^-g",
         "threshold": 0.15},
        {"id": "P3", "statement": "the factored structural identity reproduces the normal-force A",
         "rule": "|A_pred_factored/A_normal - 1| <= 0.15 on every fabric run",
         "threshold": 0.15},
        {"id": "P4", "statement": "the two geometric constants are concentration independent",
         "rule": "relative spread of a_xy and of f_norm over the 3 concentrations <= 0.15",
         "threshold": 0.15},
        {"id": "P5", "statement": "those constants predict the R2 map's total shear stress out of sample",
         "rule": "|A_pred/A_meas - 1| <= 0.15 on every gated R2 point with phi >= 0.71",
         "threshold": 0.15},
    ],
    "honesty_rules": [
        "a rejected (ungated) point is reported, never re-run until it passes",
        "no coefficient is compared to Campbell's table: it stays WITHHELD (paywalled)",
        "if P3-P5 fail the factorisation is reported as FAIL and not refitted",
    ],
}


# ------------------------------------------------------------------- utilities
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def diameter_from_kappa(kn: float, kappa: float, rho: float, gamma_dot: float) -> float:
    """Recover the length unit from the case's own definition of kappa.

    kappa = k_n/(rho d^3 gamma_dot^2)  ->  d = (k_n/(rho kappa gamma_dot^2))^(1/3).
    Used instead of hard-coding d = 1.2 so the analysis cannot silently assume a
    geometry the run did not have.
    """
    return (kn / (rho * kappa * gamma_dot ** 2)) ** (1.0 / 3.0)


def contact_geometry(x, y, r, lx, ly, s, kn) -> Dict[str, float]:
    """Touching-pair contact set of one configuration, with virial and fabric.

    Uses the module's own ``_min_image`` (the certified Lees-Edwards convention) so
    the reconstruction cannot disagree with the run about periodic images, but
    enumerates the pairs independently of the neighbour list: an independent check
    of the contact set, not a replay of the solver's bookkeeping.

    Only the **normal** force virial is reconstructed: the tangential spring
    history is not part of ``final_state``, so the friction share is measured as a
    residual (``f_norm``) and reported, never assumed.
    """
    n = x.size
    ii, jj = np.triu_indices(n, 1)
    ddx, ddy = _min_image(x, y, ii, jj, lx, ly, s)
    dist = np.hypot(ddx, ddy)
    touch = dist < (r[ii] + r[jj])
    ii, jj, ddx, ddy, dist = ii[touch], jj[touch], ddx[touch], ddy[touch], dist[touch]
    if dist.size == 0:
        raise ValueError("no contacts reconstructed")
    dc = r[ii] + r[jj]
    delta = dc - dist
    nx, ny = ddx / dist, ddy / dist
    # |r_ij| is the centre distance dist = (r_i+r_j) - delta, i.e. the separation
    # vector itself: shortening it again by delta (a bug the unit test caught) would
    # bias the virial low by ~2 <delta>/d
    branch = 1.0
    fx, fy = -kn * delta * nx, -kn * delta * ny  # force on i from j (compressive)
    area = lx * ly
    overlap_ratio = delta / dc
    return {
        "n_contacts": int(dist.size),
        "coordination_snapshot": 2.0 * dist.size / n,
        "mean_overlap_ratio_snapshot": float(np.mean(overlap_ratio)),
        "max_overlap_ratio_snapshot": float(np.max(overlap_ratio)),
        # the contact normal here points i -> j, while the virial sum below is
        # written with the force on i (j -> i); the physical fabric anisotropy is
        # therefore the *negative* of the raw mean, and is reported already flipped
        "fabric_xy": -float(np.mean(nx * ny)),
        "fabric_xx_minus_yy": -float(np.mean(nx * nx - ny * ny)),
        "overlap_weighted_fabric_xy": -float(np.mean(overlap_ratio * nx * ny)),
        "sigma_yx_normal": float(np.sum(fx * branch * ddy) / area),
        "sigma_xy_normal": float(np.sum(fy * branch * ddx) / area),
        "pressure_normal": float(-np.sum(fx * branch * ddx + fy * branch * ddy) / (2.0 * area)),
        "mean_r2": float(np.mean(r ** 2)),
        "mean_separation": float(np.mean(dist)),
        "mean_contact_diameter": float(np.mean(dc)),
        "branch_shortening": float(np.mean(delta / dist)),
        "area": area,
    }


# ---------------------------------------------------------------------- stages
def stage_prereg() -> Dict[str, Any]:
    os.makedirs(RUNS, exist_ok=True)
    payload = dict(PREREG)
    payload["written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload["script_sha256"] = _sha256(__file__)
    payload["r3_points"] = sorted(f for f in os.listdir(R3_RUNS) if f.endswith(".json"))
    path = os.path.join(RUNS, "prereg.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"prereg -> {path}  script sha256={payload['script_sha256'][:16]}")
    return payload


def run_point(phi: float, seed: int, verbose: bool = True) -> Dict[str, Any]:
    case = ElasticLimitShearCase(solid_fraction=phi, seed=seed, **BASE)
    t0 = time.time()
    out = simulate_plane_shear(case, verbose=False)
    fs = out["final_state"]
    geo = contact_geometry(fs["x"], fs["y"], case.radii, case.lx, case.ly,
                           float(fs["s"]), case.kn)
    d = case.d_mean
    a_total = out["tau_contact"] * d / case.kn       # the kappa-free coefficient
    # sigma_yx_normal is built with the force on i, i.e. the same convention the run
    # uses for tau (verified: it reproduces tau_contact to a few per mille at the
    # snapshots); it is therefore added, not subtracted
    a_norm = geo["sigma_yx_normal"] * d / case.kn
    a_pred = (phi * geo["coordination_snapshot"] * d ** 3
              / (2.0 * math.pi * geo["mean_r2"])) * geo["overlap_weighted_fabric_xy"]
    point = {
        "tag": f"fab_phi{phi:.2f}_s{seed}", "role": "fabric",
        "phi": phi, "seed": seed, "kn": case.kn, "gamma_dot": case.shear_rate,
        "d_mean": d, "kappa": case.kappa,
        "tau": out["tau"], "tau_contact": out["tau_contact"], "pressure": out["pressure"],
        "tau_star": out["tau_star"], "mu_eff": out["mu_eff"], "I": out["I"],
        "coordination": out["coordination"],
        "mean_overlap_ratio": out["mean_overlap_ratio"],
        "max_overlap_ratio": out["max_overlap_ratio"],
        "kinetic_stress_share": out["kinetic_stress_share"],
        "steady_drift_tau": out["steady_drift_tau"],
        "energy_defect": out["energy"]["relative_defect"],
        "gates": out["gates"], "runtime_s": time.time() - t0,
        "geometry": geo,
        "A_total": a_total, "A_normal": a_norm,
        "f_norm": a_norm / a_total if a_total else float("nan"),
        "a_xy": geo["overlap_weighted_fabric_xy"] / geo["mean_overlap_ratio_snapshot"],
        "A_pred_factored": a_pred,
        "factored_residual": (a_pred / a_norm - 1.0) if a_norm else float("nan"),
        "snapshot_vs_timeavg_residual": (geo["coordination_snapshot"] / out["coordination"] - 1.0),
    }
    if verbose:
        print("  %-22s phi=%.2f kappa=%.3g A=%.5f A_norm=%.5f f_norm=%.3f a_xy=%.3f "
              "P3res=%+.1f%% gates=%s (%.1fs)"
              % (point["tag"], phi, case.kappa, a_total, a_norm, point["f_norm"],
                 point["a_xy"], 100 * point["factored_residual"],
                 point["gates"]["all"], point["runtime_s"]))
    with open(os.path.join(RUNS, point["tag"] + ".json"), "w") as fh:
        json.dump(point, fh, indent=1)
    return point


def stage_fabric() -> List[Dict[str, Any]]:
    os.makedirs(RUNS, exist_ok=True)
    print(f"fabric grid: {len(PHI_GRID)} phi x {len(SEEDS)} seeds "
          f"({int(BASE['n_particles'])} disks, strain {BASE['strain_total']}, sequential)")
    pts = []
    for phi in PHI_GRID:
        for seed in SEEDS:
            pts.append(run_point(phi, seed))
    return pts


# -------------------------------------------------------------------- analysis
def load_points(directory: str, prefix: str = "") -> List[Dict[str, Any]]:
    out = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json") or name in ("analysis.json", "prereg.json"):
            continue
        if prefix and not name.startswith(prefix):
            continue
        with open(os.path.join(directory, name)) as fh:
            d = json.load(fh)
        if "kappa" in d and "tau" in d:
            out.append(d)
    return out


def coefficient(point: Dict[str, Any]) -> float:
    """A = tau*/kappa = tau d / k_n, with d recovered from the point's own kappa."""
    d = diameter_from_kappa(point["kn"], point["kappa"], 1.0, point["gamma_dot"])
    return point["tau"] * d / point["kn"]


def fit_saturation(kappas: Sequence[float], values: Sequence[float],
                   sigmas: Sequence[float]) -> Dict[str, Any]:
    """Weighted fit of A = A_inf + c kappa^-g by a grid over g (no scipy needed)."""
    k = np.asarray(kappas, float)
    a = np.asarray(values, float)
    s = np.asarray(sigmas, float)
    w = 1.0 / np.maximum(s, 1e-12) ** 2
    sw = np.sqrt(w)
    best = None
    for g in np.linspace(0.05, 2.0, 391):
        design = np.column_stack([np.ones_like(k), k ** (-g)])
        coef, *_ = np.linalg.lstsq(design * sw[:, None], a * sw, rcond=None)
        resid = (design @ coef - a) * sw
        chi2 = float(resid @ resid)
        if best is None or chi2 < best[0]:
            best = (chi2, float(g), coef)
    chi2, g, coef = best
    design = np.column_stack([np.ones_like(k), k ** (-g)])
    cov = np.linalg.inv((design * sw[:, None]).T @ (design * sw[:, None]))
    dof = max(len(k) - 2, 1)
    return {"A_inf": float(coef[0]), "c": float(coef[1]), "g": g,
            "sigma_A_inf": float(math.sqrt(cov[0, 0])),
            "chi2": chi2, "dof": dof, "reduced_chi2": chi2 / dof,
            "weights": "sigma = 6.25%/sqrt(3 seeds) of each A"}


def analyse() -> Dict[str, Any]:
    with open(os.path.join(RUNS, "prereg.json")) as fh:
        prereg = json.load(fh)
    fab_all = load_points(RUNS, "fab_")
    if len(fab_all) < 6:
        raise SystemExit("fabric points missing: run --stage fabric first")
    # the case's own gates decide validity *before* any physics is fitted; a point
    # that fails them is reported as rejected, never re-run until it passes
    fab = [p for p in fab_all if p.get("gates", {}).get("all")]
    rejected = [{"tag": p["tag"], "failed": sorted(k for k, v in p["gates"].items()
                                                   if v is False and k != "all"),
                 "coordination": p["coordination"], "phi": p["phi"], "seed": p["seed"],
                 "steady_drift_tau": p["steady_drift_tau"]}
                for p in fab_all if not p.get("gates", {}).get("all")]
    r3 = load_points(R3_RUNS)
    r2 = load_points(R2_RUNS, "map_phi")

    # ---------------- leg A: the asymptote on the R3 kappa ladder
    # the ladder is the pre-registered R3 *grid* (role == "grid"); the seed twins are
    # not extra grid points, they are the empirical noise estimate used below.
    ladder = []
    for p in r3:
        if not p.get("gates", {}).get("all") or p.get("role") != "grid":
            continue
        ladder.append({"tag": p["tag"], "kappa": p["kappa"], "A": coefficient(p),
                       "phi": p.get("phi"), "coordination": p.get("coordination"),
                       "mean_overlap_ratio": p.get("mean_overlap_ratio")})
    ladder.sort(key=lambda r: r["kappa"])
    kk = [r["kappa"] for r in ladder]
    aa = [r["A"] for r in ladder]
    twins = [p for p in r3 if p.get("role") == "noise_twin" and p.get("gates", {}).get("all")]
    twin_group = [p for p in r3 if p.get("gates", {}).get("all")
                  and abs(p["kn"] - 4.0e4) < 1.0 and abs(p["gamma_dot"] - 2.0) < 1e-9]
    twin_A = [coefficient(p) for p in twin_group]
    empirical_noise = (float(np.std(twin_A) / np.mean(twin_A)) if len(twin_A) > 1 else float("nan"))
    sigma_rel = empirical_noise if np.isfinite(empirical_noise) and empirical_noise > 0 else NOISE
    ss = [sigma_rel * r["A"] / math.sqrt(len(twin_A) if len(twin_A) > 1 else 1) for r in ladder]
    fit = fit_saturation(kk, aa, ss)
    top = [r for r in ladder if r["kappa"] >= TOP_KAPPA]
    top_spread = (max(r["A"] for r in top) / min(r["A"] for r in top) - 1.0) if len(top) > 1 else float("nan")
    a_stiffest = max(ladder, key=lambda r: r["kappa"])["A"]
    P1 = bool(np.isfinite(top_spread) and top_spread <= TOL)
    P2 = bool(abs(fit["A_inf"] - a_stiffest) / a_stiffest <= TOL)

    # ---------------- leg B: structural closure of A
    per_phi: Dict[float, List[Dict[str, Any]]] = {}
    for p in fab:
        per_phi.setdefault(round(p["phi"], 3), []).append(p)

    def spread(vals: Sequence[float]) -> float:
        vals = [v for v in vals if np.isfinite(v)]
        return (max(vals) / min(vals) - 1.0) if len(vals) > 1 else float("nan")

    phi_summary = {}
    for phi, pts in sorted(per_phi.items()):
        phi_summary[f"{phi:.3f}"] = {
            "n_runs": len(pts),
            "A_total_mean": float(np.mean([p["A_total"] for p in pts])),
            "A_total_rel_std": float(np.std([p["A_total"] for p in pts]) /
                                     np.mean([p["A_total"] for p in pts])),
            "A_normal_mean": float(np.mean([p["A_normal"] for p in pts])),
            "f_norm_mean": float(np.mean([p["f_norm"] for p in pts])),
            "f_norm_rel_std": float(np.std([p["f_norm"] for p in pts]) /
                                    np.mean([p["f_norm"] for p in pts])),
            "a_xy_mean": float(np.mean([p["a_xy"] for p in pts])),
            "a_xy_rel_std": float(np.std([p["a_xy"] for p in pts]) /
                                  np.mean([p["a_xy"] for p in pts])),
            "fabric_xy_mean": float(np.mean([p["geometry"]["fabric_xy"] for p in pts])),
            "coordination_snapshot_mean": float(np.mean(
                [p["geometry"]["coordination_snapshot"] for p in pts])),
            "coordination_run_mean": float(np.mean([p["coordination"] for p in pts])),
            "overlap_snapshot_mean": float(np.mean(
                [p["geometry"]["mean_overlap_ratio_snapshot"] for p in pts])),
            "overlap_run_mean": float(np.mean([p["mean_overlap_ratio"] for p in pts])),
            "factored_residual_absmax": float(np.max(
                [abs(p["factored_residual"]) for p in pts])),
            "all_gated": bool(all(p["gates"]["all"] for p in pts)),
        }
    a_xy_bar = float(np.mean([v["a_xy_mean"] for v in phi_summary.values()]))
    f_bar = float(np.mean([v["f_norm_mean"] for v in phi_summary.values()]))
    a_xy_spread = spread([v["a_xy_mean"] for v in phi_summary.values()])
    f_spread = spread([v["f_norm_mean"] for v in phi_summary.values()])
    P3 = bool(all(abs(p["factored_residual"]) <= TOL for p in fab))
    P4 = bool(a_xy_spread <= TOL and f_spread <= TOL)

    # ---------------- leg C: out-of-sample prediction of the R2 concentration map
    mean_r2 = float(np.mean([p["geometry"]["mean_r2"] for p in fab]))
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
    P5 = bool(predictions and worst <= TOL)

    verdicts = {k: ("PASS" if v else "FAIL") for k, v in
                (("P1", P1), ("P2", P2), ("P3", P3), ("P4", P4), ("P5", P5))}

    # ---------------- amendments: exploratory only, the verdicts above stand as-is
    phis_meas = np.array(sorted(float(k) for k in phi_summary))
    a_xy_v = np.array([phi_summary["%.3f" % q]["a_xy_mean"] for q in phis_meas])
    f_v = np.array([phi_summary["%.3f" % q]["f_norm_mean"] for q in phis_meas])
    slope_axy, icept_axy = np.polyfit(phis_meas, a_xy_v, 1)
    slope_f, icept_f = np.polyfit(phis_meas, f_v, 1)
    a_xy_fit_resid = a_xy_v / (icept_axy + slope_axy * phis_meas) - 1.0
    exploratory = []
    for r in predictions:
        a_xy_here = icept_axy + slope_axy * r["phi"]
        f_here = icept_f + slope_f * r["phi"]
        d = diameter_from_kappa(1.0e4, 5787.037, 1.0, 1.0)
        a_pred = ((r["phi"] * r["coordination"] * d ** 3 / (2.0 * math.pi * mean_r2))
                  * a_xy_here * r["mean_overlap_ratio"] / f_here)
        exploratory.append({"tag": r["tag"], "phi": r["phi"], "A_meas": r["A_meas"],
                            "A_pred_phi_dependent": a_pred,
                            "residual_phi_dependent": a_pred / r["A_meas"] - 1.0})
    worst_explor = max((abs(r["residual_phi_dependent"]) for r in exploratory), default=float("nan"))
    amendments = {
        "status": "EXPLORATORY - written after the pre-registered verdicts; does not "
                  "replace them (P4 and P5 stay FAIL)",
        "why_P4_failed": ("a_xy is not a concentration-independent constant: it falls "
                          "from %.3f to %.3f as phi rises %.2f -> %.2f (spread %.1f%%)"
                          % (a_xy_v[0], a_xy_v[-1], phis_meas[0], phis_meas[-1],
                             100 * a_xy_spread)),
        "why_P5_failed": ("with a single a_xy the prediction over-shoots the dense end, "
                          "where the predicted a_xy is too large by the same %.0f%%"
                          % (100 * a_xy_spread)),
        "a_xy_linear_fit": {"a0": float(icept_axy), "a1_per_phi": float(slope_axy),
                            "residual_max": float(np.max(np.abs(a_xy_fit_resid)))},
        "f_norm_linear_fit": {"f0": float(icept_f), "f1_per_phi": float(slope_f)},
        "phi_dependent_closure": exploratory,
        "phi_dependent_worst_residual": worst_explor,
        "interpretation": ("the virial identity (P3) holds; what fails is the assumption "
                           "that the fabric anisotropy is a material constant. With "
                           "a_xy(phi) measured on three concentrations the closure "
                           "predicts the independent R2 map within %.1f%%."
                           % (100 * worst_explor)),
    }

    result = {
        "case": CASE, "round": ROUND,
        "prereg_script_sha256": prereg.get("script_sha256"),
        "reference": prereg["reference"],
        "claim_under_test": prereg["claim_under_test"],
        "leg_a_ladder": ladder,
        "leg_a_top_kappa_cut": TOP_KAPPA,
        "leg_a_top_spread": top_spread,
        "leg_a_fit": fit,
        "leg_a_A_stiffest": a_stiffest,
        "leg_a_A_inf_vs_stiffest_rel": abs(fit["A_inf"] - a_stiffest) / a_stiffest,
        "leg_b_per_phi": phi_summary,
        "leg_b_a_xy_bar": a_xy_bar, "leg_b_f_norm_bar": f_bar,
        "leg_b_a_xy_spread": a_xy_spread, "leg_b_f_spread": f_spread,
        "leg_b_factored_residual_absmax": float(np.max(
            [abs(p["factored_residual"]) for p in fab])),
        "leg_b_mean_r2": mean_r2,
        "leg_b_n_runs": len(fab_all), "leg_b_n_gated": len(fab),
        "leg_b_rejected": rejected,
        "leg_b_branch_shortening_mean": float(np.mean(
            [p["geometry"]["branch_shortening"] for p in fab])),
        "leg_b_snapshot_vs_timeavg_absmax": float(np.max(
            [abs(p["snapshot_vs_timeavg_residual"]) for p in fab])),
        "leg_c_r2_predictions": predictions,
        "leg_c_worst_residual": worst,
        "leg_c_n_points": len(predictions),
        "seed_noise_floor": NOISE,
        "leg_a_noise_twin_A": twin_A,
        "leg_a_empirical_noise_rel": empirical_noise,
        "leg_a_sigma_used_rel": sigma_rel,
        "tolerance": TOL,
        "verdicts": verdicts,
        "amendments": amendments,
        "coefficient_leg": {
            "status": "MEASURED, no literature comparison",
            "A_inf": fit["A_inf"], "sigma_A_inf": fit["sigma_A_inf"],
            "campbell_table": "WITHHELD - paywalled (reference_anchors.json)",
            "why": ("the ledger's coefficient tolerance needs a reference; Campbell's "
                    "table is not accessible, so the coefficient is reported as a "
                    "measured asymptote plus a structural closure, not as an "
                    "agreement with Campbell"),
        },
    }
    with open(os.path.join(RUNS, "analysis.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    write_tables(result)
    make_figure(result, fab)
    return result


def write_tables(res: Dict[str, Any]) -> None:
    L = [f"# {CASE} — {ROUND}: tables", "",
         "generated by `ell_coefficient.py --analyse` from analysis.json (no hand-typed number)", "",
         "## Leg A — the coefficient A = tau*/kappa on the R3 kappa ladder", "",
         "| tag | kappa | A | phi | Z | mean ov/d |", "|---|---|---|---|---|---|"]
    for r in res["leg_a_ladder"]:
        L.append("| %s | %.3e | %.5f | %s | %.3f | %.4f |"
                 % (r["tag"], r["kappa"], r["A"], r["phi"], r["coordination"],
                    r["mean_overlap_ratio"]))
    f = res["leg_a_fit"]
    L += ["", "Saturation fit `A = A_inf + c kappa^-g`:",
          "",
          "| quantity | value |", "|---|---|",
          "| A_inf | %.5f |" % f["A_inf"],
          "| sigma(A_inf) | %.5f |" % f["sigma_A_inf"],
          "| c | %.4g |" % f["c"],
          "| g | %.3f |" % f["g"],
          "| reduced chi2 (dof %d) | %.2f |" % (f["dof"], f["reduced_chi2"]),
          "| A at stiffest gated point | %.5f |" % res["leg_a_A_stiffest"],
          "| A_inf vs stiffest | %.1f%% |" % (100 * res["leg_a_A_inf_vs_stiffest_rel"]),
          "| spread of A, kappa >= %.1e | %.1f%% (tolerance %.0f%%) |"
          % (res["leg_a_top_kappa_cut"], 100 * res["leg_a_top_spread"],
             100 * res["tolerance"]),
          "", "## Leg B — structural closure on the contact network", "",
          "| phi | runs | A total | A normal | f_norm | a_xy | factored residual (max) |",
          "|---|---|---|---|---|---|---|"]
    for phi, v in res["leg_b_per_phi"].items():
        L.append("| %s | %d | %.5f | %.5f | %.3f | %.3f | %+.1f%% |"
                 % (phi, v["n_runs"], v["A_total_mean"], v["A_normal_mean"],
                    v["f_norm_mean"], v["a_xy_mean"], 100 * v["factored_residual_absmax"]))
    L += ["", "`a_xy` spread over concentrations: %.1f%%, `f_norm` spread %.1f%%."
          % (100 * res["leg_b_a_xy_spread"], 100 * res["leg_b_f_spread"]),
          "", "## Leg C — out-of-sample prediction of the R2 concentration map", "",
          "| tag | phi | Z | mean ov/d | A measured | A predicted | residual |",
          "|---|---|---|---|---|---|---|"]
    for r in res["leg_c_r2_predictions"]:
        L.append("| %s | %.2f | %.3f | %.4f | %.5f | %.5f | %+.1f%% |"
                 % (r["tag"], r["phi"], r["coordination"], r["mean_overlap_ratio"],
                    r["A_meas"], r["A_pred"], 100 * r["residual"]))
    L += ["", "## Verdicts (pre-registered rules applied verbatim)", "",
          "| prediction | verdict |", "|---|---|"]
    for k, v in res["verdicts"].items():
        L.append("| %s | %s |" % (k, v))
    L += ["", "## Amendments (exploratory, not pre-registered)", "",
          "- %s" % res["amendments"]["status"],
          "- P4 failed because: %s" % res["amendments"]["why_P4_failed"],
          "- P5 failed because: %s" % res["amendments"]["why_P5_failed"],
          "- a_xy(phi) linear fit: %.4f %+.4f phi (max residual %.1f%%)"
          % (res["amendments"]["a_xy_linear_fit"]["a0"],
             res["amendments"]["a_xy_linear_fit"]["a1_per_phi"],
             100 * res["amendments"]["a_xy_linear_fit"]["residual_max"]),
          "- with a_xy(phi) the closure predicts the R2 map to %.1f%% (worst)"
          % (100 * res["amendments"]["phi_dependent_worst_residual"]),
          "", "| tag | phi | A measured | A predicted (a_xy(phi)) | residual |",
          "|---|---|---|---|---|"]
    for r in res["amendments"]["phi_dependent_closure"]:
        L.append("| %s | %.2f | %.5f | %.5f | %+.1f%% |"
                 % (r["tag"], r["phi"], r["A_meas"], r["A_pred_phi_dependent"],
                    100 * r["residual_phi_dependent"]))
    L += ["", "## Coefficient leg", "",
          "- status: %s" % res["coefficient_leg"]["status"],
          "- A_inf = %.5f +/- %.5f (code units)" % (res["coefficient_leg"]["A_inf"],
                                                    res["coefficient_leg"]["sigma_A_inf"]),
          "- Campbell's table: %s" % res["coefficient_leg"]["campbell_table"], ""]
    with open(os.path.join(RUNS, "tables.md"), "w") as fh:
        fh.write("\n".join(L))


def make_figure(res: Dict[str, Any], fab: List[Dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0][0]
    lad = res["leg_a_ladder"]
    ax.semilogx([r["kappa"] for r in lad], [r["A"] for r in lad], "o-", label="measured A")
    f = res["leg_a_fit"]
    kg = np.logspace(math.log10(min(r["kappa"] for r in lad)),
                     math.log10(max(r["kappa"] for r in lad)) + 0.3, 200)
    ax.semilogx(kg, f["A_inf"] + f["c"] * kg ** (-f["g"]), "--",
                label=f"A_inf={f['A_inf']:.5f} + c k^-{f['g']:.2f}")
    ax.axhline(res["leg_a_A_stiffest"] * (1 - res["tolerance"]), color="0.6", lw=0.8, ls=":")
    ax.axhline(res["leg_a_A_stiffest"] * (1 + res["tolerance"]), color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(r"$\kappa = k_n/(\rho d^3\dot\gamma^2)$")
    ax.set_ylabel(r"$A=\tau^*/\kappa=\tau d/k_n$")
    ax.set_title("(a) coefficient saturation (leg A)")
    ax.legend(fontsize=8)

    ax = axes[0][1]
    phis = sorted(res["leg_b_per_phi"].items())
    ax.plot([float(k) for k, _ in phis], [v["A_normal_mean"] for _, v in phis], "s-", label="A normal-force")
    ax.plot([float(k) for k, _ in phis], [v["A_total_mean"] for _, v in phis], "o-", label="A total")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel("A")
    ax.set_title("(b) structural closure vs concentration (leg B)")
    ax.legend(fontsize=8)

    ax = axes[1][0]
    for p in fab:
        ax.plot(p["A_normal"], p["A_pred_factored"], "o", alpha=0.8)
    lims = [0.9 * min(p["A_normal"] for p in fab), 1.1 * max(p["A_normal"] for p in fab)]
    ax.plot(lims, lims, "k--", lw=1)
    ax.plot(lims, [x * (1 + res["tolerance"]) for x in lims], color="0.6", ls=":", lw=0.8)
    ax.plot(lims, [x * (1 - res["tolerance"]) for x in lims], color="0.6", ls=":", lw=0.8)
    ax.set_xlabel("A normal-force (measured)")
    ax.set_ylabel("A factored prediction")
    ax.set_title("(c) factored identity (leg B, P3)")

    ax = axes[1][1]
    preds = res["leg_c_r2_predictions"]
    ax.plot([p["phi"] for p in preds], [p["A_meas"] for p in preds], "o-", label="R2 measured A")
    ax.plot([p["phi"] for p in preds], [p["A_pred"] for p in preds], "x--",
            label="prediction (2 constants from 9 runs)")
    ex = res["amendments"]["phi_dependent_closure"]
    ax.plot([p["phi"] for p in ex], [p["A_pred_phi_dependent"] for p in ex], "^:",
            color="tab:green", label="exploratory a_xy(phi) closure")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel("A total")
    ax.set_title("(d) out-of-sample prediction of the R2 map (leg C, P5)")
    ax.legend(fontsize=8)
    fig.suptitle(f"{CASE} — R4: elastic-limit coefficient "
                 f"A_inf={f['A_inf']:.5f}±{f['sigma_A_inf']:.5f}")
    fig.tight_layout()
    fig.savefig(os.path.join(RUNS, "fig_elastic_coefficient.png"), dpi=130)
    plt.close(fig)


def check() -> None:
    """Reproduce the shipped analysis.json from the shipped point files."""
    with open(os.path.join(RUNS, "analysis.json")) as fh:
        shipped = json.load(fh)
    fresh = analyse()
    keys = ["leg_a_top_spread", "leg_a_A_stiffest", "leg_b_a_xy_bar", "leg_b_f_norm_bar",
            "leg_c_worst_residual", "leg_b_factored_residual_absmax"]
    for k in keys:
        if abs(fresh[k] - shipped[k]) > 1e-9:
            raise SystemExit(f"analysis not reproducible: {k} {fresh[k]} != {shipped[k]}")
    if fresh["verdicts"] != shipped["verdicts"]:
        raise SystemExit("verdicts changed on re-analysis")
    if abs(fresh["leg_a_fit"]["A_inf"] - shipped["leg_a_fit"]["A_inf"]) > 1e-9:
        raise SystemExit("A_inf not reproducible")
    print("analysis reproducible; verdicts", shipped["verdicts"])


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
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
        res = analyse()
        print("verdicts:", res["verdicts"], "A_inf=%.5f" % res["leg_a_fit"]["A_inf"])
    if args.check:
        check()
    if not any([args.prereg, args.stage, args.analyse, args.check]):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())