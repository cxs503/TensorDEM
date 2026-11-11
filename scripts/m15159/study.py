"""GAP-DEM-FRICTION (m15159 retry): the direct-shear measurement study.

This module is the single post-processing/measurement path used by both the
distributed sweep (:mod:`scripts.m15159.sweep_shard`) and the single-process
entry point (:mod:`scripts.m15159.run_gap_dem_friction`).  It adds three things
the m14960 delivery lacked (see ``docs/audit/M14960_CROSS_AUDIT.md`` §7):

1. **Steady-state gate on the headline**: every level is reported with the
   m14840 stationarity leg *on* (``stationarity_tol = 0.15``); a level whose
   traction is still drifting is flaged ``valid=False`` and no mu is quoted.
2. **Degenerate interface sliding excluded by machine rule** (C7 / audit item 5):
   ``mu/mu_p in [0.98, 1.02]`` means the platen slid on the first layer and the
   bulk friction was never mobilised -> ``degenerate=True``.
3. **Compact, committed run artifacts**: level summary + a 201-point
   down-sampled shear history per run, so the gate can be re-checked from the
   committed JSON alone (full histories stay on the compute node).

Reference anchor (external): quasi-static effective friction of dense packed
spheres ``mu_s = 0.38`` with band ``0.36-0.40`` -- GDR MiDi, Eur. Phys. J. E 14,
341 (2004); Jop, Forterre & Pouliquen, Nature 441, 727 (2006).  Acceptance:
|mu_macro/ref - 1| < 0.10 (mission criterion "friction vs experiment < 10%").
The anchor is *not* tuned to the DEM micro friction: the material dependence is
reported as a separate, independent axis (mu_p = 0.3 / 0.5 / 0.7).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from tensordem.shear_cell import (
    ShearCellCase,
    prepare_bed,
    simulate_shear_cell,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO, "artifacts", "gap_dem_friction_m15159")

GEOM = {
    "small": dict(box_width=0.080, box_depth=0.032, fill_height=0.050),
    "medium": dict(box_width=0.080, box_depth=0.064, fill_height=0.080),
    "large": dict(box_width=0.160, box_depth=0.080, fill_height=0.100),
}

REF_MU = 0.38
REF_BAND = (0.36, 0.40)
TOLERANCE = 0.10
STATIONARITY_TOL = 0.15
DEGENERATE_BAND = (0.98, 1.02)   # mu/mu_p: interface sliding, not bulk friction


def _downsample(xs, ys, n=201):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size == 0:
        return [], []
    if xs.size <= n:
        idx = np.arange(xs.size)
    else:
        idx = np.unique(np.linspace(0, xs.size - 1, n).round().astype(int))
    return [float(v) for v in xs[idx]], [float(v) for v in ys[idx]]


def classify(mu, mu_p, drift_rel, stationarity_tol=STATIONARITY_TOL):
    """Machine rule: is ``mu`` a bulk-friction measurement?

    Returns ``(usable, degenerate, reasons)``.  ``mu`` is usable only when it is
    positive, the traction is steady over the averaging window and the platen did
    not simply slide on the top layer (``mu`` == ``mu_p`` to within 2%).
    """
    reasons = []
    degenerate = False
    if mu is None or mu != mu:      # nan
        return False, False, ["mu undefined (NaN)"]
    if drift_rel == drift_rel and abs(drift_rel) > stationarity_tol:
        reasons.append("traction not steady: |drift_rel| = %.3f > %.3f"
                       % (abs(drift_rel), stationarity_tol))
    ratio = mu / mu_p if mu_p else float("nan")
    if DEGENERATE_BAND[0] <= ratio <= DEGENERATE_BAND[1]:
        degenerate = True
        reasons.append("degenerate interface sliding: mu/mu_p = %.3f in [%.2f, %.2f]"
                       % (ratio, *DEGENERATE_BAND))
    if mu <= 0.0:
        reasons.append("non-positive mu")
    return (not reasons), degenerate, reasons


def run_point(tag, mu=0.5, gamma=1.0, sigma=4000.0, geom="medium", tol=STATIONARITY_TOL,
              seed=6107, mu_rolling=0.05, consol=0.04, write=True):
    """One (mu_p, gamma, sigma_n) point: prepare bed -> shear -> compact record."""
    import time
    case = ShearCellCase(mu=mu, mu_rolling=mu_rolling, seed=seed, consol_time=consol,
                         shear_strain_target=gamma, normal_stress=float(sigma),
                         stationarity_tol=tol, diagnostics=False, **GEOM[geom])
    t0 = time.time()
    bed = prepare_bed(case)
    t_bed = time.time() - t0
    t1 = time.time()
    out = simulate_shear_cell(case, bed=bed)
    t_shear = time.time() - t1

    usable, degenerate, reasons = classify(out["mu_macro"], mu, out["shear_stress_drift_rel"], tol)
    gx, gy = _downsample(out["shear_strain_history"], out["shear_stress_history"])
    disp_x, _ = _downsample(out["shear_strain_history"], out["normal_stress_history"])
    _, sig_y = _downsample(out["shear_strain_history"], out["normal_stress_history"])

    rec = {
        "tag": tag,
        "case": {"mu_p": mu, "mu_rolling": mu_rolling, "seed": seed, "gamma": gamma,
                 "sigma_target_pa": float(sigma), "geom": geom, **GEOM[geom],
                 "kn": case.kn, "kt_ratio": case.kt_ratio,
                 "restitution": case.restitution, "shear_velocity": case.shear_velocity,
                 "consol_time_s": consol, "stationarity_tol": tol,
                 "timestep_s": out["timestep"]},
        "bed": {"n_particles": bed["n_particles"], "steps": bed["steps"],
                "packing_fraction": bed["packing_fraction"], "bed_top_m": bed["bed_top"],
                "secs": t_bed},
        "level": {
            "sigma_target_pa": float(sigma),
            "sigma_pa": out["normal_stress_measured"],
            "sigma_rel_err": out["normal_stress_rel_err"],
            "sigma_sem_pa": out["normal_stress_sem"],
            "tau_pa": out["shear_stress"],
            "tau_sem_pa": out["shear_stress_sem"],
            "tau_floor_pa": out["floor_shear_stress"],
            "tau_platen_vs_floor_rel": out["tau_platen_vs_floor_rel"],
            "mu_macro": out["mu_macro"],
            "mu_over_mu_p": (out["mu_macro"] / mu) if mu else float("nan"),
            "phi_deg": out["friction_angle_deg"],
            "drift_rel": out["shear_stress_drift_rel"],
            "drift_r2": out["r_squared"],
            "shear_rate_actual": out["shear_rate_actual"],
            "inertial_number": out["inertial_number"],
            "bed_height_mean_m": out["bed_height_mean"],
            "n_particles": out["n_particles"],
            "total_steps": out["total_steps"],
            "device_valid": out["valid"],
            "device_invalid_reason": out["invalid_reason"],
            "usable": usable,
            "degenerate": degenerate,
            "reject_reasons": reasons,
        },
        "history": {"strain": gx, "tau_pa": gy, "sigma_pa": sig_y},
        "secs": {"bed": t_bed, "shear": t_shear, "total": time.time() - t0},
        "repro_cmd": ("PYTHONPATH=src python scripts/m15159/run_gap_dem_friction.py "
                      "--tag %s --mu %g --gamma %g --sigma %g --geom %s --tol %g"
                      % (tag, mu, gamma, sigma, geom, tol)),
    }
    if write:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, tag + ".json"), "w") as fh:
            json.dump(rec, fh, indent=1)
    return rec


def benchmark(rec, ref_mu=REF_MU, band=REF_BAND, tol=TOLERANCE):
    """ref vs got for one point (the machine-readable L4 row)."""
    got = rec["level"]["mu_macro"]
    err = (got / ref_mu - 1.0) if got == got else float("nan")
    return {
        "quantity": "mu_macro (quasi-static bulk friction coefficient)",
        "case": {"mu_p": rec["case"]["mu_p"], "gamma": rec["case"]["gamma"],
                 "sigma_n_pa": rec["case"]["sigma_target_pa"], "geom": rec["case"]["geom"],
                 "n_particles": rec["bed"]["n_particles"]},
        "ref": ref_mu, "ref_band": list(band),
        "ref_source": ("GDR MiDi, Eur. Phys. J. E 14, 341 (2004); "
                       "Jop, Forterre & Pouliquen, Nature 441, 727 (2006): "
                       "quasi-static effective friction of dense packed spheres 0.36-0.40"),
        "got": got,
        "abs_rel_error": abs(err) if err == err else float("nan"),
        "signed_rel_error": err,
        "tolerance": tol,
        "within_tolerance": bool(err == err and abs(err) < tol),
        "usable": rec["level"]["usable"],
        "degenerate": rec["level"]["degenerate"],
        "drift_rel": rec["level"]["drift_rel"],
        "repro_cmd": rec["repro_cmd"],
        "pass": bool(rec["level"]["usable"] and err == err and abs(err) < tol),
    }


def load_points():
    if not os.path.isdir(OUT_DIR):
        return []
    out = []
    for name in sorted(os.listdir(OUT_DIR)):
        if name.endswith(".json") and name != "benchmark.json":
            with open(os.path.join(OUT_DIR, name)) as fh:
                out.append(json.load(fh))
    return out


def envelope(points, tol=STATIONARITY_TOL):
    """Mohr-Coulomb envelope over the *steady* levels of one material."""
    rows = [p for p in points
            if p["level"]["usable"] and not p["level"]["degenerate"]
            and p["case"]["mu_p"] == 0.5]
    sig = [p["level"]["sigma_pa"] for p in rows]
    tau = [p["level"]["tau_pa"] for p in rows]
    if len(rows) < 2:
        return {"status": "insufficient_levels", "n_levels": len(rows),
                "sigmas_pa": sig, "taus_pa": tau}
    s, t = np.asarray(sig), np.asarray(tau)
    A = np.vstack([s, np.ones_like(s)]).T
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    pred = A @ coef
    ss_res = float(((t - pred) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    return {"status": "ok", "n_levels": int(len(rows)),
            "sigmas_pa": [float(v) for v in s], "taus_pa": [float(v) for v in t],
            "tan_phi": float(coef[0]),
            "phi_deg": float(math.degrees(math.atan(coef[0]))),
            "cohesion_pa": float(coef[1]),
            "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
            "mu_per_level": [float(v) for v in t / s]}