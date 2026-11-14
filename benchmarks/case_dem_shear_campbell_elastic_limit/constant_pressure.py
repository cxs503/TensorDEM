#!/usr/bin/env python3
"""Round 5 of CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT: the constant-pressure path.

Why this round (owner 2026-11-11 two-round rule: survey first, then build)
-------------------------------------------------------------------------
R1-R4 mapped ``tau*(kappa)`` in a **constant-volume** Lees-Edwards cell at
``phi = 0.80``.  Two things were left open by R4 itself:

* the cell can only reach ``kappa >= 9`` (the small-overlap wall) and it can
  never *dilate*, so Campbell's headline mechanism -- "reducing the
  concentration slightly causes a transition from an elastic to a purely
  inertial behaviour" -- cannot be exercised in it at all;
* every gated point sits in the elastic branch, so the exponent ledger is one
  single power law (``beta = 0.894``, ``n = 0.213``) and there is no second,
  independent path against which to check that ``kappa`` is really the control.

Campbell ran **pressure-controlled** shear (the open-access paraphrase in
``reference_anchors.json`` names exactly that: "one is the pressure controlled
shearing and the other is the volume controlled shearing").  This repository
already owns such a device -- :func:`tensordem.mu_i_rheology.simulate_servo_shear`,
a constant-stress direct-shear box with a servo platen -- so this round measures
the *same* observable ``tau*(kappa)`` on the pressure-controlled path and
compares it with the constant-volume fit.  No new physics kernel is written;
only the experiment moves.

Observables (both taken from the **bulk** (interior-slab) virial estimator, with
the wall estimator carried alongside as a cross-check):

    kappa  = k_n / (rho d^3 gdot^2)          (gdot = measured bulk shear rate)
    tau*   = tau / (rho d^2 gdot^2)          (tau = measured bulk shear stress)
    I      = gdot d / sqrt(P/rho)
    phi    = solid fraction of the interior slab

Pre-registered questions
------------------------
P1 reach      does the constant-pressure cell deliver >= 5 gated points over
              >= 1.5 decades of ``kappa`` inside 50 <= kappa <= 2e5?
P2 exponent   is the measured ``beta`` within 0.15 of the constant-volume value
              (``beta_CV = 0.8935``)?  If yes the two devices agree that
              ``kappa`` alone sets the exponent ("path independent"); if no,
              the concentration is a second control and R1-R4's "the exponent
              of the anchor" must be quoted together with ``phi``.
P3 coefficient is ``C`` within a factor 2 of ``C_CV = 0.04301``?
P4 concentration  is ``phi`` monotone in ``kappa`` along this path (sign test),
              and how many decades of ``tau*`` per unit ``ln phi``?
P5 resolution  (a) seed twins at two rungs: spread <= 15 %; (b) ``dt/2`` at two
              rungs: ``|Delta tau*| <= 10 %``.
P6 collapse   (diagnostic) rungs at *fixed* ``I`` and *varying* ``k_n`` (hence
              varying ``kappa`` at ~fixed ``phi``): do they fall on the same
              ``tau*(kappa)`` curve as the rate ladder?

Legs that stay *un-testable* here are declared, not guessed: Campbell's
numerical table (coefficient, and the ``kappa`` at which the abrupt collapse
happens) is behind a paywall -- see ``reference_anchors.json`` -- so the
absolute-coefficient leg stays **WITHHELD**; the 2-D/3-D deviation of R1-R4 is
unchanged (this device is 3-D, so the coefficient is not transferable to the
2-D ladder either, and that is reported as a declared deviation).

Usage
-----
    PYTHONPATH=src python3 .../constant_pressure.py --prereg
    PYTHONPATH=src python3 .../constant_pressure.py --stage grid --jobs 4
    PYTHONPATH=src python3 .../constant_pressure.py --analyse
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
from typing import Any, Dict, List, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.elastic_limit_shear import kappa_of, tau_star_of  # noqa: E402

VARIANT = os.environ.get("CP_VARIANT", "smooth")
RUNS = HERE / "runs" / ("constant_pressure_rough" if VARIANT == "rough"
                        else "constant_pressure")
POINTS = RUNS / "points"
PREREG = RUNS / "prereg.json"
R4_ANALYSIS = HERE / "runs" / "inertial_branch" / "analysis.json"
ANCHORS = HERE / "reference_anchors.json"

# ---------------------------------------------------------------------------
# design
# ---------------------------------------------------------------------------
BASE = dict(radius=0.005, density=2500.0, mu=0.5, mu_rolling=0.05,
            restitution=0.3, kt_ratio=0.5, box_width=0.12, box_depth=0.05,
            fill_height=0.06, gravity=0.0, normal_stress=6.25e5,
            settle_steps=3000, min_shear_steps=2000, shear_strain=0.25)

# (tag, kn, inertial_number, seed, dt_scale, role)
GRID: List[Dict[str, Any]] = [
    dict(tag="I0.010", kn=6.25e4, inertial_number=0.010, seed=5279, dt_scale=1.0,
         role="ladder"),
    dict(tag="I0.020", kn=6.25e4, inertial_number=0.020, seed=5279, dt_scale=1.0,
         role="ladder"),
    dict(tag="I0.040", kn=6.25e4, inertial_number=0.040, seed=5279, dt_scale=1.0,
         role="ladder"),
    dict(tag="I0.100", kn=6.25e4, inertial_number=0.100, seed=5279, dt_scale=1.0,
         role="ladder"),
    dict(tag="I0.200", kn=6.25e4, inertial_number=0.200, seed=5279, dt_scale=1.0,
         role="ladder"),
    dict(tag="I0.400", kn=6.25e4, inertial_number=0.400, seed=5279, dt_scale=1.0,
         role="ladder"),
    # seed twins (noise floor)
    dict(tag="I0.020_s2", kn=6.25e4, inertial_number=0.020, seed=6451,
         dt_scale=1.0, role="seed"),
    dict(tag="I0.200_s2", kn=6.25e4, inertial_number=0.200, seed=6451,
         dt_scale=1.0, role="seed"),
    # time-step refinement
    dict(tag="I0.020_dt2", kn=6.25e4, inertial_number=0.020, seed=5279,
         dt_scale=0.5, role="dt"),
    dict(tag="I0.200_dt2", kn=6.25e4, inertial_number=0.200, seed=5279,
         dt_scale=0.5, role="dt"),
    # stiffness rungs at fixed I (kappa moved with k_n at ~fixed phi)
    dict(tag="I0.100_kn2.5e5", kn=2.5e5, inertial_number=0.100, seed=5279,
         dt_scale=1.0, role="kn"),
    dict(tag="I0.100_kn1.0e6", kn=1.0e6, inertial_number=0.100, seed=5279,
         dt_scale=1.0, role="kn"),
]

# Round-5b: the same ladder driven by a *rough* platen (grafted monolayer,
# ``wall_mode="rough"``, da Cruz et al. 2005), written BEFORE it is run.  Round
# 5a (smooth platens) is rejected by its own shear-rate gate: the measured bulk
# rate departs from the imposed plate rate by -34 % .. +28 % and the seed twins
# at I=0.2 differ by 16 %, i.e. the smooth wall slips.  The rough wall is the
# repository's documented remedy; the gates are NOT relaxed.
GRID_ROUGH: List[Dict[str, Any]] = [
    dict(tag=f"{t}_rough", kn=kn, inertial_number=I, seed=5279, dt_scale=dt,
         role=role, wall_mode="rough")
    for (t, kn, I, dt, role) in [
        ("I0.020", 6.25e4, 0.020, 1.0, "ladder"),
        ("I0.040", 6.25e4, 0.040, 1.0, "ladder"),
        ("I0.100", 6.25e4, 0.100, 1.0, "ladder"),
        ("I0.200", 6.25e4, 0.200, 1.0, "ladder"),
        ("I0.400", 6.25e4, 0.400, 1.0, "ladder"),
        ("I0.100_kn2.5e5", 2.5e5, 0.100, 1.0, "kn"),
        ("I0.100_kn1.0e6", 1.0e6, 0.100, 1.0, "kn"),
        ("I0.200_s2", 6.25e4, 0.200, 1.0, "seed"),
        ("I0.200_dt2", 6.25e4, 0.200, 0.5, "dt"),
    ]
]
GRID = GRID_ROUGH if VARIANT == "rough" else GRID

GATES = {
    "pressure_control": 0.05,   # |P_meas/P_applied - 1|
    "shear_rate": 0.15,         # |gdot_meas/gdot_imposed - 1|
    "profile_r2": 0.90,         # linear velocity profile (homogeneous shear)
    "phi_min": 0.55,            # still a dense packing
    "overlap_proxy": 0.10,      # 1/(3 kappa I^2) ~ delta/d  (PROXY, not measured)
}
RULES = {
    "P2_tolerance_beta": 0.15,
    "P3_tolerance_factor": 2.0,
    "P5_seed_spread": 0.15,
    "P5_dt_change": 0.10,
    "P1_min_points": 5,
    "P1_min_decades": 1.5,
}


# ---------------------------------------------------------------------------
# observables and validity gates (pure functions: unit-tested)
# ---------------------------------------------------------------------------
def point_observables(*, kn: float, rho: float, d: float, gdot: float,
                      tau: float, pressure: float) -> Dict[str, float]:
    """The three dimensionless numbers of this round, from one measured point.

    ``kappa`` and ``tau_star`` are the repository's existing definitions
    (:func:`tensordem.elastic_limit_shear.kappa_of` / ``tau_star_of``) so the
    constant-pressure and constant-volume rounds cannot drift apart; the
    identity ``tau_star == (tau/pressure) / I**2`` is returned alongside as
    ``tau_star_from_mu`` because it is what makes the two devices comparable.
    """
    kappa = kappa_of(kn, rho, d, gdot)
    tau_star = tau_star_of(tau, rho, d, gdot)
    inertial = gdot * d / math.sqrt(pressure / rho)
    return {"kappa": kappa, "tau_star": tau_star, "inertial_number": inertial,
            "tau_star_from_mu": (tau / pressure) / inertial ** 2}


def overlap_proxy(kappa: float, inertial: float) -> float:
    """``delta/d ~ 1/(3 kappa I^2)`` -- an analytic PROXY, not a measurement.

    ``kappa I^2 = k_n/(d P)`` for a linear spring, so this quantity depends only
    on the stiffness-to-pressure ratio: it is the constant-pressure analogue of
    the measured overlap gate of the constant-volume cell (which cannot be
    measured here because the servo device returns no particle overlaps).
    """
    return 1.0 / (3.0 * kappa * inertial ** 2)


def gate_point(*, pressure_error: float, gdot: float, gdot_imposed: float,
               profile_r2: float, phi: float, kappa: float,
               inertial: float) -> Dict[str, Any]:
    """Apply the pre-registered validity gates to one measured point."""
    ovl = overlap_proxy(kappa, inertial)
    gates = {
        "pressure_control": abs(pressure_error) <= GATES["pressure_control"],
        "shear_rate": abs(gdot / gdot_imposed - 1.0) <= GATES["shear_rate"],
        "profile_r2": profile_r2 >= GATES["profile_r2"],
        "phi_min": phi >= GATES["phi_min"],
        "overlap_proxy": ovl <= GATES["overlap_proxy"],
    }
    return {"gates": gates, "overlap_proxy": ovl, "gated": all(gates.values())}


# ---------------------------------------------------------------------------
# pre-registration
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_prereg() -> Dict[str, Any]:
    RUNS.mkdir(parents=True, exist_ok=True)
    r4 = json.loads(R4_ANALYSIS.read_text())
    cv = r4["partition"]["single_power_law"]
    doc = {
        "case": "CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT",
        "round": ("R5b (constant-pressure path, ROUGH platen)" if VARIANT == "rough"
                  else "R5a (constant-pressure path, smooth platen)"),
        "written_before_any_point": True,
        "device": "tensordem.mu_i_rheology.simulate_servo_shear",
        "why_this_device": (
            "Campbell (2002) organised the elastic limit with pressure-controlled "
            "shear; R1-R4 used a constant-volume cell and could never dilate. "
            "This device is the repository's constant-stress servo shear box."),
        "variant": VARIANT,
        "base_case": BASE,
        "grid": GRID,
        "gates": GATES,
        "rules": RULES,
        "observables": {
            "kappa": "k_n/(rho d^3 gdot^2), gdot = measured bulk shear rate",
            "tau_star": "tau/(rho d^2 gdot^2), tau = measured bulk shear stress",
            "inertial_number": "gdot d/sqrt(P/rho), P = measured bulk pressure",
            "phi": "interior-slab solid fraction (phi_bed_steady)",
        },
        "predictions": {
            "P1_reach": ">= 5 gated points over >= 1.5 decades inside 50..2e5",
            "P2_exponent": "|beta_CP - beta_CV| <= 0.15 -> kappa is the sole control",
            "P3_coefficient": "C_CP / C_CV within a factor 2",
            "P4_concentration": "phi monotone in kappa (higher kappa -> denser)",
            "P5_resolution": "seed spread <= 15 %, dt/2 change <= 10 %",
            "P6_collapse": "kn rungs at fixed I land on the rate-ladder curve (diagnostic)",
        },
        "known_degeneracy": (
            "on this path kappa and I are not independent: kappa I^2 = k_n/(d P), so "
            "tau* = mu kappa dP/k_n and beta = 1 + d ln mu/d ln kappa + d ln P/d ln kappa. "
            "The constant-pressure exponent is therefore NOT an independent measurement "
            "of the anchor's beta; P2 is reported as a consistency check with the "
            "structural prediction, not as confirmation."),
        "known_untestable": {
            "campbell_numeric_table": "full text closed (paywall) - coefficient leg WITHHELD",
            "dimension": "this device is 3-D, R1-R4 are 2-D; coefficients are not "
                         "transferable across dimension and are compared only as "
                         "declared deviation",
            "overlap": "the servo device does not return particle overlaps; the "
                       "small-overlap gate is an analytic proxy 1/(3 kappa I^2), "
                       "declared as a proxy, not a measurement",
        },
        "fit_inputs": {
            "constant_pressure_smooth": {
                "source": "benchmarks/case_dem_shear_campbell_elastic_limit/runs/"
                          "constant_pressure/analysis.json",
                "note": "read when present; the smooth-wall round is the diagnosis "
                        "that motivates this variant",
            },
            "constant_volume": {
                "source": str(R4_ANALYSIS.relative_to(REPO)),
                "beta": cv["beta"], "C": cv["C"],
                "rms_log_residual": cv["rms_log_residual"],
            },
        },
        "sha256": {
            str(R4_ANALYSIS.relative_to(REPO)): _sha256(R4_ANALYSIS),
            str(ANCHORS.relative_to(REPO)): _sha256(ANCHORS),
            str(Path(__file__).relative_to(REPO)): _sha256(Path(__file__)),
        },
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    PREREG.write_text(json.dumps(doc, indent=1))
    return doc


# ---------------------------------------------------------------------------
# grid stage
# ---------------------------------------------------------------------------
def run_point(spec: Dict[str, Any]) -> Dict[str, Any]:
    from tensordem.mu_i_rheology import ServoShearCase, simulate_servo_shear

    out = POINTS / f"{spec['tag']}.json"
    if out.exists():
        return json.loads(out.read_text())
    case = ServoShearCase(kn=spec["kn"], inertial_number=spec["inertial_number"],
                          seed=spec["seed"],
                          wall_mode=spec.get("wall_mode", "smooth"), **BASE)
    if spec["dt_scale"] != 1.0:
        case = ServoShearCase(kn=spec["kn"],
                              inertial_number=spec["inertial_number"],
                              seed=spec["seed"],
                              wall_mode=spec.get("wall_mode", "smooth"),
                              dt=case.timestep * spec["dt_scale"], **BASE)
    res = simulate_servo_shear(case)
    rec = {"tag": spec["tag"], "spec": spec,
           "dt": case.timestep * spec["dt_scale"],
           "result": {k: v for k, v in res.items()
                      if isinstance(v, (int, float, str, bool, type(None)))}}
    POINTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    return rec


def _worker(spec: Dict[str, Any]) -> str:
    try:
        run_point(spec)
        return f"ok {spec['tag']}"
    except Exception as exc:  # pragma: no cover - reported, never swallowed
        return f"FAIL {spec['tag']}: {type(exc).__name__}: {exc}"


def stage_grid(jobs: int) -> None:
    from multiprocessing import Pool

    todo = [s for s in GRID if not (POINTS / f"{s['tag']}.json").exists()]
    print(f"[grid] {len(todo)} of {len(GRID)} points to run, jobs={jobs}")
    t0 = time.time()
    with Pool(processes=jobs) as pool:
        for msg in pool.imap_unordered(_worker, todo):
            print(f"  {msg}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"[grid] done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def _fit_power(kappa: Sequence[float], taustar: Sequence[float]) -> Dict[str, float]:
    """Least squares in log space (multiplicative errors): tau* = C kappa^beta."""
    import numpy as np

    x = np.log(np.asarray(kappa, float))
    y = np.log(np.asarray(taustar, float))
    a = np.vstack([x, np.ones_like(x)]).T
    (beta, lnC), *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - (beta * x + lnC)
    n = len(x)
    dof = max(n - 2, 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(a.T @ a)
    return {"beta": float(beta), "C": float(math.exp(lnC)),
            "sigma_beta": float(math.sqrt(max(cov[0, 0], 0.0))),
            "sigma_lnC": float(math.sqrt(max(cov[1, 1], 0.0))),
            "rms_log_residual": float(math.sqrt(float(resid @ resid) / n)),
            "n_points": n}


def analyse() -> Dict[str, Any]:
    import numpy as np

    prereg = json.loads(PREREG.read_text())
    r4 = json.loads(R4_ANALYSIS.read_text())
    cv = r4["partition"]["single_power_law"]
    beta_cv, C_cv = float(cv["beta"]), float(cv["C"])

    an_path = RUNS / "analysis.json"
    if an_path.exists() and an_path.stat().st_mtime < PREREG.stat().st_mtime:
        raise SystemExit("analysis.json predates prereg.json - refusing to fit")

    rows: List[Dict[str, Any]] = []
    for spec in GRID:
        path = POINTS / f"{spec['tag']}.json"
        if not path.exists():
            rows.append({"tag": spec["tag"], "role": spec["role"],
                         "status": "missing"})
            continue
        rec = json.loads(path.read_text())
        r = rec["result"]
        d = 2.0 * float(r["radius"])
        rho = float(r["density"])
        kn = float(r["kn"])
        gdot = r.get("shear_rate_measured")
        gdot_imp = float(r["shear_rate_imposed"])
        p_bulk = r.get("bulk_pressure_pa")
        tau_bulk = r.get("bulk_shear_pa")
        if not gdot or not p_bulk or not tau_bulk:
            rows.append({"tag": spec["tag"], "role": spec["role"],
                         "status": "incomplete"})
            continue
        gdot = abs(float(gdot))
        phi = float(r["phi_bed_steady"])
        obs = point_observables(kn=kn, rho=rho, d=d, gdot=gdot,
                                tau=float(tau_bulk), pressure=float(p_bulk))
        kappa, tau_star = obs["kappa"], obs["tau_star"]
        inertial = obs["inertial_number"]
        gt = gate_point(pressure_error=float(r["pressure_control_error_final"]),
                        gdot=gdot, gdot_imposed=gdot_imp,
                        profile_r2=float(r.get("profile_r2") or 0.0),
                        phi=phi, kappa=kappa, inertial=inertial)
        ovl, gates = gt["overlap_proxy"], gt["gates"]
        rows.append({
            "tag": spec["tag"], "role": spec["role"], "status": "ok",
            "kn": kn, "I_imposed": spec["inertial_number"],
            "gdot_measured": gdot, "gdot_imposed": gdot_imp,
            "kappa": kappa, "tau_star": tau_star,
            "tau_bulk": float(tau_bulk), "p_bulk": float(p_bulk),
            "mu_bulk": float(r.get("bulk_mu") or float("nan")),
            "mu_wall": float(r.get("mu_effective") or float("nan")),
            "tau_star_wall": (abs(float(r["shear_stress_pa"]))
                              / (rho * d ** 2 * gdot ** 2)),
            "inertial_number": inertial, "phi": phi,
            "coordination": float(r.get("coordination_number") or float("nan")),
            "overlap_proxy": ovl,
            "pressure_error": float(r["pressure_control_error_final"]),
            "profile_r2": float(r.get("profile_r2") or float("nan")),
            "wall_clock_s": float(r.get("wall_clock_s") or float("nan")),
            "tau_star_from_mu": obs["tau_star_from_mu"],
            "gates": gates, "gated": gt["gated"],
        })

    ladder = [x for x in rows if x.get("status") == "ok" and x["role"] == "ladder"]
    gated = [x for x in ladder if x["gated"]]
    out: Dict[str, Any] = {
        "case": prereg["case"], "round": prereg["round"],
        "n_submitted": len(GRID), "n_ok": sum(1 for x in rows if x.get("status") == "ok"),
        "n_gated": len(gated),
        "prereg_sha256": _sha256(PREREG),
        "points": rows,
        "gates": GATES, "rules": RULES,
    }

    # ---- P1 reach -------------------------------------------------------
    if gated:
        kap = [x["kappa"] for x in gated]
        decades = math.log10(max(kap) / min(kap)) if min(kap) > 0 else 0.0
        out["reach"] = {"kappa_min": min(kap), "kappa_max": max(kap),
                        "decades": decades,
                        "inside_50_2e5": all(50.0 <= k <= 2.0e5 for k in kap),
                        "n_gated": len(gated)}
        out["P1_verdict"] = ("PASS" if len(gated) >= RULES["P1_min_points"]
                             and decades >= RULES["P1_min_decades"]
                             and out["reach"]["inside_50_2e5"] else "NOT_SUPPORTED")
    else:
        out["reach"] = None
        out["P1_verdict"] = "NOT_SUPPORTED"

    # ---- P2/P3 exponent and coefficient --------------------------------
    if len(gated) >= 3:
        fit = _fit_power([x["kappa"] for x in gated], [x["tau_star"] for x in gated])
        out["fit_constant_pressure"] = fit
        out["fit_constant_volume"] = {"beta": beta_cv, "C": C_cv,
                                      "rms_log_residual": float(cv["rms_log_residual"]),
                                      "source": prereg["fit_inputs"]["constant_volume"]["source"]}
        dbeta = fit["beta"] - beta_cv
        out["P2_detail"] = {
            "beta_CP": fit["beta"], "sigma_beta": fit["sigma_beta"],
            "beta_CV": beta_cv, "delta_beta": dbeta,
            "n_CP": 2.0 * (1.0 - fit["beta"]), "n_CV": 2.0 * (1.0 - beta_cv),
            "tolerance": RULES["P2_tolerance_beta"],
            "z_of_delta": (abs(dbeta) / fit["sigma_beta"]) if fit["sigma_beta"] > 0 else None,
        }
        out["P2_verdict"] = ("PASS" if abs(dbeta) <= RULES["P2_tolerance_beta"]
                             else "NOT_SUPPORTED")
        ratio = fit["C"] / C_cv
        out["P3_detail"] = {"C_CP": fit["C"], "C_CV": C_cv, "ratio": ratio,
                            "tolerance_factor": RULES["P3_tolerance_factor"]}
        out["P3_verdict"] = ("PASS" if 1.0 / RULES["P3_tolerance_factor"] <= ratio
                             <= RULES["P3_tolerance_factor"] else "NOT_SUPPORTED")
        # local exponents along the ladder
        srt = sorted(gated, key=lambda x: -x["kappa"])
        loc = []
        for a, b in zip(srt, srt[1:]):
            beta_l = math.log(b["tau_star"] / a["tau_star"]) / math.log(b["kappa"] / a["kappa"])
            loc.append({"kappa_lo": a["kappa"], "kappa_hi": b["kappa"],
                        "tags": [b["tag"], a["tag"]],
                        "beta_local": beta_l, "n_local": 2.0 * (1.0 - beta_l)})
        out["local_slopes"] = loc
    else:
        out["P2_verdict"] = out["P3_verdict"] = "NOT_SUPPORTED"

    # ---- P4 concentration ----------------------------------------------
    if len(gated) >= 3:
        kk = np.log([x["kappa"] for x in gated])
        pp = np.log([x["phi"] for x in gated])
        slope = float(np.polyfit(kk, pp, 1)[0])
        # Spearman-free sign test: count concordant pairs
        conc = sum(1 for a in gated for b in gated
                   if (a["kappa"] - b["kappa"]) * (a["phi"] - b["phi"]) > 0)
        pairs = sum(1 for a in gated for b in gated if a is not b)
        out["P4_detail"] = {"d_ln_phi_d_ln_kappa": slope,
                            "concordant_fraction": conc / pairs if pairs else None,
                            "phi_min": min(x["phi"] for x in gated),
                            "phi_max": max(x["phi"] for x in gated),
                            "note": ("sign test only: the constant-pressure path "
                                     "cannot isolate phi at fixed kappa")}
        out["P4_verdict"] = "PASS" if slope > 0 and conc == pairs else "NOT_SUPPORTED"

    # ---- P5 resolution --------------------------------------------------
    def _pair(base_tag: str, other_tag: str) -> Dict[str, float]:
        base = next((x for x in rows if x["tag"] == base_tag and x.get("status") == "ok"), None)
        oth = next((x for x in rows if x["tag"] == other_tag and x.get("status") == "ok"), None)
        if base is None or oth is None:
            return {"status": "missing"}
        return {"status": "ok",
                "tau_star_base": base["tau_star"], "tau_star_other": oth["tau_star"],
                "rel_change": oth["tau_star"] / base["tau_star"] - 1.0,
                "kappa_rel_change": oth["kappa"] / base["kappa"] - 1.0}

    seeds = [_pair("I0.020", "I0.020_s2"), _pair("I0.200", "I0.200_s2")]
    dts = [_pair("I0.020", "I0.020_dt2"), _pair("I0.200", "I0.200_dt2")]
    seed_ok = all(s.get("status") == "ok" and abs(s["rel_change"]) <= RULES["P5_seed_spread"]
                  for s in seeds)
    dt_ok = all(s.get("status") == "ok" and abs(s["rel_change"]) <= RULES["P5_dt_change"]
                for s in dts)
    out["P5_detail"] = {"seeds": seeds, "dt_refinement": dts}
    out["P5_verdict"] = "PASS" if (seed_ok and dt_ok) else "NOT_SUPPORTED"

    # ---- P6 collapse (kn rungs at fixed I) ------------------------------
    kn_rows = [x for x in rows if x.get("status") == "ok" and x["role"] == "kn"]
    if kn_rows and "fit_constant_pressure" in out:
        fit = out["fit_constant_pressure"]
        pred = []
        for x in kn_rows:
            t_pred = fit["C"] * x["kappa"] ** fit["beta"]
            pred.append({"tag": x["tag"], "kappa": x["kappa"],
                         "tau_star": x["tau_star"], "tau_star_fit": t_pred,
                         "residual_log": math.log(x["tau_star"] / t_pred),
                         "gated": x["gated"]})
        worst = max(abs(p["residual_log"]) for p in pred)
        out["P6_detail"] = {"rungs": pred, "worst_abs_log_residual": worst,
                            "tolerance_log": math.log(1.5)}
        out["P6_verdict"] = "PASS" if worst <= math.log(1.5) else "NOT_SUPPORTED"
    else:
        out["P6_verdict"] = "NOT_SUPPORTED"

    an_path.write_text(json.dumps(out, indent=1, default=float))
    write_tables(out)
    make_figure(out)
    return out


def write_tables(out: Dict[str, Any]) -> None:
    lines = ["# R5 constant-pressure ladder - tables", ""]
    hdr = ("| tag | role | kn | I | gdot | kappa | tau* | mu_bulk | phi | Z | "
           "ovl proxy | gates |")
    lines += [hdr, "|" + "---|" * 12]
    for x in out["points"]:
        if x.get("status") != "ok":
            lines.append(f"| {x['tag']} | {x['role']} | - | - | - | - | - | - | - | - | - | {x['status']} |")
            continue
        bad = [k for k, v in x["gates"].items() if not v]
        lines.append(
            f"| {x['tag']} | {x['role']} | {x['kn']:.3g} | {x['inertial_number']:.4f} | "
            f"{x['gdot_measured']:.1f} | {x['kappa']:.4g} | {x['tau_star']:.4g} | "
            f"{x['mu_bulk']:.3f} | {x['phi']:.4f} | {x['coordination']:.2f} | "
            f"{x['overlap_proxy']:.4f} | {'PASS' if x['gated'] else 'FAIL(' + ','.join(bad) + ')'} |")
    lines += ["", "## verdicts", "", "| id | verdict |", "|---|---|"]
    for k in ("P1_verdict", "P2_verdict", "P3_verdict", "P4_verdict", "P5_verdict", "P6_verdict"):
        lines.append(f"| {k.split('_')[0]} | {out.get(k)} |")
    if "fit_constant_pressure" in out:
        f = out["fit_constant_pressure"]
        lines += ["", f"constant-pressure fit: tau* = {f['C']:.5g} kappa^{f['beta']:.4f} "
                      f"(beta sigma {f['sigma_beta']:.4f}), RMS(ln resid) = "
                      f"{f['rms_log_residual']*100:.1f} %"]
    (RUNS / "tables.md").write_text("\n".join(lines) + "\n")


def make_figure(out: Dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ok = [x for x in out["points"] if x.get("status") == "ok"]
    if not ok:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    fit = out.get("fit_constant_pressure")
    cvm = out.get("fit_constant_volume")
    kk = np.logspace(1.5, 5.3, 50)
    if cvm:
        ax.loglog(kk, cvm["C"] * kk ** cvm["beta"], "k--", lw=1,
                  label=f"constant volume (R1-R4): beta={cvm['beta']:.3f}")
    if fit:
        ax.loglog(kk, fit["C"] * kk ** fit["beta"], "r-", lw=1.2,
                  label=f"constant pressure (R5): beta={fit['beta']:.3f}")
    style = {"ladder": ("o", "tab:red"), "seed": ("s", "tab:orange"),
             "dt": ("^", "tab:purple"), "kn": ("D", "tab:blue")}
    for role, (m, c) in style.items():
        pts = [x for x in ok if x["role"] == role]
        if not pts:
            continue
        ax.loglog([x["kappa"] for x in pts], [x["tau_star"] for x in pts],
                  m, color=c, ms=7 if role == "ladder" else 6, ls="none",
                  mfc="none" if role != "ladder" else c,
                  label=f"{role} rungs")
    ax.set_xlabel("kappa = k_n/(rho d^3 gdot^2)")
    ax.set_ylabel("tau* = tau/(rho d^2 gdot^2)")
    ax.set_title("(a) tau*(kappa) on two devices")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for role, (m, c) in style.items():
        pts = [x for x in ok if x["role"] == role]
        if pts:
            ax.loglog([x["kappa"] for x in pts], [x["phi"] for x in pts], m,
                      color=c, ms=7 if role == "ladder" else 6, ls="none",
                      mfc="none" if role != "ladder" else c, label=role)
    ax.set_xlabel("kappa")
    ax.set_ylabel("phi (interior slab)")
    ax.set_title("(b) concentration along the path (dilation)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    lad = sorted([x for x in ok if x["role"] == "ladder"], key=lambda x: x["kappa"])
    ax.plot([x["kappa"] for x in lad], [x["mu_bulk"] for x in lad], "o-",
            color="tab:red", label="mu_bulk")
    ax.plot([x["kappa"] for x in lad], [x["mu_wall"] for x in lad], "s--",
            color="tab:gray", label="mu_wall")
    ax.set_xscale("log")
    ax.set_xlabel("kappa")
    ax.set_ylabel("mu = tau/P")
    ax.set_title("(c) friction coefficient: wall vs bulk")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="major")
    fig.tight_layout()
    (RUNS / "fig_constant_pressure.png").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(RUNS / "fig_constant_pressure.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", action="store_true")
    ap.add_argument("--stage", choices=["grid"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if a.prereg:
        doc = write_prereg()
        print(f"prereg written: {PREREG} ({len(doc['grid'])} points)")
    if a.stage == "grid":
        stage_grid(a.jobs)
    if a.analyse:
        out = analyse()
        print(json.dumps({k: v for k, v in out.items()
                          if k.endswith("_verdict") or k in ("n_gated", "reach")},
                         indent=1, default=float))


if __name__ == "__main__":
    main()