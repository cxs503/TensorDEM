"""Driver and adjudicator of the P152W1-N04 friction-contact verification surface.

Stages
------
``laws``    closed-form anchors of the two declared contact laws (seconds)
``packing`` random-close-packing anchor (Lubachevsky-Stillinger inflation)
``repose``  angle-of-repose anchor: negative control + glass-bead calibration +
            out-of-sample sand prediction
``shear``   periodic shear cell: Mohr-Coulomb envelope of peak and residual
``report``  assemble every anchor against the pre-registered criteria, write
            ``results.json`` and the figures

Usage
-----
    python -m tensordem.friction_surface.study --stage laws --out artifacts/P152W1-N04
    python -m tensordem.friction_surface.study --stage all --jobs 12

Every number in the artifact is produced by this module; nothing is typed in by
hand.  Thresholds come from ``prereg/P152W1_N04_friction_contact_surface_prereg.json``
and are never modified here.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import asdict
from multiprocessing import Pool

import numpy as np

from . import laws
from .packing import PackingCase, random_close_packing
from .repose import ReposeCase, simulate_repose
from .shear import PeriodicShearCase, mohr_coulomb_fit, simulate_shear

PREREG = os.path.join("prereg", "P152W1_N04_friction_contact_surface_prereg.json")

#: literature micro-parameters of the two materials (not fitted to any angle)
MATERIALS = {
    "glass_beads": {
        "mu": 0.30,
        "theta_exp_deg": 23.0,
        "theta_band_deg": (21.0, 25.0),
        "note": ("interparticle friction of soda-lime glass on glass from the "
                 "usual 0.2-0.4 band; the repose angle is the literature "
                 "experimental band for dry monodisperse 0.5-1.0 mm beads"),
    },
    "quartz_sand": {
        "mu": 0.50,
        "theta_exp_deg": 32.0,
        "theta_band_deg": (30.0, 34.0),
        "note": ("quartz-quartz friction ~0.5 (sub-rounded to rounded medium "
                 "sand); repose band 30-34 deg for dry sand"),
    },
}


# ---------------------------------------------------------------------------
# stage: laws
# ---------------------------------------------------------------------------

def law_anchors() -> dict:
    slider, rolling = [], []
    for mu in (0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.8):
        meas = laws.simulate_slider_onset_deg(mu)
        ana = laws.sliding_onset_angle_deg(mu)
        slider.append({"mu": mu, "measured_deg": meas, "analytic_deg": ana,
                       "abs_error_deg": abs(meas - ana)})
    for mu_r in (0.02, 0.05, 0.1, 0.2):
        meas = laws.simulate_rolling_onset_deg(mu_r)
        ana = laws.rolling_onset_angle_deg(mu_r)
        rolling.append({"mu_r": mu_r, "measured_deg": meas, "analytic_deg": ana,
                        "abs_error_deg": abs(meas - ana)})
    damping = laws.tangential_damping_effect(0.5, ct=31.6)
    return {
        "sliding_onset": slider,
        "rolling_onset": rolling,
        "sliding_onset_max_error_deg": max(r["abs_error_deg"] for r in slider),
        "rolling_onset_max_error_deg": max(r["abs_error_deg"] for r in rolling),
        "tangential_damping_effect": damping,
    }


# ---------------------------------------------------------------------------
# stage: packing
# ---------------------------------------------------------------------------

def _packing_job(args):
    n, growth, relax, seed = args
    res = random_close_packing(PackingCase(n_particles=n, rho_growth=growth,
                                           relax_steps=relax, seed=seed,
                                           under_relax=0.4))
    res.pop("history", None)
    return res


def packing_anchor(jobs: int = 5) -> dict:
    cases = [(500, 5.0e-4, 400, 1), (500, 5.0e-4, 400, 2), (500, 5.0e-4, 400, 3),
             (1000, 5.0e-4, 400, 1), (1000, 5.0e-4, 400, 2)]
    with Pool(min(jobs, len(cases))) as pool:
        runs = pool.map(_packing_job, cases)
    for r, c in zip(runs, cases):
        r["case"] = {"n_particles": c[0], "rho_growth": c[1], "relax_steps": c[2],
                     "seed": c[3]}
    jam = [r["jammed_packing_fraction"] for r in runs
           if r["jammed_packing_fraction"] is not None]
    ref = 0.64
    mean = float(np.mean(jam))
    return {
        "runs": runs,
        "phi_values": jam,
        "phi_mean": mean,
        "phi_std": float(np.std(jam)),
        "reference_phi": ref,
        "relative_error": abs(mean - ref) / ref,
        "cross_route": _geometric_route_reference(),
        "protocol": runs[0]["protocol"],
    }


def _geometric_route_reference() -> dict:
    """Cross-check against the repository's geometric packing route.

    ``tensordem.bedcoupling.packing.make_random_close_packing`` requires a
    *target* porosity, i.e. it compresses until a prescribed density is reached.
    It therefore cannot serve as an independent measurement of the jamming
    density: asking it for 0.64 returns 0.64.  Recorded here so that a later
    reader does not mistake it for a cross-check.
    """
    return {"available": False,
            "reason": ("bedcoupling.packing.make_random_close_packing takes a target "
                       "porosity/diameter and compresses to it, so it cannot measure "
                       "phi_RCP independently; the inflation route is the only "
                       "jamming-seeking route in this repository"),
            "source": "tensordem.bedcoupling.packing"}


# ---------------------------------------------------------------------------
# stage: repose
# ---------------------------------------------------------------------------

def _repose_job(args):
    mu, mu_r, seed, n_particles = args[:4]
    source_mode = args[4] if len(args) > 4 else "fixed"
    keep = args[5] if len(args) > 5 else None
    case = ReposeCase(mu=mu, mu_rolling=mu_r, seed=seed, n_particles=n_particles,
                      source_mode=source_mode)
    res = simulate_repose(case)
    positions = res.pop("final_positions", None)
    return {"mu": mu, "mu_rolling": mu_r, "seed": seed,
            "theta_r_deg": res["theta_r_deg"], "reached_rest": res["reached_rest"],
            "n_particles": res["n_particles"], "t_end": res["t_end"],
            "measurement": res["measurement"], "protocol": res["protocol"],
            "source_mode": res["source_mode"], "nozzle_z_max": res["nozzle_z_max"],
            "apex_height_m": res["apex_height_m"], "pour_closed": res["pour_closed"],
            "bursts_released": res["bursts_released"],
            "bursts_skipped": res["bursts_skipped"], "clear_steps": res["clear_steps"],
            "positions": positions if keep else None}


def repose_sweep(mu_r: float, mu_list, seeds=(15204,), n_particles=900,
                 pool: Pool | None = None, source_mode: str = "fixed") -> list:
    jobs = [(mu, mu_r, seed, n_particles, source_mode) for mu in mu_list for seed in seeds]
    if pool is None:
        return [_repose_job(j) for j in jobs]
    return pool.map(_repose_job, jobs)


def _theta(res_list, mu):
    vals = [r["theta_r_deg"] for r in res_list if abs(r["mu"] - mu) < 1e-9]
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def repose_anchor(jobs: int = 12, n_particles: int = 900,
                  source_mode: str = "fixed", seeds=(15204, 15205, 15206)) -> dict:
    """Anchor A/B: negative control + glass-bead calibration + out-of-sample sand.

    ``source_mode`` selects the pour protocol: ``"fixed"`` is the v2 protocol
    (fixed nozzle, bursts self-paced by the charge fall time, the pour closes
    when the deposit fills the nozzle); ``"rising"`` is the v1 protocol and is
    kept only so the defect test can reproduce it.
    """
    with Pool(jobs) as pool:
        # 1) negative control: no friction at all -> no stable slope
        control = pool.map(_repose_job, [(0.0, 0.0, 15204, n_particles, source_mode),
                                         (0.0, 0.0, 15205, n_particles, source_mode)])
        # 2) bisection of the single shape parameter mu_r on the glass beads
        mu_g = MATERIALS["glass_beads"]["mu"]
        target = MATERIALS["glass_beads"]["theta_exp_deg"]
        lo, hi = 0.02, 0.40
        history = []
        for _ in range(5):
            mid = 0.5 * (lo + hi)
            theta = _theta(repose_sweep(mid, [mu_g], pool=pool, n_particles=n_particles,
                                        source_mode=source_mode), mu_g)
            history.append({"mu_rolling": mid, "theta_r_deg": theta})
            if theta < target:
                lo = mid
            else:
                hi = mid
        mu_r_star = 0.5 * (lo + hi)
        # 3) final runs: 3 seeds per material at the calibrated mu_r
        final_jobs = []
        for name, mat in MATERIALS.items():
            for seed in seeds:
                final_jobs.append((mat["mu"], mu_r_star, seed, n_particles, source_mode))
        finals = pool.map(_repose_job, final_jobs)
        # 4) theta(mu) surface at the calibrated mu_r (for the figure); the
        # material runs above are points of the same surface and are re-used.
        extra = [mu for mu in (0.1, 0.2, 0.4, 0.6)
                 if not any(abs(m["mu"] - mu) < 1e-9 for m in MATERIALS.values())]
        surface = pool.map(_repose_job,
                           [(mu, mu_r_star, 15204, n_particles, source_mode)
                            for mu in extra])
        # one run per material with its positions kept, for the figure
        keep = pool.map(_repose_job,
                        [(mat["mu"], mu_r_star, seeds[0], n_particles, source_mode, True)
                         for mat in MATERIALS.values()])
    materials = {}
    for name, mat in MATERIALS.items():
        vals = [r["theta_r_deg"] for r in finals
                if abs(r["mu"] - mat["mu"]) < 1e-9 and r["theta_r_deg"] == r["theta_r_deg"]]
        mean = float(np.mean(vals)) if vals else float("nan")
        materials[name] = {
            "mu": mat["mu"],
            "mu_rolling": mu_r_star,
            "theta_exp_deg": mat["theta_exp_deg"],
            "theta_exp_band_deg": mat["theta_band_deg"],
            "theta_sim_deg": mean,
            "theta_sim_std_deg": float(np.std(vals)) if vals else float("nan"),
            "theta_sim_runs": vals,
            "relative_error": abs(mean - mat["theta_exp_deg"]) / mat["theta_exp_deg"]
            if mean == mean else float("nan"),
            "note": mat["note"],
        }
    control_theta = [r["theta_r_deg"] for r in control]
    return {
        "negative_control": {
            "mu": 0.0, "mu_rolling": 0.0,
            "theta_r_deg": float(np.nanmean(control_theta)),
            "runs": control_theta,
            "criterion": "theta_r < 5 deg",
        },
        "calibration": {
            "parameter": "mu_rolling (single shape surrogate)",
            "fitted_on": "glass_beads",
            "mu_rolling_star": mu_r_star,
            "bisection_history": history,
        },
        "materials": materials,
        "theta_mu_surface": [{"mu": r["mu"], "theta_r_deg": r["theta_r_deg"]}
                             for r in list(surface)],
        "theta_mu_surface_all": [{"mu": r["mu"], "theta_r_deg": r["theta_r_deg"]}
                                 for r in list(finals) + list(surface)],
        "protocol": {
            "source_mode": source_mode,
            "n_particles": n_particles,
            "seeds": list(seeds),
            "pour": "fixed-height nozzle; bursts self-paced by the charge fall time; "
                    "the pour closes when the deposit fills the nozzle",
            "reached_rest": [r["reached_rest"] for r in finals],
            "pour_closed": [r["pour_closed"] for r in finals],
            "bursts_released": [r["bursts_released"] for r in finals],
            "bursts_skipped": [r["bursts_skipped"] for r in finals],
            "nozzle_z_max": [r["nozzle_z_max"] for r in finals],
            "clear_steps": [r["clear_steps"] for r in finals],
        },
        "figure_positions": {
            str(m["mu"]): (None if r["positions"] is None else r["positions"].tolist())
            for m, r in zip(MATERIALS.values(), keep)
        },
        "reference_micro_parameters": "literature interparticle friction, not fitted",
    }


# ---------------------------------------------------------------------------
# stage: shear
# ---------------------------------------------------------------------------

def _shear_job(args):
    sigma, seed, velocity = args
    case = PeriodicShearCase(sigma=sigma, seed=seed, shear_velocity=velocity,
                              nx=8, ny=6, nz=6, settle_time=0.40,
                              shear_strain_target=0.45)
    res = simulate_shear(case)
    res.pop("final_positions", None)
    return res


def shear_anchor(jobs: int = 13, sigmas=(2000.0, 5000.0, 10000.0, 20000.0),
                 seeds=(152041, 152042, 152043)) -> dict:
    work = [(s, sd, 0.04) for s in sigmas for sd in seeds]
    rate = [(5000.0, 152041, 0.02)]
    with Pool(min(jobs, len(work) + len(rate))) as pool:
        runs = pool.map(_shear_job, work)
        rate_runs = pool.map(_shear_job, rate)
    peak, resid, sig_meas = [], [], []
    per_sigma = {}
    for s in sigmas:
        sel = [r for r in runs if abs(r["sigma_applied"] - s) < 1e-6]
        tp = [r["tau_peak"] for r in sel]
        tr = [r["tau_residual"] for r in sel]
        sm = [r["sigma_measured"] for r in sel]
        per_sigma[str(int(s))] = {
            "tau_peak": float(np.mean(tp)), "tau_peak_std": float(np.std(tp)),
            "tau_residual": float(np.mean(tr)), "tau_residual_std": float(np.std(tr)),
            "sigma_measured": float(np.mean(sm)),
            "tau_peak_runs": tp, "tau_residual_runs": tr,
            "phi_peak_deg": math.degrees(math.atan(float(np.mean(tp)) / s)),
            "phi_residual_deg": math.degrees(math.atan(float(np.mean(tr)) / s)),
        }
        peak.append(float(np.mean(tp)))
        resid.append(float(np.mean(tr)))
        sig_meas.append(float(np.mean(sm)))
    fit_peak_applied = mohr_coulomb_fit(list(sigmas), peak)
    fit_res_applied = mohr_coulomb_fit(list(sigmas), resid)
    fit_peak_meas = mohr_coulomb_fit(sig_meas, peak)
    fit_res_meas = mohr_coulomb_fit(sig_meas, resid)
    mu_bound = math.degrees(math.atan(MATERIAL_MU + MATERIAL_MU_R))
    return {
        "runs": runs,
        "per_sigma": per_sigma,
        "peak_fit": fit_peak_applied,
        "residual_fit": fit_res_applied,
        "peak_fit_measured_sigma": fit_peak_meas,
        "residual_fit_measured_sigma": fit_res_meas,
        "worst_relative_deviation": max(fit_peak_applied["max_relative_deviation"],
                                        fit_res_applied["max_relative_deviation"]),
        "phi_peak_deg": fit_peak_applied["phi_deg"],
        "phi_residual_deg": fit_res_applied["phi_deg"],
        "cohesion_peak_pa": fit_peak_applied["cohesion_pa"],
        "cohesion_residual_pa": fit_res_applied["cohesion_pa"],
        "cohesion_over_min_tau": abs(fit_peak_applied["cohesion_pa"]) / min(peak),
        "bound_phi_deg": mu_bound,
        "bound_violated": bool(max(fit_peak_applied["phi_deg"], fit_res_applied["phi_deg"]) > mu_bound),
        "rate_check": {
            "tau_peak_v40": per_sigma.get("5000", {}).get("tau_peak"),
            "tau_peak_v20": float(np.mean([r["tau_peak"] for r in rate_runs])),
            "runs": rate_runs,
        },
        "cell": {"nx": 8, "ny": 6, "nz": 6, "n_particles": runs[0]["n_particles"],
                 "mu": runs[0]["mu"], "mu_rolling": runs[0]["mu_rolling"],
                 "shear_velocity": runs[0]["shear_velocity"]},
    }


MATERIAL_MU = 0.5
MATERIAL_MU_R = 0.1


# ---------------------------------------------------------------------------
# adjudication
# ---------------------------------------------------------------------------

def adjudicate(anchors: dict, prereg_path: str = PREREG) -> dict:
    with open(prereg_path) as fh:
        prereg = json.load(fh)
    criteria = {c["id"]: c for c in prereg["criteria"]}
    verdicts = []

    def add(cid, value, ok, detail, anchor_present=True):
        status = ("PASS" if ok else "FAIL") if anchor_present else "WITHHELD"
        criteria[cid]["measured"] = value
        criteria[cid]["passed"] = bool(ok) if anchor_present else None
        criteria[cid]["status"] = status
        criteria[cid]["detail"] = detail
        verdicts.append({"id": cid, "status": status, "passed": criteria[cid]["passed"],
                         "measured": value, "detail": detail})

    law = anchors.get("laws", {})
    add("C0-law-onset", law.get("sliding_onset_max_error_deg"),
        law.get("sliding_onset_max_error_deg", 1e9) < 0.5,
        "max |measured - arctan(mu)| over 7 friction values")
    add("C0-law-rolling", law.get("rolling_onset_max_error_deg"),
        law.get("rolling_onset_max_error_deg", 1e9) < 0.5,
        "max |measured - arctan(mu_r)| over 4 rolling coefficients")

    rep = anchors.get("repose", {})
    has_repose = bool(rep) and bool(rep.get("materials"))
    control = rep.get("negative_control", {}).get("theta_r_deg", float("nan"))
    add("A-repose-negative-control", control,
        (control == control and control < 5.0), "mu = mu_r = 0 heap",
        anchor_present=has_repose)
    mats = rep.get("materials", {})
    errs = {k: v.get("relative_error") for k, v in mats.items()}
    worst = max([e for e in errs.values() if e is not None], default=1e9)
    add("B-repose-experiment", errs, worst < 0.10,
        "glass beads: one-parameter calibration (mu_r); quartz sand: out-of-sample prediction",
        anchor_present=has_repose)

    pack = anchors.get("packing", {})
    has_pack = bool(pack)
    perr = pack.get("relative_error", 1e9)
    add("C-packing-density", {"phi_mean": pack.get("phi_mean"),
                             "phi_std": pack.get("phi_std"),
                             "relative_error": perr},
        perr < 0.05, "Lubachevsky-Stillinger inflation, 5 runs, two system sizes",
        anchor_present=has_pack)

    shear = anchors.get("shear", {})
    has_shear = bool(shear)
    dev = shear.get("worst_relative_deviation", 1e9)
    add("D-direct-shear-mohr-coulomb", dev, dev < 0.10,
        "worst per-sigma deviation from the fitted envelope (peak and residual)",
        anchor_present=has_shear)
    add("D2-mohr-coulomb-bound", {"phi_peak_deg": shear.get("phi_peak_deg"),
                                  "phi_residual_deg": shear.get("phi_residual_deg"),
                                  "bound_deg": shear.get("bound_phi_deg")},
        not shear.get("bound_violated", True),
        "phi_macro <= arctan(mu + mu_r) for a cohesionless packing",
        anchor_present=has_shear)
    coh = shear.get("cohesion_over_min_tau", 1e9)
    add("D3-cohesion", coh, coh < 0.10,
        "|c| / min(tau) of the Mohr-Coulomb fit (informational)",
        anchor_present=has_shear)

    gated = [v for v in verdicts if v["id"] != "D3-cohesion"]
    passed = [v for v in gated if v["status"] == "PASS"]
    failed = [v for v in gated if v["status"] == "FAIL"]
    withheld = [v for v in gated if v["status"] == "WITHHELD"]
    overall = ("FAIL" if failed else ("PASS" if not withheld else "PARTIAL"))
    return {
        "node": prereg["node"],
        "verdicts": verdicts,
        "criteria": list(criteria.values()),
        "gated_passed": len(passed),
        "gated_failed": len(failed),
        "gated_withheld": len(withheld),
        "gated_total": len(gated),
        "overall": overall,
        "honest_note": ("WITHHELD means the anchor produced no adjudicable number in this "
                        "node window; it is neither a pass nor a pass-by-default"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _save(out_dir: str, name: str, payload) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    choices=["laws", "packing", "repose", "shear", "report", "all"])
    ap.add_argument("--out", default="artifacts/P152W1-N04")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--packing-jobs", type=int, default=5)
    ap.add_argument("--repose-particles", type=int, default=300)
    ap.add_argument("--source-mode", default="fixed", choices=["fixed", "rising"])
    args = ap.parse_args(argv)
    out = args.out
    if args.stage in ("laws", "all"):
        _save(out, "laws.json", law_anchors())
        print("laws: done")
    if args.stage in ("packing", "all"):
        _save(out, "packing.json", packing_anchor(jobs=args.packing_jobs))
        print("packing: done")
    if args.stage in ("repose", "all"):
        _save(out, "repose.json", repose_anchor(jobs=args.jobs,
                                               n_particles=args.repose_particles,
                                               source_mode=args.source_mode))
        print("repose: done")
    if args.stage in ("shear", "all"):
        _save(out, "shear.json", shear_anchor(jobs=args.jobs))
        print("shear: done")
    if args.stage in ("report", "all"):
        anchors = {}
        for name in ("laws", "packing", "repose", "shear"):
            path = os.path.join(out, f"{name}.json")
            if os.path.exists(path):
                with open(path) as fh:
                    anchors[name] = json.load(fh)
        verdict = adjudicate(anchors)
        _save(out, "results.json", {"anchors": anchors, "adjudication": verdict})
        print(json.dumps({x["id"]: x["status"] for x in verdict["verdicts"]}, indent=1))
        print("overall:", verdict["overall"],
              f"({verdict['gated_passed']}/{verdict['gated_total']} gated criteria)")
    return 0


if __name__ == "__main__":                       # pragma: no cover
    raise SystemExit(main())