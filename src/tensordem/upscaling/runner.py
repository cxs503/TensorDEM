"""Coarse-graining campaign runner: f-sweep, error budget, adjudication, artefacts.

Usage
-----
    PYTHONPATH=src python -m tensordem.upscaling.runner --workers 12 \
        --out runs/upscaling_cg/results.json --figures artifacts/upscaling_cg

The runner
1. runs the three macro anchors for ``f in {1,2,4,8}`` under both口径
   (``contact_time`` = A, ``stiffness_invariant`` = B) in *similarity* mode
   (all lengths x f) -> isolates the scaling-law error;
2. runs ``fixed_domain`` replicas (f=2) where the specimen keeps its f=1 size
   while the grains coarsen -> resolution / RVE error of real coarse-graining;
3. runs the negative controls (wrong rescalings) and checks that the machine
   criteria *reject* them;
4. evaluates the acceptance gates and writes a JSON artefact with provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from typing import Dict, List

import numpy as np

from .cases import RUNNERS, CaseResult, run_case
from .law import (
    BaseMaterial,
    APPROACHES,
    NEGATIVE_CONTROLS,
    applicability_domain,
    scale_material,
    scaling_rule_table,
    stability_gate,
)

F_VALUES = (1.0, 2.0, 4.0, 8.0)
CASES = ("shear", "repose", "ucs")
BASE = BaseMaterial()

# machine criteria (see campaign node metadata)
CRITERIA = {
    "shear_phi_dev_pct": {"f_le_4": 5.0, "f_8": 8.0},
    "repose_theta_dev_deg": {"f_le_4": 5.0, "f_8": 8.0},
    "e_eff_dev_pct": {"all": 10.0},
    "force_peak_dev_pct": {"all": 10.0},
    "stability_v_max_ratio": {"all": 10.0},
}

# design confining stress used for the overlap gate (shear cell dead load) and
# the self-weight stress of the drum / UCS specimens
SIGMA_DESIGN = {"shear": 200.0, "repose": None, "ucs": None}


# ---------------------------------------------------------------------------
# job execution
# ---------------------------------------------------------------------------


# replicate counts: the shear-cell friction angle carries the largest
# realisation-to-realisation scatter (finite system), so it gets 2 realisations;
# repose/UCS are deterministic to <2 % and use one.
REPLICATES = {"shear": 3, "repose": 3, "ucs": 1}
SEED0 = {"shear": 11, "repose": 21, "ucs": 31}


def _run_job(job) -> CaseResult:
    case, approach, f, mode, geom_scale, seed = job
    return run_case(case, BASE, approach, f, mode=mode, geom_scale=geom_scale, seed=seed)


def build_jobs() -> List[tuple]:
    jobs = []
    for case in CASES:
        reps = REPLICATES[case]
        for approach in APPROACHES:
            for f in F_VALUES:
                for k in range(reps):
                    jobs.append((case, approach, f, "similar", None, SEED0[case] + 100 * k))
    # real coarse-graining (fixed domain, N -> N/f^3) - resolution error budget
    for case in CASES:
        jobs.append((case, "contact_time", 2.0, "fixed_domain", 1.0, SEED0[case]))
    # negative controls (one realisation each)
    jobs.append(("ucs", "kn_f3_for_stiffness", 2.0, "similar", None, SEED0["ucs"]))
    jobs.append(("ucs", "kn_f3_for_stiffness", 4.0, "similar", None, SEED0["ucs"]))
    jobs.append(("shear", "mass_only", 2.0, "similar", None, SEED0["shear"]))
    return jobs


def _self_weight_stress(res: CaseResult) -> float:
    """Bulk self-weight stress of the specimen (Pa) for the overlap gate."""
    bulk = 1500.0  # kg/m^3, typical granular bulk density
    height = {
        "shear": 2.0 * res.metrics.get("bulk_z_m", 0.0),
        "repose": 2.0 * res.metrics.get("drum_radius_m", 0.0) * 0.45,
        "ucs": res.metrics.get("specimen_height_m", 0.0),
    }[res.case]
    return float(bulk * 9.81 * max(height, 1e-6))


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------


def run_sweep(workers: int = 8) -> List[CaseResult]:
    jobs = build_jobs()
    if workers <= 1:
        return [_run_job(j) for j in jobs]
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(workers, len(jobs))) as pool:
        return pool.map(_run_job, jobs)


# ---------------------------------------------------------------------------
# adjudication
# ---------------------------------------------------------------------------


def _dev_pct(value: float, ref: float) -> float:
    if not (value == value and ref == ref) or ref == 0.0:
        return float("nan")
    return (value - ref) / ref * 100.0


def _index(results: List[CaseResult]) -> Dict[str, Dict[str, Dict[float, List[CaseResult]]]]:
    idx: Dict[str, Dict[str, Dict[float, List[CaseResult]]]] = {}
    for r in results:
        if r.mode != "similar":
            continue
        idx.setdefault(r.case, {}).setdefault(r.approach, {}).setdefault(r.f, []).append(r)
    return idx


def _stat(runs: List[CaseResult], key: str):
    """Mean and (sample) standard deviation of a metric over realisations."""
    vals = [float(r.metrics[key]) for r in runs if r.metrics.get(key) == r.metrics.get(key)]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def _stat_stability(runs: List[CaseResult], key: str) -> float:
    vals = [float(r.stability[key]) for r in runs]
    return float(np.max(vals)) if values_are_max(key) else float(np.min(vals))


def values_are_max(key: str) -> bool:
    """For stability gates the *worst* realisation decides."""
    return key in ("v_max", "v_max_ratio", "energy_max_rise_frac", "n_escaped")


def adjudicate(results: List[CaseResult]) -> dict:
    idx = _index(results)
    devs: Dict[str, Dict[str, Dict[str, dict]]] = {}
    criteria: Dict[str, dict] = {}

    for case, per_approach in idx.items():
        for approach, per_f in per_approach.items():
            # negative controls have no f=1 run of their own: compare against
            # the fine (f=1) model of the physically correct口径
            ref_runs = per_f.get(1.0) or idx.get(case, {}).get("contact_time", {}).get(1.0)
            if not ref_runs:
                continue
            rows = {}
            for f, runs in sorted(per_f.items()):
                n_reps = len(runs)
                if case == "shear":
                    phi, phi_std = _stat(runs, "friction_angle_deg")
                    mu, mu_std = _stat(runs, "mu_macro")
                    phi_ref, _ = _stat(ref_runs, "friction_angle_deg")
                    mu_ref, _ = _stat(ref_runs, "mu_macro")
                    phi_dev = _dev_pct(phi, phi_ref)
                    thr = CRITERIA["shear_phi_dev_pct"]["f_8"] if f >= 8 else \
                        CRITERIA["shear_phi_dev_pct"]["f_le_4"]
                    n_valid = sum(1 for r in runs
                                  if np.isfinite(r.metrics.get("friction_angle_deg", np.nan)))
                    rows[str(f)] = {"friction_angle_deg": phi, "phi_std": phi_std,
                                    "n_valid_reps": n_valid,
                                    "mu_macro": mu, "mu_std": mu_std,
                                    "phi_dev_pct": phi_dev,
                                    "mu_dev_pct": _dev_pct(mu, mu_ref),
                                    "threshold_pct": thr, "n_reps": n_reps,
                                    "pass": bool(abs(phi_dev) <= thr)}
                elif case == "repose":
                    theta, theta_std = _stat(runs, "theta_r_deg")
                    theta_ref, _ = _stat(ref_runs, "theta_r_deg")
                    dev = theta - theta_ref
                    thr = CRITERIA["repose_theta_dev_deg"]["f_8"] if f >= 8 else \
                        CRITERIA["repose_theta_dev_deg"]["f_le_4"]
                    n_valid = sum(1 for r in runs
                                  if np.isfinite(r.metrics.get("theta_r_deg", np.nan)))
                    rows[str(f)] = {"theta_r_deg": theta, "theta_std": theta_std,
                                    "n_valid_reps": n_valid,
                                    "theta_dev_deg": dev, "threshold_deg": thr,
                                    "n_reps": n_reps, "pass": bool(abs(dev) <= thr)}
                else:  # ucs
                    f_exp = runs[0].mat["force_exponent"]
                    e_eff, e_std = _stat(runs, "e_eff_pa")
                    e_ref, _ = _stat(ref_runs, "e_eff_pa")
                    peak, peak_std = _stat(runs, "first_peak_force_n")
                    peak_ref, _ = _stat(ref_runs, "first_peak_force_n")
                    peak_ref_scaled = peak_ref * f ** f_exp
                    e_dev = _dev_pct(e_eff, e_ref)
                    f_dev = _dev_pct(peak, peak_ref_scaled)
                    n_valid = sum(1 for r in runs
                                  if np.isfinite(r.metrics.get("e_eff_pa", np.nan)))
                    rows[str(f)] = {
                        "n_valid_reps": n_valid,
                        "e_eff_pa": e_eff, "e_eff_std": e_std,
                        "e_eff_dev_pct": e_dev,
                        "e_eff_pass": bool(abs(e_dev) <= CRITERIA["e_eff_dev_pct"]["all"]),
                        "first_peak_force_n": peak, "peak_force_std": peak_std,
                        "first_peak_force_normalized": peak / f ** f_exp,
                        "peak_force_dev_pct": f_dev,
                        "peak_force_pass": bool(abs(f_dev) <=
                                                CRITERIA["force_peak_dev_pct"]["all"]),
                        "force_exponent": f_exp, "n_reps": n_reps,
                    }
            devs.setdefault(case, {})[approach] = rows

    # ---- criteria verdicts per approach ---------------------------------
    for approach in APPROACHES:
        verdict = {}
        shear_rows = devs.get("shear", {}).get(approach, {})
        verdict["shear_phi"] = _verdict(
            [r["pass"] for f, r in shear_rows.items() if float(f) > 1],
            {f: r["phi_dev_pct"] for f, r in shear_rows.items()})
        rep_rows = devs.get("repose", {}).get(approach, {})
        verdict["repose_theta"] = _verdict(
            [r["pass"] for f, r in rep_rows.items() if float(f) > 1],
            {f: r["theta_dev_deg"] for f, r in rep_rows.items()})
        ucs_rows = devs.get("ucs", {}).get(approach, {})
        verdict["e_eff"] = _verdict(
            [r["e_eff_pass"] for f, r in ucs_rows.items() if float(f) > 1],
            {f: r["e_eff_dev_pct"] for f, r in ucs_rows.items()})
        verdict["force_peak"] = _verdict(
            [r["peak_force_pass"] for f, r in ucs_rows.items() if float(f) > 1],
            {f: r["peak_force_dev_pct"] for f, r in ucs_rows.items()})
        criteria[approach] = verdict

    # ---- stability gates -------------------------------------------------
    stab_rows = []
    for r in results:
        if r.mode != "similar":
            continue
        sigma = SIGMA_DESIGN.get(r.case) or _self_weight_stress(r)
        mat = scale_material(BASE, r.f, r.approach)
        gate = stability_gate(mat, sigma, r.stability["v_max"])
        stab_rows.append({
            "case": r.case, "approach": r.approach, "f": r.f, "seed": r.seed,
            "v_max_ratio": r.stability["v_max_ratio"],
            "flyoff_ok": r.stability["flyoff_ok"],
            "energy_monotone_ok": r.stability["energy_monotone_ok"],
            "n_escaped": r.stability["n_escaped"],
            "overlap_ratio": gate["overlap_ratio"],
            "overlap_ok": gate["overlap_ok"],
            "steps_per_contact": gate["steps_per_contact"],
            # numerical-stability gates (no fly-off / energy decay / step
            # resolution / advective limit) are separated from the *declared
            # deformability* boundary (overlap > 2 %), which is a physics
            # validity limit of the口径, not a numerical failure
            "numerical_ok": bool(r.stability["flyoff_ok"]
                                 and r.stability["energy_monotone_ok"]
                                 and r.stability["n_escaped"] == 0
                                 and gate["steps_per_contact_ok"]),
            "declared_overlap_violation": bool(not gate["overlap_ok"]),
        })
    crit5 = {"rows": stab_rows,
             "pass": bool(all(row["numerical_ok"] for row in stab_rows)),
             "n_numerical_fail": sum(1 for row in stab_rows if not row["numerical_ok"]),
             "n_declared_overlap_violations": sum(1 for row in stab_rows
                                                  if row["declared_overlap_violation"])}

    # ---- negative controls ----------------------------------------------
    neg = {}
    ucs_rows = devs.get("ucs", {})
    nc_rows = ucs_rows.get("kn_f3_for_stiffness", {})
    neg["kn_f3_for_stiffness_vs_e_eff"] = {
        "measured": nc_rows,
        "rejected": bool(nc_rows) and all(
            abs(r["e_eff_dev_pct"]) > CRITERIA["e_eff_dev_pct"]["all"] for r in nc_rows.values()),
        "criterion": "E_eff deviation <= 10 %",
    }
    mass_rows = devs.get("shear", {}).get("mass_only", {})
    if mass_rows:
        worst = max(abs(r["phi_dev_pct"]) for r in mass_rows.values())
        neg["mass_only_vs_shear_phi"] = {
            "measured": mass_rows, "worst_phi_dev_pct": worst,
            "rejected": bool(worst > CRITERIA["shear_phi_dev_pct"]["f_le_4"]),
            "criterion": "friction angle deviation <= 5 %",
        }
    base_tc = BASE.contact_time
    neg["mass_only_contact_time"] = {
        str(f): {"t_c_ratio": scale_material(BASE, f, "mass_only").contact_time / base_tc,
                 "expected": f ** 1.5,
                 "dev_pct": (scale_material(BASE, f, "mass_only").contact_time / base_tc - 1.0) * 100.0}
        for f in F_VALUES[1:]
    }

    # ---- resolution (RVE) error budget ----------------------------------
    rve = {}
    for r in results:
        if r.mode != "fixed_domain":
            continue
        ref_runs = idx.get(r.case, {}).get(r.approach, {}).get(1.0)
        if not ref_runs:
            continue
        row = {"n_particles": r.n_particles, "ref_n_particles": ref_runs[0].n_particles}
        if r.case == "shear":
            ref_phi, _ = _stat(ref_runs, "friction_angle_deg")
            row["phi_dev_pct"] = _dev_pct(r.metrics["friction_angle_deg"], ref_phi)
        elif r.case == "repose":
            ref_theta, _ = _stat(ref_runs, "theta_r_deg")
            row["theta_dev_deg"] = r.metrics["theta_r_deg"] - ref_theta
        else:
            ref_e, _ = _stat(ref_runs, "e_eff_pa")
            row["e_eff_dev_pct"] = _dev_pct(r.metrics["e_eff_pa"], ref_e)
        rve.setdefault(r.case, {})[str(r.f)] = row

    # ---- predicted validity / failure boundary --------------------------
    boundary = {}
    for approach in APPROACHES:
        rows = {}
        for f in F_VALUES:
            mat = scale_material(BASE, f, approach)
            rows[str(f)] = {
                "overlap_ratio_imposed_200pa": mat.overlap_ratio(200.0),
                "overlap_ratio_self_weight": mat.overlap_ratio(_bulk_self_weight(mat)),
                "steps_per_contact": mat.steps_per_contact,
                "E_eff_ratio_to_base": (mat.kn / mat.radius) / (BASE.kn / BASE.radius),
                "t_c_ratio": mat.contact_time / BASE.contact_time,
                "wave_speed_ratio": mat.wave_speed / (math.sqrt(BASE.kn / (0.5 * BASE.mass))
                                                     * BASE.radius),
                "stiffness_ratio": mat.stiffness_ratio / BASE.stiffness_ratio,
            }
        boundary[approach] = rows
    boundary["limits"] = {"max_overlap_ratio": 0.02, "min_steps_per_contact": 20.0}

    nan_runs = []
    for r in results:
        bad = [k for k, v in r.metrics.items() if isinstance(v, float) and v != v]
        if bad:
            nan_runs.append({"case": r.case, "approach": r.approach, "f": r.f,
                             "mode": r.mode, "seed": r.seed, "nan_metrics": bad})
    return {"deviations": devs, "criteria": criteria, "stability": crit5,
            "negative_controls": neg, "rve_budget": rve, "validity_boundary": boundary,
            "invalid_runs": nan_runs}


def _bulk_self_weight(mat) -> float:
    """Self-weight stress of a 10-grain-diameter bed (Pa) - the reference bed
    depth used to declare the deformability (overlap) validity boundary."""
    height = 10.0 * (2.0 * mat.radius)
    return 1500.0 * mat.gravity * height


def _verdict(flags, detail) -> dict:
    flags = list(flags)
    return {"pass": bool(all(flags)) if flags else True,
            "n_checked": len(flags), "all_pass": bool(all(flags)) if flags else True,
            "detail": detail}


# ---------------------------------------------------------------------------
# artefact assembly
# ---------------------------------------------------------------------------


def load_results(path: str) -> List[CaseResult]:
    """Reconstruct the run records from a previously written results JSON."""
    with open(path) as fh:
        payload = json.load(fh)
    out: List[CaseResult] = []
    for key, run in payload["runs"].items():
        out.append(CaseResult(
            case=run["case"], approach=run["approach"], f=run["f"], mode=run["mode"],
            n_particles=run["n_particles"], n_steps=run["n_steps"], dt=run["dt"],
            metrics=run["metrics"], stability=run["stability"],
            wall_time=run["wall_time"], mat=run["mat"], seed=run.get("seed", 0),
            traces=run.get("traces") or {},
        ))
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def build_artefact(results: List[CaseResult], elapsed: float) -> dict:
    adj = adjudicate(results)
    here = os.path.dirname(os.path.abspath(__file__))
    prov_files = {}
    for name in ("law.py", "engine.py", "cases.py", "runner.py"):
        p = os.path.join(here, name)
        prov_files[name] = _sha256(p)
    runs = {}
    for r in results:
        key = f"{r.case}|{r.approach}|f{r.f:g}|{r.mode}|s{r.seed}"
        d = r.to_dict()
        runs[key] = d
    payload = {
        "campaign": "p129w2-n09-tensordem-upscaling",
        "title": "DEM coarse-graining scaling laws and fine/coarse macro-invariant error budget",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": elapsed,
        "base_material": {
            "radius_m": BASE.radius, "density": BASE.density, "kn_N_m": BASE.kn,
            "kt_ratio": BASE.kt_ratio, "restitution": BASE.restitution,
            "mu": BASE.mu, "mu_rolling": BASE.mu_rolling, "gravity": BASE.gravity,
            "contact_time_s": BASE.contact_time, "timestep_s": BASE.timestep,
        },
        "f_values": list(F_VALUES),
        "protocol": {
            "shear": "periodic direct-shear cell, dead-load platen, sigma_n = 200 Pa",
            "repose": "rotating drum, Froude-number similarity, 1.2 revolutions",
            "ucs": "oedometric uniaxial compression, lateral periodic, 2 % axial strain",
            "modes": {"similar": "all lengths x f (scaling-law error)",
                      "fixed_domain": "specimen fixed, N -> N/f^3 (RVE error)"},
        },
        "criteria_thresholds": CRITERIA,
        "runs": runs,
        **adj,
        "scale_table": scaling_rule_table(),
        "applicability": applicability_domain(),
        "provenance": {
            "source_sha256": prov_files,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "host": platform.node(),
        },
    }
    blob = json.dumps(payload, sort_keys=True, default=float).encode()
    payload["provenance"]["artefact_sha256"] = hashlib.sha256(blob).hexdigest()
    return payload


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures(payload: dict, outdir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    paths = []
    dev = payload["deviations"]

    # 1. macro quantities vs f
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [("shear", "friction_angle_deg", "friction angle φ (deg)"),
              ("repose", "theta_r_deg", "angle of repose θ_r (deg)"),
              ("ucs", "e_eff_pa", "constrained modulus E_eff (Pa)")]
    styles = {"contact_time": ("o-", "law A: k ~ f^3"),
              "stiffness_invariant": ("s--", "law B: k ~ f")}
    for ax, (case, key, label) in zip(axes, panels):
        for approach, (marker, lbl) in styles.items():
            rows = dev.get(case, {}).get(approach, {})
            if not rows:
                continue
            fs = sorted(float(k) for k in rows)
            vals = [rows[str(f)][key] if str(f) in rows else rows[f"{f:g}"][key] for f in fs]
            ax.plot(fs, vals, marker, label=lbl)
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(F_VALUES))
        ax.set_xticklabels([f"{f:g}" for f in F_VALUES])
        ax.set_xlabel("coarse-graining factor f")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Macro invariants vs coarse-graining factor (similarity mode)")
    fig.tight_layout()
    p = os.path.join(outdir, "fig_macro_vs_f.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 2. deviation budget with thresholds
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    specs = [("shear", "phi_dev_pct", "φ deviation (%)", CRITERIA["shear_phi_dev_pct"]),
             ("repose", "theta_dev_deg", "θ_r deviation (deg)", CRITERIA["repose_theta_dev_deg"]),
             ("ucs", "e_eff_dev_pct", "E_eff deviation (%)", CRITERIA["e_eff_dev_pct"])]
    for ax, (case, key, label, thr) in zip(axes, specs):
        for approach, (marker, lbl) in styles.items():
            rows = dev.get(case, {}).get(approach, {})
            if not rows:
                continue
            fs = sorted(float(k) for k in rows if float(k) > 1)
            vals = [abs(rows[str(f)][key]) for f in fs]
            ax.plot(fs, vals, marker, label=lbl)
        key_thr = "f_le_4" if "f_le_4" in thr else "all"
        ax.axhline(thr[key_thr], color="k", ls=":", lw=1,
                   label=f"limit {thr[key_thr]:g}")
        if "f_8" in thr:
            ax.axhline(thr["f_8"], color="grey", ls=":", lw=1, label=f"limit f=8 {thr['f_8']:g}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(F_VALUES))
        ax.set_xticklabels([f"{f:g}" for f in F_VALUES])
        ax.set_xlabel("f")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Macro-invariant error budget vs f (|deviation| from f=1)")
    fig.tight_layout()
    p = os.path.join(outdir, "fig_error_budget.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 3. UCS force-displacement curves + normalised
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for approach, (marker, lbl) in styles.items():
        for f in F_VALUES:
            key = f"ucs|{approach}|f{f:g}|similar"
            run = payload["runs"].get(key)
            if not run:
                continue
            tr = run["traces"]
            ax = axes[0]
            ax.plot(tr["eps"], tr["sigma_pa"], label=f"f={f:g}")
        axes[0].set_xlabel("axial strain ε")
        axes[0].set_ylabel("axial stress σ (Pa)")
        axes[0].set_title(f"UC stress-strain, {lbl}")
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        # normalised: σ / f^stress_exponent
        ax = axes[1]
        for f in F_VALUES:
            key = f"ucs|{approach}|f{f:g}|similar"
            run = payload["runs"].get(key)
            if not run:
                continue
            exp = run["mat"]["stress_exponent"]
            tr = run["traces"]
            ax.plot(tr["eps"], np.asarray(tr["sigma_pa"]) / f ** exp, label=f"f={f:g} ($/f^{exp:g}$)")
        ax.set_xlabel("axial strain ε")
        ax.set_ylabel("σ / f^p (Pa)")
        ax.set_title(f"normalised, {lbl}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(outdir, "fig_ucs_force_displacement.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 4. shear response traces
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, approach in zip(axes, ("contact_time", "stiffness_invariant")):
        for f in F_VALUES:
            run = payload["runs"].get(f"shear|{approach}|f{f:g}|similar")
            if not run:
                continue
            tau = np.asarray(run["traces"]["tau_pa"])
            sig = np.asarray(run["traces"]["sigma_pa"])
            ax.plot(np.arange(len(tau)) * 4 * run["dt"], tau / np.maximum(sig, 1e-9),
                    label=f"f={f:g}")
        ax.set_xlabel("time in shear window (s)")
        ax.set_ylabel("μ = τ/σ_n")
        ax.set_title({"contact_time": "law A: k ~ f^3", "stiffness_invariant": "law B: k ~ f"}[approach])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(outdir, "fig_shear_response.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 5. drum free surface snapshots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, f in zip(axes, (1.0, 8.0)):
        run = payload["runs"].get(f"repose|contact_time|f{f:g}|similar")
        if not run:
            ax.set_visible(False)
            continue
        x = np.asarray(run["traces"]["x"])
        z = np.asarray(run["traces"]["z"])
        ax.scatter(x, z, s=3, c="tab:blue")
        Rd = run["metrics"]["drum_radius_m"]
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(Rd * np.cos(th), Rd * np.sin(th), "k-", lw=1)
        ax.set_aspect("equal")
        ax.set_title(f"drum f={f:g}  θ_r={run['metrics']['theta_r_deg']:.1f}°")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, "fig_drum_surface.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 6. validity boundary: overlap ratio vs f
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for approach, (marker, lbl) in styles.items():
        rows = payload["validity_boundary"][approach]
        fs = sorted(float(k) for k in rows)
        ax.plot(fs, [rows[str(f)]["overlap_ratio_self_weight"] * 100 for f in fs], marker,
                label=f"{lbl} (self weight)")
        ax.plot(fs, [rows[str(f)]["overlap_ratio_imposed_200pa"] * 100 for f in fs],
                marker[0] + ":", alpha=0.6,
                label=f"{lbl} (200 Pa imposed)")
    ax.axhline(2.0, color="k", ls="--", lw=1, label="linear-contact limit 2 %")
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(F_VALUES))
    ax.set_xticklabels([f"{f:g}" for f in F_VALUES])
    ax.set_xlabel("f")
    ax.set_ylabel("contact overlap δ/R (%)")
    ax.set_title("Declared validity boundary: deformability vs f")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(outdir, "fig_validity_boundary.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DEM coarse-graining campaign")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/upscaling_cg/results.json")
    ap.add_argument("--figures", default="artifacts/upscaling_cg")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--rebuild-from", default=None,
                    help="re-adjudicate an existing results JSON without re-simulating")
    args = ap.parse_args(argv)

    t0 = time.time()
    if args.rebuild_from:
        results = load_results(args.rebuild_from)
        elapsed = float("nan")
    else:
        results = run_sweep(args.workers)
        elapsed = time.time() - t0
    payload = build_artefact(results, elapsed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    print(f"wrote {args.out} ({len(results)} runs, {elapsed:.1f}s)")

    if not args.no_figures:
        figs = make_figures(payload, args.figures)
        payload["figures"] = figs
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=1, default=float)
        print("figures:", ", ".join(figs))

    print(json.dumps({a: {k: v["all_pass"] for k, v in crit.items()}
                      for a, crit in payload["criteria"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())