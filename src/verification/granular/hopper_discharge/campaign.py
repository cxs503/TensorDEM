"""Campaign driver + acceptance criteria for the hopper verification surface (m7751).

The campaign is a fixed, reproducible matrix of DEM runs:

===============  ==========================================================
face             runs
===============  ==========================================================
``discharge``    D0 = 0.15 / 0.20 / 0.25 m  x  3 seeds   (Beverloo)
``discharge``    fill height 0.3 / 0.5 m at D0 = 0.20 m x 2 seeds (C7)
``janssen``      mu_w = 0.3 / 0.4 / 0.5  x  3 seeds   (wall stress)
``repose``       mu = 0.1 .. 0.5 (5 levels) x 3 seeds (repose angle)
``repose``       mu = 0 frictionless negative control x 3 seeds (C8)
``dtstudy``      closed cylinder at dt, dt/2, dt/4 (energy ledger, C6)
===============  ==========================================================

Each run writes one JSON record; :func:`evaluate_criteria` reduces the records
to the machine-checkable criteria C1..C9 and never reads a value that was not
measured by :mod:`kernel`.

Note: Geometries are scaled down from the prototype (10m×5m×8m) to make the
campaign tractable (~200-400 particles per case). The physics is scale-free 
for D_0/d >> 1 (here D_0/d ≈ 3-5, marginal but acceptable for verification).
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from .kernel import GranularParams, HopperCase, HopperGeometry, simulate
from .measures import (
    beverloo_fit,
    beverloo_scaling_fit,
    janssen_analytic,
    janssen_fit,
    janssen_saturation,
    repose_angle,
    saturation_metrics,
)

# ---------------------------------------------------------------------------
# Campaign specification
# ---------------------------------------------------------------------------

#: Reference data of the node (full-scale prototype), kept for provenance only.
PROTOTYPE = {
    "bin_m": [10.0, 5.0, 8.0],
    "hopper_half_angle_deg": 30.0,
    "orifice_diameter_m": [0.8, 1.0, 1.2],
    "particle_diameter_m": 0.05,
    "particle_density_kg_m3": 2500.0,
    "kn_N_m": 1.0e6,
    "kt_over_kn": 0.5,
    "restitution": 0.5,
    "mu": 0.4,
    "target_particle_count": 5.0e4,
}

SEEDS = (11, 22, 33)
DISCHARGE_D0 = (0.20, 0.25, 0.30)
JANSSEN_MU_W = (0.3, 0.4, 0.5)
REPOSE_MU = (0.1, 0.2, 0.3, 0.4, 0.5)

# Small geometries for tractable runtimes (~200-400 particles per case)
DISCHARGE_GEOM = dict(r_bin=0.20, half_angle_deg=30.0, fill_height=0.35)
JANSSEN_GEOM = dict(kind="cylinder", r_bin=0.15, fill_height=0.60)
REPOSE_GEOM = dict(kind="cylinder", r_bin=0.15, fill_height=0.25)
DTS_GEOM = dict(kind="cylinder", r_bin=0.12, fill_height=0.18)

#: Number of leading flow-rate windows dropped from the steady-rate average.
DISCHARGE_SKIP_WINDOWS = 1

BASE_PARAMS = dict(radius=0.025, density=2500.0, kn=1.0e6, kt_ratio=0.5,
                   restitution=0.5, mu=0.4, mu_rolling=0.0, gravity=9.81)


def campaign_plan() -> list[dict]:
    """Full, deterministic list of the runs that make up the surface."""
    plan: list[dict] = []

    for D0 in DISCHARGE_D0:
        for seed in SEEDS:
            plan.append({
                "case_id": f"discharge_D{int(round(D0 * 100)):03d}_s{seed}",
                "face": "discharge", "seed": seed, "orifice_diameter_m": D0,
            })
    for fill in (0.30, 0.50):
        for seed in SEEDS[:2]:
            plan.append({
                "case_id": f"discharge_D020_fill{int(round(fill * 100)):03d}_s{seed}",
                "face": "discharge", "seed": seed, "orifice_diameter_m": 0.20,
                "fill_height_m": fill, "convergence_study": True,
            })

    for mu_w in JANSSEN_MU_W:
        for seed in SEEDS:
            plan.append({
                "case_id": f"janssen_muw{int(round(mu_w * 100)):03d}_s{seed}",
                "face": "janssen", "seed": seed, "mu_wall": mu_w,
            })

    for mu in REPOSE_MU:
        for seed in SEEDS:
            plan.append({
                "case_id": f"repose_mu{int(round(mu * 100)):03d}_s{seed}",
                "face": "repose", "seed": seed, "mu": mu,
            })
    for seed in SEEDS:
        plan.append({
            "case_id": f"repose_negctl_s{seed}",
            "face": "repose", "seed": seed, "mu": 0.0, "negative_control": True,
        })
    # particle-count convergence for the repose face (smaller column)
    for seed in SEEDS[:2]:
        plan.append({
            "case_id": f"repose_mu040_small_s{seed}",
            "face": "repose", "seed": seed, "mu": 0.4,
            "column_radius_m": 0.10, "convergence_study": True,
        })

    for tag, div in (("full", 1), ("half", 2), ("quarter", 4)):
        plan.append({"case_id": f"dtstudy_{tag}", "face": "dtstudy",
                     "dt_divisor": div})
    return plan


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------

def _build_case(entry: dict) -> HopperCase:
    face = entry["face"]
    params = dict(BASE_PARAMS)
    if "mu" in entry:
        params["mu"] = float(entry["mu"])
    seed = int(entry.get("seed", 0))

    if face == "discharge":
        geom = dict(DISCHARGE_GEOM)
        geom["orifice_radius"] = 0.5 * float(entry["orifice_diameter_m"])
        if "fill_height_m" in entry:
            geom["fill_height"] = float(entry["fill_height_m"])
        return HopperCase(geometry=HopperGeometry(**geom),
                          params=GranularParams(**params), mu_wall=0.4,
                          seed=seed, mode="discharge", settle_time=0.10,
                          window=0.05, n_windows=5, packing_gap=0.01,
                          jitter=0.0015, label=entry["case_id"])
    if face == "janssen":
        return HopperCase(geometry=HopperGeometry(**JANSSEN_GEOM),
                          params=GranularParams(**params),
                          mu_wall=float(entry["mu_wall"]), seed=seed,
                          mode="static", settle_time=0.10, sample_time=0.05,
                          slab_height=0.08, packing_gap=0.01, jitter=0.0015,
                          label=entry["case_id"])
    if face == "repose":
        geom = dict(REPOSE_GEOM)
        if "column_radius_m" in entry:
            geom["r_bin"] = float(entry["column_radius_m"])
        return HopperCase(geometry=HopperGeometry(**geom),
                          params=GranularParams(**params),
                          mu_wall=float(entry.get("mu", 0.4)),
                          seed=seed, mode="heap", wall_hold_time=0.04,
                          wall_lift_time=0.03, free_settle_time=0.20,
                          packing_gap=0.01, jitter=0.0015, label=entry["case_id"])
    if face == "dtstudy":
        base = GranularParams(**params)
        return HopperCase(geometry=HopperGeometry(**DTS_GEOM),
                          params=base, mu_wall=0.4, seed=11, mode="static",
                          settle_time=0.03, sample_time=0.015,
                          slab_height=0.06, packing_gap=0.01,
                          dt=base.timestep / int(entry["dt_divisor"]),
                          label=entry["case_id"])
    raise ValueError(f"unknown face {face!r}")


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_case(entry: dict, *, keep_positions: bool = False) -> dict:
    """Run one campaign entry and return its JSON-serialisable record."""
    case = _build_case(entry)
    out = simulate(case, return_positions=keep_positions or entry["face"] == "repose")
    record = {
        "case_id": entry["case_id"],
        "face": entry["face"],
        "entry": entry,
        "n_particles": out["n_initial"],
        "bulk_density_kg_m3": out["initial_bulk_density_kg_m3"],
        "packing_fraction": out["packing_fraction"],
        "dt_s": out["dt_s"],
        "steps": out["steps"],
        "wall_clock_s": out["wall_clock_s"],
        "mass_ledger_residual": out["mass_ledger_residual"],
        "energy_ledger_residual": out["energy_ledger_residual"],
        "energy_initial_j": out["energy_initial_j"],
    }
    if entry["face"] == "discharge":
        record["flow_rate_kg_s"] = out["flow_rate_kg_s"]
        record["flow_rate_std_kg_s"] = out["flow_rate_std_kg_s"]
        record["flow_rate_windows_kg_s"] = out["flow_rate_windows_kg_s"]
        record["window_s"] = out["window_s"]
        record["n_discharged"] = out["n_discharged"]
    elif entry["face"] == "janssen":
        record["wall_stress_pa"] = out["wall_stress_pa"]
        record["slab_center_z_m"] = out["slab_center_z_m"]
        record["bottom_pressure_pa"] = out["bottom_pressure_pa"]
    elif entry["face"] == "repose":
        pos = out.get("positions")
        alive = out.get("alive")
        if pos is not None and alive is not None:
            idx = np.nonzero(alive)[0]
            theta_result = repose_angle(pos[idx], case.params.radius)
            theta = theta_result["theta_r_deg"]
            record["repose"] = {"theta_r_deg": theta}
    record["case"] = out["case"]
    return record


def run_face(face: str, out_dir: Path) -> list[Path]:
    """Run all cases of one face and write JSON records."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = [e for e in campaign_plan() if e["face"] == face]
    written = []
    for entry in plan:
        rec = run_case(entry)
        p = out_dir / f"{entry['case_id']}.json"
        with open(p, "w") as f:
            json.dump(rec, f, indent=2, default=str)
        written.append(p)
    return written


def load_records(out_dir: Path) -> dict:
    """Load all JSON records from a directory."""
    records = {}
    for p in sorted(out_dir.glob("*.json")):
        with open(p) as f:
            rec = json.load(f)
        records[rec["case_id"]] = rec
    return records


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------

def _criterion(tag: str, value, threshold: str, passed: bool, detail: str) -> dict:
    return {"tag": tag, "value": value, "threshold": threshold,
            "passed": bool(passed), "detail": detail}


def _mean_std(vals: list) -> tuple:
    arr = np.array([v for v in vals if v is not None and np.isfinite(v)])
    if len(arr) == 0:
        return float("nan"), float("nan"), 0
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, int(len(arr))


def _steady(rec: dict) -> float:
    """Mean flow rate skipping the first transient windows."""
    windows = rec.get("flow_rate_windows_kg_s", [])
    if len(windows) <= DISCHARGE_SKIP_WINDOWS:
        return rec.get("flow_rate_kg_s", float("nan"))
    return float(np.mean(windows[DISCHARGE_SKIP_WINDOWS:]))


def evaluate_criteria(records: dict) -> dict:
    """Reduce per-case records to the machine-checkable criteria C1..C9."""
    criteria: list[dict] = []
    summary: dict = {}

    def by_face(face: str) -> list:
        return [r for r in records.values() if r["face"] == face]

    # -- C1 repose angle ---------------------------------------------------
    repose = [r for r in by_face("repose") if not r.get("entry", {}).get("negative_control")]
    by_mu: dict[float, list] = {}
    for r in repose:
        mu = float(r["entry"]["mu"])
        theta = r.get("repose", {}).get("theta_r_deg")
        if theta is not None and np.isfinite(theta):
            by_mu.setdefault(mu, []).append(theta)
    repose_summary = {}
    for mu in sorted(by_mu):
        mean, std, n = _mean_std(by_mu[mu])
        repose_summary[str(mu)] = {"mu": mu, "theta_r_mean_deg": mean,
                                    "theta_r_std_deg": std, "n": n}
    summary["repose"] = repose_summary

    mus = sorted(by_mu.keys())
    thetas = [_mean_std(by_mu[mu])[0] for mu in mus]
    X = np.array(mus)
    Y = np.array(thetas)
    if len(mus) >= 2:
        A = np.stack([X, np.ones_like(X)], axis=1)
        (slope, intercept), *_ = np.linalg.lstsq(A, Y, rcond=None)
        pred = slope * X + intercept
        ss_res = float(((Y - pred)**2).sum())
        ss_tot = float(((Y - Y.mean())**2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        r2 = float("nan")
    
    # Check monotonicity
    monotonic = all(thetas[i+1] > thetas[i] for i in range(len(thetas)-1))
    
    # Check deviation from tan^-1(mu)
    deviations = []
    for mu, theta in zip(mus, thetas):
        theta_analytic = math.degrees(math.atan(mu))
        dev = abs(theta - theta_analytic)
        deviations.append(dev)
    max_dev = max(deviations) if deviations else float("nan")
    
    criteria.append(_criterion(
        "C1a_repose_monotonic", monotonic, "monotonic increase",
        monotonic, "theta_r increases with mu"))
    criteria.append(_criterion(
        "C1b_repose_R2", r2, "> 0.95",
        bool(np.isfinite(r2) and r2 > 0.95),
        f"linear fit R^2 = {r2:.3f}"))
    criteria.append(_criterion(
        "C1c_repose_deviation", max_dev, "< 10 deg",
        bool(np.isfinite(max_dev) and max_dev < 10.0),
        f"|theta_r - tan^-1(mu)| max over mu = {max_dev:.1f} deg"))

    # -- C2 Beverloo scaling -----------------------------------------------
    discharge = by_face("discharge")
    by_D0: dict[float, list] = {}
    for r in discharge:
        if r["entry"].get("convergence_study"):
            continue
        D0 = float(r["entry"]["orifice_diameter_m"])
        W = _steady(r)
        if np.isfinite(W):
            by_D0.setdefault(D0, []).append(W)
    
    D0_vals = sorted(by_D0.keys())
    W_means = [_mean_std(by_D0[D0])[0] for D0 in D0_vals]
    W_stds = [_mean_std(by_D0[D0])[1] for D0 in D0_vals]
    
    scaling = beverloo_scaling_fit(np.array(W_means), np.array(D0_vals), d=0.05)
    r2_bev = scaling["D_pow_2.5"]["r2"]
    
    summary["beverloo"] = {
        "D0_m": D0_vals,
        "flow_rate_mean_kg_s": W_means,
        "flow_rate_std_kg_s": W_stds,
        "scaling": scaling,
    }
    criteria.append(_criterion(
        "C2_beverloo_scaling_R2", r2_bev, "> 0.95",
        bool(np.isfinite(r2_bev) and r2_bev > 0.95),
        f"W vs D_0^(5/2) linear fit R^2 = {r2_bev:.3f}"))

    # -- C3 Beverloo constant C --------------------------------------------
    rho_b = discharge[0]["bulk_density_kg_m3"] if discharge else 1500.0
    g = 9.81
    d = 0.05
    fit = beverloo_fit(np.array(W_means), np.array(D0_vals), d, rho_b, g)
    C = fit["C_joint"]
    rel_err_C = abs(C - 0.58) / 0.58
    
    summary["beverloo_constant"] = fit
    criteria.append(_criterion(
        "C3_beverloo_C_constant", C, "0.58 ± 20%",
        bool(np.isfinite(C) and rel_err_C < 0.20),
        f"C = {C:.3f}, rel_err = {rel_err_C:.1%}"))

    # -- C4 Janssen --------------------------------------------------------
    janssen = by_face("janssen")
    prof: dict[float, list] = {}
    for r in janssen:
        mu_w = float(r["entry"]["mu_wall"])
        prof.setdefault(mu_w, []).append(r)
    
    R_h = 0.5 * JANSSEN_GEOM["r_bin"]
    K = 0.4  # typical Janssen ratio
    sat_devs = []
    sat_flags = []
    janssen_rows = {}
    
    for mu_w in sorted(prof):
        rows = prof[mu_w]
        rho_b_j = float(np.mean([r["bulk_density_kg_m3"] for r in rows]))
        sigma_sat_analytic = janssen_saturation(rho_b_j, g, R_h, mu_w)
        
        # Average the profiles
        grid = rows[0]["slab_center_z_m"]
        sigma_stack = np.array([r["wall_stress_pa"] for r in rows])
        sigma_mean = sigma_stack.mean(axis=0)
        sigma_std = sigma_stack.std(axis=0, ddof=1)
        
        # Fit the profile
        z_arr = np.array(grid)
        try:
            fit_j = janssen_fit(z_arr, sigma_mean)
        except ValueError:
            fit_j = {"sigma_sat_fit_pa": float("nan"), "z0_fit_m": float("nan"),
                     "r2": float("nan"), "prediction_pa": [0.0]*len(grid)}
        
        sat_metrics = saturation_metrics(z_arr, sigma_mean, ref_depth=0.75*z_arr.max())
        dev_sat = abs(fit_j["sigma_sat_fit_pa"] - sigma_sat_analytic) / sigma_sat_analytic
        sat_devs.append(dev_sat)
        sat_flags.append(sat_metrics["saturated"])
        
        janssen_rows[str(mu_w)] = {
            "mu_wall": mu_w, "R_h_m": R_h,
            "rho_b_kg_m3": rho_b_j,
            "sigma_sat_analytic_pa": sigma_sat_analytic,
            "sigma_sat_fit_pa": fit_j["sigma_sat_fit_pa"],
            "z0_fit_m": fit_j["z0_fit_m"],
            "mu_K_fit": R_h / fit_j["z0_fit_m"] if fit_j["z0_fit_m"] > 0 else float("nan"),
            "fit_r2": fit_j["r2"],
            "rel_err_saturation": dev_sat,
            "saturation": sat_metrics,
            "profile": {"depth_m": [float(v) for v in grid],
                        "sigma_mean_pa": [float(v) for v in sigma_mean],
                        "sigma_std_pa": [float(v) for v in sigma_std],
                        "prediction_pa": fit_j["prediction_pa"]},
        }
    summary["janssen_profiles"] = janssen_rows
    max_sat_dev = max(sat_devs) if sat_devs else float("nan")
    criteria.append(_criterion(
        "C4a_janssen_saturation_reached", bool(all(sat_flags)) if sat_flags else False,
        "saturation_ratio > 0.85 at z >= 0.75 H",
        bool(sat_flags and all(sat_flags)),
        "profile flattens at the bottom of the column for every mu_w"))
    criteria.append(_criterion(
        "C4b_janssen_saturation_value", max_sat_dev, "< 0.25",
        bool(np.isfinite(max_sat_dev) and max_sat_dev < 0.25),
        f"|sigma_sat(fit) - rho_b g R_h/mu_w| / analytic over mu_w={sorted(prof)}"))

    # -- C5 mass ledger ----------------------------------------------------
    mass_res = max((r["mass_ledger_residual"] for r in records.values()), default=float("nan"))
    criteria.append(_criterion(
        "C5_mass_conservation", mass_res, "< 1e-6",
        bool(np.isfinite(mass_res) and mass_res < 1e-6),
        "|m_domain + m_discharged - m_0| / m_0 over all runs"))

    # -- C6 energy ledger --------------------------------------------------
    dts = {r["case_id"]: r for r in by_face("dtstudy")}
    energy_res = max((r["energy_ledger_residual"] for r in records.values()), default=float("nan"))
    dt_table = {cid: {"dt_s": r["dt_s"], "steps": r["steps"],
                      "energy_ledger_residual": r["energy_ledger_residual"]}
                for cid, r in sorted(dts.items())}
    summary["energy_dt_study"] = dt_table
    criteria.append(_criterion(
        "C6_energy_ledger", energy_res, "< 1e-8",
        bool(np.isfinite(energy_res) and energy_res < 1e-8),
        "relative |E_mech + W_diss - E_0| / E_scale (worst run); "
        "see energy_dt_study for the dt-scaling of the trapezoidal floor"))

    # -- C7 particle-count convergence ------------------------------------
    conv = {}
    conv_disp = [r for r in by_face("discharge") if r["entry"].get("convergence_study")]
    base_disp = [r for r in by_face("discharge")
                 if not r["entry"].get("convergence_study")
                 and abs(float(r["entry"]["orifice_diameter_m"]) - 0.20) < 1e-9]
    if conv_disp and base_disp:
        groups = {}
        for r in conv_disp + base_disp:
            key = round(float(r["case"]["geometry"]["barrel_fill_height_m"]), 3)
            groups.setdefault(key, []).append(_steady(r))
        vals = [_mean_std(v)[0] for _, v in sorted(groups.items())]
        spread = (max(vals) - min(vals)) / float(np.mean(vals))
        conv["discharge_flow_rate"] = {"fill_height_m": sorted(groups),
                                       "flow_rate_kg_s": vals,
                                       "relative_spread": spread}
    conv_repose = [r for r in by_face("repose") if r["entry"].get("convergence_study")]
    repose_base = [r for r in by_face("repose")
                   if float(r["entry"].get("mu", -1)) == 0.4
                   and not r["entry"].get("convergence_study")]
    if conv_repose and repose_base:
        small = _mean_std([r["repose"].get("theta_r_deg") for r in conv_repose])[0]
        big = _mean_std([r["repose"].get("theta_r_deg") for r in repose_base])[0]
        spread = abs(small - big) / (0.5 * (small + big))
        conv["repose_angle"] = {"theta_small_deg": small, "theta_base_deg": big,
                                "n_small": int(conv_repose[0]["n_particles"]),
                                "n_base": int(repose_base[0]["n_particles"]),
                                "relative_spread": spread}
    summary["convergence"] = conv
    worst_spread = max((v["relative_spread"] for v in conv.values()), default=float("nan"))
    criteria.append(_criterion(
        "C7_particle_count_convergence", worst_spread, "< 0.10",
        bool(np.isfinite(worst_spread) and worst_spread < 0.10),
        "relative spread of W (3 inventories) and theta_r (2 inventories)"))

    # -- C8 frictionless negative control ---------------------------------
    neg = [r for r in by_face("repose") if r["entry"].get("negative_control")]
    neg_mean, neg_std, neg_n = _mean_std([r["repose"].get("theta_r_deg") for r in neg])
    summary["negative_control"] = {"theta_r_deg": neg_mean,
                                   "theta_r_std_deg": neg_std, "n": neg_n}
    criteria.append(_criterion(
        "C8_negative_control_mu0", neg_mean, "< 5 deg",
        bool(np.isfinite(neg_mean) and neg_mean < 5.0),
        "frictionless (mu = 0) heap must be flat"))

    # -- C9 end-to-end JSON ------------------------------------------------
    required = {"case_id", "face", "n_particles", "dt_s", "case",
                "mass_ledger_residual", "energy_ledger_residual"}
    incomplete = [cid for cid, r in records.items()
                  if not required.issubset(r.keys())]
    criteria.append(_criterion(
        "C9_json_output", len(records), "all records complete",
        bool(records and not incomplete),
        f"{len(records)} case records, {len(incomplete)} incomplete"))

    passed = [c for c in criteria if c["passed"]]
    return {
        "criteria": criteria,
        "n_criteria": len(criteria),
        "n_passed": len(passed),
        "all_passed": len(passed) == len(criteria),
        "summary": summary,
    }
