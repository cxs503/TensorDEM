#!/usr/bin/env python
"""CASE-DEM-BIAXIAL-FRICTION · rate-probe (H4) analysis.

Reads the single-arm rate probe written by ``scripts/m22106_rate_probe.py``
(``artifacts/CASE-DEM-BIAXIAL-FRICTION/rate_probe/iso_*.json``) and answers the
pre-registered H4 question: is the isochoric cell's energy-budget defect
(contact dissipation >> boundary work) *rate-driven*?

The R8 smoke (10x20, eps_dot=0.05, strain 0.30) measured boundary power
1.59 MW vs contact dissipation 24.53 MW (residual -1464 %).  The rate probe
runs the same geometry at eps_dot=0.0025 (20x slower).  If the residual
shrinks toward the <=15 % gate as the rate falls, the defect is kinetic
(non-affine slip is rate-driven); if it stays put, the defect is not kinetic
and the critical-state protocol needs re-examination.

Outputs (numbers injected programmatically):
    benchmarks/case_dem_biaxial_friction/results_m22258/rate_probe.json
    docs/figures/CASE-DEM-BIAXIAL-FRICTION_R9_rate_probe.png

Usage:
    PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/analyse_rate_probe.py
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from tensordem.biaxial_critical_state import (  # noqa: E402
    critical_state_from_stress_path,
    rowe_stress_dilatancy,
)

OUT_JSON = os.path.join(REPO, "benchmarks", "case_dem_biaxial_friction",
                        "results_m22258", "rate_probe.json")
FIG_PATH = os.path.join(REPO, "docs", "figures",
                        "CASE-DEM-BIAXIAL-FRICTION_R9_rate_probe.png")

# --- preregistered thresholds (prereg/m22106_mu_ladder_prereg.md, H4) --------
H4_MAX_RESIDUAL = 0.15

# R8 smoke reference (10x20, eps_dot=0.05, strain 0.30) -- the fast counterpart
SMOKE_REF = {
    "tag": "iso_smoke_10x20",
    "strain_rate": 0.05,
    "boundary_power_W": 1.595e6,
    "contact_total_power_W": 24.53e6 + 0.41e6,   # friction + damping
    "residual_fraction": -14.64,
    "source": "artifacts/CASE-DEM-BIAXIAL-FRICTION/mu_ladder_smoke/iso_smoke_10x20.json",
}


def _eng_to_log(eps_eng: float) -> float:
    return -math.log(max(1.0 - eps_eng, 1e-12))


def _mob_window(mobilisation: list, lo: float, hi: float) -> dict:
    keys_power = ("sum_friction_power", "sum_damping_power", "calls", "contacts",
                  "sum_fn", "sum_fn_abs_tanh", "saturated", "sum_vt_abs")
    sel = [s for s in mobilisation
           if lo <= _eng_to_log(float(s.get("axial_strain", 0.0))) <= hi]
    if not sel:
        return {"samples": 0}
    tot = {k: float(sum(float(s.get(k, 0.0)) for s in sel)) for k in keys_power}
    calls = max(tot["calls"], 1e-30)
    contacts = max(tot["contacts"], 1e-30)
    return {
        "samples": len(sel),
        "calls": tot["calls"],
        "mean_contacts_per_call": tot["contacts"] / calls,
        "fn_weighted_mean_abs_tanh": (tot["sum_fn_abs_tanh"] / tot["sum_fn"]
                                      if tot["sum_fn"] else float("nan")),
        "saturated_fraction": tot["saturated"] / contacts,
        "mean_abs_vt": tot["sum_vt_abs"] / contacts,
        "friction_power_W": tot["sum_friction_power"] / calls,
        "damping_power_W": tot["sum_damping_power"] / calls,
    }


def analyse_rate_arm(path: str) -> dict:
    with open(path) as fh:
        raw = json.load(fh)
    cfg = dict(raw["config"])
    sp = {k: np.asarray(v, dtype=float) for k, v in raw["stress_path"].items()
          if isinstance(v, (list, tuple))}
    mu = float(cfg["mu"])
    p0 = float(cfg["confining_stress"])
    spec = raw.get("specimen", {})
    arm = {
        "tag": raw.get("tag", os.path.basename(path)),
        "path": os.path.relpath(path, REPO),
        "mu": mu,
        "atan_mu_deg": math.degrees(math.atan(mu)),
        "specimen": {k: spec.get(k) for k in
                     ("columns", "rows", "particles", "radius", "dt")},
        "config": {k: cfg.get(k) for k in
                   ("confining_stress", "strain_rate", "strain_target",
                    "consolidation_steps", "lateral_control")},
        "elapsed_s": raw.get("elapsed_s"),
        "peak": raw.get("peak"),
        "inertial_number": raw.get("loading", {}).get("inertial_number"),
        "gate_state": "UNKNOWN",
        "phi_cv_deg": float("nan"),
    }
    try:
        cs = critical_state_from_stress_path(
            sp, sigma3_target=p0, strain_measure="log",
            boundary_control="isochoric")
    except Exception as exc:
        arm["gate_state"] = "WITHHELD"
        arm["gate_error"] = str(exc)
        return arm
    lo, hi = cs.window
    eps_eng = float(np.asarray(sp["axial_strain"], dtype=float)[-1])
    eps_log = float(np.asarray(sp["axial_strain_log"], dtype=float)[-1])
    target = float(cfg.get("strain_target") or float("nan"))
    arm["strain_budget"] = {
        "declared_target": target,
        "delivered_engineering": eps_eng,
        "delivered_log": eps_log,
        "delivered_relative_to_target_log": ((eps_log - target) / target
                                             if np.isfinite(target) and target else float("nan")),
    }
    arm.update({
        "gate_state": "PASS" if cs.converged else "WITHHELD",
        "flags": dict(cs.flags),
        "thresholds": dict(cs.thresholds),
        "window": [float(lo), float(hi)],
        "phi_cv_deg": float(cs.phi_cv_deg),
        "phi_cv_window_spread_deg": float(cs.window_spread_deg),
        "q_cv_MPa": float(cs.q_cv_Pa) / 1e6,
        "sigma_h_cv_kPa": float(cs.sigma_h_cv_Pa) / 1e3,
        "d_eps_v_d_eps_a": float(cs.d_eps_v_d_eps_a),
        "q_drift": float(cs.q_drift),
        "peak_phi_deg": float(cs.peak_phi_deg),
    })
    try:
        arm["rowe"] = rowe_stress_dilatancy(sp, mu=mu, min_strain=max(0.15, lo))
    except Exception as exc:
        arm["rowe"] = {"error": str(exc)}
    arm["relative_error_vs_rowe_atan_mu"] = (
        (arm["phi_cv_deg"] - arm["atan_mu_deg"]) / arm["atan_mu_deg"])
    arm["relative_error_vs_anchor_ref"] = (
        (arm["phi_cv_deg"] - 38.0) / 38.0)

    # --- H4: energy cross-check (same window, same contacts) ----------------
    mob = _mob_window(raw.get("mobilisation") or [], lo, hi)
    arm["mobilisation"] = mob
    width0 = float(spec.get("width0") or 0.0)
    height0 = float(spec.get("height0") or 0.0)
    rate = float(cfg["strain_rate"])
    volume = width0 * height0 * 1.0
    q_win = float(cs.q_cv_Pa)
    p_boundary = q_win * rate * volume
    p_contact = mob.get("friction_power_W", float("nan")) + \
        mob.get("damping_power_W", float("nan"))
    arm["energy_balance"] = {
        "volume_m3": volume,
        "q_window_Pa": q_win,
        "strain_rate": rate,
        "boundary_power_W": p_boundary,
        "contact_friction_power_W": mob.get("friction_power_W", float("nan")),
        "contact_damping_power_W": mob.get("damping_power_W", float("nan")),
        "contact_total_power_W": p_contact,
        "residual_fraction": ((p_boundary - p_contact) / p_boundary
                              if p_boundary else float("nan")),
        "h4_gate": "PASS" if (p_boundary and
                              abs((p_boundary - p_contact) / p_boundary) <= H4_MAX_RESIDUAL)
                      else "FAIL",
    }
    return arm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=(
        "artifacts/CASE-DEM-BIAXIAL-FRICTION/rate_probe/iso_*.json"))
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(REPO, args.glob)))
    if not paths:
        print(f"NO RATE-PROBE ARTIFACTS at {args.glob} -- not run yet")
        return 2
    arms = [analyse_rate_arm(p) for p in paths]
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump({"module": "CASE-DEM-BIAXIAL-FRICTION",
                   "round": "R9(m22258)",
                   "prereg": "prereg/m22106_mu_ladder_prereg.md",
                   "smoke_reference": SMOKE_REF,
                   "arms": arms}, fh, indent=2)
    for a in arms:
        eb = a.get("energy_balance", {})
        print(f"== {a['tag']} rate={a['config']['strain_rate']} "
              f"phi_cv={a['phi_cv_deg']:.2f}deg gate={a['gate_state']} "
              f"P_bound={eb.get('boundary_power_W',float('nan'))/1e6:.3f}MW "
              f"P_contact={eb.get('contact_total_power_W',float('nan'))/1e6:.3f}MW "
              f"resid={eb.get('residual_fraction',float('nan'))*100:.1f}% "
              f"H4={eb.get('h4_gate')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
