#!/usr/bin/env python
"""Grounded ice-ridge mobility campaign: closed form + DEM closure + criteria C1..C5.

Writes ``artifacts/ridge_mobility/metrics.json`` (every number the report cites
is produced here, no hand-copied digits) and the report figures.

Usage
-----
    python scripts/ridge_mobility_campaign.py            # full campaign (~15 min)
    python scripts/ridge_mobility_campaign.py --quick    # coarse smoke run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensordem.ridge_mobility import (  # noqa: E402
    GRAVITY, LITERATURE_KEEL_DRAFT_BAND, RHO_ICE, RHO_WATER,
    DemKeelSpec, RidgeMobilityCase, assess, dem_free_float, dem_grounding,
    dem_mobility_sweep, free_float_draft_ratio, mobility_boundary,
)

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "ridge_mobility"

SHIP_THRUST = 1.5e6      # N  (icebreaker-class thrust on the ridge)
SHIP_LENGTH = 30.0       # m  (ridge length loaded)
LOCAL_DRAFT = 1.0        # m  (ambient level-ice draft)
POROSITY = 0.30
H_OVER_HK = 0.70         # shallow grounded ridge: well-conditioned N


def _git_rev() -> dict:
    def run(cmd):
        try:
            return subprocess.check_output(cmd, cwd=Path(__file__).resolve().parents[1],
                                           text=True).strip()
        except Exception:  # pragma: no cover - diagnostics only
            return "unknown"
    return {"commit": run(["git", "rev-parse", "HEAD"]),
            "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": run(["git", "status", "--porcelain"]) != ""}


def base_case(hk_over_d: float, *, mu: float = 0.35, adhesion: float = 0.0,
              label: str = "") -> RidgeMobilityCase:
    h_k = hk_over_d * LOCAL_DRAFT
    return RidgeMobilityCase(
        keel_depth=h_k,
        porosity=POROSITY,
        seabed_friction=mu,
        local_draft=LOCAL_DRAFT,
        water_depth=H_OVER_HK * h_k,
        refreeze_adhesion=adhesion,
        ship_thrust=SHIP_THRUST,
        ship_contact_length=SHIP_LENGTH,
        label=label or f"hk/d={hk_over_d:g}",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarse smoke run")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--loaded-steps", type=int, default=6000)
    args = ap.parse_args()
    steps = 4000 if args.quick else args.steps
    loaded = 2000 if args.quick else args.loaded_steps
    r_over_h = 0.03 if args.quick else 0.015

    torch.manual_seed(20261110)
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    out: dict = {
        "campaign": "p273w3-n08-grounded-ridge-mobility",
        "repo": "TensorDEM",
        "module": "tensordem.ridge_mobility",
        "mode": "quick" if args.quick else "full",
        "protocol": {
            "flank_angle_deg": 30.0,
            "particle_radius_over_keel": r_over_h,
            "water_depth_over_keel": H_OVER_HK,
            "ship_thrust_N": SHIP_THRUST,
            "ship_contact_length_m": SHIP_LENGTH,
            "drive_per_length_N_per_m": SHIP_THRUST / SHIP_LENGTH,
            "porosity_design": POROSITY,
            "local_draft_m": LOCAL_DRAFT,
            "dem_steps": steps,
            "dt_rule": "0.25*sqrt(m/kn)",
            "damping": 0.05,
            "contact_model": "linear spring + viscous normal damping + tangential "
                             "spring capped by Coulomb (static friction)",
            "bonds": "axial freeze bonds, 5 kn, tension+compression (consolidated keel)",
        },
    }

    # ---------------------------------------------------------------- C2 free float
    print("[1/5] free float ...", flush=True)
    float_case = base_case(5.0, label="free_float")
    spec_float = DemKeelSpec(keel_depth=float_case.keel_depth, porosity=float_case.porosity,
                             radius=r_over_h * float_case.keel_depth,
                             flank_angle_deg=float_case.flank_angle_deg)
    ff_ledger, ff_cf = dem_free_float(float_case, spec_float,
                                      max_steps=max(steps, 20000), tol=1e-7)
    cf_draft_ratio = ff_cf["free_float_draft_ratio_cf"]
    area_ratio = ff_cf["archimedes_ratio_dem"]
    draft_ratio_area = math.sqrt(max(area_ratio, 0.0))
    geom_ratio = ff_ledger["draft_ratio_geometric"] * float_case.keel_depth / ff_cf["measured_height_m"]
    c2 = {
        "draft_ratio_cf": cf_draft_ratio,
        "draft_ratio_dem_area": draft_ratio_area,
        "draft_ratio_dem_geometric": geom_ratio,
        "error_area_pct": 100.0 * abs(draft_ratio_area - cf_draft_ratio) / cf_draft_ratio,
        "error_geometric_pct": 100.0 * abs(geom_ratio - cf_draft_ratio) / cf_draft_ratio,
        "grain_scale_band_pct": 100.0 * ff_cf["particle_radius_over_keel"] / cf_draft_ratio,
        "archimedes_ratio_cf": ff_cf["archimedes_ratio_cf"],
        "archimedes_ratio_dem": area_ratio,
        "archimedes_error_pct": 100.0 * abs(area_ratio - ff_cf["archimedes_ratio_cf"]) / ff_cf["archimedes_ratio_cf"],
        "ledger": ff_ledger,
        "passed": None,
    }
    c2["passed"] = c2["error_area_pct"] < 3.0
    out["C2_free_float"] = c2

    # ---------------------------------------------------------------- C1/C3 grounded
    print("[2/5] grounded hk/d=5 (mu_b=0.35) ...", flush=True)
    g_case = base_case(5.0, mu=0.35, label="grounded")
    spec_g = DemKeelSpec(keel_depth=5.0, porosity=POROSITY,
                         radius=r_over_h * 5.0, flank_angle_deg=30.0)
    g_ledger, g_cf = dem_grounding(g_case, spec_g, max_steps=steps, tol=1e-7,
                                   loaded_steps=loaded)
    out["grounded_base"] = {"case": assess(g_case), "ledger": g_ledger, "closure": g_cf}

    print("[3/5] grounded + refreezing adhesion ...", flush=True)
    a_case = base_case(5.0, mu=0.35, adhesion=3.0e3, label="grounded_adhesion")
    a_ledger, a_cf = dem_grounding(a_case, spec_g, max_steps=steps, tol=1e-7,
                                   loaded_steps=loaded)
    out["grounded_adhesion"] = {"case": assess(a_case), "ledger": a_ledger, "closure": a_cf}

    c1_cases = {"free_float": ff_ledger,
                "grounded_static": g_ledger["static"],
                "grounded_adhesion_static": a_ledger["static"]}
    c1 = {
        "reward": "max_i|F_i|/F_char < 1e-6 and max_i|M_i|/(F_char h_k) < 1e-6",
        "cases": {k: {"residual_force_ratio": v["residual_force_ratio"],
                      "residual_moment_ratio": v["residual_moment_ratio"],
                      "characteristic_force_N_per_m": None,
                      "steps": v["steps"],
                      "particles": v["particles"]} for k, v in c1_cases.items()},
    }
    c1["passed"] = all(v["residual_force_ratio"] < 1e-6 and v["residual_moment_ratio"] < 1e-6
                       for v in c1["cases"].values())
    out["C1_static_equilibrium"] = c1

    print("[4/5] mu_b sweep at hk/d=5 ...", flush=True)
    mu_grid = [0.0, 0.20, 0.50, 1.50]
    sweep = dem_mobility_sweep(g_case, mu_grid, spec=spec_g, max_steps=steps,
                               tol=1e-7, loaded_steps=loaded)
    out["mu_sweep"] = sweep

    print("[5/5] hk/d = 4, 5, 6 boundary ...", flush=True)
    boundary_rows = []
    for ratio in (4.0, 5.0, 6.0):
        case = base_case(ratio)
        if ratio == 5.0:
            dem_out = {"ledger": g_ledger, "closure": g_cf}
        else:
            spec = DemKeelSpec(keel_depth=case.keel_depth, porosity=POROSITY,
                               radius=r_over_h * case.keel_depth, flank_angle_deg=30.0)
            led, clo = dem_grounding(case, spec, max_steps=steps, tol=1e-7,
                                     loaded_steps=loaded)
            dem_out = {"ledger": led, "closure": clo}
        boundary_rows.append({
            "keel_over_draft": ratio,
            "case": assess(case),
            "mu_star_cf": case.critical_friction,
            "mu_star_cf_measured_geometry": (
                dem_out["closure"]["drive_force_per_length_N_per_m"]
                - dem_out["closure"]["adhesion_dem_N_per_m"])
            / dem_out["closure"]["normal_force_cf_measured_geometry_N_per_m"],
            "mu_star_dem": dem_out["closure"]["mu_star_dem"],
            "error_vs_cf_pct": 100.0 * abs(dem_out["closure"]["mu_star_dem"] - case.critical_friction)
            / case.critical_friction,
            "error_vs_cf_measured_pct": 100.0 * abs(
                dem_out["closure"]["mu_star_dem"]
                - (dem_out["closure"]["drive_force_per_length_N_per_m"]
                   - dem_out["closure"]["adhesion_dem_N_per_m"])
                / dem_out["closure"]["normal_force_cf_measured_geometry_N_per_m"])
            / ((dem_out["closure"]["drive_force_per_length_N_per_m"]
                - dem_out["closure"]["adhesion_dem_N_per_m"])
               / dem_out["closure"]["normal_force_cf_measured_geometry_N_per_m"]),
            "ledger": dem_out["ledger"],
            "closure": dem_out["closure"],
        })
    c3 = {
        "reward": "|mu*_DEM - mu*_CF| / mu*_CF < 5 % at hk/d = 4, 5, 6",
        "rows": [{k: v for k, v in r.items() if k not in ("ledger",)} | {
            "residual_force_ratio": r["ledger"]["residual_force_ratio"],
            "ground_normal_N_per_m": r["ledger"]["ground_normal_N_per_m"],
        } for r in boundary_rows],
    }
    c3["max_error_pct"] = max(r["error_vs_cf_pct"] for r in boundary_rows)
    c3["max_error_measured_pct"] = max(r["error_vs_cf_measured_pct"] for r in boundary_rows)
    c3["passed"] = c3["max_error_pct"] < 5.0
    out["C3_mu_star_boundary"] = c3

    # ---------------------------------------------------------------- C4 degeneracies
    c4 = {"reward": "mu_b->0 : R -> c_a b (buoyancy-dominated, mobile); "
                    "mu_b->inf : immobile", "rows": {}}
    for mu, tag in ((0.0, "mu_b_zero"), (1.50, "mu_b_large")):
        case = base_case(5.0, mu=mu)
        row = assess(case)
        sweep_row = next(r for r in sweep if abs(r["seabed_friction"] - mu) < 1e-9)
        c4["rows"][tag] = {
            "seabed_friction": mu,
            "resistance_N_per_m": row["resistance_N_per_m"],
            "adhesion_force_N_per_m": row["adhesion_force_N_per_m"],
            "friction_resistance_N_per_m": row["friction_resistance_N_per_m"],
            "resistance_minus_adhesion": row["resistance_N_per_m"] - row["adhesion_force_N_per_m"],
            "status": row["status"],
            "dem_mobile": sweep_row["mobile"],
            "dem_unbalanced_force_N_per_m": sweep_row["unbalanced_force_N_per_m"],
            "dem_mobilised_friction": sweep_row["mobilised_friction"],
            "dem_drift_m": sweep_row["drift_m"],
        }
    r0 = c4["rows"]["mu_b_zero"]
    c4["zero_check_abs_N_per_m"] = abs(r0["resistance_minus_adhesion"])
    c4["passed"] = (c4["zero_check_abs_N_per_m"] < 1e-9 and r0["status"] == "grounded_mobile"
                    and r0["dem_mobile"] and c4["rows"]["mu_b_large"]["status"] == "grounded_immobile"
                    and not c4["rows"]["mu_b_large"]["dem_mobile"])
    out["C4_degeneracies"] = c4

    # ---------------------------------------------------------------- C5 band
    band_lo, band_hi = LITERATURE_KEEL_DRAFT_BAND
    c5 = {"band": [band_lo, band_hi], "cases": []}
    for ratio in (4.0, 5.0, 6.0, 8.0):
        case = base_case(ratio)
        row = assess(case)
        c5["cases"].append({"keel_over_draft": ratio,
                            "in_band": case.draft_ratio_in_band,
                            "status": case.status,
                            "mu_star": case.critical_friction,
                            "model_form_flags": row["model_form_flags"]})
    c5["flags_registered"] = all(not c["in_band"] and c["model_form_flags"]
                                 for c in c5["cases"] if not c["in_band"])
    c5["passed"] = c5["flags_registered"] and all(
        c["in_band"] for c in c5["cases"] if band_lo <= c["keel_over_draft"] <= band_hi)
    out["C5_literature_band"] = c5

    # ---------------------------------------------------------------- boundary map
    out["boundary_curve"] = [
        {k: v for k, v in row.items() if k != "mobility_map"} | {"mobility_map": row["mobility_map"]}
        for row in mobility_boundary([2, 3, 4, 4.5, 5, 5.5, 6, 7, 8],
                                     local_draft=LOCAL_DRAFT,
                                     water_depth_ratio=H_OVER_HK,
                                     porosity=POROSITY,
                                     ship_thrust=SHIP_THRUST,
                                     ship_contact_length=SHIP_LENGTH)
    ]

    out["criteria"] = {"C1": c1["passed"], "C2": c2["passed"], "C3": c3["passed"],
                       "C4": c4["passed"], "C5": c5["passed"],
                       "all": all(out[k]["passed"] for k in
                                  ("C1_static_equilibrium", "C2_free_float",
                                   "C3_mu_star_boundary", "C4_degeneracies",
                                   "C5_literature_band"))}
    out["runtime_s"] = time.time() - t_start
    out["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "git": _git_rev(),
    }
    metrics = OUT / "metrics.json"
    metrics.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {metrics}")
    print(json.dumps(out["criteria"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())