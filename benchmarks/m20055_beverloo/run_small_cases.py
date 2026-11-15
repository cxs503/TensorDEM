#!/usr/bin/env python3
"""Run small-scale m20055 cases locally for pipeline verification.

Uses reduced silo size and shorter times to complete in <120s per case.
Results are written to runs/m20055_beverloo/ for analyse.py consumption.
"""
from __future__ import annotations

import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from tensordem.silo_discharge import SiloCase, simulate_silo, _initial_packing

REF_PATH = Path(__file__).resolve().parent / "ref_beverloo_1961.json"
OUT_DIR = ROOT / "runs" / "m20055_beverloo"

# Small-scale parameters: reduced silo size, shorter times
# D/d ratios preserved, but absolute sizes scaled down
BASE = dict(
    radius=0.010,          # d = 20 mm (same)
    density=2500.0,        # glass
    mu=0.30,
    mu_rolling=0.02,
    restitution=0.30,
    kn=1.0e5,
    kt_ratio=0.5,
    gravity=9.81,
    silo_radius=0.080,     # D_silo = 0.16 m (must be > D6 orifice 0.12m)
    fill_height=0.16,      # H/D ~ 2.7 for D5
    plate_thickness=0.010,
    wall_mu=0.0,
    packing="settled",
    plug_during_settle=True,
    settle_loose_gap=0.30,
    packing_gap=0.0,
    jitter=0.05,
    seed=20055,
    settle_time=0.20,
    window=0.10,
    n_windows=5,
    steady_quantiles=(0.20, 0.80),
)

CASES = {
    "D4":   dict(orifice_diameter=0.080, discharge_time=0.80),
    "D5":   dict(orifice_diameter=0.100, discharge_time=0.60),
    "D6":   dict(orifice_diameter=0.120, discharge_time=0.50),
    "D5_mu15": dict(orifice_diameter=0.100, discharge_time=0.60, mu=0.15),
    "D5_mu45": dict(orifice_diameter=0.100, discharge_time=0.60, mu=0.45),
    "D5_dthalf": dict(orifice_diameter=0.100, discharge_time=0.60, dt=None),
}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def build_case(case_id: str) -> SiloCase:
    params = dict(BASE)
    params.update(CASES[case_id])
    dt = params.pop("dt", None)
    case = SiloCase(**params)
    if case_id.endswith("dthalf"):
        dt = 0.5 * case.timestep
    if dt is not None:
        case = SiloCase(**{**params, "dt": dt})
    return case


def analyse(case_id: str, case: SiloCase, res: dict, ref: dict, wall: float) -> dict:
    C, k = float(ref["coefficients"]["C"]), float(ref["coefficients"]["k"])
    d = case.diameter
    D = case.orifice_diameter
    rho_b = float(res["bulk_density_bed_kg_m3"])
    g = case.gravity
    Deff = D - k * d
    W_ref = C * rho_b * math.sqrt(g) * Deff ** 2.5
    W_got = float(res["steady_flow_rate_kg_s"])
    rel = (W_got - W_ref) / W_ref if W_ref > 0 else float("nan")

    C_got = W_got / (rho_b * math.sqrt(g) * Deff ** 2.5) if Deff > 0 else float("nan")
    rel_C = (C_got - C) / C

    Fr = W_ref / (rho_b * math.sqrt(g) * D ** 2.5)
    groups = {
        "D_over_d": D / d,
        "silo_D_over_D": 2.0 * case.silo_radius / D,
        "bed_height_over_D": float(res["bed_height_m"]) / D,
        "fill_ratio_H_over_D": case.fill_height / D,
        "rho_b_over_rho_p": rho_b / case.density,
        "packing_fraction_bed": float(res["packing_fraction_bed"]),
        "C_beverloo_prefactor_Fr": Fr,
        "C_recovered": C_got,
        "k_reference": k,
        "D_over_d_minus_k": Deff / d,
        "mu_p": case.mu,
        "mu_wall": case.wall_mu,
        "restitution": case.restitution,
        "dt_over_contact_period": case.timestep / (
            2.0 * math.pi * math.sqrt(case.material.kn / case.material.mass)
        ) if hasattr(case.material, 'mass') else 0.0,
        "n_particles_initial": int(res["n_initial"]),
        "n_particles_bed": int(res["n_bed"]),
    }
    discard = {
        "rel_err_W": rel,
        "abs_rel_err_W": abs(rel),
        "rel_err_C": rel_C,
        "pass_10pct": abs(rel) < 0.10,
        "pass_15pct": abs(rel) < 0.15,
        "W_ref_kg_s": W_ref,
        "W_got_kg_s": W_got,
        "Deff_m": Deff,
        "rho_b_kg_m3": rho_b,
    }
    return {
        "case_id": case_id,
        "wall_clock_s": wall,
        "git_sha": git_sha(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "telemetry": res,
        "dimensionless": groups,
        "verdict": discard,
        "ref_source": ref["source"],
        "command": " ".join([sys.executable, *sys.argv]),
    }


def main() -> int:
    ref = json.loads(REF_PATH.read_text())
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for case_id in ["D4", "D5", "D6", "D5_mu15", "D5_mu45", "D5_dthalf"]:
        case = build_case(case_id)
        n_init = len(_initial_packing(case))
        print(f"[{case_id}] n_init={n_init}, dt={case.timestep:.2e}")
        t0 = time.time()
        res = simulate_silo(case)
        wall = time.time() - t0
        out = analyse(case_id, case, res, ref, wall)
        path = out_dir / f"{case_id}.json"
        path.write_text(json.dumps(out, indent=2, default=float))
        tele = out["telemetry"]
        print(f"  n_bed={tele['n_bed']} discharged={tele['n_discharged_total']} "
              f"W={out['verdict']['W_got_kg_s']:.4f} "
              f"W_ref={out['verdict']['W_ref_kg_s']:.4f} "
              f"rel={100*out['verdict']['rel_err_W']:+.1f}% "
              f"wall={wall:.1f}s")

    print(f"\nAll cases written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
