#!/usr/bin/env python3
"""CASE-DEM-SILO-BEVERLOO: Beverloo law validation campaign (m20707).

Uses the ``tensordem.silo_flow`` module (Cundall-Strack wall friction,
disordered packing, energy ledger) to measure the steady mass flow rate
through a flat-bottom silo orifice and compare with the Beverloo (1961)
correlation:

    W = C · rho_b · sqrt(g) · (D - k·d)^(5/2)

Reference coefficients (from ref_beverloo_1961.json):
    C = 0.58  (range 0.55-0.60 for spherical particles)
    k = 1.4   (range 1.2-1.6)

Acceptance: |W - W_ref| / W_ref < 10% for the primary case (D/d = 5).

Case matrix
-----------
| id       | D/d | mu_p | purpose                    |
|----------|-----|------|----------------------------|
| D4       | 4   | 0.50 | law check point            |
| D5       | 5   | 0.50 | primary case (verdict)     |
| D6       | 6   | 0.50 | law check point            |
| D5_mu40  | 5   | 0.40 | friction sensitivity       |
| D5_mu55  | 5   | 0.55 | friction sensitivity       |
| D5_g4    | 5   | 0.50 | gravity scaling (g/4)      |

Shared: d = 3 mm, rho_p = 2500 kg/m³, mu_r = 0.05, e = 0.30, kn = 5e3 N/m,
kt/kn = 0.5, R_silo = 0.025 m, H_fill = 0.06 m, wall_mu = 0, seed = 20707.

Usage
-----
    PYTHONPATH=src python benchmarks/m20707_beverloo/campaign.py --case D5
    PYTHONPATH=src python benchmarks/m20707_beverloo/campaign.py --list
    PYTHONPATH=src python benchmarks/m20707_beverloo/campaign.py --all
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from tensordem.silo_flow.beverloo import (
    BEVERLOO_C, BEVERLOO_K, DischargeCase, beverloo_prediction, run_discharge,
)

REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"
OUT_DIR = ROOT / "runs" / "m20707_beverloo"

# ---------------------------------------------------------------------------
# Shared base parameters
# ---------------------------------------------------------------------------
BASE = dict(
    particle_diameter=0.003,     # d = 3 mm
    silo_radius=0.025,           # R = 25 mm
    fill_height=0.060,           # H = 60 mm
    wall_mu=0.0,                 # smooth walls (Beverloo glass-bead experiments)
    seed=20707,
    settle_time=0.08,
    discard_time=0.15,
    window_time=0.20,
    n_windows=6,
    kn=5.0e3,                    # moderate stiffness
    mu=0.50,                     # inter-particle friction (primary)
    mu_rolling=0.05,
)

CASES = {
    "D4":      dict(orifice_diameter=0.012),    # D/d = 4
    "D5":      dict(orifice_diameter=0.015),    # D/d = 5, primary
    "D6":      dict(orifice_diameter=0.018),    # D/d = 6
    "D5_mu40": dict(orifice_diameter=0.015, mu=0.40),
    "D5_mu55": dict(orifice_diameter=0.015, mu=0.55),
    "D5_g4":   dict(orifice_diameter=0.015),    # gravity handled in build_case
}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def build_case(case_id: str) -> DischargeCase:
    params = dict(BASE)
    params.update(CASES[case_id])
    gravity = 9.81
    if case_id == "D5_g4":
        gravity = 9.81 / 4.0
    case = DischargeCase(**params)
    # Override gravity if needed (DischargeCase doesn't have a gravity param,
    # but the Material inside kernel.py does)
    return case, gravity


def analyse(case_id: str, case: DischargeCase, gravity: float, res: dict, wall: float) -> dict:
    ref = json.loads(REF_PATH.read_text())
    C, k = float(ref["coefficients"]["C"]), float(ref["coefficients"]["k"])
    d = case.particle_diameter
    D = case.orifice_diameter
    rho_b = float(res["rho_bulk"])
    Deff = D - k * d
    W_ref = C * rho_b * math.sqrt(gravity) * Deff ** 2.5
    W_got = float(res["flow_rate_kg_s"])
    rel = (W_got - W_ref) / W_ref if W_ref > 0 else float("nan")

    C_got = W_got / (rho_b * math.sqrt(gravity) * Deff ** 2.5) if Deff > 0 else float("nan")

    return {
        "case_id": case_id,
        "wall_clock_s": wall,
        "git_sha": git_sha(),
        "host": platform.node(),
        "gravity": gravity,
        "case": {
            "orifice_diameter": D,
            "particle_diameter": d,
            "D_over_d": D / d,
            "silo_radius": case.silo_radius,
            "fill_height": case.fill_height,
            "mu": case.mu,
            "mu_rolling": case.mu_rolling,
            "wall_mu": case.wall_mu,
            "kn": case.kn,
            "seed": case.seed,
        },
        "result": {
            "n_bed": res["n_bed"],
            "rho_bulk": rho_b,
            "flow_rate_kg_s": W_got,
            "flow_rate_std_kg_s": res["flow_rate_std_kg_s"],
            "flow_rate_windows_kg_s": res["flow_rate_windows_kg_s"],
            "particles_exited": res["particles_exited"],
            "dt": res["dt"],
            "steps": res["steps"],
        },
        "beverloo": {
            "C_ref": C,
            "k_ref": k,
            "W_ref_kg_s": W_ref,
            "C_recovered": C_got,
            "Deff_m": Deff,
            "rel_err_pct": rel * 100.0,
            "abs_rel_err_pct": abs(rel) * 100.0,
            "pass_10pct": abs(rel) < 0.10,
            "pass_15pct": abs(rel) < 0.15,
        },
    }


def run_one(case_id: str, out_dir: Path) -> dict:
    case, gravity = build_case(case_id)
    # Patch gravity into the material
    from tensordem.silo_flow.kernel import Material, SiloGeometry
    from tensordem.silo_flow.packing import build_bed

    mat = Material(
        radius=0.5 * case.particle_diameter,
        density=2500.0,
        mu=case.mu,
        mu_rolling=case.mu_rolling,
        restitution=0.3,
        kn=case.kn,
        kt_ratio=0.5,
        gravity=gravity,
    )
    geom = SiloGeometry(
        cylinder_radius=case.silo_radius,
        cylinder_height=case.fill_height + 0.015,
        outlet_radius=0.5 * case.orifice_diameter,
    )

    t0 = time.time()
    flow = build_bed(
        geom, mat, case.fill_height,
        wall_mu=case.wall_mu, seed=case.seed,
        settle_time=case.settle_time, plug_outlet=True,
    )
    rho_b = flow.bulk_density(case.fill_height)
    bed_mass = flow.bed_mass
    n_bed = flow.n

    # Open the orifice
    flow.plugged = False

    # Discard start-up
    n_discard = int(round(case.discard_time / flow.dt))
    for _ in range(n_discard):
        flow.step()

    # Measurement windows
    n_win = int(round(case.window_time / flow.dt))
    windows = []
    for _ in range(case.n_windows):
        m0 = flow.mass_exited
        n0 = flow.n_exited
        for _ in range(n_win):
            flow.step()
        windows.append({
            "mass_kg": flow.mass_exited - m0,
            "particles": flow.n_exited - n0,
            "time_s": n_win * flow.dt,
        })

    rates = np.array([w["mass_kg"] / w["time_s"] for w in windows])
    n_steady = max(1, len(rates) // 2)
    steady = rates[-n_steady:]

    wall = time.time() - t0

    res = {
        "n_bed": n_bed,
        "bed_mass_kg": bed_mass,
        "rho_bulk": rho_b,
        "dt": flow.dt,
        "steps": flow.steps,
        "n_windows": case.n_windows,
        "window_time_s": case.window_time,
        "discard_time_s": case.discard_time,
        "flow_rate_windows_kg_s": [float(r) for r in rates],
        "flow_rate_kg_s": float(np.mean(steady)),
        "flow_rate_std_kg_s": float(np.std(steady, ddof=1)) if len(steady) > 1 else 0.0,
        "flow_rate_steady_windows": n_steady,
        "particles_exited": int(sum(w["particles"] for w in windows)),
    }

    out = analyse(case_id, case, gravity, res, wall)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{case_id}.json"
    path.write_text(json.dumps(out, indent=2, default=float))

    bev = out["beverloo"]
    print(
        f"[{case_id}] D/d={out['case']['D_over_d']:.1f} "
        f"mu={out['case']['mu']:.2f} g={gravity:.2f} "
        f"n_bed={res['n_bed']} rho_b={rho_b:.0f} "
        f"W={bev['C_recovered']:.3f}*rho*sqrt(g)*Deff^2.5 "
        f"= {res['flow_rate_kg_s']:.6f} kg/s "
        f"W_ref={bev['W_ref_kg_s']:.6f} kg/s "
        f"rel={bev['rel_err_pct']:+.1f}% "
        f"{'PASS' if bev['pass_10pct'] else 'FAIL'} "
        f"wall={wall:.0f}s"
    )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="D5")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    if args.list:
        print(" ".join(sorted(CASES)))
        return 0

    out_dir = Path(args.out_dir)

    if args.all:
        for cid in sorted(CASES):
            run_one(cid, out_dir)
    else:
        run_one(args.case, out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
