#!/usr/bin/env python3
"""CASE-DEM-SILO-BEVERLOO: kn-sensitivity study (m20585).

Hypothesis: the +25% error in m20055 (kn=1e5) is caused by excessive particle
softness.  Softer particles deform more at contacts, effectively increasing the
orifice size and reducing the effective particle diameter, leading to higher
flow rates than the Beverloo prediction.

This script runs the same D5 case (D/d=5, mu=0.30) at three stiffness levels:
  kn=1e5  (m20055 baseline, expected ~+25% error)
  kn=1e6  (m15259 value, expected ~0% error)
  kn=1e7  (stiffer, should converge to kn=1e6 result)

All other parameters are held fixed (the m20055 D5 case).

Usage
-----
    PYTHONPATH=src python benchmarks/m20585_beverloo_kn_sensitivity/run_kn_sweep.py
"""
from __future__ import annotations

import json
import math
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from tensordem.silo_discharge import SiloCase, simulate_silo

REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"
OUT_DIR = ROOT / "runs" / "m20585_beverloo_kn_sensitivity"

# m20055 D5 base parameters
BASE = dict(
    radius=0.010,          # d = 20 mm
    density=2500.0,        # glass
    mu=0.30,
    mu_rolling=0.02,
    restitution=0.30,
    kt_ratio=0.5,
    gravity=9.81,
    silo_radius=0.130,
    fill_height=0.40,
    orifice_diameter=0.100,   # D/d = 5
    plate_thickness=0.010,
    wall_mu=0.0,
    packing="settled",
    plug_during_settle=True,
    settle_loose_gap=0.30,
    packing_gap=0.0,
    jitter=0.05,
    seed=20055,
    settle_time=0.60,
    discharge_time=3.0,
    window=0.50,
    n_windows=5,
    steady_quantiles=(0.20, 0.80),
)

KN_VALUES = [1e5, 1e6, 1e7]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def run_one(kn: float) -> dict:
    params = dict(BASE, kn=kn)
    case = SiloCase(**params)
    t0 = time.time()
    res = simulate_silo(case)
    wall = time.time() - t0

    ref = json.loads(REF_PATH.read_text())
    C = float(ref["coefficients"]["C"])
    k = float(ref["coefficients"]["k"])

    D = case.orifice_diameter
    d = case.diameter
    rho_b = res["bulk_density_bed_kg_m3"]
    g = case.gravity
    Deff = D - k * d
    W_ref = C * rho_b * math.sqrt(g) * Deff ** 2.5
    W_got = res["steady_flow_rate_kg_s"]
    rel = (W_got - W_ref) / W_ref if W_ref > 0 else float("nan")
    C_rec = W_got / (rho_b * math.sqrt(g) * Deff ** 2.5) if Deff > 0 else float("nan")

    return {
        "kn": kn,
        "dt": case.timestep,
        "steps": res["steps"],
        "wall_clock_s": wall,
        "n_initial": res["n_initial"],
        "n_bed": res["n_bed"],
        "bed_height_m": res["bed_height_m"],
        "packing_fraction_bed": res["packing_fraction_bed"],
        "bulk_density_bed_kg_m3": rho_b,
        "n_discharged": res["n_discharged_total"],
        "W_got_kg_s": W_got,
        "W_ref_kg_s": W_ref,
        "rel_err": rel,
        "rel_err_pct": rel * 100 if math.isfinite(rel) else None,
        "C_recovered": C_rec,
        "steady_rate_cv": res["steady_rate_cv"],
        "git_sha": git_sha(),
        "host": socket.gethostname(),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for kn in KN_VALUES:
        print(f"\n{'='*60}")
        print(f"Running kn={kn:.0e} ...")
        r = run_one(kn)
        results.append(r)
        print(f"  phi={r['packing_fraction_bed']:.3f}, rho_b={r['bulk_density_bed_kg_m3']:.0f}")
        print(f"  W={r['W_got_kg_s']:.4f}, W_ref={r['W_ref_kg_s']:.4f}")
        print(f"  err={r['rel_err_pct']:+.1f}%, C_rec={r['C_recovered']:.3f}")
        print(f"  wall={r['wall_clock_s']:.0f}s, steps={r['steps']}")

        path = OUT_DIR / f"kn_{int(kn):d}.json"
        path.write_text(json.dumps(r, indent=2, default=float))

    # Summary
    summary = {
        "study": "kn-sensitivity for CASE-DEM-SILO-BEVERLOO",
        "mission_id": 20585,
        "hypothesis": "Soft particles (kn=1e5) cause +25% error; stiffer kn converges to Beverloo",
        "reference": json.loads(REF_PATH.read_text()),
        "results": results,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # Markdown table
    lines = [
        "| kn [N/m] | dt [s] | steps | phi | rho_b [kg/m3] | W [kg/s] | W_ref [kg/s] | err [%] | C_rec | CV |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['kn']:.0e} | {r['dt']:.6e} | {r['steps']} | "
            f"{r['packing_fraction_bed']:.3f} | {r['bulk_density_bed_kg_m3']:.0f} | "
            f"{r['W_got_kg_s']:.4f} | {r['W_ref_kg_s']:.4f} | "
            f"{r['rel_err_pct']:+.1f} | {r['C_recovered']:.3f} | {r['steady_rate_cv']:.3f} |"
        )
    tbl = OUT_DIR / "report_tables.md"
    tbl.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nsummary -> {OUT_DIR/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
