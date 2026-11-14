#!/usr/bin/env python3
"""CASE-DEM-SILO-BEVERLOO: one silo-discharge case, measured against Beverloo (1961).

Runs the *existing* TensorDEM flat-bottom silo DEM (``tensordem.silo_discharge``)
with a disordered, gravity-settled bed and a plugged-then-opened orifice, measures
the steady mass flow rate from the middle of the discharge stream, and compares
it with W = C rho_b sqrt(g) (D - k d)^(5/2).

The reference coefficients are read from
``benchmarks/m20055_beverloo/ref_beverloo_1961.json`` (no number is typed into
this script or into the report).

Usage
-----
    PYTHONPATH=src python benchmarks/m20055_beverloo/run_case.py --case D5
    PYTHONPATH=src python benchmarks/m20055_beverloo/run_case.py --list
"""
from __future__ import annotations

import argparse
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

import numpy as np  # noqa: E402

from tensordem.silo_discharge import SiloCase, simulate_silo  # noqa: E402

REF_PATH = Path(__file__).resolve().parent / "ref_beverloo_1961.json"
OUT_DIR = ROOT / "runs" / "m20055_beverloo"

# ---------------------------------------------------------------------------
# Predeclared case matrix (prereg/m20055_beverloo.md).  d, material, wall
# condition and the settling protocol are shared; only D/d, mu and dt differ.
# ---------------------------------------------------------------------------
BASE = dict(
    radius=0.010,          # d = 20 mm
    density=2500.0,        # glass
    mu=0.30,               # particle friction (primary case)
    mu_rolling=0.02,
    restitution=0.30,
    kn=1.0e5,
    kt_ratio=0.5,
    gravity=9.81,
    silo_radius=0.130,     # silo D = 0.26 m
    fill_height=0.40,      # loose column -> settles to the bed measured at open
    plate_thickness=0.010,  # 0.5 d
    wall_mu=0.0,           # smooth side wall (Beverloo glass-bead experiments)
    packing="settled",
    plug_during_settle=True,
    settle_loose_gap=0.30,
    packing_gap=0.0,
    jitter=0.05,           # 0.05 r; clearance of the loose lattice is 0.6 r
    seed=20055,
    settle_time=0.60,
    window=0.50,
    n_windows=5,
    steady_quantiles=(0.20, 0.80),
)

CASES = {
    # id      overrides
    "D4":   dict(orifice_diameter=0.080, discharge_time=4.5),   # D/d = 4
    "D5":   dict(orifice_diameter=0.100, discharge_time=3.0),   # D/d = 5, primary
    "D6":   dict(orifice_diameter=0.120, discharge_time=2.0),   # D/d = 6
    "D5_mu15": dict(orifice_diameter=0.100, discharge_time=3.0, mu=0.15),
    "D5_mu45": dict(orifice_diameter=0.100, discharge_time=3.0, mu=0.45),
    "D5_dthalf": dict(orifice_diameter=0.100, discharge_time=3.0, dt=None),  # dt/2 set in main
}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:                                    # pragma: no cover
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

    # C recovered at the literature k (the dimensionless number to compare)
    C_got = W_got / (rho_b * math.sqrt(g) * Deff ** 2.5) if Deff > 0 else float("nan")
    rel_C = (C_got - C) / C

    # dimensionless groups that fully define the case
    Fr = W_ref / (rho_b * math.sqrt(g) * D ** 2.5)      # Beverloo prefactor itself
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
            2.0 * math.pi * math.sqrt(case.material.mass / case.kn)),
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="D5")
    ap.add_argument("--shard", type=int, default=0,
                    help="1-based shard index; selects a case from the predeclared matrix")
    ap.add_argument("--nshards", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    if args.list:
        print(" ".join(sorted(CASES)))
        return 0
    if args.shard and args.nshards:
        order = ["D4", "D5", "D6", "D5_mu15", "D5_mu45", "D5_dthalf"]
        if not 1 <= args.shard <= len(order):
            raise SystemExit(f"shard must be in 1..{len(order)}")
        args.case = order[args.shard - 1]

    ref = json.loads(REF_PATH.read_text())
    case = build_case(args.case)
    t0 = time.time()
    res = simulate_silo(case)
    wall = time.time() - t0
    out = analyse(args.case, case, res, ref, wall)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.case}.json"
    path.write_text(json.dumps(out, indent=2, default=float))

    tele = out["telemetry"]
    print(f"[{args.case}] D/d={out['dimensionless']['D_over_d']:.2f} "
          f"n_bed={tele['n_bed']} rho_b={out['verdict']['rho_b_kg_m3']:.0f} "
          f"discharged={tele['n_discharged_total']}/{tele['n_initial']} "
          f"W={out['verdict']['W_got_kg_s']:.4f} "
          f"W_ref={out['verdict']['W_ref_kg_s']:.4f} "
          f"rel={100*out['verdict']['rel_err_W']:+.1f}% "
          f"CV={tele['steady_rate_cv']:.3f} wall={wall:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())