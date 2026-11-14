#!/usr/bin/env python3
"""CASE-DEM-SILO-BEVERLOO (m21033): one flat-bottom silo case vs Beverloo (1961).

Reuses the existing REAL 3-D DEM ``tensordem.silo_discharge`` (settled +
plugged-then-opened protocol from m20055) with the contact stiffness raised to
``kn = 1e6 N/m`` — the value that already passed on the hopper kernel
(m15259, -0.4 %) — and the glass-bead friction ``mu = 0.15``.

Reference coefficients are read from
``benchmarks/m20055_beverloo/ref_beverloo_1961.json`` (zero hand-typed
numbers).  The predeclared matrix lives in ``prereg/m21033_beverloo.md``.

Usage
-----
    PYTHONPATH=src python benchmarks/m21033_beverloo/run_case.py --case D5
    PYTHONPATH=src python benchmarks/m21033_beverloo/run_case.py --list
"""
from __future__ import annotations

import argparse
import json
import math
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

REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"
OUT_DIR = ROOT / "runs" / "m21033_beverloo"

# ---------------------------------------------------------------------------
# Predeclared matrix (prereg/m21033_beverloo.md).  Shared material/geometry;
# only D/d, mu, kn and dt vary.  kn = 1e6 and mu = 0.15 are the two changes
# against m20055 (kn 1e5, mu 0.30) and are the hypothesis under test.
# ---------------------------------------------------------------------------
BASE = dict(
    radius=0.010,          # d = 20 mm
    density=2500.0,        # glass
    mu=0.15,               # glass-bead friction (m15259 lesson)
    mu_rolling=0.02,
    restitution=0.30,
    kn=1.0e6,              # m20585 recommendation (was 1e5 in m20055)
    kt_ratio=0.5,
    gravity=9.81,
    silo_radius=0.130,     # silo D = 0.26 m
    fill_height=0.40,
    plate_thickness=0.010,  # 0.5 d
    wall_mu=0.0,           # smooth side wall (Beverloo glass-bead experiments)
    packing="settled",
    plug_during_settle=True,
    settle_loose_gap=0.30,
    packing_gap=0.0,
    jitter=0.05,           # 0.05 r
    seed=21033,
    settle_time=0.60,
    window=0.50,
    n_windows=5,
    steady_quantiles=(0.20, 0.80),
)

CASES = {
    "D5":        dict(orifice_diameter=0.100, discharge_time=3.0),  # primary
    "D4":        dict(orifice_diameter=0.080, discharge_time=4.5),
    "D6":        dict(orifice_diameter=0.120, discharge_time=2.0),
    "D5_mu30":   dict(orifice_diameter=0.100, discharge_time=3.0, mu=0.30),
    "D5_kn1e5":  dict(orifice_diameter=0.100, discharge_time=3.0, kn=1.0e5),
    "D5_dthalf": dict(orifice_diameter=0.100, discharge_time=3.0),  # dt/2 in main
}

ORDER = ["D5", "D4", "D6", "D5_mu30", "D5_kn1e5", "D5_dthalf"]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:                                    # pragma: no cover
        return "unknown"


def build_case(case_id: str) -> SiloCase:
    params = dict(BASE)
    params.update(CASES[case_id])
    case = SiloCase(**params)
    if case_id.endswith("dthalf"):
        case = SiloCase(**{**params, "dt": 0.5 * case.timestep})
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

    groups = {
        "D_over_d": D / d,
        "silo_D_over_D": 2.0 * case.silo_radius / D,
        "bed_height_over_D": float(res["bed_height_m"]) / D,
        "fill_ratio_H_over_D": case.fill_height / D,
        "rho_b_over_rho_p": rho_b / case.density,
        "packing_fraction_bed": float(res["packing_fraction_bed"]),
        "C_beverloo_prefactor_Fr": W_ref / (rho_b * math.sqrt(g) * D ** 2.5),
        "C_recovered": C_got,
        "k_reference": k,
        "D_over_d_minus_k": Deff / d,
        "mu_p": case.mu,
        "mu_wall": case.wall_mu,
        "restitution": case.restitution,
        "kn_N_m": case.kn,
        "dt_over_contact_period": case.timestep / (
            2.0 * math.pi * math.sqrt(case.material.mass / case.kn)),
        "n_particles_initial": int(res["n_initial"]),
        "n_particles_bed": int(res["n_bed"]),
    }
    discard = {
        "rel_err_W": rel,
        "abs_rel_err_W": abs(rel),
        "rel_err_C": rel_C,
        "pass_10pct": bool(abs(rel) < 0.10) if math.isfinite(rel) else False,
        "pass_15pct": bool(abs(rel) < 0.15) if math.isfinite(rel) else False,
        "W_ref_kg_s": W_ref,
        "W_got_kg_s": W_got,
        "Deff_m": Deff,
        "rho_b_kg_m3": rho_b,
    }
    return {
        "case_id": case_id,
        "mission_id": 21033,
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
                    help="1-based shard index into the predeclared matrix")
    ap.add_argument("--nshards", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    if args.list:
        print(" ".join(ORDER))
        return 0
    if args.shard and args.nshards:
        if not 1 <= args.shard <= len(ORDER):
            raise SystemExit(f"shard must be in 1..{len(ORDER)}")
        args.case = ORDER[args.shard - 1]

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
