#!/usr/bin/env python3
"""CASE-DEM-SILO-BEVERLOO (m21421): wall-proximity scan of the flat-bottom
silo discharge law against Beverloo (1961).

One variable is changed against m21033: the silo radius.  Bed height grows with
the silo cross-section so that H/D = 4.0 stays fixed; the particle, the orifice
(D/d = 5), the contact model, the settle/plug protocol and the seed are
identical, which makes the scan a *paired* experiment.

Each case also carries the new independent mass-flux meter
(``SiloCase.measure_orifice_flux``), so the steady flow rate is measured twice,
by two independent bookkeepings.

Reference coefficients are machine-read from
``benchmarks/m20055_beverloo/ref_beverloo_1961.json`` (zero hand-typed
numbers).  The pre-registration lives in
``benchmarks/m21421_beverloo_wall/prereg.md``.

Usage
-----
    PYTHONPATH=src python benchmarks/m21421_beverloo_wall/run_case.py --case S26
    PYTHONPATH=src python benchmarks/m21421_beverloo_wall/run_case.py --shard 1 --nshards 4
    PYTHONPATH=src python benchmarks/m21421_beverloo_wall/run_case.py --list
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
OUT_DIR = ROOT / "runs" / "m21421_beverloo_wall"
MISSION_ID = 21421

# ---------------------------------------------------------------------------
# Predeclared matrix (benchmarks/m21421_beverloo_wall/prereg.md).
# Bed height is held at H = 0.400 m (H/D = 4.0) in every case, so the column of
# material above the orifice is identical and the *only* change is how far the
# cylindrical side wall sits from the orifice.  n_bed therefore grows with the
# silo cross-section (~1525 / ~2765 / ~3610).
# ---------------------------------------------------------------------------
BASE = dict(
    radius=0.010,           # d = 20 mm
    density=2500.0,
    mu=0.15,
    mu_rolling=0.02,
    restitution=0.30,
    kn=1.0e6,
    kt_ratio=0.5,
    gravity=9.81,
    orifice_diameter=0.100,  # D/d = 5
    plate_thickness=0.010,   # 0.5 d
    wall_mu=0.0,             # smooth side wall (Beverloo glass-bead condition)
    packing="settled",
    plug_during_settle=True,
    settle_loose_gap=0.30,
    packing_gap=0.0,
    jitter=0.05,
    seed=21033,              # same draw for every case -> paired scan
    settle_time=0.60,
    discharge_time=3.0,
    window=0.50,
    n_windows=5,
    steady_quantiles=(0.20, 0.80),
    measure_orifice_flux=True,
)

CASES = {
    # id        silo_radius, fill_height (H = 4.0 D)   note
    "S26": dict(silo_radius=0.130, fill_height=0.400),  # baseline == m21033 D5
    "S35": dict(silo_radius=0.175, fill_height=0.400),  # silo_D/D = 3.5
    "S40": dict(silo_radius=0.200, fill_height=0.400),  # silo_D/D = 4.0
    "F26": dict(silo_radius=0.130, fill_height=0.400,
                plug_during_settle=False),              # protocol control
}

ORDER = ["S26", "S35", "S40", "F26"]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:                                     # pragma: no cover
        return "unknown"


def build_case(case_id: str) -> SiloCase:
    params = dict(BASE)
    params.update(CASES[case_id])
    return SiloCase(**params)


def analyse(case_id: str, case: SiloCase, res: dict, ref: dict, wall: float) -> dict:
    C, k = float(ref["coefficients"]["C"]), float(ref["coefficients"]["k"])
    d = case.diameter
    D = case.orifice_diameter
    g = case.gravity
    rho_b = float(res["bulk_density_bed_kg_m3"])
    Deff = D - k * d
    W_ref = C * rho_b * math.sqrt(g) * Deff ** 2.5
    W_got = float(res["steady_flow_rate_kg_s"])
    rel = (W_got - W_ref) / W_ref if W_ref > 0 else float("nan")
    C_got = W_got / (rho_b * math.sqrt(g) * Deff ** 2.5) if Deff > 0 else float("nan")

    flux = res.get("orifice_flux", {"measured": False})
    W_flux = float(flux.get("steady_flux_kg_s", float("nan"))) if flux.get("measured") else float("nan")
    flux_gap = ((W_flux - W_got) / W_got
                if (math.isfinite(W_flux) and W_got > 0.0) else float("nan"))

    out_of_applicability = 2.0 * case.silo_radius / D < float(
        ref["validation_domain"]["silo_D_over_D_min"])

    verdict = {
        "rel_err_W": rel,
        "abs_rel_err_W": abs(rel),
        "C_recovered": C_got,
        "pass_10pct": bool(abs(rel) < 0.10),
        "pass_15pct": bool(abs(rel) < 0.15),
        "W_ref_kg_s": W_ref,
        "W_got_kg_s": W_got,
        "W_flux_kg_s": W_flux,
        "flux_vs_count_rel": flux_gap,
        "flux_consistent_3pct": bool(math.isfinite(flux_gap) and abs(flux_gap) < 0.03),
        "Deff_m": Deff,
        "rho_b_kg_m3": rho_b,
        "silo_D_over_D": 2.0 * case.silo_radius / D,
        "silo_D_over_D_in_domain": not out_of_applicability,
        "discharged_fraction": (float(res["n_discharged_total"]) / float(res["n_initial"])
                                if res["n_initial"] else float("nan")),
    }
    verdict["protocol_ok"] = bool(
        float(res["steady_rate_cv"]) < 0.30 and verdict["discharged_fraction"] >= 0.40)

    return {
        "case_id": case_id,
        "mission_id": MISSION_ID,
        "anchor": "CASE-DEM-SILO-BEVERLOO",
        "wall_clock_s": wall,
        "git_sha": git_sha(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "dimensionless": {
            "D_over_d": D / d,
            "silo_D_over_D": 2.0 * case.silo_radius / D,
            "fill_ratio_H_over_D": case.fill_height / D,
            "rho_b_over_rho_p": rho_b / case.density,
            "packing_fraction_bed": float(res["packing_fraction_bed"]),
            "mu_p": case.mu,
            "mu_wall": case.wall_mu,
            "restitution": case.restitution,
            "kn_N_m": case.kn,
            "dt_over_contact_period": case.timestep / (
                2.0 * math.pi * math.sqrt(case.material.mass / case.kn)),
            "n_particles_initial": int(res["n_initial"]),
            "n_particles_bed": int(res["n_bed"]),
        },
        "telemetry": res,
        "verdict": verdict,
        "ref_source": ref["source"],
        "command": " ".join([sys.executable, *sys.argv]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="S26")
    ap.add_argument("--shard", type=int, default=0, help="1-based shard index")
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
    (out_dir / f"{args.case}.json").write_text(
        json.dumps(out, indent=2, default=float))

    v = out["verdict"]
    print(f"[{args.case}] silo_D/D={v['silo_D_over_D']:.2f} "
          f"n_bed={out['dimensionless']['n_particles_bed']} "
          f"rho_b={v['rho_b_kg_m3']:.0f} "
          f"W={v['W_got_kg_s']:.4f} W_flux={v['W_flux_kg_s']:.4f} "
          f"W_ref={v['W_ref_kg_s']:.4f} rel={100*v['rel_err_W']:+.1f}% "
          f"C={v['C_recovered']:.3f} CV={res['steady_rate_cv']:.3f} "
          f"wall={wall:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())