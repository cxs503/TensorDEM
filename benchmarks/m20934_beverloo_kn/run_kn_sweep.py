#!/usr/bin/env python3
"""CASE-DEM-SILO-BEVERLOO (m20934): contact-stiffness (kn) sensitivity sweep.

Context
-------
The ledger anchor ``CASE-DEM-SILO-BEVERLOO`` (acceptance: |rel err| < 10% vs
Beverloo 1961) is still ``open``.  Prior evidence:

* m20055 (flat-bottom silo, ``tensordem.silo_discharge``): D/d = 4,
  ``kn = 1e5 N/m`` -> **+25.0%** (honest FAIL, documented in
  ``docs/CASE-DEM-SILO-BEVERLOO_m20055.md``).
* m15259 (hopper kernel, ``verification.granular.hopper_discharge``):
  D/d = 5, ``kn = 1e6 N/m`` -> **-0.4%** (PASS).
* m20585 hypothesised that the soft contact stiffness (``kn = 1e5``) is the
  dominant systematic error source (excess contact deformation enlarges the
  effective orifice), but the hypothesis was **never computationally verified**
  (its cluster jobs died on SSH timeouts; see ``verdict_m20585.json``).

This sweep verifies that hypothesis on the *flat-bottom silo* (the anchor
geometry): the m20055 case matrix (D/d = 4, 5, 6) is re-run with
``kn in {1e5, 3e5, 1e6}`` plus a ``dt/2`` arm for the primary D5/kn=1e6 run
(timestep-artifact guard).  Everything else (geometry, material, friction,
settling protocol, seed) is byte-identical to m20055 so the only variable is
the contact stiffness (and, in the last arm, the timestep).

The Beverloo reference coefficients are read from
``benchmarks/m20055_beverloo/ref_beverloo_1961.json`` (zero hand-copy).

Usage
-----
    PYTHONPATH=src python benchmarks/m20934_beverloo_kn/run_kn_sweep.py --list
    PYTHONPATH=src python benchmarks/m20934_beverloo_kn/run_kn_sweep.py \
        --shard 3 --nshards 10 --out-dir runs/m20934_beverloo_kn
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
DEFAULT_OUT = ROOT / "runs" / "m20934_beverloo_kn"

# ---------------------------------------------------------------------------
# Predeclared matrix.  BASE is copied verbatim from
# benchmarks/m20055_beverloo/run_case.py (prereg/m20055_beverloo.md) so the
# only variable across arms is kn (and dt in the dthalf arm).
# ---------------------------------------------------------------------------
BASE = dict(
    radius=0.010,          # d = 20 mm
    density=2500.0,        # glass
    mu=0.30,               # particle friction (m20055 primary)
    mu_rolling=0.02,
    restitution=0.30,
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
    seed=20055,
    settle_time=0.60,
    window=0.50,
    n_windows=5,
    steady_quantiles=(0.20, 0.80),
)

# (tag, orifice_diameter, discharge_time, kn, dt_scale)
ARMS = [
    ("D4_kn1e5",        0.080, 4.5, 1.0e5, 1.0),
    ("D4_kn3e5",        0.080, 4.5, 3.0e5, 1.0),
    ("D4_kn1e6",        0.080, 4.5, 1.0e6, 1.0),
    ("D5_kn1e5",        0.100, 3.0, 1.0e5, 1.0),
    ("D5_kn3e5",        0.100, 3.0, 3.0e5, 1.0),
    ("D5_kn1e6",        0.100, 3.0, 1.0e6, 1.0),
    ("D5_kn1e6_dthalf", 0.100, 3.0, 1.0e6, 0.5),
    ("D6_kn1e5",        0.120, 2.0, 1.0e5, 1.0),
    ("D6_kn3e5",        0.120, 2.0, 3.0e5, 1.0),
    ("D6_kn1e6",        0.120, 2.0, 1.0e6, 1.0),
]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:                                    # pragma: no cover
        return "unknown"


def build_arm(tag: str, D: float, t_dis: float, kn: float, dt_scale: float) -> SiloCase:
    p = dict(BASE)
    p.update(orifice_diameter=D, discharge_time=t_dis, kn=kn)
    case = SiloCase(**p)
    if dt_scale != 1.0:
        p["dt"] = dt_scale * case.timestep
        case = SiloCase(**p)
    return case


def analyse(tag: str, case: SiloCase, res: dict, ref: dict, wall: float) -> dict:
    C = float(ref["coefficients"]["C"])
    k = float(ref["coefficients"]["k"])
    d = case.diameter
    D = case.orifice_diameter
    rho_b = float(res["bulk_density_bed_kg_m3"])
    g = case.gravity
    Deff = D - k * d
    W_ref = C * rho_b * math.sqrt(g) * Deff ** 2.5
    W_got = float(res["steady_flow_rate_kg_s"])
    rel = (W_got - W_ref) / W_ref if W_ref > 0 else float("nan")
    C_got = W_got / (rho_b * math.sqrt(g) * Deff ** 2.5) if Deff > 0 else float("nan")
    return {
        "tag": tag,
        "case_id": tag.split("_kn")[0],
        "kn_N_m": case.kn,
        "dt_s": float(res["dt"]),
        "D_over_d": D / d,
        "orifice_diameter": D,
        "particle_diameter": d,
        "gravity": g,
        "n_initial": int(res["n_initial"]),
        "n_bed": int(res["n_bed"]),
        "bed_height_m": float(res["bed_height_m"]),
        "rho_b_bed_kg_m3": rho_b,
        "packing_fraction_bed": float(res["packing_fraction_bed"]),
        "n_discharged_total": int(res["n_discharged_total"]),
        "steady_interval_s": res["steady_interval_s"],
        "steady_rate_windows_kg_s": res["steady_rate_windows_kg_s"],
        "steady_rate_cv": float(res["steady_rate_cv"]),
        "discharge_time_s": [float(x) for x in res["discharge_time_s"]],
        "discharge_cumulative_n": [int(x) for x in res["discharge_cumulative_n"]],
        "W_ref_kg_s": W_ref,
        "W_got_kg_s": W_got,
        "rel_err": rel,
        "rel_err_pct": rel * 100.0 if math.isfinite(rel) else None,
        "C_recovered": C_got,
        "pass_10pct": bool(abs(rel) < 0.10) if math.isfinite(rel) else False,
        "pass_15pct": bool(abs(rel) < 0.15) if math.isfinite(rel) else False,
        "steps": int(res["steps"]),
        "wall_clock_s": wall,
        "git_sha": git_sha(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "ref_source": ref["source"],
        "command": " ".join([sys.executable, *sys.argv]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0, help="1-based shard index")
    ap.add_argument("--nshards", type=int, default=0)
    ap.add_argument("--arm", default=None, help="run a single arm by tag")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    if args.list:
        for i, (tag, *_) in enumerate(ARMS, 1):
            print(f"{i:2d}  {tag}")
        return 0

    if args.arm:
        arms = [a for a in ARMS if a[0] == args.arm]
        if not arms:
            raise SystemExit(f"unknown arm {args.arm!r}; use --list")
    elif args.shard and args.nshards:
        if not 1 <= args.shard <= len(ARMS):
            raise SystemExit(f"shard must be in 1..{len(ARMS)}")
        arms = [ARMS[args.shard - 1]]
    else:
        arms = ARMS

    ref = json.loads(REF_PATH.read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = 0
    for tag, D, t_dis, kn, dt_scale in arms:
        case = build_arm(tag, D, t_dis, kn, dt_scale)
        t0 = time.time()
        res = simulate_silo(case)
        wall = time.time() - t0
        out = analyse(tag, case, res, ref, wall)
        path = out_dir / f"{tag}.json"
        path.write_text(json.dumps(out, indent=2, default=float))
        print(f"[{tag}] D/d={out['D_over_d']:.2f} kn={kn:.1e} "
              f"n_bed={out['n_bed']} rho_b={rho_b:.0f} "
              f"discharged={out['n_discharged_total']}/{out['n_initial']} "
              f"W={out['W_got_kg_s']:.4f} W_ref={out['W_ref_kg_s']:.4f} "
              f"rel={out['rel_err_pct']:+.1f}% CV={out['steady_rate_cv']:.3f} "
              f"wall={wall:.0f}s -> {path}", flush=True)
        rc = rc or (0 if out["pass_10pct"] else 1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
