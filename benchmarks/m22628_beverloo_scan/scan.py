#!/usr/bin/env python3
"""CASE-DEM-SILO-BEVERLOO (m22628): orifice scan + fill-height independence.

Why this scan exists
--------------------
Every earlier round of the campaign (m20055 / m20707 / m21033 / m21274 /
m22124) compared the *amplitude* of the measured discharge rate with the
Beverloo prediction evaluated at the **literature** ``k = 1.4``.  That test is
blind to two of the three falsifiable statements the law actually makes:

1. the **shape** ``W ∝ (D - k d)**(5/2)`` -- the exponent 5/2;
2. the **finite-size offset** ``k d`` -- the intercept, ``k in [1.2, 1.6]``;
3. the **independence of the filling height** ``H``.

A single-orifice amplitude comparison folds all three into one number, so a
wrong ``k`` shows up as a ``D``-dependent amplitude error -- exactly the
signature observed in the campaign (``+20 %`` at ``D/d = 4``, ``-6.9 %`` /
``+5.9 %`` at ``D/d = 5 / 6``).  This module sweeps the orifice at fixed
material/geometry and sweeps the filling height at fixed orifice, so the three
statements can be tested separately and the ``(C, k, p)`` triple can be
*inverted* from the DEM data instead of assumed.

Geometry is chosen inside the law's stated domain for **every** D/d of the
scan (``D/d >= 4``, ``silo_D/D >= 2.5``, ``bed_height/D >= 1.5``); the residual
overlap of the soft-contact kernel is reported with each case.

Usage
-----
    PYTHONPATH=src python benchmarks/m22628_beverloo_scan/scan.py --list
    PYTHONPATH=src python benchmarks/m22628_beverloo_scan/scan.py --case D50
    PYTHONPATH=src python benchmarks/m22628_beverloo_scan/scan.py --shard 1 --nshards 8
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

from tensordem.silo_flow.beverloo import DischargeCase, run_discharge  # noqa: E402

REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"
OUT_DIR = ROOT / "runs" / "m22628_beverloo_scan"
PREREG = ROOT / "benchmarks" / "m22628_beverloo_scan" / "prereg.md"

# ---------------------------------------------------------------------------
# Shared configuration (identical for every case of the scan)
# ---------------------------------------------------------------------------
D = 0.003                # particle diameter d [m]
SILO_R = 0.030           # silo radius [m]
H_BASE = 0.045           # filling height of the D-scan [m]

BASE = dict(
    particle_diameter=D,
    silo_radius=SILO_R,
    fill_height=H_BASE,
    wall_mu=0.0,         # smooth wall: Beverloo's glass-bead configuration
    seed=22628,
    settle_time=0.06,
    discard_time=0.12,   # purge the plug-removal burst before measuring
    window_time=0.15,
    n_windows=5,
    kn=5.0e3,            # soft contact: overlap ~1-3 % r (see docs)
    mu=0.50,             # inter-particle friction, calibrated in m20707
    mu_rolling=0.05,
)

# ---------------------------------------------------------------------------
# Case matrix
# ---------------------------------------------------------------------------
# 5 orifice diameters at fixed filling height -> the (C, k, p) inversion
# 2 filling heights at fixed orifice              -> the H-independence test
# 1 repeat packing seed at the primary case       -> run-to-run scatter
CASES = {
    "D40":      dict(orifice_diameter=4.0 * D),
    "D45":      dict(orifice_diameter=4.5 * D),
    "D50":      dict(orifice_diameter=5.0 * D),
    "D55":      dict(orifice_diameter=5.5 * D),
    "D60":      dict(orifice_diameter=6.0 * D),
    "D50_H20":  dict(orifice_diameter=5.0 * D, fill_height=2.0 * 5.0 * D),
    "D50_H25":  dict(orifice_diameter=5.0 * D, fill_height=2.5 * 5.0 * D),
    "D50_s2":   dict(orifice_diameter=5.0 * D, seed=22629),
}

SCAN_ORDER = ["D40", "D45", "D50", "D55", "D60", "D50_H20", "D50_H25", "D50_s2"]


def case_ids() -> list[str]:
    return list(SCAN_ORDER)


def build_case(case_id: str) -> DischargeCase:
    if case_id not in CASES:
        raise KeyError(f"unknown case {case_id!r}; known: {', '.join(SCAN_ORDER)}")
    params = dict(BASE)
    params.update(CASES[case_id])
    return DischargeCase(**params)


def domain_check(case: DischargeCase, ref: dict) -> dict:
    """Applicability ratios of the Beverloo correlation for this case."""
    dom = ref["validation_domain"]
    d = case.particle_diameter
    D_o = case.orifice_diameter
    ratios = {
        "D_over_d": D_o / d,
        "silo_D_over_D": 2.0 * case.silo_radius / D_o,
        "bed_height_over_D": case.fill_height / D_o,
    }
    return {
        "ratios": ratios,
        "minima": {
            "D_over_d": dom["D_over_d_min"],
            "silo_D_over_D": dom["silo_D_over_D_min"],
            "bed_height_over_D": dom["bed_height_over_D_min"],
        },
        "in_domain": bool(
            ratios["D_over_d"] >= dom["D_over_d_min"]
            and ratios["silo_D_over_D"] >= dom["silo_D_over_D_min"]
            and ratios["bed_height_over_D"] >= dom["bed_height_over_D_min"]
        ),
    }


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def run_one(case_id: str, out_dir: Path) -> dict:
    ref = json.loads(REF_PATH.read_text())
    case = build_case(case_id)
    t0 = time.time()
    res = run_discharge(case)
    wall = time.time() - t0

    payload = {
        "campaign": "m22628",
        "case_id": case_id,
        "git_sha": git_sha(),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "wall_clock_s": wall,
        "ref_provenance": {
            "path": str(REF_PATH.relative_to(ROOT)),
            "C": ref["coefficients"]["C"],
            "k": ref["coefficients"]["k"],
            "verification_status": ref["provenance"]["verification_status"],
        },
        "domain": domain_check(case, ref),
        "case": {
            "orifice_diameter": case.orifice_diameter,
            "particle_diameter": case.particle_diameter,
            "silo_radius": case.silo_radius,
            "fill_height": case.fill_height,
            "wall_mu": case.wall_mu,
            "mu": case.mu,
            "mu_rolling": case.mu_rolling,
            "kn": case.kn,
            "seed": case.seed,
            "settle_time": case.settle_time,
            "discard_time": case.discard_time,
            "window_time": case.window_time,
            "n_windows": case.n_windows,
        },
        "result": {
            "n_bed": res["n_bed"],
            "rho_bulk": res["rho_bulk"],
            "bed_mass_kg": res["bed_mass_kg"],
            "dt": res["dt"],
            "steps": res["steps"],
            "flow_rate_kg_s": res["flow_rate_kg_s"],
            "flow_rate_std_kg_s": res["flow_rate_std_kg_s"],
            "flow_rate_windows_kg_s": res["flow_rate_windows_kg_s"],
            "flow_rate_steady_windows": res["flow_rate_steady_windows"],
            "particles_exited": res["particles_exited"],
            "n_final": res["n_final"],
            "energy": res["energy"],
        },
    }
    # mass that left the silo during the measurement, relative to the bed
    payload["result"]["mass_fraction_discharged"] = (
        res["particles_exited"] * (math.pi / 6.0) * case.particle_diameter ** 3
        * 2500.0 / res["bed_mass_kg"]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", help="single case id")
    ap.add_argument("--shard", type=int, default=None, help="1-based shard index")
    ap.add_argument("--nshards", type=int, default=len(SCAN_ORDER))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--list", action="store_true", help="print the case matrix")
    args = ap.parse_args(argv)

    ids = case_ids()
    if args.list:
        for cid in ids:
            c = build_case(cid)
            diag = domain_check(c, json.loads(REF_PATH.read_text()))
            print(
                f"{cid:9s} D/d={diag['ratios']['D_over_d']:.2f} "
                f"silo_D/D={diag['ratios']['silo_D_over_D']:.2f} "
                f"H/D={diag['ratios']['bed_height_over_D']:.2f} "
                f"in_domain={diag['in_domain']}"
            )
        return 0

    out_dir = Path(args.out)
    if args.case:
        chosen = [args.case]
    elif args.shard is not None:
        if not 1 <= args.shard <= args.nshards:
            raise SystemExit(f"shard {args.shard} outside 1..{args.nshards}")
        chosen = [cid for i, cid in enumerate(ids, start=1) if (i - 1) % args.nshards == args.shard - 1]
    else:
        chosen = ids

    for cid in chosen:
        payload = run_one(cid, out_dir)
        r = payload["result"]
        print(
            f"[{cid}] n={r['n_bed']} W={r['flow_rate_kg_s']:.6g} kg/s "
            f"+-{r['flow_rate_std_kg_s']:.3g} ({payload['wall_clock_s']:.0f} s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())