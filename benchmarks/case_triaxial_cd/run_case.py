#!/usr/bin/env python3
"""CASE-TRIAXIAL-CD: drained triaxial friction angle of a granular specimen.

One point per invocation (so a ladder can be sharded across cluster workers),
plus an ``--aggregate`` mode that turns the point JSONs into the report tables
and the figure (zero hand-copied numbers: every value in the report is emitted
by this script from the point files).

Examples
--------
    python benchmarks/case_triaxial_cd/run_case.py --tag s100_mu50_mur25 \
        --sigma3 1.0e5 --mu 0.5 --mu-r 0.25 --out artifacts/case_triaxial_cd
    python benchmarks/case_triaxial_cd/run_case.py --aggregate \
        --out artifacts/case_triaxial_cd
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensordem.triaxial_cd import (CDTriaxialCase, friction_angle_deg,  # noqa: E402
                                  mohr_coulomb_fit, run_cd_triaxial)

# Reference band (literature sand friction angle): see docs/CASE_TRIAXIAL_CD_M15235.md
REFERENCE_BAND_DEG = (30.0, 40.0)
TOLERANCE_REL = 0.10


def reference_error(phi_deg: float) -> dict[str, Any]:
    """Distance to the literature band, in the unit the case is graded in.

    The band is the *criterion*; the point target is its midpoint, declared here
    a priori (it is the geometric centre of the band, not a fitted number), so
    the grader has a single relative error per point plus an in-band flag.
    """
    lo, hi = REFERENCE_BAND_DEG
    mid = 0.5 * (lo + hi)
    inside = bool(lo <= phi_deg <= hi)
    # if outside, the nearest band edge is the reference (conservative)
    ref = mid if inside else (lo if phi_deg < lo else hi)
    err = (phi_deg - ref) / ref
    return {
        "band_deg": [lo, hi],
        "band_midpoint_deg": mid,
        "reference_used_deg": ref,
        "inside_band": inside,
        "relative_error": float(err),
        "pass_10pct": bool(inside and abs(err) <= TOLERANCE_REL),
    }


def run_point(args: argparse.Namespace) -> dict[str, Any]:
    case = CDTriaxialCase(
        sigma3=args.sigma3, mu=args.mu, mu_rolling=args.mu_r,
        radius=args.radius, density=args.density, kn=args.kn,
        nx=args.nx, ny=args.ny, nz=args.nz,
        lattice_gap=args.lattice_gap, jitter=args.jitter, seed=args.seed,
        axial_strain_rate=args.rate, axial_strain_target=args.strain,
        sample_every=args.sample_every, dt=args.dt,
        consolidate_steps=args.consolidate_steps,
    )
    t0 = time.time()
    res = run_cd_triaxial(case)
    res["tag"] = args.tag
    res["wall_time_s"] = round(time.time() - t0, 1)
    res["reference"] = reference_error(res["peak"]["phi_deg"])
    res["repro_cmd"] = (
        "python benchmarks/case_triaxial_cd/run_case.py "
        f"--tag {args.tag} --sigma3 {args.sigma3:g} --mu {args.mu:g} "
        f"--mu-r {args.mu_r:g} --nx {args.nx} --ny {args.ny} --nz {args.nz} "
        f"--radius {args.radius:g} --kn {args.kn:g} --rate {args.rate:g} "
        f"--strain {args.strain:g} --seed {args.seed} --out {args.out}")
    return res


def _fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def aggregate(out_dir: str) -> dict[str, Any]:
    files = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    points = []
    for path in files:
        if os.path.basename(path) in ("summary.json",):
            continue
        with open(path) as fh:
            rec = json.load(fh)
        if "peak" not in rec or "tag" not in rec:
            continue
        rec["_path"] = path
        points.append(rec)
    if not points:
        raise SystemExit(f"no point files in {out_dir}")

    ladder = [p for p in points if p.get("tag", "").startswith("ladder_")]
    sens_mu = [p for p in points if p.get("tag", "").startswith("sens_mu_")]
    sens_mur = [p for p in points if p.get("tag", "").startswith("sens_mur_")]
    conv = [p for p in points if p.get("tag", "").startswith("conv_")]

    def by_sigma3(rows):
        return sorted(rows, key=lambda r: r["case"]["sigma3"])

    summary: dict[str, Any] = {"n_points": len(points), "points": []}
    for p in points:
        summary["points"].append({
            "tag": p["tag"],
            "sigma3_Pa": p["case"]["sigma3"],
            "mu": p["case"]["mu"],
            "mu_rolling": p["case"]["mu_rolling"],
            "phi_peak_deg": p["peak"]["phi_deg"],
            "phi_critical_deg": p["critical_state"]["phi_deg"],
            "q_peak_Pa": p["peak"]["q_Pa"],
            "drained_pass": p["drained_gate"]["pass"],
            "drained_drift": p["drained_gate"]["max_relative_drift"],
            "reference": p["reference"],
            "wall_time_s": p["wall_time_s"],
            "path": p["_path"],
        })

    # ---- headline: the ladder point at sigma3 = 100 kPa, base parameters ----
    headline = [p for p in ladder if abs(p["case"]["sigma3"] - 1.0e5) < 1.0]
    headline = headline[0] if headline else max(ladder or points,
                                                key=lambda r: r["peak"]["phi_deg"])
    summary["headline"] = {
        "tag": headline["tag"],
        "sigma3_Pa": headline["case"]["sigma3"],
        "phi_peak_deg": headline["peak"]["phi_deg"],
        "phi_critical_deg": headline["critical_state"]["phi_deg"],
        "reference": headline["reference"],
    }

    # ---- Mohr-Coulomb consistency across the confining-pressure ladder ----
    if len(ladder) >= 2:
        rows = by_sigma3(ladder)
        s3 = [r["peak"]["sigma3_Pa"] for r in rows]
        s1 = [r["peak"]["sigma1_Pa"] for r in rows]
        eps = [r["peak"]["strain_axial"] for r in rows]
        # sigma1 = a sigma3 + b  <=>  sin(phi) = (a-1)/(a+1) for a cohesion-free soil
        mc = mohr_coulomb_fit(s3, s1)
        slope = mc["tan_phi"]
        summary["ladder_fit"] = {
            "fit": "sigma1 = a*sigma3 + b (peak, drained)",
            "slope_a": slope,
            "intercept_b_Pa": mc["cohesion_pa"],
            "sin_phi_from_slope": (slope - 1.0) / (slope + 1.0),
            "phi_from_slope_deg": friction_angle_deg(slope, 1.0),
            "r_squared": mc["r_squared"],
            "max_relative_deviation": mc["max_relative_deviation"],
            "peak_strain_axial": eps,
            "levels": [
                {"sigma3_Pa": r["peak"]["sigma3_Pa"],
                 "sigma1_Pa": r["peak"]["sigma1_Pa"],
                 "phi_peak_deg": r["peak"]["phi_deg"],
                 "phi_critical_deg": r["critical_state"]["phi_deg"],
                 "drained_drift": r["drained_gate"]["max_relative_drift"],
                 "in_band": r["reference"]["inside_band"],
                 "pass": r["reference"]["pass_10pct"]}
                for r in rows],
        }
        phis = [r["peak"]["phi_deg"] for r in rows]
        summary["ladder_fit"]["phi_spread_deg"] = max(phis) - min(phis)
        summary["ladder_fit"]["phi_spread_relative"] = (
            (max(phis) - min(phis)) / (sum(phis) / len(phis)))

    # ---- parameter sensitivity ----
    for name, rows in (("mu_sensitivity", sens_mu), ("mu_rolling_sensitivity", sens_mur)):
        if not rows:
            continue
        key = "mu" if name == "mu_sensitivity" else "mu_rolling"
        summary[name] = sorted(
            [{"value": r["case"][key],
              "phi_peak_deg": r["peak"]["phi_deg"],
              "phi_critical_deg": r["critical_state"]["phi_deg"],
              "tan_phi_peak": math.tan(math.radians(r["peak"]["phi_deg"])),
              "tag": r["tag"]} for r in rows],
            key=lambda r: r["value"])

    # ---- strain-rate convergence ----
    if conv:
        summary["strain_rate_convergence"] = sorted(
            [{"strain_rate_per_s": r["case"]["axial_strain_rate"],
              "phi_peak_deg": r["peak"]["phi_deg"],
              "phi_critical_deg": r["critical_state"]["phi_deg"],
              "inertial_number": r["quasi_static"]["inertial_number_I"],
              "tag": r["tag"]} for r in conv],
            key=lambda r: r["strain_rate_per_s"])

    summary["reference_band_deg"] = list(REFERENCE_BAND_DEG)
    summary["tolerance_relative"] = TOLERANCE_REL
    summary["all_drained_pass"] = all(
        p["drained_gate"]["pass"] for p in points)
    summary["all_reference_pass"] = all(
        p["reference"]["pass_10pct"] for p in ladder or points)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="point")
    ap.add_argument("--sigma3", type=float, default=1.0e5)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--mu-r", dest="mu_r", type=float, default=0.25)
    ap.add_argument("--radius", type=float, default=5.0e-3)
    ap.add_argument("--density", type=float, default=2650.0)
    ap.add_argument("--kn", type=float, default=1.0e6)
    ap.add_argument("--nx", type=int, default=6)
    ap.add_argument("--ny", type=int, default=6)
    ap.add_argument("--nz", type=int, default=5)
    ap.add_argument("--lattice-gap", dest="lattice_gap", type=float, default=0.15)
    ap.add_argument("--jitter", type=float, default=0.30)
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--strain", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=15235)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--sample-every", dest="sample_every", type=int, default=100)
    ap.add_argument("--consolidate-steps", dest="consolidate_steps", type=int,
                    default=40000)
    ap.add_argument("--out", default="artifacts/case_triaxial_cd")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.aggregate:
        summary = aggregate(args.out)
        path = os.path.join(args.out, "summary.json")
        with open(path, "w") as fh:
            json.dump(summary, fh, indent=1)
        print(json.dumps({k: v for k, v in summary.items() if k != "points"}, indent=1))
        return

    res = run_point(args)
    path = os.path.join(args.out, f"{args.tag}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({
        "tag": res["tag"],
        "phi_peak_deg": res["peak"]["phi_deg"],
        "phi_critical_deg": res["critical_state"]["phi_deg"],
        "reference": res["reference"],
        "drained": res["drained_gate"],
        "wall_time_s": res["wall_time_s"],
        "path": path,
    }, indent=1))


if __name__ == "__main__":
    main()