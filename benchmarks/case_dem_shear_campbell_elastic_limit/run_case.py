#!/usr/bin/env python3
"""CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT: elastic-limit shear-stress scaling.

One point per invocation (so the ladder can be sharded), plus an ``--aggregate``
mode that turns the point JSONs into the scaling fit, the tables and the figure
(zero hand-copied numbers: every value in the report is emitted from the point
files by this script).

    python benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py \
        --tag kn1e4 --kn 1e4 --out artifacts/case_dem_shear_campbell_elastic_limit
    python benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py \
        --aggregate --out artifacts/case_dem_shear_campbell_elastic_limit
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402

from tensordem.elastic_limit_shear import (  # noqa: E402
    ElasticLimitShearCase, build_configuration, configuration_report,
    shear_stress_scaling, simulate_couette_shear)

REF = {"doi": "10.1017/S002211200200109X",
       "citation": "Campbell, C. S. (2002) Granular shear flows at the elastic "
                   "limit. J. Fluid Mech. 465, 261-291.",
       "bagnold_exponent": 2.0,
       "tolerance_rel": 0.15}


def run_point(args) -> dict:
    case = ElasticLimitShearCase(
        n_particles=args.n, solid_fraction=args.phi, kn=args.kn,
        wall_velocity=args.wall_velocity, gap_over_d=args.gap_over_d,
        total_strain=args.strain, average_from=args.average_from,
        restitution=args.restitution, mu=args.mu, seed=args.seed,
        sample_every=args.sample_every)
    cfg = build_configuration(case)
    conf = configuration_report(case, cfg["pos"])
    t0 = time.time()
    res = simulate_couette_shear(case, config=cfg)
    res.update({"tag": args.tag, "config": conf,
                "wall_clock_s": time.time() - t0,
                "reference": REF})
    return res


def aggregate(out: str) -> dict:
    files = sorted(glob.glob(os.path.join(out, "point_*.json")))
    points = [json.load(open(f)) for f in files]
    points.sort(key=lambda p: p["kn"])
    payload = {"reference": REF, "points": points}
    gated = [p for p in points if p.get("gate_pass")]
    payload["n_points"] = len(points)
    payload["n_gated"] = len(gated)
    if len(gated) >= 3:
        payload["scaling"] = shear_stress_scaling(points)
    else:
        payload["scaling"] = {"error": "fewer than 3 points passed the physics "
                                       "gates; no scaling fit is legitimate",
                              "gated_tags": [p["tag"] for p in gated],
                              "failed_tags": [p["tag"] for p in points
                                              if not p.get("gate_pass")]}
    json.dump(payload, open(os.path.join(out, "scaling.json"), "w"), indent=1)

    # figure: tau* vs kappa (log-log) with the Bagnold plateau marked
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        kap = [p["kappa"] for p in points]
        tst = [p["tau_star"] for p in points]
        ok = [p.get("gate_pass") for p in points]
        ax.loglog([k for k, g in zip(kap, ok) if g], [t for t, g in zip(tst, ok) if g],
                  "o-", color="#1b6ca8", label="gated points")
        ax.loglog([k for k, g in zip(kap, ok) if not g], [t for t, g in zip(tst, ok) if not g],
                  "x", color="#b02a37", label="rejected by gate")
        ax.set_xlabel(r"$\kappa = k_n/(\rho d^3 \dot\gamma^2)$")
        ax.set_ylabel(r"$\tau^* = \tau/(\rho d^2 \dot\gamma^2)$")
        ax.set_title("CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT: tau*(kappa)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_tau_star_kappa.png"), dpi=130)
    except Exception as exc:  # pragma: no cover - plotting is optional
        payload["figure_error"] = str(exc)
    print(json.dumps({k: v for k, v in payload.items()
                      if k in ("n_points", "n_gated")}, indent=1))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--kn", type=float, default=1.0e4)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--phi", type=float, default=0.80)
    ap.add_argument("--wall-velocity", type=float, default=4.0)
    ap.add_argument("--gap-over-d", type=float, default=16.0)
    ap.add_argument("--strain", type=float, default=3.0)
    ap.add_argument("--average-from", type=float, default=0.4)
    ap.add_argument("--restitution", type=float, default=0.7)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=16136)
    ap.add_argument("--sample-every", type=int, default=25)
    ap.add_argument("--out", default="artifacts/case_dem_shear_campbell_elastic_limit")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.aggregate:
        aggregate(args.out)
        return
    if not args.tag:
        raise SystemExit("--tag is required for a single point")
    res = run_point(args)
    json.dump(res, open(os.path.join(args.out, f"point_{args.tag}.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("tag", "kappa", "tau", "pressure",
                                         "tau_star", "gate", "gate_pass")}, indent=1))


if __name__ == "__main__":
    main()
