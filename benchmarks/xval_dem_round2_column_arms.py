#!/usr/bin/env python
"""XVAL · CASE-COLUMN-COLLAPSE: macroscopic effect of the rolling-lever convention.

Runs the frozen ``a = 1`` column of ``CASE-COLUMN-COLLAPSE`` (Lube et al. 2004)
under three rolling-resistance conventions and compares the runout ratio:

    legacy            rolling_lever="particle_radius", mu_r = 0.5  (certified)
    oss_naive         rolling_lever="reduced_radius",  mu_r = 0.5  (OSS params dropped in)
    oss_mapped        rolling_lever="reduced_radius",  mu_r = 1.0  (= 2 x 0.5, the
                      lever mapping measured on the pair scene)

Usage: PYTHONPATH=src python benchmarks/xval_dem_round2_column_arms.py
Writes artifacts/xval_dem_round2_method_selection/column_arms.json
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.granular_column.axisym import (AxisymCase,   # noqa: E402
                                             lube_2004_emplacement_time,
                                             lube_2004_runout_ratio,
                                             simulate_axisymmetric_column)

OUT = os.path.join(ROOT, "artifacts", "xval_dem_round2_method_selection")
BASE = dict(aspect_ratio=1.0, grains_per_radius=8.0, radius=0.03, density=2500.0,
            mu=0.5, restitution=0.5, kn=6.0e4, kt_ratio=0.5, gravity=9.81,
            packing_gap=0.02, seed=12345, dt_scale=1.0, duration_factor=8.0,
            min_time=0.5, sample_stride=20, rest_ke_tol=1.0e-4, rest_hold=200,
            mu_wall=0.8)
ARMS = (("legacy_mu0.5", "particle_radius", 0.5),
        ("oss_naive_mu0.5", "reduced_radius", 0.5),
        ("oss_mapped_mu1.0", "reduced_radius", 1.0))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    ref_runout = lube_2004_runout_ratio(BASE["aspect_ratio"])
    r_i = BASE["grains_per_radius"] * 2.0 * BASE["radius"]
    ref_time = lube_2004_emplacement_time(BASE["aspect_ratio"], r_i, 9.81)
    res = {"reference": {"runout_ratio": ref_runout,
                         "emplacement_time_s": ref_time,
                         "source": "Lube et al. 2004 JFM 508:175-199 (evaluated at run time)"},
           "base": BASE, "arms": {}}
    for name, lever, mu_r in ARMS:
        case = AxisymCase(mu_rolling=mu_r, rolling_lever=lever, **BASE)
        t0 = time.time()
        out = simulate_axisymmetric_column(case)
        rec = {
            "rolling_lever": lever, "mu_rolling": mu_r,
            "runout_ratio": float(out["runout_ratio_p995"]),
            "runout_ratio_max": float(out["runout_ratio_max"]),
            "emplacement_time_s": float(out["t_reach"]) if "t_reach" in out else None,
            "n_particles": int(out["n_particles"]),
            "elapsed_s": round(time.time() - t0, 1),
        }
        rec["runout_err_vs_lube"] = float((rec["runout_ratio"] - ref_runout) / ref_runout)
        res["arms"][name] = rec
        print(name, rec, flush=True)
    leg = res["arms"]["legacy_mu0.5"]["runout_ratio"]
    for k, v in res["arms"].items():
        v["runout_rel_diff_vs_legacy"] = float((v["runout_ratio"] - leg) / leg)
    with open(os.path.join(OUT, "column_arms.json"), "w") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "runout_ratio_max"}
                      for k, v in res["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
