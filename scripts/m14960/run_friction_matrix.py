"""GAP-DEM-FRICTION round 2 (m14960): size scan + micro-friction calibration matrix.

One process = one (micro friction mu_p, geometry) point, swept over normal-stress
levels that share a single prepared bed (controlled variable, see m14721).

Output: ``artifacts/shear_cell_m14960/<tag>.json`` (config + per-level results +
shear histories) -- machine-readable so that every number in the report and the
tables is injected programmatically, never transcribed.

Usage
-----
    PYTHONPATH=src python scripts/m14960/run_friction_matrix.py \
        --tag mu050 --mu 0.5 --geom medium --sigmas 2000,4000,8000 --gamma 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from tensordem.shear_cell import (  # noqa: E402
    ShearCellCase,
    measure_macro_friction,
    prepare_bed,
    simulate_shear_cell,
)

# box configurations.  "small" is the m14721/m14840 cell (78 beads, 3 layers);
# "medium" doubles the depth -> 8 layers; "large" doubles x/y and doubles the
# fill again -> ~10 layers, ~1.2k beads.
GEOM = {
    "small": dict(box_width=0.080, box_depth=0.032, fill_height=0.050),
    "medium": dict(box_width=0.080, box_depth=0.064, fill_height=0.080),
    "large": dict(box_width=0.160, box_depth=0.080, fill_height=0.100),
}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                       "artifacts", "shear_cell_m14960")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--geom", default="medium", choices=sorted(GEOM))
    ap.add_argument("--sigmas", default="2000,4000,8000")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=None,
                    help="stationarity_tol of the validity gate (None = off)")
    ap.add_argument("--mu-rolling", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=6107)
    ap.add_argument("--consol", type=float, default=0.04)
    args = ap.parse_args()

    sigmas = [float(s) for s in args.sigmas.split(",") if s.strip()]
    base = ShearCellCase(mu=args.mu, mu_rolling=args.mu_rolling, seed=args.seed,
                         consol_time=args.consol, shear_strain_target=args.gamma,
                         stationarity_tol=args.tol, diagnostics=False, **GEOM[args.geom])

    t0 = time.time()
    bed = prepare_bed(base)
    t_bed = time.time() - t0
    print("[%s] bed: n=%d steps=%d phi=%.3f bed_top=%.4f  (%.0fs)"
          % (args.tag, bed["n_particles"], bed["steps"], bed["packing_fraction"],
             bed["bed_top"], t_bed), flush=True)

    levels = []
    for sigma in sigmas:
        t1 = time.time()
        out = simulate_shear_cell(replace(base, normal_stress=sigma), bed=bed)
        dt = time.time() - t1
        lvl = {
            "sigma_target_pa": sigma,
            "sigma_pa": out["normal_stress_measured"],
            "sigma_rel_err": out["normal_stress_rel_err"],
            "sigma_sem_pa": out["normal_stress_sem"],
            "tau_pa": out["shear_stress"],
            "tau_sem_pa": out["shear_stress_sem"],
            "mu": out["mu_macro"],
            "phi_deg": out["friction_angle_deg"],
            "tau_floor_pa": out["floor_shear_stress"],
            "tau_platen_vs_floor_rel": out["tau_platen_vs_floor_rel"],
            "bed_height_mean": out["bed_height_mean"],
            "shear_rate_actual": out["shear_rate_actual"],
            "inertial_number": out["inertial_number"],
            "drift_rel": out["shear_stress_drift_rel"],
            "drift_r2": out["r_squared"],
            "n_particles": out["n_particles"],
            "total_steps": out["total_steps"],
            "valid": out["valid"],
            "invalid_reason": out["invalid_reason"],
            "shear_strain_history": out["shear_strain_history"],
            "shear_stress_history": out["shear_stress_history"],
            "normal_stress_history": out["normal_stress_history"],
            "secs": dt,
        }
        levels.append(lvl)
        print("[%s] sigma %6.0f -> %8.2f (%+6.2f%%) tau %8.2f mu %6.4f phi %5.1f deg "
              "I %.4f drift %+.2f valid=%s (%.0fs)"
              % (args.tag, sigma, lvl["sigma_pa"], 100 * lvl["sigma_rel_err"],
                 lvl["tau_pa"], lvl["mu"], lvl["phi_deg"], lvl["inertial_number"],
                 lvl["drift_rel"], lvl["valid"], dt), flush=True)

    payload = {
        "tag": args.tag,
        "case": {
            "mu": args.mu, "mu_rolling": args.mu_rolling, "seed": args.seed,
            "radius": base.radius, "density": base.density, "kn": base.kn,
            "kt_ratio": base.kt_ratio, "restitution": base.restitution,
            "shear_velocity": base.shear_velocity,
            "shear_strain_target": args.gamma, "consol_time": args.consol,
            "stationarity_tol": args.tol, "geom": args.geom, **GEOM[args.geom],
        },
        "bed": {"steps": bed["steps"], "packing_fraction": bed["packing_fraction"],
                "bed_top": bed["bed_top"], "n_particles": bed["n_particles"],
                "secs": t_bed},
        "levels": levels,
        "wall_secs": time.time() - t0,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.tag + ".json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print("[%s] wrote %s (%.0fs total)" % (args.tag, path, payload["wall_secs"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())