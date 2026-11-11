"""Reduced-budget driver for the repose anchor A/B closure (protocol v2).

The full sweep (``python -m tensordem.friction_surface.study --stage repose``,
5 bisection steps x 3 seeds) is the reference run; this driver is the
compute-budgeted variant used when the shared machine is busy: 3 bisection steps,
1 seed per material, 2 extra points of the theta(mu) surface.  It writes the SAME
schema as ``study.repose_anchor`` so ``--stage report`` adjudicates it unchanged,
and it labels itself ``budget: "reduced"`` so the numbers are never mistaken for
the full sweep.

Usage
-----
    PYTHONPATH=src python scripts/p152w1_repose_v2_closure.py --particles 200 --jobs 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
from multiprocessing import Pool

import numpy as np

from tensordem.friction_surface.repose import ReposeCase, simulate_repose
from tensordem.friction_surface.study import MATERIALS, _repose_job, _theta, repose_sweep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--bisection-steps", type=int, default=3)
    ap.add_argument("--out", default="artifacts/P152W1-N04/repose.json")
    args = ap.parse_args(argv)

    n = args.particles
    seeds = (15204,)
    with Pool(args.jobs) as pool:
        control = pool.map(_repose_job, [(0.0, 0.0, 15204, n, "fixed"),
                                         (0.0, 0.0, 15205, n, "fixed")])
        mu_g = MATERIALS["glass_beads"]["mu"]
        target = MATERIALS["glass_beads"]["theta_exp_deg"]
        lo, hi = 0.02, 0.40
        history = []
        for _ in range(args.bisection_steps):
            mid = 0.5 * (lo + hi)
            th = _theta(repose_sweep(mid, [mu_g], pool=pool, n_particles=n), mu_g)
            history.append({"mu_rolling": mid, "theta_r_deg": th})
            if th == th:                       # nan never moves the bracket
                if th < target:
                    lo = mid
                else:
                    hi = mid
        mu_r_star = 0.5 * (lo + hi)
        final_jobs = [(m["mu"], mu_r_star, s, n, "fixed")
                      for m in MATERIALS.values() for s in seeds]
        finals = pool.map(_repose_job, final_jobs)
        extra = pool.map(_repose_job, [(mu, mu_r_star, 15204, n, "fixed")
                                       for mu in (0.1, 0.6)])
        keep = pool.map(_repose_job, [(m["mu"], mu_r_star, seeds[0], n, "fixed", True)
                                      for m in MATERIALS.values()])

    materials = {}
    for name, mat in MATERIALS.items():
        vals = [r["theta_r_deg"] for r in finals
                if abs(r["mu"] - mat["mu"]) < 1e-9 and r["theta_r_deg"] == r["theta_r_deg"]]
        mean = float(np.mean(vals)) if vals else float("nan")
        materials[name] = {
            "mu": mat["mu"], "mu_rolling": mu_r_star,
            "theta_exp_deg": mat["theta_exp_deg"],
            "theta_exp_band_deg": mat["theta_band_deg"],
            "theta_sim_deg": mean,
            "theta_sim_std_deg": float(np.std(vals)) if vals else float("nan"),
            "theta_sim_runs": vals,
            "relative_error": (abs(mean - mat["theta_exp_deg"]) / mat["theta_exp_deg"]
                               if mean == mean else float("nan")),
            "mu_eff_measured": (math.tan(math.radians(mean)) if mean == mean
                                else float("nan")),
            "mu_eff_experiment": math.tan(math.radians(mat["theta_exp_deg"])),
            "note": mat["note"],
        }
    payload = {
        "negative_control": {
            "mu": 0.0, "mu_rolling": 0.0,
            "theta_r_deg": float(np.nanmean([r["theta_r_deg"] for r in control])),
            "runs": [r["theta_r_deg"] for r in control],
            "criterion": "theta_r < 5 deg",
        },
        "calibration": {
            "parameter": "mu_rolling (single shape surrogate)",
            "fitted_on": "glass_beads",
            "mu_rolling_star": mu_r_star,
            "bisection_history": history,
            "bisection_steps": args.bisection_steps,
        },
        "materials": materials,
        "theta_mu_surface": [{"mu": r["mu"], "theta_r_deg": r["theta_r_deg"]}
                             for r in list(extra)],
        "theta_mu_surface_all": [{"mu": r["mu"], "theta_r_deg": r["theta_r_deg"]}
                                 for r in list(finals) + list(extra)],
        "protocol": {
            "source_mode": "fixed",
            "n_particles": n,
            "seeds": list(seeds),
            "budget": "reduced (3 bisection steps, 1 seed, 2 surface points)",
            "pour": ("fixed-height nozzle; bursts self-paced by the charge fall time; "
                     "the pour closes when the deposit fills the nozzle"),
            "reached_rest": [r["reached_rest"] for r in finals],
            "pour_closed": [r["pour_closed"] for r in finals],
            "bursts_released": [r["bursts_released"] for r in finals],
            "bursts_skipped": [r["bursts_skipped"] for r in finals],
            "nozzle_z_max": [r["nozzle_z_max"] for r in finals],
            "clear_steps": [r["clear_steps"] for r in finals],
            "n_particles_realised": [r["n_particles"] for r in finals],
            "apex_height_m": [r["apex_height_m"] for r in finals],
        },
        "figure_positions": {
            str(m["mu"]): (None if r["positions"] is None else r["positions"].tolist())
            for m, r in zip(MATERIALS.values(), keep)
        },
        "reference_micro_parameters": "literature interparticle friction, not fitted",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    print(json.dumps({k: v["relative_error"] for k, v in materials.items()}, indent=1))
    print("negative control:", payload["negative_control"]["theta_r_deg"])
    print("mu_rolling*:", mu_r_star)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
