"""Multi-seed estimate of the energy-dissipation scaling exponent (C3).

The single-realisation estimate of :mod:`tensordem.ice_ice_collision` at
C = 0.35 has a statistical uncertainty of a few per cent (the shear stress of a
124-floe cell fluctuates), so the scaling exponent is re-measured with three
independent initial packings and a longer sampling window.  Everything is
written to ``artifacts/ice_ice_collision/scaling_multiseed.json`` -- both the
per-seed values and the seed-averaged fit, so the estimate can be audited.

Usage::

    python scripts/ice_ice_collision_scaling_multiseed.py [--seeds 2 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.ice_ice_collision import FloeFieldConfig, simulate_floe_field  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts", "ice_ice_collision")
RATES = [0.5, 1.0, 1.5, 2.0, 3.0]
CONCENTRATION = 0.35


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--sample", type=int, default=3500)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "scaling_multiseed.json")
    result = {"concentration": CONCENTRATION, "rates": RATES, "rows": []}
    if os.path.exists(path):
        with open(path) as fh:
            result = json.load(fh)
    for seed in args.seeds:
        for rate in RATES:
            cfg = FloeFieldConfig(concentration=CONCENTRATION, shear_rate=rate,
                                  shear_steps=args.steps, sample_steps=args.sample,
                                  settle_steps=1500, seed=seed)
            t0 = time.time()
            out = simulate_floe_field(cfg)
            row = {"seed": seed, "shear_rate": rate,
                   "v": rate * cfg.mean_diameter,
                   "epsilon": out["diss_rate_per_mass_window"],
                   "tau_total": out["tau_total"],
                   "coordination": out["coordination"],
                   "elapsed": time.time() - t0}
            result["rows"] = [r for r in result["rows"]
                              if not (r["seed"] == seed and r["shear_rate"] == rate)]
            result["rows"].append(row)
            with open(path, "w") as fh:
                json.dump(result, fh, indent=2)
            print(f"[{time.strftime('%H:%M:%S')}] seed={seed} rate={rate} "
                  f"eps={row['epsilon']:.5g} tau={row['tau_total']:.4g} "
                  f"({row['elapsed']:.0f}s)", flush=True)

    rows = result["rows"]
    seeds = sorted({r["seed"] for r in rows})
    rates = sorted({r["shear_rate"] for r in rows})
    mean_eps, mean_tau, spread = [], [], []
    for rate in rates:
        eps = [r["epsilon"] for r in rows if r["shear_rate"] == rate]
        tau = [r["tau_total"] for r in rows if r["shear_rate"] == rate]
        mean_eps.append(float(np.mean(eps)))
        mean_tau.append(float(np.mean(tau)))
        spread.append(float(np.std(eps) / np.mean(eps)) if len(eps) > 1 else 0.0)
    v = np.array([rate * 2.0 * 0.6 for rate in rates])
    exponent = float(np.polyfit(np.log(v), np.log(mean_eps), 1)[0])
    exponent_tau = float(np.polyfit(np.log(v), np.log(mean_tau), 1)[0]) + 1.0
    result["summary"] = {
        "seeds": seeds,
        "rates": rates,
        "mean_epsilon": mean_eps,
        "relative_spread": spread,
        "exponent_multiseed": exponent,
        "exponent_from_stress": exponent_tau,
        "n_runs": len(rows),
    }
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())