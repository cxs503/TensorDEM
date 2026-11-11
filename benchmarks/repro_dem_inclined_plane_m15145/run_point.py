#!/usr/bin/env python
"""One chute-flow point (Silbert et al. 2001 reproduction).

Usage::

    PYTHONPATH=src python benchmarks/repro_dem_inclined_plane_m15145/run_point.py \
        --theta 22 --height 20 --t-end 60 --out runs/theta22_H20.json

Every number written is dimensionless in the paper's units
(``d = m = g = 1``, time in ``sqrt(d/g)``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensordem.inclined_plane import (  # noqa: E402
    InclinedPlaneCase,
    density_velocity_profile,
    simulate_chute,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, required=True)
    ap.add_argument("--height", type=float, required=True)
    ap.add_argument("--lx", type=float, default=16.0)
    ap.add_argument("--lz", type=float, default=8.0)
    ap.add_argument("--t-end", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--kn", type=float, default=1.0e4)
    ap.add_argument("--mu", type=float, default=0.50)
    ap.add_argument("--restitution", type=float, default=0.88)
    ap.add_argument("--steady-frac", type=float, default=0.4)
    ap.add_argument("--sample-every", type=int, default=40)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    case = InclinedPlaneCase(theta_deg=args.theta, height=args.height,
                             lx=args.lx, lz=args.lz, dt=args.dt, kn=args.kn,
                             mu=args.mu, restitution=args.restitution)
    t0 = time.time()
    res = simulate_chute(case, t_end=args.t_end, steady_frac=args.steady_frac,
                         sample_every=args.sample_every, verbose=True)
    res["bulk"] = density_velocity_profile(res)
    res["provenance"] = {
        "script": os.path.abspath(__file__),
        "argv": sys.argv,
        "python": sys.version.split()[0],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    m = res["metrics"]
    print(json.dumps({
        "theta": args.theta, "H": args.height,
        "flowing": m["flowing"], "ke": round(m["ke_steady"], 5),
        "vx_mean": round(m["vx_mean_steady"], 3),
        "vx_max": round(m["vx_max_steady"], 3),
        "p_bagnold": res["fits"].get("p"),
        "phi_bulk": round(res["bulk"].get("phi_bulk", float("nan")), 4),
        "wall_s": round(time.time() - t0, 1),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())