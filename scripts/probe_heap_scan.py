"""Physics probe: repose-angle response vs the calibrated contact parameters.

Development-time diagnostic for the calibration forward model.  Prints one JSON
line per case so that the output survives log truncation.

    PYTHONPATH=src python3 scripts/probe_heap_scan.py [--fast]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from tensordem.dem_calibration.heap import HeapCase, run_heap


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=0.006)
    ap.add_argument("--kn", type=float, default=800.0)
    ap.add_argument("--mu-rolling", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--fast", action="store_true", help="shorten the protocol")
    args = ap.parse_args(argv)

    base = dict(radius=args.radius, kn=args.kn, mu_rolling=args.mu_rolling,
                seed=args.seed)
    if args.fast:
        base.update(hold_time=0.02, lift_time=0.20, settle_time=0.15)

    cases = []
    for mu in (0.10, 0.20, 0.35, 0.50, 0.65, 0.80):
        cases.append(dict(mu=mu, restitution=0.30, stiffness_ratio=2.0))
    for e in (0.10, 0.50, 0.90):
        cases.append(dict(mu=0.40, restitution=e, stiffness_ratio=2.0))
    for sr in (1.0, 3.0):
        cases.append(dict(mu=0.40, restitution=0.30, stiffness_ratio=sr))
    # reproducibility: exact repeat of the first case
    cases.append(dict(cases[0]))

    for params in cases:
        case = HeapCase(**params, **base)
        t0 = time.time()
        try:
            out = run_heap(case)
            record = {
                "params": {k: params[k] for k in params},
                "theta_r_deg": round(out["theta_r_deg"], 3),
                "phi_solid": round(out["phi_solid"], 4),
                "r2": round(out["r2"], 3),
                "ke_ratio": float(f"{out['ke_ratio']:.3e}"),
                "n_particles": out["n_particles"],
                "n_steps": out["n_steps"],
                "max_speed": round(out["max_speed_m_s"], 3),
                "digest": out["digest"][:16],
                "wall_s": round(time.time() - t0, 2),
            }
        except Exception as exc:  # noqa: BLE001 - probe reports failures
            record = {"params": params, "error": f"{type(exc).__name__}: {exc}"}
        print("PROBE " + json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())