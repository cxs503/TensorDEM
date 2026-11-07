"""m6296 Beverloo silo-discharge verification sweep.

Runs the ``tensordem.silo_discharge`` module (real 3-D DEM, ``granular3d_core``
kernel) over a matrix designed to test the *literature* claims the module
makes in its own docstring:

* orifice-law exponent 2.5 in ``(D - k d)``      -> D/d matrix
* Beverloo coefficients ``C ~ 0.58``, ``k ~ 1.4`` -> fitted from the matrix
* ``W ~ sqrt(g)``                                  -> gravity case
* ``W`` independent of the fill height             -> head case

Every case is one JSON file under ``runs/m6296_silo/`` so the sweep can be
sharded (``--index i --n-total n``) across cluster nodes.

Usage
-----
    PYTHONPATH=src python scripts/m6296_silo_beverloo_sweep.py            # all
    PYTHONPATH=src python scripts/m6296_silo_beverloo_sweep.py --case D3  # one
    PYTHONPATH=src python scripts/m6296_silo_beverloo_sweep.py --shard 0 --n-total 2
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.silo_discharge import SiloCase, _initial_packing, simulate_silo  # noqa: E402

# --------------------------------------------------------------------------
# geometry (SI)
# --------------------------------------------------------------------------
D_PARTICLE = 0.015          # particle diameter [m]
R_PARTICLE = 0.5 * D_PARTICLE
SILO_RADIUS = 0.11          # 0.22 m silo diameter -> D_silo/D >= 2.9 at D/d = 5
FILL_HEIGHT = 0.12          # 0.12 m -> H/D >= 1.6 at D/d = 5, >= 2.7 at D/d = 3
SETTLE = 0.25           # let the start-up transient (~0.15 s) leave the windows

# Per-case measurement window.  ``discharge_time`` is chosen from the
# *literature* Beverloo estimate (C = 0.58, k = 1.4) so that 5 windows carry
# >= ~50 discharged particles; it only sets how long to run, it never enters
# the measured flow rate.
CASES = {
    "D3": dict(orifice_diameter=3.0 * D_PARTICLE, discharge_time=1.50, window=0.30),
    "D4": dict(orifice_diameter=4.0 * D_PARTICLE, discharge_time=0.60, window=0.12),
    "D5": dict(orifice_diameter=5.0 * D_PARTICLE, discharge_time=0.45, window=0.09),
    "grav": dict(orifice_diameter=4.0 * D_PARTICLE, discharge_time=0.60,
                 window=0.12, gravity=9.81 / 4.0),
    "head": dict(orifice_diameter=4.0 * D_PARTICLE, discharge_time=0.60,
                 window=0.12, fill_height=1.75 * FILL_HEIGHT),
}

BASE = dict(radius=R_PARTICLE, silo_radius=SILO_RADIUS, fill_height=FILL_HEIGHT,
            settle_time=SETTLE, n_windows=5)


def build_case(name: str) -> SiloCase:
    kw = dict(BASE)
    kw.update(CASES[name])
    kw["n_windows"] = 5
    return SiloCase(**kw)


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def run_one(name: str, outdir: str) -> dict:
    case = build_case(name)
    print(f"[{name}] start: n0={len(_initial_packing(case))} dt={case.timestep:.3e} "
          f"steps={int(round((case.settle_time + case.discharge_time) / case.timestep))}",
          flush=True)
    t0 = time.time()
    res = simulate_silo(case, return_history=False)
    res["wall_seconds"] = time.time() - t0
    res["case"] = name
    res["config"] = {k: getattr(case, k) for k in
                     ("radius", "silo_radius", "fill_height", "orifice_diameter",
                      "plate_thickness", "wall_mu", "mu", "mu_rolling",
                      "restitution", "kn", "gravity", "settle_time",
                      "discharge_time", "window", "n_windows")}
    res["n_particles"] = res["n_initial"]
    res["env"] = {"machine": platform.machine(), "host": platform.node(),
                  "python": platform.python_version(), "git_sha": git_sha()}
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.json")
    with open(path, "w", encoding="utf8") as fh:
        json.dump(res, fh, indent=2)
    print("RESULT_JSON " + json.dumps(res), flush=True)
    print(f"[{name}] W = {res['flow_rate_kg_s']:.5f} +- "
          f"{res['flow_rate_std_kg_s']:.5f} kg/s  "
          f"n_disch={res['n_discharged']}/{res['n_initial']}  "
          f"D/d={res['D_over_d']:.2f}  {res['wall_seconds']:.1f}s -> {path}",
          flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CASES), default=None)
    ap.add_argument("--index", type=int, default=None, help="single index 0..n-1")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--n-total", type=int, default=1)
    ap.add_argument("--outdir", default="runs/m6296_silo")
    args = ap.parse_args(argv)

    names = sorted(CASES) if args.case is None else [args.case]
    if args.index is not None:
        names = names[args.index:args.index + 1]
    elif args.shard is not None and args.n_total > 1:
        names = [n for i, n in enumerate(names) if i % args.n_total == args.shard]
    for n in names:
        run_one(n, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())