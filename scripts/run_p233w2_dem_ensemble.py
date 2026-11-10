"""Run the dynamic-DEM ensemble for node P233W2-N04 and cache it as JSON.

Usage (from the repo root)::

    PYTHONPATH=src python scripts/run_p233w2_dem_ensemble.py --out runs/dem_ensemble.json

The ensemble is deliberately small enough to be reproducible on one node:
three parent sizes x three disorder seeds at the production time step, plus a
time-step-convergence subset at ``dt/2``.  Everything downstream (kernel,
population balance, report) reads this JSON, so the report numbers are always
programme flow, never hand-copied.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from tensordem.fsd_evolution.dynamic_breakage import DynamicImpactConfig, run_impact

#: Production time step (s).  Leapfrog stability limit is ~7e-6 s; the bond
#: failure strain is 2e-4, so a 4e-6 s step advances the platen by <5 % of a
#: failure strain -- the failure sequence is time-resolved, not impulsive.
DT = 4.0e-6
#: Convergence time step: half of the production step.
DT_FINE = 2.0e-6
STRAIN_RATE = 1.0
TARGET_STRAIN = 0.08
SIZES = (24, 32, 48)
SEEDS = (0, 1, 2)
CONVERGENCE_SIZES = (32,)
NY_RATIO = 0.5


def _row(result, *, tag: str) -> dict:
    return dict(
        tag=tag,
        nx=int(result.config.nx),
        ny=int(result.config.ny),
        seed=int(result.config.seed),
        dt_s=float(result.config.dt),
        strain_rate_1_s=float(result.config.strain_rate),
        target_strain=float(result.config.t_end * result.config.strain_rate),
        parent_diameter_m=float(result.parent_diameter),
        parent_mass_kg=float(result.parent_mass),
        parent_area_m2=float(result.parent_area),
        n_fragments=int(result.n_fragments),
        largest_fraction=float(result.largest_fraction),
        n_broken_bonds=int(result.n_broken_bonds),
        t_first_break_s=result.t_first_break,
        t_split_s=result.t_split,
        mass_residual=float(result.mass_residual),
        ledger_residual=float(result.ledger_residual),
        floor_mass_fraction=float(result.floor_mass_fraction()),
        energies={k: float(v) for k, v in result.energies.items()},
        daughters=[float(v) for v in result.diameters],
        events=[
            dict(t=float(e.t), parent=float(e.parent_diameter),
                 daughters=[float(x) for x in e.daughter_diameters])
            for e in result.events
        ],
        exposure=result.exposure,
        history={k: [float(v) for v in vals] for k, vals in result.history.items()},
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/p233w2_dem_ensemble.json")
    parser.add_argument("--cache", default=None, help="reuse an existing JSON")
    args = parser.parse_args(argv)
    t0 = time.time()
    rows = []
    for nx in SIZES:
        ny = max(4, int(round(NY_RATIO * nx)))
        t_end = TARGET_STRAIN / STRAIN_RATE
        for seed in SEEDS:
            cfg = DynamicImpactConfig(
                nx=nx, ny=ny, loading="platen", strain_rate=STRAIN_RATE,
                dt=DT, t_end=t_end, seed=seed, damping_ratio=0.0,
            )
            result = run_impact(cfg)
            row = _row(result, tag="production")
            rows.append(row)
            print(
                f"production nx={nx} seed={seed}: d={row['parent_diameter_m']:.3f} m "
                f"frags={row['n_fragments']} events={len(row['events'])} "
                f"ledger={row['ledger_residual']:+.4f} mass={row['mass_residual']:.1e} "
                f"t={time.time()-t0:.0f}s",
                flush=True,
            )
    for nx in CONVERGENCE_SIZES:
        ny = max(4, int(round(NY_RATIO * nx)))
        t_end = TARGET_STRAIN / STRAIN_RATE
        for seed in SEEDS:
            cfg = DynamicImpactConfig(
                nx=nx, ny=ny, loading="platen", strain_rate=STRAIN_RATE,
                dt=DT_FINE, t_end=t_end, seed=seed, damping_ratio=0.0,
            )
            result = run_impact(cfg)
            row = _row(result, tag="convergence")
            rows.append(row)
            print(
                f"convergence nx={nx} seed={seed}: frags={row['n_fragments']} "
                f"events={len(row['events'])} ledger={row['ledger_residual']:+.4f} "
                f"t={time.time()-t0:.0f}s",
                flush=True,
            )
    payload = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        dt_s=DT,
        dt_fine_s=DT_FINE,
        strain_rate_1_s=STRAIN_RATE,
        target_strain=TARGET_STRAIN,
        sizes=list(SIZES),
        seeds=list(SEEDS),
        rows=rows,
        wall_time_s=float(time.time() - t0),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out} ({len(rows)} runs, {time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())