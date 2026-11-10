"""Run the *impact* ensemble (tensile spall) for node P233W2-N04.

The rate-controlled platen instrument (``run_p233w2_dem_ensemble.py``) measures
the inertial energy budget and the strength of a floe, but its breakage is
localised at the grips: the parent survives and chips spall off (measured
largest-daughter ratio 0.99).  A *floe that breaks* splits across its body, so
the daughter catalogue ``beta(d|d')`` and the breakage-energy threshold law are
measured on a tensile-spall impact instead: the two halves of the plate separate
at ``2 v0`` and the lattice rings down.

Outputs (cached JSON):
* one row per (size, impact velocity, seed): fragments, spall events, energies;
* the fragmentation probability ``P(break | v0, d)`` grid, from which the
  threshold velocity ``v_c(d)`` and the breakage energy ``E_b = 1/2 M v_c**2``
  follow.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from tensordem.fsd_evolution.dynamic_breakage import DynamicImpactConfig, run_impact

DT = 2.0e-6
DT_FINE = 1.0e-6
T_END = 1.5e-3
SIZES = (12, 16, 24, 32, 48)
VELOCITIES = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
SEEDS = (0, 1, 2)
NY_RATIO = 0.5


def _row(result, v0: float) -> dict:
    return dict(
        loading="stretch",
        nx=int(result.config.nx),
        ny=int(result.config.ny),
        seed=int(result.config.seed),
        dt_s=float(result.config.dt),
        impact_velocity_m_s=float(v0),
        parent_diameter_m=float(result.parent_diameter),
        parent_mass_kg=float(result.parent_mass),
        parent_area_m2=float(result.parent_area),
        broke=bool(result.broke),
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
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/p233w2_impact_ensemble.json")
    args = parser.parse_args(argv)
    t0 = time.time()
    rows = []
    for nx in SIZES:
        ny = max(4, int(round(NY_RATIO * nx)))
        for v0 in VELOCITIES:
            for seed in SEEDS:
                cfg = DynamicImpactConfig(
                    nx=nx, ny=ny, loading="stretch", impact_velocity=v0,
                    dt=DT, t_end=T_END, seed=seed, damping_ratio=0.0,
                )
                result = run_impact(cfg)
                rows.append(dict(_row(result, v0), tag="production"))
        print(f"size nx={nx} done at t={time.time()-t0:.0f}s", flush=True)
    # time-step convergence subset: the largest two sizes at dt/2
    for nx in (32, 48):
        ny = max(4, int(round(NY_RATIO * nx)))
        for v0 in VELOCITIES:
            for seed in SEEDS:
                cfg = DynamicImpactConfig(
                    nx=nx, ny=ny, loading="stretch", impact_velocity=v0,
                    dt=DT_FINE, t_end=T_END, seed=seed, damping_ratio=0.0,
                )
                result = run_impact(cfg)
                rows.append(dict(_row(result, v0), tag="convergence"))
        print(f"convergence nx={nx} done at t={time.time()-t0:.0f}s", flush=True)
    payload = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        dt_s=DT, dt_fine_s=DT_FINE, t_end_s=T_END,
        sizes=list(SIZES), velocities=list(VELOCITIES), seeds=list(SEEDS),
        rows=rows, wall_time_s=float(time.time() - t0),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out} ({len(rows)} runs, {time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())