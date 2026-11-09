#!/usr/bin/env python
"""Time-step convergence of the crushing-DEM energy ledger (base case).

The ledger residual of an explicit scheme must shrink with ``dt``; this script
runs the campaign base case at the recommended time step and at half of it and
writes ``artifacts/myi_ridge/ledger_convergence.json``.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.myi_ridge import crushing  # noqa: E402

HEIGHT = 20.0
SPECIMEN = crushing.RidgeSpecimen(height=HEIGHT, length=1.2 * HEIGHT, porosity=0.30,
                                 particle_radius=0.5, seed=20261108)
BASE = crushing.CrushParameters(ice_strength=7.5e6, bond_strength_fraction=0.10,
                                hull_velocity=2.0, advance=0.8, sail_height=0.22 * HEIGHT)


def main() -> None:
    out = {}
    for factor in (1.0, 0.5):
        dem = crushing.RidgeCrushDem(SPECIMEN, BASE)
        dt = dem.dt * factor
        dem.params = replace(BASE, dt=dt)
        dem.dt = dt
        run = dem.run()
        out[f"factor_{factor:.2f}"] = {
            "dt": dt,
            "steps": run["steps"],
            "pressure_steady": run["pressure_steady"],
            "pressure_peak": run["pressure_peak"],
            "closure_of_exchanged": run["ledger"]["closure_of_exchanged"],
            "residual": run["ledger"]["residual"],
            "dissipated": run["ledger"]["dissipated"],
            "external_work": run["ledger"]["external_work"],
        }
        print(f"factor={factor} dt={dt:.3e} steps={run['steps']} "
              f"closure={run['ledger']['closure_of_exchanged']:.4f} "
              f"p_steady={run['pressure_steady']/1e6:.3f} MPa", flush=True)
    target = ROOT / "artifacts" / "myi_ridge" / "ledger_convergence.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()