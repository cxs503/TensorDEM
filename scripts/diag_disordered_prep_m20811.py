"""Diagnostic of the settle-then-release preparation (m20811).

Cheap, small container (``column_radius = 0.24`` m, 115 grains at ``a = 0.5``):
for a sweep of jitter amplitudes, print the equilibrated packing fraction, the
overlap gate, the kinetic energy at rest and the disorder metric, and run one
collapse so the whole chain is exercised.  The production wave uses the same
code at the frozen geometry (``column_radius = 0.48`` m).

Usage: ``PYTHONPATH=src python3 scripts/diag_disordered_prep_m20811.py``
"""
from __future__ import annotations

import json
import sys

import numpy as np

from tensordem.granular_column.axisym import (AxisymCase, lube_2004_runout_ratio,
                                             simulate_axisymmetric_column)
from tensordem.granular_column.pluviation import (RelaxCase, disorder_metrics,
                                                  prepare_disordered_column)


def main() -> int:
    rows = []
    for jit in (0.0, 0.1, 0.2, 0.3):
        case = RelaxCase(column_radius=0.24, radius=0.03, aspect_ratio=0.5,
                         jitter_diameters=jit, seed=505, dt=4.8e-4,
                         label=f"diag_j{jit:g}")
        rep = prepare_disordered_column(case)
        row = {
            "jitter_diameters": jit,
            "n": rep["n_particles"],
            "phi": rep["packing_fraction"],
            "h": rep["column_height_measured"],
            "aspect_measured": rep["aspect_measured"],
            "max_overlap_frac": rep["max_overlap_frac_of_radius"],
            "hex_order": rep["hex_order"],
            "coordination": rep["coordination"],
            "ke": rep["kinetic_energy_at_rest"],
            "at_rest": rep["at_rest"],
            "admissible": rep["admissible"],
            "relax_steps": rep["relax_steps"],
        }
        if jit in (0.2, 0.3):
            pos = np.asarray(rep["positions"], dtype=float)
            ac = AxisymCase(aspect_ratio=0.5, grains_per_radius=8.0, radius=0.03,
                            density=2500.0, mu=0.5, mu_rolling=0.5, mu_wall=0.8,
                            restitution=0.5, kn=6.0e4, kt_ratio=0.5, gravity=9.81,
                            duration_factor=8.0, min_time=0.5, rest_ke_tol=1.0e-4,
                            rest_hold=200, t_reach_fraction=0.99)
            res = simulate_axisymmetric_column(ac, return_positions=False,
                                               initial_positions=pos)
            a = res["aspect_measured"]
            ref = lube_2004_runout_ratio(a)
            row.update({"runout_ratio": res["runout_ratio_p995"],
                        "reference": ref,
                        "err": res["runout_ratio_p995"] / ref - 1.0,
                        "stop_reason": res["stop_reason"]})
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({"rows": rows}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
