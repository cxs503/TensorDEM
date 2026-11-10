"""Criterion (d) of P143W2C-N02: friction-regularization cells.

Two independent measurements, both written as JSON for
``scripts/p143w2c_n02_analysis.py``:

* ``--part cell``      -- slip cells: a bonded pad dragged on the bottom platen at
  a set of slip speeds, for each of the three Coulomb laws.  The measured
  tangential reaction over the *measured* platen normal reaction is the mobilized
  coefficient, compared with the analytic coefficient of the law.
* ``--part residual``  -- bonded specimens pre-compressed axially and then sheared
  past bond failure; the traction plateau is compared with the analytic
  ``mu_d N`` / ``mu_s N`` caps.

Usage::

    python scripts/p143w2c_n02_friction.py --part cell --out artifacts/p143w2c_n02
    python scripts/p143w2c_n02_friction.py --part residual --out artifacts/p143w2c_n02
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem import bpm_identifiability as bi  # noqa: E402

LAWS = ("none", "tanh", "dual")
MU_S, MU_D, V_C = 0.6, 0.4, 1.0e-3
SLIP_SPEEDS = (0.5e-3, 1.0e-3, 2.0e-3, 5.0e-3)


def cell(outdir: str) -> str:
    rows = {}
    for law in LAWS:
        rows[law] = {}
        for v in SLIP_SPEEDS:
            r = bi.shear_cell_residual(regularization=law, mu_s=MU_S, mu_d=MU_D,
                                       transition_speed=V_C, slip_speed=v,
                                       n_steps=700)
            rows[law][f"{v:g}"] = {k: val for k, val in r.items()}
            print(f"cell {law:5s} v={v:g} mu={r['mu_measured']:.4f} "
                  f"ref={r['mu_reference']:.4f} bias={r['bias_pct']:+.2f}%",
                  flush=True)
    path = os.path.join(outdir, "friction_cell.json")
    with open(path, "w") as fh:
        json.dump({"mu_s": MU_S, "mu_d": MU_D, "transition_speed": V_C,
                   "slip_speeds": list(SLIP_SPEEDS), "rows": rows}, fh)
    return path


def residual(outdir: str) -> str:
    rows = {}
    for law in LAWS:
        r = bi.residual_shear_specimen(
            regularization=law, mu_s=MU_S, mu_d=MU_D, transition_speed=V_C,
            geometry=bi.SpecimenGeometry(n_layers=4, cross_n=3),
            mass_scaling=1600.0, axial_preload_strain=0.01,
            shear_strain_rate=2.0e-2, target_shear_strain=0.06)
        rows[law] = {k: v for k, v in r.items()
                     if k not in ("shear_strain", "tau", "bond_damage")}
        rows[law]["shear_strain"] = [float(x) for x in r["shear_strain"]]
        rows[law]["tau"] = [float(x) for x in r["tau"]]
        rows[law]["bond_damage"] = [float(x) for x in r["bond_damage"]]
        print(f"residual {law:5s} tau_pk={r['tau_peak_Pa']:.4g} "
              f"tau_res={r['tau_residual_Pa']:.4g} "
              f"cap_dyn={r['coulomb_cap_dynamic_Pa']:.4g} "
              f"dmg={r['bond_damage_fraction_end']:.2f}", flush=True)
    path = os.path.join(outdir, "friction_residual.json")
    with open(path, "w") as fh:
        json.dump({"mu_s": MU_S, "mu_d": MU_D, "transition_speed": V_C,
                   "rows": rows}, fh)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("cell", "residual"), required=True)
    ap.add_argument("--out", default="artifacts/p143w2c_n02")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    print(cell(args.out) if args.part == "cell" else residual(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())