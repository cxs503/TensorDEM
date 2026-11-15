#!/usr/bin/env python
"""m22106 probe B: is the isochoric cell's energy budget rate-limited?

Probe A (``scripts/m22106_mu_ladder_probe.py``) showed that the contact-layer
frictional dissipation power read from the friction meter is an order of
magnitude larger than the boundary work rate ``q * eps_dot_a * V`` of the same
window.  The meter's power identity itself is exact (``tests/
test_friction_meter_power_identity.py``: 27 500 W in, 27 500 W out for a
controlled sliding pair), so the mismatch is a property of the *cell*, not of
the accounting.

This probe tests the cheap explanation: the rotationless lattice is internally
faster than the boundary driving.  Same geometry (10 x 20, so the fast
counterpart already exists), same ``mu``, same target, only the axial strain
rate changes by a factor 20 (``0.05`` -> ``0.0025 1/s``).  If the residual is
rate driven it must shrink as the rate falls; if it stays put, the budget
defect is not kinetic and the critical-state protocol needs re-examination.

Usage (compute gate, one arm):

    PYTHONPATH=src python scripts/m22106_rate_probe.py --shard 1 --nshards 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.biaxial_shear_band import run_biaxial_shear_band  # noqa: E402

ARMS = [
    # tag, columns, rows, mu, strain_rate, strain_target (log)
    ("rate_slow_10x20_mu055_edot0025", 10, 20, 0.55, 0.0025, 0.60),
]

OUT_SUBDIR = os.path.join("artifacts", "CASE-DEM-BIAXIAL-FRICTION", "rate_probe")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TENSORDEM_THREADS", "4")))
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    todo = [c for i, c in enumerate(ARMS) if i % args.nshards == args.shard - 1]
    for tag, cols, rows, mu, rate, target in todo:
        started = time.time()
        res = run_biaxial_shear_band(
            confining_stress=100.0e3, columns=cols, rows=rows, radius=0.75,
            strain_rate=rate, strain_target=target, consolidation_steps=600,
            snapshot_strain=1.0e9, wall_friction=0.0,
            lateral_control="isochoric", mobilisation_every=500,
            material_overrides={"freeze_bond_fraction": 0.0,
                                "friction_coefficient": mu})
        elapsed = time.time() - started
        record = {
            "tag": tag, "arm": "rate_probe",
            "config": {"mu": mu, "columns": cols, "rows": rows,
                       "confining_stress": 100.0e3, "strain_rate": rate,
                       "strain_target": target, "consolidation_steps": 600,
                       "lateral_control": "isochoric",
                       "freeze_bond_fraction": 0.0, "wall_friction": 0.0},
            "specimen": res["specimen"], "loading": res["loading"],
            "peak": res["peak"], "stress_definitions": res["stress_definitions"],
            "mobilisation": res.get("mobilisation") or [],
            "elapsed_s": elapsed,
            "stress_path": res["stress_path"],
        }
        path = os.path.join(out_dir, f"iso_{tag}.json")
        with open(path, "w") as fh:
            json.dump(record, fh, default=lambda o: o.tolist()
                      if hasattr(o, "tolist") else str(o))
        sp = res["stress_path"]
        eps = np.asarray(sp["axial_strain_log"], dtype=float)
        print(f"== {tag} elapsed={elapsed:.0f}s steps={res['loading']['steps']} "
              f"final_log_strain={eps[-1]:.3f} I={res['loading']['inertial_number']:.2e} "
              f"peak_phi={res['peak']['friction_angle_deg']:.2f}")
        print(f"   wrote {path}")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())