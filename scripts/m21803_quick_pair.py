#!/usr/bin/env python
"""m21803 quick additivity pair -- reduced specimen, single thread, local box.

The cluster submission (``scripts/m21803_rolling_surrogate.py``, 16x32, 5 shards)
sits in the queue; this driver produces the *same* comparison on a reduced
12x24 specimen so the additivity question can be answered without waiting for
scheduling.  It is deliberately one thread and three configurations, so it stays
inside the local-compute policy (<8 cores, <10 min per configuration).

Configurations (all isochoric, sigma3 = 100 kPa, mu_r only via the surrogate):

    q_mu055_mur00   mu=0.55  mu_r=0.00   baseline
    q_mu055_mur20   mu=0.55  mu_r=0.20   surrogate split, mu_eff = 0.75
    q_mu075_mur00   mu=0.75  mu_r=0.00   pure-friction control, mu_eff = 0.75

Usage:
    PYTHONPATH=src python scripts/m21803_quick_pair.py [--only TAG ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from m21803_rolling_surrogate import summarise  # noqa: E402
from tensordem.biaxial_shear_band import run_biaxial_shear_band  # noqa: E402

OUT_DIR = "runs/m21803_quick"

CONFIGS = [
    ("q_mu055_mur00", 0.55, 0.00),
    ("q_mu055_mur20", 0.55, 0.20),
    ("q_mu075_mur00", 0.75, 0.00),
]

COMMON = dict(
    columns=12, rows=24,
    radius=0.75,
    confining_stress=100.0e3,
    strain_rate=0.02,
    strain_target=0.25,
    consolidation_steps=2000,
    snapshot_strain=1.0e9,
    wall_friction=0.0,
    lateral_control="isochoric",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    for tag, mu, mu_r in CONFIGS:
        if args.only and tag not in args.only:
            continue
        t0 = time.time()
        result = run_biaxial_shear_band(
            material_overrides={"friction_coefficient": mu,
                                "rolling_friction_coefficient": mu_r,
                                "freeze_bond_fraction": 0.0},
            record_snapshots=False, **COMMON)
        elapsed = time.time() - t0
        summary = summarise(tag, result, elapsed)
        record = {
            "tag": tag,
            "config": {**COMMON, "mu": mu, "mu_r": mu_r, "mu_eff_additive": mu + mu_r},
            "specimen": result["specimen"],
            "loading": result["loading"],
            "peak": result["peak"],
            "stress_path": result["stress_path"],
            **summary,
            "elapsed_s": round(elapsed, 1),
        }
        path = os.path.join(OUT_DIR, f"rolling_surrogate_{tag}.json")
        with open(path, "w") as fh:
            json.dump(record, fh,
                      default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print(f"[quick] wrote {path} ({elapsed:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())