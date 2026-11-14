#!/usr/bin/env python
"""m20304 probe #2: controlled-pressure biaxial shear to critical state.

After the servo defect fix (``servo_width`` now tracks the current specimen
height) the *velocity limit* is the remaining control knob: with the old limit
(2 m/s) the lateral walls slam in/out at maximum speed and the realised
confining pressure is wherever the dynamics land (130 kPa for a 100 kPa
target).  This probe uses a small limit (0.02 m/s) with a slow strain rate
(0.005 1/s), so the servo needs ~0.01 m/s to follow the specimen's dilation and
never saturates.

Usage (compute gate, one config per shard):
    PYTHONPATH=src python scripts/m20304_probe_cs2.py --shard 1 --nshards 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.biaxial_shear_band import run_biaxial_shear_band  # noqa: E402

CONFIGS = [
    # tag, columns, rows, confining stress, mu
    ("anchor_24x48_100kPa_mu055", 24, 48, 100.0e3, 0.55),
    ("res_16x32_100kPa_mu055", 16, 32, 100.0e3, 0.55),
    ("pres_16x32_200kPa_mu055", 16, 32, 200.0e3, 0.55),
    ("mu_16x32_100kPa_mu030", 16, 32, 100.0e3, 0.30),
]

COMMON = dict(radius=0.75, strain_rate=0.005, strain_target=0.50,
              consolidation_steps=6000, snapshot_strain=1.0e9,
              wall_friction=0.0, servo_gain=3.0e-4,
              servo_velocity_limit=0.02)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    todo = [c for i, c in enumerate(CONFIGS) if i % args.nshards == args.shard - 1]
    out_dir = os.path.join(os.path.dirname(__file__), "..", "runs", "m20304", "probe2")
    os.makedirs(out_dir, exist_ok=True)
    for tag, cols, rows, p, mu in todo:
        started = time.time()
        res = run_biaxial_shear_band(
            confining_stress=p, columns=cols, rows=rows,
            material_overrides={"freeze_bond_fraction": 0.0,
                                "friction_coefficient": mu},
            **COMMON)
        h = res["stress_path"]
        eps = np.asarray(h["axial_strain"])
        sv = np.asarray(h["sigma_v"])
        sh = np.asarray(h["sigma_h"])
        q = np.asarray(h["deviator"])
        ev = np.asarray(h["volumetric_strain"])
        phi = np.degrees(np.arcsin(np.clip((sv - sh) / np.maximum(sv + sh, 1e-30),
                                          -0.95, 0.95)))
        print(f"== {tag} elapsed={time.time() - started:.0f}s "
              f"steps={res['loading']['steps']} final_eps={eps[-1]:.3f} "
              f"I={res['loading']['inertial_number']:.2e} "
              f"peak_phi={res['peak']['friction_angle_deg']:.2f}")
        step = 0.05
        edges = np.arange(0.0, eps[-1] + step, step)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (eps >= lo) & (eps < hi)
            if m.sum() < 5:
                continue
            print(f"   {lo:.2f}-{hi:.2f} n={m.sum():6d} "
                  f"phi={phi[m].mean():6.2f}±{phi[m].std():4.2f} "
                  f"q={q[m].mean() / 1e3:7.1f} sv={sv[m].mean() / 1e3:7.1f} "
                  f"sh={sh[m].mean() / 1e3:6.1f} ev={ev[m].mean():+.4f} "
                  f"dEv={np.polyfit(eps[m], ev[m], 1)[0]:+.3f}")
        payload = {
            "tag": tag, "config": {**COMMON, "columns": cols, "rows": rows,
                                   "confining_stress": p, "mu": mu,
                                   "freeze_bond_fraction": 0.0},
            "elapsed_s": round(time.time() - started, 1),
            "specimen": res["specimen"], "loading": res["loading"],
            "peak": res["peak"], "stress_path": h,
        }
        with open(os.path.join(out_dir, f"{tag}.json"), "w") as fh:
            json.dump(payload, fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())