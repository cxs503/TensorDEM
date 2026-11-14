#!/usr/bin/env python
"""m20537 probe: constant-volume (isochoric) biaxial shear to critical state.

Round 2 of CASE-DEM-BIAXIAL-FRICTION failed for a *control* reason, not a
physics reason: the lateral servo had no admissible velocity limit -- large
limits made the walls slam in and out, small limits left the confining pressure
uncontrolled (measured sigma_h up to 30x the set point).  The minimal fix
recorded there was to remove the servo from the measurement loop altogether by
shearing at **constant volume** with kinematic volume-preserving walls, which
is the standard DEM critical-state test.

This probe runs that test.  ``lateral_control="isochoric"`` releases the
lateral servo after consolidation and then opens both lateral walls so that
``width * height`` stays at its consolidation value; the top platen keeps the
prescribed axial strain rate.  The lateral stress is then an *output*.

Usage (compute gate, one config per shard):
    PYTHONPATH=src python scripts/m20537_probe_isochoric.py --shard 1 --nshards 5

Each configuration writes ``runs/m20537/iso_<tag>.json`` with the full stress
path (including the logarithmic strain measure that the isochoric gate needs).
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

# tag, columns, rows, confining stress [Pa], mu, strain rate [1/s]
CONFIGS = [
    ("anchor_24x48_mu055_p100",      24, 48, 100.0e3, 0.55, 0.01),
    ("res_16x32_mu055_p100",         16, 32, 100.0e3, 0.55, 0.01),
    ("mu_24x48_mu030_p100",          24, 48, 100.0e3, 0.30, 0.01),
    ("pressure_24x48_mu055_p200",    24, 48, 200.0e3, 0.55, 0.01),
    ("rate_24x48_mu055_p100_slow",   24, 48, 100.0e3, 0.55, 0.005),
]

COMMON = dict(radius=0.75, strain_target=0.60, consolidation_steps=6000,
              snapshot_strain=1.0e9, wall_friction=0.0)


def summarise(tag: str, res: dict, elapsed: float) -> None:
    h = res["stress_path"]
    eps = np.asarray(h["axial_strain"])
    sv = np.asarray(h["sigma_v"])
    sh = np.asarray(h["sigma_h"])
    q = np.asarray(h["deviator"])
    ev = np.asarray(h["volumetric_strain_log"])
    phi = np.degrees(np.arcsin(np.clip((sv - sh) / np.maximum(sv + sh, 1e-30),
                                      -0.95, 0.95)))
    print(f"== {tag} elapsed={elapsed:.0f}s steps={res['loading']['steps']} "
          f"final_eps={eps[-1]:.3f} I={res['loading']['inertial_number']:.2e} "
          f"peak_phi={res['peak']['friction_angle_deg']:.2f}")
    for lo in np.arange(0.0, eps[-1] + 0.05, 0.05):
        m = (eps >= lo) & (eps < lo + 0.05)
        if m.sum() < 5:
            continue
        p = 0.5 * (sv[m] + sh[m])
        print(f"   {lo:.2f}-{lo + 0.05:.2f} n={m.sum():6d} "
              f"phi={phi[m].mean():6.2f}+-{phi[m].std():4.2f} "
              f"q={q[m].mean() / 1e3:7.1f} p={p.mean() / 1e3:7.1f} "
              f"sh={sh[m].mean() / 1e3:6.1f} q/p={(q[m] / np.maximum(2 * p, 1e-9)).mean():.3f} "
              f"ev_log={ev[m].mean():+.2e}")
    sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TENSORDEM_THREADS", "4")))
    out_dir = os.path.join(os.path.dirname(__file__), "..", "runs", "m20537")
    os.makedirs(out_dir, exist_ok=True)
    todo = [c for i, c in enumerate(CONFIGS) if i % args.nshards == args.shard - 1]
    for tag, cols, rows, pressure, mu, rate in todo:
        started = time.time()
        res = run_biaxial_shear_band(
            confining_stress=pressure, columns=cols, rows=rows,
            strain_rate=rate, lateral_control="isochoric",
            material_overrides={"freeze_bond_fraction": 0.0,
                                "friction_coefficient": mu},
            **COMMON)
        elapsed = time.time() - started
        summarise(tag, res, elapsed)
        record = {
            "tag": tag,
            "config": {"columns": cols, "rows": rows,
                       "confining_stress": pressure, "mu": mu,
                       "strain_rate": rate, "lateral_control": "isochoric",
                       "freeze_bond_fraction": 0.0, "wall_friction": 0.0,
                       "strain_target": COMMON["strain_target"]},
            "specimen": res["specimen"], "loading": res["loading"],
            "peak": res["peak"], "note": res["note"],
            "elapsed_s": elapsed,
            "stress_path": res["stress_path"],
        }
        path = os.path.join(out_dir, f"iso_{tag}.json")
        with open(path, "w") as fh:
            json.dump(record, fh, default=lambda o: o.tolist()
                      if hasattr(o, "tolist") else str(o))
        print(f"   wrote {path}")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())