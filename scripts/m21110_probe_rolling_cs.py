#!/usr/bin/env python
"""m21110 probe: rolling-resistance critical-state calibration.

Round 3 (m20537) established that 2D discs without rolling resistance give
phi_cv ≈ 14.3° (mu=0.55), far below the Rowe (1962) 3D limit of 28.81°.
Round 4 (m21034) integrated rolling resistance into RidgeAssembly but did not
complete the critical-state calibration.

This probe runs constant-volume (isochoric) biaxial shear with varying
rolling_friction_coefficient to establish the mu_r -- phi_cv calibration curve.

Usage (compute gate, one config per shard):
    PYTHONPATH=src python scripts/m21110_probe_rolling_cs.py --shard 1 --nshards 3

Each configuration writes ``runs/m21110/rolling_cs_<tag>.json`` with the full
stress path (including logarithmic strain measure).
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

# tag, columns, rows, confining stress [Pa], mu, mu_r, strain rate [1/s]
CONFIGS = [
    # Baseline: no rolling resistance (replicates R3 anchor)
    ("mu_r00_16x32_mu055_p100",    16, 32, 100.0e3, 0.55, 0.0, 0.01),
    # Medium rolling resistance
    ("mu_r20_16x32_mu055_p100",    16, 32, 100.0e3, 0.55, 0.2, 0.01),
    # High rolling resistance (approaching Rowe limit)
    ("mu_r40_16x32_mu055_p100",    16, 32, 100.0e3, 0.55, 0.4, 0.01),
]

COMMON = dict(radius=0.75, strain_target=0.35, consolidation_steps=4000,
              snapshot_strain=1.0e9, wall_friction=0.0)


def summarise(tag: str, res: dict, elapsed: float) -> None:
    h = res["stress_path"]
    eps = np.asarray(h["axial_strain"])
    sv = np.asarray(h["sigma_v"])
    sh = np.asarray(h["sigma_h"])
    q = np.asarray(h["deviator"])
    ev = np.asarray(h.get("volumetric_strain_log", h.get("volumetric_strain", np.zeros_like(eps))))
    phi = np.degrees(np.arcsin(np.clip((sv - sh) / np.maximum(sv + sh, 1e-30),
                                      -0.95, 0.95)))
    print(f"== {tag} elapsed={elapsed:.0f}s steps={res['loading']['steps']} "
          f"final_eps={eps[-1]:.3f} peak_phi={res['peak']['friction_angle_deg']:.2f}")
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

    # Select config for this shard
    idx = args.shard - 1
    if idx < 0 or idx >= len(CONFIGS):
        print(f"Shard {args.shard} out of range (1..{len(CONFIGS)})")
        return 1

    tag, cols, rows, sigma3, mu, mu_r, eps_dot = CONFIGS[idx]
    print(f"[m21110] shard {args.shard}/{args.nshards}: {tag}")
    print(f"  cols={cols} rows={rows} sigma3={sigma3/1e3:.0f}kPa mu={mu} mu_r={mu_r} eps_dot={eps_dot}")

    os.makedirs("runs/m21110", exist_ok=True)
    out_path = f"runs/m21110/rolling_cs_{tag}.json"

    t0 = time.time()
    result = run_biaxial_shear_band(
        confining_stress=sigma3,
        columns=cols, rows=rows,
        strain_rate=eps_dot,
        lateral_control="isochoric",
        material_overrides={
            "rolling_friction_coefficient": mu_r,
            "freeze_bond_fraction": 0.0,
        },
        record_snapshots=False,
        **COMMON,
    )
    elapsed = time.time() - t0

    # Extract stress path for analysis
    hist = result.get("stress_path", {})
    record = {
        "tag": tag,
        "config": {
            "columns": cols, "rows": rows, "sigma3_Pa": sigma3,
            "mu": mu, "mu_r": mu_r, "eps_dot": eps_dot,
            **COMMON,
        },
        "specimen": result["specimen"],
        "loading": result["loading"],
        "peak": result["peak"],
        "stress_path": hist,
        "elapsed_s": round(elapsed, 1),
    }

    summarise(tag, result, elapsed)

    with open(out_path, "w") as f:
        json.dump(record, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"[m21110] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
