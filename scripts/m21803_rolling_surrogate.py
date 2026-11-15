#!/usr/bin/env python
"""m21803 probe: is the "rolling resistance" surrogate really rolling resistance?

Round 5 (m21110) submitted a rolling-resistance calibration (phi_cv vs mu_r) but
never returned its artefacts.  Before re-running that calibration we read the
implementation (``RidgeAssembly.contact_forces``):

    rolling_limit = mu_r * F_n
    rolling_force = -rolling_limit * tanh(v_t / v_regularisation)
    total_tangential = tangential + rolling_force

i.e. the "rolling couple" is realised as an extra *sliding* friction force on the
same tangential direction.  At full slip (tanh -> 1) the contact tangential force
is therefore ``-(mu + mu_r) * F_n``: an additive friction, not a rolling torque
(particles in this model carry no rotational degree of freedom).

If that reading is right, phi_cv depends on ``mu + mu_r`` only, so

    phi_cv(mu=0.55, mu_r=0.20) ~= phi_cv(mu=0.75, mu_r=0.00)
    phi_cv(mu=0.55, mu_r=0.40) ~= phi_cv(mu=0.95, mu_r=0.00)

This probe runs exactly those five isochoric (constant-volume) biaxial shears so
the additivity hypothesis can be tested, not asserted.  Mu_r = 0 cases double as
the mu -> phi_cv baseline that the R5 calibration curve was supposed to provide.

Usage (one configuration per shard):
    PYTHONPATH=src python scripts/m21803_rolling_surrogate.py --shard 2 --nshards 5

Each configuration writes ``runs/m21803/rolling_surrogate_<tag>.json`` with the
full stress path plus the four-gate critical-state extraction.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.biaxial_critical_state import (  # noqa: E402
    critical_state_from_stress_path,
)
from tensordem.biaxial_shear_band import run_biaxial_shear_band  # noqa: E402

RUNS_DIR = "runs/m21803"

# tag, mu, mu_r
CONFIGS = [
    ("mu055_mur00", 0.55, 0.00),   # baseline (replicates R3 phi_cv at 16x32)
    ("mu055_mur20", 0.55, 0.20),   # rolling surrogate  -> mu_eff 0.75 ?
    ("mu055_mur40", 0.55, 0.40),   # rolling surrogate  -> mu_eff 0.95 ?
    ("mu075_mur00", 0.75, 0.00),   # pure-friction control for mu_eff = 0.75
    ("mu095_mur00", 0.95, 0.00),   # pure-friction control for mu_eff = 0.95
]

COMMON = dict(
    columns=16, rows=32,
    radius=0.75,
    confining_stress=100.0e3,
    strain_rate=0.01,
    strain_target=0.35,
    consolidation_steps=4000,
    snapshot_strain=1.0e9,
    wall_friction=0.0,
    lateral_control="isochoric",
)


def summarise(tag: str, res: dict, elapsed: float) -> dict:
    h = res["stress_path"]
    eps = np.asarray(h["axial_strain"], dtype=float)
    sv = np.asarray(h["sigma_v"], dtype=float)
    sh = np.asarray(h["sigma_h"], dtype=float)
    q = np.asarray(h["deviator"], dtype=float)
    phi = np.degrees(np.arcsin(np.clip((sv - sh) / np.maximum(sv + sh, 1e-30), -0.95, 0.95)))
    print(f"== {tag} elapsed={elapsed:.0f}s steps={res['loading']['steps']} "
          f"final_eps={eps[-1]:.3f} peak_phi={res['peak']['friction_angle_deg']:.2f}")
    for lo in np.arange(0.0, eps[-1] + 0.05, 0.05):
        m = (eps >= lo) & (eps < lo + 0.05)
        if m.sum() < 5:
            continue
        p = 0.5 * (sv[m] + sh[m])
        print(f"   {lo:.2f}-{lo + 0.05:.2f} n={int(m.sum()):6d} "
              f"phi={phi[m].mean():6.2f}+-{phi[m].std():4.2f} "
              f"q={q[m].mean() / 1e3:7.1f} p={p.mean() / 1e3:7.1f} "
              f"sh={sh[m].mean() / 1e3:6.1f}")
    sys.stdout.flush()

    cs = critical_state_from_stress_path(
        h, strain_measure="log", boundary_control="isochoric")
    return {
        "phi_late_mean_deg": float(np.mean(phi[eps >= 0.25])) if (eps >= 0.25).any() else None,
        "critical_state": cs.as_dict() if hasattr(cs, "as_dict") else _cs_dict(cs),
    }


def _cs_dict(cs) -> dict:
    out = {}
    for name in ("phi_deg", "q_over_p", "axial_strain", "volumetric_strain_log",
                 "gates", "accepted", "reason"):
        if hasattr(cs, name):
            out[name] = getattr(cs, name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=1, help="1-based configuration index")
    ap.add_argument("--nshards", type=int, default=len(CONFIGS))
    args = ap.parse_args()

    idx = args.shard - 1
    if idx < 0 or idx >= len(CONFIGS):
        print(f"shard {args.shard} out of range 1..{len(CONFIGS)}")
        return 2

    tag, mu, mu_r = CONFIGS[idx]
    print(f"[m21803] shard {args.shard}/{args.nshards}: {tag} mu={mu} mu_r={mu_r}")
    print(f"  {COMMON}")

    os.makedirs(RUNS_DIR, exist_ok=True)
    out_path = os.path.join(RUNS_DIR, f"rolling_surrogate_{tag}.json")

    t0 = time.time()
    result = run_biaxial_shear_band(
        material_overrides={
            "friction_coefficient": mu,
            "rolling_friction_coefficient": mu_r,
            "freeze_bond_fraction": 0.0,
        },
        record_snapshots=False,
        **COMMON,
    )
    elapsed = time.time() - t0

    summary = summarise(tag, result, elapsed)
    record = {
        "tag": tag,
        "config": {**COMMON, "mu": mu, "mu_r": mu_r,
                   "mu_eff_additive": mu + mu_r},
        "specimen": result["specimen"],
        "loading": result["loading"],
        "peak": result["peak"],
        "stress_path": result["stress_path"],
        **summary,
        "elapsed_s": round(elapsed, 1),
    }
    with open(out_path, "w") as f:
        json.dump(record, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"[m21803] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())