#!/usr/bin/env python
"""m22106 probe: interparticle-friction ladder of the isochoric biaxial test.

Question the ladder answers (preregistered in
``prereg/m22106_mu_ladder_prereg.md``, written before any of these runs):

    Is the *measured* critical-state friction angle ``phi_cv`` of the
    rotationless 2-D disc cell controlled by the interparticle Coulomb
    coefficient ``mu``, as Coulomb/Rowe theory requires
    (``phi_cv -> arctan(mu)`` at zero dilatancy), or is it pinned by the
    rotationless kinematics of the engine?

Every arm uses the *same* protocol as round 3 (``m20537``) so the numbers stay
comparable: constant-volume (isochoric) shear of an unbonded jittered disc
lattice, frictionless walls, ``sigma_3 = 100 kPa``, ``eps_dot_a = 0.01 1/s``,
``eps_a -> 0.60``, ``24 x 48`` lattice.  Only ``mu`` changes.

Two additions over round 3:

* the contact friction-mobilisation meter is sampled every 500 steps
  (``mobilisation_every``), so the macro angle can be traced to the contact law;
* the meter now also accumulates the *dissipated power* of the same contacts
  (``sum_friction_power``), which gives an independent energy route to the
  measured stress ratio: quasi-static steady shear requires
  ``q * eps_dot_a * V = P_friction + P_damping``.

Usage (compute gate -- one arm per shard, never pass ``cmd`` together with
``cmd_tpl``):

    PYTHONPATH=src python scripts/m22106_mu_ladder_probe.py --shard 1 --nshards 5

Each arm writes ``artifacts/CASE-DEM-BIAXIAL-FRICTION/mu_ladder/iso_<tag>.json``
(``artifacts/`` is what the compute gate returns; ``runs/`` is not shipped).
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

# tag, mu  -- anchor geometry/protocol identical to round 3, only mu changes
CONFIGS = [
    ("mu_010", 0.10),
    ("mu_030", 0.30),
    ("mu_055", 0.55),   # anchor value (round 3 measured 14.31 deg at this mu)
    ("mu_090", 0.90),
    ("mu_150", 1.50),   # extrapolation arm: is there any saturation ceiling?
]

COMMON = dict(confining_stress=100.0e3, columns=24, rows=48, radius=0.75,
              strain_rate=0.01, strain_target=0.60, consolidation_steps=6000,
              snapshot_strain=1.0e9, wall_friction=0.0,
              lateral_control="isochoric", mobilisation_every=500)

OUT_SUBDIR = os.path.join("artifacts", "CASE-DEM-BIAXIAL-FRICTION", "mu_ladder")


def _window_table(stress_path: dict, mobilisation: list) -> list:
    """0.05-strain window means of the quantities the verdict needs."""
    eps = np.asarray(stress_path["axial_strain"], dtype=float)
    sv = np.asarray(stress_path["sigma_v"], dtype=float)
    sh = np.asarray(stress_path["sigma_h"], dtype=float)
    q = np.asarray(stress_path["deviator"], dtype=float)
    ev = np.asarray(stress_path["volumetric_strain_log"], dtype=float)
    rows = []
    for lo in np.arange(0.0, eps[-1] + 1e-9, 0.05):
        m = (eps >= lo) & (eps < lo + 0.05)
        if m.sum() < 5:
            continue
        denom = np.maximum(sv[m] + sh[m], 1e-30)
        phi = np.degrees(np.arcsin(np.clip((sv[m] - sh[m]) / denom, -0.95, 0.95)))
        p = 0.5 * (sv[m] + sh[m])
        rows.append({"strain_lo": float(lo), "strain_hi": float(lo + 0.05),
                     "n": int(m.sum()),
                     "phi_deg_mean": float(phi.mean()),
                     "phi_deg_std": float(phi.std()),
                     "sigma_v_kpa": float(sv[m].mean() / 1e3),
                     "sigma_h_kpa": float(sh[m].mean() / 1e3),
                     "q_kpa": float(q[m].mean() / 1e3),
                     "p_kpa": float(p.mean() / 1e3),
                     "sin_phi": float((q[m] / np.maximum(2 * p, 1e-30)).mean()),
                     "strain_increment_ratio": float(q[m].mean() / np.maximum(2 * p, 1e-30).mean()),
                     "eps_v_log": float(ev[m].mean())})
    if mobilisation:
        rows.append({"mobilisation_samples": len(mobilisation)})
    return rows


def summarise(tag: str, mu: float, res: dict, elapsed: float, path: str) -> None:
    sp = res["stress_path"]
    eps = np.asarray(sp["axial_strain"], dtype=float)
    sv = np.asarray(sp["sigma_v"], dtype=float)
    sh = np.asarray(sp["sigma_h"], dtype=float)
    tail = eps >= max(0.0, eps[-1] - 0.1)
    phi_tail = float(np.degrees(np.arcsin(np.clip(
        ((sv[tail] - sh[tail]) / np.maximum(sv[tail] + sh[tail], 1e-30)).mean(),
        -0.95, 0.95))))
    print(f"== {tag} mu={mu:.2f} elapsed={elapsed:.0f}s steps={res['loading']['steps']} "
          f"final_eps={eps[-1]:.3f} I={res['loading']['inertial_number']:.2e} "
          f"peak_phi={res['peak']['friction_angle_deg']:.2f} "
          f"phi_tail(last 0.1)={phi_tail:.2f} arctan_mu={np.degrees(np.arctan(mu)):.2f}")
    for row in _window_table(sp, res.get("mobilisation") or []):
        if "strain_lo" not in row:
            continue
        print(f"   {row['strain_lo']:.2f}-{row['strain_hi']:.2f} n={row['n']:6d} "
              f"phi={row['phi_deg_mean']:6.2f}+-{row['phi_deg_std']:4.2f} "
              f"q={row['q_kpa']:7.1f} p={row['p_kpa']:7.1f} sh={row['sigma_h_kpa']:6.1f} "
              f"q/2p={row['strain_increment_ratio']:.3f} ev_log={row['eps_v_log']:+.2e}")
    sys.stdout.flush()
    print(f"   wrote {path}")
    sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--smoke", action="store_true",
                    help="run only the 10x20 short arm of the anchor mu "
                         "(83 s on 8 threads) to validate the analysis pipeline")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TENSORDEM_THREADS", "4")))
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    if args.smoke:
        smoke = {**COMMON, "columns": 10, "rows": 20, "strain_rate": 0.05,
                 "strain_target": 0.30, "consolidation_steps": 600}
        res = run_biaxial_shear_band(
            material_overrides={"freeze_bond_fraction": 0.0,
                                "friction_coefficient": 0.55}, **smoke)
        os.makedirs(os.path.join(root, "artifacts", "CASE-DEM-BIAXIAL-FRICTION",
                                 "mu_ladder_smoke"), exist_ok=True)
        path = os.path.join(root, "artifacts", "CASE-DEM-BIAXIAL-FRICTION",
                            "mu_ladder_smoke", "iso_smoke_10x20.json")
        with open(path, "w") as fh:
            json.dump({"tag": "smoke_10x20", "arm": "mu_ladder_smoke",
                       "config": {**smoke, "mu": 0.55,
                                  "freeze_bond_fraction": 0.0},
                       "specimen": res["specimen"], "loading": res["loading"],
                       "peak": res["peak"],
                       "stress_definitions": res["stress_definitions"],
                       "mobilisation": res.get("mobilisation") or [],
                       "stress_path": res["stress_path"]},
                      fh, default=lambda o: o.tolist()
                      if hasattr(o, "tolist") else str(o))
        summarise("smoke_10x20", 0.55, res, 0.0, path)
        return 0
    todo = [c for i, c in enumerate(CONFIGS) if i % args.nshards == args.shard - 1]
    if not todo:
        print(f"shard {args.shard}/{args.nshards}: nothing to do")
        return 0
    for tag, mu in todo:
        started = time.time()
        res = run_biaxial_shear_band(
            material_overrides={"freeze_bond_fraction": 0.0,
                                "friction_coefficient": mu},
            **COMMON)
        elapsed = time.time() - started
        record = {
            "tag": tag, "arm": "mu_ladder",
            "config": {**{k: v for k, v in COMMON.items()},
                       "mu": mu, "freeze_bond_fraction": 0.0},
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
        summarise(tag, mu, res, elapsed, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())