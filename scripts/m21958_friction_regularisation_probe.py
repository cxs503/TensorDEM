#!/usr/bin/env python
"""m21958 probe: is the contact-friction regularisation velocity a hidden input to phi_cv?

The unbonded contact law of the rotationless disk engine is not Coulomb but its
regularised form ``F_t = -mu F_n tanh(v_t / v_reg)`` (``ridge_dem.RidgeAssembly.forces``,
``RidgeMaterial.velocity_regularisation``, default 0.01 m/s).  That law is a
function of a *velocity ratio*, so the measured macro friction angle is allowed
to depend on a purely numerical parameter.  CASE-DEM-BIAXIAL-FRICTION sits 50 %
below Rowe(1962) ``arctan(mu) = 28.81 deg``; this probe tests whether the gap is
that regularisation.

Pre-registration: ``prereg/m21958_friction_regularisation_prereg.md`` (written
before any run).  One arm per shard.

Usage (compute gate):
    TENSORDEM_THREADS=3 PYTHONPATH=src python \\
        scripts/m21958_friction_regularisation_probe.py --shard 1 --nshards 5

Each arm writes ``runs/m21958/reg_<tag>.json`` with the full stress path and the
contact-friction mobilisation record.
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

MU = 0.55
# tag, strain rate [1/s], velocity regularisation [m/s] or None (material default)
ARMS = [
    ("A_base_rate0p01_vreg_default", 0.01, None),
    ("B_vreg_2p5e-3", 0.01, 2.5e-3),
    ("C_vreg_4e-2", 0.01, 4.0e-2),
    ("D_vreg_1e-1", 0.01, 1.0e-1),
    ("E_rate0p05_vreg_default", 0.05, None),
]
COMMON = dict(columns=24, rows=48, radius=0.75, strain_target=0.45,
              consolidation_steps=6000, snapshot_strain=1.0e9,
              wall_friction=0.0, lateral_control="isochoric",
              mobilisation_every=500)


def summarise(tag: str, res: dict, elapsed: float) -> None:
    h = res["stress_path"]
    eps = np.asarray(h["axial_strain"])
    sv = np.asarray(h["sigma_v"])
    sh = np.asarray(h["sigma_h"])
    phi = np.degrees(np.arcsin(np.clip((sv - sh) / np.maximum(sv + sh, 1e-30),
                                      -0.95, 0.95)))
    mob = res.get("mobilisation") or []
    tail = [m for m in mob if m["axial_strain"] >= 0.15]
    tanh_mean = (float(np.nanmean([m["fn_weighted_mean_abs_tanh"] for m in tail]))
                 if tail else float("nan"))
    print(f"== {tag} elapsed={elapsed:.0f}s steps={res['loading']['steps']} "
          f"final_eps={eps[-1]:.3f} I={res['loading']['inertial_number']:.2e} "
          f"peak_phi={res['peak']['friction_angle_deg']:.2f} "
          f"phi_tail={np.nanmean(phi[eps >= 0.3]):.2f} "
          f"fnwt_mean_abs_tanh={tanh_mean:.3f}")
    sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--tag", default=None,
                    help="run a single arm by tag (overrides shard selection)")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TENSORDEM_THREADS", "4")))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                           "runs", "m21958"))
    os.makedirs(out_dir, exist_ok=True)
    if args.tag:
        todo = [a for a in ARMS if a[0] == args.tag]
        if not todo:
            raise SystemExit(f"unknown tag {args.tag!r}; known: {[a[0] for a in ARMS]}")
    else:
        todo = [a for i, a in enumerate(ARMS) if i % args.nshards == args.shard - 1]
    for tag, rate, v_reg in todo:
        overrides = {"freeze_bond_fraction": 0.0, "friction_coefficient": MU}
        if v_reg is not None:
            overrides["velocity_regularisation"] = float(v_reg)
        started = time.time()
        res = run_biaxial_shear_band(
            confining_stress=100.0e3, strain_rate=rate,
            material_overrides=overrides, **COMMON)
        elapsed = time.time() - started
        summarise(tag, res, elapsed)
        record = {
            "tag": tag,
            "config": {"columns": COMMON["columns"], "rows": COMMON["rows"],
                       "confining_stress": 100.0e3, "mu": MU,
                       "strain_rate": rate,
                       "velocity_regularisation": (v_reg if v_reg is not None
                                                   else 0.01),
                       "velocity_regularisation_is_default": v_reg is None,
                       "freeze_bond_fraction": 0.0, "wall_friction": 0.0,
                       "lateral_control": "isochoric",
                       "strain_target": COMMON["strain_target"],
                       "mobilisation_every": COMMON["mobilisation_every"]},
            "specimen": res["specimen"], "loading": res["loading"],
            "peak": res["peak"],
            "consolidation_state": res["consolidation_state"],
            "mobilisation": res["mobilisation"],
            "elapsed_s": elapsed,
            "stress_path": res["stress_path"],
        }
        path = os.path.join(out_dir, f"reg_{tag}.json")
        with open(path, "w") as fh:
            json.dump(record, fh, default=lambda o: o.tolist()
                      if hasattr(o, "tolist") else str(o))
        print(f"   wrote {path}")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())