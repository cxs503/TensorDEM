#!/usr/bin/env python
"""m20304 exploratory probe: unbonded plane-strain biaxial to large strain.

Runs the production biaxial engine (`run_biaxial_shear_band`) with
`freeze_bond_fraction = 0` (unbonded frictional discs) to axial strains well
past peak, and prints the mobilised friction angle / volumetric-strain history
so the critical-state window can be located.  This is a *probe* (exploratory),
not the benchmark deliverable: it only decides where critical state lives.

Usage (compute gate):
    PYTHONPATH=src python scripts/m20304_probe_cs.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.biaxial_shear_band import run_biaxial_shear_band  # noqa: E402

CONFIGS = {
    "base_16x32_100kPa": dict(columns=16, rows=32, confining_stress=100.0e3),
    "res_24x48_100kPa": dict(columns=24, rows=48, confining_stress=100.0e3),
    "pres_16x32_200kPa": dict(columns=16, rows=32, confining_stress=200.0e3),
    "mu030_16x32_100kPa": dict(columns=16, rows=32, confining_stress=100.0e3,
                              mu=0.30),
}

STRAIN_TARGET = 0.40
CONSOLIDATION = 2000


def one(tag: str, cfg: dict) -> dict:
    mu = cfg.pop("mu", 0.55)
    started = time.time()
    res = run_biaxial_shear_band(
        confining_stress=cfg["confining_stress"],
        columns=cfg["columns"], rows=cfg["rows"], radius=0.75,
        strain_rate=0.05, strain_target=STRAIN_TARGET,
        consolidation_steps=CONSOLIDATION,
        snapshot_strain=1.0e9,          # no band snapshots in the probe
        wall_friction=0.0,              # frictionless lateral walls
        material_overrides={"freeze_bond_fraction": 0.0,
                            "friction_coefficient": mu},
    )
    elapsed = time.time() - started
    h = res["stress_path"]
    eps = np.asarray(h["axial_strain"])
    q = np.asarray(h["deviator"])
    sv = np.asarray(h["sigma_v"])
    sh = np.asarray(h["sigma_h"])
    ev = np.asarray(h["volumetric_strain"])
    ratio = np.clip((sv - sh) / np.maximum(sv + sh, 1e-30), -0.95, 0.95)
    phi = np.degrees(np.arcsin(ratio))

    # crude late-window summary at a few strain windows (probe only)
    windows = {}
    for lo, hi in ((0.30, 0.40), (0.25, 0.35), (0.35, 0.40)):
        m = (eps >= lo) & (eps <= hi)
        if m.sum() < 10:
            continue
        windows[f"{lo:.2f}-{hi:.2f}"] = {
            "phi_mean_deg": float(phi[m].mean()),
            "phi_std_deg": float(phi[m].std()),
            "q_mean_kPa": float(q[m].mean() / 1e3),
            "q_std_kPa": float(q[m].std() / 1e3),
            "sigma_h_mean_kPa": float(sh[m].mean() / 1e3),
            "d_ev_d_ea": float(np.polyfit(eps[m], ev[m], 1)[0]),
        }
    row = {
        "tag": tag, "config": {**cfg, "mu": mu, "strain_target": STRAIN_TARGET,
                               "consolidation_steps": CONSOLIDATION},
        "elapsed_s": round(elapsed, 1),
        "dt": res["specimen"]["dt"], "particles": res["specimen"]["particles"],
        "final_strain": float(eps[-1]), "steps": res["loading"]["steps"],
        "inertial_number": res["loading"]["inertial_number"],
        "peak": res["peak"], "windows": windows,
        "history": {"eps": eps.tolist(), "phi": phi.tolist(),
                    "q_kPa": (q / 1e3).tolist(),
                    "sv_kPa": (sv / 1e3).tolist(), "sh_kPa": (sh / 1e3).tolist(),
                    "ev": ev.tolist()},
    }
    print(f"[{tag}] elapsed={elapsed:.0f}s steps={res['loading']['steps']} "
          f"final_eps={eps[-1]:.3f} peak_phi={res['peak']['friction_angle_deg']:.2f}")
    for key, w in windows.items():
        print(f"    win {key}: phi={w['phi_mean_deg']:6.2f}±{w['phi_std_deg']:.2f} "
              f"q={w['q_mean_kPa']:7.1f} sh={w['sigma_h_mean_kPa']:6.1f} "
              f"dEv/dEa={w['d_ev_d_ea']:+.4f}")
    return row


def main() -> int:
    out_dir = os.path.join(os.path.dirname(__file__), "..", "runs", "m20304", "probe")
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for tag, cfg in CONFIGS.items():
        try:
            summary[tag] = one(tag, dict(cfg))
        except Exception as exc:                     # pragma: no cover - probe
            summary[tag] = {"tag": tag, "error": repr(exc)}
            print(f"[{tag}] FAILED: {exc!r}")
        with open(os.path.join(out_dir, "probe.json"), "w") as fh:
            json.dump(summary, fh, indent=1)
    print("wrote", os.path.abspath(os.path.join(out_dir, "probe.json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())