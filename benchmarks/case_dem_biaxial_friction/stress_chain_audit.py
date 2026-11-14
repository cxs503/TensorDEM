#!/usr/bin/env python
"""Stress-chain audit of the plane-strain biaxial cell (CASE-DEM-BIAXIAL-FRICTION).

Two questions are answered with numbers produced by the production engine
(:func:`tensordem.biaxial_shear_band.run_biaxial_shear_band`):

Q1  Is the consolidation phase really *isotropic*, as the case bookkeeping
    states?  The legacy cell pressurises only the lateral servo walls; the top
    and bottom platens are position fixed, so ``sigma_v`` at the end of
    consolidation is a *reaction* stress, not the prescribed 100 kPa.  The audit
    measures the end-of-consolidation stress ratio ``K = sigma_v / sigma_h``
    with and without the new axial (platen) pressure servo
    (``axial_servo=True``).

Q2  Does the externally instrumented boundary traction agree with the internal
    Love-Weber (virial) stress?  The two independent stress measures are both
    recorded by the engine; in a quasi-static cell they must agree, so their
    relative difference bounds the part of the measured friction angle that can
    be attributed to the stress *measurement* rather than to the material.

Outputs
-------
``artifacts/CASE-DEM-BIAXIAL-FRICTION/stress_chain_audit.json``  raw record
``artifacts/CASE-DEM-BIAXIAL-FRICTION/stress_chain_audit.png``   figure

Usage
-----
    PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/stress_chain_audit.py
    PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/stress_chain_audit.py \
        --columns 16 --rows 32 --consolidation-steps 4000 --strain-target 0.10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

from tensordem.biaxial_shear_band import run_biaxial_shear_band  # noqa: E402

CASE_KEY = "CASE-DEM-BIAXIAL-FRICTION"
DEFAULT_OUT = os.path.join(REPO, "artifacts", CASE_KEY)


def _run(axial_servo: bool, args, tag: str) -> dict:
    started = time.time()
    res = run_biaxial_shear_band(
        confining_stress=float(args.confining_stress),
        columns=int(args.columns), rows=int(args.rows), radius=0.75,
        strain_rate=float(args.strain_rate), strain_target=float(args.strain_target),
        consolidation_steps=int(args.consolidation_steps),
        snapshot_strain=float(args.snapshot_strain),
        servo_gain=3.0e-4, servo_velocity_limit=float(args.servo_velocity_limit),
        wall_friction=0.0,
        material_overrides={"freeze_bond_fraction": 0.35},
        record_snapshots=False, axial_servo=bool(axial_servo),
    )
    elapsed = time.time() - started
    cons = res["consolidation_state"]
    sd = res["stress_definitions"]
    hist = res["stress_path"]
    n = int(len(hist["axial_strain"]))
    idx = np.unique(np.linspace(0, n - 1, min(400, n)).round().astype(int))
    record = {
        "tag": tag,
        "axial_servo": bool(axial_servo),
        "config": {
            "columns": int(args.columns), "rows": int(args.rows),
            "confining_stress": float(args.confining_stress),
            "consolidation_steps": int(args.consolidation_steps),
            "strain_rate": float(args.strain_rate),
            "strain_target": float(args.strain_target),
            "wall_friction": 0.0,
            "freeze_bond_fraction": 0.35,
        },
        "consolidation_state": cons,
        "peak": {
            "friction_angle_deg": float(res["peak"]["friction_angle_deg"]),
            "deviator": float(res["peak"]["deviator"]),
            "axial_strain": float(res["peak"]["axial_strain"]),
            "sigma_v": float(res["peak"]["sigma_v"]),
            "sigma_h": float(res["peak"]["sigma_h"]),
            "dilation_angle_deg": float(res["peak"]["dilation_angle_deg"]),
        },
        "stress_definitions": {
            "friction_angle_internal_deg": float(sd["friction_angle_internal_deg"]),
            "sigma_v_relative_difference": float(sd["sigma_v_relative_difference"]),
            "sigma_h_relative_difference": float(sd["sigma_h_relative_difference"]),
            "friction_angle_relative_difference":
                float(sd["friction_angle_relative_difference"]),
        },
        "specimen": res["specimen"],
        "loading": res["loading"],
        "elapsed_s": round(elapsed, 1),
        "stress_path": {k: [float(v[i]) for i in idx] for k, v in hist.items()},
    }
    dphi = abs(record["stress_definitions"]["friction_angle_internal_deg"]
               - record["peak"]["friction_angle_deg"])
    record["stress_chain"] = {
        # Bound on the part of phi that the *measurement* definition can explain.
        "phi_wall_minus_internal_deg": (
            record["peak"]["friction_angle_deg"]
            - record["stress_definitions"]["friction_angle_internal_deg"]),
        "abs_phi_gap_deg": float(dphi),
        "consistent_within_2deg": bool(dphi <= 2.0),
        "consistent_within_0p1_rel": bool(
            abs(record["stress_definitions"]["friction_angle_relative_difference"]) <= 0.1),
    }
    return record


def build_figure(audit: dict, path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = audit["runs"]
    labels = [r["tag"] for r in runs]
    K = [r["consolidation_state"]["sigma_v_over_sigma_h"] for r in runs]
    phi = [r["peak"]["friction_angle_deg"] for r in runs]
    phi_i = [r["stress_definitions"]["friction_angle_internal_deg"] for r in runs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.bar(labels, K, color=["#c0504d", "#4f81bd"])
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="isotropic (K = 1)")
    ax.set_ylabel(r"$K = \sigma_v/\sigma_h$ at end of consolidation")
    ax.set_title("Q1: is the consolidation isotropic?")
    for i, v in enumerate(K):
        ax.annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=9)
    ax.legend(loc="lower right")

    ax = axes[1]
    x = np.arange(len(runs))
    ax.bar(x - 0.18, phi, width=0.36, label="wall traction", color="#4f81bd")
    ax.bar(x + 0.18, phi_i, width=0.36, label="Love-Weber virial", color="#9bbb59")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"peak friction angle $\phi$ (deg)")
    ax.set_title("Q2: boundary traction vs internal stress")
    ax.legend(loc="lower right")
    for i in range(len(runs)):
        ax.annotate(f"{abs(phi[i]-phi_i[i]):.2f}°",
                    (x[i], max(phi[i], phi_i[i])), ha="center", va="bottom", fontsize=9)

    fig.suptitle("CASE-DEM-BIAXIAL-FRICTION stress-chain audit", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--columns", type=int, default=16)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--confining-stress", type=float, default=100.0e3)
    parser.add_argument("--consolidation-steps", type=int, default=4000)
    parser.add_argument("--strain-rate", type=float, default=0.02)
    parser.add_argument("--strain-target", type=float, default=0.10)
    parser.add_argument("--snapshot-strain", type=float, default=0.02)
    parser.add_argument("--servo-velocity-limit", type=float, default=0.5)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    torch.set_num_threads(int(os.environ.get("TENSORDEM_THREADS", "4")))
    os.makedirs(args.out_dir, exist_ok=True)

    runs = []
    for tag, axial_servo in (("legacy (platens position-fixed)", False),
                             ("axial_servo=True", True)):
        print(f"[audit] running {tag} ...", flush=True)
        runs.append(_run(axial_servo, args, tag))
        print(f"[audit]   done in {runs[-1]['elapsed_s']} s: "
              f"K = {runs[-1]['consolidation_state']['sigma_v_over_sigma_h']:.3f}, "
              f"phi = {runs[-1]['peak']['friction_angle_deg']:.2f} deg", flush=True)

    audit = {
        "case_key": CASE_KEY,
        "audit": "biaxial stress-chain (consolidation isotropy + boundary/virial consistency)",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_threads": int(torch.get_num_threads()),
        },
        "runs": runs,
        "questions": {
            "Q1_consolidation_isotropy": {
                "legacy_K": runs[0]["consolidation_state"]["sigma_v_over_sigma_h"],
                "axial_servo_K": runs[1]["consolidation_state"]["sigma_v_over_sigma_h"],
                "legacy_isotropic_within_10pct":
                    runs[0]["consolidation_state"]["isotropic_within_10pct"],
                "axial_servo_isotropic_within_10pct":
                    runs[1]["consolidation_state"]["isotropic_within_10pct"],
            },
            "Q2_stress_definition": {
                "abs_phi_gap_deg": [r["stress_chain"]["abs_phi_gap_deg"] for r in runs],
                "consistent_within_2deg": [r["stress_chain"]["consistent_within_2deg"]
                                           for r in runs],
            },
        },
        "repro_cmd": ("PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/"
                      "stress_chain_audit.py --columns %d --rows %d "
                      "--consolidation-steps %d --strain-target %.3f"
                      % (args.columns, args.rows, args.consolidation_steps,
                         args.strain_target)),
    }
    json_path = os.path.join(args.out_dir, "stress_chain_audit.json")
    with open(json_path, "w") as fh:
        json.dump(audit, fh, indent=1)
    fig_path = build_figure(audit, os.path.join(args.out_dir, "stress_chain_audit.png"))
    print(f"[audit] wrote {json_path}")
    print(f"[audit] wrote {fig_path}")

    q1 = audit["questions"]["Q1_consolidation_isotropy"]
    q2 = audit["questions"]["Q2_stress_definition"]
    print(f"[audit] Q1 K legacy={q1['legacy_K']:.3f} servo={q1['axial_servo_K']:.3f}")
    print(f"[audit] Q2 |dphi| = {q2['abs_phi_gap_deg']} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())