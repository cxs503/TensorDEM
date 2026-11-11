"""Sphere (disk) baseline for the floe-attrition chain: mu_disk(C) from the
m9375 ice-ice shear cell (TensorDEM ``ice_ice_collision``).

Node p472w2-N05 needs a *roundness-independent* anchor: the effective friction
of a field of circular floes at the design concentrations.  The m9375 module
reports an equivalent viscosity ``mu_eff = |tau| / (d gamma_dot)``; the
Coulomb-like, dimensionless quantity that the attrition closure needs is the
**stress ratio** ``mu_sph = |tau_total| / p`` with ``p`` the measured normal
pressure of the same shear cell.  This script recomputes that ratio for the
three design concentrations and writes it (with provenance) to
``artifacts/floe_attrition/sphere_baseline.json``.

Deterministic: the shear cell is seeded (``FloeFieldConfig.seed``), so the
numbers are reproducible bit for bit.

Usage::

    PYTHONPATH=src python scripts/floe_attrition_sphere_baseline.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.ice_ice_collision import FloeFieldConfig, simulate_floe_field  # noqa: E402

OUT_DIR = os.path.join(ROOT, "artifacts", "floe_attrition")
CONCENTRATIONS = (0.30, 0.50, 0.70)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="short shear window (smoke test only)")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "sphere_baseline.json"))
    args = ap.parse_args(argv)

    rows = []
    for c in CONCENTRATIONS:
        cfg = FloeFieldConfig(
            concentration=c,
            settle_steps=1500 if args.quick else 2000,
            shear_steps=1500 if args.quick else 4000,
            sample_steps=800 if args.quick else 2000,
        )
        t0 = time.perf_counter()
        out = simulate_floe_field(cfg)
        wall = time.perf_counter() - t0
        pressure = float(out["pressure"])
        if not (pressure > 0.0):
            raise SystemExit(f"non-positive pressure at C={c}: {pressure}")
        mu_stress_ratio = abs(float(out["tau_total"])) / pressure
        rows.append({
            "concentration": c,
            "n_floes": int(out["n_floes"]),
            "mu_sph": mu_stress_ratio,          # |tau| / p, dimensionless
            "mu_eff_viscosity": float(out["mu_eff"]),  # m9375 definition, Pa s
            "tau_total": float(out["tau_total"]),
            "pressure": pressure,
            "coordination": float(out["coordination"]),
            "mean_contacts": float(out["mean_contacts"]),
            "bond_ruptures": int(out.get("bond_ruptures", 0)),
            "wall_seconds": wall,
            "config": {
                "box_width": cfg.box_width, "box_height": cfg.box_height,
                "thickness": cfg.thickness, "mean_radius": cfg.mean_radius,
                "friction": cfg.friction, "wall_friction": cfg.wall_friction,
                "shear_rate": cfg.shear_rate, "settle_steps": cfg.settle_steps,
                "shear_steps": cfg.shear_steps, "sample_steps": cfg.sample_steps,
                "seed": cfg.seed,
            },
        })
        print(f"C={c:.2f} N={rows[-1]['n_floes']:4d} "
              f"mu_sph={mu_stress_ratio:.4f} Z={out['coordination']:.3f} "
              f"({wall:.1f} s)", flush=True)

    mu = [r["mu_sph"] for r in rows]
    z = [r["coordination"] for r in rows]
    trend = ("increasing" if all(mu[i] < mu[i + 1] for i in range(len(mu) - 1))
             else "decreasing" if all(mu[i] > mu[i + 1] for i in range(len(mu) - 1))
             else "non-monotone")
    print(f"mu_sph(C) trend: {trend}; coordination: {z}")

    payload = {
        "provenance": {
            "node": "p472w2-N05",
            "purpose": "roundness-independent sphere baseline mu_sph(C)",
            "source_module": "tensordem.ice_ice_collision (mission 9375)",
            "definition": "mu_sph = |tau_total| / pressure (stress ratio)",
            "seed": FloeFieldConfig().seed,
            "quick": bool(args.quick),
            "wall_seconds_total": sum(r["wall_seconds"] for r in rows),
            "generated_by": "scripts/floe_attrition_sphere_baseline.py",
        },
        "rows": rows,
        "mu_sph_trend": trend,
        "coordination_trend": ("increasing" if all(z[i] < z[i + 1] for i in range(len(z) - 1))
                               else "non-monotone"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())