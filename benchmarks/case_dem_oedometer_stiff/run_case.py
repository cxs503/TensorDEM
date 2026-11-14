#!/usr/bin/env python3
"""CASE-DEM-OEDOMETER-STIFF -- run the DEM oedometer and produce artifacts.

Runs the DEM oedometer test with two configurations:
1. **Primary**: zero friction (mu=0, mu_r=0) for comparison with micromechanical theory
2. **Secondary**: standard friction (mu=0.4, mu_r=0.02) for realistic granular behavior

Writes:
* ``artifacts/case_dem_oedometer_stiff/dem_result.json`` -- primary (zero friction) result
* ``artifacts/case_dem_oedometer_stiff/dem_result_frictional.json`` -- secondary result
* ``artifacts/case_dem_oedometer_stiff/reference.json``   -- micromechanical prediction

Usage::
    PYTHONPATH=src python3 benchmarks/case_dem_oedometer_stiff/run_case.py [--frictional]
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ART = REPO / "artifacts" / "case_dem_oedometer_stiff"

from tensordem.consolidation import run_oedometer, OedometerConfig  # noqa: E402


def micromechanical_prediction(kn: float, radius: float, porosity: float,
                               coordination: float) -> dict:
    """Micromechanical prediction for constrained modulus (Chang, Kuhn & Rothenburg).

    Reference: Chang, C. S., Kuhn, M. R., & Rothenburg, L. (2004).
    "Micromechanical analysis of granular materials."
    For isotropic random packing of spheres with linear normal contacts:
        K = kn * Z * (1-n) / (18 * pi * r)
        G = kn * Z * (1-n) / (30 * pi * r)  (assuming kt=0 for frictional contacts)
        E = 9KG / (3K + G)
        nu = (3K - 2G) / (2(3K + G))
        E_oed = E * (1-nu) / ((1+nu)*(1-2*nu))
    """
    r = radius
    n = porosity
    Z = coordination

    K = kn * Z * (1 - n) / (18 * math.pi * r)
    G = kn * Z * (1 - n) / (30 * math.pi * r)
    E = 9 * K * G / (3 * K + G)
    nu = (3 * K - 2 * G) / (2 * (3 * K + G))
    E_oed = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))

    return {
        "K": K,
        "G": G,
        "E": E,
        "nu": nu,
        "E_oed": E_oed,
        "reference": "Chang, Kuhn & Rothenburg (micromechanical homogenization)",
    }


def run_one(cfg: OedometerConfig, label: str) -> dict:
    print(f"[{label}] Running DEM oedometer...")
    print(f"  Config: n_row={cfg.n_row}, n_layer={cfg.n_layer}, seed={cfg.seed}")
    print(f"  kn={cfg.kn} N/m, d={2*cfg.radius} m, friction={cfg.friction}, mu_r={cfg.mu_rolling}")
    t0 = time.time()
    result = run_oedometer(cfg, progress=lambda m: print(f"    {m}"))
    elapsed = time.time() - t0
    result["wall_time_s"] = elapsed
    result["label"] = label
    return result


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frictional", action="store_true", help="also run frictional case")
    args = parser.parse_args()

    ART.mkdir(parents=True, exist_ok=True)

    # Primary configuration: zero friction (matches micromechanical assumptions)
    cfg_primary = OedometerConfig(n_row=4, n_layer=5, seed=20261108, friction=0.0, mu_rolling=0.0)

    print("=" * 78)
    print("CASE-DEM-OEDOMETER-STIFF: DEM oedometer stress-strain stiffness benchmark")
    print("=" * 78)
    print()

    # Run primary (zero friction)
    result_primary = run_one(cfg_primary, "primary_zero_friction")

    print()
    print(f"Primary results:")
    print(f"    E_oed = {result_primary['E_oed']:.4g} Pa")
    print(f"    porosity = {result_primary['porosity']:.4f}")
    print(f"    coordination = {result_primary['coordination']:.2f}")
    print(f"    wall_time = {result_primary['wall_time_s']:.1f} s")

    # Compute micromechanical prediction for primary
    pred_primary = micromechanical_prediction(
        cfg_primary.kn, cfg_primary.radius,
        result_primary["porosity"], result_primary["coordination"]
    )
    error_pct = (result_primary["E_oed"] - pred_primary["E_oed"]) / pred_primary["E_oed"] * 100
    print(f"    E_oed (micromech) = {pred_primary['E_oed']:.4g} Pa")
    print(f"    Error = {error_pct:.2f} %")

    # Save primary result
    dem_out = {
        "E_oed": result_primary["E_oed"],
        "E_oed_mid": result_primary.get("E_oed_mid", result_primary["E_oed"]),
        "k_hyd": result_primary["k_hyd"],
        "porosity": result_primary["porosity"],
        "coordination": result_primary["coordination"],
        "strain_increment": result_primary["strain_increment"],
        "total_steps": result_primary["total_steps"],
        "wall_time_s": result_primary["wall_time_s"],
        "label": "primary_zero_friction",
        "stages": result_primary["stages"],
        "config": {
            "kn": cfg_primary.kn,
            "radius": cfg_primary.radius,
            "diameter": 2 * cfg_primary.radius,
            "density": cfg_primary.density,
            "friction": cfg_primary.friction,
            "mu_rolling": cfg_primary.mu_rolling,
            "n_row": cfg_primary.n_row,
            "n_layer": cfg_primary.n_layer,
            "seed": cfg_primary.seed,
            "particle_count": cfg_primary.particle_count,
            "box_width": cfg_primary.box_width,
            "dt": cfg_primary.timestep,
            "stress_levels": list(cfg_primary.stress_levels),
        },
    }

    # Save reference
    E_sc = cfg_primary.kn / (2 * cfg_primary.radius)
    ref_out = {
        "micromechanical": pred_primary,
        "simple_cubic_limit": {
            "E_oed": E_sc,
            "formula": "kn / d",
            "note": "Upper bound for 1D chain of linear springs",
        },
        "reference_sources": [
            "Chang, C. S., Kuhn, M. R., & Rothenburg, L. (2004). "
            "Micromechanical analysis of granular materials.",
            "Das, B. M. Advanced Soil Mechanics (for Terzaghi consolidation theory)",
        ],
    }

    dem_path = ART / "dem_result.json"
    ref_path = ART / "reference.json"
    dem_path.write_text(json.dumps(dem_out, indent=2))
    ref_path.write_text(json.dumps(ref_out, indent=2))

    # Optionally run frictional case
    if args.frictional:
        print()
        cfg_frictional = OedometerConfig(n_row=4, n_layer=5, seed=20261108, friction=0.4, mu_rolling=0.02)
        result_frictional = run_one(cfg_frictional, "secondary_frictional")

        pred_frictional = micromechanical_prediction(
            cfg_frictional.kn, cfg_frictional.radius,
            result_frictional["porosity"], result_frictional["coordination"]
        )
        error_pct_f = (result_frictional["E_oed"] - pred_frictional["E_oed"]) / pred_frictional["E_oed"] * 100
        print(f"    E_oed (micromech) = {pred_frictional['E_oed']:.4g} Pa")
        print(f"    Error = {error_pct_f:.2f} %")

        dem_out_f = {
            "E_oed": result_frictional["E_oed"],
            "porosity": result_frictional["porosity"],
            "coordination": result_frictional["coordination"],
            "k_hyd": result_frictional["k_hyd"],
            "strain_increment": result_frictional["strain_increment"],
            "wall_time_s": result_frictional["wall_time_s"],
            "label": "secondary_frictional",
            "stages": result_frictional["stages"],
            "config": {
                "kn": cfg_frictional.kn,
                "friction": cfg_frictional.friction,
                "mu_rolling": cfg_frictional.mu_rolling,
            },
        }
        (ART / "dem_result_frictional.json").write_text(json.dumps(dem_out_f, indent=2))

    print()
    print(f"Artifacts written:")
    print(f"  {dem_path}")
    print(f"  {ref_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
