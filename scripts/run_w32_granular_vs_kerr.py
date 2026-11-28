#!/usr/bin/env python3
"""W32: Granular DEM vs Kerr model for ice-ridge keel grounding.

Compares the bonded-particle keel specimen (keel_foundation_dem) with the
Kerr three-parameter foundation footing law (ridge_keel_grounding).

Procedure
=========
1. Build a DEM keel specimen (spacing_consistent bond law, seed=7).
2. Run confined_compression -> K (series modulus).
3. Run simple_shear -> G_s (skeleton shear modulus).
4. Run footing_test at b=0.5Hk, 1.0Hk, 2.0Hk -> R_dem.
5. Identify Kerr parameters: K from confined, sqrt(G2*K) from narrow footing.
6. Predict R at all widths using Kerr footing law.
7. Compare DEM vs Kerr.

Output: papers/data/w32-2026-11-16/ice/granular_vs_kerr.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tensordem.keel_foundation_dem import (
    build_keel_sample,
    confined_compression,
    footing_test,
    simple_shear,
    ICE_YOUNG,
    REFERENCE_LENGTH,
)
from tensordem.ridge_keel_grounding import (
    KerrFoundation,
    kerr_footing_resistance,
    kerr_footing_selfcheck,
)


def run_granular_vs_kerr():
    """Run the full granular vs Kerr comparison."""
    # Configuration (matching W28 spacing_consistent arm)
    n_particles = 50000
    height = 2.0  # m
    cell_ratio = 8.0
    solid_fraction = 0.62
    seed = 7
    w0 = 0.008  # m (0.4% of keel height)
    increments = 6
    steps_per_increment = 100

    # Build DEM specimen
    print(f"Building DEM keel specimen: {n_particles} particles, seed={seed}")
    t0 = time.time()
    sample = build_keel_sample(
        n_particles,
        height=height,
        cell_ratio=cell_ratio,
        solid_fraction=solid_fraction,
        seed=seed,
        ice_young=ICE_YOUNG,
        bond_reference_length=REFERENCE_LENGTH,  # spacing_consistent
    )
    build_time = time.time() - t0
    n_bonds = len(sample.bi)
    print(f"  Built in {build_time:.1f}s: {sample.n_particles} particles, "
          f"{n_bonds} bonds, spacing={sample.spacing:.4f}m")

    # Confined compression
    print("Running confined_compression...")
    t0 = time.time()
    confined = confined_compression(
        sample,
        w_max=w0,
        increments=increments,
        steps_per_increment=steps_per_increment,
    )
    confined_time = time.time() - t0
    print(f"  Done in {confined_time:.1f}s: K={confined['k_subgrade_N_per_m3']:.2e} N/m^3")

    # Simple shear (diagnostic)
    print("Running simple_shear...")
    t0 = time.time()
    shear = simple_shear(
        sample,
        gamma_max=w0 / height,
        increments=increments,
        steps_per_increment=steps_per_increment,
    )
    shear_time = time.time() - t0
    print(f"  Done in {shear_time:.1f}s: G_s={shear['G_skeleton_Pa']:.2e} Pa")

    # Footing tests
    footing_widths = [0.5 * height, 1.0 * height, 2.0 * height]  # b = 0.5Hk, 1.0Hk, 2.0Hk
    footing_results = {}
    for b in footing_widths:
        b_ratio = b / height
        print(f"Running footing_test: b={b:.2f}m ({b_ratio:.1f}*Hk)")
        t0 = time.time()
        ft = footing_test(
            sample,
            footing_width=b,
            w0=w0,
            increments=increments,
            steps_per_increment=steps_per_increment,
        )
        ft_time = time.time() - t0
        R_final = ft["resistance_N_per_m"][-1]
        print(f"  Done in {ft_time:.1f}s: R={R_final:.2e} N/m at w0={w0}m")
        footing_results[f"{b_ratio:.1f}"] = {
            "footing_width_m": b,
            "settlements_m": [float(s) for s in ft["settlements_m"]],
            "resistance_N_per_m": [float(r) for r in ft["resistance_N_per_m"]],
            "secant_stiffness_N_per_m2": ft["secant_stiffness_N_per_m2"],
            "n_footing_particles": ft["n_footing_particles"],
            "wall_time_s": ft_time,
        }

    # Identify Kerr parameters from confined + narrow footing (b=0.5Hk)
    print("\nIdentifying Kerr parameters...")
    K = confined["k_subgrade_N_per_m3"]
    b_narrow = footing_results["0.5"]["footing_width_m"]
    R_narrow = footing_results["0.5"]["resistance_N_per_m"][-1]

    sqrt_G2K = (R_narrow - K * w0 * b_narrow) / (2.0 * w0)
    G2 = sqrt_G2K ** 2 / K
    k_upper = 2.0 * K
    k_lower = 2.0 * K
    G = 2.0 * G2

    print(f"  K = {K:.2e} N/m^3")
    print(f"  sqrt(G2*K) = {sqrt_G2K:.2e} N/m^1.5")
    print(f"  G2 = {G2:.2e} N/m")

    # Build KerrFoundation and verify
    kerr_foundation = KerrFoundation(
        k_upper=k_upper,
        k_lower=k_lower,
        shear=G,
    )

    # Self-check the Kerr footing law
    print("\nKerr footing self-check...")
    selfcheck = kerr_footing_selfcheck(
        width=1.0, w0=w0, foundation=kerr_foundation
    )
    print(f"  law_inside_rel_err: {selfcheck['law_inside_rel_err']:.2e}")
    print(f"  law_outside_rel_err: {selfcheck['law_outside_rel_err']:.2e}")
    print(f"  edge_jump_rel_err: {selfcheck['edge_jump_rel_err']:.2e}")
    print(f"  integral_rel_err: {selfcheck['integral_rel_err']:.2e}")
    print(f"  all_pass: {selfcheck['all_pass']}")

    # Predict resistance at all footing widths using Kerr
    print("\nKerr predictions vs DEM results:")
    predictions = {}
    for b_ratio_str, ft_data in footing_results.items():
        b = ft_data["footing_width_m"]
        R_kerr = kerr_footing_resistance(
            width=b, w0=w0, foundation=kerr_foundation
        )
        R_dem = ft_data["resistance_N_per_m"][-1]
        rel_error = (R_kerr - R_dem) / R_dem if R_dem != 0 else float("nan")
        predictions[b_ratio_str] = {
            "footing_width_m": b,
            "R_dem_N_per_m": R_dem,
            "R_kerr_N_per_m": R_kerr,
            "rel_error": rel_error,
            "rel_error_pct": rel_error * 100,
        }
        status = "PASS" if abs(rel_error) <= 0.15 else "FAIL"
        print(f"  b={b:.2f}m ({b_ratio_str}*Hk): DEM={R_dem:.2e}, Kerr={R_kerr:.2e}, "
              f"err={rel_error*100:.1f}% [{status}]")

    # Assemble artifact
    artifact = {
        "artifact": "papers/data/w32-2026-11-16/ice/granular_vs_kerr.json",
        "mission": "paper-exp-ice-ridge-keel-grounding-5",
        "mission_id": 28180,
        "repo": "TensorDEM",
        "runner": "native",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration": {
            "keel_height_m": height,
            "cell_ratio": cell_ratio,
            "solid_fraction": solid_fraction,
            "w0_m": w0,
            "n_particles": n_particles,
            "seed": seed,
            "increments": increments,
            "steps_per_increment": steps_per_increment,
            "bond_law": "spacing_consistent",
            "bond_reference_length_m": REFERENCE_LENGTH,
            "ice_young_Pa": ICE_YOUNG,
            "footing_widths_m": footing_widths,
        },
        "dem_specimen": {
            "n_particles": sample.n_particles,
            "n_bonds": n_bonds,
            "coordination": sample.coordination,
            "length_m": sample.length,
            "height_m": sample.height,
            "spacing_m": sample.spacing,
            "mean_radius_m": float(np.mean(sample.r)),
            "solid_fraction": sample.solid_fraction,
            "kPa_per_bond": float(np.mean(sample.kn) / 1e3),
            "seed": sample.rng_seed,
            "bond_law": sample.meta["bond_law"],
            "bond_reference_length_m": sample.meta.get("bond_reference_length_m"),
        },
        "dem_results": {
            "confined_compression": {
                "test": "confined_compression",
                "settlements_m": [float(s) for s in confined["settlements_m"]],
                "pressures_Pa": [float(p) for p in confined["pressures_Pa"]],
                "k_subgrade_small_strain_N_per_m3": confined["k_subgrade_small_strain_N_per_m3"],
                "k_subgrade_N_per_m3": confined["k_subgrade_N_per_m3"],
                "wall_time_s": confined_time,
            },
            "simple_shear": {
                "test": "simple_shear",
                "gammas": [float(g) for g in shear["gammas"]],
                "taus_Pa": [float(t) for t in shear["taus_Pa"]],
                "G_skeleton_Pa": shear["G_skeleton_Pa"],
                "wall_time_s": shear_time,
            },
            "footings": footing_results,
            "total_wall_time_s": build_time + confined_time + shear_time + sum(
                ft["wall_time_s"] for ft in footing_results.values()
            ),
        },
        "kerr_identification": {
            "method": "K from confined compression, sqrt(G2*K) from narrow footing (b=0.5Hk)",
            "parameters": {
                "K_N_per_m3": K,
                "G2_N_per_m": G2,
                "sqrt_G2K_N_per_m1_5": sqrt_G2K,
                "k_upper_N_per_m3": k_upper,
                "k_lower_N_per_m3": k_lower,
                "shear_N_per_m": G,
                "c_m2": G / (k_upper + k_lower),
            },
            "foundation_dict": kerr_foundation.as_dict(),
        },
        "kerr_selfcheck": selfcheck,
        "comparison": {
            "w0_m": w0,
            "predictions": predictions,
            "summary": {
                "max_abs_rel_error": max(abs(p["rel_error"]) for p in predictions.values()),
                "max_abs_rel_error_pct": max(abs(p["rel_error_pct"]) for p in predictions.values()),
                "all_pass_15pct": all(abs(p["rel_error"]) <= 0.15 for p in predictions.values()),
            },
        },
        "limitations": [
            "Small-strain only (w0/Hk = 0.004), no bond breakage",
            "Plane strain, periodic lateral boundaries",
            "Kerr layer cannot resolve k_u/k_l split from footing test alone",
            "DEM modulus is a lattice modulus, not a continuum ice modulus",
            "Quasi-static solver: no inertia physics claimed",
        ],
    }

    # Write artifact
    output_dir = Path(__file__).resolve().parent / "papers" / "data" / "w32-2026-11-16" / "ice"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "granular_vs_kerr.json"
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nArtifact written to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for b_ratio_str, pred in predictions.items():
        print(f"  b={pred['footing_width_m']:.2f}m ({b_ratio_str}*Hk): "
              f"DEM={pred['R_dem_N_per_m']:.2e} N/m, "
              f"Kerr={pred['R_kerr_N_per_m']:.2e} N/m, "
              f"err={pred['rel_error_pct']:.1f}%")
    max_err = max(abs(p["rel_error"]) for p in predictions.values())
    print(f"\nMax absolute relative error: {max_err*100:.1f}%")
    print(f"Pass criterion (<=15%): {'PASS' if max_err <= 0.15 else 'FAIL'}")

    return artifact


if __name__ == "__main__":
    artifact = run_granular_vs_kerr()
