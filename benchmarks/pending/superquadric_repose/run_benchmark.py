"""Benchmark: Superquadric repose angle vs sphere-clump approximation.

This benchmark compares the repose angle of superquadric particles with
their sphere-clump approximations for various shape parameters.

Physical setup:
- Cylindrical drum of radius R_drum, filled with N particles
- Particles settle under gravity, then drum rotates slowly
- Measure the repose angle of the resulting pile

Parameters scanned:
- Shape exponent n1 = n2 ∈ {2, 4, 8, 16, 32, 50}
- Aspect ratio ∈ {1.0, 1.5, 2.0, 3.0}

Output:
- artifacts/dem/superquadric_repose_results.json
- benchmarks/pending/superquadric_repose/repose_angle_comparison.png
"""
from __future__ import annotations

import json
import math
import sys
import os
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tensordem.superquadric import Superquadric
from tensordem.shape_error_budget import (
    compute_geometric_error,
    compute_macroscopic_error,
    build_disable_ratio_table,
)


def run_benchmark():
    """Run the superquadric repose angle benchmark."""
    print("=" * 70)
    print("Superquadric Repose Angle Benchmark")
    print("=" * 70)
    
    # Parameters
    aspect_ratios = [1.0, 1.5, 2.0, 3.0]
    shape_exponents = [2.0, 4.0, 8.0, 16.0, 32.0, 50.0]
    n_clump_spheres = 10
    mu = 0.4  # Friction coefficient
    
    results = []
    
    print(f"\nScanning {len(aspect_ratios)} aspect ratios × {len(shape_exponents)} shape exponents")
    print(f"Clump spheres: {n_clump_spheres}, Friction coefficient: {mu}")
    print()
    
    for ar in aspect_ratios:
        for n in shape_exponents:
            # Create superquadric
            a = ar if ar >= 1.0 else 1.0
            b = 1.0
            c = 1.0
            if ar < 1.0:
                a = 1.0
                c = ar
            
            sq = Superquadric(a=a * 0.005, b=b * 0.005, c=c * 0.005, n1=n, n2=n)
            
            # Compute errors
            geom_err = compute_geometric_error(sq, n_clump_spheres, n_surface_samples=100)
            macro_err = compute_macroscopic_error(sq, n_clump_spheres, mu=mu)
            
            result = {
                "aspect_ratio": ar,
                "shape_exponent": n,
                "repose_angle_true_deg": round(macro_err.repose_angle_true_deg, 2),
                "repose_angle_clump_deg": round(macro_err.repose_angle_clump_deg, 2),
                "repose_angle_diff_deg": round(macro_err.repose_angle_diff_deg, 2),
                "force_chain_deviation_deg": round(macro_err.force_chain_deviation_deg, 2),
                "geometric_error_m": round(geom_err.max_surface_distance, 6),
                "normal_deviation_deg": round(geom_err.max_normal_deviation_deg, 2),
                "n_clump_spheres": geom_err.n_clump_spheres,
            }
            results.append(result)
            
            print(f"  AR={ar:.1f}, n={n:5.1f}: θ_true={result['repose_angle_true_deg']:5.1f}°, "
                  f"θ_clump={result['repose_angle_clump_deg']:5.1f}°, "
                  f"Δθ={result['repose_angle_diff_deg']:5.1f}°, "
                  f"geom_err={result['geometric_error_m']:.2e}m")
    
    # Build disable ratio table
    print("\n" + "=" * 70)
    print("Disable Ratio Table")
    print("=" * 70)
    
    table = build_disable_ratio_table(aspect_ratios, shape_exponents,
                                     sphere_counts=(n_clump_spheres,))
    
    print("\nCriteria:")
    for key, value in table.criteria.items():
        print(f"  {key}: {value}")
    
    print("\nEntries:")
    n_accept = sum(1 for e in table.entries if e["use_clump"])
    n_total = len(table.entries)
    print(f"  Acceptable for clump: {n_accept}/{n_total}")
    
    # Save results
    output_dir = Path(__file__).parent.parent.parent / "artifacts" / "dem"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "superquadric_repose_results.json"
    output_data = {
        "benchmark": "superquadric_repose_angle",
        "parameters": {
            "aspect_ratios": aspect_ratios,
            "shape_exponents": shape_exponents,
            "n_clump_spheres": n_clump_spheres,
            "mu": mu,
        },
        "results": results,
        "disable_ratio_table": json.loads(table.to_json()),
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    # Find the boundary where clump becomes unacceptable
    for entry in table.entries:
        if not entry["use_clump"]:
            print(f"\nFirst unacceptable case: AR={entry['aspect_ratio']}, n={entry['shape_exponent']}")
            print(f"  Repose error: {entry['repose_error_deg']}°")
            print(f"  Force chain error: {entry['force_chain_error_deg']}°")
            print(f"  Geometric error: {entry['geometric_error_m']}m")
            break
    else:
        print("\nAll cases acceptable for clump approximation.")
    
    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)
    
    return output_data


if __name__ == "__main__":
    run_benchmark()
