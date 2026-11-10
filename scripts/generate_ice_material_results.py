#!/usr/bin/env python3
"""Generate results.json for ice material constitutive model.

This script computes the strength surface, parameter tables, and uncertainty
confidence bands, then saves them to artifacts/ice_material_results.json.
"""

import json
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, "/root/tp-agent/work/m9977/src")

from tensordem.ice_material.brine_volume import brine_volume_fraction
from tensordem.ice_material.compressive_strength import (
    compressive_strength_fyi,
    compressive_strength_myi,
    pure_ice_strength,
)
from tensordem.ice_material.uncertainty import (
    monte_carlo_strength,
    strength_confidence_band,
    sensitivity_analysis,
)
from tensordem.ice_material.visualization import generate_all_figures


def main():
    """Generate all results and save to JSON."""
    print("Generating ice material constitutive model results...")

    # 1. Compute strength surface (FYI)
    print("\n1. Computing FYI strength surface...")
    T_range = np.linspace(-20, -1, 20)
    S_range = np.linspace(0, 15, 16)
    
    fyi_surface = []
    for T in T_range:
        for S in S_range:
            nu_b = brine_volume_fraction(T, S)
            sigma_fyi = compressive_strength_fyi(T, S)
            fyi_surface.append({
                "T": float(T),
                "S": float(S),
                "nu_b": float(nu_b),
                "sigma_c_MPa": float(sigma_fyi),
            })
    
    # 2. Compute strength surface (MYI)
    print("2. Computing MYI strength surface...")
    myi_surface = []
    for T in T_range:
        for S in S_range:
            nu_b = brine_volume_fraction(T, S)
            sigma_myi = compressive_strength_myi(T, S, f_desal=0.3)
            myi_surface.append({
                "T": float(T),
                "S": float(S),
                "nu_b": float(nu_b),
                "sigma_c_MPa": float(sigma_myi),
            })
    
    # 3. Parameter table
    print("3. Computing parameter table...")
    parameter_table = {
        "pure_ice_strength": {
            "sigma_ref_MPa": 25.0,
            "T_ref_C": -10.0,
            "n_exponent": 0.75,
            "description": "Reference pure ice strength at -10°C",
        },
        "brine_weakening": {
            "k_weakening": 1.87,
            "description": "Brine weakening parameter (Timco & Frederking 1984)",
        },
        "myi_desalination": {
            "f_desal_default": 0.3,
            "f_desal_range": [0.2, 0.4],
            "description": "MYI desalination factor (mature MYI)",
        },
        "cox_weeks_coefficients": {
            "c1": 49.185,
            "c2": 0.532,
            "description": "Frankenstein & Garner (1967) phase diagram coefficients",
        },
    }
    
    # 4. Uncertainty confidence bands
    print("4. Computing uncertainty confidence bands...")
    T_band = np.linspace(-20, -1, 15)
    
    fyi_band = strength_confidence_band(
        T_band, salinity=5.0, ice_type="FYI",
        T_std=0.5, S_std=0.5, n_samples_per_T=200,
    )
    
    myi_band = strength_confidence_band(
        T_band, salinity=5.0, ice_type="MYI",
        T_std=0.5, S_std=0.5, n_samples_per_T=200,
    )
    
    uncertainty_bands = {
        "FYI": {
            "temperatures": fyi_band["temperatures"].tolist(),
            "mean": fyi_band["mean"].tolist(),
            "p05": fyi_band["p05"].tolist(),
            "p50": fyi_band["p50"].tolist(),
            "p95": fyi_band["p95"].tolist(),
        },
        "MYI": {
            "temperatures": myi_band["temperatures"].tolist(),
            "mean": myi_band["mean"].tolist(),
            "p05": myi_band["p05"].tolist(),
            "p50": myi_band["p50"].tolist(),
            "p95": myi_band["p95"].tolist(),
        },
    }
    
    # 5. Sensitivity analysis
    print("5. Computing sensitivity analysis...")
    fyi_sens = sensitivity_analysis(-10.0, 5.0, ice_type="FYI")
    myi_sens = sensitivity_analysis(-10.0, 5.0, ice_type="MYI")
    
    sensitivity = {
        "FYI": {k: float(v) for k, v in fyi_sens.items()},
        "MYI": {k: float(v) for k, v in myi_sens.items()},
    }
    
    # 6. Reference validation points
    print("6. Computing reference validation points...")
    validation_points = {
        "T_minus_10_S_5_FYI": {
            "T_C": -10.0,
            "S_ppt": 5.0,
            "nu_b": float(brine_volume_fraction(-10.0, 5.0)),
            "sigma_c_MPa": float(compressive_strength_fyi(-10.0, 5.0)),
            "nu_b_ref": 0.02725,
            "sigma_c_ref": 18.4,
            "nu_b_rel_error": float(abs(brine_volume_fraction(-10.0, 5.0) - 0.02725) / 0.02725),
            "sigma_c_rel_error": float(abs(compressive_strength_fyi(-10.0, 5.0) - 18.4) / 18.4),
        },
        "T_minus_10_S_5_MYI": {
            "T_C": -10.0,
            "S_ppt": 5.0,
            "f_desal": 0.3,
            "sigma_c_MPa": float(compressive_strength_myi(-10.0, 5.0, f_desal=0.3)),
            "sigma_fyi_MPa": float(compressive_strength_fyi(-10.0, 5.0)),
            "ratio": float(compressive_strength_myi(-10.0, 5.0, f_desal=0.3) / compressive_strength_fyi(-10.0, 5.0)),
        },
    }
    
    # 7. Generate figures
    print("7. Generating visualization figures...")
    os.makedirs("/root/tp-agent/work/m9977/artifacts", exist_ok=True)
    fig_paths = generate_all_figures("/root/tp-agent/work/m9977/artifacts")
    
    # 8. Assemble final results
    results = {
        "metadata": {
            "model": "Ice Material Constitutive Model (MYI/FYI differentiation)",
            "version": "1.0.0",
            "references": [
                "Cox, G. F. N. & Weeks, W. F. (1983). CRREL Report 83-9.",
                "Timco, G. W. & Frederking, R. M. W. (1984). IAHR Ice Symposium.",
                "Frankenstein, G. & Garner, R. (1967). J. Glaciol. 7(50), 273-276.",
                "Timco, G. W. & Weeks, W. F. (2010). Cold Regions Science and Technology, 60(1), 107-129.",
            ],
            "acceptance_criteria": {
                "a_brine_volume": "ν_b at T=-10°C, S=5ppt: <2% vs literature",
                "b_fyi_strength": "σ_c_FYI at T=-10°C, S=5ppt: <10% vs literature",
                "c_myi_strength": "σ_c_MYI > σ_c_FYI (physical consistency)",
                "d_temperature_trend": "σ_c monotonically decreasing with T",
                "e_salinity_trend": "σ_c monotonically decreasing with S",
                "f_negative_controls": "S=0 → ν_b=0; T→0 → σ_c→0",
                "g_engineering": "pytest ≥15 cases; results.json; documentation",
            },
        },
        "validation_points": validation_points,
        "parameter_table": parameter_table,
        "fyi_strength_surface": fyi_surface,
        "myi_strength_surface": myi_surface,
        "uncertainty_bands": uncertainty_bands,
        "sensitivity": sensitivity,
        "figures": fig_paths,
    }
    
    # Save to JSON
    output_path = "/root/tp-agent/work/m9977/artifacts/ice_material_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print(f"✓ Generated {len(fig_paths)} figures")
    print(f"✓ FYI surface: {len(fyi_surface)} points")
    print(f"✓ MYI surface: {len(myi_surface)} points")
    
    # Print validation summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    vp = validation_points["T_minus_10_S_5_FYI"]
    print(f"\nCriterion (a) - Brine volume at T=-10°C, S=5ppt:")
    print(f"  Computed: ν_b = {vp['nu_b']:.6f}")
    print(f"  Reference: ν_b = {vp['nu_b_ref']:.6f}")
    print(f"  Relative error: {vp['nu_b_rel_error']*100:.2f}% (requirement: <2%)")
    print(f"  Status: {'✓ PASS' if vp['nu_b_rel_error'] < 0.02 else '✗ FAIL'}")
    
    print(f"\nCriterion (b) - FYI strength at T=-10°C, S=5ppt:")
    print(f"  Computed: σ_c = {vp['sigma_c_MPa']:.2f} MPa")
    print(f"  Reference: σ_c = {vp['sigma_c_ref']:.2f} MPa")
    print(f"  Relative error: {vp['sigma_c_rel_error']*100:.2f}% (requirement: <10%)")
    print(f"  Status: {'✓ PASS' if vp['sigma_c_rel_error'] < 0.10 else '✗ FAIL'}")
    
    vp_myi = validation_points["T_minus_10_S_5_MYI"]
    print(f"\nCriterion (c) - MYI vs FYI strength at T=-10°C, S=5ppt:")
    print(f"  σ_c_MYI = {vp_myi['sigma_c_MPa']:.2f} MPa")
    print(f"  σ_c_FYI = {vp_myi['sigma_fyi_MPa']:.2f} MPa")
    print(f"  Ratio: {vp_myi['ratio']:.3f}")
    print(f"  Status: {'✓ PASS' if vp_myi['ratio'] > 1.0 else '✗ FAIL'}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
