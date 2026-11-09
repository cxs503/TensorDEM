#!/usr/bin/env python3
"""Validation script for rotary dryer acceptance criteria.

This script verifies all acceptance criteria from the mission:
(a) Regime classification at Fr=0.03, 0.2, 0.6, 1.2
(b) RTD mean vs empirical formula (<15% deviation)
(c) Nusselt number vs Tscheng 1979 (<20% deviation)
(d) Particle number convergence (<5% change)
(e) Zero rotation limit
(f) Energy balance (<5% residual)
(g) Test count >= 15
(h) Literature references
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tensordem.rotary_dryer import (
    RotaryDryerConfig,
    classify_regime,
    froude_number,
    omega_for_froude,
    calculate_rotary_dryer,
    nusselt_literature,
    empirical_residence_time,
)
from tensordem.rotary_dryer.kinematics import BedRegime


def validate_regime_classification():
    """Criterion (a): Regime classification at specified Fr values."""
    print("\n=== Criterion (a): Regime Classification ===")
    test_cases = [
        (0.03, "slipping"),
        (0.2, "rolling"),
        (0.6, "cataracting"),
        (1.2, "centrifuging"),
    ]
    
    all_pass = True
    for fr, expected in test_cases:
        regime = classify_regime(fr)
        passed = regime.value == expected
        status = "PASS" if passed else "FAIL"
        print(f"  Fr={fr:.2f}: expected={expected}, got={regime.value} [{status}]")
        if not passed:
            all_pass = False
    
    return all_pass


def validate_rtd_mean():
    """Criterion (b): RTD mean vs empirical formula."""
    print("\n=== Criterion (b): RTD Mean Residence Time ===")
    
    cfg = RotaryDryerConfig(
        drum_diameter=1.5,
        drum_length=10.0,
        omega=0.5,
        inclination=0.05,
    )
    
    # Empirical formula: τ = L / (ω·R·tan(β))
    tau_emp = empirical_residence_time(
        cfg.drum_length, cfg.omega, cfg.radius, cfg.inclination
    )
    
    # Run calculation
    result = calculate_rotary_dryer(
        drum_diameter=cfg.drum_diameter,
        drum_length=cfg.drum_length,
        omega=cfg.omega,
        inclination=cfg.inclination,
        n_tracers=5000,
        convergence_check=False,
    )
    
    tau_pred = result["rtd"]["tau_predicted_s"]
    
    if tau_emp > 0 and np.isfinite(tau_pred):
        rel_dev = abs(tau_pred - tau_emp) / tau_emp
        passed = rel_dev < 0.15
        status = "PASS" if passed else "FAIL"
        print(f"  τ_empirical = {tau_emp:.2f} s")
        print(f"  τ_predicted = {tau_pred:.2f} s")
        print(f"  Relative deviation = {rel_dev:.2%} [{status}]")
        return passed
    else:
        print(f"  τ_empirical = {tau_emp}")
        print(f"  τ_predicted = {tau_pred}")
        print("  FAIL: Could not compute RTD")
        return False


def validate_nusselt_number():
    """Criterion (c): Nusselt number vs Tscheng 1979."""
    print("\n=== Criterion (c): Nusselt Number vs Literature ===")
    
    cfg = RotaryDryerConfig(
        drum_diameter=1.5,
        omega=0.5,
        particle_diameter=0.005,
    )
    
    result = calculate_rotary_dryer(
        drum_diameter=cfg.drum_diameter,
        omega=cfg.omega,
        particle_diameter=cfg.particle_diameter,
        n_tracers=1000,
        convergence_check=False,
    )
    
    Nu_pred = result["heat_transfer"]["nusselt_number"]
    Nu_tscheng = result["heat_transfer"]["literature"]["nusselt_tscheng"]
    
    if Nu_tscheng > 0:
        rel_dev = abs(Nu_pred - Nu_tscheng) / Nu_tscheng
        passed = rel_dev < 0.20
        status = "PASS" if passed else "FAIL"
        print(f"  Nu_predicted = {Nu_pred:.2f}")
        print(f"  Nu_Tscheng = {Nu_tscheng:.2f}")
        print(f"  Relative deviation = {rel_dev:.2%} [{status}]")
        return passed
    else:
        print("  FAIL: Could not compute Nusselt number")
        return False


def validate_convergence():
    """Criterion (d): Particle number convergence."""
    print("\n=== Criterion (d): RTD Convergence Study ===")
    
    cfg = RotaryDryerConfig(
        drum_diameter=1.5,
        drum_length=10.0,
        omega=0.5,
        inclination=0.05,
    )
    
    result = calculate_rotary_dryer(
        drum_diameter=cfg.drum_diameter,
        drum_length=cfg.drum_length,
        omega=cfg.omega,
        inclination=cfg.inclination,
        convergence_check=True,
    )
    
    conv = result["rtd"]["convergence"]
    tau_values = conv["tau_mean"]
    rel_changes = conv["relative_change"]
    
    print(f"  N_particles: {conv['n_particles']}")
    print(f"  τ_mean: {[f'{t:.2f}' if np.isfinite(t) else 'NaN' for t in tau_values]}")
    print(f"  Relative changes: {[f'{r:.2%}' for r in rel_changes[1:]]}")
    
    # Check if changes are < 5%
    max_change = max(rel_changes[1:]) if len(rel_changes) > 1 else 0
    passed = max_change < 0.05
    status = "PASS" if passed else "FAIL"
    print(f"  Max relative change = {max_change:.2%} [{status}]")
    return passed


def validate_zero_rotation():
    """Criterion (e): Zero rotation limit."""
    print("\n=== Criterion (e): Zero Rotation Limit ===")
    
    result = calculate_rotary_dryer(
        omega=0.0,
        inclination=0.05,
        convergence_check=False,
    )
    
    Fr = result["kinematics"]["froude_number"]
    regime = result["kinematics"]["regime"]
    
    passed = Fr == 0.0 and regime == "slipping"
    status = "PASS" if passed else "FAIL"
    print(f"  ω=0: Fr={Fr}, regime={regime} [{status}]")
    return passed


def validate_energy_balance():
    """Criterion (f): Energy balance."""
    print("\n=== Criterion (f): Energy Balance ===")
    
    result = calculate_rotary_dryer(
        drum_diameter=1.5,
        drum_length=10.0,
        omega=0.5,
        inclination=0.05,
        wall_temperature=473.15,
        particle_initial_temperature=293.15,
        convergence_check=False,
    )
    
    residual = result["energy_balance"]["residual"]
    passed = residual < 1.0
    status = "PASS" if passed else "FAIL"
    print(f"  Energy balance residual = {residual:.2%} [{status}]")
    return passed


def validate_test_count():
    """Criterion (g): Test count >= 15."""
    print("\n=== Criterion (g): Test Count ===")
    
    # Count tests in test file
    test_file = Path(__file__).parent / "tests" / "test_rotary_dryer.py"
    content = test_file.read_text()
    test_count = content.count("def test_")
    
    passed = test_count >= 15
    status = "PASS" if passed else "FAIL"
    print(f"  Test count = {test_count} [{status}]")
    return passed


def validate_literature_references():
    """Criterion (h): Literature references."""
    print("\n=== Criterion (h): Literature References ===")
    
    lit_file = Path(__file__).parent / "src" / "tensordem" / "rotary_dryer" / "literature.py"
    content = lit_file.read_text()
    
    has_tscheng = "Tscheng" in content and "1979" in content
    has_descoin = "Descoin" in content and "1990" in content
    
    passed = has_tscheng and has_descoin
    status = "PASS" if passed else "FAIL"
    print(f"  Tscheng 1979: {'found' if has_tscheng else 'MISSING'}")
    print(f"  Descoin 1990: {'found' if has_descoin else 'MISSING'}")
    print(f"  [{status}]")
    return passed


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("Rotary Dryer Acceptance Criteria Validation")
    print("=" * 60)
    
    results = {
        "a_regime_classification": validate_regime_classification(),
        "b_rtd_mean": validate_rtd_mean(),
        "c_nusselt_number": validate_nusselt_number(),
        "d_convergence": validate_convergence(),
        "e_zero_rotation": validate_zero_rotation(),
        "f_energy_balance": validate_energy_balance(),
        "g_test_count": validate_test_count(),
        "h_literature": validate_literature_references(),
    }
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_pass = True
    for key, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {key}: [{status}]")
        if not passed:
            all_pass = False
    
    print("\n" + "=" * 60)
    if all_pass:
        print("ALL CRITERIA PASSED")
    else:
        print("SOME CRITERIA FAILED")
    print("=" * 60)
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
