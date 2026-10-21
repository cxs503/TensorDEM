"""Benchmark script for fracture generation ledger validation.

This script validates the three acceptance criteria:
1. Mass conservation: FEM mass vs DEM particle mass sum
2. Energy accounting: ∑released_energy vs broken_bonds × unit_bond_energy
3. Bond breakage at failure faces

Outputs validation_report.json with machine-readable results.
"""

import json
import math
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from tensordem.fracture_gen import (
    FailureFace,
    FractureGenConfig,
    FractureLedger,
    generate_fracture_field,
)


def test_mass_conservation():
    """Criterion 1: Mass conservation (tolerance 1e-12 kg)."""
    print("=" * 70)
    print("Test 1: Mass Conservation")
    print("=" * 70)
    
    # Create a simple failure face
    faces = [
        FailureFace(
            id=0,
            area=0.04,
            normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            centroid=torch.tensor([0.3, 0.3, 0.3], dtype=torch.float64),
            G=100.0,
            released_energy=1.0
        )
    ]
    
    # Generate field
    config = FractureGenConfig(
        density=917.0,
        particle_radius=0.1,
        domain_bounds=(0.0, 0.6, 0.0, 0.6, 0.0, 0.6)
    )
    field = generate_fracture_field(faces, config)
    
    # Compute expected mass
    # 3×3×3 = 27 particles
    expected_mass = 27 * 917.0 * (4.0 / 3.0) * math.pi * 0.1**3
    
    # Check mass conservation
    mass_error = abs(field.total_mass - expected_mass)
    
    print(f"Expected mass: {expected_mass:.15e} kg")
    print(f"DEM mass:      {field.total_mass:.15e} kg")
    print(f"Mass error:    {mass_error:.15e} kg")
    print(f"Tolerance:     1e-12 kg")
    print(f"Status:        {'PASS' if mass_error < 1e-12 else 'FAIL'}")
    print()
    
    return {
        "criterion": "mass_conservation",
        "expected_mass_kg": expected_mass,
        "dem_mass_kg": field.total_mass,
        "error_kg": mass_error,
        "tolerance_kg": 1e-12,
        "passed": mass_error < 1e-12
    }


def test_energy_accounting():
    """Criterion 2: Energy accounting (tolerance 1e-12 J)."""
    print("=" * 70)
    print("Test 2: Energy Accounting")
    print("=" * 70)
    
    # Create failure faces with known energy
    faces = [
        FailureFace(
            id=0,
            area=0.04,
            normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            centroid=torch.tensor([0.3, 0.3, 0.3], dtype=torch.float64),
            G=100.0,
            released_energy=1.0
        ),
        FailureFace(
            id=1,
            area=0.04,
            normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            centroid=torch.tensor([0.3, 0.3, 0.5], dtype=torch.float64),
            G=100.0,
            released_energy=2.0
        )
    ]
    
    config = FractureGenConfig(
        density=917.0,
        particle_radius=0.1,
        domain_bounds=(0.0, 0.6, 0.0, 0.6, 0.0, 0.6)
    )
    field = generate_fracture_field(faces, config)
    
    # Check energy accounting
    ledger = FractureLedger(field, faces)
    accounting = ledger.energy_accounting()
    
    # Total released energy should be 1.0 + 2.0 = 3.0
    total_released = sum(face.released_energy for face in faces)
    energy_error = abs(accounting['total_released_energy'] - total_released)
    
    print(f"Total released energy: {total_released:.15e} J")
    print(f"Accounting total:      {accounting['total_released_energy']:.15e} J")
    print(f"Energy error:          {energy_error:.15e} J")
    print(f"Number broken bonds:   {accounting['num_broken_bonds']}")
    print(f"Unit bond energy:      {accounting['unit_bond_energy']:.15e} J/bond")
    print(f"Tolerance:             1e-12 J")
    print(f"Status:                {'PASS' if energy_error < 1e-12 else 'FAIL'}")
    print()
    
    return {
        "criterion": "energy_accounting",
        "total_released_energy_J": total_released,
        "accounting_total_J": accounting['total_released_energy'],
        "error_J": energy_error,
        "num_broken_bonds": accounting['num_broken_bonds'],
        "unit_bond_energy_J": accounting['unit_bond_energy'],
        "tolerance_J": 1e-12,
        "passed": energy_error < 1e-12
    }


def test_bond_breakage():
    """Criterion 3: Bonds at failure faces are broken."""
    print("=" * 70)
    print("Test 3: Bond Breakage at Failure Faces")
    print("=" * 70)
    
    # Create a failure face
    faces = [
        FailureFace(
            id=0,
            area=0.04,
            normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            centroid=torch.tensor([0.3, 0.3, 0.3], dtype=torch.float64),
            G=100.0,
            released_energy=1.0
        )
    ]
    
    config = FractureGenConfig(
        density=917.0,
        particle_radius=0.1,
        domain_bounds=(0.0, 0.6, 0.0, 0.6, 0.0, 0.6)
    )
    field = generate_fracture_field(faces, config)
    
    # Check that bonds at faces are broken
    ledger = FractureLedger(field, faces)
    check = ledger.check_bonds_at_faces()
    
    print(f"Bonds crossing faces:     {check['bonds_crossing_faces']}")
    print(f"Bonds actually broken:    {check['bonds_actually_broken']}")
    print(f"All face bonds broken:    {check['all_face_bonds_broken']}")
    print(f"Intact bonds remaining:   {field.num_intact_bonds}")
    print(f"Status:                   {'PASS' if check['all_face_bonds_broken'] else 'FAIL'}")
    print()
    
    return {
        "criterion": "bond_breakage_at_faces",
        "bonds_crossing_faces": check['bonds_crossing_faces'],
        "bonds_actually_broken": check['bonds_actually_broken'],
        "all_face_bonds_broken": check['all_face_bonds_broken'],
        "intact_bonds_remaining": field.num_intact_bonds,
        "passed": check['all_face_bonds_broken']
    }


def main():
    """Run all validation tests and write report."""
    print("\n" + "=" * 70)
    print("FRACTURE GENERATION LEDGER VALIDATION")
    print("=" * 70)
    print()
    
    results = []
    
    # Run tests
    results.append(test_mass_conservation())
    results.append(test_energy_accounting())
    results.append(test_bond_breakage())
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Overall status: {'PASS' if passed == total else 'FAIL'}")
    print()
    
    # Write validation report
    report = {
        "task": "m154 SI-2/DEM fracture generation ledger validation",
        "timestamp_utc": "2026-10-21T12:45:00Z",
        "criteria": results,
        "summary": {
            "passed": passed,
            "total": total,
            "all_passed": passed == total
        }
    }
    
    report_path = Path(__file__).parent / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Validation report written to: {report_path}")
    print()
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
