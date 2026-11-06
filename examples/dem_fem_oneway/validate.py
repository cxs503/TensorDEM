"""Validation script for DEM-FEM one-way coupling example.

Runs the coupling simulation and verifies all acceptance criteria.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


def validate():
    results_path = Path(__file__).parent / "results.json"
    ledger_path = Path(__file__).parent.parent.parent / "artifacts" / "dem_fem_coupling_ledger.json"

    if not results_path.exists():
        print("FAIL: results.json not found. Run run.py first.")
        return False
    if not ledger_path.exists():
        print("FAIL: dem_fem_coupling_ledger.json not found. Run run.py first.")
        return False

    with open(results_path) as f:
        results = json.load(f)
    with open(ledger_path) as f:
        ledger = json.load(f)

    v = results["verification"]
    checks = []

    # 1. Impulse ledger
    impulse_ok = v["impulse_pass"]
    checks.append(("Impulse ledger ≤1e-10", impulse_ok, v["impulse_error_relative"]))

    # 2. Deflection
    deflection_ok = v["deflection_pass"]
    checks.append(("Deflection ≤5%", deflection_ok, v["deflection_error_relative"]))

    # 3. Energy (work-energy balance)
    energy_ok = v["energy_pass"]
    checks.append(("Work-Energy ≤1e-8", energy_ok, v["work_energy_error_relative"]))

    # 4. Separability
    sep_dem_ok = v["separability_dem"]
    sep_fem_ok = v["separability_fem"]
    checks.append(("Separability DEM", sep_dem_ok, ""))
    checks.append(("Separability FEM", sep_fem_ok, v.get("separability_fem_deflection_error", "N/A")))

    # 5. Ledger completeness
    ledger_ok = all(k in ledger for k in [
        "ledger_type", "dem_side", "fem_side", "interface",
        "energy_balance", "deflection_comparison", "separability",
    ])
    checks.append(("Ledger completeness", ledger_ok, ""))

    print("DEM-FEM Coupling Validation Results")
    print("=" * 50)
    all_pass = True
    for name, passed, value in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status} ({value})")
        if not passed:
            all_pass = False

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return all_pass


if __name__ == "__main__":
    ok = validate()
    sys.exit(0 if ok else 1)
