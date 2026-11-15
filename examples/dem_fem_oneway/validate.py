"""Validation script for DEM-FEM one-way coupling example.

Runs the acceptance checks on the artifacts written by ``run.py`` and prints a
pass/fail table.  Every gate is an *observable* statement about the numbers; the
tautological ones are labelled ``numerical_verification_only`` (see
``docs/DEM_FEM_ONEWAY_CLAIM_REVIEW_m21584.md``).

The coupling ledger is read from ``results["ledger"]`` (inlined by ``run.py``)
and only falls back to ``artifacts/dem_fem_coupling_ledger.json`` when the
inline copy is absent, so this script works in a clean checkout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

LEDGER_KEYS = [
    "ledger_type", "dem_side", "fem_side", "modal_side", "interface",
    "energy_balance", "deflection_comparison", "dt_convergence", "separability",
]

# dt/T1 of the four refinement levels used by run.py (dt, dt/2, dt/4, dt/8).
DT_OVER_T1_EXPECTED = [0.1399, 0.0699, 0.0350, 0.0175]


def load_ledger(results: dict, ledger_path: Path | None) -> tuple[dict, str]:
    """Return (ledger, source).  Inline ledger wins over the artifacts dump."""
    if isinstance(results.get("ledger"), dict):
        return results["ledger"], "results.json:ledger"
    if ledger_path is not None and Path(ledger_path).exists():
        with open(ledger_path) as f:
            return json.load(f), str(ledger_path)
    raise FileNotFoundError(
        "no coupling ledger: results.json has no inline 'ledger' and "
        f"{ledger_path} is missing"
    )


def validate(results_path: Path | str | None = None,
             ledger_path: Path | str | None = None) -> bool:
    here = Path(__file__).resolve().parent
    results_path = Path(results_path) if results_path is not None else here / "results.json"
    ledger_path = (Path(ledger_path) if ledger_path is not None
                   else here.parent.parent / "artifacts" / "dem_fem_coupling_ledger.json")

    if not results_path.exists():
        print(f"FAIL: {results_path} not found. Run run.py first.")
        return False
    try:
        with open(results_path) as f:
            results = json.load(f)
        ledger, ledger_source = load_ledger(results, ledger_path)
    except FileNotFoundError as exc:
        print(f"FAIL: {exc}")
        return False

    v = results["verification"]
    checks = []

    # 1. Force history port parity (renamed from "impulse"); algebra check only.
    checks.append(("Force history port parity <=1e-10 [numerical_verification_only]",
                   v["force_history_port_parity_pass"],
                   v["force_history_port_parity_error_relative"]))

    # 2. Deflection: dynamic FEM vs continuum modal superposition <=1%
    checks.append(("Deflection dynamic vs modal <=1%", v["deflection_pass"],
                   v["deflection_error_relative"]))

    # 3. Energy (work-energy balance)
    checks.append(("Work-Energy <=1e-8", v["energy_pass"],
                   v["work_energy_error_relative"]))

    # 4. dt chain: 4 levels, per-level error against the independent modal anchor.
    chain = v.get("dt_convergence_errors_vs_modal", [])
    chain_dt = [r.get("dt_over_t1") for r in results.get("dt_convergence", [])]
    chain_len_ok = len(chain) == 4
    chain_dt_ok = (len(chain_dt) == 4 and all(
        abs(a - b) <= 5e-3 for a, b in zip(chain_dt, DT_OVER_T1_EXPECTED)))
    chain_ok = bool(chain_len_ok and chain_dt_ok and v.get("dt_convergence_pass", False))
    checks.append(("dt chain 4 levels @ dt/T1=0.14/0.07/0.035/0.0175", chain_ok,
                   "/".join(f"{100 * e:.4f}%" for e in chain)))
    checks.append(("dt chain per-level errors (<=1% @dt, <=0.1% @dt/8)",
                   bool(chain and chain[0] <= 0.01 and chain[-1] <= 1e-3),
                   (f"{100 * chain[0]:.4f}%/{100 * chain[-1]:.4f}%" if chain else "N/A")))

    # 5. Separability
    checks.append(("Separability DEM", v["separability_dem"], ""))
    checks.append(("Separability FEM", v["separability_fem"],
                   v.get("separability_fem_deflection_error", "N/A")))

    # 6. Ledger completeness (inline or artifacts copy)
    ledger_ok = all(k in ledger for k in LEDGER_KEYS)
    checks.append((f"Ledger completeness ({ledger_source})", ledger_ok,
                   ",".join(k for k in LEDGER_KEYS if k not in ledger) or "all keys"))

    # 7. Scope field present, and the interface row carries the same scope tag
    scope_ok = (results.get("scope") == "numerical_verification_only"
                and ledger.get("interface", {}).get("scope") == "numerical_verification_only")
    checks.append(("scope=numerical_verification_only (results + ledger.interface)",
                   scope_ok, results.get("scope", "MISSING")))

    # 8. Interface row must expose the port-parity name, not a bare "impulse_pass"
    iface = ledger.get("interface", {})
    naming_ok = ("force_history_port_parity_error_relative" in iface
                 and "impulse_error_relative" not in iface)
    checks.append(("interface row named force_history_port_parity", naming_ok,
                   "impulse_error_relative" in iface or "renamed"))

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