"""Claim-hygiene tests for the DEM-FEM one-way coupling example.

Locks the five items of the review anchor ``GAP-ICE-ONEWAY-CLAIM``:

1. the former "impulse" gate is reported as ``force_history_port_parity`` with
   ``scope=numerical_verification_only`` (it is an algebra check, not interface
   conservation);
2. the deflection gate is "dynamic FEM vs continuum modal superposition <= 1%"
   and is a real two-implementation comparison (error neither 0 nor above 1%);
3. a four-level ``dt`` refinement chain exists, each level carrying its own
   modal-anchor error, and the shipped gate is built from those errors instead
   of the ratio of two near-identical peaks;
4. the ``CouplingConfig`` docstring names ``platen_stiffness = 2.0e4 N/m`` as the
   contact stiffness (the old "k = 200 N/m" claim was wrong);
5. ``validate.py`` passes from a clean checkout, i.e. without
   ``artifacts/dem_fem_coupling_ledger.json``, because the ledger is inlined in
   ``results.json``.

These tests read committed artifacts only (no DEM/FEM simulation), so they run
in well under a second.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "dem_fem_oneway"
RESULTS = EXAMPLE / "results.json"

sys.path.insert(0, str(ROOT / "src"))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def results() -> dict:
    assert RESULTS.exists(), f"{RESULTS} missing: run examples/dem_fem_oneway/run.py"
    with open(RESULTS) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def validate_mod():
    return _load_module("dem_fem_validate", EXAMPLE / "validate.py")


# ---------------------------------------------------------------- item 1
def test_force_history_port_parity_renamed_and_scoped(results):
    v = results["verification"]
    assert "force_history_port_parity_error_relative" in v
    assert "force_history_port_parity_pass" in v
    assert v["force_history_port_parity_pass"] is True
    # the old bare impulse gate must not survive as a pass/fail key
    assert "impulse_pass" not in v
    assert "impulse_error_relative" not in v
    assert results["scope"] == "numerical_verification_only"


def test_ledger_interface_row_is_port_parity_not_conservation(results):
    iface = results["ledger"]["interface"]
    assert iface["scope"] == "numerical_verification_only"
    assert "force_history_port_parity_error_relative" in iface
    assert "impulse_error_relative" not in iface
    assert "algebra check" in iface["note"]
    # parity is exact by construction (same force array, same trapezoid rule)
    assert iface["force_history_port_parity_error_relative"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- item 2
def test_deflection_gate_is_dynamic_vs_modal_within_one_percent(results):
    v = results["verification"]
    assert v["deflection_pass"] is True
    err = v["deflection_error_relative"]
    assert 0.0 < err <= 0.01, err          # a real comparison, inside the 1% band
    assert v["deflection_dynamic_vs_modal_m"] == pytest.approx(
        results["modal"]["peak_deflection_m"], rel=1e-12)
    fem_peak = results["fem"]["peak_deflection_m"]
    assert err == pytest.approx(
        abs(fem_peak - results["modal"]["peak_deflection_m"])
        / results["modal"]["peak_deflection_m"], rel=1e-6)


# ---------------------------------------------------------------- item 3
def test_dt_chain_has_four_levels_with_anchor_errors(results):
    chain = results["dt_convergence"]
    assert len(chain) == 4
    assert [r["refinement"] for r in chain] == [1, 2, 4, 8]
    ratios = [r["dt_over_t1"] for r in chain]
    assert ratios == pytest.approx([0.1399, 0.0699, 0.0350, 0.0175], abs=5e-3)
    # each level carries its own independent-anchor error
    errors = [r["deflection_error_vs_modal_relative"] for r in chain]
    for r, e in zip(chain, errors):
        assert e == pytest.approx(
            abs(r["peak_deflection_m"] - r["modal_anchor_peak_m"]) / r["modal_anchor_peak_m"],
            rel=1e-6)
    assert errors[0] <= 0.01           # coarse level inside 1%
    assert errors[-1] <= 1e-3          # finest level inside 0.1%
    assert errors[1] < errors[0] and errors[-1] < errors[1]


def test_dt_chain_gate_is_not_the_tautological_peak_ratio(results):
    v = results["verification"]
    assert v["dt_convergence_pass"] is True
    assert "dt_convergence_ratio" not in v
    assert v["dt_convergence_peak_ratio_is_diagnostic_only"] is True
    assert "NOT a ratio" in v["dt_convergence_criterion"]
    assert v["dt_convergence_errors_vs_modal"] == pytest.approx(
        [r["deflection_error_vs_modal_relative"] for r in results["dt_convergence"]], rel=1e-12)
    # the diagnostic ratio of two adjacent refinements is ~1 (i.e. useless as a gate)
    assert abs(v["dt_convergence_peak_ratio_finest_over_second_finest"] - 1.0) < 1e-3


# ---------------------------------------------------------------- item 4
def test_docstring_cites_platen_stiffness_not_k_equals_200():
    src = (EXAMPLE / "run.py").read_text()
    head = src.split('"""', 2)[2]  # CouplingConfig docstring body
    doc_cfg = src[src.index("class CouplingConfig"): src.index("class CouplingConfig") + 1600]
    assert "k=200 N/m" not in head
    assert "platen_stiffness = 2.0e4" in doc_cfg
    assert "contact_stiffness" in doc_cfg  # explicit about the inert particle law
    assert "2.0e2 N/m" in doc_cfg


# ---------------------------------------------------------------- item 5
def test_validate_passes_without_artifacts_ledger(validate_mod, monkeypatch, capsys):
    """Clean checkout: no artifacts/dem_fem_coupling_ledger.json on disk."""
    missing = ROOT / "artifacts" / "__no_such_ledger_m21584.json"
    assert not missing.exists()
    assert "ledger" in json.loads(RESULTS.read_text())
    ok = validate_mod.validate(results_path=RESULTS, ledger_path=missing)
    out = capsys.readouterr().out
    assert ok, out
    assert "SOME FAILURES" not in out
    assert "results.json:ledger" in out


def test_validate_fails_loudly_when_no_ledger_anywhere(validate_mod):
    with open(RESULTS) as f:
        results = json.load(f)
    results.pop("ledger")
    tmp = RESULTS.parent / "__tmp_results_no_ledger.json"
    try:
        tmp.write_text(json.dumps(results))
        missing = ROOT / "artifacts" / "__no_such_ledger_m21584.json"
        assert validate_mod.validate(results_path=tmp, ledger_path=missing) is False
    finally:
        tmp.unlink(missing_ok=True)