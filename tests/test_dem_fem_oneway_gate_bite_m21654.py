"""Gate-bite tests for ``examples/dem_fem_oneway/validate.py`` (m21654, verifier).

The review anchor ``GAP-ICE-ONEWAY-CLAIM`` forbids tautological evidence: a gate
must be decidable from the *numbers* in ``results.json``, not from a pass/fail
boolean that the generator wrote next to them.  These tests corrupt a committed
copy of ``results.json`` in every way that matters and require ``validate()`` to
turn red, then require the untouched file to stay green.

Fast: no DEM/FEM simulation, only file I/O.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "dem_fem_oneway"
RESULTS = EXAMPLE / "results.json"
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def validate_mod():
    return _load("dem_fem_validate_bite", EXAMPLE / "validate.py")


@pytest.fixture(scope="module")
def results() -> dict:
    assert RESULTS.exists(), f"{RESULTS} missing: run examples/dem_fem_oneway/run.py"
    return json.loads(RESULTS.read_text())


def _validate_corrupted(validate_mod, results: dict, mutate, tmp_path: Path) -> bool:
    data = copy.deepcopy(results)
    mutate(data)
    path = tmp_path / "results_corrupt.json"
    path.write_text(json.dumps(data))
    return validate_mod.validate(path)


def test_committed_results_pass(validate_mod, results, tmp_path):
    path = tmp_path / "results_copy.json"
    path.write_text(json.dumps(results))
    assert validate_mod.validate(path) is True


@pytest.mark.parametrize("error", [0.0100001, 0.02, 0.2])
def test_deflection_error_above_one_percent_fails(validate_mod, results, tmp_path, error):
    """A 2 % deflection error must not be able to ride a stale ``*_pass`` flag."""
    ok = _validate_corrupted(
        validate_mod, results,
        lambda d: d["verification"].update(
            {"deflection_error_relative": error, "deflection_pass": True}),
        tmp_path)
    assert ok is False


def test_parity_error_above_tolerance_fails(validate_mod, results, tmp_path):
    ok = _validate_corrupted(
        validate_mod, results,
        lambda d: d["verification"].update(
            {"force_history_port_parity_error_relative": 0.5,
             "force_history_port_parity_pass": True}),
        tmp_path)
    assert ok is False


def test_energy_error_above_tolerance_fails(validate_mod, results, tmp_path):
    ok = _validate_corrupted(
        validate_mod, results,
        lambda d: d["verification"].update(
            {"work_energy_error_relative": 1e-2, "energy_pass": True}),
        tmp_path)
    assert ok is False


def test_declared_pass_false_with_good_number_fails(validate_mod, results, tmp_path):
    """The two sources of truth must agree; a disagreement is a FAIL, not a pass."""
    ok = _validate_corrupted(
        validate_mod, results,
        lambda d: d["verification"].update({"deflection_pass": False}),
        tmp_path)
    assert ok is False


def test_fine_level_error_above_floor_fails(validate_mod, results, tmp_path):
    ok = _validate_corrupted(
        validate_mod, results,
        lambda d: d["verification"].__setitem__(
            "dt_convergence_errors_vs_modal", [0.009, 0.002, 0.001, 5e-3]),
        tmp_path)
    assert ok is False


def test_short_chain_fails(validate_mod, results, tmp_path):
    ok = _validate_corrupted(
        validate_mod, results,
        lambda d: d["verification"].__setitem__(
            "dt_convergence_errors_vs_modal",
            d["verification"]["dt_convergence_errors_vs_modal"][:3]),
        tmp_path)
    assert ok is False


def test_scope_tag_removal_fails(validate_mod, results, tmp_path):
    ok = _validate_corrupted(validate_mod, results, lambda d: d.pop("scope"), tmp_path)
    assert ok is False


def test_interface_row_reverting_to_impulse_fails(validate_mod, results, tmp_path):
    ok = _validate_corrupted(
        validate_mod, results,
        lambda d: d["ledger"]["interface"].update({"impulse_error_relative": 0.0}),
        tmp_path)
    assert ok is False


def test_missing_inline_ledger_fails(validate_mod, results, tmp_path):
    """Without the inlined ledger (and no artifacts/), validation must not pass."""
    def drop(d):
        d.pop("ledger")
    data = copy.deepcopy(results)
    drop(data)
    path = tmp_path / "no_ledger.json"
    path.write_text(json.dumps(data))
    # point the artifacts fallback at a non-existent file
    missing = tmp_path / "artifacts" / "dem_fem_coupling_ledger.json"
    assert validate_mod.validate(path, ledger_path=missing) is False


# --------------------------------------------------- run.py docstring hygiene
def test_run_docstring_does_not_advertise_a_ratio_gate():
    """The retired ``ratio >= 0.9`` dt gate must not be re-advertised in the docstring.

    Two adjacent refinements give a peak ratio of ~1.000025, so ``ratio >= 0.9``
    is true by construction -- exactly the kind of tautology the anchor bans.
    """
    run_mod = _load("dem_fem_run_docstring", EXAMPLE / "run.py")
    doc = run_mod.__doc__ or ""
    assert "ratio ≥0.9" not in doc
    assert "ratio >= 0.9" not in doc
    assert "diagnostic only" in doc
    assert "continuum modal anchor" in doc