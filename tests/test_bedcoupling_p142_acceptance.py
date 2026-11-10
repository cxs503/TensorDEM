"""End-to-end acceptance test for the P142W3-N03 fluidised-bed node (reduced size).

The production run lives in ``artifacts/P142W3-N03`` (1200-particle bed, 3000
terminal-velocity steps, 1500 RZ/ledger steps, three resolutions up to 2400
particles).  This test uses shorter runs so the suite stays inside a normal CI
budget while still evaluating every criterion, and it imports the convergence
criterion from the dedicated convergence test instead of repeating it.
"""

from __future__ import annotations

import pytest

from tensordem.bedcoupling.study_p142 import run_acceptance_study


@pytest.fixture(scope="module")
def results() -> dict:
    return run_acceptance_study(
        n_particles=600,
        terminal_steps=800,
        rz_steps=300,
        ledger_steps=300,
        include_convergence=False,
    )


def test_all_criteria_pass(results: dict) -> None:
    failed = {k: c for k, c in results["criteria"].items() if not c["passed"]}
    assert not failed, failed
    assert results["all_criteria_passed"]


def test_ergun_error_numbers(results: dict) -> None:
    sec = results["sections"]["ergun"]
    assert abs(sec["worst_abs_error_local_voidage"]) <= 0.10
    assert len(sec["rows"]) == 13
    assert sec["rows"][0]["re_superficial"] == pytest.approx(1.0, rel=1e-9)
    assert sec["rows"][-1]["re_superficial"] == pytest.approx(1000.0, rel=1e-9)


def test_terminal_velocity_regimes(results: dict) -> None:
    sec = results["sections"]["terminal_velocity"]
    assert sec["stokes"]["re_terminal"] < 0.01
    assert abs(sec["stokes"]["rel_error"]) <= 0.05
    assert sec["transition"]["re_terminal"] > 10.0
    assert abs(sec["transition"]["rel_error"]) <= 0.10


def test_rz_curve_and_negative_control(results: dict) -> None:
    sec = results["sections"]["rz_curve"]
    assert abs(sec["worst_abs_error_mixture"]) <= 0.10
    assert sec["worst_abs_error_hydrostatic"] > 0.10  # N1 must fail
    assert abs(sec["fit_exponent_deviation"]) < 0.2


def test_ledger_and_negative_controls(results: dict) -> None:
    sec = results["sections"]["ledger"]
    assert sec["finite"]
    assert sec["bookkeeping_residual"] <= 0.01
    assert sec["equilibrium_residual"] <= 0.01
    assert sec["negative_controls"]["dropped_drag_bookkeeping"] > 10.0
    assert sec["negative_controls"]["timestep_halving"]["ratio"] < 0.7


def test_fluidization_onset(results: dict) -> None:
    sec = results["sections"]["fluidization"]
    assert abs(sec["rel_error_vs_wen_yu"]) <= 0.10
    assert sec["consistency_max_vs_ergun"] < 1e-9
    assert 1.0 < sec["simulated_local"]["re_mf"] < 100.0
