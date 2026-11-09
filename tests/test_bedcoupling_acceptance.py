"""End-to-end acceptance test for the bed-coupling node (reduced size).

The production run lived in ``artifacts/IN143-N09`` with 1500 particles and
2000 settling steps; this test uses a smaller packing (800 particles) and short
settling runs so the suite stays inside a normal CI budget while still
evaluating all six machine-checkable criteria.
"""

from __future__ import annotations

import pytest

from tensordem.bedcoupling.study import run_acceptance_study


@pytest.fixture(scope="module")
def results() -> dict:
    return run_acceptance_study(n_particles=800, n_steps=400, include_sweeps=False)


def test_all_six_criteria_pass(results: dict) -> None:
    failed = {k: c for k, c in results["criteria"].items() if not c["passed"]}
    assert not failed, failed
    assert results["all_criteria_passed"]


def test_packing_criterion_numbers(results: dict) -> None:
    sec = results["sections"]["packing"]
    stats = sec["porosity_statistics"]
    assert 0.36 <= stats["mean"] <= 0.40
    assert stats["std"] <= 0.02
    assert sec["packing"]["max_overlap_over_diameter"] <= 1e-3


def test_ergun_criteria_cover_both_regimes(results: dict) -> None:
    cases = results["sections"]["ergun"]["cases"]
    regimes = {c["regime"] for c in cases}
    assert regimes == {"laminar", "transition"}
    for c in cases:
        assert abs(c["rel_error"]) <= 0.15
        assert c["ergun_dpdz"] > 0.0


def test_terminal_velocity_criterion(results: dict) -> None:
    sec = results["sections"]["terminal_velocity"]
    assert abs(sec["rel_error"]) <= 0.05
    assert 1.0 < sec["re_terminal"] < 500.0  # transition regime, not Stokes


def test_rz_criterion_and_variants(results: dict) -> None:
    sec = results["sections"]["rz_settling"]
    assert abs(sec["acceptance"]["deviation"]) <= 0.3
    # the hydrostatic-buoyancy variant is reported, not hidden
    assert sec["variants"]["hydrostatic"]["deviation"] < -0.5


def test_static_residual_criterion(results: dict) -> None:
    sec = results["sections"]["static_bed"]
    assert sec["residual_over_total_gravity"] <= 1e-6
    assert sec["max_per_particle_residual_over_weight"] <= 1e-6


def test_timestep_criterion(results: dict) -> None:
    sec = results["sections"]["timestep"]
    assert sec["rel_change"] <= 0.03
    assert len(sec["history_dt"]["time"]) == len(sec["history_dt"]["dpdz"])
    assert len(sec["history_dt_half"]["time"]) == 2 * (len(sec["history_dt"]["time"]) - 1) + 1


def test_results_are_json_serialisable(results: dict, tmp_path) -> None:
    out = tmp_path / "results.json"
    from tensordem.bedcoupling.study import write_results

    write_results(results, out)
    text = out.read_text()
    assert "all_criteria_passed" in text
    assert results["node"] == "IN143-N09"
