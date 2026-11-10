"""Acceptance evidence: C1-C6 in one machine-checked payload, with its digest.

The report artifact is only worth anything if a third party can (a) regenerate
it and (b) detect a hand edit.  Both are tested here:

* :func:`tensordem.ice_friction_thermo.acceptance_report` is regenerated and all
  six criteria are asserted to pass, with the numbers reported in the assertion
  messages (so a regression prints the number that moved);
* the JSON written by :func:`write_evidence` is read back, its SHA-256 is
  recomputed from a fresh run, and a deliberate edit of one number is shown to
  be detected.
"""
from __future__ import annotations

import json

import pytest

from tensordem.ice_friction_thermo import (
    ERROR_GATE,
    acceptance_report,
    calibrate_from_barnes_sweep,
    canonical_digest,
    verify_evidence,
    write_evidence,
)


@pytest.fixture(scope="module")
def report() -> dict:
    return acceptance_report()


def test_all_six_criteria_pass(report):
    criteria = report["criteria"]
    assert set(criteria) == {
        "C1_contact_state_outputs",
        "C2_mu_decreases_with_temperature",
        "C3_fixed_point_convergence",
        "C4_validation_against_literature",
        "C5_friction_heat_identity",
    } | {"C5_friction_heat_identity"}
    for name, block in criteria.items():
        assert block["pass"], f"{name} failed: {block}"
    assert report["C6_provenance"]["pass"]


def test_c1_and_c5_numbers_are_physical(report):
    c1 = report["criteria"]["C1_contact_state_outputs"]
    assert c1["sliding_case_final_temperature_c"] < 0.0
    assert c1["sliding_case_final_heat_w"] > 0.0
    c5 = report["criteria"]["C5_friction_heat_identity"]
    assert c5["max_rel_error"] < 1.0e-9        # an algebraic identity, not a fit


def test_c2_is_three_out_of_three(report):
    c2 = report["criteria"]["C2_mu_decreases_with_temperature"]
    assert c2["n_decreasing"] == 3 and c2["n_cases"] == 3


def test_c3_reports_iterations_and_residuals(report):
    c3 = report["criteria"]["C3_fixed_point_convergence"]
    assert c3["max_iterations"] < 10
    assert c3["max_final_residual"] < 1.0e-2
    assert c3["max_picard_bisection_agreement"] < 1.0e-5
    for row in c3["rows"]:
        assert len(row["residual_history"]) == row["iterations"]


def test_c4_beats_the_constant_coefficient_baseline(report):
    c4 = report["criteria"]["C4_validation_against_literature"]
    assert c4["gate"] == ERROR_GATE
    assert c4["stats"]["n_within_gate"] >= 2, c4["stats"]
    # the constant mu = 0.10 baseline is scored on the same rows and is worse
    assert c4["stats"]["mean_abs_rel_error"] < c4["baseline_stats"]["mean_abs_rel_error"]
    assert c4["baseline_stats"]["n_within_gate"] <= 1
    # the calibration came from one source only (a real holdout)
    assert "Barnes" in report["calibration"]["source"]


def test_conductor_and_clapeyron_evidence(report):
    conductor = report["conductor"]
    assert conductor["steady_state"]["rel_error"] < 1e-6
    assert conductor["early_time"][-1]["rel_error"] < 1e-2
    assert conductor["clapeyron"]["slope_k_per_mpa"] == pytest.approx(0.0740, rel=1e-2)
    assert conductor["clapeyron"]["tabulated_deviation"] < 0.05
    for row in conductor["early_time"]:
        assert row["energy_relative_residual"] < 1e-8


def test_sliding_case_evidence_closes_energy(report):
    coupled = report["sliding_case"]["coupled"]
    assert abs(coupled["energy_closure_error"]) < 1e-9
    assert coupled["iterations_max"] < 10
    assert coupled["mu_final"] < coupled["mu_dry_initial"]
    assert coupled["contact_temperature_final_c"] <= coupled["melting_temperature_c"] + 1e-9


def test_pressure_melting_map_shows_the_threshold(report):
    pressure = report["pressure_melting"]
    thresholds = {row["temperature_c"]: row["pressure_melting_threshold_mpa"]
                  for row in pressure["threshold_curve"]}
    assert thresholds[-5.0] == pytest.approx(67.5, rel=1e-2)
    assert thresholds[-10.0] == pytest.approx(135.1, rel=1e-2)
    assert pressure["n_pressure_melted"] > 0
    assert pressure["max_mu_reduction"] > 0.5
    assert pressure["amontons_mu_spread"] < 1e-9


def test_evidence_digest_is_reproducible(report):
    recomputed = acceptance_report()
    assert recomputed["C6_provenance"]["digest"] == report["C6_provenance"]["digest"]
    assert canonical_digest({k: v for k, v in report.items() if k != "C6_provenance"}) \
        == report["C6_provenance"]["digest"]


def test_written_evidence_verifies_and_a_hand_edit_is_detected(tmp_path):
    model, _ = calibrate_from_barnes_sweep()
    path = tmp_path / "acceptance.json"
    payload = write_evidence(path, model)
    assert path.exists()
    stored = json.loads(path.read_text())
    assert stored["C6_provenance"]["digest"] == payload["C6_provenance"]["digest"]

    verdict = verify_evidence(path, model)
    assert verdict["digest_matches"], verdict
    assert verdict["numbers_match"], verdict["mismatches"]

    # tamper with one number: the digest must no longer match
    stored["criteria"]["C4_validation_against_literature"]["stats"]["n_within_gate"] = 99
    path.write_text(json.dumps(stored))
    tampered = verify_evidence(path, model)
    assert not tampered["digest_matches"]
    assert tampered["numbers_match"] is False