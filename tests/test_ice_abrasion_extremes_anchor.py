"""Tests for the ISO 19906 external-anchor verification of ``ice-abrasion-extremes``.

These tests pin the three things the artifact is allowed to claim:

* the *algebraic* part of the anchor (dimensional homogeneity, p(A) internal consistency,
  unit-change invariance, caliber conversion) is closed to the declared tolerance;
* the case's own pressure-area coupling is statistically zero, i.e. the case is
  structurally not a p(A) realization;
* no verbatim transcription was manufactured: with no reachable licensed copy of
  ISO 19906:2019 the anchor entries stay WITHHELD and the domain comparison stays
  NOT_ADJUDICABLE (an honest FAIL, not a silent pass).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "anchors"))

import ice_abrasion_extremes_anchor as anchor  # noqa: E402


@pytest.fixture(scope="module")
def payload():
    return anchor.build()


def test_dimensional_homogeneity():
    struct = anchor.structural_checks()
    assert struct["prefactor_dimensionally_homogeneous"] is True
    assert struct["prefactor_dims_of_C"]["M"] == pytest.approx(1.0, abs=1e-15)
    assert struct["prefactor_dims_of_C"]["T"] == pytest.approx(-2.0, abs=1e-15)
    assert struct["prefactor_dims_of_C"]["L"] == pytest.approx(
        struct["prefactor_length_exponent_expected"], abs=1e-15)
    # one dimensionless group: p/(C A^n); the exponent is a free number, not forced by units
    assert struct["buckingham_dimensionless_groups"] == 1


def test_pair_parameterization_and_unit_invariance():
    struct = anchor.structural_checks()
    assert struct["pair_parameterization_residual"] < 1e-12
    assert struct["area_unit_change_residual"] < 1e-12   # m^2 <-> ft^2, n invariant
    assert struct["pressure_unit_change_residual"] < 1e-12  # MPa <-> Pa


def test_pA_internal_consistency():
    struct = anchor.structural_checks()
    assert struct["p_at_A0_residual"] < 1e-12
    assert struct["strict_monotone_decreasing"] is True
    assert struct["loglog_slope_error"] < 1e-9
    assert struct["area_weighted_mean_residual"] < 1e-9


def test_case_caliber_conversion(payload):
    case = payload["case_caliber"]
    idr = case["identity_residuals"]
    assert idr["p_load_bearing_times_fraction_minus_p_footprint"] < 1e-9
    assert idr["load_minus_p_footprint_times_area"] < 1e-9
    assert idr["load_minus_p_load_bearing_times_area"] < 1e-9
    frac = case["parameters"]["contact_area_fraction"]
    assert case["medians"]["pressure_on_footprint_pa"] == pytest.approx(
        case["medians"]["pressure_on_load_bearing_area_pa"] * frac, rel=1e-12)


def test_case_implied_exponent_is_zero(payload):
    ci = payload["case_caliber"]["case_implied_exponent"]
    lo, hi = ci["bootstrap_95_interval"]
    # pressure and fragment area are drawn independently in the shipped sampler
    assert lo < 0.0 < hi
    assert abs(ci["slope"]) < 0.02
    assert abs(ci["slope"]) < 3.0 * ci["standard_error"]


def test_caliber_ambiguity_dominates_area_band(payload):
    cal = payload["case_caliber"]["caliber_ambiguity"]
    assert cal["factor_load_bearing_vs_footprint"] == pytest.approx(50.0, rel=1e-12)
    assert cal["abs_log_gap"] > cal["max_area_band_effect_abs_log"]
    assert cal["caliber_dominates_area_band"] is True


def test_recompute_chain_residuals(payload):
    steps = [s for s in payload["conversion_chain"] if "residual" in s]
    assert len(steps) == 3
    for step in steps:
        assert step["residual"] < 1e-9, step["id"]


def test_no_verbatim_transcription_was_manufactured(payload):
    for entry in payload["anchors"]:
        assert entry["status"] == "WITHHELD"
        assert entry["verbatim_quote"] is None
        assert entry["value_raw"] is None
        assert entry["retrieval_attempts"], "access attempts must be recorded"
        assert entry["gap"]
    # K1 stays open -> honest FAIL, and no agreement with the standard may be declared
    assert payload["verdict"]["overall"] == "HONEST_FAIL"
    assert "K1" in payload["verdict"]["open_criteria"]
    assert payload["domain_comparison"]["status"] == "NOT_ADJUDICABLE"
    assert payload["domain_comparison"]["agreement_declared"] is False


def test_value_free_criteria_pass(payload):
    passed = {c["id"]: c["passed"] for c in payload["criteria"]}
    for cid in ("K2", "K3", "K4", "K5", "K6", "K7"):
        assert passed[cid] is True, cid
    assert passed["K1"] is False


def test_artifact_on_disk_matches_script(tmp_path):
    """The shipped JSON must be the script's own output (no hand-edited numbers)."""
    written = ROOT / "benchmarks" / "anchors" / "ice-abrasion-extremes.json"
    assert written.exists()
    on_disk = json.loads(written.read_text())
    fresh = anchor.build()
    for key in ("verdict", "criteria", "conversion_chain", "candidate_registry"):
        assert json.dumps(on_disk[key], sort_keys=True) == json.dumps(fresh[key],
                                                                     sort_keys=True), key
    a = on_disk["case_caliber"]["case_implied_exponent"]
    b = fresh["case_caliber"]["case_implied_exponent"]
    for key in ("slope", "standard_error", "pearson_r"):
        assert a[key] == pytest.approx(b[key], rel=1e-12), key
    for key in ("footprint_area_m2", "pressure_on_footprint_pa"):
        assert (on_disk["case_caliber"]["medians"][key]
                == pytest.approx(fresh["case_caliber"]["medians"][key], rel=1e-12)), key