"""Verification tests for the ice-ridge grounding re-derivation (m3201).

These are verification-only tests: they exercise
:mod:`tensordem.ridge_grounding_verification`, which re-derives the closed-form
quantities behind ``docs/ice_ridge_pipeline_grounding.md`` and compares the
numbers published there with the committed artifacts.  They do not run the DEM.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensordem.ridge_geotech import SofteningVerticalSpring, skempton_undrained_nc
from tensordem.ridge_grounding_verification import (
    SQUARE_SHAPE_FACTOR, beaufort_soil, build_report, check_rate_study,
    check_winkler, legacy_undrained_nc_pre_m2983, parse_documented_numbers,
    render_markdown, skempton_fix_factor, soil_strength_linearity,
    soil_strength_sweep_closed_form)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "ridge_grounding"
DOC = ROOT / "docs" / "ice_ridge_pipeline_grounding.md"
DIAMETER = 0.914
EMBEDMENT = 0.5


def _sweep():
    rows = []
    for strength in (10.0e3, 30.0e3, 60.0e3):
        spring = SofteningVerticalSpring(beaufort_soil(strength), DIAMETER, EMBEDMENT)
        rows.append({"undrained_shear_strength": strength,
                     "soil_ultimate_capacity": spring.ultimate})
    return rows


def test_legacy_formula_reproduces_the_documented_pre_fix_sequence():
    """The audit numbers 8.076 -> 6.730 at z/B = 1.547 pin the double count."""
    depth = EMBEDMENT + DIAMETER
    legacy = legacy_undrained_nc_pre_m2983(depth, DIAMETER)
    fixed = skempton_undrained_nc(depth, DIAMETER)
    assert legacy == pytest.approx(8.076, abs=5e-4)
    assert fixed == pytest.approx(6.730, abs=5e-4)
    assert legacy / fixed == pytest.approx(SQUARE_SHAPE_FACTOR, rel=1e-12)


def test_subgrade_sweep_matches_skempton_closed_form():
    rows = _sweep()
    checks = soil_strength_sweep_closed_form(rows, diameter=DIAMETER,
                                             embedment=EMBEDMENT)
    assert len(checks) == 3
    for check in checks:
        assert check["passed"], check
        # the implied N_c is the Skempton strip value, not a shape-factor copy
        assert check["implied_nc"] == pytest.approx(check["skempton_nc"], rel=1e-12)
        assert check["implied_nc"] < 7.5


def test_subgrade_sweep_is_linear_in_shear_strength():
    result = soil_strength_linearity(_sweep())
    assert result["passed"], result


def test_pre_fix_sweep_is_exactly_twenty_percent_higher():
    rows = _sweep()
    checks = skempton_fix_factor(rows, diameter=DIAMETER, embedment=EMBEDMENT)
    for check in checks:
        assert check["passed"], check
        assert check["value"] == pytest.approx(SQUARE_SHAPE_FACTOR, rel=1e-12)
    # the published pre-fix numbers were 73.8 / 221.5 / 442.9 kN/m
    fixed_kn_per_m = [row["soil_ultimate_capacity"] / 1e3 * 1.2 for row in rows]
    assert fixed_kn_per_m == pytest.approx([73.8, 221.5, 442.9], abs=5e-2)


def test_rate_extrapolation_is_monotone_and_bounded():
    rows = [{"platen_velocity": 0.6, "capacity_per_length": 204501.0},
            {"platen_velocity": 0.3, "capacity_per_length": 184655.0},
            {"platen_velocity": 0.15, "capacity_per_length": 176000.0}]
    result = check_rate_study(rows)
    assert result["slope_per_metre_per_second"] > 0.0
    assert result["value"] < 176000.0  # extrapolation is below the slowest run
    assert result["linearity_residual_relative"] < 0.05


def test_winkler_check_rejects_a_broken_profile():
    good = {"pipe_profile": {"peak_moment": 100.0, "peak_moment_reference": 101.0,
                             "peak_deflection": 1.0, "peak_deflection_reference": 1.005,
                             "converged": True}}
    beam = {"point_load": {"moment_fe": 10.0, "moment_reference": 10.05},
            "patch_load": {"moment_fe": 20.0, "moment_reference": 20.1}}
    assert all(check["passed"] for check in check_winkler(good, beam))

    bad = json.loads(json.dumps(good))
    bad["pipe_profile"]["peak_moment"] = 150.0  # 50 % off Hetenyi
    failed = [c for c in check_winkler(bad, beam) if not c["passed"]]
    assert failed and "moment" in failed[0]["quantity"]


def test_documented_number_parser_reads_the_campaign_page():
    text = DOC.read_text(encoding="utf-8")
    numbers = parse_documented_numbers(text)
    for key in ("softening_percent", "winkler_percent", "flat_percent", "soil_sweep"):
        assert key in numbers, f"{key} not found in {DOC.name}"
    assert len(numbers["soil_sweep"]) == 3
    assert len(numbers["winkler_percent"]) == 2


needs_artifacts = pytest.mark.skipif(
    not (ARTIFACTS / "sensitivity.json").exists()
    or not (ARTIFACTS / "main_case.json").exists()
    or not (ARTIFACTS / "rate_study.json").exists(),
    reason="campaign artifacts not generated in this checkout")


@needs_artifacts
def test_committed_artifacts_pass_every_rederivation_check():
    artifacts = {
        "main_case": json.loads((ARTIFACTS / "main_case.json").read_text()),
        "sensitivity": json.loads((ARTIFACTS / "sensitivity.json").read_text()),
        "rate": json.loads((ARTIFACTS / "rate_study.json").read_text()),
        "beam": json.loads((ARTIFACTS / "beam_verification.json").read_text()),
    }
    documented = parse_documented_numbers(DOC.read_text(encoding="utf-8"))
    report = build_report(artifacts, documented, diameter=DIAMETER,
                          embedment=EMBEDMENT)
    assert report["verdict"] == "PASS", report["failed_quantities"]
    assert not report["missing_documented_numbers"]
    assert "PASS" in render_markdown(report)


@needs_artifacts
def test_committed_soil_sweep_is_the_fixed_model_not_the_pre_fix_one():
    """Guard against a stale artifact: the sweep must be 20 % below the old page."""
    sensitivity = json.loads((ARTIFACTS / "sensitivity.json").read_text())
    values = [row["soil_ultimate_capacity"] / 1e3
              for row in sensitivity["soil_strength"]]
    assert values == pytest.approx([61.5, 184.5, 369.1], abs=0.2)
    assert values[1] / values[0] == pytest.approx(3.0, rel=1e-12)
    assert math.isclose(values[2] / values[0], 6.0, rel_tol=1e-12)