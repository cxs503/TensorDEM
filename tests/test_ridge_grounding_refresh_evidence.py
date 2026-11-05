"""Doc <-> artifact drift gate for the regenerated ice-ridge grounding campaign.

The published seabed-strength numbers in ``docs/ice_ridge_pipeline_grounding.md``
had drifted from the physics by 20-33 % (they predated the Skempton capacity fix,
see ``docs/ridge_capacity_skempton_audit.md`` and
``docs/ridge_grounding_artifact_refresh.md``).  This module pins the document to
``docs/artifacts/ridge_grounding_refresh/summary.json`` and re-derives the key
quantities from the committed physics, so the drift cannot silently return.

No DEM run is performed here: the subgrade sweep and the zero-rate closure are
closed-form, and the coupled/rate numbers are compared against the committed
artifact record.
"""
from __future__ import annotations

import importlib.util
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(REPO, "docs", "artifacts", "ridge_grounding_refresh", "summary.json")
REFRESH_DOC = os.path.join(REPO, "docs", "ridge_grounding_artifact_refresh.md")
CAMPAIGN_DOC = os.path.join(REPO, "docs", "ice_ridge_pipeline_grounding.md")

#: numbers the campaign published before the Skempton fix (refuted here, kept as
#: the reference for the drift assertion)
PUBLISHED_PRE_FIX_PEAK = [73.8, 221.5, 442.9]


def _summary() -> dict:
    with open(SUMMARY, encoding="utf-8") as handle:
        return json.load(handle)


def _deck_section():
    path = os.path.join(REPO, "scripts", "run_ice_ridge_grounding.py")
    spec = importlib.util.spec_from_file_location("_ridge_deck_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BEAUFORT_SECTION


def _recompute_sweep() -> list:
    from tensordem.ridge_geotech import MohrCoulombSoil, SofteningVerticalSpring

    section = _deck_section()
    peaks = []
    for shear_strength in (10.0e3, 30.0e3, 60.0e3):
        soil = MohrCoulombSoil(friction_angle_deg=0.0, cohesion=shear_strength,
                               saturated_unit_weight=18.0e3, young_modulus=10.0e6,
                               water_unit_weight=10.06e3, undrained=True)
        spring = SofteningVerticalSpring(soil, section.outer_diameter, 0.5)
        peaks.append(spring.peak / 1.0e3)
    return peaks


def test_committed_summary_recomputes_from_physics():
    """The committed subgrade sweep must equal a fresh closed-form evaluation."""
    committed = _summary()["soil_strength_sweep"]["recomputed_peak_kN_per_m"]
    recomputed = _recompute_sweep()
    assert len(committed) == len(recomputed) == 3
    for stored, fresh in zip(committed, recomputed):
        assert stored == pytest.approx(fresh, rel=1e-4, abs=1e-3)


def test_published_numbers_are_refuted_by_a_uniform_factor():
    """The pre-fix numbers are exactly 4/3 of the fixed ones: a factor error, not a shape change."""
    sweep = _summary()["soil_strength_sweep"]
    ratios = [old / new for old, new in
              zip(sweep["published_pre_fix_peak_kN_per_m"], sweep["recomputed_peak_kN_per_m"])]
    for ratio in ratios:
        assert ratio == pytest.approx(4.0 / 3.0, rel=1e-3)
        assert ratio - 1.0 > 0.15, "the published value must be outside the 15 % band"


def test_zero_rate_extrapolation_closes_on_the_subgrade_peak():
    """At vanishing rate the coupled capacity is limited by the subgrade peak branch."""
    summary = _summary()
    intercept = summary["rate_study"]["linear_zero_rate_intercept_kN_per_m"]
    subgrade_peak = summary["main_case"]["soil_peak_kN_per_m"]
    assert abs(intercept / subgrade_peak - 1.0) < 0.03
    assert summary["zero_rate"]["closure_error"] == pytest.approx(
        intercept / subgrade_peak - 1.0, rel=1e-9)


def test_rate_rows_are_collinear_in_velocity():
    """A linear velocity law is what justifies extrapolating to zero rate."""
    rate = _summary()["rate_study"]
    velocities = rate["velocities"]
    capacities = rate["capacity_kN_per_m"]
    slope = rate["linear_slope_kN_per_m_per_mps"]
    intercept = rate["linear_zero_rate_intercept_kN_per_m"]
    assert len(velocities) == len(capacities) >= 3
    for velocity, capacity in zip(velocities, capacities):
        assert capacity == pytest.approx(intercept + slope * velocity, rel=2e-3)


def test_refresh_document_quotes_the_committed_numbers():
    """Every headline number in the refresh note must exist in the artifact."""
    summary = _summary()
    with open(REFRESH_DOC, encoding="utf-8") as handle:
        text = handle.read()
    quoted = [f"{value:.2f}" for value in
              summary["soil_strength_sweep"]["recomputed_peak_kN_per_m"]]
    quoted.append(f"{summary['main_case']['peak_load_per_length_kN_per_m']:.2f}")
    quoted.append(f"{summary['rate_study']['linear_zero_rate_intercept_kN_per_m']:.2f}")
    for number in quoted:
        assert number in text, f"{number} is quoted in docs but missing from artifact text"


def test_campaign_document_no_longer_publishes_the_pre_fix_numbers():
    """The stale 73.8 / 221.5 / 442.9 trio must be gone from the campaign report."""
    with open(CAMPAIGN_DOC, encoding="utf-8") as handle:
        text = handle.read()
    assert "the seabed\n  strength scales it monotonically (73.8 / 221.5 / 442.9" not in text
    assert "33.3 % too" in text
    assert "174.44" in text and "164.51" in text


def test_beam_criterion_is_graded_honestly():
    """Point load < 3 %, patch load inside 3-15 %: the published 1 % claim is retired."""
    beam = _summary()["beam_verification"]
    assert beam["point_load"]["moment_relative_error"] < 0.03
    assert 0.03 < beam["patch_load"]["deflection_relative_error"] < 0.15
    with open(CAMPAIGN_DOC, encoding="utf-8") as handle:
        text = handle.read()
    assert "4.47 %" in text