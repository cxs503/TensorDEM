"""Tests for the ISO 19906 external-anchor harness (ice-hull-propeller study).

These tests deliberately exercise the *refusal* path as much as the arithmetic:
the value of the harness is that an unverified anchor cannot reach a claim.
"""
from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.threefield.iso_anchor import (  # noqa: E402
    DIM_FORCE,
    DIM_LENGTH,
    DIM_LINE_LOAD,
    DIM_STRESS,
    DimensionMismatch,
    UnverifiedAnchorError,
    CaliberLevel,
    convert_pressure_area,
    dimensional_homogeneity_checks,
    global_force_locator_candidate,
    internal_consistency_checks,
    pressure_area_candidate,
    pressure_from_area,
    recompute_residual,
    require_anchor,
)

ARTIFACT = REPO / "benchmarks/anchors/ice-hull-propeller-threefield.json"
SCRIPT = REPO / "scripts/build_ice_hull_propeller_anchor.py"


# --------------------------------------------------------------------------- #
# dimensional algebra
# --------------------------------------------------------------------------- #
def test_dimension_algebra_is_exact():
    assert (DIM_STRESS * DIM_LENGTH).as_tuple() == DIM_LINE_LOAD.as_tuple()
    assert (DIM_STRESS * (DIM_LENGTH ** 2)).as_tuple() == DIM_FORCE.as_tuple()
    assert (DIM_LINE_LOAD * DIM_LENGTH).as_tuple() == DIM_FORCE.as_tuple()
    # non-integer exponents stay rational, no float drift
    dim = DIM_LENGTH ** Fraction(-3, 10)
    assert dim.length == Fraction(-3, 10)


@pytest.mark.parametrize("exponent", [-0.3, -0.2, -0.5, 0.0])
def test_dimensional_homogeneity_of_chain(exponent):
    checks = dimensional_homogeneity_checks(exponent)
    assert len(checks) == 6
    assert all(item["homogeneous"] for item in checks)


def test_dimensional_check_rejects_inhomogeneous_equation():
    from tensordem.threefield.iso_anchor import _check
    with pytest.raises(DimensionMismatch):
        _check("q = p", DIM_STRESS, DIM_LINE_LOAD, "deliberately wrong")


# --------------------------------------------------------------------------- #
# claim gate: an unverified anchor must not be readable
# --------------------------------------------------------------------------- #
def test_require_anchor_refuses_unverified_candidate():
    candidate = pressure_area_candidate(
        exponent=-0.3, c_ref_pa=1.2e6, a_ref_m2=2.5,
        candidate_id="synthetic", locator_hint="synthetic",
        provenance_note="synthetic-fixture")
    assert candidate.usable_for_claim is False
    with pytest.raises(UnverifiedAnchorError):
        require_anchor(candidate)


def test_require_anchor_refuses_verified_flag_without_quote():
    """A fabricated 'VERIFIED' badge is not enough: quote + locator are required."""
    candidate = pressure_area_candidate(
        exponent=-0.3, c_ref_pa=1.2e6, a_ref_m2=2.5,
        candidate_id="synthetic-verified", locator_hint="synthetic",
        provenance_note="synthetic-fixture")
    object.__setattr__(candidate, "verification_status", "VERIFIED")
    assert candidate.usable_for_claim is False
    with pytest.raises(UnverifiedAnchorError):
        require_anchor(candidate)


def test_require_anchor_accepts_full_provenance():
    candidate = pressure_area_candidate(
        exponent=-0.3, c_ref_pa=1.2e6, a_ref_m2=2.5,
        candidate_id="synthetic-complete", locator_hint="synthetic",
        provenance_note="synthetic-fixture")
    object.__setattr__(candidate, "verification_status", "VERIFIED")
    object.__setattr__(candidate, "quote", "synthetic quoted sentence")
    object.__setattr__(candidate, "locator", "synthetic-annex-A table A.1")
    assert require_anchor(candidate) == pytest.approx(-0.3)


def test_locator_only_anchor_carries_no_value():
    anchor = global_force_locator_candidate(
        candidate_id="synthetic-locator", locator_hint="synthetic")
    assert anchor.value is None
    assert anchor.usable_for_claim is False
    with pytest.raises(UnverifiedAnchorError):
        require_anchor(anchor)


# --------------------------------------------------------------------------- #
# conversion chain
# --------------------------------------------------------------------------- #
def _levels():
    return [CaliberLevel("unit_width_h0.30", 1.0, 0.30),
            CaliberLevel("unit_width_h0.60", 1.0, 0.60),
            CaliberLevel("full_beam_h0.60", 20.0, 0.60)]


def test_chain_recovers_reference_point_and_line_load_identity():
    chain = convert_pressure_area(exponent=-0.3, c_ref_pa=1.2e6, a_ref_m2=2.5,
                                  levels=_levels())
    assert chain["residuals"]["reference_point_recovery_relative"] < 1e-12
    for point in chain["points"]:
        assert point["line_load_n_per_m"] == pytest.approx(
            point["pressure_pa"] * point["thickness_m"], rel=1e-15)
        assert point["force_n"] == pytest.approx(
            point["line_load_n_per_m"] * point["width_m"], rel=1e-15)
        assert point["pressure_mpa"] * 1.0e6 == pytest.approx(point["pressure_pa"],
                                                             rel=1e-15)


def test_chain_size_effect_and_exponent_sensitivity():
    small = convert_pressure_area(exponent=-0.3, c_ref_pa=1.2e6, a_ref_m2=2.5,
                                  levels=[CaliberLevel("a", 1.0, 0.30)])
    big = convert_pressure_area(exponent=-0.3, c_ref_pa=1.2e6, a_ref_m2=2.5,
                                levels=[CaliberLevel("b", 1.0, 0.60)])
    # doubling the contact area lowers the pressure: exponent must be negative
    assert big["points"][0]["pressure_pa"] < small["points"][0]["pressure_pa"]
    ratio = small["points"][0]["pressure_pa"] / big["points"][0]["pressure_pa"]
    assert ratio == pytest.approx(2.0 ** 0.3, rel=1e-12)


def test_chain_flags_areas_outside_the_reference_window():
    chain = convert_pressure_area(exponent=-0.3, c_ref_pa=1.2e6, a_ref_m2=2.5,
                                  levels=_levels())
    by_label = {point["label"]: point for point in chain["points"]}
    assert by_label["unit_width_h0.30"]["outside_reference_area"] is True
    assert by_label["full_beam_h0.60"]["outside_reference_area"] is True


def test_internal_consistency_checks_pass():
    levels = _levels()
    chain = convert_pressure_area(exponent=-0.3, c_ref_pa=1.2e6, a_ref_m2=2.5,
                                  levels=levels)
    report = internal_consistency_checks(exponent=-0.3, c_ref_pa=1.2e6,
                                         a_ref_m2=2.5, levels=levels, chain=chain)
    assert report["all_ok"], report
    names = {item["check"] for item in report["checks"]}
    assert "d ln p / d ln A == ex on the converted curve" in names
    assert "pressure decreases with contact area (size effect)" in names


def test_recompute_residual_is_below_tolerance():
    levels = _levels()
    artifact = {"conversion_chain": convert_pressure_area(
        exponent=-0.3, c_ref_pa=1.2e6, a_ref_m2=2.5, levels=levels)}
    report = recompute_residual(artifact)
    assert report["ok"] is True
    assert report["worst_relative_residual"] < 1e-9


def test_recompute_detects_a_corrupted_stored_value():
    levels = _levels()
    chain = convert_pressure_area(exponent=-0.3, c_ref_pa=1.2e6, a_ref_m2=2.5,
                                  levels=levels)
    chain["points"][0]["pressure_pa"] *= 1.01
    report = recompute_residual({"conversion_chain": chain})
    assert report["ok"] is False
    assert report["worst_relative_residual"] > 1e-3


# --------------------------------------------------------------------------- #
# the landed artifact
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not generated")
def test_artifact_schema_and_verdict():
    artifact = json.loads(ARTIFACT.read_text())
    for key in ("artifact", "task_key", "case_caliber", "anchors",
                "conversion_chain", "checks", "claim_guard", "criteria", "verdict"):
        assert key in artifact, key
    assert artifact["task_key"] == "paper-exp-ice-hull-propeller-threefield-1"
    assert artifact["checks"]["recompute"]["ok"] is True
    assert artifact["checks"]["dimensional_homogeneity"]["all_ok"] is True
    assert artifact["checks"]["internal_consistency"]["all_ok"] is True
    # provenance discipline: nothing unverified may be presented as a claim
    for anchor in artifact["anchors"]:
        if anchor["verification_status"] != "VERIFIED":
            assert anchor["quote"] is None
            assert anchor["usable_for_claim"] is False
    assert artifact["claim_guard"]["claims_permitted"] == []
    assert artifact["downstream_gate"]["domain_comparison_allowed"] is False
    verified = sum(1 for a in artifact["anchors"] if a["usable_for_claim"])
    unmet = [c["criterion_id"] for c in artifact["criteria"]
             if c["status"] != "PASS"]
    if verified == 0:
        assert "C1_verbatim_transcription" in unmet
        assert artifact["verdict"]["overall"] == "HONEST_FAIL"
    else:  # once the text is transcribed this branch becomes the live one
        assert artifact["verdict"]["overall"] == "PASS"


@pytest.mark.skipif(not ARTIFACT.exists() or not SCRIPT.exists(),
                    reason="artifact/script not present")
def test_artifact_is_reproducible():
    """Regeneration must reproduce the artifact.

    The only exempt fields are the ones the artifact itself declares ``volatile``:
    they count/list worktree files that merely *mention* ISO 19906, so adding an
    unrelated document legitimately changes them.  Every number that matters
    (chain, residuals, criteria, verdict) must match exactly.
    """
    stored = json.loads(ARTIFACT.read_text())
    out = subprocess.run([sys.executable, str(SCRIPT), "--stdout"],
                         capture_output=True, text=True, check=True, cwd=str(REPO))
    fresh = json.loads(out.stdout)
    volatile = stored["generated_by"].get("volatile_fields", [])
    for path in volatile:
        head, _, leaf = path.rpartition(".")
        for doc in (stored, fresh):
            doc[head].pop(leaf, None)
    assert fresh == stored, (
        "artifact is not reproducible; regenerate with "
        "python3 scripts/build_ice_hull_propeller_anchor.py")
    assert json.dumps({k: v for k, v in stored.items() if k != "source_availability"},
                      sort_keys=True) == json.dumps(
        {k: v for k, v in fresh.items() if k != "source_availability"},
        sort_keys=True)


def test_unverified_candidates_are_never_rounded_into_values():
    """The candidate triple must round-trip to itself (no lossy rewriting)."""
    c_a = convert_pressure_area(exponent=-0.3, c_ref_pa=1.2e6, a_ref_m2=2.5,
                                levels=[CaliberLevel("a", 1.0, 0.30)])
    p_ref = pressure_from_area(c_a["pressure_area_coefficient_pa"], 2.5, -0.3)
    assert p_ref == pytest.approx(1.2e6, rel=1e-15)