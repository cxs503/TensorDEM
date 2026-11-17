"""Tests for CASE-DEM-BIAXIAL-FRICTION (anchor case plumbing).

The anchor case measures the peak Mohr-Coulomb friction angle of a
plane-strain biaxial DEM specimen and compares it against the classical
critical-state reference (Bolton 1986, dense Hostun sand, phi_ps = 38 deg,
tolerance 8 %).  These tests pin:

* the pre-declared reference / tolerance / configuration (no silent mutation),
* the acceptance arithmetic (``friction_angle_deg`` / ``judge_friction``)
  against analytic values,
* the Bolton dilatancy index arithmetic,
* the end-to-end quick pipeline (8x16 smoke specimen, not a result),
* critical-state analysis integration (phi_cv extraction + Rowe cross-check),
* self-consistency of any committed anchor artefact (``run.json``): the
  stored judgement must be reproducible from the stored peak stresses.

They do not re-assert the DEM headline number (that is the benchmark's job,
``benchmarks/case_dem_biaxial_friction/run_case.py``).
"""

from __future__ import annotations

import json
import math
import os

import pytest
import torch

from tensordem.biaxial_friction_case import (
    ANCHOR_CONFIG,
    CASE_KEY,
    MU_PARTICLE,
    PHI_CV_REF_DEG,
    PHI_CV_ROWE_DEG,
    PHI_REF_DEG,
    QUICK_CONFIG,
    TOLERANCE,
    bolton_dilatancy_index,
    friction_angle_deg,
    judge_friction,
    run_biaxial_friction_case,
)

torch.set_num_threads(1)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Run records live in two tiers: the full artefact written by ``run_case.py``
# under ``artifacts/`` (filtered out of the PR gate, so usually absent on a
# clean checkout) and the tracked distilled record under ``docs/data/``.  The
# self-consistency tests below must run against whichever exists -- guarding on
# the untracked path alone silently turned them into no-ops on ``main``.
_FULL_ARTIFACT = os.path.join(REPO, "artifacts", CASE_KEY, "run.json")
_TRACKED_ARTIFACT = os.path.join(REPO, "docs", "data",
                                 f"{CASE_KEY}_run.json")
ARTIFACT = (_FULL_ARTIFACT if os.path.exists(_FULL_ARTIFACT)
            else _TRACKED_ARTIFACT)


# ---------------------------------------------------------------------------
# Pre-declared configuration
# ---------------------------------------------------------------------------

def test_anchor_configuration_matches_the_task_board():
    """The pre-declared anchor configuration is not silently mutated."""
    assert CASE_KEY == "CASE-DEM-BIAXIAL-FRICTION"
    assert ANCHOR_CONFIG["columns"] == 24
    assert ANCHOR_CONFIG["rows"] == 48
    assert ANCHOR_CONFIG["confining_stress"] == pytest.approx(100.0e3)
    assert ANCHOR_CONFIG["wall_friction"] == pytest.approx(0.0)
    assert ANCHOR_CONFIG["consolidation_steps"] == 4000
    assert ANCHOR_CONFIG["strain_target"] == pytest.approx(0.30)
    assert ANCHOR_CONFIG["freeze_bond_fraction"] == pytest.approx(0.0)
    assert ANCHOR_CONFIG["friction_coefficient"] == pytest.approx(0.65)
    assert PHI_REF_DEG == pytest.approx(38.0)
    assert TOLERANCE == pytest.approx(0.08)


def test_quick_configuration_is_a_smoke_not_a_result():
    """Quick config must stay small (smoke only) and share the reference."""
    assert QUICK_CONFIG["columns"] * QUICK_CONFIG["rows"] < 500
    assert QUICK_CONFIG["confining_stress"] == pytest.approx(100.0e3)
    assert QUICK_CONFIG["wall_friction"] == pytest.approx(0.0)
    assert QUICK_CONFIG["freeze_bond_fraction"] == pytest.approx(0.0)
    assert QUICK_CONFIG["friction_coefficient"] == pytest.approx(0.65)


def test_critical_state_references_are_declared():
    """The critical-state reference values are pre-declared and not mutated."""
    assert PHI_CV_REF_DEG == pytest.approx(33.0)  # Bolton 1986 Table 1
    assert PHI_CV_ROWE_DEG == pytest.approx(math.degrees(math.atan(0.65)), abs=1e-9)


def test_mu_particle_matches_rowe_prediction():
    """The interparticle friction coefficient gives phi_cv matching Bolton."""
    # arctan(0.65) ≈ 33.0 deg, matching Bolton (1986) Table 1 for angular quartz sands
    assert math.degrees(math.atan(MU_PARTICLE)) == pytest.approx(PHI_CV_REF_DEG, abs=0.5)


# ---------------------------------------------------------------------------
# Acceptance arithmetic (analytic)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("s1,s3,expected_deg", [
    (3.0, 1.0, 30.0),      # sin phi = (3-1)/(3+1) = 0.5
    (1.0, 1.0, 0.0),       # hydrostatic: no deviator
    (2.0, 1.0, 19.471220634490691),  # sin phi = 1/3
    (10.0, 1.0, 54.903198772415415),  # sin phi = 9/11
])
def test_friction_angle_deg_matches_analytic(s1, s3, expected_deg):
    assert friction_angle_deg(s1, s3) == pytest.approx(expected_deg, abs=1e-9)


def test_friction_angle_deg_rejects_invalid_input():
    assert math.isnan(friction_angle_deg(float("nan"), 1.0))
    assert math.isnan(friction_angle_deg(1.0, float("inf")))
    assert math.isnan(friction_angle_deg(-1.0, -2.0))  # denom <= 0


@pytest.mark.parametrize("phi_got,expect_pass", [
    (38.0, True),                       # exact reference
    (38.0 * (1.0 + TOLERANCE) - 1e-9, True),   # just inside the boundary
    (38.0 * (1.0 + TOLERANCE) + 1e-9, False),  # just outside
    (38.0 * (1.0 - TOLERANCE) + 1e-9, True),
    (38.0 * (1.0 - TOLERANCE) - 1e-9, False),
    (float("nan"), False),
])
def test_judge_friction_acceptance_arithmetic(phi_got, expect_pass):
    j = judge_friction(phi_got)
    assert j["ref_deg"] == pytest.approx(PHI_REF_DEG)
    assert j["tolerance"] == pytest.approx(TOLERANCE)
    assert j["passed"] is expect_pass
    assert j["within_tolerance"] is expect_pass
    if math.isfinite(phi_got):
        assert j["relative_error"] == pytest.approx(
            abs(phi_got - PHI_REF_DEG) / PHI_REF_DEG, abs=1e-12)


# ---------------------------------------------------------------------------
# Bolton dilatancy index (analytic)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phi_peak,phi_cv,A,expected_IR", [
    (38.0, 33.0, 5.0, 1.0),    # Bolton plane-strain: phi_peak = 33 + 5*1 = 38
    (38.0, 33.0, 3.0, 5.0/3.0),  # triaxial A=3
    (45.0, 30.0, 5.0, 3.0),    # dense specimen
    (33.0, 33.0, 5.0, 0.0),    # critical state: no dilation
])
def test_bolton_dilatancy_index_analytic(phi_peak, phi_cv, A, expected_IR):
    assert bolton_dilatancy_index(phi_peak, phi_cv, A=A) == pytest.approx(expected_IR, abs=1e-12)


def test_bolton_dilatancy_index_rejects_unphysical():
    """phi_peak < phi_cv is unphysical (no negative dilation index)."""
    assert math.isnan(bolton_dilatancy_index(30.0, 35.0))
    assert math.isnan(bolton_dilatancy_index(float("nan"), 33.0))
    assert math.isnan(bolton_dilatancy_index(38.0, float("nan")))


def test_bolton_dilatancy_index_bolton_1986_table2():
    """Bolton (1986) Table 2: phi_ps = phi_cv + 5*I_R for plane strain.

    At D_r=80%, sigma_3=100 kPa: phi_ps=38, phi_cv=33 => I_R = 1.0.
    This is the anchor reference point.
    """
    I_R = bolton_dilatancy_index(38.0, 33.0, A=5.0)
    assert I_R == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# End-to-end quick pipeline (smoke, not a result)
# ---------------------------------------------------------------------------

def test_quick_pipeline_runs_and_judges(tmp_path):
    """The quick (8x16) pipeline must run end-to-end and emit a valid record."""
    result = run_biaxial_friction_case(quick=True)
    phi = result["friction_angle"]["phi_peak_wall_deg"]
    assert math.isfinite(phi)
    assert 0.0 < phi < 90.0
    j = result["judgement"]
    assert j["ref_deg"] == pytest.approx(PHI_REF_DEG)
    assert j["tolerance"] == pytest.approx(TOLERANCE)
    assert isinstance(j["passed"], bool)
    # the quick specimen is under-consolidated; do NOT assert the verdict
    assert result["case_key"] == CASE_KEY
    assert result["reference"].startswith("Bolton (1986)")
    # freeze_bond_fraction must be 0 (cohesionless sand)
    assert result["config"]["freeze_bond_fraction"] == pytest.approx(0.0)
    # friction_coefficient must be 0.65 (sand)
    assert result["config"]["friction_coefficient"] == pytest.approx(0.65)


def test_quick_pipeline_includes_critical_state_analysis():
    """The quick pipeline must include critical-state analysis (even if not converged)."""
    result = run_biaxial_friction_case(quick=True)
    cs = result.get("critical_state", {})
    # The analysis must be present (may or may not converge for the quick specimen)
    assert isinstance(cs, dict)
    # Quick specimen has strain_target=0.05 < min_strain=0.20, so CS analysis
    # will return an error dict
    if "error" not in cs:
        assert "phi_cv_deg" in cs
        assert "converged" in cs
        assert "rowe" in cs
        assert "phi_mu_deg" in cs["rowe"]
        assert cs["rowe"]["phi_mu_deg"] == pytest.approx(
            math.degrees(math.atan(0.65)), abs=1e-9)


def test_quick_pipeline_includes_bolton_dilatancy():
    """The quick pipeline must include Bolton dilatancy index."""
    result = run_biaxial_friction_case(quick=True)
    bd = result.get("bolton_dilatancy", {})
    assert isinstance(bd, dict)
    assert "I_R" in bd
    assert bd["A"] == pytest.approx(5.0)


def test_bolton_dilatancy_reference_index_is_predeclared():
    """The reference dilatancy index follows from the pre-declared references.

    Bolton (1986) Table 2 (dense Hostun sand, D_r = 80 %, sigma_3 = 100 kPa):
    phi_ps = 38 deg, phi_cv = 33 deg, A = 5 (plane strain)
    => I_R_ref = (38 - 33) / 5 = 1.0.
    """
    I_R_ref = bolton_dilatancy_index(PHI_REF_DEG, PHI_CV_REF_DEG)
    assert I_R_ref == pytest.approx(1.0, abs=1e-12)


def test_quick_pipeline_reports_reference_bolton_index():
    """The pipeline must report the reference I_R alongside the measured one."""
    result = run_biaxial_friction_case(quick=True)
    bd = result.get("bolton_dilatancy", {})
    assert "I_R_ref" in bd
    # I_R_ref is derived from the pre-declared references, so it is exact
    # even for the under-consolidated quick specimen (whose measured I_R is NaN).
    assert bd["I_R_ref"] == pytest.approx(
        bolton_dilatancy_index(PHI_REF_DEG, PHI_CV_REF_DEG), abs=1e-12)
    assert bd["I_R_ref"] == pytest.approx(1.0, abs=1e-12)


def test_quick_pipeline_includes_dimensionless_numbers():
    """The quick pipeline must include dimensionless numbers for quasi-static audit."""
    result = run_biaxial_friction_case(quick=True)
    dim = result.get("dimensionless", {})
    assert isinstance(dim, dict)
    assert "inertial_number" in dim
    assert dim["inertial_number"] > 0.0
    assert "confining_stress_kPa" in dim
    assert dim["confining_stress_kPa"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Committed anchor artefact self-consistency (if present)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(ARTIFACT),
                      reason="anchor artefact not committed yet")
def test_committed_artefact_judgement_is_self_consistent():
    """If run.json is committed, its judgement must follow from its own peak."""
    with open(ARTIFACT) as f:
        rec = json.load(f)
    peak = rec["peak"]
    phi_recomputed = friction_angle_deg(peak["sigma_v"], peak["sigma_h"])
    assert phi_recomputed == pytest.approx(
        rec["friction_angle"]["phi_peak_wall_deg"], abs=1e-6)
    j = judge_friction(phi_recomputed)
    assert j["passed"] is rec["judgement"]["passed"]
    assert j["relative_error"] == pytest.approx(
        rec["judgement"]["relative_error"], abs=1e-9)


@pytest.mark.skipif(not os.path.exists(ARTIFACT),
                      reason="anchor artefact not committed yet")
def test_committed_artefact_has_critical_state_analysis():
    """If run.json is committed, it should include critical-state analysis.

    Older artifacts (pre-critical-state feature) may lack this key; the test
    passes if the key is absent (documented as a known gap) but validates
    the structure when present.
    """
    with open(ARTIFACT) as f:
        rec = json.load(f)
    cs = rec.get("critical_state")
    # Allow None/missing for legacy artifacts; validate structure when present
    if cs is None:
        pytest.skip("critical_state not present in this artifact (legacy)")
    assert isinstance(cs, dict), "critical_state must be a dict when present"
    if "error" not in cs:
        assert "phi_cv_deg" in cs
        assert "converged" in cs
        assert "rowe" in cs


@pytest.mark.skipif(not os.path.exists(ARTIFACT),
                      reason="anchor artefact not committed yet")
def test_committed_artefact_is_cohesionless():
    """If run.json is committed, it must use freeze_bond_fraction=0 (sand ref)."""
    with open(ARTIFACT) as f:
        rec = json.load(f)
    assert rec["config"]["freeze_bond_fraction"] == pytest.approx(0.0)


@pytest.mark.skipif(not os.path.exists(ARTIFACT),
                      reason="anchor artefact not committed yet")
def test_committed_artefact_uses_sand_friction():
    """If run.json is committed, it must use friction_coefficient=0.65 (sand ref)."""
    with open(ARTIFACT) as f:
        rec = json.load(f)
    assert rec["config"]["friction_coefficient"] == pytest.approx(0.65)
