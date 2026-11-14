"""Tests for CASE-DEM-BIAXIAL-FRICTION (anchor case plumbing).

The anchor case measures the peak Mohr-Coulomb friction angle of a
plane-strain biaxial DEM specimen and compares it against the classical
critical-state reference (Bolton 1986, dense Hostun sand, phi_ps = 38 deg,
tolerance 8 %).  These tests pin:

* the pre-declared reference / tolerance / configuration (no silent mutation),
* the acceptance arithmetic (``friction_angle_deg`` / ``judge_friction``)
  against analytic values,
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
    PHI_CV_REF_DEG,
    PHI_CV_ROWE_DEG,
    PHI_REF_DEG,
    QUICK_CONFIG,
    TOLERANCE,
    friction_angle_deg,
    judge_friction,
    run_biaxial_friction_case,
)

torch.set_num_threads(1)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT = os.path.join(REPO, "artifacts", CASE_KEY, "run.json")


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
    assert PHI_REF_DEG == pytest.approx(38.0)
    assert TOLERANCE == pytest.approx(0.08)


def test_quick_configuration_is_a_smoke_not_a_result():
    """Quick config must stay small (smoke only) and share the reference."""
    assert QUICK_CONFIG["columns"] * QUICK_CONFIG["rows"] < 500
    assert QUICK_CONFIG["confining_stress"] == pytest.approx(100.0e3)
    assert QUICK_CONFIG["wall_friction"] == pytest.approx(0.0)


def test_critical_state_references_are_declared():
    """The critical-state reference values are pre-declared and not mutated."""
    assert PHI_CV_REF_DEG == pytest.approx(33.0)  # Bolton 1986 Table 1
    assert PHI_CV_ROWE_DEG == pytest.approx(math.degrees(math.atan(0.55)), abs=1e-9)


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
            math.degrees(math.atan(0.55)), abs=1e-9)


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
    """If run.json is committed, it must include critical-state analysis."""
    with open(ARTIFACT) as f:
        rec = json.load(f)
    cs = rec.get("critical_state", {})
    assert isinstance(cs, dict), "critical_state key must be present"
    if "error" not in cs:
        assert "phi_cv_deg" in cs
        assert "converged" in cs
        assert "rowe" in cs
