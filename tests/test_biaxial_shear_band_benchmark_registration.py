"""Tests for the CASE-BIAXIAL-SHEAR-BAND machine-referee registration.

The m15315 board audit (``docs/board_audit/m15315/AUDIT.md``) flagged that the
anchor ``CASE-BIAXIAL-SHEAR-BAND`` had **0 rows** in the platform's machine
referee table ``tensor_benchmark_cases``, so the <15 % tolerance against the
literature shear-band inclination could only be judged from the repo docstring,
not by a machine-checkable registered ref/tolerance.

``scripts/register_biaxial_shear_band_benchmark.py`` closes that registration
debt: it inserts the referee row (status=passed) and writes a program-injected
artifact ``artifacts/biaxial_shear_band/benchmark_registration/
benchmark_registration.json``.  These tests assert that the *registration
artifact* is self-consistent with the case verdict (the R10 literature
verification), so a stale or hand-edited registration cannot silently pass.

The tests are pure arithmetic on the committed artifact + the R10 verification
module; no DEM cell is run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensordem.literature_band_verification import (
    LITERATURE_BANDS,
    verify_literature_anchor,
)

ROOT = Path(__file__).resolve().parent.parent
ART = (
    ROOT
    / "artifacts"
    / "biaxial_shear_band"
    / "benchmark_registration"
    / "benchmark_registration.json"
)

# committed R7/R9 evidence (same as the registration script)
THETA = 55.00
PHI = 35.45
PSI_ROWE_32 = 4.14


@pytest.fixture(scope="module")
def reg() -> dict:
    assert ART.exists(), f"registration artifact missing: {ART}"
    return json.loads(ART.read_text())


def test_artifact_anchor_slug(reg):
    assert reg["anchor_slug"] == "CASE-BIAXIAL-SHEAR-BAND"


def test_artifact_status_passed(reg):
    assert reg["status"] == "passed"


def test_artifact_tolerance_is_owner_standard(reg):
    assert reg["tolerance"] == "<15% (owner standard)"


def test_artifact_repo_is_tensordem(reg):
    assert reg["repo"] == "TensorDEM"


def test_artifact_theta_matches_committed_evidence(reg):
    # the registration must carry the same theta as the R7/R9 committed evidence
    assert reg["theta_deg"] == THETA


def test_artifact_verdict_matches_r10_verification(reg):
    # re-derive the verdict from the same evidence; must agree with the artifact
    res = verify_literature_anchor(THETA, phi_deg=PHI, psi_deg=PSI_ROWE_32)
    assert res["verdict"] == "PASS"
    assert reg["verdict"] == res["verdict"]
    assert reg["within_15pct"] == res["within_15pct"]
    assert reg["inside_band"] == res["inside_band"]


def test_artifact_rel_error_is_8pct(reg):
    # 8.33% from the 60 deg midpoint, < 15% tolerance
    assert reg["rel_error_vs_midpoint"] == pytest.approx(0.0833333, abs=1e-4)
    assert reg["within_15pct"] is True


def test_artifact_honest_3pct_fail(reg):
    # the stricter 3% bar is NOT met (8.33% > 3%) — must be recorded honestly
    assert reg["within_3pct"] is False


def test_artifact_band_is_desrues_viggiani(reg):
    band = reg["band"]
    assert band["source"].startswith("Desrues & Viggiani")
    assert band["band_deg"] == [55.0, 60.0]
    assert band["midpoint_deg"] == 60.0


def test_artifact_theoretical_window_contains_theta(reg):
    # the measured theta must lie inside the three-law theoretical window
    win = reg["theoretical_window"]
    lo = min(win.values())
    hi = max(win.values())
    assert lo <= reg["theta_deg"] <= hi
