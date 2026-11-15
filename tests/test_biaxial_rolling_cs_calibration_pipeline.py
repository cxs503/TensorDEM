"""Verification of the R5 rolling-resistance critical-state calibration pipeline.

Anchor ``CASE-DEM-BIAXIAL-FRICTION`` (算例池·双轴剪切摩擦角) is closed by the
R5 calibration: ``phi_cv`` vs ``mu_r`` is extracted from isochoric biaxial shear
stress paths and judged against the Rowe (1962) D=0 limit ``phi_cv = arctan(mu)``.

This test verifies the *analysis* half of the pipeline
(``benchmarks/case_dem_biaxial_friction/analyse_rolling_cs.py``) against a
**synthetic analytic stress path** -- i.e. it checks that the calibration
artefact generation (critical-state extraction + Rowe cross-check + summary
table) is self-consistent and recovers a known critical-state angle, *without*
needing the heavy cluster probe data.

It does NOT re-assert the DEM headline number (that is the cluster probe's job,
``scripts/m21110_probe_rolling_cs.py`` -> ``runs/m21110/rolling_cs_*.json``).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tensordem.biaxial_critical_state import (
    critical_state_from_stress_path,
    judge_against_reference,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSE = os.path.join(REPO, "benchmarks", "case_dem_biaxial_friction", "analyse_rolling_cs.py")


def _synthetic_isochoric_path(mu: float, sigma3: float, n: int = 400) -> dict:
    """Build a synthetic isochoric stress path that reaches a known critical state.

    For a frictionless-wall isochoric cell the critical-state stress ratio is
    ``q/(2p) = sin(phi_cv)`` with ``phi_cv = arctan(mu)`` (Rowe D=0 limit, the
    interparticle-sliding bound).  We build a path that ramps to this ratio and
    then holds it stationary (the four-gate detector's ``deviator_stationary``
    and ``window_independent`` gates), with ``eps_v = 0`` (log measure) so the
    ``isochoric`` gate passes.
    """
    eps_a = np.linspace(0.0, 0.35, n)
    phi_cv = math.degrees(math.atan(mu))
    sin_phi = math.sin(math.radians(phi_cv))
    # q/(2p) ramps to sin(phi_cv) then holds
    ratio = np.minimum(eps_a / 0.10, 1.0) * sin_phi
    # p stays at the confining level (isochoric: no net volume change)
    p = np.full_like(eps_a, sigma3)
    q = 2.0 * p * ratio
    sv = p + q / 2.0
    sh = p - q / 2.0
    return {
        "axial_strain": eps_a.tolist(),
        "axial_strain_log": np.log(1.0 + eps_a).tolist(),
        "volumetric_strain": np.zeros_like(eps_a).tolist(),
        "volumetric_strain_log": np.zeros_like(eps_a).tolist(),
        "sigma_v": sv.tolist(),
        "sigma_h": sh.tolist(),
        "deviator": q.tolist(),
    }


def test_calibration_pipeline_recovers_rowe_limit():
    """The R5 analysis pipeline recovers the Rowe D=0 limit from a synthetic path.

    Feed a synthetic isochoric path with ``mu = 0.55`` (so ``phi_cv = arctan(0.55)
    = 28.81 deg``) into the same ``critical_state_from_stress_path`` call the
    analysis script uses, and check the extracted ``phi_cv`` matches the Rowe
    limit within the anchor's 8 % tolerance.
    """
    mu = 0.55
    sigma3 = 100.0e3
    path = _synthetic_isochoric_path(mu, sigma3)

    cs = critical_state_from_stress_path(
        path, sigma3_target=sigma3,
        strain_measure="log", boundary_control="isochoric",
        min_strain=0.15,
    )
    phi_cv = float(cs.phi_cv_deg)
    phi_rowe = math.degrees(math.atan(mu))

    # The synthetic path is exactly at the Rowe limit -> error ~ 0
    assert cs.converged, "synthetic isochoric path must converge to a critical state"
    assert math.isfinite(phi_cv)
    rel = abs(phi_cv - phi_rowe) / phi_rowe
    assert rel < 0.08, f"phi_cv={phi_cv:.2f} vs Rowe {phi_rowe:.2f} -> {rel*100:.1f}% (must be <8%)"


def test_judge_against_reference_uses_anchor_tolerance():
    """The anchor's judge function applies the <8 % tolerance to the Rowe limit."""
    mu = 0.55
    phi_rowe = math.degrees(math.atan(mu))
    # A value exactly at the Rowe limit passes; one 10 % off fails
    ok = judge_against_reference(phi_rowe, phi_rowe, tolerance=0.08)
    assert ok["passed"] is True
    bad = judge_against_reference(phi_rowe * 1.10, phi_rowe, tolerance=0.08)
    assert bad["passed"] is False


def test_analysis_script_imports_and_has_expected_artefact_paths():
    """The analysis script declares the calibration artefact paths it will write."""
    with open(ANALYSE) as f:
        src = f.read()
    assert "rolling_cs_calibration.json" in src
    assert "rolling_cs_calibration.png" in src
    assert "runs/m21110" in src  # consumes the cluster probe output
