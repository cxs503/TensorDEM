"""Tests for the glass-bead repose angle reference case (CASE-DEM-REPOSE-ANGLE).

Verification criteria:
C1: Analytical forward model predicts θ_r within ±3° of experimental reference (22°).
C2: Parameter sweep shows monotonic increase of θ_r with both μ and μ_r.
C3: Calibrated parameters (μ=0.35, μ_r=0.05, e=0.7) yield θ_r ≈ 22°.
C4: Results JSON is self-consistent (verdict matches error vs tolerance).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem.auto_calibration import analytical_repose_angle
from examples.glass_bead_repose_angle import (
    GLASS_BEAD_REF,
    GLASS_BEAD_DEM_PARAMS,
    analytical_validation,
    parameter_sweep_table,
)

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "glass_bead_repose" / "results.json"


# ---------------------------------------------------------------------------
# C1: Analytical model accuracy
# ---------------------------------------------------------------------------

class TestAnalyticalModelAccuracy:
    """C1: Analytical forward model predicts θ_r within ±3° of experiment."""

    def test_glass_bead_prediction_within_tolerance(self):
        """The calibrated parameters must predict θ_r within ±3° of 22°."""
        result = analytical_validation()
        assert result["passed"], (
            f"Analytical model prediction {result['theta_pred_deg']:.2f}° "
            f"deviates {result['abs_error_deg']:.2f}° from reference "
            f"{result['theta_ref_deg']:.1f}°, exceeding ±{result['tolerance_deg']}° tolerance"
        )

    def test_error_less_than_one_degree(self):
        """The error should be well within tolerance (< 1°)."""
        result = analytical_validation()
        assert result["abs_error_deg"] < 1.0, (
            f"Error {result['abs_error_deg']:.2f}° should be < 1° for a well-calibrated model"
        )


# ---------------------------------------------------------------------------
# C2: Monotonicity of parameter sweep
# ---------------------------------------------------------------------------

class TestParameterSweepMonotonicity:
    """C2: θ_r increases monotonically with μ and μ_r."""

    def test_theta_increases_with_mu(self):
        """For fixed μ_r, θ_r must increase with μ."""
        table = parameter_sweep_table()
        for mu_r in [0.02, 0.05, 0.08, 0.10]:
            vals = [r["theta_r_deg"] for r in table if r["mu_rolling"] == mu_r]
            for i in range(1, len(vals)):
                assert vals[i] > vals[i - 1], (
                    f"θ_r not monotonic in μ at μ_r={mu_r}: "
                    f"{vals[i-1]:.1f}° -> {vals[i]:.1f}°"
                )

    def test_theta_increases_with_mu_rolling(self):
        """For fixed μ, θ_r must increase with μ_r."""
        table = parameter_sweep_table()
        for mu in [0.20, 0.25, 0.30, 0.35, 0.40]:
            vals = [r["theta_r_deg"] for r in table if r["mu"] == mu]
            for i in range(1, len(vals)):
                assert vals[i] > vals[i - 1], (
                    f"θ_r not monotonic in μ_r at μ={mu}: "
                    f"{vals[i-1]:.1f}° -> {vals[i]:.1f}°"
                )


# ---------------------------------------------------------------------------
# C3: Calibrated parameters yield ~22°
# ---------------------------------------------------------------------------

class TestCalibratedParameters:
    """C3: The calibrated parameter set yields θ_r ≈ 22°."""

    def test_calibrated_params_yield_target_angle(self):
        """μ=0.35, μ_r=0.05, e=0.7 should give θ_r ≈ 22°."""
        theta = analytical_repose_angle(
            GLASS_BEAD_DEM_PARAMS["mu"],
            GLASS_BEAD_DEM_PARAMS["mu_rolling"],
            GLASS_BEAD_DEM_PARAMS["restitution"],
        )
        assert abs(theta - 22.0) <= 3.0, (
            f"Calibrated params give θ_r={theta:.1f}°, expected ~22° ± 3°"
        )

    def test_calibrated_params_in_acceptable_band(self):
        """The prediction should fall within the experimental band [20°, 25°]."""
        theta = analytical_repose_angle(
            GLASS_BEAD_DEM_PARAMS["mu"],
            GLASS_BEAD_DEM_PARAMS["mu_rolling"],
            GLASS_BEAD_DEM_PARAMS["restitution"],
        )
        lo, hi = GLASS_BEAD_REF["theta_exp_band_deg"]
        assert lo <= theta <= hi, (
            f"Prediction {theta:.1f}° outside experimental band [{lo}°, {hi}°]"
        )


# ---------------------------------------------------------------------------
# C4: Artifact self-consistency
# ---------------------------------------------------------------------------

class TestArtifactConsistency:
    """C4: Results JSON is self-consistent."""

    @pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not generated")
    def test_verdict_matches_error(self):
        """The verdict must match whether error ≤ tolerance."""
        data = json.loads(ARTIFACT.read_text())
        av = data["analytical_validation"]
        tolerance = data["reference"]["theta_tolerance_deg"]
        expected_verdict = "PASS" if av["abs_error_deg"] <= tolerance else "FAIL"
        assert data["verdict"] == expected_verdict, (
            f"Verdict {data['verdict']} inconsistent with error {av['abs_error_deg']:.2f}° "
            f"vs tolerance {tolerance}°"
        )

    @pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not generated")
    def test_sweep_table_has_expected_size(self):
        """The parameter sweep should have 5×4 = 20 entries."""
        data = json.loads(ARTIFACT.read_text())
        assert len(data["parameter_sweep"]) == 20, (
            f"Expected 20 sweep entries, got {len(data['parameter_sweep'])}"
        )

    @pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not generated")
    def test_figures_recorded(self):
        """The artifact should record figure metadata."""
        data = json.loads(ARTIFACT.read_text())
        assert len(data["figures"]) >= 2, "Expected at least 2 figures"
        names = [f["name"] for f in data["figures"]]
        assert "parameter_sweep" in names
        assert "comparison_bar" in names


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------

class TestNegativeControls:
    """Negative controls: frictionless case should give near-zero angle."""

    def test_zero_friction_zero_angle(self):
        """μ=0, μ_r=0 should give θ_r ≈ 0°."""
        theta = analytical_repose_angle(0.0, 0.0, 0.5)
        assert theta < 5.0, f"Frictionless case gives θ_r={theta:.1f}°, expected ~0°"

    def test_low_friction_low_angle(self):
        """Very low friction should give low angle (< 15°)."""
        theta = analytical_repose_angle(0.1, 0.01, 0.8)
        assert theta < 15.0, f"Low friction case gives θ_r={theta:.1f}°, expected < 15°"
