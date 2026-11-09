"""Tests for MY ice literature validation and ice ridge resistance."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.mixed_ice import aging
from tensordem.mixed_ice.field import MixedFieldSpec
from tensordem.mixed_ice.literature_validation import (
    LITERATURE_CASES,
    LiteratureCase,
    compute_validation_metrics,
    validate_my_resistance,
)
from tensordem.mixed_ice.resistance import mixed_resistance
from tensordem.mixed_ice.ridge_resistance import (
    IceRidge,
    RidgeFieldSpec,
    mixed_field_with_ridges,
    ridge_resistance,
)
from tensordem.level_ice.formula import ICEBREAKER


class TestLiteratureValidation:
    """Tests for MY ice resistance validation against literature."""

    def test_literature_cases_are_well_formed(self):
        """All literature cases have valid parameters."""
        for case in LITERATURE_CASES:
            assert case.ice_thickness_m > 0
            assert 0.0 <= case.my_fraction <= 1.0
            assert case.speed_m_s > 0
            assert case.measured_resistance_n > 0
            assert case.measured_std_n >= 0
            assert case.ice_type in ("FY", "MY", "mixed")

    def test_validate_my_resistance_returns_results(self):
        """Validation function returns expected structure."""
        validation = validate_my_resistance()
        assert "model" in validation
        assert "hull" in validation
        assert "n_cases" in validation
        assert "results" in validation
        assert validation["n_cases"] == len(LITERATURE_CASES)
        assert len(validation["results"]) == len(LITERATURE_CASES)

    def test_validation_results_have_required_fields(self):
        """Each validation result has all required fields."""
        validation = validate_my_resistance()
        for result in validation["results"]:
            assert "case_id" in result
            assert "measured_n" in result
            assert "predicted_n" in result
            assert "error_n" in result
            assert "relative_error" in result
            assert "within_15pct" in result
            assert isinstance(result["relative_error"], float)
            assert isinstance(result["within_15pct"], bool)

    def test_compute_validation_metrics(self):
        """Metrics computation returns expected structure."""
        validation = validate_my_resistance()
        metrics = compute_validation_metrics(validation)
        assert "n_cases" in metrics
        assert "mean_relative_error" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "pass_rate_15pct" in metrics
        assert "spearman_rho" in metrics
        assert "spearman_p_value" in metrics
        assert "fy_metrics" in metrics
        assert "my_metrics" in metrics
        assert "mixed_metrics" in metrics
        assert "all_within_15pct" in metrics

    def test_spearman_correlation_is_positive(self):
        """Resistance should increase monotonically with MY fraction."""
        validation = validate_my_resistance()
        metrics = compute_validation_metrics(validation)
        # Spearman rho should be positive (monotonic increase)
        assert metrics["spearman_rho"] > 0.0
        # p-value should be small (statistically significant)
        assert metrics["spearman_p_value"] < 0.05

    def test_pure_fy_degrades_to_baseline(self):
        """Pure FY case should match baseline model within 5%."""
        fy_cases = [c for c in LITERATURE_CASES if c.ice_type == "FY"]
        assert len(fy_cases) > 0
        for case in fy_cases:
            spec = MixedFieldSpec(my_fraction=0.0)
            result = mixed_resistance(spec, speed_m_s=case.speed_m_s)
            predicted = result.resistance_mean_n
            measured = case.measured_resistance_n
            relative_error = abs(predicted - measured) / measured
            # Should be within 5% for pure FY
            assert relative_error < 0.05, (
                f"FY case {case.case_id}: error {relative_error:.2%} > 5%"
            )

    def test_my_resistance_within_15_percent(self):
        """MY ice resistance should be within 15% of literature values."""
        my_cases = [c for c in LITERATURE_CASES if c.ice_type == "MY"]
        assert len(my_cases) > 0
        validation = validate_my_resistance()
        my_results = [r for r in validation["results"] if r["ice_type"] == "MY"]
        for result in my_results:
            assert result["within_15pct"], (
                f"MY case {result['case_id']}: error {result['relative_error']:.2%} > 15%"
            )

    def test_custom_validation_cases(self):
        """Validation works with custom cases."""
        custom_cases = [
            LiteratureCase(
                case_id="TEST_01",
                description="Test case",
                ice_thickness_m=1.5,
                ice_type="FY",
                my_fraction=0.0,
                speed_m_s=2.0,
                measured_resistance_n=1.0e6,
                measured_std_n=100e3,
                reference="Test",
                notes="Test case",
            )
        ]
        validation = validate_my_resistance(cases=custom_cases)
        assert validation["n_cases"] == 1
        assert validation["results"][0]["case_id"] == "TEST_01"


class TestIceRidgeResistance:
    """Tests for ice ridge resistance model."""

    def test_ice_ridge_default_values(self):
        """IceRidge has sensible defaults."""
        ridge = IceRidge()
        assert ridge.sail_height_m > 0
        assert ridge.keel_depth_m > 0
        assert ridge.width_m > 0
        assert 0.0 < ridge.sail_angle_deg < 90.0
        assert 0.0 < ridge.keel_angle_deg < 90.0
        assert 0.0 <= ridge.consolidated_fraction <= 1.0
        assert 0.0 <= ridge.porosity <= 1.0
        assert ridge.compressive_strength_pa > 0

    def test_ice_ridge_validation(self):
        """IceRidge validates inputs."""
        with pytest.raises(ValueError):
            IceRidge(sail_height_m=-1.0)
        with pytest.raises(ValueError):
            IceRidge(keel_depth_m=0.0)
        with pytest.raises(ValueError):
            IceRidge(sail_angle_deg=90.0)
        with pytest.raises(ValueError):
            IceRidge(consolidated_fraction=1.5)

    def test_ridge_resistance_components(self):
        """Ridge resistance has three components."""
        ridge = IceRidge()
        result = ridge_resistance(ridge, ICEBREAKER, speed_m_s=3.0)
        assert "sail_breaking_N" in result
        assert "keel_clearing_N" in result
        assert "core_crushing_N" in result
        assert "total_N" in result
        # Total should be sum of components
        total = (
            result["sail_breaking_N"]
            + result["keel_clearing_N"]
            + result["core_crushing_N"]
        )
        assert result["total_N"] == pytest.approx(total, rel=1e-12)

    def test_ridge_resistance_positive(self):
        """Ridge resistance is positive."""
        ridge = IceRidge()
        result = ridge_resistance(ridge, ICEBREAKER, speed_m_s=3.0)
        assert result["total_N"] > 0
        assert result["sail_breaking_N"] > 0
        assert result["keel_clearing_N"] > 0
        assert result["core_crushing_N"] >= 0

    def test_ridge_field_spec_default(self):
        """RidgeFieldSpec has sensible defaults."""
        spec = RidgeFieldSpec()
        assert 0.0 <= spec.my_fraction <= 1.0
        assert spec.ridge_density_per_km >= 0
        assert spec.level_ice_thickness_m > 0
        assert spec.ridge is not None

    def test_ridge_field_spec_validation(self):
        """RidgeFieldSpec validates inputs."""
        with pytest.raises(ValueError):
            RidgeFieldSpec(my_fraction=-0.1)
        with pytest.raises(ValueError):
            RidgeFieldSpec(my_fraction=1.5)
        with pytest.raises(ValueError):
            RidgeFieldSpec(ridge_density_per_km=-1.0)
        with pytest.raises(ValueError):
            RidgeFieldSpec(level_ice_thickness_m=0.0)

    def test_mixed_field_with_ridges(self):
        """Mixed field with ridges returns expected structure."""
        spec = RidgeFieldSpec(my_fraction=0.3)
        result = mixed_field_with_ridges(spec, speed_m_s=3.0)
        assert "level_resistance_N" in result
        assert "ridge_resistance_avg_N" in result
        assert "total_resistance_N" in result
        assert "ridge_contribution_fraction" in result
        assert "n_ridges" in result
        assert "level_details" in result
        assert "ridge_details" in result

    def test_ridges_increase_resistance(self):
        """Adding ridges increases total resistance."""
        spec_no_ridges = RidgeFieldSpec(my_fraction=0.3, ridge_density_per_km=0.0)
        spec_with_ridges = RidgeFieldSpec(my_fraction=0.3, ridge_density_per_km=5.0)
        result_no = mixed_field_with_ridges(spec_no_ridges, speed_m_s=3.0)
        result_with = mixed_field_with_ridges(spec_with_ridges, speed_m_s=3.0)
        assert result_with["total_resistance_N"] > result_no["total_resistance_N"]

    def test_ridge_density_scales_resistance(self):
        """Higher ridge density increases resistance."""
        densities = [1.0, 3.0, 5.0, 10.0]
        resistances = []
        for density in densities:
            spec = RidgeFieldSpec(my_fraction=0.3, ridge_density_per_km=density)
            result = mixed_field_with_ridges(spec, speed_m_s=3.0)
            resistances.append(result["total_resistance_N"])
        # Resistance should increase with density
        for i in range(len(resistances) - 1):
            assert resistances[i + 1] > resistances[i]

    def test_ridge_contribution_fraction(self):
        """Ridge contribution fraction is in [0, 1]."""
        spec = RidgeFieldSpec(my_fraction=0.3, ridge_density_per_km=5.0)
        result = mixed_field_with_ridges(spec, speed_m_s=3.0)
        frac = result["ridge_contribution_fraction"]
        assert 0.0 <= frac <= 1.0

    def test_no_ridges_equals_level_ice(self):
        """Zero ridge density equals pure level ice resistance."""
        spec = RidgeFieldSpec(my_fraction=0.3, ridge_density_per_km=0.0)
        result = mixed_field_with_ridges(spec, speed_m_s=3.0)
        level_spec = MixedFieldSpec(my_fraction=0.3)
        level_result = mixed_resistance(level_spec, speed_m_s=3.0)
        assert result["level_resistance_N"] == pytest.approx(
            level_result.resistance_mean_n, rel=1e-12
        )
        assert result["ridge_resistance_avg_N"] == pytest.approx(0.0, abs=1e-12)


class TestAcceptanceCriteria:
    """Tests for the three acceptance criteria."""

    def test_criterion_a_my_resistance_within_15_percent(self):
        """(a) MY ice resistance coefficient vs literature error < 15%."""
        validation = validate_my_resistance()
        metrics = compute_validation_metrics(validation)
        # Check MY cases specifically
        my_metrics = metrics["my_metrics"]
        assert my_metrics["n"] > 0
        assert my_metrics["pass_rate_15pct"] >= 1.0, (
            f"MY cases: {my_metrics['pass_rate_15pct']:.1%} within 15%, expected 100%"
        )

    def test_criterion_b_monotonic_increase_with_my_fraction(self):
        """(b) Mixed ice field resistance monotonically increases with MY ratio."""
        # Test with literature cases
        validation = validate_my_resistance()
        metrics = compute_validation_metrics(validation)
        # Spearman correlation should be +1 (perfect monotonic)
        assert metrics["spearman_rho"] > 0.9, (
            f"Spearman rho = {metrics['spearman_rho']:.3f}, expected > 0.9"
        )

        # Also test with explicit sweep
        my_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        resistances = []
        for p in my_fractions:
            spec = MixedFieldSpec(my_fraction=p)
            result = mixed_resistance(spec, speed_m_s=3.0)
            resistances.append(result.resistance_mean_n)
        # Check monotonic increase
        for i in range(len(resistances) - 1):
            assert resistances[i + 1] > resistances[i], (
                f"Resistance not monotonic: R({my_fractions[i]}) = {resistances[i]:.0f} N, "
                f"R({my_fractions[i+1]}) = {resistances[i+1]:.0f} N"
            )

    def test_criterion_c_pure_fy_degrades_to_baseline(self):
        """(c) Pure FY ice case degrades to existing model with error < 5%."""
        fy_cases = [c for c in LITERATURE_CASES if c.ice_type == "FY"]
        assert len(fy_cases) > 0, "No FY cases in literature data"
        for case in fy_cases:
            spec = MixedFieldSpec(my_fraction=0.0)
            result = mixed_resistance(spec, speed_m_s=case.speed_m_s)
            predicted = result.resistance_mean_n
            measured = case.measured_resistance_n
            relative_error = abs(predicted - measured) / measured
            assert relative_error < 0.05, (
                f"FY case {case.case_id}: error {relative_error:.2%} > 5%"
            )
