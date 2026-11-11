"""Test suite for DEM brash ice full-scale validation.

Campaign node p472w2-N04: Tests for ice.resistance.dem_straight_ahead
capability upgrade from WITHHELD to REAL.

Test coverage:
- Three-way comparison (DEM vs Lindqvist vs measured)
- Sobol sensitivity analysis
- Monte Carlo uncertainty quantification
- Acceptance criteria evaluation
- Negative controls
- Edge cases
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensordem.brash_ice_rheology.fullscale_validation import (
    OperatingPoint,
    MT_UIKKU_HULL,
    ODEN_HULL,
    FULLSCALE_DATASET,
    dem_resistance,
    lindqvist_resistance,
    three_way_comparison,
    sobol_sensitivity_analysis,
    monte_carlo_uncertainty,
    evaluate_acceptance_criteria,
)


# ============================================================================
# Test data fixtures
# ============================================================================

@pytest.fixture
def baseline_point():
    """Baseline operating point for testing."""
    return OperatingPoint("TEST-01", MT_UIKKU_HULL, 1.0, 0.5, 3.0, 520, 55)


@pytest.fixture
def high_resistance_point():
    """High resistance operating point."""
    return OperatingPoint("TEST-02", MT_UIKKU_HULL, 2.0, 0.8, 5.0, 1800, 150)


@pytest.fixture
def low_resistance_point():
    """Low resistance operating point."""
    return OperatingPoint("TEST-03", ODEN_HULL, 0.5, 0.4, 2.0, 150, 20)


# ============================================================================
# Test DEM resistance calculation
# ============================================================================

class TestDEMResistance:
    """Tests for dem_resistance function."""

    def test_dem_resistance_positive(self, baseline_point):
        """DEM resistance should be positive for valid operating point."""
        r = dem_resistance(baseline_point)
        assert r > 0, f"Expected positive resistance, got {r}"

    def test_dem_resistance_increases_with_thickness(self):
        """Resistance should increase with ice thickness."""
        pt1 = OperatingPoint("T1", MT_UIKKU_HULL, 0.8, 0.5, 3.0, 400, 40)
        pt2 = OperatingPoint("T2", MT_UIKKU_HULL, 1.2, 0.5, 3.0, 600, 60)
        r1 = dem_resistance(pt1)
        r2 = dem_resistance(pt2)
        assert r2 > r1, f"Expected r2 > r1, got {r2} <= {r1}"

    def test_dem_resistance_increases_with_speed(self):
        """Resistance should increase with ship speed."""
        pt1 = OperatingPoint("T1", MT_UIKKU_HULL, 1.0, 0.5, 2.0, 350, 35)
        pt2 = OperatingPoint("T2", MT_UIKKU_HULL, 1.0, 0.5, 4.0, 700, 70)
        r1 = dem_resistance(pt1)
        r2 = dem_resistance(pt2)
        assert r2 > r1, f"Expected r2 > r1, got {r2} <= {r1}"

    def test_dem_resistance_decreases_with_porosity(self):
        """Resistance should decrease with increasing porosity (lower concentration)."""
        pt1 = OperatingPoint("T1", MT_UIKKU_HULL, 1.0, 0.7, 3.0, 500, 50)  # low porosity
        pt2 = OperatingPoint("T2", MT_UIKKU_HULL, 1.0, 0.3, 3.0, 250, 25)  # high porosity
        r1 = dem_resistance(pt1)
        r2 = dem_resistance(pt2)
        assert r2 < r1, f"Expected r2 < r1, got {r2} >= {r1}"

    def test_dem_resistance_negative_control(self):
        """Negative control: C_ice=0 should give R_DEM=0."""
        null_point = OperatingPoint("NULL", MT_UIKKU_HULL, 1.0, 0.0, 3.0, 0.0)
        r = dem_resistance(null_point)
        assert r == 0.0, f"Expected R=0 for C=0, got {r}"

    def test_dem_resistance_micro_parameters(self, baseline_point):
        """DEM resistance should respond to micro-parameter changes."""
        r_default = dem_resistance(baseline_point)
        r_high_mu = dem_resistance(baseline_point, mu_is=0.25)
        r_high_sigma = dem_resistance(baseline_point, sigma_b=2.0e6)
        
        # Higher friction should increase resistance
        assert r_high_mu != r_default, "mu_is should affect resistance"
        # Higher bond strength should increase resistance
        assert r_high_sigma != r_default, "sigma_b should affect resistance"


# ============================================================================
# Test Lindqvist resistance calculation
# ============================================================================

class TestLindqvistResistance:
    """Tests for lindqvist_resistance function."""

    def test_lindqvist_positive(self, baseline_point):
        """Lindqvist resistance should be positive."""
        r = lindqvist_resistance(baseline_point)
        assert r > 0, f"Expected positive resistance, got {r}"

    def test_lindqvist_increases_with_thickness(self):
        """Lindqvist resistance should increase with ice thickness."""
        pt1 = OperatingPoint("T1", MT_UIKKU_HULL, 0.8, 0.5, 3.0, 400, 40)
        pt2 = OperatingPoint("T2", MT_UIKKU_HULL, 1.2, 0.5, 3.0, 600, 60)
        r1 = lindqvist_resistance(pt1)
        r2 = lindqvist_resistance(pt2)
        assert r2 > r1, f"Expected r2 > r1, got {r2} <= {r1}"


# ============================================================================
# Test three-way comparison
# ============================================================================

class TestThreeWayComparison:
    """Tests for three_way_comparison function."""

    def test_comparison_returns_list(self):
        """Comparison should return a list of results."""
        results = three_way_comparison()
        assert isinstance(results, list)
        assert len(results) == len(FULLSCALE_DATASET)

    def test_comparison_has_required_fields(self):
        """Each comparison result should have required fields."""
        results = three_way_comparison()
        required_fields = [
            "id", "ship", "H_m", "C_ice", "V_m_s",
            "R_DEM_kN", "R_Lindqvist_kN", "R_measured_kN",
            "err_DEM_vs_Lindqvist", "err_DEM_vs_measured",
        ]
        for result in results:
            for field in required_fields:
                assert field in result, f"Missing field: {field}"

    def test_comparison_baseline_accuracy(self):
        """Baseline case should have DEM vs Lindqvist error ≤15%."""
        results = three_way_comparison()
        baseline = results[0]
        err = abs(baseline["err_DEM_vs_Lindqvist"])
        assert err <= 0.15, f"Baseline error {err*100:.1f}% exceeds 15%"

    def test_comparison_mean_accuracy(self):
        """Mean DEM vs measured error should be ≤20%."""
        results = three_way_comparison()
        errors = [abs(r["err_DEM_vs_measured"]) for r in results]
        mean_err = np.mean(errors)
        assert mean_err <= 0.20, f"Mean error {mean_err*100:.1f}% exceeds 20%"


# ============================================================================
# Test Sobol sensitivity analysis
# ============================================================================

class TestSobolSensitivity:
    """Tests for sobol_sensitivity_analysis function."""

    def test_sobol_returns_dict(self, baseline_point):
        """Sobol analysis should return a dictionary."""
        result = sobol_sensitivity_analysis(baseline_point, n_samples=100)
        assert isinstance(result, dict)
        assert "S1" in result
        assert "ST" in result
        assert "ST_sum" in result

    def test_sobol_sum_equals_one(self, baseline_point):
        """Sum of total sensitivity indices should equal 1.0."""
        result = sobol_sensitivity_analysis(baseline_point, n_samples=100)
        st_sum = result["ST_sum"]
        assert abs(st_sum - 1.0) < 0.01, f"ΣS_Ti = {st_sum}, expected 1.0 ± 0.01"

    def test_sobol_has_dominant_parameter(self, baseline_point):
        """At least one parameter should have S_Ti > 0.4."""
        result = sobol_sensitivity_analysis(baseline_point, n_samples=100)
        max_st = max(result["ST"].values())
        assert max_st > 0.4, f"Max S_Ti = {max_st}, expected > 0.4"

    def test_sobol_parameter_names(self, baseline_point):
        """Sobol result should have correct parameter names."""
        result = sobol_sensitivity_analysis(baseline_point, n_samples=100)
        expected_params = ["mu_ii", "mu_is", "sigma_b", "rho_ice"]
        assert set(result["ST"].keys()) == set(expected_params)


# ============================================================================
# Test Monte Carlo uncertainty quantification
# ============================================================================

class TestMonteCarloUQ:
    """Tests for monte_carlo_uncertainty function."""

    def test_mc_returns_dict(self, baseline_point):
        """Monte Carlo should return a dictionary."""
        result = monte_carlo_uncertainty(baseline_point, n_samples=50)
        assert isinstance(result, dict)
        assert "R_mean_kN" in result
        assert "R_CI95_lo_kN" in result
        assert "R_CI95_hi_kN" in result

    def test_mc_ci_ordering(self, baseline_point):
        """CI bounds should be properly ordered."""
        result = monte_carlo_uncertainty(baseline_point, n_samples=50)
        assert result["R_CI95_lo_kN"] < result["R_CI95_hi_kN"]
        assert result["R_CI95_lo_kN"] < result["R_mean_kN"] < result["R_CI95_hi_kN"]

    def test_mc_coverage_flag(self, baseline_point):
        """Monte Carlo should report coverage flag."""
        result = monte_carlo_uncertainty(baseline_point, n_samples=50)
        assert "CI95_covers_measured" in result
        assert isinstance(result["CI95_covers_measured"], bool)


# ============================================================================
# Test acceptance criteria evaluation
# ============================================================================

class TestAcceptanceCriteria:
    """Tests for evaluate_acceptance_criteria function."""

    def test_criteria_returns_dict(self):
        """Criteria evaluation should return a dictionary."""
        comparison = three_way_comparison()
        sobol = sobol_sensitivity_analysis(n_samples=100)
        mc_results = [monte_carlo_uncertainty(pt, n_samples=50) for pt in FULLSCALE_DATASET[:3]]
        
        criteria = evaluate_acceptance_criteria(comparison, sobol, mc_results)
        assert isinstance(criteria, dict)
        assert "overall" in criteria

    def test_criteria_has_all_checks(self):
        """Criteria should include all acceptance checks (a)-(f)."""
        comparison = three_way_comparison()
        sobol = sobol_sensitivity_analysis(n_samples=100)
        mc_results = [monte_carlo_uncertainty(pt, n_samples=50) for pt in FULLSCALE_DATASET[:3]]
        
        criteria = evaluate_acceptance_criteria(comparison, sobol, mc_results)
        expected_keys = [
            "a_DEM_vs_Lindqvist",
            "b_DEM_vs_measured",
            "c_sobol_sum",
            "d_dominant_param",
            "e_ci95_coverage",
            "f_negative_control",
            "overall",
        ]
        for key in expected_keys:
            assert key in criteria, f"Missing criterion: {key}"

    def test_criteria_verdicts_are_strings(self):
        """All verdicts should be 'PASS' or 'FAIL'."""
        comparison = three_way_comparison()
        sobol = sobol_sensitivity_analysis(n_samples=100)
        mc_results = [monte_carlo_uncertainty(pt, n_samples=50) for pt in FULLSCALE_DATASET[:3]]
        
        criteria = evaluate_acceptance_criteria(comparison, sobol, mc_results)
        for name, crit in criteria.items():
            if "verdict" in crit:
                assert crit["verdict"] in ["PASS", "FAIL"], \
                    f"Invalid verdict for {name}: {crit['verdict']}"


# ============================================================================
# Test dataset properties
# ============================================================================

class TestDataset:
    """Tests for the full-scale dataset."""

    def test_dataset_has_enough_points(self):
        """Dataset should have at least 10 operating points."""
        assert len(FULLSCALE_DATASET) >= 10, \
            f"Dataset has {len(FULLSCALE_DATASET)} points, need ≥10"

    def test_dataset_has_both_ships(self):
        """Dataset should include both MT Uikku and ODEN."""
        ships = set(pt.ship.name for pt in FULLSCALE_DATASET)
        assert "MT_Uikku" in ships
        assert "ODEN" in ships

    def test_dataset_parameter_ranges(self):
        """Dataset should span required parameter ranges."""
        H_values = [pt.ice_thickness_m for pt in FULLSCALE_DATASET]
        C_values = [pt.ice_concentration for pt in FULLSCALE_DATASET]
        V_values = [pt.speed_m_s for pt in FULLSCALE_DATASET]
        
        # Ice thickness: 0.5 - 2.0 m
        assert min(H_values) <= 0.6
        assert max(H_values) >= 1.8
        
        # Concentration: 0.4 - 0.8
        assert min(C_values) <= 0.45
        assert max(C_values) >= 0.75
        
        # Speed: 2.0 - 5.0 m/s
        assert min(V_values) <= 2.1
        assert max(V_values) >= 4.5


# ============================================================================
# Integration test
# ============================================================================

class TestIntegration:
    """Integration tests for the full validation pipeline."""

    def test_full_validation_pipeline(self, tmp_path):
        """Full validation should run without errors."""
        from tensordem.brash_ice_rheology.fullscale_validation import run_full_validation
        
        output_dir = str(tmp_path / "validation_output")
        results = run_full_validation(output_dir)
        
        assert isinstance(results, dict)
        assert "acceptance_criteria" in results
        assert "three_way_comparison" in results
        assert "sobol_sensitivity" in results
        assert "monte_carlo_uq" in results
        
        # Check that results file was created
        results_file = Path(output_dir) / "results.json"
        assert results_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
