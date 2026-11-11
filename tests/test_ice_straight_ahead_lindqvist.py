"""Pytest tests for DEM ice resistance vs Lindqvist validation.

Tests cover:
1. Lindqvist reference calculator (SHA-256 anchored)
2. DEM simulation (single case)
3. Campaign (6 cases × 3 seeds = 18 runs)
4. Physical consistency (scaling with h, C)
5. Acceptance criteria (≤15% error for most cases)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensordem.lindqvist_reference import (
    LindqvistInput,
    LindqvistResult,
    compute_lindqvist,
    lindqvist_sha256,
)
from tensordem.ice_straight_ahead_final import (
    DEMStraightAheadConfig,
    DEMStraightAheadResult,
    run_dem_straight_ahead,
    run_campaign,
    concentration_correction,
)


# ---------------------------------------------------------------------------
# Lindqvist reference calculator tests
# ---------------------------------------------------------------------------


class TestLindqvistReference:
    """Tests for the Lindqvist (1989) reference calculator."""

    def test_basic_calculation(self):
        """Test that Lindqvist computes all four components."""
        inp = LindqvistInput(
            hull_beam_m=20.0,
            hull_length_m=50.0,
            hull_draught_m=8.0,
            hull_entry_angle_deg=45.0,
            ice_thickness_m=0.5,
            ice_compressive_strength_pa=1.5e6,
            ice_flexural_strength_pa=0.5e6,
            speed_m_s=3.0,
        )
        result = compute_lindqvist(inp)
        
        assert result.crushing_N > 0
        assert result.bending_N > 0
        assert result.submersion_N > 0
        assert result.added_drag_N > 0
        assert result.total_N > 0
        assert result.total_N == pytest.approx(
            result.crushing_N + result.bending_N + result.submersion_N + result.added_drag_N
        )

    def test_sha256_reproducibility(self):
        """Test that SHA-256 hash is reproducible."""
        inp1 = LindqvistInput(
            hull_beam_m=20.0, hull_length_m=50.0, hull_draught_m=8.0,
            hull_entry_angle_deg=45.0, ice_thickness_m=0.5,
            ice_compressive_strength_pa=1.5e6, ice_flexural_strength_pa=0.5e6,
            speed_m_s=3.0,
        )
        inp2 = LindqvistInput(
            hull_beam_m=20.0, hull_length_m=50.0, hull_draught_m=8.0,
            hull_entry_angle_deg=45.0, ice_thickness_m=0.5,
            ice_compressive_strength_pa=1.5e6, ice_flexural_strength_pa=0.5e6,
            speed_m_s=3.0,
        )
        
        hash1 = lindqvist_sha256(inp1)
        hash2 = lindqvist_sha256(inp2)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest

    def test_sha256_changes_with_input(self):
        """Test that SHA-256 changes when input changes."""
        inp1 = LindqvistInput(
            hull_beam_m=20.0, hull_length_m=50.0, hull_draught_m=8.0,
            hull_entry_angle_deg=45.0, ice_thickness_m=0.5,
            ice_compressive_strength_pa=1.5e6, ice_flexural_strength_pa=0.5e6,
            speed_m_s=3.0,
        )
        inp2 = LindqvistInput(
            hull_beam_m=20.0, hull_length_m=50.0, hull_draught_m=8.0,
            hull_entry_angle_deg=45.0, ice_thickness_m=0.8,  # Different
            ice_compressive_strength_pa=1.5e6, ice_flexural_strength_pa=0.5e6,
            speed_m_s=3.0,
        )
        
        hash1 = lindqvist_sha256(inp1)
        hash2 = lindqvist_sha256(inp2)
        assert hash1 != hash2

    def test_thickness_scaling(self):
        """Test that crushing resistance scales as h²."""
        inp1 = LindqvistInput(
            hull_beam_m=20.0, hull_length_m=50.0, hull_draught_m=8.0,
            hull_entry_angle_deg=45.0, ice_thickness_m=0.5,
            ice_compressive_strength_pa=1.5e6, ice_flexural_strength_pa=0.5e6,
            speed_m_s=0.0,  # Zero speed to isolate crushing
        )
        inp2 = LindqvistInput(
            hull_beam_m=20.0, hull_length_m=50.0, hull_draught_m=8.0,
            hull_entry_angle_deg=45.0, ice_thickness_m=1.0,  # 2x thickness
            ice_compressive_strength_pa=1.5e6, ice_flexural_strength_pa=0.5e6,
            speed_m_s=0.0,
        )
        
        r1 = compute_lindqvist(inp1)
        r2 = compute_lindqvist(inp2)
        
        # Crushing should scale as h²: (1.0/0.5)² = 4
        ratio = r2.crushing_N / r1.crushing_N
        assert ratio == pytest.approx(4.0, rel=0.01)

    def test_invalid_input(self):
        """Test that invalid inputs raise errors."""
        with pytest.raises(ValueError):
            LindqvistInput(
                hull_beam_m=-20.0,  # Negative
                hull_length_m=50.0, hull_draught_m=8.0,
                hull_entry_angle_deg=45.0, ice_thickness_m=0.5,
                ice_compressive_strength_pa=1.5e6, ice_flexural_strength_pa=0.5e6,
                speed_m_s=3.0,
            )


# ---------------------------------------------------------------------------
# DEM simulation tests
# ---------------------------------------------------------------------------


class TestDEMSimulation:
    """Tests for the DEM straight-ahead simulation."""

    def test_single_run(self):
        """Test that a single DEM run produces valid results."""
        config = DEMStraightAheadConfig(
            ice_thickness_m=0.5,
            concentration=0.6,
            seed=42,
            domain_length_m=100.0,
            domain_width_m=40.0,
            travel_distance_m=100.0,
            dt_s=0.02,
        )
        result = run_dem_straight_ahead(config)
        
        assert result.mean_resistance_N > 0
        assert result.std_resistance_N >= 0
        assert result.lindqvist_total_N > 0
        assert result.ratio_dem_to_lindqvist > 0
        assert result.error_percent >= 0

    def test_reproducibility(self):
        """Test that same seed gives same result."""
        config1 = DEMStraightAheadConfig(ice_thickness_m=0.5, concentration=0.6, seed=42)
        config2 = DEMStraightAheadConfig(ice_thickness_m=0.5, concentration=0.6, seed=42)
        
        result1 = run_dem_straight_ahead(config1)
        result2 = run_dem_straight_ahead(config2)
        
        assert result1.mean_resistance_N == pytest.approx(result2.mean_resistance_N)

    def test_different_seeds(self):
        """Test that different seeds give different results."""
        config1 = DEMStraightAheadConfig(ice_thickness_m=0.5, concentration=0.6, seed=42)
        config2 = DEMStraightAheadConfig(ice_thickness_m=0.5, concentration=0.6, seed=43)
        
        result1 = run_dem_straight_ahead(config1)
        result2 = run_dem_straight_ahead(config2)
        
        # Results should be different (but similar)
        assert result1.mean_resistance_N != result2.mean_resistance_N
        # But within 20% of each other
        ratio = result1.mean_resistance_N / result2.mean_resistance_N
        assert 0.8 < ratio < 1.2


# ---------------------------------------------------------------------------
# Campaign tests
# ---------------------------------------------------------------------------


class TestCampaign:
    """Tests for the full campaign."""

    def test_campaign_runs(self):
        """Test that the campaign runs without errors."""
        results = run_campaign(
            thicknesses_m=[0.5],
            concentrations=[0.6],
            n_seeds=2,
        )
        
        assert results["n_cases"] == 2
        assert len(results["results"]) == 2

    def test_campaign_acceptance(self):
        """Test that most cases are within 15% of Lindqvist.
        
        Acceptance criterion: ≥80% of cases within 15% error.
        """
        results = run_campaign(
            thicknesses_m=[0.3, 0.5, 0.8],
            concentrations=[0.6, 0.8],
            n_seeds=3,
        )
        
        errors = [r["error_percent"] for r in results["results"]]
        within_15 = sum(1 for e in errors if e <= 15)
        fraction_within_15 = within_15 / len(errors)
        
        # At least 80% of cases should be within 15%
        assert fraction_within_15 >= 0.8, (
            f"Only {within_15}/{len(errors)} cases within 15% "
            f"({fraction_within_15*100:.1f}%)"
        )


# ---------------------------------------------------------------------------
# Physical consistency tests
# ---------------------------------------------------------------------------


class TestPhysicalConsistency:
    """Tests for physical consistency of the DEM model."""

    def test_resistance_increases_with_thickness(self):
        """Test that resistance increases with ice thickness."""
        results = []
        for h in [0.3, 0.5, 0.8]:
            config = DEMStraightAheadConfig(ice_thickness_m=h, concentration=0.6, seed=42)
            result = run_dem_straight_ahead(config)
            results.append(result.mean_resistance_N)
        
        # Resistance should increase with thickness
        assert results[0] < results[1] < results[2]

    def test_resistance_increases_with_concentration(self):
        """Test that resistance increases with concentration."""
        results = []
        for C in [0.6, 0.8]:
            config = DEMStraightAheadConfig(ice_thickness_m=0.5, concentration=C, seed=42)
            result = run_dem_straight_ahead(config)
            results.append(result.mean_resistance_N)
        
        # Resistance should increase with concentration
        assert results[0] < results[1]

    def test_concentration_correction(self):
        """Test the concentration correction function."""
        assert concentration_correction(0.6) == 1.0
        assert concentration_correction(0.8) < 1.0
        assert concentration_correction(0.4) == 1.0  # Below threshold


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self):
        """Test the full pipeline from Lindqvist to DEM comparison."""
        # Compute Lindqvist reference
        lind_inp = LindqvistInput(
            hull_beam_m=20.0, hull_length_m=50.0, hull_draught_m=8.0,
            hull_entry_angle_deg=45.0, ice_thickness_m=0.5,
            ice_compressive_strength_pa=1.5e6, ice_flexural_strength_pa=0.5e6,
            speed_m_s=3.0,
        )
        lind_result = compute_lindqvist(lind_inp)
        
        # Run DEM
        config = DEMStraightAheadConfig(
            ice_thickness_m=0.5, concentration=0.6, seed=42,
            domain_length_m=100.0, domain_width_m=40.0,
            travel_distance_m=100.0, dt_s=0.02,
        )
        dem_result = run_dem_straight_ahead(config)
        
        # Compare
        assert dem_result.lindqvist_total_N == pytest.approx(lind_result.total_N)
        assert dem_result.error_percent < 20  # Within 20%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
