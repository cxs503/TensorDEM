"""Tests for ridge resistance quantile analysis.

Machine acceptance criteria:
1. Quantiles are ordered: P10 < P50 < P90
2. Total resistance = F_sail + F_keel
3. Grounding depth < keel depth (due to settlement)
4. Sensitivity indices are finite
5. SHA256 reproducibility for same input
"""
from __future__ import annotations

import hashlib
import json
import math
import pytest
import numpy as np

from tensordem.ridge_resistance_quantiles import (
    RidgeResistanceQuantileConfig,
    RidgeResistanceQuantileResult,
    compute_ridge_resistance_quantiles,
    compute_grounding_depth_quantiles,
    sensitivity_analysis,
)


class TestRidgeResistanceQuantileConfig:
    """Test configuration validation."""

    def test_valid_config(self):
        """Valid config should not raise."""
        cfg = RidgeResistanceQuantileConfig(n_samples=100)
        assert cfg.n_samples == 100

    def test_invalid_n_samples(self):
        """n_samples < 10 should raise."""
        with pytest.raises(ValueError):
            RidgeResistanceQuantileConfig(n_samples=5)


class TestComputeRidgeResistanceQuantiles:
    """Test the full quantile analysis."""

    def test_result_structure(self):
        """Result should have expected attributes."""
        cfg = RidgeResistanceQuantileConfig(n_samples=100, seed=42)
        result = compute_ridge_resistance_quantiles(cfg)
        assert len(result.F_sail_samples) == 100
        assert len(result.F_keel_samples) == 100
        assert len(result.F_total_samples) == 100
        assert len(result.grounding_depth_samples) == 100

    def test_quantiles_ordered(self):
        """P10 < P50 < P90 for all quantities."""
        cfg = RidgeResistanceQuantileConfig(n_samples=200, seed=42)
        result = compute_ridge_resistance_quantiles(cfg)

        for name in ("F_sail", "F_keel", "F_total", "grounding_depth"):
            q = result.quantiles[name]
            assert q["P10"] < q["P50"], f"{name}: P10 >= P50"
            assert q["P50"] < q["P90"], f"{name}: P50 >= P90"

    def test_total_is_sum(self):
        """F_total = F_sail + F_keel."""
        cfg = RidgeResistanceQuantileConfig(n_samples=100, seed=42)
        result = compute_ridge_resistance_quantiles(cfg)
        expected = result.F_sail_samples + result.F_keel_samples
        assert np.allclose(result.F_total_samples, expected)

    def test_positive_resistance(self):
        """All resistance samples should be positive."""
        cfg = RidgeResistanceQuantileConfig(n_samples=100, seed=42)
        result = compute_ridge_resistance_quantiles(cfg)
        assert np.all(result.F_sail_samples > 0)
        assert np.all(result.F_keel_samples > 0)
        assert np.all(result.F_total_samples > 0)

    def test_grounding_depth_less_than_keel(self):
        """Grounding depth < keel depth (due to settlement)."""
        cfg = RidgeResistanceQuantileConfig(
            n_samples=100, seed=42,
            keel_depth_mean=10.0, keel_depth_std=2.0,
        )
        result = compute_ridge_resistance_quantiles(cfg)
        # Grounding depth should be less than mean keel depth
        assert np.mean(result.grounding_depth_samples) < cfg.keel_depth_mean

    def test_statistics(self):
        """Statistics should contain expected keys."""
        cfg = RidgeResistanceQuantileConfig(n_samples=100, seed=42)
        result = compute_ridge_resistance_quantiles(cfg)

        for name in ("F_sail", "F_keel", "F_total", "grounding_depth"):
            assert "mean" in result.statistics[name]
            assert "std" in result.statistics[name]
            assert "min" in result.statistics[name]
            assert "max" in result.statistics[name]

    def test_sha256_reproducibility(self):
        """Same input → same SHA256."""
        cfg = RidgeResistanceQuantileConfig(n_samples=100, seed=42)
        result1 = compute_ridge_resistance_quantiles(cfg)
        result2 = compute_ridge_resistance_quantiles(cfg)

        # Compare quantiles
        q1 = json.dumps(result1.quantiles, sort_keys=True)
        q2 = json.dumps(result2.quantiles, sort_keys=True)
        assert hashlib.sha256(q1.encode()).hexdigest() == \
               hashlib.sha256(q2.encode()).hexdigest()


class TestGroundingDepthQuantiles:
    """Test grounding depth quantile computation."""

    def test_quantiles_ordered(self):
        """P10 < P50 < P90."""
        result = compute_grounding_depth_quantiles(n_samples=200, seed=42)
        assert result["P10"] < result["P50"] < result["P90"]

    def test_positive(self):
        """Grounding depth should be positive."""
        result = compute_grounding_depth_quantiles(n_samples=100, seed=42)
        assert result["P10"] > 0

    def test_less_than_mean_keel(self):
        """Mean grounding depth < mean keel depth."""
        result = compute_grounding_depth_quantiles(
            keel_depth_mean=10.0, n_samples=100, seed=42
        )
        assert result["mean"] < 10.0


class TestSensitivityAnalysis:
    """Test sensitivity analysis."""

    def test_structure(self):
        """Result should contain expected parameters."""
        cfg = RidgeResistanceQuantileConfig(n_samples=100, seed=42)
        result = sensitivity_analysis(cfg, n_samples=100)

        for param in ("sail_height_mean", "keel_depth_mean", "porosity_mean",
                      "sigma_f_mean", "phi_mean"):
            assert param in result
            assert "base_P50" in result[param]
            assert "plus_10pct_P50" in result[param]
            assert "minus_10pct_P50" in result[param]
            assert "sensitivity_index" in result[param]

    def test_finite_sensitivity(self):
        """Sensitivity indices should be finite."""
        cfg = RidgeResistanceQuantileConfig(n_samples=100, seed=42)
        result = sensitivity_analysis(cfg, n_samples=100)

        for param, data in result.items():
            assert math.isfinite(data["sensitivity_index"]), \
                f"{param}: sensitivity_index not finite"

    def test_sail_height_sensitive(self):
        """Sail height should have significant sensitivity."""
        cfg = RidgeResistanceQuantileConfig(n_samples=100, seed=42)
        result = sensitivity_analysis(cfg, n_samples=100)

        # Sail height should have non-zero sensitivity
        assert abs(result["sail_height_mean"]["sensitivity_index"]) > 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
