"""Tests for ice crushing against vertical structures (ISO 19906 benchmark).

Campaign node: p228w1-N04
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tensordem.ice_crushing_vertical_structure import (
    ISO19906Config,
    iso19906_nominal_load,
    iso19906_load_range,
    PorosityScalingConfig,
    fit_porosity_scaling,
    run_crushing_case,
    run_porosity_sweep,
)


class TestISO19906:
    """Test ISO 19906 ice load calculations."""

    def test_nominal_load_cylinder(self):
        """Test nominal load for cylinder."""
        cfg = ISO19906Config(sigma_c=3.0e6, h=1.0, D=5.0, shape="cylinder")
        F = iso19906_nominal_load(cfg)
        # F = C * σ_c * h * D = 2.0 * 3e6 * 1.0 * 5.0 = 30e6 N
        assert abs(F - 30.0e6) < 1.0e3

    def test_nominal_load_square(self):
        """Test nominal load for square column."""
        cfg = ISO19906Config(sigma_c=3.0e6, h=1.0, D=5.0, shape="square")
        F = iso19906_nominal_load(cfg)
        # F = C * σ_c * h * D = 2.5 * 3e6 * 1.0 * 5.0 = 37.5e6 N
        assert abs(F - 37.5e6) < 1.0e3

    def test_custom_shape_coefficient(self):
        """Test custom shape coefficient."""
        cfg = ISO19906Config(sigma_c=3.0e6, h=1.0, D=5.0, shape="cylinder", C=3.0)
        F = iso19906_nominal_load(cfg)
        # F = 3.0 * 3e6 * 1.0 * 5.0 = 45e6 N
        assert abs(F - 45.0e6) < 1.0e3

    def test_load_range(self):
        """Test load range calculation."""
        result = iso19906_load_range(
            sigma_c_range=(2.0e6, 4.0e6),
            h=1.0,
            D=5.0,
            shape="cylinder",
        )
        assert result["F_min"] < result["F_max"]
        assert result["F_mean"] == 0.5 * (result["F_min"] + result["F_max"])

    def test_invalid_parameters(self):
        """Test invalid parameter validation."""
        with pytest.raises(ValueError):
            ISO19906Config(sigma_c=-1.0, h=1.0, D=5.0)
        with pytest.raises(ValueError):
            ISO19906Config(sigma_c=3.0e6, h=0.0, D=5.0)
        with pytest.raises(ValueError):
            ISO19906Config(sigma_c=3.0e6, h=1.0, D=-5.0)
        with pytest.raises(ValueError):
            ISO19906Config(sigma_c=3.0e6, h=1.0, D=5.0, shape="invalid")


class TestPorosityScaling:
    """Test porosity-resistance scaling law."""

    def test_fit_scaling_linear(self):
        """Test fitting with α=1 (linear)."""
        porosity = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        F_ISO = 30.0e6
        # F = F_ISO * (1-n)^1
        F_steady = F_ISO * (1.0 - porosity)
        alpha, alpha_std, R2 = fit_porosity_scaling(porosity, F_steady, F_ISO)
        assert abs(alpha - 1.0) < 0.01
        assert R2 > 0.99

    def test_fit_scaling_quadratic(self):
        """Test fitting with α=2 (quadratic)."""
        porosity = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        F_ISO = 30.0e6
        # F = F_ISO * (1-n)^2
        F_steady = F_ISO * (1.0 - porosity) ** 2
        alpha, alpha_std, R2 = fit_porosity_scaling(porosity, F_steady, F_ISO)
        assert abs(alpha - 2.0) < 0.01
        assert R2 > 0.99

    def test_fit_scaling_intermediate(self):
        """Test fitting with α=1.5."""
        porosity = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        F_ISO = 30.0e6
        F_steady = F_ISO * (1.0 - porosity) ** 1.5
        alpha, alpha_std, R2 = fit_porosity_scaling(porosity, F_steady, F_ISO)
        assert abs(alpha - 1.5) < 0.01
        assert R2 > 0.99


class TestCrushingCase:
    """Test DEM crushing simulation."""

    def test_basic_case(self):
        """Test basic crushing case runs without error."""
        result = run_crushing_case(
            sigma_c=3.0e6,
            h=1.0,
            D=5.0,
            shape="cylinder",
            n_particles=50,
            duration=5.0,
            seed=42,
        )
        assert result.F_max > 0
        assert result.F_steady > 0
        assert 0.0 < result.porosity < 1.0
        assert result.F_ISO > 0
        assert len(result.time) > 0
        assert len(result.F_x) > 0

    def test_peak_force_positive(self):
        """Test that peak force is positive."""
        result = run_crushing_case(
            sigma_c=3.0e6,
            h=1.0,
            D=5.0,
            n_particles=50,
            duration=5.0,
        )
        assert result.F_max > 0

    def test_porosity_range(self):
        """Test that porosity is in valid range."""
        result = run_crushing_case(
            sigma_c=3.0e6,
            h=1.0,
            D=5.0,
            n_particles=50,
            duration=5.0,
        )
        assert 0.0 < result.porosity < 1.0

    def test_different_shapes(self):
        """Test cylinder and square shapes."""
        result_cyl = run_crushing_case(
            sigma_c=3.0e6,
            h=1.0,
            D=5.0,
            shape="cylinder",
            n_particles=50,
            duration=5.0,
        )
        result_sq = run_crushing_case(
            sigma_c=3.0e6,
            h=1.0,
            D=5.0,
            shape="square",
            n_particles=50,
            duration=5.0,
        )
        assert result_cyl.F_max > 0
        assert result_sq.F_max > 0

    def test_reproducibility(self):
        """Test that same seed gives same results."""
        result1 = run_crushing_case(
            sigma_c=3.0e6,
            h=1.0,
            D=5.0,
            n_particles=50,
            duration=5.0,
            seed=42,
        )
        result2 = run_crushing_case(
            sigma_c=3.0e6,
            h=1.0,
            D=5.0,
            n_particles=50,
            duration=5.0,
            seed=42,
        )
        assert result1.F_max == result2.F_max
        assert result1.F_steady == result2.F_steady


class TestPorositySweep:
    """Test porosity sweep functionality."""

    def test_sweep_runs(self):
        """Test that sweep runs without error."""
        config = PorosityScalingConfig(
            sigma_c=3.0e6,
            h=1.0,
            D=5.0,
            n_porosity=4,
            n_particles=50,
            drift_speed=0.5,
        )
        result = run_porosity_sweep(config)
        assert len(result.porosity) == 4
        assert len(result.F_steady) == 4
        assert result.alpha != 0
        assert result.R2 is not None

    def test_sweep_scaling_exponent_range(self):
        """Test that scaling exponent is in expected range."""
        config = PorosityScalingConfig(
            sigma_c=3.0e6,
            h=1.0,
            D=5.0,
            n_porosity=6,
            n_particles=80,
            drift_speed=0.5,
        )
        result = run_porosity_sweep(config)
        # α should be in [1.0, 2.0] for physical ice rubble
        # Allow some tolerance for DEM variability
        assert 0.5 <= result.alpha <= 2.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
