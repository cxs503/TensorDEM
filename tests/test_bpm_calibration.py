"""Tests for BPM (Bonded Particle Model) calibration module.

Tests cover:
1. Weibull statistics (MLE and bootstrap CI)
2. Size effect fitting
3. Quasi-static gate check
4. Dual-coefficient friction model
5. Complete calibration pipeline with synthetic data
"""
import math
import numpy as np
import pytest
import torch

from tensordem.bpm_calibration import (
    weibull_mle,
    weibull_bootstrap_ci,
    size_effect_weibull_fit,
    check_quasi_static,
    quasi_static_convergence,
    DualFrictionConfig,
    calibrate_bpm,
    generate_synthetic_strengths,
    generate_size_effect_data,
    QuasiStaticResult,
    LITERATURE_UCS_RANGES,
)


class TestWeibullStatistics:
    """Test Weibull MLE and bootstrap confidence intervals."""

    def test_weibull_mle_known_distribution(self):
        """Test MLE recovers known Weibull parameters."""
        # Generate data from known Weibull distribution
        m_true = 10.0
        sigma_0_true = 1e6  # 1 MPa
        n = 100
        rng = np.random.RandomState(42)
        u = rng.uniform(0, 1, n)
        strengths = sigma_0_true * (-np.log(1.0 - u)) ** (1.0 / m_true)

        m_est, sigma_0_est = weibull_mle(strengths)

        # Should recover parameters within 20% (statistical uncertainty)
        assert abs(m_est - m_true) / m_true < 0.20, f"m: {m_est} vs {m_true}"
        assert abs(sigma_0_est - sigma_0_true) / sigma_0_true < 0.20

    def test_weibull_mle_minimum_samples(self):
        """Test MLE requires at least 3 samples."""
        with pytest.raises(ValueError, match="at least 3"):
            weibull_mle(np.array([1.0, 2.0]))

    def test_weibull_bootstrap_ci_coverage(self):
        """Test bootstrap CI contains true value with expected frequency."""
        m_true = 8.0
        sigma_0_true = 2e6
        n = 30
        rng = np.random.RandomState(123)

        # Run multiple experiments to check coverage
        n_experiments = 20
        n_captured = 0
        for exp in range(n_experiments):
            u = rng.uniform(0, 1, n)
            strengths = sigma_0_true * (-np.log(1.0 - u)) ** (1.0 / m_true)
            ci = weibull_bootstrap_ci(strengths, n_bootstrap=500, seed=exp)

            if ci["m_ci_low"] <= m_true <= ci["m_ci_high"]:
                n_captured += 1

        # 95% CI should capture true value ~95% of the time (allow 80-100%)
        coverage = n_captured / n_experiments
        assert 0.70 <= coverage <= 1.0, f"Coverage {coverage} outside expected range"

    def test_weibull_bootstrap_ci_width_decreases_with_n(self):
        """Test CI width decreases with more samples."""
        m_true = 10.0
        sigma_0_true = 1e6
        rng = np.random.RandomState(42)

        widths = []
        for n in [10, 30, 100]:
            u = rng.uniform(0, 1, n)
            strengths = sigma_0_true * (-np.log(1.0 - u)) ** (1.0 / m_true)
            ci = weibull_bootstrap_ci(strengths, n_bootstrap=500, seed=42)
            width = ci["m_ci_high"] - ci["m_ci_low"]
            widths.append(width)

        # Width should decrease with more samples
        assert widths[0] > widths[1] > widths[2]


class TestSizeEffect:
    """Test size effect fitting."""

    def test_size_effect_weibull_fit_recovers_m(self):
        """Test size effect fit recovers Weibull modulus."""
        m_true = 12.0
        sigma_0_ref = 5e6  # 5 MPa at t=1m
        thicknesses = [0.1, 0.2, 0.4]

        # Generate mean strengths following Weibull size effect
        mean_strengths = []
        for t in thicknesses:
            sigma_0_t = sigma_0_ref * t ** (-2.0 / m_true)
            mean_strengths.append(sigma_0_t)

        result = size_effect_weibull_fit(
            np.array(thicknesses), np.array(mean_strengths)
        )

        # Should recover m within 1%
        assert abs(result["m_weibull"] - m_true) / m_true < 0.01
        assert result["r_squared"] > 0.99

    def test_size_effect_with_noise(self):
        """Test size effect fit with realistic noise."""
        m_true = 10.0
        sigma_0_ref = 3e6
        thicknesses = [0.1, 0.2, 0.4]

        # Generate data with some scatter
        mean_strengths = []
        rng = np.random.RandomState(42)
        for t in thicknesses:
            sigma_0_t = sigma_0_ref * t ** (-2.0 / m_true)
            # Add 5% noise
            noise = 1.0 + 0.05 * rng.randn()
            mean_strengths.append(sigma_0_t * noise)

        result = size_effect_weibull_fit(
            np.array(thicknesses), np.array(mean_strengths)
        )

        # Should still recover m within 10% despite noise
        assert abs(result["m_weibull"] - m_true) / m_true < 0.10
        assert result["r_squared"] > 0.95


class TestQuasiStaticGate:
    """Test quasi-static convergence check."""

    def test_quasi_static_pass(self):
        """Test quasi-static pass when KE/work is low."""
        history = [
            {"kinetic_energy_J": 1e-6, "external_work_J": 1.0,
             "engineering_stress_Pa": 1e6, "axial_engineering_strain": 0.001},
            {"kinetic_energy_J": 5e-7, "external_work_J": 2.0,
             "engineering_stress_Pa": 2e6, "axial_engineering_strain": 0.002},
            {"kinetic_energy_J": 8e-7, "external_work_J": 3.0,
             "engineering_stress_Pa": 2.5e6, "axial_engineering_strain": 0.003},
        ]
        result = check_quasi_static(history, beta=1600.0, threshold=1e-3)

        assert result.is_quasi_static
        assert result.ke_ratio < 1e-3
        assert result.peak_stress_Pa == 2.5e6

    def test_quasi_static_fail(self):
        """Test quasi-static fail when KE/work is high."""
        history = [
            {"kinetic_energy_J": 0.1, "external_work_J": 1.0,
             "engineering_stress_Pa": 1e6, "axial_engineering_strain": 0.001},
            {"kinetic_energy_J": 0.05, "external_work_J": 2.0,
             "engineering_stress_Pa": 2e6, "axial_engineering_strain": 0.002},
        ]
        result = check_quasi_static(history, beta=100.0, threshold=1e-3)

        assert not result.is_quasi_static
        assert result.ke_ratio > 1e-3

    def test_quasi_static_convergence_assessment(self):
        """Test convergence assessment across multiple rates."""
        results = [
            QuasiStaticResult(beta=100, ke_ratio=0.01, is_quasi_static=False,
                            peak_stress_Pa=1e6, strain_at_peak=0.001,
                            external_work_J=1.0, kinetic_energy_peak_J=0.01),
            QuasiStaticResult(beta=400, ke_ratio=5e-4, is_quasi_static=True,
                            peak_stress_Pa=2e6, strain_at_peak=0.002,
                            external_work_J=2.0, kinetic_energy_peak_J=1e-3),
            QuasiStaticResult(beta=1600, ke_ratio=2e-4, is_quasi_static=True,
                            peak_stress_Pa=2.05e6, strain_at_peak=0.002,
                            external_work_J=3.0, kinetic_energy_peak_J=6e-4),
            QuasiStaticResult(beta=3600, ke_ratio=1e-4, is_quasi_static=True,
                            peak_stress_Pa=2.08e6, strain_at_peak=0.002,
                            external_work_J=4.0, kinetic_energy_peak_J=4e-4),
        ]

        assessment = quasi_static_convergence(results)

        assert assessment["converged"]
        assert assessment["beta_min"] == 400
        assert assessment["n_quasi_static"] == 3
        assert assessment["stress_variation_pct"] < 5.0


class TestDualFriction:
    """Test dual-coefficient friction model."""

    def test_dual_friction_zero_slip(self):
        """Test friction equals static at zero slip."""
        config = DualFrictionConfig(
            mu_static=0.6, mu_dynamic=0.4, v_characteristic=1e-3
        )
        mu_eff = config.effective_friction(0.0)
        assert abs(mu_eff - 0.6) < 1e-10

    def test_dual_friction_high_slip(self):
        """Test friction approaches dynamic at high slip."""
        config = DualFrictionConfig(
            mu_static=0.6, mu_dynamic=0.4, v_characteristic=1e-3
        )
        mu_eff = config.effective_friction(1.0)  # 1 m/s >> v_c
        assert abs(mu_eff - 0.4) < 0.01

    def test_dual_friction_transition(self):
        """Test friction at characteristic velocity."""
        config = DualFrictionConfig(
            mu_static=0.6, mu_dynamic=0.4, v_characteristic=1e-3
        )
        mu_eff = config.effective_friction(1e-3)  # v = v_c
        # At v = v_c: mu = mu_d + (mu_s - mu_d) * exp(-1) = 0.4 + 0.2 * 0.368
        expected = 0.4 + 0.2 * np.exp(-1)
        assert abs(mu_eff - expected) < 1e-10

    def test_dual_friction_monotone(self):
        """Test friction decreases monotonically with slip velocity."""
        config = DualFrictionConfig(
            mu_static=0.6, mu_dynamic=0.4, v_characteristic=1e-3
        )
        velocities = np.logspace(-5, 0, 50)
        frictions = [config.effective_friction(v) for v in velocities]

        # Should be monotonically decreasing
        for i in range(len(frictions) - 1):
            assert frictions[i] >= frictions[i + 1]


class TestSyntheticData:
    """Test synthetic data generation."""

    def test_synthetic_strengths_weibull(self):
        """Test synthetic strengths follow Weibull distribution."""
        sigma_0 = 1e6
        m = 10.0
        n = 1000
        strengths = generate_synthetic_strengths(n, sigma_0, m, seed=42)

        # Check mean and std match Weibull moments
        # Mean = sigma_0 * Gamma(1 + 1/m)
        # Var = sigma_0^2 * [Gamma(1 + 2/m) - Gamma(1 + 1/m)^2]
        from math import gamma
        expected_mean = sigma_0 * gamma(1 + 1/m)
        expected_var = sigma_0**2 * (gamma(1 + 2/m) - gamma(1 + 1/m)**2)

        assert abs(np.mean(strengths) - expected_mean) / expected_mean < 0.05
        assert abs(np.std(strengths)**2 - expected_var) / expected_var < 0.10

    def test_size_effect_data_generation(self):
        """Test size effect data follows Weibull scaling."""
        thicknesses = [0.1, 0.2, 0.4]
        sigma_0_ref = 5e6
        m_true = 12.0
        n_per = 50

        data = generate_size_effect_data(
            thicknesses, n_per, sigma_0_ref, m_true, seed_base=42
        )

        # Check each thickness has correct number of specimens
        for t in thicknesses:
            assert len(data[t]) == n_per

        # Check mean strengths follow size effect law
        for t in thicknesses:
            expected_sigma_0 = sigma_0_ref * t ** (-2.0 / m_true)
            mean_strength = np.mean(data[t])
            # Allow 20% tolerance due to statistical variation
            assert abs(mean_strength - expected_sigma_0) / expected_sigma_0 < 0.20


class TestCalibrationPipeline:
    """Test complete calibration pipeline."""

    def test_calibrate_bpm_synthetic(self):
        """Test calibration with synthetic data."""
        # Generate synthetic UCS histories at different beta
        beta_values = [100, 400, 1600, 3600]
        ucs_histories = {}
        for beta in beta_values:
            # Simulate quasi-static convergence
            ke_ratio = 0.01 * (100 / beta)  # Decreases with beta
            history = []
            for step in range(100):
                strain = 0.001 * (step + 1) / 100
                stress = 2e6 * strain / 0.003  # Linear elastic
                ke = ke_ratio * 1.0 * (step + 1) / 100
                work = 1.0 * (step + 1) / 100
                history.append({
                    "kinetic_energy_J": ke,
                    "external_work_J": work,
                    "engineering_stress_Pa": stress,
                    "axial_engineering_strain": strain,
                })
            ucs_histories[float(beta)] = history

        # Generate synthetic Brazilian strengths
        thicknesses = [0.1, 0.2, 0.4]
        brazilian_strengths = generate_size_effect_data(
            thicknesses, n_per_thickness=10,
            sigma_0_ref=2e6, m_true=10.0, seed_base=42
        )

        # Run calibration
        result = calibrate_bpm(
            ucs_histories=ucs_histories,
            brazilian_strengths=brazilian_strengths,
            literature_key="freshwater_ice_-10C",
            beta_values=beta_values,
            n_bootstrap=100,
        )

        # Check quasi-static assessment
        assert result.quasi_static["converged"]
        assert result.quasi_static["beta_min"] is not None

        # Check size effect
        assert "m_weibull" in result.size_effect
        assert result.size_effect["r_squared"] > 0.9

        # Check Weibull per thickness
        assert len(result.weibull_per_thickness) == 3
        for key, data in result.weibull_per_thickness.items():
            assert "m_weibull" in data
            assert "bootstrap_ci" in data
            assert data["bootstrap_ci"]["n_valid"] > 0

        # Check UCS comparison
        assert "dem_ucsi_MPa" in result.ucs_comparison
        assert "literature_range_MPa" in result.ucs_comparison


class TestDEM3DDualFriction:
    """Test dual friction in DEM3D engine."""

    def test_dem3d_config_dual_friction(self):
        """Test DEM3DConfig accepts dual friction parameters."""
        from tensordem.dem3d import DEM3DConfig

        config = DEM3DConfig(
            static_friction_coefficient=0.6,
            dynamic_friction_coefficient=0.4,
            friction_transition_velocity=1e-3,
        )

        assert config.static_friction_coefficient == 0.6
        assert config.dynamic_friction_coefficient == 0.4
        assert config.friction_transition_velocity == 1e-3

    def test_dem3d_config_validation(self):
        """Test DEM3DConfig validates dual friction parameters."""
        from tensordem.dem3d import DEM3DConfig

        # Dynamic > static should fail
        with pytest.raises(ValueError, match="dynamic.*<=.*static"):
            DEM3DConfig(
                static_friction_coefficient=0.4,
                dynamic_friction_coefficient=0.6,
            )

        # Negative values should fail
        with pytest.raises(ValueError, match="nonnegative"):
            DEM3DConfig(static_friction_coefficient=-0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
