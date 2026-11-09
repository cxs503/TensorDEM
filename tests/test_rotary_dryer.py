"""Tests for rotary dryer module.

Test coverage:
- Kinematics: regime classification, Froude number
- Heat transfer: contact, radiation, Nusselt number
- RTD: residence time, variance, convergence
- Literature: correlation validation
- API: end-to-end calculation
- Edge cases: zero rotation, zero inclination, negative inputs
"""
from __future__ import annotations

import math
import pytest
import numpy as np

from tensordem.rotary_dryer import (
    RotaryDryerConfig,
    classify_regime,
    froude_number,
    omega_for_froude,
    contact_heat_transfer_coeff,
    radiation_heat_flux,
    total_heat_flux,
    nusselt_number,
    overall_heat_transfer_coeff,
    rtd_mean_residence_time,
    rtd_variance,
    rtd_curve,
    empirical_residence_time,
    tscheng_correlation,
    descoin_correlation,
    nusselt_literature,
    calculate_rotary_dryer,
)
from tensordem.rotary_dryer.kinematics import BedRegime
from tensordem.rotary_dryer.api import validate_energy_balance, evidence_grade


# =============================================================================
# Kinematics tests
# =============================================================================

class TestKinematics:
    """Tests for bed kinematics and regime classification."""

    def test_froude_number_calculation(self):
        """Test Froude number calculation."""
        # Fr = ω²R/g
        omega = 1.0  # rad/s
        R = 1.0  # m
        g = 9.81  # m/s²
        Fr = froude_number(omega, R, g)
        expected = omega ** 2 * R / g
        assert abs(Fr - expected) < 1e-10

    def test_froude_number_zero_omega(self):
        """Test Froude number with zero rotation."""
        Fr = froude_number(0.0, 1.0)
        assert Fr == 0.0

    def test_froude_number_invalid_radius(self):
        """Test Froude number with invalid radius."""
        with pytest.raises(ValueError, match="radius must be positive"):
            froude_number(1.0, 0.0)
        with pytest.raises(ValueError, match="radius must be positive"):
            froude_number(1.0, -1.0)

    def test_omega_for_froude(self):
        """Test inverse Froude calculation."""
        Fr_target = 0.5
        R = 0.75
        omega = omega_for_froude(Fr_target, R)
        Fr_calc = froude_number(omega, R)
        assert abs(Fr_calc - Fr_target) < 1e-10

    def test_regime_slipping(self):
        """Test slipping regime classification."""
        # Fr < 0.05
        assert classify_regime(0.01) == BedRegime.SLIPPING
        assert classify_regime(0.04) == BedRegime.SLIPPING

    def test_regime_rolling(self):
        """Test rolling regime classification."""
        # 0.05 <= Fr < 0.4
        assert classify_regime(0.05) == BedRegime.ROLLING
        assert classify_regime(0.2) == BedRegime.ROLLING
        assert classify_regime(0.39) == BedRegime.ROLLING

    def test_regime_cataracting(self):
        """Test cataracting regime classification."""
        # 0.4 <= Fr < 1.0
        assert classify_regime(0.4) == BedRegime.CATARACTING
        assert classify_regime(0.6) == BedRegime.CATARACTING
        assert classify_regime(0.99) == BedRegime.CATARACTING

    def test_regime_centrifuging(self):
        """Test centrifuging regime classification."""
        # Fr >= 1.0
        assert classify_regime(1.0) == BedRegime.CENTRIFUGING
        assert classify_regime(1.5) == BedRegime.CENTRIFUGING
        assert classify_regime(2.0) == BedRegime.CENTRIFUGING

    def test_regime_boundaries_exact(self):
        """Test exact boundary values."""
        assert classify_regime(0.05) == BedRegime.ROLLING  # boundary
        assert classify_regime(0.4) == BedRegime.CATARACTING  # boundary
        assert classify_regime(1.0) == BedRegime.CENTRIFUGING  # boundary

    def test_regime_invalid_negative(self):
        """Test regime with negative Froude number."""
        with pytest.raises(ValueError, match="non-negative"):
            classify_regime(-0.1)

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        cfg = RotaryDryerConfig()
        assert cfg.drum_diameter > 0
        assert cfg.omega >= 0

        # Invalid diameter
        with pytest.raises(ValueError):
            RotaryDryerConfig(drum_diameter=0.0)

        # Invalid omega
        with pytest.raises(ValueError):
            RotaryDryerConfig(omega=-1.0)

        # Invalid fill fraction
        with pytest.raises(ValueError):
            RotaryDryerConfig(fill_fraction=0.0)
        with pytest.raises(ValueError):
            RotaryDryerConfig(fill_fraction=1.0)

    def test_config_properties(self):
        """Test configuration derived properties."""
        cfg = RotaryDryerConfig(
            drum_diameter=2.0,
            omega=1.0,
            inclination=0.1,
        )
        assert cfg.radius == 1.0
        assert cfg.froude > 0
        assert cfg.rpm > 0
        assert cfg.inclination_deg > 0


# =============================================================================
# Heat transfer tests
# =============================================================================

class TestHeatTransfer:
    """Tests for heat transfer models."""

    def test_contact_heat_transfer_coefficient(self):
        """Test contact heat transfer coefficient calculation."""
        h = contact_heat_transfer_coeff(
            k_p=1.5, k_w=50.0, k_g=0.026, d_p=0.005, A_c=1e-10
        )
        assert h > 0
        assert math.isfinite(h)

    def test_radiation_heat_flux(self):
        """Test radiation heat flux calculation."""
        T_w = 500.0  # K
        T_p = 300.0  # K
        eps = 0.9
        q = radiation_heat_flux(T_w, T_p, eps)
        assert q > 0  # Heat flows from hot wall to cold particle
        assert math.isfinite(q)

    def test_radiation_zero_temperature_diff(self):
        """Test radiation with equal temperatures."""
        T = 400.0
        q = radiation_heat_flux(T, T, 0.9)
        assert abs(q) < 1e-10

    def test_radiation_invalid_temperature(self):
        """Test radiation with invalid temperature."""
        with pytest.raises(ValueError, match="positive"):
            radiation_heat_flux(0.0, 300.0, 0.9)
        with pytest.raises(ValueError, match="positive"):
            radiation_heat_flux(500.0, -100.0, 0.9)

    def test_total_heat_flux(self):
        """Test total heat flux calculation."""
        cfg = RotaryDryerConfig(wall_temperature=500.0, particle_initial_temperature=300.0)
        result = total_heat_flux(cfg, T_p=350.0)
        assert "q_contact" in result
        assert "q_rad" in result
        assert "q_total" in result
        assert result["q_total"] > 0  # Net heat into particle

    def test_nusselt_number(self):
        """Test Nusselt number calculation."""
        h = 100.0  # W/(m²·K)
        d_p = 0.005  # m
        k_g = 0.026  # W/(m·K)
        Nu = nusselt_number(h, d_p, k_g)
        expected = h * d_p / k_g
        assert abs(Nu - expected) < 1e-10

    def test_nusselt_invalid(self):
        """Test Nusselt number with invalid inputs."""
        with pytest.raises(ValueError):
            nusselt_number(100.0, 0.005, 0.0)
        with pytest.raises(ValueError):
            nusselt_number(100.0, 0.0, 0.026)

    def test_overall_heat_transfer(self):
        """Test overall heat transfer coefficient."""
        cfg = RotaryDryerConfig()
        result = overall_heat_transfer_coeff(cfg)
        assert "nusselt_number" in result
        assert result["nusselt_number"] > 0
        assert "regime" in result


# =============================================================================
# RTD tests
# =============================================================================

class TestRTD:
    """Tests for residence time distribution."""

    def test_empirical_residence_time_basic(self):
        """Test basic empirical residence time."""
        tau = empirical_residence_time(
            L=10.0, omega=0.5, R=0.75, beta=0.05
        )
        assert tau > 0
        assert math.isfinite(tau)

    def test_empirical_residence_time_zero_omega(self):
        """Test residence time with zero rotation."""
        tau = empirical_residence_time(
            L=10.0, omega=0.0, R=0.75, beta=0.05
        )
        assert tau == float("inf")

    def test_empirical_residence_time_zero_inclination(self):
        """Test residence time with zero inclination."""
        with pytest.raises(ValueError, match="inclination must be positive"):
            empirical_residence_time(L=10.0, omega=0.5, R=0.75, beta=0.0)

    def test_rtd_mean_residence_time(self):
        """Test RTD mean calculation."""
        # Gaussian RTD centered at 100 s
        t = np.linspace(0, 200, 1000)
        tau_mean = 100.0
        sigma = 10.0
        C = np.exp(-0.5 * ((t - tau_mean) / sigma) ** 2)
        tau_calc = rtd_mean_residence_time(t, C)
        assert abs(tau_calc - tau_mean) < 1.0  # Within 1 s

    def test_rtd_variance(self):
        """Test RTD variance calculation."""
        t = np.linspace(0, 200, 1000)
        tau_mean = 100.0
        sigma = 10.0
        C = np.exp(-0.5 * ((t - tau_mean) / sigma) ** 2)
        var_calc = rtd_variance(t, C, tau_mean)
        expected_var = sigma ** 2
        assert abs(var_calc - expected_var) < 10.0  # Within 10 s²

    def test_rtd_curve_gaussian(self):
        """Test Gaussian RTD curve."""
        t = np.linspace(0, 200, 100)
        E = rtd_curve(100.0, 10.0, t, model="gaussian")
        assert len(E) == len(t)
        assert np.all(E >= 0)
        # Should integrate to ~1
        integral = np.trapezoid(E, t)
        assert abs(integral - 1.0) < 0.1

    def test_rtd_curve_exponential(self):
        """Test exponential RTD curve."""
        t = np.linspace(0, 500, 100)
        E = rtd_curve(100.0, 100.0, t, model="exponential")
        assert len(E) == len(t)
        assert np.all(E >= 0)

    def test_rtd_curve_invalid_model(self):
        """Test RTD curve with invalid model."""
        t = np.linspace(0, 200, 100)
        with pytest.raises(ValueError, match="Unknown RTD model"):
            rtd_curve(100.0, 10.0, t, model="invalid")


# =============================================================================
# Literature correlation tests
# =============================================================================

class TestLiterature:
    """Tests for literature correlations."""

    def test_tscheng_correlation(self):
        """Test Tscheng correlation."""
        Re_p = 100.0
        Pr = 0.71
        fill = 0.15
        Nu = tscheng_correlation(Re_p, Pr, fill)
        assert Nu > 2.0  # Above conduction limit
        assert math.isfinite(Nu)

    def test_tscheng_correlation_invalid(self):
        """Test Tscheng with invalid inputs."""
        with pytest.raises(ValueError, match="non-negative"):
            tscheng_correlation(-10.0, 0.71, 0.15)
        with pytest.raises(ValueError, match="fill fraction"):
            tscheng_correlation(100.0, 0.71, 0.0)

    def test_descoin_correlation(self):
        """Test Descoin correlation."""
        Re_p = 100.0
        Pr = 0.71
        Nu = descoin_correlation(Re_p, Pr, fill=0.15, omega=0.5, R=0.75)
        assert Nu >= 2.0  # At least conduction limit
        assert math.isfinite(Nu)

    def test_nusselt_literature(self):
        """Test literature Nusselt numbers."""
        cfg = RotaryDryerConfig()
        result = nusselt_literature(cfg)
        assert "nusselt_tscheng" in result
        assert "nusselt_descoin" in result
        assert "reynolds_number" in result
        assert "prandtl_number" in result
        assert result["nusselt_tscheng"] > 0
        assert result["nusselt_descoin"] > 0


# =============================================================================
# API tests
# =============================================================================

class TestAPI:
    """Tests for the main API."""

    def test_calculate_rotary_dryer_basic(self):
        """Test basic calculation."""
        result = calculate_rotary_dryer(
            drum_diameter=1.5,
            drum_length=10.0,
            omega=0.5,
            inclination=0.05,
        )
        assert "kinematics" in result
        assert "rtd" in result
        assert "heat_transfer" in result
        assert "energy_balance" in result
        assert "evidence" in result
        assert "summary" in result

    def test_calculate_rolling_regime(self):
        """Test calculation in rolling regime."""
        # Fr = 0.2 -> rolling
        R = 0.75
        omega = omega_for_froude(0.2, R)
        result = calculate_rotary_dryer(
            drum_diameter=2 * R,
            omega=omega,
            inclination=0.05,
        )
        assert result["kinematics"]["regime"] == "rolling"

    def test_calculate_cataracting_regime(self):
        """Test calculation in cataracting regime."""
        # Fr = 0.6 -> cataracting
        R = 0.75
        omega = omega_for_froude(0.6, R)
        result = calculate_rotary_dryer(
            drum_diameter=2 * R,
            omega=omega,
            inclination=0.05,
        )
        assert result["kinematics"]["regime"] == "cataracting"

    def test_calculate_zero_rotation(self):
        """Test calculation with zero rotation (boundary case)."""
        result = calculate_rotary_dryer(
            omega=0.0,
            inclination=0.05,
        )
        # Should handle gracefully
        assert result["kinematics"]["froude_number"] == 0.0
        assert result["kinematics"]["regime"] == "slipping"

    def test_calculate_zero_inclination(self):
        """Test calculation with zero inclination (boundary case)."""
        # Zero inclination means no axial transport
        # RTD should be infinite or very large
        result = calculate_rotary_dryer(
            omega=0.5,
            inclination=0.001,  # Very small but non-zero
        )
        assert "rtd" in result

    def test_energy_balance(self):
        """Test energy balance validation."""
        cfg = RotaryDryerConfig()
        tau = 100.0  # s
        h_eff = 50.0  # W/(m²·K)
        result = validate_energy_balance(cfg, tau, h_eff)
        assert "Q_particles_W" in result
        assert "Q_wall_capacity_W" in result
        assert "residual" in result
        assert result["residual"] >= 0

    def test_evidence_grading(self):
        """Test evidence grading."""
        cfg = RotaryDryerConfig()
        Nu_pred = 50.0
        Nu_lit = {"nusselt_tscheng": 55.0}
        tau_pred = 100.0
        tau_emp = 105.0
        grades = evidence_grade(cfg, Nu_pred, Nu_lit, tau_pred, tau_emp)
        assert "regime_classification" in grades
        assert "residence_time" in grades
        assert "heat_transfer" in grades

    def test_convergence_study(self):
        """Test RTD convergence study."""
        result = calculate_rotary_dryer(
            convergence_check=True,
            n_tracers=500,
        )
        assert "convergence" in result["rtd"]
        conv = result["rtd"]["convergence"]
        assert len(conv["n_particles"]) == 3
        assert len(conv["tau_mean"]) == 3


# =============================================================================
# Negative control tests
# =============================================================================

class TestNegativeControls:
    """Negative control tests - should raise errors or handle gracefully."""

    def test_negative_omega(self):
        """Test negative angular velocity."""
        with pytest.raises(ValueError, match="non-negative"):
            RotaryDryerConfig(omega=-1.0)

    def test_negative_diameter(self):
        """Test negative drum diameter."""
        with pytest.raises(ValueError, match="positive"):
            RotaryDryerConfig(drum_diameter=-1.0)

    def test_invalid_fill_fraction(self):
        """Test invalid fill fraction."""
        with pytest.raises(ValueError):
            RotaryDryerConfig(fill_fraction=-0.1)
        with pytest.raises(ValueError):
            RotaryDryerConfig(fill_fraction=1.5)

    def test_invalid_restitution(self):
        """Test invalid restitution coefficient."""
        with pytest.raises(ValueError):
            RotaryDryerConfig(restitution=0.0)
        with pytest.raises(ValueError):
            RotaryDryerConfig(restitution=1.5)


# =============================================================================
# Integration tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_calculation_chain(self):
        """Test full calculation from config to results."""
        # Create config
        cfg = RotaryDryerConfig(
            drum_diameter=1.5,
            drum_length=10.0,
            omega=0.5,
            inclination=0.05,
            particle_diameter=0.005,
            wall_temperature=473.15,
        )

        # Check kinematics
        Fr = cfg.froude
        regime = cfg.regime
        assert Fr > 0
        assert regime in BedRegime

        # Check heat transfer
        heat = overall_heat_transfer_coeff(cfg)
        assert heat["nusselt_number"] > 0

        # Check RTD
        tau = empirical_residence_time(
            cfg.drum_length, cfg.omega, cfg.radius, cfg.inclination
        )
        assert tau > 0

        # Check literature
        lit = nusselt_literature(cfg)
        assert lit["nusselt_tscheng"] > 0

    def test_regime_transitions(self):
        """Test regime transitions with varying Froude number."""
        cfg = RotaryDryerConfig(drum_diameter=1.5)

        # Test all regimes
        fr_values = [0.01, 0.1, 0.5, 1.5]
        expected_regimes = ["slipping", "rolling", "cataracting", "centrifuging"]

        for fr, expected in zip(fr_values, expected_regimes):
            omega = omega_for_froude(fr, cfg.radius)
            cfg_test = RotaryDryerConfig(drum_diameter=1.5, omega=omega)
            assert cfg_test.regime.value == expected
