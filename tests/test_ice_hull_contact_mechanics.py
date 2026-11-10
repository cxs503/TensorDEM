"""Tests for ice-hull contact friction and adhesion mechanics.

Verification suite covering:
1. Friction model μ(p, v, T) - pressure, velocity, temperature dependence
2. Adhesion model τ_ad(T, p, t) - temperature, pressure, time dependence
3. Shear strength τ_shear = min(μ·p + τ_ad, τ_cohesive)
4. Slope sliding: friction angle θ vs T (error < 5°)
5. Tension test: adhesion force F_ad vs t (error < 20%)
6. Shear test: shear strength τ vs p (error < 15%)
7. Velocity dependence: μ(v) (error < 20%)
8. Energy conservation: friction work = heat (error < 10%)
"""
from __future__ import annotations

import math
import pytest
import numpy as np

from tensordem.ice_hull_contact_mechanics import (
    IceHullFrictionParams,
    IceHullAdhesionParams,
    IceHullContactModel,
    IceHullContactState,
    compute_friction_coefficient,
    compute_adhesion_stress,
    compute_shear_strength,
    ice_hull_contact_force,
    run_slope_sliding_verification,
    run_tension_test_verification,
    run_shear_test_verification,
    run_velocity_dependence_verification,
    run_energy_conservation_verification,
    run_all_verifications,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def model():
    """Default contact model."""
    return IceHullContactModel()


@pytest.fixture
def friction_params():
    """Default friction parameters."""
    return IceHullFrictionParams()


@pytest.fixture
def adhesion_params():
    """Default adhesion parameters."""
    return IceHullAdhesionParams()


# =============================================================================
# 1. Friction model tests
# =============================================================================

class TestFrictionModel:
    """Tests for friction coefficient μ(p, v, T)."""
    
    def test_reference_point(self, friction_params):
        """At reference conditions, μ is close to μ_ref (within pressure/velocity factors)."""
        mu = compute_friction_coefficient(
            friction_params.p_ref_pa,
            friction_params.v_ref_m_s,
            friction_params.t_ref_c,
            friction_params,
        )
        # At reference T, mu_base = mu_ref
        # But pressure and velocity factors reduce it slightly
        # So mu should be less than mu_ref but in the right ballpark
        assert mu > 0.0
        assert mu < friction_params.mu_ref
        # Should be within 50% of mu_ref (factors are 1/(1+α) ≈ 0.9)
        assert mu > 0.3 * friction_params.mu_ref
    
    def test_pressure_weakening(self, friction_params):
        """μ decreases with increasing pressure."""
        mu_low = compute_friction_coefficient(0.1e6, 1.0, -10.0, friction_params)
        mu_high = compute_friction_coefficient(10.0e6, 1.0, -10.0, friction_params)
        assert mu_high < mu_low
    
    def test_velocity_weakening(self, friction_params):
        """μ decreases with increasing velocity."""
        mu_low = compute_friction_coefficient(1.0e6, 0.01, -10.0, friction_params)
        mu_high = compute_friction_coefficient(1.0e6, 10.0, -10.0, friction_params)
        assert mu_high < mu_low
    
    def test_temperature_strengthening(self, friction_params):
        """μ increases as temperature decreases (colder = more friction)."""
        mu_warm = compute_friction_coefficient(1.0e6, 1.0, -2.0, friction_params)
        mu_cold = compute_friction_coefficient(1.0e6, 1.0, -20.0, friction_params)
        assert mu_cold > mu_warm
    
    def test_friction_bounds(self, friction_params):
        """μ stays within [mu_floor, mu_ceiling]."""
        # Very high pressure → should hit floor
        mu = compute_friction_coefficient(100e6, 100.0, 0.0, friction_params)
        assert mu >= friction_params.mu_floor
        
        # Very cold, low pressure, low velocity → should hit ceiling
        mu = compute_friction_coefficient(0.0, 0.0, -50.0, friction_params)
        assert mu <= friction_params.mu_ceiling
    
    def test_friction_positive(self):
        """μ is always positive."""
        for p in [0.0, 1e5, 1e6, 1e7]:
            for v in [0.0, 0.01, 1.0, 10.0]:
                for t in [-30.0, -10.0, -2.0]:
                    mu = compute_friction_coefficient(p, v, t)
                    assert mu > 0.0
    
    def test_pressure_dependence_monotone(self, friction_params):
        """μ(p) is monotonically decreasing."""
        pressures = [0.0, 0.1e6, 0.5e6, 1.0e6, 5.0e6, 10.0e6]
        mus = [compute_friction_coefficient(p, 1.0, -10.0, friction_params) for p in pressures]
        for i in range(len(mus) - 1):
            assert mus[i] >= mus[i + 1]
    
    def test_velocity_dependence_monotone(self, friction_params):
        """μ(v) is monotonically decreasing."""
        velocities = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
        mus = [compute_friction_coefficient(1e6, v, -10.0, friction_params) for v in velocities]
        for i in range(len(mus) - 1):
            assert mus[i] >= mus[i + 1]
    
    def test_temperature_dependence_monotone(self, friction_params):
        """μ(T) is monotonically decreasing with T (increasing with cold)."""
        temperatures = [-30.0, -20.0, -10.0, -5.0, -2.0, 0.0]
        mus = [compute_friction_coefficient(1e6, 1.0, t, friction_params) for t in temperatures]
        for i in range(len(mus) - 1):
            assert mus[i] >= mus[i + 1]


# =============================================================================
# 2. Adhesion model tests
# =============================================================================

class TestAdhesionModel:
    """Tests for adhesion stress τ_ad(T, p, t)."""
    
    def test_reference_point(self, adhesion_params):
        """At reference conditions with long contact time, τ_ad is in the right range."""
        tau = compute_adhesion_stress(
            adhesion_params.t_ref_c,
            adhesion_params.p_ref_pa,
            1000.0,  # Long contact time → f_t → 1
            adhesion_params,
        )
        # At reference T and p, should be close to tau_ref
        assert tau > 0.0
        assert tau < adhesion_params.tau_ceiling_pa
        # Should be within factor of 2 of tau_ref
        assert 0.5 * adhesion_params.tau_ref_pa < tau < 2.0 * adhesion_params.tau_ref_pa
    
    def test_temperature_dependence(self, adhesion_params):
        """τ_ad increases as temperature decreases."""
        tau_warm = compute_adhesion_stress(-2.0, 1e6, 60.0, adhesion_params)
        tau_cold = compute_adhesion_stress(-20.0, 1e6, 60.0, adhesion_params)
        assert tau_cold > tau_warm
    
    def test_pressure_dependence(self, adhesion_params):
        """τ_ad increases with pressure."""
        tau_low = compute_adhesion_stress(-10.0, 0.1e6, 60.0, adhesion_params)
        tau_high = compute_adhesion_stress(-10.0, 10.0e6, 60.0, adhesion_params)
        assert tau_high > tau_low
    
    def test_time_dependence(self, adhesion_params):
        """τ_ad grows with contact time (freeze bonding)."""
        tau_0 = compute_adhesion_stress(-10.0, 1e6, 0.0, adhesion_params)
        tau_short = compute_adhesion_stress(-10.0, 1e6, 10.0, adhesion_params)
        tau_long = compute_adhesion_stress(-10.0, 1e6, 300.0, adhesion_params)
        assert tau_0 < tau_short < tau_long
    
    def test_zero_contact_time(self, adhesion_params):
        """At t=0, adhesion is zero (no freeze bond yet)."""
        tau = compute_adhesion_stress(-10.0, 1e6, 0.0, adhesion_params)
        assert tau == pytest.approx(0.0, abs=1e-10)
    
    def test_adhesion_bounds(self, adhesion_params):
        """τ_ad stays within [tau_floor, tau_ceiling]."""
        # Extreme conditions
        tau = compute_adhesion_stress(-50.0, 100e6, 10000.0, adhesion_params)
        assert tau <= adhesion_params.tau_ceiling_pa
        
        tau = compute_adhesion_stress(0.0, 0.0, 0.0, adhesion_params)
        assert tau >= adhesion_params.tau_floor_pa
    
    def test_adhesion_positive(self):
        """τ_ad is always non-negative."""
        for t in [-30.0, -10.0, -2.0]:
            for p in [0.0, 1e5, 1e6, 1e7]:
                for tc in [0.0, 10.0, 60.0, 300.0]:
                    tau = compute_adhesion_stress(t, p, tc)
                    assert tau >= 0.0


# =============================================================================
# 3. Shear strength tests
# =============================================================================

class TestShearStrength:
    """Tests for interface shear strength τ_shear."""
    
    def test_mohr_coulomb_no_adhesion(self):
        """Without adhesion, τ_shear = μ·p (standard Coulomb)."""
        mu = 0.3
        p = 1.0e6
        tau = compute_shear_strength(mu, p, 0.0, cohesive_strength_pa=10e6)
        assert tau == pytest.approx(mu * p, rel=1e-10)
    
    def test_mohr_coulomb_with_adhesion(self):
        """With adhesion, τ_shear = μ·p + τ_ad."""
        mu = 0.3
        p = 1.0e6
        tau_ad = 0.4e6
        tau = compute_shear_strength(mu, p, tau_ad, cohesive_strength_pa=10e6)
        assert tau == pytest.approx(mu * p + tau_ad, rel=1e-10)
    
    def test_cohesive_cap(self):
        """τ_shear is capped at cohesive strength."""
        mu = 0.5
        p = 10.0e6
        tau_ad = 1.0e6
        tau_cohesive = 1.5e6
        tau = compute_shear_strength(mu, p, tau_ad, tau_cohesive)
        assert tau == tau_cohesive  # Capped
    
    def test_zero_pressure(self):
        """At zero pressure, τ_shear = τ_ad (pure adhesion)."""
        mu = 0.3
        tau_ad = 0.4e6
        tau = compute_shear_strength(mu, 0.0, tau_ad, cohesive_strength_pa=10e6)
        assert tau == pytest.approx(tau_ad, rel=1e-10)
    
    def test_shear_strength_positive(self):
        """τ_shear is always non-negative."""
        for mu in [0.0, 0.1, 0.3, 0.5]:
            for p in [0.0, 1e5, 1e6, 1e7]:
                for tau_ad in [0.0, 0.1e6, 0.5e6]:
                    tau = compute_shear_strength(mu, p, tau_ad)
                    assert tau >= 0.0


# =============================================================================
# 4. Contact model integration tests
# =============================================================================

class TestContactModel:
    """Integration tests for the complete contact model."""
    
    def test_compute_state(self, model):
        """compute_state returns a valid IceHullContactState."""
        state = model.compute_state(1.0e6, 1.0, -10.0, 60.0)
        assert isinstance(state, IceHullContactState)
        assert state.friction_coefficient > 0.0
        assert state.adhesion_stress_pa >= 0.0
        assert state.shear_strength_pa >= 0.0
        assert state.friction_power_density_w_m2 >= 0.0
    
    def test_friction_angle(self, model):
        """Friction angle θ = arctan(μ)."""
        mu = compute_friction_coefficient(1e6, 1.0, -10.0)
        theta = model.friction_angle_deg(1e6, 1.0, -10.0)
        expected = math.degrees(math.atan(mu))
        assert theta == pytest.approx(expected, rel=1e-10)
    
    def test_energy_tracking(self, model):
        """Friction power = μ·p·v."""
        state = model.compute_state(1.0e6, 2.0, -10.0)
        expected_power = state.friction_coefficient * 1.0e6 * 2.0
        assert state.friction_power_density_w_m2 == pytest.approx(expected_power, rel=1e-10)
        assert state.heat_generation_rate_w_m2 == pytest.approx(expected_power, rel=1e-10)
    
    def test_high_level_interface(self):
        """ice_hull_contact_force returns correct forces."""
        result = ice_hull_contact_force(
            normal_force_n=1000.0,
            slip_velocity_m_s=1.0,
            temperature_c=-10.0,
            contact_area_m2=0.01,
            contact_time_s=60.0,
        )
        assert "friction_force_n" in result
        assert "adhesion_force_n" in result
        assert "shear_force_n" in result
        assert result["friction_force_n"] >= 0.0
        assert result["adhesion_force_n"] >= 0.0
        assert result["shear_force_n"] >= 0.0
    
    def test_zero_area(self):
        """Zero contact area → zero forces."""
        result = ice_hull_contact_force(1000.0, 1.0, -10.0, 0.0)
        assert result["friction_force_n"] == 0.0
        assert result["adhesion_force_n"] == 0.0
        assert result["shear_force_n"] == 0.0


# =============================================================================
# 5. Verification cases
# =============================================================================

class TestVerificationSlopeSliding:
    """Verification: friction angle vs temperature."""
    
    def test_verification_runs(self):
        """Verification runs without error."""
        result = run_slope_sliding_verification()
        assert "results" in result
        assert len(result["results"]) > 0
    
    def test_criterion(self):
        """Criterion: max error < 5°."""
        result = run_slope_sliding_verification()
        assert result["max_error_deg"] < 5.0
        assert result["pass"] is True
    
    def test_temperature_range(self):
        """Test covers T ∈ [-20, -2]°C."""
        result = run_slope_sliding_verification()
        temps = [r["temperature_c"] for r in result["results"]]
        assert min(temps) <= -20.0
        assert max(temps) >= -2.0
    
    def test_all_pass(self):
        """All individual tests pass."""
        result = run_slope_sliding_verification()
        assert all(r["pass"] for r in result["results"])


class TestVerificationTensionTest:
    """Verification: adhesion force vs contact time."""
    
    def test_verification_runs(self):
        """Verification runs without error."""
        result = run_tension_test_verification()
        assert "results" in result
        assert len(result["results"]) > 0
    
    def test_criterion(self):
        """Criterion: max error < 20%."""
        result = run_tension_test_verification()
        assert result["max_error_fraction"] < 0.20
        assert result["pass"] is True
    
    def test_time_range(self):
        """Test covers t ∈ [0, 300] s."""
        result = run_tension_test_verification()
        times = [r["contact_time_s"] for r in result["results"]]
        assert min(times) == 0.0
        assert max(times) >= 300.0
    
    def test_adhesion_grows_with_time(self):
        """Adhesion increases with contact time."""
        result = run_tension_test_verification()
        stresses = [r["adhesion_stress_mpa"] for r in result["results"]]
        for i in range(len(stresses) - 1):
            assert stresses[i] <= stresses[i + 1]


class TestVerificationShearTest:
    """Verification: shear strength vs pressure."""
    
    def test_verification_runs(self):
        """Verification runs without error."""
        result = run_shear_test_verification()
        assert "results" in result
        assert len(result["results"]) > 0
    
    def test_criterion(self):
        """Criterion: max error < 15%."""
        result = run_shear_test_verification()
        assert result["max_error_fraction"] < 0.15
        assert result["pass"] is True
    
    def test_pressure_range(self):
        """Test covers p ∈ [0, 5] MPa."""
        result = run_shear_test_verification()
        pressures = [r["pressure_pa"] for r in result["results"]]
        assert min(pressures) == 0.0
        assert max(pressures) >= 5.0e6
    
    def test_shear_increases_with_pressure(self):
        """Shear strength increases with pressure (up to cohesive cap)."""
        result = run_shear_test_verification()
        strengths = [r["shear_strength_mpa"] for r in result["results"]]
        for i in range(len(strengths) - 1):
            assert strengths[i] <= strengths[i + 1]


class TestVerificationVelocityDependence:
    """Verification: friction coefficient vs velocity."""
    
    def test_verification_runs(self):
        """Verification runs without error."""
        result = run_velocity_dependence_verification()
        assert "results" in result
        assert len(result["results"]) > 0
    
    def test_criterion(self):
        """Criterion: max error < 20%."""
        result = run_velocity_dependence_verification()
        assert result["max_error_fraction"] < 0.20
        assert result["pass"] is True
    
    def test_velocity_range(self):
        """Test covers v ∈ [0.001, 10] m/s."""
        result = run_velocity_dependence_verification()
        velocities = [r["velocity_m_s"] for r in result["results"]]
        assert min(velocities) <= 0.001
        assert max(velocities) >= 10.0
    
    def test_friction_decreases_with_velocity(self):
        """Friction decreases with velocity."""
        result = run_velocity_dependence_verification()
        mus = [r["friction_coefficient"] for r in result["results"]]
        for i in range(len(mus) - 1):
            assert mus[i] >= mus[i + 1]


class TestVerificationEnergyConservation:
    """Verification: energy conservation."""
    
    def test_verification_runs(self):
        """Verification runs without error."""
        result = run_energy_conservation_verification()
        assert "friction_work_j" in result
        assert "heat_generated_j" in result
    
    def test_criterion(self):
        """Criterion: error < 10%."""
        result = run_energy_conservation_verification()
        assert result["error_fraction"] < 0.10
        assert result["pass"] is True
    
    def test_energy_balance(self):
        """Friction work equals heat generated."""
        result = run_energy_conservation_verification()
        assert result["friction_work_j"] == pytest.approx(
            result["heat_generated_j"], rel=1e-10
        )
    
    def test_positive_energy(self):
        """Energy is positive."""
        result = run_energy_conservation_verification()
        assert result["friction_work_j"] > 0.0
        assert result["heat_generated_j"] > 0.0


# =============================================================================
# 6. Full verification suite
# =============================================================================

class TestAllVerifications:
    """Run all verifications together."""
    
    def test_all_verifications_pass(self):
        """All verification criteria are met."""
        result = run_all_verifications()
        assert result["all_pass"] is True
    
    def test_all_verifications_summary(self):
        """Summary contains all verification names."""
        result = run_all_verifications()
        assert "slope_sliding" in result["summary"]
        assert "tension_test" in result["summary"]
        assert "shear_test" in result["summary"]
        assert "velocity_dependence" in result["summary"]
        assert "energy_conservation" in result["summary"]
    
    def test_all_verifications_individual_pass(self):
        """Each individual verification passes."""
        result = run_all_verifications()
        for name, passed in result["summary"].items():
            assert passed is True, f"{name} failed"


# =============================================================================
# 7. Edge cases and robustness
# =============================================================================

class TestEdgeCases:
    """Edge cases and robustness tests."""
    
    def test_zero_velocity(self, model):
        """Zero velocity is handled correctly."""
        state = model.compute_state(1e6, 0.0, -10.0)
        assert state.friction_coefficient > 0.0
        assert state.friction_power_density_w_m2 == 0.0
    
    def test_zero_pressure(self, model):
        """Zero pressure is handled correctly."""
        state = model.compute_state(0.0, 1.0, -10.0)
        assert state.friction_coefficient > 0.0
        assert state.adhesion_stress_pa >= 0.0
    
    def test_extreme_temperature(self, model):
        """Extreme temperatures are handled."""
        state = model.compute_state(1e6, 1.0, -50.0)
        assert math.isfinite(state.friction_coefficient)
        assert math.isfinite(state.adhesion_stress_pa)
    
    def test_extreme_pressure(self, model):
        """Extreme pressures are handled."""
        state = model.compute_state(100e6, 1.0, -10.0)
        assert math.isfinite(state.friction_coefficient)
        assert math.isfinite(state.shear_strength_pa)
    
    def test_custom_parameters(self):
        """Custom parameters are used correctly."""
        friction_params = IceHullFrictionParams(mu_ref=0.20, alpha_p=0.30)
        mu = compute_friction_coefficient(1e6, 1.0, -10.0, friction_params)
        assert mu > 0.0
        
        adhesion_params = IceHullAdhesionParams(tau_ref_pa=1.0e6)
        tau = compute_adhesion_stress(-10.0, 1e6, 60.0, adhesion_params)
        assert tau > 0.0
