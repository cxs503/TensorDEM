"""Pytest tests for friction contact verification.

Campaign P152W5-N02: TensorDEM friction contact verification.
"""
import pytest
import math
import numpy as np

from tensordem.friction_contact_verification import (
    verify_coulomb_friction,
    verify_rolling_resistance,
    verify_angle_of_repose,
    verify_packing_density,
    verify_direct_shear,
    verify_timestep_convergence,
    run_full_verification,
    evaluate_acceptance,
)


class TestCoulombFriction:
    """Test Coulomb friction law: |F_t| ≤ μ·|F_n|."""

    def test_saturation(self):
        """Tangential force saturates at μ·F_n."""
        result = verify_coulomb_friction()
        assert result['all_pass'], "Coulomb friction saturation failed"

    def test_zero_friction(self):
        """μ=0 → F_t=0."""
        result = verify_coulomb_friction(mu_values=[0.0], f_n_values=[1.0, 5.0])
        for r in result['results']:
            assert r['cap'] == 0.0
            assert r['f_t_final'] == 0.0

    def test_proportionality(self):
        """F_t_cap ∝ F_n for fixed μ."""
        result = verify_coulomb_friction(mu_values=[0.3], f_n_values=[1.0, 5.0, 10.0])
        caps = [r['cap'] for r in result['results']]
        assert abs(caps[1] / caps[0] - 5.0) < 1e-10
        assert abs(caps[2] / caps[0] - 10.0) < 1e-10


class TestRollingResistance:
    """Test rolling resistance: M_r ≤ μ_r·r_eff·F_n."""

    def test_saturation(self):
        """Rolling torque saturates at μ_r·r_eff·F_n."""
        result = verify_rolling_resistance()
        assert result['all_pass'], "Rolling resistance saturation failed"

    def test_zero_rolling(self):
        """μ_r=0 → M_r=0."""
        result = verify_rolling_resistance(mu_r_values=[0.0])
        for r in result['results']:
            assert r['cap'] == 0.0

    def test_proportionality(self):
        """M_r_cap ∝ r_eff·F_n."""
        result = verify_rolling_resistance(
            mu_r_values=[0.05],
            f_n_values=[1.0, 10.0],
            r_eff_values=[0.01],
        )
        caps = [r['cap'] for r in result['results']]
        assert abs(caps[1] / caps[0] - 10.0) < 1e-10


class TestAngleOfRepose:
    """Test angle of repose: tan(θ_r) ≈ μ."""

    def test_analytic_relationship(self):
        """θ_r = atan(μ) for sliding-dominated systems."""
        result = verify_angle_of_repose(mu_values=[0.3], mu_r_values=[0.0])
        r = result['results'][0]
        expected = math.degrees(math.atan(0.3))
        assert abs(r['theta_analytic_deg'] - expected) < 1e-10

    def test_negative_control(self):
        """μ=0 → θ_r < 5° (hard gate)."""
        result = verify_angle_of_repose()
        nc = result['negative_control']
        assert nc['pass'], f"Negative control failed: angle={nc['angle_deg']:.1f}°"
        assert nc['angle_deg'] < 5.0

    def test_increasing_with_mu(self):
        """θ_r increases with μ."""
        result = verify_angle_of_repose(mu_values=[0.1, 0.3, 0.5], mu_r_values=[0.0])
        angles = [r['theta_analytic_deg'] for r in result['results']]
        assert angles[0] < angles[1] < angles[2]


class TestPackingDensity:
    """Test packing density: φ vs RCP ≈ 0.64."""

    def test_rcp_limit(self):
        """μ=0 → φ ≈ 0.64 (RCP)."""
        result = verify_packing_density(mu_values=[0.0])
        r = result['results'][0]
        assert abs(r['phi_dem'] - 0.64) < 0.01

    def test_decreases_with_friction(self):
        """φ decreases with μ."""
        result = verify_packing_density(mu_values=[0.0, 0.3, 0.5])
        phis = [r['phi_dem'] for r in result['results']]
        assert phis[0] >= phis[1] >= phis[2]

    def test_error_bound(self):
        """Error vs RCP < 15% for all μ."""
        result = verify_packing_density()
        for r in result['results']:
            assert r['pass'], f"μ={r['mu']}: error={r['error_vs_rcp_pct']:.1f}%"


class TestDirectShear:
    """Test direct shear: Mohr-Coulomb τ = σ·tan(φ)."""

    def test_mohr_coulomb(self):
        """τ = σ·μ for cohesionless material."""
        result = verify_direct_shear()
        assert all(r['pass'] for r in result['results'])

    def test_linear_in_sigma(self):
        """τ ∝ σ for fixed μ."""
        result = verify_direct_shear(mu_values=[0.3], sigma_values=[500, 1000, 2000])
        taus = [r['tau_predicted_Pa'] for r in result['results']]
        assert abs(taus[1] / taus[0] - 2.0) < 1e-10
        assert abs(taus[2] / taus[0] - 4.0) < 1e-10


class TestTimestepConvergence:
    """Test time step convergence."""

    def test_stability(self):
        """All time steps are stable (dt < 0.1·dt_crit)."""
        result = verify_timestep_convergence()
        assert result['all_pass']

    def test_resolution(self):
        """All time steps well-resolve the contact period."""
        result = verify_timestep_convergence()
        for r in result['results']:
            assert r['well_resolved'], f"dt={r['dt']}: {r['steps_per_period']:.0f} steps/period"


class TestFullSuite:
    """Test the full verification suite."""

    def test_all_pass(self):
        """All acceptance criteria pass."""
        results = run_full_verification()
        acceptance = evaluate_acceptance(results)
        assert acceptance['overall']['pass'], "Not all acceptance criteria passed"

    def test_completeness(self):
        """All required tests are present."""
        results = run_full_verification()
        required = [
            'coulomb_friction',
            'rolling_resistance',
            'angle_of_repose',
            'packing_density',
            'direct_shear',
            'timestep_convergence',
        ]
        for key in required:
            assert key in results, f"Missing test: {key}"
