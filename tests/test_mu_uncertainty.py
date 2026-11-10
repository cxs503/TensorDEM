"""Tests for friction coefficient uncertainty quantification.

This test suite validates the UQ framework for DEM ice-structure contact:
- Distribution sampling and verification
- Surrogate model calibration
- Monte Carlo propagation
- Sobol sensitivity analysis
- Joint uncertainty propagation
- Physical consistency checks
"""
import pytest
import numpy as np
from scipy import stats

from tensordem.uncertainty.distributions import (
    FrictionDistribution,
    sample_mu,
)
from tensordem.uncertainty.surrogate import (
    ResistanceSurrogate,
    calibrate_surrogate,
)
from tensordem.uncertainty.propagation import (
    propagate_uncertainty,
    propagate_mu_sweep,
)
from tensordem.uncertainty.sensitivity import (
    compute_sobol_indices,
    compute_sobol_mc,
)
from tensordem.uncertainty.joint import (
    propagate_joint_uncertainty,
    three_d_mapping_closure,
)


class TestFrictionDistribution:
    """Test friction coefficient distribution sampling."""
    
    def test_normal_distribution_mean(self):
        """(a) Verify mu~N(0.1, 0.02) has mean within 0.1±0.005 for N>1000."""
        dist = FrictionDistribution(dist_type='normal', mean=0.10, std=0.02)
        samples = dist.sample(n_samples=2000, seed=42)
        
        sample_mean = np.mean(samples)
        assert abs(sample_mean - 0.10) < 0.005, \
            f"Sample mean {sample_mean:.4f} not within 0.1±0.005"
    
    def test_normal_distribution_std(self):
        """Verify standard deviation is close to specified value."""
        dist = FrictionDistribution(dist_type='normal', mean=0.10, std=0.02)
        samples = dist.sample(n_samples=5000, seed=42)
        
        sample_std = np.std(samples, ddof=1)
        assert abs(sample_std - 0.02) < 0.003, \
            f"Sample std {sample_std:.4f} not close to 0.02"
    
    def test_lognormal_distribution(self):
        """Test lognormal distribution sampling."""
        dist = FrictionDistribution(dist_type='lognormal', mean=0.10, std=0.02)
        samples = dist.sample(n_samples=2000, seed=42)
        
        # All samples should be positive
        assert np.all(samples > 0), "Lognormal samples must be positive"
        
        # Mean should be close to specified value
        sample_mean = np.mean(samples)
        assert abs(sample_mean - 0.10) < 0.01
    
    def test_uniform_distribution(self):
        """Test uniform distribution sampling."""
        dist = FrictionDistribution(
            dist_type='uniform',
            mean=0.10,
            std=0.02,
            lower=0.06,
            upper=0.14
        )
        samples = dist.sample(n_samples=2000, seed=42)
        
        # All samples within bounds
        assert np.all(samples >= 0.06) and np.all(samples <= 0.14)
    
    def test_distribution_stats(self):
        """Test distribution statistics computation."""
        dist = FrictionDistribution(dist_type='normal', mean=0.10, std=0.02)
        stats_dict = dist.stats()
        
        assert 'mean' in stats_dict
        assert 'std' in stats_dict
        assert 'median' in stats_dict
        assert 'q05' in stats_dict
        assert 'q95' in stats_dict
        assert abs(stats_dict['mean'] - 0.10) < 0.001


class TestResistanceSurrogate:
    """Test surrogate model for ice resistance."""
    
    def test_surrogate_prediction(self):
        """Test surrogate model prediction."""
        surrogate = ResistanceSurrogate(a=1e-7, b=2e-6)
        
        mu = np.array([0.0, 0.1, 0.2])
        sigma_c = np.array([2.5e6, 2.5e6, 2.5e6])
        
        r_pred = surrogate.predict(mu, sigma_c)
        
        # R = sigma_c * (a + b * mu)
        r_expected = sigma_c * (1e-7 + 2e-6 * mu)
        
        np.testing.assert_allclose(r_pred, r_expected, rtol=1e-10)
    
    def test_surrogate_calibration(self):
        """Test surrogate calibration from data."""
        # Generate synthetic data: R = sigma_c * (1e-7 + 2e-6 * mu)
        mu_vals = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
        sigma_c_vals = np.full(5, 2.5e6)
        r_vals = sigma_c_vals * (1e-7 + 2e-6 * mu_vals)
        
        surrogate = calibrate_surrogate(mu_vals, sigma_c_vals, r_vals)
        
        # Should recover exact coefficients
        assert abs(surrogate.a - 1e-7) < 1e-10
        assert abs(surrogate.b - 2e-6) < 1e-10
        assert surrogate.r_squared > 0.999
    
    def test_surrogate_monotonicity(self):
        """(b) Verify resistance increases monotonically with mu."""
        surrogate = ResistanceSurrogate(a=1e-7, b=2e-6)
        
        mu_vals = np.linspace(0.0, 0.3, 7)
        sigma_c = 2.5e6
        
        result = propagate_mu_sweep(surrogate, mu_vals, sigma_c)
        r_vals = result['r_values']
        
        # Check monotonicity
        for i in range(len(r_vals) - 1):
            assert r_vals[i+1] > r_vals[i], \
                f"Resistance not monotonic: R({mu_vals[i]})={r_vals[i]}, R({mu_vals[i+1]})={r_vals[i+1]}"
        
        # Compute Spearman correlation
        rho, _ = stats.spearmanr(mu_vals, r_vals)
        assert rho == 1.0, f"Spearman rho={rho}, expected 1.0"


class TestMonteCarloPropagation:
    """Test Monte Carlo uncertainty propagation."""
    
    def test_propagation_statistics(self):
        """Test propagation computes correct statistics."""
        surrogate = ResistanceSurrogate(a=1e-7, b=2e-6)
        mu_dist = FrictionDistribution(dist_type='normal', mean=0.10, std=0.02)
        
        result = propagate_uncertainty(
            surrogate, mu_dist, sigma_c_pa=2.5e6, n_samples=5000, seed=42
        )
        
        stats = result.compute_statistics()
        
        assert 'mean_kN' in stats
        assert 'std_kN' in stats
        assert 'q05_kN' in stats
        assert 'q95_kN' in stats
        assert stats['mean_kN'] > 0
        assert stats['std_kN'] > 0
    
    def test_ci_width_proportional_to_mu_std(self):
        """(c) Verify 95% CI width is proportional to mu std (R2>0.95)."""
        surrogate = ResistanceSurrogate(a=1e-7, b=2e-6)
        
        mu_stds = [0.01, 0.015, 0.02, 0.025, 0.03]
        ci_widths = []
        
        for mu_std in mu_stds:
            mu_dist = FrictionDistribution(dist_type='normal', mean=0.10, std=mu_std)
            result = propagate_uncertainty(
                surrogate, mu_dist, sigma_c_pa=2.5e6, n_samples=5000, seed=42
            )
            stats = result.compute_statistics()
            ci_widths.append(stats['ci_95_width_kN'])
        
        # Linear fit: CI_width = k * mu_std
        mu_stds_arr = np.array(mu_stds)
        ci_widths_arr = np.array(ci_widths)
        
        # Fit linear model
        coeffs = np.polyfit(mu_stds_arr, ci_widths_arr, 1)
        r_predicted = np.polyval(coeffs, mu_stds_arr)
        
        # Compute R-squared
        ss_res = np.sum((ci_widths_arr - r_predicted)**2)
        ss_tot = np.sum((ci_widths_arr - np.mean(ci_widths_arr))**2)
        r_squared = 1 - ss_res / ss_tot
        
        assert r_squared > 0.95, f"R²={r_squared:.4f}, expected >0.95"


class TestSensitivityAnalysis:
    """Test Sobol sensitivity analysis."""
    
    def test_sobol_indices_sum_to_one(self):
        """Verify Sobol indices sum to 1 (with interaction)."""
        surrogate = ResistanceSurrogate(a=1e-7, b=2e-6)
        mu_dist = FrictionDistribution(dist_type='normal', mean=0.10, std=0.02)
        
        sobol = compute_sobol_indices(
            surrogate, mu_dist, sigma_c_mean=2.5e6, sigma_c_std=0.5e6
        )
        
        # First-order + interaction should sum to 1
        total = sobol.s_mu + sobol.s_sigma + sobol.s_mu_sigma
        assert abs(total - 1.0) < 1e-10, f"Sobol indices sum to {total}, expected 1.0"
    
    def test_sobol_ranking(self):
        """Test sensitivity ranking."""
        surrogate = ResistanceSurrogate(a=1e-7, b=2e-6)
        mu_dist = FrictionDistribution(dist_type='normal', mean=0.10, std=0.02)
        
        sobol = compute_sobol_indices(
            surrogate, mu_dist, sigma_c_mean=2.5e6, sigma_c_std=0.5e6
        )
        
        ranking = sobol.ranking()
        assert len(ranking) == 2
        assert ranking[0][0] in ['mu', 'sigma_c']


class TestJointUncertainty:
    """Test joint uncertainty propagation."""
    
    def test_joint_propagation(self):
        """Test joint propagation of mu and sigma_c."""
        surrogate = ResistanceSurrogate(a=1e-7, b=2e-6)
        mu_dist = FrictionDistribution(dist_type='normal', mean=0.10, std=0.02)
        
        joint = propagate_joint_uncertainty(
            surrogate, mu_dist, sigma_c_mean=2.5e6, sigma_c_std=0.5e6,
            n_samples=5000, seed=42
        )
        
        stats = joint.compute_statistics()
        assert stats['mean_kN'] > 0
        assert stats['std_kN'] > 0
    
    def test_variance_decomposition(self):
        """(d) Verify variance decomposition under independence (<5% MC error)."""
        surrogate = ResistanceSurrogate(a=1e-7, b=2e-6)
        mu_dist = FrictionDistribution(dist_type='normal', mean=0.10, std=0.02)
        
        joint = propagate_joint_uncertainty(
            surrogate, mu_dist, sigma_c_mean=2.5e6, sigma_c_std=0.5e6,
            n_samples=50000, seed=42  # More samples for better MC accuracy
        )
        
        verification = joint.verify_variance_decomposition()
        
        # MC estimate should match analytical within 5%
        assert verification['relative_error'] < 0.05, \
            f"MC error {verification['relative_error']:.4f} > 5%"
        
        # Verify decomposition sums correctly
        var_sum = (verification['var_mu_kN2'] + 
                   verification['var_sigma_kN2'] + 
                   verification['var_interaction_kN2'])
        assert abs(var_sum - verification['var_total_analytical_kN2']) < 1e-10
    
    def test_three_d_mapping_closure(self):
        """Test complete 3D mapping closure."""
        surrogate = ResistanceSurrogate(a=1e-7, b=2e-6)
        mu_dist = FrictionDistribution(dist_type='normal', mean=0.10, std=0.02)
        
        results = three_d_mapping_closure(
            surrogate, mu_dist, sigma_c_mean=2.5e6, sigma_c_std=0.5e6,
            n_samples=5000, seed=42
        )
        
        assert 'joint_propagation' in results
        assert 'sobol_analysis' in results
        assert 'confidence_intervals' in results


class TestPhysicalConsistency:
    """Test physical consistency of the UQ framework."""
    
    def test_zero_friction_control(self):
        """(e) Negative control: mu=0 should give only normal contact resistance."""
        surrogate = ResistanceSurrogate(a=1e-7, b=2e-6)
        
        # At mu=0, R = sigma_c * a (only normal force contribution)
        r_mu0 = surrogate.predict_scalar(mu=0.0, sigma_c=2.5e6)
        r_mu_pos = surrogate.predict_scalar(mu=0.1, sigma_c=2.5e6)
        
        # Resistance at mu=0 should be less than at mu>0
        assert r_mu0 < r_mu_pos, "Resistance should increase with mu"
        
        # The difference is the friction contribution
        friction_contribution = r_mu_pos - r_mu0
        assert friction_contribution > 0
    
    def test_resistance_positive(self):
        """Verify resistance is always positive for physical inputs."""
        surrogate = ResistanceSurrogate(a=1e-7, b=2e-6)
        mu_dist = FrictionDistribution(dist_type='normal', mean=0.10, std=0.02)
        
        result = propagate_uncertainty(
            surrogate, mu_dist, sigma_c_pa=2.5e6, n_samples=1000, seed=42
        )
        
        assert np.all(result.r_samples > 0), "All resistance values must be positive"


class TestIntegration:
    """Integration tests for the complete UQ workflow."""
    
    def test_complete_workflow(self):
        """Test complete UQ workflow from distribution to results."""
        # 1. Define distribution
        mu_dist = FrictionDistribution(dist_type='normal', mean=0.10, std=0.02)
        
        # 2. Define surrogate (would be calibrated from DEM in practice)
        surrogate = ResistanceSurrogate(a=1e-7, b=2e-6, r_squared=0.98, n_calibration=7)
        
        # 3. Propagate uncertainty
        propagation = propagate_uncertainty(
            surrogate, mu_dist, sigma_c_pa=2.477e6, n_samples=2000, seed=42
        )
        
        # 4. Compute sensitivity
        sobol = compute_sobol_indices(
            surrogate, mu_dist, sigma_c_mean=2.477e6, sigma_c_std=0.5e6
        )
        
        # 5. Joint uncertainty
        joint = propagate_joint_uncertainty(
            surrogate, mu_dist, sigma_c_mean=2.477e6, sigma_c_std=0.5e6,
            n_samples=2000, seed=42
        )
        
        # Verify all components work together
        assert propagation.n_samples == 2000
        assert sobol.s_mu >= 0 and sobol.s_mu <= 1
        assert joint.n_samples == 2000
    
    def test_results_serialization(self):
        """Test results can be serialized to dict."""
        surrogate = ResistanceSurrogate(a=1e-7, b=2e-6, r_squared=0.98, n_calibration=7)
        mu_dist = FrictionDistribution(dist_type='normal', mean=0.10, std=0.02)
        
        joint = propagate_joint_uncertainty(
            surrogate, mu_dist, sigma_c_mean=2.5e6, sigma_c_std=0.5e6,
            n_samples=1000, seed=42
        )
        
        results_dict = joint.to_dict()
        
        assert 'n_samples' in results_dict
        assert 'mu_distribution' in results_dict
        assert 'sigma_distribution' in results_dict
        assert 'statistics' in results_dict
        assert 'variance_decomposition' in results_dict
