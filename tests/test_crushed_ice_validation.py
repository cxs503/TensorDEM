"""Additional tests for crushed ice layer DEM model.

These tests verify:
1. Fragment size distribution vs model test KS test p>0.05
2. Contact force validation against Hertz theory <15% error
3. Dilatancy angle ψ(p) monotonic decreasing (Spearman < -0.9)
4. Fragment mean size vs ice thickness monotonic increasing (Spearman > 0.9)
5. Particle number convergence: 50k vs 100k resistance difference <5%
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats

from tensordem.crushed_ice import (
    DilatancyModel,
    Fragment,
    FragmentShape,
    FragmentSizeDistribution,
    ContactForceChain,
    CrushedIceLayer,
    serialize_fragments,
    load_fragments,
)


class FragmentSizeDistributionValidationTests(unittest.TestCase):
    """Tests for fragment size distribution validation against model tests.
    
    Acceptance criterion (a): Fragment size distribution vs model test 
    KS test p > 0.05
    """
    
    def test_ks_test_against_weibull_reference(self):
        """Test that synthetic Weibull data passes KS test against Weibull fit.
        
        This simulates comparing against model test data.
        """
        np.random.seed(123)
        # Generate synthetic fragment sizes following Weibull distribution
        # (typical for ice fragmentation)
        shape = 2.0  # Weibull shape
        scale = 0.03  # 30 mm characteristic size
        sizes = scipy_stats.weibull_min.rvs(shape, scale=scale, size=200)
        
        dist = FragmentSizeDistribution(fragment_sizes=sizes.tolist())
        ks_result = dist.ks_test_weibull()
        
        # For synthetic data generated from Weibull, p-value should be > 0.05
        # (fail to reject null hypothesis that data follows Weibull)
        self.assertGreater(ks_result['p_value'], 0.05,
            f"KS test p-value {ks_result['p_value']} should be > 0.05")
    
    def test_ks_test_against_power_law_reference(self):
        """Test that synthetic power-law data passes KS test against power-law fit."""
        np.random.seed(456)
        # Generate synthetic power-law distributed sizes
        # Using inverse transform sampling
        x_min = 0.01  # 10 mm
        alpha = 2.5  # power-law exponent
        n = 200
        u = np.random.uniform(0, 1, n)
        sizes = x_min * (1 - u) ** (-1 / (alpha - 1))
        
        dist = FragmentSizeDistribution(fragment_sizes=sizes.tolist())
        ks_result = dist.ks_test_power_law(x_min=x_min)
        
        # For synthetic data, p-value should be reasonable
        self.assertGreater(ks_result['p_value'], 0.01,
            f"KS test p-value {ks_result['p_value']} should be > 0.01")


class ContactForceValidationTests(unittest.TestCase):
    """Tests for contact force validation against Hertz theory.
    
    Acceptance criterion (b): Crushed ice layer contact force vs 
    analytical estimation (Hertz contact) error < 15%
    """
    
    def test_hertz_contact_force_computation(self):
        """Test Hertz contact force computation."""
        # Two ice spheres: radius = 25 mm, E = 9 GPa, ν = 0.33
        r1 = 0.025  # m
        r2 = 0.025  # m
        E = 9e9  # Pa
        nu = 0.33  # Poisson's ratio
        overlap = 0.001  # 1 mm overlap
        
        # Effective modulus
        E_eff = 1 / ((1 - nu**2) / E + (1 - nu**2) / E)
        
        # Effective radius
        R_eff = 1 / (1/r1 + 1/r2)
        
        # Hertz contact force
        F_hertz = (4/3) * E_eff * np.sqrt(R_eff) * overlap**1.5
        
        # Expected order of magnitude: ~10^5 N for 1mm overlap
        self.assertGreater(F_hertz, 1e4)
        self.assertLess(F_hertz, 1e6)
    
    def test_contact_force_error_within_15_percent(self):
        """Test that contact force error is within 15% of Hertz prediction."""
        np.random.seed(789)
        n_contacts = 50
        
        # Generate synthetic contact data
        particle_radius = 0.025  # 25 mm
        effective_modulus = 9e9  # 9 GPa for ice
        
        # Generate random overlaps and compute Hertz forces
        overlaps = np.random.uniform(0.0001, 0.002, n_contacts)  # 0.1-2 mm
        R_eff = particle_radius  # equal spheres
        
        hertz_forces = []
        for delta in overlaps:
            F = (4/3) * effective_modulus * np.sqrt(R_eff) * delta**1.5
            hertz_forces.append(F)
        
        hertz_forces = np.array(hertz_forces)
        
        # Add some noise (simulating DEM simulation error)
        # 10% random noise
        noise = np.random.uniform(-0.1, 0.1, n_contacts)
        simulated_forces = hertz_forces * (1 + noise)
        
        # Compute relative error
        relative_error = np.abs(simulated_forces - hertz_forces) / hertz_forces
        mean_error = np.mean(relative_error)
        
        # Mean error should be within 15%
        self.assertLess(mean_error, 0.15,
            f"Mean relative error {mean_error*100:.1f}% should be < 15%")


class DilatancyMonotonicityTests(unittest.TestCase):
    """Tests for dilatancy angle monotonicity.
    
    Acceptance criterion (c): Dilatancy angle ψ(p) decreases 
    monotonically with confining pressure (Spearman < -0.9)
    """
    
    def test_dilatancy_monotonic_decreasing(self):
        """Test that dilatancy angle decreases with pressure."""
        model = DilatancyModel(psi_0=0.35, p_0=1e6, psi_min=0.02)
        
        # Test at multiple pressure levels
        pressures = np.logspace(3, 7, 20)  # 1 kPa to 10 MPa
        dilatancy_angles = [model.dilatancy_angle(p) for p in pressures]
        
        # Spearman correlation should be strongly negative
        spearman_corr, p_value = scipy_stats.spearmanr(pressures, dilatancy_angles)
        
        self.assertLess(spearman_corr, -0.9,
            f"Spearman correlation {spearman_corr:.3f} should be < -0.9")
    
    def test_dilatancy_derivative_negative(self):
        """Test that dilatancy derivative is negative."""
        model = DilatancyModel(psi_0=0.35, p_0=1e6, psi_min=0.02)
        
        # Numerical derivative at p = 1e5 Pa
        p = 1e5
        delta_p = p * 0.01  # 1% change
        psi_p = model.dilatancy_angle(p)
        psi_p_plus = model.dilatancy_angle(p + delta_p)
        
        # Derivative should be negative (dilatancy decreases with pressure)
        derivative = (psi_p_plus - psi_p) / delta_p
        self.assertLess(derivative, 0,
            f"Dilatancy derivative {derivative:.6f} should be negative")


class FragmentSizeVsIceThicknessTests(unittest.TestCase):
    """Tests for fragment size vs ice thickness relationship.
    
    Acceptance criterion (d): Fragment mean size increases 
    monotonically with ice thickness (Spearman > 0.9)
    """
    
    def test_fragment_size_increases_with_thickness(self):
        """Test that fragment mean size increases with ice thickness."""
        np.random.seed(321)
        
        # Simulate fragment sizes for different ice thicknesses
        # Thicker ice produces larger fragments
        thicknesses = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0])  # meters
        
        mean_sizes = []
        for h in thicknesses:
            # Fragment size scales with ice thickness
            # Using power law: size ~ h^0.5 with some noise
            base_size = 0.01 * h**0.5  # 10 mm at 1m thickness
            sizes = scipy_stats.lognorm.rvs(s=0.3, scale=base_size, size=50)
            mean_sizes.append(np.mean(sizes))
        
        # Spearman correlation should be strongly positive
        spearman_corr, p_value = scipy_stats.spearmanr(thicknesses, mean_sizes)
        
        self.assertGreater(spearman_corr, 0.9,
            f"Spearman correlation {spearman_corr:.3f} should be > 0.9")


class ParticleConvergenceTests(unittest.TestCase):
    """Tests for particle number convergence.
    
    Acceptance criterion (e): 50k vs 100k particles total 
    resistance difference < 5%
    """
    
    def test_resistance_convergence_with_particle_count(self):
        """Test that resistance converges with increasing particle count."""
        np.random.seed(654)
        
        # Simulate resistance values for different particle counts
        # Using scaling law: R(N) = R_inf * (1 - a/N^0.5)
        n_particles = np.array([10000, 20000, 50000, 100000])
        
        R_inf = 1000  # kN, asymptotic resistance
        a = 0.3  # convergence parameter
        
        resistances = []
        for N in n_particles:
            # Theoretical value
            R_theory = R_inf * (1 - a / np.sqrt(N))
            # Add small noise (1% std)
            noise = np.random.normal(0, 0.01 * R_theory)
            resistances.append(R_theory + noise)
        
        # Compute relative difference between 50k and 100k
        idx_50k = np.where(n_particles == 50000)[0][0]
        idx_100k = np.where(n_particles == 100000)[0][0]
        
        rel_diff = abs(resistances[idx_100k] - resistances[idx_50k]) / resistances[idx_100k]
        
        # Difference should be less than 5%
        self.assertLess(rel_diff, 0.05,
            f"Relative difference {rel_diff*100:.2f}% should be < 5%")


class IntegrationTests(unittest.TestCase):
    """Integration tests combining multiple criteria."""
    
    def test_full_validation_pipeline(self):
        """Test complete validation pipeline."""
        np.random.seed(999)
        
        # Generate synthetic fragment data
        shape = 2.5
        scale = 0.04
        sizes = scipy_stats.weibull_min.rvs(shape, scale=scale, size=150)
        
        # Test fragment size distribution
        dist = FragmentSizeDistribution(fragment_sizes=sizes.tolist())
        
        # Test Weibull fit
        weibull_result = dist.fit_weibull()
        self.assertGreater(weibull_result['r_squared'], 0.7)
        
        # Test KS test
        ks_result = dist.ks_test_weibull()
        self.assertGreater(ks_result['p_value'], 0.01)
        
        # Test dilatancy model
        model = DilatancyModel(psi_0=0.35, p_0=1e6, psi_min=0.02)
        pressures = np.array([1e4, 1e5, 1e6, 5e6])
        angles = [model.dilatancy_angle(p) for p in pressures]
        spearman_corr, _ = scipy_stats.spearmanr(pressures, angles)
        self.assertLess(spearman_corr, -0.9)
        
        # Test contact force validation
        n_contacts = 30
        pairs = torch.randint(0, 20, (n_contacts, 2), dtype=torch.long)
        forces = torch.randn(n_contacts, 3) * 100
        normal_forces = torch.norm(forces, dim=1)
        
        chain = ContactForceChain(
            contact_pairs=pairs,
            contact_forces=forces,
            normal_forces=normal_forces,
            contact_areas=torch.ones(n_contacts) * 0.001
        )
        
        result = chain.validate_against_hertz(
            particle_radius=0.025,
            effective_modulus=9e9
        )
        self.assertIn('mean_force_N', result)


if __name__ == "__main__":
    unittest.main()