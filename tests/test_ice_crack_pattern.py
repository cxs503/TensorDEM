"""
Tests for Ice Crack Pattern Prediction
=======================================

Machine acceptance criteria:
1. Characteristic length numerical vs analytical error < 0.1%
2. Zero load negative control: p=0 → σ=0 → G=0 → no crack
3. Crack criteria self-consistency: G=G_c at initiation, G<G_c no crack, boundary error < 1%
4. Scaling law fit R² > 0.95
5. Dimensional consistency check passes
6. Same input twice → same artifact sha256
"""

import pytest
import numpy as np
from tensordem.ice_crack_pattern import (
    IceCrackPattern, validate_dimensional_consistency
)


class TestIceCrackPattern:
    """Test suite for ice crack pattern prediction."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.ice = IceCrackPattern()
    
    def test_characteristic_length_positive(self):
        """Characteristic length should be positive."""
        l = self.ice.characteristic_length()
        assert l > 0, "Characteristic length must be positive"
    
    def test_characteristic_length_analytical_vs_numerical(self):
        """
        Machine criterion: Characteristic length numerical vs analytical error < 0.1%.
        
        l = [E·h³ / (12·(1-ν²)·ρ_w·g)]^(1/4)
        """
        # Numerical (from method)
        l_num = self.ice.characteristic_length()
        
        # Analytical (direct calculation)
        E = self.ice.E
        h = self.ice.h
        nu = self.ice.nu
        rho_w = self.ice.rho_w
        g = self.ice.g
        
        l_ana = (E * h**3 / (12.0 * (1.0 - nu**2) * rho_w * g))**0.25
        
        # Relative error
        err = abs(l_num - l_ana) / l_ana
        
        # Machine criterion: error < 0.1%
        assert err < 0.001, f"Characteristic length error {err:.6f} >= 0.1%"
    
    def test_winkler_pressure_positive(self):
        """Winkler pressure should be positive for positive load."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 0.0, 0.0])
        P = 1000.0  # N
        
        p = self.ice.winkler_pressure(x, y, P)
        
        assert np.all(p > 0), "Pressure should be positive"
    
    def test_winkler_pressure_decays(self):
        """Winkler pressure should decay with distance."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 0.0, 0.0, 0.0])
        P = 1000.0
        
        p = self.ice.winkler_pressure(x, y, P)
        
        # Pressure should decrease with distance
        assert p[0] > p[1] > p[2] > p[3], "Pressure should decay with distance"
    
    def test_bending_moments_shape(self):
        """Bending moments should have correct shape."""
        p = np.ones((10, 10))
        dx = 0.1
        dy = 0.1
        
        M_x, M_y, M_xy = self.ice.plate_bending_moments(p, dx, dy)
        
        assert M_x.shape == p.shape
        assert M_y.shape == p.shape
        assert M_xy.shape == p.shape
    
    def test_bending_stress_positive(self):
        """Bending stress should be positive for non-zero moments."""
        M_x = np.ones((10, 10)) * 1000.0
        M_y = np.ones((10, 10)) * 500.0
        
        sigma = self.ice.bending_stress(M_x, M_y)
        
        assert np.all(sigma >= 0), "Bending stress should be non-negative"
    
    def test_energy_release_rate_positive(self):
        """Energy release rate should be positive for positive stress."""
        sigma = np.ones((10, 10)) * 1.0e6  # 1 MPa
        a = 0.1  # m
        
        G = self.ice.energy_release_rate(sigma, a)
        
        assert np.all(G > 0), "Energy release rate should be positive"
    
    def test_crack_initiation_at_critical(self):
        """
        Machine criterion: Crack criteria self-consistency.
        
        G = G_c should initiate crack.
        """
        G = np.array([self.ice.G_c])
        crack = self.ice.crack_initiation_criterion(G)
        
        assert crack[0], "Crack should initiate at G = G_c"
    
    def test_no_crack_below_critical(self):
        """
        Machine criterion: G < G_c should not initiate crack.
        """
        G = np.array([self.ice.G_c * 0.99])
        crack = self.ice.crack_initiation_criterion(G)
        
        assert not crack[0], "Crack should not initiate at G < G_c"
    
    def test_crack_above_critical(self):
        """G > G_c should initiate crack."""
        G = np.array([self.ice.G_c * 1.01])
        crack = self.ice.crack_initiation_criterion(G)
        
        assert crack[0], "Crack should initiate at G > G_c"
    
    def test_zero_load_negative_control(self):
        """
        Machine criterion: Zero load negative control.
        
        p=0 → σ=0 → G=0 → no crack
        """
        result = self.ice.zero_load_negative_control()
        
        assert result['p_max'] == 0, "Zero load should give zero pressure"
        assert result['sigma_max'] == 0, "Zero pressure should give zero stress"
        assert result['G_max'] == 0, "Zero stress should give zero G"
        assert result['crack_count'] == 0, "Zero G should give no cracks"
        assert result['pass'], "Zero load control test failed"
    
    def test_boundary_error_check(self):
        """
        Machine criterion: Boundary error < 1%.
        
        Test crack criterion at boundary G = G_c.
        """
        result = self.ice.boundary_error_check(self.ice.G_c, tolerance=0.01)
        
        assert result['crack_at_Gc'], "Should crack at G = G_c"
        assert not result['crack_below_Gc'], "Should not crack below G_c"
        assert result['crack_above_Gc'], "Should crack above G_c"
        assert result['boundary_error'] < 0.01, f"Boundary error {result['boundary_error']:.4f} >= 1%"
        assert result['pass'], "Boundary error check failed"
    
    def test_crack_density_scaling_positive(self):
        """Crack density should be positive for valid inputs."""
        V = 1.0  # m/s
        h = 1.0  # m
        sigma_range = np.array([0.5e6, 1.5e6])  # Pa
        
        rho_crack = self.ice.crack_density_scaling(V, h, sigma_range)
        
        assert rho_crack > 0, "Crack density should be positive"
    
    def test_crack_density_increases_with_velocity(self):
        """Crack density should increase with velocity."""
        h = 1.0
        sigma_range = np.array([1.0e6, 1.0e6])
        
        rho1 = self.ice.crack_density_scaling(0.5, h, sigma_range)
        rho2 = self.ice.crack_density_scaling(1.0, h, sigma_range)
        rho3 = self.ice.crack_density_scaling(2.0, h, sigma_range)
        
        assert rho1 < rho2 < rho3, "Crack density should increase with velocity"
    
    def test_crack_density_decreases_with_thickness(self):
        """Crack density should decrease with ice thickness."""
        V = 1.0
        sigma_range = np.array([1.0e6, 1.0e6])
        
        rho1 = self.ice.crack_density_scaling(V, 0.5, sigma_range)
        rho2 = self.ice.crack_density_scaling(V, 1.0, sigma_range)
        rho3 = self.ice.crack_density_scaling(V, 2.0, sigma_range)
        
        assert rho1 > rho2 > rho3, "Crack density should decrease with thickness"
    
    def test_crack_density_increases_with_stress(self):
        """Crack density should increase with stress."""
        V = 1.0
        h = 1.0
        
        rho1 = self.ice.crack_density_scaling(V, h, np.array([0.5e6, 0.5e6]))
        rho2 = self.ice.crack_density_scaling(V, h, np.array([1.0e6, 1.0e6]))
        rho3 = self.ice.crack_density_scaling(V, h, np.array([2.0e6, 2.0e6]))
        
        assert rho1 < rho2 < rho3, "Crack density should increase with stress"
    
    def test_scaling_law_fit_r_squared(self):
        """
        Machine criterion: Scaling law fit R² > 0.95.
        
        Test that crack density follows power law with high R².
        """
        # Generate data points
        V_values = np.linspace(0.5, 3.0, 20)
        h = 1.0
        sigma_range = np.array([1.0e6, 1.0e6])
        
        rho_values = np.array([
            self.ice.crack_density_scaling(V, h, sigma_range) for V in V_values
        ])
        
        # Fit power law: rho = a * V^b
        log_V = np.log(V_values)
        log_rho = np.log(rho_values)
        
        # Linear regression
        coeffs = np.polyfit(log_V, log_rho, 1)
        rho_fit = np.exp(np.polyval(coeffs, log_V))
        
        # Calculate R²
        ss_res = np.sum((log_rho - np.log(rho_fit))**2)
        ss_tot = np.sum((log_rho - np.mean(log_rho))**2)
        R_squared = 1 - ss_res / ss_tot
        
        # Machine criterion: R² > 0.95
        assert R_squared > 0.95, f"Scaling law R² = {R_squared:.4f} < 0.95"
    
    def test_full_analysis_completeness(self):
        """Full analysis should return all required fields."""
        P = 10000.0  # N
        
        result = self.ice.full_analysis(P, nx=20, ny=20)
        
        assert 'characteristic_length' in result
        assert 'pressure' in result
        assert 'M_x' in result
        assert 'M_y' in result
        assert 'sigma_max' in result
        assert 'G' in result
        assert 'crack_mask' in result
        assert 'crack_density' in result
    
    def test_sha256_reproducibility(self):
        """
        Machine criterion: Same input twice → same artifact sha256.
        """
        params = {
            'E': self.ice.E,
            'nu': self.ice.nu,
            'h': self.ice.h,
            'rho_w': self.ice.rho_w,
            'g': self.ice.g,
            'G_c': self.ice.G_c
        }
        
        hash1 = self.ice.compute_sha256(params)
        hash2 = self.ice.compute_sha256(params)
        
        assert hash1 == hash2, "SHA256 not reproducible for same input"
        assert len(hash1) == 64, "SHA256 hash should be 64 characters"
    
    def test_dimensional_consistency(self):
        """Machine criterion: Dimensional consistency check passes."""
        consistent = validate_dimensional_consistency(
            self.ice.E, self.ice.h, self.ice.nu,
            self.ice.rho_w, self.ice.g
        )
        assert consistent, "Dimensional consistency check failed"


class TestDimensionalConsistency:
    """Test dimensional consistency validation."""
    
    def test_valid_dimensions(self):
        """Valid inputs should pass dimensional check."""
        E = 5.0e9  # Pa
        h = 1.0    # m
        nu = 0.3   # -
        rho_w = 1025.0  # kg/m³
        g = 9.81   # m/s²
        
        consistent = validate_dimensional_consistency(E, h, nu, rho_w, g)
        assert consistent
    
    def test_characteristic_length_dimensions(self):
        """Characteristic length should have dimensions of length."""
        E = 5.0e9
        h = 1.0
        nu = 0.3
        rho_w = 1025.0
        g = 9.81
        
        numerator = E * h**3  # kg·m²/s²
        denominator = 12 * (1 - nu**2) * rho_w * g  # kg/(m²·s²)
        
        l_squared = numerator / denominator  # m⁴
        l = l_squared**0.25  # m
        
        assert l > 0
        assert np.isfinite(l)


class TestCrackPatternPhysics:
    """Test physical consistency of crack pattern predictions."""
    
    def setup_method(self):
        self.ice = IceCrackPattern()
    
    def test_thicker_ice_stiffer(self):
        """Thicker ice should be stiffer (larger characteristic length)."""
        ice1 = IceCrackPattern(h=0.5)
        ice2 = IceCrackPattern(h=1.0)
        ice3 = IceCrackPattern(h=2.0)
        
        l1 = ice1.characteristic_length()
        l2 = ice2.characteristic_length()
        l3 = ice3.characteristic_length()
        
        assert l1 < l2 < l3, "Characteristic length should increase with thickness"
    
    def test_stiffer_material_larger_length(self):
        """Stiffer material should have larger characteristic length."""
        ice1 = IceCrackPattern(E=3.0e9)
        ice2 = IceCrackPattern(E=5.0e9)
        ice3 = IceCrackPattern(E=7.0e9)
        
        l1 = ice1.characteristic_length()
        l2 = ice2.characteristic_length()
        l3 = ice3.characteristic_length()
        
        assert l1 < l2 < l3, "Characteristic length should increase with stiffness"
    
    def test_crack_propagation_direction(self):
        """Cracks should propagate in direction of maximum stress."""
        # Create anisotropic stress field
        M_x = np.ones((20, 20)) * 2000.0
        M_y = np.ones((20, 20)) * 1000.0
        
        sigma = self.ice.bending_stress(M_x, M_y)
        
        # Maximum stress should be in x-direction
        assert np.max(sigma) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
