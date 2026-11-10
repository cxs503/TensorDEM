"""
Tests for Ridge Ice Resistance (RAICE Method)
==============================================

Machine acceptance criteria:
1. Ridge ice resistance components analytical vs numerical error < 5%
2. Dimensional consistency check passes
3. Same input twice → same artifact sha256
"""

import pytest
import numpy as np
import hashlib
import json
from tensordem.ice_ridge_resistance import (
    RidgeIceResistance, validate_dimensional_consistency
)


class TestRidgeIceResistance:
    """Test suite for RAICE ridge ice resistance method."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.ridge = RidgeIceResistance()
        self.h_keel = 10.0  # m
        self.h_sail = 2.0   # m
        self.B = 20.0       # m
        self.L = 5.0        # m
        self.alpha = 30.0   # deg
    
    def test_keel_resistance_positive(self):
        """Keel resistance should be positive for valid inputs."""
        F_keel = self.ridge.keel_resistance(self.h_keel, self.B, self.L, self.alpha)
        assert F_keel > 0, "Keel resistance must be positive"
    
    def test_sail_resistance_positive(self):
        """Sail resistance should be positive for valid inputs."""
        F_sail = self.ridge.sail_resistance(self.h_sail, self.B, self.L, self.alpha)
        assert F_sail > 0, "Sail resistance must be positive"
    
    def test_interference_factor_range(self):
        """Interference factor should be in (0, 1]."""
        F_keel = self.ridge.keel_resistance(self.h_keel, self.B, self.L, self.alpha)
        F_sail = self.ridge.sail_resistance(self.h_sail, self.B, self.L, self.alpha)
        k_int = self.ridge.interference_correction(F_keel, F_sail, self.h_keel, self.h_sail)
        
        assert 0 < k_int <= 1.0, f"Interference factor {k_int} out of range (0, 1]"
    
    def test_caliber_registration_positive(self):
        """Caliber registration should be positive."""
        D_caliber = self.ridge.caliber_registration(self.B, self.L, self.B)
        assert D_caliber > 0, "Caliber registration must be positive"
    
    def test_total_resistance_components(self):
        """Total resistance should include all components."""
        result = self.ridge.total_resistance(self.h_keel, self.h_sail, self.B, self.L, self.alpha)
        
        assert 'F_keel' in result
        assert 'F_sail' in result
        assert 'k_int' in result
        assert 'D_caliber' in result
        assert 'F_total' in result
    
    def test_total_resistance_positive(self):
        """Total resistance should be positive."""
        result = self.ridge.total_resistance(self.h_keel, self.h_sail, self.B, self.L, self.alpha)
        assert result['F_total'] > 0, "Total resistance must be positive"
    
    def test_analytical_vs_numerical_error(self):
        """
        Machine criterion: Analytical vs numerical error < 5%.
        
        Compare analytical solution with numerical implementation.
        """
        # Numerical solution
        result_num = self.ridge.total_resistance(self.h_keel, self.h_sail, self.B, self.L, self.alpha)
        
        # Analytical solution
        result_ana = self.ridge.analytical_solution(self.h_keel, self.h_sail, self.B, self.L, self.alpha)
        
        # Calculate relative errors
        err_keel = abs(result_num['F_keel'] - result_ana['F_keel']) / result_ana['F_keel']
        err_sail = abs(result_num['F_sail'] - result_ana['F_sail']) / result_ana['F_sail']
        err_total = abs(result_num['F_total'] - result_ana['F_total']) / result_ana['F_total']
        
        # Machine criterion: error < 5%
        assert err_keel < 0.05, f"Keel error {err_keel:.4f} >= 5%"
        assert err_sail < 0.05, f"Sail error {err_sail:.4f} >= 5%"
        assert err_total < 0.05, f"Total error {err_total:.4f} >= 5%"
    
    def test_dimensional_consistency(self):
        """Machine criterion: Dimensional consistency check passes."""
        consistent = validate_dimensional_consistency(
            self.h_keel, self.h_sail, self.B, self.L
        )
        assert consistent, "Dimensional consistency check failed"
    
    def test_sha256_reproducibility(self):
        """
        Machine criterion: Same input twice → same artifact sha256.
        """
        params = {
            'h_keel': self.h_keel,
            'h_sail': self.h_sail,
            'B': self.B,
            'L': self.L,
            'alpha': self.alpha
        }
        
        hash1 = self.ridge.compute_sha256(params)
        hash2 = self.ridge.compute_sha256(params)
        
        assert hash1 == hash2, "SHA256 not reproducible for same input"
        assert len(hash1) == 64, "SHA256 hash should be 64 characters"
    
    def test_keel_resistance_scales_with_depth(self):
        """Keel resistance should increase with keel depth."""
        F1 = self.ridge.keel_resistance(5.0, self.B, self.L, self.alpha)
        F2 = self.ridge.keel_resistance(10.0, self.B, self.L, self.alpha)
        
        assert F2 > F1, "Keel resistance should increase with depth"
    
    def test_sail_resistance_scales_with_height(self):
        """Sail resistance should increase with sail height."""
        F1 = self.ridge.sail_resistance(1.0, self.B, self.L, self.alpha)
        F2 = self.ridge.sail_resistance(2.0, self.B, self.L, self.alpha)
        
        assert F2 > F1, "Sail resistance should increase with height"
    
    def test_interference_reduces_total(self):
        """Interference correction should reduce total resistance."""
        F_keel = self.ridge.keel_resistance(self.h_keel, self.B, self.L, self.alpha)
        F_sail = self.ridge.sail_resistance(self.h_sail, self.B, self.L, self.alpha)
        k_int = self.ridge.interference_correction(F_keel, F_sail, self.h_keel, self.h_sail)
        
        F_sum = F_keel + F_sail
        F_corrected = k_int * F_sum
        
        assert F_corrected < F_sum, "Interference should reduce total resistance"
    
    def test_extreme_keel_depth(self):
        """Test with extreme keel depth (h_sail/h_keel > 2)."""
        h_keel_extreme = 0.5  # Very shallow keel
        h_sail_extreme = 2.0  # Normal sail
        
        F_keel = self.ridge.keel_resistance(h_keel_extreme, self.B, self.L, self.alpha)
        F_sail = self.ridge.sail_resistance(h_sail_extreme, self.B, self.L, self.alpha)
        k_int = self.ridge.interference_correction(F_keel, F_sail, h_keel_extreme, h_sail_extreme)
        
        # Should use minimum value
        assert k_int == 0.6, "Extreme ratio should use minimum interference factor"
    
    def test_zero_keel_depth(self):
        """Test with zero keel depth (edge case)."""
        h_keel_zero = 0.0
        
        F_keel = self.ridge.keel_resistance(h_keel_zero, self.B, self.L, self.alpha)
        
        # Should still compute (breaking force dominates)
        assert np.isfinite(F_keel), "Should handle zero keel depth"
    
    def test_parameter_sensitivity(self):
        """Test sensitivity to key parameters."""
        # Baseline
        result_base = self.ridge.total_resistance(self.h_keel, self.h_sail, self.B, self.L, self.alpha)
        
        # Increase beam by 10%
        result_beam = self.ridge.total_resistance(self.h_keel, self.h_sail, self.B * 1.1, self.L, self.alpha)
        
        # Total resistance should increase
        assert result_beam['F_total'] > result_base['F_total'], \
            "Resistance should increase with beam"


class TestDimensionalConsistency:
    """Test dimensional consistency validation."""
    
    def test_valid_dimensions(self):
        """Valid inputs should pass dimensional check."""
        consistent = validate_dimensional_consistency(
            h_keel=10.0, h_sail=2.0, B=20.0, L=5.0
        )
        assert consistent
    
    def test_positive_force_dimensions(self):
        """Force terms should have positive dimensions."""
        rho_w = 1025.0
        rho_ice = 910.0
        g = 9.81
        h_keel = 10.0
        h_sail = 2.0
        B = 20.0
        L = 5.0
        
        F_buoyancy_dim = rho_w * g * h_keel * B * L
        F_passive_dim = rho_ice * g * h_sail**2 * B
        
        assert F_buoyancy_dim > 0
        assert F_passive_dim > 0
        assert np.isfinite(F_buoyancy_dim)
        assert np.isfinite(F_passive_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
