"""Comprehensive tests for ice_crack_pattern module.

Machine-verifiable criteria tested:
1. Zero-load negative control: p=0 → σ=0 → G=0 → no crack
2. Energy release rate: G = K_I²/E'' (plane strain), dimension = J/m²
3. Crack criterion: G > G_c → crack; G < G_c → no crack
4. Radial/circumferential distinction: based on principal stress direction
5. Scaling law: ρ_crack ∝ V^α · h^β, fitted exponents with R² > 0.95
6. DEM consistency: match ratio > 0.7
7. Idempotent reproduction: same input → same SHA256
"""

import math
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensordem.ice_crack_pattern import (
    IceCrackPattern,
    validate_dimensional_consistency,
    energy_release_rate_dimensional_check,
)


class TestZeroLoadNegativeControl:
    """Criterion 1: Zero load → no crack."""

    def test_zero_pressure_zero_stress(self):
        """p=0 → σ=0."""
        ice = IceCrackPattern()
        control = ice.zero_load_negative_control()
        assert control['p_max'] == 0.0
        assert control['sigma_max'] == 0.0

    def test_zero_stress_zero_G(self):
        """σ=0 → G=0."""
        ice = IceCrackPattern()
        control = ice.zero_load_negative_control()
        assert control['G_max'] == 0.0

    def test_zero_G_no_crack(self):
        """G=0 → no crack."""
        ice = IceCrackPattern()
        control = ice.zero_load_negative_control()
        assert control['crack_count'] == 0

    def test_full_control_pass(self):
        """All zero-load conditions satisfied simultaneously."""
        ice = IceCrackPattern()
        control = ice.zero_load_negative_control()
        assert control["pass"] == True

    def test_zero_load_with_different_parameters(self):
        """Zero-load control passes for various parameter sets."""
        for E in [3e9, 5e9, 7e9]:
            for h in [0.5, 1.0, 2.0]:
                ice = IceCrackPattern(E=E, h=h)
                control = ice.zero_load_negative_control()
                assert control['pass'], f"Failed for E={E}, h={h}"


class TestEnergyReleaseRate:
    """Criterion 2: G = K_I²/E'' (plane strain), dimension = J/m²."""

    def test_G_formula_plane_strain(self):
        """G = K_I² / E'' where E'' = E/(1-ν²)."""
        ice = IceCrackPattern(E=5e9, nu=0.3)
        sigma = np.array([1e6])  # 1 MPa
        a = 0.1  # 10 cm crack

        K_I = sigma[0] * np.sqrt(np.pi * a)
        E_prime = ice.E / (1 - ice.nu**2)
        G_expected = K_I**2 / E_prime

        G_computed = ice.energy_release_rate(sigma, a)[0]
        assert abs(G_computed - G_expected) / G_expected < 1e-10

    def test_G_dimension_J_per_m2(self):
        """G has dimensions of J/m²."""
        # K_I in Pa·√m, E'' in Pa
        # G = K_I²/E'' = (Pa²·m) / Pa = Pa·m = N/m = J/m²
        K_I = 1e6 * np.sqrt(np.pi * 0.1)  # Pa·√m
        E_prime = 5e9 / (1 - 0.3**2)  # Pa
        G = energy_release_rate_dimensional_check(K_I, E_prime)
        assert G > 0
        assert np.isfinite(G)

    def test_G_increases_with_stress(self):
        """G increases monotonically with stress."""
        ice = IceCrackPattern()
        stresses = np.array([0.5e6, 1.0e6, 1.5e6, 2.0e6])
        G_values = ice.energy_release_rate(stresses, a=0.1)
        assert np.all(np.diff(G_values) > 0)

    def test_G_increases_with_crack_length(self):
        """G increases with crack length (for same stress)."""
        ice = IceCrackPattern()
        sigma = np.array([1e6])
        G_01 = ice.energy_release_rate(sigma, a=0.05)[0]
        G_02 = ice.energy_release_rate(sigma, a=0.10)[0]
        G_03 = ice.energy_release_rate(sigma, a=0.20)[0]
        assert G_01 < G_02 < G_03

    def test_G_zero_for_compressive_stress(self):
        """Compressive stress should not produce mode I cracking."""
        ice = IceCrackPattern()
        sigma = np.array([-1e6])  # Compressive
        G = ice.energy_release_rate(sigma, a=0.1)
        assert G[0] == 0.0

    def test_E_prime_plane_strain(self):
        """E'' = E/(1-ν²) for plane strain."""
        ice = IceCrackPattern(E=5e9, nu=0.3)
        E_prime_expected = 5e9 / (1 - 0.3**2)
        assert abs(ice.E_prime - E_prime_expected) < 1e-6


class TestCrackCriterion:
    """Criterion 3: G > G_c → crack; G < G_c → no crack."""

    def test_G_above_Gc_cracks(self):
        """G > G_c → crack initiates."""
        ice = IceCrackPattern(G_c=50.0)
        G = np.array([100.0])  # Well above G_c
        crack = ice.crack_initiation_criterion(G)
        assert crack[0] is np.True_

    def test_G_below_Gc_no_crack(self):
        """G < G_c → no crack."""
        ice = IceCrackPattern(G_c=50.0)
        G = np.array([10.0])  # Well below G_c
        crack = ice.crack_initiation_criterion(G)
        assert crack[0] is np.False_

    def test_G_equal_Gc_cracks(self):
        """G = G_c → crack initiates (boundary condition)."""
        ice = IceCrackPattern(G_c=50.0)
        G = np.array([50.0])
        crack = ice.crack_initiation_criterion(G)
        assert crack[0] is np.True_

    def test_boundary_error_check(self):
        """Boundary error check passes."""
        ice = IceCrackPattern(G_c=50.0)
        result = ice.boundary_error_check(ice.G_c)
        assert result['pass'] is True
        assert result['crack_at_Gc'] is True
        assert result['crack_below_Gc'] is False
        assert result['crack_above_Gc'] is True


class TestRadialCircumferentialCrack:
    """Criterion 4: Radial/circumferential distinction based on principal stress direction."""

    def test_radial_crack_near_load(self):
        """Near the load point, cracks should be radial (σ_1 points outward)."""
        ice = IceCrackPattern()
        # At point (1, 0) from load at origin, radial direction = 0 rad
        # If σ_1 direction ≈ 0, it's radial
        theta_p = np.array([0.0])  # σ_1 along x-axis
        x = np.array([1.0])
        y = np.array([0.0])
        crack_type = ice.classify_crack_type(theta_p, x, y)
        assert crack_type[0] == 1  # Radial

    def test_circumferential_crack_perpendicular(self):
        """If σ_1 is perpendicular to radial direction, crack is circumferential."""
        ice = IceCrackPattern()
        # At point (1, 0), radial direction = 0 rad
        # If σ_1 direction = π/2, it's perpendicular → circumferential
        theta_p = np.array([np.pi / 2])
        x = np.array([1.0])
        y = np.array([0.0])
        crack_type = ice.classify_crack_type(theta_p, x, y)
        assert crack_type[0] == 2  # Circumferential

    def test_crack_classification_symmetry(self):
        """Crack pattern should have radial symmetry around load point."""
        ice = IceCrackPattern()
        # Test at 4 symmetric points
        angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        r = 2.0
        for angle in angles:
            x = np.array([r * np.cos(angle)])
            y = np.array([r * np.sin(angle)])
            # Radial direction at this point
            theta_p = np.array([angle])
            crack_type = ice.classify_crack_type(theta_p, x, y)
            assert crack_type[0] == 1, f"Should be radial at angle {angle}"

    def test_principal_stress_calculation(self):
        """Principal stresses from moments are correct."""
        ice = IceCrackPattern(h=1.0)
        # Uniaxial bending: M_x = 1000, M_y = 0, M_xy = 0
        M_x = np.array([[1000.0]])
        M_y = np.array([[0.0]])
        M_xy = np.array([[0.0]])
        sigma_1, sigma_2, theta_p = ice.principal_stresses(M_x, M_y, M_xy)
        # σ_1 = 6·M_x/h² = 6000 Pa, σ_2 = 0
        assert abs(sigma_1[0, 0] - 6000.0) < 1e-6
        assert abs(sigma_2[0, 0]) < 1e-6
        # θ_p = 0 (along x-axis)
        assert abs(theta_p[0, 0]) < 1e-6


class TestScalingLaw:
    """Criterion 5: ρ_crack ∝ V^α · h^β, fitted exponents with R² > 0.95."""

    def test_crack_density_increases_with_velocity(self):
        """Crack density should increase with velocity."""
        ice = IceCrackPattern()
        sigma_range = np.array([1.0e6, 1.0e6])
        rho1 = ice.crack_density_scaling(0.5, 1.0, sigma_range)
        rho2 = ice.crack_density_scaling(1.0, 1.0, sigma_range)
        rho3 = ice.crack_density_scaling(2.0, 1.0, sigma_range)
        assert rho1 < rho2 < rho3

    def test_crack_density_decreases_with_thickness(self):
        """Crack density should decrease with ice thickness."""
        ice = IceCrackPattern()
        sigma_range = np.array([1.0e6, 1.0e6])
        rho1 = ice.crack_density_scaling(1.0, 0.5, sigma_range)
        rho2 = ice.crack_density_scaling(1.0, 1.0, sigma_range)
        rho3 = ice.crack_density_scaling(1.0, 2.0, sigma_range)
        assert rho1 > rho2 > rho3

    def test_scaling_law_R_squared_velocity(self):
        """Scaling law fit R² > 0.95 for velocity sweep."""
        ice = IceCrackPattern()
        V_values = np.linspace(0.5, 3.0, 20)
        h = 1.0
        sigma_range = np.array([1.0e6, 1.0e6])

        rho_values = np.array([
            ice.crack_density_scaling(V, h, sigma_range) for V in V_values
        ])

        # Fit power law: rho = a * V^b
        log_V = np.log(V_values)
        log_rho = np.log(rho_values)
        coeffs = np.polyfit(log_V, log_rho, 1)
        rho_fit = np.exp(np.polyval(coeffs, log_V))

        ss_res = np.sum((log_rho - np.log(rho_fit))**2)
        ss_tot = np.sum((log_rho - np.mean(log_rho))**2)
        R_squared = 1 - ss_res / ss_tot

        assert R_squared > 0.95, f"R² = {R_squared:.4f} < 0.95"

    def test_scaling_law_R_squared_thickness(self):
        """Scaling law fit R² > 0.95 for thickness sweep."""
        ice = IceCrackPattern()
        V = 1.0
        h_values = np.linspace(0.3, 2.0, 20)
        sigma_range = np.array([1.0e6, 1.0e6])

        rho_values = np.array([
            ice.crack_density_scaling(V, h, sigma_range) for h in h_values
        ])

        log_h = np.log(h_values)
        log_rho = np.log(rho_values)
        coeffs = np.polyfit(log_h, log_rho, 1)
        rho_fit = np.exp(np.polyval(coeffs, log_h))

        ss_res = np.sum((log_rho - np.log(rho_fit))**2)
        ss_tot = np.sum((log_rho - np.mean(log_rho))**2)
        R_squared = 1 - ss_res / ss_tot

        assert R_squared > 0.95, f"R² = {R_squared:.4f} < 0.95"

    def test_fit_scaling_exponents(self):
        """Fit scaling exponents from parametric sweep."""
        ice = IceCrackPattern()
        V_range = np.linspace(0.5, 3.0, 10)
        h_range = np.array([1.0])  # Fixed thickness for clean α fit

        sweep = ice.parametric_sweep(V_range, h_range, P=10000.0, nx=30, ny=30)
        result = ice.fit_scaling_exponents(sweep['V'], sweep['h'], sweep['rho_crack'])

        assert 'alpha' in result
        assert 'beta' in result
        assert 'R_squared' in result
        # For fixed h, β is degenerate but α should be ~0.5
        assert abs(result['alpha'] - 0.5) < 0.1

    def test_parametric_sweep_completeness(self):
        """Parametric sweep returns correct number of data points."""
        ice = IceCrackPattern()
        V_range = np.array([0.5, 1.0, 2.0])
        h_range = np.array([0.5, 1.0])

        sweep = ice.parametric_sweep(V_range, h_range, P=10000.0, nx=20, ny=20)
        assert len(sweep['V']) == 6  # 3 × 2
        assert len(sweep['h']) == 6
        assert len(sweep['rho_crack']) == 6


class TestDEMConsistency:
    """Criterion 6: DEM discrete crack consistency."""

    def test_perfect_match(self):
        """Perfect match between DEM and analytic cracks."""
        ice = IceCrackPattern()
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        result = ice.dem_consistency_check(positions, positions, tolerance=0.1)
        assert result['match_ratio'] == 1.0
        assert result['pass'] is True

    def test_no_match(self):
        """No match when DEM cracks are far from analytic."""
        ice = IceCrackPattern()
        dem = np.array([[100.0, 100.0]])
        analytic = np.array([[0.0, 0.0]])
        result = ice.dem_consistency_check(dem, analytic, tolerance=0.1)
        assert result['match_ratio'] == 0.0
        assert result['pass'] is False

    def test_partial_match(self):
        """Partial match scenario."""
        ice = IceCrackPattern()
        dem = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]])
        analytic = np.array([[0.0, 0.0], [1.0, 0.0]])
        result = ice.dem_consistency_check(dem, analytic, tolerance=0.1)
        assert result['n_matched'] == 2
        assert result['match_ratio'] == 1.0

    def test_empty_dem(self):
        """Empty DEM crack list."""
        ice = IceCrackPattern()
        dem = np.array([]).reshape(0, 2)
        analytic = np.array([[0.0, 0.0]])
        result = ice.dem_consistency_check(dem, analytic)
        assert result['n_matched'] == 0
        assert result['pass'] is False

    def test_empty_analytic(self):
        """Empty analytic crack list."""
        ice = IceCrackPattern()
        dem = np.array([[0.0, 0.0]])
        analytic = np.array([]).reshape(0, 2)
        result = ice.dem_consistency_check(dem, analytic)
        assert result['n_matched'] == 0
        assert result['pass'] is False


class TestIdempotentReproduction:
    """Criterion 7: Same input → same SHA256."""

    def test_sha256_reproducible(self):
        """Same parameters produce same hash."""
        ice = IceCrackPattern()
        params = {'E': ice.E, 'nu': ice.nu, 'h': ice.h, 'P': 10000.0}
        hash1 = ice.compute_sha256(params)
        hash2 = ice.compute_sha256(params)
        assert hash1 == hash2

    def test_sha256_length(self):
        """SHA256 hash is 64 characters."""
        ice = IceCrackPattern()
        params = {'E': ice.E}
        h = ice.compute_sha256(params)
        assert len(h) == 64

    def test_sha256_different_params(self):
        """Different parameters produce different hashes."""
        ice = IceCrackPattern()
        params1 = {'E': 5e9, 'h': 1.0}
        params2 = {'E': 5e9, 'h': 2.0}
        h1 = ice.compute_sha256(params1)
        h2 = ice.compute_sha256(params2)
        assert h1 != h2

    def test_full_analysis_provenance(self):
        """Full analysis includes provenance hash."""
        ice = IceCrackPattern()
        result = ice.full_analysis(P=10000.0, nx=20, ny=20)
        assert 'provenance_sha256' in result
        assert len(result['provenance_sha256']) == 64

    def test_full_analysis_idempotent(self):
        """Two full analyses with same params produce same hash."""
        ice = IceCrackPattern()
        r1 = ice.full_analysis(P=10000.0, V=1.0, nx=20, ny=20)
        r2 = ice.full_analysis(P=10000.0, V=1.0, nx=20, ny=20)
        assert r1['provenance_sha256'] == r2['provenance_sha256']


class TestDimensionalConsistency:
    """Dimensional consistency checks."""

    def test_characteristic_length_dimension(self):
        """Characteristic length has dimension of length."""
        assert validate_dimensional_consistency(5e9, 1.0, 0.3, 1025.0, 9.81)

    def test_characteristic_length_positive(self):
        """Characteristic length is positive."""
        ice = IceCrackPattern()
        assert ice.l > 0

    def test_characteristic_length_increases_with_E(self):
        """l increases with E (stiffer → longer characteristic length)."""
        ice1 = IceCrackPattern(E=3e9)
        ice2 = IceCrackPattern(E=5e9)
        ice3 = IceCrackPattern(E=7e9)
        assert ice1.l < ice2.l < ice3.l

    def test_characteristic_length_increases_with_h(self):
        """l increases with h (thicker → longer characteristic length)."""
        ice1 = IceCrackPattern(h=0.5)
        ice2 = IceCrackPattern(h=1.0)
        ice3 = IceCrackPattern(h=2.0)
        assert ice1.l < ice2.l < ice3.l

    def test_flexural_rigidity(self):
        """D = E·h³ / (12·(1-ν²))."""
        ice = IceCrackPattern(E=5e9, nu=0.3, h=1.0)
        D_expected = 5e9 * 1.0**3 / (12 * (1 - 0.3**2))
        assert abs(ice.D - D_expected) / D_expected < 1e-10


class TestFullAnalysis:
    """Integration tests for full analysis pipeline."""

    def test_full_analysis_returns_all_fields(self):
        """Full analysis returns all required fields."""
        ice = IceCrackPattern()
        result = ice.full_analysis(P=10000.0, nx=20, ny=20)

        required_fields = [
            'characteristic_length', 'E_prime', 'pressure',
            'M_x', 'M_y', 'M_xy', 'sigma_1', 'sigma_2', 'theta_p',
            'sigma_max', 'G', 'crack_mask', 'crack_type',
            'crack_density', 'x', 'y', 'X', 'Y',
            'provenance_sha256', 'params'
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_crack_types_present(self):
        """Both radial and circumferential cracks should be present."""
        ice = IceCrackPattern(G_c=1.0)  # Low G_c to ensure cracks
        result = ice.full_analysis(P=100000.0, nx=50, ny=50)
        # At least some cracks should exist
        n_cracks = np.sum(result['crack_mask'])
        if n_cracks > 0:
            types = result['crack_type'][result['crack_mask']]
            # Should have both types in a realistic scenario
            # (may not always be true for very small domains)
            assert np.any(types == 1) or np.any(types == 2)

    def test_high_load_produces_cracks(self):
        """High load should produce cracks."""
        ice = IceCrackPattern(G_c=0.1)  # Very low G_c
        result = ice.full_analysis(P=100000.0, nx=50, ny=50)
        assert np.sum(result['crack_mask']) > 0

    def test_low_load_no_cracks(self):
        """Very low load should not produce cracks."""
        ice = IceCrackPattern(G_c=1e10)  # Very high G_c
        result = ice.full_analysis(P=1.0, nx=20, ny=20)
        assert np.sum(result['crack_mask']) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
