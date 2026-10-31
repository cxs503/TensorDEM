"""Pytest tests for ice ridge mechanics module (SIR-PC2 | M02).

Tests validate:
1. Kerr (1975) ridge sailing force model
2. Timco (2011) keel drag model
3. Combined ridge resistance
4. Tensor (batch) computation
5. Acceptance criteria:
   - ridge_sailing_force_error < 0.1 [ratio]
   - keel_drag_error_vs_Timco < 0.15 [ratio]
"""
import math
import pytest
import torch

from tensordem.ice_ridge import (
    DEFAULT_GRAVITY,
    DEFAULT_ICE_DENSITY,
    DEFAULT_WATER_DENSITY,
    RidgeGeometry,
    ShipConfig,
    RidgeResistanceResult,
    kerr_ridge_sailing_force,
    kerr_ridge_sailing_force_tensor,
    timco_keel_drag,
    timco_keel_drag_tensor,
    compute_ridge_resistance,
    kerr_reference_force,
    timco_reference_drag,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def standard_ridge():
    """Standard ice ridge geometry for testing."""
    return RidgeGeometry(
        sail_height=1.5,
        keel_draft=6.0,
        ridge_width=30.0,
        ridge_length=100.0,
        porosity=0.4,
        block_size=0.5,
        sail_to_keel_ratio=0.25,
    )


@pytest.fixture
def standard_ship():
    """Standard ship configuration for testing."""
    return ShipConfig(
        beam=20.0,
        draft=8.0,
        speed=2.0,
        bow_angle=20.0,
        hull_friction_coeff=0.1,
    )


# ---------------------------------------------------------------------------
# Kerr (1975) Sailing Force Tests
# ---------------------------------------------------------------------------

class TestKerrSailingForce:
    """Tests for Kerr (1975) ridge sailing force model."""

    def test_positive_force(self, standard_ridge, standard_ship):
        """Sailing force must be positive."""
        F = kerr_ridge_sailing_force(standard_ridge, standard_ship)
        assert F > 0, f"Expected positive force, got {F}"

    def test_scales_with_sail_height_squared(self, standard_ship):
        """Force should scale approximately with h_s^2."""
        ridge1 = RidgeGeometry(sail_height=1.0)
        ridge2 = RidgeGeometry(sail_height=2.0)
        F1 = kerr_ridge_sailing_force(ridge1, standard_ship)
        F2 = kerr_ridge_sailing_force(ridge2, standard_ship)
        ratio = F2 / F1
        # Should be close to 4 (2^2), but speed correction modifies slightly
        assert 3.5 < ratio < 4.5, f"Expected ~4x scaling, got {ratio:.2f}"

    def test_scales_with_beam(self, standard_ridge):
        """Force should scale linearly with ship beam."""
        ship1 = ShipConfig(beam=10.0, speed=2.0, bow_angle=20.0)
        ship2 = ShipConfig(beam=20.0, speed=2.0, bow_angle=20.0)
        F1 = kerr_ridge_sailing_force(standard_ridge, ship1)
        F2 = kerr_ridge_sailing_force(standard_ridge, ship2)
        ratio = F2 / F1
        assert abs(ratio - 2.0) < 0.05, f"Expected 2x scaling, got {ratio:.2f}"

    def test_increases_with_speed(self, standard_ridge, standard_ship):
        """Force should increase with ship speed."""
        ship_slow = ShipConfig(beam=20.0, speed=1.0, bow_angle=20.0)
        ship_fast = ShipConfig(beam=20.0, speed=4.0, bow_angle=20.0)
        F_slow = kerr_ridge_sailing_force(standard_ridge, ship_slow)
        F_fast = kerr_ridge_sailing_force(standard_ridge, ship_fast)
        assert F_fast > F_slow, "Force should increase with speed"

    def test_decreases_with_wider_bow(self, standard_ridge, standard_ship):
        """Force should decrease with wider bow angle."""
        ship_narrow = ShipConfig(beam=20.0, speed=2.0, bow_angle=15.0)
        ship_wide = ShipConfig(beam=20.0, speed=2.0, bow_angle=30.0)
        F_narrow = kerr_ridge_sailing_force(standard_ridge, ship_narrow)
        F_wide = kerr_ridge_sailing_force(standard_ridge, ship_wide)
        assert F_narrow > F_wide, "Narrower bow should produce higher force"

    def test_reference_value(self):
        """Test against known reference value from Kerr (1975).

        For standard parameters (h_s=1.5m, B=20m, V=2m/s, phi=20°):
        F_ref = C_k * rho_i * g * h_s^2 * B * f_speed * f_bow
        where C_k=0.65, f_speed = 1 + 0.3*V/sqrt(g*h_s), f_bow = 1/sin(phi)
        """
        F_ref = kerr_reference_force(
            sail_height=1.5, beam=20.0, speed=2.0, bow_angle_deg=20.0
        )
        # Manual calculation for verification:
        h_s = 1.5
        B = 20.0
        V = 2.0
        phi = math.radians(20.0)
        C_k = 0.65
        alpha = 0.3
        f_speed = 1.0 + alpha * V / math.sqrt(DEFAULT_GRAVITY * h_s)
        f_bow = 1.0 / math.sin(phi)
        F_manual = C_k * DEFAULT_ICE_DENSITY * DEFAULT_GRAVITY * h_s**2 * B * f_speed * f_bow

        error = abs(F_ref - F_manual) / F_manual
        assert error < 0.01, f"Reference force error {error:.4f} > 1%"

    def test_acceptance_criterion_error_less_than_0_1(self, standard_ridge, standard_ship):
        """Acceptance criterion: ridge_sailing_force_error < 0.1 [ratio].

        Verify that our implementation matches the reference formula
        within 10% relative error.
        """
        F_computed = kerr_ridge_sailing_force(standard_ridge, standard_ship)
        F_reference = kerr_reference_force(
            sail_height=standard_ridge.sail_height,
            beam=standard_ship.beam,
            speed=standard_ship.speed,
            bow_angle_deg=standard_ship.bow_angle,
        )
        error = abs(F_computed - F_reference) / F_reference
        assert error < 0.1, (
            f"ridge_sailing_force_error = {error:.4f} >= 0.1 (FAIL)"
        )


# ---------------------------------------------------------------------------
# Timco (2011) Keel Drag Tests
# ---------------------------------------------------------------------------

class TestTimcoKeelDrag:
    """Tests for Timco (2011) ridge keel drag model."""

    def test_positive_drag(self, standard_ridge, standard_ship):
        """Keel drag must be positive."""
        F = timco_keel_drag(standard_ridge, standard_ship)
        assert F > 0, f"Expected positive drag, got {F}"

    def test_scales_with_keel_draft_squared(self, standard_ship):
        """Drag should scale approximately with h_k^2."""
        ridge1 = RidgeGeometry(keel_draft=4.0, porosity=0.4)
        ridge2 = RidgeGeometry(keel_draft=8.0, porosity=0.4)
        F1 = timco_keel_drag(ridge1, standard_ship)
        F2 = timco_keel_drag(ridge2, standard_ship)
        ratio = F2 / F1
        # Should be close to 4 (2^2), but speed correction modifies slightly
        assert 3.5 < ratio < 4.5, f"Expected ~4x scaling, got {ratio:.2f}"

    def test_decreases_with_porosity(self, standard_ship):
        """Drag should decrease with higher porosity."""
        ridge_low_por = RidgeGeometry(keel_draft=6.0, porosity=0.2)
        ridge_high_por = RidgeGeometry(keel_draft=6.0, porosity=0.6)
        F_low = timco_keel_drag(ridge_low_por, standard_ship)
        F_high = timco_keel_drag(ridge_high_por, standard_ship)
        assert F_low > F_high, "Higher porosity should reduce drag"

    def test_increases_with_speed(self, standard_ridge, standard_ship):
        """Drag should increase with ship speed."""
        ship_slow = ShipConfig(beam=20.0, speed=1.0, bow_angle=20.0)
        ship_fast = ShipConfig(beam=20.0, speed=4.0, bow_angle=20.0)
        F_slow = timco_keel_drag(standard_ridge, ship_slow)
        F_fast = timco_keel_drag(standard_ridge, ship_fast)
        assert F_fast > F_slow, "Drag should increase with speed"

    def test_reference_value(self):
        """Test against known reference value from Timco (2011).

        For standard parameters (h_k=6m, B=20m, V=2m/s, n=0.4, phi=20°):
        F_ref = rho_w * g * h_k^2 * B * C_R * (1-n) * f_V
        where C_R=0.85, f_V = 1 + 0.15*V
        """
        F_ref = timco_reference_drag(
            keel_draft=6.0, beam=20.0, speed=2.0,
            porosity=0.4, bow_angle_deg=20.0, ridge_width=30.0,
        )
        # Manual calculation:
        h_k = 6.0
        B = 20.0
        V = 2.0
        n = 0.4
        C_R = 0.85
        beta = 0.15
        f_V = 1.0 + beta * V
        F_manual = DEFAULT_WATER_DENSITY * DEFAULT_GRAVITY * h_k**2 * B * C_R * (1 - n) * f_V

        error = abs(F_ref - F_manual) / F_manual
        assert error < 0.01, f"Reference drag error {error:.4f} > 1%"

    def test_acceptance_criterion_error_less_than_0_15(self, standard_ridge, standard_ship):
        """Acceptance criterion: keel_drag_error_vs_Timco < 0.15 [ratio].

        Verify that our implementation matches the Timco reference formula
        within 15% relative error.
        """
        F_computed = timco_keel_drag(standard_ridge, standard_ship)
        F_reference = timco_reference_drag(
            keel_draft=standard_ridge.keel_draft,
            beam=standard_ship.beam,
            speed=standard_ship.speed,
            porosity=standard_ridge.porosity,
            bow_angle_deg=standard_ship.bow_angle,
            ridge_width=standard_ridge.ridge_width,
        )
        error = abs(F_computed - F_reference) / F_reference
        assert error < 0.15, (
            f"keel_drag_error_vs_Timco = {error:.4f} >= 0.15 (FAIL)"
        )


# ---------------------------------------------------------------------------
# Combined Ridge Resistance Tests
# ---------------------------------------------------------------------------

class TestCombinedRidgeResistance:
    """Tests for combined ridge resistance computation."""

    def test_total_is_sum(self, standard_ridge, standard_ship):
        """Total resistance should equal sailing force + keel drag."""
        result = compute_ridge_resistance(standard_ridge, standard_ship)
        expected_total = result.sailing_force + result.keel_drag
        assert abs(result.total_resistance - expected_total) < 1e-6

    def test_result_type(self, standard_ridge, standard_ship):
        """Result should be a RidgeResistanceResult."""
        result = compute_ridge_resistance(standard_ridge, standard_ship)
        assert isinstance(result, RidgeResistanceResult)

    def test_all_positive(self, standard_ridge, standard_ship):
        """All force components should be positive."""
        result = compute_ridge_resistance(standard_ridge, standard_ship)
        assert result.sailing_force > 0
        assert result.keel_drag > 0
        assert result.total_resistance > 0

    def test_typical_magnitude(self, standard_ridge, standard_ship):
        """Forces should be in reasonable range (kN to MN for typical ships)."""
        result = compute_ridge_resistance(standard_ridge, standard_ship)
        # Typical ridge resistance for a medium ship: 100 kN to 10 MN
        assert 1e5 < result.total_resistance < 1e8, (
            f"Total resistance {result.total_resistance:.0f} N out of expected range"
        )


# ---------------------------------------------------------------------------
# Tensor (Batch) Computation Tests
# ---------------------------------------------------------------------------

class TestTensorComputation:
    """Tests for vectorised tensor computation."""

    def test_kerr_tensor_matches_scalar(self):
        """Tensor computation should match scalar computation."""
        N = 5
        sail_heights = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0])
        beams = torch.tensor([15.0, 20.0, 25.0, 30.0, 35.0])
        speeds = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        bow_angles = torch.tensor([15.0, 20.0, 25.0, 30.0, 35.0])

        F_tensor = kerr_ridge_sailing_force_tensor(
            sail_heights, beams, speeds, bow_angles
        )

        for i in range(N):
            ridge = RidgeGeometry(sail_height=sail_heights[i].item())
            ship = ShipConfig(
                beam=beams[i].item(),
                speed=speeds[i].item(),
                bow_angle=bow_angles[i].item(),
            )
            F_scalar = kerr_ridge_sailing_force(ridge, ship)
            assert abs(F_tensor[i].item() - F_scalar) / F_scalar < 1e-5

    def test_timco_tensor_matches_scalar(self):
        """Tensor computation should match scalar computation."""
        N = 5
        keel_drafts = torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0])
        beams = torch.tensor([15.0, 20.0, 25.0, 30.0, 35.0])
        speeds = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        porosities = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6])
        bow_angles = torch.tensor([15.0, 20.0, 25.0, 30.0, 35.0])
        ridge_widths = torch.tensor([20.0, 25.0, 30.0, 35.0, 40.0])

        F_tensor = timco_keel_drag_tensor(
            keel_drafts, beams, speeds, porosities, bow_angles, ridge_widths
        )

        for i in range(N):
            ridge = RidgeGeometry(
                keel_draft=keel_drafts[i].item(),
                porosity=porosities[i].item(),
                ridge_width=ridge_widths[i].item(),
            )
            ship = ShipConfig(
                beam=beams[i].item(),
                speed=speeds[i].item(),
                bow_angle=bow_angles[i].item(),
            )
            F_scalar = timco_keel_drag(ridge, ship)
            assert abs(F_tensor[i].item() - F_scalar) / F_scalar < 1e-5

    def test_tensor_device_consistency(self):
        """Tensor functions should work on different devices."""
        sail_heights = torch.tensor([1.5, 2.0])
        beams = torch.tensor([20.0, 25.0])
        speeds = torch.tensor([2.0, 3.0])
        bow_angles = torch.tensor([20.0, 25.0])

        F_cpu = kerr_ridge_sailing_force_tensor(
            sail_heights, beams, speeds, bow_angles
        )
        assert F_cpu.device.type == "cpu"


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_zero_speed(self, standard_ridge):
        """At zero speed, force should still be positive (static resistance)."""
        ship = ShipConfig(beam=20.0, speed=0.0, bow_angle=20.0)
        F_sail = kerr_ridge_sailing_force(standard_ridge, ship)
        F_keel = timco_keel_drag(standard_ridge, ship)
        assert F_sail > 0, "Static sailing force should be positive"
        assert F_keel > 0, "Static keel drag should be positive"

    def test_very_small_sail(self, standard_ship):
        """Very small sail height should give small force."""
        ridge = RidgeGeometry(sail_height=0.1)
        F = kerr_ridge_sailing_force(ridge, standard_ship)
        assert F > 0
        assert F < 1e5  # Should be relatively small

    def test_high_porosity(self, standard_ship):
        """Very high porosity (0.9) should give very low keel drag."""
        ridge = RidgeGeometry(keel_draft=6.0, porosity=0.9)
        F = timco_keel_drag(ridge, standard_ship)
        assert F > 0
        # Compare with normal porosity
        ridge_normal = RidgeGeometry(keel_draft=6.0, porosity=0.4)
        F_normal = timco_keel_drag(ridge_normal, standard_ship)
        assert F < F_normal * 0.3, "High porosity should significantly reduce drag"

    def test_extreme_bow_angle(self, standard_ridge):
        """Very wide bow (45°) should give lower force than narrow bow."""
        ship_wide = ShipConfig(beam=20.0, speed=2.0, bow_angle=45.0)
        ship_narrow = ShipConfig(beam=20.0, speed=2.0, bow_angle=10.0)
        F_wide = kerr_ridge_sailing_force(standard_ridge, ship_wide)
        F_narrow = kerr_ridge_sailing_force(standard_ridge, ship_narrow)
        assert F_narrow > F_wide


# ---------------------------------------------------------------------------
# Acceptance Criterion Summary Test
# ---------------------------------------------------------------------------

class TestAcceptanceCriteria:
    """Final acceptance criterion verification."""

    def test_ridge_sailing_force_error(self, standard_ridge, standard_ship):
        """ridge_sailing_force_error < 0.1 [ratio]."""
        F_computed = kerr_ridge_sailing_force(standard_ridge, standard_ship)
        F_reference = kerr_reference_force(
            sail_height=standard_ridge.sail_height,
            beam=standard_ship.beam,
            speed=standard_ship.speed,
            bow_angle_deg=standard_ship.bow_angle,
        )
        error = abs(F_computed - F_reference) / F_reference
        assert error < 0.1, f"FAIL: ridge_sailing_force_error = {error:.4f}"

    def test_keel_drag_error_vs_Timco(self, standard_ridge, standard_ship):
        """keel_drag_error_vs_Timco < 0.15 [ratio]."""
        F_computed = timco_keel_drag(standard_ridge, standard_ship)
        F_reference = timco_reference_drag(
            keel_draft=standard_ridge.keel_draft,
            beam=standard_ship.beam,
            speed=standard_ship.speed,
            porosity=standard_ridge.porosity,
            bow_angle_deg=standard_ship.bow_angle,
            ridge_width=standard_ridge.ridge_width,
        )
        error = abs(F_computed - F_reference) / F_reference
        assert error < 0.15, f"FAIL: keel_drag_error_vs_Timco = {error:.4f}"
