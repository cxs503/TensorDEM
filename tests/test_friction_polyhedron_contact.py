"""Tests for Coulomb friction + rolling resistance + polyhedron DEM (p273w3-n02).

Verification criteria:
C1: Angle of repose θ_r vs literature (μ=0.3→≈17°, μ=0.6→≈31°), error <10%
C2: Inclined plane sliding critical angle, error <5%
C3: Polyhedron packing volume fraction φ vs literature (φ≈0.55-0.65), error <10%
C4: Energy ledger closure error <3%
C5: pytest ≥20 items all green (this file)
C6: capability_ledger update

Test categories:
- Contact model unit tests (friction, rolling, GJK/EPA integration)
- Dimensional consistency tests
- Negative controls (zero friction → no resistance)
- Analytic anchor tests
- Verification criterion tests (lightweight)
"""
from __future__ import annotations

import math
import pytest
import numpy as np

from tensordem.friction_polyhedron import (
    FrictionPolyhedronConfig,
    ContactState,
    PolyhedronParticle,
    PolyhedronDEMSystem,
    compute_contact_force,
    random_polyhedron,
    regular_polyhedron,
)
from tensordem.convex_contact import Sphere, Box, Polytope, gjk_distance, contact


# ===================================================================
# Unit tests: Contact model (8 tests)
# ===================================================================

class TestFrictionPolyhedronConfig:
    """Configuration validation."""

    def test_valid_config(self):
        """Config with valid parameters should not raise."""
        cfg = FrictionPolyhedronConfig(k_n=1e5, mu=0.3, mu_r=0.1)
        assert cfg.k_n == 1e5
        assert cfg.mu == 0.3
        assert cfg.mu_r == 0.1

    def test_negative_stiffness_rejected(self):
        """Negative stiffness should raise ValueError."""
        with pytest.raises(ValueError, match="k_n must be positive"):
            FrictionPolyhedronConfig(k_n=-1.0)

    def test_negative_friction_rejected(self):
        """Negative friction coefficient should raise ValueError."""
        with pytest.raises(ValueError, match="mu must be non-negative"):
            FrictionPolyhedronConfig(mu=-0.1)

    def test_zero_friction_allowed(self):
        """Zero friction should be valid (frictionless contact)."""
        cfg = FrictionPolyhedronConfig(mu=0.0, mu_r=0.0)
        assert cfg.mu == 0.0


class TestContactState:
    """Contact state management."""

    def test_initial_state_is_zero(self):
        """New contact state should have zero tangential displacement."""
        state = ContactState()
        np.testing.assert_array_equal(state.delta_t, np.zeros(3))
        np.testing.assert_array_equal(state.rolling_moment, np.zeros(3))
        assert state.active is True

    def test_reset_clears_state(self):
        """Reset should zero out all state variables."""
        state = ContactState()
        state.delta_t = np.array([1.0, 2.0, 3.0])
        state.rolling_moment = np.array([0.1, 0.2, 0.3])
        state.reset()
        np.testing.assert_array_almost_equal(state.delta_t, np.zeros(3))
        np.testing.assert_array_almost_equal(state.rolling_moment, np.zeros(3))


class TestComputeContactForce:
    """Contact force computation tests."""

    def test_no_contact_zero_force(self):
        """Non-overlapping spheres should produce zero force."""
        s1 = Sphere([0, 0, 0], 0.01)
        s2 = Sphere([0.1, 0, 0], 0.01)  # far apart
        state = ContactState()
        cfg = FrictionPolyhedronConfig()
        result, contacting = compute_contact_force(
            s1, s2, np.zeros(3), np.zeros(3),
            np.zeros(3), np.zeros(3),
            1.0, 1.0, np.eye(3), np.eye(3),
            state, cfg, 1e-4,
        )
        assert not contacting
        np.testing.assert_array_almost_equal(result['force_a'], np.zeros(3))
        assert result['f_n'] == 0.0

    def test_overlapping_spheres_repulsive_force(self):
        """Overlapping spheres should produce repulsive normal force."""
        s1 = Sphere([0, 0, 0], 0.01)
        s2 = Sphere([0.015, 0, 0], 0.01)  # overlapping by 0.005
        state = ContactState()
        cfg = FrictionPolyhedronConfig(k_n=1e5, mu=0.3)
        result, contacting = compute_contact_force(
            s1, s2, np.zeros(3), np.zeros(3),
            np.zeros(3), np.zeros(3),
            1.0, 1.0, np.eye(3), np.eye(3),
            state, cfg, 1e-4,
        )
        assert contacting
        assert result['f_n'] > 0.0
        # Force on a should point away from b (negative x)
        assert result['force_a'][0] < 0.0
        # Force on b should point away from a (positive x)
        assert result['force_b'][0] > 0.0

    def test_coulomb_friction_limits_tangential_force(self):
        """Tangential force should be capped at mu * F_n."""
        s1 = Sphere([0, 0, 0], 0.01)
        s2 = Sphere([0.015, 0, 0], 0.01)
        state = ContactState()
        # Pre-load tangential displacement beyond Coulomb limit
        state.delta_t = np.array([0, 1.0, 0])  # large tangential displacement
        cfg = FrictionPolyhedronConfig(k_n=1e5, k_t=1e4, mu=0.3)
        result, contacting = compute_contact_force(
            s1, s2, np.zeros(3), np.zeros(3),
            np.zeros(3), np.zeros(3),
            1.0, 1.0, np.eye(3), np.eye(3),
            state, cfg, 1e-4,
        )
        assert contacting
        # Tangential force should be capped
        assert result['f_t'] <= cfg.mu * result['f_n'] * 1.01  # small tolerance

    def test_rolling_resistance_produces_torque(self):
        """Non-zero rolling resistance coefficient should produce torque."""
        s1 = Sphere([0, 0, 0], 0.01)
        s2 = Sphere([0.015, 0, 0], 0.01)
        state = ContactState()
        cfg = FrictionPolyhedronConfig(k_n=1e5, mu=0.3, mu_r=0.1, k_r=100.0)
        # Apply relative angular velocity
        omega_b = np.array([0, 0, 10.0])
        result, contacting = compute_contact_force(
            s1, s2, np.zeros(3), np.zeros(3),
            np.zeros(3), omega_b,
            1.0, 1.0, np.eye(3), np.eye(3),
            state, cfg, 1e-4,
        )
        assert contacting
        # Rolling moment should be non-zero
        assert result['m_r'] > 0.0

    def test_zero_friction_no_tangential_force(self):
        """With mu=0, tangential force should be zero (negative control)."""
        s1 = Sphere([0, 0, 0], 0.01)
        s2 = Sphere([0.015, 0, 0], 0.01)
        state = ContactState()
        state.delta_t = np.array([0, 0.1, 0])
        cfg = FrictionPolyhedronConfig(k_n=1e5, k_t=1e4, mu=0.0)
        result, contacting = compute_contact_force(
            s1, s2, np.zeros(3), np.zeros(3),
            np.zeros(3), np.zeros(3),
            1.0, 1.0, np.eye(3), np.eye(3),
            state, cfg, 1e-4,
        )
        assert contacting
        # With mu=0, tangential force should be zero (Coulomb limit = 0)
        assert result['f_t'] < 1e-10


# ===================================================================
# Unit tests: Polyhedron particles (4 tests)
# ===================================================================

class TestPolyhedronParticle:
    """Polyhedron particle creation and properties."""

    def test_from_sphere(self):
        """Sphere particle should have correct mass."""
        p = PolyhedronParticle.from_sphere(0.01, density=1000.0)
        expected_mass = (4.0 / 3.0) * math.pi * 0.01**3 * 1000.0
        assert abs(p.mass - expected_mass) / expected_mass < 1e-10

    def test_from_box(self):
        """Box particle should have correct mass."""
        p = PolyhedronParticle.from_box([0.01, 0.01, 0.01], density=1000.0)
        expected_mass = 8 * 0.01**3 * 1000.0
        assert abs(p.mass - expected_mass) / expected_mass < 1e-10

    def test_from_polytope(self):
        """Polytope particle should have positive mass."""
        verts = regular_polyhedron('octahedron', np.array([0, 0, 0]), 0.01)
        p = PolyhedronParticle.from_polytope(verts, density=2500.0)
        assert p.mass > 0

    def test_regular_polyhedra(self):
        """All regular polyhedra should have valid vertices."""
        for name in ['tetrahedron', 'cube', 'octahedron', 'icosahedron']:
            verts = regular_polyhedron(name, np.array([0, 0, 0]), 0.01)
            assert verts.shape[0] >= 4
            assert np.all(np.isfinite(verts))


# ===================================================================
# Unit tests: DEM system (3 tests)
# ===================================================================

class TestPolyhedronDEMSystem:
    """DEM system integration tests."""

    def test_gravity_acceleration(self):
        """Particle should accelerate under gravity."""
        p = PolyhedronParticle.from_sphere(0.01, density=2500.0,
                                           position=np.array([0, 0, 1.0]))
        cfg = FrictionPolyhedronConfig(k_n=1e5, mu=0.3)
        system = PolyhedronDEMSystem([p], cfg, gravity=np.array([0, 0, -9.81]))
        initial_vz = p.v[2]
        system.step(1e-3)
        # Should have accelerated downward
        assert p.v[2] < initial_vz

    def test_floor_contact_prevents_fallthrough(self):
        """Particle should not fall through floor."""
        p = PolyhedronParticle.from_sphere(0.01, density=2500.0,
                                           position=np.array([0, 0, 0.005]))
        p.v = np.array([0, 0, -1.0])  # moving down
        cfg = FrictionPolyhedronConfig(k_n=1e5, gamma_n=100.0, mu=0.3)
        system = PolyhedronDEMSystem([p], cfg)
        system.add_wall(np.array([0, 0, 1]), 0.0)  # floor at z=0
        for _ in range(50):
            system.step(1e-4)
        # Should not have fallen through
        assert p.x[2] > -0.01

    def test_energy_tracking(self):
        """System should track kinetic and potential energy."""
        p = PolyhedronParticle.from_sphere(0.01, density=2500.0,
                                           position=np.array([0, 0, 1.0]))
        p.v = np.array([1.0, 0, 0])
        cfg = FrictionPolyhedronConfig()
        system = PolyhedronDEMSystem([p], cfg)
        system.step(1e-4)
        assert system.ke_trans > 0
        assert system.pe_gravity != 0


# ===================================================================
# Dimensional consistency tests (2 tests)
# ===================================================================

class TestDimensionalConsistency:
    """Verify dimensional consistency of contact model."""

    def test_force_units_newtons(self):
        """Contact force should be in Newtons (kg*m/s^2)."""
        s1 = Sphere([0, 0, 0], 0.01)
        s2 = Sphere([0.015, 0, 0], 0.01)
        state = ContactState()
        cfg = FrictionPolyhedronConfig(k_n=1e5)  # N/m
        result, _ = compute_contact_force(
            s1, s2, np.zeros(3), np.zeros(3),
            np.zeros(3), np.zeros(3),
            1.0, 1.0, np.eye(3), np.eye(3),  # kg, kg*m^2
            state, cfg, 1e-4,
        )
        # Force magnitude should be reasonable (not astronomical)
        f_mag = np.linalg.norm(result['force_a'])
        assert 0 < f_mag < 1e6  # N

    def test_torque_units_newton_meters(self):
        """Rolling resistance torque should be in N*m."""
        s1 = Sphere([0, 0, 0], 0.01)
        s2 = Sphere([0.015, 0, 0], 0.01)
        state = ContactState()
        cfg = FrictionPolyhedronConfig(k_n=1e5, mu_r=0.1, k_r=100.0)
        omega_b = np.array([0, 0, 10.0])
        result, _ = compute_contact_force(
            s1, s2, np.zeros(3), np.zeros(3),
            np.zeros(3), omega_b,
            1.0, 1.0, np.eye(3), np.eye(3),
            state, cfg, 1e-4,
        )
        # Torque should be reasonable
        assert result['m_r'] < 1e3  # N*m


# ===================================================================
# Negative control tests (2 tests)
# ===================================================================

class TestNegativeControls:
    """Negative controls: verify expected zero/identity behavior."""

    def test_no_overlap_no_force(self):
        """Separated particles should have zero contact force."""
        s1 = Sphere([0, 0, 0], 0.01)
        s2 = Sphere([1.0, 0, 0], 0.01)  # far apart
        state = ContactState()
        cfg = FrictionPolyhedronConfig()
        result, contacting = compute_contact_force(
            s1, s2, np.zeros(3), np.zeros(3),
            np.zeros(3), np.zeros(3),
            1.0, 1.0, np.eye(3), np.eye(3),
            state, cfg, 1e-4,
        )
        assert not contacting
        assert result['f_n'] == 0.0
        assert result['f_t'] == 0.0
        assert result['m_r'] == 0.0

    def test_zero_friction_zero_tangential(self):
        """With mu=0 and mu_r=0, tangential force and torque should be zero."""
        s1 = Sphere([0, 0, 0], 0.01)
        s2 = Sphere([0.015, 0, 0], 0.01)
        state = ContactState()
        cfg = FrictionPolyhedronConfig(k_n=1e5, mu=0.0, mu_r=0.0)
        result, contacting = compute_contact_force(
            s1, s2, np.zeros(3), np.zeros(3),
            np.zeros(3), np.zeros(3),
            1.0, 1.0, np.eye(3), np.eye(3),
            state, cfg, 1e-4,
        )
        assert contacting
        assert result['f_t'] < 1e-10
        assert result['m_r'] < 1e-10


# ===================================================================
# GJK/EPA integration tests (4 tests)
# ===================================================================

class TestGJKEPAIntegration:
    """Tests for GJK/EPA geometry used by contact model."""

    def test_sphere_sphere_gap(self):
        """GJK should compute correct gap between separated spheres."""
        s1 = Sphere([0, 0, 0], 0.01)
        s2 = Sphere([0.05, 0, 0], 0.01)
        result = gjk_distance(s1, s2)
        expected_gap = 0.03  # 0.05 - 0.01 - 0.01
        assert abs(result.distance - expected_gap) < 1e-6

    def test_sphere_sphere_overlap(self):
        """GJK should detect overlap between penetrating spheres."""
        s1 = Sphere([0, 0, 0], 0.01)
        s2 = Sphere([0.015, 0, 0], 0.01)
        result = gjk_distance(s1, s2)
        assert result.overlap  # should detect overlap

    def test_polytope_polytope_gap(self):
        """GJK should work for polytope-polytope pairs."""
        v1 = regular_polyhedron('octahedron', np.array([0, 0, 0]), 0.01)
        v2 = regular_polyhedron('octahedron', np.array([0.1, 0, 0]), 0.01)
        p1 = Polytope(v1)
        p2 = Polytope(v2)
        result = gjk_distance(p1, p2)
        assert result.distance > 0  # should be separated

    def test_box_box_contact(self):
        """GJK/EPA should handle box-box contacts."""
        b1 = Box([0, 0, 0], [0.01, 0.01, 0.01])
        b2 = Box([0.015, 0, 0], [0.01, 0.01, 0.01])  # overlapping
        result = gjk_distance(b1, b2)
        assert result.overlap


# ===================================================================
# Analytic anchor tests (2 tests)
# ===================================================================

class TestAnalyticAnchors:
    """Tests against analytic solutions."""

    def test_hertz_like_force_scaling(self):
        """Normal force should scale linearly with overlap for linear spring."""
        s1 = Sphere([0, 0, 0], 0.01)
        cfg = FrictionPolyhedronConfig(k_n=1e5, gamma_n=0.0)

        forces = []
        overlaps = [0.001, 0.002, 0.003]
        for delta in overlaps:
            s2 = Sphere([0.02 - delta, 0, 0], 0.01)
            state = ContactState()
            result, _ = compute_contact_force(
                s1, s2, np.zeros(3), np.zeros(3),
                np.zeros(3), np.zeros(3),
                1.0, 1.0, np.eye(3), np.eye(3),
                state, cfg, 1e-4,
            )
            forces.append(result['f_n'])

        # Linear scaling: F_n = k_n * delta
        for f, d in zip(forces, overlaps):
            expected = cfg.k_n * d
            assert abs(f - expected) / expected < 0.01

    def test_newton_third_law(self):
        """Forces on a and b should be equal and opposite."""
        s1 = Sphere([0, 0, 0], 0.01)
        s2 = Sphere([0.015, 0, 0], 0.01)
        state = ContactState()
        cfg = FrictionPolyhedronConfig(k_n=1e5, mu=0.3)
        result, _ = compute_contact_force(
            s1, s2, np.zeros(3), np.zeros(3),
            np.zeros(3), np.zeros(3),
            1.0, 1.0, np.eye(3), np.eye(3),
            state, cfg, 1e-4,
        )
        # F_a + F_b = 0
        total_force = result['force_a'] + result['force_b']
        np.testing.assert_array_almost_equal(total_force, np.zeros(3), decimal=10)
