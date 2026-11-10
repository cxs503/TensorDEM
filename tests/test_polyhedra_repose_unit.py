"""Ultra-fast unit tests for polyhedral repose (no DEM simulation).

These tests validate the measurement functions and basic setup without
running the full DEM simulation, completing in < 5 seconds.
"""
from __future__ import annotations

import math
import pytest
import numpy as np

from tensordem.polyhedra_repose import (
    PolyhedraReposeConfig,
    measure_repose_angle,
    measure_volume_fraction,
)
from tensordem.friction_polyhedron import (
    FrictionPolyhedronConfig,
    PolyhedronParticle,
    random_polyhedron,
)
from tensordem.convex_contact import Polytope


class TestMeasurementFunctions:
    """Unit tests for measurement functions (no simulation)."""

    def test_repose_angle_empty(self):
        """Empty list → θ = 0°."""
        result = measure_repose_angle([], container_radius=0.05)
        assert result['theta_deg'] == 0.0
        assert result['slope'] == 0.0

    def test_repose_angle_flat_layer(self):
        """Flat layer → small θ."""
        particles = []
        for i in range(10):
            x = 0.01 * (i % 5 - 2)
            y = 0.01 * (i // 5 - 0.5)
            verts = random_polyhedron(np.array([x, y, 0.005]), 0.004, seed=i)
            p = PolyhedronParticle.from_polytope(verts)
            particles.append(p)
        
        result = measure_repose_angle(particles, container_radius=0.05)
        assert 0.0 <= result['theta_deg'] < 45.0
        assert 'pile_height' in result
        assert 'pile_radius' in result

    def test_repose_angle_steep_pile(self):
        """Steep conical pile → large θ."""
        particles = []
        for i in range(15):
            r = 0.003 * (i % 5)
            theta = 2 * math.pi * i / 15
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            z = 0.02 - r * 1.5  # steep slope
            verts = random_polyhedron(np.array([x, y, z]), 0.003, seed=i)
            p = PolyhedronParticle.from_polytope(verts)
            particles.append(p)
        
        result = measure_repose_angle(particles, container_radius=0.05)
        assert result['theta_deg'] > 10.0

    def test_volume_fraction_single_particle(self):
        """Single particle → small φ."""
        verts = random_polyhedron(np.array([0, 0, 0.01]), 0.005, seed=0)
        p = PolyhedronParticle.from_polytope(verts)
        result = measure_volume_fraction([p], container_radius=0.05)
        
        assert 0.0 < result['phi'] < 0.5
        assert result['v_particles'] > 0
        assert result['v_container'] > 0
        assert result['deposit_height'] > 0

    def test_volume_fraction_multiple_particles(self):
        """Multiple particles → reasonable φ."""
        particles = []
        for i in range(10):
            x = 0.008 * (i % 3 - 1)
            y = 0.008 * (i // 3 - 1.5)
            verts = random_polyhedron(np.array([x, y, 0.01]), 0.004, seed=i)
            p = PolyhedronParticle.from_polytope(verts)
            particles.append(p)
        
        result = measure_volume_fraction(particles, container_radius=0.05)
        assert 0.0 < result['phi'] < 1.0


class TestConfiguration:
    """Configuration validation tests."""

    def test_default_config(self):
        """Default config should be valid."""
        config = PolyhedraReposeConfig()
        assert config.n_particles == 100
        assert config.mu == 0.4
        assert config.mu_r == 0.05
        assert config.dt > 0

    def test_custom_config(self):
        """Custom config should override defaults."""
        config = PolyhedraReposeConfig(
            n_particles=50,
            mu=0.5,
            mu_r=0.1,
            seed=123,
        )
        assert config.n_particles == 50
        assert config.mu == 0.5
        assert config.mu_r == 0.1
        assert config.seed == 123

    def test_friction_config(self):
        """FrictionPolyhedronConfig should validate."""
        cfg = FrictionPolyhedronConfig(k_n=1e5, mu=0.3, mu_r=0.05)
        assert cfg.k_n == 1e5
        assert cfg.mu == 0.3
        assert cfg.mu_r == 0.05

    def test_friction_config_invalid(self):
        """Invalid friction config should raise."""
        with pytest.raises(ValueError):
            FrictionPolyhedronConfig(k_n=-1.0)
        
        with pytest.raises(ValueError):
            FrictionPolyhedronConfig(mu=-0.1)


class TestParticleGeneration:
    """Particle generation tests."""

    def test_random_polyhedron(self):
        """Random polyhedron should have correct shape."""
        verts = random_polyhedron(np.array([0, 0, 0]), 0.005, n_vertices=12, seed=42)
        assert verts.shape == (12, 3)
        assert np.all(np.isfinite(verts))

    def test_polyhedron_particle_from_vertices(self):
        """PolyhedronParticle should compute mass and inertia."""
        verts = random_polyhedron(np.array([0, 0, 0.01]), 0.005, seed=0)
        p = PolyhedronParticle.from_polytope(verts, density=2500.0)
        
        assert p.mass > 0
        assert p.inertia.shape == (3, 3)
        assert np.all(np.isfinite(p.x))
        assert np.all(np.isfinite(p.v))

    def test_polytope_shape(self):
        """Polytope should store vertices."""
        verts = random_polyhedron(np.array([0, 0, 0]), 0.005, seed=0)
        shape = Polytope(verts)
        
        assert shape.vertices.shape == verts.shape
        assert shape.bounding_radius() > 0


class TestContactConfig:
    """Contact configuration tests."""

    def test_contact_state_initial(self):
        """ContactState should initialize to zero."""
        from tensordem.friction_polyhedron import ContactState
        state = ContactState()
        
        np.testing.assert_array_equal(state.delta_t, np.zeros(3))
        np.testing.assert_array_equal(state.rolling_moment, np.zeros(3))
        assert state.active is True

    def test_contact_state_reset(self):
        """ContactState reset should zero out state."""
        from tensordem.friction_polyhedron import ContactState
        state = ContactState()
        state.delta_t = np.array([1.0, 2.0, 3.0])
        state.rolling_moment = np.array([0.1, 0.2, 0.3])
        
        state.reset()
        
        np.testing.assert_array_equal(state.delta_t, np.zeros(3))
        np.testing.assert_array_equal(state.rolling_moment, np.zeros(3))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
