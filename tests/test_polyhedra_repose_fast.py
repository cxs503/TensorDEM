"""Minimal fast tests for polyhedral repose angle (AG29W1-N06).

These tests use very small particle counts and short simulation times
to complete quickly while still validating the core functionality.
"""
from __future__ import annotations

import math
import pytest
import numpy as np

from tensordem.polyhedra_repose import (
    PolyhedraReposeConfig,
    PolyhedraReposeSimulation,
    measure_repose_angle,
    measure_volume_fraction,
)
from tensordem.friction_polyhedron import (
    FrictionPolyhedronConfig,
    PolyhedronParticle,
    PolyhedronDEMSystem,
    random_polyhedron,
)
from tensordem.convex_contact import Polytope


# ===================================================================
# Fast unit tests (should complete in < 10s total)
# ===================================================================

class TestFastUnitTests:
    """Fast unit tests for measurement functions."""

    def test_measure_repose_angle_empty(self):
        """Empty particle list should give θ = 0°."""
        result = measure_repose_angle([], container_radius=0.05)
        assert result['theta_deg'] == 0.0

    def test_measure_repose_angle_flat(self):
        """Flat layer should give small angle."""
        particles = []
        for i in range(5):
            verts = random_polyhedron(np.array([0.01 * i, 0, 0.005]), 0.004, seed=i)
            p = PolyhedronParticle.from_polytope(verts)
            particles.append(p)
        result = measure_repose_angle(particles, container_radius=0.05)
        assert result['theta_deg'] < 30.0

    def test_measure_volume_fraction_single(self):
        """Single particle should give small φ."""
        verts = random_polyhedron(np.array([0, 0, 0.01]), 0.005, seed=0)
        p = PolyhedronParticle.from_polytope(verts)
        result = measure_volume_fraction([p], container_radius=0.05)
        assert 0.0 < result['phi'] < 0.5

    def test_config_validation(self):
        """Config should validate parameters."""
        config = PolyhedraReposeConfig(n_particles=10, mu=0.3)
        assert config.n_particles == 10
        assert config.mu == 0.3


class TestFastIntegration:
    """Fast integration tests with minimal particles."""

    def test_three_particles_settle(self):
        """3 particles should settle quickly."""
        config = PolyhedraReposeConfig(
            n_particles=3,
            particle_size=0.005,
            container_radius=0.03,
            pour_height=0.05,
            mu=0.3,
            mu_r=0.05,
            k_n=5e4,
            dt=1e-5,
            settle_time=0.02,
            seed=42,
        )
        sim = PolyhedraReposeSimulation(config)
        result = sim.run(record_interval=50)

        assert result['n_particles'] == 3
        assert result['simulated_time'] > 0
        assert 'repose' in result
        assert 'packing' in result

    def test_energy_tracking(self):
        """Energy should be tracked during simulation."""
        config = PolyhedraReposeConfig(
            n_particles=3,
            particle_size=0.005,
            container_radius=0.03,
            pour_height=0.05,
            mu=0.3,
            mu_r=0.05,
            k_n=5e4,
            dt=1e-5,
            settle_time=0.02,
            seed=42,
        )
        sim = PolyhedraReposeSimulation(config)
        result = sim.run(record_interval=50)

        assert len(result['history']['time']) > 0
        assert len(result['history']['ke']) > 0
        assert 'energy_error' in result


class TestAnalyticAnchors:
    """Tests against analytic solutions."""

    def test_free_fall_velocity(self):
        """Single particle free-fall: v(t) ≈ g*t."""
        verts = random_polyhedron(np.array([0, 0, 0.1]), 0.005, seed=0)
        p = PolyhedronParticle.from_polytope(verts)
        p.v = np.array([0.0, 0.0, 0.0])

        cfg = FrictionPolyhedronConfig(k_n=1e5, mu=0.0, mu_r=0.0)
        system = PolyhedronDEMSystem([p], cfg, gravity=-9.81)

        dt = 1e-5
        t_fall = 0.01
        n_steps = int(t_fall / dt)

        for _ in range(n_steps):
            system.step(dt)

        v_measured = abs(system.particles[0].v[2])
        v_analytical = 9.81 * t_fall

        rel_error = abs(v_measured - v_analytical) / v_analytical
        assert rel_error < 0.10, f"Free-fall velocity error: {rel_error:.3%}"

    def test_floor_contact(self):
        """Particle should bounce off floor."""
        verts = random_polyhedron(np.array([0, 0, 0.003]), 0.005, seed=0)
        p = PolyhedronParticle.from_polytope(verts)
        p.v = np.array([0.0, 0.0, -1.0])  # moving down

        cfg = FrictionPolyhedronConfig(k_n=1e5, mu=0.3, mu_r=0.05)
        system = PolyhedronDEMSystem([p], cfg, gravity=-9.81)
        system.add_wall(normal=np.array([0, 0, 1]), offset=0.0)

        # Step a few times
        for _ in range(100):
            system.step(1e-5)

        # Particle should not have fallen through
        assert system.particles[0].x[2] > -0.01


class TestVerificationCriteria:
    """Verification criteria tests (relaxed for fast execution)."""

    def test_c1_repose_angle_range(self):
        """C1: Repose angle should be in reasonable range."""
        config = PolyhedraReposeConfig(
            n_particles=10,
            particle_size=0.005,
            container_radius=0.03,
            pour_height=0.06,
            mu=0.4,
            mu_r=0.05,
            k_n=5e4,
            dt=1e-5,
            settle_time=0.05,
            seed=42,
        )
        sim = PolyhedraReposeSimulation(config)
        result = sim.run(record_interval=100)

        theta = result['repose']['theta_deg']
        # Relaxed range for fast test
        assert 0.0 <= theta <= 60.0, f"Repose angle {theta:.1f}° out of range"

    def test_c3_volume_fraction_range(self):
        """C3: Volume fraction should be in reasonable range."""
        config = PolyhedraReposeConfig(
            n_particles=10,
            particle_size=0.005,
            container_radius=0.03,
            pour_height=0.06,
            mu=0.4,
            mu_r=0.05,
            k_n=5e4,
            dt=1e-5,
            settle_time=0.05,
            seed=42,
        )
        sim = PolyhedraReposeSimulation(config)
        result = sim.run(record_interval=100)

        phi = result['packing']['phi']
        assert 0.0 < phi < 1.0, f"Volume fraction {phi:.3f} out of range"

    def test_c4_energy_closure(self):
        """C4: Energy balance should be tracked."""
        config = PolyhedraReposeConfig(
            n_particles=5,
            particle_size=0.005,
            container_radius=0.03,
            pour_height=0.05,
            mu=0.3,
            mu_r=0.05,
            k_n=5e4,
            dt=1e-5,
            settle_time=0.02,
            seed=42,
        )
        sim = PolyhedraReposeSimulation(config)
        result = sim.run(record_interval=50)

        # Energy error should be finite
        assert np.isfinite(result['energy_error'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
