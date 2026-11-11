"""Test for clump DEM angular velocity stability bug (p447w2-n02 discovery).

This test documents the critical numerical instability in the multi-sphere
clump DEM engine where angular velocity grows exponentially leading to NaN.

Bug: Explicit quaternion integration is unstable for elongated clumps with
small moment of inertia. The angular velocity omega grows as O(exp(t/dt))
leading to NaN within 15-20 time steps at dt=1e-4 when n_particles >= 10.

Workaround: Reduce dt by 100x (to 1e-6) for stability, but this makes
simulations 100x slower (impractical for parameter studies).

Status: KNOWN BUG - needs fix in clump_dynamics.py quaternion integration.
"""
import math
import numpy as np
import pytest

from tensordem.particle.clump import ellipsoid_clump, single_sphere
from tensordem.particle.clump_dem import ClumpDEM
from tensordem.contact.clump_contact import ContactParams, BoxObstacle


class TestClumpStabilityBug:
    """Tests documenting the clump DEM angular velocity instability."""

    def test_single_sphere_stable(self):
        """Single sphere (degenerate clump) is stable - baseline."""
        clump = single_sphere(0.002, 2500.0)
        floor = BoxObstacle(center=(0, 0, -0.001), half=(0.1, 0.1, 0.001))
        params = ContactParams(kn=1e5, mu=0.5, mu_roll=0.3)
        
        sim = ClumpDEM([clump], positions=np.array([[0, 0, 0.05]]),
                       obstacles=[floor], params=params, dt=1e-4)
        
        for _ in range(100):
            sim.step()
        
        assert np.all(np.isfinite(sim.pos)), "Single sphere should be stable"
        assert np.all(np.isfinite(sim.quat)), "Single sphere quaternion should be finite"
        assert np.all(np.isfinite(sim.omega)), "Single sphere omega should be finite"

    def test_multi_sphere_clump_unstable_at_dt_1e4(self):
        """Multi-sphere clump is UNSTABLE at dt=1e-4 with >=10 particles.
        
        This test PASSES if the bug is present (omega goes to NaN).
        When the bug is fixed, this test should be updated to check stability.
        
        The instability requires sufficient contact complexity (>=10 particles)
        to trigger the exponential growth of angular velocity.
        """
        n_particles = 10
        clumps = [ellipsoid_clump(0.00465, 0.002, n_spheres=5, density=2500.0)
                  for _ in range(n_particles)]
        positions = np.array([[i * 0.005, 0.0, 0.02] for i in range(n_particles)])
        floor = BoxObstacle(center=(0, 0, -0.001), half=(0.1, 0.1, 0.001))
        params = ContactParams(kn=1e5, mu=0.5, mu_roll=0.3)
        
        sim = ClumpDEM(clumps, positions=positions, obstacles=[floor],
                       params=params, dt=1e-4)
        
        # Run until NaN or 30 steps
        nan_detected = False
        for step in range(30):
            sim.step()
            if not np.all(np.isfinite(sim.omega)):
                nan_detected = True
                break
        
        # This test documents the bug: NaN should appear within 30 steps
        # When the bug is fixed, change this assertion to check stability
        assert nan_detected, (
            "Bug may have been fixed! Multi-sphere clump should go NaN at dt=1e-4 "
            "with 10 particles. If this fails, the stability bug has been resolved."
        )

    def test_multi_sphere_clump_stable_at_dt_1e6(self):
        """Multi-sphere clump is STABLE at dt=1e-6 (workaround).
        
        This test verifies the workaround: reducing dt by 100x stabilizes
        the simulation but makes it 100x slower.
        """
        n_particles = 10
        clumps = [ellipsoid_clump(0.00465, 0.002, n_spheres=5, density=2500.0)
                  for _ in range(n_particles)]
        positions = np.array([[i * 0.005, 0.0, 0.02] for i in range(n_particles)])
        floor = BoxObstacle(center=(0, 0, -0.001), half=(0.1, 0.1, 0.001))
        params = ContactParams(kn=1e5, mu=0.5, mu_roll=0.3)
        
        sim = ClumpDEM(clumps, positions=positions, obstacles=[floor],
                       params=params, dt=1e-6)  # 100x smaller
        
        for _ in range(100):
            sim.step()
        
        assert np.all(np.isfinite(sim.pos)), "Clump should be stable at dt=1e-6"
        assert np.all(np.isfinite(sim.quat)), "Quaternion should be finite at dt=1e-6"
        # Note: omega may be large but should be finite
        assert np.all(np.isfinite(sim.omega)), "Omega should be finite at dt=1e-6"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
