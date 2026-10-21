"""Unit tests for buoyancy_wet module."""
import math
import unittest

import torch

from tensordem.buoyancy_wet import (
    WetCouplingConfig,
    compute_buoyancy_force,
    compute_wet_drag_force,
    submerged_fraction_tensor,
)


class SubmergedFractionTests(unittest.TestCase):
    """Tests for the submerged-fraction calculation."""

    def test_fully_above_water(self):
        """Centre well above water -> f_sub = 0."""
        f = submerged_fraction_tensor(torch.tensor([1.0]), 0.5, 0.0)
        self.assertAlmostEqual(f[0].item(), 0.0, places=10)

    def test_fully_submerged(self):
        """Centre well below water -> f_sub = 1."""
        f = submerged_fraction_tensor(torch.tensor([-1.0]), 0.5, 0.0)
        self.assertAlmostEqual(f[0].item(), 1.0, places=10)

    def test_half_submerged(self):
        """Centre exactly at water level -> f_sub = 0.5."""
        f = submerged_fraction_tensor(torch.tensor([0.0]), 0.5, 0.0)
        self.assertAlmostEqual(f[0].item(), 0.5, places=6)

    def test_boundary_values(self):
        """Centre at +R (just touching) and -R (just fully in)."""
        R = 0.5
        f_above = submerged_fraction_tensor(torch.tensor([R]), R, 0.0)
        f_below = submerged_fraction_tensor(torch.tensor([-R]), R, 0.0)
        self.assertAlmostEqual(f_above[0].item(), 0.0, places=10)
        self.assertAlmostEqual(f_below[0].item(), 1.0, places=10)

    def test_batch_computation(self):
        """Batch of particles with different heights."""
        R = 1.0
        ys = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        f = submerged_fraction_tensor(ys, R, 0.0)
        self.assertAlmostEqual(f[0].item(), 1.0, places=10)  # fully in
        self.assertAlmostEqual(f[1].item(), 1.0, places=10)  # centre at -R
        self.assertAlmostEqual(f[2].item(), 0.5, places=6)   # half
        self.assertAlmostEqual(f[3].item(), 0.0, places=10)  # centre at +R
        self.assertAlmostEqual(f[4].item(), 0.0, places=10)  # fully out


class BuoyancyForceTests(unittest.TestCase):
    """Tests for the buoyancy force computation."""

    def test_no_buoyancy_when_disabled(self):
        """Disabled config -> zero force."""
        positions = torch.tensor([[0.0, 0.0]])
        cfg = WetCouplingConfig(enabled=False)
        F = compute_buoyancy_force(positions, 0.5, 1.0, 917.0, cfg)
        self.assertEqual(float(F.sum()), 0.0)

    def test_neutral_buoyancy_at_equilibrium(self):
        """At equilibrium (rho_i/rho_w submerged), buoyancy = weight.

        For a circular cross-section the equilibrium centre height h_eq
        satisfies f_sub(h_eq) = rho_i/rho_w.  We find h_eq numerically.
        """
        rho_i = 917.0
        rho_w = 1025.0
        target_f_sub = rho_i / rho_w
        thickness = 1.0
        R = 0.5
        mass = rho_i * math.pi * R ** 2 * thickness

        # Find h_eq by bisection: f_sub decreases as h increases
        lo, hi = -R, R
        for _ in range(100):
            mid = (lo + hi) / 2.0
            f = submerged_fraction_tensor(torch.tensor([mid]), R, 0.0).item()
            if f > target_f_sub:
                lo = mid  # too submerged, raise h
            else:
                hi = mid
        h_eq = (lo + hi) / 2.0

        positions = torch.tensor([[0.0, h_eq]], dtype=torch.float64)

        cfg = WetCouplingConfig(
            enabled=True, water_density=rho_w, gravity=9.81,
            water_level=0.0,
        )
        F = compute_buoyancy_force(positions, R, thickness, rho_i, cfg)

        # Buoyancy should approximately equal weight
        weight = mass * 9.81
        self.assertAlmostEqual(F[0, 1].item(), weight, delta=weight * 0.01)
        self.assertAlmostEqual(F[0, 0].item(), 0.0, places=15)

    def test_buoyancy_direction(self):
        """Buoyancy always points up (+y)."""
        positions = torch.tensor([[0.0, -0.5]], dtype=torch.float64)
        cfg = WetCouplingConfig(
            enabled=True, water_density=1025.0, gravity=9.81,
        )
        F = compute_buoyancy_force(positions, 0.5, 1.0, 917.0, cfg)
        self.assertGreater(F[0, 1].item(), 0.0)
        self.assertAlmostEqual(F[0, 0].item(), 0.0, places=15)


class WetDragForceTests(unittest.TestCase):
    """Tests for the wet drag force computation."""

    def test_no_wet_drag_when_disabled(self):
        """Disabled config -> zero drag."""
        positions = torch.tensor([[0.0, 0.0]])
        velocities = torch.tensor([[1.0, 2.0]])
        cfg = WetCouplingConfig(enabled=False, wet_drag_coeff=10.0)
        F = compute_wet_drag_force(positions, velocities, 0.5, cfg)
        self.assertEqual(float(F.sum()), 0.0)

    def test_wet_drag_opposes_velocity(self):
        """Wet drag force opposes the velocity direction."""
        positions = torch.tensor([[0.0, -0.5]], dtype=torch.float64)
        velocities = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        cfg = WetCouplingConfig(
            enabled=True, water_density=1025.0,
            wet_drag_coeff=50.0,
        )
        F = compute_wet_drag_force(positions, velocities, 0.5, cfg)
        # Drag opposes velocity
        self.assertLess(F[0, 0].item(), 0.0)
        self.assertLess(F[0, 1].item(), 0.0)

    def test_wet_drag_zero_for_dry_particle(self):
        """Dry particle (above water) feels no wet drag."""
        positions = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        velocities = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        cfg = WetCouplingConfig(
            enabled=True, water_density=1025.0,
            wet_drag_coeff=50.0,
        )
        F = compute_wet_drag_force(positions, velocities, 0.5, cfg)
        self.assertAlmostEqual(F[0, 0].item(), 0.0, places=10)
        self.assertAlmostEqual(F[0, 1].item(), 0.0, places=10)

    def test_wet_drag_zero_coeff(self):
        """Zero drag coefficient -> zero drag."""
        positions = torch.tensor([[0.0, -0.5]], dtype=torch.float64)
        velocities = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        cfg = WetCouplingConfig(
            enabled=True, water_density=1025.0, wet_drag_coeff=0.0,
        )
        F = compute_wet_drag_force(positions, velocities, 0.5, cfg)
        self.assertEqual(float(F.abs().sum()), 0.0)


class CombinedWetForceTests(unittest.TestCase):
    """Integration tests for combined buoyancy + wet drag."""

    def test_fully_wet_particle_moving(self):
        """Fully submerged moving particle gets both buoyancy and drag."""
        positions = torch.tensor([[0.0, -1.0]], dtype=torch.float64)
        velocities = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        rho_i = 917.0
        rho_w = 1025.0
        R = 0.5
        thickness = 1.0
        mass = rho_i * math.pi * R ** 2 * thickness

        cfg = WetCouplingConfig(
            enabled=True, water_density=rho_w, gravity=9.81,
            wet_drag_coeff=100.0,
        )
        from tensordem.buoyancy_wet import compute_wet_forces
        F_buoy, F_drag = compute_wet_forces(
            positions, velocities, R, thickness, rho_i, cfg,
        )
        # Buoyancy upward
        self.assertGreater(F_buoy[0, 1].item(), 0.0)
        # Drag opposes motion (leftward for rightward velocity)
        self.assertLess(F_drag[0, 0].item(), 0.0)
        # No drag in y since vy=0
        self.assertAlmostEqual(F_drag[0, 1].item(), 0.0, places=10)

    def test_fully_dry_extreme(self):
        """Fully dry particle: zero buoyancy, zero wet drag."""
        positions = torch.tensor([[0.0, 10.0]], dtype=torch.float64)
        velocities = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        cfg = WetCouplingConfig(
            enabled=True, water_density=1025.0, gravity=9.81,
            wet_drag_coeff=100.0,
        )
        from tensordem.buoyancy_wet import compute_wet_forces
        F_buoy, F_drag = compute_wet_forces(
            positions, velocities, 0.5, 1.0, 917.0, cfg,
        )
        self.assertAlmostEqual(float(F_buoy.abs().sum()), 0.0, places=10)
        self.assertAlmostEqual(float(F_drag.abs().sum()), 0.0, places=10)

    def test_fully_wet_extreme(self):
        """Fully submerged: buoyancy = rho_w * g * V, drag proportional to v."""
        positions = torch.tensor([[0.0, -10.0]], dtype=torch.float64)
        velocities = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        R = 0.5
        thickness = 1.0
        rho_w = 1025.0
        V = math.pi * R ** 2 * thickness

        cfg = WetCouplingConfig(
            enabled=True, water_density=rho_w, gravity=9.81,
            wet_drag_coeff=100.0,
        )
        from tensordem.buoyancy_wet import compute_wet_forces
        F_buoy, F_drag = compute_wet_forces(
            positions, velocities, R, thickness, 917.0, cfg,
        )
        expected_buoyancy = rho_w * 9.81 * V
        self.assertAlmostEqual(F_buoy[0, 1].item(), expected_buoyancy, delta=0.1)
        # Drag = -C_wet * v (fully submerged, f_sub=1)
        self.assertAlmostEqual(F_drag[0, 0].item(), -100.0 * 1.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
