"""Verification of core physics in :mod:`tensordem.ridge_dem`.

Protocol id: ``tensordem-ridge-dem-core-physics-v1``

Scope
-----
This file grades the *geometric* and *buoyancy* foundations of the ridge DEM
module, which are prerequisites for any grounding-force measurement:

1. **Keel geometry**: cross-section area, half-width profile, flank angle.
2. **Buoyant body force**: net vertical force per particle (ice vs water).
3. **Bond failure strain**: implied strain from strength/stiffness ratio.
4. **Recommended dt**: elastic and viscous limits.
5. **Plane displacement**: translation along normal.
6. **RidgeMaterial validation**: ductility guard, buoyancy guard.

All checks are closed-form; no DEM simulation is run.
"""
from __future__ import annotations

import math
import unittest

from tensordem.ridge_dem import RidgeKeel, RidgeMaterial, Plane


class RidgeKeelGeometryTests(unittest.TestCase):
    """Closed-form geometry of the triangular keel envelope."""

    def test_cross_section_area_is_triangle(self):
        """A = 0.5 * width * depth for a triangular cross-section."""
        keel = RidgeKeel(keel_depth=3.0, keel_width=10.0)
        self.assertAlmostEqual(keel.cross_section_area, 0.5 * 10.0 * 3.0, places=12)

    def test_half_width_is_linear(self):
        """Half-width grows linearly from tip to mouldline."""
        keel = RidgeKeel(keel_depth=4.0, keel_width=8.0)
        self.assertAlmostEqual(keel.half_width(0.0), 0.0, places=12)
        self.assertAlmostEqual(keel.half_width(4.0), 4.0, places=12)
        self.assertAlmostEqual(keel.half_width(2.0), 2.0, places=12)

    def test_half_width_clamps(self):
        """Half-width clamps to [0, keel_width/2] outside the keel."""
        keel = RidgeKeel(keel_depth=3.0, keel_width=6.0)
        self.assertAlmostEqual(keel.half_width(-1.0), 0.0, places=12)
        self.assertAlmostEqual(keel.half_width(5.0), 3.0, places=12)

    def test_flank_angle(self):
        """Flank angle = atan(0.5 * width / depth) in degrees."""
        keel = RidgeKeel(keel_depth=3.0, keel_width=6.0)
        expected = math.degrees(math.atan(0.5 * 6.0 / 3.0))
        self.assertAlmostEqual(keel.angle_deg, expected, places=12)

    def test_sail_height_default(self):
        """Default sail height is 20 % of keel depth."""
        keel = RidgeKeel(keel_depth=5.0, keel_width=10.0)
        self.assertAlmostEqual(keel.sail_height, 1.0, places=12)


class RidgeMaterialTests(unittest.TestCase):
    """Material properties and derived quantities."""

    def test_particle_mass_unit_thickness(self):
        """m = rho_i * pi * r^2 for a disk of unit thickness."""
        mat = RidgeMaterial(particle_radius=0.5, ice_density=917.0)
        expected = 917.0 * math.pi * 0.5 ** 2
        self.assertAlmostEqual(mat.particle_mass, expected, places=10)

    def test_buoyant_unit_force(self):
        """F_b = (rho_i - rho_w) * pi * r^2 * g (positive = downward in module frame)."""
        mat = RidgeMaterial(
            particle_radius=0.5, ice_density=917.0, water_density=1025.0, gravity=9.81
        )
        volume = math.pi * 0.5 ** 2
        expected = (917.0 - 1025.0) * volume * 9.81
        self.assertAlmostEqual(mat.buoyant_unit_force, expected, places=10)
        # Ice is buoyant, so the force should be negative (upward in y-up frame).
        self.assertLess(mat.buoyant_unit_force, 0.0)

    def test_bond_failure_strain(self):
        """epsilon = sigma_t / (k_b / (2r)) = 2 r sigma_t / (2 r k_b) = sigma_t / k_b."""
        mat = RidgeMaterial(
            particle_radius=0.5, bond_stiffness=5.0e6, bond_tensile_strength=50.0e3
        )
        # The code computes: sigma_t * 2r / (k_b * 2r) = sigma_t / k_b
        expected = 50.0e3 / 5.0e6
        self.assertAlmostEqual(mat.bond_failure_strain, expected, places=12)

    def test_recommended_dt_elastic_limit(self):
        """dt_elastic = 0.2 * sqrt(m / (neighbours * (k_c + 2 k_b)))."""
        # The elastic branch binds only while the viscous bound is looser, i.e.
        # while neighbours * c_d + drag < sqrt(m * k); otherwise the assertion
        # below would silently compare the elastic formula against the viscous
        # step.  The damping here sits an order of magnitude below that crossover.
        mat = RidgeMaterial(
            particle_radius=0.5, ice_density=917.0,
            contact_stiffness=1.0e7, bond_stiffness=5.0e6,
            contact_damping=1.0e4, drag=2.0e2,
        )
        neighbours = 13.0
        stiffness = neighbours * (mat.contact_stiffness + 2.0 * mat.bond_stiffness)
        damping = neighbours * mat.contact_damping + mat.drag
        self.assertLess(damping, math.sqrt(mat.particle_mass * stiffness))
        expected_elastic = 0.2 * math.sqrt(mat.particle_mass / stiffness)
        self.assertAlmostEqual(mat.recommended_dt, expected_elastic, places=10)

    def test_recommended_dt_viscous_limit(self):
        """dt_viscous = 0.2 * m / (neighbours * c_d + drag)."""
        mat = RidgeMaterial(
            particle_radius=0.1, ice_density=917.0,
            contact_stiffness=1.0e12, bond_stiffness=1.0e12,
            contact_damping=1.0e8, drag=1.0e6,
        )
        neighbours = 13.0
        damping = neighbours * mat.contact_damping + mat.drag
        expected_viscous = 0.2 * mat.particle_mass / damping
        self.assertAlmostEqual(mat.recommended_dt, expected_viscous, places=8)

    def test_explicit_dt_override(self):
        """When dt is set, recommended_dt returns it directly."""
        mat = RidgeMaterial(particle_radius=0.5, dt=1e-4)
        self.assertAlmostEqual(mat.recommended_dt, 1e-4, places=12)

    def test_bond_area(self):
        """Bond area for unit thickness = 2 * radius."""
        mat = RidgeMaterial(particle_radius=0.75)
        self.assertAlmostEqual(mat.bond_area, 1.5, places=12)


class RidgeMaterialValidationTests(unittest.TestCase):
    """Input validation guards."""

    def test_ice_must_be_buoyant(self):
        """ice_density > water_density is rejected."""
        with self.assertRaises(ValueError):
            RidgeMaterial(ice_density=1100.0, water_density=1025.0)

    def test_bond_ductility_guard(self):
        """sigma_t / k_b > 0.05 is rejected (too ductile for explicit stability)."""
        with self.assertRaises(ValueError):
            RidgeMaterial(bond_stiffness=1.0e6, bond_tensile_strength=100.0e3)

    def test_porosity_upper_bound(self):
        """Porosity >= 0.6 is rejected."""
        with self.assertRaises(ValueError):
            RidgeKeel(keel_depth=3.0, keel_width=10.0, porosity=0.6)

    def test_negative_keel_depth_rejected(self):
        with self.assertRaises(ValueError):
            RidgeKeel(keel_depth=-1.0, keel_width=10.0)


class PlaneTests(unittest.TestCase):
    """Rigid half-space wall geometry."""

    def test_displaced_along_normal(self):
        """Plane displaced by d along normal: point' = point + d * normal."""
        plane = Plane(normal=(0.0, 1.0), point=(0.0, 0.0))
        displaced = plane.displaced(5.0)
        self.assertAlmostEqual(displaced.point[0], 0.0, places=12)
        self.assertAlmostEqual(displaced.point[1], 5.0, places=12)

    def test_displaced_preserves_other_fields(self):
        """Displacement copies all other fields unchanged."""
        plane = Plane(normal=(1.0, 0.0), point=(0.0, 0.0), velocity=0.1, friction=0.4)
        displaced = plane.displaced(2.0)
        self.assertAlmostEqual(displaced.velocity, 0.1, places=12)
        self.assertAlmostEqual(displaced.friction, 0.4, places=12)
        self.assertAlmostEqual(displaced.normal[0], 1.0, places=12)

    def test_non_unit_normal_rejected(self):
        with self.assertRaises(ValueError):
            Plane(normal=(1.0, 1.0), point=(0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
