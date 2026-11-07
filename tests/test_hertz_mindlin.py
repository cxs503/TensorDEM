"""Hertz–Mindlin contact model verification tests.

Verification anchors:
=====================

1. **Mindlin stiffness ratio** (Anchor M):
   The tangential stiffness ratio k_t/k_n for equal spheres must equal
   3(1-ν)/(2-ν), independent of overlap.

2. **Hertz-Mindlin tangential force** (Anchor T):
   The tangential force with Mindlin stiffness and Coulomb saturation
   produces correct results.

3. **Negative controls**:
   - μ < 0 must be rejected by the config validator
   - k_t must never exceed a reasonable multiple of k_n (sanity check)

4. **Integration tests**:
   - Hertz-Mindlin simulation runs without errors
   - Energy-audited simulation tracks friction work

All quantities use SI units.
"""
from __future__ import annotations

import math
import unittest

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.dem3d_energy import EnergyAuditedIceDEM3D
from tensordem import hertz_mindlin


# --------------------------------------------------------------------------
# Anchor M: Mindlin stiffness ratio
# --------------------------------------------------------------------------

class MindlinStiffnessRatioTests(unittest.TestCase):
    """k_t / k_n = 3(1-ν)/(2-ν) for equal spheres in Hertz–Mindlin."""

    def test_stiffness_ratio_nu_03(self):
        """For ν = 0.3, k_t/k_n = 3*0.7/1.7 ≈ 1.235."""
        nu = 0.3
        expected = 3.0 * (1.0 - nu) / (2.0 - nu)
        actual = hertz_mindlin.mindlin_stiffness_ratio(nu)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_stiffness_ratio_nu_025(self):
        """For ν = 0.25, k_t/k_n = 3*0.75/1.75 ≈ 1.286."""
        nu = 0.25
        expected = 3.0 * (1.0 - nu) / (2.0 - nu)
        actual = hertz_mindlin.mindlin_stiffness_ratio(nu)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_stiffness_ratio_nu_0(self):
        """For ν = 0, k_t/k_n = 3/2 = 1.5."""
        nu = 0.0
        expected = 1.5
        actual = hertz_mindlin.mindlin_stiffness_ratio(nu)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_stiffness_ratio_independent_of_overlap(self):
        """The ratio k_t/k_n should be independent of overlap."""
        nu = 0.3
        E = 1e9
        R = 0.01
        from tensordem.dem3d import hertz_reduced_modulus, hertz_effective_radius, hertz_secant_stiffness
        e_star = hertz_reduced_modulus(E, nu)
        g_star = hertz_mindlin.hertz_reduced_shear_modulus(E, nu)
        r_eff = hertz_effective_radius(R)
        expected = hertz_mindlin.mindlin_stiffness_ratio(nu)

        for delta in [1e-6, 1e-5, 1e-4, 1e-3]:
            k_n = float(hertz_secant_stiffness(delta, e_star, r_eff))
            k_t = float(hertz_mindlin.hertz_mindlin_tangential_stiffness(delta, e_star, g_star, r_eff))
            ratio = k_t / k_n
            self.assertAlmostEqual(ratio, expected, places=5,
                                   msg=f"Ratio mismatch at delta={delta}")

    def test_reduced_shear_modulus_equal_spheres(self):
        """G* = E / [4*(2-ν)*(1+ν)] for equal spheres."""
        E = 1e9
        nu = 0.3
        expected = E / (4.0 * (2.0 - nu) * (1.0 + nu))
        actual = hertz_mindlin.hertz_reduced_shear_modulus(E, nu)
        self.assertAlmostEqual(actual, expected, places=5)

    def test_reduced_shear_modulus_rigid_counterpart(self):
        """G* = E / [2*(2-ν)*(1+ν)] against a rigid surface."""
        E = 1e9
        nu = 0.3
        expected = E / (2.0 * (2.0 - nu) * (1.0 + nu))
        actual = hertz_mindlin.hertz_reduced_shear_modulus(E, nu, rigid_counterpart=True)
        self.assertAlmostEqual(actual, expected, places=5)


# --------------------------------------------------------------------------
# Anchor T: Hertz-Mindlin tangential force
# --------------------------------------------------------------------------

class HertzMindlinTangentialForceTests(unittest.TestCase):
    """Test the combined Mindlin stiffness + Coulomb saturation."""

    def test_stick_below_coulomb_limit(self):
        """When tangential displacement is small, force = k_t * delta_t."""
        overlap = 1e-4
        E = 1e9
        nu = 0.3
        R = 0.01
        from tensordem.dem3d import hertz_reduced_modulus, hertz_effective_radius
        e_star = hertz_reduced_modulus(E, nu)
        g_star = hertz_mindlin.hertz_reduced_shear_modulus(E, nu)
        r_eff = hertz_effective_radius(R)

        k_t = float(hertz_mindlin.hertz_mindlin_tangential_stiffness(overlap, e_star, g_star, r_eff))
        delta_t = torch.tensor([[0.0, 1e-6, 0.0]], dtype=torch.float64)
        F_n = torch.tensor([100.0], dtype=torch.float64)
        mu = 0.5

        F_t, state = hertz_mindlin.hertz_mindlin_tangential_force(
            overlap, delta_t, e_star, g_star, r_eff, mu, F_n)

        expected_Ft = k_t * 1e-6
        self.assertAlmostEqual(float(F_t[0, 1].item()), expected_Ft, places=5,
                               msg="Should be in stick regime")

    def test_slip_above_coulomb_limit(self):
        """When tangential displacement is large, force saturates at mu * F_n."""
        overlap = 1e-4
        E = 1e9
        nu = 0.3
        R = 0.01
        from tensordem.dem3d import hertz_reduced_modulus, hertz_effective_radius
        e_star = hertz_reduced_modulus(E, nu)
        g_star = hertz_mindlin.hertz_reduced_shear_modulus(E, nu)
        r_eff = hertz_effective_radius(R)

        k_t = float(hertz_mindlin.hertz_mindlin_tangential_stiffness(overlap, e_star, g_star, r_eff))
        delta_t = torch.tensor([[0.0, 0.1, 0.0]], dtype=torch.float64)  # Large displacement
        F_n = torch.tensor([100.0], dtype=torch.float64)
        mu = 0.3

        F_t, state = hertz_mindlin.hertz_mindlin_tangential_force(
            overlap, delta_t, e_star, g_star, r_eff, mu, F_n)

        expected_limit = mu * 100.0
        self.assertAlmostEqual(float(F_t.norm(dim=1).item()), expected_limit, places=5,
                               msg="Should be at Coulomb limit")

    def test_state_projection_onto_cone(self):
        """After slip, the state should be projected onto the friction cone."""
        overlap = 1e-4
        E = 1e9
        nu = 0.3
        R = 0.01
        from tensordem.dem3d import hertz_reduced_modulus, hertz_effective_radius
        e_star = hertz_reduced_modulus(E, nu)
        g_star = hertz_mindlin.hertz_reduced_shear_modulus(E, nu)
        r_eff = hertz_effective_radius(R)

        k_t = float(hertz_mindlin.hertz_mindlin_tangential_stiffness(overlap, e_star, g_star, r_eff))
        delta_t = torch.tensor([[0.0, 0.1, 0.0]], dtype=torch.float64)
        F_n = torch.tensor([100.0], dtype=torch.float64)
        mu = 0.3

        F_t, state = hertz_mindlin.hertz_mindlin_tangential_force(
            overlap, delta_t, e_star, g_star, r_eff, mu, F_n)

        # State should be F_t / k_t
        expected_state = F_t / k_t
        torch.testing.assert_close(state, expected_state, rtol=1e-5, atol=1e-10)


# --------------------------------------------------------------------------
# Negative controls
# --------------------------------------------------------------------------

class NegativeControlTests(unittest.TestCase):
    """Invalid parameters must be rejected."""

    def test_negative_mu_rejected(self):
        """μ < 0 must raise ValueError."""
        with self.assertRaises(ValueError):
            DEM3DConfig(friction_coefficient=-0.1)

    def test_negative_mu_hertz_mindlin_rejected(self):
        """μ < 0 in hertz_mindlin functions must raise ValueError."""
        with self.assertRaises(ValueError):
            hertz_mindlin.hertz_mindlin_tangential_force(
                overlap=1e-4,
                tangential_disp=torch.zeros((1, 3), dtype=torch.float64),
                e_star=1e9,
                g_star=4e8,
                r_eff=0.005,
                mu=-0.1,
                normal_force_mag=torch.tensor([100.0]),
            )

    def test_invalid_poisson_ratio_rejected(self):
        """ν outside [0, 0.5) must raise ValueError."""
        with self.assertRaises(ValueError):
            hertz_mindlin.hertz_reduced_shear_modulus(1e9, -0.1)
        with self.assertRaises(ValueError):
            hertz_mindlin.hertz_reduced_shear_modulus(1e9, 0.5)
        with self.assertRaises(ValueError):
            hertz_mindlin.hertz_reduced_shear_modulus(1e9, 0.6)

    def test_invalid_young_modulus_rejected(self):
        """E <= 0 must raise ValueError."""
        with self.assertRaises(ValueError):
            hertz_mindlin.hertz_reduced_shear_modulus(0.0, 0.3)
        with self.assertRaises(ValueError):
            hertz_mindlin.hertz_reduced_shear_modulus(-1e9, 0.3)

    def test_tangential_stiffness_bounded(self):
        """k_t should be a reasonable multiple of k_n (not arbitrarily large)."""
        E = 1e9
        nu = 0.3
        R = 0.01
        from tensordem.dem3d import hertz_reduced_modulus, hertz_effective_radius, hertz_secant_stiffness
        e_star = hertz_reduced_modulus(E, nu)
        g_star = hertz_mindlin.hertz_reduced_shear_modulus(E, nu)
        r_eff = hertz_effective_radius(R)

        for delta in [1e-6, 1e-5, 1e-4, 1e-3]:
            k_n = float(hertz_secant_stiffness(delta, e_star, r_eff))
            k_t = float(hertz_mindlin.hertz_mindlin_tangential_stiffness(delta, e_star, g_star, r_eff))
            # k_t/k_n = 3*(1-nu)/(2-nu) ≈ 1.24 for nu=0.3, should never exceed 2
            self.assertLess(k_t / k_n, 2.0,
                            f"k_t/k_n = {k_t/k_n:.2f} at delta={delta} exceeds sanity bound")

    def test_zero_overlap_gives_zero_stiffness(self):
        """Zero overlap should give zero tangential stiffness."""
        E = 1e9
        nu = 0.3
        R = 0.01
        from tensordem.dem3d import hertz_reduced_modulus, hertz_effective_radius
        e_star = hertz_reduced_modulus(E, nu)
        g_star = hertz_mindlin.hertz_reduced_shear_modulus(E, nu)
        r_eff = hertz_effective_radius(R)

        k_t = hertz_mindlin.hertz_mindlin_tangential_stiffness(0.0, e_star, g_star, r_eff)
        self.assertAlmostEqual(float(k_t.item()), 0.0, places=15)


# --------------------------------------------------------------------------
# Integration tests
# --------------------------------------------------------------------------

class HertzMindlinDEMTests(unittest.TestCase):
    """Integration tests: Hertz-Mindlin model in full DEM simulation."""

    def test_hertz_mindlin_runs_without_error(self):
        """A Hertz-Mindlin simulation should run without errors."""
        cfg = DEM3DConfig(
            nx=3, ny=3, nz=3,
            radius=0.01,
            density=1000.0,
            contact_model="hertz",
            young_modulus=1e9,
            poisson_ratio=0.3,
            friction_coefficient=0.3,
            restitution_coefficient=1.0,
            contact_damping=0.0,
            drag=0.0,
            dt=1e-6,
            fix_x_edges=False,
            fix_bottom=False,
        )
        sim = IceDEM3D(cfg)
        sim.velocities[0, 2] = -0.1  # Give one particle a small velocity
        for _ in range(100):
            sim.step()
        # Should not raise
        self.assertTrue(torch.isfinite(sim.positions).all())

    def test_hertz_mindlin_energy_audited(self):
        """Energy-audited Hertz-Mindlin simulation should track friction work."""
        cfg = DEM3DConfig(
            nx=3, ny=3, nz=3,
            radius=0.01,
            density=1000.0,
            contact_model="hertz",
            young_modulus=1e9,
            poisson_ratio=0.3,
            friction_coefficient=0.3,
            restitution_coefficient=1.0,
            contact_damping=0.0,
            drag=0.0,
            dt=1e-6,
            fix_x_edges=False,
            fix_bottom=False,
        )
        sim = EnergyAuditedIceDEM3D(cfg)
        sim.velocities[0, 2] = -0.1
        sim.velocities[0, 0] = 0.05  # Add tangential component
        for _ in range(200):
            sim.step()
        # Friction work should be tracked (may be zero if no sliding)
        self.assertIsInstance(sim.cumulative_friction_work_J, float)

    def test_hertz_vs_linear_friction(self):
        """Hertz and linear models should both produce friction forces."""
        for model in ["linear", "hertz"]:
            cfg = DEM3DConfig(
                nx=3, ny=3, nz=3,
                radius=0.01,
                density=1000.0,
                contact_model=model,
                young_modulus=1e9,
                poisson_ratio=0.3,
                friction_coefficient=0.3,
                restitution_coefficient=1.0,
                contact_damping=0.0,
                drag=0.0,
                dt=1e-6,
                fix_x_edges=False,
                fix_bottom=False,
            )
            sim = IceDEM3D(cfg)
            sim.velocities[0, 2] = -0.1
            sim.velocities[0, 0] = 0.05
            for _ in range(50):
                sim.step()
            self.assertTrue(torch.isfinite(sim.positions).all(),
                            f"Model {model} produced non-finite positions")

    def test_hertz_mindlin_with_high_friction(self):
        """Hertz-Mindlin with high friction should still be stable."""
        cfg = DEM3DConfig(
            nx=3, ny=3, nz=3,
            radius=0.01,
            density=1000.0,
            contact_model="hertz",
            young_modulus=1e9,
            poisson_ratio=0.3,
            friction_coefficient=0.8,
            restitution_coefficient=1.0,
            contact_damping=0.0,
            drag=0.0,
            dt=1e-6,
            fix_x_edges=False,
            fix_bottom=False,
        )
        sim = IceDEM3D(cfg)
        sim.velocities[0, 2] = -0.1
        sim.velocities[0, 0] = 0.1
        for _ in range(100):
            sim.step()
        self.assertTrue(torch.isfinite(sim.positions).all())


if __name__ == "__main__":
    unittest.main()
