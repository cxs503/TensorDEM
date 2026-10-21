"""Unit tests for contact_tangential module."""
import math
import unittest

import torch

from tensordem.contact_tangential import (
    TangentialContactConfig,
    compute_tangential_force,
    damping_to_restitution,
    integrate_tangential_displacement,
    restitution_to_damping,
)


class RestitutionToDampingTests(unittest.TestCase):
    """Tests for the e -> gamma_n mapping and its inverse."""

    def test_e_1_zero_damping(self):
        """e = 1.0 => gamma_n = 0 (undamped)."""
        gamma = restitution_to_damping(1.0, 1000.0, 1.0)
        self.assertEqual(gamma, 0.0)

    def test_e_0_5(self):
        """Known value: e=0.5, k=1000, m=1 => gamma ~ 13.58."""
        gamma = restitution_to_damping(0.5, 1000.0, 1.0)
        # Manual: ln(0.5)=-0.6931, zeta=0.6931/sqrt(pi^2+0.6931^2)=0.2153
        # gamma = 2*0.2153*sqrt(1000) = 13.62 (approx)
        self.assertGreater(gamma, 10.0)
        self.assertLess(gamma, 20.0)

    def test_e_0_8(self):
        """Known value: e=0.8 => gamma ~ 4.48 for k=1000, m=1."""
        gamma = restitution_to_damping(0.8, 1000.0, 1.0)
        self.assertGreater(gamma, 2.0)
        self.assertLess(gamma, 10.0)

    def test_round_trip(self):
        """e -> gamma_n -> e should recover the original e."""
        for e_target in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            gamma = restitution_to_damping(e_target, 1000.0, 1.0)
            e_back = damping_to_restitution(gamma, 1000.0, 1.0)
            self.assertAlmostEqual(e_back, e_target, places=10,
                                   msg=f"Round trip failed for e={e_target}")

    def test_invalid_e_rejected(self):
        """e must be in (0, 1]."""
        with self.assertRaises(ValueError):
            restitution_to_damping(0.0, 1000.0, 1.0)
        with self.assertRaises(ValueError):
            restitution_to_damping(1.5, 1000.0, 1.0)
        with self.assertRaises(ValueError):
            restitution_to_damping(-0.5, 1000.0, 1.0)

    def test_critical_damping_gives_zero_e(self):
        """At critical damping (gamma = 2*sqrt(k*m)), e should be 0."""
        gamma_crit = 2.0 * math.sqrt(1000.0 * 1.0)
        e_back = damping_to_restitution(gamma_crit, 1000.0, 1.0)
        self.assertEqual(e_back, 0.0)

    def test_overdamped_gives_zero_e(self):
        """Overdamped case should return e = 0."""
        gamma_over = 3.0 * math.sqrt(1000.0 * 1.0)
        e_back = damping_to_restitution(gamma_over, 1000.0, 1.0)
        self.assertEqual(e_back, 0.0)

    def test_damping_ratio_monotonic(self):
        """Higher e => lower gamma_n."""
        g1 = restitution_to_damping(0.3, 1000.0, 1.0)
        g2 = restitution_to_damping(0.7, 1000.0, 1.0)
        self.assertGreater(g1, g2)

    def test_known_values_anchor(self):
        """Documented anchor values from the module docstring."""
        # e=0.5 => gamma ≈ 13.63
        gamma_05 = restitution_to_damping(0.5, 1000.0, 1.0)
        self.assertAlmostEqual(gamma_05, 13.626, delta=0.01)
        # e=0.8 => gamma ≈ 4.48
        gamma_08 = restitution_to_damping(0.8, 1000.0, 1.0)
        self.assertAlmostEqual(gamma_08, 4.481, delta=0.01)
        # e=1.0 => gamma = 0
        gamma_1 = restitution_to_damping(1.0, 1000.0, 1.0)
        self.assertEqual(gamma_1, 0.0)


class TangentialForceTests(unittest.TestCase):
    """Tests for the tangential friction spring."""

    def test_zero_friction_no_force(self):
        """mu=0 => zero tangential force regardless of displacement."""
        normal = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        tang_disp = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        F_t, _ = compute_tangential_force(
            normal, tang_disp, torch.tensor([100.0]), 1000.0, mu=0.0,
        )
        self.assertAlmostEqual(float(F_t.abs().sum()), 0.0, places=10)

    def test_coulomb_saturation(self):
        """Tangential force should be capped at mu * |F_n|."""
        normal = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        tang_disp = torch.tensor([[0.0, 10.0]], dtype=torch.float64)  # large disp
        F_n_mag = torch.tensor([100.0])
        mu = 0.3
        F_limit = mu * 100.0  # = 30.0
        F_t, _ = compute_tangential_force(
            normal, tang_disp, F_n_mag, k_t=1000.0, mu=mu,
        )
        # Raw force would be 1000*10=10000, but capped at 30
        self.assertAlmostEqual(float(F_t.norm(dim=1).item()), F_limit, delta=0.01)

    def test_below_coulomb_limit(self):
        """When below the friction limit, F_t = k_t * delta_t."""
        normal = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        tang_disp = torch.tensor([[0.0, 0.01]], dtype=torch.float64)
        F_n_mag = torch.tensor([100.0])
        mu = 0.5
        F_limit = mu * 100.0  # = 50
        k_t = 1000.0
        expected_Ft = k_t * 0.01  # = 10 < 50, so not capped
        F_t, _ = compute_tangential_force(
            normal, tang_disp, F_n_mag, k_t, mu,
        )
        self.assertAlmostEqual(float(F_t.norm(dim=1).item()), expected_Ft, delta=0.01)


class TangentialIntegrationTests(unittest.TestCase):
    """Tests for tangential displacement integration."""

    def test_pure_normal_velocity_no_tangential_accumulation(self):
        """Purely normal relative velocity => no tangential displacement."""
        tang_disp = torch.zeros((1, 2), dtype=torch.float64)
        rel_vel = torch.tensor([[1.0, 0.0]], dtype=torch.float64)  # along normal
        normal = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        new_disp = integrate_tangential_displacement(tang_disp, rel_vel, normal, 0.01)
        self.assertAlmostEqual(float(new_disp.abs().sum()), 0.0, places=15)

    def test_pure_tangential_velocity_accumulates(self):
        """Purely tangential relative velocity => full accumulation."""
        tang_disp = torch.zeros((1, 2), dtype=torch.float64)
        rel_vel = torch.tensor([[0.0, 1.0]], dtype=torch.float64)  # perpendicular to normal
        normal = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        dt = 0.01
        new_disp = integrate_tangential_displacement(tang_disp, rel_vel, normal, dt)
        self.assertAlmostEqual(new_disp[0, 0].item(), 0.0, places=15)
        self.assertAlmostEqual(new_disp[0, 1].item(), 1.0 * dt, places=15)

    def test_existing_displacement_accumulates(self):
        """Existing tangential displacement is incremented."""
        tang_disp = torch.tensor([[0.0, 0.05]], dtype=torch.float64)
        rel_vel = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
        normal = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        dt = 0.01
        new_disp = integrate_tangential_displacement(tang_disp, rel_vel, normal, dt)
        expected = 0.05 + 2.0 * dt
        self.assertAlmostEqual(new_disp[0, 1].item(), expected, places=15)


class TangentialContactConfigTests(unittest.TestCase):
    """Tests for the TangentialContactConfig dataclass."""

    def test_defaults(self):
        """Default config has zero friction."""
        cfg = TangentialContactConfig()
        self.assertEqual(cfg.mu, 0.0)
        self.assertIsNone(cfg.k_t)
        self.assertEqual(cfg.rolling_mu, 0.0)

    def test_custom_values(self):
        """Custom friction coefficient."""
        cfg = TangentialContactConfig(mu=0.4, k_t=500.0, rolling_mu=0.1)
        self.assertEqual(cfg.mu, 0.4)
        self.assertEqual(cfg.k_t, 500.0)
        self.assertEqual(cfg.rolling_mu, 0.1)


if __name__ == "__main__":
    unittest.main()
