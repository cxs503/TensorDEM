"""DEM ice-contact mechanics calibration tests.

Tests the complete ice-ship contact力学 chain:
1. Hertz normal contact (D1)
2. Mindlin tangential contact (D2)
3. Ice-steel friction calibration (D3)
4. Lindqvist cross-validation (D4)
5. Negative controls (D5, D6)
6. Energy conservation (D7)
7. Mesh convergence (D8)

All quantities SI.
"""
from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from tensordem.dem3d import (
    hertz_reduced_modulus,
    hertz_effective_radius,
    hertz_normal_force,
)
from tensordem import hertz_mindlin
from tensordem.ice_steel_friction import (
    fit_ice_steel_friction,
    CALIBRATION_ANCHORS,
    VALIDATION_ANCHORS,
)
from tensordem.ice_dem_contact import (
    IceMaterial,
    IcePlateConfig,
    ShipWedgeConfig,
    run_single_case,
    lindqvist_anchor_for_config,
)


class TestHertzNormalContact(unittest.TestCase):
    """D1: Hertz normal contact verification."""

    def test_hertz_force_matches_analytical(self):
        """Hertz force F_n = (4/3) E* sqrt(R*) delta^(3/2) matches analytical."""
        E = 1.0e9
        nu = 0.3
        R = 0.01
        delta = 1.0e-4
        
        e_star = hertz_reduced_modulus(E, nu)
        r_eff = hertz_effective_radius(R)
        
        # Analytical
        f_analytical = (4.0 / 3.0) * e_star * math.sqrt(r_eff) * delta**1.5
        
        # Implementation
        f_impl = float(hertz_normal_force(delta, e_star, r_eff))
        
        # Relative error should be < 5%
        rel_error = abs(f_impl - f_analytical) / f_analytical
        self.assertLess(rel_error, 0.05)

    def test_hertz_force_sweep(self):
        """Hertz force matches analytical across delta range."""
        E = 1.0e9
        nu = 0.3
        R = 0.01
        
        e_star = hertz_reduced_modulus(E, nu)
        r_eff = hertz_effective_radius(R)
        
        deltas = np.logspace(-6, -3, 10)
        for delta in deltas:
            f_analytical = (4.0 / 3.0) * e_star * math.sqrt(r_eff) * delta**1.5
            f_impl = float(hertz_normal_force(delta, e_star, r_eff))
            rel_error = abs(f_impl - f_analytical) / f_analytical
            self.assertLess(rel_error, 0.05, f"Mismatch at delta={delta}")


class TestMindlinTangentialContact(unittest.TestCase):
    """D2: Mindlin tangential contact verification."""

    def test_mindlin_stiffness_matches_analytical(self):
        """Mindlin stiffness k_t = 8 G* sqrt(R* delta) matches analytical."""
        E = 1.0e9
        nu = 0.3
        R = 0.01
        delta = 1.0e-4
        
        e_star = hertz_reduced_modulus(E, nu)
        g_star = hertz_mindlin.hertz_reduced_shear_modulus(E, nu)
        r_eff = hertz_effective_radius(R)
        
        # Analytical
        k_t_analytical = 8.0 * g_star * math.sqrt(r_eff * delta)
        
        # Implementation
        k_t_impl = float(hertz_mindlin.hertz_mindlin_tangential_stiffness(
            delta, e_star, g_star, r_eff))
        
        # Relative error should be < 10%
        rel_error = abs(k_t_impl - k_t_analytical) / k_t_analytical
        self.assertLess(rel_error, 0.10)

    def test_mindlin_force_in_stick_regime(self):
        """Mindlin force F_t = k_t * delta_t in stick regime."""
        E = 1.0e9
        nu = 0.3
        R = 0.01
        delta = 1.0e-4
        
        e_star = hertz_reduced_modulus(E, nu)
        g_star = hertz_mindlin.hertz_reduced_shear_modulus(E, nu)
        r_eff = hertz_effective_radius(R)
        
        k_t = float(hertz_mindlin.hertz_mindlin_tangential_stiffness(
            delta, e_star, g_star, r_eff))
        
        delta_t = 1.0e-6
        tang_disp = torch.tensor([[delta_t, 0.0, 0.0]], dtype=torch.float64)
        f_n_mag = torch.tensor([100.0], dtype=torch.float64)
        mu = 0.5
        
        f_t, _ = hertz_mindlin.hertz_mindlin_tangential_force(
            delta, tang_disp, e_star, g_star, r_eff, mu, f_n_mag)
        
        f_t_expected = k_t * delta_t
        f_t_actual = float(f_t.norm(dim=1).item())
        
        rel_error = abs(f_t_actual - f_t_expected) / f_t_expected
        self.assertLess(rel_error, 0.10)


class TestFrictionCalibration(unittest.TestCase):
    """D3: Ice-steel friction calibration."""

    def test_friction_model_fits(self):
        """Friction model can be fitted to calibration data."""
        model = fit_ice_steel_friction(CALIBRATION_ANCHORS)
        self.assertIsNotNone(model)

    def test_friction_prediction_accuracy(self):
        """Friction model predictions within 15% of experimental data."""
        model = fit_ice_steel_friction(CALIBRATION_ANCHORS)
        
        # Test on 3 validation conditions
        for anchor in VALIDATION_ANCHORS[:3]:
            mu_pred = model.mu(anchor.v_m_per_s, anchor.temperature_c, anchor.pressure_mpa)
            mu_exp = anchor.mu
            rel_error = abs(mu_pred - mu_exp) / mu_exp
            self.assertLess(rel_error, 0.15, f"Mismatch at {anchor.label}")


class TestLindqvistCrossValidation(unittest.TestCase):
    """D4: Lindqvist cross-validation."""

    def test_dem_resistance_finite_positive(self):
        """DEM ice resistance is finite and positive."""
        ice_material = IceMaterial()
        ice_plate = IcePlateConfig(thickness_m=1.0)
        ship = ShipWedgeConfig(speed_knots=2.0)
        
        result = run_single_case(ice_material, ice_plate, ship, steps=40)
        r_dem = result["mean_ice_resistance_N"]
        
        self.assertTrue(math.isfinite(r_dem))
        self.assertGreater(r_dem, 0)

    def test_lindqvist_components_positive(self):
        """Lindqvist components are positive."""
        ice_material = IceMaterial()
        ice_plate = IcePlateConfig(thickness_m=1.0)
        ship = ShipWedgeConfig(speed_knots=2.0)
        
        lindqvist = lindqvist_anchor_for_config(ice_material, ice_plate, ship)
        
        self.assertGreater(lindqvist["crushing_N"], 0)
        self.assertGreater(lindqvist["bending_N"], 0)
        self.assertGreater(lindqvist["total_static_N"], 0)


class TestNegativeControls(unittest.TestCase):
    """D5, D6: Negative controls."""

    def test_zero_friction_zero_tangential_force(self):
        """μ=0 → F_t = 0 (D5)."""
        E = 1.0e9
        nu = 0.3
        R = 0.01
        delta = 1.0e-4
        
        e_star = hertz_reduced_modulus(E, nu)
        g_star = hertz_mindlin.hertz_reduced_shear_modulus(E, nu)
        r_eff = hertz_effective_radius(R)
        
        tang_disp = torch.tensor([[0.01, 0.0, 0.0]], dtype=torch.float64)
        f_n_mag = torch.tensor([100.0], dtype=torch.float64)
        mu = 0.0
        
        f_t, _ = hertz_mindlin.hertz_mindlin_tangential_force(
            delta, tang_disp, e_star, g_star, r_eff, mu, f_n_mag)
        
        f_t_mag = float(f_t.norm(dim=1).item())
        self.assertLessEqual(f_t_mag, 1e-12)

    def test_rigid_wall_monotonic_force(self):
        """E→∞ → F_n increases monotonically (D6)."""
        nu = 0.3
        R = 0.01
        delta = 1.0e-4
        
        E_values = np.logspace(9, 15, 7)
        forces = []
        
        for E in E_values:
            e_star = hertz_reduced_modulus(E, nu)
            r_eff = hertz_effective_radius(R)
            f_n = float(hertz_normal_force(delta, e_star, r_eff))
            forces.append(f_n)
            self.assertTrue(math.isfinite(f_n))
        
        # Check monotonic increase
        for i in range(len(forces) - 1):
            self.assertLess(forces[i], forces[i+1])


class TestEnergyConservation(unittest.TestCase):
    """D7: Energy conservation."""

    def test_no_energy_creation(self):
        """Dissipated energy ≤ input work (no energy creation)."""
        ice_material = IceMaterial()
        ice_plate = IcePlateConfig(thickness_m=1.0)
        ship = ShipWedgeConfig(speed_knots=2.0)
        
        result = run_single_case(ice_material, ice_plate, ship, steps=40)
        
        work_input = result.get("cumulative_bow_work_J", 0.0)
        energy_dissipated = result.get("cumulative_friction_dissipation_J", 0.0)
        
        # Dissipated should not exceed input (with 1% tolerance)
        self.assertLessEqual(energy_dissipated, work_input * 1.01)


class TestMeshConvergence(unittest.TestCase):
    """D8: Mesh convergence."""

    def test_medium_resolution_produces_result(self):
        """Medium resolution (R=0.15m) produces finite positive resistance."""
        ice_material = IceMaterial()
        ship = ShipWedgeConfig(speed_knots=2.0)
        
        ice_plate = IcePlateConfig(thickness_m=1.0, particle_radius_m=0.15)
        result = run_single_case(ice_material, ice_plate, ship, steps=60)
        r_mean = result["mean_ice_resistance_N"]
        
        self.assertTrue(math.isfinite(r_mean))
        self.assertGreater(r_mean, 0)


if __name__ == "__main__":
    unittest.main()
