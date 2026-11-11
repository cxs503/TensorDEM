"""Comprehensive tests for 3D ice ridge-ship interaction module.

Tests cover:
- Ridge geometry parameterization and volume (criterion a)
- Rubble particle generation with PSD (criterion b)
- Hertz contact mechanics (criterion c)
- Ridge failure load vs Kovacs 1998 (criterion d)
- Hull FEM response (criterion e)
- Interlocking shear strength (criterion f)
- Mass conservation (criterion g)
- Negative controls (2 tests)
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from tensordem.ice_ridge_3d.geometry import (
    RidgeGeometry,
    ridge_volume,
    keel_half_width,
    sail_half_width,
)
from tensordem.ice_ridge_3d.rubble import (
    PowerLawPSD3D,
    generate_rubble_particles,
    particle_count_closed_form,
    total_particle_volume,
    measure_porosity,
)
from tensordem.ice_ridge_3d.contact import (
    hertz_normal_force_3d,
    hertz_contact_energy_3d,
    hertz_secant_stiffness_3d,
    hertz_reduced_modulus_3d,
    hertz_effective_radius_3d,
    coulomb_friction_3d,
    rolling_resistance_3d,
    interlocking_shear_strength,
    compute_contact_force,
)
from tensordem.ice_ridge_3d.interaction import (
    RidgeShipConfig,
    RidgeShipInteraction,
    ridge_crushing_load,
    iso19906_ridge_load,
    simulate_ridge_ship,
)
from tensordem.ice_ridge_3d.fem_response import (
    PlateProperties,
    plate_stress_simply_supported,
    plate_deflection_simply_supported,
    hull_panel_response,
    von_mises_stress,
)
from tensordem.ice_ridge_3d.verification import (
    AcceptanceConfig,
    run_acceptance,
    verify_ridge_volume,
    verify_particle_volume,
    verify_hertz_contact,
    verify_ridge_load,
    verify_hull_fem,
    verify_interlocking,
    verify_mass_conservation,
)


# =====================================================================
# Geometry tests
# =====================================================================
class TestRidgeGeometry(unittest.TestCase):
    def test_volume_analytical(self):
        """Criterion (a): Ridge volume matches analytical formula."""
        geom = RidgeGeometry(
            sail_height=2.0, sail_width=10.0,
            keel_depth=10.0, keel_width=40.0,
            ridge_length=100.0, porosity=0.35,
        )
        v_analytical = ridge_volume(2.0, 10.0, 10.0, 40.0, 100.0)
        v_computed = geom.sail_volume + geom.keel_volume
        error = abs(v_computed - v_analytical) / v_analytical
        self.assertLess(error, 1e-9)

    def test_keel_solid_volume(self):
        """Keel solid volume = (1-n) * V_keel."""
        geom = RidgeGeometry(
            sail_height=2.0, sail_width=10.0,
            keel_depth=10.0, keel_width=40.0,
            ridge_length=100.0, porosity=0.35,
        )
        expected = (1.0 - 0.35) * geom.keel_volume
        self.assertAlmostEqual(geom.keel_solid_volume, expected, places=6)

    def test_keel_half_width_linear(self):
        """Keel half-width decreases linearly from waterline to tip."""
        hw_top = keel_half_width(40.0, 10.0, 0.0)
        hw_mid = keel_half_width(40.0, 10.0, 5.0)
        hw_tip = keel_half_width(40.0, 10.0, 10.0)
        self.assertAlmostEqual(hw_top, 20.0)
        self.assertAlmostEqual(hw_mid, 10.0)
        self.assertAlmostEqual(hw_tip, 0.0)

    def test_sail_half_width_linear(self):
        """Sail half-width decreases linearly from waterline to peak."""
        hw_base = sail_half_width(10.0, 2.0, 0.0)
        hw_mid = sail_half_width(10.0, 2.0, 1.0)
        hw_top = sail_half_width(10.0, 2.0, 2.0)
        self.assertAlmostEqual(hw_base, 5.0)
        self.assertAlmostEqual(hw_mid, 2.5)
        self.assertAlmostEqual(hw_top, 0.0)

    def test_invalid_porosity(self):
        """Porosity must be in [0, 0.6)."""
        with self.assertRaises(ValueError):
            RidgeGeometry(
                sail_height=2.0, sail_width=10.0,
                keel_depth=10.0, keel_width=40.0,
                ridge_length=100.0, porosity=0.7,
            )

    def test_negative_dimensions(self):
        """Negative dimensions raise ValueError."""
        with self.assertRaises(ValueError):
            RidgeGeometry(
                sail_height=-1.0, sail_width=10.0,
                keel_depth=10.0, keel_width=40.0,
                ridge_length=100.0,
            )


# =====================================================================
# Rubble generation tests
# =====================================================================
class TestRubbleGeneration(unittest.TestCase):
    def test_psd_sample_range(self):
        """Sampled diameters are within [d_min, d_max] (before scaling)."""
        psd = PowerLawPSD3D(d_min=0.2, d_max=2.0, alpha=2.5)
        rng = np.random.default_rng(42)
        diameters = psd.sample(1000, rng)
        self.assertTrue(np.all(diameters >= 0.2))
        self.assertTrue(np.all(diameters <= 2.0))

    def test_particle_volume_target(self):
        """Criterion (b): Particle volume error < 2% with scaled sampling."""
        geom = RidgeGeometry(
            sail_height=2.0, sail_width=10.0,
            keel_depth=10.0, keel_width=40.0,
            ridge_length=100.0, porosity=0.35,
        )
        psd = PowerLawPSD3D(d_min=0.2, d_max=2.0, alpha=2.5)
        result = generate_rubble_particles(
            geom.keel_volume, geom.porosity, psd,
            target_count=500
        )
        self.assertLess(result["volume_error"], 0.02)

    def test_particle_count_closed_form(self):
        """Closed-form particle count is positive and finite."""
        psd = PowerLawPSD3D(d_min=0.2, d_max=2.0, alpha=2.5)
        count = particle_count_closed_form(20000.0, 0.35, psd)
        self.assertTrue(math.isfinite(count))
        self.assertGreater(count, 0)

    def test_measure_porosity(self):
        """Measured porosity is in [0, 1]."""
        por = measure_porosity(10000.0, 20000.0)
        self.assertAlmostEqual(por, 0.5)
        self.assertTrue(0.0 <= por <= 1.0)

    def test_total_particle_volume(self):
        """Total volume of spheres matches pi/6 * sum(D³)."""
        diameters = np.array([1.0, 2.0, 3.0])
        expected = math.pi / 6.0 * (1.0 + 8.0 + 27.0)
        actual = total_particle_volume(diameters)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_psd_invalid_alpha(self):
        """Alpha = 1 or 4 raises ValueError."""
        with self.assertRaises(ValueError):
            PowerLawPSD3D(alpha=1.0)
        with self.assertRaises(ValueError):
            PowerLawPSD3D(alpha=4.0)

    def test_scaled_sample_matches_volume(self):
        """Scaled sampling produces diameters matching target volume."""
        psd = PowerLawPSD3D(d_min=0.2, d_max=2.0, alpha=2.5)
        rng = np.random.default_rng(42)
        target_vol = 1000.0
        diameters = psd.sample_scaled(100, target_vol, rng)
        actual_vol = total_particle_volume(diameters)
        error = abs(actual_vol - target_vol) / target_vol
        self.assertLess(error, 1e-10)


# =====================================================================
# Contact mechanics tests
# =====================================================================
class TestContactMechanics(unittest.TestCase):
    def test_hertz_force_exponent(self):
        """Criterion (c): Hertz force scales as delta^(3/2)."""
        e_star = 1.0e9 / (2.0 * (1.0 - 0.3 ** 2))
        r_eff = 0.5
        d1, d2 = 1e-4, 1e-3
        f1 = hertz_normal_force_3d(d1, e_star, r_eff)
        f2 = hertz_normal_force_3d(d2, e_star, r_eff)
        ratio = f2 / f1
        expected_ratio = (d2 / d1) ** 1.5
        self.assertAlmostEqual(ratio, expected_ratio, places=6)

    def test_hertz_force_analytical(self):
        """Criterion (c): Hertz force matches analytical within 1%."""
        E = 1.0e9
        nu = 0.3
        R = 0.5
        delta = 1e-3

        # Independent reference
        e_eff = E / (2.0 * (1.0 - nu ** 2))
        r_eff_ref = R / 2.0
        f_ref = (4.0 / 3.0) * e_eff * math.sqrt(r_eff_ref) * delta ** 1.5

        # Computed
        e_star = hertz_reduced_modulus_3d(E, nu)
        r_eff = hertz_effective_radius_3d(R, R)
        f_computed = hertz_normal_force_3d(delta, e_star, r_eff)

        error = abs(f_computed - f_ref) / f_ref
        self.assertLess(error, 0.01)

    def test_hertz_zero_overlap(self):
        """Zero overlap gives zero force."""
        self.assertEqual(hertz_normal_force_3d(0.0, 1.0, 1.0), 0.0)
        self.assertEqual(hertz_contact_energy_3d(0.0, 1.0, 1.0), 0.0)

    def test_hertz_energy_relation(self):
        """Energy is integral of force: U = (2/5) * F * delta for Hertz."""
        e_star = 1.0e9
        r_eff = 0.5
        delta = 1e-3
        f = hertz_normal_force_3d(delta, e_star, r_eff)
        u = hertz_contact_energy_3d(delta, e_star, r_eff)
        expected_u = (2.0 / 5.0) * f * delta
        self.assertAlmostEqual(u, expected_u, places=10)

    def test_coulomb_friction_saturation(self):
        """Tangential force saturates at mu * F_n."""
        mu = 0.3
        fn = 1000.0
        k_t = 1.0e6
        # Large displacement should saturate
        disp = np.array([0.01, 0.0, 0.0])
        f_t = coulomb_friction_3d(disp, fn, mu, k_t, 1.0)
        f_t_mag = np.linalg.norm(f_t)
        self.assertLessEqual(f_t_mag, mu * fn + 1e-10)

    def test_rolling_resistance_opposes_rotation(self):
        """Rolling torque opposes angular velocity."""
        omega = np.array([1.0, 0.0, 0.0])
        torque = rolling_resistance_3d(omega, 1000.0, 0.05, 0.5)
        self.assertLess(torque[0], 0.0)  # opposite direction

    def test_interlocking_enhancement(self):
        """Criterion (f): Interlocking increases shear strength > 30%."""
        base = 1.0e5
        # With interlock_depth=0.1, D=1.0, factor=1.5, gamma=0.5:
        # enhancement = 1 + 1.5 * sqrt(0.1/1.0) = 1 + 1.5 * 0.316 = 1.474
        tau_with = interlocking_shear_strength(
            base, 0.1, 1.0, 0.3, 1.5
        )
        diff_pct = (tau_with - base) / base * 100.0
        self.assertGreater(diff_pct, 30.0)

    def test_compute_contact_force(self):
        """Full contact force computation returns valid state."""
        pos1 = np.array([0.0, 0.0, 0.0])
        pos2 = np.array([0.9, 0.0, 0.0])
        r1, r2 = 0.5, 0.5
        state = compute_contact_force(
            pos1, r1, pos2, r2,
            1.0e9, 0.3, 0.3, 0.05,
            np.zeros(3), 1.0e6, np.zeros(3)
        )
        self.assertGreater(state.normal_force, 0.0)
        self.assertTrue(np.all(np.isfinite(state.tangential_force)))


# =====================================================================
# Interaction tests
# =====================================================================
class TestRidgeShipInteraction(unittest.TestCase):
    def test_simulation_runs(self):
        """Simulation completes without error."""
        geom = RidgeGeometry(
            sail_height=2.0, sail_width=10.0,
            keel_depth=10.0, keel_width=40.0,
            ridge_length=100.0, porosity=0.35,
        )
        psd = PowerLawPSD3D(d_min=0.2, d_max=2.0, alpha=2.5)
        config = RidgeShipConfig(
            ridge_geometry=geom,
            psd=psd,
            n_steps=50,
            n_particles=50,
            seed=42,
        )
        result = simulate_ridge_ship(config)
        self.assertIn("max_force", result)
        self.assertIn("force_history", result)
        self.assertEqual(len(result["force_history"]), 50)

    def test_force_is_positive(self):
        """Resistance force is positive (opposing motion)."""
        geom = RidgeGeometry(
            sail_height=2.0, sail_width=10.0,
            keel_depth=10.0, keel_width=40.0,
            ridge_length=100.0, porosity=0.35,
        )
        psd = PowerLawPSD3D(d_min=0.2, d_max=2.0, alpha=2.5)
        config = RidgeShipConfig(
            ridge_geometry=geom,
            psd=psd,
            n_steps=100,
            n_particles=100,
            seed=42,
        )
        result = simulate_ridge_ship(config)
        # Force should be non-negative (resistance)
        self.assertTrue(all(f >= 0 for f in result["force_history"]))

    def test_ridge_crushing_load(self):
        """Crushing load = sigma_c * A_contact."""
        f = ridge_crushing_load(5.0e6, 10.0)
        self.assertAlmostEqual(f, 5.0e7)

    def test_iso19906_ridge_load(self):
        """ISO 19906 load scales as h_k^beta."""
        f1 = iso19906_ridge_load(10.0)
        f2 = iso19906_ridge_load(5.0)
        # F(10)/F(5) should be (10/5)^1.5 = 2^1.5 ≈ 2.828
        ratio = f1 / f2
        self.assertAlmostEqual(ratio, 2.0 ** 1.5, places=4)

    def test_iso19906_matches_kovacs(self):
        """ISO 19906 load for h_k=10m matches Kovacs 1998 (~5MN)."""
        f = iso19906_ridge_load(10.0)
        # Should be within 20% of 5 MN
        self.assertLess(abs(f - 5.0e6) / 5.0e6, 0.20)


# =====================================================================
# FEM response tests
# =====================================================================
class TestFEMResponse(unittest.TestCase):
    def test_plate_stress_formula(self):
        """Criterion (e): Stress formula matches closed-form."""
        pressure = 1.0e6
        b = 1.0
        t = 0.015
        a_b = 2.0
        sigma = plate_stress_simply_supported(pressure, b, t, a_b)
        # k ≈ 0.5 for a/b >= 2
        expected = 0.5 * pressure * b ** 2 / t ** 2
        self.assertAlmostEqual(sigma, expected, places=6)

    def test_plate_deflection(self):
        """Deflection formula is positive and finite."""
        w = plate_deflection_simply_supported(
            1.0e6, 1.0, 0.015, 2.0e11, 0.3, 2.0
        )
        self.assertGreater(w, 0.0)
        self.assertTrue(math.isfinite(w))

    def test_hull_panel_response(self):
        """Hull panel response returns all required fields."""
        plate = PlateProperties(
            width=1.0, length=2.0, thickness=0.015,
            young_modulus=2.0e11, poisson_ratio=0.3,
            yield_strength=355.0e6,
        )
        response = hull_panel_response(1.0e7, 10.0, plate)
        self.assertIn("sigma_max_pa", response)
        self.assertIn("deflection_m", response)
        self.assertIn("safety_margin", response)
        self.assertIn("is_safe", response)

    def test_von_mises_stress(self):
        """Von Mises stress for uniaxial tension equals applied stress."""
        sigma = von_mises_stress(100.0, 0.0, 0.0)
        self.assertAlmostEqual(sigma, 100.0)

    def test_plate_invalid_thickness(self):
        """Zero thickness raises ValueError."""
        with self.assertRaises(ValueError):
            plate_stress_simply_supported(1.0e6, 1.0, 0.0, 2.0)


# =====================================================================
# Verification / acceptance tests
# =====================================================================
class TestAcceptance(unittest.TestCase):
    def test_full_acceptance(self):
        """All acceptance criteria pass with default config."""
        result = run_acceptance()
        self.assertTrue(result.all_pass)

    def test_criterion_a(self):
        """Criterion (a): Ridge volume error < 1e-9."""
        geom = RidgeGeometry(
            sail_height=2.0, sail_width=10.0,
            keel_depth=10.0, keel_width=40.0,
            ridge_length=100.0, porosity=0.35,
        )
        r = verify_ridge_volume(geom)
        self.assertTrue(r["pass"])
        self.assertLess(r["error"], 1e-9)

    def test_criterion_b(self):
        """Criterion (b): Particle volume error < 2%."""
        geom = RidgeGeometry(
            sail_height=2.0, sail_width=10.0,
            keel_depth=10.0, keel_width=40.0,
            ridge_length=100.0, porosity=0.35,
        )
        psd = PowerLawPSD3D(d_min=0.2, d_max=2.0, alpha=2.5)
        r = verify_particle_volume(geom, psd, 500, 42)
        self.assertTrue(r["pass"])
        self.assertLess(r["error"], 0.02)

    def test_criterion_c(self):
        """Criterion (c): Hertz force error < 1%."""
        r = verify_hertz_contact(1.0e9, 0.3, 0.5, 42)
        self.assertTrue(r["pass"])
        self.assertLess(r["max_error"], 0.01)

    def test_criterion_d(self):
        """Criterion (d): Ridge load vs Kovacs error < 20%."""
        geom = RidgeGeometry(
            sail_height=2.0, sail_width=10.0,
            keel_depth=10.0, keel_width=40.0,
            ridge_length=100.0, porosity=0.35,
        )
        r = verify_ridge_load(geom, 5.0e6, 100, 42)
        self.assertTrue(r["pass"])
        self.assertLess(r["error_vs_kovacs"], 0.20)

    def test_criterion_e(self):
        """Criterion (e): Hull FEM stress error < 3%."""
        r = verify_hull_fem(
            1.0e7, 10.0, 1.0, 2.0, 0.015,
            2.0e11, 0.3, 355.0e6
        )
        self.assertTrue(r["pass"])
        self.assertLess(r["error"], 0.03)

    def test_criterion_f(self):
        """Criterion (f): Interlocking difference > 30%."""
        r = verify_interlocking(1.0e5, 0.1, 1.0, 0.3, 1.5)
        self.assertTrue(r["pass"])
        self.assertGreater(r["difference_pct"], 30.0)

    def test_criterion_g(self):
        """Criterion (g): Mass conservation < 0.1%."""
        r = verify_mass_conservation(1000.0, 1000.0)
        self.assertTrue(r["pass"])
        self.assertLess(r["change_pct"], 0.1)


# =====================================================================
# Negative control tests
# =====================================================================
class TestNegativeControls(unittest.TestCase):
    def test_zero_overlap_no_force(self):
        """Negative control: zero overlap produces zero contact force."""
        pos1 = np.array([0.0, 0.0, 0.0])
        pos2 = np.array([2.0, 0.0, 0.0])  # far apart
        r1, r2 = 0.5, 0.5
        state = compute_contact_force(
            pos1, r1, pos2, r2,
            1.0e9, 0.3, 0.3, 0.05,
            np.zeros(3), 1.0e6, np.zeros(3)
        )
        self.assertEqual(state.normal_force, 0.0)
        self.assertEqual(state.normal_overlap, 0.0)

    def test_zero_interlock_no_enhancement(self):
        """Negative control: zero interlock depth gives no enhancement."""
        base = 1.0e5
        tau = interlocking_shear_strength(
            base, 0.0, 1.0, 0.3, 1.5
        )
        self.assertAlmostEqual(tau, base)


if __name__ == "__main__":
    unittest.main()
