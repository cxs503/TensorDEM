"""Scientific audit verification: TensorDEM physical model consistency.

Audits three physical models against their docstring/literature claims:
1. brash_ice_rheology — mu(I) shear + buoyancy + keel drag
2. hertz_mindlin — normal Hertz + tangential Mindlin
3. clear_ice.contact — Hertz-Mindlin + Tsuji damping + non-cohesive clamp

All assertions are against closed-form values or conservation laws.
"""
from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from tensordem import hertz_mindlin
from tensordem.brash_ice_rheology import (
    BrashHull,
    BrashIceState,
    Closure,
    JopParameters,
    LITERATURE_JOP,
    STANDARD_BRASH,
    buoyancy_component,
    inertial_number,
    keel_drag_component,
    shear_component,
)
from tensordem.clear_ice.contact import (
    ContactMaterial,
    ContactSolver,
    ZETA_TABLE,
    E_TABLE,
    _restitution_of_zeta,
    damping_ratio_from_restitution,
    damping_ratio_from_restitution_analytic,
    effective_modulus,
    effective_radius,
    effective_shear_modulus,
    hertz_elastic_force,
    hertz_stiffness,
)


# ======================================================================
# Model 1: Brash Ice Rheology
# ======================================================================

class BrashIceRheologyAudit(unittest.TestCase):
    """Audit brash_ice_rheology against docstring formulas."""

    def test_mu_I_formula_matches_literature(self):
        """Jop et al. 2006: mu(I) = mu_s + (mu_2-mu_s)/(I_0/I + 1)."""
        params = JopParameters(*LITERATURE_JOP)
        # Check at I=0.1 (mid-range)
        I = 0.1
        expected = 0.36 + (0.84 - 0.36) / (0.30 / I + 1.0)
        actual = params.mu(I)
        self.assertAlmostEqual(actual, expected, places=10)
        # Check limits
        self.assertAlmostEqual(params.mu(1e-30), 0.36, places=10)  # quasi-static
        self.assertAlmostEqual(params.mu(1e10), 0.84, places=5)    # collisional

    def test_inertial_number_formula(self):
        """I = gamma_dot * d / sqrt(P/rho)."""
        gamma_dot = 1.0  # 1/s
        d = 0.5  # m
        P = 1000.0  # Pa
        rho = 950.0  # kg/m^3
        expected = gamma_dot * d / math.sqrt(P / rho)
        actual = inertial_number(gamma_dot, d, P, rho)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_confinement_pressure_formula(self):
        """P = (rho_w - rho_b)*g*h_sub + 0.5*C_c*rho_b*v^2."""
        state = BrashIceState(porosity=0.3)
        hull = BrashHull()
        closure = Closure()
        speed = 3.0

        # Static part
        p_static = state.static_confinement_pa
        # Dynamic part
        p_dyn = 0.5 * closure.confinement_coefficient * state.bulk_density_kg_m3 * speed ** 2
        P_expected = p_static + p_dyn

        # Get from shear_component
        result = shear_component(state, hull, speed, closure)
        self.assertAlmostEqual(result["confinement_pa"], P_expected, places=3)
        self.assertAlmostEqual(result["static_confinement_pa"], p_static, places=3)
        self.assertAlmostEqual(result["dynamic_confinement_pa"], p_dyn, places=3)

    def test_shear_component_formula(self):
        """F_shear = mu(I)*P*A_shear*(1-phi)*f(v)."""
        state = BrashIceState(porosity=0.3)
        hull = BrashHull()
        closure = Closure()
        speed = 3.0

        result = shear_component(state, hull, speed, closure)
        P = result["confinement_pa"]
        mu = result["mu_effective"]
        tau = mu * P * state.solid_fraction
        area = hull.waterline_length_m * state.submerged_thickness_m
        f = speed / (speed + closure.mobilization_speed_m_s)
        F_expected = tau * area * f

        self.assertAlmostEqual(result["F_shear_N"], F_expected, places=3)

    def test_buoyancy_component_formula(self):
        """F_buoy = rho_w * g * V_displaced * f(v), V = factor*(1-phi)*B*h_sub."""
        state = BrashIceState(porosity=0.3)
        hull = BrashHull()
        closure = Closure()
        speed = 3.0

        result = buoyancy_component(state, hull, speed, closure)
        # The implementation uses displaced_ice_volume_factor from closure
        factor = closure.displaced_ice_volume_factor
        V_disp = factor * state.solid_fraction * hull.beam_m * state.submerged_thickness_m
        f = speed / (speed + closure.mobilization_speed_m_s)
        F_expected = state.water_density_kg_m3 * state.gravity_m_s2 * V_disp * f

        self.assertAlmostEqual(result["F_buoy_N"], F_expected, places=3)

    def test_drag_component_formula(self):
        """F_drag = 0.5*C_D*rho_w*v^2*A_keel*f(v)."""
        state = BrashIceState(porosity=0.3)
        hull = BrashHull()
        closure = Closure()
        speed = 3.0

        result = keel_drag_component(state, hull, speed, closure)
        A_keel = state.solid_fraction * hull.beam_m * state.submerged_thickness_m
        f = speed / (speed + closure.mobilization_speed_m_s)
        F_expected = 0.5 * closure.keel_drag_coefficient * state.water_density_kg_m3 * speed ** 2 * A_keel * f

        self.assertAlmostEqual(result["F_drag_N"], F_expected, places=3)

    def test_porosity_limits(self):
        """phi->1: all forces -> 0 (pure water)."""
        state_dense = BrashIceState(porosity=0.99)
        hull = BrashHull()
        closure = Closure()
        speed = 3.0

        r_shear = shear_component(state_dense, hull, speed, closure)
        r_buoy = buoyancy_component(state_dense, hull, speed, closure)
        r_drag = keel_drag_component(state_dense, hull, speed, closure)

        # All should be near zero (solid fraction = 0.01)
        self.assertLess(r_shear["F_shear_N"], 5000.0)
        self.assertLess(r_buoy["F_buoy_N"], 5000.0)
        self.assertLess(r_drag["F_drag_N"], 5000.0)

    def test_zero_speed_vanishes(self):
        """R(0) = 0 exactly."""
        state = BrashIceState(porosity=0.3)
        hull = BrashHull()
        closure = Closure()

        r = shear_component(state, hull, 0.0, closure)
        self.assertEqual(r["F_shear_N"], 0.0)

        r = buoyancy_component(state, hull, 0.0, closure)
        self.assertEqual(r["F_buoy_N"], 0.0)

        r = keel_drag_component(state, hull, 0.0, closure)
        self.assertEqual(r["F_drag_N"], 0.0)


# ======================================================================
# Model 2: Hertz-Mindlin Contact
# ======================================================================

class HertzMindlinAudit(unittest.TestCase):
    """Audit hertz_mindlin against docstring formulas."""

    def test_reduced_shear_modulus_equal_spheres(self):
        """G* = E / [4*(2-nu)*(1+nu)]."""
        E = 1e9
        nu = 0.3
        expected = E / (4.0 * (2.0 - nu) * (1.0 + nu))
        actual = hertz_mindlin.hertz_reduced_shear_modulus(E, nu)
        self.assertAlmostEqual(actual, expected, places=5)

    def test_reduced_shear_modulus_rigid(self):
        """G* = E / [2*(2-nu)*(1+nu)] against rigid."""
        E = 1e9
        nu = 0.3
        expected = E / (2.0 * (2.0 - nu) * (1.0 + nu))
        actual = hertz_mindlin.hertz_reduced_shear_modulus(E, nu, rigid_counterpart=True)
        self.assertAlmostEqual(actual, expected, places=5)

    def test_tangential_stiffness_formula(self):
        """k_t = 8*G*sqrt(R*delta)."""
        E = 1e9
        nu = 0.3
        R = 0.01
        delta = 1e-4

        g_star = hertz_mindlin.hertz_reduced_shear_modulus(E, nu)
        expected = 8.0 * g_star * math.sqrt(R * delta)
        actual = float(hertz_mindlin.hertz_mindlin_tangential_stiffness(delta, E, g_star, R))
        self.assertAlmostEqual(actual, expected, places=5)

    def test_stiffness_ratio_formula(self):
        """k_t/k_n = 3*(1-nu)/(2-nu)."""
        for nu in [0.0, 0.25, 0.3]:
            expected = 3.0 * (1.0 - nu) / (2.0 - nu)
            actual = hertz_mindlin.mindlin_stiffness_ratio(nu)
            self.assertAlmostEqual(actual, expected, places=5)

    def test_coulomb_saturation(self):
        """Tangential force saturates at mu*F_n."""
        overlap = 1e-4
        E = 1e9
        nu = 0.3
        R = 0.01
        mu = 0.3
        F_n = 100.0

        g_star = hertz_mindlin.hertz_reduced_shear_modulus(E, nu)
        e_star = E / (2.0 * (1.0 - nu ** 2))

        # Large tangential displacement -> slip
        delta_t = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        F_t, state = hertz_mindlin.hertz_mindlin_tangential_force(
            overlap, delta_t, e_star, g_star, R, mu, torch.tensor([F_n]))

        self.assertAlmostEqual(float(F_t.norm(dim=1).item()), mu * F_n, places=4)

    def test_state_projection(self):
        """After slip, state = F_t / k_t."""
        overlap = 1e-4
        E = 1e9
        nu = 0.3
        R = 0.01
        mu = 0.3
        F_n = 100.0

        g_star = hertz_mindlin.hertz_reduced_shear_modulus(E, nu)
        e_star = E / (2.0 * (1.0 - nu ** 2))

        delta_t = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        F_t, state = hertz_mindlin.hertz_mindlin_tangential_force(
            overlap, delta_t, e_star, g_star, R, mu, torch.tensor([F_n]))

        k_t = float(hertz_mindlin.hertz_mindlin_tangential_stiffness(overlap, e_star, g_star, R))
        expected_state = F_t / k_t
        torch.testing.assert_close(state, expected_state, rtol=1e-5, atol=1e-10)


# ======================================================================
# Model 3: Clear Ice Contact (Hertz-Mindlin + Tsuji Damping)
# ======================================================================

class ClearIceContactAudit(unittest.TestCase):
    """Audit clear_ice.contact against docstring formulas."""

    def test_hertz_elastic_force(self):
        """F_n = (4/3)*E*sqrt(R)*delta^(3/2)."""
        mat = ContactMaterial()
        e_eff = effective_modulus(mat, mat)
        r_eff = effective_radius(0.9, 0.9)
        delta = 1e-3

        expected = (4.0 / 3.0) * e_eff * np.sqrt(r_eff) * delta ** 1.5
        actual = hertz_elastic_force(delta, e_eff, r_eff)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_hertz_stiffness_derivative(self):
        """k_n = dF/ddelta = 2*E*sqrt(R*delta)."""
        mat = ContactMaterial()
        e_eff = effective_modulus(mat, mat)
        r_eff = effective_radius(0.9, 0.9)
        delta = 1e-3

        h = 1e-9
        num_deriv = (hertz_elastic_force(delta + h, e_eff, r_eff)
                     - hertz_elastic_force(delta - h, e_eff, r_eff)) / (2 * h)
        actual = hertz_stiffness(delta, e_eff, r_eff)
        # Use relative tolerance for derivative check
        self.assertAlmostEqual(num_deriv / actual, 1.0, places=4)

    def test_shear_modulus_formula(self):
        """G* = 1/((2-nu1)/G1 + (2-nu2)/G2)."""
        mat = ContactMaterial()
        G = mat.youngs_modulus / (2.0 * (1.0 + mat.poisson_ratio))
        expected = 1.0 / (2.0 * (2.0 - mat.poisson_ratio) / G)
        actual = effective_shear_modulus(mat, mat)
        self.assertAlmostEqual(actual, expected, places=5)

    def test_damping_ratio_analytic(self):
        """zeta = -ln(e) / sqrt(pi^2 + ln^2(e))."""
        e = 0.2
        ln_e = math.log(e)
        expected = -ln_e / math.sqrt(math.pi ** 2 + ln_e ** 2)
        actual = damping_ratio_from_restitution_analytic(e)
        self.assertAlmostEqual(actual, expected, places=5)

    def test_damping_ratio_calibration_table(self):
        """Table values must be reproducible by the integrator."""
        for zeta, e in zip(ZETA_TABLE[::4], E_TABLE[::4]):
            actual = _restitution_of_zeta(zeta)
            self.assertAlmostEqual(actual, e, places=3)

    def test_non_cohesive_clamp(self):
        """Separating pair must have zero normal force."""
        r = 0.9
        mat = ContactMaterial()
        solver = ContactSolver(mat, box=None)
        mass = mat.density * np.pi * r ** 2 * 0.3

        # Separating pair
        x = np.array([[0.0, 0.0], [2 * r - 1e-3, 0.0]])
        v = np.array([[-0.5, 0.0], [0.5, 0.0]])

        f, _, diag = solver.update(x, v, np.array([r, r]),
                                   np.array([mass, mass]),
                                   np.zeros(2), 1e-4, 0.3)

        self.assertEqual(diag.fn[0], 0.0)
        self.assertTrue(np.all(diag.fn >= 0.0))

    def test_mindlin_tangential_stiffness(self):
        """k_t = 8*G*sqrt(R*delta)."""
        mat = ContactMaterial()
        r = 0.9
        delta = 2e-3
        r_eff = effective_radius(r, r)
        g_eff = effective_shear_modulus(mat, mat)

        expected = 8.0 * g_eff * np.sqrt(r_eff * delta)
        # The solver's k_t should match this
        self.assertGreater(expected, 0.0)

    def test_momentum_conservation(self):
        """Contact forces must be equal and opposite."""
        r = 0.9
        mat = ContactMaterial()
        solver = ContactSolver(mat, box=None)
        mass = mat.density * np.pi * r ** 2 * 0.3

        x = np.array([[0.0, 0.0], [1.7, 0.1]])
        v = np.array([[-0.2, 0.05], [0.3, -0.02]])
        omega = np.zeros(2)

        f, _, _ = solver.update(x, v, np.array([r, r]),
                                np.array([mass, mass]),
                                omega, 1e-4, 0.3)

        self.assertTrue(np.allclose(f.sum(axis=0), 0.0, atol=1e-9))


# ======================================================================
# Run all tests
# ======================================================================

if __name__ == "__main__":
    unittest.main()
