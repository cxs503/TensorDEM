"""Analytic verification tests for the 2-D buoyancy / wet-drag force law.

The tests assert the *dynamical* consequences of
:mod:`tensordem.buoyancy_wet` against closed-form references:

* Archimedes draft ``f_sub(h_eq) = rho_ice / rho_water`` reached by free float,
* restoring stiffness ``dF_b/dh`` against the chord-width identity,
* damped heave frequency, envelope decay rate and waveform against the
  linearised oscillator,
* pure-drag exponential velocity decay,
* timestep convergence and the O(amplitude) nonlinearity limit.

The full campaign is executed once per process (``quick`` budget) and each
assertion then inspects the corresponding sub-check.
"""
from __future__ import annotations

import json
import math
import unittest

import torch

from tensordem.buoyancy_verification import (
    PROTOCOL,
    FreeFloatParameters,
    buoyancy_stiffness_analytic,
    chord_width,
    geometry_check,
    run_campaign,
    solve_equilibrium_height,
    submerged_fraction_quadrature,
)
from tensordem.buoyancy_wet import submerged_fraction_tensor

PARAMS = FreeFloatParameters()
REPORT = run_campaign(PARAMS, include_histories=False, quick=True)


class ReferenceConvergenceTests(unittest.TestCase):
    """The analytic reference itself must be discretisation-independent."""

    def test_quadrature_panel_independence(self):
        """The angle-substituted reference must be panel-converged by default."""
        h_eq = PARAMS.equilibrium_height()
        fine = submerged_fraction_quadrature(h_eq, PARAMS.radius, panels=4000)
        default = submerged_fraction_quadrature(h_eq, PARAMS.radius)
        coarse = submerged_fraction_quadrature(h_eq, PARAMS.radius, panels=64)
        self.assertLess(abs(default - fine), 1e-10)
        self.assertLess(abs(coarse - fine), 1e-6)

    def test_naive_height_quadrature_is_biased(self):
        """Records the trap: Simpson in the height variable is O(n**-1.5).

        The chord width has square-root endpoints, so a 64-panel Simpson /
        trapezoid sum in ``y`` is biased by ~1e-4 in the submerged fraction.
        Such a bias shifts a bisected equilibrium draft by ~1e-4 m and would be
        misread as a solver error -- which is why the reference uses the
        ``y = h + R sin(phi)`` substitution.
        """
        R = PARAMS.radius
        h = PARAMS.equilibrium_height()
        reference = submerged_fraction_quadrature(h, R, panels=4000)

        def naive(hh: float, panels: int) -> float:
            lo, hi = -R + hh, min(0.0, R + hh)
            step = (hi - lo) / panels
            width = lambda y: 2.0 * math.sqrt(max(0.0, R * R - (y - hh) ** 2))  # noqa: E731
            total = 0.0
            for k in range(panels):
                total += 0.5 * (width(lo + k * step) + width(lo + (k + 1) * step))
            return total * step / (math.pi * R * R)

        self.assertGreater(abs(naive(h, 64) - reference), 1e-6)
        self.assertLess(abs(naive(h, 20000) - reference), 1e-4)

    def test_chord_width_limits(self):
        R = PARAMS.radius
        self.assertEqual(chord_width(R, R), 0.0)
        self.assertEqual(chord_width(-R, R), 0.0)
        self.assertAlmostEqual(chord_width(0.0, R), 2.0 * R, places=12)
        self.assertAlmostEqual(chord_width(0.3 * R, R), 2.0 * R * math.sqrt(1 - 0.09),
                               places=12)

    def test_equilibrium_is_monotone_and_bracketed(self):
        h_eq = PARAMS.equilibrium_height()
        self.assertLess(-PARAMS.radius, h_eq)
        self.assertLess(h_eq, PARAMS.radius)
        self.assertLess(h_eq, 0.0)  # ice is lighter than water -> centre below surface
        f_at = submerged_fraction_quadrature(h_eq, PARAMS.radius)
        self.assertAlmostEqual(f_at, PARAMS.rho_ice / PARAMS.rho_water, places=10)


class GeometryAgreementTests(unittest.TestCase):
    def test_closed_form_matches_quadrature(self):
        check = geometry_check(PARAMS.radius)
        self.assertTrue(check["passed"], check)
        self.assertLess(check["max_abs_error"], 1e-9)

    def test_reported_geometry_error(self):
        check = REPORT["checks"]["geometry_vs_quadrature"]
        self.assertLess(check["max_abs_error"], 1e-9)


class StiffnessIdentityTests(unittest.TestCase):
    def test_gradient_matches_chord_width_identity(self):
        check = REPORT["checks"]["stiffness_identity"]
        self.assertTrue(check["passed"], check)
        self.assertLess(check["max_rel_error"], 1e-6)
        # At the draft the analytic stiffness must be positive (restoring).
        k_eq = -buoyancy_stiffness_analytic(
            PARAMS.equilibrium_height(), PARAMS.radius,
            PARAMS.rho_water, PARAMS.gravity, PARAMS.thickness,
        )
        self.assertGreater(k_eq, 0.0)
        closest = min(check["rows"], key=lambda r: abs(
            r["height"] - PARAMS.equilibrium_height()))
        self.assertAlmostEqual(closest["analytic"], k_eq, places=6)


class ArchimedesTests(unittest.TestCase):
    def test_free_float_reaches_analytic_draft(self):
        check = REPORT["checks"]["archimedes_draft"]
        self.assertTrue(check["passed"], check)
        # Explicit acceptance statement: relative draft error < 3%.
        self.assertLess(check["draft_rel_error"], 3e-3)
        self.assertLess(check["f_abs_error"], 1e-3)
        self.assertLess(abs(check["v_final"]), 1e-3)  # settled, not still drifting

    def test_density_ratio_is_literature_range(self):
        """Sea-ice density ratio 0.88-0.90 (Weeks & Ackley 1986) -> f_eq."""
        f_eq = PARAMS.rho_ice / PARAMS.rho_water
        self.assertGreater(f_eq, 0.88)
        self.assertLess(f_eq, 0.90)
        self.assertAlmostEqual(f_eq, REPORT["fixture"]["f_eq_expected"], places=12)


class DampedHeaveTests(unittest.TestCase):
    def test_frequency_and_decay_against_linearised_oscillator(self):
        check = REPORT["checks"]["damped_heave"]
        self.assertTrue(check["passed"], check)
        self.assertLess(check["omega_rel_error"], 3e-2)
        self.assertLess(check["decay_rel_error"], 3e-2)
        self.assertLess(check["waveform_max_rel_error"], 3e-2)
        self.assertGreaterEqual(check["crossings"], 3)

    def test_analytic_parameters_are_self_consistent(self):
        zeta = PARAMS.damping_ratio()
        omega_n = PARAMS.omega_n()
        self.assertAlmostEqual(
            PARAMS.omega_d(), omega_n * math.sqrt(1.0 - zeta * zeta), places=12
        )
        self.assertAlmostEqual(
            PARAMS.stiffness(), PARAMS.mass * omega_n ** 2, places=6
        )


class DragDecayTests(unittest.TestCase):
    def test_exponential_velocity_decay(self):
        check = REPORT["checks"]["drag_exponential_decay"]
        self.assertTrue(check["passed"], check)
        self.assertLess(check["rel_error"], 3e-2)
        # Analytic rate is C_wet / m because f_sub = 1 for a fully submerged
        # disk; the neutral-density fixture has m = rho_water * pi R^2 * t.
        m_neutral = PARAMS.rho_water * math.pi * PARAMS.radius ** 2 * PARAMS.thickness
        self.assertAlmostEqual(
            check["rate_analytic"], PARAMS.wet_drag_coeff / m_neutral, places=12
        )


class ConvergenceTests(unittest.TestCase):
    def test_timestep_convergence(self):
        check = REPORT["checks"]["timestep_convergence"]
        self.assertTrue(check["passed"], check)
        self.assertGreaterEqual(check["observed_order"], 3.0)
        self.assertLess(check["omega_d_spread_rel"], 1e-2)

    def test_linearisation_error_shrinks_with_amplitude(self):
        check = REPORT["checks"]["linearisation_limit"]
        self.assertTrue(check["passed"], check)
        rows = check["rows"]
        self.assertLess(rows[-1]["waveform_max_rel_error"],
                        rows[0]["waveform_max_rel_error"])


class CampaignReportTests(unittest.TestCase):
    def test_all_checks_pass_and_protocol_is_stable(self):
        self.assertTrue(REPORT["all_passed"])
        self.assertEqual(REPORT["protocol"], PROTOCOL)
        self.assertEqual(
            set(REPORT["checks"]),
            {"geometry_vs_quadrature", "stiffness_identity", "archimedes_draft",
             "damped_heave", "drag_exponential_decay", "timestep_convergence",
             "linearisation_limit"},
        )

    def test_report_is_json_serialisable(self):
        encoded = json.dumps(REPORT, sort_keys=True)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["fixture"], REPORT["fixture"])
        self.assertNotIn("history", decoded["checks"]["damped_heave"])

    def test_fixture_mass_matches_geometry(self):
        fixture = REPORT["fixture"]
        expected = (fixture["rho_ice_kg_m3"] * math.pi * fixture["radius_m"] ** 2
                    * fixture["thickness_m"])
        self.assertAlmostEqual(fixture["mass_kg"], expected, places=9)
        # Sanity on the fixture the checks are run in.
        self.assertFalse(torch.isnan(torch.tensor(REPORT["fixture"]["zeta"])))


if __name__ == "__main__":
    unittest.main()