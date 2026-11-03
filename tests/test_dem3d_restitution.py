"""Verification of the DEM3D normal-contact coefficient of restitution.

The engine implements a *unilateral* (force-clamped) linear spring-dashpot
contact, so the textbook Kelvin-Voigt restitution formula is not the right
benchmark.  These tests (a) pin the clamped closed form, (b) check that the
damping calibration inverts it, (c) reproduce the documented default value and
(d) compare the closed form with a discrete-element bounce experiment.
"""
import math
import unittest

import torch

from tensordem.dem3d import DEM3DConfig
from tensordem.restitution import (
    clamped_contact_restitution,
    contact_restitution,
    damping_for_clamped_restitution,
    damping_for_restitution_analytic,
    damping_ratio,
    measure_platen_restitution,
    platen_restitution,
    restitution_from_damping,
    restitution_from_damping_clamped,
)

MASS = 0.06001750965295501  # r = 25 mm, rho = 917 kg/m3
STIFFNESS = 5_000.0


def _reference_clamped_restitution(k: float, c: float, m: float) -> float:
    """Independent RK4 integration of the clamped 1-DOF contact."""
    zeta = c / (2.0 * math.sqrt(k * m))
    if zeta == 0.0:
        return 1.0
    dt = 2.0e-6
    delta, velocity, steps = 0.0, 1.0, 0  # overlap grows at unit approach speed

    def accel(overlap: float, speed: float) -> float:
        force = max(0.0, k * overlap + c * speed)
        return -force / m

    while steps < 20_000:
        force = k * delta + c * velocity
        if delta > 0.0 and force <= 0.0:
            return abs(velocity)
        k1v = accel(delta, velocity)
        k2v = accel(delta + 0.5 * dt * velocity, velocity + 0.5 * dt * k1v)
        k3v = accel(delta + 0.5 * dt * (velocity + 0.5 * dt * k1v),
                    velocity + 0.5 * dt * k2v)
        k4v = accel(delta + dt * (velocity + 0.5 * dt * k2v),
                    velocity + dt * k3v)
        delta += dt * (velocity + dt / 6.0 * (k1v + k2v + k3v))
        velocity += dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        steps += 1
    raise RuntimeError("reference integration did not separate")


class RestitutionAnalyticsTests(unittest.TestCase):
    def test_unclamped_formula_matches_classic_value(self):
        # zeta = 0.1 -> exp(-0.1 pi / sqrt(0.99))
        damping = 0.2 * math.sqrt(STIFFNESS * MASS)
        self.assertAlmostEqual(damping_ratio(STIFFNESS, damping, MASS), 0.1, places=12)
        self.assertAlmostEqual(
            restitution_from_damping(STIFFNESS, damping, MASS),
            math.exp(-0.1 * math.pi / math.sqrt(0.99)),
            places=12,
        )

    def test_clamped_formula_matches_independent_rk4_integration(self):
        for damping in (1.0, 2.0, 5.0, 10.0):
            with self.subTest(damping=damping):
                closed = restitution_from_damping_clamped(STIFFNESS, damping, MASS)
                reference = _reference_clamped_restitution(STIFFNESS, damping, MASS)
                self.assertLess(abs(closed - reference) / reference, 1.0e-3)

    def test_clamped_restitution_exceeds_unclamped(self):
        # Removing the tensile tail can only add rebound velocity.
        for damping in (0.5, 2.0, 5.0, 10.0, 20.0):
            with self.subTest(damping=damping):
                self.assertGreater(
                    restitution_from_damping_clamped(STIFFNESS, damping, MASS),
                    restitution_from_damping(STIFFNESS, damping, MASS),
                )

    def test_bounds_and_monotonicity(self):
        self.assertEqual(restitution_from_damping_clamped(STIFFNESS, 0.0, MASS), 1.0)
        # critically damped contact cannot separate in the clamped model
        critical = 2.0 * math.sqrt(STIFFNESS * MASS)
        self.assertEqual(restitution_from_damping_clamped(STIFFNESS, critical, MASS), 0.0)
        values = [
            restitution_from_damping_clamped(STIFFNESS, c, MASS)
            for c in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0)
        ]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))

    def test_calibration_inverse_is_exact(self):
        for target in (0.9, 0.75, 0.6, 0.5, 0.4, 0.3, 0.2):
            with self.subTest(target=target):
                damping = damping_for_clamped_restitution(STIFFNESS, target, MASS)
                self.assertAlmostEqual(
                    restitution_from_damping_clamped(STIFFNESS, damping, MASS), target, places=9
                )

    def test_calibration_round_trip_unclamped(self):
        target = 0.65
        damping = damping_for_restitution_analytic(STIFFNESS, target, MASS)
        self.assertAlmostEqual(
            restitution_from_damping(STIFFNESS, damping, MASS), target, places=9
        )

    def test_invalid_inputs_rejected(self):
        for bad in (0.0, -0.1, 1.1, float("nan")):
            with self.subTest(restitution=bad):
                with self.assertRaises(ValueError):
                    damping_for_clamped_restitution(STIFFNESS, bad, MASS)
        with self.assertRaises(ValueError):
            restitution_from_damping(STIFFNESS, -1.0, MASS)

    def test_documented_default_restitutions(self):
        config = DEM3DConfig()
        self.assertAlmostEqual(contact_restitution(config), 0.63243, places=4)
        self.assertAlmostEqual(clamped_contact_restitution(config), 0.65972, places=4)
        # platens default to zero damping, i.e. a perfectly elastic contact
        self.assertEqual(platen_restitution(config), 1.0)


class RestitutionEngineTests(unittest.TestCase):
    """Discrete-element bounce probes against the closed-form benchmark."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_frictionless_platen_is_perfectly_elastic(self):
        measurement = measure_platen_restitution(damping=0.0)
        self.assertLess(measurement.relative_error_clamped, 1.0e-3)
        self.assertAlmostEqual(measurement.measured_restitution, 1.0, places=3)

    def test_measured_restitution_tracks_closed_forms(self):
        # Both closed forms bracket the measurement to a few percent; the
        # clamped form is the better predictor once the damping ratio exceeds
        # ~0.1 (see the companion test below).
        cases = ((1.0, 0.05), (2.0, 0.05), (5.0, 0.05), (10.0, 0.035))
        for damping, tolerance in cases:
            with self.subTest(damping=damping):
                measurement = measure_platen_restitution(damping=damping)
                self.assertEqual(measurement.particle_count, 8)
                self.assertLess(
                    measurement.relative_error_clamped,
                    tolerance,
                    msg=f"clamped closed form off by {100 * measurement.relative_error_clamped:.2f}%",
                )
                self.assertLess(measurement.relative_error, 0.35)

    def test_clamped_form_is_better_predictor_at_moderate_damping(self):
        for damping in (5.0, 10.0):
            with self.subTest(damping=damping):
                measurement = measure_platen_restitution(damping=damping)
                self.assertLess(
                    measurement.relative_error_clamped, measurement.relative_error
                )

    def test_unclamped_formula_fails_at_moderate_damping(self):
        # Documents why the clamped form is the calibration benchmark: the
        # textbook formula is off by ~16 % at c = 10 N.s/m.
        measurement = measure_platen_restitution(damping=10.0)
        self.assertGreater(measurement.relative_error, 0.10)
        self.assertLess(measurement.measured_restitution, measurement.analytic_restitution_clamped)

    def test_measured_restitution_is_velocity_independent(self):
        # A linear dashpot has no velocity-dependent restitution.
        slow = measure_platen_restitution(damping=5.0, impact_speed=0.1)
        fast = measure_platen_restitution(damping=5.0, impact_speed=1.0)
        self.assertLess(abs(slow.measured_restitution - fast.measured_restitution), 5.0e-3)


if __name__ == "__main__":
    unittest.main()