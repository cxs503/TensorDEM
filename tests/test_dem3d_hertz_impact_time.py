"""Time-domain verification of the DEM3D Hertz contact law (m4747).

The static Hertz anchors (``tests/test_dem3d_hertz_contact.py``) constrain
``F(delta)``; the test below constrains the *trajectory* by comparing the contact
duration produced by the shipped solver with the classical Johnson (1985, Sec.
4.2) closed form

    t_c = 2 I (5 m / (4 k_h))^(2/5) v0^(-1/5),   I = (2/5) B(2/5, 1/2),
    C = 2 I = 2.9433  (textbook 2.94).

Two anchors are asserted at once, because either alone is weak:

* the absolute prefactor ``C`` and the absolute duration ``t_c`` (a calibration
  check), and
* the exponent ``-1/5`` in ``t_c ~ v0^(-1/5)``, with a **negative control**: the
  same measurement rig driving a *linear* unilateral contact must return
  ``pi sqrt(m/k)`` with slope zero.  Without the control a slope measurement on
  its own only proves that *some* nonlinearity is present.

Every tolerance below is a physical acceptance criterion (the platform standard
is 15 %; this verification holds itself to 1 % on duration and 1e-3 on
exponents).  The reference relations are re-derived in
``tensordem.hertz_impact_time`` from textbook formulas, not imported from
``tensordem.dem3d``.
"""
from __future__ import annotations

import math
import unittest

from scipy.integrate import quad

from tensordem.hertz_impact_time import (
    JOHNSON_PREFACTOR,
    fit_loglog_exponent,
    hertz_contact_time,
    hertz_contact_time_integral,
    hertz_peak_overlap,
    linear_contact_time,
    measure_platen_impact,
    rigid_plane_hertz_stiffness,
    sweep_speeds,
)

RADIUS = 0.025
DENSITY = 917.0
YOUNG = 1.0e9
POISSON = 0.3
PLATEN_K = 5_000.0
STEPS = 600
TEXTBOOK_PREFACTOR = 2.94

MASS = DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3
K_H = rigid_plane_hertz_stiffness(YOUNG, POISSON, RADIUS)

_CACHE: dict[str, list[dict]] = {}


def sweep(model: str, speeds: tuple[float, ...]) -> list[dict]:
    """Measure once per process; tests share the (CPU-bound) campaign."""
    key = f"{model}:{','.join(str(s) for s in speeds)}"
    if key not in _CACHE:
        _CACHE[key] = sweep_speeds(speeds, contact_model=model,
                                   steps_per_contact=STEPS)
    return _CACHE[key]


class ClosedFormTests(unittest.TestCase):
    def test_contact_time_integral_matches_numerical_quadrature(self) -> None:
        reference, _ = quad(lambda u: 1.0 / math.sqrt(1.0 - u ** 2.5), 0.0, 1.0)
        self.assertLess(abs(hertz_contact_time_integral() / reference - 1.0), 1e-9)

    def test_prefactor_matches_textbook_value(self) -> None:
        self.assertLess(abs(JOHNSON_PREFACTOR / TEXTBOOK_PREFACTOR - 1.0), 2e-3)
        self.assertAlmostEqual(JOHNSON_PREFACTOR, 2.943275, places=5)

    def test_rigid_plane_stiffness_uses_single_body_modulus(self) -> None:
        # Only the particle deforms on a rigid platen: E* = E / (1 - nu^2).
        expected = (4.0 / 3.0) * (YOUNG / (1.0 - POISSON ** 2)) * math.sqrt(RADIUS)
        self.assertLess(abs(K_H / expected - 1.0), 1e-15)
        # A two-body E* = E / (2 (1 - nu^2)) would be exactly half as stiff.
        two_body = (4.0 / 3.0) * (YOUNG / (2.0 * (1.0 - POISSON ** 2))) * math.sqrt(RADIUS)
        self.assertAlmostEqual(K_H / two_body, 2.0, places=12)

    def test_peak_overlap_and_duration_scale_as_textbook_powers(self) -> None:
        speeds = [0.25, 0.5, 1.0, 2.0]
        overlaps = [hertz_peak_overlap(v, MASS, K_H) for v in speeds]
        durations = [hertz_contact_time(v, MASS, K_H) for v in speeds]
        self.assertAlmostEqual(fit_loglog_exponent(speeds, overlaps), 0.8, places=10)
        self.assertAlmostEqual(fit_loglog_exponent(speeds, durations), -0.2, places=10)

    def test_linear_contact_time_is_velocity_independent(self) -> None:
        self.assertAlmostEqual(linear_contact_time(MASS, PLATEN_K),
                               math.pi * math.sqrt(MASS / PLATEN_K), places=15)


class SolverDurationTests(unittest.TestCase):
    """Anchors B/C/E/F: the shipped solver against the closed form."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.hertz = sweep("hertz", (0.25, 0.5, 1.0, 2.0))
        cls.linear = sweep("linear", (0.5, 1.0))

    def test_hertz_duration_matches_johnson_within_one_percent(self) -> None:
        for row in self.hertz:
            with self.subTest(v_in=row["v_in"]):
                self.assertLess(abs(row["duration_rel_error"]), 0.01)
                # The residual bias must be bounded by one integration step.
                self.assertLessEqual(
                    abs(row["duration_rel_error"]),
                    1.0 / row["contact_steps"] + 1e-9,
                )

    def test_hertz_duration_exponent_is_minus_one_fifth(self) -> None:
        slope = fit_loglog_exponent([r["v_in"] for r in self.hertz],
                                    [r["measured_duration_s"] for r in self.hertz])
        self.assertLess(abs(slope + 0.2), 1e-3)

    def test_hertz_peak_overlap_matches_and_has_four_fifths_power(self) -> None:
        for row in self.hertz:
            with self.subTest(v_in=row["v_in"]):
                self.assertLess(abs(row["peak_overlap_rel_error"]), 1e-3)
        slope = fit_loglog_exponent([r["v_in"] for r in self.hertz],
                                    [r["peak_overlap_m"] for r in self.hertz])
        self.assertLess(abs(slope - 0.8), 1e-3)

    def test_undamped_hertz_impact_is_perfectly_elastic(self) -> None:
        for row in self.hertz:
            with self.subTest(v_in=row["v_in"]):
                self.assertLess(abs(row["restitution_measured"] - 1.0), 1e-9)

    def test_linear_control_reproduces_pi_sqrt_m_over_k(self) -> None:
        reference = linear_contact_time(MASS, PLATEN_K)
        for row in self.linear:
            with self.subTest(v_in=row["v_in"]):
                self.assertLess(abs(row["measured_duration_s"] / reference - 1.0), 0.01)
        slope = fit_loglog_exponent([r["v_in"] for r in self.linear],
                                    [r["measured_duration_s"] for r in self.linear])
        self.assertLess(abs(slope), 1e-6, "linear contact duration must not depend on speed")

    def test_exponent_separates_hertz_from_linear_control(self) -> None:
        hertz_slope = fit_loglog_exponent([r["v_in"] for r in self.hertz],
                                          [r["measured_duration_s"] for r in self.hertz])
        linear_slope = fit_loglog_exponent([r["v_in"] for r in self.linear],
                                           [r["measured_duration_s"] for r in self.linear])
        self.assertGreater(abs(hertz_slope - linear_slope), 0.15)

    def test_raw_bias_is_one_step_and_halves_with_the_step(self) -> None:
        t_c_ref = hertz_contact_time(1.0, MASS, K_H)
        coarse = measure_platen_impact(v_in=1.0, dt=t_c_ref / 300)
        fine = measure_platen_impact(v_in=1.0, dt=t_c_ref / 600)
        bias_coarse = coarse["measured_duration_s"] / t_c_ref - 1.0
        bias_fine = fine["measured_duration_s"] / t_c_ref - 1.0
        self.assertLess(bias_coarse, 0.0)   # quantisation always undershoots
        self.assertLess(bias_fine, 0.0)
        self.assertAlmostEqual(bias_coarse / bias_fine, 2.0, delta=0.05)


class GuardTests(unittest.TestCase):
    def test_dt_above_stability_bound_is_rejected(self) -> None:
        from tensordem.hertz_impact_time import _impact_base_config
        bound = _impact_base_config(
            radius=RADIUS, density=DENSITY, contact_model="hertz",
            young_modulus=YOUNG, poisson_ratio=POISSON,
            platen_stiffness=PLATEN_K).recommended_dt
        with self.assertRaises(ValueError):
            measure_platen_impact(v_in=1.0, dt=1.5 * bound)

    def test_invalid_inputs_rejected(self) -> None:
        with self.assertRaises(ValueError):
            hertz_peak_overlap(0.0, MASS, K_H)
        with self.assertRaises(ValueError):
            hertz_peak_overlap(1.0, MASS, 0.0)
        with self.assertRaises(ValueError):
            rigid_plane_hertz_stiffness(-1.0, POISSON, RADIUS)
        with self.assertRaises(ValueError):
            rigid_plane_hertz_stiffness(YOUNG, 0.5, RADIUS)
        with self.assertRaises(ValueError):
            fit_loglog_exponent([1.0], [1.0])
        with self.assertRaises(ValueError):
            fit_loglog_exponent([2.0, 2.0], [1.0, 3.0])


if __name__ == "__main__":
    unittest.main()