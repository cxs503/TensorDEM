"""Analytic verification tests for the tangential/normal contact models.

These tests close a verification debt: before this file the friction module was
only checked against per-formula algebra (saturation value, displacement
increment), never against a closed-form reference through time stepping.  Two
anchors are used:

* Anchor R - the coefficient-of-restitution mapping is verified by integrating
  the underlying damped impact and measuring the rebound ratio (target ``e``).
* Anchor T - the Cundall-Strack tangential spring is verified against the
  closed-form hysteresis loop area per stick-slip cycle,
  ``W = 4 mu F_n (a - mu F_n / k_t)``, plus an unclamped-state negative control
  that must (and does) dissipate nothing.

Tolerances are stated explicitly; the measured values are also written as
artifacts by ``examples/tangential_friction_verification.py``.
"""
from __future__ import annotations

import math
import unittest

import torch

from tensordem.contact_tangential import (
    compute_tangential_force,
    tangential_cycle_dissipated_work,
)
from tensordem.tangential_verification import (
    RestitutionCase,
    StickSlipCase,
    restitutions_to_damping_check,
    simulate_rebound,
    simulate_stick_slip,
)

# Verification tolerances (documented in docs/dem3d_tangential_friction_verification.md)
RESTITUTION_TOL = 5.0e-3      # 0.5 % on the rebound coefficient
FRICTION_TOL = 5.0e-3         # 0.5 % on the dissipated work per cycle
ZERO_TOL = 1.0e-9             # J, "no dissipation" band


class RestitutionAnchorTests(unittest.TestCase):
    """Anchor R: time-stepped rebound vs the target coefficient of restitution."""

    def test_measured_rebound_matches_target(self):
        for e in (0.1, 0.3, 0.5, 0.7, 0.9):
            with self.subTest(e=e):
                res = simulate_rebound(RestitutionCase(e_target=e, steps_per_period=4000))
                self.assertLess(
                    res["rel_error"], RESTITUTION_TOL,
                    msg=f"e_target={e}: e_measured={res['e_measured']}, "
                        f"rel_error={res['rel_error']:.3e}",
                )

    def test_elastic_limit_returns_full_velocity(self):
        res = simulate_rebound(RestitutionCase(e_target=1.0, steps_per_period=4000))
        self.assertEqual(res["gamma_n"], 0.0)
        self.assertLess(res["rel_error"], 1.0e-3)

    def test_error_decreases_with_time_step(self):
        coarse = simulate_rebound(RestitutionCase(e_target=0.5, steps_per_period=1000))
        fine = simulate_rebound(RestitutionCase(e_target=0.5, steps_per_period=8000))
        self.assertLess(fine["rel_error"], coarse["rel_error"])
        # first-order scheme: 8x refinement should give >= 3x error reduction
        self.assertGreater(coarse["rel_error"] / max(fine["rel_error"], 1e-12), 3.0)

    def test_mapping_roundtrip(self):
        for row in restitutions_to_damping_check():
            with self.subTest(e=row["e"]):
                self.assertLess(row["abs_error"], 1.0e-12)


class StickSlipAnchorTests(unittest.TestCase):
    """Anchor T: hysteresis loop area vs the closed-form dissipated work."""

    def test_loop_area_matches_closed_form(self):
        for a in (0.05, 0.1, 0.2, 0.4):
            with self.subTest(amplitude=a):
                res = simulate_stick_slip(
                    StickSlipCase(amplitude=a, steps_per_cycle=2000), clamp_state=True)
                self.assertGreater(res["work_analytic"], 0.0)
                self.assertLess(
                    res["rel_error"], FRICTION_TOL,
                    msg=f"a={a}: measured={res['work_measured']:.6f} J, "
                        f"analytic={res['work_analytic']:.6f} J",
                )

    def test_below_shakedown_amplitude_is_elastic(self):
        # a_stick = mu * F_n / k_t = 0.3 * 100 / 1000 = 0.03 m
        for a in (0.005, 0.02, 0.03):
            with self.subTest(amplitude=a):
                self.assertEqual(
                    tangential_cycle_dissipated_work(0.3, 100.0, 1000.0, a), 0.0)
                res = simulate_stick_slip(
                    StickSlipCase(amplitude=a, steps_per_cycle=2000), clamp_state=True)
                self.assertLess(abs(res["work_measured"]), ZERO_TOL)
                self.assertEqual(res["sliding_fraction"], 0.0)

    def test_sliding_fraction_increases_with_amplitude(self):
        fractions = [
            simulate_stick_slip(StickSlipCase(amplitude=a, steps_per_cycle=1000),
                                clamp_state=True)["sliding_fraction"]
            for a in (0.05, 0.1, 0.2)
        ]
        self.assertLess(fractions[0], fractions[1])
        self.assertLess(fractions[1], fractions[2])

    def test_convergence_with_steps_per_cycle(self):
        coarse = simulate_stick_slip(StickSlipCase(amplitude=0.2, steps_per_cycle=250),
                                     clamp_state=True)
        fine = simulate_stick_slip(StickSlipCase(amplitude=0.2, steps_per_cycle=2000),
                                   clamp_state=True)
        self.assertLess(fine["rel_error"], coarse["rel_error"])
        self.assertLess(fine["rel_error"], FRICTION_TOL)

    def test_unclamped_state_negative_control_has_no_hysteresis(self):
        """Storing the raw displacement removes the loop: W -> 0 (the bug that
        the state projection in compute_tangential_force prevents)."""
        for a in (0.05, 0.2):
            with self.subTest(amplitude=a):
                clamped = simulate_stick_slip(
                    StickSlipCase(amplitude=a, steps_per_cycle=1000), clamp_state=True)
                unclamped = simulate_stick_slip(
                    StickSlipCase(amplitude=a, steps_per_cycle=1000), clamp_state=False)
                self.assertGreater(clamped["work_measured"], 1.0)
                self.assertLess(abs(unclamped["work_measured"]), ZERO_TOL)


class TangentialStateProjectionTests(unittest.TestCase):
    """The returned state must be the clamped displacement F_t / k_t."""

    def _single(self, disp: float, mu: float, k_t: float = 1000.0, f_n: float = 100.0):
        return compute_tangential_force(
            torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            torch.tensor([[0.0, disp]], dtype=torch.float64),
            torch.tensor([f_n], dtype=torch.float64),
            k_t, mu,
        )

    def test_sticking_state_is_unchanged(self):
        f_t, state = self._single(disp=0.01, mu=0.3)  # 10 N < 30 N limit
        self.assertAlmostEqual(float(f_t[0, 1]), 10.0, places=9)
        self.assertAlmostEqual(float(state[0, 1]), 0.01, places=12)

    def test_sliding_state_is_projected_onto_cone(self):
        f_t, state = self._single(disp=0.5, mu=0.3)  # would be 500 N, limit 30 N
        self.assertAlmostEqual(float(f_t[0, 1]), 30.0, places=9)
        self.assertAlmostEqual(float(state[0, 1]), 0.03, places=12)
        self.assertAlmostEqual(
            float(state.norm(dim=1).item()), 0.3 * 100.0 / 1000.0, places=12)

    def test_force_and_state_are_consistent(self):
        f_t, state = self._single(disp=0.1, mu=0.3)
        torch.testing.assert_close(f_t, 1000.0 * state)

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            self._single(disp=0.01, mu=0.3, k_t=0.0)
        with self.assertRaises(ValueError):
            self._single(disp=0.01, mu=-1.0)
        with self.assertRaises(ValueError):
            tangential_cycle_dissipated_work(0.3, 100.0, 0.0, 0.1)

    def test_closed_form_monotone_in_amplitude(self):
        works = [tangential_cycle_dissipated_work(0.3, 100.0, 1000.0, a)
                 for a in (0.03, 0.05, 0.1, 0.2)]
        self.assertEqual(works[0], 0.0)
        self.assertTrue(all(b > a for a, b in zip(works[1:], works[2:])))
        self.assertAlmostEqual(works[-1], 20.4, places=12)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()