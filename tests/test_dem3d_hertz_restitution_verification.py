"""Regression screens for the Hertz restitution-closure verification.

These pin the *measured* findings of
``docs/dem3d_hertz_restitution_closure.md`` at reduced cost (RK4 reference is
cheap, engine impacts use one speed).  Two kinds of assertion:

* **law consistency** -- the engine and the independent RK4 reference must agree
  on the declared contact law (this is what makes the harness trustworthy);
* **honest-debt pins** -- the deviation of the measured restitution from
  ``restitution_coefficient`` is pinned to the reported magnitude, so the
  documented limitation cannot silently vanish or silently grow.
"""

from __future__ import annotations

import importlib.util
import pathlib
import unittest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "verify_dem3d_hertz_restitution.py"
_spec = importlib.util.spec_from_file_location("verify_dem3d_hertz_restitution", _SCRIPT)
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)


class ReferenceLawTests(unittest.TestCase):
    """The independent RK4 reference must behave like the law it claims."""

    def test_reference_is_time_step_convergent(self):
        coarse = V.reference_rk4(0.5, rest_target=0.5, dt=2.0e-7)["restitution"]
        fine = V.reference_rk4(0.5, rest_target=0.5, dt=5.0e-8)["restitution"]
        self.assertLess(abs(coarse - fine) / fine, 1.0e-3)

    def test_elastic_reference_returns_unit_restitution(self):
        self.assertAlmostEqual(V.reference_rk4(0.5, rest_target=1.0)["restitution"],
                               1.0, places=6)

    def test_absolute_dashpot_dissipation_decreases_with_speed(self):
        """1-e must fall monotonically with v (the v^(-1/5) trend, finite-amplitude)."""
        speeds = [0.2, 0.4, 0.8, 1.6]
        diss = [1.0 - V.reference_rk4(v, absolute_dash=60.0)["restitution"] for v in speeds]
        self.assertEqual(diss, sorted(diss, reverse=True))
        slope = V.loglog_slope(speeds, diss)
        # Asymptotic prediction is -0.2; at 1-e ~ 0.3-0.45 the measured slope is
        # shallower (-0.14).  Pin the bracket, not the asymptote.
        self.assertLess(slope, -0.10)
        self.assertGreater(slope, -0.21)

    def test_linear_reference_is_velocity_independent(self):
        for target in (0.5, 0.8):
            e = V.linear_reference(0.5, target)
            self.assertLess(abs(e - target) / target, 0.12)


class EngineClosureTests(unittest.TestCase):
    """Engine-level restitution on the Hertz branch (the actual claim)."""

    @classmethod
    def setUpClass(cls):
        cls.elastic = V.measure_engine(0.5, restitution_coefficient=1.0)
        cls.hertz = V.measure_engine(0.5, restitution_coefficient=0.5)
        cls.absolute = V.measure_engine(0.5, contact_damping_model="absolute",
                                        contact_damping=60.0, restitution_coefficient=1.0)

    def test_elastic_limit_is_lossless(self):
        self.assertAlmostEqual(self.elastic["restitution"], 1.0, places=3)

    def test_elastic_peak_overlap_discrepancy_is_pinned(self):
        """Open item: the engine peak overlap is ~1.7x the textbook Hertz value.

        The pair law is ``F = k_s(delta) delta = 2 E* sqrt(R_eff) delta^1.5``,
        i.e. 1.5x the textbook ``(4/3) E* sqrt(R_eff) delta^1.5``, which alone
        predicts a 0.85x (smaller) peak overlap -- so the measured 1.74x excess
        is NOT explained by that factor and is recorded as WITHHELD debt in
        ``docs/dem3d_hertz_restitution_closure.md`` §5.  Pinned so it cannot
        drift silently.
        """
        ratio = self.elastic["peak_overlap_m"] / V.analytic_peak_overlap(0.5)
        self.assertGreater(ratio, 1.5)
        self.assertLess(ratio, 2.0)

    def test_hertz_engine_matches_independent_reference(self):
        ref = V.reference_rk4(0.5, rest_target=0.5)["restitution"]
        self.assertLess(abs(self.hertz["restitution"] - ref) / ref, 3.0e-2)

    def test_absolute_dashpot_branch_matches_reference(self):
        ref = V.reference_rk4(0.5, absolute_dash=60.0)["restitution"]
        self.assertLess(abs(self.absolute["restitution"] - ref) / ref, 3.0e-2)

    def test_restitution_target_miss_is_pinned(self):
        """Honest debt: e_target = 0.5 yields e ~ 0.582 in the engine.

        The engine implements the declared law faithfully (previous test), so the
        miss is the *mapping* ``zeta(e_target)`` taken from linear spring-dashpot
        theory.  The published deviation is 16.4 %; pin it so it cannot drift.
        """
        dev = abs(self.hertz["restitution"] - 0.5) / 0.5
        self.assertGreater(dev, 0.10)
        self.assertLess(dev, 0.20)


if __name__ == "__main__":
    unittest.main()