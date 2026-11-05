"""Verification gates for the three-field melting core (``threefield.thermal``).

Every assertion is graded against a reference that does not reuse the code path
under test: the closed-form melting time is checked against direct quadrature of
the melting ODE, the RK2 integrator against the closed form, and the ``B -> 0``
branch against the conduction law.  See ``docs/threefield-thermal-melting-
verification.md`` (produced by ``scripts/audit_threefield_thermal.py``).
"""
from __future__ import annotations

import math
import unittest

from scipy.integrate import quad

from tensordem.threefield.thermal import (
    ThermalProperties,
    _analytic_case_constants,
    analytic_conduction_melt_time,
    analytic_diameter_conduction,
    analytic_diameter_ransom_marshall,
    analytic_ransom_marshall_time,
    diameter_rate,
    integrate_melting,
)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


def _reference_time(props: ThermalProperties, d0: float, d: float, u: float,
                    include_sensible: bool) -> float:
    """Independent reference: quadrature of dt = ∫_D^{D0} dD/(A/D + B/√D)."""
    a, b = _analytic_case_constants(props, u, include_sensible=include_sensible)
    if b <= 0.0:
        return (d0 * d0 - d * d) / (2.0 * a)
    value, _err = quad(lambda x: 1.0 / (a / x + b / math.sqrt(x)), d, d0,
                       epsabs=1e-16, epsrel=1e-13, limit=400)
    return value


class ClosedFormQuadratureTests(unittest.TestCase):
    """``analytic_ransom_marshall_time`` vs independent quadrature."""

    def setUp(self) -> None:
        self.props = ThermalProperties()

    def test_matches_quadrature_over_velocity_decades(self) -> None:
        # includes the near-degenerate regime B*sqrt(D0)/A ~ 1e-8 where the
        # unfactored primitive used to lose every significant digit
        for include_sensible in (False, True):
            for u in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 0.05, 0.25, 0.5):
                for d0, frac in ((0.05, 0.5), (0.02, 0.1), (0.30, 0.9)):
                    d = d0 * frac
                    closed = analytic_ransom_marshall_time(
                        d0, d, u, self.props, include_sensible=include_sensible)
                    reference = _reference_time(
                        self.props, d0, d, u, include_sensible)
                    self.assertLess(
                        _rel(closed, reference), 1e-10,
                        f"u={u}, d0={d0}, d={d}, sensible={include_sensible}")

    def test_near_degenerate_regime_no_cancellation(self) -> None:
        """Regression guard for the pre-fix cancellation in the primitive.

        At u ~ 1e-16 the convective correction is O(1e-6) of the conduction law,
        so the conduction law is a poor reference here: the honest reference is
        the quadrature.  The unfactored primitive used to be wrong by 1e+11 at
        these velocities, so a 1e-9 gate is decisive.
        """
        d0, d = 0.02, 0.01
        for u in (1e-16, 1e-14, 1e-12, 1e-10):
            closed = analytic_ransom_marshall_time(
                d0, d, u, self.props, include_sensible=False)
            reference = _reference_time(self.props, d0, d, u, include_sensible=False)
            self.assertLess(_rel(closed, reference), 1e-9, f"u={u}")
        # monotone in u: faster flow must melt sooner, never later
        times = [
            analytic_ransom_marshall_time(d0, d, u, self.props, include_sensible=False)
            for u in (0.0, 1e-12, 1e-8, 1e-4, 0.05, 0.5)
        ]
        self.assertTrue(all(b <= a * (1.0 + 1e-12) for a, b in zip(times, times[1:])))

    def test_ode_consistency_sign_and_scaling(self) -> None:
        """d t/d D from the closed form must equal 1/(dD/dt)."""
        h = 1e-9
        for u in (0.05, 0.25, 0.5):
            for d0 in (0.05, 0.02):
                for frac in (0.9, 0.6, 0.3):
                    d = d0 * frac
                    dtdd = (
                        analytic_ransom_marshall_time(
                            d0, d + h, u, self.props, include_sensible=False)
                        - analytic_ransom_marshall_time(
                            d0, d - h, u, self.props, include_sensible=False)
                    ) / (2.0 * h)
                    rate = diameter_rate(d, u, self.props, include_sensible=False)
                    self.assertLess(rate, 0.0)
                    self.assertLess(_rel(dtdd, 1.0 / rate), 1e-6)

    def test_degenerate_limit_equals_conduction_law(self) -> None:
        for d0, frac in ((0.05, 0.5), (0.02, 0.25), (0.01, 0.9)):
            d = d0 * frac
            via_rm = analytic_ransom_marshall_time(
                d0, d, 0.0, self.props, include_sensible=False)
            via_law = (d0 * d0 - d * d) / (2.0 * _analytic_case_constants(
                self.props, 0.0, include_sensible=False)[0])
            self.assertLess(_rel(via_rm, via_law), 1e-14)

    def test_inversion_round_trip(self) -> None:
        d0 = 0.03
        for u in (0.0, 0.05, 0.25, 0.5):
            t_full = analytic_ransom_marshall_time(
                d0, d0 * 1e-9, u, self.props, include_sensible=False)
            for frac in (0.1, 0.5, 0.95):
                t = t_full * frac
                d = analytic_diameter_ransom_marshall(
                    d0, t, u, self.props, include_sensible=False)
                t_back = analytic_ransom_marshall_time(
                    d0, d, u, self.props, include_sensible=False)
                self.assertLess(_rel(t_back, t), 1e-10)


class IntegratorOrderTests(unittest.TestCase):
    """``integrate_melting`` is an RK2 midpoint scheme: second order in dt."""

    def setUp(self) -> None:
        self.props = ThermalProperties()

    def _order(self, u: float, include_sensible: bool) -> float:
        d0, frac = 0.03, 0.4
        t_end = analytic_ransom_marshall_time(
            d0, d0 * frac, u, self.props, include_sensible=include_sensible)
        steps = (40, 80, 160, 320, 640)
        dt = [t_end / n for n in steps]
        errors = []
        for n in steps:
            hist = integrate_melting(
                d0, self.props, relative_velocity_m_s=u, duration_s=t_end,
                dt_s=t_end / n, include_sensible=include_sensible)
            exact = analytic_diameter_ransom_marshall(
                d0, t_end, u, self.props, include_sensible=include_sensible)
            errors.append(abs(hist[-1]["diameter_m"] - exact))
        # least-squares slope in log-log space without numpy
        xs = [math.log(v) for v in dt]
        ys = [math.log(e) for e in errors]
        n_pts = len(xs)
        mx = sum(xs) / n_pts
        my = sum(ys) / n_pts
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        return num / den

    def test_second_order_ransom_marshall(self) -> None:
        self.assertAlmostEqual(self._order(0.25, False), 2.0, delta=0.15)

    def test_second_order_with_sensible_heat(self) -> None:
        self.assertAlmostEqual(self._order(0.25, True), 2.0, delta=0.15)

    def test_second_order_conduction_limit(self) -> None:
        self.assertAlmostEqual(self._order(0.0, False), 2.0, delta=0.15)

    def test_operational_timestep_accuracy(self) -> None:
        """CouplingConfig.melt_dt_s = 0.25: melted diameter must stay exact."""
        d0, frac = 0.03, 0.9
        for u in (0.0, 0.1, 0.25, 0.5):
            t_end = analytic_ransom_marshall_time(
                d0, d0 * frac, u, self.props, include_sensible=False)
            hist = integrate_melting(
                d0, self.props, relative_velocity_m_s=u, duration_s=t_end,
                dt_s=t_end / max(1, round(t_end / 0.25)), include_sensible=False)
            exact = analytic_diameter_ransom_marshall(
                d0, t_end, u, self.props, include_sensible=False)
            self.assertLess(_rel(hist[-1]["diameter_m"], exact), 1e-6)


class DisappearanceTests(unittest.TestCase):
    """A fragment that melts completely inside the horizon."""

    def setUp(self) -> None:
        self.props = ThermalProperties()

    def test_vanish_time_within_one_step_and_mass_conserved(self) -> None:
        d0, u = 1.0e-4, 0.5
        t_melt = analytic_ransom_marshall_time(
            d0, d0 * 1e-9, u, self.props, include_sensible=False)
        for dt in (0.02, 0.01, 0.005):
            hist = integrate_melting(
                d0, self.props, relative_velocity_m_s=u,
                duration_s=t_melt * 2.0, dt_s=dt, include_sensible=False)
            vanish = next(r for r in hist if r["diameter_m"] <= 0.0)
            self.assertLessEqual(abs(vanish["time_s"] - t_melt), dt * (1.0 + 1e-12))
            self.assertAlmostEqual(hist[-1]["melt_fraction"], 1.0, places=12)
            self.assertLess(
                _rel(hist[-1]["melted_mass_kg"], hist[0]["mass_kg"]), 1e-12)


if __name__ == "__main__":
    unittest.main()