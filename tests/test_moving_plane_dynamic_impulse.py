"""Machine-checked contract for the closed-form *dynamic* moving-plane impulse.

The published study (``docs/dem3d_moving_plane_ledger_verification.md``) left the
dynamic ``tool_impulse`` ledger as **WITHHELD** ("no independent ``int F dt``").
This module pins the closure built by
``scripts/verify_moving_plane_dynamic_impulse.py``:

* the reduced reference model *is* the engine's force law (identity),
* the closed-form impulse reproduces an independent RK4 integration of the same
  force law (Richardson-refined) on the ramp window and over the whole
  ramp+hold window,
* the engine's incremental ledger ``-mean_force * dt`` reproduces that closed
  form to well under a part in a million, at every substep count tested,
* the one-sided contact never opens (so the frozen-active-set derivation is the
  model actually being integrated).

Tolerances are one to two decades wider than the measured residuals: the
contract fails on a regression, not on floating-point noise.
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import verify_moving_plane_dynamic_impulse as study  # noqa: E402
from tensordem.moving_plane_boundary import MovingPlaneNetwork  # noqa: E402

T_RAMP = 0.4995
# the engine advances a whole number of its own steps, so its final time is on
# that grid (1275 x dt), not at the nominal 0.55 s
_T_NOMINAL = study.RAMP + study.HOLD
T_FULL = int(round(_T_NOMINAL / MovingPlaneNetwork(8, 2).dt)) * MovingPlaneNetwork(8, 2).dt


class ReferenceModelTests(unittest.TestCase):
    def test_assembled_penalty_matches_engine_external_force(self):
        chk = study.force_identity(MovingPlaneNetwork(8, 2))
        self.assertGreater(chk["scale"], 1.0)
        self.assertLess(chk["rel"], 1e-14)

    def test_plane_kinematics_smooth_plane_derivative(self):
        A = study.AnalyticImpulse()
        h = 1e-6
        for t in (0.05, 0.2, 0.35, 0.45):
            num = (A.s(t + h) - A.s(t - h)) / (2 * h)
            self.assertAlmostEqual(num, A.sdot(t), delta=1e-9 + 1e-6 * abs(num))

    def test_plane_integrals_match_fine_quadrature(self):
        A = study.AnalyticImpulse()
        for T in (0.2, 0.5, T_FULL):
            ts = np.linspace(0.0, T, 200001)
            num = np.trapezoid([A.s(t) for t in ts], ts)
            self.assertAlmostEqual(A._s_dot_integral(T) / num, 1.0, delta=1e-9)

    def test_polynomial_exponential_integral(self):
        A = study.AnalyticImpulse()
        om = A.omega[3]
        ts = np.linspace(0.0, 0.3, 400001)
        poly = np.polyval(A.sdd_coeffs[::-1], ts)
        num = np.trapezoid(poly * np.exp(-1j * om * ts), ts)
        self.assertLess(abs(A._H(om, 0.3) - num) / abs(num), 1e-8)


class ClosedFormTests(unittest.TestCase):
    def setUp(self):
        self.A = study.AnalyticImpulse()

    def test_closed_form_matches_independent_rk4_on_ramp(self):
        ref = study.rk4_reference(self.A, self.A.net.dt / 32, T_RAMP)
        got = self.A.impulse(T_RAMP)
        self.assertLess(abs(got - ref["impulse"]) / abs(ref["impulse"]), 1e-4)

    def test_closed_form_matches_independent_rk4_over_hold(self):
        ref = study.rk4_reference(self.A, self.A.net.dt / 32, T_FULL)
        got = self.A.impulse(T_FULL)
        self.assertLess(abs(got - ref["impulse"]) / abs(ref["impulse"]), 1e-6)

    def test_contact_never_opens(self):
        ref = study.rk4_reference(self.A, self.A.net.dt / 32, T_FULL)
        self.assertEqual(ref["open_fraction"], 0.0)
        self.assertGreater(ref["min_gap"], 0.0)


class EngineLedgerTests(unittest.TestCase):
    def test_engine_dynamic_impulse_matches_closed_form(self):
        A = study.AnalyticImpulse()
        ref = A.impulse(T_FULL)
        for sub in (1, 2, 4):
            r = study.run_engine(A.net.dt, sub)
            self.assertLess(abs(r["tool_impulse"] - ref) / abs(ref), 1e-5,
                            msg=f"substeps={sub}")

    def test_engine_impulse_is_negative_of_tool_reaction(self):
        A = study.AnalyticImpulse()
        net = MovingPlaneNetwork(8, 2)
        total = 0.0
        n = 200
        for i in range(n):
            t0, t1 = i * net.dt, (i + 1) * net.dt
            a = study.smooth_plane(t0)
            b = study.smooth_plane(t1)
            total += net.moving_step(net.dt, a, b) * net.dt
        # moving_step returns -mean_force and tool_impulse accumulates -mean_force*dt
        self.assertAlmostEqual(total, net.tool_impulse,
                               delta=1e-12 + 1e-9 * abs(total))


if __name__ == "__main__":
    unittest.main()