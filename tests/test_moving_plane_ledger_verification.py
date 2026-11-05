"""Machine-checked contract for the closed-form moving-plane ledger study.

The reference itself is a linear solve of the network's own quadratic potential
(``(K + P) u = -P (x - plane)``), built from the module's own ``edges``/``k``
/``n``/``penalty``/``areas`` arrays.  These tests pin:

* the reference *is* the engine's model (Jacobian identity to 1e-12),
* the penalty law / tributary-area / penalty-energy algebra (exact),
* the static plane force is affine in penetration, so the work has the closed
  form ``0.5 F(delta) delta`` and equals the stored energy,
* the dynamic ``tool_work`` ledger converges to that closed form in the
  quasi-static limit and satisfies the energy-balance identity at every ramp
  speed tested,
* the work and impulse quadratures are exact for a frozen network.

Numbers are those produced by ``scripts/verify_moving_plane_ledger.py``; the
tolerances are one to two decades wider than the measured residuals so the
contract fails on a regression, not on floating-point noise.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

import verify_moving_plane_ledger as study  # noqa: E402
from tensordem.moving_plane_boundary import MovingPlaneNetwork  # noqa: E402


class ReferenceModelTests(unittest.TestCase):
    def setUp(self):
        self.net = MovingPlaneNetwork(8, 2)

    def test_assembled_stiffness_matches_engine_force(self):
        check = study.jacobian_check(MovingPlaneNetwork(8, 2))
        self.assertGreater(check["scale"], 1e-3)
        self.assertLess(check["rel"], 1e-12)

    def test_tributary_areas_and_penalty_law(self):
        net = MovingPlaneNetwork(8, 2)
        self.assertAlmostEqual(float(net.areas.sum()),
                               net.config["height"] * net.config["thickness"],
                               places=15)
        # hydrostatic-style equilibrium: sum(P_i A_i) clamping the left column
        plane = net.config["length"] - 3.0e-4
        f, ext, _, u_pen = net.forces(plane=plane)
        gap = (net.x[net.right, 0] + net.u[net.right, 0] - plane).clamp_min(0)
        pred = -(net.penalty * net.areas * gap)
        self.assertLess(float(((ext[net.right, 0] - pred).abs()
                               / pred.abs()).max()), 1e-12)
        self.assertEqual(int((gap > 0).sum()), len(net.right))
        self.assertEqual(float(ext[:, 1].abs().max()), 0.0)
        self.assertAlmostEqual(
            u_pen, float(0.5 * (net.penalty * net.areas * gap**2).sum()),
            places=15)

    def test_static_force_is_affine_and_work_equals_energy(self):
        net = MovingPlaneNetwork(8, 2)
        deltas = [1e-5, 1e-4, 5e-4, 1e-3]
        sols = [study.static_solution(net, d) for d in deltas]
        slopes = [s["force"] / s["delta"] for s in sols]
        self.assertLess((max(slopes) - min(slopes)) / max(slopes), 1e-8)
        self.assertEqual(study.static_solution(net, 0.0)["force"], 0.0)
        # thermodynamic consistency of the reference: dU/d(delta) == F
        for lo, hi in zip(sols[:-1], sols[1:]):
            dd = hi["delta"] - lo["delta"]
            f_mid = 0.5 * (hi["force"] + lo["force"])
            self.assertLess(abs((hi["energy"] - lo["energy"]) / dd - f_mid)
                            / f_mid, 1e-8)
        # closed-form work of the affine law vs the stored elastic energy
        s = sols[-1]
        w = 0.5 * s["force"] * s["delta"]
        self.assertLess(abs(w - s["energy"]) / s["energy"], 1e-9)


class DynamicLedgerTests(unittest.TestCase):
    def setUp(self):
        self.net = MovingPlaneNetwork(8, 2)
        self.w_exact = 0.5 * study.static_solution(
            MovingPlaneNetwork(8, 2), study.DEPTH)["force"] * study.DEPTH

    def test_quasi_static_tool_work_matches_closed_form(self):
        slow = study.run_ramp(8, 2, 1.0, 0.5 * self.net.dt)
        slow_err = abs(slow["tool_work"] - self.w_exact) / self.w_exact
        self.assertLess(slow_err, 1e-4)
        fast = study.run_ramp(8, 2, 0.05, 0.5 * self.net.dt)
        fast_err = abs(fast["tool_work"] - self.w_exact) / self.w_exact
        # a 20x faster ramp must carry a visibly larger ring-in error
        self.assertGreater(fast_err, 10.0 * slow_err)

    def test_energy_balance_identity(self):
        r = study.run_ramp(8, 2, 0.5, 0.5 * self.net.dt)
        balance = abs(r["tool_work"] - r["energy_total"]) / self.w_exact
        self.assertLess(balance, 1e-6)
        self.assertGreater(r["tool_work"], 0.0)

    def test_time_step_refinement_reduces_balance_residual(self):
        coarse = study.run_ramp(8, 2, 0.5, self.net.dt, substeps=1)
        fine = study.run_ramp(8, 2, 0.5, self.net.dt, substeps=4)
        c = abs(coarse["tool_work"] - coarse["energy_total"])
        f = abs(fine["tool_work"] - fine["energy_total"])
        self.assertLess(f, c)
        self.assertLess(f / self.w_exact, 1e-8)

    def test_frozen_network_work_and_impulse_are_exact(self):
        frozen = MovingPlaneNetwork(8, 2)
        frozen.fixed[:] = True
        frozen.v.zero_()
        plane0 = frozen.config["length"] - 2.0e-4
        steps, dpl = 100, 1.0e-7
        for i in range(steps):
            frozen.moving_step(frozen.dt, plane0 - i * dpl,
                               plane0 - (i + 1) * dpl)
        plane1 = plane0 - steps * dpl
        a_tot = frozen.penalty * float(frozen.areas.sum())
        f0 = a_tot * (frozen.config["length"] - plane0)
        f1 = a_tot * (frozen.config["length"] - plane1)
        w = 0.5 * (f0 + f1) * (plane0 - plane1)
        imp = 0.5 * (f0 + f1) * steps * frozen.dt
        self.assertLess(abs(frozen.tool_work - w) / w, 1e-12)
        self.assertLess(abs(frozen.tool_impulse - imp) / imp, 1e-12)


if __name__ == "__main__":
    unittest.main()