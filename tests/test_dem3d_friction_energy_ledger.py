"""Closed-form verification tests for the friction channel of the DEM3D ledger.

The energy ledger (``tensordem.dem3d_energy``) used to accumulate drag, normal
contact damping, tool damping and fracture release but no tangential friction
term, so with ``friction_coefficient > 0`` the entire friction work leaked
into ``energy_balance_residual_J``.  These tests pin the restored term against
the independent closed-form reference of
``scripts/verify_dem3d_friction_energy_ledger.py`` (two-sphere stick-slip
subsystem driven through the production ``step()``):

* stick phase:  ``D_f(T) = 0.5 k_t x(T)^2`` (exact harmonic solution);
* slip phase:   ``D_f(T) = int F_t v_t dt`` on the RK4 reference trajectory;
* balance:      the mu > 0 residual sits at the mu = 0 discretisation floor,
                while the pre-fix bookkeeping residual equals the friction work;
* convergence:  first-order in dt;
* bookkeeping:  probing never mutates the ledger, checkpoints round-trip the
                new term and old checkpoints (without it) still load.
"""
from __future__ import annotations

import math
import unittest

import torch

from scripts.verify_dem3d_friction_energy_ledger import (
    DT_BASE,
    K_CONTACT,
    MU_SLIP,
    build_friction_subsystem,
    convergence_sweep,
    mass,
    normal_force,
    run_balance,
    run_case,
    slip_onset,
    tracking_external_force,
)
from tensordem.dem3d import DEM3DConfig

TOLERANCE = 0.03

_CACHE: dict[str, object] = {}


def _stick():
    if "stick" not in _CACHE:
        omega = math.sqrt(K_CONTACT / 0.48014)  # mass() at RADIUS = 0.05
        _CACHE["stick"] = run_case(10.0, DT_BASE, 0.25 * 2.0 * math.pi / omega)
    return _CACHE["stick"]


def _slip():
    if "slip" not in _CACHE:
        t_s, x_s, v_s = slip_onset(MU_SLIP)
        t_end = 0.9 * (t_s + v_s * mass() / (MU_SLIP * normal_force(x_s)))
        _CACHE["slip"] = run_case(MU_SLIP, DT_BASE, t_end)
    return _CACHE["slip"]


def _balance():
    if "balance" not in _CACHE:
        _CACHE["balance"] = run_balance(DT_BASE)
    return _CACHE["balance"]


def _sweep():
    if "sweep" not in _CACHE:
        _CACHE["sweep"] = convergence_sweep()
    return _CACHE["sweep"]


class StickClosedFormTests(unittest.TestCase):
    """Stick regime: the ledger must reproduce 0.5 k_t x(T)^2."""

    def test_dissipation_matches_rk4_reference(self):
        case = _stick()
        self.assertLess(case["friction_rel_error"], TOLERANCE,
                        f"D_f error {case['friction_rel_error']:.4%}")

    def test_dissipation_matches_exact_harmonic_solution(self):
        case = _stick()
        self.assertLess(case["stick_analytic_rel_error"], TOLERANCE,
                        f"analytic error {case['stick_analytic_rel_error']:.4%}")

    def test_travel_matches_reference(self):
        case = _stick()
        self.assertLess(case["travel_rel_error"], TOLERANCE)


class SlipClosedFormTests(unittest.TestCase):
    """Stick-slip regime: the ledger must reproduce the reference work."""

    def test_dissipation_matches_reference(self):
        case = _slip()
        self.assertLess(case["friction_rel_error"], TOLERANCE,
                        f"D_f error {case['friction_rel_error']:.4%}")

    def test_travel_matches_reference(self):
        case = _slip()
        self.assertLess(case["travel_rel_error"], TOLERANCE)

    def test_dissipation_is_positive_and_bounded_by_work_budget(self):
        case = _slip()
        self.assertGreater(case["friction_dissipation_J"], 0.0)
        # friction work cannot exceed the initial kinetic energy plus the
        # (tiny) external work of the tracking load
        self.assertLess(case["friction_dissipation_J"],
                        1.5 * 0.5 * 0.48014 * 0.1**2
                        + case["external_work_J"])


class BalanceClosureTests(unittest.TestCase):
    """The friction term must close the balance to the discretisation floor."""

    def test_residual_sits_at_control_floor(self):
        bal = _balance()
        self.assertLess(bal["residual_new_relative"], 0.01,
                        f"residual {bal['residual_new_relative']:.4%}")

    def test_prefix_residual_explains_friction_work(self):
        bal = _balance()
        self.assertTrue(bal["prefix_residual_explains_friction_work"],
                        f"pre-fix residual {bal['residual_prefix_J']:.3e} J "
                        f"vs friction work {bal['friction_work_J']:.3e} J")


class ConvergenceTests(unittest.TestCase):
    def test_first_order_convergence(self):
        rows = _sweep()
        self.assertEqual(len(rows), 3)
        for coarse, fine in zip(rows, rows[1:]):
            self.assertGreater(coarse["friction_rel_error"], 0.0)
            ratio = coarse["friction_rel_error"] / fine["friction_rel_error"]
            self.assertGreaterEqual(ratio, 1.5,
                                  f"no convergence: {coarse['dt']} -> {fine['dt']}")


class BookkeepingTests(unittest.TestCase):
    def test_probe_does_not_mutate_ledger_or_trajectory(self):
        solver = build_friction_subsystem(DEM3DConfig(dt=DT_BASE), MU_SLIP)
        for _ in range(50):
            solver.step(tracking_external_force(solver))
        d_before = solver.cumulative_friction_dissipation_J
        pos_before = solver.positions.clone()
        solver.diagnostics()  # probes forces with update_history=False
        self.assertEqual(solver.cumulative_friction_dissipation_J, d_before)
        for _ in range(50):
            solver.step(tracking_external_force(solver))
        self.assertTrue(torch.isfinite(solver.positions).all())
        self.assertGreater(solver.cumulative_friction_dissipation_J, d_before)

    def test_checkpoint_roundtrip_preserves_friction_term(self):
        solver = build_friction_subsystem(DEM3DConfig(dt=DT_BASE), MU_SLIP)
        for _ in range(100):
            solver.step(tracking_external_force(solver))
        state = solver.state_dict()
        d_saved = state["energy_audit"]["cumulative_friction_dissipation_J"]
        self.assertGreater(d_saved, 0.0)
        solver2 = build_friction_subsystem(DEM3DConfig(dt=DT_BASE), MU_SLIP)
        solver2.load_state_dict(state)
        self.assertEqual(solver2.cumulative_friction_dissipation_J, d_saved)

    def test_legacy_checkpoint_without_friction_key_still_loads(self):
        solver = build_friction_subsystem(DEM3DConfig(dt=DT_BASE), MU_SLIP)
        for _ in range(100):
            solver.step(tracking_external_force(solver))
        state = solver.state_dict()
        del state["energy_audit"]["cumulative_friction_dissipation_J"]
        solver2 = build_friction_subsystem(DEM3DConfig(dt=DT_BASE), MU_SLIP)
        solver2.load_state_dict(state)
        self.assertEqual(solver2.cumulative_friction_dissipation_J, 0.0)


if __name__ == "__main__":
    unittest.main()
