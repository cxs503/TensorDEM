"""Closed-form verification of the DEM3D cumulative energy ledger.

The ledger in ``tensordem.dem3d_energy`` had only self-consistency tests before
this module (finite/non-negative terms, checkpoint reproducibilty, benchmark
repeatability).  These tests score the accumulated terms against analytical
solutions of two exactly solvable subsystems that are driven through the
production ``step()``:

* case A - single free particle with linear drag
  ``v(t) = v0 exp(-t/tau)``, ``D(t) = 0.5 m v0^2 (1 - exp(-2t/tau))``;
* case B - two particles joined by one elastic bond
  ``xi(t) = A cos(omega t)``, ``omega = sqrt(2k/m)``.

They also pin the two discretisation facts the verification established:
the ledger residual is a first-order truncation of the drag rate (predictable
in closed form), and the symplectic-Euler energy oscillation of case B has
relative amplitude ``dt * omega`` with no secular drift.

Finally ``ShearInvariantProbeTests`` pins the strain measure used by the
fracture criterion: exact on the full cubic lattice, and zero on
rank-deficient neighbourhoods (a single bond, i.e. collinear references),
where the least-squares deformation gradient is a projection and the
Green-Lagrange invariant would otherwise be a spurious 1/3 for a rigid,
undeformed configuration. The shear criterion is skipped there so thin/linear
specimens do not break on the first step.
"""
from __future__ import annotations

import math
import unittest

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D

from scripts.verify_dem3d_energy_ledger_analytic import (
    BOND_STIFFNESS,
    DRAG,
    OSC_AMPLITUDE,
    OSC_PERIODS,
    RADIUS,
    base_config,
    build_subsystem,
    convergence_sweep,
    probe_shear_invariant,
    run_drag_decay,
    run_oscillator,
)


_CACHE: dict[str, object] = {}


def _decay():
    if "decay" not in _CACHE:
        _CACHE["decay"] = run_drag_decay(base_config().recommended_dt)
    return _CACHE["decay"]


def _oscillator():
    if "osc" not in _CACHE:
        _CACHE["osc"] = run_oscillator(base_config().recommended_dt)
    return _CACHE["osc"]


def _sweep():
    if "sweep" not in _CACHE:
        _CACHE["sweep"] = convergence_sweep(3)
    return _CACHE["sweep"]


def _probe():
    if "probe" not in _CACHE:
        _CACHE["probe"] = probe_shear_invariant()
    return _CACHE["probe"]


class DragDecayClosedFormTests(unittest.TestCase):
    """Case A: the ledger must reproduce the exponential-decay solution."""

    def test_velocity_matches_closed_form(self):
        decay = _decay()
        self.assertAlmostEqual(decay["time_s"], 1.5 * decay["tau_s"], places=12)
        self.assertLess(
            decay["velocity_rel_error_pct"], 3.0,
            f"velocity error {decay['velocity_rel_error_pct']:.4g} % exceeds 3 %",
        )

    def test_cumulative_dissipation_matches_closed_form(self):
        decay = _decay()
        self.assertLess(
            decay["drag_dissipation_rel_error_pct"], 3.0,
            f"ledger dissipation error "
            f"{decay['drag_dissipation_rel_error_pct']:.4g} % exceeds 3 %",
        )
        # The ledger must record strictly positive dissipation, and no work was
        # done on the particle (isolated, drag only).
        self.assertGreater(decay["drag_dissipation_sim_J"], 0.0)
        self.assertEqual(decay["tool_work_J"], 0.0)
        self.assertEqual(decay["particle_damping_J"], 0.0)

    def test_residual_equals_first_order_truncation_prediction(self):
        """Residual = 0.5 * (drag/m) * dt * D(t): the pre-update velocity makes
        the drag rate over-record by exactly this fraction."""
        decay = _decay()
        self.assertGreater(decay["energy_balance_residual_J"], 0.0)
        self.assertLess(
            decay["residual_prediction_rel_error_pct"], 2.0,
            "closed-form residual prediction off by "
            f"{decay['residual_prediction_rel_error_pct']:.4g} %",
        )

    def test_residual_and_errors_converge_first_order(self):
        rows = _sweep()
        orders = _sweep_orders()
        for key in ("velocity_rel_error", "drag_dissipation_rel_error",
                    "drag_residual_relative"):
            self.assertAlmostEqual(
                orders[key], 1.0, delta=0.15,
                msg=f"{key} observed order {orders[key]:.3f} is not first order",
            )
        # Refining must reduce both errors monotonically.
        self.assertLess(rows[-1]["drag_dissipation_rel_error_pct"],
                        rows[0]["drag_dissipation_rel_error_pct"])

    def test_residual_relative_stays_small_at_recommended_dt(self):
        decay = _decay()
        self.assertLess(abs(decay["energy_balance_residual_relative"]), 5e-3,
                        "ledger residual exceeds 0.5 % of the reference energy")


def _sweep_orders() -> dict[str, float]:
    if "orders" in _CACHE:
        return _CACHE["orders"]  # type: ignore[return-value]
    from scripts.verify_dem3d_energy_ledger_analytic import observed_order

    rows = _sweep()
    dts = [r["dt_s"] for r in rows]
    orders = {
        "velocity_rel_error": observed_order([r["velocity_rel_error_pct"] for r in rows], dts),
        "drag_dissipation_rel_error": observed_order(
            [r["drag_dissipation_rel_error_pct"] for r in rows], dts),
        "drag_residual_relative": observed_order(
            [r["drag_residual_relative"] for r in rows], dts),
        "oscillator_period_rel_error": observed_order(
            [r["oscillator_period_rel_error_pct"] for r in rows], dts),
    }
    _CACHE["orders"] = orders
    return orders  # type: ignore[return-value]


class BondedOscillatorTests(unittest.TestCase):
    """Case B: one elastic bond, no dissipation, closed-form harmonic motion."""

    def test_period_matches_closed_form(self):
        osc = _oscillator()
        self.assertAlmostEqual(osc["omega_rad_s"],
                               math.sqrt(2.0 * BOND_STIFFNESS / base_config().mass), places=12)
        self.assertIsNotNone(osc["period_sim_s"])
        self.assertLess(
            osc["period_rel_error_pct"], 0.1,
            f"period error {osc['period_rel_error_pct']:.4g} % exceeds 0.1 %",
        )

    def test_stretch_tracks_analytic_cosine(self):
        osc = _oscillator()
        # After a non-integer number of periods the analytic stretch is a
        # non-trivial fraction of A; agreement therefore also pins the phase.
        self.assertLess(abs(osc["stretch_reference_m"]), OSC_AMPLITUDE)
        self.assertLess(osc["stretch_abs_error_m"], 0.01 * OSC_AMPLITUDE,
                        f"stretch error {osc['stretch_abs_error_m']:.3e} m > 1 % of A")

    def test_energy_is_bounded_by_symplectic_oscillation(self):
        """No secular drift: the sampled energy oscillates with relative
        amplitude dt*omega (modified-Hamiltonian behaviour)."""
        osc = _oscillator()
        self.assertLess(
            osc["energy_spread_prediction_rel_error_pct"], 1.0,
            "energy spread departs from the dt*omega prediction by "
            f"{osc['energy_spread_prediction_rel_error_pct']:.4g} %",
        )
        self.assertLess(abs(osc["energy_balance_residual_relative"]), 1e-2)

    def test_no_work_or_dissipation_is_recorded(self):
        osc = _oscillator()
        self.assertEqual(osc["tool_work_J"], 0.0)
        self.assertEqual(osc["drag_dissipation_J"], 0.0)
        self.assertEqual(osc["alive_bonds"], 1)

    def test_period_error_is_second_order(self):
        orders = _sweep_orders()
        self.assertAlmostEqual(
            orders["oscillator_period_rel_error"], 2.0, delta=0.3,
            msg=f"period order {orders['oscillator_period_rel_error']:.3f} is not second order",
        )

    def test_energy_spread_halves_with_dt(self):
        rows = _sweep()
        ratios = [rows[i]["oscillator_energy_spread_relative"]
                  / rows[i + 1]["oscillator_energy_spread_relative"]
                  for i in range(len(rows) - 1)]
        for ratio in ratios:
            self.assertAlmostEqual(ratio, 2.0, delta=0.1,
                                   msg=f"energy spread refinement ratio {ratio:.3f} != 2")


class ShearInvariantProbeTests(unittest.TestCase):
    """The fracture criterion's strain measure, scored against closed forms."""

    def test_full_lattice_matches_uniaxial_analytic_value(self):
        probe = _probe()
        self.assertAlmostEqual(
            probe["lattice_uniaxial_shear_code"],
            probe["lattice_uniaxial_shear_analytic"], places=10)
        self.assertLess(probe["lattice_uniaxial_rel_error_pct"], 1e-6)

    def test_full_lattice_is_frame_indifferent_and_zero_at_zero_strain(self):
        probe = _probe()
        self.assertLess(probe["lattice_undeformed_shear"], 1e-10)
        self.assertLess(probe["lattice_rigid_rotation_shear"], 1e-10)

    def test_collinear_neighbourhood_reports_no_spurious_strain(self):
        """Rank-deficient neighbourhoods are skipped (fixed).

        A single bond gives a rank-one reference covariance, so the least-squares
        deformation gradient would be the projection ``e e^T`` rather than the
        identity and the Green-Lagrange invariant a spurious ``1/3`` for a rigid,
        undeformed bond. The shear criterion is undefined there and is skipped
        (reported 0), so a linear (one-particle-wide) specimen does not break on
        the first step. Full-rank cubic lattices are unaffected.
        """
        probe = _probe()
        self.assertLess(probe["collinear_undeformed_shear"], 1e-6)
        self.assertLess(probe["collinear_undeformed_shear"],
                        DEM3DConfig().shear_breaking_strain)

    def test_default_thresholds_keep_a_rigid_collinear_bond(self):
        """The rank-deficiency skip must not break a rigid single bond.

        With the default breaking/short-shear strains a single collinear bond
        (rank-one reference covariance) must survive a step: the spurious 1/3
        shear invariant is skipped, so no shear fracture is triggered.
        """
        cfg = base_config(drag=0.0)  # default breaking/short-shear strains
        solver = build_subsystem(
            cfg,
            positions=[[-RADIUS, 0.0, 0.0], [RADIUS, 0.0, 0.0]],
            velocities=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            bonds=[(0, 1)],
            rest_lengths=[2.0 * RADIUS],
        )
        self.assertTrue(bool(solver.alive[0]))
        solver.step()
        self.assertTrue(bool(solver.alive[0]),
                         "the rank-deficiency skip must keep the rigid bond alive")


class LedgerBookkeepingTests(unittest.TestCase):
    """The verification harness must not perturb the production solver."""

    def test_subsystem_builder_resets_the_ledger(self):
        cfg = base_config(drag=DRAG)
        solver = build_subsystem(
            cfg,
            positions=[[0.0, 0.0, 0.0]],
            velocities=[[0.1, 0.0, 0.0]],
            bonds=[],
        )
        self.assertEqual(solver.cumulative_drag_dissipation_J, 0.0)
        self.assertEqual(solver.cumulative_tool_work_J, 0.0)
        self.assertAlmostEqual(
            solver.reference_mechanical_energy_J,
            0.5 * cfg.mass * 0.1 ** 2, places=15)
        self.assertEqual(solver.pairs.shape, (0, 2))
        self.assertAlmostEqual(solver.dt, cfg.recommended_dt, places=15)

    def test_lattice_solver_untouched_by_closed_form_cases(self):
        reference = IceDEM3D(DEM3DConfig())
        _decay()
        _oscillator()
        fresh = IceDEM3D(DEM3DConfig())
        self.assertTrue(torch.equal(reference.positions, fresh.positions))
        self.assertTrue(torch.equal(reference.pairs, fresh.pairs))
        self.assertEqual(int(fresh.broken_bonds), 0)


if __name__ == "__main__":
    unittest.main()