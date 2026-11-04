"""Invariance audit of the DEM3D contact/bond kernel, plus monitor purity.

The thresholds are the ones the audit report uses (``contact_invariants.TOLERANCES``).
Two classes of claim are pinned here:

1. *Kernel invariants* -- momentum closure, translation/rotation covariance,
   agreement with an independent ``O(N^2)`` reference, bitwise determinism,
   additive external loads and a symmetric tangent stiffness -- on randomised
   lattices whose topology is frozen before comparison.
2. *Monitor purity* -- ``diagnostics()`` must not change the trajectory of a run.
   This is a regression test for a defect found by the audit: ``diagnostics``
   went through the step-owner force path, so every probe added one
   ``v_tangent * dt`` Cundall-Strack slip increment.  Measuring the magnitude of
   that defect requires the pre-fix code, so the test asserts the fixed
   behaviour (zero divergence) and that a probe leaves the friction history
   untouched.
"""
from __future__ import annotations

import unittest

import torch

from tensordem.contact_invariants import (
    DEFAULT_CASES,
    TOLERANCES,
    CaseSpec,
    brute_force_agreement,
    determinism_repeatability,
    frozen_state,
    build_instance,
    load_superposition,
    momentum_closure,
    monitor_purity,
    rotation_equivariance,
    run_audit,
    summarise,
    tangent_reciprocity,
    translation_invariance,
    violations,
)
from tensordem.dem3d import DEM3DConfig, IceDEM3D

FRACTURED = CaseSpec(
    "linear_fractured",
    contact_model="linear",
    perturbation=1.2,
    pre_fracture=True,
    breaking_strain=0.015,
    shear_breaking_strain=0.03,
)
FRICTION = CaseSpec(
    "linear_friction_fractured",
    contact_model="linear",
    perturbation=1.2,
    pre_fracture=True,
    breaking_strain=0.015,
    shear_breaking_strain=0.03,
    friction_coefficient=0.4,
)
MIXED = CaseSpec(
    "linear_mixed",
    contact_model="linear",
    perturbation=1.2,
    pre_fracture=True,
    fracture_fraction=0.5,
    breaking_strain=0.015,
    shear_breaking_strain=0.03,
)


class KernelInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.seeds = (0, 1)

    def assertInvariant(self, result):
        tolerance = TOLERANCES[result.invariant]
        self.assertLessEqual(
            result.residual,
            tolerance,
            f"{result.invariant} on {result.case} seed {result.seed}: "
            f"residual {result.residual:.3e} > tolerance {tolerance:.1e}",
        )

    def test_momentum_closure_all_cases(self):
        for case in DEFAULT_CASES:
            for seed in self.seeds:
                with self.subTest(case=case.name, seed=seed):
                    result = momentum_closure(case, seed)
                    self.assertInvariant(result)
                    # The tool must be far enough away to contribute nothing.
                    self.assertEqual(result.detail["reaction_inf"], 0.0)

    def test_translation_invariance(self):
        for case in DEFAULT_CASES:
            with self.subTest(case=case.name):
                self.assertInvariant(translation_invariance(case, 0, (1.5, -7.25, 3.5)))

    def test_rotation_equivariance(self):
        for case in DEFAULT_CASES:
            with self.subTest(case=case.name):
                result = rotation_equivariance(case, 0)
                self.assertInvariant(result)
                # The rotation must be proper, otherwise the comparison is void.
                self.assertAlmostEqual(result.detail["det"], 1.0, places=12)

    def test_brute_force_reference_agreement(self):
        """The engine must match an independent O(N^2) pair loop."""
        for case in DEFAULT_CASES:
            with self.subTest(case=case.name):
                result = brute_force_agreement(case, 0)
                self.assertInvariant(result)

    def test_fractured_cases_actually_exercise_contacts(self):
        """Guard against a vacuous audit: contacts must be evaluated."""
        intact = brute_force_agreement(DEFAULT_CASES[0], 0)
        fractured = brute_force_agreement(FRACTURED, 0)
        # An intact perturbed lattice suppresses every contact behind a live
        # bond, so the reference sees no contact pair at all ...
        self.assertEqual(intact.detail["unbonded_overlaps"], 0.0)
        self.assertEqual(intact.detail["contact_scale"], 0.0)
        # ... while a fractured lattice must contain unbonded overlaps.
        self.assertGreater(fractured.detail["unbonded_overlaps"], 10.0)
        self.assertGreater(fractured.detail["broken_bonds"], 0.0)
        self.assertGreater(fractured.detail["contact_scale"], 0.0)

    def test_determinism_is_bitwise(self):
        for case in DEFAULT_CASES:
            with self.subTest(case=case.name):
                result = determinism_repeatability(case, 0)
                self.assertEqual(result.detail["fresh_defect"], 0.0)
                self.assertEqual(result.detail["reuse_defect"], 0.0)

    def test_external_loads_are_additive(self):
        for case in (DEFAULT_CASES[0], FRACTURED, FRICTION):
            with self.subTest(case=case.name):
                self.assertInvariant(load_superposition(case, 0))

    def test_tangent_stiffness_is_symmetric(self):
        """K = dF/dx must be symmetric for the elastic path (gradient field)."""
        for case in (DEFAULT_CASES[0], FRACTURED):
            with self.subTest(case=case.name):
                result = tangent_reciprocity(case, 0)
                self.assertInvariant(result)
                self.assertGreater(result.detail["stiffness_scale"], 0.0)

    def test_mixed_topology_exercises_both_bond_and_contact(self):
        """The mixed case must carry live bonds AND unbonded contacts together."""
        mixed = brute_force_agreement(MIXED, 0)
        self.assertGreater(mixed.detail["broken_bonds"], 0.0)
        self.assertGreater(mixed.detail["unbonded_overlaps"], 10.0)
        self.assertGreater(mixed.detail["contact_scale"], 0.0)
        # A mixed topology must still have live bonds (unlike the fully-fractured
        # cases where every bond is broken) -- otherwise the audit is vacuous.
        self.assertLess(mixed.detail["broken_bonds"], 326)

    def test_mixed_topology_all_invariants_pass(self):
        for invariant in ("momentum_closure", "translation", "rotation",
                          "brute_force", "determinism", "load_superposition"):
            for seed in self.seeds:
                with self.subTest(invariant=invariant, seed=seed):
                    result = getattr(__import__("tensordem.contact_invariants", fromlist=[invariant]), invariant)(MIXED, seed)
                    self.assertInvariant(result)

    def test_audit_summary_has_no_violations(self):
        results = run_audit(seeds=(0, 2), include_reciprocity=False)
        self.assertEqual(violations(results), [])
        rows = summarise(results)
        self.assertEqual({row["invariant"] for row in rows},
                         {"momentum_closure", "translation", "rotation",
                          "brute_force", "determinism", "load_superposition"})
        self.assertTrue(all(row["passed"] for row in rows))


class MonitorPurityTests(unittest.TestCase):
    """A probe must not change the state it is measuring."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_diagnostics_does_not_change_the_trajectory(self):
        for case in (FRICTION, DEFAULT_CASES[3], DEFAULT_CASES[4]):
            with self.subTest(case=case.name):
                purity = monitor_purity(case, seed=0, steps=40)
                self.assertEqual(purity["position_divergence_m"], 0.0)
                self.assertEqual(purity["velocity_divergence_m_s"], 0.0)
                self.assertEqual(purity["probe_repeat_defect_N"], 0.0)

    def test_probe_leaves_tangential_history_untouched(self):
        state = frozen_state(FRICTION, 0)
        sim = build_instance(FRICTION, state)
        sim.forces(update_fracture=False, update_history=True)  # step owner
        snapshot = {key: (value[0].clone(), value[1].clone())
                    for key, value in sim._tangential_state.items()}
        self.assertTrue(snapshot)
        sim.forces(update_fracture=False, update_history=False)  # probe
        self.assertEqual(set(snapshot), set(sim._tangential_state))
        for key, (displacement, normal) in snapshot.items():
            torch.testing.assert_close(sim._tangential_state[key][0], displacement, atol=0.0, rtol=0.0)
            torch.testing.assert_close(sim._tangential_state[key][1], normal, atol=0.0, rtol=0.0)

    def test_step_owner_path_still_advances_history(self):
        """The fix must not disable friction: the step path still integrates slip."""
        state = frozen_state(FRICTION, 0)
        sim = build_instance(FRICTION, state)
        sim.forces(update_fracture=False, update_history=True)
        before = {key: value[0].clone() for key, value in sim._tangential_state.items()}
        sim.velocities = sim.velocities + torch.tensor([0.05, 0.0, 0.0], dtype=sim.dtype)
        sim.forces(update_fracture=False, update_history=True)
        changed = any(
            not torch.equal(sim._tangential_state[key][0], value)
            for key, value in before.items()
            if key in sim._tangential_state
        )
        self.assertTrue(changed)

    def test_rolling_torque_accumulator_is_not_inflated_by_probes(self):
        sim = IceDEM3D(
            DEM3DConfig(
                nx=3,
                ny=3,
                nz=2,
                drag=0.0,
                fix_x_edges=False,
                friction_coefficient=0.4,
                rolling_resistance_coefficient=0.2,
            )
        )
        sim.forces(update_fracture=False, update_history=True)
        accumulated = sim._rolling_torque.clone()
        sim.forces(update_fracture=False, update_history=False)
        torch.testing.assert_close(sim._rolling_torque, accumulated, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()