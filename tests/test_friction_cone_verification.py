"""Verification tests for the Cundall-Strack Coulomb friction cone.

The analytic anchors (sliding onset travel ``delta_s = mu*N/k_t``, re-stick travel
``delta_s`` / ``2*delta_s`` after velocity reversal, friction work) are documented
in docs/dem3d_coulomb_friction_cone_verification.md.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.verify_coulomb_friction_cone import (
    DEFAULT_TOLERANCE,
    FrictionConeParams,
    run_study,
)
from tensordem.contact_tangential import (
    advance_tangential_contact,
    project_tangential_displacement,
    reset_tangential_displacement,
    saturation_displacement,
)


class FrictionConeAnalyticVerificationTests(unittest.TestCase):
    """Screen the projected state machine against analytic Coulomb anchors."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.outdir = Path(cls._tmp.name)
        cls.report = run_study(cls.outdir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_verdict_within_tolerance(self):
        self.assertEqual(self.report["verdict"], "PASS", self.report["error_budget"])
        self.assertLessEqual(self.report["max_relative_error"], DEFAULT_TOLERANCE)
        self.assertEqual(self.report["tolerance_relative"], 0.03)

    def test_stick_force_is_exact(self):
        case = self.report["cases"]["stick_no_slip"]
        self.assertEqual(case["force_rel_error"], 0.0)
        self.assertFalse(case["sliding_reported"])
        self.assertTrue(case["displacement_unchanged"])

    def test_sliding_onset_travel_matches_delta_s(self):
        case = self.report["cases"]["monotonic_sliding"]
        params = FrictionConeParams()
        self.assertAlmostEqual(
            case["measured_onset_travel_m"], params.delta_s, delta=0.03 * params.delta_s
        )

    def test_force_never_exceeds_coulomb_limit(self):
        case = self.report["cases"]["monotonic_sliding"]
        self.assertTrue(case["force_ceiling_respected"])
        self.assertLessEqual(case["max_force_over_saturation"], 1.0 + 1e-12)
        self.assertAlmostEqual(case["final_force_N"], case["analytic_final_force_N"], places=9)

    def test_friction_work_matches_analytic_dissipation(self):
        case = self.report["cases"]["monotonic_sliding"]
        self.assertLessEqual(case["friction_work_rel_error"], DEFAULT_TOLERANCE)

    def test_reversal_restick_travel_matches_analytics(self):
        case = self.report["cases"]["velocity_reversal"]
        self.assertLessEqual(case["zero_crossing_rel_error"], DEFAULT_TOLERANCE)
        self.assertLessEqual(case["minus_saturation_rel_error"], DEFAULT_TOLERANCE)

    def test_legacy_unprojected_state_is_quantifiably_wrong(self):
        """The stateless stepper keeps an unbounded sliding history."""
        case = self.report["cases"]["legacy_velocity_reversal"]
        self.assertGreater(
            case["zero_crossing_rel_error"], 10.0 * DEFAULT_TOLERANCE, case
        )
        self.assertGreater(self.report["legacy_max_relative_error"], 10.0 * DEFAULT_TOLERANCE)

    def test_artifacts_written(self):
        for name in (
            "friction_cone_verification.json",
            "friction_cone_summary.csv",
            "monotonic_sliding_history.csv",
            "reversal_history.csv",
            "legacy_reversal_history.csv",
        ):
            self.assertTrue((self.outdir / name).is_file(), name)
        payload = json.loads(
            (self.outdir / "friction_cone_verification.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["verdict"], "PASS")


class FrictionConeStateTests(unittest.TestCase):
    """Unit behaviour of the projection helpers."""

    def test_saturation_displacement_formula(self):
        value = saturation_displacement(
            torch.tensor([100.0], dtype=torch.float64), k_t=1.0e4, mu=0.3
        )
        self.assertAlmostEqual(float(value[0]), 3.0e-3, places=12)

    def test_projection_clamps_to_cone_and_reports_sliding(self):
        disp = torch.tensor([[0.0, 0.05]], dtype=torch.float64)
        clamped, sliding = project_tangential_displacement(
            disp, torch.tensor([100.0]), k_t=1.0e4, mu=0.3
        )
        self.assertTrue(bool(sliding[0]))
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(clamped[0])), 3.0e-3, places=12
        )
        # Direction is preserved by the radial projection.
        self.assertEqual(float(clamped[0, 0]), 0.0)
        self.assertAlmostEqual(float(clamped[0, 1]), 3.0e-3, places=12)

    def test_projection_leaves_sub_cone_state_untouched(self):
        disp = torch.tensor([[0.0, 1.0e-3]], dtype=torch.float64)
        clamped, sliding = project_tangential_displacement(
            disp, torch.tensor([100.0]), k_t=1.0e4, mu=0.3
        )
        self.assertFalse(bool(sliding[0]))
        self.assertTrue(torch.equal(clamped, disp))

    def test_lost_contact_resets_tangential_state(self):
        disp = torch.tensor([[0.0, 2.0e-3]], dtype=torch.float64)
        force, disp_new, sliding = advance_tangential_contact(
            disp,
            torch.zeros((1, 2), dtype=torch.float64),
            torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            torch.zeros(1),
            k_t=1.0e4,
            mu=0.3,
            dt=1.0e-4,
        )
        self.assertEqual(float(torch.linalg.vector_norm(force[0])), 0.0)
        self.assertEqual(float(torch.linalg.vector_norm(disp_new[0])), 0.0)
        self.assertFalse(bool(sliding[0]))

    def test_reset_helper_only_clears_inactive_rows(self):
        disp = torch.tensor([[0.0, 1.0e-3], [0.0, 2.0e-3]], dtype=torch.float64)
        out = reset_tangential_displacement(disp, torch.tensor([True, False]))
        self.assertAlmostEqual(float(out[0, 1]), 1.0e-3, places=15)
        self.assertEqual(float(out[1, 1]), 0.0)

    def test_three_dimensional_contacts_are_supported(self):
        disp = torch.zeros((1, 3), dtype=torch.float64)
        velocity = torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float64)
        normal = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        force, disp_new, sliding = advance_tangential_contact(
            disp, velocity, normal, torch.tensor([100.0]), 1.0e4, 0.3, 1.0e-2
        )
        self.assertEqual(force.shape, (1, 3))
        self.assertEqual(disp_new.shape, (1, 3))
        self.assertTrue(bool(sliding[0]))
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(force[0])), 30.0, places=9
        )

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            saturation_displacement(torch.tensor([100.0]), k_t=0.0, mu=0.3)
        with self.assertRaises(ValueError):
            saturation_displacement(torch.tensor([100.0]), k_t=1.0e4, mu=-0.1)
        with self.assertRaises(ValueError):
            advance_tangential_contact(
                torch.zeros((1, 2), dtype=torch.float64),
                torch.zeros((1, 2), dtype=torch.float64),
                torch.tensor([[1.0, 0.0]], dtype=torch.float64),
                torch.tensor([100.0]),
                k_t=1.0e4,
                mu=0.3,
                dt=0.0,
            )


if __name__ == "__main__":
    unittest.main()