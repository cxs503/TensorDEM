"""Regression guard for benchmarks/ice_floe_physics.py.

This file exists because the benchmark it covers was previously unreferenced:
nothing in the tree imported it, no test ran it, and two of its four anchors
verified nothing about repository code (see
``docs/ice_floe_physics_validation.md``).  Each anchor is now executed here
against its closed-form reference, and each anchor that claims to exercise a
repository force law is additionally subjected to a *negative control*: the
module under test is replaced by a stub and the anchor must fail.  A physics
anchor without a negative control can silently degrade into a tautology.
"""
from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path
from unittest import mock

import torch

torch.set_num_threads(1)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_benchmark():
    path = _REPO_ROOT / "benchmarks" / "ice_floe_physics.py"
    spec = importlib.util.spec_from_file_location("ice_floe_physics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark()


class IndependentGeometryTests(unittest.TestCase):
    """Closed-form submerged area must agree with the module's segment formula."""

    def test_closed_form_matches_segment_formula(self):
        radius = 0.37
        for i in range(41):
            u = (-1.0 + 2.0 * i / 40.0) * radius
            ref = bench.analytic_submerged_fraction(u, radius)
            code = float(bench.submerged_fraction_tensor(
                torch.tensor([u], dtype=torch.float64), radius)[0])
            self.assertAlmostEqual(code, ref, places=12,
                                   msg=f"centre height {u}")

    def test_endpoint_limits(self):
        radius = 0.5
        self.assertEqual(bench.analytic_submerged_area(radius, radius), 0.0)
        self.assertAlmostEqual(
            bench.analytic_submerged_area(-radius, radius),
            math.pi * radius ** 2, places=14)
        # Half submerged at the waterline.
        self.assertAlmostEqual(
            bench.analytic_submerged_fraction(0.0, radius), 0.5, places=15)

    def test_equilibrium_height_is_consistent(self):
        """h_eq must reproduce the Archimedes fraction through both paths."""
        h_eq = bench.analytic_equilibrium_height(917.0, 1025.0, 0.5)
        self.assertLess(h_eq, 0.0)
        self.assertAlmostEqual(
            bench.analytic_submerged_fraction(h_eq, 0.5), 917.0 / 1025.0,
            places=12)
        # The old implementation claimed -(1 - rho_i/rho_w)*R; that is wrong for
        # a circular section and would sit 6.8x too close to the surface.
        self.assertGreater(abs(h_eq), 0.5 * 0.5)


class HydrostaticAnchorTests(unittest.TestCase):
    def test_settles_to_archimedes_fraction(self):
        result = bench.anchor_hydrostatic_equilibrium(dt=2e-2, duration=8.0)
        self.assertTrue(result["passed"], result)
        self.assertLess(result["max_relative_fraction_error"], 1e-2)
        self.assertLess(result["max_abs_velocity_m_s"], 1e-3)
        for f in result["settled_f_sub"]:
            self.assertLess(abs(f - 917.0 / 1025.0) / (917.0 / 1025.0), 1e-2)

    def test_negative_control_without_buoyancy(self):
        """Stub the buoyancy force: the anchor must fail, not silently pass."""
        zero = lambda *a, **k: torch.zeros_like(a[0])  # noqa: E731
        with mock.patch.object(bench, "compute_buoyancy_force", zero):
            result = bench.anchor_hydrostatic_equilibrium(dt=2e-2, duration=4.0)
        # A fully sunk disk caps the fraction metric at 1 - rho_i/rho_w = 11.8 %,
        # so the control is judged on "clearly out of criterion", not on magnitude.
        self.assertFalse(result["passed"])
        self.assertGreater(result["max_relative_fraction_error"], 0.05)


class ArchimedesStaticTests(unittest.TestCase):
    def test_force_matches_independent_volume(self):
        result = bench.anchor_archimedes_static()
        self.assertTrue(result["passed"], result["max_relative_error"])
        self.assertLess(result["max_relative_error"], 1e-3)

    def test_negative_control(self):
        zero = lambda *a, **k: torch.zeros_like(a[0])  # noqa: E731
        with mock.patch.object(bench, "compute_buoyancy_force", zero):
            result = bench.anchor_archimedes_static()
        self.assertFalse(result["passed"])

    def test_negative_control_half_scale(self):
        """A 2x wrong buoyancy must be reported as ~50 % error, not a pass."""
        half = lambda p, r, t, d, c: 0.5 * bench._real_buoyancy(p, r, t, d, c)  # noqa: E731
        bench._real_buoyancy = bench.compute_buoyancy_force
        try:
            with mock.patch.object(bench, "compute_buoyancy_force", half):
                result = bench.anchor_archimedes_static()
        finally:
            del bench._real_buoyancy
        self.assertFalse(result["passed"])
        self.assertGreater(result["max_relative_error"], 0.4)


class HeavePeriodTests(unittest.TestCase):
    def test_period_matches_linear_theory(self):
        result = bench.anchor_heave_period(dt_levels=(1e-2, 5e-3))
        self.assertTrue(result["passed"], result["rows"])
        for row in result["rows"]:
            self.assertLess(row["relative_error"], 1e-2)

    def test_period_converges_with_dt(self):
        result = bench.anchor_heave_period(dt_levels=(1e-2, 2.5e-3))
        errors = result["relative_error_sequence"]
        self.assertLess(errors[-1], errors[0] + 1e-6)
        self.assertLess(errors[-1], 1e-3)

    def test_negative_control_free_fall_period(self):
        """Without gravity-restoring buoyancy there is no oscillation at all."""

        def falling(positions, radius, thickness, density_ice, cfg):
            return torch.zeros_like(positions)

        with mock.patch.object(bench, "compute_buoyancy_force", falling):
            with self.assertRaises(RuntimeError):
                bench.anchor_heave_period(dt_levels=(4e-3,))


class FrictionSlipTests(unittest.TestCase):
    def test_onset_and_acceleration(self):
        result = bench.anchor_friction_slip(dt_levels=(1e-4,), duration=0.02,
                                            onset_tol_deg=1.0)
        self.assertTrue(result["passed"], result["rows"])
        for row in result["rows"]:
            self.assertLess(row["accel_relative_error"], 1e-2)
            self.assertTrue(row["stick_test_bounded"])
            # Onset must land within one grid step plus the coarse-dt bias.
            self.assertLess(row["onset_error_deg"], 1.0)

    def test_negative_control_frictionless(self):
        """mu = 0 removes the tangential force: every angle must slide."""
        result = bench.anchor_friction_slip(mu_values=(0.0,), dt_levels=(1e-4,),
                                            duration=0.01, onset_tol_deg=1.0)
        row = result["rows"][0]
        flags = row["dt_levels"][0]["sliding_flags"]
        # Lanes at or above arctan(0) = 0 deg all slide; the exactly-level lane
        # (drive = 0) cannot, so it is excluded from the control.
        mid = len(flags) // 2          # lane at exactly theta = 0 deg: no drive
        self.assertTrue(all(flags[:mid] + flags[mid + 1:]), flags)
        self.assertFalse(row["stick_test_bounded"])
        self.assertFalse(row["passed"])


class RestitutionMappingTests(unittest.TestCase):
    def test_damped_oscillator_reference(self):
        result = bench.anchor_restitution_mapping(dt=1e-6)
        self.assertTrue(result["passed"], result["rows"])
        self.assertLess(result["max_ode_relative_error"], 3e-2)

    def test_negative_control_zero_damping(self):
        """e = 1 as used by a frictionless/elastic contact must not be reported."""
        stale = bench.restitution_to_damping(0.5, 1e4, 1.0)
        with mock.patch.object(bench, "restitution_to_damping",
                               lambda *a, **k: stale):
            result = bench.anchor_restitution_mapping(e_targets=(0.9,), dt=1e-6)
        self.assertFalse(result["passed"])


class WholeReportTests(unittest.TestCase):
    def test_report_schema_and_overall_pass(self):
        report = bench.run_all(fast=True)
        self.assertEqual(report["schema"], "tensordem.ice-floe-physics/2")
        self.assertTrue(report["overall_pass"], [
            (a["anchor"], a["passed"]) for a in report["anchors"]])
        names = {a["anchor"] for a in report["anchors"]}
        self.assertEqual(names, {"H_hydrostatic_equilibrium",
                                 "A_archimedes_static",
                                 "T_heave_period",
                                 "F_friction_slip",
                                 "R_restitution_mapping"})
        for anchor in report["anchors"]:
            self.assertIn("code_under_test", anchor)
            self.assertIn("reference", anchor)


if __name__ == "__main__":
    unittest.main()