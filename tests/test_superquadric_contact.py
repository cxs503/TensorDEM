"""Acceptance tests: superquadric contact maturity (campaign IN50-W3-N01).

The criteria the campaign node has to meet, each asserted on numbers rather
than on prose:

* E1 ``Hertz``: sphere-limit degeneration of the *vectorised* contact kernel
  against the analytic Hertz law, ``max rel error <= 1 %`` (measured ~1e-12).
* E1 ``flat``: cube-on-cube face compression against the analytic flat-contact
  stiffness ``E* A / (c_A + c_B)``, ``rel error <= 2 %`` (the contact area is
  measured from the geometry, not taken from the formula).
* E4 ``cost``: per-step cost of the true-shape contact ``<= 3x`` the
  multi-sphere clump contact at equal fidelity (8 spheres per clump).
* E2 ``geometry``: best-fit clump error (max surface deviation, normal
  deviation distribution) and the ``N -> inf`` negative control: the error must
  decrease monotonically as spheres are added.
* E3 ``decision``: ``|theta_true - theta_clump| <= 3 deg`` and force-chain
  (surface-normal) deviation ``<= 5 deg`` -- enforced by ``api_gate``, which
  must accept the sphere limit and reject a blocky shape represented by a
  small clump.
* E5 ``macro anchors``: the shipped artifact reproduces the 22 deg +/- 2 deg
  sphere repose calibration by column collapse, and records the drum and shear
  cell measurements with their evidence labels.

Run: ``python -m pytest tests/test_superquadric_contact.py -q``
"""
from __future__ import annotations

import json
import math
import os
import unittest

import numpy as np

from tensordem.superquadric import (
    Superquadric,
    SuperquadricBatch,
    clump_contact_batch,
    ClumpBatch,
    contact_cost_benchmark,
    cube_flat_contact_area,
    cube_flat_compression_check,
    effective_contact_radius,
    hertz_degenerate_sweep,
    hertz_force_superquadric,
    mindlin_tangential_force,
    principal_curvatures_superquadric,
    superquadric_contact,
    superquadric_contact_batch,
    superquadric_gradient,
    superquadric_implicit,
)
from tensordem.shape_error_budget import (
    CRITERIA,
    ClumpApproximationDisabled,
    DisableRatioTable,
    GeometricErrorResult,
    MacroscopicErrorResult,
    api_gate,
    build_disable_ratio_table,
    compute_geometric_error,
    compute_macroscopic_error,
    equivalent_sphere_radius,
    fit_clump_to_superquadric,
    repose_angle_from_geometry,
    sphere_volume,
)
from tensordem.shape_error_budget import _sample_surface_superquadric

ARTIFACT = os.path.join(os.path.dirname(__file__), "..", "artifacts", "dem",
                        "shape_error_table.json")


def _block(half=0.01, n=50.0):
    return Superquadric(a=half, b=half, c=half, n1=n, n2=n)


class TestHertzDegenerate(unittest.TestCase):
    """E1: the sphere limit must reproduce Hertz through the kernel."""

    def test_max_relative_error_below_one_percent(self):
        sweep = hertz_degenerate_sweep()
        self.assertLessEqual(sweep["max_rel_error"], 0.01,
                             f"criterion 1 %: {sweep['max_rel_error']:.3e}")
        # the kernel is exact in this limit, not merely within tolerance
        self.assertLess(sweep["max_rel_error"], 1e-6)
        self.assertTrue(all(case["in_contact"] for case in sweep["cases"]))

    def test_radius_and_overlap_are_recovered(self):
        sweep = hertz_degenerate_sweep()
        for case in sweep["cases"]:
            self.assertLess(case["overlap_rel_error"], 1e-9)
            self.assertLess(case["radius_eff_rel_error"], 1e-6)

    def test_rotation_invariant(self):
        # identical bodies, random orientations: a sphere cannot care
        aligned = hertz_degenerate_sweep(radii=(5e-3,), overlap_ratios=(2e-3,),
                                        orient_random=False)
        rotated = hertz_degenerate_sweep(radii=(5e-3,), overlap_ratios=(2e-3,),
                                        orient_random=True, seed=99)
        self.assertAlmostEqual(aligned["max_rel_error"], rotated["max_rel_error"],
                               delta=1e-9)

    def test_scalar_reference_agrees_for_spheres(self):
        r = 0.01
        a = Superquadric(r, r, r, 2.0, 2.0, (0.0, 0.0, 0.0))
        b = Superquadric(r, r, r, 2.0, 2.0, (0.0, 0.0, 2 * r - 1e-4))
        contact = superquadric_contact(a, b)
        self.assertTrue(contact.in_contact)
        self.assertAlmostEqual(contact.overlap, 1e-4, delta=1e-9)
        self.assertAlmostEqual(contact.effective_radius, r / 2, delta=1e-9)


class TestCubeFlatCompression(unittest.TestCase):
    """E1: face-to-face flat compression against the analytic stiffness."""

    def test_relative_error_below_two_percent(self):
        check = cube_flat_compression_check()
        self.assertLessEqual(check["rel_error"], 0.02,
                             f"criterion 2 %: {check['rel_error']:.4f}")

    def test_contact_area_is_measured_from_geometry(self):
        area = cube_flat_contact_area(0.01, 1e-5, 50.0, 400)
        self.assertLessEqual(abs(area["area_geometric_m2"] - area["area_analytic_m2"])
                             / area["area_analytic_m2"], 0.02)
        # the sampled patch must be a real 2D region, not a single point
        self.assertGreater(area["contact_points"], 0.9 * 400 ** 2)

    def test_flat_law_is_linear_in_overlap(self):
        check = cube_flat_compression_check()
        # a curvature-free contact has no Hertz non-linearity
        self.assertLess(check["stiffness_spread"], 1e-6)

    def test_flat_punch_convention_is_documented(self):
        check = cube_flat_compression_check()
        # competing (Sneddon) convention: the 2 % claim is convention-relative
        self.assertAlmostEqual(check["flat_punch_vs_column_ratio"], 1.1284, places=3)


class TestVectorisedKernel(unittest.TestCase):
    """The batch kernel is the production path; it must match the reference."""

    def test_batch_overlap_matches_scalar_for_spheres(self):
        r = 0.005
        offsets = np.linspace(2 * r - 2e-4, 2 * r - 1e-5, 5)
        body_a = SuperquadricBatch.identity(np.full(5, r), np.full(5, r), np.full(5, r),
                                           np.full(5, 2.0), np.full(5, 2.0),
                                           np.zeros((5, 3)))
        body_b = SuperquadricBatch.identity(np.full(5, r), np.full(5, r), np.full(5, r),
                                           np.full(5, 2.0), np.full(5, 2.0),
                                           np.stack([np.zeros(5), np.zeros(5), offsets], axis=1))
        res = superquadric_contact_batch(body_a, body_b)
        np.testing.assert_allclose(res["overlap"], 2 * r - offsets, rtol=1e-12)
        np.testing.assert_allclose(res["radius_eff"], r / 2, rtol=1e-9)

    def test_block_face_contact_is_exact(self):
        a = 0.01
        overlap = 1e-5
        body_a = SuperquadricBatch.identity(np.array([a]), np.array([a]), np.array([a]),
                                           np.array([50.0]), np.array([50.0]),
                                           np.zeros((1, 3)))
        body_b = SuperquadricBatch.identity(np.array([a]), np.array([a]), np.array([a]),
                                            np.array([50.0]), np.array([50.0]),
                                            np.array([[0.0, 0.0, 2 * a - overlap]]))
        res = superquadric_contact_batch(body_a, body_b)
        self.assertAlmostEqual(float(res["overlap"][0]), overlap, delta=1e-12)
        np.testing.assert_allclose(res["normal"][0], [0.0, 0.0, 1.0], atol=1e-9)

    def test_cost_ratio_within_three_times_clump(self):
        ratios = [contact_cost_benchmark(n_pairs=4096, n_sub_spheres=8,
                                        contact_fraction=0.15, repeats=7)["ratio"]
                  for _ in range(3)]
        ratio = float(np.median(ratios))
        self.assertLessEqual(ratio, 3.0,
                             f"criterion <= 3x clump: measured {ratios}")

    def test_cost_ratio_shrinks_with_larger_clumps(self):
        small = contact_cost_benchmark(n_pairs=4096, n_sub_spheres=4, repeats=5)["ratio"]
        big = contact_cost_benchmark(n_pairs=4096, n_sub_spheres=16, repeats=5)["ratio"]
        self.assertLess(big, small)


class TestGeometricLayer(unittest.TestCase):
    """E2: clump geometry error and the N -> inf negative control."""

    def test_surface_samples_lie_on_the_surface(self):
        for n1, n2 in ((2.0, 2.0), (8.0, 2.0), (2.0, 8.0), (16.0, 16.0), (50.0, 50.0)):
            sq = Superquadric(0.02, 0.01, 0.006, n1, n2)
            pts, normals = _sample_surface_superquadric(sq, 300)
            f = superquadric_implicit(sq.to_local(pts), sq.a, sq.b, sq.c, sq.n1, sq.n2)
            self.assertLess(float(np.max(np.abs(f))), 1e-12, f"n=({n1},{n2})")
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-9)

    def test_sphere_limit_clump_is_exact(self):
        sq = Superquadric(0.01, 0.01, 0.01, 2.0, 2.0)
        geom = compute_geometric_error(sq, 1, n_surface_samples=400)
        self.assertLess(geom.max_surface_distance_rel, 0.02)
        self.assertLess(geom.mean_normal_deviation_deg, 1.0)

    def test_error_decreases_monotonically_with_sphere_count(self):
        sq = Superquadric(a=3.0, b=1.0, c=1.0, n1=50.0, n2=50.0)
        fit = fit_clump_to_superquadric(sq, 24, n_surface_samples=800, return_fit=True)
        history = np.array(fit.surface_error_history)
        self.assertEqual(len(history), 24)
        self.assertTrue(np.all(np.diff(history) <= 1e-12),
                        "greedy cover must never increase the deviation")
        self.assertLess(history[-1], history[0])

    def test_more_spheres_never_make_the_fit_worse(self):
        # negative control: the min-max cover is monotone in the sphere count
        sq = Superquadric(a=2.0, b=1.0, c=1.0, n1=8.0, n2=8.0)
        coarse = compute_geometric_error(sq, 1, n_surface_samples=600)
        fine = compute_geometric_error(sq, 16, n_surface_samples=600)
        self.assertLessEqual(fine.error_history[-1], coarse.error_history[-1])
        self.assertLessEqual(fine.max_surface_distance,
                             coarse.max_surface_distance + 1e-12)

    def test_flattened_floe_needs_many_spheres(self):
        sq = Superquadric(a=5.0, b=1.0, c=1.0, n1=50.0, n2=50.0)
        coarse = compute_geometric_error(sq, 4, n_surface_samples=600)
        self.assertGreater(coarse.max_normal_deviation_deg, 5.0)

    def test_volume_matches_analytic_sphere(self):
        sq = Superquadric(0.02, 0.02, 0.02, 2.0, 2.0)
        vol = sphere_volume(sq, n_probe=200_000)
        self.assertAlmostEqual(vol / (4.0 / 3.0 * math.pi * 0.02 ** 3), 1.0, delta=0.01)
        self.assertAlmostEqual(equivalent_sphere_radius(sq), 0.02, delta=1e-3)


class TestDecisionGate(unittest.TestCase):
    """E3: the (3 deg, 5 deg) decision rule and the disable ratio table."""

    def test_gate_accepts_sphere_limit(self):
        gate = api_gate(Superquadric(0.01, 0.01, 0.01, 2.0, 2.0), 1)
        self.assertTrue(gate["allowed"], gate["reasons"])
        self.assertLessEqual(gate["repose_error_deg"], CRITERIA["max_repose_error_deg"])

    def test_gate_rejects_blocky_clump(self):
        gate = api_gate(Superquadric(a=3.0, b=1.0, c=1.0, n1=50.0, n2=50.0), 8)
        self.assertFalse(gate["allowed"])
        self.assertTrue(gate["reasons"])

    def test_gate_raises_when_required(self):
        with self.assertRaises(ClumpApproximationDisabled) as ctx:
            api_gate(Superquadric(a=5.0, b=1.0, c=1.0, n1=50.0, n2=50.0), 4,
                     raise_on_fail=True)
        self.assertIn("rejected", str(ctx.exception))
        self.assertFalse(ctx.exception.diagnostics["allowed"])

    def test_larger_clump_can_pass_the_gate(self):
        sq = Superquadric(a=2.0, b=1.0, c=1.0, n1=8.0, n2=8.0)
        small = api_gate(sq, 2)
        large = api_gate(sq, 32)
        self.assertLessEqual(large["max_surface_distance_rel"],
                             small["max_surface_distance_rel"] + 1e-12)

    def test_table_is_deterministic_and_carries_criteria(self):
        kwargs = dict(aspect_ratios=(1.0, 3.0), shape_exponents=(2.0, 50.0),
                      sphere_counts=(1, 8))
        first = build_disable_ratio_table(**kwargs)
        second = build_disable_ratio_table(**kwargs)
        self.assertEqual(first.entries, second.entries)
        self.assertEqual(first.criteria["max_repose_error_deg"], 3.0)
        self.assertEqual(first.criteria["max_force_chain_error_deg"], 5.0)
        for entry in first.entries:
            for key in ("aspect_ratio", "shape_exponent", "use_clump",
                        "repose_error_deg", "force_chain_error_deg",
                        "geometric_error_m", "normal_deviation_deg"):
                self.assertIn(key, entry)
        # sphere limit is admissible, blocky small clumps are not
        sphere_row = next(e for e in first.entries
                          if e["aspect_ratio"] == 1.0 and e["shape_exponent"] == 2.0)
        block_row = next(e for e in first.entries
                         if e["aspect_ratio"] == 3.0 and e["shape_exponent"] == 50.0)
        self.assertTrue(sphere_row["use_clump"])
        self.assertFalse(block_row["use_clump"])
        restored = DisableRatioTable.from_json(first.to_json())
        self.assertEqual(restored.criteria, first.criteria)
        self.assertEqual(len(restored.entries), len(first.entries))

    def test_macro_result_declares_evidence(self):
        sq = Superquadric(a=2.0, b=1.0, c=1.0, n1=8.0, n2=8.0)
        macro = compute_macroscopic_error(sq, 8)
        self.assertIsInstance(macro, MacroscopicErrorResult)
        self.assertEqual(macro.evidence["theta_r(mu)"], "REAL (3D DEM column collapse, monodisperse spheres)")
        self.assertIn("WITHHELD", macro.evidence["poly_shape_dem"])
        self.assertGreaterEqual(macro.repose_angle_clump_deg, macro.repose_angle_true_deg)


class TestContactMechanics(unittest.TestCase):
    """Classical checks that must keep holding in the mature module."""

    def test_hertz_force_scaling(self):
        f1 = hertz_force_superquadric(1e-4, 1e9, 0.005)
        f2 = hertz_force_superquadric(2e-4, 1e9, 0.005)
        self.assertAlmostEqual(f2 / f1, 2 ** 1.5, places=9)
        self.assertEqual(hertz_force_superquadric(0.0, 1e9, 0.005), 0.0)
        self.assertEqual(hertz_force_superquadric(-1e-4, 1e9, 0.005), 0.0)

    def test_mindlin_saturates_at_coulomb_cone(self):
        disp = np.array([1e-3, 0.0, 0.0])
        f_t, _ = mindlin_tangential_force(1e-5, disp, 1e9, 4e8, 0.005, 0.4, 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(f_t)), 0.4, places=9)

    def test_sphere_curvature_and_radius(self):
        r = 0.01
        k1, k2 = principal_curvatures_superquadric(np.array([r, 0.0, 0.0]),
                                                   r, r, r, 2.0, 2.0)
        self.assertAlmostEqual(abs(k1), 1.0 / r, delta=1e-2 / r)
        self.assertAlmostEqual(abs(k2), 1.0 / r, delta=1e-2 / r)
        r_eff = effective_contact_radius(k1, k2, k1, k2)
        self.assertAlmostEqual(r_eff, r / 2, delta=1e-4)

    def test_gradient_matches_finite_difference(self):
        sq = Superquadric(0.02, 0.012, 0.007, 6.0, 3.0)
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(6, 3)) * np.array([sq.a, sq.b, sq.c])
        analytic = np.atleast_2d(superquadric_gradient(pts, sq.a, sq.b, sq.c, sq.n1, sq.n2))
        h = 1e-8
        for i in range(6):
            numeric = np.zeros(3)
            for axis in range(3):
                e = np.zeros(3)
                e[axis] = h
                fp = superquadric_implicit((pts[i] + e).reshape(1, 3), sq.a, sq.b, sq.c, sq.n1, sq.n2)
                fm = superquadric_implicit((pts[i] - e).reshape(1, 3), sq.a, sq.b, sq.c, sq.n1, sq.n2)
                numeric[axis] = (float(fp) - float(fm)) / (2 * h)
            np.testing.assert_allclose(analytic[i], numeric, rtol=1e-4, atol=1e-6)

    def test_clump_baseline_finds_deepest_contact(self):
        centers_a = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.02]]])
        centers_b = np.array([[[0.0, 0.0, 0.0209], [0.0, 0.0, 0.06]]])
        res = clump_contact_batch(
            ClumpBatch(centers_a, np.full((1, 2), 0.005)),
            ClumpBatch(centers_b, np.full((1, 2), 0.005)))
        self.assertAlmostEqual(float(res["overlap"][0]), 0.0091, delta=1e-12)
        self.assertEqual(int(res["index_a"][0]), 1)

    def test_repose_angle_post_processing(self):
        # synthetic cone of particles with a known 25 degree slope
        rng = np.random.default_rng(1)
        pts = []
        for _ in range(400):
            x = rng.uniform(-0.2, 0.2)
            pts.append((x, rng.uniform(-0.2, 0.2), 0.2 - math.tan(math.radians(25)) * abs(x)))
        angle = repose_angle_from_geometry(np.array(pts))
        self.assertAlmostEqual(angle, 25.0, delta=2.0)


class TestArtifact(unittest.TestCase):
    """E5: the shipped artifact must carry the criteria numbers."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(ARTIFACT):
            raise unittest.SkipTest("artifact not built (run scripts/build_shape_error_table.py)")
        with open(ARTIFACT) as fh:
            cls.data = json.load(fh)

    def test_single_contact_numbers(self):
        hertz = self.data["single_contact"]["hertz_limit"]
        cube = self.data["single_contact"]["cube_flat_compression"]
        cost = self.data["single_contact"]["cost"]
        self.assertLessEqual(hertz["max_rel_error"], 0.01)
        self.assertLessEqual(cube["rel_error"], 0.02)
        ratio = cost["clump_8_spheres"]["ratio"]
        self.assertLessEqual(ratio, 3.0, f"cost criterion: {ratio:.2f}x")
        curve = cost["batch_size_curve_n8"]
        self.assertLessEqual(curve["8192"]["ratio"], 3.0)

    def test_geometry_rows_present(self):
        rows = self.data["geometry"]["rows"]
        self.assertGreater(len(rows), 4)
        for row in rows:
            self.assertIn("max_surface_distance_rel", row)
            self.assertIn("max_normal_deviation_deg", row)
            self.assertIn("n_clump_spheres", row)

    def test_repose_calibration_reproduces_22_degrees(self):
        anchor = self.data["macro"]["anchor"]
        self.assertGreaterEqual(len(anchor), 2)
        best = min(anchor, key=lambda row: abs(row["theta_r_deg"] - 22.0))
        self.assertLessEqual(abs(best["theta_r_deg"] - 22.0), 2.0,
                             f"22 +/- 2 deg criterion: {best}")

    def test_both_repose_methods_recorded(self):
        macro = self.data["macro"]
        self.assertTrue(macro["drum"], "rotating drum results missing")
        self.assertTrue(macro["collapse_column"], "column collapse results missing")
        # two independent methods: report their difference honestly
        drum_mean = macro["drum_mean_theta_deg"]
        self.assertTrue(math.isfinite(drum_mean))
        macro_evidence = macro["evidence"]
        self.assertIn("REAL", macro_evidence["drum"])
        self.assertIn("REAL", macro_evidence["collapse_column"])

    def test_shear_cell_reproduces_mobilised_friction(self):
        shear = self.data["macro"]["shear_box"]
        self.assertTrue(shear, "shear cell results missing")
        for row in shear:
            mu_in = row["mu_input"]
            mu_i = row["mu_apparent"]
            # a fully mobilised thin layer must return the inter-particle friction
            self.assertLessEqual(abs(mu_i - mu_in) / mu_in, 0.02)

    def test_table_and_gate_examples(self):
        entries = self.data["disable_ratio_table"]
        self.assertGreater(len(entries), 10)
        gates = self.data["gate_examples"]
        sphere = next(g for g in gates if g["aspect_ratio"] == 1.0)
        block = next(g for g in gates if g["aspect_ratio"] == 3.0)
        self.assertTrue(sphere["gate"]["allowed"])
        self.assertFalse(block["gate"]["allowed"])
        self.assertEqual(self.data["meta"]["criteria"]["max_repose_error_deg"], 3.0)


if __name__ == "__main__":
    unittest.main()