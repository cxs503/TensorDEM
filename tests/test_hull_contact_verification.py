"""Analytic verification of the polyline hull-contact kernel.

Every assertion here is against a closed-form geometric reference, not against a
previously stored kernel output: the point of the file is to make the kernel's
discretization error and its degenerate-case behaviour falsifiable.
"""
import math
import unittest

import torch

from tensordem.hull_contact import polyline_contact_forces
from tensordem.hull_contact_verification import (
    bow_polyline,
    contact_error_scan,
    distance_check,
    exact_circular_hull_force,
    invariance_checks,
    no_attraction_check,
    observed_order,
    point_to_polyline_distance,
    required_segments,
    sagitta,
    segments_for_magnitude_error,
    vertex_check,
)

DT = torch.float64
ZERO = torch.zeros(2, dtype=DT)


def mid_chord_probe(radius, segments, particle_radius, overlap):
    """Disks centred above every chord midpoint of an inscribed polygon."""
    vertices = bow_polyline(radius, segments)
    distance = particle_radius + radius - overlap
    angles = torch.arange(segments, dtype=DT) * (2.0 * math.pi / segments) + math.pi / segments
    positions = torch.stack([distance * torch.cos(angles), distance * torch.sin(angles)], dim=1)
    return vertices, positions


class SagittaLawTests(unittest.TestCase):
    """Magnitude error must equal sagitta / penetration -- the closed-form law."""

    def test_mid_chord_error_equals_sagitta_over_penetration(self):
        radius, rp, k = 0.5, 0.05, 1000.0
        for segments in (16, 32, 64):
            for fraction in (0.5, 0.1):
                overlap = fraction * rp
                vertices, positions = mid_chord_probe(radius, segments, rp, overlap)
                forces, _ = polyline_contact_forces(
                    positions, torch.zeros_like(positions), vertices, ZERO,
                    particle_radius=rp, hull_radius=0.0, stiffness=k)
                predicted = sagitta(radius, segments) / overlap
                measured = float(((forces.norm(dim=1) - k * overlap).abs() / (k * overlap)).max())
                if predicted >= 1.0:
                    # sagitta above the penetration: the chord never touches the disk
                    self.assertEqual(measured, 1.0, msg=f"M={segments} fraction={fraction}")
                else:
                    self.assertAlmostEqual(measured, predicted, places=10,
                                           msg=f"M={segments} fraction={fraction}")

    def test_reported_error_scan_reproduces_the_law(self):
        rows = contact_error_scan([32, 64], [0.1])
        for row in rows:
            self.assertAlmostEqual(row["max_relative_magnitude_error"],
                                   row["sagitta_m"] / row["penetration_m"], places=10)


class ConvergenceOrderTests(unittest.TestCase):
    def test_magnitude_is_second_order_and_direction_first_order(self):
        levels = (16, 32, 64, 128)
        rows = [r for r in contact_error_scan(levels, [0.5]) if r["dropout_fraction"] == 0.0]
        magnitude = observed_order([r["segments"] for r in rows],
                                   [r["max_relative_magnitude_error"] for r in rows])
        direction = observed_order([r["segments"] for r in rows],
                                   [r["max_direction_error_deg"] for r in rows])
        self.assertAlmostEqual(magnitude, -2.0, delta=0.05)
        self.assertAlmostEqual(direction, -1.0, delta=0.08)

    def test_fine_bow_reproduces_the_exact_circle(self):
        radius, rp, k = 0.5, 0.05, 1000.0
        vertices, positions = mid_chord_probe(radius, 256, rp, 0.5 * rp)
        forces, _ = polyline_contact_forces(
            positions, torch.zeros_like(positions), vertices, ZERO,
            particle_radius=rp, hull_radius=0.0, stiffness=k)
        exact, _ = exact_circular_hull_force(positions, (0.0, 0.0), radius,
                                             particle_radius=rp, stiffness=k)
        self.assertLess(float((forces - exact).abs().max()), 2.0e-3 * k * rp)


class DropoutCriterionTests(unittest.TestCase):
    """A polyline bow loses contact once its sagitta exceeds the penetration."""

    def test_no_contact_when_sagitta_exceeds_penetration(self):
        radius, rp, k = 0.5, 0.05, 1000.0
        overlap = 0.02 * rp
        for segments in (8, 16, 32):
            self.assertGreater(sagitta(radius, segments), overlap)
            vertices, positions = mid_chord_probe(radius, segments, rp, overlap)
            forces, _ = polyline_contact_forces(
                positions, torch.zeros_like(positions), vertices, ZERO,
                particle_radius=rp, hull_radius=0.0, stiffness=k)
            self.assertEqual(float(forces.norm(dim=1).max()), 0.0, msg=f"M={segments}")

    def test_contact_restored_above_sagitta_threshold(self):
        radius, rp, k = 0.5, 0.05, 1000.0
        overlap = 0.02 * rp
        segments = required_segments(radius, overlap)
        self.assertLessEqual(sagitta(radius, segments), overlap)
        vertices, positions = mid_chord_probe(radius, segments, rp, overlap)
        forces, _ = polyline_contact_forces(
            positions, torch.zeros_like(positions), vertices, ZERO,
            particle_radius=rp, hull_radius=0.0, stiffness=k)
        self.assertGreater(float(forces.norm(dim=1).min()), 0.0)

    def test_error_target_segments_are_consistent_with_the_scan(self):
        radius, rp = 0.5, 0.05
        overlap = 0.1 * rp
        predicted = segments_for_magnitude_error(0.05, radius, overlap)
        self.assertLessEqual(sagitta(radius, predicted) / overlap, 0.05)
        self.assertGreater(sagitta(radius, predicted - 1) / overlap, 0.05)
        measured = [r for r in contact_error_scan([predicted], [0.1])][0]
        self.assertLessEqual(measured["max_relative_magnitude_error"], 0.05)


class GeometryReferenceTests(unittest.TestCase):
    def test_vertex_contact_is_exact_circle_to_circle(self):
        check = vertex_check()
        self.assertLess(check["relative_error"], 1e-12)
        self.assertEqual(check["action_reaction_balance"], 0.0)

    def test_distance_matches_independent_closed_form(self):
        check = distance_check()
        self.assertGreater(check["contacting"], 0)
        self.assertLess(check["max_abs_distance_error_m"], 1e-12)

    def test_force_is_repulsive_and_balanced(self):
        check = no_attraction_check()
        self.assertGreater(check["contacting"], 0)
        self.assertEqual(check["max_action_reaction_imbalance"], 0.0)
        self.assertEqual(check["negative_magnitudes"], 0)

    def test_point_polyline_distance_matches_hand_computed_values(self):
        vertices = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=DT)
        positions = torch.tensor([[1.0, 0.75], [3.0, 0.0], [-0.5, 1.0]], dtype=DT)
        distance, index = point_to_polyline_distance(positions, vertices)
        torch.testing.assert_close(distance, torch.tensor([0.75, 1.0, math.hypot(0.5, 1.0)], dtype=DT))
        self.assertEqual([int(i) for i in index], [0, 0, 0])


class InvarianceTests(unittest.TestCase):
    """The response must depend on the geometry only, not on its representation."""

    def test_rigid_motion_mirror_and_vertex_order(self):
        report = invariance_checks()
        self.assertLess(report["rotation_0.30_max_abs_dev"], 1e-12)
        self.assertLess(report["rotation_1.10_max_abs_dev"], 1e-12)
        self.assertLess(report["rotation_3.14_max_abs_dev"], 1e-12)
        self.assertLess(report["mirror_max_abs_dev"], 1e-12)
        self.assertLess(report["vertex_reversal_max_abs_dev"], 1e-12)

    def test_degenerate_on_line_contact_is_vertex_order_independent(self):
        """Regression: the fallback normal used to flip with the vertex order."""
        hull = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=DT)
        on_line = torch.zeros((1, 2), dtype=DT)
        forward, _ = polyline_contact_forces(on_line, on_line, hull, ZERO,
                                             particle_radius=0.05, hull_radius=0.0, stiffness=1000.0)
        backward, _ = polyline_contact_forces(on_line, on_line, hull.flip(0), ZERO,
                                              particle_radius=0.05, hull_radius=0.0, stiffness=1000.0)
        torch.testing.assert_close(forward, backward, rtol=0, atol=0)
        self.assertAlmostEqual(float(forward.norm()), 1000.0 * 0.05, places=12)

    def test_degenerate_contact_opposes_the_approach_velocity(self):
        """On-line contact must not drive the disk further along its approach."""
        hull = torch.tensor([[0.0, -0.1], [0.0, 0.2]], dtype=DT)
        hull_velocity = torch.tensor([0.2, 0.0], dtype=DT)
        on_line = torch.zeros((1, 2), dtype=DT)
        forces, _ = polyline_contact_forces(on_line, on_line, hull, hull_velocity,
                                            particle_radius=0.05, hull_radius=0.0, stiffness=1000.0)
        relative_velocity = on_line[0] - hull_velocity
        # The hull advances in +x, so the disk must be pushed in +x, i.e. the force
        # opposes the relative approach velocity instead of following it.
        self.assertLess(float((forces[0] * relative_velocity).sum()), 0.0)
        self.assertGreater(float(forces[0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
