"""Tests for the exact circular-arc hull kernel and the secant error bounds.

The physical claim under test is exact, not asymptotic: for a disk whose centre
lies on the chord normal, a secant bow measures the penetration as
``delta - s`` with ``s = R (1 - cos(dtheta/2))`` the mid-chord sagitta, so the
shipped polyline kernel under-loads by ``s / delta`` and misses contact
completely for ``delta <= s``.  The arc kernel must reproduce the closed-form
circular-bow contact to machine precision.
"""
import math
import unittest

import torch

from tensordem.hull_contact import polyline_contact_forces
from tensordem.hull_geometry import (
    CircularArc,
    analytic_arc_penetration,
    arc_hull_contact_forces,
    arc_polyline,
    polyline_error_bounds,
    sagitta,
    sagitta_corrected_polyline,
    segments_for_tolerance,
)

SPAN = math.pi
START = -0.5 * math.pi


def _disk_position(radius: float, particle_radius: float, theta: float,
                   penetration: float) -> torch.Tensor:
    r = radius + particle_radius - penetration
    return torch.tensor([[r * math.cos(theta), r * math.sin(theta)]], dtype=torch.float64)


def _mid_angle(n_segments: int, segment: int) -> float:
    return START + (segment + 0.5) * (SPAN / n_segments)


class SagittaClosedFormTests(unittest.TestCase):
    """The secant penetration deficit is exactly the sagitta."""

    def test_chord_penetration_deficit_equals_sagitta(self):
        for n_segments in (4, 8, 16, 64):
            for radius_ratio in (2.0, 10.0):
                radius = radius_ratio
                particle_radius = 1.0
                vertices = arc_polyline(radius, SPAN, n_segments, start_angle=START)
                segment = n_segments // 2
                theta = _mid_angle(n_segments, segment)
                s = sagitta(radius, SPAN, n_segments)
                for penetration in (0.2, 0.1, 0.05, 0.02):
                    if penetration <= s:
                        continue
                    position = _disk_position(radius, particle_radius, theta, penetration)
                    force, _ = polyline_contact_forces(
                        position, torch.zeros_like(position), vertices,
                        torch.zeros(2, dtype=torch.float64),
                        particle_radius=particle_radius, hull_radius=0.0, stiffness=1.0)
                    measured_penetration = float(torch.linalg.vector_norm(force))
                    self.assertAlmostEqual(
                        measured_penetration, penetration - s, places=12,
                        msg=f"N={n_segments} R/rp={radius_ratio} d={penetration}")
                    expected_ratio = 1.0 - s / penetration
                    self.assertAlmostEqual(
                        measured_penetration / penetration, expected_ratio, places=12)

    def test_contact_is_lost_below_sagitta(self):
        radius, particle_radius, n_segments = 10.0, 1.0, 16
        vertices = arc_polyline(radius, SPAN, n_segments, start_angle=START)
        theta = _mid_angle(n_segments, n_segments // 2)
        s = sagitta(radius, SPAN, n_segments)
        below = _disk_position(radius, particle_radius, theta, 0.999 * s)
        above = _disk_position(radius, particle_radius, theta, 1.001 * s)
        zero_velocity = torch.zeros((1, 2), dtype=torch.float64)
        force_below, _ = polyline_contact_forces(
            below, zero_velocity, vertices, torch.zeros(2, dtype=torch.float64),
            particle_radius=particle_radius, hull_radius=0.0, stiffness=1.0)
        force_above, _ = polyline_contact_forces(
            above, zero_velocity, vertices, torch.zeros(2, dtype=torch.float64),
            particle_radius=particle_radius, hull_radius=0.0, stiffness=1.0)
        self.assertEqual(float(torch.linalg.vector_norm(force_below)), 0.0)
        self.assertGreater(float(torch.linalg.vector_norm(force_above)), 0.0)

    def test_error_is_second_order_in_segment_count(self):
        radius, particle_radius, penetration = 10.0, 1.0, 0.02
        ratios = []
        previous = None
        for n_segments in (16, 32, 64, 128):
            s = sagitta(radius, SPAN, n_segments)
            error = s / penetration
            if previous is not None:
                ratios.append(previous / error)
            previous = error
        for ratio in ratios:
            self.assertGreater(ratio, 3.9)
            self.assertLess(ratio, 4.1)

    def test_direction_error_is_bounded_by_half_segment_angle(self):
        radius, particle_radius, n_segments, penetration = 10.0, 1.0, 16, 0.05
        vertices = arc_polyline(radius, SPAN, n_segments, start_angle=START)
        segment = n_segments // 2
        bound = 0.5 * (SPAN / n_segments)
        measured = 0.0
        for index in range(41):
            xi = -1.0 + 2.0 * index / 40.0
            theta = _mid_angle(n_segments, segment) + 0.5 * xi * (SPAN / n_segments)
            position = _disk_position(radius, particle_radius, theta, penetration)
            force, _ = polyline_contact_forces(
                position, torch.zeros_like(position), vertices,
                torch.zeros(2, dtype=torch.float64),
                particle_radius=particle_radius, hull_radius=0.0, stiffness=1.0)
            reference = torch.tensor(
                [math.cos(theta), math.sin(theta)], dtype=torch.float64)
            dot = float((force[0] * reference).sum()) / float(
                torch.linalg.vector_norm(force) * torch.linalg.vector_norm(reference))
            measured = max(measured, math.acos(max(-1.0, min(1.0, dot))))
        self.assertLessEqual(measured, bound + 1e-9)
        self.assertGreater(measured, 0.5 * bound)  # the bound is nearly attained


class ArcKernelTests(unittest.TestCase):
    """The exact arc kernel must match the closed-form bow contact."""

    def test_matches_closed_form_force_and_direction(self):
        radius, particle_radius, stiffness = 10.0, 1.0, 250.0
        arc = CircularArc((0.0, 0.0), radius, START, SPAN)
        for penetration in (0.2, 0.05, 0.01, 1e-4):
            for theta in (-1.4, -0.3, 0.0, 0.7, 1.5):
                position = _disk_position(radius, particle_radius, theta, penetration)
                force, reaction = arc_hull_contact_forces(
                    position, torch.zeros_like(position), [arc],
                    torch.zeros(2, dtype=torch.float64),
                    particle_radius=particle_radius, stiffness=stiffness)
                expected = stiffness * analytic_arc_penetration(
                    radius, particle_radius, radius + particle_radius - penetration)
                self.assertAlmostEqual(
                    float(torch.linalg.vector_norm(force)), expected, places=8)
                direction = torch.tensor(
                    [math.cos(theta), math.sin(theta)], dtype=torch.float64)
                torch.testing.assert_close(
                    force[0] / torch.linalg.vector_norm(force), direction,
                    atol=1e-12, rtol=0.0)
                torch.testing.assert_close(
                    force.sum(dim=0) + reaction, torch.zeros(2, dtype=torch.float64),
                    atol=1e-12, rtol=0.0)

    def test_no_force_without_overlap(self):
        arc = CircularArc((0.0, 0.0), 10.0, START, SPAN)
        position = torch.tensor([[13.0, 0.0]], dtype=torch.float64)
        force, reaction = arc_hull_contact_forces(
            position, torch.zeros_like(position), [arc],
            torch.zeros(2, dtype=torch.float64), particle_radius=1.0, stiffness=100.0)
        self.assertEqual(float(torch.linalg.vector_norm(force)), 0.0)
        self.assertEqual(float(torch.linalg.vector_norm(reaction)), 0.0)

    def test_outside_span_falls_back_to_endpoint(self):
        radius, particle_radius, stiffness = 10.0, 1.0, 100.0
        arc = CircularArc((0.0, 0.0), radius, START, SPAN)
        theta = math.radians(135.0)
        penetration = 0.01
        position = _disk_position(radius, particle_radius, theta, penetration)
        endpoint = torch.tensor(
            [radius * math.cos(START + SPAN), radius * math.sin(START + SPAN)],
            dtype=torch.float64)
        distance = float(torch.linalg.vector_norm(position[0] - endpoint))
        force, _ = arc_hull_contact_forces(
            position, torch.zeros_like(position), [arc],
            torch.zeros(2, dtype=torch.float64),
            particle_radius=particle_radius, stiffness=stiffness)
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(force)),
            stiffness * max(0.0, particle_radius - distance), places=8)

    def test_moving_hull_damping_is_symmetric_and_never_attractive(self):
        radius, particle_radius, stiffness, damping = 10.0, 1.0, 100.0, 5.0
        arc = CircularArc((0.0, 0.0), radius, START, SPAN)
        penetration = 0.01
        position = _disk_position(radius, particle_radius, 0.0, penetration)
        into = torch.tensor([[3.0, 0.0]], dtype=torch.float64)
        away = torch.tensor([[-3.0, 0.0]], dtype=torch.float64)
        expected = stiffness * penetration + damping * 3.0
        force_into, _ = arc_hull_contact_forces(
            position, torch.zeros_like(position), [arc], into,
            particle_radius=particle_radius, stiffness=stiffness, damping=damping)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(force_into)), expected, places=9)
        force_away, _ = arc_hull_contact_forces(
            position, torch.zeros_like(position), [arc], away,
            particle_radius=particle_radius, stiffness=stiffness, damping=damping)
        self.assertEqual(float(torch.linalg.vector_norm(force_away)), 0.0)


class SagittaCorrectionTests(unittest.TestCase):
    """The cheap mitigation removes the mid-chord bias but over-loads vertices."""

    def test_correction_is_exact_at_mid_chord(self):
        radius, particle_radius, n_segments, stiffness = 10.0, 1.0, 16, 1.0
        vertices = sagitta_corrected_polyline(radius, SPAN, n_segments, start_angle=START)
        theta = _mid_angle(n_segments, n_segments // 2)
        penetration = 0.05
        position = _disk_position(radius, particle_radius, theta, penetration)
        force, _ = polyline_contact_forces(
            position, torch.zeros_like(position), vertices,
            torch.zeros(2, dtype=torch.float64),
            particle_radius=particle_radius, hull_radius=0.0, stiffness=stiffness)
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(force)), stiffness * penetration, places=9)

    def test_correction_overloads_near_vertices(self):
        radius, particle_radius, n_segments, stiffness = 10.0, 1.0, 16, 1.0
        vertices = sagitta_corrected_polyline(radius, SPAN, n_segments, start_angle=START)
        segment = n_segments // 2
        delta_theta = SPAN / n_segments
        s = sagitta(radius, SPAN, n_segments)
        penetration = 0.05
        # Disk exactly tangling the true arc at the segment joint (vertex).
        theta = START + (segment + 1.0) * delta_theta
        position = _disk_position(radius, particle_radius, theta, penetration)
        force, _ = polyline_contact_forces(
            position, torch.zeros_like(position), vertices,
            torch.zeros(2, dtype=torch.float64),
            particle_radius=particle_radius, hull_radius=0.0, stiffness=stiffness)
        relative_error = float(torch.linalg.vector_norm(force)) / (stiffness * penetration) - 1.0
        self.assertGreater(relative_error, 0.5 * s / penetration)
        # The shipped (uncorrected) polyline is exact at the same vertex.
        plain = arc_polyline(radius, SPAN, n_segments, start_angle=START)
        force_plain, _ = polyline_contact_forces(
            position, torch.zeros_like(position), plain,
            torch.zeros(2, dtype=torch.float64),
            particle_radius=particle_radius, hull_radius=0.0, stiffness=stiffness)
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(force_plain)), stiffness * penetration, places=9)


class ErrorBoundAPITests(unittest.TestCase):
    def test_segments_for_tolerance_brackets_the_bound(self):
        radius, particle_radius, penetration, tolerance = 20.0, 1.0, 0.01, 0.03
        n_required = segments_for_tolerance(radius, SPAN, penetration, tolerance)
        self.assertLessEqual(
            polyline_error_bounds(radius, SPAN, n_required, penetration)["sagitta"],
            penetration * tolerance + 1e-15)
        if n_required > 1:
            self.assertGreater(
                polyline_error_bounds(radius, SPAN, n_required - 1, penetration)["sagitta"],
                penetration * tolerance)

    def test_bounds_report_dropout(self):
        bounds = polyline_error_bounds(10.0, SPAN, 4, 0.001)
        self.assertEqual(bounds["dropout"], 1.0)
        self.assertEqual(bounds["force_ratio"], 0.0)
        self.assertEqual(bounds["relative_force_error"], -1.0)
        bounds = polyline_error_bounds(10.0, SPAN, 256, 0.01)
        self.assertEqual(bounds["dropout"], 0.0)
        self.assertAlmostEqual(
            bounds["relative_force_error"], -bounds["sagitta"] / 0.01, places=12)

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            sagitta(0.0, SPAN, 4)
        with self.assertRaises(ValueError):
            sagitta(1.0, 0.0, 4)
        with self.assertRaises(ValueError):
            sagitta(1.0, SPAN, 0)
        with self.assertRaises(ValueError):
            polyline_error_bounds(1.0, SPAN, 4, -1.0)
        with self.assertRaises(ValueError):
            segments_for_tolerance(1.0, SPAN, 0.01, 1.5)
        with self.assertRaises(ValueError):
            CircularArc((0.0, 0.0), -1.0, 0.0, SPAN)
        position = torch.zeros((1, 2), dtype=torch.float64)
        with self.assertRaises(ValueError):
            arc_hull_contact_forces(
                position, position, [], torch.zeros(2, dtype=torch.float64),
                particle_radius=1.0, stiffness=1.0)
        with self.assertRaises(ValueError):
            arc_hull_contact_forces(
                position, position, [CircularArc((0.0, 0.0), 1.0, 0.0, SPAN)],
                torch.zeros(2, dtype=torch.float64), particle_radius=0.0, stiffness=1.0)


if __name__ == "__main__":
    unittest.main()