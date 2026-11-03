"""Sagitta (facet-plane) bias of the 3-D triangle-mesh hull contact kernel.

Every reference value below is the closed form for an inscribed polyhedron, coded
locally and never taken from the module under test:

    s = R (1 - cos(pi / facets)),   delta_facet = delta_true - s,
    rel. force error = -s / delta_true,   contact lost for delta_true <= s.

The study that produced the tables in ``docs/dem3d_hull_mesh_curvature_sagitta.md``
is ``scripts/audit_hull_mesh_curvature_sagitta.py``; these tests keep the claims
cheap to re-check in CI.
"""
from __future__ import annotations

import math
import unittest

import torch

from tensordem.hull_contact import polyline_contact_forces
from tensordem.hull_mesh_contact import (
    analytic_cylinder_penetration,
    facet_sagitta,
    make_cylinder_mesh,
    mesh_contact_forces,
    required_facets,
)

RADIUS = 10.0
PARTICLE = 1.0
STIFFNESS = 1.0


def _force_3d(radius: float, facets: int, delta: float, fraction: float,
              *, offset: float = 0.0) -> float:
    """Force magnitude at face fraction ``fraction`` (0 = vertex, 0.5 = mid-chord)."""
    vertices, faces = make_cylinder_mesh(radius, 4.0 * PARTICLE, facets)
    angle = fraction * 2.0 * math.pi / facets
    r = radius + PARTICLE - delta
    position = torch.tensor([[r * math.cos(angle), r * math.sin(angle), 0.0]],
                            dtype=torch.float64)
    forces, _ = mesh_contact_forces(
        position, torch.zeros_like(position), vertices, faces,
        torch.zeros(3, dtype=torch.float64),
        particle_radius=PARTICLE, hull_radius=offset,
        stiffness=STIFFNESS, damping=0.0)
    return float(torch.linalg.vector_norm(forces[0]))


def _force_2d(radius: float, facets: int, delta: float, fraction: float) -> float:
    angles = 2.0 * math.pi * torch.arange(facets, dtype=torch.float64) / facets
    ring = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=-1)
    vertices = torch.cat([ring, ring[:1]], dim=0)
    angle = fraction * 2.0 * math.pi / facets
    r = radius + PARTICLE - delta
    position = torch.tensor([[r * math.cos(angle), r * math.sin(angle)]], dtype=torch.float64)
    forces, _ = polyline_contact_forces(
        position, torch.zeros_like(position), vertices,
        torch.zeros(2, dtype=torch.float64),
        particle_radius=PARTICLE, hull_radius=0.0,
        stiffness=STIFFNESS, damping=0.0)
    return float(torch.linalg.vector_norm(forces[0]))


class SagittaClosedFormTests(unittest.TestCase):
    def test_sagitta_matches_independent_closed_form(self) -> None:
        for radius in (2.0, 5.0, 10.0, 20.0):
            for facets in (4, 8, 16, 32, 64, 128, 256, 512):
                reference = radius * (1.0 - math.cos(math.pi / facets))
                self.assertLess(abs(facet_sagitta(radius, facets) / reference - 1.0), 1e-14)

    def test_sagitta_is_second_order_in_facet_count(self) -> None:
        # Doubling the facet count must quarter the bias (asymptotically).
        ratio = facet_sagitta(RADIUS, 64) / facet_sagitta(RADIUS, 128)
        self.assertAlmostEqual(ratio, 4.0, places=2)

    def test_invalid_inputs_rejected(self) -> None:
        for kwargs in ({"radius": 0.0, "facets": 16}, {"radius": -1.0, "facets": 16},
                       {"radius": float("nan"), "facets": 16},
                       {"radius": 1.0, "facets": 2}, {"radius": 1.0, "facets": 16.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                facet_sagitta(**kwargs)


class RequiredFacetsTests(unittest.TestCase):
    def test_returns_the_smallest_facet_count_meeting_tolerance(self) -> None:
        for radius in (2.0, 10.0, 20.0):
            for delta in (0.002, 0.01, 0.05, 0.2):
                needed = required_facets(radius, delta, 0.03)
                self.assertLessEqual(facet_sagitta(radius, needed) / delta, 0.03)
                if needed > 3:
                    self.assertGreater(facet_sagitta(radius, needed - 1) / delta, 0.03)

    def test_matches_brute_force_search(self) -> None:
        radius, delta, tolerance = 10.0, 0.05, 0.03
        brute = next(n for n in range(3, 5000)
                     if facet_sagitta(radius, n) / delta <= tolerance)
        self.assertEqual(required_facets(radius, delta, tolerance), brute)

    def test_invalid_inputs_rejected(self) -> None:
        for kwargs in ({"radius": 0.0, "overlap": 0.1, "tolerance": 0.03},
                       {"radius": 1.0, "overlap": -0.1, "tolerance": 0.03},
                       {"radius": 1.0, "overlap": 0.1, "tolerance": 0.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                required_facets(**kwargs)


class CylinderMeshTests(unittest.TestCase):
    def test_vertices_lie_on_the_circle_and_faces_are_two_per_facet(self) -> None:
        radius, facets = 3.0, 12
        vertices, faces = make_cylinder_mesh(radius, 2.0, facets)
        radial = torch.linalg.vector_norm(vertices[:, :2], dim=1)
        self.assertTrue(torch.allclose(radial, torch.full_like(radial, radius), atol=1e-15))
        self.assertEqual(vertices.shape, (2 * facets, 3))
        self.assertEqual(faces.shape, (2 * facets, 3))
        # Every mesh edge of the side wall is a chord of length 2 R sin(pi/N).
        chord = 2.0 * radius * math.sin(math.pi / facets)
        edge = torch.linalg.vector_norm(vertices[1] - vertices[0])
        self.assertAlmostEqual(float(edge), chord, places=12)

    def test_closed_option_adds_caps(self) -> None:
        _, faces = make_cylinder_mesh(1.0, 1.0, 8, closed=True)
        self.assertEqual(faces.shape, (4 * 8, 3))

    def test_invalid_inputs_rejected(self) -> None:
        for kwargs in ({"radius": 0.0, "length": 1.0, "facets": 8},
                       {"radius": 1.0, "length": -1.0, "facets": 8},
                       {"radius": 1.0, "length": 1.0, "facets": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_cylinder_mesh(**kwargs)


class KernelSagittaBiasTests(unittest.TestCase):
    def test_mid_chord_force_error_is_minus_s_over_delta(self) -> None:
        # delta must exceed the sagitta, otherwise the contact is lost (tested below)
        for facets, delta in ((32, 0.05), (64, 0.05), (128, 0.02), (256, 0.02)):
                s = facet_sagitta(RADIUS, facets)
                measured = _force_3d(RADIUS, facets, delta, 0.5) / delta - 1.0
                self.assertLess(abs(measured - (-s / delta)), 1e-9)

    def test_vertex_contact_carries_no_sagitta_bias(self) -> None:
        for facets in (8, 32, 128):
            measured = _force_3d(RADIUS, facets, 0.01, 0.0) / 0.01 - 1.0
            self.assertLess(abs(measured), 1e-9)

    def test_contact_is_lost_when_overlap_is_below_the_sagitta(self) -> None:
        facets = 16
        s = facet_sagitta(RADIUS, facets)
        self.assertEqual(_force_3d(RADIUS, facets, 0.5 * s, 0.5), 0.0)
        self.assertGreater(_force_3d(RADIUS, facets, 1.5 * s, 0.5), 0.0)

    def test_hull_radius_offset_equal_to_sagitta_cancels_mid_chord_bias(self) -> None:
        for facets in (16, 64):
            s = facet_sagitta(RADIUS, facets)
            corrected = _force_3d(RADIUS, facets, 0.02, 0.5, offset=s) / 0.02 - 1.0
            self.assertLess(abs(corrected), 1e-9)

    def test_3d_mesh_and_2d_polyline_agree_on_the_same_section(self) -> None:
        for facets in (8, 32, 128):
            for fraction in (0.0, 0.5, 1.0):
                three_d = _force_3d(RADIUS, facets, 0.05, fraction)
                two_d = _force_2d(RADIUS, facets, 0.05, fraction)
                self.assertLess(abs(three_d - two_d), 1e-12)

    def test_action_reaction_and_normal_direction(self) -> None:
        vertices, faces = make_cylinder_mesh(RADIUS, 4.0, 16)
        r = RADIUS + PARTICLE - 0.01
        position = torch.tensor([[r, 0.0, 0.0]], dtype=torch.float64)
        forces, reaction = mesh_contact_forces(
            position, torch.zeros_like(position), vertices, faces,
            torch.zeros(3, dtype=torch.float64),
            particle_radius=PARTICLE, stiffness=STIFFNESS, damping=0.0)
        self.assertLess(float(torch.linalg.vector_norm(forces[0] + reaction)), 1e-12)
        self.assertGreater(float(forces[0, 0]), 0.0)
        self.assertAlmostEqual(float(forces[0, 1]), 0.0, places=12)


class AnalyticReferenceTests(unittest.TestCase):
    def test_analytic_penetration_and_clamping(self) -> None:
        self.assertAlmostEqual(analytic_cylinder_penetration(10.05, 0.1, 10.0), 0.05, places=12)
        self.assertEqual(analytic_cylinder_penetration(10.2, 0.1, 10.0), 0.0)

    def test_analytic_penetration_rejects_bad_inputs(self) -> None:
        for kwargs in ({"point_radius": 0.0, "particle_radius": 0.1, "cylinder_radius": 1.0},
                       {"point_radius": 1.0, "particle_radius": -0.1, "cylinder_radius": 1.0},
                       {"point_radius": 1.0, "particle_radius": 0.1, "cylinder_radius": 0.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                analytic_cylinder_penetration(**kwargs)


if __name__ == "__main__":
    unittest.main()