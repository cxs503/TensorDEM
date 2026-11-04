"""Verification: degenerate (zero-area) triangles must not silently drop hull-ice contact.

Defect pinned here
------------------
``hull_mesh_contact._closest_point_triangle_batch`` follows the Ericson region
scheme.  The interior region uses ``denom = (va + vb + vc).clamp_min(eps)``: for a
zero-area triangle that denominator is *zero* and the clamp turns it into
``eps``, so ``v_int = vb / eps`` explodes and the returned "closest" point lands
far outside the triangle.  The resulting distance exceeds the contact radius and
the penalty force for that triangle is silently zero.

Such triangles are not exotic: duplicate vertices, collinear vertices and slivers
are routine in CAD/STL hull exports, and a dropped triangle means a missed
contact -- an unphysical hole in the ship hull at exactly the ice-contact patch.

Reference values
----------------
Every expectation below is an independently coded closed form (distance to a
segment / to a vertex), never the routine under test:

  * distance from P to segment AB: ``|P - (A + clamp(t,0,1) AB)|``, ``t = AP.AB/|AB|^2``
  * distance from P to a point A: ``|P - A|``
  * an AABB-aligned unit right triangle has known interior projections.

The non-degenerate path must stay bit-identical: the fallback only rewrites rows
flagged degenerate, so the valid-triangle cases below are pure regression pins.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensordem.hull_mesh_contact import (  # noqa: E402
    _SpatialHash,
    _closest_point_segment_batch,
    _closest_point_triangle_batch,
    make_flat_mesh,
    mesh_contact_forces,
)

DT = torch.float64


def segment_distance(point, a, b) -> float:
    """Independent closed form: distance from a point to a segment."""
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0.0:
        return float(torch.linalg.vector_norm(point - a))
    t = float((point - a) @ ab) / denom
    t = min(1.0, max(0.0, t))
    return float(torch.linalg.vector_norm(point - (a + t * ab)))


def closest(P, A, B, C, **kwargs):
    return _closest_point_triangle_batch(
        P.reshape(1, 3), A.reshape(1, 3), B.reshape(1, 3), C.reshape(1, 3), **kwargs
    ).reshape(3)


def t(x, y, z):
    return torch.tensor([x, y, z], dtype=DT)


class TestDegenerateTrianglesMatchTheirSegment:
    """A zero-area triangle *is* a segment (or a point); distances must agree."""

    @pytest.mark.parametrize(
        "query",
        [(0.3, 1.0, 0.2), (-2.0, 4.0, -3.0), (0.5, 1.0, 0.0), (7.0, -1.0, 0.25)],
    )
    def test_duplicate_vertex_triangle(self, query):
        """A == B (collapsed to a point) and C distinct -> segment AC."""
        P = t(*query)
        A = t(0.0, 1.0, 0.0)
        B = A.clone()
        C = t(1.0, 1.0, 0.0)
        got = closest(P, A, B, C)
        analytic = segment_distance(P, A, C)
        assert float(torch.linalg.vector_norm(got - P)) == pytest.approx(analytic, abs=1e-12)

    @pytest.mark.parametrize(
        "query",
        [(0.5, 0.3, 0.4), (2.5, -1.0, 0.0), (-4.0, 0.0, 0.0)],
    )
    def test_collinear_triangle(self, query):
        """A, B, C collinear -> the triangle degenerates to segment A-C."""
        P = t(*query)
        A = t(0.0, 0.0, 0.0)
        B = t(1.0, 0.0, 0.0)
        C = t(2.0, 0.0, 0.0)
        got = closest(P, A, B, C)
        analytic = segment_distance(P, A, C)
        assert float(torch.linalg.vector_norm(got - P)) == pytest.approx(analytic, abs=1e-12)

    def test_fully_collapsed_triangle_is_a_point(self):
        P = t(1.0, 2.0, 3.0)
        A = t(0.5, 0.5, 0.5)
        got = closest(P, A, A.clone(), A.clone())
        assert torch.allclose(got, A, atol=1e-15)

    def test_degenerate_row_is_not_answered_by_the_interior_formula(self):
        """Regression pin: the unfixed interior formula returned |P - q| >> true distance."""
        P = t(0.3, 1.0, 0.2)
        A = t(0.0, 1.0, 0.0)
        B = A.clone()
        C = t(1.0, 1.0, 0.0)
        got = closest(P, A, B, C)
        assert float(torch.linalg.vector_norm(got - P)) < 0.201
        # the buggy branch is reproducible on demand and is far worse
        with_area = closest(P, A, B, C, relative_area_tolerance=0.0)
        assert float(torch.linalg.vector_norm(got - P)) <= float(
            torch.linalg.vector_norm(with_area - P)
        ) + 1e-12

    def test_negative_tolerance_rejected(self):
        P, A, B, C = t(1.0, 1.0, 1.0), t(0.0, 0.0, 0.0), t(1.0, 0.0, 0.0), t(0.0, 1.0, 0.0)
        with pytest.raises(ValueError):
            closest(P, A, B, C, relative_area_tolerance=-1e-9)


class TestNonDegeneratePathUnchanged:
    """The fallback must not move a single valid-triangle closest point."""

    def test_unit_right_triangle_interior_projection(self):
        A, B, C = t(0.0, 0.0, 0.0), t(1.0, 0.0, 0.0), t(0.0, 1.0, 0.0)
        P = t(0.2, 0.3, 2.0)
        got = closest(P, A, B, C)
        assert torch.allclose(got, t(0.2, 0.3, 0.0), atol=1e-14)
        assert float(torch.linalg.vector_norm(got - P)) == pytest.approx(2.0, abs=1e-14)

    def test_vertex_and_edge_regions(self):
        A, B, C = t(0.0, 0.0, 0.0), t(1.0, 0.0, 0.0), t(0.0, 1.0, 0.0)
        assert torch.allclose(closest(t(-1.0, -1.0, 0.0), A, B, C), A, atol=1e-14)
        assert torch.allclose(closest(t(2.0, -1.0, 0.0), A, B, C), B, atol=1e-14)
        assert torch.allclose(closest(t(-1.0, 2.0, 0.0), A, B, C), C, atol=1e-14)
        edge = closest(t(0.45, 0.1, 0.0), A, B, C)
        assert torch.allclose(edge, t(0.45, 0.1, 0.0), atol=1e-14)

    def test_segment_helper_matches_closed_form(self):
        P = torch.rand(64, 3, dtype=DT) * 4.0 - 2.0
        A = torch.rand(64, 3, dtype=DT)
        B = torch.rand(64, 3, dtype=DT)
        got = _closest_point_segment_batch(P, A, B)
        for i in range(64):
            expected = segment_distance(P[i], A[i], B[i])
            assert float(torch.linalg.vector_norm(got[i] - P[i])) == pytest.approx(expected, abs=1e-12)


class TestContactForceIsNotDropped:
    """Physics-level pin: a sliver face must still carry a penalty contact force."""

    def test_sphere_over_duplicate_vertex_face_carries_force(self):
        vertices = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=DT
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        radius, stiffness = 0.25, 1.0e6
        position = t(0.3, 1.0, 0.2)
        penetration = radius - segment_distance(position, vertices[0], vertices[2])
        assert penetration == pytest.approx(0.05, abs=1e-14)
        force, reaction = mesh_contact_forces(
            position.reshape(1, 3),
            torch.zeros(1, 3, dtype=DT),
            vertices,
            faces,
            torch.zeros(3, dtype=DT),
            particle_radius=radius,
            stiffness=stiffness,
        )
        magnitude = float(torch.linalg.vector_norm(force))
        assert magnitude > 0.0, "degenerate face silently dropped the contact"
        assert magnitude == pytest.approx(stiffness * penetration, rel=1e-9)
        assert torch.allclose(reaction, -force.sum(dim=0), atol=1e-9)

    def test_collinear_face_carries_force(self):
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=DT
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        radius, stiffness = 0.25, 1.0e6
        position = t(0.5, 0.0, 0.2)
        force, _ = mesh_contact_forces(
            position.reshape(1, 3),
            torch.zeros(1, 3, dtype=DT),
            vertices,
            faces,
            torch.zeros(3, dtype=DT),
            particle_radius=radius,
            stiffness=stiffness,
        )
        assert float(torch.linalg.vector_norm(force)) == pytest.approx(
            stiffness * (radius - 0.2), rel=1e-9
        )


class TestBroadPhaseCompleteness:
    """The spatial hash must be a superset filter: a missed face is a missed contact."""

    def test_candidates_are_a_superset_of_brute_force(self):
        vertices, faces = make_flat_mesh(size=2.0, nx=6, ny=6)
        generator = torch.Generator().manual_seed(20261103)
        for trial in range(5):
            centres = torch.rand(120, 3, generator=generator, dtype=DT) * 1.6 - 0.8
            centres[:, 2] = 0.0
            radius = 0.05 + 0.25 * float(torch.rand(1, generator=generator))
            hash_ = _SpatialHash(vertices, faces, max(radius, 1e-6))
            candidates = hash_.query(centres, radius)
            for i in range(len(centres)):
                required = {
                    fi
                    for fi in range(len(faces))
                    if float(torch.linalg.vector_norm(vertices[faces[fi]] - centres[i], dim=1).min())
                    <= radius + 1e-12
                }
                assert required <= set(candidates[i]), f"trial {trial} sphere {i} missed faces"

    def test_empty_envelope_returns_no_candidates(self):
        vertices, faces = make_flat_mesh(size=1.0, nx=2, ny=2)
        hash_ = _SpatialHash(vertices, faces, 0.5)
        far = torch.tensor([[10.0, 10.0, 10.0]], dtype=DT)
        assert hash_.query(far, 0.1) == [[]]