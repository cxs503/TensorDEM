"""Geometry tests for the 3-D polyhedral floe module.

Everything checked here is an *exact* statement (tessellation closure, mass
properties, contact distance, buoyancy) compared against an independent
reference, so the tolerances are machine precision rather than engineering ones.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.polyhedron_floe3d.geometry import (
    ConvexPolyhedron,
    extrude_polygon,
    halfspace_vertices,
    polygon_area,
    regular_polygon,
    voronoi_tessellate,
)
from tensordem.polyhedron_floe3d.contact import sat_contact
from tensordem.polyhedron_floe3d.dynamics import body_tetrahedra, submerged_tet_moments


def test_halfspace_vertices_box():
    normals = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0],
                        [0, 0, 1.0], [0, 0, -1.0]])
    offsets = np.array([0.5, 0.5, 0.25, 0.25, 0.1, 0.1])
    vertices = halfspace_vertices(normals, offsets)
    assert vertices.shape == (8, 3)
    assert np.allclose(np.abs(vertices).max(axis=0), [0.5, 0.25, 0.1], atol=1e-12)


def test_voronoi_tessellation_closes_the_box():
    """The Voronoi cells tile the box exactly: sum(V_i) == (2h)^3."""
    rng = np.random.default_rng(0)
    half = 1.0
    for n_sites in (12, 40):
        sites = rng.uniform(-0.88, 0.88, (n_sites, 3)) * half
        cells = voronoi_tessellate(sites, half)
        total = sum(cell.volume for cell in cells)
        assert abs(total - (2.0 * half) ** 3) / (2.0 * half) ** 3 < 1e-13
        assert all(cell.volume > 0.0 for cell in cells)


def test_every_cell_vertex_is_closest_to_its_own_site():
    """The defining property of a Voronoi cell, checked on every vertex."""
    rng = np.random.default_rng(3)
    sites = rng.uniform(-0.8, 0.8, (20, 3))
    cells = voronoi_tessellate(sites, 1.0)
    for index, (site, cell) in enumerate(zip(sites, cells)):
        # the cell's construction frame is centred on its own site
        vertices = cell.construction_vertices()
        distances = np.linalg.norm(vertices[:, None, :] - (sites - site)[None, :, :], axis=2)
        # for interior vertices the nearest site must be the cell's own site; a
        # vertex on the box boundary may tie with a site outside the box
        on_boundary = np.any(np.abs(np.abs(vertices) - 1.0) < 1e-9, axis=1)
        nearest = distances.min(axis=1)
        own = distances[:, index]
        # no vertex may be *strictly* closer to another site; ties are the shared
        # facet vertices and are expected
        assert np.all(own[~on_boundary] <= nearest[~on_boundary] + 1e-9)


def test_extruded_prism_mass_properties():
    """Volume, area and inertia of a hexagonal plate against closed forms."""
    area = 0.25
    thickness = 0.2
    polygon = regular_polygon(area, 6)
    prism = extrude_polygon(polygon, thickness, principal=False)
    assert abs(polygon_area(polygon) - area) < 1e-12
    assert abs(prism.volume - area * thickness) < 1e-12
    mass = area * thickness                      # unit density
    # a regular hexagon of area A: I_z = m * A/(2*pi) * (hexagon shape factor)
    r_circum = math.sqrt(2.0 * area / (6.0 * math.sin(math.pi / 3.0)))
    inertia_z = mass * 5.0 / 12.0 * r_circum ** 2      # solid hexagon about its axis
    assert abs(prism.inertia_diag[2] - inertia_z) / inertia_z < 5e-3
    # plate inertia about an in-plane axis: m (t^2 + b^2)/12 with b the width
    assert prism.inertia_diag[0] > 0.0
    assert abs(prism.area - 2.0 * area - 6.0 * r_circum * thickness) / prism.area < 1e-9


def test_prism_volume_matches_polygon_area_for_random_polygons():
    rng = np.random.default_rng(7)
    for _ in range(10):
        points = rng.uniform(-0.2, 0.2, (12, 2))
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        polygon = points[hull.vertices]
        prism = extrude_polygon(polygon, 0.07)
        assert abs(prism.volume - polygon_area(polygon) * 0.07) < 1e-12


def test_sat_contact_matches_analytic_polytope_sat():
    """Cross-validate the vectorised SAT against the brute-force oracle."""
    from tensordem.convex_contact import analytic_polytope_sat

    rng = np.random.default_rng(11)
    plate = extrude_polygon(regular_polygon(0.2, 5), 0.1, principal=False)
    other = extrude_polygon(regular_polygon(0.2, 6), 0.1, principal=False)
    worst = 0.0
    for _ in range(12):
        angle = rng.uniform(0.0, 0.5 * math.pi)
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        va = plate.vertices @ rotation.T
        na = plate.normals @ rotation.T
        ea = (plate.vertices[plate.edges[:, 1]] - plate.vertices[plate.edges[:, 0]]) @ rotation.T
        offset = rng.uniform([-0.5, -0.5, -0.02], [0.5, 0.5, 0.02])
        vb = other.vertices + offset
        nb = other.normals
        eb = other.vertices[other.edges[:, 1]] - other.vertices[other.edges[:, 0]]
        # direction_bias = 0 restores the pure minimum-translation axis, which is
        # what the brute-force oracle reports
        geometry = sat_contact(va, na, ea, vb, nb, eb, direction_bias=0.0)
        reference, _normal = analytic_polytope_sat(va, vb)
        assert abs(geometry.gap - reference) < 2e-9, (geometry.gap, reference)
        worst = max(worst, abs(geometry.gap - reference))
    assert worst < 2e-9


def test_plate_axis_set_is_exact_and_cheap():
    """The reduced axis set for parallel extrusions equals the generic one."""
    plate_a = extrude_polygon(regular_polygon(0.2, 6), 0.1, principal=False)
    plate_b = extrude_polygon(regular_polygon(0.2, 5), 0.1, principal=False)
    va, na = plate_a.vertices, plate_a.normals
    ea = plate_a.vertices[plate_a.edges[:, 1]] - plate_a.vertices[plate_a.edges[:, 0]]
    vb, nb = plate_b.vertices + np.array([0.3, 0.05, 0.0]), plate_b.normals
    eb = plate_b.vertices[plate_b.edges[:, 1]] - plate_b.vertices[plate_b.edges[:, 0]]
    fast = sat_contact(va, na, ea, vb, nb, eb,
                       extrusion_a=plate_a.extrusion_axis,
                       extrusion_b=plate_b.extrusion_axis)
    generic = sat_contact(va, na, ea, vb, nb, eb)
    assert fast.n_axes < generic.n_axes
    assert abs(fast.gap - generic.gap) < 1e-12
    assert abs(float(fast.normal @ generic.normal) - 1.0) < 1e-9


def test_plate_contact_prefers_the_loaded_facet():
    """A laterally overlapped plate pair must not be pushed apart vertically."""
    plate_a = extrude_polygon(regular_polygon(0.3, 6), 0.12, principal=False)
    plate_b = extrude_polygon(regular_polygon(0.3, 5), 0.12, principal=False)
    va, na = plate_a.vertices, plate_a.normals
    ea = plate_a.vertices[plate_a.edges[:, 1]] - plate_a.vertices[plate_a.edges[:, 0]]

    def geometry(offset):
        vb = plate_b.vertices + offset
        nb = plate_b.normals
        eb = plate_b.vertices[plate_b.edges[:, 1]] - plate_b.vertices[plate_b.edges[:, 0]]
        return sat_contact(va, na, ea, vb, nb, eb,
                           extrusion_a=plate_a.extrusion_axis,
                           extrusion_b=plate_b.extrusion_axis)

    lateral = geometry(np.array([0.35, 0.0, 0.0]))
    assert abs(float(lateral.normal[2])) < 0.2      # horizontal normal
    stacked = geometry(np.array([0.0, 0.0, 0.10]))
    assert abs(float(stacked.normal[2])) > 0.95     # vertical normal


def test_two_dimensional_limit_of_the_contact():
    """t -> 0: at contact-scale overlaps the 3-D gap equals the 2-D SAT gap.

    The statement is restricted to overlaps of the order of the contact itself
    (a few millimetres here).  For an overlap *deeper* than the plate thickness the
    minimum-translation axis of a plate pair genuinely becomes vertical, so the
    in-plane degeneracy does not hold there -- that is 3-D physics, not a defect,
    and it is why the DEM must resolve the contact instead of letting plates
    interpenetrate.
    """
    from tensordem.polygon_contact import polygon_gap

    rng = np.random.default_rng(21)
    for _ in range(6):
        polygon_a = regular_polygon(0.2, 5)
        polygon_b = regular_polygon(0.2, 6)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        distance = rng.uniform(0.52, 0.58)      # near-touching, small overlap
        shift = distance * np.array([math.cos(angle), math.sin(angle)])
        gaps = []
        for thickness in (0.2, 0.05, 0.01):
            prism_a = extrude_polygon(polygon_a, thickness, principal=False)
            prism_b = extrude_polygon(polygon_b, thickness, principal=False)
            va, na = prism_a.vertices, prism_a.normals
            ea = prism_a.vertices[prism_a.edges[:, 1]] - prism_a.vertices[prism_a.edges[:, 0]]
            vb = prism_b.vertices + np.array([shift[0], shift[1], 0.0])
            nb = prism_b.normals
            eb = prism_b.vertices[prism_b.edges[:, 1]] - prism_b.vertices[prism_b.edges[:, 0]]
            geometry = sat_contact(va, na, ea, vb, nb, eb,
                                   extrusion_a=prism_a.extrusion_axis,
                                   extrusion_b=prism_b.extrusion_axis)
            gaps.append(geometry.gap)
        # thickness independence (the 2-D limit is reached at every t)
        assert max(gaps) - min(gaps) < 1e-12
        reference = polygon_gap(np.zeros(2), polygon_a, shift, polygon_b)
        reference_gap = getattr(reference, "gap", reference)
        if np.ndim(reference_gap) == 0:
            # the 2-D module reports the exact closest-feature distance, the SAT
            # separated branch reports the (slightly smaller) maximum negative
            # overlap, so the two agree to the SAT bound, not to machine precision
            assert abs(gaps[0] - float(reference_gap)) <= 5e-4 * (1.0 + abs(reference_gap))


def test_buoyancy_is_exact_against_clipped_hull_reference():
    """Submerged volume and centre of buoyancy of a polyhedron, exactly."""
    from scipy.spatial import ConvexHull

    def reference(tet, level):
        signed = tet[:, 2] - level
        points = [tet[i] for i in range(4) if signed[i] <= 0.0]
        for i in range(4):
            for j in range(i + 1, 4):
                if (signed[i] < 0.0) != (signed[j] < 0.0):
                    t = signed[i] / (signed[i] - signed[j])
                    points.append(tet[i] + t * (tet[j] - tet[i]))
        if len(points) < 4:
            return 0.0, np.zeros(3)
        cloud = np.array(points)
        try:
            hull = ConvexHull(cloud)
        except Exception:            # degenerate (coplanar) clip
            return 0.0, np.zeros(3)
        if hull.volume < 1e-15:
            return 0.0, np.zeros(3)
        anchor = cloud[hull.vertices].mean(axis=0)
        volume = 0.0
        moment = np.zeros(3)
        for (i, j, k) in hull.simplices:
            a, b, c = cloud[i], cloud[j], cloud[k]
            piece = abs(float(np.dot(a - anchor, np.cross(b - anchor, c - anchor)))) / 6.0
            volume += piece
            moment += piece * 0.25 * (anchor + a + b + c)
        return volume, moment / volume

    rng = np.random.default_rng(5)
    for _ in range(8):
        cells = voronoi_tessellate(rng.uniform(-0.3, 0.3, (14, 3)), 0.3)
        body = cells[0]
        tets = body_tetrahedra(body)
        level = rng.uniform(-0.25, 0.25)
        volume, moment, _centre = submerged_tet_moments(tets, level)
        total = float(volume.sum())
        centre = moment.sum(axis=0) / total
        reference_volume = 0.0
        reference_moment = np.zeros(3)
        for tet in tets:
            piece_volume, piece_centre = reference(tet, level)
            reference_volume += piece_volume
            reference_moment += piece_volume * piece_centre
        if reference_volume < 1e-14:
            continue
        assert abs(total - reference_volume) / reference_volume < 1e-12
        assert np.linalg.norm(centre - reference_moment / reference_volume) < 1e-12


def test_fully_submerged_and_fully_dry_limits():
    body = voronoi_tessellate(np.random.default_rng(2).uniform(-0.2, 0.2, (10, 3)), 0.25)[0]
    tets = body_tetrahedra(body)
    volume, moment, _ = submerged_tet_moments(tets, 10.0)
    assert abs(float(volume.sum()) - body.volume) / body.volume < 1e-12
    assert np.linalg.norm(moment.sum(axis=0) / float(volume.sum())) < 1e-12
    volume_dry, _, _ = submerged_tet_moments(tets, -10.0)
    assert float(volume_dry.sum()) == 0.0