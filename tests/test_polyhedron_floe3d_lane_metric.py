"""The C3 lane metric's instrument: ``DEMSystem3D.solid_volume_slab``.

The channel-clearing verdict ("the lane the hull swept is empty") is a ratio of
this number to a nominal layer volume, so the C3 number is only as trustworthy
as the clipper behind it.  ``solid_volume_slab`` was the measurement primitive
for rounds 2-6 without a test of its own; these cases pin it against analytic
volumes of boxes, against partition additivity (the sum over disjoint slabs must
equal the body volume, whatever the axis), and against the centroid-count
``porosity`` proxy it replaced (which is a step function -- the reason it was
replaced).

No case here runs a DEM step: the clipper is a pure geometry query.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensordem.polyhedron_floe3d.dynamics import DEMSystem3D, FloeBody
from tensordem.polyhedron_floe3d.geometry import extrude_polygon


def _box(dx: float, dy: float, dz: float):
    """Axis-aligned box centred on its centroid, ``dx x dy x dz``."""
    poly = np.array([[-dx / 2, -dy / 2], [dx / 2, -dy / 2],
                     [dx / 2, dy / 2], [-dx / 2, dy / 2]])
    return extrude_polygon(poly, dz, z0=-dz / 2)


def _system(shapes, positions, density: float = 900.0) -> DEMSystem3D:
    bodies = [FloeBody(shape=s, density=density, position=p)
              for s, p in zip(shapes, positions)]
    return DEMSystem3D(bodies, gravity=0.0, dt=1.0e-4)


def test_box_slab_matches_analytic_volume():
    """A box of 0.4 x 0.2 x 0.1 at x = 0.3: the slab volume is exact."""
    system = _system([_box(0.4, 0.2, 0.1)], [(0.3, 0.0, 0.0)])
    # x span of the box is [0.1, 0.5]
    cases = [((0.1, 0.2), 0.1 * 0.2 * 0.1),      # fully inside, lower part
             ((0.45, 0.55), 0.05 * 0.2 * 0.1),   # cut by the upper face
             ((0.00, 0.30), 0.2 * 0.2 * 0.1),    # cut by the lower face
             ((-1.0, 1.0), 0.4 * 0.2 * 0.1),     # whole body
             ((0.5, 0.6), 0.0),                  # outside
             ((0.2, 0.2), 0.0)]                  # degenerate (zero thickness)
    for (lo, hi), ref in cases:
        got = system.solid_volume_slab(0, lo, hi)
        assert got == pytest.approx(ref, abs=1e-14), (lo, hi)


def test_slab_volume_is_additive_over_a_partition():
    """Summing disjoint slabs must return the body volume, on every axis."""
    shape = _box(0.4, 0.2, 0.1)
    system = _system([shape], [(0.3, 0.0, 0.0)])
    edges = np.linspace(-0.2, 0.8, 21)
    parts = [system.solid_volume_slab(0, edges[i], edges[i + 1])
             for i in range(edges.size - 1)]
    assert sum(parts) == pytest.approx(shape.volume, rel=1e-12)
    assert all(p >= 0.0 for p in parts)
    for axis in (1, 2):
        assert system.solid_volume_slab(axis, -1.0, 1.0) == pytest.approx(
            shape.volume, rel=1e-12)


def test_slab_volume_sums_multiple_bodies():
    """Two boxes straddling the two slab faces contribute half each."""
    system = _system([_box(0.2, 0.2, 0.2), _box(0.2, 0.2, 0.2)],
                     [(-0.1, 0.0, 0.0), (0.1, 0.0, 0.0)])
    # each box spans [-0.2, 0.0] and [0.0, 0.2]; the slab [-0.1, 0.1] takes half of each
    assert system.solid_volume_slab(0, -0.1, 0.1) == pytest.approx(
        2 * 0.1 * 0.2 * 0.2, rel=1e-12)


def test_slab_volume_does_not_count_a_body_by_its_centroid():
    """A floe whose centroid is outside the slab still contributes its inside part.

    This is the defect that made the old channel observable a step function: the
    centroid-count ``porosity`` probe reports 0 for this body while half of it is
    inside the swath.
    """
    system = _system([_box(0.4, 0.2, 0.1)], [(0.3, 0.0, 0.0)])
    lo, hi = 0.34, 0.6          # centroid x = 0.3 lies outside this slab
    exact = system.solid_volume_slab(0, lo, hi)
    assert exact == pytest.approx((0.5 - 0.34) * 0.2 * 0.1, rel=1e-12)
    centroid_proxy = system.porosity([lo, -0.5, -0.5], [hi, 0.5, 0.5])
    assert centroid_proxy == pytest.approx(1.0)      # nothing counted (solid fraction 0)
    assert exact > 0.0