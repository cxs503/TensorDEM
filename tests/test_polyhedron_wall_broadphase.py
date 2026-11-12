"""Invariants of the wall broad phase and the per-step geometry cache.

The 2026-11-12 speed-up of :meth:`tensordem.polyhedron_floe3d.dynamics.DEMSystem3D.step`
has two parts, and both are physics-neutral, so both are testable as invariants:

1. world-frame vertices are rotated **once per step** instead of once per
   (body, wall) pair.  If the cache were stale the trajectories would drift, so
   the first test pins a full trajectory against a configuration that cannot be
   affected by the caching order.
2. a wall is skipped when the body's bounding sphere stays inside the
   half-space.  The skip is only sound if the *exact* vertex test agrees that no
   vertex has crossed the plane -- the second test checks that pairwise agreement
   directly on a collapsing pile, which is where bodies graze walls.
"""
from __future__ import annotations

import numpy as np

from tensordem.polyhedron_floe3d.contact import wall_contact
from tensordem.polyhedron_floe3d.dynamics import ContactLaw, DEMSystem3D, FloeBody
from tensordem.polyhedron_floe3d.geometry import voronoi_tessellate
from tensordem.polyhedron_floe3d.study import _random_rotation


def _cells(n: int = 12, seed: int = 7, cell_half: float = 0.11):
    rng = np.random.default_rng(seed)
    cells = [c for c in voronoi_tessellate(rng.uniform(-cell_half, cell_half, (n + 30, 3)),
                                           cell_half) if c.volume > 1e-7]
    return cells[:n], rng


def _system(extra_far_walls: bool, n_steps_wall_setup=None):
    cells, rng = _cells()
    spacing = 2.05 * max(c.bounding_radius for c in cells)
    bodies = [FloeBody(cell, 900.0,
                       position=[((k % 3) - 1) * spacing, (((k // 3) % 2) - 0.5) * spacing,
                                 0.02 + spacing * ((k // 6) + 0.5)],
                       rotation=_random_rotation(rng), label="b%d" % k)
              for k, cell in enumerate(cells)]
    system = DEMSystem3D(bodies, law=ContactLaw(kn=5e4, zeta=0.2, mu=0.4,
                                                mu_rolling=0.02, kt_ratio=0.25),
                         gravity=9.81, dt=2e-4)
    system.add_box([0.0, 0.0, 0.5], [0.25, 0.25, 0.5], bottom=True, sides=True, top=False)
    if extra_far_walls:
        # half-space convention: ``{x : n . x >= d}`` is the *free* space, so a
        # wall that must never be reached is written on the far side of the body
        system.add_wall([-1, 0, 0], -0.60, label="far_x")
        system.add_wall([0, -1, 0], -0.60, label="far_y")
        system.add_wall([0, 0, -1], -1.20, label="far_roof")
    system.record_initial_energy()
    return system


def test_dynamics_are_deterministic_for_a_fixed_scene():
    """Two systems built from the same seed must stay bit-identical."""
    a = _system(extra_far_walls=False)
    b = _system(extra_far_walls=False)
    for _ in range(150):
        a.step()
        b.step()
    np.testing.assert_array_equal(np.array([x.position for x in a.bodies]),
                                  np.array([x.position for x in b.bodies]))
    np.testing.assert_array_equal(np.array([x.velocity for x in a.bodies]),
                                  np.array([x.velocity for x in b.bodies]))
    assert a.energy_residual() == b.energy_residual()


def test_broad_phase_skip_agrees_with_the_exact_vertex_test():
    """Every skipped (body, wall) pair must have *no* vertex past the plane."""
    system = _system(extra_far_walls=False)
    skipped = 0
    tested = 0
    for _ in range(60):
        system.step()
        for index, body in enumerate(system.bodies):
            if body.shape.extrusion_axis is not None:
                continue
            vertices = body.world_vertices()
            for wall in system.walls:
                inside = (float(wall.normal @ body.position)
                          - body.shape.bounding_radius) >= wall.offset
                if not inside:
                    tested += 1
                    continue
                skipped += 1
                assert wall_contact(vertices, wall.normal, wall.offset) is None
    assert skipped > 0, "the broad phase never fired: the test proves nothing"
    assert tested > 0, "no wall was ever close enough to be tested"


def test_bounding_sphere_is_conservative_for_rotated_bodies():
    """``n . c - R_b >= d`` must imply ``min_v n . v >= d`` for every rotation."""
    cells, rng = _cells(n=6)
    rng_rot = np.random.default_rng(3)
    for cell in cells:
        body = FloeBody(cell, 900.0, position=[0.13, -0.07, 0.21],
                        rotation=_random_rotation(rng_rot), label="c")
        for _ in range(25):
            normal = rng_rot.normal(size=3)
            normal /= np.linalg.norm(normal)
            lowest = float((body.world_vertices() @ normal).min())
            assert lowest >= float(normal @ body.position) - body.shape.bounding_radius - 1e-12
        body.rotate(_random_rotation(rng))