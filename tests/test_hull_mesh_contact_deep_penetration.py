"""Regression surface for the deep-penetration contact defect (m16872).

Pins the defect that the shipped ``closed_surface`` path did **not** fix even
though both the module docstring and ``docs/hull_mesh_contact_signed_distance.md``
§1 promised a restoring force "at any penetration depth": the no-overlap early
exit ran *before* the inside/outside test, so every centre deeper than one
contact radius was silently dropped (17 of 48 sweep points on a capped
cylinder).  Numbers come from ``scripts/hull_signed_deep_penetration_audit.py``
(artifacts ``runs/m16872/sweep_before.json`` / ``sweep_after.json``).
"""
from __future__ import annotations

import math

import torch

from tensordem.hull_mesh_contact import mesh_contact_forces
from tensordem.hull_mesh_contact_signed import is_watertight, ray_parity_inside

F64 = torch.float64
R, K = 0.05, 1.0e6
HALF = 0.5


def box_mesh(half: float = HALF):
    c = [(-half, -half, -half), (half, -half, -half), (half, half, -half),
         (-half, half, -half), (-half, -half, half), (half, -half, half),
         (half, half, half), (-half, half, half)]
    v = torch.tensor(c, dtype=F64)
    f = torch.tensor([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=torch.int64)
    return v, f


def force(centre, verts, faces, closed=True, radius=R, stiffness=K):
    f, reaction = mesh_contact_forces(
        centre.reshape(1, 3), torch.zeros(1, 3, dtype=F64), verts, faces,
        torch.zeros(3, dtype=F64), particle_radius=radius,
        stiffness=stiffness, closed_surface=closed)
    return f[0], reaction


def centre_at_depth(depth: float):
    """Sphere centre ``depth`` metres below the top face (positive = inside)."""
    return torch.tensor([0.0, 0.0, HALF - depth], dtype=F64)


def test_box_mesh_is_watertight():
    verts, faces = box_mesh()
    assert is_watertight(verts, faces)


def test_outside_branch_unchanged():
    """Historical one-sided behaviour must be reproduced bit for bit."""
    verts, faces = box_mesh()
    f_open, _ = force(centre_at_depth(-0.04), verts, faces, closed=False)
    f_closed, _ = force(centre_at_depth(-0.04), verts, faces, closed=True)
    assert abs(float(f_open[2]) - 10000.0) < 1e-6
    assert float(f_open[2]) == float(f_closed[2])   # bit-identical, no sign change


def test_legacy_signed_path_dropped_deep_centres():
    """Control: the one-sided path still loses a deep centre (kept on purpose)."""
    verts, faces = box_mesh()
    f, _ = force(centre_at_depth(0.051), verts, faces, closed=False)
    assert float(f.abs().max()) == 0.0


def test_signed_restores_force_at_any_penetration_depth():
    """Overlap = contact_r + t for every depth, incl. t > contact_r (the defect)."""
    verts, faces = box_mesh()
    for depth in (1e-6, 0.01, 0.04, 0.049, 0.051, 0.1, 0.3):
        centre = centre_at_depth(depth)
        assert bool(ray_parity_inside(centre.reshape(1, 3), verts, faces)[0])
        f, reaction = force(centre, verts, faces, closed=True)
        expected = K * (R + depth)
        assert float(f[2]) > 0.0, f"no restoring force at t={depth}"
        assert abs(float(f[2]) - expected) / expected < 1e-9, (depth, float(f[2]))
        # outward (away from the surface the centre is behind)
        assert float(f[0]) == 0.0 and float(f[1]) == 0.0
        # equal-and-opposite reaction on the hull
        assert abs(float(reaction[2]) + float(f[2])) < 1e-6 * expected


def test_force_is_monotone_and_continuous_across_the_surface():
    verts, faces = box_mesh()
    mags = []
    for depth in (-0.04, -0.01, -1e-6, 1e-6, 0.01, 0.04, 0.3):
        f, _ = force(centre_at_depth(depth), verts, faces, closed=True)
        mags.append(float(f[2]))
    assert all(m > 0.0 for m in mags)
    assert mags == sorted(mags), mags
    jump = abs(mags[3] - mags[2]) / mags[3]
    assert jump < 1e-3, jump          # continuous at the surface (was a 100 kN jump)


def test_deep_centre_is_recovered_not_lost():
    """A sphere released 0.2 m inside a closed box must be pushed out."""
    verts, faces = box_mesh()
    mass = 4.0 / 3.0 * math.pi * R ** 3 * 2500.0
    x = centre_at_depth(0.2)
    v = torch.zeros(3, dtype=F64)
    dt = 2.0e-5
    for _ in range(int(3.0e-3 / dt)):
        f, _ = force(x, verts, faces, closed=True)
        v = v + (f / mass) * dt
        x = x + v * dt
    assert float(x[2]) > HALF, float(x[2])   # 0.815 m in the audit run
    assert float(v[2]) > 0.0


def test_global_fallback_matches_local_result_when_envelope_suffices():
    """The full-mesh fallback must not change shallow inside answers."""
    verts, faces = box_mesh()
    f_shallow, _ = force(centre_at_depth(0.01), verts, faces, closed=True)
    assert abs(float(f_shallow[2]) - K * (R + 0.01)) / (K * (R + 0.01)) < 1e-9