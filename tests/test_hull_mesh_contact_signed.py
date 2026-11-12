"""Signed (inside/outside) hull contact — GAP-DEM-HULL-SIGNED regression surface.

Pins the defect that motivated `closed_surface` in
:func:`tensordem.hull_mesh_contact.mesh_contact_forces` and the invariants of the
fix.  Numbers come from ``scripts/hull_signed_audit.py`` (artifacts:
``artifacts/m16799_hull_signed/metrics.json``).
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from tensordem.hull_mesh_contact import make_flat_mesh, mesh_contact_forces
from tensordem.hull_mesh_contact_signed import (
    is_edge_manifold,
    is_watertight,
    ray_parity_inside,
    signed_distance_to_mesh,
)

_HERE = Path(__file__).resolve()
_SPEC = importlib.util.spec_from_file_location(
    "hull_signed_audit", _HERE.parents[1] / "scripts" / "hull_signed_audit.py")
_audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_audit)

F64 = torch.float64
R, K = 0.05, 1.0e6


def _force(center, verts, faces, closed, radius=R, stiffness=K):
    f, reaction = mesh_contact_forces(
        center.reshape(1, 3), torch.zeros(1, 3, dtype=F64), verts, faces,
        torch.zeros(3, dtype=F64), particle_radius=radius, stiffness=stiffness,
        closed_surface=closed)
    return f[0], reaction


@pytest.fixture(scope="module")
def cyl():
    return _audit.capped_cylinder_mesh(0.5, 2.0, 64)


@pytest.fixture(scope="module")
def wigley():
    return _audit.wigley_mesh()


# ---------------------------------------------------------------------------
# 1. The defect, pinned as a characterisation test (baseline behaviour)
# ---------------------------------------------------------------------------

def test_baseline_reverses_force_for_centre_inside():
    """One-sided kernel pulls the particle *into* the hull past the centre depth."""
    verts, faces = make_flat_mesh(size=2.0, nx=2, ny=2)
    for depth in (0.005, 0.03):
        center = torch.tensor([0.0, 0.0, -depth], dtype=F64)   # centre below plane
        f, _ = _force(center, verts, faces, closed=False)
        assert float(f[2]) < 0.0, "documented defect: force points into the hull"


def test_baseline_drops_contact_beyond_one_radius():
    """One-sided kernel returns exactly zero once the centre is > R past the surface."""
    verts, faces = make_flat_mesh(size=2.0, nx=2, ny=2)
    center = torch.tensor([0.0, 0.0, -0.06], dtype=F64)
    f, _ = _force(center, verts, faces, closed=False)
    assert float(torch.linalg.vector_norm(f)) == 0.0


# ---------------------------------------------------------------------------
# 2. The fix: signed contact on a closed cylinder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("frac", [
    pytest.param(0.25), pytest.param(0.5), pytest.param(1.0),
    pytest.param(1.5, marks=pytest.mark.xfail(reason="OPEN: deep band delta>1R deviates from k(R+delta); see docs/hull_mesh_contact_signed_distance.md sec. 7")),
    pytest.param(2.0, marks=pytest.mark.xfail(reason="OPEN: deep band delta>1R")),
    pytest.param(3.0, marks=pytest.mark.xfail(reason="OPEN: deep band delta>1R")),
])
def test_signed_force_outward_and_on_analytic_magnitude(cyl, frac):
    """Inside the solid the force must push out with magnitude k (R + depth)."""
    verts, faces = cyl
    for az_deg in (0.0, 47.0, 133.0, 311.0):
        a = math.radians(az_deg)
        n_out = torch.tensor([math.cos(a), math.sin(a), 0.0], dtype=F64)
        rho = 0.5 - frac * R
        center = torch.tensor([n_out[0] * rho, n_out[1] * rho, 0.0], dtype=F64)
        f, reaction = _force(center, verts, faces, closed=True)
        mag = float(torch.linalg.vector_norm(f))
        analytic = K * (R + frac * R)
        assert mag > 0.0, "signed mode must never lose contact"
        # inscribed facets: the closest-point direction is off-radial by up to the
        # half-chord angle (2.8 deg at n_circ=64 -> cos 0.9988)
        assert float(torch.dot(f / mag, n_out)) > 0.995
        assert abs(mag - analytic) / analytic < 0.03
        assert float(torch.linalg.vector_norm(reaction + f)) < 1e-9


def test_signed_matches_baseline_outside(cyl):
    """The fix must not perturb the outer branch."""
    verts, faces = cyl
    center = torch.tensor([0.0, 0.53, 0.0], dtype=F64)  # 30 mm outside the wall
    f0, _ = _force(center, verts, faces, closed=False)
    f1, _ = _force(center, verts, faces, closed=True)
    assert torch.allclose(f0, f1, rtol=1e-12, atol=1e-12)


def test_signed_requires_closed_mesh():
    verts, faces = make_flat_mesh(size=2.0, nx=2, ny=2)
    assert not is_watertight(verts, faces)
    with pytest.raises(ValueError, match="watertight|closed"):
        mesh_contact_forces(torch.zeros(1, 3, dtype=F64), torch.zeros(1, 3, dtype=F64),
                            verts, faces, torch.zeros(3, dtype=F64),
                            particle_radius=R, stiffness=K, closed_surface=True)


# ---------------------------------------------------------------------------
# 3. Classifier and signed distance vs analytic references
# ---------------------------------------------------------------------------

def test_classifier_matches_analytic_containment_cylinder(cyl):
    verts, faces = cyl
    g = torch.Generator().manual_seed(7)
    pts = (torch.rand(400, 3, generator=_g(g), dtype=F64) - 0.5) * 2.6
    got = ray_parity_inside(pts, verts, faces)
    for i in range(pts.shape[0]):
        want = _audit.ref_cylinder(pts[i], 0.5, 2.0)[1]
        assert bool(got[i]) == bool(want), f"misclassified point {pts[i].tolist()}"


def _g(g):
    return g


def test_classifier_matches_analytic_containment_wigley(wigley):
    verts, faces = wigley
    g = torch.Generator().manual_seed(13)
    pts = _audit.wigley_sampler(g, 300)
    got = ray_parity_inside(pts, verts, faces)
    bad = sum(bool(got[i]) != _audit.wigley_inside(pts[i], 4.0, 0.5, 0.25)
              for i in range(pts.shape[0]))
    assert bad == 0


def test_wigley_is_closed_but_pinches_at_the_bow(wigley):
    verts, faces = wigley
    assert is_watertight(verts, faces)
    assert not is_edge_manifold(verts, faces)  # one 4-fold edge at the stem corner


def test_signed_distance_sign_and_magnitude_cylinder(cyl):
    verts, faces = cyl
    for center in (torch.tensor([0.0, 0.2, 0.0], dtype=F64),      # 300 mm inside
                   torch.tensor([0.0, 0.62, 0.0], dtype=F64),     # 120 mm outside
                   torch.tensor([0.0, 0.55, 0.0], dtype=F64)):    # 50 mm outside
        res = signed_distance_to_mesh(center.reshape(1, 3), verts, faces)
        d_ref, inside, _ = _audit.ref_cylinder(center, 0.5, 2.0)
        assert bool(res.inside[0]) == bool(inside)
        assert abs(abs(float(res.signed_distance[0])) - d_ref) / d_ref < 0.005
        assert (float(res.signed_distance[0]) < 0) == bool(inside)


# ---------------------------------------------------------------------------
# 4. Dynamic impact: baseline loses the floe, signed rebounds
# ---------------------------------------------------------------------------

def test_dynamic_impact_baseline_loses_floe_signed_rebounds():
    out = _audit.dynamic_impact(mass=480.0, R=0.5, v0=3.0, k=1.0e4, dt=2.0e-4, t_max=0.3)
    base, sig = out["baseline"], out["signed"]
    # baseline crosses the wall and ends inside the solid (wall radius 0.5 m)
    assert base["y_final"] < 0.5 - 0.5 * 0.5
    # signed: the floe is stopped and moving outward at the end of the window
    assert sig["vy_final"] > 0.0 and sig["rebound_speed"] > 0.0
    assert sig["y_final"] > base["y_final"]
    assert 0.0 < sig["restitution"] <= 1.02            # no energy creation beyond O(dt)
    assert sig["max_force"] > 0.0