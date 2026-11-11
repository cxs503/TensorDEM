"""Unit tests for the analytic clump inertia kernel (``tensordem.clump_inertia``).

The kernel exists because ``ClumpParticle.inertia`` is a voxel quadrature, and a
voxel value cannot validate itself.  These tests pin the analytic result to three
independent references:

* exact closed forms (single sphere, tangent dumbbell),
* a high-resolution voxel quadrature of the *union* for overlapping spheres,
* the inclusion-exclusion order: the third-order (triple intersection) term must
  improve agreement where it is non-zero.

They also pin the production voxel values to the analytic ones within 1 %, which
is the internal acceptance criterion of campaign node GAP-DEM-CLUMP.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.clump_inertia import (
    compare_with_voxel,
    lens_moments,
    sphere_cap_moments,
    union_mass_properties,
)
from tensordem.particle.clump import (
    SphereMember,
    ellipsoid_clump,
    single_sphere,
    voxel_mass_properties,
)

DENSITY = 2500.0


def test_sphere_cap_moments_reproduce_the_whole_sphere():
    r = 1.0
    full = sphere_cap_moments(r, -r)
    assert full["volume"] == pytest.approx(4.0 / 3.0 * math.pi * r**3, rel=1e-14)
    assert full["first_axial"] == pytest.approx(0.0, abs=1e-15)
    assert full["second_axial"] == pytest.approx(4.0 * math.pi * r**5 / 15.0, rel=1e-13)
    assert full["transverse"] == pytest.approx(8.0 * math.pi * r**5 / 15.0, rel=1e-13)
    # int(x^2) + int(y^2+z^2) == int(r^2 dV) == 4 pi r^5 / 5
    assert (full["second_axial"] + full["transverse"]) == pytest.approx(
        4.0 * math.pi * r**5 / 5.0, rel=1e-13)
    empty = sphere_cap_moments(r, r)
    assert empty["volume"] == 0.0 and empty["transverse"] == 0.0


@pytest.mark.parametrize("h", [-0.9, -0.5, 0.0, 0.4, 0.95])
def test_sphere_cap_moments_match_quadrature(h):
    """Cap volume / moments vs a dense midpoint quadrature (independent)."""
    r = 1.0
    n = 3000
    x = h + (r - h) * (np.arange(n) + 0.5) / n
    dx = (r - h) / n
    rho2 = np.clip(r * r - x * x, 0.0, None)
    vol = float(np.sum(math.pi * rho2) * dx)
    first = float(np.sum(x * math.pi * rho2) * dx)
    second = float(np.sum(x * x * math.pi * rho2) * dx)
    trans = float(np.sum(0.5 * math.pi * rho2 ** 2) * dx)
    mom = sphere_cap_moments(r, h)
    assert mom["volume"] == pytest.approx(vol, rel=1e-6)
    assert mom["second_axial"] == pytest.approx(second, rel=1e-6)
    assert mom["transverse"] == pytest.approx(trans, rel=1e-6)
    # the first moment is a small difference of large numbers near the full
    # sphere (h -> -r), so an absolute tolerance scaled by the cap volume is the
    # meaningful statement there
    assert mom["first_axial"] == pytest.approx(first, abs=1e-6 * max(vol, 1e-9))


def test_single_sphere_is_exact():
    r = 0.002
    u = union_mass_properties([[0.0, 0.0, 0.0]], [r], DENSITY)
    m = DENSITY * 4.0 / 3.0 * math.pi * r**3
    exact = 0.4 * m * r * r
    assert np.allclose(np.diag(u["inertia"]), exact, rtol=1e-14)
    assert u["mass"] == pytest.approx(m, rel=1e-14)
    assert np.allclose(u["com"], 0.0, atol=1e-18)


@pytest.mark.parametrize("separation_factor", [2.0, 2.5, 3.0])
def test_tangent_dumbbell_matches_textbook_tensor(separation_factor):
    r = 0.001
    sep = separation_factor * r
    u = union_mass_properties([[-0.5 * sep, 0, 0], [0.5 * sep, 0, 0]], [r, r], DENSITY)
    m = DENSITY * 4.0 / 3.0 * math.pi * r**3
    axial = 2.0 * 0.4 * m * r * r
    transverse = 2.0 * (0.4 * m * r * r + m * (0.5 * sep) ** 2)
    assert np.diag(u["inertia"])[0] == pytest.approx(axial, rel=1e-13)
    assert np.diag(u["inertia"])[1] == pytest.approx(transverse, rel=1e-13)
    assert np.diag(u["inertia"])[2] == pytest.approx(transverse, rel=1e-13)
    assert u["volume"] == pytest.approx(2.0 * 4.0 / 3.0 * math.pi * r**3, rel=1e-14)


def test_lens_is_containment_aware():
    r1, r2 = 0.002, 0.0008
    lens = lens_moments((0, 0, 0), r1, (1e-4, 0, 0), r2)
    assert lens is not None
    assert lens["volume"] == pytest.approx(4.0 / 3.0 * math.pi * r2**3, rel=1e-12)
    assert lens_moments((0, 0, 0), 0.001, (0.005, 0, 0), 0.001) is None


@pytest.mark.parametrize("d_factor", [1.2, 1.5, 1.8])
def test_overlapping_pair_matches_fine_voxel(d_factor):
    r = 0.001
    d = d_factor * r
    members = [SphereMember(np.array([-0.5 * d, 0.0, 0.0]), r),
               SphereMember(np.array([0.5 * d, 0.0, 0.0]), r)]
    u = union_mass_properties([m.center for m in members], [r, r], DENSITY)
    ref = voxel_mass_properties(members, DENSITY, grid=384)
    assert ref["volume"] == pytest.approx(u["volume"], rel=1e-3)
    assert np.allclose(np.diag(ref["inertia"]), np.diag(u["inertia"]), rtol=1e-3)
    # the union must be lighter than the naive member sum when there is overlap
    assert u["volume"] < 2.0 * 4.0 / 3.0 * math.pi * r**3


def test_third_order_term_improves_a_triple_intersection():
    rad = 0.001
    centres = [[0.8e-3, 0.0, 0.0], [-0.4e-3, 0.693e-3, 0.0], [-0.4e-3, -0.693e-3, 0.0]]
    u3 = union_mass_properties(centres, [rad] * 3, DENSITY, max_order=3)
    u2 = union_mass_properties(centres, [rad] * 3, DENSITY, max_order=2)
    assert u3["third_order_fraction"] > 1e-3          # there really is a triple
    assert u3["n_triples"] == 1
    ref = voxel_mass_properties([SphereMember(np.array(c), rad) for c in centres],
                                DENSITY, grid=384)
    ref_diag = np.diag(ref["inertia"])
    e3 = np.max(np.abs(np.diag(u3["inertia"]) - ref_diag)) / np.max(ref_diag)
    e2 = np.max(np.abs(np.diag(u2["inertia"]) - ref_diag)) / np.max(ref_diag)
    assert e3 < 1e-3
    assert e3 < e2


def test_union_tensor_is_physical():
    """Positive definite, principal moments obey the triangle inequality."""
    p = ellipsoid_clump(0.004, 0.0015, n_spheres=5, density=DENSITY)
    u = union_mass_properties([m.center for m in p.members],
                              [m.radius for m in p.members], DENSITY)
    mom = np.linalg.eigvalsh(u["inertia"])
    assert np.all(mom > 0.0)
    assert mom[0] + mom[1] >= mom[2] * (1.0 - 1e-12)


@pytest.mark.parametrize("ar", [1.5, 2.0, 3.0])
def test_production_clump_inertia_within_one_percent_of_analytic(ar):
    p = ellipsoid_clump(ar * 0.0012, 0.0012, n_spheres=max(3, int(round(2 * ar))),
                        density=DENSITY)
    cmp = compare_with_voxel(p, grid=192)
    assert cmp["rel_err_production"] < 0.01
    assert cmp["rel_err_mass"] < 0.01
    # transverse inertia must exceed the axial one for a prolate body
    assert p.inertia[1, 1] > p.inertia[0, 0]


def test_single_sphere_clump_keeps_exact_sphere_inertia():
    p = single_sphere(0.0015, DENSITY)
    m = DENSITY * 4.0 / 3.0 * math.pi * 0.0015**3
    assert np.allclose(np.diag(p.inertia), 0.4 * m * 0.0015**2, rtol=1e-14)