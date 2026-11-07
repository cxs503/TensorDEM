"""Tests for the IN32-N01 polygonal floe field (Laguerre + SAT + torques).

Coverage:
* Laguerre tessellation: exact box cover, prescribed-area Newton solve for all
  four FSD families, parameter/shape validation (negative controls);
* convex-polygon contact: exact signed gap for separated/overlapping bodies,
  rotated bodies, hull segments, lever arms and torques, momentum and pair-torque
  identities, KD-tree vs O(N^2) completeness (negative controls);
* field assembly: no overlap at every concentration, FSD moment agreement,
  convexity/CCW ordering, mass/inertia positivity, table export, determinism,
  backward-compatible disk view and input validation (negative controls).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.fsd_packing import BimodalFSD, DeltaFSD, LogNormalFSD, PowerLawFSD
from tensordem.laguerre_field import (LaguerreParams, jittered_sites, laguerre_cells,
                                      laguerre_prescribed_areas)
from tensordem.polygon_contact import (PolygonContactConfig, candidate_pairs,
                                       contact_forces, contact_set,
                                       coordination_number, fabric_tensor,
                                       polygon_gap, polygon_hull_gap)
from tensordem.polygon_floe_field import (PolygonFloeField, generate_from_spec,
                                          generate_polygon_floe_field,
                                          sample_floe_radii)

SQUARE = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
TRIANGLE = np.array([[0.0, 1.0], [-0.9, -0.8], [0.9, -0.8]])


def _is_convex_ccw(v: np.ndarray) -> bool:
    if v.shape[0] < 3:
        return False
    w = np.roll(v, -1, axis=0)
    cr = v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]
    return bool((cr > 0).all())


# ---------------------------------------------------------------------------
# Laguerre tessellation
# ---------------------------------------------------------------------------

def test_jittered_sites_inside_box_and_deterministic():
    box = (20.0, 12.0)
    a = jittered_sites(64, box, seed=5)
    b = jittered_sites(64, box, seed=5)
    c = jittered_sites(64, box, seed=6)
    assert a.shape == (64, 2)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)
    assert (a[:, 0] >= 0).all() and (a[:, 0] < box[0]).all()
    assert (a[:, 1] >= 0).all() and (a[:, 1] < box[1]).all()


@pytest.mark.parametrize("n,box,jitter", [(1, (10.0, 10.0), 0.5),
                                          (10, (0.0, 10.0), 0.5),
                                          (10, (10.0, 10.0), 1.0)])
def test_jittered_sites_rejects_bad_args(n, box, jitter):
    with pytest.raises(ValueError):
        jittered_sites(n, box, jitter=jitter)


def test_tessellation_covers_box_exactly():
    box = (25.0, 16.0)
    sites = jittered_sites(180, box, seed=2)
    cells = laguerre_cells(sites, None, box)
    assert abs(cells.cover - 1.0) < 1e-12
    assert cells.degenerate == 0
    assert cells.count.min() >= 3


def test_tessellation_rejects_bad_geometry():
    box = (10.0, 10.0)
    with pytest.raises(ValueError):
        laguerre_cells(np.zeros((5, 3)), None, box)
    with pytest.raises(ValueError):
        laguerre_cells(np.zeros((5, 2)), np.zeros(4), box)
    with pytest.raises(ValueError):
        laguerre_cells(np.zeros((5, 2)), None, (0.0, 10.0))


@pytest.mark.parametrize("factory", [DeltaFSD, LogNormalFSD, BimodalFSD, PowerLawFSD])
def test_prescribed_areas_converge(factory):
    box = (30.0, 20.0)
    n = 120
    sites = jittered_sites(n, box, seed=1)
    fsd = factory()
    radii = sample_floe_radii(fsd, n, seed=0)
    cells = laguerre_prescribed_areas(sites, math.pi * radii ** 2, box,
                                     params=LaguerreParams(max_iter=8,
                                                           area_tol=5e-3))
    targets = math.pi * radii ** 2
    targets = targets * (box[0] * box[1] / targets.sum())
    rel = np.abs(cells.area - targets) / targets
    assert rel.max() < 5e-3
    assert abs(cells.cover - 1.0) < 1e-6
    assert cells.degenerate == 0
    # the getter reports the neighbour count actually needed
    assert cells.n_neighbours >= 6


def test_prescribed_areas_rejects_bad_targets():
    box = (10.0, 8.0)
    sites = jittered_sites(20, box, seed=0)
    with pytest.raises(ValueError):
        laguerre_prescribed_areas(sites, -np.ones(20), box)
    with pytest.raises(ValueError):
        laguerre_prescribed_areas(sites, np.ones(19), box)
    with pytest.raises(ValueError):
        laguerre_prescribed_areas(sites, np.array([np.nan] + [1.0] * 19), box)


@pytest.mark.parametrize("kwargs", [{"n_neighbours": 4}, {"max_iter": 0},
                                    {"area_tol": 0.0}, {"area_tol": 1.5},
                                    {"pool": 4}])
def test_laguerre_params_validation(kwargs):
    with pytest.raises(ValueError):
        LaguerreParams(**kwargs)


def test_prescribed_areas_preserve_target_ratios():
    """Targets are normalised to fill the box, so their *ratios* are preserved."""
    sites = jittered_sites(60, (20.0, 15.0), seed=3)
    radii = sample_floe_radii(LogNormalFSD(0.4), 60, seed=0)
    tgt = math.pi * radii ** 2
    a = laguerre_prescribed_areas(sites, tgt, (20.0, 15.0),
                                  params=LaguerreParams(max_iter=6))
    b = laguerre_prescribed_areas(sites, 3.0 * tgt, (20.0, 15.0),
                                  params=LaguerreParams(max_iter=6))
    np.testing.assert_allclose(b.area, a.area, rtol=1e-8)
    np.testing.assert_allclose(a.area / a.area.mean(), tgt / tgt.mean(), rtol=0.02)


# ---------------------------------------------------------------------------
# convex polygon contact
# ---------------------------------------------------------------------------

def test_polygon_gap_separated_exact():
    res = polygon_gap(np.zeros(2), SQUARE, np.array([2.5, 0.0]), SQUARE)
    assert res.gap == pytest.approx(0.5, abs=1e-12)
    np.testing.assert_allclose(res.normal, [1.0, 0.0], atol=1e-12)
    # witness point sits on the facing edges
    assert res.point[0] == pytest.approx(1.25, abs=1e-12)


def test_polygon_gap_overlap_exact():
    res = polygon_gap(np.zeros(2), SQUARE, np.array([1.5, 0.0]), SQUARE)
    assert res.gap == pytest.approx(-0.5, abs=1e-12)
    assert res.overlap == pytest.approx(0.5, abs=1e-12)
    np.testing.assert_allclose(res.normal, [1.0, 0.0], atol=1e-12)


def test_polygon_gap_rotated_body():
    far = polygon_gap(np.zeros(2), SQUARE, np.array([0.2, 3.0]), TRIANGLE)
    near = polygon_gap(np.zeros(2), SQUARE, np.array([0.2, 1.2]), TRIANGLE)
    assert far.gap > 0.0 and near.gap < 0.0
    assert far.gap == pytest.approx(1.2, abs=1e-12)
    assert near.gap == pytest.approx(-0.6, abs=1e-12)


def test_polygon_gap_is_frame_consistent():
    """Translation of both bodies must not change the gap."""
    a = polygon_gap(np.zeros(2), SQUARE, np.array([3.0, 0.4]), TRIANGLE)
    shift = np.array([7.0, -2.0])
    b = polygon_gap(shift, SQUARE, shift + np.array([3.0, 0.4]), TRIANGLE)
    assert a.gap == pytest.approx(b.gap, abs=1e-12)


def test_polygon_gap_rejects_degenerate_input():
    with pytest.raises(ValueError):
        polygon_gap(np.zeros(2), SQUARE, np.zeros(2), np.zeros((0, 2)))
    with pytest.raises(ValueError):
        polygon_gap(np.zeros(2), np.array([[np.nan, 0.0]]), np.zeros(2), SQUARE)


def test_hull_contact_carries_lever_arm():
    res = polygon_hull_gap(np.zeros(2), SQUARE, np.array([1.6, -5.0]),
                           np.array([1.6, 5.0]))
    assert res.gap == pytest.approx(0.6, abs=1e-12)
    assert float(np.linalg.norm(res.r_a)) > 0.0
    np.testing.assert_allclose(res.r_b, [0.0, 0.0])


def test_contact_forces_momentum_and_pair_torque():
    pos = np.array([[0.0, 0.0], [1.4, 0.0]])
    mass = np.array([10.0, 10.0])
    contacts = {"i": np.array([0]), "j": np.array([1]), "gap": np.array([-0.1]),
                "normal": np.array([[1.0, 0.0]]), "point": np.array([[0.7, 0.85]]),
                "disp": np.array([[1.4, 0.0]])}
    out = contact_forces(pos, mass, contacts)
    assert out["f_n"][0] > 0.0
    assert abs(out["body_force"].sum()) < 1e-10
    d = contacts["disp"][0]
    f = out["force"][0]
    ident = (d[0] * f[1] - d[1] * f[0]) - (out["torque_i"][0] + out["torque_j"][0])
    assert abs(ident) < 1e-9 * max(1.0, abs(out["torque_i"][0]))
    # the offset contact point produces a torque that a disk model cannot have
    assert abs(out["torque_i"][0]) > 0.0
    assert out["torque_arm"][0] == pytest.approx(math.hypot(0.7, 0.85))


def test_contact_forces_coulomb_cap_and_zero_gap():
    pos = np.array([[0.0, 0.0], [1.2, 0.0]])
    mass = np.full(2, 5.0)
    cfg = PolygonContactConfig(mu=0.5)
    contacts = {"i": np.array([0]), "j": np.array([1]), "gap": np.array([-0.01]),
                "normal": np.array([[1.0, 0.0]]), "point": np.zeros((1, 2)),
                "disp": np.array([[1.2, 0.0]])}
    out = contact_forces(pos, mass, contacts, config=cfg, slip=np.array([10.0]))
    assert abs(out["f_t"][0]) <= 0.5 * out["f_n"][0] + 1e-12
    # no overlap -> no normal force at all
    contacts["gap"] = np.array([0.01])
    out2 = contact_forces(pos, mass, contacts, config=cfg)
    assert out2["f_n"][0] == 0.0


def test_contact_config_validation():
    with pytest.raises(ValueError):
        PolygonContactConfig(kn=-1.0)
    with pytest.raises(ValueError):
        PolygonContactConfig(zeta=-0.1)


def test_candidate_pairs_periodic_matches_brute_force():
    box = (12.0, 9.0)
    rng = np.random.default_rng(0)
    pos = rng.random((40, 2)) * np.array(box)
    reach = np.full(40, 1.2)
    i, j, disp = candidate_pairs(pos, reach, box)
    fast = {(int(a), int(b)) for a, b in zip(i, j)}
    d = pos[None, :, :] - pos[:, None, :]
    d[..., 0] -= box[0] * np.round(d[..., 0] / box[0])
    d[..., 1] -= box[1] * np.round(d[..., 1] / box[1])
    dist = np.linalg.norm(d, axis=2)
    bound = reach[:, None] + reach[None, :]
    bf = {(int(a), int(b)) for a, b in zip(*np.nonzero(np.triu(dist <= bound, k=1)))}
    assert fast == bf
    # displacements are the minimum-image ones
    for a, b, dd in zip(i, j, disp):
        assert np.linalg.norm(dd) <= np.linalg.norm(d[a, b]) + 1e-12


def test_contact_set_fast_equals_brute():
    field = generate_polygon_floe_field(LogNormalFSD(0.4), coverage=0.90,
                                        n_floes=220, seed=11)
    cut = 0.25 * float(np.median(field.r_eq))
    fast = field.contacts(cutoff=cut)
    brute = field.contacts(cutoff=cut, brute=True)
    kf = {(int(a), int(b)) for a, b in zip(fast["i"], fast["j"])}
    kb = {(int(a), int(b)) for a, b in zip(brute["i"], brute["j"])}
    assert kf == kb
    assert kf, "the near-contact shell must not be empty at phi=0.90"


def test_contact_set_is_nested_in_cutoff():
    field = generate_polygon_floe_field(DeltaFSD(5.0), coverage=0.85,
                                        n_floes=150, seed=1)
    r = float(np.median(field.r_eq))
    small = field.contacts(cutoff=0.1 * r)
    big = field.contacts(cutoff=0.4 * r)
    ks = {(int(a), int(b)) for a, b in zip(small["i"], small["j"])}
    kb = {(int(a), int(b)) for a, b in zip(big["i"], big["j"])}
    assert ks <= kb
    assert len(kb) > len(ks)


def test_fabric_tensor_and_coordination():
    normals = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    fab = fabric_tensor(normals)
    assert fab["phi_xx"] == pytest.approx(0.5)
    assert fab["phi_yy"] == pytest.approx(0.5)
    assert fab["anisotropy"] == pytest.approx(0.0, abs=1e-12)
    assert coordination_number(10, np.array([0, 1]), np.array([1, 2])) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        coordination_number(0, np.zeros(0), np.zeros(0))


# ---------------------------------------------------------------------------
# field assembly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phi", [0.60, 0.80, 0.85, 0.90])
def test_field_has_no_overlap(phi):
    field = generate_polygon_floe_field(LogNormalFSD(0.5), coverage=phi,
                                        n_floes=180, seed=4)
    assert field.min_signed_gap() >= -1e-9
    assert field.realised_coverage() == pytest.approx(phi, rel=2e-2)


@pytest.mark.parametrize("factory", [DeltaFSD, LogNormalFSD, BimodalFSD, PowerLawFSD])
def test_field_moments_match_fsd(factory):
    field = generate_polygon_floe_field(factory(), coverage=0.85, n_floes=400, seed=1)
    err = field.moment_errors()
    assert err["m1"] < 1e-2
    assert err["m2"] < 1e-2


def test_field_polygons_are_convex_and_ccw():
    field = generate_polygon_floe_field(BimodalFSD(), coverage=0.85, n_floes=150, seed=2)
    for k in range(field.n):
        assert _is_convex_ccw(field.verts[k, : field.count[k]])


def test_field_positive_mass_inertia_and_area_consistency():
    field = generate_polygon_floe_field(LogNormalFSD(0.4), coverage=0.85,
                                        n_floes=120, seed=3)
    assert (field.mass > 0).all()
    assert (field.inertia > 0).all()
    # polygon area from the shoelace formula must equal the stored area
    for k in range(field.n):
        v = field.verts[k, : field.count[k]]
        a = 0.5 * abs(np.sum(v[:, 0] * np.roll(v, -1, axis=0)[:, 1]
                             - v[:, 1] * np.roll(v, -1, axis=0)[:, 0]))
        assert a == pytest.approx(field.area[k], rel=1e-10)


def test_field_determinism_and_seed_sensitivity():
    a = generate_polygon_floe_field(LogNormalFSD(0.4), coverage=0.85,
                                    n_floes=100, seed=7)
    b = generate_polygon_floe_field(LogNormalFSD(0.4), coverage=0.85,
                                    n_floes=100, seed=7)
    c = generate_polygon_floe_field(LogNormalFSD(0.4), coverage=0.85,
                                    n_floes=100, seed=8)
    np.testing.assert_allclose(a.centroid, b.centroid)
    np.testing.assert_allclose(a.area, b.area)
    assert not np.allclose(a.centroid, c.centroid)


@pytest.mark.parametrize("kwargs", [{"coverage": 1.2}, {"coverage": 0.0},
                                    {"n_floes": 2}, {"density": -1.0},
                                    {"box": (0.0, 5.0)}])
def test_field_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        generate_polygon_floe_field(DeltaFSD(5.0), **kwargs)


def test_field_rejects_bad_radii():
    with pytest.raises(ValueError):
        generate_polygon_floe_field(radii=np.array([1.0, -1.0, 2.0, 2.0]),
                                    n_floes=4)
    with pytest.raises(ValueError):
        generate_polygon_floe_field(radii=np.array([1.0, 1.0, 1.0]), n_floes=4)


def test_floe_table_roundtrip():
    field = generate_polygon_floe_field(LogNormalFSD(0.4), coverage=0.85,
                                        n_floes=90, seed=5)
    table = field.floe_table()
    assert table["id"].shape == (90,)
    assert table["vertices"].shape[0] == 90
    np.testing.assert_allclose(table["area"], field.area)
    np.testing.assert_allclose(table["r_eq"], np.sqrt(field.area / math.pi))
    assert int(table["n_vertices"].max()) <= table["vertices"].shape[1]


def test_disc_equivalent_backward_compatible():
    field = generate_polygon_floe_field(DeltaFSD(5.0), coverage=0.85,
                                        n_floes=64, seed=0)
    disc = field.as_disc_equivalent()
    assert disc["pos"].shape == (64, 2)
    assert disc["radius"].shape == (64,)
    np.testing.assert_allclose(disc["radius"], 5.0, rtol=1e-3)


def test_vertex_scaling_preserves_topology():
    field = generate_polygon_floe_field(LogNormalFSD(0.4), coverage=0.64,
                                        n_floes=80, seed=6)
    big = field.with_vertex_scale(1.25)
    np.testing.assert_allclose(big.area, field.area * 1.25 ** 2)
    np.testing.assert_allclose(big.centroid, field.centroid)
    assert big.min_signed_gap() < field.min_signed_gap()
    with pytest.raises(ValueError):
        field.with_vertex_scale(0.0)


def test_generate_from_spec():
    spec = {"fsd": {"kind": "log_normal", "cv": 0.3}, "coverage": 0.8, "n_floes": 64}
    field = generate_from_spec(spec, seed=0)
    assert isinstance(field, PolygonFloeField)
    assert field.n == 64
    assert field.realised_coverage() == pytest.approx(0.8, rel=2e-2)


def test_adjacency_and_coordination_consistency():
    field = generate_polygon_floe_field(DeltaFSD(5.0), coverage=0.85,
                                        n_floes=120, seed=2)
    cut = 0.3 * float(np.median(field.r_eq))
    adj = field.adjacency(cutoff=cut)
    z = field.coordination_number(cutoff=cut)
    assert z == pytest.approx(adj.mean(), rel=1e-12)
    assert z > 3.0