"""Tests for :mod:`tensordem.floe_compaction` (IN26-N01).

Coverage of the module:

* batched SAT kernel against the reference scalar routine (depth exact, normal up
  to orientation, separation verdict identical);
* vectorised periodic pair search against the reference ``candidate_pairs``;
* rasterised union coverage against analytic areas, incl. periodic seam and
  double-covered area;
* the consolidated-zone construction: exact coverage, zero overlap, a contact
  network that percolates, FSD moments, determinism, and the negative controls
  (lead zone too high, buffered field contact-free, invalid parameters);
* the attraction/projection solver: area/box invariance, pinning, parameter
  validation, and the monodisperse regression lock (bit-for-bit against the
  IN32-N01 generator plus the H1 single-dispersion field).
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from tensordem.floe_compaction import (
    CompactionParams,
    ConsolidationParams,
    _polygon_area,
    _select_by_area,
    compact_floe_field,
    contact_percolation,
    fast_pairs,
    generate_compacted_floe_field,
    generate_consolidated_floe_field,
    h1_reference_digest,
    sat_pairs,
    union_coverage,
    zone_order,
)
from tensordem.fsd_packing import BimodalFSD, DeltaFSD, LogNormalFSD, PowerLawFSD
from tensordem.laguerre_field import LaguerreParams
from tensordem.polygon_contact import candidate_pairs, polygon_gap
from tensordem.polygon_floe_field import generate_polygon_floe_field

#: SHA-256 over the monodisperse (DeltaFSD(5 m)) polygonal field produced by
#: :func:`generate_polygon_floe_field` with ``coverage=0.85, n_floes=200, seed=7``.
#: Recorded with this node (IN26-N01, 2026-11-10); the monodisperse regression lock
#: requires the new module to reproduce it bit-for-bit.
MONODISPERSE_FIELD_DIGEST = (
    "70141922c3f1117cd90a37efff0df440419e218a29ddd70433877c7c1919ce2f")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _random_convex_polygons(rng, m, cap, radius=(1.0, 2.0)):
    verts = np.zeros((m, cap, 2))
    valid = np.zeros((m, cap), dtype=bool)
    for k in range(m):
        nside = int(rng.integers(3, cap + 1))
        ang = np.sort(rng.uniform(0.0, 2.0 * math.pi, nside))
        rad = rng.uniform(*radius) * (0.75 + 0.5 * rng.random(nside))
        pts = np.stack([np.cos(ang), np.sin(ang)], axis=1) * rad[:, None]
        pts = pts[ConvexHull(pts).vertices]
        verts[k, : pts.shape[0]] = pts
        valid[k, : pts.shape[0]] = True
    return verts, valid


def _field_digest(field) -> str:
    hasher = hashlib.sha256()
    for arr in (field.centroid, field.verts, field.count, field.area, field.r_eq):
        hasher.update(np.ascontiguousarray(arr).tobytes())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# 1-3: contact kernels
# ---------------------------------------------------------------------------

def test_sat_pairs_matches_reference_depth():
    rng = np.random.default_rng(20261110)
    m, cap = 120, 10
    a, va = _random_convex_polygons(rng, m, cap)
    b, vb = _random_convex_polygons(rng, m, cap)
    disp = np.stack([rng.uniform(-3.0, 3.0, m), rng.uniform(-3.0, 3.0, m)], axis=1)
    depth, normal, separated = sat_pairs(a, b + disp[:, None, :], va, vb, disp)
    err_depth = 0.0
    err_align = 0.0
    n_overlap = 0
    for k in range(m):
        ref = polygon_gap(np.zeros(2), a[k, : va[k].sum()],
                          np.zeros(2), b[k, : vb[k].sum()], disp[k])
        if ref.gap < 0.0:
            n_overlap += 1
            err_depth = max(err_depth, abs(ref.overlap - depth[k]))
            # the reference orients the normal by the CFP witness, the batched
            # kernel by the centroid line: they agree up to the sign
            err_align = max(err_align, 1.0 - abs(float(np.dot(ref.normal, normal[k]))))
        else:
            assert separated[k] and depth[k] == 0.0
    assert n_overlap > 10
    assert err_depth < 1e-12, err_depth
    assert err_align < 1e-9, err_align


def test_sat_pairs_rejects_bad_shapes():
    a = np.zeros((3, 4, 2))
    b = np.zeros((3, 4, 2))
    good = np.ones((3, 4), dtype=bool)
    disp = np.zeros((3, 2))
    with pytest.raises(ValueError):
        sat_pairs(a, np.zeros((3, 5, 2)), good, good, disp)
    with pytest.raises(ValueError):
        sat_pairs(a, b, np.ones((3, 3), dtype=bool), good, disp)
    with pytest.raises(ValueError):
        sat_pairs(a, b, good, good, np.zeros((3, 3)))
    empty = sat_pairs(np.zeros((0, 4, 2)), np.zeros((0, 4, 2)),
                      np.zeros((0, 4), dtype=bool), np.zeros((0, 4), dtype=bool),
                      np.zeros((0, 2)))
    assert empty[0].size == 0


def test_fast_pairs_matches_reference():
    field = generate_consolidated_floe_field(LogNormalFSD(0.4), coverage=0.80,
                                            n_floes=120, seed=5, diagnose=False)
    verts = field.verts
    count = field.count
    reach = np.array([float(np.linalg.norm(verts[k, : count[k]], axis=1).max())
                      for k in range(field.n)])
    for pad in (0.0, 1.5):
        i, j, disp = fast_pairs(field.centroid, reach, field.box, pad)
        i2, j2, disp2 = candidate_pairs(field.centroid, reach, field.box, pad=pad)
        assert set(zip(i.tolist(), j.tolist())) == set(zip(i2.tolist(), j2.tolist()))
        lookup = {(int(a), int(b)): d for a, b, d in zip(i.tolist(), j.tolist(),
                                                         disp.tolist())}
        for a, b, d in zip(i2.tolist(), j2.tolist(), disp2.tolist()):
            got = np.asarray(lookup[(int(a), int(b))])
            assert np.allclose(got, np.asarray(d), atol=1e-12)


# ---------------------------------------------------------------------------
# 4-7: union coverage rasteriser
# ---------------------------------------------------------------------------

def _square(centre, side):
    h = 0.5 * side
    return np.array([[-h, -h], [h, -h], [h, h], [-h, h]])


def test_union_coverage_single_square_is_analytic():
    box = (10.0, 10.0)
    verts = _square((5.0, 5.0), 4.0)[None, :, :]
    out = union_coverage(np.array([[5.0, 5.0]]), verts, np.array([4]), box,
                         resolution=400)
    assert out["union_coverage"] == pytest.approx(16.0 / 100.0, abs=2e-3)
    assert out["double_covered"] == 0.0
    assert out["nominal_coverage"] == pytest.approx(16.0 / 100.0, rel=1e-12)


def test_union_coverage_counts_double_covered_area():
    box = (10.0, 10.0)
    verts = np.stack([_square((5.0, 5.0), 4.0), _square((6.0, 5.0), 4.0)])
    centroid = np.array([[5.0, 5.0], [6.0, 5.0]])
    out = union_coverage(centroid, verts, np.array([4, 4]), box, resolution=400)
    # union = 2*16 - overlap(3 x 4) = 20 m^2
    assert out["union_coverage"] == pytest.approx(20.0 / 100.0, abs=3e-3)
    assert out["double_covered"] == pytest.approx(12.0 / 100.0, abs=3e-3)
    assert out["overlap_fraction"] == pytest.approx(32.0 / 100.0 - 20.0 / 100.0,
                                                   abs=3e-3)


def test_union_coverage_handles_periodic_seam():
    box = (10.0, 10.0)
    # square of side 4 centred exactly on the x seam: half of it wraps
    verts = _square((0.0, 0.0), 4.0)[None, :, :]
    out = union_coverage(np.array([[10.0, 5.0]]), verts, np.array([4]), box,
                         resolution=400)
    assert out["union_coverage"] == pytest.approx(16.0 / 100.0, abs=3e-3)
    assert out["double_covered"] == 0.0


def test_union_coverage_rejects_small_resolution():
    with pytest.raises(ValueError):
        union_coverage(np.zeros((1, 2)), _square((0, 0), 1.0)[None], np.array([4]),
                       (2.0, 2.0), resolution=8)


def test_polygon_area_analytic():
    assert _polygon_area(np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0]])) \
        == pytest.approx(3.0)
    assert _polygon_area(_square((0.0, 0.0), 2.0)) == pytest.approx(4.0)
    assert _polygon_area(np.zeros((2, 2))) == 0.0


# ---------------------------------------------------------------------------
# 8-13: the consolidated-zone construction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phi", [0.60, 0.70, 0.80, 0.85])
def test_consolidated_coverage_is_exact_and_overlap_free(phi):
    field = generate_consolidated_floe_field(
        LogNormalFSD(0.5), coverage=phi, n_floes=200, seed=11, diagnose=False,
        params=ConsolidationParams(laguerre=LaguerreParams(area_tol=1e-3)))
    realised = field.realised_coverage()
    assert abs(realised - phi) / phi < 1e-5          # criterion (a): << 0.5 %
    assert field.min_signed_gap() > -1e-9            # criterion (b): << 0.01 m
    assert field.compaction["nominal_coverage"] == pytest.approx(phi, rel=1e-5)


def test_consolidated_field_has_contact_network_the_buffer_field_cannot():
    buffer = generate_polygon_floe_field(LogNormalFSD(0.5), coverage=0.85,
                                         n_floes=200, seed=11)
    assert buffer.contacts(cutoff=1e-2)["i"].size == 0          # IN32-N01: no contacts
    field = generate_consolidated_floe_field(LogNormalFSD(0.5), coverage=0.85,
                                             n_floes=200, seed=11, diagnose=False)
    cs = field.contacts(cutoff=1e-2)
    assert cs["i"].size > 100
    coordination = 2.0 * cs["i"].size / field.n
    assert coordination > 3.0
    assert np.allclose(cs["gap"], 0.0, atol=1e-9)   # edge-sharing cells: exact contacts


def test_consolidated_zone_percolates_but_the_lead_zone_does_not():
    field = generate_consolidated_floe_field(LogNormalFSD(0.5), coverage=0.85,
                                             n_floes=250, seed=11,
                                             diagnose=False)
    perco = contact_percolation(field, contact_gap=1e-2)
    assert perco["percolating"]
    assert perco["largest_cluster_fraction"] > 0.5
    assert 4.0 < perco["coordination"] < 6.5
    # the isolated-floe negative control: a buffered field has no contacts at all
    buffer = generate_polygon_floe_field(LogNormalFSD(0.5), coverage=0.85,
                                         n_floes=250, seed=11)
    dead = contact_percolation(buffer, contact_gap=1e-2)
    assert dead["contacts"] == 0
    assert dead["clusters"] == buffer.n
    assert not dead["percolating"]


def test_coordination_grows_with_coverage():
    z = []
    for phi in (0.60, 0.75, 0.85):
        field = generate_consolidated_floe_field(LogNormalFSD(0.5), coverage=phi,
                                                 n_floes=200, seed=2,
                                                 diagnose=False)
        z.append(contact_percolation(field, contact_gap=1e-2)["coordination"])
    assert z[0] < z[1] < z[2]
    assert z[0] > 0.5                     # even the 60 % cover keeps a contact net


@pytest.mark.parametrize("factory", [LogNormalFSD(0.5), PowerLawFSD(2.5),
                                     BimodalFSD(), DeltaFSD(5.0)])
def test_consolidated_floe_areas_follow_the_fsd(factory):
    field = generate_consolidated_floe_field(factory, coverage=0.80, n_floes=200,
                                             seed=4, diagnose=False)
    moments = factory.moments()
    # the realised moments carry the tessellation area tolerance (criterion (b) of
    # IN32-N01 was 1 %); m2 weights the large floes, so it is the looser one
    assert field.r_eq.mean() == pytest.approx(moments["m1"], rel=5e-3)
    assert (field.r_eq ** 2).mean() == pytest.approx(moments["m2"], rel=1e-2)


def test_consolidated_field_is_deterministic_and_seed_sensitive():
    a = generate_consolidated_floe_field(LogNormalFSD(0.5), coverage=0.80,
                                         n_floes=120, seed=6, diagnose=False)
    b = generate_consolidated_floe_field(LogNormalFSD(0.5), coverage=0.80,
                                         n_floes=120, seed=6, diagnose=False)
    c = generate_consolidated_floe_field(LogNormalFSD(0.5), coverage=0.80,
                                         n_floes=120, seed=7, diagnose=False)
    assert _field_digest(a) == _field_digest(b)
    assert _field_digest(a) != _field_digest(c)


def test_consolidated_zone_geometry_options():
    for zone in ("band", "disc", "random"):
        field = generate_consolidated_floe_field(
            LogNormalFSD(0.5), coverage=0.80, n_floes=120, seed=3, diagnose=False,
            params=ConsolidationParams(zone=zone))
        assert field.compaction["nominal_coverage"] == pytest.approx(0.80, rel=1e-5)
        assert field.min_signed_gap() > -1e-9
    multi = generate_consolidated_floe_field(
        DeltaFSD(5.0), coverage=0.85, n_floes=120, seed=3, diagnose=False,
        params=ConsolidationParams(n_bands=3, band_axis=0))
    assert multi.compaction["n_bands"] == 3


# ---------------------------------------------------------------------------
# 14-16: zone ordering and area selection helpers
# ---------------------------------------------------------------------------

def test_zone_order_band_is_contiguous_and_random_is_not():
    rng = np.random.default_rng(1)
    box = (40.0, 25.0)
    sites = np.stack([rng.uniform(0.0, box[0], 200), rng.uniform(0.0, box[1], 200)],
                     axis=1)
    order, key = zone_order(sites, box, zone="band", n_bands=1, band_axis=1, seed=0)
    assert np.all(np.diff(key[order]) >= -1e-12)          # monotone coordinate
    head = sites[order[:40], 1]
    assert head.max() - head.min() < 0.35 * box[1]         # a band, not scattered
    order_r, _ = zone_order(sites, box, zone="random", seed=0)
    assert sorted(order_r.tolist()) == list(range(200))
    _, kd = zone_order(sites, box, zone="disc", seed=0)
    assert kd.min() >= 0.0


def test_zone_order_rejects_bad_arguments():
    sites = np.zeros((4, 2))
    with pytest.raises(ValueError):
        zone_order(sites, (1.0, 1.0), zone="spiral")
    with pytest.raises(ValueError):
        zone_order(sites, (-1.0, 1.0), zone="band")
    with pytest.raises(ValueError):
        zone_order(np.zeros((4, 3)), (1.0, 1.0), zone="band")
    with pytest.raises(ValueError):
        zone_order(sites, (1.0, 1.0), zone="band", band_axis=2)


def test_select_by_area_hits_the_target():
    areas = np.full(10, 2.0)
    order = np.arange(10)
    mask, realised = _select_by_area(order, areas, 11.0)
    assert mask.sum() == 6 and realised == pytest.approx(12.0)   # 10 vs 12 -> 12? no
    # 5 floes carry 10 (< 11) and 6 carry 12: |10-11| == |12-11| -> include 6
    mask2, realised2 = _select_by_area(order, areas, 10.2)
    assert mask2.sum() == 5 and realised2 == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 17-22: solver, locks, negative controls
# ---------------------------------------------------------------------------

def test_monodisperse_limit_reproduces_the_in32_field_bit_for_bit():
    plain = generate_polygon_floe_field(DeltaFSD(5.0), coverage=0.85, n_floes=200,
                                        seed=7)
    locked = generate_compacted_floe_field(DeltaFSD(5.0), coverage=0.85, n_floes=200,
                                           seed=7, compaction=None)
    for name in ("centroid", "verts", "count", "area", "perimeter", "r_eq", "mass",
                 "inertia", "thickness"):
        assert np.array_equal(np.asarray(getattr(plain, name)),
                              np.asarray(getattr(locked, name))), name
    assert _field_digest(plain) == _field_digest(locked)
    if MONODISPERSE_FIELD_DIGEST != "PLACEHOLDER":
        assert _field_digest(plain) == MONODISPERSE_FIELD_DIGEST


def test_h1_monodisperse_field_is_reproduced():
    lock = h1_reference_digest(coverage=0.80, domain_length=120.0,
                               domain_width=80.0, seed=0)
    assert lock["method"] == "h1_lattice"
    assert lock["deterministic"]
    assert lock["field_digest"] == lock["rerun_field_digest"]
    assert 0.75 < lock["realised_coverage"] < 0.85
    # the H1 lattice is monodisperse: every floe carries the same radius
    field = generate_consolidated_floe_field(DeltaFSD(5.0), coverage=0.85,
                                            n_floes=200, seed=7, diagnose=False)
    assert np.allclose(field.r_eq, 5.0, rtol=5e-3)


def test_compact_floe_field_preserves_areas_box_and_nominal_coverage():
    base = generate_polygon_floe_field(LogNormalFSD(0.5), coverage=0.80,
                                       n_floes=120, seed=9)
    comp = compact_floe_field(base, CompactionParams(stages=1, steps_per_stage=5,
                                                    projection_sweeps=4))
    assert np.array_equal(np.asarray(comp.area), np.asarray(base.area))
    assert np.array_equal(np.asarray(comp.verts), np.asarray(base.verts))
    assert np.array_equal(np.asarray(comp.count), np.asarray(base.count))
    assert comp.box == base.box
    assert comp.realised_coverage() == pytest.approx(base.realised_coverage(),
                                                     rel=1e-12)
    assert comp.compaction["residual_overlap"] <= 0.01
    assert comp.compaction["steps"] >= 1


def test_compact_floe_field_respects_pinning():
    base = generate_polygon_floe_field(LogNormalFSD(0.4), coverage=0.80,
                                       n_floes=120, seed=9)
    pinned = np.arange(base.n) < base.n // 2
    comp = compact_floe_field(base, CompactionParams(stages=1, steps_per_stage=8,
                                                     projection_sweeps=4),
                              pinned=pinned)
    assert np.allclose(np.asarray(comp.centroid)[pinned],
                       np.asarray(base.centroid)[pinned], atol=1e-12)
    assert comp.compaction["pinned"] == int(pinned.sum())
    assert comp.min_signed_gap() > -0.01


def test_compact_floe_field_rejects_bad_arguments():
    base = generate_polygon_floe_field(DeltaFSD(5.0), coverage=0.70, n_floes=40,
                                       seed=1)
    with pytest.raises(TypeError):
        compact_floe_field({"not": "a field"})
    with pytest.raises(ValueError):
        compact_floe_field(base, pinned=np.ones(base.n, dtype=bool))
    with pytest.raises(ValueError):
        compact_floe_field(base, pinned=np.ones(7, dtype=bool))


@pytest.mark.parametrize("kwargs", [
    {"stages": 0}, {"steps_per_stage": 0}, {"attraction": 0.0}, {"attraction": 1.5},
    {"relax": 0.0}, {"contact_tol": -1.0}, {"preload": -1.0},
    {"overlap_tol": -1.0}, {"final_overlap_tol": -1.0},
    {"final_overlap_tol": 1.0, "overlap_tol": 0.1}, {"max_step_rel": 0.0},
    {"projection_sweeps": 0}, {"polish_sweeps": -1}, {"energy_tol": -1.0},
    {"max_pair_gap": 1e-3},
])
def test_compaction_params_validation(kwargs):
    with pytest.raises(ValueError):
        CompactionParams(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"lead_coverage": 0.0}, {"lead_coverage": 1.0}, {"lead_coverage": -0.3},
    {"zone": "spiral"}, {"n_bands": 0}, {"band_axis": 3}, {"aspect": -1.0},
    {"jitter": -0.5},
])
def test_consolidation_params_validation(kwargs):
    with pytest.raises(ValueError):
        ConsolidationParams(**kwargs)


def test_generate_consolidated_rejects_bad_arguments():
    with pytest.raises(ValueError):
        generate_consolidated_floe_field(LogNormalFSD(0.5), coverage=1.2)
    with pytest.raises(ValueError):
        generate_consolidated_floe_field(LogNormalFSD(0.5), coverage=0.5,
                                         params=ConsolidationParams(lead_coverage=0.6))
    with pytest.raises(ValueError):
        generate_consolidated_floe_field(LogNormalFSD(0.5), coverage=0.8, n_floes=2)
    with pytest.raises(ValueError):
        generate_consolidated_floe_field(LogNormalFSD(0.5), coverage=0.8, n_floes=10,
                                         radii=np.ones(9))
    with pytest.raises(ValueError):
        generate_consolidated_floe_field(LogNormalFSD(0.5), coverage=0.8, n_floes=10,
                                         radii=-np.ones(10))


def test_lead_compaction_pinned_keeps_the_consolidated_sheet_rigid():
    field = generate_consolidated_floe_field(
        LogNormalFSD(0.5), coverage=0.80, n_floes=150, seed=8, diagnose=False,
        params=ConsolidationParams(lead_compaction=True,
                                   compaction=CompactionParams(stages=1,
                                                               steps_per_stage=6,
                                                               projection_sweeps=4)))
    assert "lead_compaction" in field.compaction
    assert field.compaction["lead_compaction"]["pinned"] > 0
    assert field.realised_coverage() == pytest.approx(0.80, rel=1e-5)
    assert field.min_signed_gap() > -0.01