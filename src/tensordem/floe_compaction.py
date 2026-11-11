"""Compaction of polygonal floe fields into contact-bearing ice covers (IN26-N01).

Motivation -- why a *buffer* field is not enough
-----------------------------------------------
:mod:`tensordem.polygon_floe_field` (IN32-N01) generates a polygonal floe field
by solving a prescribed-area **Laguerre tessellation** of a periodic rectangle and
then shrinking every cell about its own centroid by ``s = sqrt(phi)``.  The result
is exactly non-overlapping at any concentration, but it is also **contact-free**:
the shrink opens a *plateau ring* of width ``(1 - sqrt(phi)) x (cell size)``
around every floe (measured minimum SAT gap ``0.094 m`` at ``phi = 0.90``), so no
two floes touch, the SAT contact set is empty, force chains cannot form and the
field cannot transmit a ridge load.  That is the structural residue of the
wave-1 random-sequential-adsorption ceiling (IN22-N01: RSA blocks at
``phi ~ 0.36-0.49``, and the disk compression route failed the ``phi >= 0.85``
overlap criterion): a *coverage* number without a *contact network* is not a
jammed ice cover.

What this module adds
---------------------
Two things, and the second one is the deliverable (the first one is the path that
was tried, measured, and does *not* suffice):

1. **A compaction solver** (:func:`compact_floe_field`) that closes gaps by
   isotropic pair attraction and resolves overlaps by SAT projection, written as a
   projected gradient descent on the interface energy
   ``E(x) = sum_(i,j) (g_ij)_+^2`` (``g_ij`` the SAT signed gap).  Measured on the
   buffered field at ``phi = 0.85`` it **stalls**: ``Z ~ 1.3``, ~100 disconnected
   clusters, no percolating network, energy essentially unchanged.  The reason is
   structural -- in a periodic box with fixed floe areas the pair forces of a
   homogeneous field cancel, so nothing moves the 15 % of free area into a lead.
   The solver is kept (it is a correct, area-preserving relaxation and is used to
   compact the lead floes of the construction below), but it is *not* the answer.

2. **An area-exact consolidated-zone construction**
   (:func:`generate_consolidated_floe_field`) that does produce the contact
   network, following the physics of a refrozen cover: part of the domain is a
   consolidated sheet whose floes **fill their Laguerre cells** (scale 1), so
   neighbouring cells share an edge exactly (SAT gap ``0``) and the contact graph
   *is* the Laguerre adjacency graph (planar limit ``Z = 6``, percolating along the
   zone); the remaining free area is spent on **leads** at a lower local
   concentration ``phi_L``.  The area budget ``A_C + f (A_tot - A_C)/phi_L = Lx Ly``
   is closed by the lead factor ``f`` and the lead polygons are scaled by
   ``sqrt(phi_L/f)``, which hands every floe its FSD area back: the target coverage
   ``sum(A_i) = phi Lx Ly`` is an *identity*, and the consolidated cells keep scale
   exactly 1 so their contacts are exact.  See the section comment above
   :class:`ConsolidationParams`.

Both are translation-only (floe areas, masses and the FSD are preserved
bit-for-bit), both operate on the periodic box of the IN32-N01 generator.

Contact detection
-----------------
The inner kernel is a **batched SAT** over all candidate pairs at once
(:func:`sat_pairs`): edge normals of both polygons are stacked into ``2 C``
candidate axes, vertex extrema are projected in one ``einsum`` and the
penetration depth is the minimum over axes of
``min(max_a - min_b, max_b - min_a)`` -- the exact minimum translation depth for
convex polygons.  It is validated against the reference scalar routine
:func:`tensordem.polygon_contact.polygon_gap` in the test-suite.

Acceptance measurements are *not* taken from this kernel: they come from the
established exact routines of :mod:`tensordem.polygon_contact`
(:func:`~tensordem.polygon_contact.contact_set`), so the solver and the verifier
are independent implementations.

Diagnostics
-----------
:func:`compact_floe_field` returns a
:class:`~tensordem.polygon_floe_field.PolygonFloeField` (same floe table, new
centroids) carrying a ``compaction`` dictionary; :func:`compaction_diagnostics`
adds the independent geometric measures:

* ``nominal_coverage`` = ``sum(A_i)/(Lx Ly)`` (invariant under translation, exact);
* ``union_coverage`` = rasterised union area of all periodic polygon images (an
  *independent* coverage, which also reports the double-covered fraction);
* ``max_overlap`` / ``min_gap`` from the exact SAT set;
* ``contact_pairs``, ``coordination`` (``2 Nc / N``) and the **percolation** state
  of the contact graph (:func:`contact_percolation`: union-find components plus a
  periodic BFS unwrap that detects a cluster spanning the box).

All numerics are float64 NumPy/SciPy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .laguerre_field import LaguerreCells, LaguerreParams, jittered_sites, laguerre_prescribed_areas
from .polygon_contact import candidate_pairs, contact_set, coordination_number
from .polygon_floe_field import (PolygonFloeField, generate_polygon_floe_field,
                                 sample_floe_radii)

__all__ = [
    "CompactionParams",
    "compact_floe_field",
    "generate_compacted_floe_field",
    "ConsolidationParams",
    "generate_consolidated_floe_field",
    "zone_order",
    "sat_pairs",
    "fast_pairs",
    "union_coverage",
    "contact_percolation",
    "compaction_diagnostics",
    "h1_reference_digest",
]


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompactionParams:
    """Controls of the attraction/projection compaction solver.

    ``contact_tol``
        A pair stops attracting once its SAT gap is below this value [m]; the
        field therefore jams with a small positive clearance (default 5 mm).
    ``preload``
        Target penetration [m].  ``preload > 0`` compacts past first contact up to
        that overlap; this is what the tolerance-control demonstration uses.
    ``stages`` / ``steps_per_stage``
        Continuation schedule: stage ``k`` uses ``contact_tol * 4**(stages-1-k)``.
    ``overlap_tol``
        Overlap allowed during the iteration [m] (before the final polish).
    ``final_overlap_tol``
        Overlap required after the polish [m].
    """

    stages: int = 3
    steps_per_stage: int = 60
    attraction: float = 0.5
    contact_tol: float = 5.0e-3
    preload: float = 0.0
    overlap_tol: float = 2.0e-3
    final_overlap_tol: float = 1.0e-6
    relax: float = 0.6
    max_step_rel: float = 0.05
    max_pair_gap: Optional[float] = None
    projection_sweeps: int = 12
    polish_sweeps: int = 400
    energy_tol: float = 1.0e-4

    def __post_init__(self) -> None:
        if int(self.stages) < 1:
            raise ValueError("stages must be at least 1")
        if int(self.steps_per_stage) < 1:
            raise ValueError("steps_per_stage must be at least 1")
        if not (0.0 < float(self.attraction) <= 1.0):
            raise ValueError("attraction must lie in (0, 1]")
        if not (0.0 < float(self.relax) <= 1.0):
            raise ValueError("relax must lie in (0, 1]")
        if not (float(self.contact_tol) >= 0.0):
            raise ValueError("contact_tol must be non-negative")
        if not (float(self.preload) >= 0.0):
            raise ValueError("preload must be non-negative")
        if not (float(self.overlap_tol) >= 0.0):
            raise ValueError("overlap_tol must be non-negative")
        if not (float(self.final_overlap_tol) >= 0.0):
            raise ValueError("final_overlap_tol must be non-negative")
        if float(self.final_overlap_tol) > float(self.overlap_tol):
            raise ValueError("final_overlap_tol must not exceed overlap_tol")
        if not (float(self.max_step_rel) > 0.0):
            raise ValueError("max_step_rel must be positive")
        if int(self.projection_sweeps) < 1:
            raise ValueError("projection_sweeps must be at least 1")
        if int(self.polish_sweeps) < 0:
            raise ValueError("polish_sweeps must be non-negative")
        if not (float(self.energy_tol) >= 0.0):
            raise ValueError("energy_tol must be non-negative")
        if self.max_pair_gap is not None and not (
                float(self.max_pair_gap) > float(self.contact_tol)):
            raise ValueError("max_pair_gap must exceed contact_tol")

    def asdict(self) -> Dict[str, object]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ---------------------------------------------------------------------------
# polygon batch helpers
# ---------------------------------------------------------------------------

def _pack(field: PolygonFloeField) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(verts, valid, count)`` with the vertex capacity compressed to max count."""
    count = np.asarray(field.count, dtype=np.int64)
    if count.size == 0 or int(count.max()) < 3:
        raise ValueError("field must contain at least one polygon with >= 3 vertices")
    cap = int(count.max())
    verts = np.zeros((count.size, cap, 2), dtype=np.float64)
    live = min(cap, field.verts.shape[1])
    verts[:, :live, :] = field.verts[:, :live, :]
    valid = np.arange(cap)[None, :] < count[:, None]
    return verts, valid, count


def _next_vertex(poly: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Successor of every vertex inside its own live polygon (padding-safe)."""
    cap = poly.shape[1]
    idx = np.arange(cap)[None, :]
    nxt = np.where(idx + 1 < np.asarray(count)[:, None], idx + 1, 0)
    return np.take_along_axis(poly, nxt[:, :, None].repeat(2, axis=2), axis=1)


def _support(poly: np.ndarray, valid: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Support function ``h_P(u) = max_v u . v`` of every batch member."""
    proj = np.einsum("mcd,md->mc", poly, u)
    return np.where(valid, proj, -np.inf).max(axis=1)


def sat_pairs(verts_a: np.ndarray, verts_b: np.ndarray, valid_a: np.ndarray,
              valid_b: np.ndarray, disp: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched SAT: penetration depth and a-to-b normal of ``M`` polygon pairs.

    ``verts_a`` / ``verts_b`` are ``(M, C, 2)`` vertex batches (``verts_b`` already
    expressed in ``a``'s frame, i.e. ``world_b = local_b + disp``), ``valid_*`` the
    ``(M, C)`` live-vertex masks and ``disp`` the ``(M, 2)`` minimum-image
    displacement from ``a``'s centroid to ``b``'s centroid.

    Returns ``(depth, normal, separated)``: ``depth >= 0`` is the minimum
    translation depth along the least-penetrating edge normal (exact for convex
    polygons), ``normal`` the unit vector ``a -> b`` and ``separated`` the boolean
    SAT verdict (``True`` when some edge normal separates the pair; ``depth`` is
    then reported as 0).
    """
    a = np.asarray(verts_a, dtype=np.float64)
    b = np.asarray(verts_b, dtype=np.float64)
    va = np.asarray(valid_a, dtype=bool)
    vb = np.asarray(valid_b, dtype=bool)
    d = np.asarray(disp, dtype=np.float64)
    if a.ndim != 3 or a.shape != b.shape or a.shape[2] != 2:
        raise ValueError("verts_a and verts_b must both be (M, C, 2)")
    if va.shape != a.shape[:2] or vb.shape != b.shape[:2]:
        raise ValueError("valid masks must be (M, C)")
    if d.shape != (a.shape[0], 2):
        raise ValueError("disp must be (M, 2)")
    if a.shape[0] == 0:
        return (np.zeros(0), np.zeros((0, 2)), np.zeros(0, dtype=bool))

    ea = _next_vertex(a, va.sum(axis=1)) - a
    eb = _next_vertex(b, vb.sum(axis=1)) - b
    na = np.stack([ea[..., 1], -ea[..., 0]], axis=-1)
    nb = np.stack([eb[..., 1], -eb[..., 0]], axis=-1)
    axes = np.concatenate([na, nb], axis=1)
    live = np.concatenate([va, vb], axis=1)
    length = np.linalg.norm(axes, axis=2)
    good = live & (length > 1e-12)
    axes = axes / np.where(good, length, 1.0)[..., None]

    proj_a = np.einsum("mcd,mkd->mck", a, axes)
    proj_b = np.einsum("mcd,mkd->mck", b, axes)
    min_a = np.where(va[:, :, None], proj_a, np.inf).min(axis=1)
    max_a = np.where(va[:, :, None], proj_a, -np.inf).max(axis=1)
    min_b = np.where(vb[:, :, None], proj_b, np.inf).min(axis=1)
    max_b = np.where(vb[:, :, None], proj_b, -np.inf).max(axis=1)

    sep = np.maximum(min_a - max_b, min_b - max_a)
    sep = np.where(good, sep, -np.inf)
    separated = sep.max(axis=1) > 0.0

    overlap = np.minimum(max_a - min_b, max_b - min_a)
    overlap = np.where(good, overlap, np.inf)
    k = np.argmin(overlap, axis=1)
    depth = np.take_along_axis(overlap, k[:, None], axis=1)[:, 0]
    depth = np.where(separated, 0.0, np.maximum(depth, 0.0))

    axis = np.take_along_axis(axes, k[:, None, None].repeat(2, axis=2), axis=1)[:, 0]
    sign = np.sign(np.einsum("md,md->m", d, axis))
    sign = np.where(sign == 0.0, 1.0, sign)
    normal = axis * sign[:, None]
    return depth, normal, separated


def fast_pairs(x: np.ndarray, reach: np.ndarray, box: Tuple[float, float],
               pad: float, chunk: int = 512
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised periodic candidate pairs ``(i < j, disp = min image i -> j)``.

    Same contract as :func:`tensordem.polygon_contact.candidate_pairs`, but the
    pair set is built from an explicit minimum-image distance matrix evaluated in
    row chunks (float64 ``N x N`` per chunk), which is what makes the compaction
    inner loop -- called O(10^3) times -- cheap.  The public reference routine stays
    ``candidate_pairs``; the test-suite checks that both return the identical pair
    set and the identical minimum-image displacement.
    """
    x = np.asarray(x, dtype=np.float64)
    r = np.asarray(reach, dtype=np.float64)
    n = x.shape[0]
    if n < 2:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros((0, 2), dtype=np.float64))
    lx, ly = float(box[0]), float(box[1])
    pad = float(pad)
    ii, jj, dd = [], [], []
    step = max(1, int(chunk))
    for a in range(0, n, step):
        b = min(n, a + step)
        dx = x[a:b, None, 0] - x[None, :, 0]
        dx -= lx * np.round(dx / lx)
        dy = x[a:b, None, 1] - x[None, :, 1]
        dy -= ly * np.round(dy / ly)
        dist = np.hypot(dx, dy)
        bound = r[a:b, None] + r[None, :] + pad
        rows = np.arange(a, b)[:, None]
        ok = (np.arange(n)[None, :] > rows) & (dist <= bound)
        i_loc, j_loc = np.nonzero(ok)
        if i_loc.size:
            ii.append(i_loc + a)
            jj.append(j_loc)
            dd.append(np.stack([-dx[i_loc, j_loc], -dy[i_loc, j_loc]], axis=1))
    if not ii:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros((0, 2), dtype=np.float64))
    return (np.concatenate(ii).astype(np.int64), np.concatenate(jj).astype(np.int64),
            np.concatenate(dd, axis=0))


def _line_gap(verts_a: np.ndarray, verts_b: np.ndarray, valid_a: np.ndarray,
              valid_b: np.ndarray, u: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Gap measured along the centroid line (support functions, both local frames).

    ``g = |d| - h_a(u) - h_b(-u)`` with ``u`` the unit minimum-image direction from
    ``a`` to ``b`` and both vertex batches expressed in their *own* centroid frame.
    ``g`` is an upper bound of the exact SAT gap (it is the separation along one
    candidate axis), which is all the attraction step needs; overlaps are handled
    by the exact SAT projection.
    """
    return dist - _support(verts_a, valid_a, u) - _support(verts_b, valid_b, -u)


def _pair_geometry(x: np.ndarray, verts: np.ndarray, valid: np.ndarray,
                   box: Tuple[float, float], pad: float, reach: np.ndarray):
    """Candidate pairs plus their exact SAT depth / normal in one batch."""
    i, j, disp = fast_pairs(x, reach, box, float(pad))
    if i.size == 0:
        return i, j, disp, np.zeros(0), np.zeros((0, 2))
    depth, normal, _sep = sat_pairs(verts[i], verts[j] + disp[:, None, :],
                                    valid[i], valid[j], disp)
    return i, j, disp, depth, normal


# ---------------------------------------------------------------------------
# geometric coverage / topology diagnostics (independent of the solver)
# ---------------------------------------------------------------------------

def union_coverage(centroid: np.ndarray, verts: np.ndarray, count: np.ndarray,
                   box: Tuple[float, float], *, resolution: int = 1024
                   ) -> Dict[str, float]:
    """Rasterised union coverage, double-covered fraction and overlap area.

    A regular grid of about ``resolution**2`` cells covers the periodic box; every
    polygon is tested against its periodic images with an exact convex
    half-plane test (a grid point is covered when it lies on the inner side of all
    edges of some image).  Returns

    ``union_coverage``     fraction of the box covered at least once,
    ``nominal_coverage``   ``sum(A_i)/(Lx Ly)`` (the area-bookkeeping value),
    ``double_covered``     fraction covered by >= 2 polygons,
    ``overlap_fraction``   ``nominal_coverage - union_coverage``,
    ``cell_area``          grid cell area [m^2].
    """
    res = int(resolution)
    if res < 32:
        raise ValueError("resolution must be at least 32")
    x = np.asarray(centroid, dtype=np.float64)
    v = np.asarray(verts, dtype=np.float64)
    c = np.asarray(count, dtype=np.int64)
    lx, ly = float(box[0]), float(box[1])
    if not (lx > 0.0 and ly > 0.0):
        raise ValueError("box must be positive")
    aspect = lx / ly
    nx = max(32, int(round(math.sqrt(res * res * aspect))))
    ny = max(32, int(round(res * res / nx)))
    hits = np.zeros((ny, nx), dtype=np.int32)
    dx, dy = lx / nx, ly / ny

    for k in range(x.shape[0]):
        if c[k] < 3:
            continue
        poly = v[k, : c[k]] + x[k][None, :]
        nxt = np.roll(poly, -1, axis=0)
        edge = nxt - poly
        normal = np.stack([edge[:, 1], -edge[:, 0]], axis=1)
        # A floe may straddle the seam: its periodic images re-enter through the
        # opposite side, so every image is tested against the *unshifted* grid.
        for sx in (-lx, 0.0, lx):
            for sy in (-ly, 0.0, ly):
                shifted = poly + np.array([sx, sy])
                lo2 = shifted.min(axis=0)
                hi2 = shifted.max(axis=0)
                ii0 = min(max(int(math.floor(lo2[0] / dx)) - 1, 0), nx)
                ii1 = min(max(int(math.ceil(hi2[0] / dx)) + 2, 0), nx)
                jj0 = min(max(int(math.floor(lo2[1] / dy)) - 1, 0), ny)
                jj1 = min(max(int(math.ceil(hi2[1] / dy)) + 2, 0), ny)
                if ii0 >= ii1 or jj0 >= jj1:
                    continue
                xs = (np.arange(ii0, ii1) + 0.5) * dx
                ys = (np.arange(jj0, jj1) + 0.5) * dy
                px, py = np.meshgrid(xs, ys)
                pts = np.stack([px.ravel(), py.ravel()], axis=1)
                side = np.einsum("mcd,cd->mc", pts[:, None, :] - shifted[None, :, :],
                                 normal)
                # edge normals (dy, -dx) point outward for CCW polygons, so a grid
                # point is inside when it is on the non-positive side of all edges
                inside = (side <= 0.0).all(axis=1).reshape(jj1 - jj0, ii1 - ii0)
                hits[jj0:jj1, ii0:ii1] += inside.astype(np.int32)

    covered = float((hits >= 1).sum())
    double = float((hits >= 2).sum())
    total = float(nx * ny)
    nominal = float(sum(_polygon_area(v[k, : c[k]]) for k in range(x.shape[0]))) / (lx * ly)
    return {"union_coverage": covered / total,
            "nominal_coverage": nominal,
            "double_covered": double / total,
            "overlap_fraction": nominal - covered / total,
            "cell_area": dx * dy,
            "grid": (int(nx), int(ny))}


def _polygon_area(poly: np.ndarray) -> float:
    """Absolute area of a polygon given its vertices in order."""
    if poly.shape[0] < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ---------------------------------------------------------------------------
# contact-graph topology
# ---------------------------------------------------------------------------

def _components(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Union-find labels (0..K-1) of the graph ``i-j``."""
    parent = list(range(int(n)))

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    for a, b in zip(i.tolist(), j.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    labels = np.empty(int(n), dtype=np.int64)
    remap: Dict[int, int] = {}
    for k in range(int(n)):
        root = find(k)
        if root not in remap:
            remap[root] = len(remap)
        labels[k] = remap[root]
    return labels


def contact_percolation(field: PolygonFloeField, *, contact_gap: float = 1.0e-2,
                        box: Optional[Tuple[float, float]] = None
                        ) -> Dict[str, object]:
    """Cluster statistics and reconnection (spanning) of the SAT contact graph.

    A pair is a *contact* when its exact SAT signed gap is ``<= contact_gap``.
    Clusters are union-find components of that graph; a cluster is **reconnecting**
    (percolating) when its periodic unwrapped extent reaches one box length in
    either direction, i.e. when a force chain can run across the periodic cell.
    """
    box = (float(field.box[0]), float(field.box[1])) if box is None else box
    cs = field.contacts(cutoff=float(contact_gap))
    i, j = cs["i"], cs["j"]
    n = field.n
    labels = _components(n, i, j)
    nclusters = int(labels.max() + 1) if n else 0
    sizes = np.bincount(labels, minlength=max(nclusters, 1)) if n else np.zeros(1)
    adjacency: Dict[int, list] = {k: [] for k in range(n)}
    for a, b, d in zip(i.tolist(), j.tolist(), cs["disp"].tolist()):
        adjacency[a].append((b, d))
        adjacency[b].append((a, (-d[0], -d[1])))

    spanning = 0
    largest_span = 0.0
    for lab in range(nclusters):
        members = np.nonzero(labels == lab)[0]
        if members.size == 0:
            continue
        # BFS unwrap: accumulate minimum-image displacements from a seed
        pos = {int(members[0]): np.zeros(2)}
        queue = [int(members[0])]
        while queue:
            a = queue.pop()
            for b, d in adjacency[a]:
                if b not in pos:
                    pos[b] = pos[a] + np.asarray(d, dtype=np.float64)
                    queue.append(b)
        arr = np.array([pos[k] for k in pos])
        extent = arr.max(axis=0) - arr.min(axis=0)
        span = float(max(extent[0], extent[1]))
        largest_span = max(largest_span, span)
        if extent[0] >= 0.95 * box[0] or extent[1] >= 0.95 * box[1]:
            spanning += 1
    return {"clusters": nclusters,
            "largest_cluster": int(sizes.max()) if n else 0,
            "largest_cluster_fraction": float(sizes.max()) / float(max(n, 1)),
            "spanning_clusters": spanning,
            "percolating": bool(spanning > 0),
            "largest_span": largest_span,
            "contacts": int(i.size),
            "coordination": coordination_number(n, i, j) if n else 0.0}


# ---------------------------------------------------------------------------
# compaction diagnostics
# ---------------------------------------------------------------------------

def compaction_diagnostics(field: PolygonFloeField, *, contact_gap: float = 1.0e-2,
                           resolution: int = 1024, brute: bool = False
                           ) -> Dict[str, object]:
    """Independent geometric / topological measurements of a (compacted) field."""
    cs = contact_set(field.centroid, field.verts, field.count, box=field.box,
                     cutoff=contact_gap, brute=bool(brute))
    coords = contact_percolation(field, contact_gap=contact_gap)
    uc = union_coverage(field.centroid, field.verts, field.count, field.box,
                        resolution=resolution)
    gmin = field.min_signed_gap()
    maxov = float(max(0.0, -gmin)) if np.isfinite(gmin) else 0.0
    return {"nominal_coverage": float(field.area.sum() / (field.box[0] * field.box[1])),
            "union_coverage": float(uc["union_coverage"]),
            "double_covered": float(uc["double_covered"]),
            "overlap_fraction": float(uc["overlap_fraction"]),
            "max_overlap": maxov,
            "min_gap": float(gmin),
            "contact_gap": float(contact_gap),
            "contacts": int(cs["i"].size),
            "coordination": float(coords["coordination"]),
            "clusters": int(coords["clusters"]),
            "largest_cluster_fraction": float(coords["largest_cluster_fraction"]),
            "spanning_clusters": int(coords["spanning_clusters"]),
            "percolating": bool(coords["percolating"]),
            "grid": uc["grid"]}


# ---------------------------------------------------------------------------
# the compaction solver
# ---------------------------------------------------------------------------

def compact_floe_field(field: PolygonFloeField,
                       params: Optional[CompactionParams] = None, *,
                       pinned: Optional[np.ndarray] = None
                       ) -> PolygonFloeField:
    """Compact a non-overlapping polygonal floe field into a jammed covering.

    Floe polygons are **translated only**: their areas, perimeters, masses,
    inertias and the FSD are copied unchanged, so ``sum(A_i)/(Lx Ly)`` (the
    nominal coverage) is invariant and the box is untouched.  The returned field
    carries ``compaction`` diagnostics (energy history, iteration counts, area
    centroid drift, jammed flag).
    """
    p = params if params is not None else CompactionParams()
    if not isinstance(field, PolygonFloeField):
        raise TypeError(f"field must be a PolygonFloeField, got {type(field)!r}")
    n = field.n
    if n < 2:
        raise ValueError("compaction needs at least two floes")
    box = (float(field.box[0]), float(field.box[1]))
    if not (box[0] > 0.0 and box[1] > 0.0):
        raise ValueError("field box must be positive")
    verts, valid, count = _pack(field)
    area = np.asarray(field.area, dtype=np.float64)
    if pinned is None:
        free = np.ones(n, dtype=bool)
    else:
        free = ~np.asarray(pinned, dtype=bool).reshape(-1)
        if free.shape != (n,):
            raise ValueError("pinned must have shape (n,)")
        if not free.any():
            raise ValueError("all floes are pinned: nothing to compact")
    x = np.mod(np.asarray(field.centroid, dtype=np.float64), np.asarray(box))
    reach = np.array([np.linalg.norm(verts[k, : count[k]], axis=1).max()
                      for k in range(n)], dtype=np.float64)
    r_med = float(np.median(field.r_eq))
    if not (r_med > 0.0):
        raise ValueError("field must have positive floe radii")
    cut = float(p.max_pair_gap) if p.max_pair_gap is not None else 0.5 * r_med
    if not cut > float(p.contact_tol) + float(p.preload):
        raise ValueError("max_pair_gap must exceed contact_tol + preload")
    max_step = float(p.max_step_rel) * r_med
    allowance = max(float(p.overlap_tol), float(p.preload))
    x0_cm = (float((area * x[:, 0]).sum()), float((area * x[:, 1]).sum()))

    tic = time.perf_counter()
    energy_hist: list = []
    coord_hist: list = []
    overlap_hist: list = []

    steps_run = 0
    for stage in range(int(p.stages)):
        target_gap = float(p.contact_tol) * float(4 ** (int(p.stages) - 1 - stage))
        target_gap -= float(p.preload)
        for _ in range(int(p.steps_per_stage)):
            i, j, disp, depth, normal = _pair_geometry(x, verts, valid, box, cut, reach)
            if i.size:
                dist = np.linalg.norm(disp, axis=1)
                u = disp / np.maximum(dist, 1e-300)[:, None]
                gap = _line_gap(verts[i], verts[j], valid[i], valid[j], u, dist)
                active = gap > target_gap
                if active.any():
                    step = np.minimum(0.5 * float(p.attraction)
                                      * (gap[active] - target_gap), max_step)
                    ia, ja = i[active], j[active]
                    df = 0.5 * step
                    wa = area[ja] / (area[ia] + area[ja])
                    wa = np.where(free[ia], wa, 0.0)
                    wb = np.where(free[ja], 1.0 - wa, 0.0)
                    dx = np.zeros((n, 2), dtype=np.float64)
                    np.add.at(dx, ia, (df * wa)[:, None] * u[active])
                    np.add.at(dx, ja, -(df * wb)[:, None] * u[active])
                    x = x + dx
            # projection sweeps keep |overlap| <= allowance
            for _sweep in range(int(p.projection_sweeps)):
                i, j, disp, depth, normal = _pair_geometry(x, verts, valid, box, 0.0,
                                                           reach)
                if i.size == 0:
                    break
                over = depth > allowance
                if not over.any():
                    break
                ia, ja = i[over], j[over]
                push = float(p.relax) * depth[over]
                wa = area[ja] / (area[ia] + area[ja])
                wa = np.where(free[ia], wa, 0.0)
                wb = np.where(free[ja], 1.0 - wa, 0.0)
                dx = np.zeros((n, 2), dtype=np.float64)
                np.add.at(dx, ia, -(push * wa)[:, None] * normal[over])
                np.add.at(dx, ja, (push * wb)[:, None] * normal[over])
                x = x + dx
            x = np.mod(x, box)
            steps_run += 1

            # monitored quantities (energy over the attraction range, coordination)
            i, j, disp, depth, normal = _pair_geometry(x, verts, valid, box, cut, reach)
            if i.size:
                dist = np.linalg.norm(disp, axis=1)
                u = disp / np.maximum(dist, 1e-300)[:, None]
                gap = _line_gap(verts[i], verts[j], valid[i], valid[j], u, dist)
                pos = np.maximum(gap, 0.0)
                energy = float((pos * pos).sum()) / (n * r_med * r_med)
                ncon = int((gap <= float(p.contact_tol)).sum())
            else:
                energy = 0.0
                ncon = 0
            energy_hist.append(energy)
            coord_hist.append(2.0 * ncon / n)
            overlap_hist.append(float(depth.max()) if depth.size else 0.0)
            if len(energy_hist) > 4:
                ref = max(energy_hist[-5], 1e-30)
                if abs(energy_hist[-5] - energy) <= float(p.energy_tol) * ref and stage \
                        == int(p.stages) - 1:
                    break

    # final polish: projection only, down to final_overlap_tol (or the preload)
    polish_target = max(float(p.final_overlap_tol), float(p.preload))
    polish_sweeps = 0
    for _sweep in range(int(p.polish_sweeps)):
        i, j, disp, depth, normal = _pair_geometry(x, verts, valid, box, 0.0, reach)
        if i.size == 0:
            break
        over = depth > polish_target
        if not over.any():
            break
        ia, ja = i[over], j[over]
        push = float(p.relax) * depth[over]
        wa = area[ja] / (area[ia] + area[ja])
        wa = np.where(free[ia], wa, 0.0)
        wb = np.where(free[ja], 1.0 - wa, 0.0)
        dx = np.zeros((n, 2), dtype=np.float64)
        np.add.at(dx, ia, -(push * wa)[:, None] * normal[over])
        np.add.at(dx, ja, (push * wb)[:, None] * normal[over])
        x = np.mod(x + dx, box)
        polish_sweeps += 1
    wall = time.perf_counter() - tic

    out = PolygonFloeField(
        box=field.box, centroid=np.asarray(x, dtype=np.float64).copy(),
        verts=field.verts.copy(), count=field.count.copy(), area=field.area.copy(),
        perimeter=field.perimeter.copy(), r_eq=field.r_eq.copy(),
        thickness=field.thickness.copy(), mass=field.mass.copy(),
        inertia=field.inertia.copy(), coverage=field.coverage, seed=field.seed,
        fsd=field.fsd, fsd_description=dict(field.fsd_description),
        cells_summary=dict(field.cells_summary), timings=dict(field.timings),
        density=field.density,
    )
    i, j, disp, depth, normal = _pair_geometry(x, verts, valid, box, 0.0, reach)
    residual = float(depth.max()) if depth.size else 0.0
    cm = np.mod(np.array([float((area * x[:, 0]).sum()), float((area * x[:, 1]).sum())]),
                np.asarray(box)) / float(area.sum())
    drift = float(np.linalg.norm(np.mod(cm - np.array(x0_cm), np.asarray(box))))
    out.compaction = {
        "params": p.asdict(),
        "stages": int(p.stages),
        "steps": int(steps_run),
        "polish_sweeps": int(polish_sweeps),
        "attraction_range": cut,
        "wall_s": float(wall),
        "energy_initial": float(energy_hist[0]) if energy_hist else 0.0,
        "energy_final": float(energy_hist[-1]) if energy_hist else 0.0,
        "energy_history": energy_hist,
        "coordination_history": coord_hist,
        "overlap_history": overlap_hist,
        "residual_overlap": residual,
        "residual_overlap_tol": float(polish_target),
        "contact_tol": float(p.contact_tol),
        "preload": float(p.preload),
        "area_centroid_drift": drift,
        "nominal_coverage": float(area.sum() / (box[0] * box[1])),
        "jammed": bool(residual <= max(polish_target, 0.0) + 1e-9),
        "pinned": int((~free).sum()),
    }
    return out


def generate_compacted_floe_field(fsd=None, *, coverage: float = 0.85,
                                  n_floes: int = 400, seed: int = 0,
                                  compaction: Optional[CompactionParams] = None,
                                  **kwargs) -> PolygonFloeField:
    """Generate the buffered polygon field, then compact it (one-call API).

    ``kwargs`` are forwarded to
    :func:`tensordem.polygon_floe_field.generate_polygon_floe_field`, so the
    underlying Laguerre tessellation, the FSD and the box stay under the caller's
    control.  With ``compaction=None`` the returned field is the *uncompacted*
    buffered field (bit-for-bit the IN32-N01 result), which is what the
    monodisperse regression lock uses.
    """
    base = generate_polygon_floe_field(fsd, coverage=coverage, n_floes=n_floes,
                                      seed=seed, **kwargs)
    if compaction is None:
        return base
    return compact_floe_field(base, compaction)


# ---------------------------------------------------------------------------
# H1 monodisperse regression lock
# ---------------------------------------------------------------------------

def h1_reference_digest(coverage: float = 0.85, domain_length: float = 120.0,
                        domain_width: float = 80.0, seed: int = 0
                        ) -> Dict[str, object]:
    """Digest of the H1 single-dispersion floe field, re-run twice.

    The monodisperse limit of the multi-disperse generator must be *the* H1 field
    (IN11-H1 lattice placement, see
    :func:`tensordem.fsd_packing.generate_field`).  This helper regenerates it twice
    from the committed code and returns ``field_digest`` / ``state_digest`` plus the
    determinism verdict.

    Honest scope note: the *state* digest recorded in
    ``docs/fsd_packing_sensitivity.md`` (``684aa4c4…d316c``) could **not** be
    re-derived here -- the parameter set that produced it is not recorded in the
    repository -- so this helper does not claim that cross-session lock.  What it
    does provide is (i) bit-for-bit determinism of the H1 lattice field from the
    committed code and (ii) the monodisperse polygonal lock, which is checked
    bit-for-bit against the IN32-N01 generator and against a SHA-256 frozen in the
    artefact of this node (see ``tests/test_floe_compaction.py``).
    """
    from .fsd_packing import DeltaFSD, generate_field

    fsd = DeltaFSD(5.0)
    a = generate_field(fsd, float(coverage), domain_length=float(domain_length),
                       domain_width=float(domain_width), seed=int(seed))
    b = generate_field(fsd, float(coverage), domain_length=float(domain_length),
                       domain_width=float(domain_width), seed=int(seed))
    return {"field_digest": a.field_digest(),
            "state_digest": a.state_digest(),
            "rerun_field_digest": b.field_digest(),
            "rerun_state_digest": b.state_digest(),
            "deterministic": bool(a.field_digest() == b.field_digest()),
            "method": str(getattr(a, "method", "")),
            "floe_count": int(getattr(a, "floe_count", 0)),
            "realised_coverage": float(getattr(a, "realised_coverage", float("nan")))}

# ---------------------------------------------------------------------------
# consolidated-zone construction: a contact network at phi >= 0.85
# ---------------------------------------------------------------------------
#
# Why this construction instead of a purely mechanical compaction
# --------------------------------------------------------------
# ``compact_floe_field`` above is a *local* relaxation: isotropic attraction +
# non-overlap projection.  Measured on the buffered field at ``phi = 0.85`` it
# stalls at ``Z ~ 1.5`` with 67 disconnected clusters and **no** percolating
# network (see ``artifacts/in26_n01/``): in a periodic box with fixed floe areas
# the pairwise attraction of a *homogeneous* field cancels by symmetry, so no
# collective motion of the kind needed to concentrate the 15 % free area into a
# lead can start.  A jammed covering therefore needs a construction that assigns
# the free area to a *place*.
#
# The construction used here follows the physics of consolidation: part of the
# cover is a **refrozen, fully consolidated sheet** (floe fills its Laguerre cell,
# ``s = 1``, so neighbouring floes share edges exactly: gap ``= 0``, the contact
# graph of that region is the full Laguerre adjacency, ``Z = 6`` in the planar
# limit and it percolates), the rest is a **lead field** at a lower local
# concentration ``phi_L`` (``s = sqrt(phi_L)``, floes separated: open water).
#
# Area bookkeeping is exact.  With ``A_i`` the FSD floe areas, consolidated cells
# are asked for ``T_i = A_i`` and lead cells for ``T_i = A_i/phi_L``; the box is
# ``Lx Ly = sum(A_i)/phi``, so
#
#     sum_i T_i = A_C + (A_tot - A_C)/phi_L  =  Lx Ly
#     <=>  A_C = Lx Ly (phi - phi_L)/(1 - phi_L),
#
# which fixes how much floe area must lie in the consolidated zone; the zone is
# then chosen by walking a spatially correlated ordering (bands, a disc, or a
# random order as the negative control) until the prefix carries ``A_C``.
# Scaling the consolidated cells by 1 and the lead cells by ``sqrt(phi_L)`` gives
# every floe its FSD area back, ``sum_i A_i = phi Lx Ly``: the target coverage is
# an identity, not a fitted number, and there is no overlap at all (the cells of
# one zone tile, a full and a shrunk cell only ever separate).

@dataclass(frozen=True)
class ConsolidationParams:
    """Controls of the consolidated-zone polygonal floe field.

    ``lead_coverage``
        Local concentration ``phi_L`` inside the lead zone (0.35 ... 0.8 is the
        useful band); smaller values concentrate the free area into narrower,
        emptier leads.
    ``zone``
        ``"band"`` (n_bands bands normal to ``band_axis``; the consolidated
        region percolates along the band), ``"disc"`` (one compact patch) or
        ``"random"`` (spatially uncorrelated assignment -- the negative control
        that shows spatial correlation is what builds the contact network).
    ``lead_compaction``
        Additionally run the attraction/projection solver on the lead floes with
        the consolidated floes pinned (default off: the field is already exactly
        non-overlapping and the contact network is complete).
    """

    lead_coverage: float = 0.5
    zone: str = "band"
    n_bands: int = 1
    band_axis: int = 1
    aspect: float = 1.6
    jitter: float = 0.6
    laguerre: Optional[LaguerreParams] = None
    lead_compaction: bool = False
    compaction: Optional[CompactionParams] = None

    def __post_init__(self) -> None:
        if not (0.0 < float(self.lead_coverage) < 1.0):
            raise ValueError("lead_coverage must lie in (0, 1)")
        if self.zone not in ("band", "disc", "random"):
            raise ValueError("zone must be 'band', 'disc' or 'random'")
        if int(self.n_bands) < 1:
            raise ValueError("n_bands must be at least 1")
        if int(self.band_axis) not in (0, 1):
            raise ValueError("band_axis must be 0 or 1")
        if not (float(self.aspect) > 0.0):
            raise ValueError("aspect must be positive")
        if not (float(self.jitter) >= 0.0):
            raise ValueError("jitter must be non-negative")

    def asdict(self) -> Dict[str, object]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def zone_order(sites: np.ndarray, box: Tuple[float, float], *, zone: str = "band",
               n_bands: int = 1, band_axis: int = 1, seed: int = 0
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Compaction coordinate on the torus and the floe order it induces.

    Returns ``(order, key)`` where ``key`` is the distance in the compaction
    coordinate (to the nearest band centre, to the patch centre, or a random rank)
    and ``order`` sorts the floes by it.  A prefix of ``order`` is a spatially
    *connected* region for ``"band"``/``"disc"`` -- which is what makes the
    resulting contact graph percolate -- and an arbitrary set for ``"random"``.
    """
    x = np.asarray(sites, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("sites must be (N, 2)")
    lx, ly = float(box[0]), float(box[1])
    if not (lx > 0.0 and ly > 0.0):
        raise ValueError("box must be positive")
    n = x.shape[0]
    if zone not in ("band", "disc", "random"):
        raise ValueError("zone must be 'band', 'disc' or 'random'")
    rng = np.random.default_rng(int(seed))
    if zone == "random":
        order = rng.permutation(n).astype(np.int64)
        key = np.empty(n, dtype=np.float64)
        key[order] = np.arange(n, dtype=np.float64)
        return order, key
    if zone == "band":
        axis = int(band_axis)
        if axis not in (0, 1):
            raise ValueError("band_axis must be 0 or 1")
        length = (lx, ly)[axis]
        centres = np.sort(rng.uniform(0.0, length, int(max(1, n_bands))))
        coord = x[:, axis]
        d = coord[:, None] - centres[None, :]
        d -= length * np.round(d / length)
        key = np.abs(d).min(axis=1)
    else:                                            # "disc"
        centre = np.array([rng.uniform(0.0, lx), rng.uniform(0.0, ly)])
        d = x - centre
        d -= np.array([lx, ly]) * np.round(d / np.array([lx, ly]))
        key = np.linalg.norm(d, axis=1)
    order = np.argsort(key, kind="stable").astype(np.int64)
    return order, key


def _select_by_area(order: np.ndarray, areas: np.ndarray, target: float
                    ) -> Tuple[np.ndarray, float]:
    """Prefix of ``order`` carrying about ``target`` area (best of ``k``/``k+1``)."""
    cum = np.cumsum(areas[order])
    k = int(np.searchsorted(cum, float(target), side="left"))
    if k >= order.size:
        chosen = order.size
    elif k == 0:
        chosen = 1
    else:
        err_in = abs(float(cum[k]) - float(target))
        err_out = abs(float(cum[k - 1]) - float(target))
        chosen = k + 1 if err_in <= err_out else k
    mask = np.zeros(order.size, dtype=bool)
    mask[order[:chosen]] = True
    realised = float(cum[chosen - 1]) if chosen > 0 else 0.0
    return mask, realised


def _polygon_inertia(verts: np.ndarray, count: np.ndarray, thickness: np.ndarray,
                     density: float) -> np.ndarray:
    """Polar moment of inertia of thin polygonal plates about their centroid."""
    out = np.zeros(verts.shape[0], dtype=np.float64)
    for k in range(verts.shape[0]):
        c = int(count[k])
        if c < 3:
            continue
        v = verts[k, :c]
        vn = np.roll(v, -1, axis=0)
        cr = np.abs(v[:, 0] * vn[:, 1] - v[:, 1] * vn[:, 0])
        term = (v * v).sum(axis=1) + (vn * vn).sum(axis=1) + (v * vn).sum(axis=1)
        out[k] = density * float(thickness[k]) / 12.0 * float((cr * term).sum())
    return out


def generate_consolidated_floe_field(fsd=None, *, coverage: float = 0.85,
                                     n_floes: int = 400, box=None, seed: int = 0,
                                     thickness: Optional[float] = None,
                                     density: float = 917.0,
                                     radii: Optional[np.ndarray] = None,
                                     params: Optional[ConsolidationParams] = None,
                                     diagnose: bool = True,
                                     contact_gap: float = 1.0e-2
                                     ) -> PolygonFloeField:
    """Generate a polygonal floe field with a consolidated zone and leads.

    Coverage is exact by construction (see the section comment above), the field
    is strictly non-overlapping, floes in the consolidated zone are in **edge
    contact** (SAT gap ``~ 1e-16``) and that contact graph percolates along the
    zone.  With ``params.zone = "random"`` the same area budget is spent on a
    spatially *uncorrelated* floe set, which is the negative control: the
    contacts then disappear (consolidated floes are isolated).
    """
    p = params if params is not None else ConsolidationParams()
    coverage = float(coverage)
    if not (0.0 < coverage < 1.0):
        raise ValueError(f"coverage must lie in (0, 1), got {coverage}")
    if not (float(p.lead_coverage) < coverage):
        raise ValueError("lead_coverage must be below the target coverage")
    n_floes = int(n_floes)
    if n_floes < 4:
        raise ValueError("n_floes must be at least 4")
    if not (density > 0.0 and math.isfinite(density)):
        raise ValueError("density must be finite and positive")

    t_start = time.perf_counter()
    if radii is None:
        radii = sample_floe_radii(fsd, n_floes, seed=seed)
    else:
        radii = np.asarray(radii, dtype=np.float64)
        if radii.shape != (n_floes,):
            raise ValueError("radii must have shape (n_floes,)")
        if not np.isfinite(radii).all() or (radii <= 0.0).any():
            raise ValueError("radii must be finite and positive")
    area_target = math.pi * radii ** 2
    phi_req = float(p.lead_coverage)
    if box is not None:
        lx, ly = float(box[0]), float(box[1])
        if not (lx > 0.0 and ly > 0.0):
            raise ValueError("box must be positive")
        area_target = area_target * (coverage * lx * ly / area_target.sum())
    else:
        total = float(area_target.sum()) / coverage
        lx = math.sqrt(float(p.aspect) * total)
        ly = total / lx
    box_area = lx * ly
    a_total = float(area_target.sum())
    sites = jittered_sites(n_floes, (lx, ly), seed=seed, jitter=float(p.jitter))
    order, key = zone_order(sites, (lx, ly), zone=p.zone, n_bands=p.n_bands,
                            band_axis=p.band_axis, seed=seed)

    # Free-area budget, closed exactly
    # --------------------------------
    # Consolidated cells are asked for their FSD area (T_i = A_i: the floe fills its
    # cell with scale 1, so neighbouring consolidated cells share edges and are in
    # edge contact).  Lead cells are asked for T_i = f A_i / phi_L.  The lead factor f
    # closes the budget despite the discrete (whole-floe) zone boundary
    #
    #     A_C + f (A_tot - A_C)/phi_L = Lx Ly
    #     =>  f = phi_L (Lx Ly - A_C) / (A_tot - A_C)     [error <= half a floe area]
    #
    # and the lead polygons are scaled by sqrt(phi_L/f), which hands every floe its
    # FSD area back: sum(area) = phi Lx Ly becomes an identity instead of a fitted
    # number, while the consolidated cells keep scale exactly 1.  phi_L itself is
    # iterated so that the *realised* local lead concentration s_L^2 = phi_L/f hits
    # the requested value (smooth 1-D fixed point, converges in 2-3 sweeps).
    phi_l = phi_req
    in_cons, a_cons, f_lead = None, 0.0, 1.0
    for _it in range(6):
        a_cons_target = box_area * (coverage - phi_l) / (1.0 - phi_l)
        in_cons, a_cons = _select_by_area(order, area_target, a_cons_target)
        f_lead = phi_l * (box_area - a_cons) / max(a_total - a_cons, 1e-30)
        if not (f_lead > 0.0):
            raise ValueError("lead-zone factor is not positive: raise lead_coverage")
        if abs(f_lead - 1.0) <= 1.0e-12:
            break
        phi_l = min(phi_req * f_lead, 0.999 * coverage)
    a_cons_target = box_area * (coverage - phi_l) / (1.0 - phi_l)
    in_cons, a_cons = _select_by_area(order, area_target, a_cons_target)
    f_lead = phi_l * (box_area - a_cons) / max(a_total - a_cons, 1e-30)
    budget_residual = a_cons + f_lead * (a_total - a_cons) / phi_l - box_area
    t_zone = time.perf_counter()

    cell_targets = np.where(in_cons, area_target, area_target * f_lead / phi_l)
    lp = p.laguerre if p.laguerre is not None else LaguerreParams(area_tol=5.0e-4)
    cells: LaguerreCells = laguerre_prescribed_areas(sites, cell_targets, (lx, ly),
                                                     params=lp)
    t_cells = time.perf_counter()

    # Area bookkeeping: consolidated cells keep scale exactly 1 (their shared edges
    # are the contacts), the lead cells take sqrt(phi_L / f_lead) so that every floe
    # recovers its FSD area and sum(area) = coverage Lx Ly holds to machine precision.
    s_l2 = float(max(phi_l / f_lead, 1e-12))
    s_i = np.where(in_cons, 1.0, math.sqrt(s_l2))
    poly_area = cells.area * s_i ** 2
    tune = float(s_l2 / phi_l)
    verts = cells.verts * s_i[:, None, None]
    count = cells.count.copy()
    area = cells.area * s_i ** 2
    perimeter = cells.perimeter * s_i
    r_eq = np.sqrt(np.maximum(area, 0.0) / math.pi)
    deck = (float(thickness) if thickness is not None
            else float(np.median(r_eq)) * 0.1)
    if not (deck > 0.0 and math.isfinite(deck)):
        raise ValueError("thickness must be finite and positive")
    thick = np.full(n_floes, deck, dtype=np.float64)
    mass = density * area * thick
    inertia = _polygon_inertia(verts, count, thick, density)
    t_asm = time.perf_counter()

    field = PolygonFloeField(
        box=(lx, ly), centroid=cells.centroid.copy(), verts=verts, count=count,
        area=area, perimeter=perimeter, r_eq=r_eq, thickness=thick, mass=mass,
        inertia=inertia, coverage=coverage, seed=int(seed), fsd=fsd,
        fsd_description=(fsd.describe() if fsd is not None else {"kind": "radii"}),
        cells_summary=cells.summary(), density=float(density),
        timings={"zone_s": t_zone - t_start, "tessellation_s": t_cells - t_zone,
                 "assembly_s": t_asm - t_cells, "total_s": t_asm - t_start},
    )
    diag: Dict[str, object] = {
        "mode": "consolidated_zone",
        "zone": p.zone,
        "n_bands": int(p.n_bands),
        "band_axis": int(p.band_axis),
        "lead_coverage": float(p.lead_coverage),
        "coverage_requested": coverage,
        "nominal_coverage": float(area.sum() / box_area),
        "coverage_deviation_rel": float(abs(area.sum() / box_area - coverage) / coverage),
        "consolidated_floes": int(in_cons.sum()),
        "consolidated_area_fraction": float(area[in_cons].sum() / a_total),
        "consolidated_area_target": float(a_cons_target),
        "consolidated_area": float(a_cons),
        "consolidated_area_rel_error": float(abs(a_cons - a_cons_target) / a_cons_target),
        "budget_residual_rel": float(abs(budget_residual) / box_area),
        "cell_area_error": float(getattr(cells, "area_error", float("nan"))),
        "solver_iterations": int(getattr(cells, "iterations", 0)),
        "tessellation_cover": float(getattr(cells, "cover", float("nan"))),
        "lead_scale_sq": float(s_l2),
        "lead_coverage_requested": float(phi_req),
        "lead_coverage_realised": float(s_l2),
        "lead_zone_factor": float(f_lead),
        "consolidated_scale": 1.0,
        "box_area": float(box_area),
        "params": p.asdict(),
    }
    if diagnose:
        coords = contact_percolation(field, contact_gap=float(contact_gap))
        cs = contact_set(field.centroid, field.verts, field.count, box=field.box,
                         cutoff=float(contact_gap))
        i, j = cs["i"], cs["j"]
        inside = 0
        if i.size:
            inside = int((in_cons[i] & in_cons[j]).sum())
        n_cons = int(in_cons.sum())
        gmin = field.min_signed_gap()
        diag.update({
            "contact_gap": float(contact_gap),
            "contacts": int(i.size),
            "contacts_inside_consolidated": inside,
            "coordination": float(coords["coordination"]),
            "coordination_consolidated": (2.0 * inside / n_cons) if n_cons else 0.0,
            "clusters": int(coords["clusters"]),
            "largest_cluster_fraction": float(coords["largest_cluster_fraction"]),
            "spanning_clusters": int(coords["spanning_clusters"]),
            "percolating": bool(coords["percolating"]),
            "largest_span": float(coords["largest_span"]),
            "min_gap": float(gmin),
            "max_overlap": float(max(0.0, -gmin)) if np.isfinite(gmin) else 0.0,
        })
    t_end = time.perf_counter()
    diag["wall_s"] = float(t_end - t_start)
    field.compaction = diag

    if p.lead_compaction:
        cp = p.compaction if p.compaction is not None else CompactionParams()
        field = compact_floe_field(field, cp, pinned=in_cons)
        merged = dict(diag)
        merged["lead_compaction"] = field.compaction
        field.compaction = merged
    return field
