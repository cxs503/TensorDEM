"""Exact geometry for three-dimensional polyhedral ice floes.

The 2-D floe field of :mod:`tensordem.polygon_floe_field` (m7768 wave) is a set
of convex polygons produced by a Laguerre (power) tessellation.  Lifting that
capability to three dimensions needs three ingredients that a 2-D polygon does
not have:

1. **a body** with volume, centroid and a full inertia tensor (not only a polar
   moment), so that out-of-plane rotations and buoyancy torques are physical;
2. **a generative geometry**: convex polyhedra obtained either from a genuine
   3-D Laguerre/Voronoi tessellation of the ice volume or by extruding a 2-D
   floe polygon (with plate thickness :math:`t`), which is what makes the
   :math:`t \\to 0` degeneracy check possible at all;
3. **exact mass properties** (divergence-theorem tetrahedra), so that the mass
   and inertia used by the dynamics are not approximations.

Representation
--------------
Every body is stored as an intersection of half-spaces
:math:`\\{x : n_k \\cdot x \\le d_k\\}` with outward unit normals.  Clipping
(by a tessellation bisector, by a wall, by the water plane) is then *appending
one half-space*, which keeps the representation closed under every geometric
operation the DEM needs.  Vertices are recovered as the feasible triple-plane
intersections; for the low plane counts of a floe (8-24) this is both exact and
cheap.

Frame conventions
-----------------
Bodies are recentred on their centroid and rotated into their **principal
inertia frame** at construction.  Consequently ``vertices``, ``centroid == 0``,
``inertia_diag`` (principal moments) and ``principal_axes`` (columns map the
body frame to the construction frame) are all exact.

All quantities are SI (m, kg, s).  ``ConvexPolyhedron`` is dimensionless
(unit density); multiply by the ice density to get mass properties.
"""
from __future__ import annotations

import itertools
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import ConvexHull

__all__ = [
    "ConvexPolyhedron",
    "halfspace_vertices",
    "voronoi_cell_polyhedron",
    "voronoi_tessellate",
    "extrude_polygon",
    "regular_polygon",
    "polygon_area",
    "disc_equivalent_polygon",
    "prism_from_polygon",
    "equivalent_disc_prism",
    "body_frames",
]

_TINY = 1.0e-12


# ---------------------------------------------------------------------------
# half-space -> vertices
# ---------------------------------------------------------------------------
def halfspace_vertices(normals: np.ndarray, offsets: np.ndarray,
                       tol: float = 1.0e-9) -> np.ndarray:
    """Vertices of the convex polytope ``{x : n_k . x <= d_k}``.

    Exact for the low plane counts of a floe: every vertex of a polytope in
    ``R^3`` is the intersection of three of its facet planes, so intersecting
    all plane triples and keeping the feasible points enumerates the vertex set
    without any tolerance-dependent iteration.  Degenerate (near parallel)
    triples are dropped by the determinant filter, duplicate points are merged
    at ``1e-9 * scale``.
    """
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    offsets = np.asarray(offsets, dtype=np.float64).reshape(-1)
    k = normals.shape[0]
    if k < 4:
        raise ValueError("a bounded 3-D polytope needs at least 4 half-spaces")
    scale = float(np.max(np.abs(offsets))) + 1.0
    tri = np.array(list(itertools.combinations(range(k), 3)), dtype=int)
    a = normals[tri]
    det = np.linalg.det(a)
    good = np.abs(det) > 1.0e-10
    if not np.any(good):
        return np.zeros((0, 3))
    a = a[good]
    b = offsets[tri[good]]
    pts = np.linalg.solve(a, b[..., None])[..., 0]
    feasible = np.all(pts @ normals.T <= offsets[None, :] + tol * scale, axis=1)
    pts = pts[feasible]
    if pts.shape[0] == 0:
        return pts
    quant = max(tol * scale * 10.0, _TINY)
    keys = np.round(pts / quant).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(first)]


def _unique_rows(values: np.ndarray, quant: float = 1.0e-9) -> np.ndarray:
    """Deduplicate rows after quantisation (deterministic order preserved)."""
    if values.shape[0] == 0:
        return values
    keys = np.round(values / quant).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return values[np.sort(first)]


# ---------------------------------------------------------------------------
# mass properties from a vertex cloud (divergence theorem)
# ---------------------------------------------------------------------------
def _mass_properties(vertices: np.ndarray):
    """Exact volume, centroid, inertia tensor and area of a convex hull.

    A tetrahedral fan from an interior point ``g0`` gives
    ``V = sum v_t``, ``centroid = sum v_t g_t / V`` and, with
    ``u_i = v_i^(t) - g_t``,

    .. math::
        \\int_{T}(r-g_T)(r-g_T)^T \\, dV = \\frac{V_T}{20}\\sum_{i=1}^{4} u_i u_i^T

    (barycentric integration of :math:`\\lambda_i\\lambda_j` on a simplex), so
    the inertia tensor is exact, not a sphere/box surrogate.
    """
    hull = ConvexHull(vertices)
    simplices = hull.simplices
    g0 = vertices[simplices].mean(axis=(0, 1))
    volume = 0.0
    first = np.zeros(3)
    second = np.zeros((3, 3))
    area = 0.0
    for (i, j, k) in simplices:
        a, b, c = vertices[i], vertices[j], vertices[k]
        vol = float(np.dot(a - g0, np.cross(b - g0, c - g0))) / 6.0
        if vol <= 0.0:
            # orientation flip: hull faces are outward, keep |vol| safety net
            vol = abs(vol)
        gt = 0.25 * (g0 + a + b + c)
        volume += vol
        first += vol * gt
        u = np.stack([g0 - gt, a - gt, b - gt, c - gt])
        second += (vol / 20.0) * (u.T @ u) + vol * np.outer(gt, gt)
        area += 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
    centroid = first / volume
    # second moment about the centroid, then the inertia tensor
    second -= volume * np.outer(centroid, centroid)
    inertia = np.trace(second) * np.eye(3) - second
    return volume, centroid, inertia, area, simplices


def _face_normals(vertices: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Outward unit normals of the hull, deduplicated (planar facets merged)."""
    a = vertices[simplices[:, 0]]
    b = vertices[simplices[:, 1]]
    c = vertices[simplices[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > _TINY
    normals = normals[keep] / lengths[keep, None]
    # canonical orientation so that the outward direction wins deduplication
    centre = 0.5 * (a[keep] + b[keep] + c[keep])
    flip = np.einsum("ij,ij->i", normals, centre) < 0.0
    normals[flip] *= -1.0
    return _unique_rows(normals, 1.0e-6)


def _unique_edges(simplices: np.ndarray, n_vertices: int) -> np.ndarray:
    """Unique undirected vertex-index pairs of the hull surface."""
    seg = np.vstack([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]])
    seg = np.sort(seg, axis=1)
    return _unique_rows(seg.astype(np.float64), 1.0e-9).astype(int) if seg.size else \
        np.zeros((0, 2), dtype=int)


# ---------------------------------------------------------------------------
# the body
# ---------------------------------------------------------------------------
class ConvexPolyhedron:
    """A convex polyhedron in its principal centroidal frame (unit density).

    Attributes
    ----------
    vertices : (v, 3) float64
        Body-frame vertices; the centroid is at the origin.
    faces : (t, 3) int
        Triangular facets of the hull (outward oriented).
    normals : (f, 3) float64
        Deduplicated outward facet normals.
    edges : (e, 2) int
        Unique edge vertex-index pairs.
    volume, area : float
        Volume and surface area (unit density -> these are also mass and area).
    inertia_diag : (3,) float64
        Principal moments of inertia.
    principal_axes : (3, 3) float64
        Columns map body coordinates to the coordinates the planes were given in.
    """

    __slots__ = ("vertices", "faces", "normals", "edges", "volume", "area",
                 "inertia_diag", "principal_axes", "bounding_radius",
                 "sphericity", "planes_n", "planes_d", "n_vertices", "n_faces",
                 "centroid_in_planes", "extrusion_axis")

    def __init__(self, normals: np.ndarray, offsets: np.ndarray, *,
                 recentre: bool = True, principal: bool = True):
        normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        offsets = np.asarray(offsets, dtype=np.float64).reshape(-1)
        length = np.linalg.norm(normals, axis=1)
        if np.any(length < _TINY):
            raise ValueError("degenerate half-space normal")
        normals = normals / length[:, None]
        offsets = offsets / length

        verts = halfspace_vertices(normals, offsets)
        if verts.shape[0] < 4:
            raise ValueError("half-spaces do not bound a 3-D body")
        volume, centroid, inertia, area, simplices = _mass_properties(verts)

        if principal:
            moments, axes = np.linalg.eigh(inertia)
            rot = axes
        else:
            moments = np.diag(inertia).copy()
            rot = np.eye(3)
        if recentre:
            verts = (verts - centroid) @ rot
        else:
            verts = verts @ rot

        faces = simplices
        normals_b = _face_normals(verts, faces)
        edges = _unique_edges(faces, verts.shape[0])

        self.vertices = verts
        self.faces = faces
        self.normals = normals_b
        self.edges = edges
        self.volume = float(volume)
        self.area = float(area)
        self.inertia_diag = np.asarray(moments, dtype=np.float64)
        self.principal_axes = rot
        self.bounding_radius = float(np.max(np.linalg.norm(verts, axis=1)))
        self.sphericity = float(math.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / area)
        self.planes_n = normals
        self.planes_d = offsets
        self.n_vertices = int(verts.shape[0])
        self.n_faces = int(normals_b.shape[0])
        #: extrusion direction for plate-like bodies (``None`` for general polyhedra)
        self.extrusion_axis = None
        #: centroid expressed in the frame in which ``planes_n/planes_d`` are given
        self.centroid_in_planes = np.asarray(centroid, dtype=np.float64)

    # -- derived quantities -------------------------------------------------
    @property
    def mass_properties(self) -> dict:
        return {
            "volume": self.volume,
            "area": self.area,
            "inertia_diag": self.inertia_diag.tolist(),
            "sphericity": self.sphericity,
            "n_vertices": self.n_vertices,
            "n_faces": self.n_faces,
            "bounding_radius": self.bounding_radius,
        }

    def clipped(self, normal: Sequence[float], offset: float) -> "ConvexPolyhedron":
        """Intersect with one more half-space (in this body's own frame)."""
        normal = np.asarray(normal, dtype=np.float64).reshape(1, 3)
        offset = np.asarray([offset], dtype=np.float64)
        return ConvexPolyhedron(np.vstack([self.planes_n, normal]),
                                np.concatenate([self.planes_d, offset]))

    def scaled(self, factor: float) -> "ConvexPolyhedron":
        return ConvexPolyhedron(self.planes_n, self.planes_d * float(factor))

    def construction_vertices(self) -> np.ndarray:
        """Vertices in the frame in which the defining planes were given.

        The body itself is stored recentred and rotated into the principal frame,
        so ``vertices`` alone is not comparable with ``planes_n/planes_d``.  This
        method undoes both transforms, which is what geometric assertions about
        half-space sets need.
        """
        return self.vertices @ self.principal_axes.T + self.centroid_in_planes

    def aabb(self) -> tuple:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (f"ConvexPolyhedron(V={self.volume:.6g}, A={self.area:.6g}, "
                f"nv={self.n_vertices}, nf={self.n_faces})")


# ---------------------------------------------------------------------------
# generative geometries
# ---------------------------------------------------------------------------
def voronoi_cell_polyhedron(site: Sequence[float], others: np.ndarray,
                            box_half: float, n_neighbours: int = 24,
                            max_refine: int = 6, tol: float = 1.0e-10
                            ) -> ConvexPolyhedron:
    """Voronoi cell of ``site`` clipped to the box ``[-h, h]^3``, exactly.

    All half-spaces are written in the frame centred on ``site`` (which is the
    frame :class:`ConvexPolyhedron` expects); the centroid in the original
    frame is ``centroid_in_planes + site``.  The bisector plane between ``site``
    and a neighbour ``q`` contains the points equidistant from both, so a point
    ``y = x - site`` satisfies ``n . y <= |site - q| / 2`` with
    ``n = (q - site)/|q - site|`` (a *positive* normal towards ``q``; the
    opposite sign would keep the far half-space and inflate the cell).

    Starting from the ``n_neighbours`` nearest sites, the cell is refined until
    no omitted site can cut it (feasibility test on the current vertices), so
    the result is the exact Voronoi cell rather than a nearest-neighbour
    approximation.  Exactness is what the volume-closure test checks:
    ``sum(V_i) == (2h)^3`` to machine precision.
    """
    site = np.asarray(site, dtype=np.float64)
    others = np.asarray(others, dtype=np.float64)
    delta = others - site
    dist = np.linalg.norm(delta, axis=1)
    order = np.argsort(dist)
    order = order[dist[order] > _TINY][:n_neighbours]

    h = float(box_half)
    normals = [np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]),
               np.array([0, 1.0, 0]), np.array([0, -1.0, 0]),
               np.array([0, 0, 1.0]), np.array([0, 0, -1.0])]
    offsets = [h - site[0], h + site[0], h - site[1], h + site[1], h - site[2], h + site[2]]
    used = set(int(i) for i in order)

    def _bisector(index: int):
        d = others[index] - site
        length = float(np.linalg.norm(d))
        return d / length, 0.5 * length

    for j in order:
        n, d = _bisector(int(j))
        normals.append(n)
        offsets.append(d)

    normals = np.asarray(normals, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    vertices = halfspace_vertices(normals, offsets)
    scale = 2.0 * h
    for _ in range(max_refine):
        if vertices.shape[0] < 4:
            break
        # a vertex y violates the bisector of q when 2 y.(q - s) > |q - s|^2
        proj = vertices @ delta.T                       # (n_vertices, n_sites)
        growth = (dist * dist)[None, :]
        violated = np.any(2.0 * proj > growth + tol * scale * scale, axis=0)
        violated[[int(i) for i in used]] = False
        violated[dist <= _TINY] = False
        if not np.any(violated):
            break
        extra = np.flatnonzero(violated)
        # add the most violated first (ray-cast order), keep the count modest
        extra = extra[np.argsort(-dist[extra])][:8]
        for j in extra:
            n, d = _bisector(int(j))
            normals = np.vstack([normals, n])
            offsets = np.concatenate([offsets, [d]])
            used.add(int(j))
        vertices = halfspace_vertices(normals, offsets)
    return ConvexPolyhedron(normals, offsets)


def voronoi_tessellate(sites: np.ndarray, box_half: float, n_neighbours: int = 24):
    """Voronoi tessellation of a box as a list of :class:`ConvexPolyhedron`."""
    sites = np.asarray(sites, dtype=np.float64)
    cells = []
    for i, site in enumerate(sites):
        others = np.delete(sites, i, axis=0)
        cells.append(voronoi_cell_polyhedron(site, others, box_half, n_neighbours))
    return cells


def polygon_area(polygon: np.ndarray) -> float:
    p = np.asarray(polygon, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _ccw_ordered(polygon: np.ndarray) -> np.ndarray:
    p = np.asarray(polygon, dtype=np.float64)
    if p.shape[0] == 1:
        raise ValueError("polygon needs at least 2 distinct points")
    hull = ConvexHull(p)
    ordered = p[hull.vertices]
    # ConvexHull returns counter-clockwise ordering in 2-D
    return ordered


def extrude_polygon(polygon_xy: np.ndarray, thickness: float, *, z0: float = 0.0,
                    principal: bool = True) -> ConvexPolyhedron:
    """Extrude a convex 2-D polygon into a plate of thickness ``thickness``.

    This is the exact lift of the 2-D floe field: the footprint (and hence the
    in-plane contact geometry) is inherited unchanged, the third dimension only
    adds thickness.  It is therefore the geometry used for the :math:`t \\to 0`
    degeneracy check.
    """
    thickness = float(thickness)
    if thickness <= 0.0:
        raise ValueError("thickness must be positive")
    ordered = _ccw_ordered(polygon_xy)
    m = ordered.shape[0]
    edge = np.roll(ordered, -1, axis=0) - ordered
    outward = np.stack([edge[:, 1], -edge[:, 0]], axis=1)
    lengths = np.linalg.norm(outward, axis=1)
    if np.any(lengths < _TINY):
        raise ValueError("degenerate polygon edge")
    outward = outward / lengths[:, None]
    offsets_xy = np.einsum("ij,ij->i", outward, ordered)
    normals = np.zeros((m + 2, 3))
    normals[:m, :2] = outward
    normals[m, 2] = -1.0
    normals[m + 1, 2] = 1.0
    offsets = np.zeros(m + 2)
    offsets[:m] = offsets_xy
    offsets[m] = -z0
    offsets[m + 1] = z0 + thickness
    body = ConvexPolyhedron(normals, offsets, principal=principal)
    # the extrusion axis in the body frame (principal rotation applied)
    body.extrusion_axis = body.principal_axes.T @ np.array([0.0, 0.0, 1.0])
    body.extrusion_axis /= np.linalg.norm(body.extrusion_axis)
    return body


def regular_polygon(area: float, n_sides: int, *, phase: float = 0.0,
                    centre: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    """Regular ``n_sides``-gon of prescribed area (the 'disc' reference shape)."""
    if n_sides < 3:
        raise ValueError("n_sides must be >= 3")
    r = math.sqrt(2.0 * float(area) / (n_sides * math.sin(2.0 * math.pi / n_sides)))
    a = np.linspace(0.0, 2.0 * math.pi, n_sides, endpoint=False) + phase
    return np.stack([centre[0] + r * np.cos(a), centre[1] + r * np.sin(a)], axis=1)


def disc_equivalent_polygon(area: float, n_sides: int = 24, **kwargs) -> np.ndarray:
    """Polygonal approximation of a circular disc of the same area."""
    return regular_polygon(area, n_sides, **kwargs)


def prism_from_polygon(polygon_xy: np.ndarray, thickness: float, *,
                       footprint_area: float | None = None) -> ConvexPolyhedron:
    p = _ccw_ordered(polygon_xy)
    if footprint_area is not None and abs(polygon_area(p) - footprint_area) > 1e-9 * footprint_area:
        raise ValueError("polygon area does not match the requested footprint")
    return extrude_polygon(p, thickness)


def equivalent_disc_prism(volume: float, thickness: float, n_sides: int = 24, *,
                          phase: float = 0.0) -> ConvexPolyhedron:
    """Disc-like plate with the *same volume and thickness* as a polyhedral floe.

    Used for the shape-effect comparison: only the plan-form (shape) changes,
    every mass property except the inertia stays comparable.
    """
    area = float(volume) / float(thickness)
    return extrude_polygon(regular_polygon(area, n_sides, phase=phase), thickness)


# ---------------------------------------------------------------------------
# world frames
# ---------------------------------------------------------------------------
def body_frames(shape: ConvexPolyhedron, rotation: np.ndarray, position: np.ndarray):
    """World-frame vertices, facet normals and edge vectors of one body."""
    rotation = np.asarray(rotation, dtype=np.float64)
    vertices = shape.vertices @ rotation.T + position
    normals = shape.normals @ rotation.T
    edges = shape.vertices[shape.edges[:, 1]] - shape.vertices[shape.edges[:, 0]]
    edges = edges @ rotation.T
    return vertices, normals, edges