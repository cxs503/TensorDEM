"""Triangle-to-triangle interface clipping in 3D, including *non-coplanar* facets.

Motivation (campaign node IN52-N01)
-----------------------------------
The 1D interface contract of TensorPhysics m5207 and its coplanar 2D/3D
generalisation (TensorDEM m5436, ``interface_operators.mortar``) assume that both
discretisations of the interface lie in **one** plane.  A curved interface
(sphere/hull, cylinder/silo wall, ice-floe surface) is represented by two
*different* polyhedral approximations, one per subdomain, and their facets are
**not** coplanar.  The intersection of two non-coplanar triangles is a segment,
i.e. a measure-zero object, so an area-based mortar cannot use the intersection
directly: the standard device is the projection (mortar-segment) approach, used
here as

    m_ij   = | S_i  ∩  P_{S_i}( T_j ) |              (donor-plane measure)
    m_ij^T = m_ij / | n_i . n_j |                    (area transported to T_j)

with ``P_{S_i}`` the orthogonal projection onto the plane of the donor facet.
The projection defect is ``O(theta^2)`` in the dihedral angle ``theta`` between
the two approximations, and ``theta = O(h)`` for two inscribed polyhedra of the
same curved patch, so the geometric bias is ``O(h^2)``.  That is *measured* here
(gate D4), not assumed.

Measure-normalisation trap of m5207 section 5.2, on the critical path
---------------------------------------------------------------------
For a curved interface the donor-plane overlaps do **not** partition the
acceptor facet (``sum_i m_ij != |T_j|``), so the raw transfer is not conservative
and the deficit has to be backfilled.  The correct (measure-proportional) form is

    dv_j = (sent - recv) * |T_j| / sum_k |T_k|^2

so that ``sum_j dv_j |T_j| = sent - recv`` exactly.  Normalising with
``sum_k |T_k|`` (the trap) leaves a percentage-level arithmetic residual that
reads like a physics error; both branches are executable here, so the trap is a
regression test rather than a comment.

Everything below works on *lists of planar triangles*, so a flat triangulation
(two cubes sharing a face, the two facet families rotated 17 deg against each
other) and a curved triangulation (two polyhedral sphere approximations) share
one code path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import cell_overlap

__all__ = [
    "TriSide",
    "SurfaceMortarOperator",
    "clip_homogeneous",
    "facet_gradients",
    "ico_sphere_tris",
    "measure_backfill",
    "mirrored_tris",
    "surface_cell_averages",
    "surface_coverage",
    "surface_integral",
    "surface_l2_error",
    "surface_overlap_entries",
    "tri_area",
    "tri_centroid",
    "tri_frame",
    "tri_normal",
    "tri_patch_side",
    "tri_pair_overlap",
    "triangulate_polygon",
    "uv_sphere_tris",
    "clip_tris_by_box",
]

# barycentric Gauss points, exact for polynomials of degree 2 on a triangle
_GAUSS_BARY = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
_GAUSS_W = np.full(3, 1.0 / 3.0)


# ---------------------------------------------------------------------------
# triangle primitives
# ---------------------------------------------------------------------------
def tri_area(tri: np.ndarray) -> float:
    """Area of a planar triangle given its three vertices."""
    v = np.asarray(tri, dtype=float)
    return 0.5 * float(np.linalg.norm(np.cross(v[1] - v[0], v[2] - v[0])))


def tri_centroid(tri: np.ndarray) -> np.ndarray:
    return np.asarray(tri, dtype=float).mean(axis=0)


def tri_normal(tri: np.ndarray) -> np.ndarray:
    """Unit normal of a triangle (right-handed with respect to vertex order)."""
    v = np.asarray(tri, dtype=float)
    n = np.cross(v[1] - v[0], v[2] - v[0])
    nrm = float(np.linalg.norm(n))
    if nrm <= 0.0:
        raise ValueError("degenerate (zero-area) triangle")
    return n / nrm


def tri_frame(tri: np.ndarray):
    """Local frame of a triangle: ``(centroid, basis (2,3), unit normal)``."""
    v = np.asarray(tri, dtype=float)
    c = v.mean(axis=0)
    _, _, vt = np.linalg.svd(v - c, full_matrices=False)
    n = np.cross(vt[0], vt[1])
    n = n / float(np.linalg.norm(n))
    return c, vt[:2], n


def triangulate_polygon(poly: np.ndarray, tol: float = 1e-17) -> list:
    """Fan-triangulate a convex planar polygon (keeps the region exactly)."""
    p = np.asarray(poly, dtype=float)
    out = []
    for k in range(1, len(p) - 1):
        t = np.array([p[0], p[k], p[k + 1]])
        if tri_area(t) > tol:
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# homogeneous Sutherland-Hodgman clipping
# ---------------------------------------------------------------------------
def clip_homogeneous(subject: np.ndarray, planes: np.ndarray,
                     tol: float = 1e-14) -> np.ndarray:
    """Sutherland-Hodgman clipping of a homogeneous polygon by half-planes.

    Parameters
    ----------
    subject : (nv, 3) array
        Homogeneous coordinates ``(x, y, w)``.  Affine geometry has ``w = 1``; the
        ``w`` column is what makes this the *homogeneous* (rather than purely
        planar) formulation: the signed distance of a vertex to a clip plane is
        ``(a x + b y + c w) / w``, so a vertex at infinity (``w = 0``) has no
        finite clip parameter.  The classical pitfalls are (i) the interpolation
        ``t = d_a / (d_a - d_b)`` when the edge is parallel to the clip plane and
        (ii) a non-finite homogeneous coordinate; both guards are explicit, and a
        degenerate edge is *dropped* (measure zero) instead of producing a NaN
        that would contaminate the overlap table.
    planes : (m, 3) array
        Half-planes ``(a, b, c)`` whose inside is ``a x + b y + c w >= 0``.
    """
    sub = np.asarray(subject, dtype=float)
    pl = np.asarray(planes, dtype=float)
    if len(sub) < 3:
        return np.zeros((0, 3))
    out = sub
    for (a, b, c) in pl:
        if len(out) < 3:
            return np.zeros((0, 3))
        w = out[:, 2]
        finite = np.abs(w) > tol
        if not np.all(finite):
            out = out[finite]
            if len(out) < 3:
                return np.zeros((0, 3))
            w = out[:, 2]
        d = (a * out[:, 0] + b * out[:, 1] + c * out[:, 2]) / w
        nxt = np.roll(out, -1, axis=0)
        dn = np.roll(d, -1)
        keep_p, keep_q = d >= 0.0, dn >= 0.0
        new = []
        for i in range(len(out)):
            if keep_p[i]:
                new.append(out[i])
            if keep_p[i] != keep_q[i]:
                denom = d[i] - dn[i]
                if abs(denom) <= tol * max(1.0, abs(float(d[i]))):
                    continue          # crossing at infinity -> no finite vertex
                t = d[i] / denom
                new.append(out[i] + t * (nxt[i] - out[i]))
        out = np.asarray(new, dtype=float) if len(new) else np.zeros((0, 3))
    return out


def _clipper_planes_local(clip_local: np.ndarray) -> np.ndarray:
    """Half-plane coefficients ``(a, b, c)`` of a convex counter-clockwise polygon."""
    p = np.asarray(clip_local, dtype=float)
    area2 = float(np.dot(p[:, 0], np.roll(p[:, 1], -1))
                  - np.dot(np.roll(p[:, 0], -1), p[:, 1]))
    if area2 < 0.0:
        p = p[::-1]
    m = len(p)
    planes = np.empty((m, 3))
    for k in range(m):
        a, b = p[k], p[(k + 1) % m]
        e = b - a
        # inside half-plane: cross(e, x - a) >= 0
        planes[k] = (-e[1], e[0], e[1] * a[0] - e[0] * a[1])
    return planes


# ---------------------------------------------------------------------------
# one triangle pair
# ---------------------------------------------------------------------------
def tri_pair_overlap(sa: np.ndarray, sb: np.ndarray, *, cos_coplanar: float = 1.0 - 1e-12,
                     cos_min: float = 0.15, tol: float = 1e-14) -> dict:
    """Overlap of two planar triangles, coplanar *or* not.

    Returns a dict with

    ``measure_src``   area of ``S_a ∩ P(S_b)`` in the plane of ``S_a``
    ``measure_tgt``   the same region transported to ``S_b`` (``/|cos|``)
    ``moment``        ``∫ (x - c_a) dx`` over that region, in ``S_a``'s frame
    ``cos_dihedral``  ``|n_a . n_b|``
    ``path``          ``"coplanar"`` (exact intersection), ``"projected"``
                      (non-coplanar, projection mortar) or ``"rejected"``
                      (dihedral too large: the two facets are not the same
                      interface patch; reported, never silently mixed)

    The clip runs in ``S_a``'s local frame on the projected vertices of ``S_b``
    through ``geometry.cell_overlap``, i.e. the coplanar branch is *identical* to
    the planar operator of m5436 and the non-coplanar branch differs only by the
    projection and the ``1/|cos|`` transport.
    """
    a = np.asarray(sa, dtype=float)
    b = np.asarray(sb, dtype=float)
    na, nb = tri_normal(a), tri_normal(b)
    cos = float(abs(np.dot(na, nb)))
    path = "coplanar" if cos >= cos_coplanar else "projected"
    if cos < cos_min:
        return {"measure_src": 0.0, "measure_tgt": 0.0, "moment": np.zeros(2),
                "cos_dihedral": cos, "path": "rejected"}
    c, basis, _ = tri_frame(a)
    b_proj = b - np.outer((b - c) @ na, na)      # orthogonal projection
    measure, moment = cell_overlap(a, b_proj, basis, c, tol)
    if measure <= 0.0:
        return {"measure_src": 0.0, "measure_tgt": 0.0, "moment": np.zeros(2),
                "cos_dihedral": cos, "path": path}
    return {"measure_src": measure, "measure_tgt": measure / cos, "moment": moment,
            "cos_dihedral": cos, "path": path}


# ---------------------------------------------------------------------------
# a whole triangulated side
# ---------------------------------------------------------------------------
@dataclass
class TriSide:
    """One triangulated discretisation of an interface surface.

    Unlike ``mortar.InterfaceSide`` the cells need **not** be coplanar; that is
    what allows two different polyhedral approximations of a curved interface to
    be coupled.  Every cell is a planar triangle carrying its own frame.
    """

    tris: list
    label: str = ""
    measures: np.ndarray = field(default_factory=lambda: np.zeros(0))
    centroids: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    lo: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    hi: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    def __post_init__(self) -> None:
        self.tris = [np.asarray(t, dtype=float) for t in self.tris]
        if not self.tris:
            raise ValueError("TriSide needs at least one triangle")
        areas, cents, nrms, lo, hi = [], [], [], [], []
        for t in self.tris:
            if t.shape != (3, 3):
                raise ValueError("every cell must be a (3, 3) triangle")
            a = tri_area(t)
            if a <= 0.0:
                raise ValueError(f"degenerate triangle in side {self.label!r}")
            areas.append(a)
            cents.append(tri_centroid(t))
            nrms.append(tri_normal(t))
            lo.append(t.min(axis=0))
            hi.append(t.max(axis=0))
        self.measures = np.asarray(areas)
        self.centroids = np.asarray(cents)
        self.normals = np.asarray(nrms)
        self.lo = np.asarray(lo)
        self.hi = np.asarray(hi)

    @property
    def n(self) -> int:
        return len(self.tris)

    @property
    def area(self) -> float:
        return float(self.measures.sum())

    def characteristic_length(self) -> float:
        return float(math.sqrt(self.area / self.n))

    def mirrored(self, point, normal, label=None) -> "TriSide":
        """Reflect the side about a plane (the *same* plane for both sides)."""
        p = np.asarray(point, dtype=float)
        nrm = np.asarray(normal, dtype=float)
        nrm = nrm / np.linalg.norm(nrm)
        tris = []
        for t in self.tris:
            refl = t - np.outer(2.0 * ((t - p) @ nrm), nrm)
            tris.append(refl[::-1])             # reverse vertex order, keep orientation
        return TriSide(tris, label=self.label if label is None else label)


def mirrored_tris(tris, point, normal) -> list:
    """Reflect a raw triangle list about a plane."""
    p = np.asarray(point, dtype=float)
    nrm = np.asarray(normal, dtype=float)
    nrm = nrm / np.linalg.norm(nrm)
    out = []
    for t in tris:
        t = np.asarray(t, dtype=float)
        out.append((t - np.outer(2.0 * ((t - p) @ nrm), nrm))[::-1])
    return out


# ---------------------------------------------------------------------------
# mesh generators
# ---------------------------------------------------------------------------
def _is_convex_2d(quad: np.ndarray, tol: float = 1e-14) -> bool:
    """True if a 2D quad (given in order) is strictly convex."""
    q = np.asarray(quad, dtype=float)
    signs = []
    for k in range(4):
        a, b, c = q[k], q[(k + 1) % 4], q[(k + 2) % 4]
        e1, e2 = b - a, c - b
        cr = e1[0] * e2[1] - e1[1] * e2[0]
        if abs(cr) <= tol:
            return False
        signs.append(np.sign(cr))
    return len(set(signs)) == 1


def tri_patch_side(origin, e1, e2, n1: int, n2: int, *, rotate_deg: float = 0.0,
                   jitter: float = 0.0, seed: int = 0, label: str = "") -> TriSide:
    """Triangulated planar patch covering exactly the unit square ``[0,1]^2``.

    The square is parametrised by ``origin + s*e1 + t*e2`` for ``(s, t)`` in the
    unit square.  ``rotate_deg`` rotates the **facet family** with respect to the
    square by that angle (the rotated lattice is clipped back to the square), so
    two sides with different ``rotate_deg`` give generic, vertex-misaligned
    overlaps: no nesting, no shared nodes, and the two partitions cover the same
    region exactly.  ``jitter`` perturbs interior nodes by ``O(jitter*h)`` from a
    seeded distribution.

    A lattice plus ``O(h)`` jitter is deliberate: a random node cloud does not
    refine and reports a fake order plateau (m5207 trap list, re-measured in
    m5436 section 5.2).
    """
    o = np.asarray(origin, dtype=float)
    E1 = np.asarray(e1, dtype=float)
    E2 = np.asarray(e2, dtype=float)
    E1 = E1 / np.linalg.norm(E1)
    E2 = E2 / np.linalg.norm(E2)
    length1 = float(np.linalg.norm(np.asarray(e1, dtype=float)))
    length2 = float(np.linalg.norm(np.asarray(e2, dtype=float)))
    th = math.radians(rotate_deg)
    f1 = math.cos(th) * E1 + math.sin(th) * E2
    f2 = -math.sin(th) * E1 + math.cos(th) * E2
    # coordinates of the square corners in the (f1, f2) frame
    corners = [s * length1 * E1 + t * length2 * E2 for s in (0.0, 1.0) for t in (0.0, 1.0)]
    cc = np.array([[np.dot(c, f1), np.dot(c, f2)] for c in corners])
    lo, hi = cc.min(axis=0), cc.max(axis=0)
    h1, h2 = (hi[0] - lo[0]) / n1, (hi[1] - lo[1]) / n2
    c1 = lo[0] + np.arange(n1 + 1) * h1
    c2 = lo[1] + np.arange(n2 + 1) * h2
    rng = np.random.default_rng(seed)
    h = min(h1, h2)
    # lattice + O(h) jitter, with the amplitude reduced until every quad stays
    # convex: a non-convex quad split along a diagonal would double count or lose
    # area, which would destroy the coverage gate for a purely geometric reason.
    base = np.zeros((n1 + 1, n2 + 1, 2))
    for a in range(n1 + 1):
        for b in range(n2 + 1):
            base[a, b] = (c1[a], c2[b])
    delta = {(a, b): rng.uniform(-1.0, 1.0, size=2)
             for a in range(1, n1) for b in range(1, n2)}
    amp = jitter * h
    for _ in range(60):
        nodes2 = base.copy()
        for (a, b), d in delta.items():
            nodes2[a, b] = base[a, b] + amp * d
        ok = True
        for a in range(n1):
            for b in range(n2):
                quad = np.array([nodes2[a, b], nodes2[a + 1, b],
                                 nodes2[a + 1, b + 1], nodes2[a, b + 1]])
                if not _is_convex_2d(quad):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            break
        amp *= 0.5
    else:                                              # pragma: no cover
        raise RuntimeError("could not build a convex jittered triangulation")
    nodes = o + nodes2[..., 0:1] * f1 + nodes2[..., 1:2] * f2
    tris = []
    for a in range(n1):
        for b in range(n2):
            q = [nodes[a, b], nodes[a + 1, b], nodes[a + 1, b + 1], nodes[a, b + 1]]
            if (a + b) % 2 == 0:
                tris.extend([np.array([q[0], q[1], q[2]]),
                             np.array([q[0], q[2], q[3]])])
            else:
                tris.extend([np.array([q[0], q[1], q[3]]),
                             np.array([q[1], q[2], q[3]])])
    # clip the (possibly rotated) lattice back to the square: both sides of a
    # pair then partition the *same* region, so a coverage defect is a property
    # of the operator and not of a mismatched support.
    twin = np.cross(f1, f2)
    twin = twin / np.linalg.norm(twin)
    planes = [(o, E1), (o + length1 * E1, -E1), (o, E2), (o + length2 * E2, -E2)]
    # clipped-away slivers are dropped below 1e-14 of the mean unclipped cell
    # area: such facets have a garbage normal (their two edges are numerically
    # parallel) and, being divided by their own measure, they would poison the
    # cell averages; the area they remove is < 1e-14 of the patch, i.e. far below
    # every gate below.
    a_ref = float(np.mean([tri_area(t) for t in tris])) if tris else 1.0
    clipped = []
    for t in tris:
        poly = _clip_polygon_by_planes(t, planes)
        if len(poly) >= 3:
            clipped.extend(triangulate_polygon(poly, tol=1e-14 * a_ref))
    return TriSide(clipped, label=label or f"tri{n1}x{n2}@{rotate_deg:g}deg")


def uv_sphere_tris(radius: float = 1.0, n_theta: int = 8, n_phi: int = 16,
                   center=(0.0, 0.0, 0.0)) -> list:
    """Latitude/longitude triangulation of a sphere (pole-degenerate cells dropped)."""
    c = np.asarray(center, dtype=float)
    thetas = np.linspace(0.0, math.pi, n_theta + 1)
    phis = np.linspace(0.0, 2.0 * math.pi, n_phi + 1)

    def vert(i, j):
        th, ph = thetas[i], phis[j]
        return c + radius * np.array([math.sin(th) * math.cos(ph),
                                      math.sin(th) * math.sin(ph),
                                      math.cos(th)])

    tris = []
    for i in range(n_theta):
        for j in range(n_phi):
            v00, v10 = vert(i, j), vert(i + 1, j)
            v11, v01 = vert(i + 1, j + 1), vert(i, j + 1)
            tris.append(np.array([v00, v10, v11]))
            tris.append(np.array([v00, v11, v01]))
    return [t for t in tris if tri_area(t) > 1e-15 * radius * radius]


def ico_sphere_tris(radius: float = 1.0, level: int = 2, center=(0.0, 0.0, 0.0)) -> list:
    """Subdivided octahedron projected onto the sphere.

    Deliberately a *different topology* from :func:`uv_sphere_tris` (no shared
    vertices in general), so the pair (uv, icosphere) is a genuinely non-matching
    discretisation of the same curved patch.
    """
    c = np.asarray(center, dtype=float)
    v = [np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
         np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]),
         np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
             (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    for _ in range(level):
        cache = {}
        new_faces = []

        def mid(i, j):
            key = (min(i, j), max(i, j))
            if key not in cache:
                m = v[i] + v[j]
                cache[key] = len(v)
                v.append(m / np.linalg.norm(m))
            return cache[key]

        for (a, b, d) in faces:
            ab, bd, da = mid(a, b), mid(b, d), mid(d, a)
            new_faces += [(a, ab, da), (ab, b, bd), (da, bd, d), (ab, bd, da)]
        faces = new_faces
    tris = [radius * np.array([v[a], v[b], v[d]]) + c for (a, b, d) in faces]
    return [t for t in tris if tri_area(t) > 1e-15 * radius * radius]


def _clip_polygon_by_planes(poly: np.ndarray, planes) -> np.ndarray:
    """Clip a 3D convex polygon by half-spaces ``(point, inward normal)``."""
    out = np.asarray(poly, dtype=float)
    for (q, nrm) in planes:
        if len(out) < 3:
            return np.zeros((0, 3))
        d = (out - q) @ nrm                     # >= 0 inside
        nxt = np.roll(out, -1, axis=0)
        dn = np.roll(d, -1)
        keep_p, keep_q = d >= 0.0, dn >= 0.0
        new = []
        for i in range(len(out)):
            if keep_p[i]:
                new.append(out[i])
            if keep_p[i] != keep_q[i]:
                denom = d[i] - dn[i]
                if abs(denom) <= 1e-300:
                    continue
                t = d[i] / denom
                new.append(out[i] + t * (nxt[i] - out[i]))
        out = np.asarray(new, dtype=float) if len(new) else np.zeros((0, 3))
    return out


def clip_tris_by_box(tris, box, drop_rel: float = 1e-14) -> list:
    """Clip a triangle list by an axis-aligned box ``((x0,x1),(y0,y1),(z0,z1))``.

    ``+-inf`` bounds are allowed.  The clipped convex polygon is fan-triangulated,
    so the clipped surface covers the box region **exactly** -- the coverage of
    each triangulation is therefore limited only by the polyhedral approximation
    of the sphere, which is what gate D4 measures.
    """
    planes = []
    for axis, (lo, hi) in enumerate(box):
        e = np.zeros(3)
        e[axis] = 1.0
        if np.isfinite(lo):
            planes.append((-lo * e, e))
        if np.isfinite(hi):
            planes.append((hi * e, -e))
    tlist = [np.asarray(t, dtype=float) for t in tris]
    a_ref = float(np.mean([tri_area(t) for t in tlist])) if tlist else 1.0
    out = []
    for t in tlist:
        poly = _clip_polygon_by_planes(t, planes)
        if len(poly) >= 3:
            out.extend(triangulate_polygon(poly, tol=drop_rel * a_ref))
    return out


# ---------------------------------------------------------------------------
# overlap table, with spatial hashing
# ---------------------------------------------------------------------------
def _bucket_pairs(src: TriSide, tgt: TriSide, size: float) -> list:
    """Candidate (i, j) overlap pairs via a uniform hash grid on bounding boxes."""
    def keys(lo, hi):
        k_lo = np.floor(lo / size).astype(np.int64)
        k_hi = np.floor(hi / size).astype(np.int64)
        return k_lo, k_hi

    grid: dict = {}
    tlo, thi = keys(tgt.lo, tgt.hi)
    for j in range(tgt.n):
        for a in range(tlo[j, 0], thi[j, 0] + 1):
            for b in range(tlo[j, 1], thi[j, 1] + 1):
                for c in range(tlo[j, 2], thi[j, 2] + 1):
                    grid.setdefault((a, b, c), []).append(j)
    slo, shi = keys(src.lo, src.hi)
    pairs = set()
    for i in range(src.n):
        for a in range(slo[i, 0], shi[i, 0] + 1):
            for b in range(slo[i, 1], shi[i, 1] + 1):
                for c in range(slo[i, 2], shi[i, 2] + 1):
                    for j in grid.get((a, b, c), ()):
                        pairs.add((i, j))
    return sorted(pairs)


def surface_overlap_entries(src: TriSide, tgt: TriSide, *, cos_min: float = 0.15,
                            tol: float = 1e-14, verbose: bool = False) -> list:
    """Pairs ``(i, j, measure_src, moment)`` with non-degenerate overlap.

    Candidate pairs come from a uniform hash grid on the facet bounding boxes
    (the pairwise bbox pre-screen of m5436 is ``O(n^2)``, which is 20 M pairs for
    the finest curved case here); the dihedral pre-screen then rejects facets that
    belong to different interface patches instead of mixing them.
    """
    size = 1.5 * max(src.characteristic_length(), tgt.characteristic_length())
    entries = []
    n_rejected = 0
    for (i, j) in _bucket_pairs(src, tgt, size):
        if np.any(src.lo[i] > tgt.hi[j] + tol) or np.any(tgt.lo[j] > src.hi[i] + tol):
            continue
        res = tri_pair_overlap(src.tris[i], tgt.tris[j], cos_min=cos_min, tol=tol)
        if res["path"] == "rejected":
            n_rejected += 1
            continue
        if res["measure_src"] > 0.0:
            entries.append((i, j, res["measure_src"], res["moment"]))
    if verbose:
        print(f"[overlap] {src.label} -> {tgt.label}: {len(entries)} pairs, "
              f"{n_rejected} dihedral-rejected")
    return entries


def surface_coverage(src: TriSide, tgt: TriSide, entries) -> dict:
    """Per-side coverage defect: ``sum_j m_ij / |S_i| - 1`` and the target view.

    A *zero* value means the overlaps partition the side exactly (structural
    conservation); a non-zero value is the geometric defect of the projection
    mortar on a curved interface, and it is what the backfill has to absorb.
    """
    cov_s = np.zeros(src.n)
    cov_t = np.zeros(tgt.n)
    for (i, j, m, _mom) in entries:
        cov_s[i] += m
        cov_t[j] += m / max(float(np.dot(src.normals[i], tgt.normals[j])), 1e-300)
    return {"src_rel": float(np.max(np.abs(cov_s / src.measures - 1.0))),
            "tgt_rel": float(np.max(np.abs(cov_t / tgt.measures - 1.0))),
            "src_abs": float(np.max(np.abs(cov_s - src.measures)) / src.area),
            "tgt_abs": float(np.max(np.abs(cov_t - tgt.measures)) / tgt.area),
            "src_total": float(cov_s.sum() / src.area - 1.0),
            "tgt_total": float(cov_t.sum() / tgt.area - 1.0),
            "note": ("``*_rel`` is the per-facet relative defect -- it is dominated "
                     "by measure-zero slivers and is reported as a diagnostic; "
                     "``*_abs`` normalises the worst facet deficit by the total "
                     "area and is the quantity that can actually break a "
                     "conservation gate, so the gate uses ``*_abs`` / ``*_total``")}


# ---------------------------------------------------------------------------
# field helpers
# ---------------------------------------------------------------------------
def surface_cell_averages(side: TriSide, field) -> np.ndarray:
    """Cell means ``∫_cell f dA / |cell|`` by a degree-2 exact Gauss rule."""
    out = np.empty(side.n)
    for k, t in enumerate(side.tris):
        vals = field(_GAUSS_BARY @ t)
        out[k] = float(np.dot(_GAUSS_W, vals) / _GAUSS_W.sum())
    return out


def surface_integral(values: np.ndarray, measures: np.ndarray) -> float:
    """Measure-weighted integral over a triangulated surface."""
    return float(np.dot(np.asarray(values, dtype=float),
                        np.asarray(measures, dtype=float)))


def surface_l2_error(values: np.ndarray, exact: np.ndarray,
                     measures: np.ndarray) -> float:
    """Measure-weighted discrete L2 error (cell-wise)."""
    v = np.asarray(values, dtype=float)
    e = np.asarray(exact, dtype=float)
    m = np.asarray(measures, dtype=float)
    return float(math.sqrt(float(np.dot((v - e) ** 2, m)) / float(m.sum())))


def facet_gradients(side: TriSide, k: int = 8, radius_factor: float = 3.0) -> np.ndarray:
    """Least-squares surface gradient per facet, returned in each facet's frame.

    Shape ``(n, 2, n)``: ``G[i, :, m]`` is the gradient contribution of donor
    value ``u_m`` at facet ``i``, expressed in facet ``i``'s own tangent basis.
    Neighbours are the ``k`` nearest facet centroids within
    ``radius_factor * h_local``; the tangent-plane least-squares fit is exact for
    a linearly varying field on a flat patch, which is what makes the order-2
    patch test (gate D2) machine-exact.
    """
    n = side.n
    G = np.zeros((n, 2, n))
    h = side.characteristic_length()
    scale = max(h, 1e-300)
    for i in range(n):
        c, basis, _ = tri_frame(side.tris[i])
        d = side.centroids - c
        l2 = np.einsum("ij,ij->i", d, d)
        # exclude self (distance 0) and restrict to a local ball
        order = np.argsort(l2)
        near = [m for m in order[:k + 1] if m != i][:k]
        dist = np.sqrt(l2[near])
        keep = dist <= radius_factor * h
        near = [m for m, ok in zip(near, keep) if ok]
        if len(near) < 3:
            near = [m for m in order[:6] if m != i]
        loc = (d[near] @ basis.T) / scale            # (k, 2), O(1) scaled
        A = np.column_stack([loc[:, 0], loc[:, 1], np.ones(len(near))])
        # minimum-norm least squares (exact for linear data); the coordinates are
        # scaled by h so that sliver cells do not wreck the conditioning
        pinv = np.linalg.pinv(A, rcond=1e-12) / scale
        wx, wy = pinv[0], pinv[1]
        G[i, 0, near] = wx
        G[i, 1, near] = wy
        G[i, 0, i] = -wx.sum()
        G[i, 1, i] = -wy.sum()
    return G


def measure_backfill(v: np.ndarray, sent: float, recv: float, measures: np.ndarray,
                     broken: str | None = None) -> np.ndarray:
    """Conservative backfill: ``dv_j = (sent - recv) |T_j| / sum_k |T_k|^2``.

    This is the m5207 section 5.2 fix, re-stated for curved interfaces where the
    deficit is **not** zero: the correction is proportional to the cell measure so
    that ``sum_j dv_j |T_j| = sent - recv`` exactly, independent of the cell-size
    distribution.

    ``broken="backfill_linear"`` normalises by ``sum_k |T_k|`` (the trap): the
    integral of the correction becomes ``(sent - recv) * sum|T|^2 / (sum|T|)^2``,
    which for a mesh of ``N`` comparable cells leaves ``~ (sent-recv)(1 - 1/N)``
    behind -- a percent-level arithmetic residual that looks like a physics error.
    Kept here as an executable negative control.
    """
    m = np.asarray(measures, dtype=float)
    v = np.asarray(v, dtype=float)
    deficit = float(sent) - float(recv)
    if broken is None:
        denom = float(np.sum(m * m))
    elif broken == "backfill_linear":
        denom = float(np.sum(m))
    else:
        raise ValueError(f"unknown backfill mode {broken!r}")
    if denom == 0.0:
        return v
    return v + deficit * m / denom


# ---------------------------------------------------------------------------
# mortar on triangulated (possibly curved) interfaces
# ---------------------------------------------------------------------------
@dataclass
class SurfaceMortarOperator:
    """Mortar projection between two triangulated surfaces (coplanar or not).

    Parameters
    ----------
    order : {1, 2}
        1 = piecewise-constant donor reconstruction (overlap-weighted average);
        2 = adds the dipole term ``g_i . (x - c_i)`` with the surface gradient of
        the donor data, exact for fields that are linear in position.
    backfill : {None, "measure"}
        ``"measure"`` applies the measure-proportional deficit correction; it is
        **required** on a curved interface, where the donor-plane overlaps do not
        partition the acceptor facet.
    broken : {None, "midpoint", "backfill_linear"}
        Negative controls: ``"midpoint"`` point-samples the donor reconstruction at
        the acceptor centroid (classic non-conservative non-matching flux);
        ``"backfill_linear"`` mis-normalises the backfill.
    """

    order: int = 2
    backfill: str | None = None
    broken: str | None = None
    cos_min: float = 0.15
    name: str = "mortar-tri3d"

    def __post_init__(self) -> None:
        if self.order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        if self.backfill not in (None, "measure"):
            raise ValueError("unknown backfill mode")
        if self.broken not in (None, "midpoint", "backfill_linear"):
            raise ValueError("unknown broken mode")

    def transfer(self, src: TriSide, u: np.ndarray, tgt: TriSide,
                 entries=None) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        if u.shape != (src.n,):
            raise ValueError(f"expected {src.n} donor values, got {u.shape}")
        if self.broken == "midpoint":
            v = self._transfer_midpoint(src, u, tgt)
        else:
            ents = surface_overlap_entries(src, tgt, cos_min=self.cos_min) \
                if entries is None else entries
            acc = np.zeros(tgt.n)
            G = facet_gradients(src) if self.order == 2 else None
            for (i, j, m, mom) in ents:
                acc[j] += m * u[i]
                if G is not None:
                    acc[j] += float(mom @ (G[i] @ u))
            v = acc / tgt.measures
        if self.backfill == "measure":
            v = measure_backfill(v, surface_integral(u, src.measures),
                                 surface_integral(v, tgt.measures), tgt.measures,
                                 broken=("backfill_linear"
                                         if self.broken == "backfill_linear" else None))
        return v

    def _transfer_midpoint(self, src: TriSide, u: np.ndarray, tgt: TriSide) -> np.ndarray:
        G = facet_gradients(src) if self.order == 2 else np.zeros((src.n, 2, src.n))
        g = np.einsum("inl,l->in", G, u)
        out = np.empty(tgt.n)
        for j in range(tgt.n):
            c, basis, _ = tri_frame(tgt.tris[j])
            d = src.centroids - c
            k = int(np.argmin(np.einsum("ij,ij->i", d, d)))
            out[j] = u[k] + float((d[k] @ basis.T) @ (G[k] @ u)) if self.order == 2 \
                else u[k]
        return out


# ---------------------------------------------------------------------------
# closed-form references for the curved benchmark
# ---------------------------------------------------------------------------
def octant_reference(radius: float = 1.0) -> dict:
    """Closed-form ``(area, ∫x dA, ∫(x+y+z) dA)`` on the octant ``x,y,z >= 0``.

    ``dA = R dx dφ`` in the axis coordinate, hence
    ``∫_{octant} x dA = R³ (π/2)(1/2) = πR³/4`` and the symmetric companions.
    """
    r = float(radius)
    return {"area": 0.5 * math.pi * r * r,
            "int_x": 0.25 * math.pi * r ** 3,
            "int_sum": 0.75 * math.pi * r ** 3}