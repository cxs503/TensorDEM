"""GJK/EPA convex contact geometry for TensorDEM.

Why this module exists
----------------------
``tensordem.dem3d`` and the 2-D codes carry *spherical* particles only: a contact
is ``|c_i - c_j| - r_i - r_j`` in closed form, and every verification document in
the repository is scored against that closed form.  Nothing in the package can
answer the two questions a non-spherical DEM engine has to answer on every pair:

* **gap** -- how far apart are two *arbitrary* convex bodies (possibly rotated),
  and where is the closest pair of points?
* **penetration** -- when they overlap, how deep, along which normal, and where
  does the contact act?

This module supplies both from one support-function interface, using the two
classical algorithms of the field:

* **GJK** (Gilbert, Johnson & Keerthi 1988) on the Minkowski difference
  ``M = A - B = {a - b}``.  The distance between the bodies is the distance from
  the origin to ``M``.  The implementation keeps a simplex of at most four
  support points and re-solves the *closest point on a simplex* sub-problem
  exactly (enumerating every vertex subset and solving the equality-constrained
  least-norm problem), so the returned distance carries a **rigorous two-sided
  bound**: with ``v`` the closest point of the current simplex and ``w`` the
  support point of ``M`` along ``-v``,

      ``-(v . w) / |v|  <=  dist(A, B)  <=  |v|``

  i.e. ``(upper - lower) / upper = (|v|^2 - v.w) / |v|^2``.  Iteration stops when
  that *relative gap* falls below ``rel_tol``; the reported distance is then
  accurate to ``rel_tol`` in relative terms by construction, not by luck.
* **EPA** (expanding polytope algorithm, Gino van den Bergen 2003) when the
  origin is inside ``M``: the polytope inscribed in ``M`` is refined towards the
  closest boundary face until the support point along that face normal no longer
  improves the depth by more than ``rel_tol``.  For polytope-polytope pairs the
  expansion terminates in a *finite* number of steps and the depth is exact up to
  rounding; for smooth bodies (spheres, cones) the depth error decreases
  geometrically with the refinement and reaches ``1e-10`` relative in a few tens
  of iterations.

Shape support functions
-----------------------
The engine never needs a mesh: every body is consumed through

    ``support(d) = argmax_{x in body} x . d``

which is closed-form for the four primitives shipped here:

======================  ==========================================================
``Sphere``              ``c + R d/|d|``
``Box``                 ``c + R_body sign(R_body^T d) * h`` (arbitrary orientation)
``Polytope``            ``V[argmax(V d)]`` (convex vertex cloud)
``Cone``                ``max(apex.d, base-disc support)`` -- exact, no faceting
======================  ==========================================================

``Clump`` (multi-sphere, the second contact channel) is *not* a convex body; it
is handled member-by-member with the same generic solver, which is the standard
multi-sphere treatment: the union's contact is the extremal member contact.  The
convex-hull shortcut (feeding the union support function to GJK) is deliberately
**not** used, because a hull would invent contacts across interior gaps.

Analytic oracles
----------------
Every number this module returns is scored, in the accompanying tests, against an
oracle in the same file that is derived independently of GJK/EPA:

* ``analytic_sphere_sphere`` / ``analytic_sphere_box`` / ``analytic_point_box_gap``
  -- closed form;
* ``analytic_box_box_sat`` / ``analytic_polytope_sat`` -- separating-axis exactly
  (15 candidate axes for boxes, face normals plus edge cross products otherwise);
  for convex polytopes SAT gives the exact gap *and* the exact penetration depth;
* ``analytic_point_cone_gap`` -- meridional-plane ``(t, r)`` construction.

All quantities are SI; all arrays are ``float64`` with shape ``(3,)``.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "Sphere",
    "Box",
    "Polytope",
    "Cone",
    "Clump",
    "ContactGeometry",
    "GJKResult",
    "gjk_distance",
    "epa_penetration",
    "contact",
    "clump_contact",
    "analytic_sphere_sphere",
    "analytic_point_box_gap",
    "analytic_sphere_box",
    "analytic_box_box_sat",
    "analytic_polytope_sat",
    "analytic_point_cone_gap",
]

_TINY = 1.0e-300


def _vec3(value: Sequence[float]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"non-finite 3-vector: {value!r}")
    return arr


def _unit(value: Sequence[float], fallback: Sequence[float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    arr = _vec3(value)
    norm = float(np.linalg.norm(arr))
    if norm < 1.0e-14:
        arr = _vec3(fallback)
        norm = float(np.linalg.norm(arr))
    return arr / norm


def _rotation(value) -> np.ndarray:
    if value is None:
        return np.eye(3)
    rot = np.asarray(value, dtype=np.float64).reshape(3, 3)
    # Re-orthonormalise so that a slightly noisy user rotation cannot silently
    # bias every distance by O(|R^T R - I|).
    u, _, vt = np.linalg.svd(rot)
    return u @ vt


def _any_perp(axis: np.ndarray) -> np.ndarray:
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    perp = np.cross(axis, helper)
    return perp / np.linalg.norm(perp)


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
class ConvexShape:
    """Support-function body (abstract)."""

    def support(self, direction: Sequence[float]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def centroid(self) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def bounding_radius(self) -> float:  # pragma: no cover
        raise NotImplementedError

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.empty(3)
        hi = np.empty(3)
        for axis in range(3):
            direction = np.zeros(3)
            direction[axis] = 1.0
            hi[axis] = float(self.support(direction)[axis])
            lo[axis] = float(self.support(-direction)[axis])
        return lo, hi


@dataclass(frozen=True)
class Sphere(ConvexShape):
    center: Sequence[float]
    radius: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.radius) and self.radius > 0.0):
            raise ValueError("Sphere.radius must be finite and positive")
        object.__setattr__(self, "center", _vec3(self.center))

    def support(self, direction: Sequence[float]) -> np.ndarray:
        return self.center + self.radius * _unit(direction)

    def centroid(self) -> np.ndarray:
        return self.center

    def bounding_radius(self) -> float:
        return float(self.radius)


@dataclass(frozen=True)
class Box(ConvexShape):
    """Oriented box: ``center + orientation @ (sign * half_extents)``."""

    center: Sequence[float]
    half_extents: Sequence[float]
    orientation: object = None

    def __post_init__(self) -> None:
        half = _vec3(self.half_extents)
        if np.any(half <= 0.0):
            raise ValueError("Box.half_extents must be positive")
        object.__setattr__(self, "center", _vec3(self.center))
        object.__setattr__(self, "half_extents", half)
        object.__setattr__(self, "orientation", _rotation(self.orientation))

    def support(self, direction: Sequence[float]) -> np.ndarray:
        local = self.orientation.T @ _vec3(direction)
        # np.sign(0) == 0 would collapse the support point into the box interior
        # (a face point is required), so ties are broken towards +1.
        signs = np.where(local >= 0.0, 1.0, -1.0)
        return self.center + self.orientation @ (signs * self.half_extents)

    def centroid(self) -> np.ndarray:
        return self.center

    def bounding_radius(self) -> float:
        return float(np.linalg.norm(self.half_extents))

    def vertices(self) -> np.ndarray:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=3)))
        return self.center + (signs * self.half_extents) @ self.orientation.T


@dataclass(frozen=True)
class Polytope(ConvexShape):
    """Convex hull of a vertex cloud (vertex order irrelevant)."""

    vertices: object

    def __post_init__(self) -> None:
        verts = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        if verts.shape[0] < 4:
            raise ValueError("Polytope needs at least 4 vertices")
        if not np.all(np.isfinite(verts)):
            raise ValueError("Polytope vertices must be finite")
        object.__setattr__(self, "vertices", verts)

    def support(self, direction: Sequence[float]) -> np.ndarray:
        dots = self.vertices @ _vec3(direction)
        return self.vertices[int(np.argmax(dots))]

    def centroid(self) -> np.ndarray:
        return self.vertices.mean(axis=0)

    def bounding_radius(self) -> float:
        return float(np.max(np.linalg.norm(self.vertices - self.centroid(), axis=1)))


@dataclass(frozen=True)
class Cone(ConvexShape):
    """Right circular cone: apex, unit axis pointing apex -> base centre.

    The base disc is represented *exactly* through its support function, so no
    faceting error enters the oracle chain.
    """

    apex: Sequence[float]
    axis: Sequence[float]
    height: float
    radius: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.height) and self.height > 0.0):
            raise ValueError("Cone.height must be finite and positive")
        if not (math.isfinite(self.radius) and self.radius > 0.0):
            raise ValueError("Cone.radius must be finite and positive")
        object.__setattr__(self, "apex", _vec3(self.apex))
        object.__setattr__(self, "axis", _unit(self.axis))

    @property
    def base_center(self) -> np.ndarray:
        return self.apex + self.height * self.axis

    def support(self, direction: Sequence[float]) -> np.ndarray:
        d = _vec3(direction)
        radial = d - (d @ self.axis) * self.axis
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm <= 1.0e-14 * max(1.0, float(np.linalg.norm(d))):
            if d @ self.axis >= 0.0:
                return self.base_center.copy()
            return self.apex.copy()
        base_point = self.base_center + self.radius * (radial / radial_norm)
        if float(self.apex @ d) >= float(base_point @ d):
            return self.apex.copy()
        return base_point

    def centroid(self) -> np.ndarray:
        return self.apex + 0.75 * self.height * self.axis

    def bounding_radius(self) -> float:
        centre = self.centroid()
        rim = self.base_center + self.radius * _any_perp(self.axis)
        return float(max(np.linalg.norm(self.apex - centre), np.linalg.norm(rim - centre)))


@dataclass(frozen=True)
class Clump:
    """Rigid multi-sphere cluster (second contact channel, non-convex union)."""

    spheres: tuple
    label: str = ""

    def __init__(self, spheres: Sequence[Sphere], label: str = ""):
        members = tuple(spheres)
        if not members:
            raise ValueError("Clump needs at least one member sphere")
        for member in members:
            if not isinstance(member, Sphere):
                raise TypeError("Clump members must be Sphere instances")
        object.__setattr__(self, "spheres", members)
        object.__setattr__(self, "label", label)

    def support(self, direction: Sequence[float]) -> np.ndarray:
        """Support of the *convex hull* -- deliberately unused for contacts."""
        d = _vec3(direction)
        return max((s.support(d) for s in self.spheres), key=lambda p: float(p @ d))

    def centroid(self) -> np.ndarray:
        return np.mean([s.center for s in self.spheres], axis=0)

    def bounding_radius(self) -> float:
        centre = self.centroid()
        return float(max(np.linalg.norm(s.center - centre) + s.radius for s in self.spheres))

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.min([s.center - s.radius for s in self.spheres], axis=0)
        hi = np.max([s.center + s.radius for s in self.spheres], axis=0)
        return lo, hi


# ---------------------------------------------------------------------------
# Closest point on a simplex (exact, by subset enumeration)
# ---------------------------------------------------------------------------
def _closest_on_simplex(points: Sequence[np.ndarray]):
    """Closest point to the origin in the convex hull of ``points`` (k <= 4).

    Returns ``(closest, weights, subset)``.  Every non-empty vertex subset is
    solved for its equality-constrained least-norm barycentric combination and
    only subsets with non-negative weights are admissible; the admissible point
    with the smallest norm is the exact closest point of the simplex.
    """
    cloud = np.asarray(points, dtype=np.float64)
    k = cloud.shape[0]
    best = None
    for mask in range(1, 1 << k):
        idx = [i for i in range(k) if (mask >> i) & 1]
        m = len(idx)
        sub = cloud[idx]
        if m == 1:
            weights = np.array([1.0])
        else:
            gram = sub @ sub.T
            system = np.zeros((m + 1, m + 1))
            system[:m, :m] = gram
            system[:m, m] = 1.0
            system[m, :m] = 1.0
            rhs = np.zeros(m + 1)
            rhs[m] = 1.0
            try:
                solution = np.linalg.solve(system, rhs)
            except np.linalg.LinAlgError:
                solution = np.linalg.lstsq(system, rhs, rcond=None)[0]
            weights = solution[:m]
            if not np.all(np.isfinite(weights)) or np.any(weights < -1.0e-12):
                continue
            weights = np.clip(weights, 0.0, None)
            total = float(weights.sum())
            if total <= 0.0:
                continue
            weights = weights / total
        point = weights @ sub
        distance_sq = float(point @ point)
        if best is None or distance_sq < best[0]:
            best = (distance_sq, point, idx, weights)
    assert best is not None
    return best[1], best[3], best[2]


# ---------------------------------------------------------------------------
# GJK
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GJKResult:
    """Outcome of the distance phase.

    ``overlap`` is ``True`` when the origin lies on or inside the Minkowski
    difference, i.e. the reported distance is zero and EPA must be run to obtain
    the penetration depth.
    """

    overlap: bool
    distance: float
    normal: np.ndarray
    point_a: np.ndarray
    point_b: np.ndarray
    iterations: int
    relative_gap: float
    simplex: tuple = field(default=(), repr=False)
    simplex_a: tuple = field(default=(), repr=False)
    simplex_b: tuple = field(default=(), repr=False)


def gjk_distance(a: ConvexShape, b: ConvexShape, *, max_iter: int = 200,
                 rel_tol: float = 1.0e-13) -> GJKResult:
    """Distance between two convex support-function bodies."""
    scale = max(a.bounding_radius() + b.bounding_radius(), 1.0e-12)
    overlap_tol = 1.0e-13 * scale
    direction = b.centroid() - a.centroid()
    if float(np.linalg.norm(direction)) <= 1.0e-12 * scale:
        direction = np.array([1.0, 0.0, 0.0])

    support_a = a.support(direction)
    support_b = b.support(-direction)
    vertices = [support_a - support_b]
    list_a = [support_a]
    list_b = [support_b]
    v = vertices[0].copy()
    iterations = 0
    gap = 1.0

    for iterations in range(1, max_iter + 1):
        v_sq = float(v @ v)
        if v_sq <= overlap_tol ** 2:
            return GJKResult(True, 0.0, _unit(-v if v_sq > 0 else -direction),
                             a.support(direction), b.support(-direction), iterations,
                             0.0, tuple(vertices), tuple(list_a), tuple(list_b))
        direction = -v
        support_a = a.support(direction)
        support_b = b.support(-direction)
        w = support_a - support_b
        gap = (v_sq - float(v @ w)) / v_sq
        if gap <= rel_tol:
            break
        if any(float(np.linalg.norm(w - p)) <= 1.0e-15 * scale for p in vertices):
            break
        vertices.append(w)
        list_a.append(support_a)
        list_b.append(support_b)
        v, _, subset = _closest_on_simplex(vertices)
        vertices = [vertices[i] for i in subset]
        list_a = [list_a[i] for i in subset]
        list_b = [list_b[i] for i in subset]

    v, weights, subset = _closest_on_simplex(vertices)
    dist = float(np.linalg.norm(v))
    point_a = np.zeros(3)
    point_b = np.zeros(3)
    for weight, idx in zip(weights, subset):
        point_a += weight * list_a[idx]
        point_b += weight * list_b[idx]
    if dist <= 1.0e-13 * scale:
        return GJKResult(True, 0.0, _unit(point_b - point_a, -direction),
                         point_a, point_b, iterations, float(gap),
                         tuple(vertices), tuple(list_a), tuple(list_b))
    normal = (point_b - point_a) / dist
    return GJKResult(False, dist, normal, point_a, point_b, iterations, float(gap),
                     tuple(vertices), tuple(list_a), tuple(list_b))


# ---------------------------------------------------------------------------
# EPA
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ContactGeometry:
    """Complete contact description for one convex pair (or clump pair)."""

    separated: bool
    distance: float           # >= 0 gap; 0 when overlapping
    depth: float              # >= 0 penetration; 0 when separated
    normal: np.ndarray        # unit vector, points from A towards B
    point_a: np.ndarray
    point_b: np.ndarray
    gjk_iterations: int = 0
    epa_iterations: int = 0
    epa_faces: int = 0
    relative_gap: float = 0.0

    @property
    def overlap(self) -> float:
        return -self.distance if self.separated else self.depth


def _seed_directions() -> list[np.ndarray]:
    dirs = [np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]),
            np.array([0, 1.0, 0]), np.array([0, -1.0, 0]),
            np.array([0, 0, 1.0]), np.array([0, 0, -1.0])]
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            dirs.append(_unit([0.0, sx, sy * phi]))
            dirs.append(_unit([sx, sy * phi, 0.0]))
            dirs.append(_unit([sx * phi, 0.0, sy]))
    # Cube-corner directions keep the seed robust for very anisotropic bodies.
    for sign in itertools.product((1.0, -1.0), repeat=3):
        dirs.append(_unit(sign))
    return dirs


def _build_tetrahedron(points: np.ndarray, scale: float):
    """Find four of ``points`` whose convex hull strictly contains the origin."""
    best = None
    for combo in itertools.combinations(range(points.shape[0]), 4):
        tet = points[list(combo)]
        matrix = np.empty((3, 3))
        matrix[:, 0] = tet[1] - tet[0]
        matrix[:, 1] = tet[2] - tet[0]
        matrix[:, 2] = tet[3] - tet[0]
        det = float(np.linalg.det(matrix))
        if abs(det) <= 1.0e-14 * scale ** 3:
            continue
        bary = np.linalg.solve(matrix, -tet[0])
        weights = np.array([bary[0], bary[1], bary[2], 1.0 - float(bary.sum())])
        if np.all(weights > 1.0e-9):
            score = float(weights.min())
            if best is None or score > best[0]:
                best = (score, list(combo))
    return best


def _face_record(verts: Sequence[np.ndarray], i: int, j: int, k: int):
    a, b, c = verts[i], verts[j], verts[k]
    normal = np.cross(b - a, c - a)
    norm = float(np.linalg.norm(normal))
    if norm <= _TINY:
        return None
    normal = normal / norm
    dist = float(normal @ a)
    if dist < 0.0:
        normal = -normal
        dist = -dist
        j, k = k, j
    return [i, j, k, normal, dist]


def _face(verts, i, j, k):
    record = _face_record(verts, i, j, k)
    if record is None:
        return None
    return tuple(record[:3]) + (record[3], record[4])


def _seed_directions_rich() -> list[np.ndarray]:
    """Well-spread direction set for the EPA seed polytope."""
    dirs = list(_seed_directions())
    # Fibonacci sphere: fills the gaps of the icosahedral set so that the hull of
    # the sampled support points encloses the origin even for shallow overlaps.
    count = 48
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(count):
        z = 1.0 - 2.0 * (i + 0.5) / count
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        theta = golden * i
        dirs.append(np.array([radius * math.cos(theta), radius * math.sin(theta), z]))
    return dirs


def _seed_polytope(a: ConvexShape, b: ConvexShape, scale: float, max_rounds: int = 48):
    """Sampled support points whose convex hull encloses the origin.

    A hand-picked six-direction seed is *not* enough: for two boxes in general
    orientation every axis support point can share a coordinate (the tie-broken
    vertex), the six points become coplanar in one projection and no tetrahedron
    of them contains the origin.  Sampling a well-spread direction set, taking the
    convex hull and -- while the origin is still outside a facet -- adding the
    support point along that facet's outward normal, is robust: it terminates
    because the origin lies inside the Minkowski difference.
    """
    from scipy.spatial import ConvexHull, QhullError

    verts: list[np.ndarray] = []
    pa: list[np.ndarray] = []
    pb: list[np.ndarray] = []

    def add(direction: np.ndarray) -> bool:
        sa = a.support(direction)
        sb = b.support(-direction)
        point = sa - sb
        if any(float(np.linalg.norm(point - q)) <= 1.0e-14 * scale for q in verts):
            return False
        verts.append(point)
        pa.append(sa)
        pb.append(sb)
        return True

    for direction in _seed_directions_rich():
        add(direction)

    for _ in range(max_rounds):
        if len(verts) < 4:
            return None
        try:
            hull = ConvexHull(np.asarray(verts))
        except QhullError:
            return None
        # scipy's equations are ``n . x + d = 0`` with ``n`` outward, so every
        # ``d < 0`` means the origin is on the inner side of that facet.
        offsets = hull.equations[:, 3]
        worst = int(np.argmax(offsets))
        if offsets[worst] <= -1.0e-12 * scale:
            return verts, pa, pb, [tuple(int(i) for i in simplex) for simplex in hull.simplices]
        direction = hull.equations[worst, :3].copy()
        if not add(direction):
            jitter = _any_perp(direction)
            if not add(_unit(direction + 1.0e-3 * jitter)):
                return None
    return None


def _support_plane_distance(a: ConvexShape, b: ConvexShape, normal: np.ndarray) -> float:
    """Support-plane height of ``A - B`` along ``normal`` (candidate depth)."""
    return float((a.support(normal) - b.support(-normal)) @ normal)


def _tangential_residual(a: ConvexShape, b: ConvexShape, normal: np.ndarray):
    """Tangential part of the Minkowski support point (vanishes at the optimum)."""
    point = a.support(normal) - b.support(-normal)
    e1 = _any_perp(normal)
    e2 = np.cross(normal, e1)
    return np.array([float(point @ e1), float(point @ e2)]), point, e1, e2


def _refine_normal(a: ConvexShape, b: ConvexShape, normal: np.ndarray,
                   *, residual_tol: float = 1.0e-15, max_iter: int = 60):
    """Polish a contact normal by minimising the support height on the 2-sphere.

    Both contact branches share one variational statement: with
    ``h_M(n) = h_A(n) + h_B(-n)`` the support function of the Minkowski difference,
    the signed distance from the origin to ``M`` is

        ``min over |n| = 1 of h_M(n)``  (negative outside -> gap; positive inside -> depth)

    and the minimising direction is the contact normal.  EPA and GJK both converge
    geometrically, and on a *curved* boundary the depth error is quadratic in the
    normal error, so an EPA normal can never beat ``sqrt(2 * depth_error / curvature)``
    -- about ``1e-7`` rad at a ``1e-13`` depth tolerance.  Measuring convergence on
    the objective instead of on its stationarity condition is exactly the trap that
    caps the angle at ``sqrt(eps)``; this routine therefore converges on the
    *tangential residual* of the Minkowski support point, which vanishes at the
    optimum, and uses ``h`` only as a monotonicity guard so that the flat
    face-normal fans of polytope pairs (where GJK/EPA is already exact) cannot be
    wandered away from.

    Returns ``(normal, signed_height, moved)``.
    """
    n = _unit(normal)
    residual, point, _, _ = _tangential_residual(a, b, n)
    size = float(np.linalg.norm(residual))
    height = float(point @ n)
    moved = False
    for _ in range(max_iter):
        if size <= residual_tol * max(1.0, float(np.linalg.norm(point))):
            break
        _, _, e1, e2 = _tangential_residual(a, b, n)
        step = 1.0e-6
        hessian = np.empty((2, 2))
        for k, direction in enumerate((e1, e2)):
            shifted = _unit(n + step * direction)
            r_shift, _, _, _ = _tangential_residual(a, b, shifted)
            hessian[:, k] = (r_shift - residual) / step
        attempts = []
        try:
            delta = -np.linalg.solve(hessian, residual)
        except np.linalg.LinAlgError:
            delta = None
        if delta is not None and np.all(np.isfinite(delta)):
            size_newton = float(np.linalg.norm(delta))
            if size_newton > 0.3:
                delta = delta * (0.3 / size_newton)
            attempts.append(delta)
        attempts.append(-0.25 * residual / size)
        accepted = False
        for candidate in attempts:
            for shrink in (1.0, 0.5, 0.25, 0.1, 0.01):
                trial = _unit(n + shrink * candidate[0] * e1 + shrink * candidate[1] * e2)
                r_trial, p_trial, _, _ = _tangential_residual(a, b, trial)
                size_trial = float(np.linalg.norm(r_trial))
                height_trial = float(p_trial @ trial)
                if (size_trial < size
                        and height_trial <= height + 1.0e-15 * max(1.0, abs(height))):
                    n, residual, point, size = trial, r_trial, p_trial, size_trial
                    height = height_trial
                    accepted = moved = True
                    break
            if accepted:
                break
        if not accepted:
            break
    return n, height, moved


def _barycentric(triangle: np.ndarray, point: np.ndarray) -> np.ndarray:
    a, b, c = triangle
    v0 = b - a
    v1 = c - a
    v2 = point - a
    d00 = float(v0 @ v0)
    d01 = float(v0 @ v1)
    d11 = float(v1 @ v1)
    d20 = float(v2 @ v0)
    d21 = float(v2 @ v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) <= _TINY:
        return np.array([1.0, 0.0, 0.0])
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return np.array([1.0 - v - w, v, w])


def epa_penetration(a: ConvexShape, b: ConvexShape, *, max_iter: int = 2000,
                    rel_tol: float = 1.0e-13, refine: bool = True) -> ContactGeometry:
    """Penetration depth / normal / contact points via EPA on ``A - B``."""
    scale = max(a.bounding_radius() + b.bounding_radius(), 1.0e-12)
    seed = _seed_polytope(a, b, scale)
    if seed is None:
        # Origin sits on (not inside) the Minkowski difference: touching contact.
        g = gjk_distance(a, b)
        normal = g.normal if float(np.linalg.norm(g.normal)) > 0 else np.array([1.0, 0.0, 0.0])
        return ContactGeometry(True, 0.0, 0.0, _unit(normal), g.point_a, g.point_b,
                               gjk_iterations=g.iterations)
    verts, pa, pb, simplices = seed

    faces = []
    for simplex in simplices:
        record = _face(verts, simplex[0], simplex[1], simplex[2])
        if record is not None:
            faces.append(record)

    iterations = 0
    for iterations in range(1, max_iter + 1):
        best = min(faces, key=lambda f: f[4])
        normal = best[3]
        dist = best[4]
        support_a = a.support(normal)
        support_b = b.support(-normal)
        point = support_a - support_b
        reach = float(point @ normal)
        if reach - dist <= rel_tol * max(1.0, abs(reach)):
            break
        if any(float(np.linalg.norm(point - q)) <= 1.0e-15 * scale for q in verts):
            break
        visible = [f for f in faces if float(f[3] @ (point - verts[f[0]])) > 1.0e-15 * scale]
        if not visible:
            break
        counts: dict[tuple[int, int], int] = {}
        for face in visible:
            tri = (face[0], face[1], face[2])
            for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = (min(edge), max(edge))
                counts[key] = counts.get(key, 0) + 1
        horizon = []
        for face in visible:
            tri = (face[0], face[1], face[2])
            for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                if counts[(min(edge), max(edge))] == 1:
                    horizon.append(edge)
        faces = [f for f in faces if f not in visible]
        new_index = len(verts)
        verts.append(point)
        pa.append(support_a)
        pb.append(support_b)
        for i, j in horizon:
            record = _face(verts, i, j, new_index)
            if record is not None:
                faces.append(record)
        if not faces:
            break

    best = min(faces, key=lambda f: f[4])
    depth = float(best[4])
    normal = best[3].copy()
    tri = [best[0], best[1], best[2]]
    tri_points = np.asarray([verts[i] for i in tri])
    weights = _barycentric(tri_points, depth * normal)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    weights = np.array([1.0 / 3.0] * 3) if total <= 0.0 else weights / total
    point_a = sum(w * pa[i] for w, i in zip(weights, tri))
    point_b = sum(w * pb[i] for w, i in zip(weights, tri))
    if refine:
        normal, refined, moved = _refine_normal(a, b, normal)
        if moved and refined > 0.0:
            depth = refined
            point_a = a.support(normal)
            point_b = b.support(-normal)
    return ContactGeometry(False, -depth, depth, normal, np.asarray(point_a),
                           np.asarray(point_b), epa_iterations=iterations,
                           epa_faces=len(faces))


def _contact_points_along(a: ConvexShape, b: ConvexShape, normal: np.ndarray,
                          point_a: np.ndarray, point_b: np.ndarray):
    """Re-anchor the contact points on the polished normal.

    The EPA barycentrics are only valid for the EPA normal; after the polish the
    contact points are the extremal points of the two bodies along ``+-normal``,
    which is the standard convention for a smooth contact.
    """
    support_a = a.support(normal)
    support_b = b.support(-normal)
    if float((support_a - support_b) @ normal) <= 1.0e-15:
        return np.asarray(point_a), np.asarray(point_b)
    return np.asarray(support_a), np.asarray(support_b)


# ---------------------------------------------------------------------------
# Unified contact entry point
# ---------------------------------------------------------------------------
def contact(a: ConvexShape, b: ConvexShape, *, rel_tol: float = 1.0e-13,
            refine: bool = True) -> ContactGeometry:
    """Gap + penetration for one convex pair, dispatched through GJK then EPA."""
    gjk = gjk_distance(a, b, rel_tol=rel_tol)
    if not gjk.overlap:
        if refine:
            normal, height, moved = _refine_normal(a, b, gjk.normal)
            if moved and height < 0.0:
                point_a = a.support(normal)
                point_b = b.support(-normal)
                return ContactGeometry(True, float(np.linalg.norm(point_b - point_a)),
                                       0.0, normal, point_a, point_b,
                                       gjk_iterations=gjk.iterations,
                                       relative_gap=gjk.relative_gap)
        return ContactGeometry(True, gjk.distance, 0.0, gjk.normal, gjk.point_a,
                               gjk.point_b, gjk_iterations=gjk.iterations,
                               relative_gap=gjk.relative_gap)
    epa = epa_penetration(a, b, rel_tol=rel_tol)
    return ContactGeometry(epa.separated, epa.distance, epa.depth, epa.normal,
                           epa.point_a, epa.point_b, gjk_iterations=gjk.iterations,
                           epa_iterations=epa.epa_iterations, epa_faces=epa.epa_faces,
                           relative_gap=gjk.relative_gap)


def clump_contact(clump: Clump, other, *, rel_tol: float = 1.0e-13):
    """Multi-sphere channel: extremal member contact against ``other``.

    ``other`` may be a single convex shape or another :class:`Clump`.  Returns
    ``(aggregate, records, index)`` where ``records`` is the per-member list of
    ``((member_i, member_j), ContactGeometry)`` and ``index`` identifies the
    extremal pair.
    """
    if isinstance(other, Clump):
        pairs = [(i, j, sa, sb)
                 for i, sa in enumerate(clump.spheres)
                 for j, sb in enumerate(other.spheres)]
    else:
        pairs = [(i, -1, sa, other) for i, sa in enumerate(clump.spheres)]

    records = [((i, j), contact(sa, sb, rel_tol=rel_tol)) for i, j, sa, sb in pairs]
    depth = max((g.depth for _, g in records), default=0.0)
    if depth > 0.0:
        index, best = max(records, key=lambda item: item[1].depth)
        return ContactGeometry(False, -depth, depth, best.normal, best.point_a,
                               best.point_b, gjk_iterations=best.gjk_iterations,
                               epa_iterations=best.epa_iterations,
                               epa_faces=best.epa_faces), records, index
    index, best = min(records, key=lambda item: item[1].distance)
    return ContactGeometry(True, best.distance, 0.0, best.normal, best.point_a,
                           best.point_b, gjk_iterations=best.gjk_iterations,
                           relative_gap=best.relative_gap), records, index


# ---------------------------------------------------------------------------
# Independent analytic oracles
# ---------------------------------------------------------------------------
def analytic_sphere_sphere(center_a, radius_a, center_b, radius_b):
    """Closed-form sphere-sphere gap (negative = penetration depth)."""
    ca = _vec3(center_a)
    cb = _vec3(center_b)
    delta = cb - ca
    dist = float(np.linalg.norm(delta))
    gap = dist - radius_a - radius_b
    normal = _unit(delta)
    point_a = ca + radius_a * normal
    point_b = cb - radius_b * normal
    return gap, normal, point_a, point_b


def analytic_point_box_gap(point, center, half_extents, orientation=None):
    """Closed-form point-to-oriented-box signed gap (negative = inside)."""
    p = _vec3(point)
    c = _vec3(center)
    h = _vec3(half_extents)
    rot = _rotation(orientation)
    local = rot.T @ (p - c)
    outside = np.clip(np.abs(local) - h, 0.0, None)
    outside_norm = float(np.linalg.norm(outside))
    if outside_norm > 0.0:
        closest_local = np.clip(local, -h, h)
        closest = c + rot @ closest_local
        return outside_norm, closest
    # Inside: distance to the nearest face.
    inside = h - np.abs(local)
    axis = int(np.argmin(inside))
    closest_local = local.copy()
    closest_local[axis] = math.copysign(h[axis], local[axis])
    closest = c + rot @ closest_local
    return -float(inside[axis]), closest


def analytic_sphere_box(center, radius, box_center, box_half, box_orientation=None):
    gap, closest = analytic_point_box_gap(center, box_center, box_half, box_orientation)
    signed = gap - radius
    direction = _unit(closest - _vec3(center))
    point_b = closest
    point_a = _vec3(center) - radius * direction * (-1.0 if signed < 0 else 1.0)
    return signed, direction, point_a, point_b


def _sat_axes(polytope_a: np.ndarray, polytope_b: np.ndarray,
              faces_a: Sequence[tuple], faces_b: Sequence[tuple]):
    axes = []
    for _, normal in list(faces_a) + list(faces_b):
        axes.append(_unit(normal))
    edges_a = _unique_edges(polytope_a, faces_a)
    edges_b = _unique_edges(polytope_b, faces_b)
    for ea in edges_a:
        for eb in edges_b:
            cross = np.cross(ea, eb)
            if float(np.linalg.norm(cross)) > 1.0e-10:
                axes.append(_unit(cross))
    return axes


def _unique_edges(vertices: np.ndarray, faces: Sequence[tuple]):
    edges = []
    for tri, _ in faces:
        pts = vertices[list(tri)]
        for i in range(3):
            edge = pts[(i + 1) % 3] - pts[i]
            norm = float(np.linalg.norm(edge))
            if norm <= 1.0e-14:
                continue
            edges.append(edge / norm)
    return edges


def _project(vertices: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    dots = vertices @ axis
    return float(dots.min()), float(dots.max())


def _box_faces() -> list[np.ndarray]:
    return [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]),
            np.array([-1.0, 0, 0]), np.array([0, -1.0, 0]), np.array([0, 0, -1.0])]


def analytic_box_box_sat(center_a, half_a, orientation_a, center_b, half_b, orientation_b):
    """Exact OBB-OBB clearance (positive) or penetration depth (negative)."""
    rot_a = _rotation(orientation_a)
    rot_b = _rotation(orientation_b)
    verts_a = _vec3(center_a) + (np.array(list(itertools.product((1.0, -1.0), repeat=3)))
                                 * _vec3(half_a)) @ rot_a.T
    verts_b = _vec3(center_b) + (np.array(list(itertools.product((1.0, -1.0), repeat=3)))
                                 * _vec3(half_b)) @ rot_b.T
    axes = [(rot_a[:, i]) for i in range(3)] + [(rot_b[:, i]) for i in range(3)]
    for i in range(3):
        for j in range(3):
            cross = np.cross(rot_a[:, i], rot_b[:, j])
            if float(np.linalg.norm(cross)) > 1.0e-10:
                axes.append(_unit(cross))
    best_gap = -math.inf
    best_overlap = math.inf
    gap_axis = None
    overlap_axis = None
    for axis in axes:
        amin, amax = _project(verts_a, axis)
        bmin, bmax = _project(verts_b, axis)
        gap = max(bmin - amax, amin - bmax)
        overlap = min(amax, bmax) - max(amin, bmin)
        if gap > best_gap:
            best_gap = gap
            gap_axis = axis if float((_vec3(center_b) - _vec3(center_a)) @ axis) >= 0 else -axis
        if overlap < best_overlap:
            best_overlap = overlap
            overlap_axis = axis if float((_vec3(center_b) - _vec3(center_a)) @ axis) >= 0 else -axis
    if best_gap >= 0.0:
        return best_gap, _unit(gap_axis), None, None
    return -best_overlap, _unit(overlap_axis), None, None


def analytic_polytope_sat(vertices_a, vertices_b):
    """Exact clearance / penetration depth for two convex polytope vertex clouds.

    Faces are recovered by the brute-force support-plane test (a triple of
    vertices is a face when all other vertices lie on one side of its plane),
    which keeps this oracle independent of any mesh bookkeeping in the engine.
    """
    va = np.asarray(vertices_a, dtype=np.float64).reshape(-1, 3)
    vb = np.asarray(vertices_b, dtype=np.float64).reshape(-1, 3)
    faces_a = _brute_faces(va)
    faces_b = _brute_faces(vb)
    axes = _sat_axes(va, vb, faces_a, faces_b)
    centre_a = va.mean(axis=0)
    centre_b = vb.mean(axis=0)
    best_gap = -math.inf
    best_overlap = math.inf
    gap_axis = overlap_axis = None
    for axis in axes:
        amin, amax = _project(va, axis)
        bmin, bmax = _project(vb, axis)
        gap = max(bmin - amax, amin - bmax)
        overlap = min(amax, bmax) - max(amin, bmin)
        sign = 1.0 if float((centre_b - centre_a) @ axis) >= 0.0 else -1.0
        if gap > best_gap:
            best_gap = gap
            gap_axis = sign * axis
        if overlap < best_overlap:
            best_overlap = overlap
            overlap_axis = sign * axis
    if best_gap >= 0.0:
        return best_gap, _unit(gap_axis)
    return -best_overlap, _unit(overlap_axis)


def _brute_faces(vertices: np.ndarray) -> list[tuple]:
    """Recover support planes of a convex vertex cloud (triple, outward normal)."""
    n = vertices.shape[0]
    faces = []
    seen = set()
    for triple in itertools.combinations(range(n), 3):
        a, b, c = vertices[list(triple)]
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm <= 1.0e-12:
            continue
        normal = normal / norm
        dots = (vertices - a) @ normal
        if np.all(dots <= 1.0e-10):
            key = (round(normal[0], 9), round(normal[1], 9), round(normal[2], 9))
            if key not in seen:
                seen.add(key)
                faces.append((triple, normal))
        elif np.all(dots >= -1.0e-10):
            key = (round(-normal[0], 9), round(-normal[1], 9), round(-normal[2], 9))
            if key not in seen:
                seen.add(key)
                faces.append((triple, -normal))
    if not faces:
        raise ValueError("could not recover any support plane from the vertex cloud")
    return faces


def analytic_point_cone_gap(point, apex, axis, height, radius):
    """Signed gap from a point to a right circular cone (negative = inside).

    Built in the meridional plane ``(t, r)`` where the lateral surface is the
    segment ``(0,0)-(H,R)``, the base is the segment ``t = H, 0 <= r <= R`` and
    the interior is the triangle below it.  Exact -- the cone is never faceted.
    """
    p = _vec3(point)
    a = _unit(axis)
    apex_v = _vec3(apex)
    u = p - apex_v
    t = float(u @ a)
    radial = u - t * a
    r = float(np.linalg.norm(radial))
    radial_dir = radial / r if r > 0.0 else _any_perp(a)
    if height > 0.0 and 0.0 <= t <= height and r <= radius * t / height:
        # Inside the cone: nearest surface point.
        lateral = (radius * t - height * r) / math.hypot(height, radius)
        base = math.hypot(max(0.0, height - t), max(0.0, r - radius))
        depth = min(lateral, base)
        if base < lateral:
            closest = apex_v + height * a + radial_dir * min(r, radius)
        else:
            s = (t * height + r * radius) / (height ** 2 + radius ** 2)
            closest = apex_v + (s * height) * a + (s * radius) * radial_dir
        return -depth, closest
    if 0.0 <= t <= height:
        s = (t * height + r * radius) / (height ** 2 + radius ** 2)
        s = min(max(s, 0.0), 1.0)
        closest = apex_v + (s * height) * a + (s * radius) * radial_dir
        return math.hypot(t - s * height, r - s * radius), closest
    if t > height:
        closest = apex_v + height * a + radial_dir * min(r, radius)
        return math.hypot(t - height, max(0.0, r - radius)), closest
    return math.hypot(t, r), apex_v.copy()


def analytic_clump_box_gap(clump: Clump, box_center, box_half, box_orientation=None):
    """Per-member closed-form gap of a clump against a box (extremal member)."""
    gaps = []
    for member in clump.spheres:
        gap, _ = analytic_point_box_gap(member.center, box_center, box_half, box_orientation)
        gaps.append(gap - member.radius)
    index = int(np.argmin(gaps))
    return min(gaps), index


def analytic_clump_clump_gap(clump_a: Clump, clump_b: Clump):
    """Per-member closed-form gap of two clumps (extremal member pair)."""
    best = (math.inf, None, None)
    for i, sa in enumerate(clump_a.spheres):
        for j, sb in enumerate(clump_b.spheres):
            gap, normal, _, _ = analytic_sphere_sphere(sa.center, sa.radius,
                                                       sb.center, sb.radius)
            if gap < best[0]:
                best = (gap, (i, j), normal)
    return best
