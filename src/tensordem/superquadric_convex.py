"""Superquadric bodies as support-function shapes for the GJK/EPA engine (m6292).

``tensordem.superquadric.superquadric_contact_batch`` finds the contact of a
superquadric pair with a four-sweep fixed point on the common-normal
condition.  That iteration is fast (1-3 us/pair) but its fixed point is **not
unique** for the flat-facet family: the reported overlap of one and the same
pair can differ by up to ``2.4 R`` between two sweep counts, including a sign
flip (contact reported as separation).  See
``docs/superquadric_flat_facet_reference.md``.

This module supplies the missing *independent* path.  A superquadric is a
convex, centrally symmetric body with a closed-form support function

    ``support(d) = c + R ( s(u) u )``,   ``u = R^T d / |R^T d|``

with ``s`` the radial function already used by the kernel
(:func:`tensordem.superquadric._radius_along`).  Feeding that through
:class:`tensordem.convex_contact.ConvexShape` makes the whole GJK (gap) / EPA
(penetration depth) machinery -- together with its independently coded
analytic oracles ``analytic_sphere_sphere`` and ``analytic_box_box_sat`` --
usable on superquadrics, so the fast kernel finally has a reference it can be
scored against.

Nothing here changes the production fast path: the adapter is additive, and the
only reuse is the radial function itself, so the two paths cannot disagree
about *geometry*, only about *which common-normal solution* they report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .convex_contact import (
    ConvexShape,
    ContactGeometry,
    contact as _convex_contact,
    analytic_box_box_sat,
    analytic_sphere_sphere,
)
from .superquadric import (
    SuperquadricBatch,
    _BLOCK_EXPONENT,
    _radius_along,
)

__all__ = [
    "SuperquadricShape",
    "superquadric_shape",
    "contact_superquadric_pair",
    "batch_to_shapes",
    "support_distance",
    "support_point",
    "analytic_sphere_limit",
    "analytic_block_limit",
    "geometry_dict",
    "support_distance_many",
    "fibonacci_sphere",
    "support_height",
    "signed_gap_along",
    "contact_by_direction_min",
]


def support_distance(
    direction_local: Sequence[float],
    a: float, b: float, c: float,
    n1: float, n2: float,
) -> float:
    """Support function ``h(d) = max_{x in body} x . d`` in the local frame.

    The body is the unit ball of the composed norm

        ``phi(x) = ( ( |x/a|^n2 + |y/b|^n2 )^(n1/n2) + |z/c|^n1 )^(1/n1)``

    (the kernel's implicit function is ``F = phi^n1 - 1``), and the dual of a
    composition of norms is the composition of the dual norms, with Holder
    conjugate exponents ``n* = n/(n-1)``::

        ``h(d) = [ ( (a|d1|)^p + (b|d2|)^p )^(q/p) + (c|d3|)^q ]^(1/q)``,
        ``p = n2/(n2-1)``, ``q = n1/(n1-1)``

    Limits: ``n1 = n2 = 2`` gives ``h = |diag(a,b,c) d|`` (ellipsoid/sphere),
    ``n1, n2 -> inf`` gives ``h = max(a|d1|, b|d2|, c|d3|)`` (box).  This is a
    different function from the radial distance ``s(u)`` used by the fixed-point
    kernel: ``s(u) u`` is the surface point along ``u``, which is a support
    point only for the sphere.  Verified against a brute-force max over sampled
    surface points to <= 1e-4 relative (sampling-limited).
    """
    d = np.asarray(direction_local, dtype=float).reshape(3)
    if not np.any(d):
        raise ValueError("support direction must be non-zero")
    if n1 > _BLOCK_EXPONENT and n2 > _BLOCK_EXPONENT:
        # dual of the max-norm ball is the L1 norm: h = a|d1| + b|d2| + c|d3|
        return float(abs(d[0]) * a + abs(d[1]) * b + abs(d[2]) * c)
    p = n2 / (n2 - 1.0)
    q = n1 / (n1 - 1.0)
    u = np.abs(d[0]) * a
    v = np.abs(d[1]) * b
    w = np.abs(d[2]) * c
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inner = (u ** p + v ** p) ** (1.0 / p)
        return float((inner ** q + w ** q) ** (1.0 / q))


def support_point(
    direction_local: Sequence[float],
    a: float, b: float, c: float,
    n1: float, n2: float,
) -> np.ndarray:
    """Support point ``argmax_{x in body} x . d`` (local frame), via ``grad h``.

    For the block branch (``n1, n2 > 40``) the box corner is returned, with the
    ``d_i = 0`` tie broken towards ``+1`` -- the convention
    ``convex_contact.Box`` documents (a zero sign would put the point in the
    interior of a face and silently weaken GJK).
    """
    d = np.asarray(direction_local, dtype=float).reshape(3)
    if not np.any(d):
        raise ValueError("support direction must be non-zero")
    if n1 > _BLOCK_EXPONENT and n2 > _BLOCK_EXPONENT:
        signs = np.where(d >= 0.0, 1.0, -1.0)
        return signs * np.array([a, b, c], dtype=float)
    p = n2 / (n2 - 1.0)
    q = n1 / (n1 - 1.0)
    u = np.abs(d[0]) * a
    v = np.abs(d[1]) * b
    w = np.abs(d[2]) * c
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inner = (u ** p + v ** p) ** (1.0 / p)
        h = (inner ** q + w ** q) ** (1.0 / q)
        # grad_d h  (h is positively homogeneous of degree 1)
        common = h ** (1.0 - q)
        A_factor = common * inner ** (q - p) if inner > 0.0 else 0.0 * common
        gx = A_factor * u ** (p - 1.0) * a * np.sign(d[0])
        gy = A_factor * v ** (p - 1.0) * b * np.sign(d[1])
        gz = common * w ** (q - 1.0) * c * np.sign(d[2])
    return np.array([gx, gy, gz], dtype=float)


@dataclass(frozen=True)
class SuperquadricShape(ConvexShape):
    """Convex superquadric body usable by GJK/EPA.

    Args:
        center: World-frame centre ``c``.
        half_extents: ``(a, b, c)`` semi-axes (local frame).
        exponents: ``(n1, n2)``; ``n1 = n2 = 2`` is a sphere, both ``> 40``
            selects the exact block (max-norm) branch of the implicit function.
        orientation: ``(3, 3)`` rotation mapping the local frame to world.
    """

    center: Sequence[float]
    half_extents: Sequence[float]
    exponents: Sequence[float] = (2.0, 2.0)
    orientation: object = None

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float).reshape(3)
        ext = np.asarray(self.half_extents, dtype=float).reshape(3)
        expo = np.asarray(self.exponents, dtype=float).reshape(2)
        if np.any(ext <= 0.0) or not np.all(np.isfinite(ext)):
            raise ValueError("half_extents must be finite and positive")
        if np.any(expo < 1.0) or not np.all(np.isfinite(expo)):
            raise ValueError("exponents must be finite and >= 1")
        if self.orientation is None:
            rot = np.eye(3)
        else:
            rot = np.asarray(self.orientation, dtype=float).reshape(3, 3)
            if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-9):
                raise ValueError("orientation must be orthonormal")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "half_extents", ext)
        object.__setattr__(self, "exponents", expo)
        object.__setattr__(self, "orientation", rot)

    # -- support-function interface ----------------------------------------
    def support(self, direction: Sequence[float]) -> np.ndarray:
        d = np.asarray(direction, dtype=float).reshape(3)
        if not np.any(d):
            raise ValueError("support direction must be non-zero")
        local = self.orientation.T @ d
        point = support_point(local, *self.half_extents, *self.exponents)
        return self.center + self.orientation @ point

    def centroid(self) -> np.ndarray:
        return self.center

    def bounding_radius(self) -> float:
        """``max_{x in body} |x - c|``: the corner (vertex) distance.

        Corners are the eight combinations ``(+-a, +-b, +-c)`` for the block
        branch and the surface points along the same eight directions
        otherwise; both are evaluated and the larger is returned, which is an
        upper bound for every exponent (the radial distance ``s(u)`` peaks on a
        corner direction).
        """
        best = 0.0
        for signs in ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
                      (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)):
            u = np.array(signs, dtype=float) * np.asarray(self.half_extents)
            best = max(best, float(np.linalg.norm(
                support_point(u, *self.half_extents, *self.exponents))))
            best = max(best, float(_radius_along(
                (u / np.linalg.norm(u))[None, :],
                np.array([self.half_extents[0]]), np.array([self.half_extents[1]]),
                np.array([self.half_extents[2]]),
                np.array([self.exponents[0]]), np.array([self.exponents[1]]))[0]))
        for axis in np.eye(3):
            best = max(best, float(np.linalg.norm(
                support_point(axis, *self.half_extents, *self.exponents))))
        return float(best) * (1.0 + 1e-12)

    # -- convenience -------------------------------------------------------
    def support_point_distance(self, direction: Sequence[float]) -> float:
        """Centre-to-support-point distance along a world direction."""
        return float(np.linalg.norm(self.support(direction) - self.center))


def superquadric_shape(
    a: float, b: float, c: float, n1: float, n2: float,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    orientation=None,
) -> SuperquadricShape:
    """Shorthand constructor with the argument order of ``SuperquadricBatch``."""
    return SuperquadricShape(center, (a, b, c), (n1, n2), orientation)


def batch_to_shapes(body: SuperquadricBatch) -> list[SuperquadricShape]:
    """Convert a batched geometry (one entry per pair) into GJK/EPA shapes."""
    return [
        SuperquadricShape(body.center[i], (body.a[i], body.b[i], body.c[i]),
                          (body.n1[i], body.n2[i]), body.rot[i])
        for i in range(body.n_pairs)
    ]


def contact_superquadric_pair(
    body_a: SuperquadricBatch,
    body_b: SuperquadricBatch,
    pair: int = 0,
    *,
    rel_tol: float = 1e-13,
) -> ContactGeometry:
    """Reference contact (GJK gap / EPA depth) of one pair of a batched setup."""
    sa, sb = batch_to_shapes(body_a), batch_to_shapes(body_b)
    return _convex_contact(sa[pair], sb[pair], rel_tol=rel_tol)


# -- analytic limits ------------------------------------------------------
def analytic_sphere_limit(shape_a: SuperquadricShape, shape_b: SuperquadricShape) -> dict:
    """Exact contact of the ``n1 = n2 = 2`` limit (a sphere pair).

    Returns ``{"gap": signed clearance (negative = penetration), "depth": >= 0,
    "normal": unit vector from A to B}``.
    """
    for shape in (shape_a, shape_b):
        if not np.allclose(shape.exponents, 2.0):
            raise ValueError("sphere limit requires n1 = n2 = 2")
        if not np.allclose(shape.half_extents, shape.half_extents[0]):
            raise ValueError("sphere limit requires a = b = c")
    gap, normal, _pa, _pb = analytic_sphere_sphere(
        shape_a.center, shape_a.half_extents[0],
        shape_b.center, shape_b.half_extents[0],
    )
    return _oracle_dict(gap, normal)


def analytic_block_limit(shape_a: SuperquadricShape, shape_b: SuperquadricShape) -> dict:
    """Exact contact of the block limit (``n1, n2 > 40``), via separating axes."""
    for shape in (shape_a, shape_b):
        if not np.all(shape.exponents > _BLOCK_EXPONENT):
            raise ValueError("block limit requires n1, n2 > 40")
    gap, normal, _pa, _pb = analytic_box_box_sat(
        shape_a.center, shape_a.half_extents, shape_a.orientation,
        shape_b.center, shape_b.half_extents, shape_b.orientation,
    )
    return _oracle_dict(gap, normal)


def _oracle_dict(gap: float, normal: np.ndarray) -> dict:
    gap = float(gap)
    n = np.asarray(normal, dtype=float).reshape(3)
    return {"gap": gap, "depth": max(0.0, -gap), "normal": n / np.linalg.norm(n)}


def geometry_dict(geo: ContactGeometry) -> dict:
    """Normalise a :class:`ContactGeometry` to the oracle dict convention."""
    gap = -float(geo.depth) if not geo.separated else float(geo.distance)
    return {"gap": gap, "depth": float(geo.depth),
            "normal": np.asarray(geo.normal, dtype=float).reshape(3),
            "separated": bool(geo.separated)}


# ---------------------------------------------------------------------------
# Vectorised support + the direction-minimisation contact reference
# ---------------------------------------------------------------------------
def support_distance_many(dirs_local: np.ndarray, a: float, b: float, c: float,
                          n1: float, n2: float) -> np.ndarray:
    """Support function ``h(d)`` for an ``(N, 3)`` block of local directions."""
    d = np.atleast_2d(np.asarray(dirs_local, dtype=float))
    u = np.abs(d[:, 0]) * a
    v = np.abs(d[:, 1]) * b
    w = np.abs(d[:, 2]) * c
    if n1 > _BLOCK_EXPONENT and n2 > _BLOCK_EXPONENT:
        return u + v + w
    p = n2 / (n2 - 1.0)
    q = n1 / (n1 - 1.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inner = (u ** p + v ** p) ** (1.0 / p)
        return (inner ** q + w ** q) ** (1.0 / q)


def fibonacci_sphere(n: int) -> np.ndarray:
    """Deterministic, near-uniform ``(n, 3)`` direction set on the unit sphere."""
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


def support_height(shape: ConvexShape, normal: np.ndarray) -> float:
    """``h(n) = max_{x} x . n`` for one unit world direction."""
    return float(np.asarray(shape.support(normal)) @ normal)


def signed_gap_along(shape_a: ConvexShape, shape_b: ConvexShape, normal: np.ndarray) -> float:
    """``h_A(n) + h_B(-n)``: penetration depth (>= 0) or ``-distance`` along ``n``.

    For convex bodies the signed contact gap is

        ``g = min_{|n| = 1} [ h_A(n) + h_B(-n) ]``

    which is ``>= 0`` (and equal to the minimum translation depth) when the
    bodies overlap and ``= -distance`` when they are apart.  The minimising
    ``n`` is the contact normal, pointing from A to B.
    """
    n = np.asarray(normal, dtype=float).reshape(3)
    return float(shape_a.support(n) @ n + shape_b.support(-n) @ (-n))


def contact_by_direction_min(shape_a: ConvexShape, shape_b: ConvexShape,
                             n_dirs: int = 20000, refine: bool = True,
                             step_floor: float = 1e-10) -> dict:
    """Reference contact of a convex pair by minimising over directions.

    Independent of the fixed-point kernel *and* of GJK/EPA: it evaluates the
    definition ``g(n) = h_A(n) + h_B(-n)`` on a Fibonacci grid and then runs a
    deterministic shrinking pattern search in the tangent plane, so the answer
    depends only on the geometry.  ``n`` (grid size) sets how many separated
    contact directions can be missed; the pattern search removes the grid bias
    once the basin has been found.

    Returns ``{"gap", "depth", "normal", "grid_dirs", "refine_steps"}`` with
    ``gap < 0`` meaning a separation of ``-gap``.
    """
    dirs = fibonacci_sphere(int(n_dirs))
    if isinstance(shape_a, SuperquadricShape) and isinstance(shape_b, SuperquadricShape):
        # vectorised grid: h_A(n) + h_B(-n) for all directions at once
        locals_a = dirs @ shape_a.orientation
        locals_b = (-dirs) @ shape_b.orientation
        h_a = support_distance_many(locals_a, *shape_a.half_extents, *shape_a.exponents)
        h_b = support_distance_many(locals_b, *shape_b.half_extents, *shape_b.exponents)
        # support distance is measured from each body's own centre, so the
        # centre offset contributes (c_A - c_B) . n
        values = h_a + h_b + dirs @ (shape_a.center - shape_b.center)
    else:
        values = np.array([signed_gap_along(shape_a, shape_b, d) for d in dirs])
    best = int(np.argmin(values))
    n = dirs[best].copy()
    folds = 0
    if refine:
        current = float(values[best])
        t1, t2 = _tangent_basis(n)
        diag = 1.0 / math.sqrt(2.0)
        moves = (t1, -t1, t2, -t2,
                 diag * (t1 + t2), diag * (t1 - t2),
                 diag * (-t1 + t2), -diag * (t1 + t2))
        step = 0.2
        while step > step_floor:
            t1, t2 = _tangent_basis(n)
            moves = (t1, -t1, t2, -t2,
                     diag * (t1 + t2), diag * (t1 - t2),
                     diag * (-t1 + t2), -diag * (t1 + t2))
            best_dir, best_value = n, current
            for tangent in moves:
                candidate = _normalise_direction(n + step * tangent)
                value = signed_gap_along(shape_a, shape_b, candidate)
                if value < best_value - 1e-18:
                    best_dir, best_value = candidate, value
            if best_value < current - 1e-18:
                n, current = best_dir, best_value
                folds += 1
            else:
                step *= 0.5
    gap = float(signed_gap_along(shape_a, shape_b, n))
    return {"gap": gap, "depth": max(0.0, gap), "normal": n,
            "grid_dirs": int(n_dirs), "bins": folds}


def _tangent_basis(n: np.ndarray):
    """Orthonormal pair spanning the plane perpendicular to the unit vector ``n``."""
    seed = np.array([0.0, 0.0, 1.0]) if abs(float(n[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = _unit(np.cross(n, seed))
    t2 = np.cross(n, t1)
    return t1, t2


def _normalise_direction(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / float(np.linalg.norm(v))


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm <= 0.0:
        raise ValueError("cannot normalise a zero vector")
    return v / norm
