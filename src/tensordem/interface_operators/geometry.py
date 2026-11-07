"""Exact geometric primitives for 2D/3D mortar interface operators.

Everything in this module is *exact up to floating point*: overlap measures of
polygonal cells come from convex clipping plus closed-form shoelace areas (no
quadrature noise), and the curved (spherical) area of a sphere/box intersection
comes from a semi-analytic stratification in which the inner integral is
analytic and only the outer (azimuthal) integral is quadratured.

Why this matters for the acceptance gates
-----------------------------------------
The mortar conservation residual of an exactly-partitioning overlap decomposition
is a *round-off* quantity (~1e-16).  If the overlap measures carried quadrature
error, the measured "residual" would just measure the quadrature and the gate
would be meaningless.  Hence analytic overlaps here, and Gauss quadrature only
where the integrand is genuinely smooth (field averages).

Conventions
-----------
* A "cell" is a convex polygon in ``dim``-space; for a 1D interface manifold in
  2D/3D the cell is a *segment* (two vertices) and polygon clipping degenerates
  to interval clipping along the interface tangent.
* Interface manifolds are described by an orthonormal tangent ``basis`` of shape
  ``(nb, dim)`` with ``nb = 1`` (curve) or ``nb = 2`` (surface); all clipping is
  done in those local coordinates, which keeps the code dimension-agnostic.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "TOL",
    "GL_X",
    "GL_W",
    "gl_rule",
    "signed_area",
    "polygon_area",
    "polygon_centroid",
    "clip_convex_polygon",
    "interval_clip",
    "cell_overlap",
    "plane_frame",
    "project_to_local",
    "sphere_box_area",
    "sphere_box_area_monte_carlo",
    "adaptive_gauss",
]

TOL = 1e-13

_GAUSS_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def gl_rule(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights on [-1, 1] (cached)."""
    if n not in _GAUSS_CACHE:
        _GAUSS_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _GAUSS_CACHE[n]


GL_X, GL_W = gl_rule(8)


# ---------------------------------------------------------------------------
# 2D polygon helpers
# ---------------------------------------------------------------------------
def signed_area(poly: np.ndarray) -> float:
    """Signed shoelace area of a 2D polygon (CCW > 0)."""
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(poly: np.ndarray) -> float:
    """Unsigned area of a 2D polygon."""
    return abs(signed_area(poly))


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    """Centroid of a 2D polygon; falls back to the vertex mean if degenerate."""
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return p.mean(axis=0)
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = 0.5 * cross.sum()
    if abs(a) < 1e-300:
        return p.mean(axis=0)
    cx = ((x + x1) * cross).sum() / (6.0 * a)
    cy = ((y + y1) * cross).sum() / (6.0 * a)
    return np.array([cx, cy])


def clip_convex_polygon(subject: np.ndarray, clipper: np.ndarray,
                        tol: float = 1e-14) -> np.ndarray:
    """Sutherland-Hodgman clipping of ``subject`` by a convex ``clipper``.

    Both polygons are 2D; the clipper may be given in either orientation.  The
    result is the vertex list of the intersection polygon (possibly empty).
    Edges are closed half-planes with a tolerance scaled by the polygon extent,
    so the shared edges of a *partition* are preserved rather than shaved off.
    """
    sub = np.asarray(subject, dtype=float)
    clip = np.asarray(clipper, dtype=float)
    if len(sub) < 3 or len(clip) < 3:
        return np.zeros((0, 2))
    orient = 1.0 if signed_area(clip) >= 0.0 else -1.0
    scale = max(float(np.ptp(clip, axis=0).max()), 1e-300)
    eps = tol * scale * scale
    out = sub
    m = len(clip)
    for k in range(m):
        if len(out) == 0:
            return np.zeros((0, 2))
        a = clip[k]
        e = clip[(k + 1) % m] - a
        d = orient * (e[0] * (out[:, 1] - a[1]) - e[1] * (out[:, 0] - a[0]))
        nxt = np.roll(out, -1, axis=0)
        dn = np.roll(d, -1)
        keep_p = d >= -eps
        keep_q = dn >= -eps
        new = []
        for i in range(len(out)):
            if keep_p[i]:
                new.append(out[i])
            if keep_p[i] != keep_q[i]:
                denom = d[i] - dn[i]
                if abs(denom) < 1e-300:
                    continue
                t = d[i] / denom
                new.append(out[i] + t * (nxt[i] - out[i]))
        out = np.asarray(new, dtype=float) if len(new) else np.zeros((0, 2))
    return out


def interval_clip(a: tuple[float, float] | np.ndarray,
                  b: tuple[float, float] | np.ndarray,
                  tol: float = 1e-14) -> tuple[float, float] | None:
    """Closed intersection of two 1D intervals; ``None`` if (numerically) empty."""
    lo = max(float(a[0]), float(b[0]))
    hi = min(float(a[1]), float(b[1]))
    if hi - lo <= tol * max(1.0, abs(hi), abs(lo)):
        return None
    return lo, hi


# ---------------------------------------------------------------------------
# dimension-agnostic cell overlap
# ---------------------------------------------------------------------------
def plane_frame(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best-fit frame of a vertex set: returns ``(origin, basis (nb, dim))``.

    ``nb = 2`` for a genuine surface patch, ``nb = 1`` for a collinear vertex
    set (a segment); the same clipping code then serves curves and surfaces.
    """
    v = np.asarray(verts, dtype=float)
    if len(v) < 3:
        raise ValueError("plane_frame needs >= 3 vertices")
    c = v.mean(axis=0)
    _, s, vt = np.linalg.svd(v - c, full_matrices=False)
    keep = s > 1e-12 * max(s[0], 1e-300)
    return c, vt[keep]


def project_to_local(poly: np.ndarray, origin: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project ambient points onto the local tangent basis."""
    p = np.asarray(poly, dtype=float)
    return (p - origin) @ basis.T


def _moment_local(poly_local: np.ndarray) -> tuple[float, np.ndarray]:
    """Measure and measure-weighted first moment of a 2D local polygon."""
    a = polygon_area(poly_local)
    if a <= 0.0:
        return 0.0, np.zeros(2)
    return a, a * polygon_centroid(poly_local)


def cell_overlap(poly_s: np.ndarray, poly_t: np.ndarray,
                 basis: np.ndarray, origin: np.ndarray,
                 tol: float = 1e-12) -> tuple[float, np.ndarray]:
    """Intersection measure of two convex cells and its first moment.

    Parameters
    ----------
    poly_s, poly_t
        Ambient vertex arrays of the two cells (``(nv, dim)``).  All vertices of
        both cells must lie in the interface manifold spanned by ``basis``.
    basis
        Orthonormal tangent basis ``(nb, dim)``, ``nb`` in ``{1, 2}``.
    origin
        Origin of the local frame.

    Returns
    -------
    (measure, moment)
        ``measure`` is the intersection length (``nb == 1``) or area
        (``nb == 2``); ``moment`` has shape ``(nb,)`` and equals
        ``int_{S_i & T_j} (x - c_i) dx`` with ``c_i`` the source-cell centroid
        projected onto the manifold.  That moment integrates to *zero* over an
        unclipped cell, which is exactly why the linear-reconstruction slope
        term of the mortar does not break conservation.
    """
    s_loc = project_to_local(poly_s, origin, basis)
    t_loc = project_to_local(poly_t, origin, basis)
    nb = basis.shape[0]
    if nb == 1:
        s = s_loc[:, 0]
        t = t_loc[:, 0]
        seg = interval_clip((s.min(), s.max()), (t.min(), t.max()), tol)
        if seg is None:
            return 0.0, np.zeros(1)
        lo, hi = seg
        cs = 0.5 * (s.min() + s.max())
        mom = 0.5 * (hi * hi - lo * lo) - cs * (hi - lo)   # int (x - cs) dx
        return hi - lo, np.array([mom])
    if nb == 2:
        clipped = clip_convex_polygon(t_loc, s_loc)
        if len(clipped) < 3:
            return 0.0, np.zeros(2)
        a, mom = _moment_local(clipped)
        cs = polygon_centroid(s_loc)
        return a, mom - a * cs
    raise ValueError(f"unsupported nb={nb}")


# ---------------------------------------------------------------------------
# adaptive quadrature
# ---------------------------------------------------------------------------
def adaptive_gauss(f, a: float, b: float, rtol: float = 1e-13,
                   atol: float = 1e-15, max_depth: int = 40) -> float:
    """Adaptive Gauss-Legendre (8 vs 16 nodes) with recursive bisection.

    The curved-clipping integrand is continuous but only piecewise smooth (its
    derivative jumps where a box plane becomes the active constraint), so a fixed
    composite rule would converge like O(h^2) across the kinks; adaptive
    bisection localises them and reaches round-off.
    """
    x8, w8 = gl_rule(8)
    x16, w16 = gl_rule(16)

    def rule(x, w, lo, hi):
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        return half * float(np.dot(w, f(mid + half * x)))

    def rec(lo, hi, depth):
        i16 = rule(x16, w16, lo, hi)
        i8 = rule(x8, w8, lo, hi)
        if abs(i16 - i8) <= rtol * abs(i16) + atol or depth >= max_depth:
            return i16
        mid = 0.5 * (lo + hi)
        return rec(lo, mid, depth + 1) + rec(mid, hi, depth + 1)

    return rec(a, b, 0)


# ---------------------------------------------------------------------------
# sphere / box surface-area clipping
# ---------------------------------------------------------------------------
def _interval_list_intersect(a: list[tuple[float, float]],
                             b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for lo1, hi1 in a:
        for lo2, hi2 in b:
            lo, hi = max(lo1, lo2), min(hi1, hi2)
            if hi > lo:
                out.append((lo, hi))
    return out


def _mu_constraint(value_lo: float, value_hi: float, direction: float,
                   tol: float = 1e-15) -> list[tuple[float, float]]:
    """Allowed ``mu = cos(theta)`` intervals for ``sqrt(1-mu^2) * direction``
    inside ``[value_lo, value_hi]`` (coordinates normalised by the radius)."""
    if abs(direction) < tol:
        return [(-1.0, 1.0)] if value_lo <= 0.0 <= value_hi else []
    a = value_lo / direction
    b = value_hi / direction
    lo, hi = (min(a, b), max(a, b))
    lo = max(lo, 0.0)
    hi = min(hi, 1.0)
    if hi < lo:
        return []
    mu2_lo = max(1.0 - hi * hi, 0.0)
    mu2_hi = min(1.0 - lo * lo, 1.0)
    if mu2_hi < mu2_lo:
        return []
    r_lo = float(np.sqrt(mu2_lo))
    r_hi = float(np.sqrt(mu2_hi))
    out = []
    if r_hi > r_lo:
        out.append((r_lo, r_hi))
        out.append((-r_hi, -r_lo))
    return out


def sphere_box_area(radius: float, box, rtol: float = 1e-13) -> float:
    """Area of the part of a sphere that lies inside an axis-aligned box.

    ``box`` is three ``(lo, hi)`` pairs (``+-inf`` allowed); ``rtol`` controls the
    outer (azimuthal) quadrature.

    The patch is ``{p : |p| = R, p in box}``.  With ``mu = cos(theta)`` the area
    element is ``R^2 dmu dphi`` and, for fixed azimuth, every box constraint is an
    *interval* constraint on ``mu`` only, so the inner integral is analytic and
    only the outer one is quadratured.  This is the curved-clipping reference used
    by the 3D gate: box planes cut the sphere along small circles, which no
    flat-facet mesh can represent, so a polygonal path converges here only O(h^2).
    """
    R = float(radius)
    if R <= 0.0:
        raise ValueError("radius must be positive")
    bounds = [(lo / R, hi / R) for lo, hi in box]
    (x1, x2), (y1, y2), (z1, z2) = bounds
    z1c = max(z1, -1.0)
    z2c = min(z2, 1.0)
    if z2c <= z1c:
        return 0.0
    z_int = [(z1c, z2c)]

    def integrand(phi_arr):
        phi_arr = np.atleast_1d(np.asarray(phi_arr, dtype=float))
        out = np.zeros_like(phi_arr)
        for k, ph in enumerate(phi_arr):
            c = float(np.cos(ph))
            s = float(np.sin(ph))
            allowed = _interval_list_intersect(
                z_int, _interval_list_intersect(
                    _mu_constraint(x1, x2, c), _mu_constraint(y1, y2, s)))
            out[k] = R * R * sum(hi - lo for lo, hi in allowed)
        return out

    total = 0.0
    for a, b in ((0.0, 0.5 * np.pi), (0.5 * np.pi, np.pi),
                 (np.pi, 1.5 * np.pi), (1.5 * np.pi, 2.0 * np.pi)):
        total += adaptive_gauss(integrand, a, b, rtol=rtol, atol=1e-16)
    return float(total)


def sphere_box_area_monte_carlo(radius: float, box, n: int = 2_000_000,
                                seed: int = 0) -> tuple[float, float]:
    """Monte-Carlo cross-check of :func:`sphere_box_area` (returns (area, stderr)).

    Samples the sphere uniformly and counts the fraction inside the box; used as
    an *independent* sanity check of the semi-analytic reference, not as a gate
    (its 1/sqrt(n) convergence cannot certify 1e-6).
    """
    R = float(radius)
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 3))
    p /= np.linalg.norm(p, axis=1)[:, None]
    inside = np.ones(n, dtype=bool)
    for k, (lo, hi) in enumerate(box):
        inside &= (R * p[:, k] >= lo) & (R * p[:, k] <= hi)
    frac = inside.mean()
    area = 4.0 * np.pi * R * R * frac
    stderr = 4.0 * np.pi * R * R * np.sqrt(max(frac * (1 - frac), 0.0) / n)
    return float(area), float(stderr)