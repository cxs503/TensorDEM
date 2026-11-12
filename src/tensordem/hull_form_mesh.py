"""Analytic ship-hull surface + triangulation + exact distance (verification aid).

Why this module exists
----------------------
:mod:`tensordem.hull_mesh_contact` takes a *triangulated* hull.  Whether the
contact forces it produces are right depends on two independent things: the
narrow-phase geometry (already verified on planes, single triangles and a
faceted cylinder, ``docs/dem3d_hull_mesh_contact_curved_audit.md``) and the
*digitisation* of a curved **ship hull** into facets, which was still open in
the gap ledger.  Scoring the second needs a hull whose surface is known in
closed form, so that the reference distance is independent of the kernel -- a
digitised offset table (SUBOFF etc.) is not.

The hull used here is the classical **Wigley** form (parabolic waterlines,
parabolic sections, vertical stem, keelline at ``z = -T``; the Gothenburg
1980 / Tokyo 1994 CFD-workshop validation hull):

    x(u) = (u - 1/2) L                          u in [0, 1]   (aft -> fore)
    z(v) = -T + v T                             v in [0, 1]   (keel -> waterline)
    y(u,v) = (B/2) (1 - (2u-1)^2) (2v - v^2)    y >= 0, mirrored to y < 0

The surface is *doubly curved* (waterline + section curvature) and closes at
the keel line and at both ends -- the geometric features a real hull
triangulation has and a cylinder does not.

What the module offers
----------------------
``wigley_half_breadth``      pointwise hull surface
``wigley_mesh``              inscribed triangulation (vertices exactly on the surface)
``wigley_distance``          distance from a point to the surface, by vectorised
                             coarse bracketing + local polish (see below)
``wigley_inside``            point-in-body test
``wigley_curvature``         principal curvatures, for the facet-size rule
                             ``chord <= sqrt(8 delta / kappa_max)``
``wigley_contact_reference`` analytic penalty force for a sphere vs the hull

The mesh is inscribed (its vertices lie on the analytic surface), so the mesh
distance is always ``>=`` the analytic distance and the penalty force is
*under*-predicted -- the conservative direction for ice-hull loads.  That bias,
not numerical noise, is what the audit measures.

Distance routine
----------------
The squared distance is a smooth 2-D function of ``(u, v)``; it can be
multimodal (a point near the keel can be nearest to either half-breadth side or
to the keel line itself), so a single local solve is not enough.  The routine
evaluates a vectorised coarse grid (default 241 x 121), brackets the global
minimum, then polishes with bounded L-BFGS-B + Nelder-Mead inside the
bracketing window.  ``tests/test_hull_form_mesh_contact.py`` and the audit
script check it against a brute-force dense grid over random points, so the
reference itself is falsifiable (the trap recorded in the curved-hull audit:
never take a hand-written reference on trust).

Evidence level: REAL (closed-form geometry, no fitting).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "WigleyHull",
    "wigley_directional_curvature",
    "wigley_half_breadth",
    "wigley_mesh",
    "wigley_distance",
    "wigley_inside",
    "wigley_curvature",
    "wigley_contact_reference",
]


@dataclass(frozen=True)
class WigleyHull:
    """Ship-form parameters.  Default = L/B = 10, B/T = 1.6 (Wigley ratios)."""

    length: float = 4.0
    beam: float = 0.40
    draft: float = 0.25

    def x_of(self, u):
        return (np.asarray(u, dtype=float) - 0.5) * self.length

    def z_of(self, v):
        return -self.draft + np.asarray(v, dtype=float) * self.draft

    def half_breadth(self, u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        return 0.5 * self.beam * (1.0 - (2.0 * u - 1.0) ** 2) * (2.0 * v - v * v)

    def surface(self, u, v) -> np.ndarray:
        """Point(s) on the ``y >= 0`` half of the surface, shape (..., 3)."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        return np.stack([self.x_of(u), self.half_breadth(u, v), self.z_of(v)], axis=-1)


def wigley_half_breadth(x, z, *, length=4.0, beam=0.40, draft=0.25):
    """Half-breadth at longitudinal ``x`` and height ``z`` (0 outside the box)."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    u = x / length + 0.5
    v = (z + draft) / draft
    y = 0.5 * beam * (1.0 - (2.0 * u - 1.0) ** 2) * (2.0 * v - v * v)
    inside_box = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    return np.where(inside_box, np.maximum(y, 0.0), 0.0)


def wigley_inside(point, hull: WigleyHull | None = None) -> bool:
    """True if ``point`` lies strictly inside the (solid) hull."""
    hull = hull or WigleyHull()
    x, y, z = (float(c) for c in point)
    if not (-0.5 * hull.length < x < 0.5 * hull.length):
        return False
    if not (-hull.draft < z < 0.0):
        return False
    hb = float(wigley_half_breadth(x, z, length=hull.length,
                                   beam=hull.beam, draft=hull.draft))
    return abs(y) < hb


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------

def wigley_mesh(n_u: int = 60, n_v: int = 24, hull: WigleyHull | None = None):
    """Inscribed triangulation of the full (port + starboard) wetted surface.

    Parameters
    ----------
    n_u, n_v : panels along x and z (``(n_u+1)(n_v+1)`` vertices per side).
        Vertices sit exactly on the analytic surface, so every triangle is a
        chord plane of the hull -- an inscribed digitisation, as from offsets.

    Returns
    -------
    vertices : (V, 3) float64 ndarray
    faces    : (F, 3) int64 ndarray, 0-based
    """
    hull = hull or WigleyHull()
    if n_u < 2 or n_v < 1:
        raise ValueError("need n_u >= 2 and n_v >= 1")
    u = np.linspace(0.0, 1.0, n_u + 1)
    v = np.linspace(0.0, 1.0, n_v + 1)
    UU, VV = np.meshgrid(u, v, indexing="ij")          # (n_u+1, n_v+1)
    pts = hull.surface(UU, VV)                         # (n_u+1, n_v+1, 3)
    side = pts.reshape(-1, 3)
    mirrored = side.copy()
    mirrored[:, 1] *= -1.0
    vertices = np.concatenate([side, mirrored], axis=0).astype(np.float64)

    def vid(i, j, off):
        return off + i * (n_v + 1) + j

    faces = []
    for off in (0, side.shape[0]):
        for i in range(n_u):
            for j in range(n_v):
                a, b = vid(i, j, off), vid(i + 1, j, off)
                c, d = vid(i, j + 1, off), vid(i + 1, j + 1, off)
                faces.append((a, b, d))
                faces.append((a, d, c))
    return vertices, np.asarray(faces, dtype=np.int64)


# ---------------------------------------------------------------------------
# Distance to the analytic surface
# ---------------------------------------------------------------------------

def _dist_sq_grid(p, hull: WigleyHull, u: np.ndarray, v: np.ndarray):
    """Vectorised squared distance from point ``p`` to the grid of surface points."""
    U, V = np.meshgrid(u, v, indexing="ij")
    S = hull.surface(U, V)
    d = S - p
    return U, V, (d * d).sum(axis=-1)


def wigley_distance(point, hull: WigleyHull | None = None, *,
                    n_coarse_u: int = 241, n_coarse_v: int = 121,
                    n_refine: int = 61):
    """Distance from ``point`` to the hull surface, with the foot point.

    Coarse vectorised bracket -> three shrinking-window dense refinements ->
    bounded L-BFGS-B polish.  Cost ~15 ms per query, accuracy at the level of
    the brute-force dense-grid minimum (checked in the tests and the audit).

    Returns ``(distance, foot_point, (u, v))``.
    """
    from scipy.optimize import minimize

    hull = hull or WigleyHull()
    raw = np.asarray(point, dtype=float).reshape(3)
    p = raw.copy()
    p[1] = abs(p[1])                  # symmetry: solve on the y >= 0 half

    ug = np.linspace(0.0, 1.0, n_coarse_u)
    vg = np.linspace(0.0, 1.0, n_coarse_v)
    _U, _V, D = _dist_sq_grid(p, hull, ug, vg)
    i, j = np.unravel_index(np.argmin(D), D.shape)

    du, dv = 1.0 / (n_coarse_u - 1), 1.0 / (n_coarse_v - 1)
    lo = np.array([max(ug[i] - 2 * du, 0.0), max(vg[j] - 2 * dv, 0.0)])
    hi = np.array([min(ug[i] + 2 * du, 1.0), min(vg[j] + 2 * dv, 1.0)])

    def g(w):
        s = hull.surface(float(w[0]), float(w[1]))
        return float(((s - p) ** 2).sum())

    best = (float(D[i, j]), (float(ug[i]), float(vg[j])))
    for _ in range(3):
        us = np.linspace(lo[0], hi[0], n_refine)
        vs = np.linspace(lo[1], hi[1], n_refine)
        _U2, _V2, D2 = _dist_sq_grid(p, hull, us, vs)
        a, b = np.unravel_index(np.argmin(D2), D2.shape)
        if D2[a, b] < best[0]:
            best = (float(D2[a, b]), (float(us[a]), float(vs[b])))
        span_u = (hi[0] - lo[0]) / (n_refine - 1) * 3.0
        span_v = (hi[1] - lo[1]) / (n_refine - 1) * 3.0
        lo = np.array([max(us[a] - span_u, 0.0), max(vs[b] - span_v, 0.0)])
        hi = np.array([min(us[a] + span_u, 1.0), min(vs[b] + span_v, 1.0)])

    res = minimize(g, x0=np.array(best[1]), method="L-BFGS-B",
                   bounds=list(zip(lo, hi)),
                   options={"ftol": 1e-18, "gtol": 1e-14, "maxiter": 400})
    fval = min(float(res.fun), best[0])
    uv = res.x if res.fun <= best[0] else np.array(best[1])

    d = math.sqrt(max(fval, 0.0))
    foot = hull.surface(float(uv[0]), float(uv[1]))
    if raw[1] < 0:
        foot = foot * np.array([1.0, -1.0, 1.0])
    return d, foot, (float(uv[0]), float(uv[1]))


# ---------------------------------------------------------------------------
# Curvature (facet-size rule)
# ---------------------------------------------------------------------------

def wigley_curvature(u, v, hull: WigleyHull | None = None, h: float = 1e-5):
    """Principal curvatures ``(k_max, k_min)`` of the ``y >= 0`` hull surface.

    From the first/second fundamental forms with central differences of the
    analytic parametrisation; used to state the facet-size rule
    ``chord <= sqrt(8 delta / kappa_max)``.  ``inf`` at the degenerate stem and
    keel lines, where no single facet can represent the crease.
    """
    hull = hull or WigleyHull()

    def X(uu, vv):
        return hull.surface(uu, vv)

    Xu = (X(u + h, v) - X(u - h, v)) / (2 * h)
    Xv = (X(u, v + h) - X(u, v - h)) / (2 * h)
    Xuu = (X(u + h, v) - 2 * X(u, v) + X(u - h, v)) / (h * h)
    Xvv = (X(u, v + h) - 2 * X(u, v) + X(u, v - h)) / (h * h)
    Xuv = (X(u + h, v + h) - X(u + h, v - h) - X(u - h, v + h) + X(u - h, v - h)) / (4 * h * h)

    n = np.cross(Xu, Xv)
    norm = np.linalg.norm(n)
    if norm < 1e-14:
        return math.inf, math.inf
    n = n / norm
    E = float(Xu @ Xu)
    F = float(Xu @ Xv)
    G = float(Xv @ Xv)
    L = float(n @ Xuu)
    M = float(n @ Xuv)
    N = float(n @ Xvv)
    denom = E * G - F * F
    if denom <= 1e-18:
        return math.inf, math.inf
    K = (L * N - M * M) / denom
    H = (E * N - 2 * F * M + G * L) / (2 * denom)
    root = math.sqrt(max(H * H - K, 0.0))
    return abs(H) + root, abs(abs(H) - root)


def wigley_directional_curvature(u, v, hull: WigleyHull | None = None, h: float = 1e-5):
    """Normal curvatures ``(k_u, k_v)`` along the two parameter directions.

    These are the curvatures a *rectangular* panel of the digitisation actually
    faces: ``k_u`` with the chord along ``u`` (waterline direction) and ``k_v``
    with the chord along ``v`` (section direction).  The facet offset (sagitta)
    of a panel is then, to second order,

        s = (k_u h_u^2 + k_v h_v^2) / 8 ,

    and the relative force error of the mesh kernel is ``s / penetration``
    (the analytic penalty force is ``k * pen`` while the mesh sees
    ``k * (pen - s)``).  Using ``k_max`` with the wrong chord over-predicts the
    error by an order of magnitude -- measured in the audit.
    """
    hull = hull or WigleyHull()
    X = lambda uu, vv: hull.surface(uu, vv)
    Xu = (X(u + h, v) - X(u - h, v)) / (2 * h)
    Xv = (X(u, v + h) - X(u, v - h)) / (2 * h)
    Xuu = (X(u + h, v) - 2 * X(u, v) + X(u - h, v)) / (h * h)
    Xvv = (X(u, v + h) - 2 * X(u, v) + X(u, v - h)) / (h * h)
    n = np.cross(Xu, Xv)
    nrm = np.linalg.norm(n)
    if nrm < 1e-14:
        return math.inf, math.inf
    n = n / nrm
    nu = float(np.linalg.norm(Xu)) ** 2
    nv = float(np.linalg.norm(Xv)) ** 2
    ku = abs(float(Xuu @ n)) / nu if nu > 1e-30 else math.inf
    kv = abs(float(Xvv @ n)) / nv if nv > 1e-30 else math.inf
    return ku, kv


# ---------------------------------------------------------------------------
# Analytic contact reference
# ---------------------------------------------------------------------------

def wigley_contact_reference(centres, radius, *, stiffness, hull: WigleyHull | None = None,
                             hull_position=None):
    """Penalty contact force from the *analytic* hull, not from a mesh.

    ``F = k (R - d) n`` with ``d`` the distance to the surface and ``n`` the
    unit vector from foot point to sphere centre.  Nothing from
    :mod:`tensordem.hull_mesh_contact` is used, so this is an independent
    reference for the mesh kernel.  Returns ``(forces (N,3), distances (N,))``.
    """
    hull = hull or WigleyHull()
    centres = np.atleast_2d(np.asarray(centres, dtype=float))
    shift = np.zeros(3) if hull_position is None else np.asarray(hull_position, dtype=float)
    forces = np.zeros_like(centres)
    dists = np.zeros(centres.shape[0])
    for i, c in enumerate(centres):
        local = c - shift
        d, foot, _ = wigley_distance(local, hull)
        dists[i] = d
        overlap = radius - d
        if overlap <= 0.0:
            continue
        delta = local - foot
        nrm = np.linalg.norm(delta)
        n = delta / nrm if nrm > 1e-15 else np.array([0.0, 0.0, 1.0])
        forces[i] = stiffness * overlap * n
    return forces, dists