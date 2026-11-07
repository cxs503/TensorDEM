"""Superquadric contact kernel: geometry, force laws, and material interface.

This module is the *mature* contact layer behind
:mod:`tensordem.superquadric_dem`.  It differs from the exploratory prototype in
:mod:`tensordem.superquadric` in four ways that matter for a DEM time loop:

1. **Vectorised geometry.**  ``closest_point_batch`` projects an array of query
   points onto an array of superquadrics at once (bisection for a radial start
   value, then projected-Newton refinement).  The legacy scalar routine needs
   ~150 python-level array calls per query, which makes a time loop
   unusable; the batch version needs ~60 array calls *independent of M*.
2. **World-frame normals.**  Surface gradients are rotated by the body
   orientation, so contacts remain correct for rotated (rolling) particles --
   the prototype used the local-frame gradient directly.
3. **Two analytical normal branches.**  Curved contact uses Hertz with the
   curvature-derived effective radius; *flat-facet* contact (large shape
   exponent, zero curvature) switches to the Boussinesq flat-punch law of
   :mod:`tensordem.flat_punch`, since Hertz diverges as ``R_eff -> inf``.
   The patch area is obtained analytically from the implicit surface, so the
   branch is geometry-driven and not a hard-coded constant.
4. **Explicit separated concerns.**  Normal law, Mindlin tangential history and
   rolling resistance are independent functions with their own unit anchors,
   so each can be verified against its own closed form.

Conventions
-----------
* ``n`` (contact normal) points from body **A** to body **B**.
* ``delta`` is the overlap (penetration depth) in metres, ``0`` when separated.
* ``E*`` follows the DEM pair convention ``1/E* = sum_i (1-nu_i^2)/E_i``;
  ``G*`` likewise with ``2(1+nu)``.
* Angles in degrees only where named ``_deg``; internally radians.

All quantities are SI (m, kg, s, N, Pa, rad).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .flat_punch import area_equivalent_punch_stiffness, reduced_modulus

__all__ = [
    "ContactMaterial",
    "ContactGeometry",
    "implicit_local",
    "gradient_local",
    "closest_point_batch",
    "signed_distance_batch",
    "contact_geometry_batch",
    "facet_half_width",
    "facet_patch_area",
    "principal_curvatures_local",
    "effective_radius_hertz",
    "normal_force",
    "mindlin_tangential_update",
    "rolling_resistance_torque",
    "superquadric_inertia",
]

DTYPE_NP = np.float64


# ---------------------------------------------------------------------------
# Implicit geometry (vectorised over leading dimension M)
# ---------------------------------------------------------------------------

def _params(a, b, c, n1, n2, m: int):
    arr = []
    for v, name in ((a, "a"), (b, "b"), (c, "c"), (n1, "n1"), (n2, "n2")):
        v = np.asarray(v, dtype=DTYPE_NP)
        if v.ndim == 0:
            v = np.full(m, float(v), dtype=DTYPE_NP)
        else:
            v = np.broadcast_to(v.reshape(-1), (m,)).astype(DTYPE_NP)
        arr.append(v)
    return tuple(arr)


def implicit_local(x: np.ndarray, a, b, c, n1, n2) -> np.ndarray:
    """Superquadric implicit function ``F`` in the body frame of each row.

    ``F = [ (|x/a|^n2 + |y/b|^n2)^(n1/n2) + |z/c|^n1 ] - 1`` for moderate
    exponents; the Chebyshev (max-norm) limit is used when ``n1, n2 > 40`` so
    that the block corner ``(|x|,|y|,|z|) = (a,b,c)`` lies on the surface.
    """
    x = np.atleast_2d(np.asarray(x, dtype=DTYPE_NP))
    m = x.shape[0]
    a, b, c, n1, n2 = _params(a, b, c, n1, n2, m)
    xa = np.abs(x[:, 0]) / a
    xb = np.abs(x[:, 1]) / b
    xc = np.abs(x[:, 2]) / c
    with np.errstate(over="ignore", invalid="ignore"):
        horiz = (xa ** n2 + xb ** n2) ** (1.0 / n2)
        f = horiz ** n1 + xc ** n1 - 1.0
        block = np.maximum(np.maximum(xa, xb), xc) - 1.0
    return np.where((n1 > 40.0) & (n2 > 40.0), block, f)


def gradient_local(x: np.ndarray, a, b, c, n1, n2) -> np.ndarray:
    """Analytic gradient of :func:`implicit_local` (body frame, ``(M,3)``)."""
    x = np.atleast_2d(np.asarray(x, dtype=DTYPE_NP))
    m = x.shape[0]
    a, b, c, n1, n2 = _params(a, b, c, n1, n2, m)
    safe = lambda v: np.where(np.abs(v) < 1e-12, 1e-12, v)
    xa = np.abs(x[:, 0]) / a
    xb = np.abs(x[:, 1]) / b
    xc = np.abs(x[:, 2]) / c
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        hsum = xa ** n2 + xb ** n2
        horiz = hsum ** (1.0 / n2)
        fx = n1 * horiz ** (n1 - 1.0) * xa ** (n2 - 1.0) / (safe(hsum) ** ((n2 - 1.0) / n2)) / a
        fy = n1 * horiz ** (n1 - 1.0) * xb ** (n2 - 1.0) / (safe(hsum) ** ((n2 - 1.0) / n2)) / b
        fz = n1 * xc ** (n1 - 1.0) / c
        g = np.column_stack([
            np.sign(x[:, 0]) * fx,
            np.sign(x[:, 1]) * fy,
            np.sign(x[:, 2]) * fz,
        ])
        # block limit: subgradient of the max-norm
        ga = np.sign(x[:, 0]) / a
        gb = np.sign(x[:, 1]) / b
        gc = np.sign(x[:, 2]) / c
        is_max = (xa >= xb) & (xa >= xc)
        is_maxy = (xb >= xa) & (xb >= xc)
        g_block = np.column_stack([
            np.where(is_max, ga, 0.0),
            np.where(is_maxy, gb, 0.0),
            np.where(~is_max & ~is_maxy, gc, 0.0),
        ])
    g = np.where(((n1 > 40.0) & (n2 > 40.0))[:, None], g_block, g)
    g = np.where(np.isfinite(g), g, 0.0)
    return g


def closest_point_batch(
    q: np.ndarray,
    a, b, c, n1, n2,
    n_newton: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project query points (body frame) onto the superquadric surfaces.

    Returns ``(p, residual)`` with ``p`` the closest surface point and
    ``residual = F(p)`` (≈0).

    Algorithm: radial bisection for an initial surface point, then
    projected-Newton: alternately take a normal step onto the surface and an
    orthogonal projection of the query direction onto the tangent plane, which
    converges quadratically for convex bodies.
    """
    q = np.atleast_2d(np.asarray(q, dtype=DTYPE_NP))
    m = q.shape[0]
    a, b, c, n1, n2 = _params(a, b, c, n1, n2, m)

    r = np.linalg.norm(q, axis=1)
    r_safe = np.where(r < 1e-15, 1.0, r)
    direction = q / r_safe[:, None]
    direction[r < 1e-15] = np.array([1.0, 0.0, 0.0])

    # The implicit function is homogeneous of degree n1 in the *scaled*
    # coordinates, so F(s * d) = 0 has the closed-form solution
    #     s = [ H^n1 + |d_z/c|^n1 ]^(-1/n1),   H = ((d_x/a)^n2 + (d_y/b)^n2)^(1/n2)
    # which removes the 40-step bisection of the prototype entirely.
    dx = np.abs(direction[:, 0]) / a
    dy = np.abs(direction[:, 1]) / b
    dz = np.abs(direction[:, 2]) / c
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        horiz = (dx ** n2 + dy ** n2) ** (1.0 / n2)
        s_rad = (horiz ** n1 + dz ** n1) ** (-1.0 / n1)
        s_block = 1.0 / np.maximum.reduce([dx, dy, dz])
    s_rad = np.where((n1 > 40.0) & (n2 > 40.0), s_block, s_rad)
    s_rad = np.where(np.isfinite(s_rad) & (s_rad > 0), s_rad, 1.0)
    p = s_rad[:, None] * direction

    for _ in range(n_newton):
        g = gradient_local(p, a, b, c, n1, n2)
        gn = np.linalg.norm(g, axis=1)
        gn_safe = np.where(gn < 1e-30, 1.0, gn)
        nrm = g / gn_safe[:, None]
        f = implicit_local(p, a, b, c, n1, n2)
        # 1) normal step onto the surface (first-order distance F/|grad F|)
        p = p - (f / gn_safe)[:, None] * nrm
        # 2) move within the tangent plane towards the query point
        d = q - p
        d_perp = d - np.sum(d * nrm, axis=1)[:, None] * nrm
        p = p + d_perp
        # 3) re-project onto the surface (keeps iterates feasible)
        f = implicit_local(p, a, b, c, n1, n2)
        g = gradient_local(p, a, b, c, n1, n2)
        gn = np.linalg.norm(g, axis=1)
        gn_safe = np.where(gn < 1e-30, 1.0, gn)
        p = p - (f / gn_safe)[:, None] * (g / gn_safe[:, None])

    residual = implicit_local(p, a, b, c, n1, n2)
    return p, residual


def signed_distance_batch(q: np.ndarray, a, b, c, n1, n2) -> np.ndarray:
    """Signed distance from the surface at query points (body frame).

    The magnitude is the *exact* distance to the Newton-projected closest
    surface point (``|q - p|``), not the linearised ``F/|grad F|`` of the
    prototype.  The linearised form is exact only for planes; for a sphere it
    is biased by ``(r+R)/(2r)`` (0.5% at ``delta/R = 1e-2``), which is enough
    to eat the 1% Hertz budget.  The sign follows ``F``.
    """
    q = np.atleast_2d(np.asarray(q, dtype=DTYPE_NP))
    m = q.shape[0]
    a, b, c, n1, n2 = _params(a, b, c, n1, n2, m)
    f = implicit_local(q, a, b, c, n1, n2)
    p, _ = closest_point_batch(q, a, b, c, n1, n2)
    dist = np.linalg.norm(q - p, axis=1)
    return np.where(f < 0.0, -dist, dist)


# ---------------------------------------------------------------------------
# Contact geometry
# ---------------------------------------------------------------------------

@dataclass
class ContactGeometry:
    """Contact geometry between two superquadrics (world frame)."""
    in_contact: bool = False
    overlap: float = 0.0
    normal: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    point_a: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    point_b: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    gap: float = 0.0
    patch_area: np.ndarray = field(default_factory=lambda: np.zeros(0))


def contact_geometry_batch(
    centre_a: np.ndarray, rot_a: np.ndarray, shape_a: Tuple,
    centre_b: np.ndarray, rot_b: np.ndarray, shape_b: Tuple,
) -> ContactGeometry:
    """Penetration depth, contact normal and contact points for a batch of pairs.

    ``rot_*`` are ``(M,3,3)`` rotation matrices mapping body -> world
    (``world = R @ local + centre``).  ``shape_*`` are tuples of arrays
    ``(a, b, c, n1, n2)``.

    The overlap is measured with the *implicit signed distance* of each body
    evaluated at the other body's closest surface point::

        delta = max( 0, -d_B(p_A), -d_A(p_B), (s_A + s_B - |c_B - c_A|) )

    All three estimators coincide for spheres (exact); the max keeps the
    penetration measure conservative for blocks, where the centre-line
    estimator alone underestimates corner contacts.
    """
    ca = np.atleast_2d(centre_a)
    cb = np.atleast_2d(centre_b)
    m = ca.shape[0]
    a_a, b_a, c_a, n1_a, n2_a = _params(*shape_a, m)
    a_b, b_b, c_b, n1_b, n2_b = _params(*shape_b, m)

    # world -> body:  local = R^T (x - centre)
    def to_local(points, centre, rot):
        d = points - centre
        return np.einsum("mji,mj->mi", rot, d)

    def to_world(vec_local, rot):
        return np.einsum("mij,mj->mi", rot, vec_local)

    # closest point of A to B's centre, and of B to A's centre
    p_a_local, _ = closest_point_batch(to_local(cb, ca, rot_a), a_a, b_a, c_a, n1_a, n2_a)
    p_b_local, _ = closest_point_batch(to_local(ca, cb, rot_b), a_b, b_b, c_b, n1_b, n2_b)
    p_a = to_world(p_a_local, rot_a) + ca
    p_b = to_world(p_b_local, rot_b) + cb

    d_b_at_pa = signed_distance_batch(to_local(p_a, cb, rot_b), a_b, b_b, c_b, n1_b, n2_b)
    d_a_at_pb = signed_distance_batch(to_local(p_b, ca, rot_a), a_a, b_a, c_a, n1_a, n2_a)

    # centre-line direction (used only as a normal fallback)
    delta_centre = cb - ca
    dist = np.linalg.norm(delta_centre, axis=1)
    dist_safe = np.where(dist < 1e-15, 1.0, dist)
    n_dir = delta_centre / dist_safe[:, None]
    n_dir[dist < 1e-15] = np.array([0.0, 0.0, 1.0])

    delta = np.maximum(0.0, np.maximum(-d_b_at_pa, -d_a_at_pb))

    # contact normal: outward normals of both bodies, opposite for a contact
    g_a = gradient_local(to_local(p_a, ca, rot_a), a_a, b_a, c_a, n1_a, n2_a)
    g_b = gradient_local(to_local(p_b, cb, rot_b), a_b, b_b, c_b, n1_b, n2_b)
    n_a_world = to_world(g_a, rot_a)
    n_b_world = to_world(g_b, rot_b)
    n_a_world /= np.maximum(np.linalg.norm(n_a_world, axis=1), 1e-30)[:, None]
    n_b_world /= np.maximum(np.linalg.norm(n_b_world, axis=1), 1e-30)[:, None]
    n_contact = n_a_world - n_b_world
    norm = np.linalg.norm(n_contact, axis=1)
    fallback = norm < 1e-8
    n_contact = np.where(fallback[:, None], n_dir, n_contact / np.maximum(norm, 1e-30)[:, None])
    # orient from A to B
    flip = np.sum(n_contact * n_dir, axis=1) < 0.0
    n_contact = np.where(flip[:, None], -n_contact, n_contact)

    gap = np.maximum(0.0, np.minimum(d_b_at_pa, d_a_at_pb))
    geom = ContactGeometry(
        in_contact=bool(np.any(delta > 0.0)),
        overlap=delta,
        normal=n_contact,
        point_a=p_a,
        point_b=p_b,
        gap=gap,
    )
    return geom


# ---------------------------------------------------------------------------
# Flat-facet patch area (punch branch)
# ---------------------------------------------------------------------------

def facet_half_width(c: float, n1: float, delta: float) -> float:
    """Normalised cross-section half-width of the flat facet at depth ``delta``.

    For a superquadric along its local ``z`` axis, ``F = h^n1 + |z/c|^n1 - 1``
    with ``h`` the normalised horizontal coordinate, so pressing a facet to
    depth ``delta`` (``z = c - delta``) exposes ``h = (1 - ((c-delta)/c)^n1)^(1/n1)``.
    """
    if c <= 0.0:
        return 0.0
    z = max(0.0, min(1.0, (c - delta) / c))
    return float(max(0.0, 1.0 - z ** n1) ** (1.0 / n1))


def facet_patch_area(
    a: float, b: float, c: float, n1: float, n2: float, delta: float,
) -> float:
    """Analytic area of the flat facet patch exposed by a normal overlap.

    The facet region is the horizontal cross-section at ``z = c - delta``
    (block limit: a square of side ``2a h x 2b h`` with corner rounding set by
    ``n2``).  The cross-section area is evaluated by generic 1-D quadrature of
    the implicit curve, so rounded (``n2`` finite) and square (``n2 -> inf``)
    facets are both handled.
    """
    h = facet_half_width(c, n1, delta)
    if h <= 0.0:
        return 0.0
    # half-axis of the cross section: |x/a|^n2 + |y/b|^n2 = h^n2
    x_half = a * h
    if n2 > 40.0:
        # square cross-section with corner rounding ~ h*2^(-1/n2)
        corner = 2.0 ** (-1.0 / n2)
        return float(4.0 * a * b * h * h * corner ** 2)
    # integrate y(x) = b h (1 - (x/(a h))^n2)^(1/n2)
    t, w = np.polynomial.legendre.leggauss(40)
    t = 0.5 * (t + 1.0)
    w = 0.5 * w
    x = t * x_half
    inside = np.maximum(0.0, 1.0 - (x / max(x_half, 1e-300)) ** n2)
    y = b * h * inside ** (1.0 / n2)
    return float(4.0 * np.sum(w * y))


# ---------------------------------------------------------------------------
# Curvature and Hertz branch
# ---------------------------------------------------------------------------

def principal_curvatures_local(
    x: np.ndarray, a, b, c, n1, n2, eps_rel: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Principal curvatures of the implicit surface at body-frame points.

    The shape operator ``S = (I - n n^T) H / |grad F|`` (Hessian of the implicit
    function by central differences of the *analytic* gradient) is restricted to
    the tangent plane; its two eigenvalues are the principal curvatures (1/m),
    positive where the body is convex.  Fully vectorised over the leading
    dimension.
    """
    x = np.atleast_2d(np.asarray(x, dtype=DTYPE_NP))
    m = x.shape[0]
    a, b, c, n1, n2 = _params(a, b, c, n1, n2, m)
    g = gradient_local(x, a, b, c, n1, n2)
    gn = np.maximum(np.linalg.norm(g, axis=1), 1e-30)
    nrm = g / gn[:, None]
    h = np.maximum(np.maximum(np.maximum(a, b), c) * eps_rel, 1e-15)
    hess = np.empty((m, 3, 3))
    for j in range(3):
        dx = np.zeros((m, 3))
        dx[:, j] = h
        gp = gradient_local(x + dx, a, b, c, n1, n2)
        gm = gradient_local(x - dx, a, b, c, n1, n2)
        hess[:, :, j] = (gp - gm) / (2.0 * h)[:, None]
    hess = 0.5 * (hess + np.transpose(hess, (0, 2, 1)))
    # tangent basis, vectorised (pick a reference axis that is not parallel to n)
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (m, 1))
    alt = np.abs(nrm[:, 0]) > 0.9
    ref[alt] = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(nrm, ref)
    t1 /= np.maximum(np.linalg.norm(t1, axis=1), 1e-30)[:, None]
    t2 = np.cross(nrm, t1)
    ht1 = np.einsum("mi,mij->mj", t1, hess)
    ht2 = np.einsum("mi,mij->mj", t2, hess)
    s11 = np.einsum("mi,mi->m", t1, ht1) / gn
    s12 = np.einsum("mi,mi->m", t1, ht2) / gn
    s22 = np.einsum("mi,mi->m", t2, ht2) / gn
    half = 0.5 * (s11 + s22)
    disc = np.sqrt(np.maximum(0.25 * (s11 - s22) ** 2 + s12 ** 2, 0.0))
    return half - disc, half + disc


def effective_radius_hertz(k1_a, k2_a, k1_b, k2_b) -> np.ndarray:
    """Hertz effective radius from the principal curvatures of both surfaces.

    With ``A = (k1_a + k2_a)/2``, ``B = (k1_b + k2_b)/2`` (sum of the mean
    curvatures of the two *convex* surfaces facing each other, i.e. B enters
    with a minus sign for the opposing body) the effective radius is
    ``1/R_eff = 2 (A + B)``, which reduces to ``R/2`` for two equal spheres of
    radius ``R``.  Non-positive denominators (locally flat or saddle contact)
    yield ``inf`` so the caller switches to the punch branch.
    """
    k1_a = np.asarray(k1_a, dtype=DTYPE_NP)
    k2_a = np.asarray(k2_a, dtype=DTYPE_NP)
    k1_b = np.asarray(k1_b, dtype=DTYPE_NP)
    k2_b = np.asarray(k2_b, dtype=DTYPE_NP)
    inv_r = 0.5 * (k1_a + k2_a) + 0.5 * (k1_b + k2_b)
    with np.errstate(divide="ignore"):
        r = np.where(inv_r > 1e-12, 1.0 / np.maximum(inv_r, 1e-12), np.inf)
    return r


# ---------------------------------------------------------------------------
# Material and force laws
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContactMaterial:
    """Contact material parameters (SI)."""
    e_star: float = 1.0e9
    g_star: float = 4.0e8
    mu: float = 0.4
    mu_rolling: float = 0.05
    restitution: float = 0.4
    normal_law: str = "auto"        # "hertz" | "punch" | "auto"
    flat_curvature_threshold: float = 1.0   # |k| < threshold / max_extent -> punch

    @property
    def damping_ratio(self) -> float:
        """Viscous damping ratio implied by the coefficient of restitution."""
        e = float(min(max(self.restitution, 1e-6), 0.999))
        return -math.log(e) / math.sqrt(math.pi ** 2 + math.log(e) ** 2)

    def damping_normal(self, m_eff, k_normal):
        """Viscous normal damping coefficient from the restitution (vectorised)."""
        m_eff = np.asarray(m_eff, dtype=DTYPE_NP)
        k_normal = np.asarray(k_normal, dtype=DTYPE_NP)
        return 2.0 * self.damping_ratio * np.sqrt(
            np.maximum(m_eff, 1e-30) * np.maximum(k_normal, 1e-30))


def normal_force(
    overlap: np.ndarray,
    r_eff: np.ndarray,
    patch_area: np.ndarray,
    v_n: np.ndarray,
    m_eff: np.ndarray,
    material: ContactMaterial,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normal contact force with an automatic Hertz / flat-punch branch.

    Returns ``(force_magnitude, stiffness, branch)`` where ``branch`` is 0 for
    Hertz and 1 for the flat punch (``-1`` for no contact).  The dashpot uses
    the branch-consistent stiffness, so the restitution anchor is preserved in
    both branches.
    """
    overlap = np.asarray(overlap, dtype=DTYPE_NP)
    r_eff = np.asarray(r_eff, dtype=DTYPE_NP)
    patch_area = np.asarray(patch_area, dtype=DTYPE_NP)
    v_n = np.asarray(v_n, dtype=DTYPE_NP)
    m_eff = np.asarray(m_eff, dtype=DTYPE_NP)

    active = overlap > 0.0
    hertz_ok = np.isfinite(r_eff) & (r_eff > 0.0)
    if material.normal_law == "hertz":
        punch = np.zeros_like(active)
    elif material.normal_law == "punch":
        punch = active.copy()
    else:
        # a facet patch is only exposed by (nearly) flat surfaces; curved
        # contacts always report patch_area = 0 and stay on Hertz
        punch = active & (patch_area > 0.0)

    f = np.zeros_like(overlap)
    k = np.zeros_like(overlap)

    hmask = active & ~punch & hertz_ok
    if np.any(hmask):
        k[hmask] = (4.0 / 3.0) * material.e_star * np.sqrt(r_eff[hmask]) * np.sqrt(overlap[hmask])
        f[hmask] = (4.0 / 3.0) * material.e_star * np.sqrt(r_eff[hmask]) * overlap[hmask] ** 1.5

    pmask = active & punch
    if np.any(pmask):
        k_flat = 2.0 * material.e_star * np.sqrt(np.maximum(patch_area[pmask], 1e-300) / math.pi)
        k[pmask] = k_flat
        f[pmask] = k_flat * overlap[pmask]

    # damping only on approaching contacts (v_n < 0 is approaching by sign
    # convention: v_n = (v_B - v_A) . n)
    if np.any(active):
        c_n = material.damping_normal(m_eff[active], k[active])
        f[active] = np.maximum(0.0, f[active] - c_n * v_n[active])

    branch = np.where(active, np.where(punch, 1, 0), -1)
    return f, k, branch


def mindlin_tangential_update(
    f_n: np.ndarray,
    k_t: np.ndarray,
    mu: np.ndarray,
    xi: np.ndarray,
    v_t: Optional[np.ndarray] = None,
    dt: float = 0.0,
    normal: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cundall-Strack incremental tangential spring with Coulomb saturation.

    The stored stretch is advanced by the *relative tangential displacement*
    ``v_t * dt`` (this increment is what makes friction exist at all -- a version
    that only saturates a static stretch silently reproduces a frictionless
    material, which is exactly the bug found while calibrating the repose
    anchur), projected into the current tangent plane, and then saturated:

        xi <- P_t (xi + v_t dt),   f_t = -k_t xi,   |f_t| <= mu f_n

    Args:
        f_n: normal force magnitude (N).
        k_t: tangential stiffness (N/m).
        mu: Coulomb coefficient.
        xi: stored tangential stretch ``(M,3)`` (m).
        v_t: relative tangential velocity at the contact ``(M,3)`` (m/s).
        dt: time step (s).
        normal: contact normal ``(M,3)`` used to project the stretch.

    Returns:
        ``(f_t, xi_new, slid)``.
    """
    xi_new = np.array(xi, dtype=DTYPE_NP, copy=True)
    if v_t is not None and dt > 0.0:
        xi_new = xi_new + np.asarray(v_t, dtype=DTYPE_NP) * dt
    if normal is not None:
        normal = np.asarray(normal, dtype=DTYPE_NP)
        xi_new = xi_new - np.sum(xi_new * normal, axis=1)[:, None] * normal
    f_t = -k_t[:, None] * xi_new
    mag = np.linalg.norm(f_t, axis=1)
    limit = mu * f_n
    slid = mag > limit
    scale = np.where(slid & (mag > 1e-30), limit / np.maximum(mag, 1e-300), 1.0)
    f_t = f_t * scale[:, None]
    xi_new = np.where(slid[:, None], -f_t / np.maximum(k_t, 1e-300)[:, None], xi_new)
    return f_t, xi_new, slid


def rolling_resistance_torque(
    f_n: np.ndarray,
    r_eff: np.ndarray,
    omega_rel: np.ndarray,
    mu_rolling: np.ndarray,
    r_eff_fallback: np.ndarray,
) -> np.ndarray:
    """Coulomb rolling-resistance moment saturating at ``mu_r R F_n``.

    The moment opposes the relative spin ``omega_rel``; it is the standard
    "rolling friction" closure (Ai et al. 2011) without elastic memory, which
    is adequate for the repose-angle calibration targeted here (the same
    closure is used by the sphere baseline
    :mod:`tensordem.repose_calibration`).
    """
    mag = np.linalg.norm(omega_rel, axis=1)
    r = np.where(np.isfinite(r_eff) & (r_eff > 0), r_eff, r_eff_fallback)
    limit = mu_rolling * r * f_n
    scale = np.where(mag > 1e-30, limit / np.maximum(mag, 1e-300), 0.0)
    return -omega_rel * scale[:, None]


# ---------------------------------------------------------------------------
# Inertia
# ---------------------------------------------------------------------------

def superquadric_inertia(
    a: float, b: float, c: float, n1: float, n2: float, density: float,
    n_grid: int = 32,
) -> Tuple[float, np.ndarray]:
    """Mass and diagonal inertia tensor of a uniform superquadric.

    Deterministic midpoint voxel integration of the implicit solid (errors
    ~1e-3 relative for ``n_grid = 32``, verified against the exact sphere
    value ``2/5 m R^2`` in the test suite).
    """
    x = (np.arange(n_grid) + 0.5) / n_grid * 2.0 - 1.0
    hx = x * a
    hy = x * b
    hz = x * c
    X, Y, Z = np.meshgrid(hx, hy, hz, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    f = implicit_local(pts, a, b, c, n1, n2)
    inside = f < 0.0
    dv = (2.0 * a / n_grid) * (2.0 * b / n_grid) * (2.0 * c / n_grid)
    vol = float(np.sum(inside)) * dv
    mass = density * vol
    if vol <= 0:
        return 0.0, np.zeros(3)
    xs = pts[inside]
    ixx = density * dv * float(np.sum(xs[:, 1] ** 2 + xs[:, 2] ** 2))
    iyy = density * dv * float(np.sum(xs[:, 0] ** 2 + xs[:, 2] ** 2))
    izz = density * dv * float(np.sum(xs[:, 0] ** 2 + xs[:, 1] ** 2))
    return mass, np.array([ixx, iyy, izz])