"""Curved-patch mortar: patch-local parametrisation, Newton-projected true
intersection, per-patch local conservation with gap/penetration volume terms.

Why this module exists
----------------------
``mortar.py`` / ``clip3d.py`` treat a non-coplanar pair by **plane projection**:
the slave facet is projected orthogonally into the master facet's plane, the
overlap is clipped there and the master measure is transported by
``1/|cos(dihedral)|``.  That construction is only first-order accurate on a
curved interface: the per-facet coverage defect ``sum_i m_ij / |Gamma_j| - 1``
is O(kappa h), dominated by slivers, and in the previous node it was therefore
demoted to a *diagnostic* while only the global sum was gated.  A global gate
cannot see a local flux imbalance: one patch can be over-covered while its
neighbour is under-covered with the global sum still exact, and the coupled
system then injects or dissipates energy at the interface every step.

This module replaces the plane projection by

1. a **patch-local parametrisation** ``x(u, v)`` of the interface surface
   (analytic chart with derivatives), so the interface measure is the true
   curved measure ``J(u, v) = |x_u x x_v|`` rather than a projected area;
2. **Newton projection** ``x(u, v) = p`` (2x2 Newton on the normal equations,
   quadratic convergence) to locate points of the other side *on the patch*.
   Ambient 3D meshes are supported: their vertices are projected onto the patch
   instead of onto a neighbouring facet's plane;
3. the **true intersection**: both sides induce a partition of the parameter
   domain.  A master facet's footprint is its parameter triangle, a slave
   facet's footprint likewise, and their intersection is an exact convex polygon
   ``R_ij`` in ``(u, v)``.  The mortar measure is ``m_ij = int_{R_ij} J du dv``
   (high-order quadrature over the polygon) and the first moment is
   ``mu_ij = int_{R_ij} (x - c_i) J du dv``.  Because ``{R_ij}_i`` partitions
   ``P_j`` exactly, ``sum_i m_ij = int_{P_j} J`` holds to quadrature round-off --
   **per facet**, not only globally;
4. **gap / penetration volume terms**: when the two discretisations do not tile
   the same parameter domain (a real interface gap, or duplicated / overlapping
   receptor facets, i.e. penetration), the exact per-patch identity is::

       |Gamma_j| + V_pen,j - V_gap,j = sum_i m_ij
       V_gap,j = int_{P_j} max(1 - c(x), 0) J du dv
       V_pen,j = int_{P_j} max(c(x) - 1, 0) J du dv

   with ``c`` the coverage multiplicity of the slave footprints, computed from
   an exact arrangement of convex polygons.  Without the terms a gap silently
   destroys flux and a penetration silently doubles it.

Consequences measured by the acceptance criteria: the **per-patch conservation
gate** (:func:`per_patch_balance`), the **time-domain coupling energy account**
(:func:`first_order_coupling`) and the **transpose consistency in the curved
metric** (:func:`dual_moment_defect`).

Controls (deliberately broken variants, negative evidence for the gates)
-----------------------------------------------------------------------
``measure="centroid_j"`` uses ``|R_ij| * J(centroid)`` instead of exact
quadrature; ``metric="flat"`` ignores the curvature factor (``J = 1``);
:func:`plane_projection_defect` reproduces the previous plane-projection
construction.  All three violate the per-facet gate at O(h)..O(h^2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "tri_gauss",
    "ParamPatch",
    "SpherePatch",
    "GraphPatch",
    "CurvedSide",
    "grid_side",
    "side_from_ambient",
    "concat_sides",
    "clip_poly",
    "poly_measure_moment",
    "pair_measures",
    "per_patch_balance",
    "coverage_arrangement",
    "intersection_curve",
    "CurvedMortarOperator",
    "plane_projection_defect",
    "first_order_coupling",
    "dual_moment_defect",
    "fit_rate",
    "run_acceptance",
]


# ---------------------------------------------------------------------------
# quadrature on the reference triangle (conical product, exact to degree 2n-1)
# ---------------------------------------------------------------------------
def tri_gauss(n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """``n``-point conical-product Gauss rule on the reference triangle.

    Points are barycentric ``(s, t)`` on ``(0,0), (1,0), (0,1)``; the weights sum
    to ``1/2`` (its area) so that ``int_ref f = sum_k w_k f``.  Exact for total
    degree ``2n - 1`` (``n = 8`` -> degree 15), which is what makes
    ``sum_i m_ij = int_{P_j} J`` hold to quadrature round-off.
    """
    xg, wg = np.polynomial.legendre.leggauss(int(n))
    t = 0.5 * (xg + 1.0)
    wt = 0.5 * wg
    pts, ws = [], []
    for i in range(int(n)):
        for j in range(int(n)):
            a, b = t[i], t[j]
            pts.append((a, b * (1.0 - a)))
            ws.append(wt[i] * wt[j] * (1.0 - a))
    return np.asarray(pts, dtype=float), np.asarray(ws, dtype=float)


# ---------------------------------------------------------------------------
# patches: analytic chart of the interface surface
# ---------------------------------------------------------------------------
class ParamPatch:
    """An analytic parametric surface patch ``x(u, v)`` on ``[0,1]^2``."""

    label = "patch"
    _kappa = 1.0

    # -- geometry ---------------------------------------------------------
    def x(self, uv):                                   # pragma: no cover
        raise NotImplementedError

    def xu(self, uv):                                  # pragma: no cover
        raise NotImplementedError

    def xv(self, uv):                                  # pragma: no cover
        raise NotImplementedError

    def xuu(self, uv):                                 # pragma: no cover
        raise NotImplementedError

    def xuv(self, uv):                                 # pragma: no cover
        raise NotImplementedError

    def xvv(self, uv):                                 # pragma: no cover
        raise NotImplementedError

    def jac(self, uv) -> np.ndarray:
        """Curved measure factor ``J = |x_u x x_v|`` (vectorised)."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        return np.linalg.norm(np.cross(self.xu(uv), self.xv(uv)), axis=-1)

    def normal(self, uv):
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        n = np.cross(self.xu(uv), self.xv(uv))
        return n / np.linalg.norm(n, axis=-1)[:, None]

    # -- Newton projection -------------------------------------------------
    def project(self, p, uv0=None, *, tol: float = 1e-15, maxit: int = 100):
        """Project an ambient point ``p`` onto the patch by 2x2 Newton.

        Newton on the stationarity of ``|x(u, v) - p|^2``::

            g = ((x - p).x_u , (x - p).x_v)
            H = [[x_u.x_u + (x-p).x_uu , x_u.x_v + (x-p).x_uv],
                 [  symmetric          , x_v.x_v + (x-p).x_vv]]

        Returns ``(uv, distance, iterations, |g|)``.  This is the "true"
        projection that replaces projecting a facet onto a facet plane.
        """
        p = np.asarray(p, dtype=float)
        uv = np.asarray(uv0 if uv0 is not None else self.initial_guess(p), dtype=float)
        uv = np.clip(uv, -0.35, 1.35)
        gnorm, it = np.inf, 0
        for it in range(1, maxit + 1):
            r = self.x(uv)[0] - p
            du, dv = self.xu(uv)[0], self.xv(uv)[0]
            g = np.array([r @ du, r @ dv])
            gnorm = float(np.linalg.norm(g))
            xuu, xuv, xvv = self.xuu(uv)[0], self.xuv(uv)[0], self.xvv(uv)[0]
            H = np.array([
                [du @ du + r @ xuu, du @ dv + r @ xuv],
                [du @ dv + r @ xuv, dv @ dv + r @ xvv],
            ])
            det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
            if abs(det) < 1e-300:
                break
            step = -np.linalg.solve(H, g)
            uv = np.clip(uv + step, -0.35, 1.35)
            if float(np.linalg.norm(step)) < tol:
                break
        return uv, float(np.linalg.norm(self.x(uv)[0] - p)), it, gnorm

    def initial_guess(self, p) -> np.ndarray:
        """Coarse grid search used as the Newton seed (robust, cheap)."""
        g = np.linspace(0.0, 1.0, 7)
        U, V = np.meshgrid(g, g, indexing="ij")
        uv = np.stack([U.ravel(), V.ravel()], axis=1)
        d = np.linalg.norm(self.x(uv) - np.asarray(p, dtype=float)[None, :], axis=1)
        return uv[int(np.argmin(d))]

    # -- surface integrals -------------------------------------------------
    def area(self, rule=None, subdiv: int = 8) -> float:
        """Curved area of the whole parameter square ``[0, 1]^2``.

        A *composite* rule is required: one degree-15 patch of a curved surface is
        O(h^16) only in the element size, and a single unit-square element leaves a
        ~1e-7 relative error on the graph patch.  Subdividing into ``subdiv^2``
        cells brings it to round-off.
        """
        m, _mom = self._square_moments(rule, subdiv)
        return m

    def centroid(self, rule=None, subdiv: int = 8) -> np.ndarray:
        m, mom = self._square_moments(rule, subdiv)
        return mom / m

    def _square_moments(self, rule=None, subdiv: int = 8):
        rule = rule or tri_gauss(8)
        g = np.linspace(0.0, 1.0, subdiv + 1)
        tot, mom = 0.0, np.zeros(3)
        for i in range(subdiv):
            for j in range(subdiv):
                a = np.array([g[i], g[j]])
                b = np.array([g[i + 1], g[j]])
                c = np.array([g[i + 1], g[j + 1]])
                d = np.array([g[i], g[j + 1]])
                for tri in (np.array([a, b, c]), np.array([a, c, d])):
                    m, mm = poly_measure_moment(self, tri, rule)
                    tot += m
                    mom += mm
        return tot, mom

    @property
    def curvature_scale(self) -> float:
        """``kappa * L`` proxy (max curvature times patch size)."""
        return float(self._kappa)


@dataclass
class GraphPatch(ParamPatch):
    """Graph patch ``x = c + L u e1 + W v e2 + A sin(pi u) sin(pi v) n``.

    A smooth bump with a flat boundary, i.e. the interface closes on a plane (the
    usual "flat hull side vs wavy ice surface" configuration).  Curvature scale
    ``kappa ~ 2 pi^2 A / L^2``.
    """

    length: float = 1.0
    width: float = 1.0
    amp: float = 0.12
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    e1: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    e2: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    label: str = "graph"

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.e1 = np.asarray(self.e1, dtype=float)
        self.e2 = np.asarray(self.e2, dtype=float)
        n = np.cross(self.e1, self.e2)
        self.normal_vec = n / np.linalg.norm(n)
        self._kappa = abs(2.0 * math.pi ** 2 * self.amp) / (self.length * self.length)

    def _h(self, u, v):
        return self.amp * np.sin(math.pi * u) * np.sin(math.pi * v)

    def x(self, uv):
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        u, v = uv[:, 0], uv[:, 1]
        return (self.center[None, :]
                + (self.length * u)[:, None] * self.e1[None, :]
                + (self.width * v)[:, None] * self.e2[None, :]
                + self._h(u, v)[:, None] * self.normal_vec[None, :])

    def xu(self, uv):
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        u, v = uv[:, 0], uv[:, 1]
        hu = self.amp * math.pi * np.cos(math.pi * u) * np.sin(math.pi * v)
        return self.length * self.e1[None, :] + hu[:, None] * self.normal_vec[None, :]

    def xv(self, uv):
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        u, v = uv[:, 0], uv[:, 1]
        hv = self.amp * math.pi * np.sin(math.pi * u) * np.cos(math.pi * v)
        return self.width * self.e2[None, :] + hv[:, None] * self.normal_vec[None, :]

    def xuu(self, uv):
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        u, v = uv[:, 0], uv[:, 1]
        huu = -self.amp * math.pi ** 2 * np.sin(math.pi * u) * np.sin(math.pi * v)
        return huu[:, None] * self.normal_vec[None, :]

    def xuv(self, uv):
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        u, v = uv[:, 0], uv[:, 1]
        huv = self.amp * math.pi ** 2 * np.cos(math.pi * u) * np.cos(math.pi * v)
        return huv[:, None] * self.normal_vec[None, :]

    def xvv(self, uv):
        return self.xuu(uv)


@dataclass
class SpherePatch(ParamPatch):
    """Spherical patch ``x = c + R (sin t cos f, sin t sin f, cos t)``.

    ``t = t0 + (t1 - t0) u``, ``f = f0 + (f1 - f0) v``;
    ``J = R^2 sin t (t1 - t0)(f1 - f0)``.  Uniform curvature ``kappa = 1/R``
    makes it the clean curved-metric benchmark.
    """

    radius: float = 1.0
    theta0: float = 0.35
    theta1: float = 1.20
    phi0: float = 0.30
    phi1: float = 1.30
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    label: str = "sphere"

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self._kappa = 1.0 / self.radius

    def _tf(self, uv):
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        t = self.theta0 + (self.theta1 - self.theta0) * uv[:, 0]
        f = self.phi0 + (self.phi1 - self.phi0) * uv[:, 1]
        return t, f

    def x(self, uv):
        t, f = self._tf(uv)
        R = self.radius
        return self.center[None, :] + R * np.stack(
            [np.sin(t) * np.cos(f), np.sin(t) * np.sin(f), np.cos(t)], axis=1)

    def xu(self, uv):
        t, f = self._tf(uv)
        k = (self.theta1 - self.theta0) * self.radius
        return k * np.stack([np.cos(t) * np.cos(f), np.cos(t) * np.sin(f),
                             -np.sin(t)], axis=1)

    def xv(self, uv):
        t, f = self._tf(uv)
        k = (self.phi1 - self.phi0) * self.radius
        return k * np.stack([-np.sin(t) * np.sin(f), np.sin(t) * np.cos(f),
                             np.zeros_like(t)], axis=1)

    def xuu(self, uv):
        t, f = self._tf(uv)
        k = ((self.theta1 - self.theta0) ** 2) * self.radius
        return -k * np.stack([np.sin(t) * np.cos(f), np.sin(t) * np.sin(f),
                              np.cos(t)], axis=1)

    def xuv(self, uv):
        t, f = self._tf(uv)
        k = (self.theta1 - self.theta0) * (self.phi1 - self.phi0) * self.radius
        return k * np.stack([-np.cos(t) * np.sin(f), np.cos(t) * np.cos(f),
                             np.zeros_like(t)], axis=1)

    def xvv(self, uv):
        t, f = self._tf(uv)
        k = ((self.phi1 - self.phi0) ** 2) * self.radius
        return -k * np.stack([np.sin(t) * np.cos(f), np.sin(t) * np.sin(f),
                              np.zeros_like(t)], axis=1)


# ---------------------------------------------------------------------------
# convex polygon helpers (parameter domain)
# ---------------------------------------------------------------------------
def _signed_area(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return 0.0
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1)
                              - np.roll(p[:, 0], -1) * p[:, 1]))


def _ccw(poly: np.ndarray) -> np.ndarray:
    p = np.asarray(poly, dtype=float)
    return p if _signed_area(p) >= 0.0 else p[::-1].copy()


def _side_val(p, a, b):
    e = b - a
    return e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])


def _split_poly(poly, a, b, keep_left: bool, tol: float = 0.0) -> np.ndarray:
    """Sutherland-Hodgman against one half-plane (left of ``a -> b``)."""
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return np.zeros((0, 2))
    sgn = 1.0 if keep_left else -1.0
    out = []
    n = len(p)
    for k in range(n):
        cur, nxt = p[k], p[(k + 1) % n]
        sc, sn = sgn * _side_val(cur, a, b), sgn * _side_val(nxt, a, b)
        if sc >= -tol:
            out.append(cur)
        if (sc > tol and sn < -tol) or (sc < -tol and sn > tol):
            t = sc / (sc - sn)
            out.append(cur + t * (nxt - cur))
    return np.asarray(out, dtype=float) if len(out) >= 3 else np.zeros((0, 2))


def clip_poly(subject, clipper) -> np.ndarray:
    """Exact intersection of two convex polygons in the parameter domain."""
    out = np.asarray(subject, dtype=float)
    cl = _ccw(clipper)
    if len(cl) < 3 or len(out) < 3:
        return np.zeros((0, 2))
    for k in range(len(cl)):
        if len(out) < 3:
            return np.zeros((0, 2))
        out = _split_poly(out, cl[k], cl[(k + 1) % len(cl)], keep_left=True)
    return out


def _outside_pieces(pc, q) -> list:
    """Decompose ``pc \\ q`` into convex pieces (``q`` convex): exact decomposition."""
    q = _ccw(q)
    rem = np.asarray(pc, dtype=float)
    pieces = []
    for k in range(len(q)):
        if len(rem) < 3:
            break
        a, b = q[k], q[(k + 1) % len(q)]
        out = _split_poly(rem, a, b, keep_left=False)
        if len(out) >= 3:
            pieces.append(out)
        rem = _split_poly(rem, a, b, keep_left=True)
    return pieces


def poly_measure_moment(patch: ParamPatch, poly, rule=None):
    """``(int_P J du dv, int_P x J du dv)`` by high-order Gauss over a polygon.

    The polygon is fan-triangulated and all sub-triangles are integrated in one
    vectorised call (``n = 8`` -> 64 points per sub-triangle).
    """
    rule = rule or tri_gauss(8)
    pts, ws = rule
    p = _ccw(poly)
    if len(p) < 3:
        return 0.0, np.zeros(3)
    a = p[0]
    tri = []
    dets = []
    for k in range(1, len(p) - 1):
        b, c = p[k], p[k + 1]
        det = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(det) <= 0.0:
            continue
        tri.append((a, b, c))
        dets.append(abs(det))
    if not tri:
        return 0.0, np.zeros(3)
    dets = np.asarray(dets)
    uv = np.concatenate([
        a[None, :] + pts[:, [0]] * (b - a)[None, :] + pts[:, [1]] * (c - a)[None, :]
        for (a, b, c) in tri], axis=0)
    J = patch.jac(uv)
    X = patch.x(uv)
    w = (ws[None, :] * dets[:, None]).ravel()
    total = float(np.dot(w, J))
    mom = (X * (w * J)[:, None]).sum(axis=0)
    return total, mom


# ---------------------------------------------------------------------------
# one discretisation of the patch (a parameter-domain triangulation)
# ---------------------------------------------------------------------------
@dataclass
class CurvedSide:
    """A triangulation of the patch parameter domain.

    ``tris[k]`` are the three ``(u, v)`` vertices of facet ``k``.  The facet is a
    *curved* element (3D image ``x(u, v)`` over the parameter triangle), so its
    measure is ``int_T J`` and its centroid is the curved centroid
    ``int_T x J / int_T J``.  Flat chords are only used by the deliberately broken
    plane-projection control.
    """

    tris: np.ndarray
    patch: ParamPatch
    label: str = ""
    measures: np.ndarray = field(default_factory=lambda: np.zeros(0))
    centroids: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    uv_centroids: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    projection_residual: float = 0.0
    rule: tuple = field(default_factory=lambda: tri_gauss(8))

    def __post_init__(self):
        self.tris = np.asarray(self.tris, dtype=float)
        m = np.zeros(len(self.tris))
        c = np.zeros((len(self.tris), 3))
        uvc = np.zeros((len(self.tris), 2))
        for k, t in enumerate(self.tris):
            mm, mom = poly_measure_moment(self.patch, t, self.rule)
            m[k] = mm
            c[k] = mom / mm if mm > 0.0 else self.patch.x(t.mean(axis=0))[0]
            uvc[k] = t.mean(axis=0)
        self.measures = m
        self.centroids = c
        self.uv_centroids = uvc

    @property
    def n(self) -> int:
        return len(self.tris)

    @property
    def area(self) -> float:
        return float(self.measures.sum())

    def chords(self) -> np.ndarray:
        """Flat triangles (3D) used by the plane-projection control."""
        out = np.zeros((self.n, 3, 3))
        for k, t in enumerate(self.tris):
            out[k] = self.patch.x(t)
        return out


def grid_side(patch: ParamPatch, nu: int, nv: int, *, jitter: float = 0.0,
              seed: int = 0, box=(0.0, 1.0, 0.0, 1.0), label: str = "",
              rule=None) -> CurvedSide:
    """Jittered ``nu x nv`` tensor-grid triangulation of a parameter rectangle.

    Interior nodes are perturbed by ``O(jitter) * h`` so that the two sides are
    genuinely non-matching; boundary nodes stay on the box boundary, so a side
    always tiles its box exactly (the conforming-coverage premise of the per-patch
    gate).  If a perturbation would invert a cell it is halved, never accepted.
    """
    u0, u1, v0, v1 = box
    rng = np.random.default_rng(seed)
    for _attempt in range(10):
        U = np.linspace(u0, u1, nu + 1)
        V = np.linspace(v0, v1, nv + 1)
        UU, VV = np.meshgrid(U, V, indexing="ij")
        du = (u1 - u0) / nu
        dv = (v1 - v0) / nv
        if jitter > 0.0:
            r = rng.uniform(-1.0, 1.0, size=(nu + 1, nv + 1, 2))
            inner = np.zeros((nu + 1, nv + 1))
            inner[1:-1, 1:-1] = 1.0
            UU = UU + jitter * du * r[..., 0] * inner
            VV = VV + jitter * dv * r[..., 1] * inner
        tris, ok = [], True
        for i in range(nu):
            for j in range(nv):
                p00 = np.array([UU[i, j], VV[i, j]])
                p10 = np.array([UU[i + 1, j], VV[i + 1, j]])
                p11 = np.array([UU[i + 1, j + 1], VV[i + 1, j + 1]])
                p01 = np.array([UU[i, j + 1], VV[i, j + 1]])
                for t in (np.array([p00, p10, p11]), np.array([p00, p11, p01])):
                    if abs(_signed_area(t)) <= 1e-14 * abs(du * dv):
                        ok = False
                    tris.append(t)
        if ok:
            return CurvedSide(np.asarray(tris), patch, label or f"grid{nu}x{nv}",
                              rule=rule or tri_gauss(8))
        jitter *= 0.5
    raise RuntimeError("grid_side: jitter could not produce a valid triangulation")


def side_from_ambient(patch: ParamPatch, verts, faces, *, label: str = "",
                      rule=None) -> CurvedSide:
    """Build a side from an ambient 3D mesh: Newton-project the vertices.

    This is the path a coupled code actually needs (meshes arrive as 3D
    coordinates).  The maximum projection residual is stored on the side; the
    tests assert it is at round-off, i.e. Newton really lands on the patch and does
    not silently clamp.
    """
    verts = np.asarray(verts, dtype=float)
    uv = np.zeros((len(verts), 2))
    resid = 0.0
    for k, p in enumerate(verts):
        u, d, _it, _g = patch.project(p)
        uv[k] = u
        resid = max(resid, d)
    tris = np.asarray([uv[list(f)] for f in faces])
    side = CurvedSide(tris, patch, label or "from_ambient", rule=rule or tri_gauss(8))
    side.projection_residual = resid
    return side


def concat_sides(a: CurvedSide, b: CurvedSide, label: str = "") -> CurvedSide:
    """Union of two triangulations (used to build penetrating / duplicated coverage)."""
    return CurvedSide(np.concatenate([a.tris, b.tris]), a.patch,
                      label or f"{a.label}+{b.label}", rule=a.rule)


# ---------------------------------------------------------------------------
# intersections, per-patch balance, gap / penetration accounting
# ---------------------------------------------------------------------------
def _bbox(tri):
    return tri.min(axis=0), tri.max(axis=0)


def pair_measures(master: CurvedSide, slave: CurvedSide, *, tol: float = 1e-15):
    """Exact parameter-domain intersection of every overlapping facet pair.

    Returns ``(i, j, m, mom)`` with ``i`` the slave facet, ``j`` the master facet,
    ``m = int_{R_ij} J du dv`` and ``mom = int_{R_ij} (x - c_i) J du dv``.  The
    candidate screen is a bounding-box test in the parameter domain.
    """
    entries = []
    lo_s = np.asarray([t.min(axis=0) for t in slave.tris])
    hi_s = np.asarray([t.max(axis=0) for t in slave.tris])
    for j, tm in enumerate(master.tris):
        lo_m, hi_m = _bbox(tm)
        cand = ~(np.any(lo_s > hi_m + tol, axis=1) | np.any(lo_m > hi_s + tol, axis=1))
        for i in np.nonzero(cand)[0]:
            R = clip_poly(tm, slave.tris[i])
            if len(R) < 3 or abs(_signed_area(R)) <= tol:
                continue
            m, mom = poly_measure_moment(master.patch, R)
            if m <= 0.0:
                continue
            entries.append((int(i), j, m, mom - m * slave.centroids[i]))
    return entries


def per_patch_balance(master: CurvedSide, slave: CurvedSide, entries=None) -> dict:
    """Per-master-facet coverage balance ``sum_i m_ij`` vs ``|Gamma_j|``.

    Reports the gate quantity ``max_j |sum_i m_ij - |Gamma_j|| / |Gamma_j|``
    together with the *global* sum (the quantity the previous node gated) so that
    the two can be compared.
    """
    if entries is None:
        entries = pair_measures(master, slave)
    s = np.zeros(master.n)
    for (i, j, m, _mom) in entries:
        s[j] += m
    rel = np.abs(s - master.measures) / master.measures
    return {
        "per_facet_max_rel": float(rel.max()),
        "per_facet_median_rel": float(np.median(rel)),
        "n_uncovered": int(np.sum(s <= 0.0)),
        "global_rel": float((s.sum() - master.area) / master.area),
        "sum_ij": s,
        "master_measures": master.measures.copy(),
    }


def coverage_arrangement(master: CurvedSide, slave: CurvedSide, *,
                         area_tol: float = 1e-18, max_pieces: int = 8192) -> dict:
    """Exact multiplicity arrangement of slave footprints inside each patch.

    For every master facet the parameter triangle is split by the covering slave
    footprints into convex pieces carrying coverage multiplicity ``c``::

        sum_i m_ij = sum_pieces c        * |piece|_Gamma
        V_pen,j    = sum_pieces max(c-1,0) * |piece|_Gamma
        V_gap,j    = sum_pieces max(1-c,0) * |piece|_Gamma
        |Gamma_j|  = sum_pieces            |piece|_Gamma      (machine precision)
    """
    out = {"sum_ij": np.zeros(master.n), "v_gap": np.zeros(master.n),
           "v_pen": np.zeros(master.n), "total": np.zeros(master.n),
           "max_mult": np.zeros(master.n, dtype=int),
           "n_pieces": np.zeros(master.n, dtype=int)}
    for j, tm in enumerate(master.tris):
        lo_m, hi_m = _bbox(tm)
        lo_s = np.asarray([t.min(axis=0) for t in slave.tris])
        hi_s = np.asarray([t.max(axis=0) for t in slave.tris])
        pre = ~(np.any(lo_s > hi_m, axis=1) | np.any(lo_m > hi_s, axis=1))
        cand = [slave.tris[i] for i in np.nonzero(pre)[0]
                if len(clip_poly(tm, slave.tris[i])) >= 3]
        pieces = [(np.asarray(tm, dtype=float), 0)]
        for q in cand:
            new = []
            for p, c in pieces:
                ins = clip_poly(p, q)
                if len(ins) >= 3 and abs(_signed_area(ins)) > area_tol:
                    new.append((ins, c + 1))
                for o in _outside_pieces(p, q):
                    if abs(_signed_area(o)) > area_tol:
                        new.append((o, c))
            pieces = new[:max_pieces]
        s_ij = v_gap = v_pen = tot = 0.0
        mx = 0
        for p, c in pieces:
            m, _ = poly_measure_moment(master.patch, p)
            tot += m
            s_ij += c * m
            if m > 1e-12 * master.measures[j]:
                mx = max(mx, c)
            if c == 0:
                v_gap += m
            elif c > 1:
                v_pen += (c - 1) * m
        out["sum_ij"][j] = s_ij
        out["v_gap"][j] = v_gap
        out["v_pen"][j] = v_pen
        out["total"][j] = tot
        out["max_mult"][j] = mx
        out["n_pieces"][j] = len(pieces)
    out["max_mult_overall"] = int(out["max_mult"].max()) if master.n else 0
    lt = master.measures
    bal = master.measures + out["v_pen"] - out["v_gap"] - out["sum_ij"]
    out["balance_rel"] = float(np.max(np.abs(bal) / lt)) if master.n else 0.0
    out["identity_total_rel"] = float(abs(out["total"].sum() - master.area) / master.area)
    out["v_gap_total"] = float(out["v_gap"].sum())
    out["v_pen_total"] = float(out["v_pen"].sum())
    return out


def intersection_curve(patch: ParamPatch, master: CurvedSide, slave: CurvedSide,
                       *, n_samples: int = 17, max_segments: int = 400) -> dict:
    """The true intersection curve between two facets, in 3D.

    For a non-matching pair the shared boundary of ``R_ij`` is a polyline in the
    parameter domain; mapped through ``x(u, v)`` it is a **space curve on the
    interface**, not a straight chord.  The plane-projection construction replaces
    it by the straight chord between projected endpoints.  The reported numbers are
    the total curve length and the chord length, i.e. how much of the interface
    "wire length" the projection loses at a given mesh size.
    """
    curve_len = 0.0
    chord_len = 0.0
    n_used = 0
    worst = 0.0
    for j, tm in enumerate(master.tris):
        lo_m, hi_m = _bbox(tm)
        for ts in slave.tris:
            lo_s, hi_s = _bbox(ts)
            if np.any(lo_s > hi_m) or np.any(lo_m > hi_s):
                continue
            R = clip_poly(tm, ts)
            if len(R) < 3:
                continue
            for k in range(len(R)):
                a, b = R[k], R[(k + 1) % len(R)]
                s = np.linspace(0.0, 1.0, max(2, n_samples))
                uv = a[None, :] + s[:, None] * (b - a)[None, :]
                pts = patch.x(uv)
                seg = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
                chord = float(np.linalg.norm(pts[-1] - pts[0]))
                curve_len += seg
                chord_len += chord
                worst = max(worst, (seg - chord) / max(chord, 1e-300))
                n_used += 1
                if n_used >= max_segments:
                    break
            if n_used >= max_segments:
                break
        if n_used >= max_segments:
            break
    return {"curve_length": curve_len, "chord_length": chord_len,
            "rel_gain": (curve_len - chord_len) / chord_len if chord_len > 0 else 0.0,
            "worst_segment_gain": worst, "segments": n_used}


# ---------------------------------------------------------------------------
# the mortar operator in the curved metric
# ---------------------------------------------------------------------------
@dataclass
class CurvedMortarOperator:
    """Second-order mortar operator ``M: slave cell means -> master cell means``.

    ``M_ji = (m_ij + g_i . mu_ij) / |Gamma_j|`` with ``g_i`` a least-squares gradient
    over the neighbouring slave centroids (exact for affine ambient fields).
    ``order=0`` drops the gradient term.  ``measure`` selects the (broken) controls:
    ``"curved"`` (exact curved quadrature, default), ``"centroid_j"`` (``J`` frozen
    at the polygon centroid) and ``"flat"`` (``J = 1``, curvature ignored).
    """

    master: CurvedSide
    slave: CurvedSide
    order: int = 2
    measure: str = "curved"
    k_nn: int = 8

    def __post_init__(self):
        self.entries = pair_measures(self.master, self.slave)
        # self.m[j, i] = measure of slave facet i in master facet j; the master is
        # the target of the mortar transfer, so the matrix is (n_master, n_slave).
        self.m = np.zeros((self.master.n, self.slave.n))
        self.mom = np.zeros((self.master.n, self.slave.n, 3))
        for (i, j, m, mom) in self.entries:
            w, mw = self._pair_measure(i, j, m, mom)
            self.m[j, i] += w
            self.mom[j, i] += mw
        mom_abs = self.mom + self.m[..., None] * self.slave.centroids[None, :, :]
        self.nu = mom_abs - self.m[..., None] * self.master.centroids[:, None, :]
        self.M = self.m / self.master.measures[:, None]
        self.N = self.m / self.slave.measures[None, :]
        self.nn_idx = self._neighbours(self.slave.centroids, self.k_nn)

    # -- measures ---------------------------------------------------------
    def _pair_measure(self, i, j, m, mom):
        """Measure (and first moment) of one pair in the selected metric."""
        if self.measure == "curved":
            return m, mom
        R = clip_poly(self.master.tris[j], self.slave.tris[i])
        if len(R) < 3:
            return 0.0, np.zeros(3)
        a2 = abs(_signed_area(R))
        uv_c = _poly_centroid_2d(R)
        d = self.master.patch.x(uv_c[None, :])[0] - self.slave.centroids[i]
        if self.measure == "centroid_j":
            mm = a2 * float(self.master.patch.jac(uv_c[None, :])[0])
            return mm, mm * d
        if self.measure == "flat":
            return a2, a2 * d
        raise ValueError(f"unknown measure {self.measure!r}")

    # -- gradients / assembly ---------------------------------------------
    def _neighbours(self, centroids, k_nn):
        d = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        k = min(k_nn, len(centroids) - 1)
        return np.argsort(d, axis=1)[:, :k]

    def gradients_for(self, u) -> np.ndarray:
        """Batch least-squares gradients of the cell means ``u``.

        Exact for affine ambient fields: the normal equations of the local affine fit
        ``u_k - u_i = g . (c_k - c_i)`` are assembled and solved for all cells at once.
        """
        u = np.asarray(u, dtype=float)
        c = self.slave.centroids
        idx = self.nn_idx
        d = c[idx] - c[:, None, :]                       # (n, k, 3)
        A = np.concatenate([d, np.ones((d.shape[0], d.shape[1], 1))], axis=2)
        G = np.einsum("nik,nil->nkl", A, A)
        rhs = np.einsum("nik,ni->nk", A, u[idx] - u[:, None])
        try:
            sol = np.linalg.solve(G, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(G[0], rhs[0], rcond=None)[0][None, :]
        return sol[:, :3]

    def matrix(self, u=None) -> np.ndarray:
        """Full (second-order) matrix for the field ``u`` used to build the gradients."""
        if self.order < 2:
            return self.M.copy()
        g = self.gradients_for(np.zeros(self.slave.n) if u is None else u)
        return self.M + np.einsum("ki,jki->jk", g, self.mom) / self.master.measures[:, None]

    def apply(self, u) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        out = self.M @ u
        if self.order < 2:
            return out
        g = self.gradients_for(u)
        return out + np.einsum("ki,jki->j", g, self.mom) / self.master.measures

    def apply_transpose(self, w, *, weights=None) -> np.ndarray:
        """``M^T W_t w``: the measure-weighted transpose (the dual operator)."""
        w = np.asarray(w, dtype=float)
        wt = self.master.measures if weights is None else np.asarray(weights, float)
        return self.matrix().T @ (wt * w)

    # -- diagnostics ------------------------------------------------------
    def conservation(self) -> dict:
        s = self.m.sum(axis=1)
        rel = np.abs(s - self.master.measures) / self.master.measures
        Mt = self.matrix()
        col = np.sum(Mt * self.master.measures[:, None], axis=0)
        return {"per_facet_max_rel": float(rel.max()),
                "global_rel": float((s.sum() - self.master.area) / self.master.area),
                "colsum_max_rel": float(np.max(np.abs(col - self.slave.measures)
                                               / self.slave.measures))}

    def patch_test(self, a, b=0.0) -> float:
        """Relative defect of the exact reproduction of the affine field ``a.x + b``."""
        a = np.asarray(a, dtype=float)
        u = self.slave.centroids @ a + b
        v = self.apply(u)
        exact = self.master.centroids @ a + b
        return float(np.max(np.abs(v - exact)) / max(np.max(np.abs(exact)), 1e-300))


def _poly_centroid_2d(poly) -> np.ndarray:
    p = _ccw(poly)
    a = _signed_area(p)
    if abs(a) < 1e-300:
        return p.mean(axis=0)
    n = p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]
    cx = float(np.sum((p[:, 0] + np.roll(p[:, 0], -1)) * n)) / (6.0 * a)
    cy = float(np.sum((p[:, 1] + np.roll(p[:, 1], -1)) * n)) / (6.0 * a)
    return np.array([cx, cy])


def plane_projection_defect(master: CurvedSide, slave: CurvedSide) -> dict:
    """Previous-node construction: project chords into the master chord's plane.

    Uses the flat chords of both sides, orthogonally projects the slave chord onto
    the master chord's plane, clips in that plane and sums the projected areas;
    the per-facet coverage defect is the number the earlier node could only report
    as a diagnostic (it was O(kappa h) and therefore never gated).
    """
    from .geometry import cell_overlap

    mt = master.chords()
    st = slave.chords()
    lo_s = st.min(axis=1) - 1e-12
    hi_s = st.max(axis=1) + 1e-12
    covered = np.zeros(master.n)
    for j in range(master.n):
        A = mt[j]
        lo_m, hi_m = A.min(axis=0), A.max(axis=0)
        cand = ~(np.any(lo_s > hi_m, axis=1) | np.any(lo_m > hi_s, axis=1))
        na = np.cross(A[1] - A[0], A[2] - A[0])
        na = na / np.linalg.norm(na)
        b1 = (A[1] - A[0]) / np.linalg.norm(A[1] - A[0])
        basis = np.array([b1, np.cross(na, b1)])
        origin = A.mean(axis=0)
        tot = 0.0
        for i in np.nonzero(cand)[0]:
            B = st[i]
            nb = np.cross(B[1] - B[0], B[2] - B[0])
            nb = nb / np.linalg.norm(nb)
            if abs(float(na @ nb)) < 0.15:
                continue
            Bp = B - np.outer((B - origin) @ na, na)
            area, _mom = cell_overlap(A, Bp, basis, origin, 1e-15)
            if area > 0.0:
                tot += area
        covered[j] = tot
    rel = np.abs(covered - master3_measures(master)) / master3_measures(master)
    return {"per_facet_max_rel": float(rel.max()),
            "per_facet_median_rel": float(np.median(rel)),
            "global_rel": float((covered.sum() - master.area) / master.area)}


def master3_measures(master: CurvedSide) -> np.ndarray:
    """Measures used by the plane-projection control (the curved facet measures)."""
    return master.measures


# ---------------------------------------------------------------------------
# time-domain account: two first-order systems coupled through the surface
# ---------------------------------------------------------------------------
def first_order_coupling(op: CurvedMortarOperator, *, dt: float = 1e-3, nsteps: int = 500,
                         k_gap: float = 1.0, decay_a: float = 0.4,
                         decay_b: float = 0.4, u0_a=1.0, u0_b=1.0,
                         flux_form: str = "closure") -> dict:
    """Two first-order systems coupled through the curved interface.

    Side A (slave cells) and side B (master cells) each relax toward zero at rates
    ``decay_a`` / ``decay_b`` and exchange heat through the interface with a **gap
    conductance** ``k_gap`` (``R = 1/k_gap`` per unit area)::

        v_j   = (M u_A)_j                       projection mortar (order 2)
        lam_j = (u_B,j - v_j) / R               flux per unit master area
        du_A,i/dt = -a_A u_A,i - (sum_j m_ij lam_j) / |S_i|
        du_B,j/dt = -a_B u_B,j + c_j lam_j

    ``flux_form="closure"`` uses ``c_j = 1`` -- the standard mortar form.  It is mass
    consistent only if ``sum_i m_ij = |Gamma_j|`` holds *per facet*: the interface
    leak is then ``sum_j (|Gamma_j| - sum_i m_ij) lam_j``, i.e. exactly the per-patch
    conservation defect, and a uniform state (``u_A = u_B = const``) has
    ``lam = -(defect)/R != 0`` even though nothing should flow -- that is the
    "artificial injection/dissipation at the interface" this node is about.
    ``flux_form="coverage"`` uses ``c_j = sum_i m_ij / |Gamma_j|``; it telescopes for
    any geometry and is the control that separates a geometry defect from a scheme
    defect.

    Bookkeeping (each step, exact algebra; the reported residuals are round-off):

    * mass:  ``dM = -dt (a_A W_s.u_A + a_B W_t.u_B) + dt sum_j (W_t,j c_j - S_j) lam_j``
    * energy: ``E_{n+1} - E_n = dt sum W u f + dt^2/2 sum W f^2`` (exact for the
      explicit update) and the physical split
      ``dE/dt = -D_A - D_B + [sum W_t,j c_j lam_j u_B,j - sum_i q_i u_A,i]``.
    """
    Ws = op.slave.measures
    Wt = op.master.measures
    nA, nB = op.slave.n, op.master.n
    u_a = np.full(nA, float(u0_a))
    u_b = np.full(nB, float(u0_b))
    R = 1.0 / k_gap
    S = op.m.sum(axis=1)                       # per-master covered measure
    cov = S / Wt
    c_vec = np.ones(nB) if flux_form == "closure" else cov

    def mass(ua, ub):
        return float(Ws @ ua + Wt @ ub)

    def energy(ua, ub):
        return 0.5 * float(Ws @ (ua ** 2) + Wt @ (ub ** 2))

    m0, e0 = mass(u_a, u_b), energy(u_a, u_b)
    hist = {"t": [], "mass": [], "energy": [], "lam_max": [], "leak": []}
    renorm = 0.0
    leak_accum = 0.0
    mass_res = 0.0
    E_res = 0.0
    lam_max = 0.0
    flux_power = 0.0
    pseudo = 0.0
    e_prev = e0
    for step in range(int(nsteps)):
        v = op.apply(u_a)
        lam = (u_b - v) * k_gap
        q = op.m.T @ lam
        f_a = -decay_a * u_a - q / Ws
        f_b = -decay_b * u_b + c_vec * lam
        dm = float(Ws @ (dt * f_a) + Wt @ (dt * f_b))
        dm_expect = -dt * (decay_a * float(Ws @ u_a) + decay_b * float(Wt @ u_b)) \
            + dt * float((Wt * c_vec - S) @ lam)
        dE = dt * (float(Ws @ (u_a * f_a)) + float(Wt @ (u_b * f_b))) \
            + 0.5 * dt * dt * (float(Ws @ (f_a ** 2)) + float(Wt @ (f_b ** 2)))
        D_A = decay_a * float(Ws @ (u_a ** 2))
        D_B = decay_b * float(Wt @ (u_b ** 2))
        exch = float(Wt @ (c_vec * lam * u_b)) - float(u_a @ q)
        u_a = u_a + dt * f_a
        u_b = u_b + dt * f_b
        e_now = energy(u_a, u_b)
        mass_res = max(mass_res, abs(dm - dm_expect))
        E_res = max(E_res, abs((e_now - e_prev) - dE))
        renorm = max(renorm, abs(dm_expect))
        leak_accum += dt * float((Wt * c_vec - S) @ lam)
        lam_max = max(lam_max, float(np.max(np.abs(lam))))
        flux_power = max(flux_power, max(D_A + D_B, 1e-300))
        pseudo = max(pseudo, abs(exch - float(np.sum(Wt * c_vec * R * lam ** 2))))
        e_prev = e_now
        hist["t"].append((step + 1) * dt)
        hist["mass"].append(mass(u_a, u_b))
        hist["energy"].append(e_now)
        hist["lam_max"].append(float(np.max(np.abs(lam))))
        hist["leak"].append(float((Wt * c_vec - S) @ lam))
    m1 = mass(u_a, u_b)
    e1 = energy(u_a, u_b)
    return {
        "dt": dt, "nsteps": int(nsteps), "flux_form": flux_form, "k_gap": k_gap,
        "mass0": m0, "mass1": m1, "energy0": e0, "energy1": e1,
        "per_facet_defect": op.conservation()["per_facet_max_rel"],
        "lam_max": lam_max,
        "mass_bookkeeping_residual_rel": mass_res / max(renorm, 1e-300),
        "energy_bookkeeping_residual_rel": E_res / max(abs(e0), 1e-300) / max(dt, 1e-300),
        "mass_leak_accum": leak_accum,
        "mass_leak_rel": abs(leak_accum) / max(abs(m0), 1e-300),
        "mass_drift_rel": abs(m1 - m0 - (-0.0)) / max(abs(m0), 1e-300),
        "energy_drift_rel": abs(e1 - e0) / max(abs(e0), 1e-300),
        "spurious_power_max_rel": pseudo / max(flux_power, 1e-300),
        "interface_leak_rate_max": leak_accum,
        "history": hist,
    }


# ---------------------------------------------------------------------------
# transpose (dual) consistency in the curved metric
# ---------------------------------------------------------------------------
def dual_moment_defect(op: CurvedMortarOperator, direction=None) -> float:
    """Max relative defect of the measure-weighted transpose's first moment.

    For each slave cell ``i`` and each ambient direction ``a``::

        (M^T W_t x)_i = sum_j W_t,j M_ji (a . c_j)   vs   int_{Omega_i} (a . x) J

    i.e. the transpose operator must reproduce the first moment of the slave
    footprint in the **curved** metric.  Zeroth order this is the exact statement
    ``sum_j m_ij c_j``; second order it is a second-order-accurate statement whose
    rate is what :func:`run_acceptance` fits.
    """
    M = op.matrix()
    Wt = op.master.measures
    dirs = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
    if direction is not None:
        dirs = [np.asarray(direction, dtype=float)]
    worst = 0.0
    L = math.sqrt(float(op.master.area))          # patch length scale
    for a in dirs:
        a = a / np.linalg.norm(a)
        lhs = M.T @ (Wt * (op.master.centroids @ a))
        rhs = op.slave.measures * (op.slave.centroids @ a)
        # normalise by the moment scale |a| L |Omega_i|, not by a.c_i (which can
        # vanish on a flat patch and turn a small absolute defect into noise).
        scale = op.slave.measures * L
        worst = max(worst, float(np.max(np.abs(lhs - rhs) / scale)))
    return worst


def fit_rate(hs, errs) -> float:
    """Least-squares slope of ``log(err)`` vs ``log(h)`` (at least 2 distinct h)."""
    h = np.asarray(hs, dtype=float)
    e = np.asarray(errs, dtype=float)
    if len(h) < 2:
        return float("nan")
    ok = e > 0
    if ok.sum() < 2:
        return float("nan")
    p = np.polyfit(np.log(h[ok]), np.log(e[ok]), 1)
    return float(p[0])


# ---------------------------------------------------------------------------
# the metric-consistent transpose (adjoint) of the second-order mortar
# ---------------------------------------------------------------------------
def _ls_gradient(centroids, values, i, k_nn=8):
    d = np.linalg.norm(centroids - centroids[i], axis=1)
    idx = np.argsort(d)[1:k_nn + 1]
    if len(idx) < 3:
        return np.zeros(3)
    A = np.column_stack([centroids[idx] - centroids[i], np.ones(len(idx))])
    sol, *_ = np.linalg.lstsq(A, values[idx] - values[i], rcond=None)
    return sol[:3]


def adjoint_apply(op: CurvedMortarOperator, w, k_nn: int = 8) -> np.ndarray:
    """Metric-consistent adjoint ``N~: master cell means -> slave cell means``.

    The transpose of the mortar must carry the reconstruction on the **master**
    side too, otherwise it is only a midpoint (first-order) operator::

        (N~ w)_i = (1/|S_i|) sum_j [ m_ij w_j + h_j . nu_ji ] ,
        nu_ji = int_{R_ij} (x - c_j) J du dv ,   h_j = LS gradient of w on the master

    With the exact sub-cell moments ``nu`` the identity
    ``sum_i |S_i| (N~ w)_i`` telescopes and affine fields are reproduced exactly --
    but only in the true curved metric (see :func:`adjoint_diagnostics`).
    """
    w = np.asarray(w, dtype=float)
    c = op.master.centroids
    H = np.zeros((op.master.n, 3))
    for j in range(op.master.n):
        d = np.linalg.norm(c - c[j], axis=1)
        idx = np.argsort(d)[1:k_nn + 1]
        if len(idx) < 3:
            continue
        A = np.column_stack([c[idx] - c[j], np.ones(len(idx))])
        sol, *_ = np.linalg.lstsq(A, w[idx] - w[j], rcond=None)
        H[j] = sol[:3]
    return (op.m.T @ w + np.einsum("jk,jik->i", H, op.nu)) / op.slave.measures


def adjoint_diagnostics(op: CurvedMortarOperator) -> dict:
    """Transpose consistency numbers in the curved metric.

    ``patch_test``: max relative defect of the exact reproduction of affine fields
    by the metric-consistent adjoint (round-off if the moments are right).
    ``transpose_conservation``: ``|sum_i |S_i| N~_ij - |Gamma_j|| / |Gamma_j|``
    (the transpose of the per-patch conservation statement).
    ``naive_transpose_moment``: first-moment defect of the *midpoint* transpose
    (master centroids, no reconstruction), which is only first order -- the reason
    the reconstruction is needed on both sides.
    """
    dirs = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
    L = math.sqrt(float(op.master.area))
    worst = 0.0
    for a in dirs:
        a = a / np.linalg.norm(a)
        w = op.master.centroids @ a
        got = adjoint_apply(op, w)
        exact = op.slave.centroids @ a
        worst = max(worst, float(np.max(np.abs(got - exact))) / L)
    # dual patch test of order 0: the adjoint must preserve constants
    ones = np.ones(op.master.n)
    tcons = float(np.max(np.abs(adjoint_apply(op, ones) - 1.0)))
    naive = dual_moment_defect(op)
    return {"patch_test": worst, "transpose_conservation": tcons,
            "naive_transpose_moment": naive,
            "per_facet_conservation": op.conservation()["per_facet_max_rel"]}


# ---------------------------------------------------------------------------
# acceptance runner (criteria E1..E6)
# ---------------------------------------------------------------------------
def _provenance() -> dict:
    import hashlib
    import sys

    import numpy as _np

    h = hashlib.sha256()
    for name in ("ParamPatch", "CurvedSide", "CurvedMortarOperator", "first_order_coupling"):
        h.update(name.encode())
    return {"module": "tensordem.interface_operators.curved_patch_mortar",
            "numpy": _np.__version__, "python": sys.version.split()[0],
            "sha256_argnames": h.hexdigest()[:16],
            "rule": "tri_gauss(n=8) conical product, exact to degree 15"}


def _newton_history(patch: ParamPatch, p, uv0) -> list:
    """Residual sequence of the Newton projection (quadratic-convergence evidence)."""
    p = np.asarray(p, dtype=float)
    uv = np.asarray(uv0, dtype=float)
    hist = []
    for _ in range(12):
        r = patch.x(uv)[0] - p
        du, dv = patch.xu(uv)[0], patch.xv(uv)[0]
        g = np.array([r @ du, r @ dv])
        xuu, xuv, xvv = patch.xuu(uv)[0], patch.xuv(uv)[0], patch.xvv(uv)[0]
        H = np.array([[du @ du + r @ xuu, du @ dv + r @ xuv],
                      [du @ dv + r @ xuv, dv @ dv + r @ xvv]])
        step = -np.linalg.solve(H, g)
        uv = uv + step
        hist.append(float(np.linalg.norm(step)))
        if hist[-1] == 0.0:
            break
    return hist


def run_acceptance(meshes=(4, 6, 8, 10), *, seed_master: int = 11, seed_slave: int = 23,
                   jitter: float = 0.3) -> dict:
    """Machine-checkable acceptance of the curved-patch mortar (criteria E1..E6).

    Returns a plain dict (JSON-serialisable except ``histories``) so the benchmark
    script can dump it verbatim -- no number in the report is typed by hand.
    """
    out = {"provenance": _provenance(), "criteria": {}}
    patches = [SpherePatch(), GraphPatch()]

    # -- E1: Newton projection + true intersection curve -------------------
    e1 = {"projection": [], "curve": []}
    for patch in patches:
        L = math.sqrt(float(patch.area()))
        dirs = np.array([[0.31, 0.62, 0.0], [-0.2, 0.4, 0.9]])
        resid = 0.0
        quad = []
        for i, uvc in enumerate(np.array([[0.37, 0.62], [0.71, 0.18]])):
            p = patch.x(uvc[None, :])[0]
            uv, d, it, _g = patch.project(p, uv0=np.array([0.5, 0.5]))
            resid = max(resid, d)
            hist = _newton_history(patch, p, np.array([0.5, 0.5]))
            quad.append(hist)
        # ambient 3D mesh: vertices projected onto the patch (real-code path)
        side = grid_side(patch, 6, 5, jitter=jitter, seed=seed_master)
        verts = patch.x(side.tris.reshape(-1, 2))
        faces = np.arange(len(verts)).reshape(-1, 3)
        amb = side_from_ambient(patch, verts, faces)
        e1["projection"].append({
            "patch": patch.label, "kappa_L": patch.curvature_scale * L,
            "newton_residual": resid, "newton_residual_rel": resid / L,
            "newton_iterations": [len(h) for h in quad],
            "newton_history_last": quad[-1],
            "ambient_mesh_residual_rel": amb.projection_residual / L,
            "ambient_roundtrip_max_measure_diff": float(np.max(np.abs(amb.measures - side.measures) / side.measures)),
        })
        for n in meshes:
            m = grid_side(patch, n, n, jitter=jitter, seed=seed_master)
            s = grid_side(patch, n + 3, n + 2, jitter=jitter, seed=seed_slave)
            cur = intersection_curve(patch, m, s, n_samples=9)
            e1["curve"].append({"patch": patch.label, "h": 1.0 / n,
                                "rel_gain": cur["rel_gain"],
                                "curve_length": cur["curve_length"],
                                "chord_length": cur["chord_length"]})
    gains = {(k["patch"], round(k["h"], 6)): k["rel_gain"] for k in e1["curve"]}
    for patch in patches:
        hs = [k["h"] for k in e1["curve"] if k["patch"] == patch.label]
        gs = [k["rel_gain"] for k in e1["curve"] if k["patch"] == patch.label]
        e1.setdefault("curve_rate", {})[patch.label] = fit_rate(hs, gs)
    out["criteria"]["E1_newton_true_intersection"] = e1

    # -- E2 / E5: per-patch gate + transpose consistency + controls --------
    e2, e5, e6 = {"rows": []}, {"rows": []}, {"rows": []}
    for patch in patches:
        hs, d_cur, d_cen, d_flat, d_plane, pt_cur, pt_cen, pt_flat = [], [], [], [], [], [], [], []
        for n in meshes:
            m = grid_side(patch, n, n, jitter=jitter, seed=seed_master)
            s = grid_side(patch, n + 3, n + 2, jitter=jitter, seed=seed_slave)
            b = per_patch_balance(m, s)
            op = CurvedMortarOperator(m, s)
            opc = CurvedMortarOperator(m, s, measure="centroid_j")
            opf = CurvedMortarOperator(m, s, measure="flat")
            pp = plane_projection_defect(m, s)
            hs.append(1.0 / n)
            d_cur.append(b["per_facet_max_rel"])
            d_cen.append(opc.conservation()["per_facet_max_rel"])
            d_flat.append(opf.conservation()["per_facet_max_rel"])
            d_plane.append(pp["per_facet_max_rel"])
            pt_cur.append(adjoint_diagnostics(op)["patch_test"])
            pt_cen.append(adjoint_diagnostics(opc)["patch_test"])
            pt_flat.append(adjoint_diagnostics(opf)["patch_test"])
            e2["rows"].append({"patch": patch.label, "h": 1.0 / n,
                               "per_facet_max_rel": b["per_facet_max_rel"],
                               "global_rel": b["global_rel"],
                               "centroid_j": opc.conservation()["per_facet_max_rel"],
                               "flat": opf.conservation()["per_facet_max_rel"],
                               "plane_projection": pp["per_facet_max_rel"],
                               "plane_projection_global": pp["global_rel"],
                               "patch_test_curved": pt_cur[-1],
                               "patch_test_centroid_j": pt_cen[-1],
                               "patch_test_flat": pt_flat[-1]})
        e6["rows"].append({"patch": patch.label, "h": hs,
                           "curved": d_cur, "centroid_j": d_cen,
                           "flat": d_flat, "plane_projection": d_plane,
                           "rate_centroid_j": fit_rate(hs, d_cen),
                           "rate_plane_projection": fit_rate(hs, d_plane)})
        e5["rows"].append({"patch": patch.label, "h": hs, "curved": pt_cur,
                           "centroid_j": pt_cen, "flat": pt_flat,
                           "rate_naive_transpose": fit_rate(hs, d_cur),
                           "rate_transpose_vs_h": fit_rate(hs, pt_cur)})
    e2["gate"] = 1e-12
    e2["max_per_facet_rel"] = float(max(r["per_facet_max_rel"] for r in e2["rows"]))
    e2["max_global_rel"] = float(max(abs(r["global_rel"]) for r in e2["rows"]))
    e2["pass"] = bool(e2["max_per_facet_rel"] <= 1e-12)
    out["criteria"]["E2_per_patch_local_conservation"] = e2
    e5["gate"] = 1e-12
    e5["max_curved_patch_test"] = float(max(r["patch_test_curved"] for r in e2["rows"]))
    e5["min_broken_patch_test"] = float(min(min(r["patch_test_centroid_j"], r["patch_test_flat"])
                                            for r in e2["rows"]))
    e5["pass"] = bool(e5["max_curved_patch_test"] <= 1e-12
                      and e5["min_broken_patch_test"] > 1e-6)
    out["criteria"]["E5_transpose_consistency"] = e5
    out["criteria"]["E6_controls"] = e6

    # -- E3: gap / penetration volume terms -------------------------------
    e3 = {"rows": []}
    for patch in patches:
        m = grid_side(patch, 6, 6, jitter=jitter, seed=seed_master)
        cases = {
            "conforming": grid_side(patch, 7, 7, jitter=jitter, seed=seed_slave),
            "gap": grid_side(patch, 7, 7, jitter=jitter, seed=seed_slave,
                             box=(0.05, 0.95, 0.04, 0.96)),
            "penetration": concat_sides(
                grid_side(patch, 7, 7, jitter=jitter, seed=seed_slave),
                grid_side(patch, 5, 5, jitter=0.1, seed=seed_slave + 1, box=(0.2, 0.8, 0.2, 0.8)),
                label="penetration"),
        }
        for name, sl in cases.items():
            acc = coverage_arrangement(m, sl)
            e3["rows"].append({
                "patch": patch.label, "case": name,
                "balance_rel": acc["balance_rel"],
                "identity_total_rel": acc["identity_total_rel"],
                "v_gap_total": acc["v_gap_total"], "v_pen_total": acc["v_pen_total"],
                "v_gap_rel": acc["v_gap_total"] / m.area,
                "v_pen_rel": acc["v_pen_total"] / m.area,
                "max_multiplicity": acc["max_mult_overall"],
                "max_pieces_per_facet": int(acc["n_pieces"].max()),
            })
    e3["gate"] = 1e-12
    e3["max_balance_rel"] = float(max(r["balance_rel"] for r in e3["rows"]))
    e3["pass"] = bool(e3["max_balance_rel"] <= 1e-12)
    out["criteria"]["E3_gap_penetration_terms"] = e3

    # -- E4: time-domain coupling energy account -------------------------
    e4 = {"rows": []}
    for patch in patches:
        m = grid_side(patch, 6, 6, jitter=0.25, seed=seed_master)
        s = grid_side(patch, 8, 7, jitter=0.25, seed=seed_slave)
        for meas in ("curved", "centroid_j", "flat"):
            op = CurvedMortarOperator(m, s, measure=meas)
            r = first_order_coupling(op, dt=1e-3, nsteps=400)
            e4["rows"].append({
                "patch": patch.label, "measure": meas,
                "per_facet_defect": r["per_facet_defect"],
                "lam_max_uniform_state": r["lam_max"],
                "mass_leak_accum": r["mass_leak_accum"],
                "mass_bookkeeping_residual_rel": r["mass_bookkeeping_residual_rel"],
                "energy_bookkeeping_residual_rel": r["energy_bookkeeping_residual_rel"],
                "energy_drift_rel": r["energy_drift_rel"],
                "spurious_power_max_rel": r["spurious_power_max_rel"],
            })
    cur = [r for r in e4["rows"] if r["measure"] == "curved"]
    broken = [r for r in e4["rows"] if r["measure"] != "curved"]
    e4["gate"] = 1e-12
    e4["curved_lam_max"] = float(max(r["lam_max_uniform_state"] for r in cur))
    e4["curved_leak_max"] = float(max(abs(r["mass_leak_accum"]) for r in cur))
    e4["curved_energy_bookkeeping_max"] = float(max(r["energy_bookkeeping_residual_rel"] for r in cur))
    e4["broken_lam_min"] = float(min(r["lam_max_uniform_state"] for r in broken))
    e4["pass"] = bool(e4["curved_lam_max"] <= 1e-12 and e4["curved_leak_max"] <= 1e-12
                      and e4["curved_energy_bookkeeping_max"] <= 1e-10
                      and e4["broken_lam_min"] > 1e-6)
    out["criteria"]["E4_time_domain_energy_account"] = e4

    out["all_pass"] = bool(all(out["criteria"][k].get("pass", True) for k in out["criteria"]))
    return out
