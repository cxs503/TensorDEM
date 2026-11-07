"""Dimension-agnostic mortar (measure-weighted) interface operator, 1D/2D/3D.

The 1D operator of TensorPhysics mission m5207 is the ``nb = 1`` / ``dim = 2``
special case of this module: an interface is a ``nb``-dimensional manifold
(``nb`` = number of tangent directions) embedded in a ``dim``-dimensional ambient
space.  The three cases exercised here are

===============  ======  =====  ===========================================
case             nb      dim    geometry
===============  ======  =====  ===========================================
1D reference     1       1      two partitions of one interval
2D oblique       1       2      straight interface at 17 deg, non-matching
3D two cubes     2       3      shared square face of two unit cubes
===============  ======  =====  ===========================================

Operator definition
-------------------
Target cell mean::

    v_j = (1/|T_j|) * sum_i int_{S_i & T_j} [ u_i + g_i . (x - c_i) ] dx ,
    g_i = gradient reconstructed from the donor cell means (least squares)

Because the overlaps ``S_i & T_j`` partition every source cell exactly and
``int_{S_i}(x - c_i) dx = 0`` for the cell centroid, ``sum_j v_j |T_j| =
sum_i u_i |S_i|`` for **any** data: conservation is structural, not enforced.
The linear reconstruction makes the operator exact for linear data (patch test)
and second order for smooth data on non-matching meshes.

The measure-normalisation trap (mission m5207, section 5.2)
-----------------------------------------------------------
Operators that are *not* conservative by construction (RBF/MLS, momentum exchange
with a compact kernel) need an explicit global deficit backfill.  The deficit must
be spread **proportionally to measure divided by the sum of squared measures**::

    dv_j = (sent - recv) * |T_j| / sum_k |T_k|^2
    => sum_j dv_j |T_j| = (sent - recv) * sum_j |T_j|^2 / sum_k |T_k|^2
                        = (sent - recv)                     [exact]

The plausible-looking variant ``dv_j = (sent - recv) * |T_j| / sum_k |T_k|``
leaves ``(sent - recv) * (1 - sum|T|^2 / (sum|T|)^2)`` behind -- an arithmetic
residual of ~1e-2 that reads like a physics bug.  ``broken="backfill_linear"``
reproduces that trap on purpose; see ``tests/test_mortar_2d3d.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import (cell_overlap, gl_rule, plane_frame, signed_area)

__all__ = [
    "InterfaceSide",
    "segment_side",
    "grid_side",
    "tri_patch_side",
    "from_polygons",
    "MortarOperator",
    "overlap_entries",
    "gradient_matrix",
    "measure_proportional_backfill",
    "integral",
    "cell_averages",
    "l2_error",
    "measure_weighted_symmetry",
    "reflect_side",
    "rotate_side",
]


# ---------------------------------------------------------------------------
# side (one partition of the interface manifold)
# ---------------------------------------------------------------------------
@dataclass
class InterfaceSide:
    """One discretisation of an interface manifold.

    Attributes
    ----------
    polys : list of (nv, dim) arrays
        Ambient vertex coordinates of each convex cell.  For an interface curve
        (``nb = 1``) a cell is a two-vertex segment.
    basis : (nb, dim) array
        Orthonormal tangent basis of the manifold.
    origin : (dim,) array
        Origin of the local tangent frame.
    label : str
        Human-readable tag used in reports.
    local, measures, centroids, local_centroids
        Derived in ``__post_init__``: cell vertices/centroids in local tangent
        coordinates and the cell length (``nb == 1``) or area (``nb == 2``).
    """

    polys: list
    basis: np.ndarray
    origin: np.ndarray
    label: str = ""
    local: list = field(default_factory=list)
    measures: np.ndarray = field(default_factory=lambda: np.zeros(0))
    centroids: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    local_centroids: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def __post_init__(self) -> None:
        self.basis = np.asarray(self.basis, dtype=float)
        self.origin = np.asarray(self.origin, dtype=float)
        nb = self.basis.shape[0]
        self.local = [np.atleast_2d((np.asarray(p, dtype=float) - self.origin) @ self.basis.T)
                      for p in self.polys]
        meas, cents, lcents = [], [], []
        for p, lp in zip(self.polys, self.local):
            p = np.asarray(p, dtype=float)
            if nb == 1:
                x = lp[:, 0]
                meas.append(float(x.max() - x.min()))
                cents.append(0.5 * (p.min(axis=0) + p.max(axis=0)))
                lcents.append(np.array([0.5 * (x.max() + x.min())]))
            elif nb == 2:
                meas.append(abs(signed_area(lp)))
                c2 = _polygon_centroid_2d(lp)
                lcents.append(c2)
                cents.append(self.origin + self.basis.T @ c2)
            else:
                raise ValueError(f"unsupported nb={nb}")
        self.measures = np.asarray(meas, dtype=float)
        self.centroids = np.asarray(cents, dtype=float)
        self.local_centroids = np.asarray(lcents, dtype=float)
        if np.any(self.measures <= 0.0):
            raise ValueError("degenerate cell (zero length/area) in side")
        if nb == 2:
            nrm = np.cross(self.basis[0], self.basis[1])
            nrm = nrm / np.linalg.norm(nrm)
            out = max(float(np.max(np.abs((np.asarray(p, dtype=float)
                                            - self.origin) @ nrm)))
                      for p in self.polys)
            scale = max(float(np.sqrt(np.mean(self.measures))), 1e-300)
            if out > 1e-10 * max(scale, 1.0):
                raise ValueError(
                    f"non-planar cell in side {self.label!r}: out-of-plane "
                    f"deviation {out:.3e} exceeds 1e-10*scale")

    @property
    def n(self) -> int:
        return len(self.polys)

    @property
    def nb(self) -> int:
        return self.basis.shape[0]

    @property
    def dim(self) -> int:
        return self.basis.shape[1]

    def bbox_local(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([lp.min(axis=0) for lp in self.local])
        hi = np.array([lp.max(axis=0) for lp in self.local])
        return lo, hi


def _polygon_centroid_2d(poly: np.ndarray) -> np.ndarray:
    """Centroid of a 2D polygon (vertex mean fallback when degenerate)."""
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = 0.5 * cross.sum()
    if abs(a) < 1e-300:
        return p.mean(axis=0)
    return np.array([((x + x1) * cross).sum() / (6.0 * a),
                     ((y + y1) * cross).sum() / (6.0 * a)])


def _is_convex_local(poly: np.ndarray) -> bool:
    """Convexity test for a planar cell in local 2D coordinates."""
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return True
    if len(p) == 3:
        return abs(signed_area(p)) > 0.0
    n = len(p)
    sgn = 0.0
    for k in range(n):
        a, b, c = p[k], p[(k + 1) % n], p[(k + 2) % n]
        u1 = b - a
        u2 = c - b
        cr = float(u1[0] * u2[1] - u1[1] * u2[0])
        if abs(cr) < 1e-14:
            continue
        if sgn == 0.0:
            sgn = np.sign(cr)
        elif np.sign(cr) != sgn:
            return False
    return True


# ---------------------------------------------------------------------------
# factories
# ---------------------------------------------------------------------------
def segment_side(p0, p1, fractions, label: str = "") -> InterfaceSide:
    """Straight interface ``p0 -> p1`` partitioned at arclength ``fractions``.

    ``fractions`` are strictly increasing values in ``[0, 1]`` (cell breaks, both
    ends included).  Works in any ambient dimension, which is how the 1D
    reference case (``dim = 1``) and the 2D 17-degree oblique case share one code
    path.
    """
    p0 = np.atleast_1d(np.asarray(p0, dtype=float))
    p1 = np.atleast_1d(np.asarray(p1, dtype=float))
    fr = np.asarray(fractions, dtype=float)
    if abs(fr[0]) > 1e-12 or abs(fr[-1] - 1.0) > 1e-12:
        raise ValueError("fractions must start at 0 and end at 1")
    fr[0], fr[-1] = 0.0, 1.0
    if np.any(np.diff(fr) <= 0.0):
        raise ValueError("fractions must be strictly increasing")
    polys = [[p0 + fr[k] * (p1 - p0), p0 + fr[k + 1] * (p1 - p0)]
             for k in range(len(fr) - 1)]
    d = p1 - p0
    basis = (d / np.linalg.norm(d))[None, :]
    return InterfaceSide(polys=polys, basis=basis, origin=p0, label=label)


def _build_quads(origin, e1, e2, c1, c2, amp_abs, rng):
    """Node-jittered quad grid with convexity guaranteed by amplitude reduction."""
    n1, n2 = len(c1) - 1, len(c2) - 1
    base = np.zeros((n1 + 1, n2 + 1, 3))
    for a in range(n1 + 1):
        for b in range(n2 + 1):
            base[a, b] = origin + c1[a] * e1 + c2[b] * e2
    e1h = np.asarray(e1, dtype=float) / np.linalg.norm(e1)
    e2h = np.asarray(e2, dtype=float) / np.linalg.norm(e2)
    interior = [(a, b) for a in range(1, n1) for b in range(1, n2)]
    # jitter stays *inside* the interface plane: an out-of-plane component would
    # make the cells non-planar and silently corrupt area/moment integrals.
    u = {ab: (rng.uniform(-1.0, 1.0) * e1h + rng.uniform(-1.0, 1.0) * e2h)
         for ab in interior}
    scale = amp_abs
    for _ in range(60):
        nodes = base.copy()
        for (a, b), dlt in u.items():
            nodes[a, b] = base[a, b] + scale * dlt
        polys = []
        ok = True
        for a in range(n1):
            for b in range(n2):
                quad = np.array([nodes[a, b], nodes[a + 1, b],
                                 nodes[a + 1, b + 1], nodes[a, b + 1]])
                loc = np.array([(q - origin) @ np.vstack([e1, e2]).T for q in quad])
                if not _is_convex_local(loc) or abs(signed_area(loc)) <= 0.0:
                    ok = False
                    break
                polys.append(quad)
            if not ok:
                break
        if ok:
            return polys if polys else []
        scale *= 0.5
    raise RuntimeError("could not build a convex jittered grid")


def grid_side(origin, e1, e2, n1: int, n2: int, jitter: float = 0.0,
              seed: int = 0, label: str = "", cuts1=None, cuts2=None) -> InterfaceSide:
    """Quadrilateral surface patch: ``n1 x n2`` cells spanned by ``e1, e2``.

    ``jitter`` perturbs *interior* nodes by O(``jitter * h``) from a seeded uniform
    distribution; boundary nodes stay on the patch boundary so the partition of
    the interface patch remains exact -- that is what makes the coverage error a
    meaningful gate.  The amplitude is reduced automatically until every cell is
    convex (Sutherland-Hodgman clipping needs convex clippers).

    A random *cloud* would be the wrong tool for a convergence study: its
    clustering noise saturates and reports a fake order plateau (m5207 trap list).
    """
    origin = np.asarray(origin, dtype=float)
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    rng = np.random.default_rng(seed)
    c1 = np.linspace(0.0, 1.0, n1 + 1) if cuts1 is None else np.asarray(cuts1, float)
    c2 = np.linspace(0.0, 1.0, n2 + 1) if cuts2 is None else np.asarray(cuts2, float)
    h = min(float(np.diff(c1).min()) * np.linalg.norm(e1),
            float(np.diff(c2).min()) * np.linalg.norm(e2))
    polys = _build_quads(origin, e1, e2, c1, c2, jitter * h, rng)
    basis = np.vstack([e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)])
    return InterfaceSide(polys=polys, basis=basis, origin=origin, label=label)


def tri_patch_side(origin, e1, e2, n1: int, n2: int, jitter: float = 0.0,
                   seed: int = 0, label: str = "") -> InterfaceSide:
    """Triangulated surface patch (each quad split along a diagonal).

    A genuinely different cell type from :func:`grid_side`, so the 3D acceptance
    case exercises quad-triangle clipping in both roles.
    """
    quads = grid_side(origin, e1, e2, n1, n2, jitter=jitter, seed=seed).polys
    polys = []
    for k, q in enumerate(quads):
        if k % 2 == 0:
            polys.append(np.array([q[0], q[1], q[2]]))
            polys.append(np.array([q[0], q[2], q[3]]))
        else:
            polys.append(np.array([q[0], q[1], q[3]]))
            polys.append(np.array([q[1], q[2], q[3]]))
    origin = np.asarray(origin, dtype=float)
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    basis = np.vstack([e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)])
    return InterfaceSide(polys=polys, basis=basis, origin=origin, label=label)


def from_polygons(polys, label: str = "") -> InterfaceSide:
    """Build a side from arbitrary convex polygons (frame inferred by SVD)."""
    flat = np.vstack([np.asarray(p, dtype=float) for p in polys])
    origin, basis = plane_frame(flat)
    return InterfaceSide(polys=list(polys), basis=basis, origin=origin, label=label)


def reflect_side(side: InterfaceSide, point, normal) -> InterfaceSide:
    """Reflect a side about the plane through ``point`` with unit ``normal``.

    The *same* mirror axis must be used for both sides of an interface; reflecting
    each side about its own extent gives a spurious ``symmetry_flip`` of order 2
    (m5207 trap list).
    """
    point = np.asarray(point, dtype=float)
    nrh = np.asarray(normal, dtype=float)
    nrh = nrh / np.linalg.norm(nrh)

    def refl(x):
        x = np.asarray(x, dtype=float)
        return x - 2.0 * ((x - point) @ nrh)[..., None] * nrh

    polys = [refl(p) for p in side.polys][::-1]
    return InterfaceSide(polys=polys, basis=refl(side.basis + point) - point,
                         origin=refl(side.origin), label=side.label + "-mirrored")


def rotate_side(side: InterfaceSide, rot: np.ndarray, center=(0.0, 0.0, 0.0)) -> InterfaceSide:
    """Rigidly rotate a side about ``center`` by the 3D rotation ``rot``.

    Cell ordering is preserved, so a rotation-equivariance test can compare
    transferred fields element by element.
    """
    c = np.asarray(center, dtype=float)
    rot = np.asarray(rot, dtype=float)
    polys = [(np.asarray(p, dtype=float) - c) @ rot.T + c for p in side.polys]
    return InterfaceSide(polys=polys, basis=side.basis @ rot.T,
                         origin=(side.origin - c) @ rot.T + c,
                         label=side.label + "-rotated")


# ---------------------------------------------------------------------------
# overlap decomposition
# ---------------------------------------------------------------------------
def overlap_entries(src: InterfaceSide, tgt: InterfaceSide,
                    tol: float = 1e-12) -> list[tuple[int, int, float, np.ndarray]]:
    """Sparse overlap list ``(i, j, measure, moment)`` of two partitions.

    A bounding-box prefilter keeps this O(n) for structured meshes; each pair is
    then clipped in closed form, so ``sum(measure)`` equals the common support
    measure to round-off -- that is gate C4's coverage test.
    """
    if src.nb != tgt.nb:
        raise ValueError(f"manifold dimension mismatch: {src.nb} vs {tgt.nb}")
    lo_s, hi_s = src.bbox_local()
    lo_t, hi_t = tgt.bbox_local()
    scale = max(float(np.abs(lo_s).max()), float(np.abs(hi_s).max()), 1.0)
    pad = 1e-12 * scale
    cand = np.ones((src.n, tgt.n), dtype=bool)
    for k in range(src.nb):
        cand &= lo_s[:, k][:, None] <= hi_t[None, :, k] + pad
        cand &= lo_t[None, :, k] <= hi_s[:, k][:, None] + pad
    ii, jj = np.nonzero(cand)
    out = []
    for i, j in zip(ii, jj):
        m, mom = cell_overlap(src.polys[i], tgt.polys[j], src.basis, src.origin,
                              tol=tol)
        if m > 0.0:
            out.append((int(i), int(j), float(m), mom))
    return out


def gradient_matrix(side: InterfaceSide, k_neighbors: int = 6,
                    reg: float = 1e-12) -> np.ndarray:
    """Least-squares gradient operator ``G``, ``g_i = sum_l G[i, :, l] u_l``.

    Shape ``(n, nb, n)``.  The stencil is the set of nearest cell centroids on the
    same side (excluding the cell itself), which makes the reconstruction exact for
    linear data on any graded / jittered / triangulated mesh -- that is the 2D/3D
    patch test.  ``reg`` is a Tikhonov floor activated only for degenerate
    stencils (condition number above 1e10); it perturbs the exactness by O(reg).
    """
    n, nb = side.n, side.nb
    S = np.zeros((n, nb, n))
    if n < 2:
        return S
    x = side.local_centroids
    k = min(k_neighbors, n - 1)
    for i in range(n):
        d = x - x[i]
        dist = np.linalg.norm(d, axis=1)
        dist[i] = np.inf
        idx = np.argsort(dist)[:k]
        w = 1.0 / np.maximum(dist[idx], 1e-300) ** 2
        dd = d[idx]
        A = (w[:, None, None] * dd[:, :, None] * dd[:, None, :]).sum(axis=0)
        rhs = (w[:, None] * dd).T                       # (nb, k)
        cond = np.linalg.cond(A)
        if not np.isfinite(cond) or cond > 1e10:
            A = A + reg * max(np.trace(A), 1e-300) * np.eye(nb)
        coef = np.linalg.solve(A, rhs)                  # (nb, k)
        S[i][:, idx] = coef
        S[i][:, i] = -coef.sum(axis=1)
    return S


# ---------------------------------------------------------------------------
# conservative backfill
# ---------------------------------------------------------------------------
def measure_proportional_backfill(v: np.ndarray, sent: float, recv: float,
                                  measures: np.ndarray,
                                  broken: str | None = None) -> np.ndarray:
    """Spread the global conservation deficit proportionally to cell measure.

    ``dv_j = (sent - recv) * |T_j| / sum_k |T_k|^2``   (correct)

    ``broken="backfill_linear"`` uses ``|T_j| / sum_k |T_k|`` instead, which
    leaves ``(sent - recv) * (1 - sum|T|^2 / (sum|T|)^2)`` behind: the
    measure-normalisation trap of m5207 section 5.2, a ~1e-2 arithmetic residual
    that reads like a physics error.  Kept as an executable negative control.
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


def integral(values: np.ndarray, measures: np.ndarray) -> float:
    """Measure-weighted integral ``sum_k v_k |T_k|``."""
    return float(np.dot(np.asarray(values, dtype=float), np.asarray(measures, dtype=float)))


# ---------------------------------------------------------------------------
# mortar operator
# ---------------------------------------------------------------------------
@dataclass
class MortarOperator:
    """Mortar projection between two discretisations of the same manifold.

    Parameters
    ----------
    order : {1, 2}
        1 = piecewise-constant donor reconstruction (pure overlap weighting);
        2 = piecewise-linear donor reconstruction (adds the dipole term
        ``g_i . (x - c_i)``, exact for linear data).
    backfill : {None, "measure"}
        ``"measure"`` adds the measure-proportional deficit correction after the
        transfer (needed only when the grid pair does not partition the support
        exactly, and for the RBF/ME operators built on the same helper).
    broken : {None, "midpoint", "backfill_linear"}
        Deliberate negative controls: ``"midpoint"`` point-samples the donor
        reconstruction at the acceptor centroid (classic non-conservative flux on
        a non-matching interface); ``"backfill_linear"`` mis-normalises the
        backfill.
    """

    order: int = 2
    backfill: str | None = None
    broken: str | None = None
    name: str = "mortar-2d3d"

    def __post_init__(self) -> None:
        if self.order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        if self.backfill not in (None, "measure"):
            raise ValueError(f"unknown backfill mode {self.backfill!r}")
        if self.broken not in (None, "midpoint", "backfill_linear"):
            raise ValueError(f"unknown broken mode {self.broken!r}")

    # -- assembly --------------------------------------------------------
    def matrix(self, src: InterfaceSide, tgt: InterfaceSide) -> np.ndarray:
        """Dense mortar matrix ``A`` with ``v = A u`` (shape ``(nt, ns)``)."""
        A = np.zeros((tgt.n, src.n))
        S = gradient_matrix(src) if self.order == 2 else None
        for i, j, m, mom in overlap_entries(src, tgt):
            A[j, i] += m / tgt.measures[j]
            if S is not None:
                A[j, :] += (mom @ S[i]) / tgt.measures[j]
        return A

    # -- application -----------------------------------------------------
    def transfer(self, src: InterfaceSide, src_values: np.ndarray,
                 tgt: InterfaceSide) -> np.ndarray:
        """Map donor cell means ``src_values`` onto acceptor cell means."""
        u = np.asarray(src_values, dtype=float)
        if u.shape != (src.n,):
            raise ValueError(f"expected {src.n} donor values, got shape {u.shape}")
        if self.broken == "midpoint":
            v = self._transfer_midpoint(src, u, tgt)
        else:
            acc = np.zeros(tgt.n)
            S = gradient_matrix(src) if self.order == 2 else None
            for i, j, m, mom in overlap_entries(src, tgt):
                acc[j] += m * u[i]
                if S is not None:
                    acc[j] += float(mom @ (S[i] @ u))
            v = acc / tgt.measures
        if self.backfill == "measure":
            sent = integral(u, src.measures)
            recv = integral(v, tgt.measures)
            v = measure_proportional_backfill(
                v, sent, recv, tgt.measures,
                broken="backfill_linear" if self.broken == "backfill_linear" else None)
        return v

    def _transfer_midpoint(self, src: InterfaceSide, u: np.ndarray,
                           tgt: InterfaceSide) -> np.ndarray:
        """Negative control: point-sample the donor reconstruction.

        No overlap integration at all, so the received integral does not match the
        sent one on non-matching meshes (and the error does not shrink when one
        partition is shifted with respect to the other).
        """
        S = gradient_matrix(src) if self.order == 2 else np.zeros((src.n, src.nb, src.n))
        g = np.einsum("inl,l->in", S, u)
        x = src.local_centroids
        y = tgt.local_centroids
        d = np.linalg.norm(y[:, None, :] - x[None, :, :], axis=2)
        k = np.argmin(d, axis=1)
        return u[k] + np.einsum("in,in->i", g[k], y - x[k])


def cell_averages(side: InterfaceSide, field, n_gauss: int = 8) -> np.ndarray:
    """Cell means of an analytic field: ``int_{cell} f / |cell|``.

    ``nb == 1``: Gauss along the cell.  ``nb == 2``: the cell is fan-triangulated
    and each triangle integrated from the reference triangle with a tensor-product
    Gauss rule (exact for polynomials of degree ``2*n_gauss - 1`` per variable, so
    smooth fields are integrated to round-off).  Used both to *build* donor data
    and to measure the L2 error against the exact acceptor means.
    """
    xg, wg = gl_rule(n_gauss)
    out = np.zeros(side.n)
    for k, poly in enumerate(side.polys):
        poly = np.asarray(poly, dtype=float)
        if side.nb == 1:
            p0, p1 = poly[0], poly[-1]
            pts = 0.5 * (p0 + p1)[None, :] + 0.5 * xg[:, None] * (p1 - p0)[None, :]
            out[k] = 0.5 * float(np.dot(wg, np.asarray(field(pts), dtype=float)))
        else:
            acc = 0.0
            v0 = poly[0]
            for t in range(1, len(poly) - 1):
                a, b = poly[t], poly[t + 1]
                area = 0.5 * float(np.linalg.norm(np.cross(a - v0, b - v0)))
                if area <= 0.0:
                    continue
                # Duffy transform (r, s) in [0,1]^2 -> reference triangle,
                # jacobian (1-r); avoids the diagonal double counting of a
                # naively masked tensor rule on the half square.
                uu = 0.5 * (xg + 1.0)
                pts, wts = [], []
                for p in range(n_gauss):
                    for q in range(n_gauss):
                        r, sv = uu[p], uu[q]
                        pts.append(v0 + r * (a - v0) + (1.0 - r) * sv * (b - v0))
                        wts.append(0.25 * wg[p] * wg[q] * (1.0 - r))
                vals = np.asarray(field(np.asarray(pts)), dtype=float)
                acc += 2.0 * area * float(np.dot(np.asarray(wts), vals))
            out[k] = acc / side.measures[k]
    return out


def l2_error(values: np.ndarray, exact: np.ndarray, measures: np.ndarray) -> float:
    """Measure-weighted discrete L2 norm of ``values - exact``."""
    diff = np.asarray(values, dtype=float) - np.asarray(exact, dtype=float)
    m = np.asarray(measures, dtype=float)
    return float(np.sqrt(np.sum(diff * diff * m) / np.sum(m)))


def measure_weighted_symmetry(A_st: np.ndarray, A_ts: np.ndarray,
                              m_s: np.ndarray, m_t: np.ndarray) -> float:
    """Relative inconsistency between the two directions of a mortar pair.

    The two transfer matrices are adjoints *up to measure*::

        A_st[j, i] |T_j| == A_ts[i, j] |S_i| == |S_i & T_j|

    Returns ``max |A_st m_t - A_ts^T m_s| / max |A_st m_t|``.  This is the C5
    "M_ij vs M_ji" gate and is a machine-precision identity for order-1 mortar,
    because both entries carry the same clipped overlap measure.
    """
    lhs = np.asarray(A_st, dtype=float) * np.asarray(m_t, dtype=float)[:, None]
    rhs = (np.asarray(A_ts, dtype=float) * np.asarray(m_s, dtype=float)[:, None]).T
    denom = float(np.max(np.abs(lhs))) or 1.0
    return float(np.max(np.abs(lhs - rhs)) / denom)