"""Convex-polygon contact: SAT gap, witness pair, lever arms and contact law
(IN32-N01).

The floe field of :mod:`tensordem.polygon_floe_field` is a set of **convex
polygons** on the periodic ice channel.  Contacts between them (and between a
floe edge and the ship hull) are resolved with the separating-axis theorem (SAT)
plus a closest-feature-pair (CFP) witness computation.  Unlike a disk contact,
the contact of two polygons

* has a **signed gap** that is exact for both the separated and the overlapping
  branch (separated: minimum vertex/edge distance, exact for convex polygons;
  overlapping: minimum translation depth over all edge normals, i.e. the SAT
  penetration),
* carries a **contact point** off the line joining the centroids, hence a
  **lever arm** :math:`r_i = p - x_i` and a torque :math:`r_i \\times f_i`.

Sign and normal convention (same as :mod:`tensordem.granular2d`)
---------------------------------------------------------------
``gap > 0`` means a positive clearance, ``gap < 0`` an overlap of depth
``-gap``.  The unit normal ``n`` points from body ``a`` to body ``b``; pushing
``a`` by ``-gap * n`` (and ``b`` by ``+gap * n``) removes the overlap.  Forces on
``a`` are ``+f`` and on ``b`` are ``-f`` with ``f = f_n n + f_t t``.

Contact law
-----------
The normal/tangential law of :mod:`tensordem.granular2d` is reused verbatim:

.. math::
    f_n = \\max(0, k_n \\delta + c_n \\, n \\cdot v_{rel}), \\quad
    c_n = 2 \\zeta \\sqrt{k_n m_{eff}}, \\quad
    |f_t| \\le \\mu f_n, \\quad
    |M_r| \\le \\mu_r r_{eff} f_n

with :math:`\\delta = -gap` the penetration, ``v_rel`` the relative velocity of
the contact points and ``m_eff`` the pair reduced mass.  The **torque** on a
floe is :math:`\\tau_i = r_i \\times f_i` (plus rolling resistance), which is what
a disk model cannot produce: two disks touch on the centroid line, so their
contact force never has a lever arm.

Periodic boundaries
-------------------
All pair searches use the minimum-image displacement; a candidate list from a
KD-tree over the 3x3 image set is exact (see
:func:`candidate_pairs`), and the set of true contacts can be compared
element-by-element with the ``O(N^2)`` brute-force enumeration
(:func:`contact_set` with ``brute=True``).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "ContactGeometry",
    "PolygonContactConfig",
    "polygon_gap",
    "polygon_hull_gap",
    "candidate_pairs",
    "contact_set",
    "coordination_number",
    "fabric_tensor",
    "contact_forces",
]


# ---------------------------------------------------------------------------
# geometry primitives
# ---------------------------------------------------------------------------

def _point_segment_distance(points: np.ndarray, segments: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Distance ``(P, S)`` from every point to every segment, plus foot points.

    ``points`` is ``(P, 2)``, ``segments`` is ``(S, 2, 2)``; returns the distance
    matrix and the closest point on each segment (shape ``(P, S, 2)``).
    """
    a = segments[:, 0, :]
    ab = segments[:, 1, :] - a
    ap = points[:, None, :] - a[None, :, :]
    denom = np.maximum(np.einsum("sk,sk->s", ab, ab), 1e-300)
    t = np.einsum("psk,sk->ps", ap, ab) / denom[None, :]
    t = np.clip(t, 0.0, 1.0)
    foot = a[None, :, :] + t[..., None] * ab[None, :, :]
    dist = np.linalg.norm(points[:, None, :] - foot, axis=2)
    return dist, foot


def _edges(verts: np.ndarray) -> np.ndarray:
    return np.stack([verts, np.roll(verts, -1, axis=0)], axis=1)


@dataclass
class ContactGeometry:
    """Signed gap, normal, contact point and lever arms of one pair."""

    gap: float                     # signed gap [m] (>0 clearance, <0 overlap)
    normal: np.ndarray             # unit normal, a -> b
    point: np.ndarray              # contact point (world, a frame)
    r_a: np.ndarray                # point - x_a  (lever arm of a)
    r_b: np.ndarray                # point - x_b  (lever arm of b)
    axis: np.ndarray               # SAT axis of least penetration
    tangent: np.ndarray            # unit tangent (n rotated by +90 deg)

    @property
    def overlap(self) -> float:
        return max(0.0, -self.gap)


def polygon_gap(centroid_a: np.ndarray, verts_a: np.ndarray,
                centroid_b: np.ndarray, verts_b: np.ndarray,
                offset: Optional[np.ndarray] = None) -> ContactGeometry:
    """Exact signed gap and CFP witness of two convex polygons.

    ``verts_*`` are local vertices relative to ``centroid_*``.  ``offset`` is the
    minimum-image displacement that brings ``b`` into the neighbourhood of ``a``
    (defaults to ``centroid_b - centroid_a``).
    """
    ca = np.asarray(centroid_a, dtype=np.float64).reshape(2)
    cb = np.asarray(centroid_b, dtype=np.float64).reshape(2)
    va = np.asarray(verts_a, dtype=np.float64)
    vb = np.asarray(verts_b, dtype=np.float64)
    if va.ndim != 2 or va.shape[1] != 2 or va.shape[0] < 2:
        raise ValueError("verts_a must be (M >= 2, 2)")
    if vb.ndim != 2 or vb.shape[1] != 2 or vb.shape[0] < 2:
        raise ValueError("verts_b must be (M >= 2, 2)")
    if not (np.isfinite(va).all() and np.isfinite(vb).all()):
        raise ValueError("vertices must be finite")
    disp = (cb - ca) if offset is None else np.asarray(offset, dtype=np.float64)
    A = va                                    # a frame (centred on a's centroid)
    B = vb + disp                              # b in a's frame

    ea = _edges(A)
    eb = _edges(B)
    na = np.stack([ea[:, 1, 1] - ea[:, 0, 1], -(ea[:, 1, 0] - ea[:, 0, 0])], axis=1)
    nb = np.stack([eb[:, 1, 1] - eb[:, 0, 1], -(eb[:, 1, 0] - eb[:, 0, 0])], axis=1)
    axes = np.concatenate([na, nb], axis=0)
    norm = np.linalg.norm(axes, axis=1)
    keep = norm > 1e-12
    axes = axes[keep] / norm[keep][:, None]
    if axes.shape[0] == 0:
        raise ValueError("degenerate polygons: no edge normal")

    pa = A @ axes.T                                   # (Ma, S)
    pb = B @ axes.T
    min_a, max_a = pa.min(axis=0), pa.max(axis=0)
    min_b, max_b = pb.min(axis=0), pb.max(axis=0)
    sep = np.maximum(min_a - max_b, min_b - max_a)     # >0 -> separated on that axis
    if sep.max() > 0.0:
        # separated: exact distance from vertex-edge pairs (both directions)
        da, fa = _point_segment_distance(A, eb)
        db, fb = _point_segment_distance(B, ea)
        if da.min() <= db.min():
            i, j = np.unravel_index(np.argmin(da), da.shape)
            p_a = A[i]
            p_b = fa[i, j]
        else:
            i, j = np.unravel_index(np.argmin(db), db.shape)
            p_b = B[i]
            p_a = fb[i, j]
        d = float(np.linalg.norm(p_b - p_a))
        n = (p_b - p_a) / d if d > 1e-15 else np.asarray([1.0, 0.0])
        point = 0.5 * (p_a + p_b)
        axis = n
    else:
        overlap = np.minimum(max_a - min_b, max_b - min_a)
        k = int(np.argmin(overlap))
        axis = axes[k]
        depth = float(overlap[k])
        if (max_a[k] - min_b[k]) <= (max_b[k] - min_a[k]):
            n = axis if _center_side(A, B, axis) >= 0.0 else -axis
        else:
            n = axis if _center_side(A, B, axis) >= 0.0 else -axis
        # CFP witness: deepest vertex of one body projected on the other body
        p_a, p_b = _deepest_witness(A, B, eb, ea, n)
        point = 0.5 * (p_a + p_b)
        # orient the normal from a to b
        if np.dot(p_b - p_a, n) < 0.0:
            n = -n
        return ContactGeometry(gap=-depth, normal=n, point=point,
                               r_a=point - np.zeros(2), r_b=point - disp,
                               axis=axis, tangent=_perp(n))

    return ContactGeometry(gap=d, normal=n, point=point,
                           r_a=point - np.zeros(2), r_b=point - disp,
                           axis=axis, tangent=_perp(n))


def _perp(n: np.ndarray) -> np.ndarray:
    return np.asarray([-n[1], n[0]], dtype=np.float64)


def _center_side(A: np.ndarray, B: np.ndarray, axis: np.ndarray) -> float:
    """Sign of the b-centroid displacement measured along ``axis``."""
    return float(np.dot(B.mean(axis=0) - A.mean(axis=0), axis))


def _deepest_witness(A: np.ndarray, B: np.ndarray, eb: np.ndarray,
                     ea: np.ndarray, n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Witness pair for the overlapping branch: deepest vertex vs its projection.

    Returned in ``a``'s frame.  The deepest vertex of ``b`` (measured along ``n``)
    is projected onto ``A``'s boundary, and the symmetric candidate is evaluated
    as well; the pair with the larger inter-penetration is used.
    """
    best = None
    for pts, segs in ((B, ea), (A, eb)):
        proj = pts @ n
        i = int(np.argmin(proj) if pts is B else np.argmax(proj))
        vertex = pts[i]
        dist, foot = _point_segment_distance(vertex[None, :], segs)
        j = int(np.argmin(dist[0]))
        cand = (vertex, foot[0, j])
        depth = float(np.dot(cand[1] - cand[0], n)) * (1.0 if pts is B else -1.0)
        if best is None or depth > best[0]:
            best = (depth, cand[0].copy(), cand[1].copy())
    return best[1], best[2]


def polygon_hull_gap(centroid: np.ndarray, verts: np.ndarray,
                     seg_a: np.ndarray, seg_b: np.ndarray) -> ContactGeometry:
    """Contact of a convex polygon with one hull segment (a 2-vertex polygon)."""
    sa = np.asarray(seg_a, dtype=np.float64).reshape(2)
    sb = np.asarray(seg_b, dtype=np.float64).reshape(2)
    mid = 0.5 * (sa + sb)
    res = polygon_gap(centroid, verts, mid, np.stack([sa - mid, sb - mid]), None)
    # translate the lever arms back to the world frame: r_b is meaningless for a
    # static segment, only r_a (the floe arm) is used by the force model
    res.r_b = np.zeros(2)
    return res


# ---------------------------------------------------------------------------
# periodic pair search
# ---------------------------------------------------------------------------

def candidate_pairs(centroids: np.ndarray, reach: np.ndarray,
                    box: Optional[Tuple[float, float]] = None, *,
                    pad: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All pairs ``i < j`` whose bounding circles are within ``reach_i + reach_j``.

    ``reach`` is the per-body circumradius; a pair is returned when
    ``|x_j - x_i|_min <= reach_i + reach_j + pad``.  The displacement returned is
    the minimum-image vector ``x_j - x_i`` (the shortest periodic one), which is
    what the SAT witness expects.
    """
    x = np.asarray(centroids, dtype=np.float64)
    r = np.asarray(reach, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("centroids must be (N, 2)")
    if r.shape != (x.shape[0],):
        raise ValueError("reach must be (N,)")
    n = x.shape[0]
    if n < 2:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros((0, 2), dtype=np.float64))
    if box is None:
        tree = cKDTree(x)
        cutoff = float(r.max() + 0.5 * (r.max() + r.min())) + pad
        pairs = tree.query_pairs(cutoff + float(r.max()), output_type="ndarray")
        if pairs.size == 0:
            return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                    np.zeros((0, 2), dtype=np.float64))
        i, j = pairs[:, 0], pairs[:, 1]
        disp = x[j] - x[i]
        ok = np.linalg.norm(disp, axis=1) <= r[i] + r[j] + pad
        return i[ok], j[ok], disp[ok]
    lx, ly = float(box[0]), float(box[1])
    offsets = np.array([[a * lx, b * ly] for a in (-1, 0, 1) for b in (-1, 0, 1)],
                       dtype=np.float64)
    images = (x[None, :, :] + offsets[:, None, :]).reshape(-1, 2)
    owner = np.tile(np.arange(n), offsets.shape[0])
    tree = cKDTree(images)
    best: Dict[Tuple[int, int], Tuple[float, np.ndarray]] = {}
    for k in range(n):
        radius = float(r[k] + r.max() + pad)
        for idx in tree.query_ball_point(x[k], radius):
            j = int(owner[idx])
            if j == k:
                continue
            d = images[idx] - x[k]
            i_, j_ = (k, j) if k < j else (j, k)
            dd = float(np.dot(d, d))
            cur = best.get((i_, j_))
            if cur is None or dd < cur[0]:
                best[(i_, j_)] = (dd, d if k < j else -d)
    keys = sorted(best)
    if not keys:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros((0, 2), dtype=np.float64))
    i = np.array([k[0] for k in keys], dtype=np.int64)
    j = np.array([k[1] for k in keys], dtype=np.int64)
    disp = np.array([best[k][1] for k in keys], dtype=np.float64)
    ok = np.linalg.norm(disp, axis=1) <= r[i] + r[j] + pad
    return i[ok], j[ok], disp[ok]


def contact_set(centroids: np.ndarray, verts: np.ndarray, count: np.ndarray,
                *, box: Optional[Tuple[float, float]] = None,
                cutoff: float = 0.0, brute: bool = False
                ) -> Dict[str, np.ndarray]:
    """All pairs whose signed gap is ``<= cutoff``, with SAT geometry.

    ``brute=True`` enumerates all ``N (N-1) / 2`` pairs through the same geometric
    filter (``O(N^2)`` broadcasting) instead of the KD-tree, which is the
    independent cross-check used in the acceptance tests.
    """
    x = np.asarray(centroids, dtype=np.float64)
    v = np.asarray(verts, dtype=np.float64)
    c = np.asarray(count, dtype=np.int64)
    n = x.shape[0]
    reach = np.array([np.linalg.norm(v[k, : c[k]], axis=1).max() if c[k] else 0.0
                      for k in range(n)], dtype=np.float64)
    if brute:
        if box is None:
            dist = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
            disp = x[None, :, :] - x[:, None, :]
        else:
            lx, ly = box
            d = x[None, :, :] - x[:, None, :]
            d[..., 0] -= lx * np.round(d[..., 0] / lx)
            d[..., 1] -= ly * np.round(d[..., 1] / ly)
            disp = d
            dist = np.linalg.norm(d, axis=2)
        bound = reach[:, None] + reach[None, :] + cutoff
        iu, ju = np.nonzero(np.triu(dist <= bound, k=1))
        i, j = iu, ju
        disp = disp[i, j]
    else:
        i, j, disp = candidate_pairs(x, reach, box, pad=cutoff)
    gaps = np.empty(i.size, dtype=np.float64)
    normals = np.empty((i.size, 2), dtype=np.float64)
    points = np.empty((i.size, 2), dtype=np.float64)
    keep = np.zeros(i.size, dtype=bool)
    for m in range(i.size):
        a, b = int(i[m]), int(j[m])
        res = polygon_gap(x[a], v[a, : c[a]], x[b], v[b, : c[b]], disp[m])
        gaps[m] = res.gap
        normals[m] = res.normal
        points[m] = res.point + x[a]
        keep[m] = res.gap <= cutoff
    return {"i": i[keep], "j": j[keep], "gap": gaps[keep],
            "normal": normals[keep], "point": points[keep],
            "disp": disp[keep]}


def coordination_number(n_particles: int, i: np.ndarray, j: np.ndarray) -> float:
    """Mean coordination number ``2 Nc / N``."""
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    return 2.0 * float(len(i)) / float(n_particles)


def fabric_tensor(normals: np.ndarray, *, mask: Optional[np.ndarray] = None
                  ) -> Dict[str, float]:
    """Second-order fabric tensor ``Phi_ab = <n_a n_b>`` and its anisotropy."""
    nrm = np.asarray(normals, dtype=np.float64)
    if nrm.size == 0:
        return {"phi_xx": 0.0, "phi_xy": 0.0, "phi_yy": 0.0,
                "anisotropy": 0.0, "principal_angle": 0.0, "n": 0.0}
    if mask is not None:
        nrm = nrm[np.asarray(mask, dtype=bool)]
    if nrm.size == 0:
        return {"phi_xx": 0.0, "phi_xy": 0.0, "phi_yy": 0.0,
                "anisotropy": 0.0, "principal_angle": 0.0, "n": 0.0}
    phi = (nrm.T @ nrm) / nrm.shape[0]
    ev = np.linalg.eigvalsh(phi)
    anis = float((ev[-1] - ev[0]))
    angle = 0.5 * math.atan2(2.0 * phi[0, 1], phi[0, 0] - phi[1, 1])
    return {"phi_xx": float(phi[0, 0]), "phi_xy": float(phi[0, 1]),
            "phi_yy": float(phi[1, 1]), "anisotropy": anis,
            "principal_angle": float(angle), "n": float(nrm.shape[0])}


# ---------------------------------------------------------------------------
# contact law with lever arms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolygonContactConfig:
    """Material/contact parameters (SI, same conventions as granular2d)."""

    kn: float = 3.0e4
    kt_ratio: float = 0.5
    zeta: float = 0.2
    mu: float = 0.5
    mu_rolling: float = 0.05
    density: float = 917.0
    thickness: float = 0.03

    def __post_init__(self) -> None:
        for name in ("kn", "kt_ratio", "density", "thickness"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("zeta", "mu", "mu_rolling"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


def contact_forces(centroids: np.ndarray, masses: np.ndarray,
                   contacts: Dict[str, np.ndarray], *,
                   config: Optional[PolygonContactConfig] = None,
                   velocity: Optional[np.ndarray] = None,
                   omega: Optional[np.ndarray] = None,
                   slip: Optional[np.ndarray] = None,
                   rolling: Optional[np.ndarray] = None
                   ) -> Dict[str, np.ndarray]:
    """Normal/tangential contact forces **and torques** on every body.

    Returns per-contact ``f_n``, ``f_t``, the world force ``force`` on body ``i``
    and the world torque about the centroid of both bodies (``torque_i``,
    ``torque_j``), plus per-body accumulations ``body_force`` (N,2) and
    ``body_torque`` (N,).  ``slip`` is the accumulated tangential displacement
    ``xi_t`` of every contact (kept by the caller); ``f_t = -k_t xi_t`` clipped to
    the Coulomb cone ``mu f_n``.  ``rolling`` is an optional per-contact rolling
    resistance torque.  With neither the model is frictionless-normal only.
    """
    cfg = config or PolygonContactConfig()
    x = np.asarray(centroids, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    if m.shape != (x.shape[0],):
        raise ValueError("masses must be (N,)")
    n_body = x.shape[0]
    i = np.asarray(contacts["i"], dtype=np.int64)
    j = np.asarray(contacts["j"], dtype=np.int64)
    gap = np.asarray(contacts["gap"], dtype=np.float64)
    nrm = np.asarray(contacts["normal"], dtype=np.float64)
    pt = np.asarray(contacts["point"], dtype=np.float64)
    nc = i.size
    tng = np.stack([-nrm[:, 1], nrm[:, 0]], axis=1) if nc else np.zeros((0, 2))

    r_i = pt - x[i] if nc else np.zeros((0, 2))
    if nc and contacts.get("disp") is not None:
        # minimum-image position of body j: the torque arm must be measured from
        # the *image* of j that actually touches i, not from the folded centroid
        x_j_img = x[i] + np.asarray(contacts["disp"], dtype=np.float64)
    else:
        x_j_img = x[j] if nc else np.zeros((0, 2))
    r_j = pt - x_j_img if nc else np.zeros((0, 2))

    delta = np.maximum(0.0, -gap)
    if velocity is None:
        v_i = np.zeros((nc, 2))
        v_j = np.zeros((nc, 2))
    else:
        v = np.asarray(velocity, dtype=np.float64)
        v_i, v_j = v[i], v[j]
    w = np.zeros(n_body) if omega is None else np.asarray(omega, dtype=np.float64)

    # relative velocity of the two contact points
    if nc:
        omega_i = w[i][:, None] * np.stack([-r_i[:, 1], r_i[:, 0]], axis=1)
        omega_j = w[j][:, None] * np.stack([-r_j[:, 1], r_j[:, 0]], axis=1)
        v_rel = (v_i + omega_i) - (v_j + omega_j)
    else:
        v_rel = np.zeros((0, 2))
    v_n = np.einsum("ck,ck->c", v_rel, nrm)
    m_eff = 0.5 * (m[i] + m[j]) if nc else np.zeros(0)
    c_n = 2.0 * cfg.zeta * np.sqrt(cfg.kn * m_eff)
    f_n = np.maximum(0.0, cfg.kn * delta + c_n * v_n)
    if slip is None:
        f_t = np.zeros(nc)
    else:
        f_t = -cfg.kt_ratio * cfg.kn * np.asarray(slip, dtype=np.float64)
    cap = cfg.mu * f_n
    f_t = np.clip(f_t, -cap, cap) if nc else f_t
    m_roll = np.zeros(nc)
    if rolling is not None:
        m_roll = -cfg.mu_rolling * np.abs(np.asarray(rolling, dtype=np.float64))

    f_vec = f_n[:, None] * nrm + f_t[:, None] * tng if nc else np.zeros((0, 2))
    torque_i = (r_i[:, 0] * f_vec[:, 1] - r_i[:, 1] * f_vec[:, 0]) if nc else np.zeros(0)
    torque_j = -(r_j[:, 0] * f_vec[:, 1] - r_j[:, 1] * f_vec[:, 0]) if nc else np.zeros(0)
    torque_i = torque_i + m_roll
    torque_j = torque_j - m_roll

    body_force = np.zeros((n_body, 2))
    body_torque = np.zeros(n_body)
    if nc:
        np.add.at(body_force, i, f_vec)
        np.add.at(body_force, j, -f_vec)
        np.add.at(body_torque, i, torque_i)
        np.add.at(body_torque, j, torque_j)
    return {"i": i, "j": j, "gap": gap, "normal": nrm, "tangent": tng,
            "point": pt, "r_i": r_i, "r_j": r_j, "f_n": f_n, "f_t": f_t,
            "force": f_vec, "torque_i": torque_i, "torque_j": torque_j,
            "body_force": body_force, "body_torque": body_torque,
            "torque_arm": np.linalg.norm(r_i, axis=1) if nc else np.zeros(0)}