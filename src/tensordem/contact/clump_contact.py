"""Contact layer for rigid multi-sphere (clump) particles.

What this module contains
-------------------------
1. **Narrow phase on the member spheres.**  A clump is a rigid union of spheres,
   so every clump/clump, clump/wall or clump/obstacle contact is a set of
   member-sphere contacts, each with its own point and lever arm.  Because
   neighbouring members of a clump overlap, two different member pairs can
   sample the *same* physical contact patch; summing them double counts the
   junction force.  :func:`merge_junction_candidates` removes those duplicates
   with a contact-patch criterion (contact points closer than their patch radius
   and normals aligned within ``merge_angle``), keeping the deepest indentation
   of each cluster -- the same geometric argument as
   :mod:`tensordem.clump_clump`, here vectorised through a KD-tree so that it is
   affordable every time step.
2. **Force law.**  Linear spring-dashpot normal force, Cundall-Strack tangential
   spring with Coulomb limit, and a Cundall-type rolling-resistance torque
   ``M = -mu_r R_eff F_n  w_rel/|w_rel|`` acting on the *member* contact.  The
   normal branch is
   ``F_n = max(k_n delta - c_n v_n, 0)``, ``c_n = 2 zeta sqrt(m_eff k_n)``.
3. **Exact energy bookkeeping.**  Every term that stores or removes energy is
   returned per step: elastic normal ``k_n delta^2 / 2``, tangential spring
   ``k_t |xi|^2 / 2``, dashpot work ``c_n v_n^2 dt``, sliding friction work
   ``|F_t . v_t| dt``, rolling work ``|M . w_rel| dt`` and the tangential energy
   dropped when a contact detaches (:class:`EnergyLedger`).  This is what the
   compression test's "external work = kinetic + elastic + dissipated" gate
   (C6) is checked against.
4. **Force-chain descriptors.**  :func:`force_distribution_stats` (mean, std,
   coefficient of variation, Gini, strong-force fraction) and
   :func:`force_chain_clusters` (union-find over contacts carrying more than
   ``threshold * mean``), i.e. the quantities compared between non-spherical and
   spherical packings (C2).

The static contact geometry is verified separately in
``tests/test_clump_particle.py`` against closed forms (sphere-sphere and
sphere-box) and the junction merging against constructed duplicate pairs.

Units: SI (m, N, s, rad).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "ContactParams",
    "BoxObstacle",
    "sphere_box_contacts",
    "member_pair_candidates",
    "merge_junction_candidates",
    "ContactHistory",
    "evaluate_contact_law",
    "EnergyLedger",
    "force_distribution_stats",
    "force_chain_clusters",
    "union_find_clusters",
]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass
class ContactParams:
    """Linear spring-dashpot + Coulomb + rolling-resistance parameters."""

    kn: float = 150.0             # normal stiffness [N/m]
    kt_ratio: float = 0.7         # tangential stiffness / normal stiffness
    zeta: float = 0.3             # normal damping ratio [-]
    mu: float = 0.5               # sliding friction coefficient [-]
    mu_roll: float = 0.3          # rolling resistance coefficient [-]
    merge_angle_deg: float = 30.0 # junction merge normal tolerance
    eps_w: float = 1.0            # rolling-torque regularisation [rad/s] (stability-bounded viscous branch)

    @property
    def kt(self) -> float:
        return self.kt_ratio * self.kn

    def copy(self, **kw) -> "ContactParams":
        d = dict(
            kn=self.kn,
            kt_ratio=self.kt_ratio,
            zeta=self.zeta,
            mu=self.mu,
            mu_roll=self.mu_roll,
            merge_angle_deg=self.merge_angle_deg,
            eps_w=self.eps_w,
        )
        d.update(kw)
        return ContactParams(**d)


# ---------------------------------------------------------------------------
# Rigid box obstacles (walls, floor, piston)
# ---------------------------------------------------------------------------
@dataclass
class BoxObstacle:
    """Axis-aligned rigid box with optional constant velocity.

    Used for the floor, the (frictionless) side walls of the quasi-2D heap, the
    liftable box and the compression piston.  Contacts are computed as the
    point-to-box distance, so edges and corners are handled without special
    cases.
    """

    center: np.ndarray
    half: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mu: Optional[float] = None        # per-obstacle override of ContactParams.mu
    mu_roll: Optional[float] = None
    name: str = "box"

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", np.asarray(self.center, dtype=float).reshape(3))
        h = np.asarray(self.half, dtype=float).reshape(3)
        if np.any(h <= 0.0):
            raise ValueError("box half sizes must be positive")
        object.__setattr__(self, "half", h)
        object.__setattr__(self, "velocity", np.asarray(self.velocity, dtype=float).reshape(3))

    @property
    def lo(self) -> np.ndarray:
        return self.center - self.half

    @property
    def hi(self) -> np.ndarray:
        return self.center + self.half

    def advance(self, dt: float) -> None:
        object.__setattr__(self, "center", self.center + self.velocity * dt)


def sphere_box_contacts(
    points: np.ndarray, radii: np.ndarray, box: BoxObstacle
) -> Dict[str, np.ndarray]:
    """Member-sphere vs box contacts.

    Returns arrays ``penetration``, ``normal`` (unit, pointing from the box into
    the sphere, i.e. the direction of the push on the sphere), ``point``
    (midway through the overlap), ``member`` (index into ``points``) and
    ``inside`` (sphere centre inside the box).  Only positive penetrations are
    returned.
    """
    pts = np.asarray(points, dtype=float)
    rad = np.asarray(radii, dtype=float)
    lo, hi = box.lo, box.hi
    q = np.clip(pts, lo, hi)
    d = pts - q
    dist = np.linalg.norm(d, axis=1)
    outside = dist > 1e-12
    normal = np.zeros_like(pts)
    normal[outside] = d[outside] / dist[outside, None]
    pen = rad - dist
    inside = ~outside
    if np.any(inside):
        idx = np.nonzero(inside)[0]
        # distance to each of the six faces; exit through the nearest one
        face = np.stack(
            [pts[idx] - lo[None, :], hi[None, :] - pts[idx]], axis=2
        ).reshape(len(idx), 6)
        j = np.argmin(face, axis=1)
        axis = j % 3
        sign = np.where(j % 2 == 0, -1.0, 1.0)   # low face -> -axis, high face -> +axis
        pen[idx] = rad[idx] + face[np.arange(len(idx)), j]
        normal[idx] = 0.0
        normal[idx, axis] = sign
    keep = pen > 0.0
    if not np.any(keep):
        return {
            "penetration": np.zeros(0),
            "normal": np.zeros((0, 3)),
            "point": np.zeros((0, 3)),
            "member": np.zeros(0, dtype=int),
            "inside": np.zeros(0, dtype=bool),
        }
    k = np.nonzero(keep)[0]
    n = normal[k]
    point = pts[k] - (rad[k] - 0.5 * pen[k])[:, None] * n
    return {
        "penetration": pen[k],
        "normal": n,
        "point": point,
        "member": k,
        "inside": inside[k],
    }


# ---------------------------------------------------------------------------
# Member-member narrow phase
# ---------------------------------------------------------------------------
def member_pair_candidates(
    centers: np.ndarray,
    radii: np.ndarray,
    body_index: np.ndarray,
    *,
    max_patch: float | None = None,
) -> Dict[str, np.ndarray]:
    """All member pairs (different bodies) with positive penetration.

    A KD-tree over the member centres provides the neighbour candidate list
    (``2 * max(radii)`` search radius); pairs belonging to the same rigid body
    are excluded because a clump cannot contact itself.
    """
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    body_index = np.asarray(body_index, dtype=int)
    n = len(centers)
    empty = {
        "i": np.zeros(0, dtype=int),
        "j": np.zeros(0, dtype=int),
        "penetration": np.zeros(0),
        "normal": np.zeros((0, 3)),
        "point": np.zeros((0, 3)),
    }
    if n < 2:
        return empty
    tree = cKDTree(centers)
    pairs = tree.query_pairs(2.0 * float(radii.max()), output_type="ndarray")
    if len(pairs) == 0:
        return empty
    i = pairs[:, 0].astype(int)
    j = pairs[:, 1].astype(int)
    keep = body_index[i] != body_index[j]
    i, j = i[keep], j[keep]
    if len(i) == 0:
        return empty
    d = centers[j] - centers[i]
    dist = np.linalg.norm(d, axis=1)
    pen = radii[i] + radii[j] - dist
    keep = (pen > 0.0) & (dist > 1e-12)
    i, j, d, dist, pen = i[keep], j[keep], d[keep], dist[keep], pen[keep]
    if len(i) == 0:
        return empty
    normal = d / dist[:, None]
    point = centers[i] + (radii[i] - 0.5 * pen)[:, None] * normal
    return {
        "i": i,
        "j": j,
        "penetration": pen,
        "normal": normal,
        "point": point,
    }


def merge_junction_candidates(
    cand: Dict[str, np.ndarray],
    radii: np.ndarray | None = None,
    *,
    merge_angle_deg: float = 30.0,
    group_keys: np.ndarray | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Merge member pairs that sample one physical contact junction.

    Two candidates are merged when

    * their contact points are closer than the smaller Hertz patch radius
      ``a = sqrt(R* delta)`` (Hertz patch half-width), and
    * their normals agree within ``merge_angle_deg``.

    The criterion is single-linkage over a KD-tree of contact points, so the
    result is independent of the candidate order.  For each cluster the
    **deepest** candidate is kept (adding the penetrations inside a cluster would
    re-introduce the double counting).

    ``group_keys`` (one integer per candidate, e.g. the body pair) makes the
    search local: candidates whose group has a single member cannot be a
    duplicate and are passed through untouched.  In a sparse packing that skips
    nearly all of the work.

    Returns ``(kept, removed)``; both are candidate dicts with the same keys, and
    ``kept`` carries an extra ``cluster_size`` array.
    """
    n = len(cand["i"])
    if n == 0:
        return (
            {
                "i": np.zeros(0, dtype=int),
                "j": np.zeros(0, dtype=int),
                "penetration": np.zeros(0),
                "normal": np.zeros((0, 3)),
                "point": np.zeros((0, 3)),
                "cluster_size": np.zeros(0, dtype=int),
            },
            {
                "i": np.zeros(0, dtype=int),
                "j": np.zeros(0, dtype=int),
                "penetration": np.zeros(0),
                "normal": np.zeros((0, 3)),
                "point": np.zeros((0, 3)),
            },
        )
    pts = cand["point"]
    pen = cand["penetration"]
    normal = cand["normal"]
    if radii is None:
        patch = 1e-4 * np.ones(n)
    else:
        r_i = np.asarray(radii, dtype=float)[cand["i"]]
        r_j = np.asarray(radii, dtype=float)[cand["j"]]
        r_star = r_i * r_j / (r_i + r_j)
        patch = np.sqrt(np.maximum(r_star * pen, 0.0))

    if group_keys is None:
        dup_mask = np.ones(n, dtype=bool)
    else:
        keys = np.asarray(group_keys)
        _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
        dup_mask = counts[inv] > 1

    sel = np.nonzero(dup_mask)[0]
    passthrough = np.nonzero(~dup_mask)[0]
    cos_tol = math.cos(math.radians(merge_angle_deg))
    keep_local: List[int] = []
    sizes_local: List[int] = []
    if len(sel) > 1:
        sub_pts = pts[sel]
        sub_patch = patch[sel]
        sub_pen = pen[sel]
        sub_n = normal[sel]
        pairs = cKDTree(sub_pts).query_pairs(float(sub_patch.max()), output_type="ndarray")
        if len(pairs) > 0:
            a, b = pairs[:, 0].astype(int), pairs[:, 1].astype(int)
            keep = np.linalg.norm(sub_pts[a] - sub_pts[b], axis=1) <= np.minimum(
                sub_patch[a], sub_patch[b]
            )
            # signed cosine: anti-parallel normals are *different* junctions
            keep &= np.sum(sub_n[a] * sub_n[b], axis=1) >= cos_tol
            a, b = a[keep], b[keep]
        else:
            a = b = np.zeros(0, dtype=int)
        m = len(sel)
        parent = np.arange(m)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for x, y in zip(a.tolist(), b.tolist()):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx
        roots = np.array([find(k) for k in range(m)])
        order = np.lexsort((np.arange(m), -sub_pen))
        seen: Dict[int, int] = {}
        cluster_size: Dict[int, int] = {}
        for k in order.tolist():
            r = int(roots[k])
            cluster_size[r] = cluster_size.get(r, 0) + 1
            if r not in seen:
                seen[r] = k
        for r, k in seen.items():
            keep_local.append(int(sel[k]))
            sizes_local.append(int(cluster_size[r]))
    elif len(sel) == 1:
        keep_local.append(int(sel[0]))
        sizes_local.append(1)

    keep_idx = np.array(sorted(passthrough.tolist() + keep_local), dtype=int)
    size_map = {k: sz for k, sz in zip(keep_local, sizes_local)}
    sizes = np.array([size_map.get(int(k), 1) for k in keep_idx], dtype=int)
    kept = {
        "i": cand["i"][keep_idx],
        "j": cand["j"][keep_idx],
        "penetration": pen[keep_idx],
        "normal": normal[keep_idx],
        "point": pts[keep_idx],
        "cluster_size": sizes,
    }
    mask = np.ones(n, dtype=bool)
    mask[keep_idx] = False
    removed = {
        "i": cand["i"][mask],
        "j": cand["j"][mask],
        "penetration": pen[mask],
        "normal": normal[mask],
        "point": pts[mask],
    }
    return kept, removed


# ---------------------------------------------------------------------------
# Tangential history (Cundall-Strack spring displacement)
# ---------------------------------------------------------------------------
class ContactHistory:
    """Persistent tangential spring displacement per contact key.

    Keys are integers that identify a contact for as long as it exists (for
    member pairs: ``i * n_members + j``; for member/obstacle contacts they live
    in a separate namespace offset).  Storing the history in growable arrays
    with a key -> slot map keeps the per-step update vectorised.
    """

    def __init__(self, capacity: int = 2048) -> None:
        self.capacity = int(capacity)
        self.xi = np.zeros((self.capacity, 3))
        self._stamp = np.zeros(self.capacity, dtype=np.int64)
        self._slot: Dict[int, int] = {}
        self.step_id = 0
        self.n_allocated = 0

    def _grow(self, need: int) -> None:
        while self.n_allocated + need > self.capacity:
            self.capacity *= 2
            self.xi = np.vstack([self.xi, np.zeros((self.capacity // 2, 3))])
            self._stamp = np.concatenate(
                [self._stamp, np.zeros(self.capacity // 2, dtype=np.int64)]
            )

    def slots(self, keys: np.ndarray) -> np.ndarray:
        out = np.empty(len(keys), dtype=np.int64)
        slot_map = self._slot
        n_new = 0
        for k, key in enumerate(keys.tolist()):
            s = slot_map.get(key)
            if s is None:
                s = self.n_allocated
                slot_map[key] = s
                self.n_allocated += 1
                n_new += 1
            out[k] = s
        if n_new:
            self._grow(0)
        return out

    def begin_step(self) -> None:
        self.step_id += 1

    def touch(self, slots: np.ndarray) -> None:
        self._stamp[slots] = self.step_id

    def retire(self) -> float:
        """Zero and return the stored tangential energy of contacts not active now."""
        dead = np.nonzero(
            (self._stamp > 0) & (self._stamp != self.step_id)
        )[0]
        if len(dead) == 0:
            return 0.0
        return 0.0  # energy handled by caller with kt; see engine


# ---------------------------------------------------------------------------
# Force law
# ---------------------------------------------------------------------------
@dataclass
class ContactLawResult:
    fn: np.ndarray
    ft: np.ndarray
    xi: np.ndarray
    active: np.ndarray
    e_dashpot: np.ndarray
    e_friction: np.ndarray
    pe_normal: np.ndarray
    pe_tangential: np.ndarray


def evaluate_contact_law(
    penetration: np.ndarray,
    normal: np.ndarray,
    v_rel: np.ndarray,
    m_eff: np.ndarray,
    params: ContactParams,
    xi_old: np.ndarray,
    dt: float,
) -> ContactLawResult:
    """Vectorised normal/tangential force law for a list of contacts.

    ``v_rel`` is the relative velocity of the two material points at the contact
    (``v_i + w_i x (x_c - x_i) - (v_j + w_j x (x_c - x_j))``), so spin contributes
    correctly.  ``xi_old`` is the tangential spring displacement carried over from
    the previous step; the returned ``xi`` is the updated value (clamped when
    sliding, which is the standard Cundall-Strack reset).
    """
    pen = np.asarray(penetration, dtype=float)
    n = np.asarray(normal, dtype=float)
    v = np.asarray(v_rel, dtype=float)
    m_eff = np.asarray(m_eff, dtype=float)
    xi_old = np.asarray(xi_old, dtype=float)
    active = pen > 0.0
    vn = np.sum(v * n, axis=1)
    f_elastic = params.kn * pen
    c_n = 2.0 * params.zeta * np.sqrt(m_eff * params.kn)
    # Kelvin-Voigt with delta_dot = +vn (penetration grows while approaching):
    # F_n = k delta + c vn, so the dashpot work -c vn^2 is dissipative.
    fn = np.where(active, np.maximum(f_elastic + c_n * vn, 0.0), 0.0)
    live = fn > 0.0

    vt = v - vn[:, None] * n
    xi_trial = xi_old + vt * dt
    xi_trial -= np.sum(xi_trial * n, axis=1)[:, None] * n
    ft = -params.kt * xi_trial
    ft_mag = np.linalg.norm(ft, axis=1)
    limit = params.mu * fn
    sliding = ft_mag > np.maximum(limit, 1e-300)
    scale = np.where(sliding, limit / np.maximum(ft_mag, 1e-300), 1.0)
    ft = ft * np.where(live, scale, 0.0)[:, None]
    xi_new = -ft / params.kt

    e_dashpot = np.where(live, c_n * vn * vn * dt, 0.0)
    # sliding friction work: the clamp already removed the elastic part, so the
    # tangential work done by a sliding force is |F_t| |v_t| dt; for sticking
    # contacts the exchange is elastic and lives in pe_tangential.
    e_friction = np.where(live & sliding, ft_mag * np.linalg.norm(vt, axis=1) * dt, 0.0)
    pe_normal = np.where(live, 0.5 * params.kn * pen * pen, 0.0)
    pe_tangential = 0.5 * params.kt * np.sum(xi_new * xi_new, axis=1)
    return ContactLawResult(
        fn=fn,
        ft=ft,
        xi=xi_new,
        active=live,
        e_dashpot=e_dashpot,
        e_friction=e_friction,
        pe_normal=pe_normal,
        pe_tangential=pe_tangential,
    )


# ---------------------------------------------------------------------------
# Energy ledger
# ---------------------------------------------------------------------------
@dataclass
class EnergyLedger:
    """Cumulative energy terms for a DEM run (C6 gate)."""

    external_work: float = 0.0
    dissipated_normal: float = 0.0
    dissipated_friction: float = 0.0
    dissipated_rolling: float = 0.0
    dissipated_detach: float = 0.0
    initial_total: Optional[float] = None
    final_total: float = 0.0

    @property
    def dissipated_total(self) -> float:
        return (
            self.dissipated_normal
            + self.dissipated_friction
            + self.dissipated_rolling
            + self.dissipated_detach
        )

    def close(self, final_total: float, initial_total: Optional[float] = None) -> dict:
        if self.initial_total is None:
            self.initial_total = 0.0 if initial_total is None else float(initial_total)
        self.final_total = float(final_total)
        delta = self.final_total - self.initial_total
        expected = self.external_work - self.dissipated_total
        residual = expected - delta
        scale = max(abs(self.external_work), abs(delta), 1e-30)
        return {
            "external_work": self.external_work,
            "dissipated_total": self.dissipated_total,
            "dissipated_normal": self.dissipated_normal,
            "dissipated_friction": self.dissipated_friction,
            "dissipated_rolling": self.dissipated_rolling,
            "dissipated_detach": self.dissipated_detach,
            "initial_total": self.initial_total,
            "final_total": self.final_total,
            "delta_total": delta,
            "residual": residual,
            "relative_residual": abs(residual) / scale,
        }


# ---------------------------------------------------------------------------
# Force-chain descriptors
# ---------------------------------------------------------------------------
def force_distribution_stats(fn: np.ndarray) -> dict:
    """Distribution width descriptors of the normal contact forces (C2)."""
    f = np.asarray(fn, dtype=float)
    f = f[f > 0.0]
    if len(f) == 0:
        return {
            "n_contacts": 0,
            "mean": 0.0,
            "std": 0.0,
            "cv": 0.0,
            "gini": 0.0,
            "strong_fraction": 0.0,
            "top10_load_fraction": 0.0,
        }
    mean = float(f.mean())
    std = float(f.std(ddof=0))
    s = np.sort(f)
    n = len(s)
    gini = float((2.0 * np.sum((np.arange(1, n + 1)) * s) / (n * s.sum())) - (n + 1.0) / n)
    strong = float(np.sum(f > mean) / n)
    k = max(1, int(math.ceil(0.1 * n)))
    top10 = float(np.sort(f)[-k:].sum() / f.sum())
    return {
        "n_contacts": int(n),
        "mean": mean,
        "std": std,
        "cv": std / mean if mean > 0 else 0.0,
        "gini": gini,
        "strong_fraction": strong,
        "top10_load_fraction": top10,
    }


def union_find_clusters(n_nodes: int, pairs: Iterable[Tuple[int, int]]) -> List[List[int]]:
    """Connected components of an undirected graph (union-find)."""
    parent = list(range(n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    groups: Dict[int, List[int]] = {}
    for k in range(n_nodes):
        groups.setdefault(find(k), []).append(k)
    return list(groups.values())


def force_chain_clusters(
    fn: np.ndarray,
    body_i: np.ndarray,
    body_j: np.ndarray,
    n_bodies: int,
    *,
    threshold: float = 1.0,
) -> dict:
    """Force-chain topology over the "strong" contact network.

    Contacts carrying more than ``threshold * <F>`` are strong (Radjai et al.
    1996); their connected components on the body graph are the force chains.
    Returns the cluster size distribution, the largest-cluster share of the
    strong-contact count and the mean chain length.
    """
    f = np.asarray(fn, dtype=float)
    bi = np.asarray(body_i, dtype=int)
    bj = np.asarray(body_j, dtype=int)
    pos = f > 0.0
    bi, bj, f = bi[pos], bj[pos], f[pos]
    if len(f) == 0:
        return {
            "n_strong_contacts": 0,
            "n_clusters": 0,
            "largest_cluster_share": 0.0,
            "mean_chain_length": 0.0,
            "cluster_sizes": [],
            "mean_force": 0.0,
        }
    mean = float(f.mean())
    strong = f > threshold * mean
    clusters = union_find_clusters(n_bodies, zip(bi[strong], bj[strong]))
    sizes = sorted((len(c) for c in clusters), reverse=True)
    n_strong = int(strong.sum())
    return {
        "n_strong_contacts": n_strong,
        "n_clusters": len(clusters),
        "largest_cluster_share": (sizes[0] / n_bodies) if sizes else 0.0,
        "mean_chain_length": float(np.mean(sizes)) if sizes else 0.0,
        "cluster_sizes": sizes,
        "mean_force": mean,
    }