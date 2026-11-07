"""Periodic Laguerre (power-diagram) tessellation with prescribed cell areas
(IN32-N01).

The wave-1 floe-field generator :mod:`tensordem.fsd_packing` represents every
floe by a **circle** of equal area on a size-graded lattice.  Real fragmented ice
covers are *polygonal*: floes are the fragments of a broken plate, they interlock
along straight edges and those edges carry the contact forces and the torques.
This module supplies the geometric substrate of a polygonal floe field: a
**Laguerre tessellation** (power diagram) of a periodic rectangle whose convex
cells have areas prescribed by the floe-size distribution (FSD).

Definition
----------
With sites :math:`x_i \\in \\mathbb{R}^2` and weights :math:`w_i` the power
(Laguerre) cell of site ``i`` is

.. math::
    C_i = \\{ y : |y - x_i|^2 - w_i \\le |y - x_j|^2 - w_j \\;\\forall j \\ne i \\}.

Every inequality is a half-plane.  In coordinates :math:`y' = y - x_i` and with
:math:`d = x_j' - x_i` (image of ``j`` seen from ``i``) it reads

.. math::
    d \\cdot y' \\le \\tfrac12 \\bigl(|d|^2 + w_i - w_j\\bigr),

so a cell is the intersection of neighbour half-planes: a convex polygon (a
quadrilateral for four independent half-planes, up to ~8 edges for random
sites).  The cells tile the torus exactly -- their interiors are disjoint and
their areas sum to ``Lx Ly`` (:attr:`LaguerreCells.cover`, an internal
consistency check that must equal 1 to machine precision).

Periodic box and polygon storage
--------------------------------
Neighbours are searched among the 3x3 periodic images, so no cell is cut by a
spurious "wall" plane: the cell is built in the *infinite periodic plane* and a
half-box window centred on its site only acts as a safety bound.  Each polygon is
stored in **centroid + local-vertex** form with the centroid folded into the box
(on the torus a cell may straddle the seam, which is harmless because every
distance test uses the minimum-image displacement, see
:mod:`tensordem.polygon_contact`).

Prescribed areas
----------------
The FSD gives a target area :math:`A_i` per floe.  The floe field is *buffered*
by a plateau gap (every cell is winched inwards by :math:`s = \\sqrt{\\phi}`
about its own centroid), so the tessellation is asked for

.. math::   T_i = A_i / \\phi, \\qquad \\sum_i T_i = L_x L_y,

i.e. the cells fill the box and every cell shrunk by ``s`` has exactly the
prescribed area ``A_i``.

The weights are recovered by a **Newton iteration on the exact area Jacobian**.
For an edge of length :math:`\\ell_{ij}` lying on the bisector of the pair
``(i, j)``, shifting the plane by :math:`\\delta w/(2|d_{ij}|)` sweeps an area
``\\ell_{ij} \\delta w / (2|d_{ij}|)``, hence

.. math::
    \\frac{\\partial A_i}{\\partial w_i} = \\sum_j \\frac{\\ell_{ij}}{2|d_{ij}|},
    \\qquad \\frac{\\partial A_i}{\\partial w_j} = -\\frac{\\ell_{ij}}{2|d_{ij}|},

which is a (singular) **graph Laplacian** -- the classical semi-discrete optimal
transport Hessian.  One sparse solve per iteration replaces hundreds of fixed
point sweeps, and a backtracking line search keeps the iteration inside the
regime where the diagram is well posed.  Weights are initialised from the local
plane-balance estimate :math:`w_i \\approx 1.074\\sqrt{T_i \\bar T} - \\bar T`
(a hexagon of area ``T`` has apothem ``0.537 sqrt(T)``, so the bisector must sit
at distance ``0.537 sqrt(T)`` from the site).

Neighbour relevance
-------------------
Sites are pre-selected from a KD-tree pool and then ordered by **plane distance**
``|0.5 |d| + (w_i - w_j)/(2|d|)|``, not by Euclidean distance: a large cell is
bounded by far-away large floes whose bisectors are close to its site, so
Euclidean ordering needs a much larger ``k``.  If the resulting tessellation does
not cover the box to ``1e-9`` the neighbour count is automatically increased
(:func:`laguerre_cells`), which makes the construction self-validating: a
non-unit cover is a hard error, never a silent overlap.

All numerics are float64 NumPy.  The batched Sutherland-Hodgman clipper
(:func:`_clip_batch`) advances all ``N`` cells simultaneously, so a tessellation
costs ``O(N k)`` vectorised work instead of ``O(N^2)`` Python loops.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "LaguerreParams",
    "LaguerreCells",
    "jittered_sites",
    "laguerre_cells",
    "laguerre_prescribed_areas",
]


# ---------------------------------------------------------------------------
# parameters / containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LaguerreParams:
    """Solver controls for the prescribed-area power diagram."""

    n_neighbours: int = 16
    max_iter: int = 60
    area_tol: float = 2.0e-3
    pool: Optional[int] = None

    def __post_init__(self) -> None:
        if int(self.n_neighbours) < 6:
            raise ValueError("n_neighbours must be at least 6")
        if int(self.max_iter) < 1:
            raise ValueError("max_iter must be positive")
        if not (0.0 < float(self.area_tol) < 1.0):
            raise ValueError("area_tol must lie in (0, 1)")
        if self.pool is not None and int(self.pool) < int(self.n_neighbours):
            raise ValueError("pool must be at least n_neighbours")


@dataclass
class LaguerreCells:
    """A tessellation of the periodic box.

    ``verts`` are **local** vertices relative to ``centroid`` (shape
    ``(N, cap, 2)``; only the first ``count[i]`` rows are meaningful and they are
    ordered counter-clockwise).  ``centroid`` is folded into the box.
    """

    box: Tuple[float, float]
    sites: np.ndarray
    weights: np.ndarray
    verts: np.ndarray
    count: np.ndarray
    area: np.ndarray
    perimeter: np.ndarray
    centroid: np.ndarray
    cell_centroid_site: np.ndarray      # cell centroid in site-relative coords
    iterations: int = 0
    area_error: float = 0.0
    cover: float = 1.0
    n_neighbours: int = 0

    @property
    def n(self) -> int:
        return int(self.sites.shape[0])

    @property
    def degenerate(self) -> int:
        return int((self.count < 3).sum())

    def world_vertices(self, index: int) -> np.ndarray:
        """Vertices of cell ``index`` in world (box) coordinates."""
        i = int(index)
        return self.verts[i, : self.count[i]] + self.centroid[i]

    def equivalent_radius(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.area, 0.0) / math.pi)

    def summary(self) -> Dict[str, object]:
        return {
            "n": self.n,
            "box": [float(self.box[0]), float(self.box[1])],
            "iterations": int(self.iterations),
            "area_error_max_rel": float(self.area_error),
            "cover": float(self.cover),
            "degenerate_cells": self.degenerate,
            "n_neighbours": int(self.n_neighbours),
            "area_min": float(self.area.min()),
            "area_max": float(self.area.max()),
            "vertices_mean": float(np.mean(self.count)),
        }


# ---------------------------------------------------------------------------
# site sets
# ---------------------------------------------------------------------------

def jittered_sites(n: int, box: Tuple[float, float], *, seed: int = 0,
                   jitter: float = 0.6) -> np.ndarray:
    """``n`` disordered sites in the box from a jittered hexagonal lattice.

    A pure Poisson sample produces sliver cells (bad conditioning of the area
    solve and a degenerate contact graph); a jittered lattice keeps a controlled
    minimum separation while removing the lattice symmetry, exactly like the
    jittered rows of the wave-1 generator.
    """
    n = int(n)
    if n < 2:
        raise ValueError("n must be at least 2")
    lx, ly = float(box[0]), float(box[1])
    if not (lx > 0.0 and ly > 0.0) or not math.isfinite(lx + ly):
        raise ValueError("box must be finite and positive")
    if not (0.0 <= jitter < 1.0) or not math.isfinite(jitter):
        raise ValueError("jitter must lie in [0, 1)")

    aspect = lx / ly
    ny = max(1, int(round(math.sqrt(n / (aspect * 0.8660254037844386)))))
    nx = max(1, int(math.ceil(n / ny)))
    while nx * ny < n:
        ny += 1
        nx = int(math.ceil(n / ny))
    sx, sy = lx / nx, ly / ny
    pts = []
    for row in range(ny):
        shift = 0.5 * sx if (row % 2) else 0.0
        for col in range(nx):
            pts.append(((col + 0.5) * sx + shift, (row + 0.5) * sy))
    pts = np.asarray(pts, dtype=np.float64)[:n]
    pts[:, 0] = np.mod(pts[:, 0], lx)
    if jitter > 0.0:
        rng = np.random.default_rng(int(seed))
        pts += (rng.random(pts.shape) - 0.5) * jitter * np.array([sx, sy])
    pts[:, 0] = np.mod(pts[:, 0], lx)
    pts[:, 1] = np.mod(pts[:, 1], ly)
    return pts


# ---------------------------------------------------------------------------
# batched Sutherland-Hodgman clipping of convex polygons
# ---------------------------------------------------------------------------

def _next_vertex(poly: np.ndarray, dot: Optional[np.ndarray], count: np.ndarray
                 ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Successor vertex of every polygon slot, wrapping past the last valid one.

    ``np.roll`` cannot be used because the tail of the capacity is padding: the
    successor of slot ``count[i] - 1`` is slot ``0``, not ``count[i]``.
    """
    cap = poly.shape[1]
    safe = np.maximum(count, 1)[:, None]
    m_idx = np.arange(cap)[None, :]
    nxt_idx = np.where(m_idx + 1 >= safe, m_idx + 1 - safe, m_idx + 1)
    rows = np.arange(poly.shape[0])[:, None]
    nxt = poly[rows, nxt_idx]
    dot_nxt = None if dot is None else dot[rows, nxt_idx]
    return nxt, dot_nxt


def _clip_batch(poly: np.ndarray, count: np.ndarray, a: np.ndarray,
                b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Clip every polygon of the batch by the half-plane ``a . y <= b``.

    ``poly`` has shape ``(N, cap, 2)`` (ordered vertices, ``count[i]`` valid) and
    the planes are per-polygon (``(N, 2)`` / ``(N,)``).  A convex polygon cut by
    one line gains at most two vertices (one per crossing), so ``count_out <=
    count + 2``; the caller reserves slack.
    """
    n, cap, _ = poly.shape
    m_idx = np.arange(cap)[None, :]
    valid = m_idx < count[:, None]
    dot = np.einsum("nmd,nd->nm", poly, a) - b[:, None]
    inside = dot <= 0.0
    nxt, dot_nxt = _next_vertex(poly, dot, count)
    keep = valid & inside
    cross = valid & (inside != (dot_nxt <= 0.0))
    denom = dot - dot_nxt
    safe = np.where(np.abs(denom) > 0.0, denom, 1.0)
    t = np.where(np.abs(denom) > 0.0, dot / safe, 0.0)
    inter = poly + t[..., None] * (nxt - poly)

    keep_before = np.cumsum(keep, axis=1) - keep
    cross_before = np.cumsum(cross, axis=1) - cross
    pos_keep = keep_before + cross_before
    pos_cross = pos_keep + keep

    out = np.zeros_like(poly)
    r, m = np.nonzero(keep)
    out[r, pos_keep[r, m]] = poly[r, m]
    r, m = np.nonzero(cross)
    out[r, pos_cross[r, m]] = inter[r, m]
    count_out = (keep.sum(axis=1) + cross.sum(axis=1)).astype(np.int64)
    if count_out.size and int(count_out.max()) > cap:
        raise RuntimeError("polygon clipping exceeded capacity")
    return out, count_out


def _cell_moments(poly: np.ndarray, count: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area, area centroid (local coordinates) and perimeter of the batch."""
    cap = poly.shape[1]
    m_idx = np.arange(cap)[None, :]
    mask = m_idx < count[:, None]
    nxt, _ = _next_vertex(poly, None, count)
    cross = poly[..., 0] * nxt[..., 1] - poly[..., 1] * nxt[..., 0]
    cross = np.where(mask, cross, 0.0)
    signed = 0.5 * cross.sum(axis=1)
    orient = np.where(signed < 0.0, -1.0, 1.0)
    area = np.abs(signed)
    safe = np.where(area > 0.0, area, 1.0)
    cx = (np.where(mask, (poly[..., 0] + nxt[..., 0]) * cross, 0.0)).sum(axis=1)
    cy = (np.where(mask, (poly[..., 1] + nxt[..., 1]) * cross, 0.0)).sum(axis=1)
    local = np.stack([orient * cx / (6.0 * safe), orient * cy / (6.0 * safe)], axis=1)
    seg = np.linalg.norm(nxt - poly, axis=2)
    perimeter = np.where(mask, seg, 0.0).sum(axis=1)
    degenerate = count < 3
    area = np.where(degenerate, 0.0, area)
    local = np.where(degenerate[:, None], 0.0, local)
    perimeter = np.where(degenerate, 0.0, perimeter)
    return area, local, perimeter


# ---------------------------------------------------------------------------
# neighbour planes
# ---------------------------------------------------------------------------

_CAND_CACHE: Dict[tuple, tuple] = {}


def _image_candidates(sites: np.ndarray, box: Tuple[float, float], pool: int
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cached ``pool`` nearest periodic images of every site (Euclidean)."""
    key = (round(float(box[0]), 9), round(float(box[1]), 9), int(pool),
           sites.shape[0], float(sites.sum()), float((sites * sites).sum()))
    hit = _CAND_CACHE.get(key)
    if hit is not None:
        return hit
    n = sites.shape[0]
    offsets = np.array([[i * box[0], j * box[1]]
                        for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=np.float64)
    images = (sites[None, :, :] + offsets[:, None, :]).reshape(-1, 2)
    owner = np.tile(np.arange(n), offsets.shape[0])
    tree = cKDTree(images)
    _, idx = tree.query(sites, k=min(pool + 1, images.shape[0]))
    nb_owner = owner[idx[:, 1:]]
    nb_vec = images[idx[:, 1:]] - sites[:, None, :]
    dist = np.sqrt(np.maximum(np.einsum("nkd,nkd->nk", nb_vec, nb_vec), 1e-300))
    _CAND_CACHE.clear()
    _CAND_CACHE[key] = (nb_vec, nb_owner, dist)
    return nb_vec, nb_owner, dist


def _neighbour_planes(sites: np.ndarray, weights: np.ndarray,
                      box: Tuple[float, float], k: int, *,
                      pool: Optional[int] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Half-plane data ``(a, b, owner)`` of the ``k`` most relevant neighbours.

    Candidates are the ``pool`` nearest periodic images (Euclidean, cached across
    weight updates); they are then ordered by **plane distance** ``|0.5|d| +
    (w_i - w_j)/(2|d|)|``.  Euclidean order is wrong for polydisperse fields -- a
    large cell is bounded by far-away large floes whose bisectors are close to its
    site -- so plane ordering is what keeps a small ``k`` sufficient.
    """
    n = sites.shape[0]
    pool = int(pool if pool is not None else max(2 * k, 48))
    pool = int(min(max(pool, k + 1), max(n - 1, 1)))
    nb_vec, nb_owner, dist = _image_candidates(sites, box, pool)
    plane = 0.5 * dist + (weights[:, None] - weights[nb_owner]) / (2.0 * dist)
    plane = np.where(np.isfinite(plane), np.abs(plane), np.inf)
    order = np.argsort(plane, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    nb_vec = nb_vec[rows, order]
    nb_owner = nb_owner[rows, order]
    b = 0.5 * (np.einsum("nkd,nkd->nk", nb_vec, nb_vec)
               + weights[:, None] - weights[nb_owner])
    return nb_vec, b, nb_owner


# ---------------------------------------------------------------------------
# tessellation
# ---------------------------------------------------------------------------

def _tessellate(sites: np.ndarray, weights: np.ndarray, box: Tuple[float, float],
                k: int, pool: Optional[int]) -> LaguerreCells:
    n = sites.shape[0]
    lx, ly = float(box[0]), float(box[1])
    a, b, _owner = _neighbour_planes(sites, weights, (lx, ly), k, pool=pool)

    cap = 2 * k + 16
    poly = np.zeros((n, cap, 2), dtype=np.float64)
    hx, hy = 0.5 * lx, 0.5 * ly
    poly[:, 0] = (-hx, -hy)
    poly[:, 1] = (hx, -hy)
    poly[:, 2] = (hx, hy)
    poly[:, 3] = (-hx, hy)
    count = np.full(n, 4, dtype=np.int64)

    def attach(p: np.ndarray, c: np.ndarray) -> np.ndarray:
        # Work on a view whose capacity is the live vertex count plus slack so
        # the clipper touches the minimum number of elements (a cut adds <= 1
        # vertex, a spike up to 2, hence slack 6).
        need = int(c.max()) + 6 if c.size else 6
        if need <= p.shape[1]:
            return p[:, :need]
        big = np.zeros((p.shape[0], need, 2), dtype=np.float64)
        big[:, : p.shape[1]] = p
        return big

    # Plane distance of every neighbour: a plane whose (signed) distance from the
    # site exceeds the current polygon circumradius cannot cut the cell, which is
    # what lets finished cells leave the batch below.
    alen = np.linalg.norm(a, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_plane = 0.5 * alen + (weights[:, None] - weights[_owner]) / (2.0 * alen)
    s_plane = np.where(np.isfinite(s_plane), s_plane, -np.inf)

    idx = np.arange(n)
    cap0 = poly.shape[1]
    active_poly, active_count = poly, count
    poly_full = np.zeros((n, cap0, 2), dtype=np.float64)
    count_full = np.zeros(n, dtype=np.int64)
    for j in range(k):
        active_poly, active_count = _clip_batch(
            attach(active_poly, active_count), active_count,
            a[idx, j, :], b[idx, j])
        if j + 1 < k and active_poly.shape[0] > 16 and (j % 8 == 7):
            cap = active_poly.shape[1]
            valid = np.arange(cap)[None, :] < active_count[:, None]
            pts = np.where(valid[:, :, None], active_poly, 0.0)
            radius = np.sqrt((pts * pts).sum(axis=2).max(axis=1))
            remaining = s_plane[idx, j + 1:]
            finished = np.all(remaining > 1.05 * radius[:, None], axis=1)
            if finished.any() and not finished.all():
                live = ~finished
                done = idx[finished]
                poly_full[done, : active_poly.shape[1]] = active_poly[finished]
                count_full[done] = active_count[finished]
                idx = idx[live]
                active_poly = active_poly[live]
                active_count = active_count[live]
                k_active = int(np.maximum(active_count.max(), 4)) + 6
                active_poly = active_poly[:, : min(k_active, active_poly.shape[1])]
    poly_full[idx, : active_poly.shape[1]] = active_poly
    count_full[idx] = active_count
    poly, count = poly_full, count_full
    area, local, perimeter = _cell_moments(poly, count)
    m_idx = np.arange(poly.shape[1])[None, :, None]
    poly = np.where(m_idx < count[:, None, None], poly - local[:, None, :], 0.0)
    centroid = sites + local
    centroid[:, 0] = np.mod(centroid[:, 0], lx)
    centroid[:, 1] = np.mod(centroid[:, 1], ly)
    cells = LaguerreCells(
        box=(lx, ly), sites=sites, weights=weights, verts=poly, count=count,
        area=area, perimeter=perimeter, centroid=centroid,
        cell_centroid_site=local, n_neighbours=int(k),
    )
    cells.cover = float(area.sum() / (lx * ly))
    return cells


def laguerre_cells(sites: np.ndarray, weights: Optional[np.ndarray],
                   box: Tuple[float, float], *, n_neighbours: int = 16,
                   pool: Optional[int] = None, expand: bool = True
                   ) -> LaguerreCells:
    """Tessellate the periodic box for the given sites and weights.

    If the achieved box coverage differs from 1 by more than ``1e-6`` the
    neighbour count is doubled and the tessellation is rebuilt: a non-unit cover
    means the finite neighbour list over-claimed area, i.e. a silent overlap.  A
    ``RuntimeError`` is raised if even the full candidate pool cannot restore
    exact coverage -- never a silently overlapping field.
    """
    sites = np.asarray(sites, dtype=np.float64)
    if sites.ndim != 2 or sites.shape[1] != 2:
        raise ValueError("sites must have shape (N, 2)")
    if not np.isfinite(sites).all():
        raise ValueError("sites must be finite")
    n = sites.shape[0]
    lx, ly = float(box[0]), float(box[1])
    if not (lx > 0.0 and ly > 0.0) or not math.isfinite(lx + ly):
        raise ValueError("box must be finite and positive")
    if weights is None:
        weights = np.zeros(n, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (n,):
        raise ValueError("weights must have shape (N,)")
    if not np.isfinite(weights).all():
        raise ValueError("weights must be finite")

    k = int(min(max(int(n_neighbours), 6), max(n - 1, 6)))
    k_cap = max(n - 1, 6)
    cells = None
    while True:
        cells = _tessellate(sites, weights, (lx, ly), k, pool)
        if not expand or abs(cells.cover - 1.0) <= 1e-6 or k >= k_cap:
            break
        k = int(min(2 * k, k_cap))
        pool = None if pool is None else max(2 * pool, 2 * k)
    if abs(cells.cover - 1.0) > 1e-3:
        raise RuntimeError(
            f"Laguerre tessellation does not cover the box (cover={cells.cover:.6f}, "
            f"n={n}, n_neighbours={k}): weights are too disparate for a local "
            f"neighbour list")
    return cells


# ---------------------------------------------------------------------------
# prescribed-area solve
# ---------------------------------------------------------------------------

def _edge_flux(cells: LaguerreCells, sites: np.ndarray, weights: np.ndarray,
               box: Tuple[float, float], n_neighbours: int,
               pool: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared-edge fluxes ``l_ij / (2 |d_ij|)`` that form the area Jacobian."""
    a, b, owner = _neighbour_planes(sites, weights, box, n_neighbours, pool=pool)
    cap = cells.verts.shape[1]
    m_idx = np.arange(cap)[None, :]
    mask = m_idx < cells.count[:, None]
    nxt, _ = _next_vertex(cells.verts, None, cells.count)
    length = np.where(mask, np.linalg.norm(nxt - cells.verts, axis=2), 0.0)
    mid = 0.5 * (cells.verts + nxt) + cells.cell_centroid_site[:, None, :]
    resid = np.einsum("nmd,nkd->nmk", mid, a) - b[:, None, :]
    pick = np.argmin(np.abs(resid), axis=2)
    best = np.take_along_axis(np.abs(resid), pick[..., None], axis=2)[..., 0]
    dist = np.linalg.norm(np.take_along_axis(a, pick[..., None], axis=1), axis=2)
    r_eq = np.sqrt(np.maximum(cells.area, 1e-30) / math.pi)[:, None]
    on_plane = mask & (best <= 1e-9 * np.maximum(dist, 1e-12) * r_eq) & (dist > 0.0)
    flux = np.where(on_plane, length / (2.0 * np.maximum(dist, 1e-30)), 0.0)
    rows = np.broadcast_to(np.arange(cells.n)[:, None], cells.verts.shape[:2])
    cols = np.take_along_axis(owner, pick, axis=1)
    keep = on_plane & (flux > 0.0)
    return rows[keep], cols[keep], flux[keep]


def _check_valid(cells: LaguerreCells) -> bool:
    return (abs(cells.cover - 1.0) <= 1e-3 and cells.degenerate == 0
            and np.isfinite(cells.area).all())


def _initial_weights(targets: np.ndarray) -> np.ndarray:
    """Plane-balance initial guess ``w ~ 1.074 sqrt(T Tbar) - Tbar``."""
    t_bar = float(targets.mean())
    return 1.074 * np.sqrt(targets * t_bar) - t_bar


def laguerre_prescribed_areas(sites: np.ndarray, targets: np.ndarray,
                              box: Tuple[float, float], *,
                              params: Optional[LaguerreParams] = None
                              ) -> LaguerreCells:
    """Solve for the weights that give every cell its target area.

    Targets are rescaled to fill the box (``sum(targets) = Lx Ly``).  The solve is
    a **homotopy** (continuation) in the target vector, from the uniform
    tessellation ``w = 0`` (which is exact for equal areas) to the requested
    distribution, with a Newton step on the exact area Laplacian
    (:func:`_edge_flux`) at every stage, safeguarded by a backtracking line
    search on the relative-weighted residual ``sum((A-T)^2/T)``.  Continuation is
    what keeps the iterate out of the regime in which heavily under-weighted
    cells vanish (an empty cell has no shared edge, hence no Jacobian row).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    sites = np.asarray(sites, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape != (sites.shape[0],):
        raise ValueError("targets must have shape (N,)")
    if not np.isfinite(targets).all() or (targets <= 0.0).any():
        raise ValueError("targets must be finite and positive")
    params = params or LaguerreParams()
    lx, ly = float(box[0]), float(box[1])
    total = lx * ly
    n = sites.shape[0]
    tgt = targets * (total / targets.sum())
    k = int(min(max(params.n_neighbours, 6), max(n - 1, 6)))
    pool = params.pool

    k_state = [k]

    def areas(w: np.ndarray):
        cells_ = laguerre_cells(sites, w, (lx, ly), n_neighbours=k_state[0],
                                pool=pool)
        # remember the neighbour count that produced an exact cover: re-deriving
        # it from scratch at every weight update would cost a factor ~3
        k_state[0] = max(k_state[0], cells_.n_neighbours)
        return cells_

    def merit(a: np.ndarray, ref: np.ndarray) -> float:
        return float(np.sum((a - ref) ** 2 / ref))

    weights = np.zeros(n, dtype=np.float64)
    cells = areas(weights)
    stages = list(np.linspace(0.0, 1.0, 5))[1:]
    iterations = 0
    err = float(np.max(np.abs(cells.area - tgt) / tgt))
    for lam in stages:
        ref = (1.0 - lam) * tgt.mean() + lam * tgt
        ref = ref * (total / ref.sum())
        cur_err = float(np.max(np.abs(cells.area - ref) / ref))
        # intermediate homotopy stages only need to stay inside the basin of the
        # final Newton iteration; the strict tolerance is applied at lambda = 1
        stage_tol = params.area_tol if lam >= 1.0 else 20.0 * params.area_tol
        for _ in range(int(params.max_iter)):
            iterations += 1
            if cur_err <= stage_tol:
                break
            rows, cols, flux = _edge_flux(cells, sites, weights, (lx, ly),
                                          k_state[0], pool)
            diag = np.bincount(rows, weights=flux, minlength=n)
            positive = diag[diag > 0.0]
            floor = float(np.median(positive)) if positive.size else 1.0
            diag = np.maximum(diag, floor)
            jac = coo_matrix(
                (np.concatenate([flux, -flux]),
                 (np.concatenate([rows, rows]), np.concatenate([rows, cols]))),
                shape=(n, n)).tocsr()
            eye = coo_matrix((np.full(n, 1e-9 * float(diag.mean())),
                              (np.arange(n), np.arange(n))), shape=(n, n)).tocsr()
            rhs = ref - cells.area
            try:
                step = spsolve((jac + eye).tocsr(), rhs)
                if not np.isfinite(step).all():
                    raise RuntimeError("non-finite Newton step")
            except Exception:                                  # pragma: no cover
                step = rhs / diag
            step = step - step.mean()

            base = merit(cells.area, ref)
            best = None
            lam_step = 1.0
            for _try in range(4):
                cand = weights + lam_step * step
                cand = cand - cand.mean()
                try:
                    trial = areas(cand)
                except RuntimeError:
                    lam_step *= 0.5
                    continue
                m = merit(trial.area, ref)
                e = float(np.max(np.abs(trial.area - ref) / ref))
                if best is None or m < best[0]:
                    best = (m, e, trial, cand)
                if m < base:
                    break
                lam_step *= 0.5
            if best is None:
                raise RuntimeError("prescribed-area solve failed: no admissible step")
            _, cur_err, cells, weights = best
        err = float(np.max(np.abs(cells.area - tgt) / tgt))
    cells.iterations = iterations
    cells.area_error = err
    return cells
