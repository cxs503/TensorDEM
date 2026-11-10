"""Vectorised counting-sort cell list for DEM contact search (IN383-DEM01).

The contact search is the only super-linear step in a DEM time step.  A naive
``O(N^2)`` pair enumeration is kept here as a *reference* implementation: it is
used by the verification tests and by the negative controls to prove that the
``O(N)`` cell list returns exactly the same contact set.

Design notes
------------
* Cell size ``c = 2*r_max + skin`` so that every pair closer than
  ``r_i + r_j + skin`` is guaranteed to sit in the same or an adjacent cell.
* Particles are ordered by a stable sort on the linear cell id (radix-like on
  integer keys); the partner ranges of every particle are then found with two
  ``searchsorted`` calls and expanded raggedly with ``np.repeat``.  No Python
  loop over cells => runtime is ``O(N + C)`` with ``C`` the candidate count.
* Pairs are returned with ``i < j`` and sorted lexicographically, which makes
  the pair list order deterministic (bit-level reproducibility, criterion C6)
  and lets a moving particle keep its tangential-history slot between rebuilds.
"""

from __future__ import annotations

import numpy as np

#: 13 half-space cell offsets + the self cell (used when every axis has >= 3 cells)
HALF_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    (dx, dy, dz)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dx, dy, dz) > (0, 0, 0)
)

FULL_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    (dx, dy, dz)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
)


def _grid(lengths, cell_size):
    return tuple(max(1, int(np.floor(lengths[a] / cell_size[a]))) for a in range(3))


def _cell_index(pos, lengths, cell_size, n_cell):
    """Return linear cell id and per-axis indices."""
    idx = []
    for a in range(3):
        raw = pos[:, a] / cell_size[a]
        # Guard against non-finite or extreme values that would overflow int64
        raw = np.where(np.isfinite(raw), raw, 0.0)
        raw = np.clip(raw, -1e15, 1e15)
        j = np.floor(raw).astype(np.int64)
        idx.append(np.clip(j, 0, n_cell[a] - 1))
    ids = idx[0] + n_cell[0] * (idx[1] + n_cell[1] * idx[2])
    return ids, idx


def cell_list_pairs(
    pos: np.ndarray,
    lengths: np.ndarray,
    radii: np.ndarray,
    *,
    skin: float,
    periodic: tuple[bool, bool, bool] = (False, True, False),
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return candidate pairs ``(i, j)`` with ``i < j`` from a uniform cell list.

    Parameters
    ----------
    pos : (N, 3) float64
    lengths : (3,) float64 box edge lengths
    radii : (N,) float64
    skin : float   Verlet skin added to the geometric reach
    periodic : per-axis periodicity (wrapping is applied in ``y`` by default)
    """
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    if not np.all(np.isfinite(pos)):
        bad = int((~np.isfinite(pos)).any(axis=1).sum())
        raise ValueError(
            f"contact search received {bad} non-finite particle position(s): the "
            "integration has diverged (check dt / contact stiffness) — refusing to "
            "build a cell list from NaN/inf coordinates"
        )
    n = pos.shape[0]
    rmax = float(radii.max()) if n else 0.0
    c = 2.0 * rmax + skin
    cell_size = np.full(3, c)
    n_cell = _grid(lengths, cell_size)
    ids, axis_idx = _cell_index(pos, lengths, cell_size, n_cell)
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]

    small_axis = any(n_cell[a] < 3 for a in range(3) if periodic[a])
    offsets = FULL_OFFSETS if small_axis else HALF_OFFSETS + ((0, 0, 0),)

    self_idx = np.arange(n, dtype=np.int64)
    out_i: list[np.ndarray] = []
    out_j: list[np.ndarray] = []
    for off in offsets:
        alive = np.ones(n, dtype=bool)
        axes = []
        for a in range(3):
            j = axis_idx[a] + off[a]
            if periodic[a]:
                j = np.mod(j, n_cell[a])
            else:
                alive &= (j >= 0) & (j < n_cell[a])
                j = np.clip(j, 0, n_cell[a] - 1)
            axes.append(j)
        # same mixed-radix encoding as ``_cell_index``: ix + ncx*(iy + ncy*iz)
        target = axes[0] + n_cell[0] * (axes[1] + n_cell[1] * axes[2])
        lo = np.searchsorted(sorted_ids, target, side="left")
        hi = np.searchsorted(sorted_ids, target, side="right")
        cnt = np.where(alive, hi - lo, 0)
        total = int(cnt.sum())
        if total == 0:
            continue
        first = np.repeat(lo, cnt)
        base = np.arange(total, dtype=np.int64)
        starts = np.cumsum(cnt) - cnt
        local = base - np.repeat(starts, cnt)
        partner = order[first + local]
        src = np.repeat(self_idx, cnt)
        if off == (0, 0, 0):
            keep = partner > src
            src, partner = src[keep], partner[keep]
        # a particle cannot pair with itself even through wrapping
        keep = partner != src
        out_i.append(src[keep])
        out_j.append(partner[keep])

    if not out_i:
        empty = np.zeros(0, dtype=np.int64)
        stats = {"n_candidate_raw": 0.0, "n_cells": float(np.prod(n_cell)), "cell_size": float(c)}
        return empty, empty, stats

    i = np.concatenate(out_i)
    j = np.concatenate(out_j)
    lo = np.minimum(i, j)
    hi = np.maximum(i, j)
    key = lo.astype(np.int64) * n + hi.astype(np.int64)
    key = np.unique(key)
    i = key // n
    j = key % n
    # geometric filter: keep only pairs within the Verlet reach
    d = pos[j] - pos[i]
    for a in range(3):
        if periodic[a]:
            d[:, a] -= lengths[a] * np.round(d[:, a] / lengths[a])
    dist = np.sqrt(np.einsum("ij,ij->i", d, d))
    reach = radii[i] + radii[j] + skin
    keep = dist <= reach
    i, j = i[keep], j[keep]
    stats = {
        "n_candidate_raw": float(key.size),
        "n_candidate": float(i.size),
        "n_cells": float(np.prod(n_cell)),
        "cell_size": float(c),
    }
    return i, j, stats


def brute_force_pairs(
    pos: np.ndarray,
    lengths: np.ndarray,
    radii: np.ndarray,
    *,
    skin: float = 0.0,
    periodic: tuple[bool, bool, bool] = (False, True, False),
    block: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """``O(N^2)`` reference search — verification / negative-control only."""
    n = pos.shape[0]
    ii, jj = [], []
    for start in range(0, n, block):
        stop = min(n, start + block)
        d = pos[None, :, :] - pos[start:stop, None, :]
        for a in range(3):
            if periodic[a]:
                d[..., a] -= lengths[a] * np.round(d[..., a] / lengths[a])
        dist = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
        reach = radii[start:stop, None] + radii[None, :] + skin
        overlap = (dist < reach) & (np.arange(n)[None, :] > np.arange(start, stop)[:, None])
        a, b = np.nonzero(overlap)
        ii.append(a + start)
        jj.append(b)
    if not ii:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(ii), np.concatenate(jj)