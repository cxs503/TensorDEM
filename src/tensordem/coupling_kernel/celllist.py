"""Cell-linked-list contact detection core (numpy, host side).

The detector is deliberately implemented in **numpy only** and executed on the
host for every backend: integer cell bookkeeping is exactly reproducible, so
the numpy and torch backends see the *same* candidate pair list, step after
step.  This is what makes the backend-equivalence claim testable at the state
level rather than only at the force level.

Reference: Allen & Tildesley, *Computer Simulation of Liquids*, ch. 5 (linked
cell lists); the implementation below uses a stable sort of integer cell keys
plus a ragged gather over the 13 half-space cell offsets, so the pair list is
deterministic (no Python set/dict iteration order involved).
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = ["candidate_pairs", "brute_force_pairs", "cell_geometry"]

_HALF_OFFSETS = tuple(
    off
    for off in itertools.product((-1, 0, 1), repeat=3)
    if off > (0, 0, 0)
)  # 13 offsets


def cell_geometry(box_lo, box_hi, cutoff: float):
    """Return ``(n_cells, cell_size)`` for a cutoff-based cell decomposition."""

    lo = np.asarray(box_lo, dtype=np.float64)
    hi = np.asarray(box_hi, dtype=np.float64)
    length = hi - lo
    if np.any(length <= 0.0):
        raise ValueError("box_hi must exceed box_lo in every direction")
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    nc = np.maximum(1, np.floor(length / cutoff).astype(np.int64))
    size = length / nc
    return nc, size


def candidate_pairs(positions, cutoff: float, box_lo, box_hi):
    """Return ``(i, j)`` candidate pairs with ``i < j`` and ``|r_ij| <= cutoff``.

    Only pairs inside the box are considered; particles are assumed to be
    clamped to the box by the wall model.  The returned arrays are int64 and
    sorted by ``i * n + j`` (canonical, reproducible order).
    """

    pos = np.asarray(positions, dtype=np.float64)
    n = pos.shape[0]
    if n < 2:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy()
    lo = np.asarray(box_lo, dtype=np.float64)
    hi = np.asarray(box_hi, dtype=np.float64)
    nc, cell = cell_geometry(lo, hi, cutoff)

    ijk = np.floor((pos - lo) / cell).astype(np.int64)
    ijk = np.clip(ijk, 0, nc - 1)
    keys = ijk[:, 0] + nc[0] * (ijk[:, 1] + nc[1] * ijk[:, 2])
    ncells = int(nc[0] * nc[1] * nc[2])

    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    counts = np.bincount(keys, minlength=ncells)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    ijk_sorted = ijk[order]
    rows = np.arange(n, dtype=np.int64)

    collected_i: list[np.ndarray] = []
    collected_j: list[np.ndarray] = []

    def _push(rows_sel: np.ndarray, target_keys: np.ndarray) -> None:
        if rows_sel.size == 0:
            return
        cnt = counts[target_keys]
        total = int(cnt.sum())
        if total == 0:
            return
        row_rep = np.repeat(rows_sel, cnt)
        ends = np.cumsum(cnt)
        start_rep = np.repeat(ends - cnt, cnt)
        local = np.arange(total, dtype=np.int64) - start_rep
        sorted_idx = np.repeat(starts[target_keys], cnt) + local
        cols = order[sorted_idx]
        collected_i.append(order[row_rep])
        collected_j.append(cols)

    # self cell: keep i < j only
    same = np.arange(n, dtype=np.int64)
    _push(same, sorted_keys)

    for off in _HALF_OFFSETS:
        tgt = ijk_sorted + np.asarray(off, dtype=np.int64)
        valid = np.all((tgt >= 0) & (tgt < nc), axis=1)
        if not np.any(valid):
            continue
        tgt_valid = tgt[valid]
        target_keys = (
            tgt_valid[:, 0] + nc[0] * (tgt_valid[:, 1] + nc[1] * tgt_valid[:, 2])
        )
        _push(rows[valid], target_keys)

    if not collected_i:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy()

    i = np.concatenate(collected_i)
    j = np.concatenate(collected_j)
    if i.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy()

    d = pos[j] - pos[i]
    keep = (i != j) & (np.einsum("ij,ij->i", d, d) <= cutoff * cutoff)
    i = i[keep]
    j = j[keep]
    i, j = np.minimum(i, j), np.maximum(i, j)
    key = i.astype(np.int64) * n + j
    uniq = np.unique(key)  # sorted + de-duplicated (safety against double counting)
    return (uniq // n).astype(np.int64), (uniq % n).astype(np.int64)


def brute_force_pairs(positions, cutoff: float):
    """O(n^2) reference detector used to verify :func:`candidate_pairs`."""

    pos = np.asarray(positions, dtype=np.float64)
    n = pos.shape[0]
    if n < 2:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy()
    iu, ju = np.triu_indices(n, k=1)
    d = pos[ju] - pos[iu]
    keep = np.einsum("ij,ij->i", d, d) <= cutoff * cutoff
    return iu[keep].astype(np.int64), ju[keep].astype(np.int64)