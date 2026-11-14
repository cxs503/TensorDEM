"""Geometric admissibility of a disordered initial packing (m20259).

Why this exists
---------------
The ``CASE-COLUMN-COLLAPSE`` ensemble wave (m19922) turns realisation
uncertainty on by *positional jitter*: every particle is displaced by a normal
deviate with amplitude ``jitter * r_p`` per coordinate.  Any such displacement
larger than half the inter-particle clearance of the prepared lattice puts
neighbouring spheres into overlap, and an overlapping initial state stores
elastic energy that the DEM releases impulsively at ``t = 0``.  A "disordered
packing" produced that way is not a physical realisation of the same granular
column: it is a different initial energy state.

``admissibility`` answers, with pure geometry and no time integration, whether a
prepared state is physically admissible, and how large the jitter amplitude may
be before overlap appears.  It is the gate used by
``benchmarks/case_column_collapse_ensemble_admissible_m20259.py`` and by
``scripts/analyze_column_collapse_ensemble_m20259.py``.

Definitions (all lengths in metres)
-----------------------------------
``gap_ij``      ``|x_i - x_j| - (r_i + r_j)``; positive = clearance.
``min_gap``     the smallest ``gap_ij`` over all pairs (lattice: ``packing_gap``).
``overlap_pairs`` number of pairs with ``gap_ij < -tol``; the admissibility
                criterion is ``overlap_pairs == 0`` with ``tol = 0``.
``max_overlap`` the deepest penetration ``max(-gap_ij)``, i.e. the stored
                overlap depth that the contact law will push apart.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

#: Pre-registered admissibility rule: a state with any overlapping pair is
#: rejected.  Not a tunable tolerance — a contact law may never start in
#: penetration without having been asked to.
ADMISSIBLE_OVERLAP_PAIRS = 0


def overlap_stats(pos: np.ndarray, radius: float,
                  tol: float = 0.0) -> dict:
    """Geometric admissibility report for a monodisperse packing.

    ``pos`` is ``(N, 3)``; the neighbour search is a KD-tree over a cutoff of
    one diameter, so the cost is ``O(N log N)`` and this never integrates
    anything.
    """
    pos = np.asarray(pos, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must be (N, 3)")
    n = len(pos)
    if n < 2:
        return {"n_particles": n, "min_gap": float("inf"), "overlap_pairs": 0,
                "max_overlap": 0.0, "admissible": True, "tol": tol}
    from scipy.spatial import cKDTree

    tree = cKDTree(pos)
    dist, _ = tree.query(pos, k=2)
    min_gap = float(dist[:, 1].min() - 2.0 * radius)
    pairs = tree.query_pairs(2.0 * radius + max(tol, 0.0), output_type="ndarray")
    if len(pairs):
        sep = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
        gaps = sep - 2.0 * radius
        n_overlap = int(np.sum(gaps < -tol))
        max_overlap = float(max(0.0, -gaps.min()))
    else:
        n_overlap, max_overlap = 0, 0.0
    return {"n_particles": int(n),
            "min_gap": min_gap,
            "overlap_pairs": n_overlap,
            "max_overlap": max_overlap,
            "admissible": bool(n_overlap <= ADMISSIBLE_OVERLAP_PAIRS),
            "tol": float(tol)}


def max_admissible_jitter(build: Callable[[float], np.ndarray], radius: float,
                          levels: Iterable[float]) -> dict:
    """Largest jitter amplitude in ``levels`` that still yields no overlap.

    ``build(jitter)`` must return the initial positions for that amplitude
    (same case, same seed).  The scan is monotone in the amplitude in practice,
    so the result is reported as the largest acceptable level *and* the first
    rejecting level, which is a bracketing interval rather than a fitted
    threshold.
    """
    rows = []
    for j in sorted(float(x) for x in levels):
        st = overlap_stats(build(j), radius)
        rows.append({"jitter": j, **st})
    ok = [r for r in rows if r["admissible"]]
    bad = [r for r in rows if not r["admissible"]]
    return {
        "levels": rows,
        "jitter_max_admissible": (max(r["jitter"] for r in ok) if ok else None),
        "jitter_first_rejecting": (min(r["jitter"] for r in bad) if bad else None),
        "rule": "overlap_pairs == 0 (geometry only, no time integration)",
    }