"""Analyst-independent solid-fraction estimators for a settled DEM packing.

The deposited-packing solid fraction is a *measurement*, not a simulation
output: it is the occupied volume divided by a chosen control volume, and a
pipeline has to choose (i) which control volume (band, free-surface handling,
wall layers dropped) and (ii) how the occupied volume is quadratured (binned
profile with linear weights, per-sphere volume balance, ...).  Either choice is
an analyst degree of freedom that can move a reported ``phi`` by more than a
loose acceptance band allows.

This module supplies the two estimators that do **not** depend on those choices,
so a reported number can be confronted with them:

``band_solid_fraction_exact``
    Exact quadrature of ``sum_i Vol(sphere_i cap slab)`` -- the analytic
    antiderivative of the sphere cross-section.  No binning, no
    centre-in-bin assignment.  Counts pair overlaps twice, so it is an upper
    bound for the true occupied (union) volume.

``band_solid_fraction_mc``
    Monte-Carlo/union estimator: uniform points in the control volume, count
    the fraction inside *at least one* sphere (with x/y minimum image, matching
    a laterally periodic cell and a floor without a ceiling).  This is the
    unbiased union-volume estimator; its statistical error is returned with the
    number.  Restricting the sampling region (``interior_margin``) measures how
    much the lateral cell boundary contributes.

Both are pure functions of the settled configuration, so they can be applied to
stored ``runs/*.npz`` without re-running a simulation.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "sphere_slab_volume",
    "band_solid_fraction_exact",
    "band_solid_fraction_mc",
]


def sphere_slab_volume(z_centre, radius, z_lo, z_hi):
    """Exact volume of one sphere intersected with the slab ``[z_lo, z_hi]``.

    ``V = pi (r^2 h - h^3/3)`` evaluated at the two clamped offsets
    ``h = clip(z - z_centre, -r, r)``: the exact integral of the circular
    cross-section ``pi (r^2 - h^2)`` over the part of the sphere inside the
    slab.  Vectorised over the sphere axis.
    """
    z_centre = np.asarray(z_centre, dtype=float)
    r = np.broadcast_to(np.asarray(radius, dtype=float), z_centre.shape).astype(float)
    a = np.clip(np.asarray(z_lo, dtype=float) - z_centre, -r, r)
    b = np.clip(np.asarray(z_hi, dtype=float) - z_centre, -r, r)
    vol = math.pi * (r * r * (b - a) - (b ** 3 - a ** 3) / 3.0)
    return np.maximum(vol, 0.0)


def band_solid_fraction_exact(positions, radius, box_width, z_lo, z_hi,
                              radii=None):
    """``sum_i Vol(sphere_i cap band) / (box_width^2 (z_hi - z_lo))``.

    ``radii`` overrides a scalar ``radius`` (per-particle radii, e.g. the
    overlap-deflated geometry).  Returns ``nan`` for an empty band.
    """
    positions = np.asarray(positions, dtype=float)
    height = float(z_hi) - float(z_lo)
    if height <= 0.0:
        return float("nan")
    rad = radius if radii is None else np.asarray(radii, dtype=float)
    vol = sphere_slab_volume(positions[:, 2], rad, z_lo, z_hi)
    return float(np.sum(vol) / (float(box_width) ** 2 * height))


def _replicate_xy(positions: np.ndarray, box_width: float, radii):
    """Nine in-plane images, so a nearest-centre query obeys the x/y min image."""
    shifts = np.array([[dx, dy, 0.0]
                       for dx in (-box_width, 0.0, box_width)
                       for dy in (-box_width, 0.0, box_width)])
    centres = (positions[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    rad = None if radii is None else np.tile(np.asarray(radii, float), 9)
    return centres, rad


def band_solid_fraction_mc(positions, box_width, z_lo, z_hi, radius,
                           n_points: int = 400_000, seed: int = 0,
                           chunk: int = 200_000, workers: int = 1,
                           radii=None, interior_margin: float | None = None):
    """Monte-Carlo union-volume solid fraction of the slab ``[z_lo, z_hi]``.

    Points are uniform in ``[0, box_width)^2 x [z_lo, z_hi)``; a point counts as
    occupied when it lies inside at least one sphere (``k`` nearest centres with
    min-image in x/y).  With ``radii`` given, spheres have per-particle radii,
    so ``k`` neighbours are tested instead of one.

    ``interior_margin`` (in length units): also return the fraction computed only
    from points at least that far from the lateral boundary, i.e. the same
    quantity with the periodic cell wall excluded.

    Returns ``{"phi", "stderr", "n_points", "n_inside", "phi_interior"}``;
    ``phi_interior`` is ``nan`` when no margin is requested.
    """
    positions = np.asarray(positions, dtype=float)
    height = float(z_hi) - float(z_lo)
    if height <= 0.0:
        raise ValueError("empty band")
    centres, rad = _replicate_xy(positions, float(box_width), radii)
    tree = cKDTree(centres)
    k = 1 if rad is None else min(8, centres.shape[0])
    r_scalar = float(radius)
    rng = np.random.default_rng(seed)
    n_inside = 0
    n_int_inside = 0
    n_int_total = 0
    done = 0
    while done < n_points:
        m = min(chunk, n_points - done)
        pts = np.empty((m, 3))
        pts[:, 0] = rng.random(m) * box_width
        pts[:, 1] = rng.random(m) * box_width
        pts[:, 2] = z_lo + rng.random(m) * height
        dist, idx = tree.query(pts, k=k, workers=workers)
        if rad is None:
            inside = dist < r_scalar
        else:
            dist = np.atleast_2d(dist.T).T if dist.ndim == 1 else dist
            inside = np.any(dist < rad[idx], axis=1)
        n_inside += int(np.count_nonzero(inside))
        if interior_margin is not None:
            margin = float(interior_margin)
            ok = ((pts[:, 0] > margin) & (pts[:, 0] < box_width - margin)
                  & (pts[:, 1] > margin) & (pts[:, 1] < box_width - margin))
            n_int_total += int(np.count_nonzero(ok))
            n_int_inside += int(np.count_nonzero(inside & ok))
        done += m
    p = n_inside / float(n_points)
    stderr = math.sqrt(max(p * (1.0 - p), 0.0) / float(n_points))
    phi_int = (n_int_inside / float(n_int_total)) if n_int_total else float("nan")
    return {
        "phi": float(p),
        "stderr": float(stderr),
        "n_points": int(n_points),
        "n_inside": int(n_inside),
        "phi_interior": float(phi_int),
        "n_interior_points": int(n_int_total),
    }