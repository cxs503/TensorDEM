"""Unbiased jamming (RCP) measurement for the geometric compression protocol.

Why this module exists
----------------------
:func:`tensordem.bedcoupling.packing.make_random_close_packing` sets the cell
volume to *hit* the requested target porosity: the returned ``solid_fraction``
equals ``1 - target_porosity`` to machine precision, for every ``N`` and every
seed.  A benchmark that compares that number against the literature RCP value
(0.6366, Scott 1960 / Bernal & Mason 1960) therefore measures its own input,
not the packing: choosing ``target_porosity = 0.36`` makes the "agreement"
with 0.64 a tautology, and the reported ``N``-independence follows from the
same identity.

What is measurable is the *protocol's* jamming limit: the highest solid
fraction at which the overlap relaxation still removes the contacts it created
while compressing.  That number is *not* imposed anywhere in the code, it
depends on the relaxation budget and on the overlap tolerance one is willing
to call "in contact", and it must be reported as a function of that tolerance.

Public API
----------
* :func:`overlap_statistics` -- residual-overlap metrics of one packing.
* :func:`overlap_scan` -- those metrics on a solid-fraction grid.
* :func:`jamming_from_scan` -- tolerance-interpolated jamming solid fraction.
* :func:`jamming_curve` -- the same for several tolerances at once.

All routines are deterministic in ``seed``; nothing here asserts a target
solid fraction.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree

from .packing import make_random_close_packing

__all__ = [
    "ScanPoint",
    "overlap_statistics",
    "overlap_scan",
    "jamming_from_scan",
    "jamming_curve",
]


@dataclass(frozen=True)
class ScanPoint:
    """Residual-overlap metrics at one *imposed* solid fraction."""

    solid_fraction: float
    max_overlap_over_diameter: float
    mean_overlap_over_diameter: float
    overlap_pair_fraction: float
    n_contacts: int
    wall_time_s: float

    def as_dict(self) -> dict:
        return asdict(self)


def _contact_overlaps(positions, box_length, diameter):
    """Overlaps of all *touching* pairs (``r < d``) under minimum image."""
    tree = cKDTree(positions, boxsize=box_length)
    contact = tree.query_pairs(diameter, output_type="ndarray")
    n = int(len(positions))
    n_pairs = n * (n - 1) / 2.0
    if len(contact) == 0:
        return np.zeros(0), n_pairs
    dvec = positions[contact[:, 1]] - positions[contact[:, 0]]
    dvec -= box_length * np.round(dvec / box_length)
    dist = np.linalg.norm(dvec, axis=1)
    return np.maximum(diameter - dist, 0.0), n_pairs


def overlap_statistics(bed) -> dict:
    """Residual-overlap metrics of a :class:`~.packing.PackedBed`.

    ``max_overlap_over_diameter`` is the strict admissibility measure used by
    the packing module; ``mean_overlap_over_diameter`` averages over *touching*
    pairs only and behaves like a contact pressure proxy, which is the quantity
    that grows smoothly through the jamming transition.
    """
    overlaps, n_pairs = _contact_overlaps(bed.positions, bed.box_length, bed.diameter)
    d = bed.diameter
    touching = int(overlaps.size)
    return {
        "max_overlap_over_diameter": float(overlaps.max() / d) if touching else 0.0,
        "mean_overlap_over_diameter": float(overlaps.mean() / d) if touching else 0.0,
        "overlap_pair_fraction": float(touching / n_pairs) if n_pairs > 0 else 0.0,
        "n_contacts": touching,
    }


def overlap_scan(
    n_particles: int,
    diameter: float = 1e-3,
    phi_grid=(0.60, 0.62, 0.64, 0.66),
    *,
    seed: int = 42,
    final_relax_iterations: int = 6000,
) -> list[ScanPoint]:
    """Impose each solid fraction in ``phi_grid`` and measure the residual overlap.

    The solid fractions are *imposed* (that is what the compression protocol
    does); the measured quantities are the overlaps, which are free.
    """
    import time

    points: list[ScanPoint] = []
    for phi in phi_grid:
        phi = float(phi)
        t0 = time.time()
        bed = make_random_close_packing(
            n_particles,
            diameter,
            1.0 - phi,
            seed=seed,
            final_relax_iterations=final_relax_iterations,
            overlap_tolerance=1e-3,
        )
        stats = overlap_statistics(bed)
        points.append(
            ScanPoint(
                solid_fraction=bed.solid_fraction,
                wall_time_s=time.time() - t0,
                **stats,
            )
        )
    return points


def jamming_from_scan(points, tolerance: float, key: str = "max_overlap_over_diameter") -> dict:
    """Solid fraction at which ``key`` crosses ``tolerance`` (log-linear).

    Returns the interpolated crossing ``phi_j`` together with the bracketing
    scan points and a ``status`` of ``"interpolated"``, ``"below_grid"`` (every
    scanned point is clean, the limit is above the grid) or ``"above_grid"``
    (the coarsest point already overlaps, the limit is below the grid).  No
    value is ever returned by convention when the scan does not bracket it.
    """
    pts = sorted(points, key=lambda p: p.solid_fraction)
    xs = np.array([p.solid_fraction for p in pts], dtype=float)
    ys = np.array([getattr(p, key) for p in pts], dtype=float)
    if np.all(ys <= tolerance):
        return {"status": "below_grid", "phi_j": float(xs[-1]), "bracket": [float(xs[-2]), float(xs[-1])], "tolerance": tolerance, "key": key}
    if ys[0] > tolerance:
        return {"status": "above_grid", "phi_j": float(xs[0]), "bracket": [float(xs[0]), float(xs[1])], "tolerance": tolerance, "key": key}

    # first index above the tolerance; interpolate against the previous point
    idx = int(np.argmax(ys > tolerance))
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = max(ys[idx - 1], 1e-30), ys[idx]
    frac = (math.log(tolerance) - math.log(y0)) / (math.log(y1) - math.log(y0))
    phi_j = x0 + frac * (x1 - x0)
    return {
        "status": "interpolated",
        "phi_j": float(phi_j),
        "bracket": [float(x0), float(x1)],
        "bracket_values": [float(y0), float(y1)],
        "tolerance": float(tolerance),
        "key": key,
    }


def jamming_curve(points, tolerances=(1e-4, 1e-3, 1e-2, 5e-2), key: str = "max_overlap_over_diameter") -> dict:
    """Tolerance dependence of the jamming solid fraction.

    The spread between the loosest and the strictest tolerance is the honest
    uncertainty of any single ``phi_J`` quoted from this protocol.
    """
    out = {f"{tol:.0e}": jamming_from_scan(points, tol, key=key) for tol in tolerances}
    vals = [v["phi_j"] for v in out.values() if v["status"] == "interpolated"]
    out["spread"] = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
    out["key"] = key
    return out