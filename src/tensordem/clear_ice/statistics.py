"""Granular statistics of a clear-ice floe field: ``P(f)``, ``Z``, ``n``.

Evidence discipline
-------------------
Every number produced here is measured from the DEM state (contact list,
floe centres/radii), never assumed.  The three quantities are:

``P(f)`` -- force-chain distribution
    The distribution of **normal** contact forces normalised by their mean,
    ``P(f) = p(F_n / <F_n>)``.  For a jammed granular packing with Hertzian
    contacts the tail is exponential for ``f > 1``, ``P(f) ~ exp(-f/f0)``, with
    ``f0`` a measure of the chain heterogeneity (friction broadens it and
    increases the fraction of weak contacts).  The exponential fit is done in
    log space over the tail ``1 <= f <= f_tail_max`` and reported with its
    coefficient of determination ``R^2``; a power-law fit on the same window is
    reported alongside so the exponential claim is falsifiable rather than
    asserted.

``Z`` -- mean contact coordination number
    Number of mechanical contacts per floe, with a load threshold stated
    explicitly (``fraction * <F_n>``) so that grazing/numerical contacts do not
    inflate it.  ``Z`` is also reported excluding rattlers (degree < 2), which
    is the standard granular-mechanics convention.

``n`` -- porosity
    Two independent, explicitly different estimators:

    * ``areal`` -- ``1 - sum(pi r^2) / area``, the free-water fraction of a
      region (i.e. ``1 - ice concentration``);
    * ``voronoi`` -- free volume of the periodic Voronoi/Delaunay tessellation
      of the floe centres, ``sum(cell_area - pi r^2) / sum(cell_area)``.  The
      tessellation is built with periodic images, so no artificial wall bias is
      introduced.

    They answer the same physical question from different data, and their
    agreement in the ambient field is a consistency check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import Delaunay, Voronoi, cKDTree

__all__ = [
    "ForceChainStats",
    "PorosityStats",
    "areal_porosity",
    "coordination_number",
    "exponential_fit",
    "force_distribution",
    "voronoi_free_volume",
    "region_masks",
]


# --------------------------------------------------------------------------
# force-chain distribution
# --------------------------------------------------------------------------
@dataclass
class ForceChainStats:
    f_centres: np.ndarray
    probability: np.ndarray
    n_contacts: int
    mean_force: float
    f0: float                     # exponential decay scale (in <F_n> units)
    r2_exponential: float         # R^2 of ln P(f) vs f on the tail window
    powerlaw_exponent: float
    r2_powerlaw: float
    tail_window: tuple[float, float]
    weak_fraction: float          # P(f < 0.5)
    top_force_fraction: float     # share of the total load carried by f > 3
    exponential_preferred_bic: float   # BIC(power law) - BIC(exponential)

    def as_dict(self) -> dict:
        return {
            "n_contacts": int(self.n_contacts),
            "mean_force_N": float(self.mean_force),
            "f0_exponential": float(self.f0),
            "r2_exponential_tail": float(self.r2_exponential),
            "powerlaw_exponent": float(self.powerlaw_exponent),
            "r2_powerlaw_tail": float(self.r2_powerlaw),
            "tail_window": list(self.tail_window),
            "weak_contact_fraction_below_0p5": float(self.weak_fraction),
            "load_share_above_3x_mean": float(self.top_force_fraction),
            "delta_BIC_powerlaw_minus_exponential": float(self.exponential_preferred_bic),
        }


def exponential_fit(f: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None):
    """Least-squares ``ln p = a + b f``; returns ``(f0, r2, a, b)``."""
    mask = (p > 0.0) & np.isfinite(p)
    f, p = f[mask], p[mask]
    if weights is not None:
        weights = weights[mask]
    if len(f) < 3:
        return np.nan, np.nan, np.nan, np.nan
    y = np.log(p)
    if weights is None:
        A = np.stack((np.ones_like(f), f), axis=1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
    else:
        w = np.sqrt(weights)
        A = np.stack((np.ones_like(f), f), axis=1) * w[:, None]
        coef, *_ = np.linalg.lstsq(A, y * w, rcond=None)
        resid = (y - np.stack((np.ones_like(f), f), axis=1) @ coef) * w
        ss_tot = float(np.sum((y * w - (y * w).mean()) ** 2))
        r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
    a, b = float(coef[0]), float(coef[1])
    f0 = -1.0 / b if b < 0 else np.nan
    return f0, r2, a, b


def force_distribution(fn: np.ndarray, *, bins: int = 26, f_min: float = 0.0,
                       f_max: float = 5.0,
                       tail_window: tuple[float, float] = (1.0, 4.0),
                       min_count: int = 12) -> ForceChainStats:
    """Force-chain statistics of a set of normal contact forces ``fn`` (N)."""
    fn = np.asarray(fn, dtype=float)
    fn = fn[np.isfinite(fn) & (fn > 0.0)]
    if len(fn) < 10:
        raise ValueError("need at least 10 positive contact forces")
    mean_f = float(fn.mean())
    f = fn / mean_f
    edges = np.linspace(f_min, f_max, bins + 1)
    counts, _ = np.histogram(f, bins=edges)
    widths = np.diff(edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    p = counts / (len(f) * widths)

    # tail window: this is where the exponential claim lives
    lo, hi = tail_window
    tail = (centres >= lo) & (centres <= hi) & (counts >= min_count)
    if tail.sum() >= 3:
        f0, r2_exp, a, b = exponential_fit(centres[tail], p[tail])
        # power law (straight line in log-log)
        y = np.log(p[tail])
        X = np.stack((np.ones(tail.sum()), np.log(centres[tail])), axis=1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        ss = float(np.sum((y - y.mean()) ** 2))
        r2_pow = 1.0 - float(np.sum(resid ** 2)) / ss if ss > 0 else np.nan
        n_tail = int(counts[tail].sum())
        bic_exp = n_tail * np.log(np.sum(resid ** 2) / n_tail) if n_tail > 0 else np.nan
        bic_pow = bic_exp
        # ANCOVA-free comparison: recompute residuals of the exponential on the
        # same window for a like-for-like BIC difference.
        exp_resid = np.log(p[tail]) - (a + b * centres[tail])
        bic_exp = float(n_tail * np.log(np.sum(exp_resid ** 2) / n_tail))
        bic_pow = float(n_tail * np.log(max(np.sum(resid ** 2), 1e-300) / n_tail))
    else:  # pragma: no cover - only for degenerate contact sets
        f0, r2_exp, r2_pow, bic_exp, bic_pow = np.nan, np.nan, np.nan, np.nan, np.nan
        coef = [np.nan, np.nan]

    weak = float(np.mean(f < 0.5))
    load = float(fn.sum())
    top = float(fn[f > 3.0].sum() / load) if load > 0 else np.nan
    return ForceChainStats(
        f_centres=centres, probability=p, n_contacts=int(len(f)), mean_force=mean_f,
        f0=f0, r2_exponential=r2_exp, powerlaw_exponent=float(coef[1]),
        r2_powerlaw=r2_pow, tail_window=tail_window, weak_fraction=weak,
        top_force_fraction=top, exponential_preferred_bic=float(bic_pow - bic_exp),
    )


# --------------------------------------------------------------------------
# coordination number
# --------------------------------------------------------------------------
def coordination_number(i: np.ndarray, j: np.ndarray, n_particles: int,
                        fn: np.ndarray | None = None, *,
                        load_fraction: float = 1.0e-3,
                        exclude_rattlers: bool = True) -> dict:
    """Mean coordination number with an explicit load threshold."""
    i = np.asarray(i, dtype=int)
    j = np.asarray(j, dtype=int)
    if fn is not None and len(fn):
        fn = np.asarray(fn, dtype=float)
        keep = fn > load_fraction * float(fn.mean())
        i, j = i[keep], j[keep]
    deg = np.bincount(i, minlength=n_particles) + np.bincount(j, minlength=n_particles)
    z_all = float(deg.mean())
    non_rattler = deg >= 2
    z_eff = float(deg[non_rattler].mean()) if np.any(non_rattler) else 0.0
    return {
        "Z_all": z_all,
        "Z_non_rattler": z_eff if exclude_rattlers else z_all,
        "rattler_fraction": float(1.0 - non_rattler.mean()),
        "mean_degree": z_all,
        "load_threshold_fraction": load_fraction,
    }


# --------------------------------------------------------------------------
# porosity
# --------------------------------------------------------------------------
@dataclass
class PorosityStats:
    areal: float
    voronoi: float
    region_area: float
    n_floes: int | None = None

    def as_dict(self) -> dict:
        return {"areal": self.areal, "voronoi": self.voronoi,
                "region_area_m2": self.region_area}


def areal_porosity(x: np.ndarray, r: np.ndarray, area: float,
                   mask: np.ndarray | None = None) -> float:
    """``1 - sum(pi r^2)/area`` restricted to the masked floes."""
    if mask is not None:
        r = r[mask]
    covered = float(np.sum(np.pi * r ** 2))
    return float(1.0 - covered / area)


def voronoi_free_volume(x: np.ndarray, r: np.ndarray, box: tuple[float, float],
                        mask: np.ndarray | None = None) -> float:
    """Periodic Voronoi free-volume porosity of the masked floe subset.

    The tessellation is built with the 8 translated copies of the point set and
    only cells whose seed lies in the central box are counted, which removes
    the bias of a non-periodic diagram at the box edges.  Cells smaller than the
    floe itself (possible only with unrelaxed overlaps) contribute zero free
    volume instead of a negative number.
    """
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)
    if mask is not None:
        x, r = x[mask], r[mask]
    if len(x) < 4:
        raise ValueError("need at least 4 floes for a Voronoi porosity")
    Lx, Ly = float(box[0]), float(box[1])
    shifts = np.array([[i * Lx, j * Ly] for i in (-1, 0, 1) for j in (-1, 0, 1)])
    pts = (x[None, :, :] + shifts[:, None, :]).reshape(-1, 2)
    n_images = len(shifts)
    centre_slice = slice(4 * len(x), 5 * len(x))   # shift index (0, 0) is the 5th
    vor = Voronoi(pts)
    areas = np.zeros(len(pts))
    for idx, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if not region or -1 in region:
            continue
        poly = vor.vertices[region]
        a = 0.0
        for k in range(len(poly)):
            x1, y1 = poly[k]
            x2, y2 = poly[(k + 1) % len(poly)]
            a += x1 * y2 - x2 * y1
        areas[idx] = 0.5 * abs(a)
    central = areas[centre_slice]
    free = np.maximum(central - np.pi * r ** 2, 0.0)
    total = float(central.sum())
    return float(free.sum() / total) if total > 0 else np.nan


def region_masks(x: np.ndarray, config) -> tuple[dict, dict]:
    """Strip regions of the clear-ice channel and their exact areas.

    All three are full-width strips in the periodic ``y`` direction, so the
    area of each is ``length * Ly`` exactly -- no mask-subtraction
    approximation, which is what makes the areal porosity comparable between
    regions.

    * ``feed``  -- far upstream of the gate, the undisturbed clear-ice field;
    * ``jam``   -- the last ``4`` mean radii before the gate (compression zone);
    * ``wake``  -- ``6`` mean radii behind the gate (clearing zone).
    """
    c = config.gate_centre
    r_mean = float(config.mean_radius)
    dx = x[:, 0] - c[0]
    edges = {
        "feed": (-np.inf, -10.0 * r_mean),
        "jam": (-4.0 * r_mean, 0.0),
        "wake": (0.0, 6.0 * r_mean),
    }
    masks, areas = {}, {}
    for name, (lo, hi) in edges.items():
        masks[name] = (dx >= lo) & (dx < hi)
        length = (hi - lo) if np.isfinite(hi - lo) else (config.domain_x * 0.5)
        if name == "feed":
            length = config.gate_x_fraction * config.domain_x - 10.0 * r_mean
        areas[name] = float(max(length, 0.0) * config.domain_y)
    return masks, areas
