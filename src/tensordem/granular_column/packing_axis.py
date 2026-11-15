"""Initial packing fraction as an independent axis for CASE-COLUMN-COLLAPSE.

Why this exists
---------------
``CASE-COLUMN-COLLAPSE`` is adjudicated by a *resolution* ladder
(``benchmarks/case_column_collapse_grid_study_m21656.py``): the column radius
``r_i`` is held fixed and only the number of grains per radius ``gpr`` moves.
That ladder carries a confound which m21656 reported but never treated
(audit note ⑤.4): **the HCP lattice fills the cylinder differently at different
resolution**, so ``phi_0`` is not constant along the ladder.  Recomputed from
the committed geometry (no integration needed, :func:`solid_fraction` on the
built packing):

===========  ==================  ==================
grains/``r_i``  ``phi_0`` (a=3)     ``phi_0`` (a=1)
===========  ==================  ==================
4            0.5457              0.5255
5            0.5744              0.5620
8            0.6263              0.6179
10           0.6337              0.6268
12           0.6371              0.6312
14           0.6572              0.6523
16           0.6615              0.6570
===========  ==================  ==================

i.e. a ``phi_0`` drift of 11.6 percentage points (21 % relative) *across the
very axis that is supposed to be "resolution only"*.  If the final runout
responds to ``phi_0``, then part of what m21656 read as "resolution scatter"
(and its non-monotone ``a = 3`` sequence) is a packing-fraction trend, and the
grid-convergence statement is confounded.

What this module provides
-------------------------
Pure, closed-form bookkeeping -- no DEM, no file IO, no hidden state:

``solid_fraction``
    ``phi_0 = N * (4/3 pi r_p^3) / (pi r_i^2 h_i)`` with the *measured*
    ``r_i``/``h_i`` of the built packing (never an analytic lattice value).
``ols_slope``
    Ordinary least squares slope with an honest refusal vocabulary: fewer than
    three finite points, fewer than three distinct abscissae, or a degenerate
    abscissa range are ``indeterminate`` -- not a slope with a big error bar.
``difference_quotient``
    A two-point slope, explicitly labelled as *carrying no significance
    statement* (it is reported because a two-level arm is still evidence about
    the sign and magnitude of the response).
``slope_is_resolved``
    Requires ``|slope| > sigma * stderr(slope)``; a fit whose residual degrees
    of freedom are zero (as many data points as parameters) is **not**
    resolved, because its stderr is undefined.
``confound_decomposition``
    Turns ``(slope, Delta phi_0, observed spread)`` into the runout shift that
    the packing drift alone implies, its share of the observed spread, and a
    three-way verdict.  The share thresholds (0.2 / 0.5) are pre-registered
    here, before any of the sweep's numbers exist, so the verdict cannot be
    tuned after the fact.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "CONFOUND_SHARE_COMPARABLE",
    "CONFOUND_SHARE_DOMINATED",
    "RESOLUTION_SIGMA",
    "confound_decomposition",
    "difference_quotient",
    "ols_slope",
    "slope_is_resolved",
    "solid_fraction",
]

#: ``|slope|`` must exceed this many standard errors to count as resolved.
RESOLUTION_SIGMA = 2.0
#: Share of the observed spread that is called "comparable" (lower bound).
CONFOUND_SHARE_COMPARABLE = 0.2
#: Share of the observed spread that is called "packing-dominated".
CONFOUND_SHARE_DOMINATED = 0.5


def _finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def solid_fraction(n_particles: int, radius: float,
                   r_i: float, h_i: float) -> float:
    """Initial solid fraction of a cylindrical column of spheres.

    ``phi_0 = N (4/3 pi r_p^3) / (pi r_i^2 h_i)`` with ``r_i``/``h_i`` the
    **measured** outer radius and height of the built packing (``r_i`` is the
    99.5 % radial quantile plus one grain radius, as everywhere else in this
    case).  Raises ``ValueError`` rather than returning a silent ``nan``.
    """
    n = int(n_particles)
    radius = float(radius)
    r_i = float(r_i)
    h_i = float(h_i)
    if n < 1:
        raise ValueError("n_particles must be >= 1")
    if not all(math.isfinite(v) and v > 0.0 for v in (radius, r_i, h_i)):
        raise ValueError("radius, r_i and h_i must be finite and positive")
    volume_grain = 4.0 / 3.0 * math.pi * radius ** 3
    volume_column = math.pi * r_i ** 2 * h_i
    return float(n * volume_grain / volume_column)


def ols_slope(xs: Sequence[float], ys: Sequence[float], *,
              min_points: int = 3, min_distinct: int = 3,
              rel_span_floor: float = 1e-9) -> dict:
    """Least-squares slope of ``ys`` on ``xs``, with refusal vocabulary.

    Returns a dict with ``status`` in ``{"fit", "indeterminate"}``.  No fitted
    number is reported unless the data can support it: a slope over two points
    is ``indeterminate`` here (use :func:`difference_quotient` and say so).
    """
    xs = list(xs)
    ys = list(ys)
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys)
             if _finite(x) and _finite(y)]
    n = len(pairs)
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    out = {"status": "indeterminate", "slope": float("nan"),
           "intercept": float("nan"), "stderr": float("nan"),
           "r2": float("nan"), "n": n, "n_distinct": 0,
           "x_span": float("nan"), "reason": ""}
    if n == 0:
        out["reason"] = "no finite (x, y) pairs"
        return out
    distinct = np.unique(x)
    span = float(x.max() - x.min())
    out["n_distinct"] = int(distinct.size)
    out["x_span"] = span
    if n < max(int(min_points), 3):
        out["reason"] = f"fewer than {max(int(min_points), 3)} finite points"
        return out
    if distinct.size < max(int(min_distinct), 3):
        out["reason"] = (f"x has fewer than {max(int(min_distinct), 3)} distinct "
                         f"values: a response is not identifiable from replicates")
        return out
    if span <= rel_span_floor * max(abs(float(x.mean())), 1e-300):
        out["reason"] = "x range is degenerate"
        return out
    xbar = float(x.mean())
    ybar = float(y.mean())
    sxx = float(np.sum((x - xbar) ** 2))
    if sxx <= 0.0:
        out["reason"] = "zero abscissa variance"
        return out
    slope = float(np.sum((x - xbar) * (y - ybar)) / sxx)
    intercept = ybar - slope * xbar
    resid = y - (intercept + slope * x)
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - ybar) ** 2))
    dof = n - 2
    out.update({"status": "fit", "slope": slope, "intercept": intercept})
    if dof > 0:
        out["stderr"] = float(math.sqrt(sse / dof / sxx))
    out["r2"] = (float(1.0 - sse / sst) if sst > 0.0
                 else (1.0 if sse == 0.0 else float("nan")))
    return out


def difference_quotient(x0: float, y0: float, x1: float, y1: float) -> dict:
    """Two-point slope.  Carries **no** uncertainty estimate, by construction."""
    for v in (x0, y0, x1, y1):
        if not _finite(v):
            raise ValueError("difference_quotient needs four finite numbers")
    dx = float(x1) - float(x0)
    if dx == 0.0:
        raise ValueError("difference_quotient needs two distinct abscissae")
    return {"status": "difference_quotient",
            "slope": (float(y1) - float(y0)) / dx,
            "intercept": float("nan"), "stderr": float("nan"),
            "r2": float("nan"), "n": 2, "n_distinct": 2,
            "x_span": abs(dx),
            "reason": "two levels only: sign and magnitude, no significance test"}


def slope_is_resolved(fit: dict, *, sigma: float = RESOLUTION_SIGMA,
                      abs_floor: float = 0.0) -> dict:
    """Can ``fit`` support a statement that the response is non-zero?"""
    if not isinstance(fit, dict):
        raise TypeError("fit must be a dict from ols_slope/difference_quotient")
    slope = fit.get("slope", float("nan"))
    stderr = fit.get("stderr", float("nan"))
    if fit.get("status") != "fit":
        return {"resolved": False,
                "reason": f"fit status {fit.get('status')!r}: "
                          f"{fit.get('reason', 'no slope reported')}"}
    if not _finite(slope):
        return {"resolved": False, "reason": "slope is not finite"}
    if not _finite(stderr):
        return {"resolved": False,
                "reason": "no residual degrees of freedom, so no slope stderr: "
                          "the fit cannot be distinguished from an exact two-parameter "
                          "interpolation"}
    if abs(float(slope)) <= float(sigma) * float(stderr):
        return {"resolved": False,
                "reason": f"|slope| = {abs(float(slope)):.6g} <= {sigma:g} * "
                          f"stderr = {sigma * float(stderr):.6g}"}
    if abs(float(slope)) <= float(abs_floor):
        return {"resolved": False,
                "reason": f"|slope| = {abs(float(slope)):.6g} <= floor "
                          f"{float(abs_floor):.6g}"}
    return {"resolved": True, "reason": ""}


def confound_decomposition(*, slope: float, dphi: float, observed_spread: float,
                           reference: float, resolved: bool,
                           tolerance: float = 0.15,
                           share_comparable: float = CONFOUND_SHARE_COMPARABLE,
                           share_dominated: float = CONFOUND_SHARE_DOMINATED) -> dict:
    """How much of an observed spread can be *attributed* to the packing drift?

    Parameters
    ----------
    slope
        ``d(rho_inf) / d(phi_0)`` in runout-ratio units per unit solid fraction,
        measured on a dedicated ``phi_0`` arm at fixed resolution.
    dphi
        The solid-fraction span that the *other* axis (the resolution ladder)
        sweeps, ``max(phi_0) - min(phi_0)``.
    observed_spread
        The runout spread actually observed along that other axis
        (``max - min`` of ``rho_inf``), same units as ``slope * phi_0``.
    reference
        The reference runout ratio ``rho_inf``, used to convert the implied
        shift into a relative one.
    resolved
        Output of :func:`slope_is_resolved`; ``False`` forces
        ``verdict = "indeterminate"`` -- an unresolved slope may not be used to
        explain anything.

    Verdict vocabulary (pre-registered thresholds 0.2 / 0.5)
    --------------------------------------------------------
    ``packing_dominated``   ``share >= 0.5``: the packing drift alone can
                            produce at least half of the observed spread.
    ``comparable``          ``0.2 <= share < 0.5``: neither axis dominates.
    ``resolution_dominated````share < 0.2``: the spread is not explained by the
                            packing drift.
    ``indeterminate``       the slope is not resolved, or an input is not finite.
    """
    for name, val in (("slope", slope), ("dphi", dphi),
                      ("observed_spread", observed_spread), ("reference", reference)):
        if not _finite(val):
            return {"verdict": "indeterminate",
                    "reason": f"{name} is not finite",
                    "implied_shift": float("nan"),
                    "implied_relative": float("nan"), "share": float("nan"),
                    "exceeds_tolerance_alone": False}
    dphi = abs(float(dphi))
    observed_spread = abs(float(observed_spread))
    reference = abs(float(reference))
    implied = abs(float(slope)) * dphi
    implied_rel = implied / reference if reference > 0.0 else float("nan")
    share = implied / observed_spread if observed_spread > 0.0 else float("nan")
    out = {"implied_shift": implied, "implied_relative": implied_rel,
           "share": share,
           "observed_spread": observed_spread,
           "dphi": dphi,
           "exceeds_tolerance_alone": bool(_finite(implied_rel)
                                           and implied_rel >= tolerance),
           "tolerance": float(tolerance),
           "reason": ""}
    if not resolved:
        out["verdict"] = "indeterminate"
        out["reason"] = "the phi_0 slope is not resolved, so it cannot explain a spread"
        return out
    if not _finite(share):
        out["verdict"] = "indeterminate"
        out["reason"] = "observed spread is zero: nothing to apportion"
        return out
    if share >= share_dominated:
        out["verdict"] = "packing_dominated"
    elif share >= share_comparable:
        out["verdict"] = "comparable"
    else:
        out["verdict"] = "resolution_dominated"
    out["reason"] = (f"implied shift {implied:.4g} ({implied_rel:+.2%} of "
                     f"{reference:.4g}) is {share:.0%} of the observed spread "
                     f"{observed_spread:.4g}")
    return out