"""Free-surface (flank) fitting for a granular / rubble pile -> angle of repose.

Why this module exists
----------------------
The angle of repose is a *metrology* quantity: it is the slope of the linear
flank of a settled pile, not the average slope of everything that happens to
carry the density of ice.  A poured or collapsed pile always has three regions

    apex / plateau      h(r) ~ const          (top of the heap)
    flank               h(r) = H - r tan(a)   (the repose angle lives here)
    apron / foot        h(r) ~ 1 particle     (frustum spread on the floor)

Fitting one straight line through all three regions averages the flank with a
flat apron and reports an angle that is far too small.  Measured with the
shipped whole-profile fit: a *known* 30 deg cone is recovered as 28.67 deg (no
apron), but the same cone with 40 % of its bodies in a one-particle apron is
recovered as **24.09 deg**; with a thicker apron it falls to 16.05 deg -- a
metrology error of 6-14 deg with no physics in it.  That is precisely the regime
the H1 C2 report landed in (0-16 deg), so the number could not be used to accept
or reject the physics.  Every number here is produced by
``benchmarks/h1_broken_ice_dem/make_figures_round3.py`` splitting the profile of
a fixed-seed synthetic pile; see ``docs/H1_BROKEN_ICE_DEM_ROUND3.md``.

What is fitted here
-------------------
* :func:`bin_free_surface` -- annular bins of the pile top (90th percentile of
  the per-body top height, as before, so the estimator stays comparable with
  earlier reports).
* :func:`fit_flank` -- locate the apex bin, then split the outward profile into
  *flank* and *apron* at the break that minimises the total squared error of a
  two-segment linear model (:func:`_two_segment_break`).  The angle is the
  Theil-Sen (median of pairwise slopes) slope of the flank segment only;
  Theil-Sen is used so that one noisy annulus cannot swing the result the way a
  least-squares fit can.

The rules are structural (best two-segment description of the free surface), not
calibrated against the 25-40 deg literature band used by the H1 case --
:func:`fit_flank` never looks at the band.  A pile that has no descending
segment (a monolayer spread on the floor) returns ``angle_deg = nan`` with
``reason`` set instead of a bogus 0 deg: "I cannot see a pile" is a measurement
outcome, not an angle.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

__all__ = ["bin_free_surface", "fit_flank", "NO_PILE"]

NO_PILE = "no_descending_flank"


def bin_free_surface(radii: Sequence[float], tops: Sequence[float],
                     n_bins: int = 7, min_bodies: int = 3,
                     outer_fraction: float = 0.95) -> Dict[str, list]:
    """Annular free-surface profile of a pile.

    ``radii`` and ``tops`` are per-body horizontal distance from the pile
    centroid and per-body top height.  Bins span ``[0, outer_fraction * r_max]``
    so that a single body on the floor cannot stretch the outer bins (which
    would coarsen the profile) and an annulus that holds fewer than
    ``min_bodies`` bodies is dropped instead of being given a percentile of its
    own.

    Returns ``{"radius", "height", "count"}`` lists, one entry per annulus that
    survived, ordered by radius.
    """
    radii = np.asarray(radii, dtype=np.float64)
    tops = np.asarray(tops, dtype=np.float64)
    if radii.size == 0:
        return {"radius": [], "height": [], "count": []}
    r_max = float(np.max(radii))
    if r_max <= 0.0:
        return {"radius": [], "height": [], "count": []}
    edges = np.linspace(0.0, r_max * float(outer_fraction), int(n_bins) + 1)
    out: Dict[str, list] = {"radius": [], "height": [], "count": []}
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (radii >= lo) & (radii < hi)
        count = int(np.count_nonzero(mask))
        if count >= int(min_bodies):
            out["radius"].append(0.5 * (lo + hi))
            out["height"].append(float(np.percentile(tops[mask], 90)))
            out["count"].append(count)
    return out


def _theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Median of all pairwise slopes (robust simple linear regression)."""
    slopes = []
    for i in range(x.size):
        for j in range(i + 1, x.size):
            dx = x[j] - x[i]
            if abs(dx) > 0.0:
                slopes.append((y[j] - y[i]) / dx)
    if not slopes:
        return float("nan")
    return float(np.median(slopes))


def _sse(x: np.ndarray, y: np.ndarray) -> float:
    """Residual sum of squares of the least-squares line through ``(x, y)``."""
    if x.size < 2:
        return 0.0
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    return float(np.sum(resid * resid))


def _segment_bic(n_points: int, sse: float, n_params: int) -> float:
    """BIC of a piecewise-linear model (lower is better).

    ``n * ln(SSE/n) + k * ln(n)`` with ``k`` free parameters (2 per segment).
    The penalty is what stops the break search from cutting a clean cone into
    two short segments: splitting a straight profile only pays for itself when
    it explains materially more of the profile than the extra parameters cost.
    """
    if n_points <= 0:
        return float("inf")
    return float(n_points * math.log(max(sse / n_points, 1e-30))
                 + n_params * math.log(n_points))


def _two_segment_break(radius: np.ndarray, height: np.ndarray, apex: int,
                       min_segment: int = 3,
                       require_flatter_apron: bool = True,
                       min_apron_bins: int = 2):
    """Split ``[apex:]`` into flank + apron by BIC (see :func:`_segment_bic`).

    The single-segment model (the whole outward profile is one flank) competes
    with every *admissible* two-segment break; the winner is returned.  A clean
    cone keeps a single segment (break ``None``), a pile with a foot gets a
    break at the toe.

    Admissibility (``require_flatter_apron``) has two structural parts:

    * an apron is the *flattened* foot of the pile, so the projection's outward
      descent cannot be steeper than the flank's.  A break that puts the steeper
      segment outboard is a convexity of one single cone, not an apron; splitting
      there and reporting the inner (shallower) slope is exactly how a clean
      40 deg cone used to be reported as 35.10 deg (see
      ``runs/h1_broken_ice_dem/round3/round3_figures.json``, column
      ``flank_clean``, and ``fig1_pre_fix_ablation``);
    * the apron must be resolved by at least ``min_apron_bins`` annuli -- a
      single annulus is one point, not a segment, so it cannot be shown to be a
      foot (and it truncates the flank of a clean cone).

    Pass ``False`` to recover the unconstrained (pre-fix) behaviour, i.e. no
    slope test and a one-annulus apron allowed; the tests use that as baseline.

    Returns ``(break_index, sse_flank, sse_apron, bic)`` where ``break_index`` is
    the last flank index (``None`` = no break, whole profile is the flank).
    """
    n = radius.size
    m = n - apex
    # a one-annulus "apron" is only reachable when the structural rules are off
    min_apron = int(min_apron_bins) if require_flatter_apron else 1
    if m < int(min_segment) + min_apron:
        return None, float("nan"), float("nan"), float("inf")
    best_bic = _segment_bic(m, _sse(radius[apex:], height[apex:]), 2)
    best = (None, _sse(radius[apex:], height[apex:]), float("nan"), best_bic)
    for k in range(apex + int(min_segment) - 1, n - min_apron):
        flank = slice(apex, k + 1)
        apron = slice(k + 1, n)
        if flank.stop - flank.start < int(min_segment):
            continue
        sse_flank = _sse(radius[flank], height[flank])
        n_apron = apron.stop - apron.start
        if n_apron >= 2:
            if require_flatter_apron:
                flank_slope = _theil_sen_slope(radius[flank], height[flank])
                apron_slope = _theil_sen_slope(radius[apron], height[apron])
                if (math.isfinite(flank_slope) and math.isfinite(apron_slope)
                        and apron_slope < flank_slope):
                    # steeper outboard segment: not a foot, so not a split
                    continue
            sse_apron = _sse(radius[apron], height[apron])
            sse, n_params = sse_flank + sse_apron, 4
        elif n_apron == 1:
            sse_apron, sse, n_params = 0.0, sse_flank, 3
        else:
            sse_apron, sse, n_params = float("nan"), sse_flank, 3
        bic = _segment_bic(m, sse, n_params)
        if bic < best_bic - 1e-12:
            best = (k, sse_flank, sse_apron, bic)
            best_bic = bic
    return best


def fit_flank(radius: Sequence[float], height: Sequence[float],
              count: Sequence[int] | None = None,
              min_flank_bins: int = 3,
              min_drop: float = 0.0,
              require_flatter_apron: bool = True,
              min_apron_bins: int = 2) -> Dict[str, object]:
    """Angle of the descending flank of a free-surface profile.

    Steps
    -----
    1. Start from the bin with the largest height (the apex of the pile).
    2. Split the outward profile into *flank* and *apron* with the BIC-scored
       two-segment model (:func:`_two_segment_break`), admitting only splits
       whose apron is no steeper than the flank and covers at least
       ``min_apron_bins`` annuli (``require_flatter_apron``).
    3. Take the Theil-Sen slope of the flank and convert it to degrees.

    A flank shorter than ``min_flank_bins`` annuli, one whose slope is not
    negative, or one whose height drop is below ``min_drop`` (pass a particle
    radius: a heap with less relief than one particle is not a heap) means no
    pile was resolved; the function then returns ``angle_deg = nan`` with
    ``reason = NO_PILE`` so callers cannot mistake "no pile" for "a pile with a
    0 deg slope".

    Returns ``angle_deg``, ``n_flank_bins``, ``n_profile_bins``,
    ``apex_radius_m``, ``toe_radius_m``, ``flank_height_drop_m``,
    ``apron_angle_deg`` (slope of what was excluded, so a reader can check the
    exclusion) and ``apron_body_share``.
    """
    r = np.asarray(radius, dtype=np.float64)
    h = np.asarray(height, dtype=np.float64)
    c = np.asarray(count if count is not None else np.ones_like(r), dtype=np.int64)
    result: Dict[str, object] = {
        "angle_deg": float("nan"), "provisional_angle_deg": float("nan"),
        "n_flank_bins": 0, "n_profile_bins": int(r.size),
        "apex_radius_m": float("nan"), "toe_radius_m": float("nan"),
        "flank_height_drop_m": float("nan"), "apron_angle_deg": float("nan"),
        "apron_body_share": float("nan"), "reason": NO_PILE,
    }
    if r.size < int(min_flank_bins):
        return result

    apex = int(np.argmax(h))
    result["apex_radius_m"] = float(r[apex])
    if r.size - apex < int(min_flank_bins):
        return result

    break_index, _flank_sse, _apron_sse, _bic = _two_segment_break(
        r, h, apex, min_segment=int(min_flank_bins),
        require_flatter_apron=bool(require_flatter_apron),
        min_apron_bins=int(min_apron_bins))
    stop = break_index + 1 if break_index is not None else r.size
    if stop - apex < int(min_flank_bins):
        return result

    slope = _theil_sen_slope(r[apex:stop], h[apex:stop])
    if not math.isfinite(slope) or slope >= 0.0:
        return result

    drop = float(h[apex] - h[stop - 1])
    if drop < float(min_drop):
        # a "flank" with less relief than the measurement resolution is a
        # monolayer with noise on it, not a pile
        result["flank_height_drop_m"] = drop
        result["provisional_angle_deg"] = float(math.degrees(math.atan(-slope)))
        result["n_flank_bins"] = int(stop - apex)
        result["toe_radius_m"] = float(r[stop - 1])
        return result

    result["angle_deg"] = float(math.degrees(math.atan(-slope)))
    result["n_flank_bins"] = int(stop - apex)
    result["toe_radius_m"] = float(r[stop - 1])
    result["flank_height_drop_m"] = drop
    result["reason"] = "ok"
    if r.size - stop >= 2:
        apron_slope = _theil_sen_slope(r[stop:], h[stop:])
        if math.isfinite(apron_slope):
            result["apron_angle_deg"] = float(math.degrees(math.atan(max(-apron_slope, 0.0))))
    if c.sum() > 0:
        result["apron_body_share"] = float(c[stop:].sum() / c.sum())
    return result