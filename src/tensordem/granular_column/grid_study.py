"""Grid (resolution) convergence assessment for the column-collapse metrics.

Why this module exists
----------------------
``CASE-COLUMN-COLLAPSE`` is judged by comparing a discrete quantity (the
runout ratio ``r_inf/r_i`` and the emplacement time) against the empirical
laws of Lube et al. (2004) with a **15 % tolerance**.  Every previous round of
this anchor compared the *finest available* resolution against the reference
and declared band / out-of-band.  That test is only meaningful if the
discretisation error is itself smaller than the tolerance, and until this
module existed nobody had quantified it: the anchor's single out-of-band point
(``a = 3``, 12 grains/``r_i``, +15.4 %) sits *between* two in-band points
(8 grains/``r_i``: +8.1 %, 16 grains/``r_i``: +11.1 %), i.e. the resolution
sequence is not monotone, so the classic grid-convergence estimators do not
apply to it as-is.

What is computed
----------------
Three estimators, in decreasing order of strictness.  The caller never picks
one silently: :func:`assess_ladder` returns *which* estimator fired and why.

1. **Observed order + Richardson extrapolation + GCI** (Roache 1994, 1998;
   ASME V&V 20).  Requires three levels in a *uniform* refinement ratio
   ``r = h_coarse/h_mid = h_mid/h_fine`` and a *monotone* sequence
   ``(f3 - f2)/(f2 - f1) > 0``.  Then

   .. math::

       p = \\frac{\\ln[(f_2-f_1)/(f_3-f_2)]}{\\ln r},\\qquad
       f_{\\rm exact} \\simeq f_3 + \\frac{f_3-f_2}{r^p-1},\\qquad
       U_{\\rm GCI} = F_s\\frac{|(f_3-f_2)/f_3|}{r^p-1}

   with the safety factor ``F_s = 1.25`` of ASME V&V 20.  The asymptotic
   consistency check ``GCI_21 / (r^p GCI_32) ~ 1`` is returned so the caller
   can see whether the triple is actually in the asymptotic range.

2. **Oscillatory range band** — when the sequence is non-monotone the observed
   order is undefined (``p`` would be complex).  GCI is then *not applicable*;
   reporting it anyway is a known way to produce a confident wrong number.  We
   fall back to the half-spread of **all** levels of the ladder,

   .. math:: U_{\\rm band} = \\tfrac12 (\\max_i f_i - \\min_i f_i)

   which is an honest *range*, not an uncertainty at a stated coverage
   probability: it says "the answer moves by this much when the only thing
   changed is the resolution".  It is reported as such.

3. **Insufficient** — fewer than three levels: no statement is possible.  The
   verdict is ``"insufficient"`` and no uncertainty is invented.

The verdict for the anchor
--------------------------
:func:`adjudicate` combines a signed error ``e = (f - f_ref)/f_ref`` with an
uncertainty fraction ``u`` and the card tolerance ``tol``:

* ``"in_band"``            -- ``|e| <= tol``;
* ``"out_of_band"``        -- ``|e| > tol`` **and** ``|e| - u > tol``: the miss
  survives the discretisation uncertainty;
* ``"indeterminate"``      -- ``|e| > tol`` but ``|e| - u <= tol``: the miss is
  inside the resolution scatter, so the reference comparison cannot adjudicate
  it.  This is the case the anchor's ``a = 3`` point falls into if the ladder
  is band-limited.

Units: ``h`` in metres (any consistent length unit), ``f`` and ``u``
dimensionless.  No DEM, no numpy-only helpers that need the kernel: this module
is pure arithmetic so the tests can pin it against closed-form synthetic
sequences.
"""

from __future__ import annotations

import math

__all__ = [
    "ASME_FS",
    "uniform_triples",
    "LADDER_INSUFFICIENT",
    "LADDER_GCI",
    "LADDER_BAND",
    "refinement_ratios",
    "observed_order",
    "richardson_extrapolate",
    "gci",
    "range_band",
    "assess_ladder",
    "adjudicate",
]

#: Safety factor of ASME V&V 20-2009 (applied to the GCI).
ASME_FS = 1.25

#: Estimator tags returned by :func:`assess_ladder`.
LADDER_GCI = "gci"
LADDER_BAND = "oscillatory-band"
LADDER_INSUFFICIENT = "insufficient"

#: Relative tolerance on the uniformity of the refinement ratio.
_RATIO_RTOL = 1.0e-6


class GridStudyError(ValueError):
    """Raised for malformed ladders (wrong ordering, duplicate levels, ...)."""


def _sorted_levels(levels):
    """Validate ``[(h, f), ...]`` and sort coarse -> fine (decreasing ``h``)."""
    pairs = []
    for lev in levels:
        h, f = lev
        h = float(h)
        f = float(f)
        if not math.isfinite(h) or h <= 0.0:
            raise GridStudyError(f"level spacing h must be positive finite, got {h!r}")
        if not math.isfinite(f):
            raise GridStudyError(f"level value f must be finite, got {f!r}")
        pairs.append((h, f))
    if len(pairs) < 2:
        raise GridStudyError("a ladder needs at least two levels")
    pairs.sort(key=lambda t: -t[0])
    for a, b in zip(pairs, pairs[1:]):
        if abs(a[0] - b[0]) <= _RATIO_RTOL * max(a[0], b[0]):
            raise GridStudyError(f"duplicate level spacing h = {a[0]!r}")
    return pairs


def refinement_ratios(levels):
    """Successive ratios ``h_coarse/h_fine`` of a ladder (coarse -> fine)."""
    pairs = _sorted_levels(levels)
    return [a[0] / b[0] for a, b in zip(pairs, pairs[1:])]


def uniform_triples(levels) -> list:
    """All index triples ``(i, j, k)`` of a coarse->fine ladder with equal ratios.

    A ladder assembled for a grid study often contains a *designed* uniform
    triple (e.g. grains-per-radius 4, 8, 16 -> ``r = 2``) plus filler rungs.
    Looking only at the first three levels would then miss the triple the
    ladder was built around and report a spurious "non-uniform ratio".
    """
    pairs = _sorted_levels(levels)
    out = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            for k in range(j + 1, len(pairs)):
                r1, r2 = pairs[i][0] / pairs[j][0], pairs[j][0] / pairs[k][0]
                if abs(r1 - r2) <= _RATIO_RTOL * max(r1, r2):
                    out.append((i, j, k))
    return out


def observed_order(h1: float, h2: float, h3: float,
                   f1: float, f2: float, f3: float) -> dict:
    """Three-level observed order of accuracy ``p`` (uniform refinement only).

    ``(h1, f1)`` is the coarsest level and ``(h3, f3)`` the finest.  Returns
    ``{"p": float|None, "r": float, "monotone": bool, "reason": str}``.  ``p``
    is ``None`` whenever the triple cannot support a real order: non-uniform
    ratio, or a non-monotone / equal-successive-change sequence.
    """
    h1, h2, h3 = float(h1), float(h2), float(h3)
    f1, f2, f3 = float(f1), float(f2), float(f3)
    if not (h1 > h2 > h3 > 0.0):
        raise GridStudyError("require h1 > h2 > h3 > 0 (coarse -> fine)")
    r1, r2 = h1 / h2, h2 / h3
    if abs(r1 - r2) > _RATIO_RTOL * max(r1, r2):
        return {"p": None, "r": None, "monotone": None,
                "reason": (f"non-uniform refinement ratio ({r1:.6g} vs {r2:.6g}); "
                           "three-level Richardson requires a uniform ratio")}
    d12, d23 = f2 - f1, f3 - f2
    if d12 == 0.0:
        return {"p": None, "r": r1, "monotone": None,
                "reason": "f2 == f1: the sequence does not move, order undefined"}
    ratio = d23 / d12
    if ratio <= 0.0:
        return {"p": None, "r": r1, "monotone": False,
                "reason": ("non-monotone sequence (successive changes change sign); "
                           "the observed order is undefined and any GCI would be "
                           "an artefact")}
    if abs(ratio - 1.0) <= 1.0e-12:
        return {"p": None, "r": r1, "monotone": True,
                "reason": "successive changes are equal: p -> 0, extrapolation diverges"}
    # ``d23/d12 = r**-p`` (the changes scale as h**-p when refining), so the
    # order is ``ln(d12/d23)/ln r``.  Getting this ratio upside down flips the
    # sign of p; the closed-form test with a known p pins it.
    p = math.log(d12 / d23) / math.log(r1)
    return {"p": p, "r": r1, "monotone": True, "reason": "ok"}


def richardson_extrapolate(f2: float, f3: float, p: float, r: float) -> float:
    """Richardson (h -> 0) extrapolation of the two finest levels."""
    if p == 0.0:
        raise GridStudyError("p == 0 makes the Richardson correction singular")
    return float(f3) + (float(f3) - float(f2)) / (float(r) ** float(p) - 1.0)


def gci(f1: float, f2: float, f3: float, p: float, r: float,
        fs: float = ASME_FS) -> dict:
    """Grid Convergence Index of the fine pair, and the asymptotic ratio check.

    Returns ``{"gci_fine", "gci_mid", "asymptotic_ratio", "f_extrapolated",
    "u_rel"}``; ``u_rel`` is the GCI of the fine level expressed as a fraction
    of ``f3`` -- the number to compare against a tolerance.
    """
    if p <= 0.0:
        raise GridStudyError("GCI requires p > 0")
    f1, f2, f3 = float(f1), float(f2), float(f3)
    rp = float(r) ** float(p)
    if abs(rp - 1.0) <= 1.0e-14:
        raise GridStudyError("r**p == 1 makes the GCI singular")
    g_fine = fs * abs((f3 - f2) / f3) / (rp - 1.0)
    g_mid = fs * abs((f2 - f1) / f2) / (rp - 1.0)
    ratio = g_mid / (rp * g_fine) if g_fine > 0.0 else float("nan")
    return {
        "gci_fine": g_fine,
        "gci_mid": g_mid,
        "asymptotic_ratio": ratio,
        "f_extrapolated": richardson_extrapolate(f2, f3, p, r),
        "u_rel": g_fine,
    }


def range_band(values) -> dict:
    """Half-spread of a set of level values (the oscillatory fallback)."""
    vals = [float(v) for v in values]
    if len(vals) < 2:
        raise GridStudyError("a band needs at least two values")
    if any(not math.isfinite(v) for v in vals):
        raise GridStudyError("band values must be finite")
    lo, hi = min(vals), max(vals)
    finest = vals[-1]
    half = 0.5 * (hi - lo)
    return {
        "min": lo, "max": hi, "half_spread": half,
        "mean": sum(vals) / len(vals),
        "u_rel": half / abs(finest) if finest != 0.0 else float("nan"),
    }


def assess_ladder(levels, reference: float | None = None,
                  tol: float = 0.15, fs: float = ASME_FS) -> dict:
    """Assess one metric along a resolution ladder.

    ``levels`` is ``[(h, f), ...]`` with ``h`` the resolution length (e.g. the
    particle diameter) and ``f`` the measured quantity; order does not matter.
    ``reference`` (optional) is the experimental/analytic target, used to
    produce the signed error and the :func:`adjudicate` verdict for the finest
    level.

    Returns a dict with ``method`` in ``{"gci", "oscillatory-band",
    "insufficient"}``, the supporting numbers, and -- when a reference is given
    -- ``error_finest``, ``u_rel``, ``adjudication`` and the tolerance
    comparison.  ``u_rel`` is always *the estimator that the method tag names*;
    it is never silently swapped.
    """
    pairs = _sorted_levels(levels)
    out = {
        "n_levels": len(pairs),
        "h": [p[0] for p in pairs],
        "f": [p[1] for p in pairs],
        "h_finest": pairs[-1][0],
        "f_finest": pairs[-1][1],
        "tolerance": float(tol),
        "safety_factor": float(fs),
        "method": LADDER_INSUFFICIENT,
        "u_rel": None,
        "reason": "",
    }
    band = range_band([p[1] for p in pairs]) if len(pairs) >= 2 else None
    if band is not None:
        out["band"] = band

    if len(pairs) < 3:
        out["reason"] = "fewer than three levels: no convergence statement possible"
    else:
        trips = uniform_triples(pairs)
        # Prefer the triple whose finest level is the finest available: the GCI
        # is a statement about the finest resolution we actually paid for.
        trips.sort(key=lambda t: (pairs[t[2]][0], -t[0]))
        if trips:
            i, j, k = trips[0]
            out["triple"] = {"indices": [i, j, k],
                             "h": [pairs[i][0], pairs[j][0], pairs[k][0]]}
            order = observed_order(pairs[i][0], pairs[j][0], pairs[k][0],
                                   pairs[i][1], pairs[j][1], pairs[k][1])
        else:
            out["triple"] = None
            order = {"p": None, "r": None, "monotone": None,
                     "reason": ("no uniform-ratio triple in the ladder, so no "
                                "three-level order of accuracy exists")}
        out["order"] = order
        if order["p"] is not None and order["p"] > 0.0:
            i0, j0, k0 = out["triple"]["indices"]
            est = gci(pairs[i0][1], pairs[j0][1], pairs[k0][1],
                      order["p"], order["r"], fs=fs)
            out["method"] = LADDER_GCI
            out["gci"] = est
            out["u_rel"] = est["u_rel"]
            out["reason"] = (f"monotone triple, observed order p = {order['p']:.4g}, "
                             f"r = {order['r']:.6g}")
        else:
            out["method"] = LADDER_BAND
            out["u_rel"] = band["u_rel"]
            out["reason"] = (order["reason"] + "; fell back to the half-spread of "
                             "all levels (a range, not a coverage interval)")

    if reference is not None:
        ref = float(reference)
        finest = pairs[-1][1]
        err = (finest - ref) / ref if ref != 0.0 else float("nan")
        out["reference"] = ref
        out["error_finest"] = err
        out["u_rel"] = out["u_rel"]
        out["adjudication"] = adjudicate(err, out["u_rel"], tol)
        out["error_minus_u"] = abs(err) - (out["u_rel"] or 0.0)
    return out


def adjudicate(error: float, u_rel: float | None, tol: float) -> str:
    """Classify a signed reference error against tolerance and uncertainty.

    ``"in_band"`` / ``"out_of_band"`` / ``"indeterminate"`` as documented in
    the module docstring.  ``u_rel is None`` (fewer than three levels) yields
    ``"indeterminate"`` whenever the error leaves the band: with no
    discretisation estimate the miss cannot be attributed.
    """
    e = float(error)
    if not math.isfinite(e):
        return "indeterminate"
    if abs(e) <= tol:
        return "in_band"
    if u_rel is None:
        return "indeterminate"
    u = abs(float(u_rel))
    return "out_of_band" if (abs(e) - u) > tol else "indeterminate"