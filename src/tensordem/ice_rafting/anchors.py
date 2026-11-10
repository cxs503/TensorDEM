"""Literature anchors for the ice rafting / ridging verification face.

Honesty contract
----------------
Every anchor states its central value, its observed range and its citation.  The
rafting-onset pressures of thin ice are scattered by roughly a factor of two
between events (floe size distribution, convergence rate, temperature, snow
load), so a point-wise comparison at 15 % is only meaningful *inside* that
scatter: :func:`compare` reports both the range test and the point relative
error, and the report quotes both.
"""
from __future__ import annotations

__all__ = [
    "RAFTING_ANCHORS",
    "RIDGE_GEOMETRY_ANCHORS",
    "compare",
    "bracket_verdict",
]

#: Field-observation anchors for the rafting / ridge-building onset pressure of
#: a compressive ice sheet.  The onset pressure is quoted per unit ice thickness
#: so the same anchor family can be compared with any thickness.
RAFTING_ANCHORS = {
    "rafting_onset_h050": {
        "quantity": "rafting/ridge-building onset sheet stress, h = 0.5 m",
        "value": 1.5e4,
        "range": (1.0e4, 2.5e4),
        "unit": "Pa",
        "citation": "Keinonen et al. (1996) field ice-pressure compilation: "
                    "rafting and ridge building of thin first-year ice begin at "
                    "sheet stresses of order 10-25 kPa for h ~ 0.5 m; event-to-"
                    "event scatter ~ +/-50 %.",
    },
    "rafting_onset_h100": {
        "quantity": "rafting/ridge-building onset sheet stress, h = 1.0 m",
        "value": 2.5e4,
        "range": (1.8e4, 4.0e4),
        "unit": "Pa",
        "citation": "Keinonen et al. (1996) field ice-pressure compilation: "
                    "onset stress grows with thickness at roughly 20-40 kPa per "
                    "metre of ice for first-year ice.",
    },
}

#: Ridge geometry anchors (sail above water, keel below).
RIDGE_GEOMETRY_ANCHORS = {
    "sail_keel_ratio": {
        "quantity": "sail height / keel depth H_s / H_k",
        "value": 1.0 / 4.4,
        "range": (0.18, 0.33),
        "unit": "-",
        "citation": "Timco & Burden (1994), A review of the mechanical "
                    "properties of sea ice ridges: keel depth H_k = 4.4 H_s, "
                    "i.e. H_s/H_k = 0.227 (observed 0.18-0.33).",
    },
    "sail_aspect": {
        "quantity": "sail height / sail width",
        "value": 0.15,
        "range": (0.08, 0.30),
        "unit": "-",
        "citation": "Ridge sail slope statistics (Timco & Burden 1994; "
                    "Lepparanta, The Drift of Sea Ice): sail flanks stand at "
                    "~1:4 to 1:10, i.e. H_s/W_s ~ 0.1-0.25.",
    },
}


def compare(value: float, anchor: dict) -> dict:
    """Range test plus point relative error against one anchor."""
    central = float(anchor["value"])
    lo, hi = anchor["range"]
    err = (float(value) - central) / central if central else float("inf")
    return {
        "value": float(value),
        "anchor": central,
        "anchor_range": [float(lo), float(hi)],
        "rel_error": float(err),
        "abs_rel_error": float(abs(err)),
        "within_range": bool(lo <= value <= hi),
        "citation": anchor["citation"],
    }


def bracket_verdict(lower: float, upper: float, anchor: dict) -> dict:
    """Does ``[lower, upper]`` bracket the anchor's observed range?

    Used for the two-limit rafting threshold: the rigid-floe DEM and the closed
    form give the upper bound (the whole floe must be lifted), the flexural limit
    gives the lower bound (a long floe bends and breaks, so only the elastic
    foundation length takes part).
    """
    lo, hi = anchor["range"]
    ok = min(lower, upper) <= lo and max(lower, upper) >= hi
    return {
        "lower": float(min(lower, upper)),
        "upper": float(max(lower, upper)),
        "anchor_range": [float(lo), float(hi)],
        "anchor_value": float(anchor["value"]),
        "brackets": bool(ok),
        "citation": anchor["citation"],
    }