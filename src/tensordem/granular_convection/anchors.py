"""Literature anchors for the granular-convection verification face.

Every anchor carries an explicit ``citation`` string and an ``uncertainty``
field.  The acceptance bookkeeping distinguishes two verdicts that are easy to
confuse:

``within_range``
    the model value falls inside the reported experimental window -- the
    strongest statement that can honestly be made against a scattered
    experimental literature;
``rel_error`` / ``rel_error_ok``
    the point-wise relative error against the reported central value, with the
    acceptance tolerance of the node (10 % for the segregation timescale).

Recording both avoids the trap of quoting a 10 % point error from data whose own
scatter is 50 %.
"""
from __future__ import annotations

import math

__all__ = [
    "BNE_ANCHORS",
    "RBNE_BOUNDARY_ANCHOR",
    "ONSET_ANCHOR",
    "anchor_errors",
    "point_error",
    "interpolated_boundary",
]

#: Onset of Brazil-nut convection in a vertically shaken bed: the bed must
#: fluidise in the dilation phase, which happens for ``Gamma`` above ~1.
ONSET_ANCHOR = {
    "name": "bne_onset_gamma",
    "quantity": "critical shaking intensity Gamma_c for segregation onset",
    "value": 1.2,
    "range": (1.0, 1.5),
    "unit": "-",
    "citation": "Fluidisation onset of a vertically shaken granular bed, "
                "Gamma_c ~ 1 (widely reproduced; also the value used by "
                "tensordem.bidisperse_segregation and tensordem.segregation)",
}

#: Segregation timescale for a large intruder in a bed ~10-20 particle diameters
#: deep.  The literature reports "a few tens of cycles"; the scatter of the
#: published values for comparable cells is large, hence the wide range.
BNE_ANCHORS = {
    "segregation_timescale_cycles": {
        "name": "bne_segregation_timescale",
        "quantity": "cycles to reach ~63 % of the asymptotic Brazil-nut rise",
        "value": 30.0,
        "range": (15.0, 60.0),
        "unit": "cycles",
        "citation": "Experimental Brazil-nut effect in a vertically vibrated "
                    "bed: the large intruder reaches the free surface within a "
                    "few tens of vibration cycles (Shinbrot et al. 1996 regime "
                    "of granular segregation; Shinbrot & Muzzio reviews). "
                    "Reported scatter for comparable cells: +/-50 %.",
    },
    "size_ratio_for_bne": {
        "name": "bne_size_ratio",
        "quantity": "minimum size ratio for size-driven segregation",
        "value": 1.2,
        "range": (1.1, 1.5),
        "unit": "-",
        "citation": "Size segregation needs d_large/d_small >~ 1.1-1.5; below it "
                    "percolation cannot open a persistent void.",
    },
}

#: Reverse Brazil-nut boundary: for a size ratio of 2 the density contrast that
#: overturns the size-driven rise sits around 2 (Hong, Quinn & Luding 2001
#: report the BNE->RBNE turnover as a competition between the percolation and
#: the condensation mechanisms, with the critical density ratio increasing with
#: the size ratio).
RBNE_BOUNDARY_ANCHOR = {
    "name": "rbne_density_boundary",
    "quantity": "critical density ratio rho_large/rho_small for BNE->RBNE",
    "size_ratio": 2.0,
    "value": 2.0,
    "range": (1.4, 3.4),
    "unit": "-",
    "citation": "Hong, D.C., Quinn, P.V. & Luding, S. (2001) Phys. Rev. Lett. "
                "86, 3423: reverse Brazil-nut effect for a dense large species; "
                "the crossover density ratio grows with the size ratio and lies "
                "in the 1.4-3.4 band for d_l/d_s = 2.",
}


def point_error(value: float, anchor: dict) -> dict:
    """Relative error of ``value`` against an anchor's central value."""
    central = float(anchor["value"])
    lo, hi = anchor["range"]
    err = (value - central) / central if central else float("inf")
    return {
        "value": float(value),
        "anchor": central,
        "anchor_range": [float(lo), float(hi)],
        "rel_error": float(err),
        "abs_rel_error": float(abs(err)),
        "within_range": bool(lo <= value <= hi),
        "citation": anchor["citation"],
    }


def anchor_errors(values: dict, tolerance: float) -> dict:
    """Apply :func:`point_error` to a mapping ``{anchor_name: value}``.

    Returns the per-anchor verdict plus an overall ``pass`` that requires both
    inclusion in the experimental range and the point tolerance.
    """
    results = {}
    for name, value in values.items():
        if name not in BNE_ANCHORS:
            raise KeyError(f"unknown anchor {name!r}")
        block = point_error(float(value), BNE_ANCHORS[name])
        block["tolerance"] = float(tolerance)
        block["pass"] = bool(block["within_range"] and block["abs_rel_error"] <= tolerance)
        results[name] = block
    return {"anchors": results,
            "all_pass": all(b["pass"] for b in results.values())}


def interpolated_boundary(points, *, rtol: float = 1e-9) -> list[dict]:
    """Locate the BNE/RBNE sign change along density ratio at fixed ``Gamma``.

    ``points`` is a sequence of objects/dicts with ``gamma``, ``density_ratio``
    and ``dz`` (or ``direction``).  The boundary is linearly interpolated between
    the last Brazil-nut and the first reverse-Brazil-nut density ratio, so the
    estimate is not quantised by the scan grid.
    """
    rows = []
    for p in points:
        if isinstance(p, dict):
            rows.append((float(p["gamma"]), float(p["density_ratio"]), float(p["dz"])))
        else:
            rows.append((float(p.gamma), float(p.density_ratio), float(p.dz)))
    by_gamma: dict[float, list[tuple[float, float]]] = {}
    for gamma, ratio, dz in rows:
        by_gamma.setdefault(gamma, []).append((ratio, dz))
    out = []
    for gamma in sorted(by_gamma):
        seq = sorted(by_gamma[gamma])
        boundary = None
        for (r0, d0), (r1, d1) in zip(seq, seq[1:]):
            if d0 == 0.0:
                boundary = r0
                break
            if (d0 > 0.0) != (d1 > 0.0):
                frac = abs(d0) / (abs(d0) + abs(d1))
                boundary = r0 + frac * (r1 - r0)
                break
        out.append({"gamma": gamma, "rho_boundary": boundary,
                    "scan": [[float(r), float(d)] for r, d in seq]})
    return out