"""Reference anchors the brash-channel model is compared against.

Primary reference (criterion b)
-------------------------------
:func:`reference_anchor` wraps the in-repository, hashed broken-ice/brash closed
form :func:`tensordem.ice_resistance_scaling.broken_ice_anchor`

.. math:: R = R_c + R_s + R_m + R_b ,
          R_c = 0.5\\,\\sigma_c H^2(\\tan\\alpha+\\mu/\\cos\\alpha),\\quad
          R_s = \\rho_w g H^2 B(0.9+0.05\\mu L/B),\\quad
          R_m = 0.1\\,\\rho_i H B v^2 ,

with the level-ice bending term suppressed for floe diameters below the
characteristic bending length (always true here: floe 0.5 m << lambda).

Why this anchor and not Suyuthi et al. (2013)
---------------------------------------------
The campaign node names Suyuthi et al. (2013).  That paper and the FSICR
commentary are **not readable in this offline sandbox** - the same provenance
policy as :mod:`tensordem.level_ice.formula`, whose ledger marks literature
agreement WITHHELD.  The node's comparison is therefore run against the
in-repository brash anchor above (FSICR-family structure: crushing + submersion
+ momentum flux), and the substitution is recorded in the report rather than
papered over.

Component-level caveat (kept visible on purpose)
------------------------------------------------
The anchor's crushing term carries the *intact* compressive strength over the
full ice thickness and no beam dependence, so for the standard case it makes up
``2/3`` of the anchor total at B = 15 m and only ``1/2`` at B = 35 m.  Its
B-exponent is 0.25 against the 1.0 of any geometrically similar model, which is
the origin of the residual in criterion (b) - see
:func:`scaling_diagnostic`.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

__all__ = [
    "reference_anchor",
    "anchor_components",
    "level_ice_reference",
    "scaling_diagnostic",
]

from .material import BrashIceState
from .model import (BrashHull, Closure, STANDARD_BEAMS_M, brash_resistance,
                    standard_hull)


def _ship_case(state: BrashIceState, hull: BrashHull, speed: float):
    from tensordem.ice_resistance_scaling import ShipCase
    return ShipCase(
        length_m=hull.length_m, beam_m=hull.beam_m, draught_m=hull.draught_m,
        speed_m_s=float(speed), thickness_m=state.layer_thickness_m,
        concentration=1.0, entry_angle_deg=hull.entry_angle_deg,
        friction_coefficient=0.15, strength_pa=1.0e6,
        flexural_strength_pa=0.5e6, elastic_modulus_pa=5.0e9,
        floe_diameter_m=state.floe_diameter_m,
    )


def reference_anchor(state: BrashIceState, hull: BrashHull,
                     speed: float = 3.0) -> Dict[str, object]:
    """In-repository brash/broken-ice reference anchor (criterion b target)."""
    from tensordem.ice_resistance_scaling import broken_ice_anchor
    payload = broken_ice_anchor(_ship_case(state, hull, speed))
    payload = dict(payload)
    payload["total_kN"] = payload["total_N"] / 1.0e3
    payload["source"] = ("tensordem.ice_resistance_scaling.broken_ice_anchor "
                         "(FSICR-family structure)")
    return payload


def anchor_components(state: BrashIceState, hull: BrashHull,
                      speed: float = 3.0) -> Dict[str, float]:
    """Anchor total split into its own components, in N."""
    payload = reference_anchor(state, hull, speed)
    return {
        "crushing_N": float(payload["crushing_N"]),
        "bending_N": float(payload["bending_N"]),
        "submersion_N": float(payload["submersion_N"]),
        "momentum_N": float(payload["momentum_N"]),
        "total_N": float(payload["total_N"]),
    }


def level_ice_reference(state: BrashIceState, hull: BrashHull,
                        speed: float = 3.0) -> Dict[str, object]:
    """Lindqvist intact level-ice anchor - an *upper* bound, for context only."""
    from tensordem.level_ice.formula import HullForm, IceState, lindqvist_components
    ice = IceState(thickness_m=state.layer_thickness_m,
                   compressive_strength_pa=1.0e6,
                   flexural_strength_pa=0.5e6,
                   elastic_modulus_pa=5.0e9,
                   density_kg_m3=state.solid_density_kg_m3)
    ship = HullForm(length_m=hull.length_m, beam_m=hull.beam_m,
                    draught_m=hull.draught_m)
    payload = dict(lindqvist_components(ice, ship, speed,
                                        water_density_kg_m3=state.water_density_kg_m3,
                                        gravity_m_s2=state.gravity_m_s2))
    payload["total_kN"] = payload["total_N"] / 1.0e3
    payload["source"] = "tensordem.level_ice.formula.lindqvist_components"
    return payload


def _loglog_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pairs) < 2:
        raise ValueError("need at least two positive points")
    mx = sum(p[0] for p in pairs) / len(pairs)
    my = sum(p[1] for p in pairs) / len(pairs)
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    return sxy / sxx


def scaling_diagnostic(state: BrashIceState = None,
                       beams: Sequence[float] = STANDARD_BEAMS_M,
                       speed: float = 3.0,
                       closure: Closure = None) -> Dict[str, object]:
    """Beam-scaling exponents of the model and of the reference anchor.

    The residual of criterion (b) is a *slope* mismatch: all three model
    components scale with the beam (geometric similarity, exponent ~1), while
    the anchor's crushing term is beam independent, dragging its exponent down
    to ~0.4 over the 15-35 m beam range.  Quantifying that here keeps the comparison honest instead of
    hiding it in a fitted constant.
    """
    from .material import STANDARD_BRASH
    state = state or STANDARD_BRASH
    closure = closure or Closure.from_ledger()
    model_totals, anchor_totals = [], []
    rows = []
    for beam in beams:
        hull = standard_hull(beam)
        model = brash_resistance(state, hull, speed, closure)
        anchor = reference_anchor(state, hull, speed)
        model_totals.append(model.total_N)
        anchor_totals.append(float(anchor["total_N"]))
        rows.append({
            "beam_m": float(beam),
            "model_total_kN": model.total_kN,
            "anchor_total_kN": float(anchor["total_N"]) / 1.0e3,
            "relative_error": model.total_N / float(anchor["total_N"]) - 1.0,
            "shares": model.shares,
        })
    return {
        "rows": rows,
        "model_beam_exponent": _loglog_slope(beams, model_totals),
        "anchor_beam_exponent": _loglog_slope(beams, anchor_totals),
        "max_abs_relative_error": max(abs(r["relative_error"]) for r in rows),
    }
