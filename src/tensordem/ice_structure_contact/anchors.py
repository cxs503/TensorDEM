"""Analytical anchors for the ice-structure contact verification surface.

``lindqvist_slice_anchor``
    The in-repository Lindqvist three-component anchor
    (:func:`tensordem.level_ice.formula.lindqvist_components`) reduced to the
    out-of-plane width of a longitudinal slice.

    The anchor itself documents that its crushing term is a *total* stem force
    whose implicit contact width is the ice thickness ``h`` (that is why
    ``h**2`` appears and the ship beam ``B`` does not), while bending,
    submersion and added drag scale with the waterline breadth ``B``.  A slice
    of out-of-plane width ``t`` therefore carries

        ``R_slice = R_c * (t / h) + (R_b + R_s + R_v) * (t / B)``

    which is the only width-consistent way of comparing a 2-D slice with the
    3-D anchor.  For the verification case ``t = h`` (the slice thickness is
    the ice thickness, i.e. the anchor's own stem-contact convention), so the
    crushing term passes through unchanged.

``fit_power_law``
    Least-squares ``R = a h^n`` fit (log-log) used for the thickness
    sensitivity criterion (theory ``n = 1.5 ... 2.0``).

All quantities SI.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

from ..level_ice.formula import HullForm, IceState, lindqvist_components

__all__ = [
    "lindqvist_slice_anchor",
    "fit_power_law",
    "slice_anchor_from_case",
]


def lindqvist_slice_anchor(
    ice: IceState,
    hull: HullForm,
    speed_m_s: float,
    *,
    out_of_plane_width_m: float,
) -> dict[str, Any]:
    """Lindqvist components reduced to a slice of width ``t`` (see module doc)."""
    comp = lindqvist_components(ice, hull, speed_m_s)
    t = out_of_plane_width_m
    h = ice.thickness_m
    b = hull.beam_m
    r_c = comp["crushing_N"] * (t / h)
    r_b = comp["bending_N"] * (t / b)
    r_s = comp["submersion_N"] * (t / b)
    r_v = comp["added_drag_N"] * (t / b)
    total = r_c + r_b + r_s + r_v
    return {
        "model": "lindqvist-3component-slice",
        "out_of_plane_width_m": t,
        "ice_thickness_m": h,
        "hull_beam_m": b,
        "speed_m_s": speed_m_s,
        "crushing_N": r_c,
        "bending_N": r_b,
        "submersion_N": r_s,
        "added_drag_N": r_v,
        "total_N": total,
        "full_3d_total_N": comp["total_N"],
        "crushing_fraction": r_c / total if total else 0.0,
        "submersion_fraction": r_s / total if total else 0.0,
        "bending_fraction": r_b / total if total else 0.0,
        "added_drag_fraction": r_v / total if total else 0.0,
        "anchor_ledger_sha256": comp["ledger_sha256"],
        "friction_coefficient": hull.friction_coefficient,
        "entry_angle_deg": hull.entry_angle_deg,
    }


def slice_anchor_from_case(case: dict[str, Any]) -> dict[str, Any]:
    """Build the slice anchor for a case dict produced by the campaign runner."""
    ice = IceState(
        thickness_m=case["ice_thickness_m"],
        compressive_strength_pa=case["compressive_strength_pa"],
        flexural_strength_pa=case["flexural_strength_pa"],
        elastic_modulus_pa=case["young_modulus_pa"],
        density_kg_m3=case["ice_density_kg_m3"],
    )
    hull = HullForm(
        length_m=case["hull_length_m"],
        beam_m=case["hull_beam_m"],
        draught_m=case.get("draught_m", 7.0),
        entry_angle_deg=case["rake_angle_deg"],
        stem_angle_deg=case.get("stem_angle_deg", 30.0),
        friction_coefficient=case["friction_coefficient"],
    )
    return lindqvist_slice_anchor(
        ice, hull, case["speed_m_s"], out_of_plane_width_m=case["slice_width_m"]
    )


def fit_power_law(
    thicknesses: Sequence[float], resistances: Sequence[float]
) -> dict[str, float]:
    """Least-squares fit ``R = a h^n``; returns ``n``, ``a`` and ``r2``."""
    h = np.asarray(thicknesses, dtype=float)
    r = np.asarray(resistances, dtype=float)
    if h.size < 2 or r.size != h.size:
        raise ValueError("need >= 2 matched thickness/resistance samples")
    if np.any(h <= 0.0) or np.any(r <= 0.0):
        raise ValueError("power-law fit needs strictly positive samples")
    x = np.log(h)
    y = np.log(r)
    n, log_a = np.polyfit(x, y, 1)
    pred = n * x + log_a
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {"exponent_n": float(n), "coefficient_a": float(math.exp(log_a)), "r2": r2}


def relative_deviation(model: float, reference: float) -> float:
    """``|model - reference| / reference`` (guarded against zero)."""
    if reference == 0.0:
        return float("inf")
    return abs(model - reference) / abs(reference)