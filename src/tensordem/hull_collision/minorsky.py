"""Minorsky (1959) empirical collision-damage anchor and its cross-check.

Minorsky's classical relation between the kinetic energy lost in a ship
collision and the volume of damaged structure is

    E [MJ] = 47.1 * R_T [m^3] + 0.5

fitted to the collision damage of 1950s vessels (his data set spans roughly
10-400 MJ; below ~10 MJ the relation is an extrapolation).  ``R_T`` is the
*volume of the plastically deformed steel*, plating plus framing, which is the
same quantity this package computes from its own mechanism
(:func:`tensordem.hull_collision.resistance.damage_geometry`).

The mission specifies the yield-stress scaled form

    V_damage = C * M_striking * V^2 / sigma_y

Combining the two without any fitted parameter gives

    V_damage = (0.5*M*V^2) / (47.1e6) * (sigma_y_ref / sigma_y),  i.e.
    C = sigma_y_ref / (2 * 47.1e6)

with ``sigma_y_ref`` the yield stress of Minorsky's reference material.  Taking
``sigma_y_ref = 235 MPa`` (mild steel of his era) makes the scaled form reduce
*exactly* to Minorsky's line at 235 MPa and gives the documented ``1/sigma_y``
scaling for higher-strength hull steels.  ``C`` is therefore an inherited
constant, not a calibration of this package's model - the comparison below is a
genuine independent cross-check, not a fit.

Known limitation: Minorsky's ``47.1 MJ/m^3`` embeds the fracture/tearing of the
plated region of his samples; a rigid-plastic model without fracture energy can
be expected to agree only in the severe-collision range where the whole damaged
region is fully plastic.  The comparison therefore reports, per case, whether
the collision lies inside Minorsky's calibration range.
"""
from __future__ import annotations

import math

__all__ = [
    "MINORSKY_1959_MJ_PER_M3",
    "MINORSKY_1959_INTERCEPT_MJ",
    "REFERENCE_YIELD_STRESS_PA",
    "MINORSKY_VALID_ENERGY_RANGE_J",
    "minorsky_energy_J",
    "minorsky_damage_volume",
    "minorsky_coefficient",
    "compare_damage_volume",
    "kinetic_energy",
]

MINORSKY_1959_MJ_PER_M3 = 47.1
MINORSKY_1959_INTERCEPT_MJ = 0.5
REFERENCE_YIELD_STRESS_PA = 235.0e6
MINORSKY_VALID_ENERGY_RANGE_J = (10.0e6, 400.0e6)


def _real(name: str, value: float, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def kinetic_energy(mass: float, velocity: float) -> float:
    """``0.5*m*V^2`` [J]."""
    m = _real("mass", mass, positive=True)
    v = _real("velocity", velocity, nonnegative=True)
    return 0.5 * m * v * v


def minorsky_energy_J(damaged_volume: float) -> float:
    """Minorsky 1959 energy for a damaged steel volume [J]."""
    volume = _real("damaged_volume", damaged_volume, nonnegative=True)
    return (MINORSKY_1959_MJ_PER_M3 * volume + MINORSKY_1959_INTERCEPT_MJ) * 1.0e6


def minorsky_damage_volume(energy_J: float) -> float:
    """Inverse of :func:`minorsky_energy_J` (clamped at zero) [m^3]."""
    e = _real("energy_J", energy_J, nonnegative=True)
    return max((e / 1.0e6 - MINORSKY_1959_INTERCEPT_MJ) / MINORSKY_1959_MJ_PER_M3, 0.0)


def minorsky_coefficient() -> float:
    """Dimensionless ``C`` of ``V_damage = C*M*V^2/sigma_y`` (derived, not fitted).

    ``C = sigma_y_ref / (2*47.1e6)`` so that at ``sigma_y = 235 MPa`` the expression
    reproduces Minorsky's line exactly.  The ``1/sigma_y`` scaling for other
    steels is the documented extension of the 1959 relation.
    """
    return REFERENCE_YIELD_STRESS_PA / (2.0 * MINORSKY_1959_MJ_PER_M3 * 1.0e6)


def compare_damage_volume(
    *,
    model_volume_m3: float,
    mass: float,
    velocity: float,
    sigma_y: float,
) -> dict:
    """Compare a model damaged volume with the yield-scaled Minorsky estimate.

    Returns the energy, both volumes, the signed relative error
    ``(V_model - V_Minorsky)/V_Minorsky`` and whether the case lies inside
    Minorsky's calibration energy range.
    """
    v_model = _real("model_volume_m3", model_volume_m3, nonnegative=True)
    m = _real("mass", mass, positive=True)
    v = _real("velocity", velocity, nonnegative=True)
    sy = _real("sigma_y", sigma_y, positive=True)

    energy = kinetic_energy(m, v)
    c = minorsky_coefficient()
    v_minorsky = c * m * v * v / sy
    # The task form has no intercept, so it must reproduce Minorsky's *zero-intercept*
    # line exactly; the 1959 fitted intercept (+0.5 MJ) is reported separately and is
    # the only difference between the two routes.
    v_minorsky_zero_intercept = energy / (MINORSKY_1959_MJ_PER_M3 * 1.0e6) * (REFERENCE_YIELD_STRESS_PA / sy)
    v_minorsky_with_intercept = minorsky_damage_volume(energy) * (REFERENCE_YIELD_STRESS_PA / sy)
    rel_error = (v_model - v_minorsky) / v_minorsky if v_minorsky else math.inf
    lo, hi = MINORSKY_VALID_ENERGY_RANGE_J
    return {
        "mass_kg": m,
        "velocity_m_s": v,
        "kinetic_energy_J": energy,
        "sigma_y_Pa": sy,
        "minorsky_coefficient_C": c,
        "volume_model_m3": v_model,
        "volume_minorsky_m3": v_minorsky,
        "volume_minorsky_zero_intercept_m3": v_minorsky_zero_intercept,
        "volume_minorsky_with_intercept_m3": v_minorsky_with_intercept,
        "consistent_formulations": abs(v_minorsky - v_minorsky_zero_intercept) <= 1e-12 * max(v_minorsky, 1e-30),
        "relative_error": rel_error,
        "in_minorsky_range": bool(lo <= energy <= hi),
    }
