"""Level-ice resistance with an explicit temperature-dependent hull-ice friction.

The resistance is written as the classical three-term Lindqvist (1989)
decomposition plus a hull-ice friction term that carries the temperature
dependence of the friction coefficient:

    R(T) = R_crushing + R_breaking + R_submersion + R_friction(T)

with

    R_crushing    = f_c * sigma_c * h^2
    R_breaking    = f_b * B * h^(3/2) * sigma_f
    R_submersion  = f_s * rho_w * g * h * (B + T_draft) * sqrt(T_draft / B)
    R_friction    = mu(T) * cot(beta) * (R_crushing + R_breaking)

``beta`` is the mean inclination of the bow plating against the ice sheet: the
normal force transmitted by the hull is ``N = (R_crushing + R_breaking)/sin
beta``, so its tangential (friction) component adds ``mu * cot(beta) * N *
sin(beta) = mu * cot(beta) * (R_crushing + R_breaking)`` to the resistance.
That is the standard closed-form way of embedding a hull friction coefficient
in a force-balance ice-resistance formula (Lindqvist 1989; Riska 1997 discuss
the same term).

Published empirical formulas (Lindqvist, Riska, Jeong, Keinonen) absorb the
hull friction into coefficients fitted to full-scale trials, so they cannot be
reopened for a temperature sweep without refitting; the decomposition above
keeps the friction term explicit instead.  **The absolute level of the formula
is therefore not a validation target** -- only its friction share and its
temperature sensitivity are used here (see ``docs/hull_ice_friction_temperature.md``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from .hull_ice_friction import (
    T_MELT_C,
    _require_finite,
    _require_positive,
    friction_temperature_model,
    micro_contact_friction,
)

#: Lindqvist-type coefficients (dimensionless unless noted).
F_CRUSHING = 0.5
F_BREAKING = 0.003  # 1/sqrt(m) in the published m-kN-s form
F_SUBMERSION = 0.7


@dataclass(frozen=True)
class VesselIceGeometry:
    """Vessel and ice-sheet parameters for the level-ice resistance estimate."""

    beam_m: float = 19.6
    draft_m: float = 7.5
    bow_inclination_deg: float = 30.0
    ice_thickness_m: float = 0.77
    ice_compressive_strength_Pa: float = 1.748e6
    ice_flexural_strength_Pa: float = 0.724e6
    ice_density_kg_m3: float = 900.0
    water_density_kg_m3: float = 1025.0
    gravity_m_s2: float = 9.81

    def __post_init__(self) -> None:
        for name in (
            "beam_m",
            "draft_m",
            "ice_thickness_m",
            "ice_compressive_strength_Pa",
            "ice_flexural_strength_Pa",
            "ice_density_kg_m3",
            "water_density_kg_m3",
            "gravity_m_s2",
        ):
            _require_positive(getattr(self, name), name)
        _require_finite(self.bow_inclination_deg, "bow_inclination_deg")
        if not 0.0 < self.bow_inclination_deg < 90.0:
            raise ValueError("bow_inclination_deg must lie in (0, 90)")

    @property
    def bow_cotangent(self) -> float:
        return 1.0 / math.tan(math.radians(self.bow_inclination_deg))


def ice_resistance_components(
    temperature_C: float,
    *,
    geometry: Optional[VesselIceGeometry] = None,
    mu: Optional[float] = None,
    mu_model: Optional[Callable[[float], float]] = None,
    mu_parameters: Optional[dict] = None,
) -> dict:
    """Return the resistance components and the total at one temperature.

    The friction coefficient is either given directly (``mu``), produced by
    ``mu_model(temperature_C)``, or evaluated from the calibrated parametric
    law with ``mu_parameters = {"mu0": ..., "alpha": ..., "beta": ...}``.
    """
    geo = geometry or VesselIceGeometry()
    _require_finite(temperature_C, "temperature_C")
    if mu is not None and (mu_model is not None or mu_parameters is not None):
        raise ValueError("pass either mu or a friction model, not both")
    if mu is None:
        if mu_parameters is not None:
            mu = float(
                friction_temperature_model(temperature_C, **mu_parameters)
            )
        elif mu_model is not None:
            mu = float(mu_model(temperature_C))
        else:
            mu = micro_contact_friction(temperature_C)
    _require_finite(mu, "mu")
    if mu < 0.0:
        raise ValueError("mu must be nonnegative")

    h = geo.ice_thickness_m
    crushing = F_CRUSHING * geo.ice_compressive_strength_Pa * h**2
    breaking = (
        F_BREAKING
        * geo.beam_m
        * h**1.5
        * geo.ice_flexural_strength_Pa
    )
    submersion = (
        F_SUBMERSION
        * geo.water_density_kg_m3
        * geo.gravity_m_s2
        * h
        * (geo.beam_m + geo.draft_m)
        * math.sqrt(geo.draft_m / geo.beam_m)
    )
    friction = mu * geo.bow_cotangent * (crushing + breaking)
    total = crushing + breaking + submersion + friction
    return {
        "temperature_C": float(temperature_C),
        "friction_coefficient": float(mu),
        "bow_cotangent": geo.bow_cotangent,
        "crushing_resistance_N": crushing,
        "breaking_resistance_N": breaking,
        "submersion_resistance_N": submersion,
        "friction_resistance_N": friction,
        "total_resistance_N": total,
        "friction_share": friction / total if total > 0.0 else 0.0,
        "non_friction_resistance_N": crushing + breaking + submersion,
    }


def resistance_temperature_comparison(
    warm_temperature_C: float = -5.0,
    cold_temperature_C: float = -20.0,
    *,
    geometry: Optional[VesselIceGeometry] = None,
    mu_parameters: Optional[dict] = None,
) -> dict:
    """Compare the level-ice resistance at two temperatures.

    Returns both states, the absolute and relative increase of the resistance
    for the colder state, and the friction share at each state.
    """
    _require_finite(warm_temperature_C, "warm_temperature_C")
    _require_finite(cold_temperature_C, "cold_temperature_C")
    if not cold_temperature_C < warm_temperature_C <= T_MELT_C:
        raise ValueError(
            "require cold_temperature_C < warm_temperature_C <= 0 degC"
        )
    warm = ice_resistance_components(
        warm_temperature_C, geometry=geometry, mu_parameters=mu_parameters
    )
    cold = ice_resistance_components(
        cold_temperature_C, geometry=geometry, mu_parameters=mu_parameters
    )
    delta = cold["total_resistance_N"] - warm["total_resistance_N"]
    return {
        "warm": warm,
        "cold": cold,
        "delta_resistance_N": delta,
        "delta_relative_to_warm": delta / warm["total_resistance_N"],
        "delta_relative_to_cold": delta / cold["total_resistance_N"],
        "friction_coefficient_ratio_cold_over_warm": cold["friction_coefficient"]
        / warm["friction_coefficient"],
        "friction_share_warm": warm["friction_share"],
        "friction_share_cold": cold["friction_share"],
    }


def resistance_temperature_sweep(
    temperatures_C,
    *,
    geometry: Optional[VesselIceGeometry] = None,
    mu_parameters: Optional[dict] = None,
) -> list[dict]:
    """Resistance components at every temperature of a sweep."""
    return [
        ice_resistance_components(
            float(temperature), geometry=geometry, mu_parameters=mu_parameters
        )
        for temperature in temperatures_C
    ]