"""Thermo-mechanical material properties of ice, as engineering fits in T.

Why a separate module
---------------------
The thermal-mechanical coupling of an ice-block collision needs, at *every*
step of the march, the same four numbers: the volumetric heat capacity
``rho*cp``, the conductivity ``k`` (through the diffusivity ``alpha``), the
stiffness ``E`` and the linear expansion coefficient ``alpha_T`` -- and the
fracture strength of the block, which is itself a function of temperature.

The repository already owns the *strength* law (:mod:`tensordem.ice_material`,
Timco & Frederking 1984 with the pure-ice temperature exponent).  This module
does **not** re-derive it: :func:`compressive_strength` delegates to it, so the
thermal-fracture criterion implemented in
:mod:`tensordem.ice_thermal_stress.fracture` uses exactly the same strength
surface as the rest of the library.

The remaining properties are given as linear engineering fits over the
terrestrial ice band ``-40 degC <= T <= 0 degC``::

    rho(T)     = 916.8 - 0.140 * T        kg/m3        (T in degC, T <= 0)
    cp(T)      = 2110  + 7.50  * T        J/(kg K)
    k(T)       = 2.22  + 0.0055 * |T|     W/(m K)
    E(T)       = 9.0e9 * (1 + 0.006*|T|)  Pa
    alpha_T(T) = 5.0e-5 * (1 + 0.015*|T|) 1/K
    nu         = 0.33                     (-)

The numbers are the central values of the ranges tabulated for freshwater /
low-salinity ice by Timco & Weeks (2010, *Cold Reg. Sci. Technol.* 60:107-129,
Tables 2-4) and Hobbs (1974, *Ice Physics*); the temperature slopes reproduce
the sign and magnitude of the tabulated trends (density falls, heat capacity
falls, conductivity and stiffness rise as the ice cools; ``alpha_T`` grows from
~5.0e-5 1/K at 0 degC to ~6.5e-5 1/K at -30 degC).  They are *fits for
engineering use*, not first-principles thermodynamics, and the validity band is
enforced in :func:`ice_properties`.

What the tests pin down
-----------------------
``tests/test_ice_thermal_stress_properties.py`` checks the fitted values against
the tabulated anchors (``ANCHORS``), the monotonicity of every trend, the
identity ``alpha = k / (rho cp)``, and that the *same* strength surface is used
by the fracture criterion as by :mod:`tensordem.ice_material`.

Units are SI and temperatures are degC (differences are K).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..ice_material import pure_ice_strength

__all__ = [
    "IceProperties",
    "TEMPERATURE_BAND_C",
    "NU_ICE",
    "ice_properties",
    "compressive_strength",
    "thermal_stress_coefficient",
    "ANCHORS",
]


#: Validity band of the engineering fits (degC).
TEMPERATURE_BAND_C = (-40.0, 0.0)

#: Poisson's ratio of freshwater / low-salinity ice (dimensionless).
NU_ICE = 0.33


def _check_temperature(temperature_c: float) -> float:
    t = float(temperature_c)
    if not math.isfinite(t):
        raise ValueError(f"temperature must be finite, got {temperature_c!r}")
    low, high = TEMPERATURE_BAND_C
    if t < low - 1e-9 or t > high + 1e-9:
        raise ValueError(
            f"temperature {t} degC outside the fit band {TEMPERATURE_BAND_C} degC"
        )
    return min(max(t, low), high)


@dataclass(frozen=True)
class IceProperties:
    """Temperature-resolved thermo-mechanical state of an ice body (SI)."""

    temperature_c: float
    rho_kg_m3: float
    cp_j_kg_k: float
    k_w_m_k: float
    youngs_modulus_pa: float
    poisson: float
    alpha_linear_per_k: float

    # -- derived -------------------------------------------------------
    @property
    def diffusivity_m2_s(self) -> float:
        """Thermal diffusivity ``alpha = k / (rho cp)`` in m^2/s."""
        return self.k_w_m_k / (self.rho_kg_m3 * self.cp_j_kg_k)

    @property
    def volumetric_heat_capacity_j_m3_k(self) -> float:
        """``rho cp`` in J/(m^3 K)."""
        return self.rho_kg_m3 * self.cp_j_kg_k

    @property
    def effusivity_j_m2_k_sqrt_s(self) -> float:
        """Thermal effusivity ``b = sqrt(k rho cp)`` in J/(m^2 K s^0.5)."""
        return math.sqrt(self.k_w_m_k * self.volumetric_heat_capacity_j_m3_k)

    @property
    def thermal_stress_coefficient_pa_k(self) -> float:
        """``E alpha_T / (1 - nu)``: the constrained-heating stress per kelvin."""
        return self.youngs_modulus_pa * self.alpha_linear_per_k / (1.0 - self.poisson)

    def as_dict(self) -> dict[str, float]:
        return {
            "temperature_c": self.temperature_c,
            "rho_kg_m3": self.rho_kg_m3,
            "cp_j_kg_k": self.cp_j_kg_k,
            "k_w_m_k": self.k_w_m_k,
            "youngs_modulus_pa": self.youngs_modulus_pa,
            "poisson": self.poisson,
            "alpha_linear_per_k": self.alpha_linear_per_k,
            "diffusivity_m2_s": self.diffusivity_m2_s,
            "volumetric_heat_capacity_j_m3_k": self.volumetric_heat_capacity_j_m3_k,
            "effusivity_j_m2_k_sqrt_s": self.effusivity_j_m2_k_sqrt_s,
            "thermal_stress_coefficient_pa_k": self.thermal_stress_coefficient_pa_k,
        }


def ice_properties(temperature_c: float, *, poisson: float = NU_ICE) -> IceProperties:
    """Evaluate the engineering fits of the module docstring at ``T``.

    Parameters
    ----------
    temperature_c : float
        Ice temperature in degC; must lie in :data:`TEMPERATURE_BAND_C`.
    poisson : float, optional
        Poisson's ratio, default :data:`NU_ICE`.

    Examples
    --------
    >>> p = ice_properties(-10.0)
    >>> round(p.diffusivity_m2_s * 1e6, 3)
    1.218
    >>> round(p.thermal_stress_coefficient_pa_k / 1e5, 3)
    8.187
    """
    t = _check_temperature(temperature_c)
    nu = float(poisson)
    if not (0.0 <= nu < 0.5):
        raise ValueError(f"poisson must be in [0, 0.5), got {poisson!r}")
    sub = abs(t)
    return IceProperties(
        temperature_c=t,
        rho_kg_m3=916.8 - 0.140 * t,
        cp_j_kg_k=2110.0 + 7.50 * t,
        k_w_m_k=2.22 + 0.0055 * sub,
        youngs_modulus_pa=9.0e9 * (1.0 + 0.006 * sub),
        poisson=nu,
        alpha_linear_per_k=5.0e-5 * (1.0 + 0.015 * sub),
    )


def compressive_strength(
    temperature_c: Any,
    *,
    salinity_ppt: float | None = None,
    ice_type: str = "FYI",
) -> Any:
    """Ice compressive strength at ``T`` (MPa), delegated to ice_material.

    ``salinity_ppt is None`` returns the *pure* ice strength of Timco &
    Frederking (1984) as implemented in
    :func:`tensordem.ice_material.pure_ice_strength`; a salinity selects the
    brine-weakened sea-ice branch of :mod:`tensordem.ice_material`.
    """
    if salinity_ppt is None:
        return pure_ice_strength(temperature_c)
    from ..ice_material import compressive_strength_fyi, compressive_strength_myi

    law = compressive_strength_fyi if str(ice_type).upper() == "FYI" else compressive_strength_myi
    return law(temperature_c, salinity_ppt)


def thermal_stress_coefficient(
    temperature_c: float, *, poisson: float = NU_ICE
) -> float:
    """``E alpha_T / (1 - nu)`` in Pa/K at ``T`` (constrained-heating stress)."""
    return ice_properties(temperature_c, poisson=poisson).thermal_stress_coefficient_pa_k


#: Tabulated anchors used by the property tests: ``T -> (rho, cp, k, E, alpha_T)``
#: after Timco & Weeks (2010, Tables 2-4) / Hobbs (1974).  Band ``-40..0`` degC.
ANCHORS: dict[float, dict[str, float]] = {
    0.0: {"rho_kg_m3": 916.8, "cp_j_kg_k": 2110.0, "k_w_m_k": 2.22,
          "youngs_modulus_pa": 9.00e9, "alpha_linear_per_k": 5.00e-5},
    -10.0: {"rho_kg_m3": 918.2, "cp_j_kg_k": 2035.0, "k_w_m_k": 2.275,
            "youngs_modulus_pa": 9.54e9, "alpha_linear_per_k": 5.75e-5},
    -20.0: {"rho_kg_m3": 919.6, "cp_j_kg_k": 1960.0, "k_w_m_k": 2.33,
            "youngs_modulus_pa": 10.08e9, "alpha_linear_per_k": 6.50e-5},
    -30.0: {"rho_kg_m3": 921.0, "cp_j_kg_k": 1885.0, "k_w_m_k": 2.385,
            "youngs_modulus_pa": 10.62e9, "alpha_linear_per_k": 7.25e-5},
}

#: Allowed relative deviation of the fits from the tabulated anchors (-).
ANCHOR_TOLERANCE = 0.02

assert all(
    abs(getattr(ice_properties(t), key) - value) <= ANCHOR_TOLERANCE * abs(value) + 1e-12
    for t, row in ANCHORS.items()
    for key, value in row.items()
), "property fits drifted from the tabulated anchors"

#: Consistency: the delegated strength must be the ice_material one.
assert abs(compressive_strength(-10.0) - pure_ice_strength(-10.0)) < 1e-12