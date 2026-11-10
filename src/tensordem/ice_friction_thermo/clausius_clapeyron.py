"""Pressure melting of ice: the Clausius-Clapeyron branch of the contact state.

Why this module exists
----------------------
Constant-friction DEM contact models evaluate ``mu`` at the *bulk* ice
temperature and treat the melting point as a constant ``0 degC``.  At a sliding
ice-structure interface neither is true:

* The melting point depends on the **local contact pressure**,

      ``dT_m/dp = T_m * (v_water - v_ice) / L_f``   (Clausius-Clapeyron),

  which for ice is ``-7.40e-8 K/Pa = -0.074 K/MPa``: water is *denser* than ice,
  so pressure *lowers* the melting point.  The depression is negligible at the
  nominal pressure of a hull contact (1 MPa -> 0.07 K) but not at the **real
  contact pressure** of the asperities, where the load is carried on a small
  area fraction ``phi = p_nominal / H``.  With an ice hardness of 20 MPa and a
  nominal 1 MPa, ``p_real = H = 20 MPa`` and ``T_m = -1.5 degC``; at 100 MPa
  (cold, hard ice) ``T_m = -7.4 degC``.
* Melting therefore starts when the contact temperature reaches ``T_m(p)``, not
  ``0 degC``, and the amount of water in the interface -- the quantity that
  destroys friction -- is a function of ``T - T_m(p)``.

The signature of this mechanism is the observed **pressure weakening** of ice
friction (Oksanen & Keinonen 1982): at fixed temperature and speed, ``mu``
decreases with contact pressure because ``T_m(p)`` drops toward the interface
temperature and the melt fraction rises.

Everything is SI internally.  Temperatures are degC (only differences and
derivatives matter); pressures are Pa.

References
----------
Petrenko, V.F. & Whitworth, R.W. (1999). Physics of Ice. Oxford Univ. Press.
    (Clapeyron slope, latent heat, densities.)
Barnes, P., Tabor, D. & Walker, J.C.F. (1971). The friction and creep of
    polycrystalline ice. Proc. R. Soc. Lond. A 324, 127-155.
Oksanen, P. & Keinonen, J. (1982). The mechanism of friction of ice. Wear 78,
    315-324.
"""
from __future__ import annotations

import math

from .properties import (
    ICE_DENSITY_KG_M3,
    LATENT_HEAT_J_KG,
    MELT_TEMPERATURE_C,
    MELT_TEMPERATURE_K,
    WATER_DENSITY_KG_M3,
)

__all__ = [
    "cc_slope_k_per_pa",
    "clapeyron_slope_check",
    "melting_temperature_c",
    "pressure_melting_threshold_pa",
    "melt_fraction",
    "real_contact_pressure",
    "stefan_melt_rate_m_per_s",
    "film_latent_energy_j_m2",
    "CC_SLOPE_K_PER_PA",
    "CC_SLOPE_K_PER_MPA",
]


def cc_slope_k_per_pa(
    melt_temperature_k: float = MELT_TEMPERATURE_K,
    rho_water_kg_m3: float = WATER_DENSITY_KG_M3,
    rho_ice_kg_m3: float = ICE_DENSITY_KG_M3,
    latent_heat_j_kg: float = LATENT_HEAT_J_KG,
) -> float:
    """``dT_m/dp`` of the ice-water transition [K/Pa] (negative for ice).

    Derived from the Clapeyron relation ``dT/dp = T * dv / L`` with
    ``dv = 1/rho_water - 1/rho_ice = -9.05e-5 m^3/kg``.  No fitted parameter:
    the value follows from the three material properties.
    """
    for name, value in (("melt_temperature_k", melt_temperature_k),
                        ("rho_water_kg_m3", rho_water_kg_m3),
                        ("rho_ice_kg_m3", rho_ice_kg_m3),
                        ("latent_heat_j_kg", latent_heat_j_kg)):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    dv = 1.0 / rho_water_kg_m3 - 1.0 / rho_ice_kg_m3
    return melt_temperature_k * dv / latent_heat_j_kg


#: Derived at import from the material properties; -7.402e-8 K/Pa.
CC_SLOPE_K_PER_PA: float = cc_slope_k_per_pa()
#: The same number in the engineering unit used in the report; 0.074 K/MPa.
CC_SLOPE_K_PER_MPA: float = -CC_SLOPE_K_PER_PA * 1.0e6


def clapeyron_slope_check() -> float:
    """Relative deviation of the derived slope from the tabulated 0.0072 K/atm.

    The textbook pressure-melting coefficient is ``0.0072 K per atmosphere``
    (Petrenko & Whitworth 1999, ch. 4).  Returning the relative deviation makes
    the property-derived slope a *checked* number instead of an assumed one:
    the value obtained here is 0.0075 K/atm, i.e. 4 % above the tabulated one,
    the difference being the temperature dependence of ``dv`` and ``L`` which a
    constant-property Clapeyron relation ignores.
    """
    tabulated_k_per_atm = 0.0072
    derived = -CC_SLOPE_K_PER_PA * 101325.0
    return abs(derived - tabulated_k_per_atm) / tabulated_k_per_atm


def melting_temperature_c(pressure_pa: float,
                          slope_k_per_pa: float = CC_SLOPE_K_PER_PA,
                          melt_temperature_c0: float = MELT_TEMPERATURE_C) -> float:
    """Melting point [degC] of ice under ``pressure_pa`` (linear Clapeyron).

    ``T_m(p) = T_m(0) + (dT_m/dp) p``.  The linearisation is exact for constant
    specific volumes; over 0..200 MPa the non-linearity of ``dv(T)`` and
    ``L(T)`` changes the result by a few percent, which is far below the
    experimental scatter of the friction data used downstream.
    """
    p = float(pressure_pa)
    if not math.isfinite(p):
        raise ValueError("pressure_pa must be finite")
    if p < 0.0:
        raise ValueError("pressure_pa must be non-negative")
    return melt_temperature_c0 + slope_k_per_pa * p


def pressure_melting_threshold_pa(temperature_c: float,
                                  slope_k_per_pa: float = CC_SLOPE_K_PER_PA,
                                  melt_temperature_c0: float = MELT_TEMPERATURE_C) -> float:
    """Contact pressure [Pa] that depresses the melting point to ``temperature_c``.

    This is the pressure-melting onset: ice at ``-5 degC`` is melted by pressure
    alone once the real contact pressure exceeds ~68 MPa.  A negative (or zero)
    temperature is required -- ice above its melting point is already liquid.
    """
    t = float(temperature_c)
    if not math.isfinite(t):
        raise ValueError("temperature_c must be finite")
    if t >= melt_temperature_c0:
        raise ValueError("pressure melting is only defined below the melting point")
    return (t - melt_temperature_c0) / slope_k_per_pa


def melt_fraction(temperature_c: float, pressure_pa: float,
                  onset_width_k: float = 1.0,
                  slope_k_per_pa: float = CC_SLOPE_K_PER_PA,
                  melt_temperature_c0: float = MELT_TEMPERATURE_C) -> float:
    """Water fraction at the interface, ``0`` (dry) to ``1`` (fully melted).

    ``s = clip((T - T_m(p) + w) / w, 0, 1)``: melting begins ``w`` kelvin below
    the pressure-dependent melting point and is complete *at* it.  Two
    consequences are the physics this module exists for:

    * at fixed ``T``, raising ``p`` lowers ``T_m(p)`` and therefore raises ``s``
      (pressure weakening);
    * at fixed ``p``, raising ``T`` raises ``s`` (frictional heating weakening).

    ``onset_width_k`` is the one phenomenological length in the model (the
    width of the melting transition); it is scanned in the sensitivity table
    rather than hidden.
    """
    width = float(onset_width_k)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("onset_width_k must be finite and positive")
    t_melt = melting_temperature_c(pressure_pa, slope_k_per_pa, melt_temperature_c0)
    s = (float(temperature_c) - t_melt + width) / width
    return min(1.0, max(0.0, s))


def real_contact_pressure(nominal_pressure_pa: float, hardness_pa: float,
                          max_fraction: float = 1.0) -> tuple[float, float]:
    """Real (asperity) contact pressure and area fraction.

    Returns ``(p_real, phi)``.  A plastic/idealised-hardness argument: the load
    is carried by an area fraction ``phi = p_nominal / H`` at the hardness
    pressure ``H`` (Bowden & Tabor).  Until ``phi`` reaches 1 the real pressure
    saturates at ``H`` -- which is why the pressure dependence of friction
    enters through the *real* pressure, not the nominal one.
    """
    p_n = float(nominal_pressure_pa)
    hardness = float(hardness_pa)
    if not math.isfinite(p_n) or p_n < 0.0:
        raise ValueError("nominal_pressure_pa must be finite and non-negative")
    if not math.isfinite(hardness) or hardness <= 0.0:
        raise ValueError("hardness_pa must be finite and positive")
    fraction = min(max_fraction, max(p_n / hardness, 0.0))
    if fraction <= 0.0:
        return 0.0, 0.0
    return p_n / fraction, fraction


def stefan_melt_rate_m_per_s(excess_flux_w_m2: float,
                            rho_ice_kg_m3: float = ICE_DENSITY_KG_M3,
                            latent_heat_j_kg: float = LATENT_HEAT_J_KG) -> float:
    """Film growth rate ``dh/dt = q_excess / (rho L)`` [m/s].

    Only the part of the frictional flux that cannot be conducted away (because
    the interface is pinned at ``T_m``) goes into latent heat.
    """
    flux = float(excess_flux_w_m2)
    if not math.isfinite(flux):
        raise ValueError("excess_flux_w_m2 must be finite")
    if flux <= 0.0:
        return 0.0
    return flux / (rho_ice_kg_m3 * latent_heat_j_kg)


def film_latent_energy_j_m2(film_thickness_m: float,
                            rho_ice_kg_m3: float = ICE_DENSITY_KG_M3,
                            latent_heat_j_kg: float = LATENT_HEAT_J_KG) -> float:
    """Latent energy stored in a melt film of thickness ``film_thickness_m``."""
    h = float(film_thickness_m)
    if not math.isfinite(h) or h < 0.0:
        raise ValueError("film_thickness_m must be finite and non-negative")
    return h * rho_ice_kg_m3 * latent_heat_j_kg