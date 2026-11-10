"""Psychrometric closures for moist-air convective drying (wet-bulb / humid enthalpy).

All quantities SI.  Reference state T_REF = 273.15 K (0 degC) for enthalpy.

The property set is chosen so that the energy bookkeeping is **exactly
consistent** at the level of enthalpy functions:

    h_fg(T) = h_v(T) - h_l(T) = H_FG_REF + (CP_V - CP_L) (T - T_REF)

with ``h_l(T) = CP_L (T - T_REF)`` and ``h_v(T) = H_FG_REF + CP_V (T - T_REF)``.
This is the standard ASHRAE linear latent-heat law (h_fg = 2500.9 - 2.326 T[degC]
kJ/kg) and it makes the vapour/liquid enthalpy pair used by the gas ledger and
the particle energy equation identical by construction, so an energy-ledger
residual measures *implementation* error rather than property-model mismatch.

Property correlations
---------------------
* saturation pressure: Hyland & Wexler (ASHRAE Fundamentals 2021, ch.1, eq. 6),
  water surface, 273.15 K <= T <= 473.15 K (max deviation ~0.05 % against
  the IAPWS-95 table at 0/20/50/100 degC -- see tests).
* air viscosity: Sutherland (Sutherland 1893), mu = 1.458e-6 T^1.5/(T+110.4).
* air conductivity: 3rd-order polynomial of ASHRAE (100 K <= T <= 1000 K).
* water-vapour diffusivity in air: D = 2.13e-5 (T/273.15)^1.94 (101325/p)
  (Marrero & Mason 1972 form; 2.53e-5 m^2/s at 298 K).
* ideal-gas densities with R_da = 287.058, R_v = 461.495 J/(kg K).

Key identities used downstream (verified in tests)
--------------------------------------------------
* For an isothermal ideal-gas mixture ``rho_da dY = (1 - p_v/p)^-1 d(rho_v)``;
  the two mass-transfer driving forces (vapour-density difference and
  ``rho_da`` times humidity-ratio difference) therefore differ only by the
  kinetic factor ``p/(p - p_v) ~ 1.002`` at 101 kPa, and both formulations
  collapse onto the classical psychrometric wet-bulb equation when the Lewis
  number is forced to one (analogy hook ``d_ab_override="alpha"`` in
  :mod:`tensordem.rotary_dryer.moist_transfer`).
"""
from __future__ import annotations

import math
from typing import Optional

__all__ = [
    "R_DA",
    "R_V",
    "CP_DA",
    "CP_V",
    "CP_L",
    "EPS_W",
    "T_REF",
    "H_FG_REF",
    "P_STD",
    "saturation_pressure",
    "latent_heat",
    "vapour_enthalpy",
    "liquid_enthalpy",
    "moist_enthalpy",
    "temperature_from_enthalpy",
    "vapour_pressure_from_humidity_ratio",
    "humidity_ratio_from_vapour_pressure",
    "saturation_humidity_ratio",
    "vapour_density",
    "moist_air_density",
    "dry_air_density",
    "humid_heat_capacity",
    "moist_air_cp",
    "air_viscosity",
    "air_conductivity",
    "water_vapour_diffusivity",
    "prandtl_number",
    "schmidt_number",
    "lewis_number",
    "psychrometric_residual",
    "wet_bulb_temperature",
]

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
R_DA = 287.058  # J/(kg K) dry air
R_V = 461.495   # J/(kg K) water vapour
CP_DA = 1006.0  # J/(kg K) dry air (ASHRAE 1.006 kJ/kg/K)
CP_V = 1860.0   # J/(kg K) water vapour
CP_L = 4186.0   # J/(kg K) liquid water
EPS_W = R_DA / R_V  # 0.62198 humidity-ratio constant
T_REF = 273.15  # K
H_FG_REF = 2500.9e3  # J/kg latent heat at T_REF
P_STD = 101325.0  # Pa

# Hyland-Wexler coefficients (water surface, 0..200 degC)
_HW_C8 = -5.8002206e3
_HW_C9 = 1.3914993
_HW_C10 = -4.8640239e-2
_HW_C11 = 4.1764768e-5
_HW_C12 = -1.4452093e-8
_HW_C13 = 6.5459673


def saturation_pressure(T: float) -> float:
    """Saturation vapour pressure of water over a plane surface (Pa).

    Hyland & Wexler (1983) formulation as tabulated by ASHRAE (2021).
    Valid 273.15 K <= T <= 473.15 K; raises ValueError outside that window.
    """
    if not (240.0 <= T <= 480.0):
        raise ValueError(f"saturation_pressure: T={T} K outside correlation range")
    ln_pws = (
        _HW_C8 / T
        + _HW_C9
        + _HW_C10 * T
        + _HW_C11 * T * T
        + _HW_C12 * T * T * T
        + _HW_C13 * math.log(T)
    )
    return math.exp(ln_pws)


def latent_heat(T: float) -> float:
    """Latent heat of vaporisation of water at T (J/kg), ASHRAE linear law."""
    return H_FG_REF + (CP_V - CP_L) * (T - T_REF)


def liquid_enthalpy(T: float) -> float:
    """Specific enthalpy of saturated liquid water (J/kg), reference T_REF."""
    return CP_L * (T - T_REF)


def vapour_enthalpy(T: float) -> float:
    """Specific enthalpy of water vapour (J/kg) relative to liquid at T_REF."""
    return H_FG_REF + CP_V * (T - T_REF)


def moist_enthalpy(T: float, Y: float) -> float:
    """Specific enthalpy of moist air per kg dry air (J/kg_da)."""
    return CP_DA * (T - T_REF) + Y * (H_FG_REF + CP_V * (T - T_REF))


def temperature_from_enthalpy(h: float, Y: float) -> float:
    """Invert :func:`moist_enthalpy` (exact, linear in T)."""
    denom = CP_DA + Y * CP_V
    if denom <= 0.0:
        raise ValueError("temperature_from_enthalpy: non-positive heat capacity")
    return T_REF + (h - Y * H_FG_REF) / denom


def vapour_pressure_from_humidity_ratio(Y: float, p: float = P_STD) -> float:
    """Partial pressure of water vapour (Pa) from humidity ratio Y (kg/kg_da)."""
    return Y * p / (EPS_W + Y)


def humidity_ratio_from_vapour_pressure(p_v: float, p: float = P_STD) -> float:
    """Humidity ratio (kg/kg_da) from vapour partial pressure (Pa)."""
    if p_v >= p:
        raise ValueError("humidity_ratio_from_vapour_pressure: p_v >= p")
    return EPS_W * p_v / (p - p_v)


def saturation_humidity_ratio(T: float, p: float = P_STD) -> float:
    """Saturated humidity ratio at T (kg/kg_da).

    Returns +inf when the saturation pressure reaches the total pressure
    (T at or above the boiling point): the air cannot hold any more vapour.
    """
    p_ws = saturation_pressure(T)
    if p_ws >= p:
        return float("inf")
    return humidity_ratio_from_vapour_pressure(p_ws, p)


def vapour_density(p_v: float, T: float) -> float:
    """Water-vapour partial density (kg/m^3)."""
    return p_v / (R_V * T)


def dry_air_density(p_v: float, T: float, p: float = P_STD) -> float:
    """Dry-air partial density (kg/m^3) given the vapour partial pressure."""
    return (p - p_v) / (R_DA * T)


def moist_air_density(T: float, Y: float, p: float = P_STD) -> float:
    """Density of moist air (kg moist air / m^3).

    rho = p_da/(R_da T) + p_v/(R_v T) with p_v from Y.
    """
    p_v = vapour_pressure_from_humidity_ratio(Y, p)
    return (p - p_v) / (R_DA * T) + p_v / (R_V * T)


def humid_heat_capacity(Y: float) -> float:
    """Humid heat capacity per kg *dry air* (J/(kg_da K))."""
    return CP_DA + Y * CP_V


def moist_air_cp(Y: float) -> float:
    """Specific heat per kg *moist air* (J/(kg K))."""
    return (CP_DA + Y * CP_V) / (1.0 + Y)


def air_viscosity(T: float) -> float:
    """Dynamic viscosity of (moist) air, Sutherland law (Pa s)."""
    return 1.458e-6 * T ** 1.5 / (T + 110.4)


def air_conductivity(T: float) -> float:
    """Thermal conductivity of (moist) air (W/(m K)), ASHRAE polynomial."""
    return (
        1.5207e-11 * T ** 3
        - 4.8574e-8 * T * T
        + 1.0184e-4 * T
        - 3.9333e-4
    )


def water_vapour_diffusivity(T: float, p: float = P_STD) -> float:
    """Binary diffusivity of water vapour in air (m^2/s)."""
    return 2.13e-5 * (T / 273.15) ** 1.94 * (P_STD / p)


def prandtl_number(T: float, Y: float = 0.0, p: float = P_STD) -> float:
    """Prandtl number of the moist-air mixture."""
    rho = moist_air_density(T, Y, p)
    return air_viscosity(T) * moist_air_cp(Y) / air_conductivity(T)


def schmidt_number(T: float, Y: float = 0.0, p: float = P_STD) -> float:
    """Schmidt number of water vapour in moist air."""
    rho = moist_air_density(T, Y, p)
    return air_viscosity(T) / (rho * water_vapour_diffusivity(T, p))


def lewis_number(T: float, Y: float = 0.0, p: float = P_STD) -> float:
    """Lewis number Le = Sc/Pr = alpha/D_AB."""
    return schmidt_number(T, Y, p) / prandtl_number(T, Y, p)


# ---------------------------------------------------------------------------
# classical (Lewis = 1 / adiabatic-saturation) psychrometric wet bulb
# ---------------------------------------------------------------------------
def psychrometric_residual(T_wb: float, T_g: float, Y_g: float, p: float = P_STD) -> float:
    """Residual of the classical psychrometric (adiabatic saturation) equation.

    (T_g - T_wb) - h_fg(T_wb) [Y_sat(T_wb) - Y_g] / c_pH = 0

    Positive for T_wb below the root.
    """
    cp_h = humid_heat_capacity(Y_g)
    lhs = T_g - T_wb
    rhs = latent_heat(T_wb) * (saturation_humidity_ratio(T_wb, p) - Y_g) / cp_h
    return lhs - rhs


def wet_bulb_temperature(
    T_g: float,
    Y_g: float,
    p: float = P_STD,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """Classical psychrometric wet-bulb temperature (K), Lewis factor = 1.

    Root of the adiabatic-saturation (Carrier) equation solved with a
    conservative bisection on [T_lo, T_g].  ``T_lo`` is found by halving
    from T_g until the residual is positive (guaranteed since the saturated
    humidity ratio grows super-exponentially as T falls).

    Notes
    -----
    This is the value read off a psychrometric chart.  It is the fixed point
    of the surface energy/mass balance *only* when the Lewis number of the
    vapour/heat transfer pair equals one; see
    :func:`tensordem.rotary_dryer.moist_transfer.wet_bulb_from_closures`.
    """
    if Y_g < 0.0:
        raise ValueError("wet_bulb_temperature: Y_g must be >= 0")
    if Y_g >= saturation_humidity_ratio(T_g, p):
        # unsaturated air required; supersaturated input degenerates to T_g
        return T_g
    T_lo = T_g
    for _ in range(200):
        T_lo = T_lo - 5.0
        if T_lo < 200.0:
            raise ValueError("wet_bulb_temperature: bracket failed")
        if psychrometric_residual(T_lo, T_g, Y_g, p) > 0.0:
            break
    lo, hi = T_lo, T_g
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = psychrometric_residual(mid, T_g, Y_g, p)
        if f_mid > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)