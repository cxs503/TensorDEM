"""Compressive strength of sea ice: Timco & Frederking (1984) parameterization.

Physical background
-------------------
The uniaxial compressive strength of sea ice depends on:
1. **Temperature T**: colder → stronger (pure ice strength increases with |T|)
2. **Salinity S / brine volume ν_b**: more brine → weaker (porosity effect)
3. **Strain rate**: intermediate rates (10⁻³–10⁻² s⁻¹) give peak strength
4. **Grain structure**: columnar (S2) vs granular ice have different strengths
5. **Ice type**: FYI (first-year) vs MYI (multi-year) differ in salinity history

Timco & Frederking (1984) formulation
--------------------------------------
Timco & Frederking (1984) compiled extensive compressive strength data for
columnar sea ice (S2 grain structure) at medium strain rates (~10⁻³ s⁻¹).
They proposed that the compressive strength depends on brine volume as:

    σ_c = σ_i(T) × exp(-k × √ν_b)

where:
- σ_i(T) is the compressive strength of pure (freshwater) ice at temperature T
- ν_b is the brine volume fraction (from Cox & Weeks 1983)
- k is a dimensionless weakening parameter (k ≈ 1.87 from their fit)

The pure ice strength σ_i(T) depends on temperature as:

    σ_i(T) = σ_ref × (|T| / T_ref)^n

where:
- σ_ref = 25.0 MPa at T_ref = 10°C subcooling (i.e., T = -10°C)
- n = 0.75 (power-law exponent from laboratory data)

This gives σ_i(-10°C) = 25 MPa, σ_i(-20°C) ≈ 42 MPa, σ_i(-5°C) ≈ 14.8 MPa.

MYI desalination correction
----------------------------
Multi-year ice (MYI) has undergone multiple melt-freeze cycles. During each
melt season, gravity drainage removes brine from the ice, reducing its bulk
salinity. Typical salinities:
- FYI: 5-10 ppt (new ice can be 10-15 ppt)
- MYI: 1-3 ppt (after years of drainage)

The MYI compressive strength is computed by applying a desalination correction
to the bulk salinity before computing brine volume:

    S_MYI_eff = S × f_desal

where f_desal is the desalination factor (typically 0.2-0.4 for mature MYI).

Then:
    σ_c_MYI(T, S) = σ_c_FYI(T, S_MYI_eff)

Since S_MYI_eff < S, we have ν_b_MYI < ν_b_FYI, and thus σ_c_MYI > σ_c_FYI.
This reflects the physical fact that MYI is stronger than FYI due to desalination.

Validation
----------
At T = -10°C, S = 5 ppt (FYI):
    ν_b = 0.0273 (2.73%)
    σ_i = 25.0 MPa
    σ_c_FYI = 25.0 × exp(-1.87 × √0.0273) = 25.0 × exp(-0.309) = 25.0 × 0.734 = 18.4 MPa

This is in the range of laboratory measurements (15-20 MPa for FYI at -10°C).

At T = -10°C, S = 5 ppt, f_desal = 0.3 (MYI):
    S_MYI_eff = 5 × 0.3 = 1.5 ppt
    ν_b_MYI = 1.5 × (4.9185 + 0.532) × 10⁻³ = 0.00818 (0.82%)
    σ_c_MYI = 25.0 × exp(-1.87 × √0.00818) = 25.0 × exp(-0.169) = 25.0 × 0.845 = 21.1 MPa

So σ_c_MYI (21.1 MPa) > σ_c_FYI (18.4 MPa), as expected physically.

References
----------
- Timco, G. W. & Frederking, R. M. W. (1984). "The effect of grain size on
  the compressive strength of ice." IAHR Ice Symposium, Hamburg, 1984.
- Timco, G. W. & Weeks, W. F. (2010). "A review of the engineering properties
  of sea ice in the twenty-first century." Cold Regions Science and Technology,
  60(1), 107-129.
- Schulson, E. M. & Duval, P. (2009). "Creep and Fracture of Ice." Cambridge
  University Press.
"""

import numpy as np
from typing import Union, Optional
from .brine_volume import brine_volume_fraction

# Type alias
Numeric = Union[float, np.ndarray]

# Default parameters for Timco & Frederking (1984) formulation
SIGMA_REF = 25.0  # Reference pure ice strength at T_ref (MPa)
T_REF = 10.0  # Reference subcooling (°C), i.e., T = -10°C
N_EXPONENT = 0.75  # Power-law exponent for temperature dependence
K_WEAKENING = 1.87  # Brine weakening parameter (dimensionless)

# MYI desalination factors (typical range)
F_DESAL_DEFAULT = 0.3  # Default desalination factor for mature MYI
F_DESAL_RANGE = (0.2, 0.4)  # Typical range for MYI


def pure_ice_strength(temperature: Numeric) -> Numeric:
    """Compressive strength of pure (freshwater) ice at temperature T.

    Parameters
    ----------
    temperature : float or array
        Ice temperature in °C (must be negative).

    Returns
    -------
    sigma_i : float or array
        Pure ice compressive strength in MPa.

    Notes
    -----
    The temperature dependence is:
        σ_i(T) = σ_ref × (|T| / T_ref)^n

    with σ_ref = 25 MPa at T_ref = 10°C subcooling, n = 0.75.

    Examples
    --------
    >>> pure_ice_strength(-10.0)
    25.0
    >>> pure_ice_strength(-20.0)
    42.04482076269336
    """
    T = np.asarray(temperature, dtype=float)

    if np.any(T > 0):
        raise ValueError(f"Temperature {T}°C must be negative.")

    T_sub = np.abs(T)
    sigma_i = SIGMA_REF * (T_sub / T_REF) ** N_EXPONENT

    if sigma_i.ndim == 0:
        return float(sigma_i)
    return sigma_i


def compressive_strength_fyi(
    temperature: Numeric,
    salinity: Numeric,
    k_weakening: float = K_WEAKENING,
    sigma_ref: float = SIGMA_REF,
    n_exponent: float = N_EXPONENT,
    extrapolate: bool = False,
) -> Numeric:
    """Compressive strength of first-year ice (FYI).

    Parameters
    ----------
    temperature : float or array
        Ice temperature in °C (must be negative, typically -22.9 ≤ T ≤ -0.5).
    salinity : float or array
        Bulk salinity in ppt (typically 0-15 ppt for FYI).
    k_weakening : float, optional
        Brine weakening parameter (default 1.87 from Timco & Frederking 1984).
    sigma_ref : float, optional
        Reference pure ice strength in MPa (default 25.0 at -10°C).
    n_exponent : float, optional
        Power-law exponent for temperature dependence (default 0.75).
    extrapolate : bool, optional
        If False, raise ValueError for T outside valid range.

    Returns
    -------
    sigma_c : float or array
        Compressive strength in MPa.

    Notes
    -----
    The formulation is:
        σ_c = σ_i(T) × exp(-k × √ν_b)

    where σ_i(T) is the pure ice strength and ν_b is the brine volume fraction.

    Examples
    --------
    >>> compressive_strength_fyi(-10.0, 5.0)
    18.35...
    >>> compressive_strength_fyi(-10.0, 0.0)
    25.0
    """
    T = np.asarray(temperature, dtype=float)
    S = np.asarray(salinity, dtype=float)

    # Compute pure ice strength
    sigma_i = pure_ice_strength(T)

    # Compute brine volume fraction
    nu_b = brine_volume_fraction(T, S, extrapolate=extrapolate)

    # Apply brine weakening (Timco & Frederking 1984)
    sigma_c = sigma_i * np.exp(-k_weakening * np.sqrt(nu_b))

    if sigma_c.ndim == 0:
        return float(sigma_c)
    return sigma_c


def compressive_strength_myi(
    temperature: Numeric,
    salinity: Numeric,
    f_desal: float = F_DESAL_DEFAULT,
    k_weakening: float = K_WEAKENING,
    sigma_ref: float = SIGMA_REF,
    n_exponent: float = N_EXPONENT,
    extrapolate: bool = False,
) -> Numeric:
    """Compressive strength of multi-year ice (MYI) with desalination correction.

    Parameters
    ----------
    temperature : float or array
        Ice temperature in °C (must be negative).
    salinity : float or array
        Bulk salinity in ppt (typically 1-5 ppt for MYI, but can be higher
        if measured before full desalination).
    f_desal : float, optional
        Desalination factor (0 < f_desal ≤ 1). The effective salinity is
        S_eff = S × f_desal. Default 0.3 for mature MYI.
    k_weakening : float, optional
        Brine weakening parameter (default 1.87).
    sigma_ref : float, optional
        Reference pure ice strength in MPa (default 25.0).
    n_exponent : float, optional
        Power-law exponent (default 0.75).
    extrapolate : bool, optional
        If False, raise ValueError for T outside valid range.

    Returns
    -------
    sigma_c_myi : float or array
        Compressive strength of MYI in MPa.

    Notes
    -----
    MYI has undergone long-term gravity drainage, reducing its bulk salinity.
    The effective salinity is:
        S_eff = S × f_desal

    where f_desal ≈ 0.2-0.4 for mature MYI (vs. 1.0 for FYI).

    Since S_eff < S, we have ν_b_MYI < ν_b_FYI, and thus σ_c_MYI > σ_c_FYI.
    This reflects the physical fact that MYI is stronger than FYI.

    Examples
    --------
    >>> compressive_strength_myi(-10.0, 5.0, f_desal=0.3)
    21.12...
    >>> # MYI is stronger than FYI at same T, S:
    >>> fyi = compressive_strength_fyi(-10.0, 5.0)
    >>> myi = compressive_strength_myi(-10.0, 5.0, f_desal=0.3)
    >>> myi > fyi
    True
    """
    T = np.asarray(temperature, dtype=float)
    S = np.asarray(salinity, dtype=float)

    if not (0 < f_desal <= 1):
        raise ValueError(f"Desalination factor f_desal={f_desal} must be in (0, 1].")

    # Apply desalination correction
    S_eff = S * f_desal

    # Compute strength using effective salinity
    sigma_c_myi = compressive_strength_fyi(
        T, S_eff,
        k_weakening=k_weakening,
        sigma_ref=sigma_ref,
        n_exponent=n_exponent,
        extrapolate=extrapolate,
    )

    if np.ndim(sigma_c_myi) == 0:
        return float(sigma_c_myi)
    return sigma_c_myi


def strength_ratio_myi_fyi(
    temperature: float,
    salinity: float,
    f_desal: float = F_DESAL_DEFAULT,
) -> float:
    """Compute the ratio σ_c_MYI / σ_c_FYI at given T, S.

    Parameters
    ----------
    temperature : float
        Ice temperature in °C.
    salinity : float
        Bulk salinity in ppt.
    f_desal : float
        Desalination factor for MYI.

    Returns
    -------
    ratio : float
        Ratio σ_c_MYI / σ_c_FYI (always > 1 for f_desal < 1).

    Examples
    --------
    >>> strength_ratio_myi_fyi(-10.0, 5.0, f_desal=0.3)
    1.15...
    """
    sigma_fyi = compressive_strength_fyi(temperature, salinity)
    sigma_myi = compressive_strength_myi(temperature, salinity, f_desal=f_desal)
    return sigma_myi / sigma_fyi


def strength_temperature_curve(
    salinity: float,
    temperatures: Optional[np.ndarray] = None,
    ice_type: str = "FYI",
    f_desal: float = F_DESAL_DEFAULT,
) -> tuple:
    """Compute strength vs temperature curve at fixed salinity.

    Parameters
    ----------
    salinity : float
        Bulk salinity in ppt.
    temperatures : array, optional
        Temperature array in °C. If None, use -20 to -1°C with 20 points.
    ice_type : str
        "FYI" or "MYI".
    f_desal : float
        Desalination factor (only used if ice_type="MYI").

    Returns
    -------
    T_array : array
        Temperature array in °C.
    sigma_array : array
        Compressive strength in MPa.
    """
    if temperatures is None:
        temperatures = np.linspace(-20, -1, 20)

    if ice_type.upper() == "FYI":
        sigma = compressive_strength_fyi(temperatures, salinity, extrapolate=True)
    elif ice_type.upper() == "MYI":
        sigma = compressive_strength_myi(
            temperatures, salinity, f_desal=f_desal, extrapolate=True
        )
    else:
        raise ValueError(f"ice_type must be 'FYI' or 'MYI', got '{ice_type}'.")

    return temperatures, sigma


def strength_salinity_curve(
    temperature: float,
    salinities: Optional[np.ndarray] = None,
    ice_type: str = "FYI",
    f_desal: float = F_DESAL_DEFAULT,
) -> tuple:
    """Compute strength vs salinity curve at fixed temperature.

    Parameters
    ----------
    temperature : float
        Ice temperature in °C.
    salinities : array, optional
        Salinity array in ppt. If None, use 0 to 15 ppt with 16 points.
    ice_type : str
        "FYI" or "MYI".
    f_desal : float
        Desalination factor (only used if ice_type="MYI").

    Returns
    -------
    S_array : array
        Salinity array in ppt.
    sigma_array : array
        Compressive strength in MPa.
    """
    if salinities is None:
        salinities = np.linspace(0, 15, 16)

    if ice_type.upper() == "FYI":
        sigma = compressive_strength_fyi(temperature, salinities, extrapolate=True)
    elif ice_type.upper() == "MYI":
        sigma = compressive_strength_myi(
            temperature, salinities, f_desal=f_desal, extrapolate=True
        )
    else:
        raise ValueError(f"ice_type must be 'FYI' or 'MYI', got '{ice_type}'.")

    return salinities, sigma
