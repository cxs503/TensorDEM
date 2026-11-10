"""Brine volume fraction of sea ice: Cox & Weeks (1983) parameterization.

Physical background
-------------------
Sea ice is a composite of pure ice crystals, brine inclusions, air bubbles,
and solid salt precipitates. The **brine volume fraction** ν_b is the volume
of liquid brine per unit total volume of ice. It is the key microstructural
parameter controlling ice strength: more brine → weaker ice.

The brine volume depends on:
1. **Temperature T** (°C, negative): colder → less brine (more freezing)
2. **Bulk salinity S** (ppt, parts per thousand by mass): more salt → more brine

The equilibrium thermodynamics are governed by the NaCl-H₂O phase diagram.
At temperature T, the brine salinity S_brine is fixed by the liquidus curve.
The brine volume fraction is then:

    ν_b = (S / S_brine) × (ρ_ice / ρ_brine)

where ρ_ice ≈ 917 kg/m³ and ρ_brine depends on S_brine.

Cox & Weeks (1983) formulation
-------------------------------
Cox & Weeks (1983), building on Frankenstein & Garner (1967), provide empirical
equations for the brine volume as a function of T and S. For the temperature
range -22.9°C ≤ T ≤ -0.5°C (where NaCl·2H₂O is the solid phase in equilibrium
with brine):

    ν_b = S × (49.185 / |T| + 0.532) × 10⁻³

where:
- S is bulk salinity in ppt (kg salt / 1000 kg ice-seawater)
- T is temperature in °C (negative, so |T| = -T)
- ν_b is dimensionless (fraction, 0 to 1)

The coefficients 49.185 and 0.532 come from fitting the NaCl-H₂O phase diagram
(Frankenstein & Garner 1967, Eqs. 3-4).

For T < -22.9°C, solid NaCl·2H₂O precipitates and the brine volume decreases
more rapidly. For T > -0.5°C, the ice is near melting and the formulation
breaks down.

Validation
----------
At T = -10°C, S = 5 ppt:
    ν_b = 5 × (49.185/10 + 0.532) × 10⁻³
        = 5 × (4.9185 + 0.532) × 10⁻³
        = 5 × 5.4505 × 10⁻³
        = 0.02725 ≈ 2.73%

This matches literature values (Timco & Weeks 2010, Table 2): ~2.5-3.0% for
low-salinity ice at -10°C.

References
----------
- Cox, G. F. N. & Weeks, W. F. (1983). "Equations for determining the gas
  content and density of sea ice." CRREL Report 83-9, US Army Corps of Engineers.
- Frankenstein, G. & Garner, R. (1967). "Equations for determining the brine
  volume of sea ice from -0.5 to -22.9°C." J. Glaciol. 7(50), 273-276.
- Timco, G. W. & Weeks, W. F. (2010). "A review of the engineering properties
  of sea ice in the twenty-first century." Cold Regions Science and Technology,
  60(1), 107-129.
"""

import numpy as np
from typing import Union

# Type alias for scalar or array inputs
Numeric = Union[float, np.ndarray]

# Physical constants
RHO_ICE = 917.0  # Density of pure ice (kg/m³)
T_MIN_VALID = -22.9  # Minimum valid temperature (°C)
T_MAX_VALID = -0.5  # Maximum valid temperature (°C)


def brine_volume_fraction(
    temperature: Numeric,
    salinity: Numeric,
    extrapolate: bool = False,
) -> Numeric:
    """Compute brine volume fraction using Cox & Weeks (1983).

    Parameters
    ----------
    temperature : float or array
        Ice temperature in °C (must be negative, typically -22.9 ≤ T ≤ -0.5).
    salinity : float or array
        Bulk salinity in ppt (parts per thousand by mass, typically 0-15 ppt).
    extrapolate : bool, optional
        If False (default), raise ValueError for T outside [-22.9, -0.5]°C.
        If True, extrapolate (use with caution).

    Returns
    -------
    nu_b : float or array
        Brine volume fraction (dimensionless, 0 to 1).

    Notes
    -----
    The formula is:
        ν_b = S × (49.185 / |T| + 0.532) × 10⁻³

    For S = 0 (freshwater ice), ν_b = 0 (no brine).
    For T → 0⁻, ν_b → ∞ (unphysical; formula breaks down near melting).

    Examples
    --------
    >>> brine_volume_fraction(-10.0, 5.0)
    0.0272525
    >>> brine_volume_fraction(-10.0, 0.0)
    0.0
    """
    T = np.asarray(temperature, dtype=float)
    S = np.asarray(salinity, dtype=float)

    # Validate inputs
    if not extrapolate:
        if np.any(T > T_MAX_VALID):
            raise ValueError(
                f"Temperature {T}°C exceeds valid range (T ≤ {T_MAX_VALID}°C). "
                "Set extrapolate=True to override."
            )
        if np.any(T < T_MIN_VALID):
            raise ValueError(
                f"Temperature {T}°C below valid range (T ≥ {T_MIN_VALID}°C). "
                "Set extrapolate=True to override."
            )

    if np.any(S < 0):
        raise ValueError(f"Salinity {S} ppt must be non-negative.")

    # Handle T = 0 (avoid division by zero)
    T_abs = np.abs(T)
    T_safe = np.where(T_abs < 1e-10, 1e-10, T_abs)

    # Cox & Weeks (1983) / Frankenstein & Garner (1967) formula
    nu_b = S * (49.185 / T_safe + 0.532) * 1e-3

    # Ensure non-negative (S=0 → nu_b=0)
    nu_b = np.maximum(nu_b, 0.0)

    # Return scalar if inputs were scalar
    if nu_b.ndim == 0:
        return float(nu_b)
    return nu_b


def cox_weeks_nu_b(
    temperature: Numeric,
    salinity: Numeric,
    **kwargs,
) -> Numeric:
    """Alias for :func:`brine_volume_fraction` (Cox & Weeks 1983).

    This function is provided for convenience and backward compatibility.
    See :func:`brine_volume_fraction` for full documentation.
    """
    return brine_volume_fraction(temperature, salinity, **kwargs)


def brine_volume_fraction_with_uncertainty(
    temperature: float,
    salinity: float,
    T_std: float = 0.5,
    S_std: float = 0.5,
    n_samples: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute brine volume with uncertainty propagation.

    Parameters
    ----------
    temperature : float
        Mean temperature in °C.
    salinity : float
        Mean salinity in ppt.
    T_std : float
        Standard deviation of temperature uncertainty (°C).
    S_std : float
        Standard deviation of salinity uncertainty (ppt).
    n_samples : int
        Number of Monte Carlo samples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'mean': mean brine volume fraction
        - 'std': standard deviation
        - 'p05': 5th percentile
        - 'p50': 50th percentile (median)
        - 'p95': 95th percentile
        - 'samples': array of all samples
    """
    rng = np.random.default_rng(seed)

    # Sample T and S from normal distributions
    T_samples = rng.normal(temperature, T_std, n_samples)
    S_samples = rng.normal(salinity, S_std, n_samples)

    # Clip to valid ranges
    T_samples = np.clip(T_samples, T_MIN_VALID + 0.1, T_MAX_VALID - 0.1)
    S_samples = np.maximum(S_samples, 0.0)

    # Compute brine volume for each sample
    nu_b_samples = brine_volume_fraction(T_samples, S_samples, extrapolate=True)

    return {
        "mean": float(np.mean(nu_b_samples)),
        "std": float(np.std(nu_b_samples)),
        "p05": float(np.percentile(nu_b_samples, 5)),
        "p50": float(np.percentile(nu_b_samples, 50)),
        "p95": float(np.percentile(nu_b_samples, 95)),
        "samples": nu_b_samples,
    }
