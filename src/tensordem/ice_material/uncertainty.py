"""Monte Carlo uncertainty propagation for ice strength predictions.

Physical motivation
-------------------
Ice strength predictions depend on uncertain inputs:
1. **Temperature T**: measurement uncertainty, spatial variability
2. **Salinity S**: measurement uncertainty, spatial variability
3. **Model parameters**: k (weakening), σ_ref (pure ice strength), f_desal (MYI)

This module propagates input uncertainties through the ice strength model
using Monte Carlo sampling, producing confidence intervals for strength
predictions.

Method
------
For each uncertain input, we specify a probability distribution (typically
normal with specified mean and standard deviation). We then:
1. Draw N random samples from the joint input distribution
2. Compute the strength for each sample
3. Compute statistics (mean, std, percentiles) of the output

This gives a full probability distribution for the strength, not just a
point estimate.

References
----------
- Rubinstein, R. Y. & Kroese, D. P. (2016). "Simulation and the Monte Carlo
  Method." 3rd ed., Wiley.
"""

import numpy as np
from typing import Optional, Dict, Any
from .compressive_strength import (
    compressive_strength_fyi,
    compressive_strength_myi,
    K_WEAKENING,
    SIGMA_REF,
    N_EXPONENT,
    F_DESAL_DEFAULT,
)


def monte_carlo_strength(
    temperature: float,
    salinity: float,
    ice_type: str = "FYI",
    T_std: float = 0.5,
    S_std: float = 0.5,
    k_std: float = 0.1,
    sigma_ref_std: float = 2.0,
    f_desal_std: float = 0.05,
    f_desal_mean: float = F_DESAL_DEFAULT,
    n_samples: int = 1000,
    seed: int = 42,
    extrapolate: bool = True,
) -> Dict[str, Any]:
    """Monte Carlo uncertainty propagation for ice compressive strength.

    Parameters
    ----------
    temperature : float
        Mean temperature in °C.
    salinity : float
        Mean salinity in ppt.
    ice_type : str
        "FYI" or "MYI".
    T_std : float
        Standard deviation of temperature uncertainty (°C).
    S_std : float
        Standard deviation of salinity uncertainty (ppt).
    k_std : float
        Standard deviation of weakening parameter k (dimensionless).
    sigma_ref_std : float
        Standard deviation of reference pure ice strength (MPa).
    f_desal_std : float
        Standard deviation of MYI desalination factor (only for MYI).
    f_desal_mean : float
        Mean of MYI desalination factor (only for MYI).
    n_samples : int
        Number of Monte Carlo samples.
    seed : int
        Random seed for reproducibility.
    extrapolate : bool
        If True, allow extrapolation outside valid T range.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'mean': mean strength (MPa)
        - 'std': standard deviation (MPa)
        - 'p05': 5th percentile (MPa)
        - 'p50': 50th percentile / median (MPa)
        - 'p95': 95th percentile (MPa)
        - 'cov': coefficient of variation (std/mean)
        - 'samples': array of all strength samples
        - 'input_samples': dict of input parameter samples
    """
    rng = np.random.default_rng(seed)

    # Sample input parameters from normal distributions
    T_samples = rng.normal(temperature, T_std, n_samples)
    S_samples = rng.normal(salinity, S_std, n_samples)
    k_samples = rng.normal(K_WEAKENING, k_std, n_samples)
    sigma_ref_samples = rng.normal(SIGMA_REF, sigma_ref_std, n_samples)

    # Clip to physically reasonable ranges
    T_samples = np.clip(T_samples, -40.0, -0.1)
    S_samples = np.maximum(S_samples, 0.0)
    k_samples = np.maximum(k_samples, 0.0)
    sigma_ref_samples = np.maximum(sigma_ref_samples, 0.0)

    if ice_type.upper() == "MYI":
        f_desal_samples = rng.normal(f_desal_mean, f_desal_std, n_samples)
        f_desal_samples = np.clip(f_desal_samples, 0.05, 1.0)
    else:
        f_desal_samples = np.full(n_samples, F_DESAL_DEFAULT)

    # Compute strength for each sample
    sigma_samples = np.zeros(n_samples)

    for i in range(n_samples):
        if ice_type.upper() == "FYI":
            sigma_samples[i] = compressive_strength_fyi(
                T_samples[i],
                S_samples[i],
                k_weakening=k_samples[i],
                sigma_ref=sigma_ref_samples[i],
                extrapolate=extrapolate,
            )
        elif ice_type.upper() == "MYI":
            sigma_samples[i] = compressive_strength_myi(
                T_samples[i],
                S_samples[i],
                f_desal=f_desal_samples[i],
                k_weakening=k_samples[i],
                sigma_ref=sigma_ref_samples[i],
                extrapolate=extrapolate,
            )
        else:
            raise ValueError(f"ice_type must be 'FYI' or 'MYI', got '{ice_type}'.")

    # Compute statistics
    mean_sigma = float(np.mean(sigma_samples))
    std_sigma = float(np.std(sigma_samples))
    cov = std_sigma / mean_sigma if mean_sigma > 0 else 0.0

    return {
        "mean": mean_sigma,
        "std": std_sigma,
        "p05": float(np.percentile(sigma_samples, 5)),
        "p50": float(np.percentile(sigma_samples, 50)),
        "p95": float(np.percentile(sigma_samples, 95)),
        "cov": cov,
        "samples": sigma_samples,
        "input_samples": {
            "temperature": T_samples,
            "salinity": S_samples,
            "k_weakening": k_samples,
            "sigma_ref": sigma_ref_samples,
            "f_desal": f_desal_samples,
        },
    }


def strength_confidence_band(
    temperatures: np.ndarray,
    salinity: float,
    ice_type: str = "FYI",
    T_std: float = 0.5,
    S_std: float = 0.5,
    n_samples_per_T: int = 500,
    seed: int = 42,
    extrapolate: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute confidence band for strength vs temperature.

    Parameters
    ----------
    temperatures : array
        Temperature array in °C.
    salinity : float
        Mean salinity in ppt.
    ice_type : str
        "FYI" or "MYI".
    T_std : float
        Standard deviation of temperature uncertainty (°C).
    S_std : float
        Standard deviation of salinity uncertainty (ppt).
    n_samples_per_T : int
        Number of Monte Carlo samples per temperature point.
    seed : int
        Random seed for reproducibility.
    extrapolate : bool
        If True, allow extrapolation outside valid T range.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'temperatures': input temperature array
        - 'mean': mean strength at each T (array)
        - 'p05': 5th percentile at each T (array)
        - 'p50': 50th percentile at each T (array)
        - 'p95': 95th percentile at each T (array)
    """
    n_T = len(temperatures)
    mean_arr = np.zeros(n_T)
    p05_arr = np.zeros(n_T)
    p50_arr = np.zeros(n_T)
    p95_arr = np.zeros(n_T)

    for i, T in enumerate(temperatures):
        mc_result = monte_carlo_strength(
            temperature=float(T),
            salinity=salinity,
            ice_type=ice_type,
            T_std=T_std,
            S_std=S_std,
            n_samples=n_samples_per_T,
            seed=seed + i,  # Different seed for each T
            extrapolate=extrapolate,
        )
        mean_arr[i] = mc_result["mean"]
        p05_arr[i] = mc_result["p05"]
        p50_arr[i] = mc_result["p50"]
        p95_arr[i] = mc_result["p95"]

    return {
        "temperatures": temperatures,
        "mean": mean_arr,
        "p05": p05_arr,
        "p50": p50_arr,
        "p95": p95_arr,
    }


def sensitivity_analysis(
    temperature: float,
    salinity: float,
    ice_type: str = "FYI",
    n_samples: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute sensitivity of strength to each input parameter.

    Uses one-at-a-time (OAT) sensitivity: vary each parameter by ±1 std
    while holding others fixed, and compute the resulting change in strength.

    Parameters
    ----------
    temperature : float
        Reference temperature in °C.
    salinity : float
        Reference salinity in ppt.
    ice_type : str
        "FYI" or "MYI".
    n_samples : int
        Number of samples for Monte Carlo (not used in OAT, kept for API).
    seed : int
        Random seed (not used in OAT, kept for API).

    Returns
    -------
    sensitivities : dict
        Dictionary with keys:
        - 'd_sigma_d_T': sensitivity to temperature (MPa/°C)
        - 'd_sigma_d_S': sensitivity to salinity (MPa/ppt)
        - 'd_sigma_d_k': sensitivity to weakening parameter (MPa)
        - 'd_sigma_d_sigma_ref': sensitivity to reference strength (dimensionless)
    """
    # Base case
    if ice_type.upper() == "FYI":
        sigma_base = compressive_strength_fyi(temperature, salinity)
    else:
        sigma_base = compressive_strength_myi(temperature, salinity)

    # Temperature sensitivity
    dT = 1.0  # 1°C perturbation
    if ice_type.upper() == "FYI":
        sigma_T_plus = compressive_strength_fyi(temperature + dT, salinity, extrapolate=True)
        sigma_T_minus = compressive_strength_fyi(temperature - dT, salinity, extrapolate=True)
    else:
        sigma_T_plus = compressive_strength_myi(temperature + dT, salinity, extrapolate=True)
        sigma_T_minus = compressive_strength_myi(temperature - dT, salinity, extrapolate=True)
    d_sigma_d_T = (sigma_T_plus - sigma_T_minus) / (2 * dT)

    # Salinity sensitivity
    dS = 1.0  # 1 ppt perturbation
    if ice_type.upper() == "FYI":
        sigma_S_plus = compressive_strength_fyi(temperature, salinity + dS, extrapolate=True)
        sigma_S_minus = compressive_strength_fyi(temperature, max(0, salinity - dS), extrapolate=True)
    else:
        sigma_S_plus = compressive_strength_myi(temperature, salinity + dS, extrapolate=True)
        sigma_S_minus = compressive_strength_myi(temperature, max(0, salinity - dS), extrapolate=True)
    d_sigma_d_S = (sigma_S_plus - sigma_S_minus) / (2 * dS)

    # Weakening parameter sensitivity
    dk = 0.1
    if ice_type.upper() == "FYI":
        sigma_k_plus = compressive_strength_fyi(
            temperature, salinity, k_weakening=K_WEAKENING + dk, extrapolate=True
        )
        sigma_k_minus = compressive_strength_fyi(
            temperature, salinity, k_weakening=K_WEAKENING - dk, extrapolate=True
        )
    else:
        sigma_k_plus = compressive_strength_myi(
            temperature, salinity, k_weakening=K_WEAKENING + dk, extrapolate=True
        )
        sigma_k_minus = compressive_strength_myi(
            temperature, salinity, k_weakening=K_WEAKENING - dk, extrapolate=True
        )
    d_sigma_d_k = (sigma_k_plus - sigma_k_minus) / (2 * dk)

    # Reference strength sensitivity
    dsigma_ref = 1.0  # 1 MPa perturbation
    if ice_type.upper() == "FYI":
        sigma_ref_plus = compressive_strength_fyi(
            temperature, salinity, sigma_ref=SIGMA_REF + dsigma_ref, extrapolate=True
        )
        sigma_ref_minus = compressive_strength_fyi(
            temperature, salinity, sigma_ref=SIGMA_REF - dsigma_ref, extrapolate=True
        )
    else:
        sigma_ref_plus = compressive_strength_myi(
            temperature, salinity, sigma_ref=SIGMA_REF + dsigma_ref, extrapolate=True
        )
        sigma_ref_minus = compressive_strength_myi(
            temperature, salinity, sigma_ref=SIGMA_REF - dsigma_ref, extrapolate=True
        )
    d_sigma_d_sigma_ref = (sigma_ref_plus - sigma_ref_minus) / (2 * dsigma_ref)

    return {
        "d_sigma_d_T": d_sigma_d_T,
        "d_sigma_d_S": d_sigma_d_S,
        "d_sigma_d_k": d_sigma_d_k,
        "d_sigma_d_sigma_ref": d_sigma_d_sigma_ref,
        "sigma_base": sigma_base,
    }
