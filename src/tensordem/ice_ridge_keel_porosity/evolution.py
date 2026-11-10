"""Porosity evolution: exponential decay fitting and analysis.

The porosity evolution of a consolidating ice ridge keel follows:

    n_keel(t) = n_∞ + (n_0 - n_∞) · exp(-t / τ)

where:
- n_0 = initial porosity (t=0), typically ≈ 0.4
- n_∞ = asymptotic porosity (t→∞), typically ≈ 0.2
- τ = consolidation time scale (s)

This module provides:
- Analytical exponential decay function
- Least-squares fitting to DEM time series data
- Goodness-of-fit metrics (R², residual)
- End-to-end evolution analysis pipeline
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple, Optional

import numpy as np
from scipy.optimize import curve_fit

from .geometry import KeelGeometry
from .packing import KeelPackingConfig, generate_rubble_packing
from .consolidation import (
    ConsolidationConfig,
    ConsolidationResult,
    run_consolidation,
)

__all__ = [
    "PorosityEvolutionResult",
    "exponential_decay",
    "fit_exponential_decay",
    "run_evolution_analysis",
]


@dataclass
class PorosityEvolutionResult:
    """Result of porosity evolution analysis.

    Attributes
    ----------
    n_0 : float
        Initial porosity (fitted or measured).
    n_inf : float
        Asymptotic porosity (fitted).
    tau : float
        Consolidation time scale (s).
    r_squared : float
        Coefficient of determination R².
    residual_rms : float
        Root-mean-square residual.
    time : np.ndarray
        Time stamps from DEM (s).
    porosity_dem : np.ndarray
        Porosity from DEM simulation.
    porosity_fit : np.ndarray
        Porosity from exponential fit.
    consolidation : ConsolidationResult
        Full consolidation result.
    """
    n_0: float
    n_inf: float
    tau: float
    r_squared: float
    residual_rms: float
    time: np.ndarray
    porosity_dem: np.ndarray
    porosity_fit: np.ndarray
    consolidation: ConsolidationResult


def exponential_decay(t: np.ndarray, n_0: float, n_inf: float, tau: float) -> np.ndarray:
    """Exponential decay model for porosity evolution.

    n(t) = n_∞ + (n_0 - n_∞) · exp(-t / τ)

    Parameters
    ----------
    t : np.ndarray
        Time values (s).
    n_0 : float
        Initial porosity.
    n_inf : float
        Asymptotic porosity.
    tau : float
        Time scale (s).

    Returns
    -------
    n : np.ndarray
        Porosity at each time.
    """
    return n_inf + (n_0 - n_inf) * np.exp(-t / tau)


def fit_exponential_decay(
    time: np.ndarray,
    porosity: np.ndarray,
    p0: Tuple[float, float, float] = None,
    bounds: Tuple = None,
) -> Tuple[float, float, float, float, float]:
    """Fit exponential decay to porosity time series.

    Parameters
    ----------
    time : np.ndarray
        Time stamps (s).
    porosity : np.ndarray
        Porosity values.
    p0 : tuple, optional
        Initial guess (n_0, n_inf, tau). Default (0.4, 0.2, 100).
    bounds : tuple, optional
        Parameter bounds ((lower,), (upper,)). Default enforces physical ranges.

    Returns
    -------
    n_0 : float
        Fitted initial porosity.
    n_inf : float
        Fitted asymptotic porosity.
    tau : float
        Fitted time scale (s).
    r_squared : float
        Coefficient of determination.
    residual_rms : float
        RMS residual.
    """
    if len(time) < 3:
        raise ValueError("Need at least 3 data points for fitting")

    if p0 is None:
        p0 = (porosity[0], porosity[-1], 100.0)

    if bounds is None:
        # Physical bounds: porosities in [0.05, 0.95], tau > 0
        bounds = (
            [0.05, 0.05, 1.0],      # lower bounds
            [0.95, 0.95, 1e6],      # upper bounds
        )

    # Fit
    popt, pcov = curve_fit(
        exponential_decay,
        time,
        porosity,
        p0=p0,
        bounds=bounds,
        maxfev=10000,
    )
    n_0_fit, n_inf_fit, tau_fit = popt

    # Goodness of fit
    porosity_fit = exponential_decay(time, *popt)
    ss_res = np.sum((porosity - porosity_fit) ** 2)
    ss_tot = np.sum((porosity - np.mean(porosity)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    residual_rms = np.sqrt(np.mean((porosity - porosity_fit) ** 2))

    return n_0_fit, n_inf_fit, tau_fit, r_squared, residual_rms


def run_evolution_analysis(
    geom: KeelGeometry = None,
    packing_config: KeelPackingConfig = None,
    consolidation_config: ConsolidationConfig = None,
) -> PorosityEvolutionResult:
    """Run end-to-end porosity evolution analysis.

    1. Generate initial rubble packing
    2. Run DEM consolidation simulation
    3. Fit exponential decay to porosity time series
    4. Return comprehensive result

    Parameters
    ----------
    geom : KeelGeometry, optional
        Keel geometry. Default: 10m×10m×5m.
    packing_config : KeelPackingConfig, optional
        Packing configuration.
    consolidation_config : ConsolidationConfig, optional
        Consolidation simulation configuration.

    Returns
    -------
    PorosityEvolutionResult
        Complete evolution analysis result.
    """
    if geom is None:
        geom = KeelGeometry(L_top=10.0, W_top=10.0, L_bot=2.0, W_bot=2.0, depth=5.0)

    if packing_config is None:
        packing_config = KeelPackingConfig(n_particles=500, seed=42)

    if consolidation_config is None:
        consolidation_config = ConsolidationConfig(
            total_time=500.0,
            record_interval=5.0,
            wave_amplitude=0.5,
            wave_period=10.0,
        )

    # Step 1: Generate packing
    packing = generate_rubble_packing(geom, packing_config)

    # Step 2: Run consolidation
    consolidation = run_consolidation(packing, geom, consolidation_config)

    # Step 3: Fit exponential decay
    time = consolidation.time
    porosity = consolidation.porosity

    n_0_fit, n_inf_fit, tau_fit, r_squared, residual_rms = fit_exponential_decay(
        time, porosity
    )

    # Compute fit curve
    porosity_fit = exponential_decay(time, n_0_fit, n_inf_fit, tau_fit)

    return PorosityEvolutionResult(
        n_0=n_0_fit,
        n_inf=n_inf_fit,
        tau=tau_fit,
        r_squared=r_squared,
        residual_rms=residual_rms,
        time=time,
        porosity_dem=porosity,
        porosity_fit=porosity_fit,
        consolidation=consolidation,
    )
