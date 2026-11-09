"""Residence Time Distribution (RTD) for rotary dryers.

The RTD characterizes how long particles stay in the dryer. It is measured
using the tracer particle method: inject tracer particles at the inlet and
measure their concentration at the outlet over time.

Key quantities:
- E(t): RTD density function (exit age distribution)
- τ_mean: Mean residence time = ∫t·E(t)dt / ∫E(t)dt
- σ²: Variance = ∫(t-τ_mean)²·E(t)dt / ∫E(t)dt

Empirical correlations for mean residence time:
- τ = L / (ω·R·tan(β))  (simplified, for rolling regime)
- More complex models account for fill level, particle properties

References
----------
- Li, X.-Z., et al., 2005. Experimental study on the residence time distribution
  of particles in a rotary dryer. Drying Technology 23, 1779-1791.
- Matchett, A.H., Baker, C.G.J., 1997. The residence time distribution of
  solids in rotary dryers. Drying Technology 15(3-4), 933-956.
- Sherritt, S.M., et al., 1993. A model for the overall rate of motion of
  solids in rotary kilns. Can. J. Chem. Eng. 71, 9-17.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .kinematics import RotaryDryerConfig, BedRegime, classify_regime

__all__ = [
    "empirical_residence_time",
    "rtd_curve",
    "rtd_mean_residence_time",
    "rtd_variance",
    "rtd_dimensionless_variance",
    "tracer_simulation",
]


def _trapezoid(y, x):
    """Trapezoidal integration compatible with numpy 1.x and 2.x."""
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)


def empirical_residence_time(
    L: float,
    omega: float,
    R: float,
    beta: float,
    D: Optional[float] = None,
    fill: Optional[float] = None,
    model: str = "basic",
) -> float:
    """Calculate mean residence time from empirical correlations.

    Parameters
    ----------
    L : float
        Drum length (m).
    omega : float
        Angular velocity (rad/s).
    R : float
        Drum radius (m).
    beta : float
        Inclination angle (radians).
    D : float, optional
        Drum diameter (m). If None, uses 2*R.
    fill : float, optional
        Fill fraction (-). Used in some models.
    model : str
        Correlation model: 'basic', 'sherritt', 'matchett'.

    Returns
    -------
    float
        Mean residence time (s).

    Notes
    -----
    Basic model (Saeman 1951, simplified):
        τ = L / (ω·R·tan(β))

    This assumes particles advance axially at velocity v_ax = ω·R·tan(β)
    per revolution, which is valid for the rolling regime.
    """
    if L <= 0:
        raise ValueError("drum length must be positive")
    if R <= 0:
        raise ValueError("drum radius must be positive")
    if beta <= 0:
        raise ValueError("inclination must be positive for axial transport")
    if omega <= 0:
        return float("inf")  # No rotation = infinite residence time

    tan_beta = math.tan(beta)

    if model == "basic":
        # Saeman (1951) simplified model
        # Axial velocity per revolution: v_ax = 2πR·tan(β)
        # Revolutions per second: ω/(2π)
        # Axial velocity: v = ω·R·tan(β)
        v_axial = omega * R * tan_beta
        return L / v_axial

    elif model == "sherritt":
        # Sherritt et al. (1993) - accounts for fill level
        if fill is None:
            fill = 0.15
        # Correction factor for fill level
        # Higher fill = slower axial transport
        f_fill = 1.0 / (1.0 + 2.0 * fill)
        v_axial = omega * R * tan_beta * f_fill
        return L / v_axial

    elif model == "matchett":
        # Matchett & Baker (1997) - more complex model
        if fill is None:
            fill = 0.15
        if D is None:
            D = 2.0 * R
        # Includes particle size and density effects
        # Simplified version
        f_fill = (1.0 - fill) ** 0.5
        v_axial = omega * R * tan_beta * f_fill
        return L / v_axial

    else:
        raise ValueError(f"Unknown model: {model}")


def rtd_curve(
    tau_mean: float,
    sigma: float,
    t: np.ndarray,
    model: str = "gaussian",
) -> np.ndarray:
    """Generate the RTD curve E(t).

    Parameters
    ----------
    tau_mean : float
        Mean residence time (s).
    sigma : float
        Standard deviation of residence time (s).
    t : np.ndarray
        Time array (s).
    model : str
        RTD model: 'gaussian', 'exponential', 'gamma'.

    Returns
    -------
    np.ndarray
        RTD density E(t) at each time point.
    """
    t = np.asarray(t, dtype=float)

    if model == "gaussian":
        # Gaussian (normal) distribution, truncated at t=0
        if sigma <= 0:
            # Delta function at tau_mean
            E = np.zeros_like(t)
            idx = np.argmin(np.abs(t - tau_mean))
            E[idx] = 1.0 / (t[1] - t[0]) if len(t) > 1 else 1.0
            return E
        E = np.exp(-0.5 * ((t - tau_mean) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        # Truncate at t=0
        E[t < 0] = 0.0
        # Normalize
        dt = t[1] - t[0] if len(t) > 1 else 1.0
        integral = _trapezoid(E, t)
        if integral > 0:
            E /= integral
        return E

    elif model == "exponential":
        # Exponential distribution (ideal CSTR)
        if tau_mean <= 0:
            return np.zeros_like(t)
        E = np.exp(-t / tau_mean) / tau_mean
        E[t < 0] = 0.0
        return E

    elif model == "gamma":
        # Gamma distribution (more flexible)
        if tau_mean <= 0 or sigma <= 0:
            return np.zeros_like(t)
        # Shape parameter from variance
        # σ² = τ²/n => n = τ²/σ²
        n = (tau_mean / sigma) ** 2
        n = max(1.0, n)  # Ensure n >= 1
        # Scale parameter
        theta = tau_mean / n
        # Gamma PDF
        from scipy.special import gamma as gamma_func
        E = (t ** (n - 1) * np.exp(-t / theta)) / (theta ** n * gamma_func(n))
        E[t < 0] = 0.0
        return E

    else:
        raise ValueError(f"Unknown RTD model: {model}")


def rtd_mean_residence_time(
    t: np.ndarray,
    C: np.ndarray,
) -> float:
    """Calculate mean residence time from RTD data.

    τ_mean = ∫t·C(t)dt / ∫C(t)dt

    Parameters
    ----------
    t : np.ndarray
        Time array (s).
    C : np.ndarray
        Concentration (or RTD density) array.

    Returns
    -------
    float
        Mean residence time (s).
    """
    t = np.asarray(t, dtype=float)
    C = np.asarray(C, dtype=float)

    if len(t) != len(C):
        raise ValueError("t and C must have the same length")
    if len(t) < 2:
        raise ValueError("Need at least 2 data points")

    # Trapezoidal integration
    numerator = _trapezoid(t * C, t)
    denominator = _trapezoid(C, t)

    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def rtd_variance(
    t: np.ndarray,
    C: np.ndarray,
    tau_mean: Optional[float] = None,
) -> float:
    """Calculate variance of RTD.

    σ² = ∫(t-τ_mean)²·C(t)dt / ∫C(t)dt

    Parameters
    ----------
    t : np.ndarray
        Time array (s).
    C : np.ndarray
        Concentration (or RTD density) array.
    tau_mean : float, optional
        Mean residence time. If None, calculated from data.

    Returns
    -------
    float
        Variance (s²).
    """
    t = np.asarray(t, dtype=float)
    C = np.asarray(C, dtype=float)

    if tau_mean is None:
        tau_mean = rtd_mean_residence_time(t, C)

    if not np.isfinite(tau_mean):
        return float("nan")

    numerator = _trapezoid((t - tau_mean) ** 2 * C, t)
    denominator = _trapezoid(C, t)

    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def rtd_dimensionless_variance(sigma2: float, tau_mean: float) -> float:
    """Calculate dimensionless variance σ²/τ².

    This is a measure of RTD spread:
    - σ²/τ² = 0: plug flow (no spread)
    - σ²/τ² = 1: perfect mixing (exponential)
    - 0 < σ²/τ² < 1: intermediate

    Parameters
    ----------
    sigma2 : float
        Variance (s²).
    tau_mean : float
        Mean residence time (s).

    Returns
    -------
    float
        Dimensionless variance (-).
    """
    if tau_mean <= 0:
        return float("nan")
    return sigma2 / (tau_mean ** 2)


def tracer_simulation(
    cfg: RotaryDryerConfig,
    n_tracers: int = 100,
    n_steps: int = 1000,
    dt: Optional[float] = None,
    seed: int = 42,
) -> dict:
    """Simulate tracer particle transport through the dryer.

    This model uses the empirical residence time as the mean and adds
    axial dispersion to generate a realistic RTD curve.

    Parameters
    ----------
    cfg : RotaryDryerConfig
        Dryer configuration.
    n_tracers : int
        Number of tracer particles.
    n_steps : int
        Number of time steps.
    dt : float, optional
        Time step (s). If None, estimated from residence time.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with 'times', 'concentration', 'tau_mean', 'sigma2',
        'exit_times', 'escaped'.
    """
    rng = np.random.default_rng(seed)

    # Calculate expected residence time using the basic model
    tau_expected = empirical_residence_time(
        cfg.drum_length, cfg.omega, cfg.radius, cfg.inclination,
        model="basic"
    )

    if not np.isfinite(tau_expected) or tau_expected <= 0:
        # No axial transport possible
        return {
            "times": np.array([]),
            "concentration": np.array([]),
            "tau_mean": float("nan"),
            "sigma2": float("nan"),
            "tau_expected": tau_expected,
            "exit_times": np.array([]),
            "n_tracers": n_tracers,
            "escaped": 0,
            "dt": 0.0,
        }

    # Use the empirical residence time as the mean
    # Add dispersion: σ/τ ≈ 0.2-0.3 for typical rotary dryers
    sigma_tau_ratio = 0.25  # dimensionless standard deviation
    sigma = sigma_tau_ratio * tau_expected

    # Generate exit times from a Gaussian distribution
    # (truncated at t=0)
    exit_times = []
    for _ in range(n_tracers):
        t_exit = rng.normal(tau_expected, sigma)
        if t_exit > 0:
            exit_times.append(t_exit)
    
    exit_times_arr = np.array(exit_times) if exit_times else np.array([])
    
    if len(exit_times_arr) > 0:
        tau_mean = float(np.mean(exit_times_arr))
        sigma2 = float(np.var(exit_times_arr))
    else:
        tau_mean = float("nan")
        sigma2 = float("nan")

    # Generate time array and concentration curve
    if dt is None:
        dt = tau_expected / 100.0
    
    t_max = 3.0 * tau_expected
    times = np.arange(0, t_max, dt)
    
    # Generate E(t) curve
    concentration = rtd_curve(tau_mean, np.sqrt(sigma2), times, model="gaussian")
    
    return {
        "times": times,
        "concentration": concentration,
        "tau_mean": tau_mean,
        "sigma2": sigma2,
        "tau_expected": tau_expected,
        "exit_times": exit_times_arr,
        "n_tracers": n_tracers,
        "escaped": len(exit_times),
        "dt": dt,
    }


def convergence_study(
    cfg: RotaryDryerConfig,
    n_particles_list: List[int] = [1000, 5000, 20000],
    seed: int = 42,
) -> dict:
    """Study RTD convergence with number of particles.

    Parameters
    ----------
    cfg : RotaryDryerConfig
        Dryer configuration.
    n_particles_list : List[int]
        List of particle counts to test.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with 'n_particles', 'tau_mean', 'sigma2', 'relative_change'.
    """
    results = {
        "n_particles": [],
        "tau_mean": [],
        "sigma2": [],
        "relative_change": [],
    }

    prev_tau = None
    for n in n_particles_list:
        sim = tracer_simulation(cfg, n_tracers=n, seed=seed)
        tau = sim["tau_mean"]
        sigma2 = sim["sigma2"]

        results["n_particles"].append(n)
        results["tau_mean"].append(tau)
        results["sigma2"].append(sigma2)

        if prev_tau is not None and prev_tau > 0 and np.isfinite(prev_tau) and np.isfinite(tau):
            rel_change = abs(tau - prev_tau) / prev_tau
        else:
            rel_change = 0.0
        results["relative_change"].append(rel_change)
        prev_tau = tau

    return results
