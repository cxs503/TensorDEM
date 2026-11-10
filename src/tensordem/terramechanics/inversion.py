"""Parameter inversion for Bekker and Janosi models from experimental data.

This module provides least-squares fitting routines to extract terramechanics
parameters from plate penetration tests (Bekker) and direct shear tests
(Janosi).

Bekker inversion
----------------
Given pressure–sinkage data ``(z_i, p_i)``, fit the Bekker equation:

    p = (k_c / b + k_phi) * z^n

by nonlinear least squares. The width ``b`` must be known (plate width).

Janosi inversion
----------------
Given shear displacement–stress data ``(j_i, tau_i)`` at known normal pressure
``p``, fit the Janosi equation:

    tau = tau_max * (1 - exp(-j / K))

by nonlinear least squares. The maximum shear stress ``tau_max`` can either be
fixed (from Mohr–Coulomb) or fitted.

Units are SI (m, Pa, -).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

__all__ = [
    "BekkerInversionResult",
    "invert_bekker_parameters",
    "invert_janosi_parameters",
]


def _finite(name: str, value: float, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class BekkerInversionResult:
    """Result of Bekker parameter inversion."""

    k_c: float          # N/m^{n+1}
    k_phi: float        # N/m^{n+2}
    n: float            # dimensionless
    r_squared: float    # goodness of fit
    rmse: float         # root mean square error (Pa)
    n_data: int         # number of data points

    @property
    def stiffness(self) -> float:
        """Combined stiffness k_c/b + k_phi for unit width."""
        return self.k_c + self.k_phi


def _bekker_model(z: np.ndarray, log_k_combined: float, n: float) -> np.ndarray:
    """Bekker model in log-space for fitting: log(p) = log(k_combined) + n * log(z)."""
    return np.exp(log_k_combined) * np.power(np.maximum(z, 1e-15), n)


def invert_bekker_parameters(
    sinkage: np.ndarray, pressure: np.ndarray, width: float,
    *, k_c_guess: float = 100.0, k_phi_guess: float = 10000.0, n_guess: float = 1.0,
) -> BekkerInversionResult:
    """Invert Bekker parameters from plate penetration test data.

    Parameters
    ----------
    sinkage : np.ndarray
        Sinkage data ``z`` (m), must be > 0.
    pressure : np.ndarray
        Pressure data ``p`` (Pa), must be > 0.
    width : float
        Plate width ``b`` (m).
    k_c_guess, k_phi_guess, n_guess : float
        Initial guesses for the parameters.

    Returns
    -------
    BekkerInversionResult
        Fitted parameters with goodness-of-fit metrics.
    """
    z = np.asarray(sinkage, dtype=float)
    p = np.asarray(pressure, dtype=float)
    b = _finite("width", width, positive=True)

    if len(z) < 3:
        raise ValueError("at least 3 data points required for Bekker inversion")
    if np.any(z <= 0) or np.any(p <= 0):
        raise ValueError("sinkage and pressure must be positive")

    # Log-linear fit for initial estimate: log(p) = log(k_combined) + n * log(z)
    log_z = np.log(z)
    log_p = np.log(p)
    coeffs = np.polyfit(log_z, log_p, 1)
    n_init = coeffs[0]
    log_k_init = coeffs[1]
    k_combined_init = np.exp(log_k_init)

    # Split k_combined into k_c and k_phi using guesses
    k_c_init = max(k_c_guess, 1e-6)
    k_phi_init = max(k_combined_init - k_c_init / b, 1e-6)
    n_init = max(min(n_init, 3.0), 0.1)

    def bekker_func(z_arr, k_c_fit, k_phi_fit, n_fit):
        k_comb = k_c_fit / b + k_phi_fit
        return k_comb * np.power(np.maximum(z_arr, 1e-15), n_fit)

    try:
        popt, pcov = curve_fit(
            bekker_func, z, p,
            p0=[k_c_init, k_phi_init, n_init],
            bounds=([0, 0, 0.05], [np.inf, np.inf, 3.0]),
            maxfev=10000,
        )
        k_c_fit, k_phi_fit, n_fit = popt

        # Compute goodness of fit
        p_pred = bekker_func(z, k_c_fit, k_phi_fit, n_fit)
        ss_res = np.sum((p - p_pred) ** 2)
        ss_tot = np.sum((p - np.mean(p)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(ss_res / len(z)))

        return BekkerInversionResult(
            k_c=float(k_c_fit),
            k_phi=float(k_phi_fit),
            n=float(n_fit),
            r_squared=float(r_squared),
            rmse=rmse,
            n_data=len(z),
        )
    except Exception:
        # Fall back to log-linear estimate
        k_c_fallback = k_c_init
        k_phi_fallback = max(k_combined_init - k_c_init / b, 1e-6)
        n_fallback = n_init
        p_pred = (k_c_fallback / b + k_phi_fallback) * z ** n_fallback
        ss_res = np.sum((p - p_pred) ** 2)
        ss_tot = np.sum((p - np.mean(p)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(ss_res / len(z)))

        return BekkerInversionResult(
            k_c=k_c_fallback,
            k_phi=k_phi_fallback,
            n=n_fallback,
            r_squared=float(r_squared),
            rmse=rmse,
            n_data=len(z),
        )


@dataclass(frozen=True)
class JanosiInversionResult:
    """Result of Janosi parameter inversion."""

    tau_max: float      # Pa
    K: float            # m
    r_squared: float
    rmse: float         # Pa
    n_data: int


def invert_janosi_parameters(
    shear_displacement: np.ndarray, shear_stress: np.ndarray,
    *, tau_max_guess: float = 50000.0, K_guess: float = 0.03,
    tau_max_fixed: float | None = None,
) -> JanosiInversionResult:
    """Invert Janosi parameters from direct shear test data.

    Parameters
    ----------
    shear_displacement : np.ndarray
        Shear displacement ``j`` (m), must be >= 0.
    shear_stress : np.ndarray
        Shear stress ``tau`` (Pa), must be >= 0.
    tau_max_guess, K_guess : float
        Initial guesses.
    tau_max_fixed : float or None
        If provided, tau_max is fixed to this value and only K is fitted.

    Returns
    -------
    JanosiInversionResult
        Fitted parameters with goodness-of-fit metrics.
    """
    j = np.asarray(shear_displacement, dtype=float)
    tau = np.asarray(shear_stress, dtype=float)

    if len(j) < 3:
        raise ValueError("at least 3 data points required for Janosi inversion")
    if np.any(j < 0) or np.any(tau < 0):
        raise ValueError("shear displacement and stress must be nonnegative")

    def janosi_func(j_arr, tmax, K_fit):
        return tmax * (1.0 - np.exp(-j_arr / K_fit))

    if tau_max_fixed is not None:
        tmax = _finite("tau_max_fixed", tau_max_fixed, positive=True)

        def janosi_fixed_K(j_arr, K_fit):
            return tmax * (1.0 - np.exp(-j_arr / K_fit))

        try:
            popt, _ = curve_fit(
                janosi_fixed_K, j, tau,
                p0=[K_guess],
                bounds=([1e-6], [1.0]),
                maxfev=10000,
            )
            K_fit = float(popt[0])
        except Exception:
            K_fit = K_guess

        tau_pred = janosi_func(j, tmax, K_fit)
    else:
        tmax_init = max(tau_max_guess, np.max(tau) * 1.1)
        try:
            popt, _ = curve_fit(
                janosi_func, j, tau,
                p0=[tmax_init, K_guess],
                bounds=([0, 1e-6], [np.inf, 1.0]),
                maxfev=10000,
            )
            tmax, K_fit = popt
        except Exception:
            tmax = tmax_init
            K_fit = K_guess

        tau_pred = janosi_func(j, tmax, K_fit)

    ss_res = np.sum((tau - tau_pred) ** 2)
    ss_tot = np.sum((tau - np.mean(tau)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(ss_res / len(j)))

    return JanosiInversionResult(
        tau_max=float(tmax),
        K=float(K_fit),
        r_squared=float(r_squared),
        rmse=rmse,
        n_data=len(j),
    )
