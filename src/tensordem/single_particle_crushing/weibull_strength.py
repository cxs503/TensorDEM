"""Weibull strength–size effect model for single-particle crushing.

The Weibull theory of brittle strength postulates that failure is governed by
the weakest-link principle: a larger specimen has a higher probability of
containing a critical flaw.  The resulting size effect on the characteristic
failure stress is

    σ_f = σ_0 (V / V_0)^{-1/m}

where:
  - σ_0 is the characteristic strength at reference volume V_0
  - m is the Weibull modulus (shape parameter; higher = less scatter)
  - V is the particle volume

The failure probability under applied stress σ is

    P_f = 1 − exp[−(σ/σ_0)^m (V/V_0)]

For a collection of particles at the same stress level, the fraction that
survives is exp[−(σ/σ_0)^m (V/V_0)].

This module provides:
  - Analytical functions for the Weibull strength–size law
  - Maximum-likelihood estimation of m and σ_0 from test data
  - Synthetic data generation for verification
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class WeibullStrengthModel:
    """Parameters of the Weibull strength–size effect model.

    Attributes
    ----------
    sigma_0 : float
        Characteristic strength (Pa) at reference volume V_0.
    m : float
        Weibull modulus (dimensionless). Typical values: 5–10 for rocks,
        3–5 for concrete, 10–20 for ceramics.
    V_0 : float
        Reference volume (m^3).
    """
    sigma_0: float
    m: float
    V_0: float = 1.0e-9  # default 1 mm^3

    def __post_init__(self) -> None:
        if self.sigma_0 <= 0:
            raise ValueError("sigma_0 must be positive")
        if self.m <= 0:
            raise ValueError("m must be positive")
        if self.V_0 <= 0:
            raise ValueError("V_0 must be positive")


def weibull_strength_size(model: WeibullStrengthModel, V: float) -> float:
    """Characteristic failure stress at volume V.

    σ_f(V) = σ_0 (V / V_0)^{-1/m}

    Parameters
    ----------
    model : WeibullStrengthModel
        Weibull model parameters.
    V : float
        Particle volume (m^3).

    Returns
    -------
    float
        Characteristic failure stress (Pa).
    """
    if V <= 0:
        raise ValueError("Volume must be positive")
    return model.sigma_0 * (V / model.V_0) ** (-1.0 / model.m)


def weibull_failure_probability(
    model: WeibullStrengthModel,
    sigma: float,
    V: float,
) -> float:
    """Failure probability under stress σ for a particle of volume V.

    P_f = 1 − exp[−(σ/σ_0)^m (V/V_0)]

    Parameters
    ----------
    model : WeibullStrengthModel
        Weibull model parameters.
    sigma : float
        Applied stress (Pa). Must be non-negative.
    V : float
        Particle volume (m^3). Must be positive.

    Returns
    -------
    float
        Failure probability in [0, 1].
    """
    if sigma < 0:
        raise ValueError("Stress must be non-negative")
    if V <= 0:
        raise ValueError("Volume must be positive")
    if sigma == 0:
        return 0.0
    exponent = (sigma / model.sigma_0) ** model.m * (V / model.V_0)
    return 1.0 - math.exp(-exponent)


def fit_weibull_modulus(
    strengths: Sequence[float],
    volumes: Sequence[float] | None = None,
    V_0: float = 1.0e-9,
) -> tuple[float, float]:
    """Maximum-likelihood estimation of Weibull modulus m and σ_0.

    For a set of failure stresses {σ_i} at (possibly different) volumes {V_i},
    the MLE for m is obtained from the linearized form:

        ln[−ln(1 − P_f)] = m ln(σ) − m ln(σ_0) + ln(V/V_0)

    Using median-rank approximation P_f,i ≈ (i − 0.5) / N for ordered data.

    Parameters
    ----------
    strengths : sequence of float
        Failure stresses (Pa), one per test specimen.
    volumes : sequence of float, optional
        Volumes (m^3) for each specimen. If None, all volumes are assumed equal
        to V_0.
    V_0 : float
        Reference volume (m^3).

    Returns
    -------
    m : float
        Estimated Weibull modulus.
    sigma_0 : float
        Estimated characteristic strength (Pa).
    """
    sigmas = np.asarray(strengths, dtype=np.float64)
    n = len(sigmas)
    if n < 3:
        raise ValueError("Need at least 3 data points for Weibull fit")

    if volumes is None:
        vols = np.full(n, V_0, dtype=np.float64)
    else:
        vols = np.asarray(volumes, dtype=np.float64)
        if len(vols) != n:
            raise ValueError("strengths and volumes must have same length")

    # Sort by stress
    order = np.argsort(sigmas)
    sigmas_sorted = sigmas[order]
    vols_sorted = vols[order]

    # Median-rank approximation for failure probability
    ranks = np.arange(1, n + 1, dtype=np.float64)
    Pf = (ranks - 0.5) / n

    # Linearize: ln[-ln(1-Pf)] = m*ln(sigma) - m*ln(sigma_0) + ln(V/V_0)
    # => ln[-ln(1-Pf)] - ln(V/V_0) = m*ln(sigma) - m*ln(sigma_0)
    y = np.log(-np.log(1.0 - Pf)) - np.log(vols_sorted / V_0)
    x = np.log(sigmas_sorted)

    # Linear regression: y = m*x + b, where b = -m*ln(sigma_0)
    # => sigma_0 = exp(-b/m)
    coeffs = np.polyfit(x, y, 1)
    m_est = coeffs[0]
    b_est = coeffs[1]
    sigma_0_est = math.exp(-b_est / m_est)

    return float(m_est), float(sigma_0_est)


def generate_weibull_strengths(
    model: WeibullStrengthModel,
    n_samples: int,
    V: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate synthetic failure stresses from the Weibull distribution.

    The failure stress at volume V follows a Weibull distribution with
    shape parameter m and scale parameter σ_f(V) = σ_0 (V/V_0)^{-1/m}.

    Parameters
    ----------
    model : WeibullStrengthModel
        Weibull model parameters.
    n_samples : int
        Number of samples to generate.
    V : float, optional
        Particle volume (m^3). If None, uses V_0.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of failure stresses (Pa).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if V is None:
        V = model.V_0

    sigma_f = weibull_strength_size(model, V)
    # Weibull-distributed random variates
    uniforms = rng.random(n_samples)
    strengths = sigma_f * (-np.log(1.0 - uniforms)) ** (1.0 / model.m)
    return strengths


def weibull_r_squared(
    strengths: Sequence[float],
    volumes: Sequence[float] | None = None,
    V_0: float = 1.0e-9,
) -> float:
    """Compute R² of the Weibull linear fit.

    Parameters
    ----------
    strengths : sequence of float
        Failure stresses (Pa).
    volumes : sequence of float, optional
        Volumes (m^3). If None, all equal to V_0.
    V_0 : float
        Reference volume (m^3).

    Returns
    -------
    float
        R² value of the linearized Weibull fit.
    """
    sigmas = np.asarray(strengths, dtype=np.float64)
    n = len(sigmas)

    if volumes is None:
        vols = np.full(n, V_0, dtype=np.float64)
    else:
        vols = np.asarray(volumes, dtype=np.float64)

    order = np.argsort(sigmas)
    sigmas_sorted = sigmas[order]
    vols_sorted = vols[order]

    ranks = np.arange(1, n + 1, dtype=np.float64)
    Pf = (ranks - 0.5) / n

    y = np.log(-np.log(1.0 - Pf)) - np.log(vols_sorted / V_0)
    x = np.log(sigmas_sorted)

    coeffs = np.polyfit(x, y, 1)
    y_pred = coeffs[0] * x + coeffs[1]

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        return 1.0
    return float(1.0 - ss_res / ss_tot)
