"""Gaudin–Schuhmann破碎粒度分布模型。

The Gaudin-Schuhmann distribution (also called the power-law or
size-number distribution) is a widely used empirical model for the
fragment size distribution produced by comminution processes:

    P(d) = (d / d_max)^α

where:
  - P(d) is the cumulative fraction passing (by number or mass)
  - d is the particle diameter
  - d_max is the maximum fragment size
  - α is the distribution exponent (typically 0.5–3.0)

The corresponding PDF is:
    p(d) = α d^{α-1} / d_max^α

This module provides:
  - Analytical CDF and PDF functions
  - Maximum-likelihood estimation of α from fragment size data
  - Synthetic data generation for verification
  - Goodness-of-fit metrics (R², KS statistic)
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class GaudinSchuhmannDistribution:
    """Parameters of the Gaudin-Schuhmann fragment size distribution.

    Attributes
    ----------
    alpha : float
        Distribution exponent (dimensionless). Typical values:
        0.5–1.5 for coarse crushing, 1.5–3.0 for fine grinding.
    d_max : float
        Maximum fragment size (m).
    """
    alpha: float
    d_max: float

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.d_max <= 0:
            raise ValueError("d_max must be positive")


def gaudin_cdf(d: float | np.ndarray, dist: GaudinSchuhmannDistribution) -> float | np.ndarray:
    """Gaudin-Schuhmann cumulative distribution function.

    P(d) = (d / d_max)^α  for 0 ≤ d ≤ d_max
    P(d) = 1               for d > d_max
    P(d) = 0               for d < 0

    Parameters
    ----------
    d : float or np.ndarray
        Particle diameter(s) (m).
    dist : GaudinSchuhmannDistribution
        Distribution parameters.

    Returns
    -------
    float or np.ndarray
        Cumulative probability (fraction passing).
    """
    d = np.asarray(d, dtype=np.float64)
    result = np.where(
        d <= 0,
        0.0,
        np.where(
            d >= dist.d_max,
            1.0,
            (d / dist.d_max) ** dist.alpha
        )
    )
    if result.ndim == 0:
        return float(result)
    return result


def gaudin_pdf(d: float | np.ndarray, dist: GaudinSchuhmannDistribution) -> float | np.ndarray:
    """Gaudin-Schuhmann probability density function.

    p(d) = α d^{α-1} / d_max^α  for 0 ≤ d ≤ d_max
    p(d) = 0                      otherwise

    Parameters
    ----------
    d : float or np.ndarray
        Particle diameter(s) (m).
    dist : GaudinSchuhmannDistribution
        Distribution parameters.

    Returns
    -------
    float or np.ndarray
        Probability density.
    """
    d = np.asarray(d, dtype=np.float64)
    result = np.where(
        (d >= 0) & (d <= dist.d_max),
        dist.alpha * d ** (dist.alpha - 1) / dist.d_max ** dist.alpha,
        0.0
    )
    if result.ndim == 0:
        return float(result)
    return result


def fit_gaudin_exponent(
    sizes: Sequence[float],
    d_max: float | None = None,
) -> tuple[float, float]:
    """Maximum-likelihood estimation of the Gaudin exponent α.

    For the Gaudin-Schuhmann distribution, the MLE of α given data {d_i}
    and known d_max is:

        α = N / Σ ln(d_max / d_i)

    If d_max is not provided, it is estimated as the maximum observed size.

    Parameters
    ----------
    sizes : sequence of float
        Fragment diameters (m). Must be positive and ≤ d_max.
    d_max : float, optional
        Maximum fragment size (m). If None, uses max(sizes).

    Returns
    -------
    alpha : float
        Estimated distribution exponent.
    alpha_stderr : float
        Standard error of the estimate.
    """
    data = np.asarray(sizes, dtype=np.float64)
    data = data[data > 0]  # Remove non-positive values
    n = len(data)

    if n < 2:
        raise ValueError("Need at least 2 data points")

    if d_max is None:
        d_max = float(data.max())
    else:
        # Filter out values exceeding d_max
        data = data[data <= d_max]
        n = len(data)
        if n < 2:
            raise ValueError("Too few valid data points")

    # MLE: α = N / Σ ln(d_max / d_i)
    log_ratios = np.log(d_max / data)
    # Avoid division by zero for d_i = d_max
    log_ratios = log_ratios[log_ratios > 1e-10]
    n_valid = len(log_ratios)

    if n_valid < 2:
        raise ValueError("Insufficient variation in data for MLE")

    alpha = n_valid / log_ratios.sum()
    # Asymptotic standard error
    alpha_stderr = alpha / math.sqrt(n_valid)

    return float(alpha), float(alpha_stderr)


def generate_gaudin_sizes(
    dist: GaudinSchuhmannDistribution,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate synthetic fragment sizes from the Gaudin-Schuhmann distribution.

    Uses inverse CDF sampling:
        d = d_max * U^{1/α}
    where U ~ Uniform(0, 1).

    Parameters
    ----------
    dist : GaudinSchuhmannDistribution
        Distribution parameters.
    n_samples : int
        Number of samples to generate.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Fragment diameters (m).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    uniforms = rng.random(n_samples)
    sizes = dist.d_max * uniforms ** (1.0 / dist.alpha)
    return sizes


def gaudin_r_squared(
    sizes: Sequence[float],
    d_max: float | None = None,
    n_bins: int = 20,
) -> float:
    """Compute R² of the Gaudin-Schuhmann fit.

    Compares the empirical CDF (from sorted data) with the fitted
    Gaudin-Schuhmann CDF.

    Parameters
    ----------
    sizes : sequence of float
        Fragment diameters (m).
    d_max : float, optional
        Maximum fragment size. If None, uses max(sizes).
    n_bins : int
        Number of bins for the comparison.

    Returns
    -------
    float
        R² value of the CDF fit.
    """
    data = np.sort(np.asarray(sizes, dtype=np.float64))
    data = data[data > 0]
    n = len(data)

    if n < 5:
        raise ValueError("Need at least 5 data points")

    if d_max is None:
        d_max = float(data.max())

    alpha, _ = fit_gaudin_exponent(data, d_max)
    dist = GaudinSchuhmannDistribution(alpha=alpha, d_max=d_max)

    # Empirical CDF at data points
    empirical_cdf = np.arange(1, n + 1) / n

    # Model CDF at data points
    model_cdf = gaudin_cdf(data, dist)

    # R² computation
    ss_res = np.sum((empirical_cdf - model_cdf) ** 2)
    ss_tot = np.sum((empirical_cdf - np.mean(empirical_cdf)) ** 2)

    if ss_tot == 0:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def gaudin_ks_statistic(
    sizes: Sequence[float],
    d_max: float | None = None,
) -> float:
    """Compute the Kolmogorov-Smirnov statistic for Gaudin-Schuhmann fit.

    Parameters
    ----------
    sizes : sequence of float
        Fragment diameters (m).
    d_max : float, optional
        Maximum fragment size. If None, uses max(sizes).

    Returns
    -------
    float
        KS statistic (maximum absolute difference between empirical and
        model CDF).
    """
    data = np.sort(np.asarray(sizes, dtype=np.float64))
    data = data[data > 0]
    n = len(data)

    if n < 2:
        raise ValueError("Need at least 2 data points")

    if d_max is None:
        d_max = float(data.max())

    alpha, _ = fit_gaudin_exponent(data, d_max)
    dist = GaudinSchuhmannDistribution(alpha=alpha, d_max=d_max)

    empirical_cdf = np.arange(1, n + 1) / n
    model_cdf = gaudin_cdf(data, dist)

    return float(np.max(np.abs(empirical_cdf - model_cdf)))
