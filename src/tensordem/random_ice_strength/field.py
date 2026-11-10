"""Spatially correlated Gaussian field along a beam strip and the Weibull copula field.

Why this module exists
----------------------
The strength of an ice plate is not a single number: two unit-width beam strips
10 m apart do not break at the same stress, and the *distance over which they
stay similar* is what decides how strongly the breaking force of a wide
structure averages down (the pressure-area / size effect).  Criterion (b) of
campaign node ``P295W1-N07`` is exactly this: generate a Gaussian random field
whose correlation function matches the prescribed one to an ``L2`` error of 10 %
or better, and show the correlation length has the effect it is supposed to
have.

Method (exact, no tuning)
-------------------------
For a stationary kernel ``rho(d)`` on a grid ``x`` the discretised covariance
matrix ``C_ij = rho(x_i - x_j)`` is squared-exponential/positive-definite (up to
round-off), so its eigen-decomposition is the *exact* Karhunen-Loeve basis of
the discretised process::

    C = sum_i lambda_i phi_i phi_i^T ,    Z = sum_i sqrt(lambda_i) phi_i xi_i ,
    xi_i ~ N(0, 1) i.i.d.

Sampling that sum with **every** eigenvalue above a round-off tolerance is
therefore exact at the grid points -- the only numerical content is the eigen
solve, and the retained energy fraction is reported so the truncation cannot
hide.  There is no circulant embedding, no spectral bump, no windowing: those
are legitimate alternatives, but they all trade accuracy for speed, and here the
grids are small (O(100) points) so accuracy is free.

Notes that matter downstream
----------------------------
* **Domain restriction is exact.**  A Gaussian process restricted to a
  sub-interval has the kernel restricted to that sub-interval.  So one
  decomposition on ``[0, B_max]`` gives, by simply slicing columns, exact
  realisations of the same process on ``[0, B]`` for every ``B <= B_max``.  The
  size-effect sweep uses this as a common-random-numbers device (nested
  realisations), which is why the small quantile differences between adjacent
  widths are resolvable at all.
* **The kernel does not know about the domain.**  ``rho`` is a function of the
  separation only, so the same ``L`` means the same physical correlation length
  at every width -- no re-parameterisation when the size sweep changes ``B``.
* **Marginal transform.**  The Weibull field is ``sigma(x) = F^{-1}(Phi(Z(x)))``
  (Gaussian copula).  The marginal is exact by construction and the *rank*
  correlation is inherited from the Gaussian field exactly;
  :func:`empirical_rank_correlation` compares it with the analytic target
  ``(6/pi) asin(rho/2)`` from :func:`tensordem.random_ice_strength.distributions.spearman_from_gaussian_correlation`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .distributions import (
    WeibullMarginal,
    gaussian_copula_uniforms,
    spearman_from_gaussian_correlation,
)

__all__ = [
    "KERNELS",
    "correlation_kernel",
    "correlation_matrix",
    "correlation_length_1e",
    "KlField1D",
    "gaussian_field_1d",
    "empirical_correlation",
    "empirical_rank_correlation",
    "correlation_l2_error",
    "correlated_weibull_field",
]


def correlation_kernel(d, length: float, model: str = "gaussian"):
    """Stationary correlation kernel ``rho(d)``.

    ``gaussian``     ``exp(-(d/L)^2)``   -- ``L`` is the ``1/e`` crossing;
    ``exponential``  ``exp(-|d|/L)``     -- ``L`` is the integral scale.

    The Gaussian kernel is the default because the brief asks for a *Gaussian*
    random field and because its ``1/e`` length is unambiguous (equal to the
    scale parameter), which makes the measured-versus-prescribed comparison
    direct.  The exponential kernel is offered so the same machinery can be run
    on the rougher (Markov) alternative without a second implementation.
    """
    d = np.asarray(d, dtype=float)
    if length <= 0.0:
        raise ValueError("length must be positive")
    if model == "gaussian":
        return np.exp(-(d / length) ** 2)
    if model == "exponential":
        return np.exp(-np.abs(d) / length)
    raise ValueError(f"unknown correlation model {model!r}")


KERNELS = ("gaussian", "exponential")


def correlation_matrix(x, length: float, model: str = "gaussian") -> np.ndarray:
    """Dense covariance matrix of the kernel on the grid ``x``."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a 1D grid with at least two points")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("x must be strictly increasing")
    return correlation_kernel(x[:, None] - x[None, :], length, model)


def correlation_length_1e(rho, lags) -> float:
    """Separation at which an empirical/target correlation has fallen to ``1/e``.

    Linear interpolation in ``d`` between the bracketing lags (no fit), with
    ``inf`` returned if the curve never reaches ``1/e`` inside the lag window.
    This is the length the report compares against the prescribed ``L``.
    """
    rho = np.asarray(rho, dtype=float)
    lags = np.asarray(lags, dtype=float)
    target = math.exp(-1.0)
    below = np.nonzero(rho <= target)[0]
    if below.size == 0:
        return float("inf")
    i = int(below[0])
    if i == 0:
        return float(lags[0])
    r0, r1 = rho[i - 1], rho[i]
    d0, d1 = lags[i - 1], lags[i]
    if r1 == r0:
        return float(d0)
    frac = (r0 - target) / (r0 - r1)
    return float(d0 + frac * (d1 - d0))


@dataclass(frozen=True)
class KlField1D:
    """Exact (grid-)Karhunen-Loeve representation of a 1D stationary field."""

    x: np.ndarray            # (n,)   grid, strictly increasing
    eigenvalues: np.ndarray  # (m,)   descending
    modes: np.ndarray        # (m, n) sqrt(lambda_i) * phi_i
    model: str
    length: float
    energy_fraction: float   # retained / total positive variance
    n_modes_available: int
    n_modes: int

    # -- sampling ---------------------------------------------------------
    def sample(self, n_samples: int, *, rng: Optional[np.random.Generator] = None,
               seed: int = 20260101) -> np.ndarray:
        """``(n_samples, nx)`` realisations of the zero-mean unit-variance field."""
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        rng = np.random.default_rng(seed) if rng is None else rng
        xi = rng.standard_normal((n_samples, self.n_modes))
        return xi @ self.modes

    # -- diagnostics ------------------------------------------------------
    @property
    def variance(self) -> float:
        return float(np.sum(self.eigenvalues))

    @property
    def n_points(self) -> int:
        return int(self.x.size)

    def target_correlation(self) -> np.ndarray:
        return correlation_kernel(self.x[None, :] - self.x[:, None], self.length, self.model)


def gaussian_field_1d(x, length: float, model: str = "gaussian", *,
                      energy_target: float = 1.0,
                      eigenvalue_tol: float = 1e-12) -> KlField1D:
    """Karhunen-Loeve basis of the kernel on the grid ``x``.

    Eigenvalues are sorted descending and all modes with
    ``lambda > eigenvalue_tol * lambda_max`` are kept; ``energy_target < 1``
    additionally truncates at the smallest mode count that reaches that
    fraction of the discrete variance (the truncated energy is reported in
    ``energy_fraction``, never silently assumed to be 1).
    """
    if not (0.0 < energy_target <= 1.0):
        raise ValueError("energy_target must be in (0, 1]")
    x = np.asarray(x, dtype=float)
    cov = correlation_matrix(x, length, model)
    w, v = np.linalg.eigh(cov)                     # ascending
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]
    w_max = float(w[0])
    keep = w > eigenvalue_tol * w_max
    w = w[keep]
    v = v[:, keep]
    total = float(np.sum(w))
    if energy_target < 1.0:
        cumulative = np.cumsum(w) / total
        m = int(np.searchsorted(cumulative, energy_target) + 1)
        w = w[:m]
        v = v[:, :m]
    modes = (np.sqrt(w)[:, None] * v.T).copy()
    retained = float(np.sum(w))
    return KlField1D(
        x=x,
        eigenvalues=w,
        modes=modes,
        model=model,
        length=float(length),
        energy_fraction=retained / total if total > 0.0 else float("nan"),
        n_modes_available=int(np.count_nonzero(keep)),
        n_modes=int(w.size),
    )


def empirical_correlation(fields, x=None, *, n_lags: Optional[int] = None,
                          reference: str = "ensemble") -> Tuple[np.ndarray, np.ndarray]:
    """Lag-resolved circular-free autocorrelation of a stack of fields on a line.

    Returns ``(lags, rho)`` with ``rho[0] = 1``.  The reference for the
    fluctuation is the **ensemble mean field** (``reference="ensemble"``), not
    each realisation's own spatial mean: subtracting the latter removes the
    ``k = 0`` mode of the KL expansion and biases small-lag correlations
    downwards -- an O(1) error at the first lag, as documented in
    :mod:`tensordem.stochastic_pressure_field.correlation`.
    """
    f = np.asarray(fields, dtype=float)
    if f.ndim == 1:
        f = f[None, :]
    if f.ndim != 2:
        raise ValueError("fields must be (n_samples, nx)")
    n_samples, nx = f.shape
    if n_samples < 2:
        raise ValueError("need at least two realisations")
    if reference == "ensemble":
        base = f.mean(axis=0)
    elif reference == "own":
        base = f.mean(axis=1, keepdims=True)
    else:
        raise ValueError("reference must be 'ensemble' or 'own'")
    fluct = f - base
    var = float(np.mean(fluct ** 2))
    if var <= 0.0:
        raise ValueError("fields have zero variance")
    max_lag = nx - 1 if n_lags is None else min(int(n_lags), nx - 1)
    lags = np.arange(max_lag + 1, dtype=float)
    rho = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        rho[k] = float(np.mean(fluct[:, : nx - k] * fluct[:, k:])) / var
    if x is not None:
        x = np.asarray(x, dtype=float)
        lags = lags * float(np.mean(np.diff(x)))   # uniform grid spacing -> distance
    return lags, rho


def empirical_rank_correlation(fields, x=None, *, n_lags: Optional[int] = None
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Spearman correlation vs separation, estimated on pooled pairs.

    For each lag the pairs ``(Z[s, i], Z[s, i+k])`` are pooled over every
    realisation ``s`` and position ``i`` and ranked on the pooled column, so the
    statistic estimates the *population* Spearman correlation of the pair
    distribution.  Under the Gaussian copula that population value is exactly
    ``(6/pi) asin(rho/2)`` -- see
    :func:`distributions.spearman_from_gaussian_correlation` -- which turns the
    comparison into a test of the generator rather than of an estimator.

    (Ranking each realisation against itself and averaging would instead
    estimate a *sample* Spearman of size ``nx``, biased low for strongly
    correlated pairs -- measured bias at ``nx = 101``, ``rho = 0.37``: -0.11.)
    """
    from scipy.stats import rankdata

    f = np.asarray(fields, dtype=float)
    if f.ndim == 1:
        f = f[None, :]
    n_samples, nx = f.shape
    max_lag = nx - 1 if n_lags is None else min(int(n_lags), nx - 1)
    lags = np.arange(max_lag + 1, dtype=float)
    rho = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        # pooled pairs: ranks on the pooled column estimate the *population*
        # Spearman of the pair distribution, which is the quantity the analytic
        # (6/pi) asin(rho/2) formula describes
        ra = rankdata(f[:, : nx - k].ravel()).astype(float)
        rb = rankdata(f[:, k:].ravel()).astype(float)
        ra -= ra.mean()
        rb -= rb.mean()
        denom = math.sqrt(float(np.dot(ra, ra)) * float(np.dot(rb, rb)))
        rho[k] = float(np.dot(ra, rb) / denom) if denom > 0.0 else float("nan")
    if x is not None:
        x = np.asarray(x, dtype=float)
        lags = lags * float(np.mean(np.diff(x)))
    return lags, rho


def correlation_l2_error(rho_emp, rho_target, *, lags=None, max_lag: Optional[float] = None
                         ) -> Dict[str, float]:
    """Relative ``L2`` error between an empirical and a target correlation curve.

    ``||rho_emp - rho_tgt||_2 / ||rho_tgt||_2`` over the lag window (default: all
    lags given).  The discrete normalisation is the pointwise one; it is the
    number the acceptance criterion quotes, and it is stated explicitly in the
    report so that "L2 error" cannot be read two ways.
    """
    a = np.asarray(rho_emp, dtype=float)
    b = np.asarray(rho_target, dtype=float)
    if a.shape != b.shape:
        raise ValueError("empirical and target correlation must share the same lag grid")
    mask = np.ones(a.size, dtype=bool)
    if max_lag is not None:
        if lags is None:
            raise ValueError("max_lag requires the lags array")
        mask = np.asarray(lags, dtype=float) <= float(max_lag)
    diff = a[mask] - b[mask]
    denom = float(np.linalg.norm(b[mask]))
    if denom == 0.0:
        raise ValueError("target correlation is identically zero in the window")
    l1 = float(np.max(np.abs(diff)))
    return {
        "l2_relative": float(np.linalg.norm(diff) / denom),
        "max_abs": l1,
        "n_lags": float(np.count_nonzero(mask)),
    }


def correlated_weibull_field(marginal: WeibullMarginal, kl: KlField1D,
                             n_samples: int, *, rng: Optional[np.random.Generator] = None,
                             seed: int = 20260101) -> np.ndarray:
    """``(n_samples, nx)`` field of Weibull strengths via the Gaussian copula.

    ``sigma = F^{-1}(Phi(Z))`` with ``Z`` from :meth:`KlField1D.sample`.  The
    marginal is exactly ``Weibull(k, sigma_0)`` at every grid point and the rank
    correlation is inherited from ``Z`` unchanged; what changes with ``L`` is
    *which* strengths sit next to each other, i.e. exactly the quantity the wide
    structure's breaking force averages over.
    """
    z = kl.sample(n_samples, rng=rng, seed=seed)
    u = gaussian_copula_uniforms(z)
    return marginal.ppf(u)


def spearman_target(kl: KlField1D) -> np.ndarray:
    """Analytic rank-correlation target of the copula field at every lag."""
    lags = kl.x[None, :] - kl.x[:, None]
    return spearman_from_gaussian_correlation(
        correlation_kernel(lags, kl.length, kl.model))