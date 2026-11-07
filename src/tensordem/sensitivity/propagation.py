"""Uncertainty propagation: Laplace posterior -> predictive band -> coverage.

The calibration data are a *single* measurement vector (one calibration trace),
so the posterior is obtained with the standard linearised (Laplace) update of a
Gaussian prior under a Gaussian likelihood:

``Sigma_post^-1 = J^T Sigma^-1 J + Gamma_prior^-1``
``u_post        = u0 + Sigma_post (J^T Sigma^-1 r - Gamma_prior^-1 (u0 - u_prior))``

where ``r = y - f(u0)`` is the residual at the linearisation point.  Posterior
draws are mapped through the (validated) surrogate to a resistance distribution;
adding the declared observation noise gives the posterior-predictive band whose
empirical coverage is tested against the L2 anchor corpus.

Coverage is reported with a Wilson score interval so that a small anchor corpus
(``M`` points) cannot be over-interpreted: with ``M = 40`` the 95 % Wilson
interval is roughly +/-7 percentage points.
"""

from __future__ import annotations

import math

import numpy as np

from .identifiability import fisher_information

__all__ = [
    "laplace_posterior",
    "sample_gaussian",
    "posterior_predictive",
    "wilson_interval",
    "coverage_test",
]


def laplace_posterior(J, residual, sigma, u0, *, prior_mean=None,
                      prior_sd: float = 0.25) -> dict:
    """Linearised Gaussian posterior in unit-cube coordinates."""
    J = np.asarray(J, dtype=float)
    residual = np.asarray(residual, dtype=float).ravel()
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), residual.shape)
    if residual.size != J.shape[0]:
        raise ValueError("residual length must match the Jacobian rows")
    if prior_sd <= 0.0:
        raise ValueError("prior_sd must be positive")
    u0 = np.asarray(u0, dtype=float).ravel()
    prior_mean = np.full_like(u0, 0.5) if prior_mean is None else np.asarray(prior_mean, dtype=float)
    fim = fisher_information(J, sigma)
    prior_precision = np.eye(u0.size) / prior_sd**2
    post_precision = fim + prior_precision
    rhs = J.T @ (residual / sigma**2) - prior_precision @ (u0 - prior_mean)
    covariance = np.linalg.inv(post_precision)
    delta = covariance @ rhs
    mean = u0 + delta
    return {
        "mean": mean,
        "covariance": covariance,
        "fim": fim,
        "posterior_std": np.sqrt(np.clip(np.diag(covariance), 0.0, None)),
        "prior_sd": float(prior_sd),
        "min_eigenvalue": float(np.linalg.eigvalsh(post_precision)[0]),
    }


def sample_gaussian(mean, covariance, n: int, *, seed: int = 0,
                    lower: float = 0.0, upper: float = 1.0,
                    max_rounds: int = 8) -> np.ndarray:
    """Rejection-truncated Gaussian draws kept inside the parameter box."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    mean = np.asarray(mean, dtype=float).ravel()
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (mean.size, mean.size):
        raise ValueError("covariance shape does not match the mean")
    # Symmetrise and jitter to keep the Cholesky factorisation well posed.
    covariance = 0.5 * (covariance + covariance.T)
    jitter = 1e-12 * max(float(np.trace(covariance)), 1e-30)
    rng = np.random.default_rng(seed)
    accepted = []
    total = 0
    for _ in range(max_rounds):
        draws = rng.multivariate_normal(mean, covariance + jitter * np.eye(mean.size),
                                        size=n, method="cholesky")
        inside = np.all((draws >= lower) & (draws <= upper), axis=1)
        accepted.append(draws[inside])
        total += int(inside.sum())
        if total >= n:
            break
    if total == 0:
        raise RuntimeError("all posterior draws fell outside the parameter box")
    pooled = np.vstack(accepted)
    if len(pooled) < n:
        # Recycle accepted draws rather than biasing towards the mode.
        pooled = np.vstack([pooled, pooled[: n - len(pooled)]])
    return pooled[:n]


def posterior_predictive(model, samples, *, sigma_obs, level: float = 0.95) -> dict:
    """Posterior-predictive summary of a scalar response.

    ``sigma_obs`` is folded in as an additive Gaussian observation error on top
    of the surrogate-induced spread, so the returned interval is a *predictive*
    (not a credible) interval.
    """
    values = np.asarray(model(samples), dtype=float).ravel()
    if values.size != len(samples):
        raise ValueError("model must return one value per posterior sample")
    if sigma_obs < 0.0 or not math.isfinite(sigma_obs):
        raise ValueError("sigma_obs must be finite and nonnegative")
    mean = float(values.mean())
    sd_model = float(values.std(ddof=1))
    sd_pred = math.sqrt(sd_model**2 + sigma_obs**2)
    alpha = 0.5 * (1.0 - level)
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    z = 1.959963984540054 if abs(level - 0.95) < 1e-6 else float(
        -np.sqrt(2.0) * _erfinv(level - 1.0))
    return {
        "mean": mean,
        "sd_model": sd_model,
        "sd_predictive": sd_pred,
        "u95_half_width": float(z * sd_pred),
        "quantile_interval": [float(lower), float(upper)],
        "level": level,
        "n_samples": int(len(samples)),
        "samples": values,
    }


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must lie in [0, trials]")
    p = successes / trials
    denom = 1.0 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def coverage_test(truth, lower, upper) -> dict:
    """Empirical coverage of the interval ``[lower, upper]`` against ``truth``."""
    truth = np.asarray(truth, dtype=float).ravel()
    lower = np.broadcast_to(np.asarray(lower, dtype=float), truth.shape)
    upper = np.broadcast_to(np.asarray(upper, dtype=float), truth.shape)
    if np.any(upper < lower):
        raise ValueError("upper interval bound must not be below lower")
    inside = (truth >= lower) & (truth <= upper)
    k, n = int(inside.sum()), int(truth.size)
    lo, hi = wilson_interval(k, n)
    return {
        "coverage": k / n,
        "covered": k,
        "trials": n,
        "wilson95": [lo, hi],
        "expected": 0.95,
        "inside_nominal": bool(lo <= 0.95 <= hi),
        "misses_low": int(np.sum(truth < lower)),
        "misses_high": int(np.sum(truth > upper)),
        "inside": inside.tolist(),
    }


def _erfinv(x: float) -> float:
    """Inverse error function (Newton on a robust initial guess)."""
    from scipy.special import erfinv as _scipy_erfinv

    return float(_scipy_erfinv(x))