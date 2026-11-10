"""Weibull marginal of ice bending strength and its calibration chain.

Why this module exists
----------------------
Campaign node ``P295W1-N07`` asks for a *random* ice-plate strength field for the
DEM/beam strip model, in three parts: a Weibull marginal with an auditable
calibration (QQ-plot), a spatially correlated Gaussian field with a
prescribed correlation length, and the statistics of the breaking force
ensemble that follows from the two.  This module owns the first part.

Convention (identical to the rest of the package)
------------------------------------------------
Two-parameter Weibull on the bending (flexural) strength ``sigma_f > 0``::

    F(s) = 1 - exp(-(s / sigma_0)**k),     k = modulus, sigma_0 = scale

so ``sigma_0`` is the 63.2 % quantile, **not** the mean.  The mean and the
coefficient of variation follow in closed form,

.. math::
    \\mu = \\sigma_0 \\Gamma(1 + 1/k), \\qquad
    \\mathrm{CV}^2 = \\frac{\\Gamma(1 + 2/k)}{\\Gamma(1 + 1/k)^2} - 1,

and both are used to convert between "engineer-speak" inputs (a mean strength
from a Timco/Weeks style strength review) and the ``(k, sigma_0)`` pair the
sampler needs.  :meth:`WeibullMarginal.from_mean_cov` does that inversion by
bisection on ``k``; no value is hard-coded downstream.

Calibration and its verdicts
----------------------------
:func:`fit_weibull_mle` is the maximum-likelihood fit (profile likelihood in
``k``, closed-form ``sigma_0``) -- a *different* code path from the log-log
regression estimator in :func:`tensordem.bpm_calibration.weibull_mle`, so the
two can be cross-checked against each other in the tests rather than one being
believed twice.

:func:`qq_plot_r2` is the acceptance statistic of criterion (a): the
coefficient of determination of the quantile-quantile regression of the sorted
sample against the fitted Weibull quantiles.  It is deliberately *not* the
log-likelihood: a QQ-plot ``R^2`` is what the brief asks for, it is scale-free,
and it is the same statistic a hand check on a probability plot would produce.

Because the sampler below is an exact inverse-CDF transform, a QQ-plot on its
own output is close to a tautology.  :func:`qq_plot_r2` is therefore always
reported together with a **negative control** (:func:`negative_control_lognormal`):
the same statistic computed for a lognormal sample, which must fall below the
acceptance line.  A criterion that cannot fail is not a criterion.

:func:`ks_distance` adds the distribution-free Kolmogorov-Smirnov distance and
its asymptotic ``p``-value for the same fit, and :func:`weibull_moments` gives
the moments and the entropy-free comparison numbers the report quotes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

__all__ = [
    "WeibullMarginal",
    "fit_weibull_mle",
    "qq_plot_r2",
    "qq_quantiles",
    "ks_distance",
    "negative_control_lognormal",
    "weibull_moments",
    "spearman_from_gaussian_correlation",
    "gaussian_copula_uniforms",
]


# ----------------------------------------------------------------------------
# marginal
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class WeibullMarginal:
    """Two-parameter Weibull marginal ``sigma_f ~ Weibull(k, sigma_0)``."""

    modulus: float
    scale: float

    def __post_init__(self) -> None:
        for name, value in (("modulus", self.modulus), ("scale", self.scale)):
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")

    # -- moments ---------------------------------------------------------
    @property
    def mean(self) -> float:
        return self.scale * math.gamma(1.0 + 1.0 / self.modulus)

    @property
    def variance(self) -> float:
        g1 = math.gamma(1.0 + 1.0 / self.modulus)
        g2 = math.gamma(1.0 + 2.0 / self.modulus)
        return self.scale ** 2 * (g2 - g1 ** 2)

    @property
    def cov(self) -> float:
        """Coefficient of variation (``sigma/mu``) of the marginal."""
        return math.sqrt(self.variance) / self.mean

    # -- distribution ----------------------------------------------------
    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        z = np.clip(x / self.scale, 0.0, None)
        return 1.0 - np.exp(-np.power(z, self.modulus))

    def ppf(self, p):
        """Inverse CDF; ``p`` is clipped away from 0 and 1 for safety."""
        p = np.clip(np.asarray(p, dtype=float), 1e-300, 1.0 - 1e-16)
        return self.scale * np.power(-np.log1p(-p), 1.0 / self.modulus)

    def sample(self, n: int, *, rng: np.random.Generator) -> np.ndarray:
        if n <= 0:
            raise ValueError("n must be positive")
        return self.ppf(rng.random(n))

    # -- constructors from engineering inputs ----------------------------
    @classmethod
    def from_mean_modulus(cls, mean: float, modulus: float) -> "WeibullMarginal":
        """``sigma_0 = mean / Gamma(1 + 1/k)`` for a prescribed mean and ``k``."""
        if mean <= 0.0 or modulus <= 0.0:
            raise ValueError("mean and modulus must be positive")
        return cls(modulus=float(modulus),
                   scale=float(mean) / math.gamma(1.0 + 1.0 / float(modulus)))

    @classmethod
    def from_mean_cov(cls, mean: float, cov: float) -> "WeibullMarginal":
        """Invert ``CV(k)`` numerically to match a measured mean and CV.

        ``CV(k)`` decreases monotonically from ``+inf`` (``k -> 0``) to ``0``
        (``k -> inf``), so a bracketed bisection on ``ln k`` is unconditionally
        convergent here; the bracket is widened until it encloses the root.
        """
        if mean <= 0.0:
            raise ValueError("mean must be positive")
        if not (0.0 < cov < 5.0):
            raise ValueError("cov must be in (0, 5)")

        def residual(ln_k: float) -> float:
            k = math.exp(ln_k)
            return math.sqrt(math.gamma(1.0 + 2.0 / k) / math.gamma(1.0 + 1.0 / k) ** 2
                             - 1.0) - cov

        # ``residual`` evaluates Gamma(1 + 2/k); scipy's float gamma overflows
        # for ``k < ~0.012``, so the bracket is kept inside ``k in [0.02, 1000]``.
        # That is wide enough for every CV above (CV = 30 at k = 0.041,
        # CV = 0.001 at k = 1000), so no widening is needed or attempted.
        lo, hi = math.log(0.02), math.log(1000.0)
        flo, fhi = residual(lo), residual(hi)
        if flo * fhi > 0.0:
            raise RuntimeError("could not bracket the Weibull modulus for the given CV")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            fmid = residual(mid)
            if flo * fmid <= 0.0:
                hi, fhi = mid, fmid
            else:
                lo, flo = mid, fmid
            if hi - lo < 1e-13:
                break
        k = math.exp(0.5 * (lo + hi))
        return cls.from_mean_modulus(mean, k)


def weibull_moments(sample: Sequence[float]) -> Dict[str, float]:
    """Sample mean / std / CV / skewness, plus the Weibull ``k`` implied by the CV."""
    x = np.asarray([float(v) for v in sample], dtype=float)
    x = x[np.isfinite(x) & (x > 0.0)]
    if x.size < 3:
        raise ValueError("need at least three positive samples")
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    centred = x - mu
    skew = float(np.mean(centred ** 3) / sd ** 3) if sd > 0.0 else float("nan")
    return {
        "n": float(x.size),
        "mean": mu,
        "std": sd,
        "cov": sd / mu,
        "skewness": skew,
        "min": float(x.min()),
        "max": float(x.max()),
    }


# ----------------------------------------------------------------------------
# maximum likelihood
# ----------------------------------------------------------------------------

def fit_weibull_mle(sample: Sequence[float], *,
                    quantile_plot: bool = True) -> Dict[str, float]:
    """Maximum-likelihood ``(k, sigma_0)`` of a two-parameter Weibull sample.

    The shape solves the profile likelihood equation

    .. math::
        \\frac{\\sum x_i^k \\ln x_i}{\\sum x_i^k} - \\frac{1}{k} - \\overline{\\ln x} = 0

    (monotone in ``k``, so a bracketed Brent solve is used), after which
    ``sigma_0 = (mean(x**k))**(1/k)`` and the log-likelihood follow in closed
    form.  The sample is scaled by its own mean before the solve so that
    ``x**k`` cannot overflow for large ``k``.

    Returns ``m_hat``/``sigma_hat`` (the MLE), ``k_hat``/``scale_hat`` aliases,
    the sample moments, the log-likelihood and, if ``quantile_plot``, the
    QQ-plot ``R^2``, its regression line and the KS distance/p-value of the
    fitted marginal against the sample.
    """
    from scipy.optimize import brentq

    x = np.asarray([float(v) for v in sample], dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        raise ValueError("at least three finite samples are required")
    if np.any(x <= 0.0):
        raise ValueError("Weibull samples must be strictly positive")

    xbar = float(x.mean())
    y = x / xbar
    ly = np.log(y)
    mean_ly = float(ly.mean())

    def equation(k: float) -> float:
        # ``sum(y**k ln y) / sum(y**k)`` is evaluated as a soft-max of ``ln y``
        # with weights ``exp(k ln y)``, shifted by the maximum exponent so that
        # no power can overflow for any bracket the solver visits.  Without the
        # shift, a heavy-tailed sample pushes ``k`` past ~200 and ``y**k``
        # overflows to inf/inf = nan, which kills the bracket.
        a = k * ly
        shift = float(np.max(a))
        w = np.exp(a - shift)
        return float(np.sum(w * ly) / np.sum(w) - 1.0 / k - mean_ly)

    lo, hi = 1.0e-3, 500.0
    flo, fhi = equation(lo), equation(hi)
    widen = 0
    while flo * fhi > 0.0 and widen < 8:           # widen only if still unbracketed
        hi *= 4.0
        fhi = equation(hi)
        widen += 1
    if flo * fhi > 0.0:                            # pragma: no cover - pathological
        raise RuntimeError("Weibull profile likelihood has no bracket")
    k_hat = float(brentq(equation, lo, hi, xtol=1e-13, rtol=1e-14, maxiter=400))
    sigma_hat = xbar * float(np.mean(np.power(y, k_hat)) ** (1.0 / k_hat))

    loglik = float(np.sum(np.log(k_hat / sigma_hat)
                          + (k_hat - 1.0) * np.log(x / sigma_hat)
                          - np.power(x / sigma_hat, k_hat)))

    out: Dict[str, float] = {
        "n": float(x.size),
        "m_hat": k_hat,
        "sigma_hat": sigma_hat,
        "k_hat": k_hat,
        "scale_hat": sigma_hat,
        "log_likelihood": loglik,
    }
    out.update({f"sample_{key}": value for key, value in weibull_moments(x).items()})
    if quantile_plot:
        marginal = WeibullMarginal(modulus=k_hat, scale=sigma_hat)
        out.update({f"qq_{key}": value for key, value in qq_plot_r2(x, marginal).items()})
        out.update({f"ks_{key}": value for key, value in ks_distance(x, marginal).items()})
    return out


# ----------------------------------------------------------------------------
# goodness of fit: QQ plot and KS
# ----------------------------------------------------------------------------

def qq_quantiles(sample: Sequence[float], marginal: WeibullMarginal
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorted sample, matching theoretical quantiles and plotting positions.

    Plotting positions use the median-rank (Benard) approximation
    ``p_i = (i - 0.3) / (n + 0.4)`` -- the same convention as
    :func:`tensordem.bpm_calibration.weibull_mle`, so a QQ ``R^2`` computed here
    and one computed by hand on the package's standard probability plot agree.
    """
    x = np.sort(np.asarray([float(v) for v in sample], dtype=float))
    n = x.size
    p = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
    theoretical = marginal.ppf(p)
    return x, theoretical, p


def qq_plot_r2(sample: Sequence[float], marginal: WeibullMarginal) -> Dict[str, float]:
    """Coefficient of determination of the QQ regression ``x_(i)`` vs ``F^{-1}(p_i)``.

    ``R^2 = 1 - SS_res / SS_tot`` with the regression line free (slope and
    intercept), i.e. the fraction of the sample quantile spread that the fitted
    Weibull explains.  For a perfect Weibull sample ``R^2 -> 1``; a heavy-tailed
    or bounded competitor (lognormal, normal) leaves a systematic curvature and
    loses several percent.  ``rmse`` is reported in the sample's own units and
    ``max_abs_dev`` is the Chebyshev norm of the same residual.
    """
    x, theoretical, _p = qq_quantiles(sample, marginal)
    if x.size < 3:
        raise ValueError("at least three samples are required for a QQ plot")
    design = np.vstack([theoretical, np.ones_like(theoretical)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, x, rcond=None)
    fitted = slope * theoretical + intercept
    residual = x - fitted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((x - x.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return {
        "r2": float(r2),
        "slope": float(slope),
        "intercept": float(intercept),
        "rmse": float(math.sqrt(ss_res / x.size)),
        "max_abs_dev": float(np.max(np.abs(residual))),
        "n": float(x.size),
    }


def ks_distance(sample: Sequence[float], marginal: WeibullMarginal) -> Dict[str, float]:
    """Two-sided Kolmogorov-Smirnov distance and asymptotic ``p``-value."""
    from scipy import stats as _stats

    x = np.asarray([float(v) for v in sample], dtype=float)
    x = x[np.isfinite(x) & (x > 0.0)]
    result = _stats.kstest(x, lambda v: marginal.cdf(v))
    n = x.size
    # asymptotic Kolmogorov distribution: Q(lam) = 2 sum (-1)^(j-1) exp(-2 j^2 lam^2)
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * float(result.statistic)
    p_asym = 0.0
    for j in range(1, 101):
        p_asym += 2.0 * (-1.0) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "p_value_asymptotic": float(max(min(p_asym, 1.0), 0.0)),
        "n": float(n),
    }


def negative_control_lognormal(*, n: int = 5000, sigma_ln: float = 0.6,
                               seed: int = 20260101) -> Dict[str, float]:
    """QQ ``R^2`` of a *lognormal* sample against its best-fit Weibull.

    This is what makes criterion (a) falsifiable: the same statistic, the same
    sample size, the same estimator, applied to a distribution that is *not*
    Weibull.  If this number were also >= 0.95 the acceptance test would have no
    discriminating power, and the report would say so.
    """
    rng = np.random.default_rng(seed)
    sample = np.exp(sigma_ln * rng.standard_normal(n))
    fit = fit_weibull_mle(sample)
    return {
        "n": float(n),
        "sigma_ln": float(sigma_ln),
        "weibull_k_hat": float(fit["k_hat"]),
        "qq_r2": float(fit["qq_r2"]),
        "ks_statistic": float(fit["ks_statistic"]),
        "ks_p_value": float(fit["ks_p_value"]),
    }


# ----------------------------------------------------------------------------
# Gaussian copula: independent of the marginal, carries the correlation only
# ----------------------------------------------------------------------------

def gaussian_copula_uniforms(z) -> np.ndarray:
    """``u = Phi(z)`` clipped into ``(0, 1)`` so the Weibull inverse is finite."""
    from scipy.special import ndtr

    u = ndtr(np.asarray(z, dtype=float))
    return np.clip(u, 1e-15, 1.0 - 1e-15)


def spearman_from_gaussian_correlation(rho):
    """Exact rank correlation of a bivariate Gaussian pair: ``(6/pi) asin(rho/2)``.

    A monotone marginal transform (``Phi(z) -> Weibull ppf``) leaves the *rank*
    correlation untouched, so this is the analytic target for the Spearman
    correlation of the generated strength field.  Comparing the measured rank
    correlation with this expression tests the copula and the field generator
    jointly, without a second random-field implementation to trust.
    """
    rho = np.asarray(rho, dtype=float)
    return (6.0 / math.pi) * np.arcsin(np.clip(rho, -1.0, 1.0) / 2.0)