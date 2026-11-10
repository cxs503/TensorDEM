"""Daughter size distributions ``b(v, v')`` for particle breakage.

Two families are implemented, both *mass conserving by construction*
(``sum_i v_i = v'`` for every event, to machine precision):

``UniformDaughter``
    The classical uniform (in volume) binary daughter distribution

    .. math:: b(v, v') = 2/v', \\qquad 0 < v < v' ,

    i.e. the split fraction ``f = v/v'`` is ``U(0, 1)`` for both daughters
    (mean daughter count 2, so ``int v b dv = v'``).  This is the family for
    which the pure-breakage population balance has a closed-form solution
    (see :mod:`tensordem.breakage_pbm.dpbm`).

``LognormalDaughter``
    ``n_d`` daughters whose *weights* are lognormal: ``z_i ~ LogNormal(-s^2/2,
    s)`` (unit mean) and ``f_i = z_i / sum_j z_j``.  This is the standard
    "lognormal daughter distribution" of comminution modelling; the
    normalisation makes ``sum_i f_i = 1`` exact, which is what the population
    balance's mass conservation requires.

Both families expose :meth:`sample` (one event -> daughter volumes),
:meth:`fraction_pdf`/:meth:`fraction_cdf` where a closed form exists, and are
compared against DEM fragment statistics with the two-sample Kolmogorov-Smirnov
test (:func:`ks_two_sample`) and a parametric-bootstrap calibration of the
p-value (:func:`ks_two_sample_bootstrap_pvalue`) because the lognormal family
has parameters estimated from the very sample being tested.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "UniformDaughter",
    "LognormalDaughter",
    "daughter_volume_fractions",
    "ks_two_sample",
    "ks_two_sample_bootstrap_pvalue",
    "ks_uniform",
    "wasserstein_1d",
    "empirical_cdf",
    "daughter_mass_defect",
]


def empirical_cdf(sample, grid) -> np.ndarray:
    """Right-continuous empirical CDF of ``sample`` evaluated on ``grid``."""
    x = np.sort(np.asarray(sample, dtype=np.float64).ravel())
    if x.size == 0:
        raise ValueError("empty sample")
    return np.searchsorted(x, np.asarray(grid, dtype=np.float64), side="right") / x.size


def ks_two_sample(a, b) -> dict:
    """Two-sample Kolmogorov-Smirnov test (asymptotic p-value)."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size < 2 or y.size < 2:
        raise ValueError("need >= 2 samples on each side")
    result = stats.ks_2samp(x, y, alternative="two-sided", method="asymp")
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue),
            "n_a": int(x.size), "n_b": int(y.size)}


def ks_two_sample_bootstrap_pvalue(observed_statistic: float, reference_sample,
                                   n_model_samples: int, draw, *, n_boot: int = 200,
                                   seed: int = 0) -> float:
    """Parametric-bootstrap p-value of a two-sample KS statistic.

    ``draw`` is a callable ``draw(rng, n) -> sample`` from the *fitted* model.
    When model parameters are estimated on the data being tested, the plain
    two-sample p-value is optimistic; the bootstrap distribution of the
    statistic under the fitted model is the honest reference.
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(reference_sample, dtype=np.float64).ravel()
    stats_null = np.empty(int(n_boot))
    for k in range(int(n_boot)):
        model = np.asarray(draw(rng, data.size), dtype=np.float64)
        simulated = np.asarray(draw(rng, data.size), dtype=np.float64)
        stats_null[k] = stats.ks_2samp(model, simulated, method="asymp").statistic
    return float(np.mean(stats_null >= float(observed_statistic)))


def ks_uniform(sample) -> dict:
    """One-sample KS test against ``U(0, 1)`` (no fitted parameters)."""
    x = np.asarray(sample, dtype=np.float64).ravel()
    result = stats.kstest(x, "uniform")
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue),
            "n_a": int(x.size)}


def wasserstein_1d(a, b) -> float:
    """First Wasserstein distance between two empirical 1-D distributions."""
    return float(stats.wasserstein_distance(
        np.asarray(a, dtype=np.float64).ravel(),
        np.asarray(b, dtype=np.float64).ravel(),
    ))


def daughter_mass_defect(rule, parent_volume: float, rng, n_events: int = 200) -> float:
    """Maximum relative mass defect ``|sum v_i - v'| / v'`` over random events."""
    v = float(parent_volume)
    worst = 0.0
    for _ in range(int(n_events)):
        daughters = np.asarray(rule.sample(v, rng), dtype=np.float64)
        worst = max(worst, abs(float(np.sum(daughters)) - v) / v)
    return worst


@dataclass(frozen=True)
class UniformDaughter:
    """Uniform-in-volume daughter distribution, ``b(v, v') = 2/v'``."""

    name: str = "uniform"
    mean_daughters: float = 2.0
    n_parameters: int = 0

    def sample(self, parent_volume, rng: np.random.Generator) -> np.ndarray:
        v = float(parent_volume)
        if v <= 0.0:
            raise ValueError("parent volume must be positive")
        u = float(rng.random())
        return np.array([u * v, (1.0 - u) * v])

    @staticmethod
    def fraction_pdf(f):
        """Density of the pooled daughter volume fraction ``f = v/v'``."""
        f = np.asarray(f, dtype=np.float64)
        return np.where((f > 0.0) & (f < 1.0), 1.0, 0.0)

    @staticmethod
    def fraction_cdf(f):
        return np.clip(np.asarray(f, dtype=np.float64), 0.0, 1.0)

    def sample_fractions(self, rng: np.random.Generator, n_events: int = 1000) -> np.ndarray:
        u = rng.random(int(n_events))
        return np.concatenate([u, 1.0 - u])


@dataclass(frozen=True)
class LognormalDaughter:
    """Normalised-lognormal daughter weights with exact mass conservation.

    ``n_daughters`` fragments receive volumes ``f_i v'`` with
    ``f_i = z_i / sum_j z_j`` and ``z_i ~ LogNormal(-s^2/2, s)`` (``E[z] = 1``,
    so ``E[f_i] = 1/n_daughters``).  ``sigma_ln`` is the log-weight spread; the
    limit ``sigma_ln -> 0`` is equal-size breakage and ``sigma_ln -> large``
    approaches the uniform family.
    """

    sigma_ln: float = 1.0
    n_daughters: int = 3
    name: str = "lognormal"
    n_parameters: int = 1

    def __post_init__(self) -> None:
        if self.sigma_ln < 0.0:
            raise ValueError("sigma_ln must be non-negative")
        if int(self.n_daughters) < 2:
            raise ValueError("n_daughters must be >= 2")

    @property
    def mean_daughters(self) -> float:
        return float(self.n_daughters)

    def sample_fractions(self, rng: np.random.Generator, n_events: int = 1000) -> np.ndarray:
        k = int(self.n_daughters)
        n = int(n_events)
        z = rng.lognormal(mean=-0.5 * self.sigma_ln ** 2, sigma=self.sigma_ln,
                          size=(n, k))
        f = z / z.sum(axis=1, keepdims=True)
        return f.ravel()

    def sample(self, parent_volume, rng: np.random.Generator) -> np.ndarray:
        v = float(parent_volume)
        if v <= 0.0:
            raise ValueError("parent volume must be positive")
        k = int(self.n_daughters)
        z = rng.lognormal(mean=-0.5 * self.sigma_ln ** 2, sigma=self.sigma_ln, size=k)
        f = z / z.sum()
        return f * v

    @classmethod
    def fit_fractions(cls, fractions, n_daughters: int = 3) -> "LognormalDaughter":
        """Moment fit of ``sigma_ln`` to observed daughter fractions.

        For the normalised-lognormal family the coefficient of variation of the
        weights is ``sqrt(exp(s^2) - 1)``; the observed fractions are rescaled
        to unit mean per event before matching, which makes the estimator a
        plain method-of-moments inversion (no fitting to the PBM target).
        """
        f = np.asarray(fractions, dtype=np.float64).ravel()
        f = f[f > 0.0]
        if f.size < 3:
            raise ValueError("need >= 3 positive fractions")
        z = f * float(n_daughters)
        cv = float(np.std(z, ddof=1) / np.mean(z))
        sigma = float(np.sqrt(np.log1p(cv * cv)))
        return cls(sigma_ln=max(sigma, 1e-3), n_daughters=int(n_daughters))


def daughter_volume_fractions(fragment_volumes, parent_volume) -> np.ndarray:
    """Fractions ``v_i / v'`` of one breakage event."""
    v = np.asarray(fragment_volumes, dtype=np.float64).ravel()
    parent = float(parent_volume)
    if parent <= 0.0:
        raise ValueError("parent volume must be positive")
    return v / parent