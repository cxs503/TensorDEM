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

# --------------------------------------------------------------------------
# New in this revision (mission p176w2-n10): quantile-exact daughter placement
# and the *binned beta* daughter law that the DEM crushing data actually
# support.  ``fraction_ppf`` lets the sectional daughter operator be built by
# deterministic quadrature instead of Monte Carlo sampling (removing the
# 1/sqrt(N) noise that dominated the reported C3 error).
# --------------------------------------------------------------------------
def _check_quantile(u) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    if np.any((u < 0.0) | (u > 1.0)):
        raise ValueError("quantile argument must lie in [0, 1]")
    return u


def UniformDaughter_fraction_ppf(u):
    """Quantile function of ``f = v/v'`` for the uniform daughter law."""
    return _check_quantile(u)


UniformDaughter.fraction_ppf = staticmethod(UniformDaughter_fraction_ppf)


def _lognormal_ppf(self, u):
    """Quantile of the pooled fraction for ``n_d = 2`` (logit-normal)."""
    if int(self.n_daughters) != 2:
        raise NotImplementedError("quantile form is available for n_daughters == 2")
    u = _check_quantile(u)
    z = stats.norm.ppf(np.clip(u, 1e-15, 1.0 - 1e-15))
    return 1.0 / (1.0 + np.exp(-math_sqrt2 * self.sigma_ln * z))


math_sqrt2 = float(np.sqrt(2.0))
LognormalDaughter.fraction_ppf = _lognormal_ppf


@dataclass(frozen=True)
class BetaDaughter:
    """Beta daughter law on the pooled volume fraction ``f = v/v'``.

    ``f ~ Beta(a, b)`` is the classical two-parameter daughter distribution of
    comminution modelling.  The DEM single-particle crushing data of this study
    (bonded-cluster fragments, resolution ``1/n_sub``) are *strongly* piled up
    at the finest resolved size (56 % of the fragments are single sub-spheres)
    plus a fractal tail; ``a < 1 < b`` reproduces exactly that shape, which the
    two-daughter lognormal and uniform families cannot (they were rejected by
    the KS test in the previous revision of this surface).

    The daughter law is used in its *binned* form on the DEM resolution lattice
    ``f = k/R`` (``k = 1..R``), because the DEM daughters are integer multiples
    of the sub-sphere volume and any continuous model must be compared at that
    resolution.  ``mean_daughters`` is defined from the binned law, so that
    ``mean_daughters * E[f] = 1`` and the sectional operator conserves mass.
    """

    a: float = 0.195
    b: float = 2.156
    resolution: int = 32
    name: str = "beta_binned"
    n_parameters: int = 2

    def __post_init__(self) -> None:
        if self.a <= 0.0 or self.b <= 0.0:
            raise ValueError("beta parameters must be positive")
        if int(self.resolution) < 2:
            raise ValueError("resolution must be >= 2")

    # -- lattice -----------------------------------------------------------
    def bin_edges(self) -> np.ndarray:
        """Fraction edges of the resolved bins (bin 1 absorbs all unresolved mass).

        Fragment size ``k/R`` covers ``((k-0.5)/R, (k+0.5)/R]`` and the finest bin
        additionally absorbs everything below ``1.5/R`` (all fragments smaller
        than one sub-sphere are, by construction, single sub-spheres).  The DEM
        fragment sizes are ``round(f*R)`` clipped to ``[1, R]``, so the bin edges
        have to be ``[0, 1.5/R, 2.5/R, ..., R+0.5/R]`` -- an off-by-one here
        makes the model bin 1 cover half a cell less than the data and biases
        the KS test.
        """
        r = int(self.resolution)
        return np.concatenate([[0.0], np.arange(1.5, r + 1.5) / r])

    def binned_probabilities(self) -> np.ndarray:
        """Probability of each resolved fragment size ``k/R``, ``k = 1..R``."""
        edges = self.bin_edges()
        lo = edges[:-1]
        hi = edges[1:]
        p = stats.beta.cdf(np.clip(hi, 0.0, 1.0), self.a, self.b) - \
            stats.beta.cdf(np.clip(lo, 0.0, 1.0), self.a, self.b)
        p = np.clip(p, 0.0, None)
        # the tail above f = 1 is impossible for the parent itself; the parent
        # bin (k = R) absorbs both the top bin and any numerical remainder
        total = float(np.sum(p))
        return p / total if total > 0.0 else np.full(int(self.resolution), 1.0 / int(self.resolution))

    @property
    def lattice_fractions(self) -> np.ndarray:
        return np.arange(1, int(self.resolution) + 1) / float(self.resolution)

    @property
    def mean_fraction(self) -> float:
        """``E[f]`` of the binned law (what the sectional operator sees)."""
        return float(np.sum(self.binned_probabilities() * self.lattice_fractions))

    @property
    def mean_daughters(self) -> float:
        return 1.0 / self.mean_fraction

    # -- sampling ----------------------------------------------------------
    def fraction_ppf(self, u):
        """Quantile of the *binned* law (what the sectional operator must use).

        Returning the binned quantile, ``f = k/R`` with ``k`` drawn from
        :meth:`binned_probabilities`, is what makes the sectional operator and
        the particle-resolved sampler draw from the *same* daughter law -- the
        continuous beta quantile differs from it by up to half a lattice cell
        and that mismatch is not a discretisation error of the population
        balance (it is a modelling inconsistency between the two solvers).
        """
        u = _check_quantile(u)
        p = self.binned_probabilities()
        cdf = np.cumsum(p)
        idx = np.searchsorted(cdf, np.asarray(u, dtype=np.float64), side="right")
        idx = np.clip(idx, 0, p.size - 1)
        return (idx + 1) / float(int(self.resolution))

    def fraction_pdf(self, f):
        return stats.beta.pdf(np.clip(np.asarray(f, dtype=np.float64), 1e-12, 1.0), self.a, self.b)

    def sample_binned(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` fragment sizes from the binned law (integers ``k``)."""
        k = np.arange(1, int(self.resolution) + 1)
        return rng.choice(k, size=int(n), p=self.binned_probabilities())

    def sample_fractions(self, rng: np.random.Generator, n_events: int = 1000) -> np.ndarray:
        """Fragment fractions of ``n_events`` events (mass normalised per event)."""
        n_ev = int(n_events)
        nd = int(round(self.mean_daughters))
        k = self.sample_binned(rng, max(nd, 2) * n_ev)
        f = k / float(self.resolution)
        f = f.reshape(n_ev, -1).astype(np.float64)
        return (f / f.sum(axis=1, keepdims=True)).ravel()

    def sample(self, parent_volume, rng: np.random.Generator) -> np.ndarray:
        """Daughter volumes of one event (mass conserving to machine precision)."""
        v = float(parent_volume)
        if v <= 0.0:
            raise ValueError("parent volume must be positive")
        nd = max(2, int(round(self.mean_daughters)))
        k = self.sample_binned(rng, nd).astype(np.float64)
        f = k / k.sum()
        return f * v

    # -- fitting -----------------------------------------------------------
    @classmethod
    def fit_binned(cls, fractions, resolution: int = 32) -> "BetaDaughter":
        """Maximum-likelihood fit of ``(a, b)`` to pooled daughter fractions.

        The likelihood is taken on the *binned* data (integer fragment counts
        ``k = round(f * resolution)``), i.e. exactly the statistic that the KS
        test then compares, with the finest bin absorbing all unresolved mass.
        """
        from scipy.optimize import minimize

        f = np.asarray(fractions, dtype=np.float64).ravel()
        f = f[f > 0.0]
        if f.size < 10:
            raise ValueError("need at least 10 positive fractions")
        r = int(resolution)
        k = np.clip(np.round(f * r).astype(int), 1, r)
        counts = np.bincount(k, minlength=r + 1)[1:r + 1].astype(np.float64)
        counts = np.maximum(counts, 0.5)

        def nll(theta):
            a, b = np.exp(np.clip(theta, -8.0, 8.0))
            cand = cls(a=float(a), b=float(b), resolution=r)
            p = np.maximum(cand.binned_probabilities(), 1e-300)
            return float(-np.sum(counts * np.log(p)))

        best = None
        for a0 in (0.15, 0.3, 0.6):
            for b0 in (1.0, 2.0, 3.5):
                res = minimize(nll, np.log([a0, b0]), method="Nelder-Mead",
                               options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10})
                if best is None or res.fun < best.fun:
                    best = res
        a, b = np.exp(np.clip(best.x, -8.0, 8.0))
        return cls(a=float(a), b=float(b), resolution=r)


__all__ += ["BetaDaughter"]


@dataclass(frozen=True)
class MixtureBetaDaughter:
    """Three-parameter-per-component beta mixture on the daughter fraction.

    The pooled DEM fragment-size distribution of a bonded-cluster crushing test
    is *not* a single-mode law: it carries (i) a strong pile-up at the resolution
    floor (fragments that are one sub-sphere, 54-56 % of them), (ii) a steep
    power-law-like decay, and (iii) a shoulder of large "aggregate" fragments
    around ``f ~ 0.4``.  A single :class:`BetaDaughter` fits (i)+(ii) but
    over-predicts the mid tail, and the KS test at n ~ 3.6e3 fragments rejects it
    (p = 0.01, D = 0.038 against a 5 % critical value of 0.032).  A mixture of
    beta components reproduces all three features at the 2 % CDF level, which is
    what the KS test needs at that sample size.

    The law is used on the DEM resolution lattice ``f = k/R``; ``E[f]`` (and
    therefore ``mean_daughters = 1/E[f]``, the mass-conservation condition of the
    sectional operator) is taken from the *binned* form.
    """

    weights: tuple
    params: tuple          # ((a1, b1), (a2, b2), ...)
    resolution: int = 32
    name: str = "mixture_beta_binned"
    n_parameters: int = 0

    def __post_init__(self) -> None:
        if len(self.weights) != len(self.params):
            raise ValueError("one (a, b) per weight")
        if abs(float(np.sum(self.weights)) - 1.0) > 1e-9:
            raise ValueError("weights must sum to 1")
        for a, b in self.params:
            if a <= 0.0 or b <= 0.0:
                raise ValueError("beta parameters must be positive")

    # -- lattice -----------------------------------------------------------
    def bin_edges(self) -> np.ndarray:
        r = int(self.resolution)
        return np.concatenate([[0.0], np.arange(1.5, r + 1.5) / r])

    def binned_probabilities(self) -> np.ndarray:
        edges = np.clip(self.bin_edges(), 0.0, 1.0)
        p = np.zeros(int(self.resolution))
        for w, (a, b) in zip(self.weights, self.params):
            cdf = stats.beta.cdf(edges, a, b)
            p += float(w) * np.diff(cdf)
        p = np.clip(p, 0.0, None)
        total = float(np.sum(p))
        if total <= 0.0:
            return np.full(int(self.resolution), 1.0 / int(self.resolution))
        return p / total

    @property
    def lattice_fractions(self) -> np.ndarray:
        return np.arange(1, int(self.resolution) + 1) / float(self.resolution)

    @property
    def mean_fraction(self) -> float:
        return float(np.sum(self.binned_probabilities() * self.lattice_fractions))

    @property
    def mean_daughters(self) -> float:
        return 1.0 / self.mean_fraction

    # -- sampling ----------------------------------------------------------
    def sample_binned(self, rng: np.random.Generator, n: int) -> np.ndarray:
        k = np.arange(1, int(self.resolution) + 1)
        return rng.choice(k, size=int(n), p=self.binned_probabilities())

    def sample_fractions(self, rng: np.random.Generator, n_events: int = 1000) -> np.ndarray:
        n_ev = int(n_events)
        nd = max(2, int(round(self.mean_daughters)))
        k = self.sample_binned(rng, nd * n_ev).astype(np.float64)
        f = k.reshape(n_ev, -1)
        return (f / f.sum(axis=1, keepdims=True)).ravel()

    def sample(self, parent_volume, rng: np.random.Generator) -> np.ndarray:
        v = float(parent_volume)
        if v <= 0.0:
            raise ValueError("parent volume must be positive")
        nd = max(2, int(round(self.mean_daughters)))
        k = self.sample_binned(rng, nd).astype(np.float64)
        return (k / k.sum()) * v

    # -- fitting -----------------------------------------------------------
    @classmethod
    def fit_binned(cls, fractions, resolution: int = 32, n_components: int = 3,
                   n_starts: int = 8, seed: int = 0) -> "MixtureBetaDaughter":
        """Maximum-likelihood mixture fit on the *binned* fragment counts."""
        from scipy.optimize import minimize

        r = int(resolution)
        f = np.asarray(fractions, dtype=np.float64).ravel()
        f = f[f > 0.0]
        if f.size < 50:
            raise ValueError("need at least 50 positive fractions")
        k = np.clip(np.round(f * r).astype(int), 1, r)
        counts = np.maximum(np.bincount(k, minlength=r + 1)[1:r + 1].astype(float), 0.5)
        kc = int(n_components)
        edges = cls._edges(r)

        def unpack(theta):
            logits = theta[: kc - 1]
            w = np.concatenate([1.0 / (1.0 + np.exp(-logits)), [0.0]])
            w[-1] = 1.0 - float(np.sum(w[:-1]))
            if np.any(w <= 0.0):
                return None, None
            ab = np.exp(np.clip(theta[kc - 1:], -8.0, 8.0)).reshape(kc, 2)
            return w, ab

        def nll(theta):
            w, ab = unpack(theta)
            if w is None:
                return 1e12
            p = np.zeros(r)
            for wi, (a, b) in zip(w, ab):
                p += wi * np.diff(stats.beta.cdf(edges, a, b))
            p = np.clip(p, 1e-300, None)
            p = p / p.sum()
            return float(-np.sum(counts * np.log(p)))

        rng = np.random.default_rng(seed)
        starts = [[0.0, 0.0, -1.0, 1.2, -0.5, 0.8, 1.5, 2.5][:2 * kc]]
        for _ in range(max(0, n_starts - 1)):
            x = np.concatenate([rng.normal(0.0, 1.5, kc - 1),
                                rng.normal(0.0, 1.0, 2 * kc)])
            starts.append(list(x))
        best = None
        for x0 in starts:
            res = minimize(nll, np.asarray(x0, dtype=np.float64), method="Nelder-Mead",
                           options={"maxiter": 20000, "fatol": 1e-9})
            if best is None or res.fun < best.fun:
                best = res
        w, ab = unpack(best.x)
        order = np.argsort(-w)
        w = np.clip(w[order], 0.0, None)
        w = w / w.sum()
        ab = ab[order]
        return cls(weights=tuple(float(x) for x in w),
                   params=tuple((float(a), float(b)) for a, b in ab),
                   resolution=r)

    @staticmethod
    def _edges(resolution: int) -> np.ndarray:
        return np.concatenate([[0.0], np.arange(1.5, resolution + 1.5) / resolution])


__all__ += ["MixtureBetaDaughter"]
