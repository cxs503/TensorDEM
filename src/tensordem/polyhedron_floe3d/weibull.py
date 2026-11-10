"""Weibull weakest-link strength statistics for ice fragmentation.

Ice is a brittle, flaw-dominated material: its strength is not a material
constant but the strength of its *weakest* flaw inside the loaded volume, so the
measured strength of a body of volume :math:`V` follows a two-parameter Weibull
law

.. math::
    P_f(\\sigma; V) = 1 - \\exp\\left[-\\frac{V}{V_0}
        \\left(\\frac{\\sigma}{\\sigma_0}\\right)^{m}\\right],

with shape (modulus) :math:`m` and scale :math:`\\sigma_0` referred to
:math:`V_0`.  Two consequences are *falsifiable predictions* and are used as the
acceptance tests of this module:

* **size effect** (weakest link): the scale of a body of volume ``V`` is
  :math:`\\sigma_0 (V_0/V)^{1/m}`, i.e. a log-log slope of :math:`-1/m`;
* **fragmentation**: if every sub-volume breaks at its own weakest-link
  strength, the resulting fragment-size distribution is itself Weibullian
  (Charles/Gilvarry), which is checked on a Weibull probability plot
  (:math:`\\ln(-\\ln(1-F))` vs :math:`\\ln x`) whose linear-fit
  :math:`R^2` must exceed 0.95.

Everything is numpy-only and deterministic given a ``numpy.random.Generator``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "WeibullFit",
    "FlawFragmentation",
    "fragment_flaw_model",
    "weibull_cdf",
    "weibull_sf",
    "weibull_pdf",
    "sample_strengths",
    "fit_mle",
    "probability_plot_points",
    "fit_probability_plot",
    "weakest_link_scale",
    "weakest_link_survival",
    "kolmogorov_smirnov",
    "fragment_sizes_from_mask",
    "fragmentation_monte_carlo",
    "FragmentationResult",
    "size_effect_study",
]

_STATUSES = ("REAL",)


@dataclass(frozen=True)
class WeibullFit:
    """Result of a two-parameter Weibull calibration."""

    modulus: float
    scale: float
    n: int
    method: str
    r_squared: float = float("nan")
    log_likelihood: float = float("nan")
    slope_stderr: float = float("nan")

    def as_dict(self) -> Dict[str, float]:
        return {
            "modulus": self.modulus,
            "scale": self.scale,
            "n": self.n,
            "method": self.method,
            "r_squared": self.r_squared,
            "log_likelihood": self.log_likelihood,
            "slope_stderr": self.slope_stderr,
        }


# ---------------------------------------------------------------------------
# distribution functions
# ---------------------------------------------------------------------------
def weibull_cdf(x, modulus: float, scale: float):
    x = np.asarray(x, dtype=np.float64)
    z = np.clip(x / float(scale), 0.0, None)
    return 1.0 - np.exp(-(z ** float(modulus)))


def weibull_sf(x, modulus: float, scale: float):
    x = np.asarray(x, dtype=np.float64)
    z = np.clip(x / float(scale), 0.0, None)
    return np.exp(-(z ** float(modulus)))


def weibull_pdf(x, modulus: float, scale: float):
    x = np.asarray(x, dtype=np.float64)
    m = float(modulus)
    s = float(scale)
    z = np.clip(x / s, 1e-300, None)
    return (m / s) * z ** (m - 1.0) * np.exp(-(z ** m))


def sample_strengths(n: int, modulus: float, scale: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Inverse-CDF sample of ``Weibull(modulus, scale)`` (unit lowest value)."""
    if n <= 0:
        raise ValueError("n must be positive")
    u = rng.random(int(n))
    u = np.clip(u, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
    return float(scale) * (-np.log1p(-u)) ** (1.0 / float(modulus))


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def fit_mle(samples: np.ndarray, *, tol: float = 1.0e-12, max_iter: int = 200) -> WeibullFit:
    """Maximum-likelihood calibration of ``(m, sigma_0)``.

    Solves the profile-likelihood equation

    .. math::
        \\frac{1}{m} + \\overline{\\ln x}
        - \\frac{\\sum x_i^m \\ln x_i}{\\sum x_i^m} = 0

    by bisection on :math:`\\ln m` (monotone, hence globally convergent), with
    :math:`\\sigma_0 = \\left(\\overline{x^m}\\right)^{1/m}`.
    """
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    x = x[x > 0.0]
    n = x.size
    if n < 5:
        raise ValueError("at least 5 positive samples are required")
    logx = np.log(x)
    mean_logx = float(logx.mean())

    def profile(m: float) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            xm = np.exp(m * (logx - float(logx.max())))     # rescaled, never overflows
        s = xm.sum()
        if not np.isfinite(s) or s <= 0.0:
            return float("inf")
        # d/dm sum(x^m) / sum(x^m) is shift invariant, so the rescaling is exact
        return 1.0 / m + mean_logx - float((xm * logx).sum() / s)

    lo, hi = 1.0e-3, 1.0e3
    f_lo, f_hi = profile(lo), profile(hi)
    if f_lo * f_hi > 0.0:  # extremely degenerate data: fall back to plot fit
        plot = fit_probability_plot(samples)
        return WeibullFit(plot.modulus, plot.scale, n, "plot-fallback",
                          plot.r_squared, float("nan"), plot.slope_stderr)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = profile(mid)
        if abs(f_mid) < tol or (hi - lo) < tol * mid:
            break
        if f_lo * f_mid <= 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    m = 0.5 * (lo + hi)
    scale = float(np.exp(np.log(np.exp(m * logx).mean()) / m))
    loglik = float(n * (math.log(m) - m * math.log(scale))
                   + (m - 1.0) * logx.sum() - np.exp(m * (logx - math.log(scale))).sum())
    return WeibullFit(m, scale, n, "mle", float("nan"), loglik, float("nan"))


def probability_plot_points(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Weibull probability-plot coordinates ``(ln x, ln(-ln(1-F)))``.

    Plotting positions use the median-rank approximation
    :math:`F_i = (i-0.5)/n`, the standard unbiased choice for probability
    plotting.
    """
    x = np.sort(np.asarray(samples, dtype=np.float64).reshape(-1))
    x = x[x > 0.0]
    n = x.size
    if n < 3:
        raise ValueError("at least 3 positive samples are required")
    f = (np.arange(1, n + 1) - 0.5) / n
    y = np.log(-np.log1p(-f))
    return np.log(x), y


def fit_probability_plot(samples: np.ndarray) -> WeibullFit:
    """Least-squares fit on the Weibull probability plot.

    Returns the modulus (slope), the scale
    :math:`\\sigma_0 = \\exp(-\\text{intercept}/m)` and the coefficient of
    determination of the straight-line fit (the acceptance metric ``R^2``).
    """
    x, y = probability_plot_points(samples)
    n = x.size
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    stderr = math.nan
    if n > 2 and ss_tot > 0.0 and slope != 0.0:
        stderr = math.sqrt(ss_res / (n - 2) / float(np.sum((x - x.mean()) ** 2)))
    modulus = float(slope)
    scale = float(np.exp(-intercept / slope)) if slope > 0.0 else math.inf
    return WeibullFit(modulus, scale, n, "probability-plot", r2, float("nan"), stderr)


# ---------------------------------------------------------------------------
# weakest-link size effect
# ---------------------------------------------------------------------------
def weakest_link_scale(scale: float, size: float, size_ref: float, modulus: float) -> float:
    """Strength scale of a body of ``size`` from a reference ``size_ref``."""
    return float(scale) * (float(size_ref) / float(size)) ** (1.0 / float(modulus))


def weakest_link_survival(stress, modulus: float, scale: float, size: float,
                          size_ref: float):
    """Weakest-link survival probability of a body of volume ``size``."""
    return weibull_sf(stress, modulus, weakest_link_scale(scale, size, size_ref, modulus))


def kolmogorov_smirnov(samples: np.ndarray, modulus: float, scale: float) -> float:
    """Max distance between the empirical CDF and the Weibull CDF."""
    x = np.sort(np.asarray(samples, dtype=np.float64).reshape(-1))
    n = x.size
    cdf = weibull_cdf(x, modulus, scale)
    upper = np.arange(1, n + 1) / n
    lower = np.arange(0, n) / n
    return float(max(np.max(np.abs(cdf - upper)), np.max(np.abs(cdf - lower))))


# ---------------------------------------------------------------------------
# weakest-link fragmentation
# ---------------------------------------------------------------------------
@dataclass
class FragmentationResult:
    """Outcome of one weakest-link fragmentation realisation."""

    fragments: np.ndarray          # (k,) fragment cell counts
    masses: np.ndarray             # (k,) fragment volumes (unit density)
    sizes: np.ndarray              # (k,) equivalent sphere diameters (m)
    broken_fraction: float         # fraction of the *active* cells that failed
    intact: bool                   # True when the body did not split
    broken_volume: float = 0.0     # failed-cell volume inside the active body

    @property
    def n_fragments(self) -> int:
        return int(self.fragments.size)


def fragment_sizes_from_mask(mask: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    """Connected components of an unbroken-cell mask -> fragment volumes.

    ``mask`` is True where a cell survived.  Fragments are 6-connected
    components (:func:`scipy.ndimage.label`), which keeps two fragments that only
    touch at an edge/corner apart -- the physically meaningful separation for a
    crack that must cut a full face.
    """
    from scipy import ndimage

    structure = ndimage.generate_binary_structure(3, 1)
    labels, count = ndimage.label(mask, structure=structure)
    if count == 0:
        return np.zeros(0)
    cell_volume = float(np.prod(np.asarray(spacing, dtype=np.float64)))
    return np.asarray(ndimage.sum(mask, labels, index=np.arange(1, count + 1)),
                      dtype=np.float64) * cell_volume


def fragmentation_monte_carlo(n_cells: Sequence[int], *, cell_size: float,
                              stress_field: np.ndarray,
                              modulus: float, scale: float, size_ref: float,
                              rng: np.random.Generator,
                              active: Optional[np.ndarray] = None) -> FragmentationResult:
    """One weakest-link fragmentation realisation on a rectilinear cell lattice.

    Every cell draws its own strength from the volume-scaled Weibull law
    (``size = cell volume``), and fails where the imposed stress field exceeds
    it.  The surviving cells are labelled into fragments, so the fragment size
    distribution is a *consequence* of the Weibull law and not a fitted input --
    that is what makes the probability-plot check falsifiable.
    """
    n_cells = tuple(int(v) for v in n_cells)
    n = int(np.prod(n_cells))
    stress_field = np.asarray(stress_field, dtype=np.float64)
    if stress_field.size != n:
        raise ValueError("stress field size does not match the cell lattice")
    cell_volume = float(cell_size) ** 3
    cell_scale = weakest_link_scale(scale, cell_volume, float(size_ref), modulus)
    strengths = sample_strengths(n, modulus, cell_scale, rng).reshape(n_cells)
    survived = stress_field.reshape(n_cells) < strengths
    active_mask = np.ones(n_cells, dtype=bool) if active is None else \
        np.asarray(active, dtype=bool).reshape(n_cells)
    # fragments must not leak through inactive (outside-body) cells
    survived_active = np.logical_and(survived, active_mask)
    volumes = fragment_sizes_from_mask(survived_active, (cell_size,) * 3)
    broken_volume = float(np.count_nonzero(active_mask) - np.count_nonzero(survived_active)) \
        * cell_volume
    broken_fraction = broken_volume / max(float(np.count_nonzero(active_mask)) * cell_volume,
                                          1.0e-300)
    if volumes.size == 0:
        return FragmentationResult(np.zeros(0), np.zeros(0), np.zeros(0),
                                   broken_fraction, False, broken_volume)
    counts = volumes / cell_volume
    diameters = (6.0 * volumes / math.pi) ** (1.0 / 3.0)
    return FragmentationResult(counts, volumes, diameters, broken_fraction,
                               volumes.size == 1, broken_volume)


def size_effect_study(scales_by_size: Dict[float, float]) -> Dict[str, float]:
    """Fit ``ln sigma_0 = const - (1/m) ln V`` and report the implied modulus.

    Returns ``{"modulus_implied", "r_squared", "slope", "intercept"}``.
    """
    sizes = np.array(sorted(scales_by_size), dtype=np.float64)
    scales = np.array([scales_by_size[s] for s in sizes], dtype=np.float64)
    slope, intercept = np.polyfit(np.log(sizes), np.log(scales), 1)
    pred = slope * np.log(sizes) + intercept
    ss_res = float(np.sum((np.log(scales) - pred) ** 2))
    ss_tot = float(np.sum((np.log(scales) - np.log(scales).mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {"modulus_implied": float(-1.0 / slope) if slope < 0.0 else math.inf,
            "slope": float(slope), "intercept": float(intercept), "r_squared": r2}

# ---------------------------------------------------------------------------
# flaw-influence (Poisson-Voronoi) fragmentation -- the classical weakest-link
# fragment-size model
# ---------------------------------------------------------------------------
@dataclass
class FlawFragmentation:
    """One realisation of the flaw-influence fragmentation model."""

    volumes: np.ndarray      # (k,) fragment volumes (m^3)
    masses: np.ndarray       # (k,) fragment masses at unit density (= volumes)
    sizes: np.ndarray        # (k,) equivalent sphere diameters (m)
    n_flaws: int             # activated flaws in this realisation
    intact: bool
    rate_sum: float          # Poisson mean of the activated-flaw count


def fragment_flaw_model(centres: np.ndarray, active: np.ndarray, stress: np.ndarray,
                        cell_volume: float, *, modulus: float, scale: float,
                        size_ref: float, rng: np.random.Generator,
                        jitter: float = 0.4) -> FlawFragmentation:
    """Fragmentation as the influence zones of the *activated* flaws.

    Classical weakest-link fragmentation (Gilvarry 1961; Grady 1985): flaws with a
    Weibull strength distribution are a Poisson field of intensity
    ``(sigma/sigma_0)^m / V_0`` per unit volume, so under a stress field
    ``sigma(x)`` the activated flaws form an inhomogeneous Poisson process with
    mean ``mu = sum_i (sigma_i/sigma_0)^m V_cell/V_0``.  Every fragment is the
    region of influence of one activated flaw, realised here by a discrete
    Voronoi assignment of the body's cells.

    Two properties matter for the audit:

    * **mass conservation is exact by construction** -- every active cell is
      assigned to exactly one activated flaw, so ``sum(V_frag) == V_body``;
    * the fragment size distribution is *not* an input: it follows from Poisson
      tessellation statistics, and its scale follows ``sigma^{-m}`` because
      ``mu ~ sigma^m``.  Both are checked in :mod:`tensordem.polyhedron_floe3d.study`.

    Contrast :func:`fragmentation_monte_carlo`, which draws one independent
    strength per lattice cell: that is a percolation model and yields
    percolation (not Weibullian) cluster statistics -- measured, not assumed.
    """
    centres = np.asarray(centres, dtype=np.float64)
    active = np.asarray(active, dtype=bool).reshape(-1)
    stress = np.asarray(stress, dtype=np.float64).reshape(-1)
    points = centres[active]
    sigma = np.clip(stress[active], 0.0, None)
    n_active = points.shape[0]
    body_volume = n_active * cell_volume
    if n_active == 0:
        return FlawFragmentation(np.zeros(0), np.zeros(0), np.zeros(0), 0, False, 0.0)

    rate = (sigma / float(scale)) ** float(modulus) * cell_volume / float(size_ref)
    mu = float(rate.sum())
    n_flaws = int(rng.poisson(mu))
    # a fragment cannot be finer than one lattice cell; capping keeps the
    # discrete Voronoi assignment well posed and bounds the cost
    capped = n_flaws > n_active
    n_flaws = min(n_flaws, n_active)
    if n_flaws < 2:
        size = (6.0 * body_volume / math.pi) ** (1.0 / 3.0)
        return FlawFragmentation(np.array([body_volume]), np.array([body_volume]),
                                 np.array([size]), n_flaws, True, mu)

    probability = rate / mu
    owners = rng.choice(n_active, size=n_flaws, p=probability)
    spacing = cell_volume ** (1.0 / 3.0)
    seeds = points[owners] + rng.uniform(-jitter, jitter, (n_flaws, 3)) * spacing
    delta = points[:, None, :] - seeds[None, :, :]
    label = np.argmin(np.einsum("ijk,ijk->ij", delta, delta), axis=1)
    counts = np.bincount(label, minlength=n_flaws)
    volumes = counts[counts > 0].astype(np.float64) * cell_volume
    diameters = (6.0 * volumes / math.pi) ** (1.0 / 3.0)
    if capped:  # record the cap through the rate sum instead of hiding it
        mu = max(mu, float(n_flaws))
    return FlawFragmentation(volumes, volumes.copy(), diameters, n_flaws,
                             volumes.size == 1, mu)
