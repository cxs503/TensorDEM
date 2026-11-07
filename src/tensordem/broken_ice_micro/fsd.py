"""Power-law floe-size-distribution estimation.

The observable is the *number* distribution of floe diameters,
``n(D) dD ~ D**(-alpha) dD`` (equivalently ``N(D > x) ~ x**(1-alpha)``).  Two
estimators are produced for every census so that the result is not an artefact
of one fitting choice:

* **maximum likelihood with an objective cutoff** (Clauset, Shalizi & Newman
  2009, *SIAM Review* 51, 661): for a candidate ``D_min`` the exponent is
  ``alpha = 1 + n / sum(log(D_i / D_min))`` and the candidate cutoff is chosen
  to minimise the Kolmogorov-Smirnov distance between the empirical tail CDF and
  the fitted power law.  The standard error is ``(alpha - 1)/sqrt(n)``.
* **log-binned least squares**: counts in logarithmic bins divided by the bin
  width give the density, and the ``R**2`` of the straight-line fit in
  ``log``-``log`` space is the dispersion measure the campaign gates on
  (``R**2 > 0.9``).

Both are self-contained (no SciPy dependency on the import path) so the module
can be used from a minimal environment.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

__all__ = [
    "FSD_FIT_ALPHA_RANGE",
    "PowerLawFit",
    "fit_power_law",
    "log_binned_histogram",
]

#: Campaign window for the FSD exponent (IN34-W3-N01 acceptance criterion (a)).
FSD_FIT_ALPHA_RANGE: tuple[float, float] = (2.5, 3.5)


@dataclass
class PowerLawFit:
    """Result of the two-track power-law estimate of a fragment census."""

    alpha: float
    alpha_stderr: float
    x_min: float
    x_max: float
    n_tail: int
    ks_distance: float
    r2: float
    slope_binned: float
    binned_x: np.ndarray
    binned_y: np.ndarray
    n_total: int

    @property
    def in_campaign_range(self) -> bool:
        lo, hi = FSD_FIT_ALPHA_RANGE
        return lo <= self.alpha <= hi

    @property
    def passes(self) -> bool:
        """Campaign criterion (a): alpha in range *and* R**2 > 0.9."""
        return self.in_campaign_range and self.r2 > 0.9

    def as_dict(self) -> dict:
        return dict(
            alpha=float(self.alpha),
            alpha_stderr=float(self.alpha_stderr),
            x_min_m=float(self.x_min),
            x_max_m=float(self.x_max),
            n_tail=int(self.n_tail),
            n_total=int(self.n_total),
            ks_distance=float(self.ks_distance),
            r2=float(self.r2),
            slope_binned=float(self.slope_binned),
            in_campaign_range=bool(self.in_campaign_range),
            passes=bool(self.passes),
            alpha_window=list(FSD_FIT_ALPHA_RANGE),
            binned_x=[float(v) for v in self.binned_x],
            binned_y=[float(v) for v in self.binned_y],
        )


def log_binned_histogram(
    sizes,
    *,
    x_min: float,
    x_max: float | None = None,
    bins_per_decade: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Density of a power-law tail in logarithmic bins.

    Returns ``(centres, density)`` with ``density`` the number of samples per
    unit size (so a pure ``x**-alpha`` law is a straight line on a log-log
    plot).  Empty bins are dropped.
    """
    data = np.asarray(sizes, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data) & (data > 0.0)]
    if x_max is None:
        x_max = float(data.max())
    if not (x_max > x_min > 0.0):
        raise ValueError("require 0 < x_min < x_max")
    n_bins = max(3, int(math.ceil(math.log10(x_max / x_min) * bins_per_decade)))
    edges = x_min * (x_max / x_min) ** (np.arange(n_bins + 1) / n_bins)
    counts, _ = np.histogram(data, bins=edges)
    widths = np.diff(edges)
    centres = 0.5 * (edges[1:] + edges[:-1])
    density = counts / widths
    keep = counts > 0
    return centres[keep], density[keep]


def _ks_distance(sorted_tail: np.ndarray, alpha: float, x_min: float) -> float:
    n = sorted_tail.size
    if n == 0:
        return float("nan")
    cdf_model = 1.0 - (sorted_tail / x_min) ** (1.0 - alpha)
    empirical = np.arange(1, n + 1) / n
    return float(np.max(np.abs(empirical - cdf_model)))


def fit_power_law(
    sizes,
    *,
    x_min: float | None = None,
    min_tail: int = 20,
    max_candidates: int = 40,
    x_max: float | None = None,
    bins_per_decade: float = 4.0,
) -> PowerLawFit:
    """Estimate the FSD exponent of a fragment-size sample.

    Parameters
    ----------
    sizes
        Fragment equivalent diameters (m).
    x_min
        Lower cutoff.  ``None`` (default) selects it by minimising the KS
        distance over candidate cutoffs drawn from the sample itself.
    min_tail
        Minimum number of fragments above the cutoff.
    x_max
        Upper cutoff (defaults to the largest fragment).
    """
    data = np.sort(np.asarray(sizes, dtype=np.float64).reshape(-1))
    data = data[np.isfinite(data) & (data > 0.0)]
    if data.size < min_tail:
        raise ValueError(f"need at least {min_tail} fragments, got {data.size}")
    hi = float(x_max) if x_max is not None else float(data[-1])

    def mle(tail: np.ndarray, lo: float) -> float:
        return 1.0 + tail.size / float(np.log(tail / lo).sum())

    if x_min is None:
        candidates = data[: max(1, data.size - min_tail)][:: max(1, data.size // max_candidates)]
        candidates = np.unique(candidates)
        best = (math.inf, float(candidates[0]), float("nan"))
        for lo in candidates:
            tail = data[data >= lo]
            if tail.size < min_tail:
                continue
            a = mle(tail, lo)
            d = _ks_distance(tail, a, lo)
            if d < best[0]:
                best = (d, float(lo), a)
        if not math.isfinite(best[0]):
            raise ValueError("no candidate cutoff held the minimum tail length")
        ks, lo, _ = best
    else:
        lo = float(x_min)
        tail = data[data >= lo]
        if tail.size < min_tail:
            raise ValueError("too few fragments above the requested x_min")
        ks = _ks_distance(tail, mle(tail, lo), lo)

    tail = data[(data >= lo) & (data <= hi)]
    alpha = mle(tail, lo)
    stderr = (alpha - 1.0) / math.sqrt(tail.size)

    centres, density = log_binned_histogram(
        tail, x_min=lo, x_max=hi if hi > lo else tail.max(), bins_per_decade=bins_per_decade
    )
    if centres.size >= 3:
        lx = np.log10(centres)
        ly = np.log10(density)
        slope, intercept = np.polyfit(lx, ly, 1)
        pred = slope * lx + intercept
        ss_res = float(((ly - pred) ** 2).sum())
        ss_tot = float(((ly - ly.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    else:
        slope, r2 = float("nan"), 0.0

    return PowerLawFit(
        alpha=float(alpha),
        alpha_stderr=float(stderr),
        x_min=float(lo),
        x_max=float(hi),
        n_tail=int(tail.size),
        ks_distance=float(ks),
        r2=float(r2),
        slope_binned=float(-slope),
        binned_x=centres,
        binned_y=density,
        n_total=int(data.size),
    )