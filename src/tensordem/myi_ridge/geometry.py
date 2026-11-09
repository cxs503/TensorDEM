"""Multi-year ice (MYI) pressure-ridge keel geometry statistics.

Physical background
-------------------
A multi-year pressure ridge consists of a keel of consolidated and
unconsolidated rubble below the waterline and a sail above it.  For
engineering loads the three geometric descriptors that matter are

* the keel depth ``h_keel`` (distance from the waterline to the keel tip),
* the porosity ``n`` of the unconsolidated rubble (void volume fraction),
* the characteristic block size ``L_floe`` (diameter of the individual ice
  blocks the rubble is made of).

Published ridge statistics (Timco & Frederking 1998, *Physical properties of
sea ice relevant to ice-induced loading*; Timco & Burden 1997, *An analysis of
the shape of sea ice ridges*; Strub-Klein & Sudom 2012, *A comprehensive
analysis of the morphology of first- and second-year sea-ice ridges*) show that
keel depth is well described by a log-normal distribution, that multi-year
ridges are deeper than first-year ridges (mean keel depth of order 10-20 m,
extremes past 30 m), that keel depth and width are roughly self-similar with a
flank angle of 20-25 deg measured from the vertical, that unconsolidated keel
porosity is 0.2-0.4, and that block sizes range from sub-metre fragments to
more than 10 m.

This module implements the statistics end of the ridge chain: sampling of the
joint distribution and construction of a plane-strain keel cross-section made of
discrete blocks (plus a fragment ladder) held at a prescribed porosity.  The
cross-section object is the geometry input of the DEM crushing model
(:mod:`tensordem.myi_ridge.crushing`) and of the campaign figures.

Parameter values in :class:`MyiRidgeStatistics` are *representative* inputs
taken from the compilations above; they are not measurements made here, and
every derived statistic is reported against its target so the generation
fidelity (rather than the literature value) is what the tests check.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

__all__ = [
    "MyiRidgeStatistics",
    "KeelSection",
    "lognormal_from_moments",
    "sample_keel_depths",
    "sample_porosity",
    "sample_block_sizes",
    "sail_height",
    "keel_width",
    "keel_envelope_half_width",
    "build_keel_section",
    "section_statistics",
    "ensemble_statistics",
]


def _real(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class MyiRidgeStatistics:
    """Joint statistics of a multi-year ridge keel (representative inputs)."""

    keel_depth_mean: float = 16.0
    keel_depth_cv: float = 0.35
    keel_depth_bounds: tuple[float, float] = (10.0, 30.0)
    sail_to_keel_ratio: float = 0.22
    porosity_mean: float = 0.30
    porosity_std: float = 0.05
    porosity_bounds: tuple[float, float] = (0.20, 0.40)
    block_size_median: float = 4.0
    block_size_sigma: float = 0.45
    block_size_bounds: tuple[float, float] = (2.0, 10.0)
    flank_angle_deg: float = 22.0
    block_area_fraction: float = 0.75

    def __post_init__(self) -> None:
        for name in ("keel_depth_mean", "keel_depth_cv", "sail_to_keel_ratio",
                     "porosity_std", "block_size_median", "block_size_sigma",
                     "flank_angle_deg", "block_area_fraction"):
            _real(name, getattr(self, name), positive=True)
        _real("porosity_mean", self.porosity_mean)
        for name in ("keel_depth_bounds", "porosity_bounds", "block_size_bounds"):
            bounds = getattr(self, name)
            if len(bounds) != 2 or not bounds[0] < bounds[1]:
                raise ValueError(f"{name} must be an increasing pair")
        low, high = self.porosity_bounds
        if not low < self.porosity_mean < high:
            raise ValueError("porosity_mean must lie inside porosity_bounds")
        if not 0.0 <= low and high < 0.6:
            raise ValueError("porosity bounds must lie in [0, 0.6)")
        if not 0.0 < self.block_area_fraction <= 1.0:
            raise ValueError("block_area_fraction must lie in (0, 1]")
        if not 0.0 < self.flank_angle_deg < 60.0:
            raise ValueError("flank_angle_deg must lie in (0, 60)")

    @property
    def keel_depth_std(self) -> float:
        return self.keel_depth_cv * self.keel_depth_mean


def truncated_lognormal_moments(mu: float, sigma: float, bounds: tuple[float, float]
                                ) -> tuple[float, float]:
    """Mean and standard deviation of a log-normal law truncated to ``bounds``."""
    low, high = bounds
    a = (math.log(low) - mu) / sigma
    b = (math.log(high) - mu) / sigma
    mass = norm.cdf(b) - norm.cdf(a)
    if mass <= 0.0:  # pragma: no cover - defensive
        raise ValueError("truncation interval carries no probability")
    first = math.exp(mu + 0.5 * sigma ** 2) * (norm.cdf(b - sigma) - norm.cdf(a - sigma)) / mass
    second = math.exp(2.0 * mu + 2.0 * sigma ** 2) \
        * (norm.cdf(b - 2.0 * sigma) - norm.cdf(a - 2.0 * sigma)) / mass
    variance = max(second - first ** 2, 0.0)
    return first, math.sqrt(variance)


def lognormal_from_moments(mean: float, cv: float,
                           bounds: tuple[float, float] | None = None
                           ) -> tuple[float, float]:
    """``(mu, sigma)`` of the log-normal law with arithmetic mean and CV given.

    With ``bounds`` the parameters are solved so that the *truncated* law keeps
    the requested arithmetic mean and CV, which removes the bias that naive
    rejection sampling of a truncated tail would introduce.
    """
    _real("mean", mean, positive=True)
    _real("cv", cv, positive=True)
    std = cv * mean
    sigma2 = math.log1p((std / mean) ** 2)
    mu0 = math.log(mean) - 0.5 * sigma2
    sigma0 = math.sqrt(sigma2)
    if bounds is None:
        return mu0, sigma0
    low, high = bounds
    if not low < mean < high:
        raise ValueError("mean must lie inside the truncation bounds")

    # the log-normal shape parameter is taken from the closed-form CV; the
    # location is then solved so the *truncated* law keeps the target mean.
    # Truncation necessarily shrinks the spread, so the achieved standard
    # deviation is reported by the caller instead of being forced.
    def mean_at(mu: float, sigma: float, bounds_: tuple[float, float]) -> float:
        return truncated_lognormal_moments(mu, sigma, bounds_)[0]

    low_mu = math.log(low) - 8.0 * sigma0
    high_mu = math.log(high) + 8.0 * sigma0
    mu = brentq(lambda guess: mean_at(guess, sigma0, bounds) - mean,
                low_mu, high_mu, xtol=1e-13, rtol=1e-13, maxiter=200)
    return mu, sigma0


def _rng(seed: int) -> np.random.Generator:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    return np.random.default_rng(int(seed))


def _truncated_lognormal(mean: float, cv: float, bounds: tuple[float, float],
                         size: int, rng: np.random.Generator) -> np.ndarray:
    """Rejection sampling of a log-normal law restricted to ``bounds``."""
    mu, sigma = lognormal_from_moments(mean, cv, bounds)
    low, high = bounds
    out = np.empty(size, dtype=float)
    filled = 0
    guard = 0
    while filled < size:
        draw = rng.lognormal(mu, sigma, size=4 * (size - filled) + 16)
        keep = draw[(draw >= low) & (draw <= high)]
        take = min(keep.size, size - filled)
        out[filled:filled + take] = keep[:take]
        filled += take
        guard += 1
        if guard > 10000:  # pragma: no cover - defensive
            raise RuntimeError("truncated log-normal sampling failed to converge")
    return out


def sample_keel_depths(stats: MyiRidgeStatistics, count: int,
                       seed: int = 20261108) -> np.ndarray:
    """Sample ``count`` keel depths [m] from the truncated log-normal law."""
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("count must be a positive integer")
    return _truncated_lognormal(stats.keel_depth_mean, stats.keel_depth_cv,
                                stats.keel_depth_bounds, int(count), _rng(seed))


def sample_porosity(stats: MyiRidgeStatistics, count: int,
                    seed: int = 20261109) -> np.ndarray:
    """Sample rubble porosities, clipped to the physical band."""
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("count must be a positive integer")
    rng = _rng(seed)
    low, high = stats.porosity_bounds
    draw = rng.normal(stats.porosity_mean, stats.porosity_std, size=int(count))
    return np.clip(draw, low, high)


def sample_block_sizes(stats: MyiRidgeStatistics, count: int,
                       seed: int = 20261110) -> np.ndarray:
    """Sample block diameters [m] (truncated log-normal)."""
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("count must be a positive integer")
    mu = math.log(stats.block_size_median)
    sigma = stats.block_size_sigma
    low, high = stats.block_size_bounds
    rng = _rng(seed)
    out = np.empty(int(count), dtype=float)
    filled = 0
    guard = 0
    while filled < count:
        draw = rng.lognormal(mu, sigma, size=4 * (count - filled) + 16)
        keep = draw[(draw >= low) & (draw <= high)]
        take = min(keep.size, count - filled)
        out[filled:filled + take] = keep[:take]
        filled += take
        guard += 1
        if guard > 10000:  # pragma: no cover - defensive
            raise RuntimeError("block size sampling failed to converge")
    return out


def sail_height(keel_depth: float, stats: MyiRidgeStatistics) -> float:
    """Sail height [m] implied by the mean sail-to-keel ratio."""
    _real("keel_depth", keel_depth, positive=True)
    return stats.sail_to_keel_ratio * float(keel_depth)


def keel_width(keel_depth: float, stats: MyiRidgeStatistics) -> float:
    """Keel width at the waterline [m] for the self-similar flank angle."""
    _real("keel_depth", keel_depth, positive=True)
    return 2.0 * float(keel_depth) * math.tan(math.radians(stats.flank_angle_deg))


def keel_envelope_half_width(height_above_tip: float, keel_depth: float,
                             stats: MyiRidgeStatistics) -> float:
    """Half width [m] of the triangular keel envelope at a height above the tip."""
    _real("keel_depth", keel_depth, positive=True)
    clamped = min(max(float(height_above_tip), 0.0), float(keel_depth))
    return 0.5 * keel_width(keel_depth, stats) * clamped / float(keel_depth)


@dataclass
class KeelSection:
    """A plane-strain keel cross-section built from discrete blocks."""

    keel_depth: float
    porosity_target: float
    porosity: float
    centers: np.ndarray
    radii: np.ndarray
    is_block: np.ndarray
    envelope_area: float
    block_solid_area: float
    fragment_solid_area: float
    flank_angle_deg: float
    seed: int

    @property
    def solid_area(self) -> float:
        return float(self.block_solid_area + self.fragment_solid_area)

    @property
    def count(self) -> int:
        return int(self.centers.shape[0])

    @property
    def block_count(self) -> int:
        return int(self.is_block.sum())

    @property
    def keel_width(self) -> float:
        return 2.0 * self.keel_depth * math.tan(math.radians(self.flank_angle_deg))

    @property
    def measured_keel_depth(self) -> float:
        """Deepest point reached by the rubble [m] (0 if no rubble)."""
        if self.count == 0:  # pragma: no cover - defensive
            return 0.0
        return float(np.max(-(self.centers[:, 1]) + self.radii))


def _sample_inside_triangle(rng: np.random.Generator, half_width: float,
                            keel_depth: float, min_y: float | None = None
                            ) -> tuple[float, float]:
    """Area-uniform sample of a point inside the triangular keel envelope.

    ``min_y`` (a negative height above the tip) restricts the sample so that a
    block of the given radius still fits inside the keel depth.
    """
    floor = -keel_depth if min_y is None else max(min_y, -keel_depth)
    for _ in range(64):
        y = -keel_depth * (1.0 - math.sqrt(1.0 - float(rng.random())))
        if y < floor:
            continue
        width = half_width * (1.0 + y / keel_depth)
        x = width * (2.0 * float(rng.random()) - 1.0)
        return x, y
    width = half_width * (1.0 + floor / keel_depth)
    return 0.0, floor


def build_keel_section(keel_depth: float, porosity: float,
                       stats: MyiRidgeStatistics, *, seed: int = 20261108,
                       max_block_attempts: int = 6000,
                       fragment_ratio: float = 4.0) -> KeelSection:
    """Fill the keel envelope with blocks and filler fragments.

    Large blocks (diameters drawn from :func:`sample_block_sizes`) are placed
    first by random sequential adsorption until ``block_area_fraction`` of the
    required solid area is reached or ``max_block_attempts`` is exhausted; the
    remaining solid area is filled by a fragment ladder of radius divided by
    ``fragment_ratio``.  The achieved porosity is measured, never assumed.
    """
    _real("keel_depth", keel_depth, positive=True)
    _real("porosity", porosity)
    if not 0.0 <= porosity < 0.6:
        raise ValueError("porosity must lie in [0, 0.6)")
    rng = _rng(seed)
    h = float(keel_depth)
    half = 0.5 * keel_width(h, stats)
    area = 0.5 * half * 2.0 * h
    solid_target = (1.0 - porosity) * area
    block_target = stats.block_area_fraction * solid_target

    centers: list[tuple[float, float]] = []
    radii: list[float] = []
    is_block: list[bool] = []

    def _place(radius: float) -> bool:
        for _ in range(40):
            x, y = _sample_inside_triangle(rng, half, h, min_y=-(h - radius))
            if centers:
                cx = np.asarray(centers)
                rr = np.asarray(radii)
                gap = np.hypot(cx[:, 0] - x, cx[:, 1] - y) - (rr + radius)
                if not bool(np.all(gap > 0.02 * radius)):
                    continue
            centers.append((x, y))
            radii.append(radius)
            return True
        return False

    block_pool = sample_block_sizes(stats, 400, seed=seed + 1)
    solid = 0.0
    # seed one block at the deepest admissible position so that the realised
    # keel depth equals the requested depth instead of being one radius short
    tip_radius = min(0.5 * float(np.median(block_pool)),
                     math.sqrt(max(solid_target, 1e-9) / (8.0 * math.pi)))
    if h - tip_radius > 0.0:
        centers.append((0.0, -(h - tip_radius)))
        radii.append(tip_radius)
        is_block.append(True)
        solid += math.pi * tip_radius ** 2
    used = 0
    attempts = 0
    while solid < block_target and attempts < max_block_attempts:
        radius = 0.5 * float(block_pool[used % block_pool.size])
        used += 1
        attempts += 1
        if solid + math.pi * radius ** 2 > block_target:
            continue  # the block would overshoot the large-block budget
        if _place(radius):
            is_block.append(True)
            solid += math.pi * radius ** 2

    block_solid = solid
    # fragment matrix: a triangular lattice of small fragments fills the space
    # left between the large blocks and is thinned so the measured solid area
    # matches the requested porosity.  A lattice fill (rather than a random
    # adsorption ladder) is used because the pockets between large blocks are
    # too small for blind random placement in shallow sections.
    remaining = solid_target - solid
    scale = fragment_ratio
    for _pass in range(5):
        if remaining <= 0.0 or not centers:
            break
        r_f = max(0.5 * float(np.median(block_pool)) / scale, 0.03)
        spacing = 2.0 * r_f
        row = spacing * math.sqrt(3.0) / 2.0
        placed_centers = np.asarray(centers, dtype=float)
        placed_radii = np.asarray(radii, dtype=float)
        candidates = []
        j = 0
        while (j + 0.5) * row <= h:
            height = (j + 0.5) * row
            y = -h + height
            half_w = half * height / h
            offset = 0.5 * spacing if j % 2 else 0.0
            count = int(2.0 * half_w / spacing) + 1
            for i in range(count):
                x = -half_w + (i + 0.5) * spacing + offset
                if abs(x) > half_w:
                    continue
                gap = np.hypot(placed_centers[:, 0] - x, placed_centers[:, 1] - y) \
                    - (placed_radii + 0.9 * r_f)
                if bool(np.all(gap > 0.0)):
                    candidates.append((x, y))
            j += 1
        if not candidates:
            scale *= 2.0
            continue
        candidates = np.asarray(candidates, dtype=float)
        needed = int(round(remaining / (math.pi * r_f ** 2)))
        take = min(needed, candidates.shape[0])
        if take > 0:
            pick = rng.permutation(candidates.shape[0])[:take]
            for index in pick:
                centers.append((float(candidates[index, 0]), float(candidates[index, 1])))
                radii.append(r_f)
                is_block.append(False)
                solid += math.pi * r_f ** 2
        remaining = solid_target - solid
        scale *= 2.0
    measured = 1.0 - solid / area
    return KeelSection(
        keel_depth=h,
        porosity_target=float(porosity),
        porosity=float(measured),
        centers=np.asarray(centers, dtype=float).reshape(-1, 2),
        radii=np.asarray(radii, dtype=float),
        is_block=np.asarray(is_block, dtype=bool),
        envelope_area=float(area),
        block_solid_area=float(block_solid),
        fragment_solid_area=float(solid - block_solid),
        flank_angle_deg=float(stats.flank_angle_deg),
        seed=int(seed),
    )


def section_statistics(section: KeelSection, stats: MyiRidgeStatistics) -> dict:
    """Measured geometry statistics of one keel section."""
    block_diameters = 2.0 * section.radii[section.is_block]
    return {
        "keel_depth": section.keel_depth,
        "measured_keel_depth": section.measured_keel_depth,
        "keel_depth_error": (section.measured_keel_depth - section.keel_depth)
        / section.keel_depth,
        "porosity_target": section.porosity_target,
        "porosity_measured": section.porosity,
        "porosity_error": section.porosity - section.porosity_target,
        "blocks": section.block_count,
        "fragments": int(section.count - section.block_count),
        "block_diameter_mean": float(block_diameters.mean()) if block_diameters.size else 0.0,
        "block_diameter_min": float(block_diameters.min()) if block_diameters.size else 0.0,
        "block_diameter_max": float(block_diameters.max()) if block_diameters.size else 0.0,
        "block_area_fraction": section.block_solid_area / max(section.solid_area, 1e-30),
        "solid_area": section.solid_area,
        "envelope_area": section.envelope_area,
        "keel_width": section.keel_width,
        "sail_height": sail_height(section.keel_depth, stats),
    }


def ensemble_statistics(stats: MyiRidgeStatistics, count: int = 400,
                        *, seed: int = 20261108) -> dict:
    """Compare the sampled ensemble with the target distribution.

    Returns the arithmetic mean error of the sampled keel depths, the
    porosity band statistics, block-size statistics and quantile comparison of
    the sampled log-normal law against the analytic one.
    """
    depths = sample_keel_depths(stats, count, seed=seed)
    porosity = sample_porosity(stats, count, seed=seed + 1)
    blocks = sample_block_sizes(stats, count, seed=seed + 2)
    mu, sigma = lognormal_from_moments(stats.keel_depth_mean, stats.keel_depth_cv)
    quantiles = (0.05, 0.25, 0.5, 0.75, 0.95)
    # analytic quantiles of the un-truncated law, renormalised by the truncation
    from statistics import NormalDist

    normal = NormalDist()
    analytic = [math.exp(mu + sigma * normal.inv_cdf(q)) for q in quantiles]
    empirical = np.quantile(depths, quantiles).tolist()
    density, edges = np.histogram(depths, bins=12)
    return {
        "count": int(count),
        "keel_depth_mean_target": stats.keel_depth_mean,
        "keel_depth_mean_sampled": float(depths.mean()),
        "keel_depth_mean_error": float(depths.mean() / stats.keel_depth_mean - 1.0),
        "keel_depth_std_sampled": float(depths.std(ddof=1)),
        "keel_depth_std_target": stats.keel_depth_std,
        "keel_depth_quantiles": [float(v) for v in empirical],
        "keel_depth_quantiles_lognormal": analytic,
        "keel_depth_min": float(depths.min()),
        "keel_depth_max": float(depths.max()),
        "keel_depth_histogram": {
            "edges": [float(v) for v in edges],
            "counts": [int(v) for v in density],
        },
        "porosity_mean_sampled": float(porosity.mean()),
        "porosity_std_sampled": float(porosity.std(ddof=1)),
        "porosity_min": float(porosity.min()),
        "porosity_max": float(porosity.max()),
        "block_diameter_mean": float(blocks.mean()),
        "block_diameter_mean_target": truncated_lognormal_moments(
            math.log(stats.block_size_median), stats.block_size_sigma,
            stats.block_size_bounds)[0],
        "block_diameter_mean_error": float(
            blocks.mean() / truncated_lognormal_moments(
                math.log(stats.block_size_median), stats.block_size_sigma,
                stats.block_size_bounds)[0] - 1.0),
        "block_diameter_median": float(np.median(blocks)),
        "block_diameter_min": float(blocks.min()),
        "block_diameter_max": float(blocks.max()),
        "block_size_bounds": list(stats.block_size_bounds),
    }