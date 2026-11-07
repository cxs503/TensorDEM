"""Statistics of the structure pressure field ``p(y, z, t)``.

Three quantities are produced, matching the three engineering questions:

``gamma`` (peak factor)
    ``max(p) / mean(p)`` over the loaded part of the structure.  Three
    definitions are kept apart because they answer different questions and
    differ by a large factor if confused:

    * ``gamma_spatial`` -- peak of the **time-averaged** field over its mean
      (the patch factor of a sustained loading; the design-relevant one),
    * ``gamma_instant`` -- snapshot peak over snapshot mean, averaged over time
      (the instantaneous localisation of the crushing zone),
    * ``gamma_extreme`` -- absolute space-time maximum over the grand mean (an
      upper bound that grows with the record length and the grid resolution,
      and is therefore *not* used for design).

``L_c`` (correlation length)
    e-folding radius of the azimuthally averaged autocorrelation of the
    time-averaged pressure field, plus the one-dimensional integral scales along
    ``y`` and ``z``.  Biased/unbiased FFT autocorrelation with overlap
    correction; see :func:`spatial_autocorrelation`.

Design extreme
    Gumbel block-maxima fit of the spatial-mean pressure and of the local peak
    pressure, with a return level for a target exceedance probability.  The
    limits of a short, strongly autocorrelated record are stated in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve

__all__ = [
    "PressureStatistics",
    "spatial_autocorrelation",
    "radial_profile",
    "correlation_length",
    "block_maxima",
    "gumbel_fit",
    "gumbel_return_level",
    "pressure_statistics",
]


def spatial_autocorrelation(field: np.ndarray) -> np.ndarray:
    """Unbiased, normalised spatial autocorrelation of a 2-D field.

    ``R(my, mz) = <p'(r) p'(r + m)> / <p'^2>`` with ``p' = p - <p>``, returned as
    a *centred* array of shape ``(2*nz-1, 2*ny-1)`` whose centre
    ``(nz-1, ny-1)`` is the zero lag.  The overlap-count division makes the
    estimate unbiased at large separations (where only a few pairs survive), so
    the correlation length is not biased low by tapering.
    """
    f = np.asarray(field, dtype=float)
    if f.ndim != 2:
        raise ValueError("field must be 2-D (z, y)")
    fluct = f - f.mean()
    nz, ny = fluct.shape
    if not np.any(fluct):
        return np.ones((2 * nz - 1, 2 * ny - 1))
    ac = fftconvolve(fluct, fluct[::-1, ::-1], mode="full")
    mz = np.abs(np.arange(2 * nz - 1) - (nz - 1))
    my = np.abs(np.arange(2 * ny - 1) - (ny - 1))
    counts = np.outer(nz - mz, ny - my).astype(float)
    unbiased = ac / np.maximum(counts, 1.0)
    return unbiased / unbiased[nz - 1, ny - 1]


def radial_profile(
    field_ac: np.ndarray, dy: float, dz: float, *, nbins: int = 24
) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthal average of a centred 2-D autocorrelation."""
    nz2, ny2 = field_ac.shape
    cz, cy = (nz2 - 1) // 2, (ny2 - 1) // 2
    zz = (np.arange(nz2) - cz) * dz
    yy = (np.arange(ny2) - cy) * dy
    zz, yy = np.meshgrid(zz, yy, indexing="ij")
    r = np.sqrt(zz**2 + yy**2)
    rmax = float(np.percentile(r, 95))
    if rmax <= 0.0:
        return np.zeros(0), np.zeros(0)
    edges = np.linspace(0.0, rmax, nbins + 1)
    which = np.digitize(r.ravel(), edges) - 1
    values = field_ac.ravel()
    radii = np.zeros(nbins)
    profile = np.full(nbins, np.nan)
    for b in range(nbins):
        sel = which == b
        if np.any(sel):
            radii[b] = 0.5 * (edges[b] + edges[b + 1])
            profile[b] = values[sel].mean()
    ok = np.isfinite(profile)
    return radii[ok], profile[ok]


def correlation_length(
    radii: np.ndarray, profile: np.ndarray, *, target: float = np.exp(-1.0)
) -> float:
    """First crossing of ``profile`` below ``target``, by linear interpolation."""
    radii = np.asarray(radii)
    profile = np.asarray(profile)
    below = np.nonzero(profile <= target)[0]
    if below.size == 0:
        return float("nan")
    k = int(below[0])
    if k == 0:
        return float(radii[0])
    r0, r1 = radii[k - 1], radii[k]
    p0, p1 = profile[k - 1], profile[k]
    if p1 == p0:
        return float(r1)
    return float(r0 + (target - p0) * (r1 - r0) / (p1 - p0))


def integral_scale(radii: np.ndarray, profile: np.ndarray) -> float:
    """Integral scale ``int_0^{r0} R dr`` up to the first zero crossing."""
    radii = np.asarray(radii)
    profile = np.asarray(profile)
    negative = np.nonzero(profile < 0.0)[0]
    stop = int(negative[0]) if negative.size else len(profile)
    if stop < 2:
        return float("nan")
    return float(np.trapezoid(profile[:stop], radii[:stop]))


def block_maxima(series: np.ndarray, block: int) -> np.ndarray:
    """Non-overlapping block maxima of a 1-D series."""
    s = np.asarray(series, dtype=float)
    if block < 1:
        raise ValueError("block must be >= 1")
    n = len(s) // block
    if n < 1:
        return np.array([s.max()])
    return s[: n * block].reshape(n, block).max(axis=1)


def gumbel_fit(sample: np.ndarray) -> tuple[float, float]:
    """Gumbel (max) parameters ``(mu, beta)`` by the method of moments."""
    s = np.asarray(sample, dtype=float)
    if s.size < 3:
        raise ValueError("Gumbel fit needs at least 3 block maxima")
    beta = float(np.std(s, ddof=1) * np.sqrt(6.0) / np.pi)
    mu = float(s.mean() - 0.5772156649015329 * beta)
    return mu, max(beta, 1e-12)


def gumbel_return_level(mu: float, beta: float, exceedance: float) -> float:
    """Gumbel return level for a per-block exceedance probability ``p``."""
    p = float(exceedance)
    if not 0.0 < p < 1.0:
        raise ValueError("exceedance must be in (0, 1)")
    return float(mu - beta * np.log(-np.log(1.0 - p)))


@dataclass
class PressureStatistics:
    """Summary of a pressure field ``(nt, nz, ny)`` on the structure."""

    mean_field: np.ndarray
    std_field: np.ndarray
    active_mask: np.ndarray
    mean_pressure: float
    peak_field_pressure: float
    peak_pressure: float
    gamma_spatial: float
    gamma_instant: float
    gamma_extreme: float
    autocorrelation: np.ndarray
    radii: np.ndarray
    radial_profile: np.ndarray
    L_c: float
    L_c_y: float
    L_c_z: float
    integral_scale: float
    autocorrelation_fluctuation: np.ndarray
    fluctuation_L_c: float
    peak_factor_cells: np.ndarray
    peak_factor_mean: float
    peak_factor_std: float
    block_peak_factor: float
    gumbel: tuple[float, float]
    loaded_area_fraction: float
    active_area_fraction: float
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "mean_pressure": self.mean_pressure,
            "peak_pressure": self.peak_pressure,
            "peak_field_pressure": self.peak_field_pressure,
            "gamma_spatial": self.gamma_spatial,
            "gamma_instant": self.gamma_instant,
            "gamma_extreme": self.gamma_extreme,
            "L_c": self.L_c,
            "L_c_y": self.L_c_y,
            "L_c_z": self.L_c_z,
            "integral_scale": self.integral_scale,
            "fluctuation_L_c": self.fluctuation_L_c,
            "peak_factor_mean": self.peak_factor_mean,
            "peak_factor_std": self.peak_factor_std,
            "block_peak_factor": self.block_peak_factor,
            "gumbel_mu": self.gumbel[0],
            "gumbel_beta": self.gumbel[1],
            "loaded_area_fraction": self.loaded_area_fraction,
            "active_area_fraction": self.active_area_fraction,
            "meta": self.meta,
        }


def pressure_statistics(
    pressure: np.ndarray,
    dy: float,
    dz: float,
    *,
    transient: float = 0.25,
    active_fraction: float = 0.25,
    design_exceedance: float = 0.1,
) -> PressureStatistics:
    """Full statistical summary of a ``(nt, nz, ny)`` pressure history.

    Parameters
    ----------
    transient
        Fraction of the record discarded before statistics (the initial
        compaction transient).
    active_fraction
        A cell is *active* when its time-mean pressure exceeds this fraction of
        the maximum time-mean pressure.  The inactive tail (structurally
        irrelevant, numerically tiny) is excluded from the peak/mean ratios.
    design_exceedance
        Per-block exceedance probability used for the Gumbel return level.
    """
    p = np.asarray(pressure, dtype=float)
    if p.ndim != 3:
        raise ValueError("pressure must be (nt, nz, ny)")
    nt = p.shape[0]
    start = int(round(transient * (nt - 1)))
    window = p[start:]
    if window.shape[0] < 4:
        raise ValueError("record too short after the transient")
    mean_field = window.mean(axis=0)
    std_field = window.std(axis=0)
    if not np.any(mean_field > 0.0):
        raise ValueError("no contact with the structure")
    active = mean_field >= active_fraction * mean_field.max()
    mean_pressure = float(mean_field[active].mean())
    peak_field_pressure = float(mean_field.max())
    peak_pressure = float(window.max())

    gamma_spatial = peak_field_pressure / max(mean_pressure, 1e-30)
    flat = window[:, active]
    snap_max = flat.max(axis=1)
    snap_mean = flat.mean(axis=1)
    # snapshots with (almost) no load carry no information about localisation;
    # they are excluded instead of producing a 1/0 ratio.
    load = snap_mean > 0.2 * mean_pressure
    if np.any(load):
        gamma_instant = float(np.median(snap_max[load] / snap_mean[load]))
    else:
        gamma_instant = float(peak_field_pressure / max(mean_pressure, 1e-30))
    active_mean_all = float(window[:, active].mean())
    gamma_extreme = float(window.max() / max(active_mean_all, 1e-30))

    ac = spatial_autocorrelation(mean_field)
    radii, profile = radial_profile(ac, dy, dz)
    l_c = correlation_length(radii, profile)
    l_int = integral_scale(radii, profile)

    cz, cy = (ac.shape[0] - 1) // 2, (ac.shape[1] - 1) // 2
    ac_y = ac[cz, cy:]
    ac_z = ac[cz:, cy]
    l_y = correlation_length(np.arange(len(ac_y)) * dy, ac_y)
    l_z = correlation_length(np.arange(len(ac_z)) * dz, ac_z)

    fluct_field = std_field
    ac_fl = spatial_autocorrelation(fluct_field)
    fl_radii, fl_profile = radial_profile(ac_fl, dy, dz)
    fl_lc = correlation_length(fl_radii, fl_profile)

    cell_peak = flat.max(axis=1)
    denominator = np.maximum(mean_field[active], 1e-30)
    peak_factor_cells = flat.max(axis=0) / denominator
    spatial_mean_series = snap_mean
    n_blocks = max(4, min(12, window.shape[0] // 8))
    block = max(1, window.shape[0] // n_blocks)
    maxima = block_maxima(spatial_mean_series, block)
    mu, beta = gumbel_fit(maxima)
    design_level = gumbel_return_level(mu, beta, design_exceedance)
    block_peak_factor = float(design_level / max(spatial_mean_series.mean(), 1e-30))

    return PressureStatistics(
        mean_field=mean_field,
        std_field=std_field,
        active_mask=active,
        mean_pressure=mean_pressure,
        peak_field_pressure=peak_field_pressure,
        peak_pressure=peak_pressure,
        gamma_spatial=float(gamma_spatial),
        gamma_instant=gamma_instant,
        gamma_extreme=gamma_extreme,
        autocorrelation=ac,
        radii=radii,
        radial_profile=profile,
        L_c=l_c,
        L_c_y=float(l_y),
        L_c_z=float(l_z),
        integral_scale=l_int,
        autocorrelation_fluctuation=ac_fl,
        fluctuation_L_c=float(fl_lc),
        peak_factor_cells=peak_factor_cells,
        peak_factor_mean=float(np.mean(peak_factor_cells)),
        peak_factor_std=float(np.std(peak_factor_cells)),
        block_peak_factor=block_peak_factor,
        gumbel=(mu, beta),
        loaded_area_fraction=float(active.mean()),
        active_area_fraction=float(active.mean()),
        meta={
            "samples_used": int(window.shape[0]),
            "samples_total": int(nt),
            "transient_fraction": float(transient),
            "active_fraction_threshold": float(active_fraction),
            "design_exceedance": float(design_exceedance),
            "block_length_samples": int(block),
            "n_blocks": int(len(maxima)),
            "instant_peak_median": float(np.median(snap_max)),
            "spatial_mean_median": float(np.median(spatial_mean_series)),
            "cell_peak_median": float(np.median(cell_peak)),
            "loaded_snapshots": int(np.count_nonzero(load)),
        },
    )