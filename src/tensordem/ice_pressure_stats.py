"""Spatial statistics of the ice-structure contact pressure field and the
ISO 19906 pressure-area law.

Physical problem
----------------
Ice crushing loads are *not* uniformly distributed over a wide structure face.
Two classical observations follow from that:

* the contact pressure field ``p(y, z, t)`` has a finite **spatial correlation
  length** ``L_c`` -- contact is carried by a set of quasi-independent
  high-pressure patches whose extent is set by fragmentation, not by the
  structure size (Sanderson 1988; Sodhi; Dempsey);
* the *design* pressure averaged over an area ``A`` decays with the area,
  ``p_d(A) = p_0 (A / A_0) ** (-n)``, the ISO 19906 pressure-area relation
  (ISO 19906:2019 Annex A; Masterson & Frederking).  ``n`` is small
  (``0.05 .. 0.15`` for most data sets) and is a *consequence* of the spatial
  statistics, not an independent input.

This module makes both quantities **computed** from a DEM contact-pressure
field: it estimates the autocorrelation function (ACF) ``R(dx)``, extracts the
correlation length ``L_c`` from ``R(L_c) = R(0) / e``, forms the area-averaged
design pressure curve over the resolvable area range, and fits ``n``.  Every
number is produced by the estimators below from a DEM field; the ISO 19906 band
is a comparison anchor, never a fitting target.

Estimators
----------
``acf_1d`` / ``acf_2d``
    FFT autocorrelation with **unbiased** count normalisation
    (``sum_k x_k x_{k+d} / (N - d)``), normalised so that ``R(0) = 1`` exactly
    (the value is divided by itself, hence bit-exact, not merely close).
``pooled ACF`` (``PressureRecord.pooled_acf``)
    The ensemble-and-space averaged ACF: the unnormalised covariance of every
    snapshot is accumulated (FFT), summed over snapshots, and only then
    normalised.  This is the estimator that converges in the number of
    snapshots ``N`` -- a single-snapshot ACF is dominated by sampling noise --
    and it is exactly decomposable over time shards (a sum of sums).
``radial_profile`` / ``correlation_length``
    Azimuthal average of the 2-D ACF and its ``1/e`` crossing
    (``R(L_c) = R(0)/e``, linear interpolation).  A field that never decays
    (uniform pressure) returns ``inf`` instead of a spurious finite number.
``patch_average`` / ``pressure_area_curve``
    Exact sliding-window patch mean through an integral image (no wrap-around,
    no FFT leakage), with per-cell areas so that a non-uniform surface grid
    (the curved structures) is handled exactly.  ``p_d(A)`` is the quantile
    (default: the maximum) of the patch-mean pressure over all patch positions
    and all snapshots -- the standard "global pressure over area" definition.
``fit_pressure_area_law``
    Log-log OLS fit of the ISO 19906 form, returning ``p_0``, ``n`` and the
    log-log ``R^2``.  Areas that cannot be formed inside the grid are *dropped*
    (reported as NaN), never silently clamped: clamping would fabricate
    exponent.

DEM scenarios
-------------
``IceStructureCrush`` extends :class:`tensordem.ice_contact.sim.IceWallCrush`
with a rigid **solid of revolution** structure (vertical cylinder, cone).
Contact is resolved against the exact surface of revolution
``r = r_s(z)``: the signed normal distance of every contact point is
``(r - r_s(z)) / sqrt(1 + r_s'(z)**2)`` and the contact normal is the true
surface normal ``n = (r_hat - r_s' z_hat)/sqrt(1 + r_s'**2)``.  The pressure
field is deposited on the unwrapped front-half surface grid ``(theta, z)`` with
per-row cell areas ``r_s(z) dtheta dz``, so the area bookkeeping on the curved
surface is exact.  The flat wall keeps the repository's verified plane contact
(:mod:`tensordem.ice_contact.sim`) unchanged.

Evidence grade: REAL.  Contact forces come from the DEM; the ACF, ``L_c``,
``p_d(A)`` and ``n`` are fitted from those forces.  Known approximation: the
cylinder/cone are modelled as *bosses protruding from the flat wall* (their
axis lies on the wall plane) so that no ice escapes behind the structure; the
curved surface itself is resolved geometrically, but the far-field is the same
bounded slab as the wall case.

Units: SI (Pa, m, s).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .ice_contact.sim import CrushParams, IceWallCrush, pack_slab
from .ice_contact.sim import _crush_law

__all__ = [
    "acf_1d",
    "acf_2d",
    "radial_profile",
    "correlation_length",
    "patch_average",
    "fit_pressure_area_law",
    "PressureAreaFit",
    "PressureRecord",
    "RevolvedStructure",
    "IceStructureCrush",
    "shard_ranges",
    "run_ice_plate_scenario",
]

E_FOLD = 1.0 / math.e


# ----------------------------------------------------------------------
# autocorrelation
# ----------------------------------------------------------------------
def acf_1d(x: np.ndarray, dx: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Unbiased, normalised 1-D autocorrelation of ``x`` via FFT.

    Returns ``(lag, R)`` with ``R[0] == 1.0`` exactly.
    """
    a = np.asarray(x, dtype=float).ravel()
    n = a.size
    if n < 2:
        raise ValueError("need at least two samples")
    a = a - a.mean()
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    spec = np.fft.rfft(a, nfft)
    cov = np.fft.irfft(spec * np.conj(spec), nfft)[:n]
    counts = np.arange(n, 0, -1, dtype=float)
    cov = cov / counts                       # unbiased
    return np.arange(n) * float(dx), _normalised(cov)


def _normalised(cov: np.ndarray) -> np.ndarray:
    """Normalise a covariance to ``R[0] == 1``; a zero-variance field is
    perfectly correlated, so it returns ``R = 1`` everywhere (not an error and
    not a spurious finite length)."""
    c0 = float(cov.flat[0])
    if c0 <= 0.0:
        return np.ones_like(cov)
    return cov / c0


def _lag_counts(nz: int, ny: int) -> np.ndarray:
    return np.outer(np.arange(nz, 0, -1), np.arange(ny, 0, -1)).astype(float)


def _cov2d(a: np.ndarray) -> np.ndarray:
    """Unnormalised (non-circular) 2-D autocovariance, shape ``a.shape``."""
    nz, ny = a.shape
    pad = (2 * nz, 2 * ny)
    spec = np.fft.rfft2(a, pad)
    full = np.fft.irfft2(spec * np.conj(spec), pad)
    return full[:nz, :ny]


def acf_2d(field: np.ndarray) -> np.ndarray:
    """Unbiased, normalised 2-D autocorrelation of one field, ``R[0,0] = 1``."""
    f = np.asarray(field, dtype=float)
    if f.ndim != 2:
        raise ValueError("field must be 2-D")
    f = f - f.mean()
    return _normalised(_cov2d(f) / _lag_counts(*f.shape))


def radial_profile(
    acf: np.ndarray,
    dx: float = 1.0,
    dz: float | None = None,
    n_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthal average of a 2-D ACF: returns ``(r, R(r))``.

    ``r`` is the physical radius ``sqrt((iz*dz)**2 + (iy*dx)**2)``.  Bins are
    uniform in ``r`` with width ``min(dx, dz)`` by default.
    """
    a = np.asarray(acf, dtype=float)
    if a.ndim != 2:
        raise ValueError("acf must be 2-D")
    dz = float(dx if dz is None else dz)
    nz, ny = a.shape
    iz, iy = np.indices(a.shape)
    r = np.sqrt((iz * dz) ** 2 + (iy * dx) ** 2)
    step = min(float(dx), float(dz))
    if n_bins is None:
        n_bins = max(2, int(math.ceil(r.max() / step)))
    edges = np.arange(n_bins + 1) * step
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, n_bins - 1)
    weight = np.bincount(idx, minlength=n_bins).astype(float)
    total = np.bincount(idx, weights=a.ravel(), minlength=n_bins)
    ok = weight > 0.0
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres[ok], total[ok] / weight[ok]


def correlation_length(r: np.ndarray, R: np.ndarray, level: float = E_FOLD) -> float:
    """Radius where ``R`` first drops to ``level * R(0)`` (linear interpolation).

    Returns ``inf`` when ``R`` never decays to that level (e.g. a uniform
    field): the physical statement is "no resolvable correlation length",
    which must not be disguised as a finite number.
    """
    r = np.asarray(r, dtype=float)
    R = np.asarray(R, dtype=float)
    if r.size < 2:
        raise ValueError("need at least two radii")
    target = level * R[0]
    below = np.nonzero(R <= target)[0]
    if below.size == 0:
        return math.inf
    k = int(below[0])
    if k == 0:
        return float(r[0])
    r0, r1 = r[k - 1], r[k]
    R0, R1 = R[k - 1], R[k]
    if R0 == R1:
        return float(r1)
    return float(r0 + (R0 - target) * (r1 - r0) / (R0 - R1))


# ----------------------------------------------------------------------
# pressure-area
# ----------------------------------------------------------------------
def _integral_image(f: np.ndarray) -> np.ndarray:
    s = np.zeros((f.shape[0] + 1, f.shape[1] + 1))
    s[1:, 1:] = np.cumsum(np.cumsum(f, axis=0), axis=1)
    return s


def _window_sums(s: np.ndarray, nz: int, ny: int) -> np.ndarray:
    return (s[nz:, ny:] - s[:-nz, ny:] - s[nz:, :-ny] + s[:-nz, :-ny])


def patch_average(
    field: np.ndarray,
    side: float,
    dx: float,
    dz: float | None = None,
    cell_area: np.ndarray | None = None,
) -> np.ndarray:
    """Sliding rectangular patch mean of ``field`` (pressure) over area ``side**2``.

    Exact through an integral image (no wrap-around).  With ``cell_area``
    given, the mean is the area-weighted mean
    ``sum(p * A) / sum(A)``, which is the correct average on a surface grid
    whose rows have different physical width.
    """
    p = np.asarray(field, dtype=float)
    if p.ndim != 2:
        raise ValueError("field must be 2-D")
    dz = float(dx if dz is None else dz)
    nz, ny = p.shape
    area = np.full(p.shape, dx * dz) if cell_area is None else np.asarray(cell_area, float)
    if area.shape != p.shape:
        raise ValueError("cell_area shape mismatch")
    side = float(side)
    if side <= 0.0:
        raise ValueError("side must be positive")
    nr = min(nz, max(1, int(round(side / dz))))
    nc = min(ny, max(1, int(round(side / dx))))
    sf = _window_sums(_integral_image(p * area), nr, nc)
    sa = _window_sums(_integral_image(area), nr, nc)
    return sf / sa


def fit_pressure_area_law(
    areas: Sequence[float] | np.ndarray,
    p_d: Sequence[float] | np.ndarray,
    A0: float = 1.0,
) -> "PressureAreaFit":
    """Log-log OLS fit of ``p_d = p_0 (A / A_0) ** (-n)``."""
    A = np.atleast_1d(np.asarray(areas, dtype=float))
    p = np.atleast_1d(np.asarray(p_d, dtype=float))
    m = np.isfinite(A) & np.isfinite(p) & (A > 0.0) & (p > 0.0)
    if int(m.sum()) < 3:
        raise ValueError("need at least 3 positive finite (A, p_d) pairs")
    x = np.log(A[m] / A0)
    y = np.log(p[m])
    if np.ptp(x) <= 0.0:
        raise ValueError("areas must span a range")
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return PressureAreaFit(
        p0=float(math.exp(intercept)),
        n=float(-slope),
        r2=float(r2),
        n_points=int(m.sum()),
        A0=float(A0),
    )


@dataclass(frozen=True)
class PressureAreaFit:
    """Result of the log-log fit ``p_d = p_0 (A / A_0) ** (-n)``."""

    p0: float  # design pressure at A = A_0 [Pa]
    n: float  # pressure-area exponent [-]
    r2: float  # coefficient of determination in log-log space [-]
    n_points: int
    A0: float = 1.0

    def predict(self, A: float) -> float:
        return self.p0 * (float(A) / self.A0) ** (-self.n)


# ----------------------------------------------------------------------
# sharding helpers
# ----------------------------------------------------------------------
def shard_ranges(n: int, n_shards: int) -> list[tuple[int, int]]:
    """Contiguous split of ``[0, n)`` into ``n_shards`` blocks (sizes differ <= 1)."""
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    k = min(int(n_shards), int(n))
    base, extra = divmod(int(n), k)
    out: list[tuple[int, int]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        out.append((start, start + size))
        start += size
    return out


# ----------------------------------------------------------------------
# record
# ----------------------------------------------------------------------
@dataclass
class PressureRecord:
    """Time series of structure-surface pressure fields ``p(s, z, t)`` [Pa].

    ``fields``
        ``(T, nz, ns)`` pressure snapshots.
    ``dx, dz``
        nominal cell size along the horizontal (arc-length) and vertical
        directions [m].
    ``cell_area``
        ``(nz, ns)`` physical area of each cell [m^2]; uniform for the flat
        wall, ``r_s(z) dtheta dz`` for the curved structures.
    """

    fields: np.ndarray
    dx: float
    dz: float
    structure: str = "wall"
    cell_area: np.ndarray | None = None
    meta: dict = field(default_factory=dict)
    raw_windows: np.ndarray | None = None   # (T, nz, ns) normal-force sums per window
    raw_steps: np.ndarray | None = None     # (T,) steps in each window

    def __post_init__(self) -> None:
        self.fields = np.asarray(self.fields, dtype=float)
        if self.fields.ndim != 3:
            raise ValueError("fields must be (T, nz, ns)")
        if self.cell_area is None:
            self.cell_area = np.full(self.fields.shape[1:], float(self.dx) * float(self.dz))
        else:
            self.cell_area = np.asarray(self.cell_area, dtype=float)
            if self.cell_area.shape != self.fields.shape[1:]:
                raise ValueError("cell_area shape must match fields[0]")
        if np.any(self.cell_area <= 0.0):
            raise ValueError("cell areas must be positive")

    # -- basics -------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return int(self.fields.shape[0])

    @property
    def mean_cell_area(self) -> float:
        return float(self.cell_area.mean())

    def mean_field(self, n_shards: int = 1) -> np.ndarray:
        """Time-averaged pressure field (shard-exact: sums of blocks)."""
        total = np.zeros_like(self.fields[0])
        for a, b in shard_ranges(self.n_samples, n_shards):
            total += self.fields[a:b].sum(axis=0)
        return total / self.n_samples

    def max_field(self, n_shards: int = 1) -> np.ndarray:
        out = np.full_like(self.fields[0], -np.inf)
        for a, b in shard_ranges(self.n_samples, n_shards):
            np.maximum(out, self.fields[a:b].max(axis=0), out=out)
        return out

    # -- ACF ----------------------------------------------------------
    def _cov_sum(self, start: int, stop: int) -> np.ndarray:
        """Sum of unnormalised snapshot autocovariances over ``[start, stop)``."""
        acc = np.zeros(self.fields.shape[1:])
        for t in range(start, stop):
            f = self.fields[t] - self.fields[t].mean()
            acc += _cov2d(f)
        return acc

    def pooled_cov(self, n_shards: int = 1) -> np.ndarray:
        """Ensemble-averaged covariance sums over all snapshots (shard-exact)."""
        acc = np.zeros(self.fields.shape[1:])
        for a, b in shard_ranges(self.n_samples, n_shards):
            acc += self._cov_sum(a, b)
        return acc

    def pooled_acf(self, n_shards: int = 1, n_samples: int | None = None) -> np.ndarray:
        """Ensemble-and-space averaged, unbiased, normalised 2-D ACF.

        ``R[0, 0] == 1.0`` exactly (division by itself).  ``n_samples`` limits
        the number of snapshots used (for convergence studies).
        """
        T = self.n_samples if n_samples is None else int(n_samples)
        if T > self.n_samples or T < 1:
            raise ValueError("n_samples out of range")
        acc = np.zeros(self.fields.shape[1:])
        for a, b in shard_ranges(T, n_shards):
            acc += self._cov_sum(a, b)
        return _normalised(acc / _lag_counts(*self.fields.shape[1:]) / T)

    def radial_acf(self, n_shards: int = 1, n_samples: int | None = None):
        """``(r, R(r))`` of the pooled ACF (azimuthal average)."""
        return radial_profile(
            self.pooled_acf(n_shards=n_shards, n_samples=n_samples), dx=self.dx, dz=self.dz
        )

    def correlation_length(self, n_shards: int = 1, n_samples: int | None = None) -> float:
        r, R = self.radial_acf(n_shards=n_shards, n_samples=n_samples)
        return correlation_length(r, R)

    def correlation_length_convergence(
        self, counts: Iterable[int] | None = None
    ) -> list[dict]:
        """``L_c`` versus number of snapshots used (convergence evidence)."""
        if counts is None:
            counts = [10, 25, 50, 100, 200, self.n_samples]
        out: list[dict] = []
        prev: float | None = None
        for n in [c for c in counts if c <= self.n_samples]:
            lc = self.correlation_length(n_samples=n)
            change = math.inf if prev is None else abs(lc - prev) / lc
            out.append({"n_samples": int(n), "L_c": float(lc), "rel_change": float(change)})
            prev = lc
        return out

    def directional_acf(self, axis: int = 1, n_shards: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """1-D ACF along ``axis`` of the pooled covariance, normalised to 1.

        ``axis=1`` is the horizontal (arc) direction, ``axis=0`` the vertical.
        The pooled covariance is averaged over the other axis first, so this is
        the space-and-time averaged 1-D ACF -- directly comparable with the
        classic ``R(dx)`` plots.
        """
        cov = self.pooled_cov(n_shards=n_shards) / _lag_counts(*self.fields.shape[1:]) / self.n_samples
        prof = cov.mean(axis=1 - axis)
        step = self.dz if axis == 0 else self.dx
        return np.arange(prof.size) * float(step), prof / prof[0]

    # -- pressure-area ------------------------------------------------
    def area_bounds(self) -> tuple[float, float]:
        """Resolvable patch-area range ``(A_min, A_max)`` of the grid [m^2]."""
        nz, ny = self.fields.shape[1:]
        a_min = float(self.cell_area.min())
        a_max = float(self.cell_area.sum())
        return a_min, a_max

    def _window_field(self, t: int, nr: int, nc: int) -> np.ndarray:
        ca = self.cell_area
        sizes = _window_sums(_integral_image(ca), nr, nc)
        return _window_sums(_integral_image(self.fields[t] * ca), nr, nc) / sizes

    def _patch_geometry(self, A: float) -> tuple[int, int] | None:
        nz, ny = self.fields.shape[1:]
        a_min, a_max = self.area_bounds()
        if not np.isfinite(A) or A <= 0.0:
            raise ValueError("areas must be positive and finite")
        if A < 0.5 * a_min or A > a_max:
            return None
        nr = min(nz, max(1, int(round(math.sqrt(A) / self.dz))))
        nc = min(ny, max(1, int(round(A / (nr * self.dz * self.dx)))))
        return nr, nc

    def peak_patch_field(self, A: float, n_shards: int = 1) -> tuple[np.ndarray, float] | None:
        """Per-position maximum over time of the area-``A`` patch mean pressure.

        Returns ``(field, A_eff)`` where ``field[i, j]`` is the largest patch
        mean pressure ever seen with its upper-left cell at ``(i, j)``.  The
        maximum is associative, so the result is exact under time sharding.
        """
        geom = self._patch_geometry(A)
        if geom is None:
            return None
        nr, nc = geom
        ca = self.cell_area
        sizes = _window_sums(_integral_image(ca), nr, nc)
        A_eff = float(sizes.mean())
        out = np.full(sizes.shape, -np.inf)
        for a, b in shard_ranges(self.n_samples, n_shards):
            for t in range(a, b):
                w = _window_sums(_integral_image(self.fields[t] * ca), nr, nc) / sizes
                np.maximum(out, w, out=out)
        return out, A_eff

    def pressure_area_curve(
        self,
        areas: Sequence[float] | np.ndarray,
        reduce: str = "mean",
        n_shards: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Design pressure ``p_d(A)`` for each requested area [m^2].

        ``p_d(A)`` is built from the per-position peak-over-time patch-mean
        field (:meth:`peak_patch_field`) and reduced over patch positions:

        ``reduce="mean"`` (default)
            the **typical** peak pressure of an area-``A`` patch -- the direct
            analogue of fitting a power law to an ensemble of measured peak
            pressures on structures of nominal area ``A``.
        ``reduce="max"``
            the worst patch on the face (upper envelope).
        ``reduce="median"``
            the median patch (robust to a single hot spot).

        Returns ``(A_eff, p_d, n_windows)``.  ``A_eff`` is the *measured* mean
        patch area; areas that cannot be formed inside the grid return ``NaN``
        -- they are never clamped.
        """
        areas = np.atleast_1d(np.asarray(areas, dtype=float))
        A_eff = np.full(areas.size, np.nan)
        p_d = np.full(areas.size, np.nan)
        n_win = np.zeros(areas.size, dtype=int)
        for k, A in enumerate(areas):
            got = self.peak_patch_field(A, n_shards=n_shards)
            if got is None:
                continue
            field, a_eff = got
            vals = field[np.isfinite(field)]
            if vals.size == 0:
                continue
            A_eff[k] = a_eff
            n_win[k] = int(vals.size)
            if reduce == "max":
                p_d[k] = float(vals.max())
            elif reduce == "median":
                p_d[k] = float(np.median(vals))
            elif reduce == "mean":
                p_d[k] = float(vals.mean())
            else:
                raise ValueError(f"unknown reduce {reduce!r}")
        return A_eff, p_d, n_win

    def fit_pressure_area(
        self,
        areas: Sequence[float] | np.ndarray,
        reduce: str = "mean",
        n_shards: int = 1,
        A0: float = 1.0,
    ) -> tuple[PressureAreaFit, np.ndarray, np.ndarray]:
        A_eff, p_d, _ = self.pressure_area_curve(areas, reduce=reduce, n_shards=n_shards)
        fit = fit_pressure_area_law(A_eff, p_d, A0=A0)
        return fit, A_eff, p_d

    # -- windowing ----------------------------------------------------
    def slice(self, start: int, stop: int | None = None, structure: str | None = None) -> "PressureRecord":
        """Time sub-window ``[start, stop)`` as a new record (for stationarity)."""
        stop = self.n_samples if stop is None else int(stop)
        if not 0 <= int(start) < stop <= self.n_samples:
            raise ValueError("invalid window")
        raw = None if self.raw_windows is None else self.raw_windows[int(start):stop]
        steps = None if self.raw_steps is None else self.raw_steps[int(start):stop]
        meta = dict(self.meta)
        meta["window"] = [int(start), int(stop)]
        return PressureRecord(
            fields=self.fields[int(start):stop],
            dx=self.dx,
            dz=self.dz,
            structure=self.structure if structure is None else structure,
            cell_area=self.cell_area,
            meta=meta,
            raw_windows=raw,
            raw_steps=steps,
        )

    # -- coarse-graining ----------------------------------------------
    def regrain(self, coarse_length: float) -> "PressureRecord":
        """Re-coarse-grain the stored raw force windows at another scale.

        ``p = G_sigma * F / (G_sigma * A) / n_window`` -- the same normalised
        convolution the DEM run used, evaluated post hoc, so the sensitivity of
        the statistics to the coarse-graining scale costs no new DEM run.
        """
        if self.raw_windows is None or self.raw_steps is None:
            raise RuntimeError("raw windows were not stored for this record")
        from scipy.ndimage import gaussian_filter

        sig = float(coarse_length) / float(self.dz)
        area = np.asarray(self.cell_area, dtype=float)
        if sig > 1e-9:
            den = gaussian_filter(area, sig, mode="constant")
        else:
            den = area
        fields = np.empty_like(self.raw_windows)
        for t in range(self.raw_windows.shape[0]):
            num = (
                gaussian_filter(self.raw_windows[t], sig, mode="constant")
                if sig > 1e-9
                else self.raw_windows[t]
            )
            fields[t] = num / np.maximum(den, 1e-30) / float(self.raw_steps[t])
        meta = dict(self.meta)
        meta["coarse_length"] = float(coarse_length)
        return PressureRecord(
            fields=fields,
            dx=self.dx,
            dz=self.dz,
            structure=self.structure,
            cell_area=self.cell_area,
            meta=meta,
            raw_windows=self.raw_windows,
            raw_steps=self.raw_steps,
        )

    # -- provenance ---------------------------------------------------
    def summary(self) -> dict:
        a_min, a_max = self.area_bounds()
        lc = self.correlation_length()
        return {
            "structure": self.structure,
            "n_samples": self.n_samples,
            "shape": [int(v) for v in self.fields.shape],
            "dx": float(self.dx),
            "dz": float(self.dz),
            "mean_cell_area": self.mean_cell_area,
            "area_range": [a_min, a_max],
            "p_max": float(self.fields.max()),
            "p_mean": float(self.fields.mean()),
            "L_c": None if math.isinf(lc) else lc,
            **{k: v for k, v in self.meta.items() if k not in ("structure",)},
        }


# ----------------------------------------------------------------------
# curved structures: rigid solid of revolution
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RevolvedStructure:
    """Rigid solid of revolution about a vertical axis on the wall plane.

    ``r_s(z) = radius + (z_ref - z) * tan(half_angle)``: ``half_angle = 0``
    gives a vertical cylinder, ``half_angle > 0`` a cone with its apex up
    (radius decreasing with height).
    """

    kind: str = "cylinder"  # "cylinder" | "cone"
    radius: float = 0.4  # radius at z = z_ref [m]
    z_ref: float = 0.5  # reference height [m]
    half_angle_deg: float = 0.0  # cone half angle [deg]
    y_center: float = 1.0  # axis lateral position [m]
    x_center: float = 0.0  # axis distance from the wall plane [m]

    def __post_init__(self) -> None:
        if self.kind not in ("cylinder", "cone"):
            raise ValueError("kind must be 'cylinder' or 'cone'")
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")
        if not 0.0 <= self.half_angle_deg < 90.0:
            raise ValueError("half_angle_deg must be in [0, 90)")

    @property
    def slope(self) -> float:
        """``dr_s/dz`` (negative for an apex-up cone)."""
        return -math.tan(math.radians(self.half_angle_deg))

    def radius_at(self, z: np.ndarray | float) -> np.ndarray | float:
        """Surface radius at height ``z`` [m]."""
        return self.radius + (self.z_ref - z) * math.tan(math.radians(self.half_angle_deg))


class IceStructureCrush(IceWallCrush):
    """DEM of an ice slab crushing against a rigid solid of revolution.

    The structure contact is resolved on the exact surface of revolution; the
    flat wall at ``x = 0`` still acts as the backing plane (so no ice escapes
    behind the structure) but contributes nothing to the recorded pressure
    field, which lives on the unwrapped front-half surface grid.
    """

    def __init__(
        self,
        *args,
        structure: RevolvedStructure | None = None,
        coarse_length: float = 0.10,
        store_raw: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.revolved = structure
        self.coarse_length = float(coarse_length)
        self.store_raw = bool(store_raw)
        self._last_record_step = 0
        self.raw_windows: list[np.ndarray] = []
        self.raw_steps: list[int] = []
        if structure is not None:
            self._setup_surface_grid()

    # -- grid ---------------------------------------------------------
    def _setup_surface_grid(self) -> None:
        st = self.revolved
        assert st is not None
        self.nz = max(1, int(round(self.H / self.dz)))
        self.dz_eff = self.H / self.nz
        self.nth = max(8, int(round(math.pi * st.radius / self.dy)))
        self.dtheta = math.pi / self.nth
        z_c = (np.arange(self.nz) + 0.5) * self.dz_eff
        widths = np.asarray(st.radius_at(z_c), dtype=float) * self.dtheta  # arc per cell
        self.cell_area_rows = widths * self.dz_eff
        self.cell_area_field = np.repeat(self.cell_area_rows[:, None], self.nth, axis=1)
        self.cell_area = float(self.cell_area_field.mean())
        self.ny = self.nth
        self.y_center = st.y_center
        self.x_center = st.x_center
        # ``_p_acc`` accumulates the normal FORCE per cell [N]; pressure is
        # formed at record time by dividing by the per-cell area.
        self._p_acc = np.zeros((self.nz, self.nth))

    # -- recording ----------------------------------------------------
    def _pressure_now(self, step: int) -> np.ndarray:
        """Time-averaged, coarse-grained pressure of the window ending at ``step``.

        Two conventions matter for the absolute magnitude:

        * the accumulator holds the **sum of normal forces** over the window,
          so it is divided by the number of steps in the window -- the record
          is the *time-averaged* pressure, i.e. directly comparable with a
          measured crushing pressure [Pa];
        * the force density is coarse-grained with a Gaussian kernel of
          ``coarse_length`` (normalised convolution, ``G*F / G*A``), because a
          single contact force deposited on one grid cell is a discretisation
          spike, not a continuum pressure; the kernel is the scale at which
          the DEM field is interpreted as a stress field.
        """
        area = (
            np.asarray(self.cell_area_field, dtype=float)
            if self.revolved is not None
            else np.full((self.nz, self.ny), float(self.cell_area))
        )
        npers = max(1, int(step - self._last_record_step))
        sig = self.coarse_length / (self.dz_eff if self.revolved is not None else self.dz)
        if sig > 1e-9:
            from scipy.ndimage import gaussian_filter

            num = gaussian_filter(self._p_acc, sig, mode="constant")
            den = gaussian_filter(area, sig, mode="constant")
            p = num / np.maximum(den, 1e-30)
        else:
            p = self._p_acc / area
        return np.asarray(p, dtype=float) / npers

    def _record(self, step: int, wall_normal: float, platen_normal: float) -> None:  # noqa: D401
        if self.store_raw:
            self.raw_windows.append(self._p_acc.copy())
            self.raw_steps.append(max(1, int(step - self._last_record_step)))
        self.pressure.append(self._pressure_now(step))
        self._last_record_step = int(step)
        self._p_acc[:] = 0.0
        self.force_history.append(float(wall_normal))
        self.tangential.append(self._ft_acc.copy())
        self._ft_acc[:] = 0.0
        self.platen_force_history.append(float(platen_normal))
        self.ke_history.append(self.kinetic_energy())
        self.time_history.append(step * self.dt)

    # -- surface deposition -------------------------------------------
    def _deposit_surface(self, point: np.ndarray, weight: np.ndarray) -> None:
        if weight.size == 0:
            return
        x = point[:, 0] - self.x_center
        y = point[:, 1] - self.y_center
        z = point[:, 2]
        keep = (x > 0.0) & (z >= 0.0) & (z <= self.H)
        if not np.any(keep):
            return
        x, y, z, weight = x[keep], y[keep], z[keep], weight[keep]
        theta = np.arctan2(y, x)  # front half: (-pi/2, pi/2)
        fi = (theta + 0.5 * math.pi) / self.dtheta - 0.5
        fj = z / self.dz_eff - 0.5
        ii = np.floor(fi).astype(int)
        jj = np.floor(fj).astype(int)
        ti = fi - ii
        tj = fj - jj
        for oi, oj, wgt in ((0, 0, (1 - ti) * (1 - tj)), (1, 0, ti * (1 - tj)),
                            (0, 1, (1 - ti) * tj), (1, 1, ti * tj)):
            a = ii + oi
            b = jj + oj
            ok = (a >= 0) & (a < self.nth) & (b >= 0) & (b < self.nz)
            if not np.any(ok):
                continue
            flat = b[ok] * self.nth + a[ok]
            self._p_acc += np.bincount(
                flat, weights=(weight * wgt)[ok], minlength=self.nz * self.nth
            ).reshape(self.nz, self.nth)

    # -- structure contact --------------------------------------------
    def _plane_contact(self, normal, offset, plane_velocity, mu, *, structure=False, world=None):  # noqa: D401
        if not structure or self.revolved is None:
            return super()._plane_contact(
                normal, offset, plane_velocity, mu, structure=structure, world=world
            )
        if world is None:
            world = self._contact_world()
        # backing plane: reacts the ice, contributes no recorded pressure
        f_plane = super()._plane_contact(
            normal, offset, plane_velocity, mu, structure=False, world=world
        )
        f_boss = self._surface_contact(world)
        return f_plane + f_boss

    def _surface_contact(self, world: np.ndarray) -> float:
        st = self.revolved
        assert st is not None
        x = world[:, 0] - st.x_center
        y = world[:, 1] - st.y_center
        z = world[:, 2]
        r = np.maximum(np.hypot(x, y), 1e-12)
        rs = np.asarray(st.radius_at(z), dtype=float)
        drs = st.slope
        lam = math.sqrt(1.0 + drs * drs)
        dist = (r - rs) / lam  # signed distance, positive outside the solid
        delta = self.cp_rad - dist
        active = delta > 0.0
        if not np.any(active):
            return 0.0
        idx = np.nonzero(active)[0]
        pid = self.cp_pid[idx]
        delta = delta[idx]
        area = self.cp_area[idx]
        share = area / np.maximum(self.cp_ref[idx], 1e-30)
        rr = r[idx]
        nrm = np.stack(
            [x[idx] / rr / lam, y[idx] / rr / lam, np.full(rr.size, -drs / lam)], axis=1
        )
        point = world[idx] - dist[idx][:, None] * nrm
        lever = point - self.pos[pid]
        v_rel = self.vel[pid] + np.cross(self.omega[pid], lever)
        vn = np.einsum("ij,ij->i", v_rel, nrm)
        damp = 2.0 * self.params.zeta * np.sqrt(self.params.k_n * self.mass[pid] * share)
        f_n = _crush_law(
            self.params.k_n * delta * share - damp * vn,
            self.params.sigma_c * area,
            self.params.crush_hardening,
        )
        v_t = v_rel - vn[:, None] * nrm
        v_t_mag = np.linalg.norm(v_t, axis=1)
        f_t = (
            -self.params.mu_wall
            * f_n[:, None]
            * np.tanh(v_t_mag / self.params.v_smooth)[:, None]
            * (v_t / np.maximum(v_t_mag, 1e-12)[:, None])
        )
        f = f_n[:, None] * nrm + f_t
        self._scatter3(self.force, pid, f)
        self._scatter3(self.torque, pid, np.cross(lever, f))
        self._ft_acc += f_t.sum(axis=0)
        self._deposit_surface(point, f_n)
        return float(f_n.sum())

    def surface_pressure(self) -> np.ndarray:
        """Accumulated surface pressure ``p(theta, z)`` since the last record."""
        area = self.cell_area_field if self.revolved is not None else self.cell_area
        return self._p_acc / area


# ----------------------------------------------------------------------
# scenario driver
# ----------------------------------------------------------------------
def run_ice_plate_scenario(
    structure: str = "wall",
    *,
    seed: int = 20261109,
    n_max: int = 240,
    thickness: float = 1.0,
    width: float = 2.0,
    length_factor: float = 1.0,
    drive_speed: float = 1.0,
    stroke: float = 0.08,
    dt: float = 6.0e-5,
    save_every: int | None = None,
    grid: tuple[float, float] = (0.05, 0.05),
    struct_radius: float = 0.40,
    cone_half_angle_deg: float = 30.0,
    sigma_c: float = 8.0e6,
    k_n: float = 5.0e6,
    mass_scale: float = 1.0,
    zeta: float = 0.30,
    bulk_damping: float = 12.0,
    solid_fraction: float = 0.55,
    settle_steps: int = 200,
    coarse_length: float = 0.10,
    keep_raw: bool = True,
) -> PressureRecord:
    """Run one ice-slab vs rigid-structure DEM case and return its record.

    ``structure`` is ``"wall"`` (flat plane: the verified IN27-S2 configuration),
    ``"cylinder"`` (vertical cylinder of radius ``struct_radius``) or ``"cone"``
    (cone with ``cone_half_angle_deg``, apex up).  The assembly is the dense
    jittered-lattice crushed-ice slab of
    :func:`tensordem.ice_contact.study.build_assembly`; the structure is rigid.
    For the curved structures the fragments that start inside the solid are
    removed before the run (the DEM resolves the residual overlaps).
    """
    if structure not in ("wall", "cylinder", "cone"):
        raise ValueError(f"unknown structure {structure!r}")
    from .ice_contact.shapes import sphere_shape
    from .ice_contact.study import build_assembly

    box = (max(length_factor * thickness, 0.45), float(width), float(thickness))
    volumes, positions, d50 = build_assembly(
        seed, box, n_max=int(n_max), solid_fraction=float(solid_fraction)
    )
    shapes = [sphere_shape(float(v)) for v in volumes]

    spec = None
    if structure in ("cylinder", "cone"):
        spec = RevolvedStructure(
            kind=structure,
            radius=float(struct_radius),
            z_ref=0.5 * float(thickness),
            half_angle_deg=float(cone_half_angle_deg) if structure == "cone" else 0.0,
            y_center=0.5 * float(width),
            x_center=0.0,
        )
        rs = np.asarray(spec.radius_at(positions[:, 2]), dtype=float)
        rr = np.hypot(positions[:, 0] - spec.x_center, positions[:, 1] - spec.y_center)
        keep = rr > rs + np.array([s.r_circ for s in shapes])
        shapes = [s for s, k in zip(shapes, keep) if k]
        positions = positions[keep]
        if len(shapes) < 20:
            raise RuntimeError("structure removed too many fragments")

    n_steps = int(round(stroke / (drive_speed * dt)))
    save_every = int(save_every) if save_every else max(1, n_steps // 100)
    sim = IceStructureCrush(
        shapes,
        params=CrushParams(
            sigma_c=float(sigma_c),
            k_n=float(k_n),
            mass_scale=float(mass_scale),
            zeta=float(zeta),
            bulk_damping=float(bulk_damping),
        ),
        box=box,
        drive_speed=float(drive_speed),
        seed=seed,
        grid=grid,
        positions=positions,
        stroke=float(stroke),
        dt=float(dt),
        save_every=int(save_every),
        settle_steps=int(settle_steps),
        structure=spec,
        coarse_length=float(coarse_length),
    ).run()
    p = np.asarray(sim.pressure_array, dtype=float)
    if p.size == 0 or not np.any(p > 0.0):
        raise RuntimeError(f"scenario {structure!r} produced no contact pressure")
    cell_area = (
        np.full(p.shape[1:], float(sim.cell_area)) if spec is None
        else np.array(sim.cell_area_field, dtype=float)
    )
    rec = PressureRecord(
        fields=p,
        dx=float(sim.dy),
        dz=float(sim.dz_eff if spec is not None else sim.dz),
        structure=structure,
        cell_area=cell_area,
        meta={
            "scenario": structure,
            "seed": int(seed),
            "n_fragments": int(sim.n),
            "box": [float(v) for v in box],
            "drive_speed": float(drive_speed),
            "stroke": float(stroke),
            "dt": float(dt),
            "save_every": int(save_every),
            "sigma_c": float(sigma_c),
            "k_n": float(k_n),
            "mass_scale": float(mass_scale),
            "zeta": float(zeta),
            "bulk_damping": float(bulk_damping),
            "solid_fraction": float(solid_fraction),
            "coarse_length": float(coarse_length),
            "pressure_normalisation": "time-averaged over the save window, "
            "Gaussian-coarse-grained force density (G*F / G*A)",
            "d50_ratio_to_thickness": 0.18,
            "struct_radius": float(struct_radius),
            "cone_half_angle_deg": float(cone_half_angle_deg) if structure == "cone" else 0.0,
            "t_end": float(sim.steps * sim.dt),
            "n_snapshots": int(p.shape[0]),
            "wall_force_max": float(np.max(sim.force_history)) if sim.force_history else 0.0,
            "platen_force_max": float(np.max(sim.platen_force_history))
            if sim.platen_force_history
            else 0.0,
        },
    )
    if len(sim.raw_windows):
        rec.raw_windows = np.asarray(sim.raw_windows, dtype=float)
        rec.raw_steps = np.asarray(sim.raw_steps, dtype=int)
    if keep_raw:
        rec.meta["force_history_max"] = float(np.max(sim.force_history))
    return rec
