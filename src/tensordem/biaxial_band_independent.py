"""Independent shear-band orientation estimators for the biaxial case (R8).

Why this module exists
----------------------
Every published round of CASE-BIAXIAL-SHEAR-BAND (R4 .. R7) measures the band
inclination with **one** operator:
:func:`tensordem.biaxial_shear_band.radon_band_angle`, a Radon/projection focus
that maximises the fraction ``eta2`` of the ``gamma`` variance explained by
projection-bin means.  The R7 adversarial review
(``docs/ADVERSARIAL_REVIEW_BIAXIAL_SHEAR_BAND_R7.md`` §4.3, item 2) recorded the
consequence as an un-run check: the operator was never reproduced
independently, so an operator-specific bias cannot be separated from the
physical band angle by the published evidence.

This module supplies two estimators built on **different statistics** of the
same local-maximum-shear-strain field ``gamma(x)``:

``band_model_fit``
    Matched-filter / non-linear least-squares fit of an explicit band model,
    ``gamma = c0 + A1 exp(-u1^2 / 2 sigma^2) + A2 exp(-u2^2 / 2 sigma^2)``,
    where ``u1`` is the distance to a line at inclination ``alpha`` and ``u2``
    the distance to its **conjugate** (mirror) line.  The Gaussian ridge profile
    is the standard idealisation of a shear band; the conjugate term makes the
    operator well posed on the X-shaped localisation that post-peak biaxial
    specimens actually show (see :func:`recover_angle_sweep`).  Nothing is
    binned and no variance fraction is maximised: the operator is a model fit,
    and it reports its own goodness of fit ``R^2`` and the conjugate amplitude
    ratio, so a two-band field is visible in the output rather than averaged
    away.

``spectral_band_angle``
    2-D FFT of the ``gamma`` field on a raster whose cell is two particle
    spacings (the lattice period itself is then outside the retained wavevector
    band).  A ridge has its power concentrated on wavevectors perpendicular to
    it, so the **largest** eigenvalue eigenvector of the spectral inertia tensor
    is the band normal.  No particle-level projection and no binning.

Both are reported in the published convention: inclination in degrees from
``+x`` (the ``sigma_3`` axis of the biaxial cell), folded into ``[0, 90]`` by
:func:`tensordem.biaxial_shear_band.fold_angle_deg`, so a band and its conjugate
share an inclination.

Validation
----------
:func:`recover_angle_sweep` measures known inclinations laid on a jittered
triangular lattice -- the geometry class of the specimens -- for a range of band
widths, particle noise levels, and (for the conjugate term) X-shaped
localisation.  It returns the bias and MAE of every operator, including the
published one, so an operator-specific bias is visible *before* the operator is
used on a specimen.  The known limitation found this way is documented in the
module docstring of the R8 report: the spectral operator is single-band only.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .biaxial_shear_band import (BandMeasurementError, fold_angle_deg,
                                 local_shear_strain, radon_band_angle)

__all__ = [
    "band_model_fit",
    "spectral_band_angle",
    "measure_band_independent",
    "synthetic_lattice",
    "synthetic_band_field",
    "recover_angle_sweep",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_samples(x: Sequence[float], y: Sequence[float],
                weight: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    ws = np.asarray(weight, dtype=np.float64).ravel()
    if not (xs.size == ys.size == ws.size):
        raise BandMeasurementError("x, y and weight must have the same length")
    if xs.size < 30:
        raise BandMeasurementError(f"need at least 30 samples, got {xs.size}")
    if not (np.isfinite(xs).all() and np.isfinite(ys).all() and np.isfinite(ws).all()):
        raise BandMeasurementError("samples must be finite")
    if (ws < 0.0).any() or not ws.sum() > 0.0:
        raise BandMeasurementError("weights must be non-negative with a positive sum")
    return xs, ys, ws


def _normal(angle_deg: float) -> tuple[float, float]:
    a = math.radians(float(angle_deg))
    return -math.sin(a), math.cos(a)


# ---------------------------------------------------------------------------
# estimator 1: matched-filter band-model fit (conjugate aware)
# ---------------------------------------------------------------------------
def band_model_fit(x: Sequence[float], y: Sequence[float],
                   weight: Sequence[float], *,
                   alpha_step: float = 1.0,
                   sigma_range: tuple[float, float] = (1.0, 15.0),
                   n_sigma: int = 14,
                   conjugate: bool = True) -> dict[str, Any]:
    """Fit ``gamma = c0 + A1 g(alpha, sigma) + A2 g(conjugate, sigma)``.

    ``g`` is a unit-amplitude Gaussian ridge of width ``sigma`` in the distance
    to a line, ``u1 = (X - <X>) . n(alpha)``; the conjugate ridge uses
    ``n(180 - alpha)``, the mirror inclination, because a biaxial specimen's
    conjugate band shares the folded inclination but not the normal.  For every
    ``(alpha, sigma)`` on the search grid the amplitudes ``(c0, A1, A2)`` are
    obtained by linear least squares, and the grid point with the largest
    ``R^2`` is returned.

    A negative ``A1`` (or ``A2``) is meaningless for a shear ridge, so a ridge
    counts as *present* only when its amplitude is positive; the returned
    ``angle_deg`` is the inclination of the **dominant positive** ridge.

    Returns ``angle_deg``/``folded_angle_deg``, ``r2``, ``sigma``,
    ``amplitudes``, ``conjugate_ratio`` (= min/max of the two ridge
    amplitudes; ~1 means a symmetric X), and the full ``r2_curve`` over alpha at
    the best ``sigma`` for auditing.
    """
    xs, ys, ws = _as_samples(x, y, weight)
    step = float(alpha_step)
    if not step > 0.0:
        raise BandMeasurementError("alpha_step must be positive")
    sigmas = np.geomspace(float(sigma_range[0]), float(sigma_range[1]), int(n_sigma))
    centre = np.array([xs.mean(), ys.mean()])
    d = np.column_stack([xs, ys]) - centre
    total_ss = float(((ws - ws.mean()) ** 2).sum())
    if not total_ss > 0.0:
        raise BandMeasurementError("gamma field has no variance")

    alphas = np.arange(0.0, 90.0, step)
    best = {"r2": -np.inf}
    ones = np.ones(xs.size)
    for alpha in alphas:
        nx, ny = _normal(float(alpha))
        u1 = d[:, 0] * nx + d[:, 1] * ny
        # conjugate line: normal n(180 - alpha) = (-sin a, -cos a)
        u2 = -(d[:, 0] * math.sin(math.radians(float(alpha)))
               + d[:, 1] * math.cos(math.radians(float(alpha))))
        for sigma in sigmas:
            g1 = np.exp(-0.5 * (u1 / sigma) ** 2)
            g2 = (np.exp(-0.5 * (u2 / sigma) ** 2) if conjugate
                  else np.zeros_like(g1))
            design = np.column_stack([ones, g1, g2])
            coef, *_ = np.linalg.lstsq(design, ws, rcond=None)
            resid = ws - design @ coef
            sse = float(resid @ resid)
            r2 = 1.0 - sse / total_ss
            if r2 > best["r2"]:
                best = {"r2": r2, "alpha": float(alpha),
                        "sigma": float(sigma),
                        "amplitudes": [float(v) for v in coef]}

    alpha = best["alpha"]
    a1, a2 = best["amplitudes"][1], best["amplitudes"][2]
    # inclination of the dominant *positive* ridge; its conjugate shares the
    # folded inclination, so either way the reported angle is alpha
    dominant_is_conjugate = bool(a2 > a1)
    ridge_amplitudes = [v for v in (a1, a2) if v > 0.0]
    conjugate_ratio = (float(min(ridge_amplitudes) / max(ridge_amplitudes))
                       if len(ridge_amplitudes) == 2 and max(ridge_amplitudes) > 0.0
                       else (0.0 if len(ridge_amplitudes) < 2 else 1.0))
    return {
        "estimator": "band_model_fit",
        "angle_deg": float(alpha),
        "folded_angle_deg": fold_angle_deg(float(alpha)),
        "r2": float(best["r2"]),
        "sigma": float(best["sigma"]),
        "amplitudes": best["amplitudes"],
        "background": float(best["amplitudes"][0]),
        "dominant_is_conjugate": dominant_is_conjugate,
        "conjugate_ratio": conjugate_ratio,
        "alpha_step_deg": step,
        "sigma_range": [float(sigma_range[0]), float(sigma_range[1])],
    }


# ---------------------------------------------------------------------------
# estimator 2: spectral inertia of the gamma field
# ---------------------------------------------------------------------------
def _raster(xs: np.ndarray, ys: np.ndarray, ws: np.ndarray, cell: float
            ) -> tuple[np.ndarray, float, float]:
    """Mean weight per cell on a grid with cell size ``cell`` (mass units)."""
    width = float(np.ptp(xs))
    height = float(np.ptp(ys))
    if width <= 0.0 or height <= 0.0:
        raise BandMeasurementError("degenerate sample extent")
    nx = int(max(8, round(width / cell)))
    ny = int(max(8, round(height / cell)))
    rng = [[float(xs.min()), float(xs.max())], [float(ys.min()), float(ys.max())]]
    wsum, _, _ = np.histogram2d(xs, ys, bins=[nx, ny], range=rng, weights=ws)
    count, _, _ = np.histogram2d(xs, ys, bins=[nx, ny], range=rng)
    occupied = count > 0
    if occupied.sum() < 16:
        raise BandMeasurementError("too few occupied grid cells for a spectrum")
    field = np.full((nx, ny), float(ws.mean()))
    field[occupied] = wsum[occupied] / count[occupied]
    return field, width / nx, height / ny


def spectral_band_angle(x: Sequence[float], y: Sequence[float],
                        weight: Sequence[float], *,
                        cell: float | None = None,
                        k_cut_fraction: float = 0.45) -> dict[str, Any]:
    """Band inclination from the spectral inertia tensor of the ``gamma`` field.

    ``cell`` is the raster cell size in model length units; ``None`` uses
    ``2 * sqrt(A / N / 0.866)``, i.e. two mean particle spacings of a triangular
    lattice, so that the lattice period itself is safely above the retained
    wavevector band.  Retained wavevectors are
    ``k1 <= |k| <= k_cut_fraction * k_Nyquist_min`` with ``k1`` the fundamental
    of the window: the mean level and the lattice peaks are both excluded.
    A Hann taper suppresses rectangular-window leakage.

    The band normal is the eigenvector of the power-weighted ``k`` covariance
    with the **largest** eigenvalue (power is concentrated on wavevectors
    perpendicular to the ridge); the inclination is that normal minus 90 deg.

    Limitation (measured in :func:`recover_angle_sweep`): a *conjugate* pair
    makes the spectral inertia isotropic in the crossing region, so this
    operator is only valid for a single dominant band (``r2`` /
    ``conjugate_ratio`` of :func:`band_model_fit` are the gate).
    """
    xs, ys, ws = _as_samples(x, y, weight)
    if cell is None:
        area = float(np.ptp(xs) * np.ptp(ys))
        cell = 2.0 * math.sqrt(max(area / xs.size / 0.866, 1e-12))
    if not float(cell) > 0.0:
        raise BandMeasurementError("cell must be positive")
    field, dx, dy = _raster(xs, ys, ws, float(cell))
    nx, ny = field.shape
    tapered = (field - field.mean()) * np.outer(np.hanning(nx), np.hanning(ny))
    power = np.abs(np.fft.fft2(tapered)) ** 2
    kx = 2.0 * math.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * math.pi * np.fft.fftfreq(ny, d=dy)
    kxx, kyy = np.meshgrid(kx, ky, indexing="ij")
    kmag = np.hypot(kxx, kyy)
    k1 = min(2.0 * math.pi / float(np.ptp(xs)), 2.0 * math.pi / float(np.ptp(ys)))
    k_hi = float(k_cut_fraction) * min(np.abs(kx).max(), np.abs(ky).max())
    keep = (kmag >= k1) & (kmag <= k_hi)
    modes = int(keep.sum())
    if modes < 6:
        raise BandMeasurementError(
            f"only {modes} wavevectors retained; raster too coarse "
            f"(cell={cell:.3g}) for this window")
    w = np.where(keep, power, 0.0)
    total = float(w.sum())
    if not total > 0.0:
        raise BandMeasurementError("retained spectrum carries no power")
    w = w / total
    mx = float((w * kxx).sum())
    my = float((w * kyy).sum())
    cxx = float((w * (kxx - mx) ** 2).sum())
    cyy = float((w * (kyy - my) ** 2).sum())
    cxy = float((w * (kxx - mx) * (kyy - my)).sum())
    evals, evecs = np.linalg.eigh(np.array([[cxx, cxy], [cxy, cyy]]))
    normal = evecs[:, int(np.argmax(evals))]
    if normal[1] < 0.0:
        normal = -normal
    normal_angle = math.degrees(math.atan2(float(normal[1]), float(normal[0])))
    band_angle = (normal_angle - 90.0) % 180.0
    anisotropy = (float(evals.max() / evals.min()) if evals.min() > 0.0 else float("inf"))
    return {
        "estimator": "spectral_inertia",
        "angle_deg": float(band_angle),
        "folded_angle_deg": fold_angle_deg(float(band_angle)),
        "normal_angle_deg": float(normal_angle),
        "cell": float(cell),
        "grid": [int(nx), int(ny)],
        "modes": modes,
        "k_anisotropy": anisotropy,
    }


# ---------------------------------------------------------------------------
# measurement wrapper (same interface as ``measure_band``)
# ---------------------------------------------------------------------------
def measure_band_independent(x_ref: np.ndarray, x_now: np.ndarray, *,
                             bounds: tuple[float, float, float, float] | None = None,
                             k: int = 8) -> dict[str, Any]:
    """Measure the band with the published operator and both independent ones.

    ``bounds`` = ``(xmin, xmax, ymin, ymax)`` of the measurement window, as in
    :func:`tensordem.biaxial_shear_band.measure_band`; the boundary layer is
    normally excluded because the rough walls nucleate their own bands.
    """
    ref = np.asarray(x_ref, dtype=np.float64)
    now = np.asarray(x_now, dtype=np.float64)
    if bounds is not None:
        xmin, xmax, ymin, ymax = (float(v) for v in bounds)
        inside = ((ref[:, 0] >= xmin) & (ref[:, 0] <= xmax)
                  & (ref[:, 1] >= ymin) & (ref[:, 1] <= ymax))
    else:
        inside = np.ones(ref.shape[0], dtype=bool)
    if int(inside.sum()) < 30:
        raise BandMeasurementError("too few particles inside the measurement window")
    gamma = local_shear_strain(ref[inside], now[inside], k=k)
    if not np.isfinite(gamma).all() or not gamma.mean() > 0.0:
        raise BandMeasurementError("local strain field is degenerate")
    xs, ys = ref[inside, 0], ref[inside, 1]
    published = radon_band_angle(xs, ys, gamma)
    published = {k: v for k, v in published.items()
                 if k not in ("angles_deg", "eta2_curve")}
    model = band_model_fit(xs, ys, gamma)
    out: dict[str, Any] = {
        "particles": int(inside.sum()),
        "gamma_mean": float(gamma.mean()),
        "localisation_index": float(gamma.std() / gamma.mean()),
        "radon": published,
        "model": model,
        "folded_angles_deg": {
            "radon": float(published["folded_angle_deg"]),
            "model": float(model["folded_angle_deg"]),
        },
    }
    try:
        spectral = spectral_band_angle(xs, ys, gamma)
    except BandMeasurementError as exc:          # coarse window: report, do not fabricate
        out["spectral"] = {"estimator": "spectral_inertia", "error": str(exc)}
    else:
        out["spectral"] = spectral
        out["folded_angles_deg"]["spectral"] = float(spectral["folded_angle_deg"])
    angles = list(out["folded_angles_deg"].values())
    out["spread_deg"] = float(np.ptp(angles)) if len(angles) > 1 else 0.0
    return out


# ---------------------------------------------------------------------------
# synthetic validation on the specimen geometry class
# ---------------------------------------------------------------------------
def synthetic_lattice(*, columns: int = 24, rows: int = 48,
                      spacing: float = 1.5, jitter: float = 0.03,
                      seed: int = 0) -> np.ndarray:
    """Jittered triangular disc lattice, the geometry class of the specimens."""
    rng = np.random.default_rng(int(seed))
    x, y = [], []
    for j in range(int(rows)):
        for i in range(int(columns)):
            x.append((i + 0.5 * (j % 2)) * spacing)
            y.append(j * spacing * math.sqrt(3.0) / 2.0)
    pos = np.column_stack([x, y])
    pos += rng.normal(scale=float(jitter) * spacing, size=pos.shape)
    return pos


def synthetic_band_field(pos: np.ndarray, *, angle_deg: float,
                         half_width: float, amplitude: float = 1.0,
                         contrast: float = 1.0, conjugate_deg: float | None = None,
                         conjugate_fraction: float = 0.0,
                         noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """``gamma`` field of a Gaussian ridge at ``angle_deg`` (+ optional conjugate).

    A shear band is a ridge of local maximum shear strain whose profile across
    the band is roughly Gaussian; ``conjugate_deg`` adds the mirror band at
    ``conjugate_fraction`` of the amplitude, which is how post-peak specimens
    actually localise.
    """
    pos = np.asarray(pos, dtype=np.float64)
    centre = pos.mean(axis=0)
    rng = np.random.default_rng(int(seed))

    def ridge(alpha_deg: float) -> np.ndarray:
        nx, ny = _normal(alpha_deg)
        u = (pos[:, 0] - centre[0]) * nx + (pos[:, 1] - centre[1]) * ny
        return np.exp(-0.5 * (u / float(half_width)) ** 2)

    field = ridge(angle_deg)
    if conjugate_deg is not None and conjugate_fraction > 0.0:
        field = field + float(conjugate_fraction) * ridge(conjugate_deg)
    field = amplitude * (1.0 + contrast * field)
    if noise > 0.0:
        field = field * (1.0 + rng.normal(scale=float(noise), size=field.size))
    return field


def recover_angle_sweep(*, angles_deg: Sequence[float] = (0.0, 15.0, 30.0, 45.0, 52.0,
                                                          60.0, 75.0, 85.0),
                        columns: int = 24, rows: int = 48,
                        half_widths: Sequence[float] = (2.0, 4.0, 7.0),
                        noise: float = 0.05,
                        conjugate_fractions: Sequence[float] = (0.0, 0.7, 1.0),
                        seeds: Sequence[int] = (0, 1, 2)) -> dict[str, Any]:
    """Recover known inclinations with all three operators (bias envelope).

    Every combination of ``half_width`` x ``conjugate_fraction`` is a row of the
    table; for each known angle the per-seed median, bias and MAE of the
    published Radon focus, :func:`band_model_fit` and
    :func:`spectral_band_angle` are reported.  This is the evidence that decides
    which operator may be applied to a specimen field and in which regime.
    """
    cases: list[dict[str, Any]] = []
    for half_width in half_widths:
        for conjugate_fraction in conjugate_fractions:
            table: list[dict[str, Any]] = []
            for alpha in angles_deg:
                per_op: dict[str, list[float]] = {"radon": [], "model": [], "spectral": []}
                for seed in seeds:
                    pos = synthetic_lattice(columns=columns, rows=rows, seed=int(seed))
                    field = synthetic_band_field(
                        pos, angle_deg=float(alpha), half_width=float(half_width),
                        noise=noise,
                        conjugate_deg=180.0 - float(alpha) if conjugate_fraction > 0 else None,
                        conjugate_fraction=float(conjugate_fraction), seed=int(seed))
                    published = radon_band_angle(pos[:, 0], pos[:, 1], field)
                    per_op["radon"].append(float(published["folded_angle_deg"]))
                    per_op["model"].append(float(
                        band_model_fit(pos[:, 0], pos[:, 1], field)["folded_angle_deg"]))
                    try:
                        spectral = spectral_band_angle(pos[:, 0], pos[:, 1], field)
                    except BandMeasurementError:
                        continue
                    per_op["spectral"].append(float(spectral["folded_angle_deg"]))
                entry: dict[str, Any] = {"angle_true_deg": float(alpha)}
                for op, vals in per_op.items():
                    arr = np.asarray(vals, dtype=np.float64)
                    if arr.size == 0:
                        entry[op] = {"median_deg": None, "bias_deg": None,
                                     "mae_deg": None, "values_deg": []}
                        continue
                    entry[op] = {
                        "median_deg": float(np.median(arr)),
                        "values_deg": [float(v) for v in arr],
                        "bias_deg": float(np.median(arr) - float(alpha)),
                        "mae_deg": float(np.mean(np.abs(arr - float(alpha)))),
                    }
                table.append(entry)
            summary = {}
            for op in ("radon", "model", "spectral"):
                mae = [r[op]["mae_deg"] for r in table if r[op]["mae_deg"] is not None]
                bias = [r[op]["bias_deg"] for r in table if r[op]["bias_deg"] is not None]
                summary[op] = {
                    "mae_deg": float(np.mean(mae)) if mae else None,
                    "max_abs_bias_deg": float(np.max(np.abs(bias))) if bias else None,
                }
            cases.append({"half_width": float(half_width),
                          "conjugate_fraction": float(conjugate_fraction),
                          "sweep": table, "summary": summary})
    return {"noise": float(noise), "seeds": [int(s) for s in seeds],
            "cases": cases}