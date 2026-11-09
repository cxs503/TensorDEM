"""Spatial correlation function of the DEM pressure field and its fitted model.

A random-field load model needs two things before it can generate anything:
the **variance** of the pressure fluctuation and its **spatial correlation
function** ``R(dx, dy)``.  Both are estimated here from the DEM snapshots (no
assumed form), and then a small family of admissible models is fitted:

``exponential``  ``R = exp(-|dx|/Lx) exp(-|dy|/Ly)``
``matern``       separable product of 1D Matern kernels with a shared
                 smoothness ``nu``; ``nu = 1/2`` is the exponential, and larger
                 ``nu`` is smoother.
``matern_radial`` anisotropic radial Matern (one smoothness, elliptical
                 scaling) - reported for completeness; it is *not* used for the
                 KL expansion because a radial kernel is not separable.

Estimator
---------
The DEM cell is periodic in ``x`` and ``y`` (see
:mod:`tensordem.stochastic_pressure_field.dem_field`), so the field is
periodic too and the **circular** autocorrelation is the exact estimator - no
tapering, no zero padding, no window bias:

.. math::

    R(dx, dy) = \\frac{\\langle \\tilde p (x,y) \\tilde p (x+dx, y+dy)\\rangle}
                     {\\langle \\tilde p^2 \\rangle},
    \\qquad \\tilde p = p - \\langle p \\rangle

with the average over the grid *and* over snapshots.  That is the field's
two-point statistic; the fit residual is the only place where a parametric
assumption enters, and it is reported (``R2``, ``rmse``) wherever it is used.

Definition of the correlation length
-----------------------------------
For the exponential model the scale parameter *is* the ``1/e`` length.  For the
Matern model the scale parameter ``L`` is the standard Matern length; the
``1/e`` crossing is reported separately (``corr_length_1e``) so that the number
that enters the report is comparable between models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy import special
from scipy.optimize import least_squares

__all__ = [
    "CorrelationFit",
    "autocorrelation_2d",
    "fit_correlation_model",
    "matern_1d",
    "separable_covariance",
    "fitted_2d_model",
]


def autocorrelation_2d(fields: np.ndarray) -> np.ndarray:
    """Circular (periodic) 2D autocorrelation of a stack of fields.

    ``fields`` has shape ``(n_snap, ny, nx)``.  The fluctuation about the
    **ensemble mean field** is correlated circularly with itself (exact for a
    periodic field) and averaged over snapshots, normalised so ``R[0, 0] = 1``.

    Subtracting each snapshot's *own* spatial mean would remove the ``k = 0``
    mode of the expansion and make every field look shorter-correlated than it
    is (measured: an O(1) bias in ``R`` at the first lag); the ensemble mean is
    the correct reference because it carries the (possibly spatially
    non-uniform) mean pressure field.
    """
    f = np.asarray(fields, dtype=float)
    if f.ndim != 3:
        raise ValueError("fields must be (n_snap, ny, nx)")
    if f.shape[0] < 1:
        raise ValueError("need at least one snapshot")
    ny, nx = f.shape[1], f.shape[2]
    acc = np.zeros((ny, nx))
    var = 0.0
    mean_field = f.mean(axis=0)
    for snap in f:
        fl = snap - mean_field
        F = np.fft.rfft2(fl)
        ac = np.fft.irfft2(F * np.conj(F), s=(ny, nx)) / float(nx * ny)
        acc += ac
        var += float(np.mean(fl ** 2))
    acc /= float(f.shape[0])
    var /= float(f.shape[0])
    if var <= 0.0:
        raise ValueError("field has zero variance")
    return acc / var


def matern_1d(d: np.ndarray, length: float, nu: float) -> np.ndarray:
    """Matern correlation function of a scalar distance argument."""
    d = np.abs(np.asarray(d, dtype=float))
    length = float(length)
    nu = float(nu)
    if length <= 0.0:
        return (d <= 0.0).astype(float)
    x = np.sqrt(2.0 * nu) * d / length
    # value at x = 0 is 1 exactly
    out = np.ones_like(x, dtype=float)
    pos = x > 0.0
    if np.any(pos):
        out[pos] = (2.0 ** (1.0 - nu) / special.gamma(nu)) * x[pos] ** nu * special.kv(nu, x[pos])
    return np.clip(out, 0.0, 1.0)


def separable_covariance(lx: np.ndarray, ly: np.ndarray, length_x: float,
                         length_y: float, nu: float = 0.5) -> np.ndarray:
    """Separable covariance matrix ``Cx (x) Cy`` on two coordinate vectors."""
    dx = np.abs(lx[:, None] - lx[None, :])
    dy = np.abs(ly[:, None] - ly[None, :])
    if nu == 0.5:
        cx = np.exp(-dx / max(length_x, 1e-12))
        cy = np.exp(-dy / max(length_y, 1e-12))
    else:
        cx = matern_1d(dx, length_x, nu)
        cy = matern_1d(dy, length_y, nu)
    return np.kron(cx, cy)


@dataclass(frozen=True)
class CorrelationFit:
    """Fitted correlation model and its goodness of fit."""

    model: str
    length_x: float
    length_y: float
    nu: float
    r2: float
    rmse: float
    max_abs_error: float
    corr_length_1e_x: float
    corr_length_1e_y: float
    variance: float
    n_lag_cells: int
    lag_limit_x: float
    lag_limit_y: float
    params: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "model": self.model,
            "length_x": self.length_x,
            "length_y": self.length_y,
            "nu": self.nu,
            "r2": self.r2,
            "rmse": self.rmse,
            "max_abs_error": self.max_abs_error,
            "corr_length_1e_x": self.corr_length_1e_x,
            "corr_length_1e_y": self.corr_length_1e_y,
            "variance": self.variance,
            "n_lag_cells": self.n_lag_cells,
            "lag_limit_x": self.lag_limit_x,
            "lag_limit_y": self.lag_limit_y,
            **self.params,
        }


def _predict(model: str, dx: np.ndarray, dy: np.ndarray, p: Iterable[float]) -> np.ndarray:
    p = list(p)
    if model == "exponential":
        lx, ly = p
        return np.exp(-np.abs(dx) / lx - np.abs(dy) / ly)
    if model == "matern":
        lx, ly, nu = p
        return matern_1d(dx, lx, nu) * matern_1d(dy, ly, nu)
    if model == "matern_radial":
        lx, ly, nu = p
        r = np.sqrt((dx / lx) ** 2 + (dy / ly) ** 2)
        return matern_1d(r, 1.0, nu)
    raise ValueError(f"unknown correlation model: {model}")


def _r2(data: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> Tuple[float, float, float]:
    resid = data - pred
    ss_res = float(np.sum(weights * resid ** 2))
    mean = float(np.sum(weights * data) / np.sum(weights))
    ss_tot = float(np.sum(weights * (data - mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return r2, rmse, float(np.max(np.abs(resid)))


def fit_correlation_model(
    autocorr: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    *,
    models: Tuple[str, ...] = ("exponential", "matern", "matern_radial"),
    lag_limit_fraction: float = 0.5,
) -> Dict[str, CorrelationFit]:
    """Fit correlation models to an estimated ``R(dx, dy)`` table.

    Lags are restricted to ``lag_limit_fraction`` of the domain in each
    direction: beyond that a periodic estimator has too few independent pairs
    (the same lag appears only once around the ring) and the estimate is
    biased low.
    """
    ac = np.asarray(autocorr, dtype=float)
    ny, nx = ac.shape
    dx_grid = (np.arange(nx) - 0) * float(x_centers[1] - x_centers[0])
    dy_grid = (np.arange(ny) - 0) * float(y_centers[1] - y_centers[0])
    # circular lags -> signed lags (smallest image)
    dx_signed = np.where(dx_grid <= 0.5 * nx * (x_centers[1] - x_centers[0]),
                         dx_grid, dx_grid - nx * (x_centers[1] - x_centers[0]))
    dy_signed = np.where(dy_grid <= 0.5 * ny * (y_centers[1] - y_centers[0]),
                         dy_grid, dy_grid - ny * (y_centers[1] - y_centers[0]))
    DX, DY = np.meshgrid(np.abs(dx_signed), np.abs(dy_signed))
    lx_max = lag_limit_fraction * float(np.max(np.abs(dx_signed)))
    ly_max = lag_limit_fraction * float(np.max(np.abs(dy_signed)))
    mask = (DX <= lx_max) & (DY <= ly_max)
    # weight by the number of periodic lag pairs available (edges have few)
    counts = np.where(DX == 0, 1, 2) * np.where(DY == 0, 1, 2)
    counts = np.broadcast_to(counts, (ny, nx)).astype(float)
    # number of grid cells realising a given (dx, dy) lag: (nx - dx)(ny - dy)
    cells = (nx - np.round(DX / (x_centers[1] - x_centers[0]))) * \
            (ny - np.round(DY / (y_centers[1] - y_centers[0])))
    weights = np.where(mask, np.maximum(cells, 1.0), 0.0)

    data = ac
    x0_map = {
        "exponential": [max(lx_max / 3.0, 1e-6), max(ly_max / 3.0, 1e-6)],
        "matern": [max(lx_max / 3.0, 1e-6), max(ly_max / 3.0, 1e-6), 1.0],
        "matern_radial": [max(lx_max / 3.0, 1e-6), max(ly_max / 3.0, 1e-6), 1.0],
    }
    bounds_map = {
        "exponential": ([1e-6, 1e-6], [10.0 * lx_max, 10.0 * ly_max]),
        "matern": ([1e-6, 1e-6, 0.55], [10.0 * lx_max, 10.0 * ly_max, 6.0]),
        "matern_radial": ([1e-6, 1e-6, 0.55], [10.0 * lx_max, 10.0 * ly_max, 6.0]),
    }

    m = mask & (weights > 0)
    out: Dict[str, CorrelationFit] = {}
    for model in models:
        lo, hi = bounds_map[model]
        w = np.sqrt(weights[m])

        def resid(p):
            return (_predict(model, DX[m], DY[m], p) - data[m]) * w

        try:
            sol = least_squares(resid, x0=np.array(x0_map[model]), bounds=(lo, hi),
                                max_nfev=2000, xtol=1e-12, ftol=1e-12)
            p = sol.x
        except Exception:                                   # pragma: no cover
            p = np.array(x0_map[model])
        pred = _predict(model, DX, DY, p)
        r2, rmse, max_err = _r2(data[m], pred[m], weights[m])
        if model == "exponential":
            nu = 0.5
            lx, ly = p
            c1x, c1y = lx, ly
        elif model == "matern":
            lx, ly, nu = p
            c1x = _crossing_length(lx, nu)
            c1y = _crossing_length(ly, nu)
        else:
            lx, ly, nu = p
            c1x = _crossing_length(lx, nu)
            c1y = _crossing_length(ly, nu)
        out[model] = CorrelationFit(
            model=model,
            length_x=float(lx),
            length_y=float(ly),
            nu=float(nu),
            r2=float(r2),
            rmse=float(rmse),
            max_abs_error=float(max_err),
            corr_length_1e_x=float(c1x),
            corr_length_1e_y=float(c1y),
            variance=float(ac[0, 0]),
            n_lag_cells=int(np.count_nonzero(m)),
            lag_limit_x=float(lx_max),
            lag_limit_y=float(ly_max),
        )
    return out


def _crossing_length(length: float, nu: float) -> float:
    """Distance where the 1D Matern correlation drops to ``1/e``."""
    from scipy.optimize import brentq
    if nu <= 0.5 + 1e-9:
        return float(length)
    f = lambda d: float(matern_1d(np.array([d]), length, nu)[0]) - np.exp(-1.0)
    try:
        return float(brentq(f, 0.0, 20.0 * length))
    except Exception:                                        # pragma: no cover
        return float(length)


def fitted_2d_model(fit: CorrelationFit, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Evaluate the fitted 2D correlation surface (the KL kernel)."""
    return _predict(fit.model, dx, dy, [fit.length_x, fit.length_y, fit.nu]) \
        if fit.model != "exponential" else _predict(fit.model, dx, dy,
                                                    [fit.length_x, fit.length_y])