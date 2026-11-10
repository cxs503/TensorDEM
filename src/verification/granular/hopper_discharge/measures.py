"""Measurement operators for the hopper verification surface (TensorDEM m7751).

Every operator here turns a *measured* quantity into a comparison against a
closed-form prediction; none of them returns the analytic value as if it had
been measured.

* :func:`repose_angle` -- free-surface slope of a DEM heap (theta_r).
* :func:`beverloo_scaling_fit` -- W vs (D0 - k d)^(5/2) linear fit (exponent
  check) and W vs D0^(5/2) (the criterion as written).
* :func:`beverloo_fit` -- recover the constant C for a prescribed k (and
  jointly for a grid of k) from measured flow rates.
* :func:`janssen_analytic` / :func:`janssen_fit` -- closed-form wall-stress
  profile and the two-parameter (sigma_sat, z0) fit of a DEM profile.
* :func:`saturation_metrics` -- machine-checkable saturation diagnostics.

All quantities SI; angles returned in degrees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Repose angle
# ---------------------------------------------------------------------------

def heap_height_map(positions: np.ndarray, radius: float, *,
                    cell: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Surface height map of a heap: (cell radius, cell top height).

    The XY plane is binned into cells of ``1.5 r``; the height of a cell is the
    highest particle centre in it.  This is the standard DEM heap-surface
    estimator: it needs no neighbour search and cannot mistake a *buried*
    particle for a surface particle (which is what a narrow vertical probe
    does in an ordered packing).
    """
    pos = np.asarray(positions, dtype=np.float64)
    if len(pos) == 0:
        return np.zeros(0), np.zeros(0)
    h = 1.5 * radius if cell is None else float(cell)
    span = int(np.ceil(float(np.abs(pos[:, :2]).max()) / h)) + 2
    off = span + 1
    ix = (np.floor(pos[:, 0] / h).astype(np.int64) + off)
    iy = (np.floor(pos[:, 1] / h).astype(np.int64) + off)
    stride = int(2 * off + 3)
    key = ix * stride + iy
    order = np.argsort(key, kind="stable")
    keys = key[order]
    zs = pos[order, 2]
    uniq, start = np.unique(keys, return_index=True)
    top = np.maximum.reduceat(zs, start)
    cx = (uniq // stride).astype(np.float64) - off
    cy = (uniq % stride).astype(np.float64) - off
    rho = np.hypot((cx + 0.5) * h, (cy + 0.5) * h)
    return rho, top


def repose_angle(positions: np.ndarray, radius: float, *,
                 n_bins: int = 12, flank_low: float = 0.25,
                 flank_high: float = 0.90) -> dict:
    """Angle of repose of a DEM heap from its height map.

    Method: (1) height map of the free surface (see :func:`heap_height_map`);
    (2) radial bins of ``2 r``; the bin height is the *median* cell height
    (robust against stray particles); (3) least-squares fit
    ``h = h0 - tan(theta) rho`` over the flank, ``flank_low .. flank_high`` of
    the peak height (the apex plateau and the toe are excluded because a
    height map is not conical there); (4) ``theta = arctan(-slope)``.

    The apex is the *median* of the three highest occupied bins (a single dust
    particle that bounced away must not set the scale).

    Returns the angle in degrees together with the fit quality and the bin
    table, so the measurement can be audited or re-plotted without re-running
    the DEM.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if len(pos) < 20:
        raise ValueError("heap too small for a repose-angle measurement")
    rho_c, top = heap_height_map(pos, radius)
    if len(rho_c) < 6:
        raise ValueError("height map too coarse for a repose-angle measurement")
    z_floor = float(np.min(pos[:, 2])) - radius
    h = top - z_floor
    h_peak = float(np.percentile(h, 80.0))
    if h_peak <= 2.0 * radius:
        raise ValueError("heap height below two particle diameters")

    r_max = float(np.percentile(rho_c, 99.0))
    edges = np.linspace(0.0, r_max + 1e-9, n_bins + 1)
    bin_h, bin_rho, bin_n = [], [], []
    for i in range(n_bins):
        sel = (rho_c >= edges[i]) & (rho_c < edges[i + 1])
        if sel.sum() >= 2:
            bin_h.append(float(np.median(h[sel])))
            bin_rho.append(float(0.5 * (edges[i] + edges[i + 1])))
            bin_n.append(int(sel.sum()))
    bin_h = np.asarray(bin_h)
    bin_rho = np.asarray(bin_rho)
    if len(bin_h) < 3:
        raise ValueError("not enough occupied radial bins")
    flank = (bin_h >= flank_low * h_peak) & (bin_h <= flank_high * h_peak)
    if flank.sum() < 3:
        flank = np.ones_like(bin_h, dtype=bool)
    x, y = bin_rho[flank], bin_h[flank]
    w = np.asarray(bin_n, dtype=np.float64)[flank]
    w = w / w.sum()
    A = np.stack([np.ones_like(x), x], axis=1) * np.sqrt(w)[:, None]
    b = y * np.sqrt(w)
    (h0, slope), *_ = np.linalg.lstsq(A, b, rcond=None)
    pred = h0 + slope * x
    ss_res = float(((y - pred) ** 2 * w).sum())
    ss_tot = float(((y - (y * w).sum()) ** 2 * w).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    theta = math.degrees(math.atan(max(-slope, 0.0)))
    return {
        "theta_r_deg": theta,
        "slope": float(-slope),
        "intercept_m": float(h0),
        "r2": r2,
        "heap_height_m": h_peak,
        "heap_radius_m": r_max,
        "n_cells": int(len(rho_c)),
        "n_flank_bins": int(flank.sum()),
        "bin_radius_m": [float(v) for v in bin_rho],
        "bin_height_m": [float(v) for v in bin_h],
        "bin_count": [int(v) for v in bin_n],
    }


# ---------------------------------------------------------------------------
# Beverloo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BeverlooResult:
    C: float
    k: float
    r2: float
    rel_err_vs_literature: float
    prediction_kg_s: Sequence[float]

    def as_dict(self) -> dict:
        return {
            "C": self.C, "k": self.k, "r2": self.r2,
            "rel_err_vs_C0": self.rel_err_vs_literature,
            "prediction_kg_s": [float(x) for x in self.prediction_kg_s],
        }


def beverloo_prediction(rho_b: float, g: float, D: np.ndarray, d: float,
                        C: float, k: float) -> np.ndarray:
    """W = C rho_b sqrt(g) (D - k d)^(5/2) (Beverloo et al. 1961)."""
    D = np.asarray(D, dtype=np.float64)
    eff = np.maximum(D - k * d, 1e-9)
    return C * rho_b * math.sqrt(g) * eff ** 2.5


def beverloo_fit(W: np.ndarray, D: np.ndarray, d: float, rho_b: float,
                 g: float, *, C_literature: float = 0.58,
                 k_literature: float = 1.4, k_ranges: Sequence[tuple[float, float]] = ((1.0, 2.0),)
                 ) -> dict:
    """Fit C (fixed k) and the joint (C, k) pair to *measured* flow rates.

    The joint fit scans ``k`` on a grid and solves ``C`` in closed form for each
    ``k`` (least squares in the ``W`` metric), returning the best pair and the
    coefficient of determination of the fit.  ``C_fixed_k`` is the value obtained
    when the literature ``k`` is imposed -- that is the number to compare with
    ``C_literature`` for the acceptance criterion.
    """
    W = np.asarray(W, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    if W.shape != D.shape or W.size < 2:
        raise ValueError("need >= 2 (W, D) pairs of matching shape")
    if not np.all(np.isfinite(W)) or np.any(W <= 0.0):
        raise ValueError("flow rates must be finite and positive")

    def _solve(k: float) -> tuple[float, np.ndarray, float]:
        base = rho_b * math.sqrt(g) * np.maximum(D - k * d, 1e-9) ** 2.5
        C = float((W * base).sum() / (base * base).sum())
        pred = C * base
        ss_res = float(((W - pred) ** 2).sum())
        ss_tot = float(((W - W.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
        return C, pred, r2

    C_fixed, pred_fixed, r2_fixed = _solve(k_literature)
    best = None
    for lo, hi in k_ranges:
        for k in np.linspace(lo, hi, 201):
            C, pred, r2 = _solve(float(k))
            if best is None or r2 > best[2]:
                best = (float(k), C, r2, pred)
    k_best, C_best, r2_best, pred_best = best
    return {
        "C_fixed_k": C_fixed,
        "k_fixed": float(k_literature),
        "r2_fixed_k": r2_fixed,
        "prediction_fixed_k_kg_s": [float(x) for x in pred_fixed],
        "C_joint": C_best,
        "k_joint": k_best,
        "r2_joint": r2_best,
        "prediction_joint_kg_s": [float(x) for x in pred_best],
        "rel_err_C_fixed_k_vs_C0": abs(C_fixed - C_literature) / C_literature,
        "rel_err_C_joint_vs_C0": abs(C_best - C_literature) / C_literature,
        "C_literature": float(C_literature),
        "k_literature": float(k_literature),
    }


def beverloo_scaling_fit(W: np.ndarray, D: np.ndarray, d: float = 0.0,
                         k: float = 0.0) -> dict:
    """Linear fit ``W = A X + B`` for two abscissae, with R^2 for each.

    ``X = (D - k d)^(5/2)`` reproduces the Beverloo exponent
    (``k = 0`` -> plain ``D^(5/2)``, the acceptance criterion as written;
    ``k = 1.4`` -> orifice-diameter correction).  A collapse onto a straight
    line through the origin is what the 5/2 exponent predicts; the intercept
    quantifies the departure (finite particle size / hopper-angle effect).
    """
    W = np.asarray(W, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    out = {}
    for tag, kk in (("D_pow_2.5", 0.0), ("D_minus_kd_pow_2.5", float(k))):
        X = np.maximum(D - kk * d, 1e-12) ** 2.5
        A = np.stack([X, np.ones_like(X)], axis=1)
        (slope, intercept), *_ = np.linalg.lstsq(A, W, rcond=None)
        pred = slope * X + intercept
        ss_res = float(((W - pred) ** 2).sum())
        ss_tot = float(((W - W.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
        # through-origin variant
        slope0 = float((W * X).sum() / (X * X).sum())
        pred0 = slope0 * X
        ss_res0 = float(((W - pred0) ** 2).sum())
        r2_0 = 1.0 - ss_res0 / ss_tot if ss_tot > 0.0 else float("nan")
        out[tag] = {
            "slope": float(slope), "intercept": float(intercept), "r2": r2,
            "slope_origin": slope0, "r2_origin": r2_0,
            "abscissa": [float(v) for v in X],
            "prediction_kg_s": [float(v) for v in pred],
        }
    return out


# ---------------------------------------------------------------------------
# Janssen
# ---------------------------------------------------------------------------

def janssen_analytic(z: np.ndarray, rho_b: float, g: float, R_h: float,
                     mu_w: float, K: float) -> np.ndarray:
    """sigma_w(z) = (rho_b g R_h / mu_w) [1 - exp(-mu_w K z / R_h)]."""
    z = np.asarray(z, dtype=np.float64)
    if mu_w <= 0.0:
        raise ValueError("mu_w must be positive for the Janssen profile")
    sat = rho_b * g * R_h / mu_w
    return sat * (1.0 - np.exp(-mu_w * K * z / R_h))


def janssen_saturation(rho_b: float, g: float, R_h: float, mu_w: float) -> float:
    """Asymptotic wall stress rho_b g R_h / mu_w (independent of K)."""
    return rho_b * g * R_h / mu_w


def janssen_fit(z: np.ndarray, sigma: np.ndarray) -> dict:
    """Least-squares fit of ``sigma = sigma_sat (1 - exp(-z / z0))``.

    ``z`` is the depth below the free surface.  Non-finite / non-positive
    entries are dropped (slabs with no wall contact).  Returns the fitted
    parameters, R^2 and the derived ``mu_w K = R_h / z0`` diagnostic.
    """
    from scipy.optimize import curve_fit

    z = np.asarray(z, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    ok = np.isfinite(z) & np.isfinite(s) & (s > 0.0)
    if ok.sum() < 3:
        raise ValueError("need >= 3 valid wall-stress samples")
    zf, sf = z[ok], s[ok]

    def model(zz, sat, z0):
        return sat * (1.0 - np.exp(-zz / np.maximum(z0, 1e-9)))

    p0 = (float(sf.max()) * 1.2, float(max(zf.max() / 2.0, 1e-3)))
    bounds = ([0.0, 1e-4], [np.inf, np.inf])
    popt, _ = curve_fit(model, zf, sf, p0=p0, bounds=bounds, maxfev=20000)
    sat, z0 = float(popt[0]), float(popt[1])
    pred = model(zf, sat, z0)
    ss_res = float(((sf - pred) ** 2).sum())
    ss_tot = float(((sf - sf.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return {
        "sigma_sat_fit_pa": sat,
        "z0_fit_m": z0,
        "r2": r2,
        "n_points": int(ok.sum()),
        "depth_m": [float(v) for v in zf],
        "sigma_pa": [float(v) for v in sf],
        "prediction_pa": [float(v) for v in pred],
    }


def saturation_metrics(z: np.ndarray, sigma: np.ndarray, *,
                       ref_depth: float) -> dict:
    """Machine-checkable saturation diagnostics of a measured profile.

    * ``saturation_ratio`` = sigma(z >= ref_depth) / sigma_at_max_depth;
    * ``relative_slope`` = |dsigma/dz| at the deepest point * z0_measured,
      normalised by sigma there (0 for a fully saturated profile);
    * ``dsigma_rel_per_m`` = relative change of sigma per metre at the bottom.
    """
    z = np.asarray(z, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    ok = np.isfinite(z) & np.isfinite(s) & (s > 0.0)
    z, s = z[ok], s[ok]
    if len(z) < 3:
        raise ValueError("need >= 3 valid samples")
    order = np.argsort(z)
    z, s = z[order], s[order]
    deep = z >= ref_depth
    if not deep.any():
        deep = z >= 0.8 * z.max()
    ratio = float(s[deep].mean() / s[-1]) if deep.any() else float("nan")
    if len(z) >= 2:
        slope = (s[-1] - s[-2]) / max(z[-1] - z[-2], 1e-12)
    else:
        slope = float("nan")
    rel_slope_per_m = float(slope / s[-1]) if s[-1] > 0 else float("nan")
    return {
        "ref_depth_m": float(ref_depth),
        "saturation_ratio": ratio,
        "dsigma_dz_pa_per_m": float(slope),
        "dsigma_rel_per_m": rel_slope_per_m,
        "sigma_at_depth_pa": float(s[-1]),
        "max_depth_m": float(z[-1]),
        "saturated": bool(ratio > 0.85 and abs(rel_slope_per_m) < 0.15),
    }