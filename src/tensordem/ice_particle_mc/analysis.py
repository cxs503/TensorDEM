"""Estimators for the ice-particle Mohr-Coulomb verification surface.

Every function here is a pure numerical estimator (fit or statistic) with an
explicit, stated convention, so a report can cite both the number and the way it
was obtained:

``fit_linear_modulus``      -- OLS slope of ``sigma(eps)`` over a strain window.
``exponential_tail_fit``    -- exponential fit of the force PDF for ``f > <f>``.
``fit_power_law``           -- log-log OLS ``y = y0 (x/x0)^n``.
``pressure_area_exponent``  -- Korzhavin-form pressure-area exponent ``n``.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["fit_linear_modulus", "exponential_tail_fit", "fit_power_law",
           "pressure_area_exponent", "relative_error"]


def fit_linear_modulus(strain, stress, peak_fraction_lo: float = 0.25,
                       peak_fraction_hi: float = 0.70) -> dict[str, float]:
    """Young's modulus from the pre-peak linear segment of a stress-strain curve.

    The window is a *fraction of the peak stress* (default 25%-70%), which is
    fixed before the run and reported; the slope is plain OLS.  ``r2`` is returned
    so a non-linear window announces itself instead of silently biasing ``E``.
    """
    eps = np.asarray(strain, dtype=float)
    sig = np.asarray(stress, dtype=float)
    if eps.size < 4:
        return {"E_Pa": float("nan"), "r2": float("nan"), "n_points": float(eps.size),
                "window_Pa": (float("nan"), float("nan"))}
    peak = float(np.nanmax(sig))
    lo, hi = peak_fraction_lo * peak, peak_fraction_hi * peak
    mask = (sig >= lo) & (sig <= hi)
    if mask.sum() < 4:
        return {"E_Pa": float("nan"), "r2": float("nan"), "n_points": float(mask.sum()),
                "window_Pa": (lo, hi)}
    slope, intercept = np.polyfit(eps[mask], sig[mask], 1)
    predicted = slope * eps[mask] + intercept
    ss_res = float(np.sum((sig[mask] - predicted) ** 2))
    ss_tot = float(np.sum((sig[mask] - sig[mask].mean()) ** 2))
    return {"E_Pa": float(slope), "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "n_points": float(mask.sum()), "window_Pa": (lo, hi)}


def exponential_tail_fit(force, n_bins: int = 30, min_count: int = 8,
                         upper_percentile: float = 99.5) -> dict[str, float]:
    """Fit ``P(f) ~ exp(-f/<f>)`` on the tail ``f > <f>`` of a force PDF.

    Convention (stated because it changes the number): the histogram runs from
    ``<f>`` to the ``upper_percentile`` of the normalised force, the PDF is a
    density (counts / (A * df)) so its integral over the tail is 1 within the
    window, and the fit is OLS of ``ln P`` on ``f``.  ``R2`` is computed in log
    space (the space the fit minimises) and ``r2_linear`` after exponentiating,
    so neither can hide a bad fit.  The decay length is ``-1/slope`` in units of
    ``<f>``; the literature exponential tail has decay length ~``<f>``.
    """
    f = np.asarray(force, dtype=float)
    f = f[np.isfinite(f) & (f > 0.0)]
    if f.size < 50:
        return {"decay_length": float("nan"), "r2": float("nan"),
                "n_forces": float(f.size)}
    mean = float(f.mean())
    x = f / mean
    hi = float(np.percentile(x, upper_percentile))
    if not hi > 1.0:
        return {"decay_length": float("nan"), "r2": float("nan"),
                "n_forces": float(f.size)}
    bins = np.linspace(1.0, hi, n_bins + 1)
    counts, edges = np.histogram(x, bins=bins)
    widths = np.diff(edges)
    density = counts / (counts.sum() * widths)
    centres = 0.5 * (edges[1:] + edges[:-1])
    keep = (counts >= min_count) & (density > 0.0)
    if keep.sum() < 4:
        return {"decay_length": float("nan"), "r2": float("nan"),
                "n_forces": float(f.size)}
    y = np.log(density[keep])
    slope, intercept = np.polyfit(centres[keep], y, 1)
    predicted = slope * centres[keep] + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_log = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    linear = density[keep]
    linear_pred = np.exp(predicted)
    ss_res_lin = float(np.sum((linear - linear_pred) ** 2))
    ss_tot_lin = float(np.sum((linear - linear.mean()) ** 2))
    return {
        "decay_length": float(-1.0 / slope) if slope < 0 else float("inf"),
        "r2": float(r2_log),
        "r2_linear": float(1.0 - ss_res_lin / ss_tot_lin) if ss_tot_lin > 0 else float("nan"),
        "slope": float(slope),
        "n_forces": float(f.size),
        "mean_force_N": mean,
        "n_bins_used": float(keep.sum()),
        "centres": centres.tolist(),
        "densities": density.tolist(),
        "counts": counts.tolist(),
        "bin_edges": edges.tolist(),
    }


def fit_power_law(x, y) -> dict[str, float]:
    """Log-log OLS ``log y = log y0 + n log(x/x0)``; returns ``n`` and ``R2``."""
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    good = (xv > 0) & (yv > 0) & np.isfinite(xv) & np.isfinite(yv)
    if good.sum() < 3:
        return {"exponent": float("nan"), "r2": float("nan"), "n_points": float(good.sum())}
    lx = np.log(xv[good])
    ly = np.log(yv[good])
    slope, intercept = np.polyfit(lx, ly, 1)
    predicted = slope * lx + intercept
    ss_res = float(np.sum((ly - predicted) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return {"exponent": float(slope), "prefactor": float(math.exp(intercept)),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "n_points": float(good.sum()),
            "log_log_pairs": np.stack([lx, ly], axis=1).tolist()}


def pressure_area_exponent(areas, pressures) -> dict[str, float]:
    """Korzhavin-form exponent ``n`` in ``P(A) = P0 (A/A0)^n`` (expect -0.5)."""
    fit = fit_power_law(areas, pressures)
    fit["reference_pressure_Pa"] = fit.get("prefactor", float("nan"))
    fit["target_exponent"] = -0.5
    fit["exponent_error"] = (abs(fit["exponent"] + 0.5)
                             if math.isfinite(fit["exponent"]) else float("nan"))
    fit["area_range"] = (float(np.min(areas)), float(np.max(areas)))
    return fit


def relative_error(value: float, reference: float) -> float:
    if reference == 0.0 or not math.isfinite(reference):
        return float("nan")
    return abs(value - reference) / abs(reference)