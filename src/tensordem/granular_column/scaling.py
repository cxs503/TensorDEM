"""Runout / Midi scaling fits and the machine-checkable acceptance criteria.

Everything here is a pure function of the numbers produced by
:mod:`tensordem.granular_column.column3d` and
:mod:`tensordem.granular_column.rheology`; no simulation is run, so the fits
can be recomputed from the archived case JSONs (``runs/``) without re-running
the DEM.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "fit_power_law",
    "local_exponents",
    "fit_midi_piecewise",
    "plateau_test",
    "fit_exponent",
    "relative_error",
    "jop_mu_i",
    "fit_jop_law",
    "criterion",
]

#: Lube et al. (2004) large-aspect-ratio exponent for column runout.
LUBE_EXPONENT = 2.0 / 3.0
#: Jop, Forterre & Pouliquen (2006) reference mu(I) parameters (glass beads).
JOP_MU_S = 0.36
JOP_MU_2 = 0.84
JOP_I_0 = 0.30


def fit_power_law(x, y) -> dict:
    """Least-squares ``y = c x^p`` in log-log space; returns p, c, r2, stderr."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        raise ValueError("need at least two positive points")
    lx, ly = np.log(x), np.log(y)
    p, ln_c = np.polyfit(lx, ly, 1)
    pred = p * lx + ln_c
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof = max(1, x.size - 2)
    stderr = math.sqrt(ss_res / dof / max(ss_tot, 1e-300)) * (x.size ** 0.5) if ss_tot > 0 else float("nan")
    return {
        "exponent": float(p),
        "prefactor": float(math.exp(ln_c)),
        "r2": float(r2),
        "stderr": float(stderr),
        "n": int(x.size),
        "x": x.tolist(),
        "y": y.tolist(),
    }


def local_exponents(x, y) -> list[dict]:
    """Consecutive log-log slopes (the discrete d ln y / d ln x)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    out = []
    for k in range(1, x.size):
        if x[k] <= 0 or x[k - 1] <= 0 or y[k] <= 0 or y[k - 1] <= 0:
            continue
        p = math.log(y[k] / y[k - 1]) / math.log(x[k] / x[k - 1])
        out.append({"a_low": float(x[k - 1]), "a_high": float(x[k]),
                    "a_mid": float(math.sqrt(x[k] * x[k - 1])),
                    "exponent": float(p)})
    return out


def fit_midi_piecewise(aspect, runout, *, small_max: float = 1.0) -> dict:
    """Piecewise runout law ``alpha a`` (small a) / ``beta a^p`` (large a).

    The large-``a`` exponent and prefactor are fitted on ``a > small_max``; the
    small-``a`` slope is fitted on ``a <= small_max``.  A saturated plateau of
    the runout itself is *not* predicted by the Lube/Midi laws -- the runout
    keeps growing as ``a^(2/3)`` -- so the "plateau" of the campaign node is
    tested on the *local exponent* (:func:`plateau_test`).
    """
    aspect = np.asarray(aspect, dtype=float)
    runout = np.asarray(runout, dtype=float)
    small = aspect <= small_max
    large = aspect >= 2.0 * small_max
    small_fit = fit_power_law(aspect[small], runout[small]) if small.sum() >= 2 else None
    large_fit = fit_power_law(aspect[large], runout[large]) if large.sum() >= 2 else None
    return {
        "small": small_fit,
        "large": large_fit,
        "local": local_exponents(aspect, runout),
        "small_max": small_max,
        "lube_exponent": LUBE_EXPONENT,
    }


def plateau_test(exponents: list[dict], *, tau: float = 0.05) -> dict:
    """Is the local runout exponent saturating?

    Returns the last two local exponents, their relative change and whether the
    change is below ``tau`` (the ``<= 5 %`` of criterion (1b)).
    """
    if len(exponents) < 2:
        raise ValueError("need at least two local exponents")
    last, prev = exponents[-1], exponents[-2]
    change = abs(last["exponent"] - prev["exponent"]) / max(abs(prev["exponent"]), 1e-12)
    return {
        "exponent_last": last["exponent"],
        "exponent_prev": prev["exponent"],
        "a_last": last["a_mid"],
        "relative_change": float(change),
        "plateau": bool(change <= tau),
        "tau": tau,
    }


def fit_exponent(x, y) -> dict:
    """Alias of :func:`fit_power_law` used for the time/velocity scalings."""
    return fit_power_law(x, y)


def relative_error(value: float, reference: float) -> float:
    return abs(float(value) - float(reference)) / abs(float(reference))


def jop_mu_i(inertial_number, mu_s: float = JOP_MU_S, mu_2: float = JOP_MU_2,
             i_0: float = JOP_I_0):
    """Jop et al. (2006) ``mu(I) = mu_s + (mu_2 - mu_s) / (I_0/I + 1)``."""
    i = np.maximum(np.asarray(inertial_number, dtype=float), 0.0)
    return mu_s + (mu_2 - mu_s) / (1.0 + i_0 / np.maximum(i, 1e-12))


def fit_jop_law(inertial_number, mu, *, fix_mu_2: float | None = None) -> dict:
    """Fit ``mu(I)`` to the Jop law by bounded non-linear least squares.

    ``fix_mu_2`` implements the negative control: with ``mu_2 = 0`` the law
    cannot saturate above ``mu_s`` and the fit residual stays large.
    """
    from scipy.optimize import curve_fit

    i = np.asarray(inertial_number, dtype=float)
    m = np.asarray(mu, dtype=float)
    mask = np.isfinite(i) & np.isfinite(m) & (i > 0)
    i, m = i[mask], m[mask]
    if i.size < 3:
        raise ValueError("need at least three mu(I) samples")

    if fix_mu_2 is None:
        def model(x, mu_s, mu_2, i_0):
            return jop_mu_i(x, mu_s, mu_2, i_0)

        p0 = [float(np.min(m)), float(np.max(m)) * 1.2, 0.3]
        bounds = ([0.0, 0.0, 1e-4], [2.0, 4.0, 50.0])
        popt, _ = curve_fit(model, i, m, p0=p0, bounds=bounds, maxfev=20000)
        mu_s, mu_2, i_0 = (float(v) for v in popt)
    else:
        def model_s(x, mu_s, i_0):
            return jop_mu_i(x, mu_s, fix_mu_2, i_0)

        p0 = [float(np.min(m)), 0.3]
        bounds = ([0.0, 1e-4], [2.0, 50.0])
        popt, _ = curve_fit(model_s, i, m, p0=p0, bounds=bounds, maxfev=20000)
        mu_s, i_0 = (float(v) for v in popt)
        mu_2 = float(fix_mu_2)
    pred = jop_mu_i(i, mu_s, mu_2, i_0)
    ss_res = float(np.sum((m - pred) ** 2))
    ss_tot = float(np.sum((m - m.mean()) ** 2))
    return {
        "mu_s": mu_s,
        "mu_2": mu_2,
        "i_0": i_0,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "rms": float(math.sqrt(ss_res / i.size)),
        "n": int(i.size),
        "i": i.tolist(),
        "mu": m.tolist(),
        "mu_fit": pred.tolist(),
    }


def criterion(name: str, value: float, target: str, tol: float | None,
              passed: bool, note: str = "") -> dict:
    """Uniform criterion record for the report JSON."""
    return {"name": name, "value": float(value), "target": target,
            "tolerance": tol, "pass": bool(passed), "note": note}
