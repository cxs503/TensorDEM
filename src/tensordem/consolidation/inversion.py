"""Back-analysis of ``C_v`` and ``s_f`` from a measured consolidation curve.

The inversion is the second half of the chain: a consolidation curve
(``U(t)`` or ``s(t)``) is *read back* as the two numbers the design office
wants -- the coefficient of consolidation ``C_v`` and the final settlement
``s_f`` -- and those numbers are compared with the micro-mechanically
predicted ones.  Inverse problems need a stated estimator and a stated
uncertainty, so both are explicit here:

``t50`` / ``t90``
    the classical construction: ``C_v = T_v(U) H_dr^2 / t_U`` with the
    tabulated time factors (``T_v = 0.197`` at 50 %, ``0.848`` at 90 %);
``least_squares``
    a one-parameter fit of the whole curve against the Terzaghi series
    (log-space golden-section search, no scipy dependency needed);
``ci95``
    the confidence band reported with the inverted ``C_v``.  It is built from
    the *ensemble* of independent calibrations (DEM realisations), i.e. a
    propagation of the measured ``E_oed`` / ``k_hyd`` spread -- not a
    curve-fitting residual, which would understate the uncertainty by orders
    of magnitude.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .terzaghi import Tv_of_U, U_series

__all__ = [
    "ConsolidationInversion",
    "cv_from_time_factor",
    "time_to_degree",
    "invert_cv",
    "invert_settlement",
    "ci95",
    "propagate_cv_uncertainty",
]


def cv_from_time_factor(t, Tv: float, h_dr: float) -> float:
    """``C_v = T_v H_dr^2 / t`` (the classical t50 / t90 construction)."""
    t = float(t)
    if t <= 0.0:
        raise ValueError("time must be positive")
    return float(Tv) * float(h_dr) ** 2 / t


def time_to_degree(t: Sequence[float], U: Sequence[float], degree: float) -> float:
    """Interpolate the time at which ``U`` first reaches ``degree``."""
    t_arr = np.asarray(t, dtype=np.float64)
    U_arr = np.asarray(U, dtype=np.float64)
    order = np.argsort(t_arr)
    t_arr, U_arr = t_arr[order], U_arr[order]
    if U_arr[0] >= degree:
        return float(t_arr[0])
    idx = np.nonzero(U_arr >= degree)[0]
    if idx.size == 0:
        raise ValueError(f"curve never reaches U = {degree}")
    j = int(idx[0])
    if j == 0:
        return float(t_arr[0])
    u0, u1 = U_arr[j - 1], U_arr[j]
    w = (degree - u0) / (u1 - u0) if u1 > u0 else 0.0
    return float(t_arr[j - 1] + w * (t_arr[j] - t_arr[j - 1]))


def invert_cv(
    t: Sequence[float],
    U: Sequence[float],
    h_dr: float,
    *,
    method: str = "t50",
    degree: float = 0.5,
) -> dict:
    """Invert ``C_v`` from a simulated ``U(t)`` history.

    ``method``: ``"t50"``, ``"t90"`` or ``"least_squares"`` (full-curve fit).
    Returns the estimate plus the estimator diagnostics so a reader can see
    how much the choice of estimator matters.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    U_arr = np.asarray(U, dtype=np.float64)
    keep = (t_arr > 0.0) & (U_arr > 0.0) & (U_arr < 1.0)
    if keep.sum() < 3:
        raise ValueError("not enough points on the curve to invert C_v")

    estimates: dict[str, float] = {}
    for name, target in (("t50", 0.5), ("t90", 0.9)):
        try:
            t_star = time_to_degree(t_arr[keep], U_arr[keep], target)
            Tv_star = Tv_of_U(target)
            estimates[name] = cv_from_time_factor(t_star, Tv_star, h_dr)
        except ValueError:
            continue

    if method == "least_squares" or method == "fit":
        estimates["least_squares"] = _fit_cv(t_arr[keep], U_arr[keep], h_dr)
        chosen = estimates["least_squares"]
    elif method in ("t50", "t90"):
        if method not in estimates:
            raise ValueError(f"the curve does not reach the {method} point")
        chosen = estimates[method]
    else:
        raise ValueError(f"unknown method {method!r}")

    residuals = _residuals(t_arr[keep], U_arr[keep], h_dr, chosen)
    return {
        "method": method,
        "C_v": chosen,
        "estimates": estimates,
        "t50": time_to_degree(t_arr[keep], U_arr[keep], 0.5) if "t50" in estimates else None,
        "t90": time_to_degree(t_arr[keep], U_arr[keep], 0.9) if "t90" in estimates else None,
        "max_abs_residual": float(np.max(np.abs(residuals))),
        "rms_residual": float(np.sqrt(np.mean(residuals ** 2))),
        "n_points": int(keep.sum()),
    }


def _residuals(t: np.ndarray, U: np.ndarray, h_dr: float, cv: float) -> np.ndarray:
    Tv = cv * t / h_dr ** 2
    return np.asarray(U_series(np.clip(Tv, 1e-12, None)), dtype=np.float64) - U


def _fit_cv(t: np.ndarray, U: np.ndarray, h_dr: float, *,
            lo_years: float = -30.0, hi_years: float = 30.0) -> float:
    """Golden-section minimisation of the log-space sum of squared residuals."""
    sse = lambda log_cv: float(np.sum(_residuals(t, U, h_dr, math.exp(log_cv)) ** 2))
    a, b = lo_years, hi_years
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = sse(c), sse(d)
    for _ in range(200):
        if abs(b - a) < 1e-10:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = sse(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = sse(d)
    return float(math.exp(0.5 * (a + b)))


def invert_settlement(t: Sequence[float], settlement: Sequence[float]) -> dict:
    """Back out the final settlement and the 50 % time from an ``s(t)`` curve.

    ``s_f`` is taken as the plateau of the measured curve (mean of the last
    10 % of samples), which is what a laboratory oedometer reports, and the
    time factor is then anchored on ``s_f/2``.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    s_arr = np.asarray(settlement, dtype=np.float64)
    if t_arr.size < 5:
        raise ValueError("not enough points to invert the settlement curve")
    tail = max(3, t_arr.size // 10)
    s_final = float(np.mean(s_arr[-tail:]))
    t50 = time_to_degree(t_arr, s_arr / s_final, 0.5)
    return {
        "s_final": s_final,
        "t50": t50,
        "tail_samples": tail,
        "plateau_spread": float(np.max(s_arr[-tail:]) - np.min(s_arr[-tail:])),
    }


def ci95(samples: Iterable[float], *, confidence: float = 0.95) -> dict:
    """Mean and two-sided confidence interval of a small sample (Student t)."""
    x = np.asarray(list(samples), dtype=np.float64)
    n = x.size
    if n == 0:
        raise ValueError("empty sample")
    mean = float(np.mean(x))
    if n == 1:
        return {"n": 1, "mean": mean, "se": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "confidence": confidence, "relative_se": float("nan")}
    se = float(np.std(x, ddof=1) / math.sqrt(n))
    # two-sided Student-t quantile without scipy
    t_q = _student_t_quantile(0.5 + 0.5 * confidence, n - 1)
    return {
        "n": n,
        "mean": mean,
        "se": se,
        "ci_low": mean - t_q * se,
        "ci_high": mean + t_q * se,
        "confidence": confidence,
        "relative_se": se / abs(mean) if mean else float("nan"),
    }


def _student_t_quantile(p: float, dof: int) -> float:
    """Student-t quantile by bisection on the regularised incomplete beta."""
    if dof <= 0:
        return float("nan")
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, dof) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _t_cdf(x: float, dof: int) -> float:
    x = abs(float(x))
    beta = _incomplete_beta(0.5 * dof, 0.5, dof / (dof + x * x))
    return 1.0 - 0.5 * beta


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta ``I_x(a,b)`` by continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log(1.0 - x))
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(a, b, x) / a
    return 1.0 - front * _beta_cf(b, a, 1.0 - x) / b


def _beta_cf(a: float, b: float, x: float, *, max_iter: int = 300) -> float:
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / (qap + tiny)
    d = 1.0 / (abs(d) < tiny and tiny or d)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2) + tiny)
        d = 1.0 + aa * d
        d = 1.0 / (abs(d) < tiny and tiny or d)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2) + tiny)
        d = 1.0 + aa * d
        d = 1.0 / (abs(d) < tiny and tiny or d)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def propagate_cv_uncertainty(
    E_samples: Iterable[float],
    k_samples: Iterable[float],
    gamma_w: float,
    *,
    paired: bool = True,
) -> dict:
    """``C_v = k E_oed / gamma_w`` with a propagated confidence band.

    ``paired=True`` treats ``E_samples[i]``/``k_samples[i]`` as one and the
    same DEM realisation (the honest case: a run that gives a stiff skeleton
    also gives the porosity, hence the conductivity, of that same packing).
    """
    E = np.asarray(list(E_samples), dtype=np.float64)
    k = np.asarray(list(k_samples), dtype=np.float64)
    if E.size != k.size:
        raise ValueError("E_samples and k_samples must have the same length")
    cv = k * E / float(gamma_w)
    stats = ci95(cv)
    if E.size > 1:
        rel_E = float(np.std(E, ddof=1) / math.sqrt(E.size)) / abs(float(np.mean(E)))
        rel_k = float(np.std(k, ddof=1) / math.sqrt(k.size)) / abs(float(np.mean(k)))
    else:
        rel_E = rel_k = float("nan")
    stats.update({
        "C_v_samples": cv.tolist(),
        "independent_propagation_relative_se": math.hypot(rel_E, rel_k),
        "paired": paired,
        "gamma_w": float(gamma_w),
    })
    return stats