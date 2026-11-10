"""Scaling analysis helpers: complexity exponent, Amdahl, weak/memory scaling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class LogLogFit:
    """``log(t) = slope * log(N) + intercept`` least-squares fit."""

    slope: float
    intercept: float
    r2: float
    n_points: int
    residuals: list[float]

    def as_dict(self) -> dict[str, float | int]:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "r2": self.r2,
            "n_points": self.n_points,
            "max_abs_residual_log": max((abs(r) for r in self.residuals), default=0.0),
        }


def fit_loglog(n: np.ndarray | list[float], t: np.ndarray | list[float]) -> LogLogFit:
    """Fit ``log t`` against ``log N`` and return slope / R^2.

    A contact search that is ``O(N)`` gives ``slope = 1``; the pure pair loop of
    a naive implementation gives ``slope = 2``.  The residual array is returned
    in log space so that the report can show the fit quality per point.
    """
    n_arr = np.asarray(n, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    if n_arr.size != t_arr.size:
        raise ValueError("n and t must have the same length")
    if n_arr.size < 2:
        raise ValueError("at least two points are required")
    if np.any(n_arr <= 0) or np.any(t_arr <= 0):
        raise ValueError("inputs must be positive")
    x = np.log(n_arr)
    y = np.log(t_arr)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    resid = y - pred
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - float((resid**2).sum()) / ss_tot
    return LogLogFit(
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        n_points=int(n_arr.size),
        residuals=[float(r) for r in resid],
    )


def strong_scaling(threads: list[int], times: list[float], *, base_threads: int) -> dict:
    """Speed-up and parallel efficiency relative to a ``base_threads`` baseline.

    ``efficiency = (T_base / T_p) / (p / base)`` — i.e. the fraction of the ideal
    ``p/base``-fold gain actually realised when growing from the base thread
    count.  Both this and the ``T(1)``-based definition are reported because the
    mission protocol starts at 8 threads while reviewers often expect ``T(1)``.
    """
    t = np.asarray(times, dtype=np.float64)
    p = np.asarray(threads, dtype=np.float64)
    base = float(base_threads)
    base_time = float(t[np.argmin(np.abs(p - base))])
    speedup_rel = base_time / t
    ideal_rel = p / base
    eff_rel = speedup_rel / ideal_rel
    if p.min() <= 1.0:
        t1 = float(t[np.argmin(p)])
        speedup_abs = t1 / t
        eff_abs = speedup_abs / p
    else:
        speedup_abs = [float("nan")] * len(threads)
        eff_abs = [float("nan")] * len(threads)
    return {
        "threads": [int(x) for x in threads],
        "time_s": [float(x) for x in t],
        "speedup_vs_base": [float(x) for x in speedup_rel],
        "efficiency_vs_base": [float(x) for x in eff_rel],
        "speedup_vs_1": [float(x) for x in speedup_abs],
        "efficiency_vs_1": [float(x) for x in eff_abs],
        "base_threads": int(base_threads),
    }


def amdahl_serial_fraction(threads: list[int], times: list[float]) -> dict:
    """Least-squares Amdahl fit ``T(p) = T_serial + T_par / p``.

    Returns the implied serial fraction ``s = T_serial / T(1)`` and the implied
    asymptotic speed-up cap ``1/s``.
    """
    p = np.asarray(threads, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    a = np.stack([np.ones_like(p), 1.0 / p], axis=1)
    coef, *_ = np.linalg.lstsq(a, t, rcond=None)
    t_serial, t_par = float(coef[0]), float(coef[1])
    t1 = t_serial + t_par
    s = t_serial / t1 if t1 else float("nan")
    return {
        "t_serial_s": t_serial,
        "t_parallel_s": t_par,
        "serial_fraction": s,
        "speedup_cap": (1.0 / s) if s > 0 else float("inf"),
        "t1_model_s": t1,
        "t1_measured_s": float(t[np.argmin(p)]),
    }


def weak_scaling(labels: list[int], times: list[float]) -> dict:
    """Weak-scaling ratio: per-step time growth relative to the smallest case."""
    t = np.asarray(times, dtype=np.float64)
    ratio = t / t.min()
    return {
        "size": [int(x) for x in labels],
        "time_s": [float(x) for x in t],
        "ratio_to_min": [float(x) for x in ratio],
        "max_ratio": float(ratio.max()),
        "efficiency": float(1.0 / ratio.max()),
    }


def memory_linearity(n: list[int], peak_mb: list[float]) -> dict:
    """Check that peak memory is affine in ``N`` (``peak = a + b N``).

    Returns the relative deviation of every measurement from the fit, the
    per-particle marginal cost ``b`` and the coefficient of determination.
    """
    n_arr = np.asarray(n, dtype=np.float64)
    m_arr = np.asarray(peak_mb, dtype=np.float64)
    a = np.stack([np.ones_like(n_arr), n_arr], axis=1)
    coef, *_ = np.linalg.lstsq(a, m_arr, rcond=None)
    pred = a @ coef
    rel_dev = np.abs(m_arr - pred) / np.maximum(pred, 1e-12)
    ss_tot = float(((m_arr - m_arr.mean()) ** 2).sum())
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - float(((m_arr - pred) ** 2).sum()) / ss_tot
    return {
        "n": [int(x) for x in n],
        "peak_mb": [float(x) for x in m_arr],
        "intercept_mb": float(coef[0]),
        "bytes_per_particle": float(coef[1] * 1e6 / 1.0),
        "kb_per_particle": float(coef[1] * 1e3),
        "fit_r2": float(r2),
        "max_relative_deviation": float(rel_dev.max()),
        "relative_deviation": [float(x) for x in rel_dev],
    }


def geometric_series_counts(max_n: int, *, ratio: int = 4, count: int = 5) -> list[int]:
    """Particle counts ``{2k, 8k, 32k, 64k, 128k}``-style ladder."""
    n = max_n
    out = []
    for _ in range(count):
        out.append(int(n))
        n = max(1000, n // ratio) if ratio > 1 else n
    return sorted(set(out))


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def ceil_pow2(value: float) -> int:
    return int(2 ** math.ceil(math.log2(max(value, 1.0))))