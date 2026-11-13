"""Pre-registered arrival (``t_inf``) definitions for the granular-column case.

Why this module exists
----------------------
``CASE-COLUMN-COLLAPSE`` (Lube, Huppert, Sparks & Freundt 2004, JFM **508**,
175-199, doi:10.1017/S0022112004009036) carries **two** closed-form predictions:

* runout radius ``r_inf = r_i (1 + 1.24 a)`` for ``a < 1.7``, ``r_i (1 + 1.6
  sqrt(a))`` for ``a > 1.7``  -- the *first* prediction;
* emplacement time ``t_inf = 3 sqrt(h_i / g) = 3 sqrt(r_i / g) a^(1/2)`` --
  the *second* prediction.

The second prediction compares a measured time against a measured number, so it
is only meaningful together with an **explicit, pre-registered definition of
"the grains have reached r_inf"**.  Rounds 1-5 of the case used a *fraction of
the final runout* (90 / 95 / 99 %), chosen per run and never declared, and
``docs/AUDIT_COLUMN_COLLAPSE_TIME_LAW_m17534.md`` showed the verdict moves by
21 percentage points across that family -- i.e. the definition, not the
physics, dominated the verdict.  The audit's proposed remedy was a
velocity-threshold definition.  This module implements that definition family
so it can be **frozen before** the runs and adjudicated mechanically.

Definitions implemented
-----------------------
``fraction``
    ``t`` at which the (unsmoothed) front first reaches ``r_i + f * (r_f -
    r_i)``.  The historical round 1-5 definition (``f`` in 0.90/0.95/0.99).
``velocity``
    ``t`` at which the front speed, measured on a time-window-smoothed front
    history, first drops below ``f * v_peak`` *after* the speed peak.  The
    audit's recommended definition family.  ``v_peak`` is the maximum of the
    smoothed speed, so the definition is scale-free.
``length``
    ``t`` at which the front first comes within ``n * d`` of its final value
    (``d`` = grain diameter).  This mimics the experimental resolution limit:
    a video cannot distinguish "one grain short" from "arrived".

All three return ``math.nan`` when the criterion is never met (no silent
fallbacks), and every function validates its inputs: a definition that can be
silently satisfied by a truncated or non-monotone history would be worthless as
an acceptance criterion.

Smoothing
---------
The front is the 99.5 % radial quantile plus one radius — a statistic of a few
dozen grains near the edge, so its sample-to-sample scatter is not small.  Its
numerical derivative on a densely sampled history is therefore dominated by
quantile noise, and a raw ``v < f v_peak`` test would fire on the noise floor.
:func:`smooth_front` applies a **fixed-time** moving average (default 10 ms,
~20 integration steps at the case's ``dt``), so the smoothing strength does not
change when the sampling stride changes.  This is a deliberate, documented
choice: it is part of the pre-registration, not a free parameter.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "ArrivalError",
    "smooth_front",
    "front_speed",
    "time_to_fraction",
    "MONOTONE_TOLERANCE",
    "time_to_velocity_threshold",
    "time_to_length_tolerance",
    "arrival_report",
]

#: Pre-registered definition families (frozen by
#: ``benchmarks/case_column_collapse_time_m18261.py`` before any run).
FRACTION_LEVELS = (0.50, 0.75, 0.90, 0.95, 0.99)
VELOCITY_FRACTIONS = (0.10, 0.05, 0.02, 0.01, 0.005)
LENGTH_TOLERANCES_D = (0.5, 1.0, 2.0)
#: Default smoothing window (s) and the primary pre-registered members.
DEFAULT_WINDOW_S = 0.010
PRIMARY_FRACTION = 0.90
PRIMARY_VELOCITY_FRACTION = 0.01
PRIMARY_LENGTH_TOLERANCE_D = 1.0


class ArrivalError(ValueError):
    """Raised when a front history cannot support an arrival definition."""


#: Allowed backward excursion of the raw front, as a fraction of its total
#: excursion.  The front is a 99.5 % radial *quantile*, so it can recede by a
#: fraction of a grain as the edge rearranges; a gross recession (a genuinely
#: receding flow) is still rejected.
MONOTONE_TOLERANCE = 0.01


def _as_history(time, front, monotone_tolerance: float = MONOTONE_TOLERANCE
                ) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=np.float64).ravel()
    front = np.asarray(front, dtype=np.float64).ravel()
    if time.size != front.size:
        raise ArrivalError("time and front histories must have equal length")
    if time.size < 4:
        raise ArrivalError("front history needs at least 4 samples")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(front)):
        raise ArrivalError("front history must be finite")
    dt = np.diff(time)
    if np.any(dt <= 0.0):
        raise ArrivalError("time history must be strictly increasing")
    excursion = float(np.ptp(front))
    if excursion > 0.0 and np.any(np.diff(front) < -monotone_tolerance * excursion):
        raise ArrivalError(
            "front history must be non-decreasing within monotone_tolerance="
            f"{monotone_tolerance:g} of its excursion (a receding front is not a "
            "column collapse)")
    return time, front


def smooth_front(time, front, window_s: float = DEFAULT_WINDOW_S) -> np.ndarray:
    """Moving average of the front over a fixed **time** window.

    The window is converted to a sample count from the median sampling
    interval, so the same ``window_s`` means the same physical smoothing at any
    ``sample_stride``.  Edges are padded with the end values (a zero pad would
    manufacture a huge spurious speed at ``t = 0``).
    """
    time, front = _as_history(time, front)
    if not math.isfinite(window_s) or window_s < 0.0:
        raise ArrivalError("window_s must be finite and nonnegative")
    if window_s == 0.0:
        return front.copy()
    sample_dt = float(np.median(np.diff(time)))
    if sample_dt <= 0.0:
        raise ArrivalError("median sampling interval must be positive")
    width = int(round(window_s / sample_dt))
    width = max(1, min(width, front.size))
    if width % 2 == 0:
        width += 1
    if width > front.size:
        width = front.size if front.size % 2 else front.size - 1
    half = width // 2
    padded = np.concatenate([np.full(half, front[0]), front,
                             np.full(half, front[-1])])
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def front_speed(time, front, window_s: float = DEFAULT_WINDOW_S) -> np.ndarray:
    """Time derivative of :func:`smooth_front` (m/s)."""
    time, front = _as_history(time, front)
    smoothed = smooth_front(time, front, window_s)
    return np.gradient(smoothed, time)


def time_to_fraction(time, front, radius: float, final: float | None = None,
                     fraction: float = PRIMARY_FRACTION) -> float:
    """First time the front reaches ``radius + fraction * (final - radius)``."""
    time, front = _as_history(time, front)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ArrivalError("radius must be finite and positive")
    if not 0.0 < fraction <= 1.0:
        raise ArrivalError("fraction must lie in (0, 1]")
    target_final = float(front[-1] if final is None else final)
    runout = target_final - float(radius)
    if runout <= 0.0:
        raise ArrivalError("final front must exceed the initial radius")
    target = float(radius) + fraction * runout
    idx = np.nonzero(front >= target)[0]
    return float(time[idx[0]]) if idx.size else math.nan


def time_to_velocity_threshold(time, front,
                               fraction: float = PRIMARY_VELOCITY_FRACTION,
                               window_s: float = DEFAULT_WINDOW_S) -> float:
    """First time after the speed peak at which ``|v| < fraction * v_peak``.

    ``v_peak`` is the maximum smoothed front speed of *this* run, so the test is
    scale-free.  Returns ``nan`` if the front never slows that far inside the
    run (a truncated run may not declare arrival).
    """
    time, front = _as_history(time, front)
    if not 0.0 < fraction < 1.0:
        raise ArrivalError("velocity fraction must lie in (0, 1)")
    speed = front_speed(time, front, window_s)
    peak_idx = int(np.argmax(speed))
    peak = float(speed[peak_idx])
    if not math.isfinite(peak) or peak <= 0.0:
        raise ArrivalError("front speed peak must be positive and finite")
    below = np.nonzero(speed[peak_idx:] < fraction * peak)[0]
    return float(time[peak_idx + below[0]]) if below.size else math.nan


def time_to_length_tolerance(time, front, tolerance: float,
                             final: float | None = None) -> float:
    """First time the front is within ``tolerance`` (m) of its final value."""
    time, front = _as_history(time, front)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ArrivalError("tolerance must be finite and positive")
    target = float(front[-1] if final is None else final) - float(tolerance)
    idx = np.nonzero(front >= target)[0]
    return float(time[idx[0]]) if idx.size else math.nan


def arrival_report(time, front, *, radius: float, grain_diameter: float,
                   final: float | None = None,
                   window_s: float = DEFAULT_WINDOW_S,
                   fraction_levels=FRACTION_LEVELS,
                   velocity_fractions=VELOCITY_FRACTIONS,
                   length_tolerances_d=LENGTH_TOLERANCES_D) -> dict:
    """Every pre-registered arrival definition of one run, in one dict."""
    time, front = _as_history(time, front)
    if not math.isfinite(grain_diameter) or grain_diameter <= 0.0:
        raise ArrivalError("grain_diameter must be finite and positive")
    final_front = float(front[-1] if final is None else final)
    report = {
        "radius_initial": float(radius),
        "front_final": final_front,
        "runout": final_front - float(radius),
        "front_final_overshoot": float(front[-1]) - final_front,
        "window_s": float(window_s),
        "sample_dt": float(np.median(np.diff(time))),
        "n_samples": int(time.size),
        "peak_front_speed_smoothed": float(front_speed(time, front,
                                                       window_s).max()),
        "fraction": {f"{f:g}": time_to_fraction(time, front, radius, final_front, f)
                     for f in fraction_levels},
        "velocity": {f"{f:g}": time_to_velocity_threshold(time, front, f, window_s)
                     for f in velocity_fractions},
        "length": {f"{n:g}": time_to_length_tolerance(time, front,
                                                      n * grain_diameter, final_front)
                   for n in length_tolerances_d},
    }
    report["primary"] = {
        "fraction_0.9": report["fraction"][f"{PRIMARY_FRACTION:g}"],
        "velocity_0.01": report["velocity"][f"{PRIMARY_VELOCITY_FRACTION:g}"],
        "length_1d": report["length"][f"{PRIMARY_LENGTH_TOLERANCE_D:g}"],
    }
    return report