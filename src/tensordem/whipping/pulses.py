"""Segmentation and parametric reconstruction of ice-hull contact-force pulses.

The module turns a sampled DEM ice-load record ``F(t)`` into a list of discrete
pulse events and, for every event, an *analytic* parametric time history that
reproduces the four engineering pulse features

* ``t_r``      rise time (10 % level -> peak),
* ``F_peak``   peak force,
* ``tau``      pulse width (full width at half maximum), and
* ``I``        impulse ``int F dt`` over the event.

The parametric pulse is a two-branch ``sin^p`` / ``cos^p`` shape

    F_p(t) = F_peak sin^p(pi/2 * (t - t0)/t_r)          t0 <= t <= t_peak
             F_peak cos^p(pi/2 * (t - t_peak)/t_d)      t_peak < t <= t_end

with ``t_r`` and the decay time ``t_d`` taken from the measurement.  The single
free exponent ``p`` is fixed by requiring the *exact* impulse

    I_p = F_peak c_p (t_r + t_d),   c_p = int_0^1 sin^p(pi u / 2) du
        = Gamma((p+1)/2) / (sqrt(pi) Gamma((p+2)/2))

to equal the measured impulse; ``c_p`` is evaluated from the Gamma function, so
the impulse match is limited only by the root-finder tolerance (no quadrature
error enters the parametrisation).  ``c_p`` decreases monotonically from
``c_0 = 1`` to ``c_inf = 0``, hence a solution exists for every physically
admissible ratio ``I / (F_peak (t_r + t_d)) in (0, 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln

__all__ = [
    "PulseFeatures",
    "ParametricPulse",
    "detect_pulses",
    "resample_uniform",
    "sine_power_integral",
    "pulse_train",
]


# --------------------------------------------------------------------------- #
# sampling helpers
# --------------------------------------------------------------------------- #
def resample_uniform(times, values, sampling_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample ``(times, values)`` onto a uniform grid of ``sampling_rate_hz``.

    The source record is interpolated piecewise-linearly; the returned grid
    spans the source record so that no part of the signal is extrapolated.
    """
    t = np.asarray(times, dtype=float).ravel()
    v = np.asarray(values, dtype=float).ravel()
    if t.size != v.size or t.size < 2:
        raise ValueError("times and values must be equally sized with >= 2 samples")
    if not (np.isfinite(t).all() and np.isfinite(v).all()):
        raise ValueError("record must be finite")
    if not math.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    if not np.all(np.diff(t) > 0):
        raise ValueError("times must be strictly increasing")
    dt = 1.0 / sampling_rate_hz
    n = int(math.floor((t[-1] - t[0]) / dt)) + 1
    grid = t[0] + dt * np.arange(n)
    return grid, np.interp(grid, t, v)


def _crossings(t: np.ndarray, y: np.ndarray, level: float, rising: bool) -> np.ndarray:
    """Linear-interpolated times at which ``y`` crosses ``level``."""
    f = y - level
    sign = np.sign(f)
    sign[sign == 0.0] = 1.0
    prod = sign[:-1] * sign[1:]
    if rising:
        idx = np.flatnonzero((prod < 0) & (sign[1:] > 0))
    else:
        idx = np.flatnonzero((prod < 0) & (sign[1:] < 0))
    if idx.size == 0:
        return np.empty(0, dtype=float)
    y0, y1 = y[idx], y[idx + 1]
    frac = (level - y0) / (y1 - y0)
    return t[idx] + frac * (t[idx + 1] - t[idx])


def _interp_level(t: np.ndarray, y: np.ndarray, i: int, level: float, forward: bool,
                  lo: int = 0, hi: int | None = None) -> float:
    """Time where ``y`` reaches ``level`` walking from index ``i`` inside [lo, hi]."""
    if hi is None:
        hi = t.size - 1
    if forward:
        j = i
        while j + 1 <= hi and y[j + 1] < level:
            j += 1
        if j + 1 > hi:
            return float(t[hi])
        y0, y1 = y[j], y[j + 1]
        frac = (level - y0) / (y1 - y0) if y1 != y0 else 0.0
        return float(t[j] + frac * (t[j + 1] - t[j]))
    j = i
    while j - 1 >= lo and y[j - 1] < level:
        j -= 1
    if j - 1 < lo:
        return float(t[lo])
    y0, y1 = y[j], y[j - 1]
    frac = (level - y0) / (y1 - y0) if y1 != y0 else 0.0
    return float(t[j] + frac * (t[j - 1] - t[j]))


# --------------------------------------------------------------------------- #
# pulse description
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PulseFeatures:
    """Four-feature description of one ice-impact pulse event."""

    index: int
    t_onset: float
    t_peak: float
    t_end: float
    peak: float
    rise_time: float
    decay_time: float
    width_fwhm: float
    impulse: float
    threshold: float

    @property
    def pulse_width(self) -> float:
        return self.width_fwhm

    def as_dict(self) -> dict:
        return {
            "index": int(self.index),
            "t_onset_s": self.t_onset,
            "t_peak_s": self.t_peak,
            "t_end_s": self.t_end,
            "F_peak_N": self.peak,
            "rise_time_s": self.rise_time,
            "decay_time_s": self.decay_time,
            "width_fwhm_s": self.width_fwhm,
            "impulse_Ns": self.impulse,
            "threshold_N": self.threshold,
        }


def sine_power_integral(p: float) -> float:
    """``int_0^1 sin^p(pi u / 2) du`` evaluated through the Gamma function."""
    if not math.isfinite(p) or p <= 0.0:
        raise ValueError("exponent must be finite and positive")
    log_c = -0.5 * math.log(math.pi) + gammaln(0.5 * (p + 1.0)) - gammaln(0.5 * (p + 2.0))
    return math.exp(log_c)


@dataclass(frozen=True)
class ParametricPulse:
    """Analytic ``sin^p``/``cos^p`` reconstruction of :class:`PulseFeatures`."""

    features: PulseFeatures
    exponent: float
    impulse_ratio: float = field(default=1.0)

    @classmethod
    def from_features(cls, features: PulseFeatures) -> "ParametricPulse":
        t_r = features.rise_time
        t_d = features.decay_time
        if not (t_r > 0.0 and t_d > 0.0 and features.peak > 0.0):
            raise ValueError("pulse must have positive rise/decay times and peak")
        span = features.peak * (t_r + t_d)
        ratio = features.impulse / span
        if not (0.0 < ratio < 1.0):
            raise ValueError(
                f"impulse ratio {ratio:.4f} outside (0, 1); record is not a single pulse"
            )
        lo, hi = 1.0e-3, 1.0e3
        while sine_power_integral(lo) < ratio:
            lo *= 0.5
        while sine_power_integral(hi) > ratio:
            hi *= 2.0
        p = brentq(lambda q: sine_power_integral(q) - ratio, lo, hi, xtol=1e-13, rtol=1e-15)
        return cls(features=features, exponent=p)

    # -- evaluation ------------------------------------------------------- #
    def __call__(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        f = self.features
        out = np.zeros_like(t)
        p = self.exponent
        up = (t >= f.t_onset) & (t <= f.t_peak)
        if np.any(up):
            u = np.clip((t[up] - f.t_onset) / f.rise_time, 0.0, 1.0)
            out[up] = f.peak * np.clip(np.sin(0.5 * math.pi * u), 0.0, None) ** p
        down = (t > f.t_peak) & (t <= f.t_end)
        if np.any(down):
            u = np.clip((t[down] - f.t_peak) / f.decay_time, 0.0, 1.0)
            out[down] = f.peak * np.clip(np.cos(0.5 * math.pi * u), 0.0, None) ** p
        return out

    def impulse(self) -> float:
        f = self.features
        return f.peak * sine_power_integral(self.exponent) * (f.rise_time + f.decay_time)

    def width_fwhm(self) -> float:
        p = self.exponent
        u_half = (2.0 / math.pi) * math.asin(0.5 ** (1.0 / p))
        f = self.features
        return (1.0 - u_half) * (f.rise_time + f.decay_time)

    def as_dict(self) -> dict:
        f = self.features
        return {
            **f.as_dict(),
            "exponent_p": self.exponent,
            "impulse_reconstructed_Ns": self.impulse(),
            "impulse_rel_error": (self.impulse() - f.impulse) / f.impulse if f.impulse else 0.0,
            "width_fwhm_reconstructed_s": self.width_fwhm(),
            "width_rel_error": (
                (self.width_fwhm() - f.width_fwhm) / f.width_fwhm if f.width_fwhm else 0.0
            ),
        }


# --------------------------------------------------------------------------- #
# pulse detection
# --------------------------------------------------------------------------- #
def detect_pulses(
    times,
    force,
    *,
    threshold_fraction: float = 0.10,
    min_separation_s: float = 0.005,
    min_peak_fraction: float = 0.05,
) -> list[PulseFeatures]:
    """Split a DEM contact-force record into pulse events.

    ``threshold_fraction`` is the fraction of the record maximum that defines
    both the event onset level and the "still in the same event" hysteresis
    level.  Events whose peaks stay below ``min_peak_fraction`` of the record
    maximum are discarded, and events separated by less than
    ``min_separation_s`` are merged (they belong to one contact episode).
    """
    t = np.asarray(times, dtype=float).ravel()
    f = np.asarray(force, dtype=float)
    if f.ndim != 1 or f.size != t.size:
        raise ValueError("force must be a 1-D record matching times")
    if t.size < 3 or not np.all(np.diff(t) > 0):
        raise ValueError("times must be strictly increasing with >= 3 samples")
    if not (math.isfinite(threshold_fraction) and 0.0 < threshold_fraction < 1.0):
        raise ValueError("threshold_fraction must lie in (0, 1)")
    peak_max = float(np.max(f))
    if not peak_max > 0.0:
        raise ValueError("record must contain a positive force")

    level = threshold_fraction * peak_max
    onsets = _crossings(t, f, level, rising=True)
    offsets = _crossings(t, f, level, rising=False)

    events: list[list[float]] = []
    for t_on in onsets:
        later = offsets[offsets > t_on]
        t_off = float(later[0]) if later.size else float(t[-1])
        if events and t_on - events[-1][1] < min_separation_s:
            events[-1][1] = max(events[-1][1], t_off)
        else:
            events.append([float(t_on), t_off])

    pulses: list[PulseFeatures] = []
    for t_on, t_off in events:
        mask = (t >= t_on) & (t <= t_off)
        if not np.any(mask):
            continue
        tt, ff = t[mask], f[mask]
        i_peak = int(np.argmax(ff))
        peak = float(ff[i_peak])
        if peak < min_peak_fraction * peak_max:
            continue
        t_peak = float(tt[i_peak])
        half = 0.5 * peak
        # walk out only as far as the neighbouring local minima (a later, larger
        # event must not stretch this pulse's half-width)
        j = i_peak
        while j + 1 < ff.size and ff[j + 1] <= ff[j]:
            j += 1
        right = _interp_level(tt, ff, i_peak, half, forward=True, lo=i_peak, hi=j)
        j = i_peak
        while j - 1 >= 0 and ff[j - 1] <= ff[j]:
            j -= 1
        left = _interp_level(tt, ff, i_peak, half, forward=False, lo=j, hi=i_peak)
        pulse = PulseFeatures(
            index=len(pulses),
            t_onset=t_on,
            t_peak=t_peak,
            t_end=float(t_off),
            peak=peak,
            rise_time=max(t_peak - t_on, 0.0),
            decay_time=max(float(t_off) - t_peak, 0.0),
            width_fwhm=max(right - left, 0.0),
            impulse=float(np.trapezoid(ff, tt)),
            threshold=level,
        )
        if pulse.rise_time > 0.0 and pulse.decay_time > 0.0:
            pulses.append(pulse)
    return pulses


def pulse_train(pulses, times) -> np.ndarray:
    """Superpose parametric pulses on the ``times`` grid."""
    out = np.zeros_like(np.asarray(times, dtype=float))
    for pulse in pulses:
        out += pulse(times)
    return out