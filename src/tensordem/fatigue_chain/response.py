"""Exact SDOF / modal transfer of a force history to hot-spot stress.

The chain between the DEM force ``F(t)`` and the S-N curve is a linear
structural operator.  Two references are provided:

* :func:`sdof_response` -- the **exact** piecewise-linear (Nigam-Jennings)
  recurrence for

      q'' + 2 zeta wn q' + wn^2 q = wn^2 F(t)

  normalised so that the DC gain is 1 (a static force ``F`` gives ``q = F``).
  Over a step where ``F`` is linear the particular solution is
  ``q_p = f0 + df (tau/h - 2 zeta / wn)`` and the homogeneous part is the usual
  damped sinusoid; the recurrence is exact for the interpolated input, i.e. the
  only error of the transfer is the piecewise-linear representation of the
  record.
* :func:`half_sine_closed_form` -- the textbook undamped Duhamel solution for a
  half-sine pulse ``F0 sin(pi t / td)``:

      q(t) = [sin(w t) - r sin(wn t)] / (1 - r^2),   r = w / wn, w = pi/td

  used as the independent anchor for the single-pulse judge
  (peak displacement / peak stress vs closed form < 3 %).

A :class:`ModalChannel` is one mode of the hot-spot transfer: natural frequency,
damping ratio and static gain in Pa per newton.  Superposing the channels gives
the stress history

      sigma(t) = sum_i  g_i * q_i(t)

which is what the rainflow counter sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2 renamed trapz

__all__ = [
    "ModalChannel",
    "sdof_response",
    "half_sine_closed_form",
    "modal_stress_history",
    "duhamel_convolution",
]


@dataclass(frozen=True)
class ModalChannel:
    """One mode of the force -> stress transfer."""

    freq_hz: float
    damping: float
    static_gain_Pa_per_N: float

    def __post_init__(self) -> None:
        if not (self.freq_hz > 0.0):
            raise ValueError("freq_hz must be positive")
        if not (0.0 <= self.damping < 1.0):
            raise ValueError("damping must be in [0, 1)")
        if self.static_gain_Pa_per_N == 0.0:
            raise ValueError("static_gain_Pa_per_N must be non-zero")

    @property
    def omega_n(self) -> float:
        return 2.0 * math.pi * self.freq_hz


def sdof_response(t, force, freq_hz: float, damping: float = 0.0) -> np.ndarray:
    """Exact SDOF response (DC gain 1) to a piecewise-linear force record.

    Initial condition: the system is in static equilibrium with the first
    sample, ``q(0) = force[0]``, ``q'(0) = 0``.  For a record that starts from
    rest with ``force[0] = 0`` -- the case of every load history in this
    package -- that is zero initial state, which is what
    :func:`duhamel_convolution` assumes; the two then agree to ~1e-7
    (verified in the tests).
    """
    tt = np.asarray(t, dtype=float).ravel()
    ff = np.asarray(force, dtype=float).ravel()
    if tt.size != ff.size:
        raise ValueError("t and force must have the same length")
    if tt.size == 0:
        raise ValueError("empty record")
    wn = 2.0 * math.pi * float(freq_hz)
    zeta = float(damping)
    if wn <= 0.0:
        raise ValueError("freq_hz must be positive")
    if not (0.0 <= zeta < 1.0):
        raise ValueError("damping must be in [0, 1)")
    q = np.empty(tt.size, dtype=float)
    q[0] = ff[0]
    v = 0.0
    wd = wn * math.sqrt(1.0 - zeta * zeta)
    for i in range(1, tt.size):
        h = tt[i] - tt[i - 1]
        if h <= 0.0:
            q[i] = q[i - 1]
            continue
        f0, f1 = ff[i - 1], ff[i]
        df = f1 - f0
        # particular solution of the ramp: q_p(tau) = C + D tau with
        # D = df/h and C = f0 - 2 zeta df/(wn h); the 1/h term is what the
        # damped case needs (an earlier version dropped it and was 3.4 % low
        # against both the Duhamel convolution and RK4)
        shift = 2.0 * zeta * df / (wn * h)
        qp0 = f0 - shift
        a = q[i - 1] - qp0
        b = (v - df / h + zeta * wn * a) / wd
        e = math.exp(-zeta * wn * h)
        c, s = math.cos(wd * h), math.sin(wd * h)
        q[i] = e * (a * c + b * s) + f1 - shift
        # derivative at the new state
        v = e * ((-zeta * wn * a + b * wd) * c + (-zeta * wn * b - a * wd) * s) + df / h
    return q


def half_sine_closed_form(t, force_peak: float, duration_s: float, freq_hz: float,
                          damping: float = 0.0) -> np.ndarray:
    """Undamped closed-form response to a half-sine pulse (``q``, DC gain 1)."""
    tt = np.asarray(t, dtype=float).ravel()
    if damping != 0.0:
        raise ValueError("half_sine_closed_form is the undamped Duhamel solution")
    wn = 2.0 * math.pi * float(freq_hz)
    w = math.pi / float(duration_s)
    r = w / wn
    if abs(1.0 - r * r) < 1e-12:
        raise ValueError("resonant half-sine pulse (r = 1)")
    q = np.empty(tt.size, dtype=float)
    inside = tt <= duration_s
    ti = tt[inside]
    q[inside] = (np.sin(w * ti) - r * np.sin(wn * ti)) / (1.0 - r * r)
    if tt.size and not np.all(inside):
        qd = (math.sin(w * duration_s) - r * math.sin(wn * duration_s)) / (1.0 - r * r)
        vd = (w * math.cos(w * duration_s) - r * wn * math.cos(wn * duration_s)) / (1.0 - r * r)
        tau = tt[~inside] - duration_s
        q[~inside] = qd * np.cos(wn * tau) + (vd / wn) * np.sin(wn * tau)
    return force_peak * q


def duhamel_convolution(t, force, freq_hz: float, damping: float = 0.0) -> np.ndarray:
    """Reference response by numerical Duhamel convolution (fine grid)."""
    tt = np.asarray(t, dtype=float).ravel()
    ff = np.asarray(force, dtype=float).ravel()
    if tt.size != ff.size or tt.size == 0:
        raise ValueError("t and force must be non-empty and equal length")
    wn = 2.0 * math.pi * float(freq_hz)
    zeta = float(damping)
    wd = wn * math.sqrt(1.0 - zeta * zeta)
    # response = int_0^t h(t - tau) wn^2 F(tau) / wn^2 dtau, h impulse response
    # of the DC-normalised system: h(s) = wn^2 * e^{-zeta wn s} sin(wd s)/(m wd)
    # with m = 1 evaluated on the same record: q(t) = int_0^t wn/sqrt(1-zeta^2)
    #   e^{-zeta wn (t-tau)} sin(wd (t-tau)) F(tau) dtau
    q = np.zeros(tt.size, dtype=float)
    for i in range(tt.size):
        if i == 0:
            q[i] = ff[0]
            continue
        s = tt[i] - tt[: i + 1]
        kernel = (wn / math.sqrt(1.0 - zeta * zeta)) * np.exp(-zeta * wn * s) * np.sin(wd * s)
        vals = ff[: i + 1] * kernel
        if i == 1:
            q[i] = vals[0] * (tt[1] - tt[0])
            continue
        q[i] = _trapz(vals, tt[: i + 1])
    return q


def modal_stress_history(t, force, channels: Sequence[ModalChannel]) -> np.ndarray:
    """Hot-spot stress history = sum over modes of gain * SDOF response."""
    tt = np.asarray(t, dtype=float).ravel()
    ff = np.asarray(force, dtype=float).ravel()
    if not channels:
        raise ValueError("at least one modal channel is required")
    sigma = np.zeros_like(ff, dtype=float)
    for ch in channels:
        q = sdof_response(tt, ff, ch.freq_hz, ch.damping)
        sigma += ch.static_gain_Pa_per_N * q
    return sigma