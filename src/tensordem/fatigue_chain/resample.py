"""Sampling-rate compression of a load history: naive vs peak-preserving.

Two reductions of a uniformly sampled record to a lower sampling rate ``fs``:

``naive_decimate``
    the record a real DAQ delivers at ``fs``: a zero-phase 4th-order
    Butterworth anti-alias low-pass at ``0.4 fs`` followed by point sampling on
    the ``fs`` grid (``anti_alias=False`` reproduces a raw logger without
    filtering, which is reported separately because it aliases).  This is the
    reduction that loses pulse peaks.

``peak_preserving_decimate``
    min/max decimation with bin-boundary anchors: inside every ``1/fs`` bin the
    two extremes are kept *at their original times* and the bin boundaries keep
    their true sample values, so the excursion is confined to its bin (a naive
    min/max stream, which drops the boundary anchors, manufactures a
    low-frequency sawtooth out of two adjacent bin extremes -- measured as a
    +10^2 damage error at 1 Hz on the m5405 record).  A pulse shorter than the
    sampling interval is therefore still seen at its true amplitude.

Both are transport-level operations on ``(t, x)``: they return a compressed
sample set which :func:`reconstruct` maps back onto an analysis grid (linear
interpolation, i.e. the anti-imaging filter a logger's replay/plot uses).  The
damage chain in :mod:`tensordem.fatigue_chain.damage` then consumes the
reconstructed record, so the reported ``dD/D`` is the error of the *compressed
record*, not of a hypothetical idealised sampler.

Known limitation: linear reconstruction cannot restore the amplitude between
two retained samples, so peak-preserving compression systematically *widens*
pulses to the bin width.  That widening is part of the measured error and is
reported as such (it is also why the method is not free at very low rates).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

__all__ = [
    "naive_decimate",
    "peak_preserving_decimate",
    "compress",
    "reconstruct",
    "METHODS",
]

METHODS = ("naive", "raw", "peak")


def _check(t, x) -> tuple[np.ndarray, np.ndarray]:
    tt = np.asarray(t, dtype=float).ravel()
    xx = np.asarray(x, dtype=float).ravel()
    if tt.size != xx.size:
        raise ValueError("t and x must have the same length")
    if tt.size < 2:
        raise ValueError("at least two samples are required")
    if np.any(np.diff(tt) <= 0.0):
        raise ValueError("t must be strictly increasing")
    return tt, xx


def naive_decimate(t, x, rate_hz: float, *, anti_alias: bool = True
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Anti-aliased point sampling on a ``rate_hz`` grid."""
    tt, xx = _check(t, x)
    if not rate_hz > 0.0:
        raise ValueError("rate_hz must be positive")
    step = 1.0 / rate_hz
    if step <= (tt[1] - tt[0]):
        return tt.copy(), xx.copy()
    signal = xx
    if anti_alias:
        dt = float(np.median(np.diff(tt)))
        signal = _lowpass(xx, dt, 0.4 * rate_hz)
    n = int(np.floor((tt[-1] - tt[0]) / step)) + 1
    grid = tt[0] + step * np.arange(n)
    idx = np.clip(np.searchsorted(tt, grid), 0, tt.size - 1)
    keep = np.unique(idx)
    return tt[keep], signal[keep]


def _lowpass(x: np.ndarray, dt: float, cutoff_hz: float) -> np.ndarray:
    """Zero-phase Butterworth low-pass of a uniformly sampled record."""
    fs = 1.0 / dt
    wn = 2.0 * cutoff_hz / fs
    if wn >= 1.0 or wn <= 0.0 or x.size < 15:
        return np.asarray(x, dtype=float)
    b, a = butter(4, wn, btype="low")
    return filtfilt(b, a, np.asarray(x, dtype=float))


def peak_preserving_decimate(t, x, rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Min/max per bin with the extremes at their original times.

    Each bin contributes ``(bin start, extreme 1, extreme 2, bin end)`` in
    occurrence order, values taken from the true samples, so the extreme
    amplitude survives and the excursion stays inside its own bin.
    """
    tt, xx = _check(t, x)
    if not rate_hz > 0.0:
        raise ValueError("rate_hz must be positive")
    step = 1.0 / rate_hz
    if step <= (tt[1] - tt[0]):
        return tt.copy(), xx.copy()
    bins = np.floor((tt - tt[0]) / step).astype(int)
    out_t: list = []
    out_x: list = []
    for b in np.unique(bins):
        sel = np.flatnonzero(bins == b)
        if sel.size == 0:
            continue
        lo = int(sel[np.argmin(xx[sel])])
        hi = int(sel[np.argmax(xx[sel])])
        t_bin = float(tt[0] + b * step)
        pts: list[tuple[float, float]] = [(t_bin, float(xx[sel[0]]))]
        for idx in sorted({lo, hi}, key=lambda k: tt[k]):
            if float(tt[idx]) > pts[-1][0]:
                pts.append((float(tt[idx]), float(xx[idx])))
        t_end_bin = t_bin + step
        if t_end_bin > pts[-1][0]:
            pts.append((t_end_bin, float(xx[sel[-1]])))
        for tv, xv in pts:
            if out_t and tv <= out_t[-1]:
                continue
            out_t.append(tv)
            out_x.append(xv)
    return np.asarray(out_t), np.asarray(out_x)


def compress(t, x, rate_hz: float, method: str = "peak", *,
             anti_alias: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to ``naive`` or ``peak`` compression."""
    if method == "naive":
        return naive_decimate(t, x, rate_hz, anti_alias=anti_alias)
    if method == "raw":
        return naive_decimate(t, x, rate_hz, anti_alias=False)
    if method == "peak":
        return peak_preserving_decimate(t, x, rate_hz)
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


def reconstruct(t_target, t_compressed, x_compressed) -> np.ndarray:
    """Linear reconstruction of a compressed record on the analysis grid."""
    tt = np.asarray(t_target, dtype=float).ravel()
    tc = np.asarray(t_compressed, dtype=float).ravel()
    xc = np.asarray(x_compressed, dtype=float).ravel()
    if tc.size == 0:
        raise ValueError("compressed record is empty")
    if tc.size == 1:
        return np.full_like(tt, xc[0])
    return np.interp(tt, tc, xc)