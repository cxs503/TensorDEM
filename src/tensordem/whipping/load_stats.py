"""Statistical characterisation of ice-hull contact-load time histories.

The whipping chain (:mod:`tensordem.whipping.pipeline`) answers "what does one
DEM pulse train do to the hull girder".  This module answers the preceding
question: *does the load history itself look like a real ice-pressure record?*
Peak force alone does not: a record made of a few isolated spikes and a record
of continuous crushing can share the same peak and differ completely in
whipping content, fatigue content and spectral shape.

Definitions implemented here (all on the same 1 kHz record):

``R``          load ratio ``mean(F) / max(F)``.  Full-scale ice-pressure
               records are neither a constant (``R -> 1``) nor a sparse spike
               train (``R -> 0``); the campaign band is ``R in [0.30, 0.50]``.
``t_r``        rise time of an event, measured between the 10 % and the 90 %
               level of *that event's* peak (10 %-90 %, not 10 %-peak).
``t_h``        hold time of an event: full width at half maximum of the same
               event (time spent at or above 50 % of its peak).  ``t_h90``
               (dwell above 90 %) is reported as well for peak-plateau checks.
``dt_on``      inter-arrival intervals between successive event onsets
               (distribution summarised by mean / coefficient of variation /
               percentiles).
``slope``      least-squares slope of ``log10 S_FF`` against ``log10 f`` over a
               prescribed band (0.5-10 Hz by default), in decade/decade.
``N_rf``       ASTM E1049-85 rainflow cycle count of the record.
``phi_qs``     quasi-static energy fraction, i.e. the share of the load
               variance carried by ``f < 1 Hz``.

Everything is a pure function of the sampled record, so each definition can be
checked against an analytic signal (see ``tests/test_ice_load_stats_calibration.py``):
a sine wave has ``R = 2/pi`` exactly, a ramp of known slope has a known rise
time, band-limited white noise has a known spectral slope of zero and a
superposition of exponential decays with a ``1/tau`` density has
``slope = -1`` decade/decade.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np
from scipy.signal import find_peaks, welch

from .rainflow import rainflow as _rainflow, total_count as _total_count

__all__ = [
    "Event",
    "detect_events",
    "dispersion",
    "record_statistics",
    "segment_statistics",
    "spectral_slope",
    "welch_psd",
]

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _crossings(t: np.ndarray, y: np.ndarray, level: float, rising: bool) -> np.ndarray:
    """Linearly interpolated crossing times of ``y`` through ``level``."""
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


def _first(time: np.ndarray, mask) -> float | None:
    return float(time[mask][0]) if np.any(mask) else None


@dataclass(frozen=True)
class Event:
    """One ice-hull contact event and its 10 %/50 %/90 % feature set."""

    index: int
    t_onset: float          # first upward crossing of the detection level
    t_end: float            # downward crossing of the detection level
    t_peak: float
    peak: float
    level: float            # detection level (Pa or N)
    rise_time: float        # 10 % -> 90 % of this event's peak
    hold_time: float        # FWHM (time above 50 % of this event's peak)
    hold_time_90: float     # dwell above 90 %
    decay_time: float       # peak -> 10 % on the trailing edge
    impulse: float          # int F dt over the event
    complete: bool          # both 10 % and 90 % levels were crossed

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# event segmentation
# --------------------------------------------------------------------------- #
def _rise_cross(t, y, lo_i: int, hi_i: int, level: float):
    """Last upward crossing of ``level`` before ``hi_i`` (interpolated).

    The *last* crossing (not the first) is the one belonging to the peak at
    ``hi_i``: in an oscillating record the first crossing in the window belongs
    to an earlier cycle.
    """
    seg = y[lo_i:hi_i + 1]
    below = np.flatnonzero(seg < level)
    if below.size == 0:
        return float(t[lo_i])
    k = int(below[-1]) + 1 + lo_i
    if k > hi_i:
        return None
    if k == lo_i:
        return float(t[k])
    y0, y1 = float(y[k - 1]), float(y[k])
    frac = (level - y0) / (y1 - y0) if y1 != y0 else 0.0
    return float(t[k - 1] + frac * (t[k] - t[k - 1]))


def _fall_cross(t, y, lo_i: int, hi_i: int, level: float):
    """First downward crossing of ``level`` on ``y[lo_i:hi_i+1]`` (interpolated)."""
    seg = y[lo_i:hi_i + 1]
    below = np.flatnonzero(seg <= level)
    if below.size == 0:
        return None
    k = int(below[0]) + lo_i
    if k == lo_i:
        return float(t[k])
    y0, y1 = float(y[k - 1]), float(y[k])
    frac = (level - y0) / (y1 - y0) if y1 != y0 else 0.0
    return float(t[k - 1] + frac * (t[k] - t[k - 1]))


def detect_events(times, force, *, prominence_fraction: float = 0.03,
                  trough_half_window_s: float = 0.30) -> list[Event]:
    """Split a load record into contact events, one per local force maximum.

    Dense crushing records overlap: a record can stay above any fixed level for
    seconds, so level-crossing segmentation merges everything into one event
    and the rise time degenerates.  Each event is therefore attached to a local
    maximum found by ``scipy.signal.find_peaks`` with a prominence of
    ``prominence_fraction`` of the record's dynamic range, and its pre-event
    level is the **left base** returned by the same prominence test: the lowest
    point between this peak and the previous higher peak.  That baseline is the
    physically relevant one (the load level the pulse started from) and it is
    insensitive both to the crushing noise floor and to slow quasi-static
    wander.  ``trough_half_window_s`` only caps how far the base may be
    searched (it is passed to ``find_peaks`` as ``wlen``).

    The 10 %, 50 % and 90 % levels are fractions of the peak above its own
    left base; the onset is the last 10 % upward crossing before the peak, so
    inter-arrival statistics are onset-to-onset intervals.
    """
    t = np.asarray(times, dtype=float).ravel()
    f = np.asarray(force, dtype=float).ravel()
    if f.size != t.size or t.size < 16:
        raise ValueError("force must match times with >= 16 samples")
    if not np.all(np.diff(t) > 0):
        raise ValueError("times must be strictly increasing")
    if not (math.isfinite(prominence_fraction) and 0.0 < prominence_fraction < 1.0):
        raise ValueError("prominence_fraction must lie in (0, 1)")
    base_rec = float(np.min(f))
    span = float(np.max(f)) - base_rec
    if not span > 0.0:
        raise ValueError("record has no dynamic range")
    dt = float(t[1] - t[0])
    wlen = max(int(round(2.0 * trough_half_window_s / dt)) + 1, 3)
    peaks, props = find_peaks(f, prominence=prominence_fraction * span,
                              distance=2, wlen=wlen)
    if peaks.size == 0:
        return []
    left = props["left_bases"]
    right = props["right_bases"]

    events: list[Event] = []
    for k, i_peak in enumerate(peaks):
        pk = float(f[i_peak])
        base = float(f[left[k]])
        if not pk > base:
            continue
        lv10 = base + 0.10 * (pk - base)
        lv50 = base + 0.50 * (pk - base)
        lv90 = base + 0.90 * (pk - base)
        t10 = _rise_cross(t, f, int(left[k]), int(i_peak), lv10)
        t90 = _rise_cross(t, f, int(left[k]), int(i_peak), lv90)
        up50 = _rise_cross(t, f, int(left[k]), int(i_peak), lv50)
        down50 = _fall_cross(t, f, int(i_peak), int(right[k]), lv50)
        down90 = _fall_cross(t, f, int(i_peak), int(right[k]), lv90)
        t10_down = _fall_cross(t, f, int(i_peak), int(right[k]), lv10)
        complete = t10 is not None and t90 is not None
        events.append(Event(
            index=len(events),
            t_onset=float(t10) if t10 is not None else float(t[int(left[k])]),
            t_end=float(t10_down) if t10_down is not None else float(t[int(right[k])]),
            t_peak=float(t[i_peak]),
            peak=pk,
            level=base,
            rise_time=float(t90 - t10) if complete else 0.0,
            hold_time=max(float(down50 - up50), 0.0)
            if (up50 is not None and down50 is not None) else 0.0,
            hold_time_90=max(float(down90 - t90), 0.0)
            if (t90 is not None and down90 is not None) else 0.0,
            decay_time=max(float(t10_down - t[i_peak]), 0.0) if t10_down is not None else 0.0,
            impulse=float(np.trapezoid(f[int(left[k]):int(right[k]) + 1] - base,
                                       t[int(left[k]):int(right[k]) + 1])),
            complete=bool(complete),
        ))
    return events


def _pct(values, q: float) -> float:
    v = np.asarray(values, dtype=float)
    return float(np.percentile(v, q)) if v.size else 0.0


def dispersion(values) -> float:
    """``(max - min) / mean(|values|)`` -- the campaign dispersion metric."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return 0.0
    mean = float(np.mean(np.abs(v)))
    return float((v.max() - v.min()) / mean) if mean > 0.0 else 0.0


# --------------------------------------------------------------------------- #
# spectra
# --------------------------------------------------------------------------- #
def welch_psd(times, force, *, nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Welch PSD of a uniformly sampled record."""
    t = np.asarray(times, dtype=float).ravel()
    f = np.asarray(force, dtype=float).ravel()
    dt = float(t[1] - t[0])
    if nperseg is None:
        nperseg = int(min(8192, 2 ** int(math.floor(math.log2(max(f.size // 4, 16))))))
    nperseg = int(min(nperseg, f.size))
    freq, psd = welch(f - f.mean(), fs=1.0 / dt, nperseg=nperseg,
                      noverlap=nperseg // 2, detrend="constant")
    return freq, psd


def spectral_slope(freq, psd, band=(0.5, 10.0)) -> dict:
    """Least-squares slope of ``log10 S`` against ``log10 f`` inside ``band``."""
    freq = np.asarray(freq, dtype=float)
    psd = np.asarray(psd, dtype=float)
    lo, hi = float(band[0]), float(band[1])
    mask = (freq >= lo) & (freq <= hi) & (psd > 0.0)
    n = int(mask.sum())
    if n < 3:
        return {"slope_decade_per_decade": float("nan"), "r2": float("nan"),
                "points": n, "band_hz": [lo, hi]}
    x = np.log10(freq[mask])
    y = np.log10(psd[mask])
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0.0 else 1.0
    return {"slope_decade_per_decade": float(slope), "r2": float(r2),
            "points": n, "band_hz": [lo, hi]}


# --------------------------------------------------------------------------- #
# record statistics
# --------------------------------------------------------------------------- #
def record_statistics(times, force, *, quasi_static_hz: float = 1.0,
                      slope_band_hz=(0.5, 10.0), rainflow: bool = True,
                      prominence_fraction: float = 0.03,
                      trough_half_window_s: float = 0.30) -> dict:
    """All campaign load-history statistics of one record.

    Returns a JSON-serialisable dict; the keys are the campaign symbols
    (``ratio_R``, ``rise_time_median_s``, ``hold_time_median_s``,
    ``interarrival_*``, ``spectral_slope_decade_per_decade``,
    ``rainflow_cycles``, ``quasi_static_energy_fraction``) plus the diagnostics
    needed to audit them (event list summary, PSD fit quality, band edges).
    """
    t = np.asarray(times, dtype=float).ravel()
    f = np.asarray(force, dtype=float).ravel()
    if f.size != t.size or t.size < 8:
        raise ValueError("force must match times with >= 8 samples")
    if not np.all(np.diff(t) > 0):
        raise ValueError("times must be strictly increasing")
    if not np.allclose(np.diff(t), t[1] - t[0], rtol=1e-9, atol=1e-15):
        raise ValueError("time grid must be uniform")
    dt = float(t[1] - t[0])

    events = detect_events(t, f, prominence_fraction=prominence_fraction,
                           trough_half_window_s=trough_half_window_s)
    rises = [e.rise_time for e in events if e.complete]
    holds = [e.hold_time for e in events]
    decays = [e.decay_time for e in events]
    onsets = np.array([e.t_onset for e in events], dtype=float)
    intervals = np.diff(onsets) if onsets.size > 1 else np.empty(0)
    n_ev = len(events)
    duration = float(t[-1] - t[0])

    freq, psd = welch_psd(t, f)
    slope = spectral_slope(freq, psd, slope_band_hz)
    total_energy = float(np.trapezoid(psd, freq))
    qs_mask = freq < quasi_static_hz
    qs_energy = float(np.trapezoid(psd[qs_mask], freq[qs_mask])) if np.any(qs_mask) else 0.0
    phi_qs = qs_energy / total_energy if total_energy > 0.0 else 0.0

    stats = {
        "duration_s": duration,
        "sampling_rate_hz": 1.0 / dt,
        "samples": int(f.size),
        "peak_force_N": float(f.max()),
        "mean_force_N": float(f.mean()),
        "ratio_R": float(f.mean() / f.max()) if f.max() > 0.0 else float("nan"),
        "std_force_N": float(f.std()),
        "pulse_count": n_ev,
        "pulse_rate_hz": n_ev / duration if duration > 0.0 else 0.0,
        "duty_cycle": float(sum(e.t_end - e.t_onset for e in events) / duration)
        if duration > 0.0 else 0.0,
        "rise_time_median_s": _pct(rises, 50),
        "rise_time_p10_s": _pct(rises, 10),
        "rise_time_p90_s": _pct(rises, 90),
        "hold_time_median_s": _pct(holds, 50),
        "hold_time_90_median_s": _pct([e.hold_time_90 for e in events], 50),
        "decay_time_median_s": _pct(decays, 50),
        "interarrival_mean_s": float(intervals.mean()) if intervals.size else 0.0,
        "interarrival_std_s": float(intervals.std()) if intervals.size else 0.0,
        "interarrival_cv": (float(intervals.std() / intervals.mean())
                            if intervals.size and intervals.mean() > 0.0 else 0.0),
        "interarrival_p50_s": _pct(intervals, 50),
        "interarrival_p10_s": _pct(intervals, 10),
        "interarrival_p90_s": _pct(intervals, 90),
        "spectral_slope_decade_per_decade": slope["slope_decade_per_decade"],
        "spectral_slope_r2": slope["r2"],
        "spectral_slope_points": slope["points"],
        "spectral_slope_band_hz": list(slope["band_hz"]),
        "quasi_static_energy_fraction": float(phi_qs),
        "quasi_static_cutoff_hz": float(quasi_static_hz),
        "rainflow_cycles": 0.0,
    }
    if rainflow:
        cycles = _rainflow(f)
        stats["rainflow_cycles"] = _total_count(cycles)
        stats["rainflow_cycles_per_s"] = (stats["rainflow_cycles"] / duration
                                          if duration > 0.0 else 0.0)
    return stats


def segment_statistics(times, force, n_segments: int = 4, *,
                       quasi_static_hz: float = 1.0,
                       slope_band_hz=(0.5, 10.0), rainflow: bool = True) -> list[dict]:
    """Split a record into ``n_segments`` equal-time windows and describe each."""
    t = np.asarray(times, dtype=float).ravel()
    f = np.asarray(force, dtype=float).ravel()
    if not (2 <= n_segments <= len(t) // 8):
        raise ValueError("n_segments must be >= 2 and leave >= 8 samples each")
    edges = np.linspace(t[0], t[-1], n_segments + 1)
    out = []
    for k in range(n_segments):
        last = k == n_segments - 1
        mask = (t >= edges[k]) & ((t <= edges[k + 1]) if last else (t < edges[k + 1]))
        seg_t, seg_f = t[mask], f[mask]
        stats = record_statistics(seg_t, seg_f, quasi_static_hz=quasi_static_hz,
                                  slope_band_hz=slope_band_hz, rainflow=rainflow)
        stats["segment"] = k
        stats["t_start_s"] = float(seg_t[0])
        stats["t_end_s"] = float(seg_t[-1])
        out.append(stats)
    return out