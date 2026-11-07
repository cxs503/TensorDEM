"""Torque-pulsation spectral analysis for propeller-ice milling.

The acceptance test for the milling transient is spectral: the shaft torque must
pulsate at the blade-passing frequency

    f_BPF = n [rev/s] * Z

and its harmonics.  This module supplies the (small, testable) tool chain:

``detrend`` / ``hann_window`` / ``periodogram``
    mean-plus-linear detrend, Hann taper, one-sided PSD with the window power
    loss corrected (``sum(Pxx) * df`` equals the variance of the detrended
    signal up to the taper bias -- asserted in the tests);
``find_peaks_spectral``
    local maxima with parabolic (sub-bin) interpolation and a prominence gate;
``analyse_torque_spectrum``
    the acceptance report: dominant frequency, BPF, integer harmonic match, the
    harmonic amplitude ratio (peak amplitude over the broadband floor) and the
    pulsation metrics of the time series.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "detrend",
    "harmonic_amplitudes",
    "hann_window",
    "periodogram",
    "find_peaks_spectral",
    "blade_passing_frequency",
    "analyse_torque_spectrum",
    "harmonic_index",
]


def detrend(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Remove the mean and the least-squares linear trend of ``x(t)``."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.size != x.size:
        raise ValueError("t and x must have the same length")
    if t.size < 3:
        return x - np.mean(x)
    slope, intercept = np.polyfit(t, x, 1)
    return x - (slope * t + intercept)


def hann_window(n: int) -> np.ndarray:
    """Periodic Hann taper of length ``n``."""
    if n < 2:
        return np.ones(max(n, 1))
    return 0.5 - 0.5 * np.cos(2.0 * math.pi * np.arange(n) / n)


def periodogram(t: np.ndarray, x: np.ndarray, *, remove_trend: bool = True,
                taper: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """One-sided PSD of ``x(t)`` [unit^2/Hz] on the natural FFT grid.

    Returns ``(freqs, psd)`` with ``freqs[1] = 1 / (N dt)``.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.size != x.size or t.size < 8:
        raise ValueError("need at least 8 samples with matching t and x")
    dt = float(np.median(np.diff(t)))
    if not dt > 0.0:
        raise ValueError("time must increase")
    y = detrend(t, x) if remove_trend else x - np.mean(x)
    window = hann_window(y.size) if taper else np.ones(y.size)
    scale = 1.0 / (np.sum(window ** 2))
    spectrum = np.fft.rfft(y * window)
    # |X_k|^2 / (fs * sum(w^2)); one-sided doubling below
    psd = (np.abs(spectrum) ** 2) * scale * dt
    freqs = np.fft.rfftfreq(y.size, d=dt)
    if psd.size > 2:
        psd[1:-1] *= 2.0
    return freqs, psd


def find_peaks_spectral(freqs: np.ndarray, psd: np.ndarray, *, n_peaks: int = 6,
                        min_freq: float | None = None,
                        prominence_ratio: float = 1.5) -> list[tuple[float, float]]:
    """Local PSD maxima with parabolic sub-bin interpolation.

    Returns a list of ``(frequency, amplitude)`` sorted by decreasing amplitude,
    where the amplitude is the interpolated PSD value at the peak.
    """
    f = np.asarray(freqs, dtype=float)
    p = np.asarray(psd, dtype=float)
    if f.size != p.size:
        raise ValueError("freqs and psd must have the same length")
    lo = 1 if min_freq is None else int(np.searchsorted(f, min_freq))
    peaks: list[tuple[float, float]] = []
    for k in range(max(lo, 1), p.size - 1):
        if p[k] <= p[k - 1] or p[k] < p[k + 1]:
            continue
        denom = p[k - 1] - 2.0 * p[k] + p[k + 1]
        shift = 0.5 * (p[k - 1] - p[k + 1]) / denom if abs(denom) > 0.0 else 0.0
        shift = float(np.clip(shift, -0.5, 0.5))
        df = f[k + 1] - f[k]
        peaks.append((float(f[k] + shift * df), float(p[k])))
    if not peaks:
        return []
    peaks.sort(key=lambda item: item[1], reverse=True)
    floor = float(np.median(p))
    selected = [pk for pk in peaks if pk[1] >= prominence_ratio * floor]
    if not selected:
        selected = peaks[:1]
    return selected[:n_peaks]


def harmonic_amplitudes(t: np.ndarray, x: np.ndarray, f0: float, n_harmonics: int = 4,
                        *, n_fit: int = 6) -> dict[str, Any]:
    """Least-squares amplitudes of the harmonics of ``f0`` in a time signal.

    Fits ``x(t) = a0 + sum_k [a_k cos(2 pi k f0 t) + b_k sin(2 pi k f0 t)]`` for
    ``k = 1 .. n_harmonics`` (plus ``n_fit`` nuisance harmonics above the band so
    that unmodelled content does not leak into the reported amplitudes) and
    returns the amplitudes ``A_k = sqrt(a_k^2 + b_k^2)``, the fraction of the
    detrended variance explained by the harmonic set, and the phase of each term.
    This is the time-domain counterpart of the PSD reading and is exact for a
    signal built from those harmonics (asserted in the tests).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if t.size != x.size or t.size < 32:
        raise ValueError("need at least 32 samples with matching t and x")
    y = detrend(t, x)
    columns = [np.ones_like(t)]
    for k in range(1, n_harmonics + n_fit + 1):
        columns.append(np.cos(2.0 * math.pi * k * f0 * t))
        columns.append(np.sin(2.0 * math.pi * k * f0 * t))
    design = np.stack(columns, axis=1)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    model = design @ coefficients
    residual = float(np.sum((y - model) ** 2))
    total = float(np.sum(y ** 2))
    amplitudes = {}
    phases = {}
    for k in range(1, n_harmonics + 1):
        a_k = float(coefficients[2 * k - 1])
        b_k = float(coefficients[2 * k])
        amplitudes[k] = math.hypot(a_k, b_k)
        phases[k] = math.atan2(-b_k, a_k)
    a1 = amplitudes.get(1, 0.0)
    return {
        "f0_Hz": float(f0),
        "amplitudes": amplitudes,
        "phases_rad": phases,
        "ratios_to_fundamental": {k: (v / a1 if a1 > 0.0 else float("inf"))
                                  for k, v in amplitudes.items()},
        "harmonic_fit_r2": 1.0 - residual / total if total > 0.0 else 1.0,
        "rms": float(np.sqrt(np.mean(y ** 2))),
    }


def blade_passing_frequency(rpm: float, n_blades: int) -> float:
    """``f_BPF = rpm/60 * Z`` [Hz]."""
    return float(rpm) / 60.0 * int(n_blades)


def harmonic_index(frequency: float, f_bpf: float, *, tol: float = 0.06) -> int:
    """Nearest integer harmonic ``k`` with ``|f - k f_BPF| <= tol k f_BPF``, else ``0``."""
    if f_bpf <= 0.0:
        raise ValueError("f_bpf must be positive")
    k = int(round(frequency / f_bpf))
    if k < 1:
        return 0
    return k if abs(frequency - k * f_bpf) <= tol * k * f_bpf else 0


@dataclass(frozen=True)
class SpectrumAnalysis:
    """Spectral acceptance report for one torque history."""

    f_peak: float
    f_bpf: float
    dominant_harmonic: int
    harmonics: tuple[tuple[int, float, float], ...]
    psd_floor: float
    pulse: dict[str, float]
    fundamental: float = 1.0

    @property
    def peak_matches_bpf(self) -> bool:
        """``True`` when the dominant peak is an exact multiple of the BPF."""
        return self.dominant_harmonic >= 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "f_peak_Hz": self.f_peak,
            "f_bpf_Hz": self.f_bpf,
            "f_peak_over_bpf": self.f_peak / self.f_bpf,
            "dominant_harmonic": self.dominant_harmonic,
            "peak_matches_bpf": self.peak_matches_bpf,
            "harmonics": [
                {"k": int(k), "f_Hz": float(f),
                 "amplitude_over_floor": float(amp / max(self.psd_floor, 1e-300)),
                 "amplitude_over_fundamental": float(amp / max(self.fundamental, 1e-300))}
                for k, f, amp in self.harmonics
            ],
            "psd_floor": self.psd_floor,
            **{k: float(v) for k, v in self.pulse.items()},
        }


def analyse_torque_spectrum(time: np.ndarray, torque: np.ndarray, *, rpm: float,
                            n_blades: int, window: np.ndarray | None = None,
                            n_peaks: int = 6,
                            tol: float = 0.06) -> SpectrumAnalysis:
    """Analyse the shaft-torque spectrum against the BPF and its harmonics."""
    t = np.asarray(time, dtype=float)
    q = np.asarray(torque, dtype=float)
    if window is not None:
        t = t[window]
        q = q[window]
    if t.size < 32:
        raise ValueError("need at least 32 samples in the analysis window")
    f_bpf = blade_passing_frequency(rpm, n_blades)
    freqs, psd = periodogram(t, q)
    floor = float(np.percentile(psd[1:], 50.0))
    if floor <= 0.0:
        floor = float(np.mean(psd[1:]))
    peaks = find_peaks_spectral(freqs, psd, n_peaks=n_peaks, min_freq=0.5 * f_bpf)
    if not peaks:
        raise ValueError("no spectral peak found above the prominence gate")
    f_dom, a_dom = peaks[0]
    harmonic = harmonic_index(f_dom, f_bpf, tol=tol)
    harmonics: list[tuple[int, float, float]] = []
    df = float(np.median(np.diff(freqs)))
    for k in (1, 2, 3, 4):
        target = k * f_bpf
        if target >= freqs[-1]:
            continue
        lo = int(np.searchsorted(freqs, target - max(df, 0.02 * target)))
        hi = int(np.searchsorted(freqs, target + max(df, 0.02 * target))) + 1
        hi = min(hi, psd.size)
        if hi <= lo:
            continue
        amp = float(np.max(psd[lo:hi]))
        harmonics.append((k, target, amp))
    fundamental = harmonics[0][2] if harmonics else 1.0
    q_mean = float(np.mean(q))
    q05, q95 = (float(v) for v in np.percentile(q, [5.0, 95.0]))
    rms = float(np.sqrt(np.mean((q - q_mean) ** 2)))
    pulse = {
        "Q_mean_Nm": q_mean,
        "delta_q_over_mean_p95": (q95 - q05) / q_mean if q_mean else float("nan"),
        "delta_q_over_mean_rms": rms / q_mean if q_mean else float("nan"),
        "dominant_peak_over_floor": a_dom / max(floor, 1e-30),
    }
    return SpectrumAnalysis(
        f_peak=float(f_dom),
        f_bpf=float(f_bpf),
        dominant_harmonic=int(harmonic),
        harmonics=tuple(harmonics),
        psd_floor=floor,
        pulse=pulse,
        fundamental=fundamental,
    )