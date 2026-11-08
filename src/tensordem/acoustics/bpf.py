"""Blade-passing-frequency (BPF) source model and spectral peak picking.

A rotor with ``Z`` blades turning at ``N`` rev/min produces a load pulse train
whose fundamental is the blade-passing frequency

.. math::  f_\\mathrm{BPF} = \\frac{N}{60}\\, Z  \\;[\\mathrm{Hz}],

with harmonics ``k f_BPF``.  Blake's propeller-noise hierarchy uses exactly this
tonal ladder: thickness/monsoon noise (volume displacement, monopole), the
steady/unsteady load dipole at the BPF and the broadband trailing-edge
quadrupole.  For a face sheet wrapped around the shaft the pulse arrives at
azimuth ``psi`` with the sweep phase ``Z psi``, so the load history of a face is

.. math::

    F(\\psi,t) = F_0 \\Bigl[c_0 + \\sum_{k=1}^{K} c_k
        \\cos\\bigl(2\\pi k f_\\mathrm{BPF}\\, t + k Z \\psi\\bigr)\\Bigr],

which is what :func:`rotor_face_forces` builds.  The acceptance test is that the
face-level FFT of that history peaks at ``f_BPF`` (criterion (c), < 1 %).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "shaft_frequency_hz",
    "blade_passing_frequency_hz",
    "default_harmonic_ratios",
    "rotor_face_forces",
    "rotor_face_turbulence",
    "find_spectral_peak",
    "third_octave_levels",
    "ToneDetection",
]


def shaft_frequency_hz(rpm: float) -> float:
    """Shaft rotation frequency ``N/60`` [Hz]."""
    rpm = float(rpm)
    if not math.isfinite(rpm) or rpm <= 0.0:
        raise ValueError("rpm must be positive")
    return rpm / 60.0


def blade_passing_frequency_hz(rpm: float, blades: int) -> float:
    """Blade-passing frequency ``f_BPF = N Z / 60`` [Hz]."""
    if int(blades) != blades or int(blades) < 1:
        raise ValueError("blades must be a positive integer")
    return shaft_frequency_hz(rpm) * int(blades)


def default_harmonic_ratios(n_harmonics: int = 3) -> np.ndarray:
    """Decaying BPF harmonic ladder (1, 1/2, 1/3, ...) including the mean ``c_0``.

    Returns ``(c_0, c_1, ..., c_K)`` normalised so that the peak is 1.
    """
    if n_harmonics < 1:
        raise ValueError("need at least one harmonic")
    ratios = np.array([0.5] + [1.0 / k for k in range(1, n_harmonics + 1)])
    return ratios / ratios.max()


def rotor_face_forces(
    t: Sequence[float],
    azimuth: Any,
    *,
    rpm: float,
    blades: int,
    force_peak_n: float,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    harmonic_ratios: Any = None,
    blade_scatter: Any = None,
) -> np.ndarray:
    """Per-face force histories ``(n_faces, 3, n_samples)`` [N] of a BPF pulse train.

    ``azimuth`` gives the circumferential position of each face; the sweep phase
    is ``k Z psi``.  All faces share the same waveform shape, so the array is the
    outer product of the (3,) load direction with the ``(n_faces, n_samples)``
    scalar histories.
    """
    tt = np.asarray(t, dtype=float).ravel()
    if tt.size < 8:
        raise ValueError("need at least 8 time samples")
    psi = np.asarray(azimuth, dtype=float).ravel()
    if psi.size == 0:
        raise ValueError("azimuth must not be empty")
    if force_peak_n <= 0.0:
        raise ValueError("force_peak_n must be positive")
    a = np.asarray(axis, dtype=float).ravel()
    if a.size != 3 or not np.any(a):
        raise ValueError("axis must be a non-zero 3-vector")
    a = a / np.linalg.norm(a)
    ratios = (default_harmonic_ratios(3) if harmonic_ratios is None
              else np.asarray(harmonic_ratios, dtype=float).ravel())
    if ratios.size < 2:
        raise ValueError("harmonic_ratios must contain c0 and at least one harmonic")
    f_bpf = blade_passing_frequency_hz(rpm, blades)
    omega = 2.0 * math.pi * f_bpf
    z = int(blades)
    shape = (psi.size, tt.size)
    scalar = np.full(shape, ratios[0], dtype=float)
    for k in range(1, ratios.size):
        scalar += ratios[k] * np.cos(k * omega * tt[None, :] + k * z * psi[:, None])
    if blade_scatter is not None:
        scat = np.asarray(blade_scatter, dtype=float).ravel()
        if scat.size != psi.size:
            raise ValueError("blade_scatter must have one entry per face")
        if np.any(scat <= 0.0):
            raise ValueError("blade_scatter factors must be positive")
        scalar = ratios[0] + (scalar - ratios[0]) * scat[:, None]
    scalar *= force_peak_n / np.max(np.abs(ratios))
    return scalar[:, None, :] * a[None, :, None]


def rotor_face_turbulence(
    t: Sequence[float],
    azimuth: Any,
    *,
    rpm: float,
    blades: int,
    u_rms: float,
    u_mean: Sequence[float] = (0.0, 0.0, 0.0),
    harmonic_ratios: Any = None,
    rng: np.random.Generator | None = None,
    broadband_fraction: float = 0.05,
    blade_scatter: Any = None,
) -> np.ndarray:
    """Per-face velocity histories ``(n_faces, 3, n_samples)`` [m/s] in the wake.

    The tonal part is the BPF pulse train (blade-to-blade unsteadiness, which is
    what a Lighthill quadrupole of the wake sees); a small seeded broadband part
    keeps the spectrum from being a pure line so the peak finder is exercised on
    a realistic floor.
    """
    if u_rms <= 0.0:
        raise ValueError("u_rms must be positive")
    tt = np.asarray(t, dtype=float).ravel()
    psi = np.asarray(azimuth, dtype=float).ravel()
    ratios = (default_harmonic_ratios(3) if harmonic_ratios is None
              else np.asarray(harmonic_ratios, dtype=float).ravel())
    f_bpf = blade_passing_frequency_hz(rpm, blades)
    omega = 2.0 * math.pi * f_bpf
    z = int(blades)
    mean = np.asarray(u_mean, dtype=float).ravel()
    if mean.size != 3:
        raise ValueError("u_mean must be a 3-vector")
    base = np.zeros((psi.size, tt.size), dtype=float)
    for k in range(1, ratios.size):
        base += ratios[k] * np.cos(k * omega * tt[None, :] + k * z * psi[:, None])
    base *= u_rms / max(float(np.max(np.abs(ratios[1:]))), 1.0e-12)
    if blade_scatter is not None:
        scat = np.asarray(blade_scatter, dtype=float).ravel()
        if scat.size != psi.size:
            raise ValueError("blade_scatter must have one entry per face")
        if np.any(scat <= 0.0):
            raise ValueError("blade_scatter factors must be positive")
        base = base * scat[:, None]
    if broadband_fraction > 0.0:
        gen = rng if rng is not None else np.random.default_rng(0)
        base = base + broadband_fraction * u_rms * gen.standard_normal(base.shape)
    out = np.empty((psi.size, 3, tt.size), dtype=float)
    # axial mean flow plus the transverse tonal flutter (rotating blade wake)
    out[:, 0] = mean[0] + base
    out[:, 1] = mean[1] + 0.3 * base
    out[:, 2] = mean[2] + 0.15 * base
    return out


# --------------------------------------------------------------------------- #
# peak picking
# --------------------------------------------------------------------------- #
@dataclass
class ToneDetection:
    """Result of a spectral peak search around an expected tone."""

    f_expected: float
    f_peak: float
    relative_error: float
    peak_amplitude: float
    prominence_ratio: float
    bin_width: float
    within_tolerance: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "f_expected_hz": self.f_expected,
            "f_peak_hz": self.f_peak,
            "relative_error": self.relative_error,
            "peak_amplitude": self.peak_amplitude,
            "prominence_ratio": self.prominence_ratio,
            "bin_width_hz": self.bin_width,
            "within_tolerance": self.within_tolerance,
        }


def find_spectral_peak(
    freqs: Sequence[float],
    amplitude: Sequence[float],
    *,
    f_expected: float,
    tolerance: float = 0.08,
    floor_quantile: float = 0.5,
) -> ToneDetection:
    """Locate the tone near ``f_expected`` with parabolic (sub-bin) refinement.

    The search window is ``f_expected (1 +- tolerance)``; the returned
    ``relative_error`` is ``|f_peak - f_expected| / f_expected`` and
    ``prominence_ratio`` is the peak over the median amplitude of the window
    (a negative control on broadband-only input gives a ratio near 1).
    """
    f = np.asarray(freqs, dtype=float).ravel()
    a = np.asarray(amplitude, dtype=float).ravel()
    if f.size != a.size or f.size < 8:
        raise ValueError("freqs and amplitude must match and hold >= 8 bins")
    if f_expected <= 0.0:
        raise ValueError("f_expected must be positive")
    df = float(np.median(np.diff(f)))
    lo = f_expected * (1.0 - tolerance)
    hi = f_expected * (1.0 + tolerance)
    sel = np.where((f >= lo) & (f <= hi))[0]
    if sel.size < 3:
        raise ValueError("search window holds fewer than 3 spectral bins")
    k = int(sel[np.argmax(a[sel])])
    amp = float(a[k])
    f_peak = float(f[k])
    if 0 < k < a.size - 1:
        y0, y1, y2 = a[k - 1], a[k], a[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom < 0.0:
            delta = 0.5 * (y0 - y2) / denom
            delta = float(np.clip(delta, -0.5, 0.5))
            f_peak = float(f[k] + delta * df)
    floor = float(np.quantile(a[sel], floor_quantile))
    prominence = amp / floor if floor > 0.0 else float("inf")
    return ToneDetection(
        f_expected=float(f_expected),
        f_peak=f_peak,
        relative_error=abs(f_peak - f_expected) / f_expected,
        peak_amplitude=amp,
        prominence_ratio=prominence,
        bin_width=df,
        within_tolerance=abs(f_peak - f_expected) / f_expected < 0.01,
    )


def third_octave_levels(
    freqs: Sequence[float],
    amplitude: Sequence[float],
    *,
    f_min: float = 5.0,
    f_max: float = 2000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Band levels of the peak-amplitude spectrum on IEC third-octave centres.

    Returns ``(centre_freqs, band_amplitudes)`` with the band amplitude the
    energy sum ``sqrt(sum |A_k|^2 / 2)`` of its bins (RMS of band).
    """
    f = np.asarray(freqs, dtype=float).ravel()
    a = np.asarray(amplitude, dtype=float).ravel()
    if f.size != a.size or f.size < 4:
        raise ValueError("freqs and amplitude must match and hold >= 4 bins")
    if f_min <= 0.0 or f_max <= f_min:
        raise ValueError("need 0 < f_min < f_max")
    # IEC 61260 1/3-octave centres fc = 1000 * 10^(k/10), restricted to the band
    k_lo = int(math.ceil(10.0 * math.log10(f_min / 1000.0)))
    k_hi = int(math.floor(10.0 * math.log10(f_max / 1000.0)))
    centres_arr = np.asarray([1000.0 * 10.0 ** (k / 10.0)
                              for k in range(k_lo, k_hi + 1)], dtype=float)
    levels = np.zeros_like(centres_arr)
    for i, fc in enumerate(centres_arr):
        lo, hi = fc * 2.0 ** (-1.0 / 6.0), fc * 2.0 ** (1.0 / 6.0)
        mask = (f >= lo) & (f < hi)
        levels[i] = math.sqrt(0.5 * float(np.sum(a[mask] ** 2))) if np.any(mask) else 0.0
    return centres_arr, levels