"""O(N) face-level FFT machinery for the acoustic source terms.

The far field of a face-distributed compact source is a *sum over faces of
retarded, spectrally filtered source histories*:

.. math::

    p_a(\\mathbf{x}, f) = \\sum_{f} a_f(\\mathbf{n}) S_f(f) e^{-i \\omega r_f/c_0},

and in the time domain the same quantity needs an explicit retarded-time loop
``sum_f h_f(t - r_f/c0)`` for every observer sample.  Doing that inner product in
the time domain costs ``O(N_t N_faces)`` and, for the classical discrete
Fourier transform, ``O(N^2)`` per face; the FFT replaces the transform with
``O(N log N)`` and the retardation becomes a single complex phase multiply in the
frequency domain.  This module ships

``RingBuffer``
    fixed-capacity per-face history buffer (circular write, ordered read), i.e.
    the streaming front end that feeds the FFT without reallocating;
``face_spectrum``
    one-sided peak-amplitude FFT of a ``(n_faces, n_samples)`` block in a single
    vectorised call -- the O(N log N) path;
``direct_dft``
    the O(N^2) reference transform (same convention) used for the equivalence and
    timing comparison;
``quadrupole_field_time`` / ``dipole_field_time``
    explicit retarded-time reference for the propagation step;
``benchmark_fft_vs_direct``
    acceptance numbers of criterion (d): relative error and wall-clock ratio.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .source_terms import Medium

__all__ = [
    "RingBuffer",
    "face_spectrum",
    "face_spectrum_complex",
    "peak_to_fft_coefficients",
    "direct_dft",
    "quadrupole_field_time",
    "dipole_field_time",
    "benchmark_fft_vs_direct",
    "propagation_equivalence",
    "hann_window",
]


# --------------------------------------------------------------------------- #
# ring buffer (streaming front end)
# --------------------------------------------------------------------------- #
class RingBuffer:
    """Fixed-capacity circular buffer of per-face histories.

    ``capacity`` samples are kept; pushing more overwrites the oldest ones.  The
    stored array is ``(n_faces, capacity)`` (or ``(n_faces, n_components,
    capacity)``) so that a whole face block is one contiguous row and the FFT can
    be taken along the last axis in one call.
    """

    def __init__(self, n_faces: int, capacity: int, n_components: int = 1) -> None:
        if n_faces <= 0 or capacity <= 0 or n_components <= 0:
            raise ValueError("n_faces, capacity and n_components must be positive")
        self.n_faces = int(n_faces)
        self.capacity = int(capacity)
        self.n_components = int(n_components)
        shape = ((self.n_faces, self.capacity) if self.n_components == 1
                 else (self.n_faces, self.n_components, self.capacity))
        self._data = np.zeros(shape, dtype=float)
        self._head = 0          # next write position
        self._count = 0         # valid samples, <= capacity

    # ------------------------------------------------------------------ core
    def push(self, block: Any) -> None:
        """Append ``m`` samples; ``block`` is ``(n_faces, m)`` (or with components)."""
        arr = np.asarray(block, dtype=float)
        expected_head = (self.n_faces, self.n_components) if self.n_components > 1 \
            else (self.n_faces,)
        if arr.ndim != len(expected_head) + 1 or arr.shape[:-1] != expected_head:
            raise ValueError(f"block must have leading shape {expected_head}")
        m = arr.shape[-1]
        if m > self.capacity:
            arr = arr[..., -self.capacity:]
            m = self.capacity
            self._data[...] = 0.0
            self._head = 0
            self._count = 0
        first = min(m, self.capacity - self._head)
        self._data[..., self._head:self._head + first] = arr[..., :first]
        rest = m - first
        if rest > 0:
            self._data[..., :rest] = arr[..., first:]
        self._head = (self._head + m) % self.capacity
        self._count = min(self.capacity, self._count + m)

    # ------------------------------------------------------------------ read
    @property
    def count(self) -> int:
        return self._count

    def snapshot(self, n: int | None = None) -> np.ndarray:
        """Ordered copy of the newest ``n`` samples (oldest first)."""
        take = self._count if n is None else int(n)
        if take <= 0 or take > self._count:
            raise ValueError("requested sample count not available in the buffer")
        ordered = np.concatenate([self._data[..., self._head:],
                                  self._data[..., :self._head]], axis=-1)
        return ordered[..., -take:].copy()

    def reset(self) -> None:
        self._data[...] = 0.0
        self._head = 0
        self._count = 0


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
def hann_window(n: int) -> np.ndarray:
    """Periodic Hann taper (amplitude-loss corrected by the caller)."""
    if n < 2:
        return np.ones(max(n, 1))
    return 0.5 - 0.5 * np.cos(2.0 * math.pi * np.arange(n) / n)


def face_spectrum_complex(x: Any, dt: float, *, taper: bool = True
                          ) -> tuple[np.ndarray, np.ndarray]:
    """One-sided **complex peak-amplitude** spectrum of every face (one ``rfft``).

    ``x`` is ``(n_faces, n_samples)``; a harmonic ``A cos(2 pi f t + phi)`` on an FFT
    bin yields ``X_k = A exp(i phi)``, i.e. the *source phase between faces is
    preserved* -- essential for a rotating load pattern, whose azimuthal phase
    ``k Z psi`` is what makes it an azimuthal multipole.  Cost
    ``O(n_faces * N log N)`` with no Python loop over faces or samples.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("x must be (n_faces, n_samples)")
    if not dt > 0.0:
        raise ValueError("dt must be positive")
    n = arr.shape[1]
    if n < 8:
        raise ValueError("need at least 8 samples")
    window = hann_window(n) if taper else np.ones(n)
    scale = 2.0 / float(np.sum(window))       # peak amplitude for interior bins
    spec = np.fft.rfft((arr - arr.mean(axis=1, keepdims=True)) * window, axis=1)
    spec = spec * scale
    spec[:, 0] *= 0.5
    if n % 2 == 0:
        spec[:, -1] *= 0.5
    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs, spec


def face_spectrum(x: Any, dt: float, *, taper: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude of :func:`face_spectrum_complex` (peak-amplitude convention)."""
    freqs, spec = face_spectrum_complex(x, dt, taper=taper)
    return freqs, np.abs(spec)


def peak_to_fft_coefficients(spec: Any, n_samples: int) -> np.ndarray:
    """Convert a one-sided **peak-amplitude** spectrum to numpy ``rfft`` coefficients.

    :func:`face_spectrum_complex` stores peak amplitudes (DC and Nyquist halved);
    ``np.fft.irfft`` expects the raw transform, i.e. ``N/2`` times the interior
    amplitudes and ``N`` times DC / Nyquist.
    """
    S = np.asarray(spec, dtype=complex)
    out = S * (0.5 * n_samples)
    out[..., 0] = S[..., 0] * n_samples
    if n_samples % 2 == 0:
        out[..., -1] = S[..., -1] * n_samples
    return out


def direct_dft(x: Any, dt: float, freqs: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """O(N^2) reference transform with the same amplitude convention.

    ``X_k = 2/N |sum_m x_m exp(-2 pi i f_k t_m)|`` for ``k >= 1`` (0.5 weight at
    DC/Nyquist), matching :func:`face_spectrum` with ``taper=False``.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("x must be (n_faces, n_samples)")
    n = arr.shape[1]
    t = np.arange(n) * float(dt)
    f = np.fft.rfftfreq(n, d=dt) if freqs is None else np.asarray(freqs, dtype=float)
    kernel = np.exp(-2j * math.pi * np.outer(f, t))   # (n_freq, n_samples)
    centred = arr - arr.mean(axis=1, keepdims=True)
    spec = centred @ kernel.T                          # (n_faces, n_freq)
    amp = 2.0 * np.abs(spec) / n
    amp[:, 0] *= 0.5
    if n % 2 == 0:
        amp[:, -1] *= 0.5
    return f, amp


# --------------------------------------------------------------------------- #
# explicit retarded-time propagation (reference for the spectral path)
# --------------------------------------------------------------------------- #
def _retarded(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    """``signal(t - delay)`` on a uniform grid, linear interpolation."""
    n = signal.shape[0]
    idx = np.arange(n, dtype=float) - delay_samples
    i0 = np.floor(idx).astype(int)
    w = idx - i0
    valid = (i0 >= 0) & (i0 <= n - 2)
    out = np.zeros(n, dtype=signal.dtype)
    i0c = np.clip(i0, 0, n - 2)
    val = (1.0 - w) * signal[i0c] + w * signal[i0c + 1]
    out[valid] = val[valid]
    return out


def _d_dt(signal: np.ndarray, dt: float, order: int) -> np.ndarray:
    """Central-difference time derivative (``order`` 1 or 2), *not* wrapping."""
    y = np.asarray(signal, dtype=float)
    for _ in range(order):
        y = np.gradient(y, dt)
    return y


def quadrupole_field_time(
    q_history: Any,
    positions: Any,
    direction: Any,
    distance: float,
    dt: float,
    medium: Medium,
) -> np.ndarray:
    """Explicit retarded-time quadrupole far field, ``O(N_t * n_faces)``.

    ``q_history`` is ``(n_faces, 3, 3, N)`` (peak-scaled histories), ``positions``
    ``(n_faces, 3)``, ``direction`` a unit 3-vector.  Reference path for
    :func:`tensordem.acoustics.source_terms.quadrupole_far_field_spectrum`.
    """
    Q = np.asarray(q_history, dtype=float)
    pos = np.asarray(positions, dtype=float)
    n = np.asarray(direction, dtype=float)
    if Q.ndim != 4 or Q.shape[1:3] != (3, 3):
        raise ValueError("q_history must be (n_faces, 3, 3, N)")
    if pos.shape != (Q.shape[0], 3):
        raise ValueError("positions must be (n_faces, 3)")
    if distance <= 0.0:
        raise ValueError("distance must be positive")
    n = n / np.linalg.norm(n)
    n_faces, _, _, n_samp = Q.shape
    # far-field retarded distance of each face: r - n.x_f (same model as the
    # spectral path, otherwise the two paths differ by the array phase)
    delays = (distance - pos @ n) / (medium.c0 * dt)
    proj = np.einsum("i,j,fijt->ft", n, n, Q)        # (n_faces, N)
    out = np.zeros(n_samp, dtype=float)
    pref = 1.0 / (4.0 * math.pi * medium.c0 ** 2 * distance)
    for f in range(n_faces):
        second = _d_dt(proj[f], dt, 2)
        out += _retarded(second, float(delays[f]))
    return pref * out


def dipole_field_time(
    f_history: Any,
    positions: Any,
    direction: Any,
    distance: float,
    dt: float,
    medium: Medium,
) -> np.ndarray:
    """Explicit retarded-time Curle dipole far field ``O(N_t * n_faces)``."""
    F = np.asarray(f_history, dtype=float)
    pos = np.asarray(positions, dtype=float)
    n = np.asarray(direction, dtype=float)
    if F.ndim != 3 or F.shape[1] != 3:
        raise ValueError("f_history must be (n_faces, 3, N)")
    if pos.shape != (F.shape[0], 3):
        raise ValueError("positions must be (n_faces, 3)")
    if distance <= 0.0:
        raise ValueError("distance must be positive")
    n = n / np.linalg.norm(n)
    n_faces, _, n_samp = F.shape
    delays = (distance - pos @ n) / (medium.c0 * dt)
    proj = np.einsum("i,fit->ft", n, F)              # (n_faces, N)
    out = np.zeros(n_samp, dtype=float)
    pref = 1.0 / (4.0 * math.pi * medium.c0 * distance)
    for f in range(n_faces):
        first = _d_dt(proj[f], dt, 1)
        out += _retarded(first, float(delays[f]))
    return pref * out


# --------------------------------------------------------------------------- #
# acceptance helpers
# --------------------------------------------------------------------------- #
def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """RMS relative error, normalised by the reference RMS."""
    ref = np.sqrt(np.mean(np.abs(b) ** 2))
    if ref <= 0.0:
        raise ValueError("reference is identically zero")
    return float(np.sqrt(np.mean(np.abs(a - b) ** 2)) / ref)


@dataclass
class FFTBenchmark:
    """Numbers behind acceptance criterion (d)."""

    n_faces: int
    n_samples: int
    max_rel_error: float = 0.0
    rms_rel_error: float = 0.0
    fft_seconds: float = 0.0
    direct_seconds: float = 0.0
    repeats: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def speedup(self) -> float:
        if self.fft_seconds <= 0.0 or self.direct_seconds <= 0.0:
            return float("nan")
        return self.direct_seconds / self.fft_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_faces": self.n_faces,
            "n_samples": self.n_samples,
            "max_rel_error": self.max_rel_error,
            "rms_rel_error": self.rms_rel_error,
            "fft_seconds": self.fft_seconds,
            "direct_seconds": self.direct_seconds,
            "speedup": self.speedup,
            "repeats": self.repeats,
            **self.extra,
        }


def benchmark_fft_vs_direct(
    x: Any,
    dt: float,
    *,
    repeats: int = 3,
    max_samples: int = 4096,
) -> FFTBenchmark:
    """Compare :func:`face_spectrum` against :func:`direct_dft` in error and time."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("x must be (n_faces, n_samples)")
    arr = arr[:, :max_samples]
    n_faces, n_samples = arr.shape
    if arr.shape[1] < 16:
        raise ValueError("need at least 16 samples")

    f_fft, a_fft = face_spectrum(arr, dt, taper=False)
    t0 = time.perf_counter()
    for _ in range(repeats):
        f_fft, a_fft = face_spectrum(arr, dt, taper=False)
    fft_seconds = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        f_dir, a_dir = direct_dft(arr, dt, freqs=f_fft)
    direct_seconds = (time.perf_counter() - t0) / repeats

    peak = float(a_dir.max()) if a_dir.size else 0.0
    active = a_dir.max(axis=0) > 1.0e-9 * max(peak, 1.0e-300)
    if not np.any(active):
        raise ValueError("benchmark block carries no spectral content")
    denom = np.sqrt(np.mean(a_dir[:, active] ** 2))
    max_rel = float(np.max(np.abs(a_fft - a_dir)) / denom)
    rms_rel = _relative_error(a_fft, a_dir)
    return FFTBenchmark(
        n_faces=n_faces,
        n_samples=n_samples,
        max_rel_error=max_rel,
        rms_rel_error=rms_rel,
        fft_seconds=fft_seconds,
        direct_seconds=direct_seconds,
        repeats=repeats,
        extra={"n_freq_bins": int(f_fft.size),
               "active_bins": int(active.sum())},
    )


def propagation_equivalence(
    kind: str,
    history: Any,
    positions: Any,
    direction: Any,
    distance: float,
    dt: float,
    medium: Medium,
    *,
    n_grow: int = 0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Spectral (O(N log N)) vs explicit retarded-time propagation.

    Returns ``(rms_relative_error, p_spectral_time, p_reference_time)`` where the
    spectral path is transformed back to the time domain for the comparison.
    """
    from .source_terms import dipole_far_field_spectrum, quadrupole_far_field_spectrum

    n = history.shape[-1]
    freqs = np.fft.rfftfreq(n, d=dt)
    spec = np.fft.rfft(history, axis=-1) / n * 2.0   # peak amplitude
    spec[..., 0] *= 0.5
    if n % 2 == 0:
        spec[..., -1] *= 0.5
    spec = np.moveaxis(spec, -1, 0)                  # (n_freq, n_faces, ...)
    dirs = np.asarray(direction, dtype=float)[None, :]
    if kind == "quadrupole":
        p_spec = quadrupole_far_field_spectrum(
            spec, freqs, dirs, distance=distance, medium=medium, positions=positions)[:, 0]
        p_time_ref = quadrupole_field_time(history, positions, dirs[0], distance, dt, medium)
    elif kind == "dipole":
        p_spec = dipole_far_field_spectrum(
            spec, freqs, dirs, distance=distance, medium=medium, positions=positions)[:, 0]
        p_time_ref = dipole_field_time(history, positions, dirs[0], distance, dt, medium)
    else:
        raise ValueError("kind must be 'quadrupole' or 'dipole'")
    p_time_spec = np.fft.irfft(peak_to_fft_coefficients(p_spec, n), n=n)
    if n_grow:
        p_time_spec = p_time_spec[n_grow:]
        p_time_ref = p_time_ref[n_grow:]
        n = p_time_spec.size
    return _relative_error(p_time_spec, p_time_ref), p_time_spec, p_time_ref