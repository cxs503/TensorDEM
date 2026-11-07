"""Ice-load inversion from recovered modal coordinates.

Given the modal coordinates ``q_n(t)`` recovered by the modal filter, the modal
equation of motion

    q_n'' + 2 zeta_n omega_n q_n' + omega_n^2 q_n = p_n(t)

is solved *for the modal force*,

    p_n(t) = q_n'' + 2 zeta_n omega_n q_n' + omega_n^2 q_n ,

because the generalised mass is normalised to unity in :mod:`.model`.  The
physical load is then the minimum-residual projection of ``p(t)`` onto the
point-load pattern ``phi(x_F)``,

    F_ice(t) = sum_n phi_n(x_F) p_n(t) / sum_n phi_n(x_F)^2 ,

and the distributed estimate ``f_ice(x, t) = sum_n p_n(t) phi_n(x) /
sum_n phi_n(x_F)^2`` is available for spatial display.

Derivatives are the numerical weak point of any strain-based load inversion:
double differentiation of a noisy signal amplifies noise with the square of the
frequency.  A Savitzky-Golay differentiator is used because it is a local
polynomial fit, i.e. a differentiator *and* a low-pass with a flat passband, and
the passband is reported explicitly (``cutoff_hz``) so that the error metrics
below are always evaluated against the identically band-limited reference.  The
raw (unfiltered) reference errors are reported alongside.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

__all__ = [
    "LoadInversionResult",
    "smooth",
    "differentiate",
    "invert_modal_force",
    "load_error_metrics",
    "bandlimit",
    "savgol_cutoff_hz",
]


def savgol_cutoff_hz(window: int, dt: float) -> float:
    """Half-power-ish cut-off (``0.44 / (window/2)``) of a Savitzky-Golay fit.

    The conventional figure of merit for a least-squares polynomial of length
    ``window`` is that the filter attenuates above ``~0.44 / (window * dt / 2)``;
    stating it makes the bandwidth of the inversion explicit instead of implied.
    """
    if window < 3:
        raise ValueError("window must be at least 3 samples")
    return 0.44 / (0.5 * window * dt)


def smooth(signal, *, window: int = 51, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing (deriv=0) of the same local-polynomial family.

    Applied to the recovered modal coordinates before the stress is formed for
    fatigue counting, so that the stress record carries the same bandwidth as the
    load inversion instead of the full sensor noise band.
    """
    x = np.asarray(signal, dtype=float)
    if window % 2 == 0 or window < 5:
        raise ValueError("window must be odd and >= 5")
    if polyorder >= window:
        raise ValueError("polyorder must be smaller than window")
    if x.shape[-1] < window:
        raise ValueError("signal is shorter than the smoothing window")
    return savgol_filter(x, window_length=window, polyorder=polyorder,
                         deriv=0, axis=-1, mode="interp")


def differentiate(signal, dt: float, *, deriv: int = 1, window: int = 11,
                  polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay derivative along the last axis."""
    x = np.asarray(signal, dtype=float)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if deriv not in (1, 2):
        raise ValueError("only first and second derivatives are supported")
    if window % 2 == 0 or window < 5:
        raise ValueError("window must be odd and >= 5")
    if polyorder >= window:
        raise ValueError("polyorder must be smaller than window")
    if x.shape[-1] < window:
        raise ValueError("signal is shorter than the differentiation window")
    return savgol_filter(x, window_length=window, polyorder=polyorder,
                         deriv=deriv, delta=dt, axis=-1, mode="interp")


@dataclass(frozen=True)
class LoadInversionResult:
    """Modal and physical load estimates plus the derivative bandwidth."""

    time: np.ndarray            # (n_time,)
    q: np.ndarray               # (n_modes, n_time)
    qdot: np.ndarray
    qddot: np.ndarray
    modal_force: np.ndarray     # (n_modes, n_time)
    point_force: np.ndarray     # (n_time,)
    cutoff_hz: float
    window: int

    def load_field(self, rom, x) -> np.ndarray:
        """Distributed load estimate ``f(x, t)`` for the abscissae ``x``."""
        out = np.empty((np.size(x), self.point_force.size))
        for k in range(self.point_force.size):
            out[:, k] = rom.load_field(x, self.modal_force[:, k])
        return out

    def as_dict(self) -> dict:
        return {"cutoff_hz": self.cutoff_hz, "window": self.window,
                "n_samples": int(self.time.size),
                "peak_modal_force": [float(v) for v in np.max(np.abs(self.modal_force), axis=1)]}


def invert_modal_force(q, dt: float, rom, *, window: int = 11,
                       polyorder: int = 3) -> LoadInversionResult:
    """Invert modal force and point load from modal coordinates ``q``."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    if q.shape[0] != rom.n_modes:
        raise ValueError("q must have one row per mode")
    qdot = differentiate(q, dt, deriv=1, window=window, polyorder=polyorder)
    qddot = differentiate(q, dt, deriv=2, window=window, polyorder=polyorder)
    omega = rom.omega[:, None]
    zeta = rom.zeta[:, None]
    modal_force = qddot + 2.0 * zeta * omega * qdot + omega**2 * q
    point_force = rom.load_projection() @ modal_force
    n_time = q.shape[1]
    return LoadInversionResult(time=np.arange(n_time) * dt, q=q, qdot=qdot,
                               qddot=qddot, modal_force=modal_force,
                               point_force=point_force,
                               cutoff_hz=savgol_cutoff_hz(window, dt),
                               window=window)


def bandlimit(signal, dt: float, cutoff_hz: float, *, order: int = 4,
              axis: int = -1) -> np.ndarray:
    """Zero-phase Butterworth low-pass used to compare like with like."""
    from scipy.signal import butter, sosfiltfilt

    x = np.asarray(signal, dtype=float)
    fs = 1.0 / dt
    if not 0.0 < cutoff_hz < 0.5 * fs:
        raise ValueError("cutoff must lie strictly inside the Nyquist band")
    sos = butter(order, cutoff_hz / (0.5 * fs), btype="low", output="sos")
    return sosfiltfilt(sos, x, axis=axis)


def load_error_metrics(estimate, reference, dt: float, *, cut_edges: float = 0.0,
                       bandlimit_hz: float | None = None) -> dict:
    """Peak / energy / correlation error of an inverted load against a reference.

    ``cut_edges`` trims that fraction of the record at both ends (filter start-up).
    ``bandlimit_hz`` applies the same low-pass to both signals first, which is the
    fair comparison for an inversion whose differentiation bandwidth is finite.
    """
    est = np.asarray(estimate, dtype=float).ravel()
    ref = np.asarray(reference, dtype=float).ravel()
    if est.size != ref.size:
        raise ValueError("estimate and reference must have equal length")
    if est.size < 8:
        raise ValueError("record too short")
    if bandlimit_hz is not None:
        est = bandlimit(est, dt, bandlimit_hz)
        ref = bandlimit(ref, dt, bandlimit_hz)
    n = est.size
    lo = int(round(cut_edges * n))
    hi = n - lo
    est, ref = est[lo:hi], ref[lo:hi]
    peak_ref = float(np.max(np.abs(ref)))
    peak_est = float(np.max(np.abs(est)))
    if peak_ref <= 0.0:
        raise ValueError("reference load is identically zero")
    # fluctuating (AC) energy: the DC component of an ice-load record is the
    # quasi-static mean thrust and is compared separately as ``mean_bias_N``,
    # so that a small offset cannot masquerade as an energy error
    energy_ref = float(np.var(ref))
    energy_est = float(np.var(est))
    denom = float(np.sqrt(np.sum(ref**2) * np.sum(est**2)))
    corr = float(np.sum(est * ref) / denom) if denom > 0.0 else 0.0
    return {
        "peak_reference_N": peak_ref,
        "peak_estimate_N": peak_est,
        "peak_error": abs(peak_est - peak_ref) / peak_ref,
        "peak_bias": (peak_est - peak_ref) / peak_ref,
        "energy_reference_N2": energy_ref,
        "energy_estimate_N2": energy_est,
        "energy_error": abs(energy_est - energy_ref) / energy_ref,
        "energy_bias": (energy_est - energy_ref) / energy_ref,
        "rms_error": float(np.sqrt(np.mean((est - ref) ** 2)) / np.sqrt(energy_ref)),
        "mean_bias_N": float(est.mean() - ref.mean()),
        "mean_bias": float((est.mean() - ref.mean()) / np.sqrt(energy_ref)),
        "correlation": corr,
        "n_samples": int(est.size),
        "bandlimit_hz": bandlimit_hz,
        "edge_trim_fraction": cut_edges,
    }