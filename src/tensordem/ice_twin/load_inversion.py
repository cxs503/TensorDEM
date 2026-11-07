"""Ice-load inversion from modal coordinates.

Given the (filtered) modal coordinates ``q(t)`` the modal forces follow from
the reduced equations of motion,

    F_n(t) = m_n q_n'' + c_n q_n' + k_n q_n,

and the total ice force under the contact patch is recovered by the
Galerkin projection onto the patch participation vector,

    f_hat(t) = sum_n gamma_n F_n(t) / sum_n gamma_n^2.

For an exact modal superposition this is the identity (the completeness of
the mass-normalised shapes makes ``sum_n gamma_n^2`` the squared norm of the
patch in the modal basis), so the only errors are modal truncation,
differentiation of the measured coordinates, and measurement noise.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .rom import FreeFreeBeamROM

__all__ = [
    "smooth",
    "differentiate",
    "invert_modal_force",
    "reconstruct_total_force",
    "peak_error",
    "energy_error",
]


def smooth(signal, dt: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass (guards against short records)."""
    s = np.asarray(signal, dtype=float)
    fs = 1.0 / dt
    if not 0.0 < cutoff_hz < 0.5 * fs:
        return s.copy()
    if s.size < 4 * (2 * order + 1):
        return s.copy()
    sos = butter(order, cutoff_hz / (0.5 * fs), btype="low", output="sos")
    return sosfiltfilt(sos, s)


def differentiate(signal, dt: float, cutoff_hz: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference (q', q'') with optional pre-smoothing.

    Returns ``(qdot, qddot)``.  Pre-smoothing at ``cutoff_hz`` removes the
    high-frequency noise that differentiation would otherwise amplify.
    """
    s = np.asarray(signal, dtype=float)
    if cutoff_hz is not None:
        s = smooth(s, dt, cutoff_hz)
    qdot = np.gradient(s, dt, axis=-1, edge_order=2)
    qddot = np.gradient(qdot, dt, axis=-1, edge_order=2)
    return qdot, qddot


def invert_modal_force(rom: FreeFreeBeamROM, q, qdot, qddot) -> np.ndarray:
    """F_n(t) = m_n q_n'' + c_n q_n' + k_n q_n, shape (n_modes, n_t)."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    n = q.shape[0]
    m = np.ones(n)
    c = np.diag(rom.C)
    k = np.diag(rom.K)
    return m[:, None] * qddot + c[:, None] * qdot + k[:, None] * q


def reconstruct_total_force(rom: FreeFreeBeamROM, F_modal,
                           x0: float, width: float) -> np.ndarray:
    """f_hat(t) = (gamma . F_modal) / ||gamma||^2 (Galerkin projection)."""
    F = np.atleast_2d(np.asarray(F_modal, dtype=float))
    g = rom.patch_participation(x0, width)
    return (g @ F) / float(g @ g)


def peak_error(f_hat, f) -> float:
    """max |f_hat - f| / max |f|  (0 for a zero reference)."""
    f = np.asarray(f, dtype=float)
    f_hat = np.asarray(f_hat, dtype=float)
    scale = float(np.abs(f).max())
    if scale == 0.0:
        return float(np.abs(f_hat).max()) if f_hat.size else 0.0
    return float(np.abs(f_hat - f).max() / scale)


def energy_error(f_hat, f) -> float:
    """|E(f_hat) - E(f)| / E(f) with E = int x^2 dt (0 for a zero reference)."""
    f = np.asarray(f, dtype=float)
    f_hat = np.asarray(f_hat, dtype=float)
    e = float(np.trapezoid(f * f, dx=1.0))
    if e == 0.0:
        return float(np.trapezoid(f_hat * f_hat, dx=1.0)) if f_hat.size else 0.0
    return abs(float(np.trapezoid(f_hat * f_hat, dx=1.0)) - e) / e
