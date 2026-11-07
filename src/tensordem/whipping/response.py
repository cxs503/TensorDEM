"""Modal superposition with piecewise-exact Duhamel integration.

Each retained mode is a SDOF oscillator with unit modal mass

    q'' + 2 zeta omega q' + omega^2 q = p(t),

solved *exactly* between samples for a load that is piecewise linear on the
sampling grid.  Within one step ``[t_n, t_n + h]`` with ``p(tau) = p0 + (p1-p0)
tau/h`` the particular solution is linear, so the homogeneous constants follow
from the state at ``t_n``:

    a = p0/omega^2 - 2 zeta (p1-p0) / (omega^3 h),   b = (p1-p0)/(omega^2 h),
    C = q0 - a,   D = (q0' - b + zeta omega C) / omega_d,
    q(h) = a + b h + e^{-zeta omega h}(C cos(omega_d h) + D sin(omega_d h)).

No numerical quadrature, no stability limit, and the energy ledger of
:func:`energy_ledger` is exact apart from the (convergent) quadrature of
``F(t) * v_load(t)`` and ``2 zeta omega q'^2``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.signal import butter, sosfiltfilt

__all__ = [
    "ModalSolution",
    "modal_response",
    "stress_history",
    "energy_ledger",
    "lowpass",
    "bandpass",
    "rms",
]


def _step_update(q0, v0, p0, p1, omega, zeta, h):
    """Exact state update for piecewise-linear modal load (arrays over modes)."""
    omega = np.asarray(omega, dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    zeta = np.minimum(zeta, 0.999999)
    wd = omega * np.sqrt(1.0 - zeta * zeta)
    dp = p1 - p0
    b = dp / (omega**2 * h)
    a = p0 / omega**2 - 2.0 * zeta * dp / (omega**3 * h)
    C = q0 - a
    D = (v0 - b + zeta * omega * C) / wd
    e = np.exp(-zeta * omega * h)
    c, s = np.cos(wd * h), np.sin(wd * h)
    q1 = a + b * h + e * (C * c + D * s)
    v1 = b + e * ((-zeta * omega * C + D * wd) * c + (-zeta * omega * D - C * wd) * s)
    return q1, v1


@dataclass(frozen=True)
class ModalSolution:
    """Modal coordinates and modal velocities on the sampling grid."""

    time: np.ndarray
    q: np.ndarray        # (n_modes, n_time)
    qdot: np.ndarray
    omega: np.ndarray    # rad/s
    zeta: np.ndarray
    modal_force: np.ndarray   # (n_modes, n_time)

    def modal_energy(self) -> dict:
        ke = 0.5 * np.sum(self.qdot[:, -1] ** 2)
        se = 0.5 * np.sum(self.omega ** 2 * self.q[:, -1] ** 2)
        return {"kinetic_J": float(ke), "strain_J": float(se)}


def modal_response(time, modal_force, omega, zeta) -> ModalSolution:
    """Integrate every mode exactly over a piecewise-linear modal force."""
    t = np.asarray(time, dtype=float)
    p = np.atleast_2d(np.asarray(modal_force, dtype=float))
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    zeta = np.broadcast_to(np.atleast_1d(np.asarray(zeta, dtype=float)), omega.shape).copy()
    if p.shape[1] != t.size:
        raise ValueError("modal_force must have one column per sample")
    if np.any(omega <= 0.0) or np.any(zeta < 0.0):
        raise ValueError("omega must be positive and zeta nonnegative")
    h = t[1] - t[0]
    if h <= 0.0 or not np.allclose(np.diff(t), h, rtol=1e-9, atol=1e-15):
        raise ValueError("time grid must be uniform")
    n_modes = omega.size
    q = np.zeros((n_modes, t.size))
    v = np.zeros((n_modes, t.size))
    for n in range(t.size - 1):
        q[:, n + 1], v[:, n + 1] = _step_update(q[:, n], v[:, n], p[:, n], p[:, n + 1],
                                                omega, zeta, h)
    return ModalSolution(time=t, q=q, qdot=v, omega=omega, zeta=zeta, modal_force=p)


def stress_history(solution: ModalSolution, curvature_weight) -> np.ndarray:
    """Deck stress ``sigma(t) = sum_i w_i q_i(t)`` at one longitudinal station.

    ``curvature_weight[i] = E * y_deck * d2phi_i/dx2(x_station)`` (Pa per unit
    modal coordinate), i.e. the classic bending-stress modal participation.
    """
    w = np.asarray(curvature_weight, dtype=float)
    if w.size != solution.q.shape[0]:
        raise ValueError("curvature_weight must have one entry per mode")
    return w @ solution.q


def lowpass(signal, dt: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass (no phase distortion of the record)."""
    fs = 1.0 / dt
    if not 0.0 < cutoff_hz < 0.5 * fs:
        raise ValueError("cutoff must lie strictly inside the Nyquist band")
    sos = butter(order, cutoff_hz / (0.5 * fs), btype="low", output="sos")
    return sosfiltfilt(sos, np.asarray(signal, dtype=float))


def bandpass(signal, dt: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass."""
    fs = 1.0 / dt
    if not 0.0 < low_hz < high_hz < 0.5 * fs:
        raise ValueError("band must lie strictly inside the Nyquist band")
    sos = butter(order, [low_hz / (0.5 * fs), high_hz / (0.5 * fs)],
                 btype="band", output="sos")
    return sosfiltfilt(sos, np.asarray(signal, dtype=float))


def rms(signal) -> float:
    x = np.asarray(signal, dtype=float)
    return float(math.sqrt(float(np.mean(x * x))))


def energy_ledger(solution: ModalSolution, force, load_shape, dt: float) -> dict:
    """Input work against kinetic + strain + damping energy of the modal system.

    ``force`` is the physical bow load ``F(t)`` and ``load_shape[i]`` the value
    of dry (wet) mode ``i`` at the load position, so that the modal force is
    ``p_i = load_shape[i] F``.  The input work is evaluated with the load-point
    velocity ``v_load = sum_i load_shape_i q_i'``, and the damping dissipation
    with a separate quadrature of ``2 zeta_i omega_i q_i'^2``.

    Returns the two energy ports, their relative residual and the work
    computed at a halved step (``work_refined``) as a quadrature-convergence
    indicator.
    """
    f = np.asarray(force, dtype=float)
    s = np.asarray(load_shape, dtype=float)
    if f.size != solution.q.shape[1]:
        raise ValueError("force must match the modal solution sampling")
    n = f.size
    if n % 2 == 0:      # composite Simpson needs an odd number of samples
        f = f[:-1]
        q_use = solution.q[:, :-1]
        qd_use = solution.qdot[:, :-1]
        n -= 1
    else:
        q_use = solution.q
        qd_use = solution.qdot
    v_load = s @ qd_use
    power = f * v_load
    work = float(np.trapezoid(power, dx=dt))
    work_simpson = _simpson_uniform(power, dt)
    diss = float(np.sum(2.0 * solution.zeta * solution.omega
                        * _simpson_uniform_axis(qd_use**2, dt)))
    diss_trapz = float(np.sum(2.0 * solution.zeta * solution.omega
                              * np.trapezoid(solution.qdot**2, dx=dt, axis=1)))
    ke = 0.5 * float(np.sum(qd_use[:, -1] ** 2))
    se = 0.5 * float(np.sum(solution.omega ** 2 * q_use[:, -1] ** 2))
    residual = abs(work_simpson - (ke + se + diss)) / max(abs(work_simpson), 1e-300)
    return {
        "work_input_J": work,
        "work_input_simpson_J": work_simpson,
        "damping_trapz_J": diss_trapz,
        "residual_trapz": abs(work - (ke + se + diss_trapz)) / max(abs(work), 1e-300),
        "kinetic_J": ke,
        "strain_J": se,
        "damping_J": diss,
        "residual": float(residual),
        "peak_energy_J": float(ke + se),
    }


def _simpson_uniform_axis(y: np.ndarray, dx: float) -> np.ndarray:
    n = y.shape[-1]
    if n % 2 == 0:
        return np.trapezoid(y, dx=dx, axis=-1)
    return dx / 3.0 * (y[..., 0] + y[..., -1] + 4.0 * y[..., 1:-1:2].sum(-1)
                       + 2.0 * y[..., 2:-2:2].sum(-1))


def _simpson_uniform(y: np.ndarray, dx: float) -> float:
    n = y.size
    if n % 2 == 0:
        return float(np.trapezoid(y, dx=dx))
    return float(dx / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-2:2].sum()))