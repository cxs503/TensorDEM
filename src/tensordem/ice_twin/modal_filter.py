"""Modal filtering: strain measurements -> modal coordinates.

Two complementary estimators are provided.

``extract`` -- the spatial least-squares inversion prescribed by the campaign
specification,

    min_q  || eps_meas(t) - H q(t) ||^2   for every sample t,

with ``H[i, n] = phi_n'(x_i)``.  It is exact for any (noise-free) modal
superposition and is the workhorse of the load-inversion chain.

``fit_mode`` -- a *synchronous* (matched) filter for the analytic anchor:
the least-squares coordinate of one mode is projected onto the two basis
functions ``sin(omega_n t)``, ``cos(omega_n t)`` over the whole window.  This
is the classical modal filter of vibration analysis; it averages the white
measurement noise over the window, so the recovered amplitude/phase of a
single-mode sinusoid converges as ``1/sqrt(T)`` and beats the per-sample
least-squares bound (which is fixed at ``10^(-SNR/20)`` by the Cramer-Rao
limit of a single frequency bin).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rom import FreeFreeBeamROM

__all__ = ["ModalFilter", "fit_sinusoid"]


def fit_sinusoid(signal, t, omega: float) -> tuple[float, float]:
    """Least-squares fit ``A sin(omega t + phase)``; returns (A, phase)."""
    s = np.asarray(signal, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    if s.size != t.size or s.size < 4:
        raise ValueError("signal and t must match and have >= 4 samples")
    A = np.column_stack([np.sin(omega * t), np.cos(omega * t)])
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    amp = math_hypot(coef[0], coef[1])
    phase = float(np.arctan2(coef[1], coef[0]))
    return float(amp), phase


def math_hypot(a: float, b: float) -> float:
    return float(np.hypot(a, b))


@dataclass
class ModalFilter:
    """Least-squares modal coordinate extraction from strain gauges."""

    rom: FreeFreeBeamROM
    x_gauges: np.ndarray

    def __post_init__(self):
        self.x_gauges = np.atleast_1d(np.asarray(self.x_gauges, dtype=float))
        H = self.rom.strain_matrix(self.x_gauges)
        if H.shape[0] < H.shape[1]:
            raise ValueError("need at least as many gauges as retained modes")
        self.H = H
        self.H_pinv = np.linalg.pinv(H)          # (n_modes, n_gauges)

    def extract(self, eps) -> np.ndarray:
        """q_hat(t) = H^+ eps(t), shape (n_modes, n_t)."""
        eps = np.atleast_2d(np.asarray(eps, dtype=float))
        if eps.shape[0] != self.H.shape[0]:
            raise ValueError("eps must have one row per gauge")
        return self.H_pinv @ eps

    def fit_mode(self, eps, t, n: int) -> np.ndarray:
        """Synchronous fit of mode n over the window (anchor path)."""
        q_ls = self.extract(eps)
        amp, phase = fit_sinusoid(q_ls[n, :], t, self.rom.omega[n])
        return amp * np.sin(self.rom.omega[n] * t + phase)
