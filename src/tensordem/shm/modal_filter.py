"""Modal filtering: least-squares inversion of measured strain to modal coordinates.

The sensor model is linear and memoryless,

    eps(x_k, t) = sum_n H[k, n] q_n(t) + noise_k(t),

so the modal coordinates are recovered from a (possibly regularised) least
squares problem

    q_hat = argmin_q || eps - H q ||^2 + lambda || q ||^2 .

Because the modal mass is normalised to unity, ``q`` is in metres (per unit mode
shape) and the residual has strain units.  The class also reports the *noise
gain* ``||A||_F`` of the estimator (``A = (H^T H + lambda I)^-1 H^T``), which is
the factor by which sensor noise is amplified into the modal coordinates; it is
the quantity that sets the usable signal-to-noise ratio of the inversion and is
measured in the acceptance tests.

Ill-conditioned sensor sets are rejected instead of silently amplified: with the
default Tikhonov-free estimator the condition number of ``H`` must stay below
``max_condition`` (a rank-deficient layout with, e.g., a single gauge per mode is
physically unsolvable, not a numerical detail).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ModalFilter", "ModalFilterResult", "harmonic_modal_estimate"]


@dataclass(frozen=True)
class ModalFilterResult:
    """Recovered modal coordinates and inversion diagnostics."""

    q: np.ndarray                 # (n_modes, n_time)
    residual: np.ndarray          # (n_time,) strain L2 residual per sample
    noise_gain: float             # ||A||_F, noise amplification factor
    condition_number: float       # cond(H)
    regularization: float

    def residual_rms(self) -> float:
        return float(np.sqrt(np.mean(self.residual**2)))

    def as_dict(self) -> dict:
        return {"noise_gain": self.noise_gain,
                "condition_number": self.condition_number,
                "regularization": self.regularization,
                "residual_rms": self.residual_rms(),
                "n_samples": int(self.q.shape[1])}


class ModalFilter:
    """Least-squares (optionally Tikhonov-regularised) strain -> modal filter."""

    def __init__(self, strain_mode_matrix, *, regularization: float = 0.0,
                 max_condition: float = 1.0e4):
        H = np.atleast_2d(np.asarray(strain_mode_matrix, dtype=float))
        if H.ndim != 2:
            raise ValueError("strain_mode_matrix must be two-dimensional")
        n_sensors, n_modes = H.shape
        if n_sensors < n_modes:
            raise ValueError("need at least as many sensors as modes")
        if not np.isfinite(H).all():
            raise ValueError("strain_mode_matrix must be finite")
        if regularization < 0.0:
            raise ValueError("regularization must be nonnegative")
        gram = H.T @ H
        cond = float(np.linalg.cond(gram))
        if cond > max_condition and regularization == 0.0:
            raise ValueError(
                f"strain-mode Gram matrix is ill conditioned (cond={cond:.3e}); "
                "add sensors or a Tikhonov regularization")
        self.H = H
        self.n_modes = n_modes
        self.n_sensors = n_sensors
        self.regularization = float(regularization)
        self.condition_number = cond
        if regularization > 0.0:
            # scaled regularisation: lambda = regularization * mean(diag(H^T H))
            lam = regularization * float(np.mean(np.diag(gram)))
            self._A = np.linalg.solve(gram + lam * np.eye(n_modes), H.T)
        else:
            self._A = np.linalg.pinv(H)
        self.noise_gain = float(np.linalg.norm(self._A, ord="fro"))

    # ------------------------------------------------------------------ apply
    def apply(self, strain) -> ModalFilterResult:
        """Invert one sample (n_sensors,) or a history (n_sensors, n_time)."""
        eps = np.asarray(strain, dtype=float)
        if eps.ndim == 1:
            if eps.size != self.n_sensors:
                raise ValueError("strain sample has the wrong length")
            q = self._A @ eps
            resid = eps - self.H @ q
            return ModalFilterResult(q=q[:, None], residual=np.array([np.linalg.norm(resid)]),
                                     noise_gain=self.noise_gain,
                                     condition_number=self.condition_number,
                                     regularization=self.regularization)
        if eps.ndim != 2 or eps.shape[0] != self.n_sensors:
            raise ValueError("strain history must have shape (n_sensors, n_time)")
        if not np.isfinite(eps).all():
            raise ValueError("strain history must be finite")
        q = self._A @ eps
        resid = eps - self.H @ q
        return ModalFilterResult(q=q,
                                 residual=np.linalg.norm(resid, axis=0),
                                 noise_gain=self.noise_gain,
                                 condition_number=self.condition_number,
                                 regularization=self.regularization)

    def predict_strain(self, q) -> np.ndarray:
        """Forward sensor strain from modal coordinates."""
        return self.H @ np.asarray(q, dtype=float)

def harmonic_modal_estimate(strain, dt: float, omega, *,
                           window_s: float = 10.0, estimator: "ModalFilter" | None = None,
                           modes=None) -> np.ndarray:
    """Sliding-window harmonic (lock-in) modal estimator.

    The instantaneous least-squares filter is *unbiased* but its random error is
    set by the sensor noise, ``~ noise_gain * sigma_noise / ||q||``; at a 20 dB
    sensor SNR that is a ~10 % error, no matter how many sensors are used.  When a
    mode rings at a known frequency ``omega_n`` — the situation of the analytic
    anchor and of springing monitoring — the noise can be averaged out by a local
    least-squares fit of

        q_n(t) ~ a0 + b sin(omega_n t) + c cos(omega_n t)

    over a window of ``window_s``.  The fitted value at the window centre is the
    estimate of ``q_n(t)``; white noise is reduced by ``~ sqrt(2 / (window_s*fs))``
    while a pure sinusoid at ``omega_n`` is reproduced without bias, which is what
    makes the 1 % anchor of the node reachable.

    Returns an array shaped like the modal-coordinate history ``(n_modes, n_time)``.
    """
    eps = np.atleast_2d(np.asarray(strain, dtype=float))
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    if estimator is None:
        raise ValueError("an fitted ModalFilter instance is required to know H")
    if estimator.H.shape[0] != eps.shape[0]:
        raise ValueError("strain has the wrong number of sensors")
    q_inst = estimator.apply(eps).q
    n_modes, n_time = q_inst.shape
    window = int(round(window_s / dt))
    if window % 2 == 0:
        window += 1
    window = max(window, 5)
    if window >= n_time:
        raise ValueError("window_s must be shorter than the record")
    if modes is None:
        modes = range(n_modes)
    out = np.array(q_inst, dtype=float, copy=True)
    half = (window - 1) // 2
    tau = (np.arange(window) - half) * dt
    for n in modes:
        w = omega[n]
        basis = np.stack([np.ones_like(tau), np.sin(w * tau), np.cos(w * tau)], axis=1)
        pinv_basis = np.linalg.pinv(basis)
        # fitted value at the window centre: basis(0) = [1, sin 0, cos 0]
        proj = pinv_basis[0] + pinv_basis[2]
        padded = np.pad(q_inst[n], (half, half), mode="reflect")
        out[n] = np.convolve(padded, proj[::-1], mode="valid")
    return out
