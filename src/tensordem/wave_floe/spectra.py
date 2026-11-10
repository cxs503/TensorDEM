"""Sea-state spectra, harmonic realisations and spectral fidelity metrics.

Why this module exists
----------------------
The shipped wave support in :mod:`tensordem.buoyancy_wet` is a *single* Airy
wave (one amplitude, one period).  Real ice-zone seas are irregular, and every
statement of the form "the floe responds to the sea state" presupposes that the
code can (a) build a target spectrum, (b) realise it as a wave surface and
(c) recover the spectrum back from the realised surface.  Nothing in the tree
does (a)-(c), so a spectral-input requirement could not be machine-checked at
all.  This module supplies all three, with the normalisation of every spectrum
*pinned by an analytically integrable limit* rather than by a remembered
coefficient:

* the Pierson-Moskowitz shape is written in ``H_s/T_p`` form so that
  ``int S dw = H_s^2/16`` is an exact integral (checked in closed form in the
  tests, see :func:`pierson_moskowitz`);
* the JONSWAP peak enhancement carries a shape-only factor ``gamma**r`` whose
  normalisation constant ``alpha`` is obtained by *numerically integrating the
  shape* and rescaling to the requested ``H_s`` -- so no hand-copied ``alpha``
  formula is trusted and ``H_s`` is exact by construction;
* the Welch estimator is normalised so that ``int_0^inf S(w) dw`` recovers the
  variance of the signal (Parseval), which is what makes the ``R^2`` fidelity
  metric meaningful rather than cosmetic.

Conventions: ``omega`` is angular frequency [rad/s], ``S`` is the one-sided
angular-frequency PSD [m^2 s/rad], zero up-crossing ``H_s = 4 sqrt(m0)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x renamed trapz

GRAVITY = 9.81


# ---------------------------------------------------------------------------
# Spectral shapes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Spectrum:
    """A discretised one-sided spectrum (omega grid + PSD values)."""

    omega: np.ndarray
    S: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "omega", np.asarray(self.omega, dtype=float))
        object.__setattr__(self, "S", np.asarray(self.S, dtype=float))
        if self.omega.shape != self.S.shape:
            raise ValueError("omega and S must have the same shape")
        if np.any(self.omega <= 0.0):
            raise ValueError("omega must be strictly positive")

    # -- moments ---------------------------------------------------------
    def moment(self, n: int = 0) -> float:
        """Spectral moment ``m_n = int omega^n S(omega) d omega`` (trapezoid)."""
        return float(_trapz(self.S * self.omega ** n, self.omega))

    @property
    def m0(self) -> float:
        return self.moment(0)

    @property
    def significant_height(self) -> float:
        """``H_s = 4 sqrt(m0)``."""
        return 4.0 * math.sqrt(max(self.m0, 0.0))

    @property
    def peak_omega(self) -> float:
        return float(self.omega[int(np.argmax(self.S))])

    @property
    def peak_period(self) -> float:
        return 2.0 * math.pi / self.peak_omega

    @property
    def mean_period(self) -> float:
        """``T_1 = 2 pi m0 / m1``."""
        return 2.0 * math.pi * self.m0 / self.moment(1)

    def variance(self) -> float:
        return self.m0

    def to_dict(self) -> dict:
        return {
            "significant_height": self.significant_height,
            "peak_period": self.peak_period,
            "peak_omega": self.peak_omega,
            "m0": self.m0,
            "n_components": int(self.omega.size),
        }


def pierson_moskowitz(omega, significant_height: float = 2.0,
                      peak_period: float = 8.0) -> Spectrum:
    """Pierson-Moskowitz spectrum in ``H_s/T_p`` form.

    ``S(w) = (5/16) (H_s^2/w_p) (w_p/w)^5 exp(-5/4 (w_p/w)^4)``

    The prefactor is *not* remembered: ``int_0^inf S dw = H_s^2/16`` follows from
    the substitution ``t = (w_p/w)^4`` and ``int_0^inf exp(-5t/4) dt = 4/5``,
    giving ``(5/64)(4/5) H_s^2 = H_s^2/16`` exactly (:meth:`Spectrum.m0` is
    checked against that closed form in the tests).
    """
    w = np.asarray(omega, dtype=float)
    wp = 2.0 * math.pi / peak_period
    S = (5.0 / 16.0) * (significant_height ** 2 / wp) * (wp / w) ** 5 \
        * np.exp(-1.25 * (wp / w) ** 4)
    return Spectrum(w, S)


def jonswap(omega, significant_height: float = 2.0, peak_period: float = 8.0,
            gamma: float = 3.3, sigma_low: float = 0.07,
            sigma_high: float = 0.09) -> Spectrum:
    """JONSWAP spectrum normalised to the requested ``H_s``.

    ``S(w) = alpha g^2 / w^5 * exp(-5/4 (w_p/w)^4) * gamma**r``,
    ``r = exp(-(w - w_p)^2 / (2 sigma^2 w_p^2))``.

    ``alpha`` is *solved* from the normalisation ``16 int S dw = H_s^2`` by
    integrating the shape numerically on a wide logarithmic grid, so the returned
    spectrum has the requested ``H_s`` to quadrature accuracy for any ``gamma``
    (the closed-form alpha curve often quoted in the literature is an
    approximation and is deliberately not used here).
    """
    w = np.asarray(omega, dtype=float)
    wp = 2.0 * math.pi / peak_period
    sigma = np.where(w <= wp, sigma_low, sigma_high)
    r = np.exp(-((w - wp) ** 2) / (2.0 * sigma ** 2 * wp ** 2))
    shape = GRAVITY ** 2 / w ** 5 * np.exp(-1.25 * (wp / w) ** 4) * gamma ** r

    # Integral of the shape on a very wide log grid -> alpha for the target Hs.
    w_int = np.logspace(math.log10(w[0] * 1e-2), math.log10(w[-1] * 1e2), 40001)
    s_int = np.where(
        w_int <= wp, sigma_low, sigma_high)
    r_int = np.exp(-((w_int - wp) ** 2) / (2.0 * s_int ** 2 * wp ** 2))
    shape_int = GRAVITY ** 2 / w_int ** 5 * np.exp(-1.25 * (wp / w_int) ** 4) \
        * gamma ** r_int
    scale = float(_trapz(shape_int, w_int))
    alpha = significant_height ** 2 / (16.0 * scale)
    return Spectrum(w, alpha * shape)


def spectrum_grid(peak_period: float = 8.0, n: int = 512,
                  low_factor: float = 0.2, high_factor: float = 6.0) -> np.ndarray:
    """Logarithmic omega grid bracketing the spectral peak."""
    wp = 2.0 * math.pi / peak_period
    return np.logspace(math.log10(low_factor * wp), math.log10(high_factor * wp), n)


# ---------------------------------------------------------------------------
# Realisation
# ---------------------------------------------------------------------------

def discretize(spec: Spectrum, phase_seed: int | None = None) -> tuple[np.ndarray, ...]:
    """Turn a continuous spectrum into harmonic components.

    Component amplitude from the one-sided density:
    ``A_i = sqrt(2 S(w_i) dw_i)``, so ``sum A_i^2 / 2 = int S dw = m0`` up to
    quadrature error (asserted in the tests).
    """
    w = spec.omega
    dw = np.gradient(w)
    amp = np.sqrt(2.0 * spec.S * dw)
    rng = np.random.default_rng(phase_seed)
    phases = rng.uniform(0.0, 2.0 * math.pi, size=w.size)
    return w, amp, phases


def synthesize_eta(omega: np.ndarray, amp: np.ndarray, phases: np.ndarray,
                   t: np.ndarray, x: float = 0.0,
                   water_depth: float | None = None) -> np.ndarray:
    """Sum the harmonics into a surface elevation time history at station ``x``.

    Each component uses its own wavenumber from the finite-depth dispersion
    relation (deep water when ``water_depth`` is None), so the realisation is
    dispersive and phase-coherent in exactly the way the floe will feel it.
    """
    from .waves import solve_wavenumber  # local import: avoids a cycle

    t = np.asarray(t, dtype=float)
    eta = np.zeros_like(t)
    for wi, ai, pi in zip(omega, amp, phases):
        ki = solve_wavenumber(wi, water_depth)
        eta += ai * np.cos(ki * x - wi * t + pi)
    return eta


# ---------------------------------------------------------------------------
# Spectral estimation
# ---------------------------------------------------------------------------

def welch_psd(signal: Sequence[float] | np.ndarray, dt: float,
              segment: int | None = None, overlap: float = 0.5,
              window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    """One-sided angular-frequency PSD by Welch averaging.

    The normalisation ``S(f) = 2 |X|^2 / (f_s N U)`` with ``U = mean(w^2)``
    makes ``int_0^inf S(f) df`` equal the signal variance (Parseval); the
    returned ``S(omega) = S(f) / (2 pi)`` therefore satisfies
    ``int_0^inf S(omega) d omega`` = variance, matching the input spectra above.
    """
    x = np.asarray(signal, dtype=float)
    n = x.size
    if segment is None:
        segment = max(32, n // 8)
    segment = int(min(segment, n))
    step = max(1, int(round(segment * (1.0 - overlap))))
    if window == "hann":
        win = np.hanning(segment)
    elif window == "rect":
        win = np.ones(segment)
    else:
        raise ValueError(f"unknown window {window!r}")
    U = float(np.mean(win ** 2))
    starts = list(range(0, n - segment + 1, step))
    if not starts:
        starts = [0]
    acc = None
    for s in starts:
        chunk = x[s:s + segment] * win
        X = np.fft.rfft(chunk)
        # S(f) = 2|X|^2/(fs N U) with fs = 1/dt  ->  2|X|^2 dt/(N U)
        P = (np.abs(X) ** 2) * 2.0 * dt / (segment * U)
        acc = P if acc is None else acc + P
    P = acc / len(starts)
    freqs = np.fft.rfftfreq(segment, d=dt)
    # rfft: the DC and Nyquist bins are not doubled; drop DC (omega = 0) anyway.
    omega = 2.0 * math.pi * freqs[1:]
    S_omega = P[1:] / (2.0 * math.pi)
    return omega, S_omega


def spectral_fidelity(target: Spectrum, est_omega: np.ndarray,
                      est_S: np.ndarray, band: tuple[float, float] | None = None,
                      n_grid: int = 128) -> dict:
    """Compare an estimated PSD with the target spectrum on a common grid.

    ``R^2 = 1 - sum(S_est - S_tgt)^2 / sum(S_tgt - mean(S_tgt))^2`` evaluated on
    a log-spaced omega grid inside ``band`` (default ``0.5 w_p .. 3 w_p``, the
    energy-carrying band).  Also reports the ``H_s`` ratio and peak-period error,
    so a high ``R^2`` with a wrong variance cannot pass unnoticed.
    """
    wp = target.peak_omega
    lo, hi = band if band is not None else (0.5 * wp, 3.0 * wp)
    grid = np.logspace(math.log10(lo), math.log10(hi), n_grid)
    s_tgt = np.interp(grid, target.omega, target.S)
    s_est = np.interp(grid, est_omega, est_S)
    resid = float(np.sum((s_est - s_tgt) ** 2))
    denom = float(np.sum((s_tgt - s_tgt.mean()) ** 2))
    r2 = 1.0 - resid / denom if denom > 0 else float("nan")
    m0_est = float(_trapz(est_S, est_omega))
    hs_est = 4.0 * math.sqrt(max(m0_est, 0.0))
    peak_est = float(est_omega[int(np.argmax(est_S))])
    return {
        "r2": r2,
        "rms_error": math.sqrt(resid / grid.size),
        "hs_target": target.significant_height,
        "hs_estimate": hs_est,
        "hs_ratio": hs_est / target.significant_height,
        "peak_omega_target": wp,
        "peak_omega_estimate": peak_est,
        "peak_omega_error": abs(peak_est - wp) / wp,
        "band": [lo, hi],
        "n_grid": n_grid,
    }


def ensemble_psd(omega: np.ndarray, amp: np.ndarray, phases: np.ndarray,
                 dt: float, duration: float, n_realisations: int = 16,
                 segment: int | None = None, depth: float | None = None,
                 x: float = 0.0, seed0: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble-averaged Welch PSD of ``n_realisations`` independent seas.

    A single realisation of a random sea has chi-square scatter in its
    periodogram (relative spread ~ 1 at each bin), so the fidelity metric is only
    meaningful for an *ensemble* estimate -- the standard ocean-engineering
    practice (each realisation = a different random phase set, same spectrum).
    Returns the omega grid of the estimator and the mean PSD.
    """
    t = np.arange(0.0, duration, dt)
    acc = None
    for r in range(n_realisations):
        ph = np.random.default_rng(seed0 + r).uniform(0.0, 2.0 * math.pi,
                                                      size=omega.size)
        eta = synthesize_eta(omega, amp, ph, t, x=x, water_depth=depth)
        w_est, s_est = welch_psd(eta, dt, segment=segment)
        acc = s_est if acc is None else acc + s_est
    return w_est, acc / n_realisations