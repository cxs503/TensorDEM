"""Axial banding model for a long rotating cylinder (reduced-order, verified).

Physical background
-------------------
In a long rotating cylinder, size-bidisperse granular beds also segregate *along
the axis*: bands of coarse and fine material appear and then slowly coarsen
(Savage 1979; Hill & Tan 2013).  The accepted mechanism is a differential axial
flux driven by the difference of the dynamic repose angles of the two species
plus axial diffusion of the flowing layer.

This module deliberately does **not** attempt a 3D DEM band simulation (that is
out of scope for one verification node and would not give a machine-checkable
anchor).  Instead the axial dynamics is written in the conserved
Cahn-Hilliard form used for banding in the segregation literature

    dphi/dt = M * d^2/dx^2 [ f'(phi) - kappa * d^2phi/dx^2 ],
    f(phi)  = c/4 * (phi^2 - 1)^2,   f'(phi) = c * phi * (phi^2 - 1)

where ``phi`` is the coarse-species fraction along the axis, ``M`` the axial
mobility (repose-angle-driven drift), ``c`` sets the amplitude of the
segregation "double well" and ``kappa`` the band-interface energy.  The model is
verified against its own closed-form linear theory:

    sigma(k) = M*k^2 * (c - kappa*k^2)          (about a 50/50 mixture, phi0=0)
    k_max    = sqrt(c / (2*kappa)),   lambda_max = 2*pi*sqrt(2*kappa/c)
    spinodal: unstable iff 3*phi0^2 - 1 < 0  (i.e. |phi0| < 1/sqrt(3))

Verification surface (see ``axial_verification``):
  A1 measured growth rate of a single k-mode vs ``sigma(k)``          (<=3 %)
  A2 selected band wavelength vs ``lambda_max``                       (<=10 %)
  A3 conservation of ``integral phi dx``                              (<=1e-10)
  A4 negative control: phi0 = 0.8 (outside the spinodal) -> no bands, decay
  A5 negative control: kappa -> 0 removes wavelength selection (band count
     collapses onto grid noise / no separation of scales)

Calibration of ``M``, ``c``, ``kappa`` to a real 3D drum is **not** performed
here and the model is therefore reported as a verified *mechanism* model only
(WITHHELD as a quantitative axial-segregation predictor).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["AxialBandingConfig", "simulate_bands", "linear_growth_rate",
           "dominant_wavelength", "axial_verification"]


class AxialBandingConfig:
    def __init__(self, n=256, length=1.0, c=1.0, kappa=5.0e-4, mobility=1.0,
                 phi0=0.0, noise=1.0e-3, dt=2.0e-6, n_steps=4000, seed=0):
        self.n = int(n)
        self.length = float(length)
        self.c = float(c)
        self.kappa = float(kappa)
        self.mobility = float(mobility)
        self.phi0 = float(phi0)
        self.noise = float(noise)
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.seed = int(seed)

    @property
    def dx(self):
        return self.length / self.n

    @property
    def k_max_theory(self):
        return math.sqrt(self.c / (2.0 * self.kappa))

    @property
    def wavelength_theory(self):
        return 2.0 * math.pi / self.k_max_theory

    @property
    def spinodal_limit(self):
        return 1.0 / math.sqrt(3.0)


def linear_growth_rate(k, c, kappa, mobility):
    """sigma(k) = M k^2 (c - kappa k^2) about the phi0 = 0 state."""
    return mobility * k ** 2 * (c - kappa * k ** 2)


def _rhs_step(phi_hat, k, cfg):
    """One semi-implicit step of the conserved Cahn-Hilliard model."""
    phi = np.real(np.fft.ifft(phi_hat))
    nl = phi ** 3 - phi
    nl_hat = np.fft.fft(nl)
    # dphi/dt = -M k^2 [ c*NL - kappa*(-k^2) phi ]  -> implicit in the k^4 term
    num = phi_hat - cfg.mobility * k ** 2 * cfg.c * nl_hat * cfg.dt
    den = 1.0 + cfg.mobility * cfg.kappa * k ** 4 * cfg.dt
    return num / den


def simulate_bands(cfg: AxialBandingConfig, mode_index: int | None = None):
    """Integrate the axial model.

    ``mode_index`` selects a single cosine mode as initial condition (for the
    growth-rate check); ``None`` uses broadband noise (for band selection).

    Returns dict with the final field, its FFT power spectrum, the band count,
    conservation error and the field history (a few snapshots).
    """
    rng = np.random.default_rng(cfg.seed)
    x = np.arange(cfg.n) * cfg.dx
    k = 2.0 * math.pi * np.fft.fftfreq(cfg.n, d=cfg.dx)

    if mode_index is None:
        phi = cfg.phi0 + cfg.noise * rng.standard_normal(cfg.n)
    else:
        phi = cfg.phi0 + cfg.noise * np.cos(2.0 * math.pi * mode_index * x / cfg.length)
    phi_hat = np.fft.fft(phi)
    mass0 = float(np.sum(phi) * cfg.dx)

    history = []
    amp = []
    for step in range(cfg.n_steps):
        phi_hat = _rhs_step(phi_hat, k, cfg)
        if step % max(1, cfg.n_steps // 40) == 0:
            history.append(np.real(np.fft.ifft(phi_hat)).copy())
            amp.append(float(np.std(np.real(np.fft.ifft(phi_hat)))))
    phi_f = np.real(np.fft.ifft(phi_hat))
    mass = float(np.sum(phi_f) * cfg.dx)

    power = np.abs(np.fft.rfft(phi_f - phi_f.mean())) ** 2
    kpos = 2.0 * math.pi * np.fft.rfftfreq(cfg.n, d=cfg.dx)
    power[0] = 0.0
    dominant = float(kpos[int(np.argmax(power))]) if power.size > 1 else 0.0
    wavelength = 2.0 * math.pi / dominant if dominant > 0 else float("inf")
    band_count = float(cfg.length / wavelength) if wavelength not in (0.0, float("inf")) else 0.0

    return {
        "phi": phi_f, "x": x, "power": power, "k": kpos,
        "dominant_k": dominant, "wavelength": wavelength,
        "band_count": band_count,
        "conservation_error": abs(mass - mass0) / max(abs(mass0), 1e-30),
        "amplitude_final": float(np.std(phi_f)),
        "amplitude_history": amp,
        "history": history,
        "config": cfg,
    }


def dominant_wavelength(phi, length):
    """Wavelength of the strongest non-zero FFT mode of a 1D field."""
    y = np.asarray(phi, float)
    power = np.abs(np.fft.rfft(y - y.mean())) ** 2
    k = 2.0 * math.pi * np.fft.rfftfreq(y.size, d=length / y.size)
    power[0] = 0.0
    i = int(np.argmax(power))
    return 2.0 * math.pi / k[i] if k[i] > 0 else float("inf")


def _measured_growth_rate(cfg_factory, k_mode, steps=400, amplitude=1e-4):
    """Growth rate of one k-mode, from a log-linear fit of the mode amplitude.

    The mode amplitude is extracted with a phase-projected Fourier coefficient
    ``a(t) = 2|sum(phi exp(-i k x))|/N`` and ``d(ln a)/dt`` is fitted over the
    whole window, so the estimate does not depend on the two-sample endpoint
    difference and stays inside the small-amplitude (linear) regime.
    """
    cfg = cfg_factory(n_steps=steps, noise=amplitude, dt=2e-6)
    x = np.arange(cfg.n) * cfg.dx
    k = 2.0 * math.pi * np.fft.fftfreq(cfg.n, d=cfg.dx)
    phi = cfg.phi0 + amplitude * np.cos(2.0 * math.pi * k_mode * x / cfg.length)
    phi_hat = np.fft.fft(phi)
    # projector on the driven mode (mode index k_mode == k[k_mode] for L=1)
    basis = np.exp(-1j * 2.0 * math.pi * k_mode * x / cfg.length)
    ts, amps = [], []
    n_fit = min(steps, 200)
    for step in range(n_fit):
        phi_hat = _rhs_step(phi_hat, k, cfg)
        field = np.real(np.fft.ifft(phi_hat))
        ts.append((step + 1) * cfg.dt)
        amps.append(abs(2.0 * np.sum(field * basis) / cfg.n))
    ts = np.asarray(ts)
    amps = np.asarray(amps)
    keep = amps > 1e-14
    if keep.sum() < 5:
        return float("nan")
    slope = np.polyfit(ts[keep], np.log(amps[keep]), 1)[0]
    return float(slope)


def axial_verification(n=256, c=1.0, kappa=5.0e-4, mobility=1.0):
    """Run the analytic-anchored verification of the axial model.

    Returns a dict of checks; each check is {value, reference, rel_error, pass}.
    """
    checks = {}
    lam = 2.0 * math.pi * math.sqrt(2.0 * kappa / c)
    k_th = math.sqrt(c / (2.0 * kappa))

    def factory(**kw):
        base = dict(n=n, length=1.0, c=c, kappa=kappa, mobility=mobility,
                    phi0=0.0, noise=1.0e-4, dt=2e-6, n_steps=4000, seed=1)
        base.update(kw)
        return AxialBandingConfig(**base)

    # A1 growth rate of a single mode vs theory
    growth = {}
    for m in (3, 6, 10):
        k = 2.0 * math.pi * m  # length = 1 -> k = 2 pi m
        measured = _measured_growth_rate(factory, m, steps=400, amplitude=1e-4)
        theory = linear_growth_rate(k, c, kappa, mobility)
        rel = abs(measured - theory) / abs(theory) if theory != 0 else float("nan")
        growth[m] = {"k": k, "measured": measured, "theory": theory, "rel_error": rel,
                     "pass": bool(rel <= 0.03)}
    checks["A1_growth_rate"] = growth

    # A2 wavelength selection from broadband noise
    cfg = factory(noise=1e-3, n_steps=6000)
    out = simulate_bands(cfg)
    rel = abs(out["wavelength"] - lam) / lam
    checks["A2_wavelength"] = {
        "measured": out["wavelength"], "theory": lam,
        "k_max_measured": out["dominant_k"], "k_max_theory": k_th,
        "rel_error": rel, "pass": bool(rel <= 0.10),
    }

    # A3 conservation
    checks["A3_conservation"] = {
        "value": out["conservation_error"], "reference": 0.0,
        "pass": bool(out["conservation_error"] <= 1e-10),
    }

    # A4 negative control: outside the spinodal (phi0 = 0.8) -> no banding
    cfg4 = factory(noise=1e-3, phi0=0.8, n_steps=6000)
    out4 = simulate_bands(cfg4)
    amp_ratio = out4["amplitude_final"] / 1e-3
    checks["A4_spinodal_negative_control"] = {
        "phi0": 0.8, "spinodal_limit": 1.0 / math.sqrt(3.0),
        "amplitude_ratio_final_over_initial": amp_ratio,
        "amplitude_final": out4["amplitude_final"],
        "pass": bool(amp_ratio < 1.0),   # no growth: perturbation decays
    }

    # A5 no interface energy -> no wavelength selection (band count tracks grid)
    cfg5 = factory(noise=1e-3, kappa=1e-9, n_steps=3000)
    out5 = simulate_bands(cfg5)
    checks["A5_no_kappa_limit"] = {
        "wavelength": out5["wavelength"], "band_count": out5["band_count"],
        "wavelength_theory_unbounded": float("inf"),
        "pass": bool(out5["band_count"] >= 5.0),   # collapses to grid scale
    }

    checks["all_pass"] = bool(
        all(v["pass"] for v in checks["A1_growth_rate"].values())
        and checks["A2_wavelength"]["pass"]
        and checks["A3_conservation"]["pass"]
        and checks["A4_spinodal_negative_control"]["pass"]
        and checks["A5_no_kappa_limit"]["pass"]
    )
    return checks