"""Synthetic one-dimensional fields with an *exactly known* exponential ACF.

Two generators of the same target autocorrelation ``R(r) = exp(-|r| / lam)`` are
provided, because they differ in the way a *finite window* interacts with the
generator and therefore cross-check the bias model:

``"spectral"``
    Spectral synthesis: white Gaussian noise is coloured with the Lorentzian
    two-sided PSD of the target ACF, ``S(f) = 2 lam / (1 + (2 pi f lam)**2)``,
    and inverse transformed.  The realisation is **periodic** on the record
    length, so a window shorter than the synthesis length is *not* affected by
    a mean drift; the window is cut from a much longer parent record.
``"ar1"``
    Exact first-order autoregression ``x_n = phi x_{n-1} + sqrt(1-phi**2) w_n``
    with ``phi = exp(-dx / lam)``.  Its stationary ACF on the grid is exactly
    ``exp(-k dx / lam)``, but the record is aperiodic and carries a genuine
    low-frequency wander.

Both generators return unit-variance fields (the amplitude normalisation of the
spectral synthesis is fixed by standardising the realisation; only the *shape*
of the spectrum enters, so no analytic normalisation constant is asserted).

Because the target ACF is analytic, every correlation-length estimate in this
package can be scored against a known truth -- that is the whole point: the
estimator bias is measured, not assumed.
"""
from __future__ import annotations

import numpy as np

__all__ = ["exponential_acf", "spectral_field", "ar1_field", "windowed_field"]


def exponential_acf(r: np.ndarray, lam: float) -> np.ndarray:
    """Target autocorrelation ``exp(-|r|/lam)``."""
    return np.exp(-np.abs(np.asarray(r, dtype=float)) / float(lam))


def spectral_field(
    n: int, dx: float, lam: float, rng: np.random.Generator
) -> np.ndarray:
    """One periodic realisation of length ``n*dx`` with ACF ``exp(-r/lam)``."""
    if n < 8:
        raise ValueError("need at least 8 samples")
    fk = np.fft.fftfreq(int(n), d=float(dx))
    shape = 2.0 * float(lam) / (1.0 + (2.0 * np.pi * fk * float(lam)) ** 2)
    spec = np.sqrt(shape) * (
        rng.standard_normal(int(n)) + 1j * rng.standard_normal(int(n))
    )
    x = np.fft.ifft(spec).real
    x -= x.mean()
    sd = x.std()
    return x / sd if sd > 0.0 else x


def ar1_field(
    n: int,
    dx: float,
    lam: float,
    rng: np.random.Generator,
    *,
    burn: int | None = None,
) -> np.ndarray:
    """Exact AR(1) realisation: stationary ACF ``exp(-k dx / lam)``."""
    n = int(n)
    if n < 8:
        raise ValueError("need at least 8 samples")
    phi = float(np.exp(-float(dx) / float(lam)))
    if burn is None:
        burn = int(max(50, min(5000, round(20.0 * lam / float(dx)))))
    noise = rng.standard_normal(n + burn) * np.sqrt(max(1.0 - phi * phi, 1e-30))
    x = np.empty(n + burn)
    x[0] = rng.standard_normal()
    for i in range(1, n + burn):
        x[i] = phi * x[i - 1] + noise[i]
    out = x[burn:]
    sd = out.std()
    return (out - out.mean()) / sd if sd > 0.0 else out


def windowed_field(
    n: int,
    dx: float,
    lam: float,
    rng: np.random.Generator,
    *,
    kind: str = "spectral",
    parent_factor: float = 200.0,
) -> np.ndarray:
    """A length-``n`` window cut from a parent record of ``parent_factor*lam``.

    The parent is long compared with both ``n`` and ``lam`` so the window is a
    fair draw from the stationary ensemble (no reuse of the same realisation
    across windows: a fresh parent is synthesised for every call).
    """
    n = int(n)
    if kind == "ar1":
        return ar1_field(n, dx, lam, rng)
    if kind != "spectral":
        raise ValueError(f"unknown field kind {kind!r}")
    parent = int(max(2 * n, round(parent_factor * lam / float(dx))))
    big = spectral_field(parent, dx, lam, rng)
    start = int(rng.integers(0, parent - n + 1))
    win = big[start : start + n].copy()
    win -= win.mean()
    sd = win.std()
    return win / sd if sd > 0.0 else win