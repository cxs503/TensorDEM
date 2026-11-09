"""Correlation-length estimators plus a regime diagnostic.

All estimators take a *line record* ``x`` sampled at spacing ``dx`` and return a
correlation length in the same length unit as ``dx``.  The definitions are fixed
**before** any bias is measured, so the calibration in :mod:`.calibration` is a
test of the estimators, not a tuned fit:

``"integral"`` (primary)
    ``lam = int_0^{r_end} R(r) dr`` with ``R`` the *unbiased* (overlap-count
    corrected) normalised autocorrelation and ``r_end`` the first lag where
    ``R`` turns negative (capped at ``0.5 L``).  Standard integral scale; the
    truncation at the first zero crossing is what makes it finite on a noisy
    record, and it is also the main source of its finite-window bias.
``"expfit"`` (primary)
    Weighted least-squares fit of ``R(r) = A exp(-r/lam)`` in *linear* space over
    ``r <= min(2.5 lam_e, 0.5 L)`` with weights ``(N-k)^(1/2)``, where
    ``lam_e`` is the e-folding lag.  Linear-space fitting is used because the
    log-space fit is unstable when the ACF is noise dominated at large lags (the
    log amplifies the noise; measured in this repository's estimator trials).
``"spectral"`` (primary)
    Log-space Lorentzian fit ``ln S(f) = ln A - ln(1 + (2 pi f lam)**2)`` to the
    Hann-windowed periodogram, over the frequency band up to 12 half-power
    widths (``f <= 12 / (2 pi lam_e)``), with ``lam`` bounded to
    ``[0.05, 5] lam_e``.  For ``R(r) = exp(-|r|/lam)`` the spectrum is exactly
    Lorentzian, so the model is analytic rather than assumed.  ``lam_e`` enters
    only as the scale of the fit band and of the bounds -- this coupling with
    the e-folding estimator is documented, not hidden.
``"efold"`` (legacy, for comparison)
    First crossing of ``R`` below ``1/e`` by linear interpolation.  This is the
    definition already used by ``tensordem.ice_contact.statistics``, carried here
    so the campaign can state how the legacy number is biased on a finite
    record.

``diagnose_regime`` flags the two configurations in which *none* of these
estimators is meaningful -- white noise (no correlation length to measure) and a
strictly periodic record (the full-period integral of the ACF vanishes, so a
truncated integral scale is an artefact).  Flagged records must not be fed to
the bias correction; that is a declared scope limit, not a silent fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, signal

__all__ = [
    "autocorrelation",
    "lambda_integral",
    "lambda_expfit",
    "lambda_spectral",
    "lambda_efold",
    "lambda_lagslope",
    "lambda_crossing",
    "estimate_all",
    "diagnose_regime",
    "Regime",
    "METHODS",
    "PRIMARY_METHODS",
]

METHODS = ("integral", "expfit", "spectral", "efold", "lagslope", "crossing")
PRIMARY_METHODS = ("integral", "expfit", "crossing")

_NAN = float("nan")


def autocorrelation(
    x: np.ndarray, dx: float = 1.0, *, unbiased: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised autocorrelation ``R(k)`` for lags ``k = 0 .. n-1``.

    ``unbiased=True`` divides the lag-``k`` sum by the number of surviving pairs
    ``n - k`` and normalises by the lag-0 variance, which removes the linear
    taper of the biased estimator; the price is a growing variance at large
    lags, which is why every estimator below truncates its summation.
    """
    f = np.asarray(x, dtype=float).ravel()
    n = f.size
    if n < 8:
        raise ValueError("record too short (need >= 8 samples)")
    f = f - f.mean()
    var = float(np.mean(f * f))
    if var <= 0.0:
        return np.arange(n) * dx, np.ones(n)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(f, nfft)
    raw = np.fft.irfft(spec * np.conj(spec), nfft)[:n]
    lags = np.arange(n)
    if unbiased:
        R = (raw / (n - lags)) / (raw[0] / n)
    else:
        R = raw / raw[0]
    return lags * dx, R


def _truncation_index(R: np.ndarray, max_lag_fraction: float) -> int:
    n = R.size
    kcap = max(2, int(np.floor(max_lag_fraction * (n - 1))))
    neg = np.nonzero(R[1 : kcap + 1] < 0.0)[0]
    k_end = int(neg[0]) if neg.size else kcap
    return max(k_end, 1)


def lambda_efold(x: np.ndarray, dx: float = 1.0) -> float:
    """E-folding lag of the ACF (legacy definition, linear interpolation)."""
    lags, R = autocorrelation(x, dx, unbiased=True)
    below = np.nonzero(R <= np.exp(-1.0))[0]
    if below.size == 0:
        return _NAN
    k = int(below[0])
    if k == 0:
        return float(lags[0])
    r0, r1 = lags[k - 1], lags[k]
    p0, p1 = R[k - 1], R[k]
    if p1 == p0:
        return float(r1)
    return float(r0 + (np.exp(-1.0) - p0) * (r1 - r0) / (p1 - p0))


def _record_length(x: np.ndarray, dx: float) -> float:
    return float(np.asarray(x).size) * float(dx)


def _physical(x: np.ndarray, dx: float, lam: float) -> float:
    """Reject estimates that cannot be a correlation length of this record.

    A fitted correlation length at or beyond the record length is a *fit
    divergence* (the exponential model is then flat over the whole record and
    any ``lam`` fits), and keeping it in an ensemble mean corrupts the bias model
    (observed: a single diverging realisation moved the ensemble ratio of
    ``expfit`` at ``K = 1`` to O(10^3) and destroyed the fit).  Such a value is
    reported as ``nan`` and counted as a failed realisation.
    """
    if not np.isfinite(lam) or lam <= 0.0:
        return _NAN
    if lam > _record_length(x, dx):
        return _NAN
    return float(lam)


def lambda_integral(
    x: np.ndarray, dx: float = 1.0, *, max_lag_fraction: float = 0.5
) -> float:
    """Integral scale of the (unbiased) autocorrelation."""
    lags, R = autocorrelation(x, dx, unbiased=True)
    k_end = _truncation_index(R, max_lag_fraction)
    return _physical(x, dx, float(np.trapezoid(R[: k_end + 1], lags[: k_end + 1])))


def lambda_lagslope(
    x: np.ndarray, dx: float = 1.0, *, lag: int = 1, min_decay: float = 0.02
) -> float:
    """Leading-lag log-slope (initial-slope) correlation length.

    For an exponential ACF ``R(r) = exp(-r/lam)`` the leading lag inverts exactly,
    ``lam = -r / ln R(r)``, so this estimator carries **no truncation**: it is a
    fixed-number-of-lags (local) functional of the record.  That matters for the
    variance-scaling law: the first-zero-truncated integral scale
    (:func:`lambda_integral`) sums a *noise-driven random number of lags*, so its
    variance saturates instead of falling like ``lam/L`` (measured in
    ``docs/ice_pressure_band_lambda_x.md`` §3.4).  With a fixed lag the asymptotic
    result ``Var(lam_hat/lam) ~ 2 / (L/lam)`` holds, i.e. ``Var(lam_hat) ~
    lam**3 / L``.

    The price is a larger variance than a windowed fit at very long records and a
    dependence on the ACF being locally exponential; it is therefore reported
    together with the integral/expfit estimators, never alone.  ``min_decay``
    rejects records without measurable memory at the leading lag (white noise).
    """
    k = int(max(1, lag))
    lags, R = autocorrelation(x, dx, unbiased=True)
    if k >= R.size:
        return _NAN
    r = float(R[k])
    if not np.isfinite(r) or r <= float(min_decay) or r >= 1.0:
        return _NAN
    return float(-k * dx / np.log(r))


def lambda_crossing(
    x: np.ndarray, dx: float = 1.0, *, target: float = 0.6
) -> float:
    """Level-crossing correlation length: the lag ``r0`` at which ``R`` hits ``target``.

    For ``R(r) = exp(-r/lam)`` the crossing solves ``r0 = lam ln(1/target)``, so
    ``lam = r0 / ln(1/target)``.  The lag is located by log-linear interpolation
    of the unbiased ACF, which makes the estimator

    * **well conditioned** -- the sensitivity of ``lam`` to an error in ``R`` is
      ``1 / (target ln(1/target)) ~ 3`` at ``target = 0.6``, versus ~20 for the
      leading-lag inversion at a sampling step of ``0.05 lam`` (see
      :func:`lambda_lagslope`);
    * **shape-only** -- it uses the *position* of a level crossing, not the
      absolute level of the ACF at a fixed lag, so a sub-percent disagreement
      between a generator's grid ACF and the analytic one does not turn into a
      percent-level bias in ``lam``;
    * **step independent** -- the crossing lag is a physical length, so the bias
      does not depend on how finely the record is sampled.

    ``target = 0.6`` is pre-registered (the legacy e-folding estimator is the
    ``target = exp(-1)`` special case, kept for comparison).
    """
    lags, R = autocorrelation(x, dx, unbiased=True)
    t = float(target)
    if not 0.0 < t < 1.0:
        raise ValueError("target must be in (0, 1)")
    below = np.nonzero(R <= t)[0]
    if below.size == 0:
        return _NAN
    k = int(below[0])
    if k == 0:
        return float(lags[0])
    r0, r1 = float(R[k - 1]), float(R[k])
    if r0 <= 0.0 or r1 <= 0.0:
        return _NAN
    l0, l1 = np.log(r0), np.log(r1)
    frac = (l0 - np.log(t)) / (l0 - l1) if l0 != l1 else 0.0
    frac = float(min(max(frac, 0.0), 1.0))
    r_star = float(lags[k - 1] + frac * (lags[k] - lags[k - 1]))
    return _physical(x, dx, float(r_star / np.log(1.0 / t)))


def lambda_expfit(
    x: np.ndarray,
    dx: float = 1.0,
    *,
    max_lag_fraction: float = 0.5,
    range_factor: float = 2.5,
    floor: float = 0.02,
) -> float:
    """Correlation length from a weighted linear-space exponential ACF fit."""
    lags, R = autocorrelation(x, dx, unbiased=True)
    n = R.size
    lam_e = lambda_efold(x, dx)
    if not np.isfinite(lam_e) or lam_e <= 0.0:
        return _NAN
    r_max = min(max_lag_fraction * (n - 1) * dx, range_factor * lam_e)
    k = np.arange(1, n)
    mask = (lags[1:] <= r_max) & (R[1:] > floor) & (k <= 0.5 * (n - 1))
    if mask.sum() < 4:
        return _NAN
    r_fit = lags[1:][mask]
    y = R[1:][mask]
    sigma = 1.0 / np.sqrt(np.maximum(n - k[mask], 1.0))

    def model(r, amp, lam):
        return amp * np.exp(-r / lam)

    try:
        popt, _ = optimize.curve_fit(
            model, r_fit, y, p0=(1.0, lam_e), sigma=sigma,
            bounds=([0.0, 1e-6], [3.0, np.inf]), maxfev=4000,
        )
    except (RuntimeError, ValueError):
        return _NAN
    return _physical(x, dx, float(popt[1]))


def lambda_spectral(
    x: np.ndarray,
    dx: float = 1.0,
    *,
    half_power_widths: float = 12.0,
    min_bins: int = 4,
    lower_bound: float = 0.05,
    upper_bound: float = 5.0,
) -> float:
    """Correlation length from a log-space Lorentzian fit of the periodogram."""
    f = np.asarray(x, dtype=float).ravel()
    n = f.size
    if n < 16:
        return _NAN
    lam_e = lambda_efold(f, dx)
    if not np.isfinite(lam_e) or lam_e <= 0.0:
        lam_e = max(lambda_integral(f, dx), 2.0 * dx)
    lam_e = max(float(lam_e), 2.0 * dx)
    freq, psd = signal.periodogram(
        f, fs=1.0 / dx, window="hann", detrend="linear", scaling="density"
    )
    df = freq[1] - freq[0] if freq.size > 1 else 1.0
    f_max = max(min_bins * df, half_power_widths / (2.0 * np.pi * lam_e))
    sel = (freq > 0.0) & (freq <= f_max)
    if sel.sum() < 3:
        return _NAN
    ff = freq[sel]
    yy = np.log(np.maximum(psd[sel], 1e-300))

    def model(fq, log_amp, log_theta):
        lam = lam_e * np.exp(log_theta)
        return log_amp - np.log(1.0 + (2.0 * np.pi * fq * lam) ** 2)

    try:
        popt, _ = optimize.curve_fit(
            model, ff, yy,
            p0=(float(np.log(max(np.median(psd[sel][:3]), 1e-300))), 0.0),
            bounds=([-np.inf, np.log(lower_bound)], [np.inf, np.log(upper_bound)]),
            maxfev=4000,
        )
    except (RuntimeError, ValueError):
        return _NAN
    return _physical(x, dx, float(lam_e * np.exp(popt[1])))


@dataclass(frozen=True)
class Regime:
    """Diagnostic flags for one record."""

    label: str            # "ok" | "white_noise" | "periodic"
    r_lag2: float
    noise_level: float
    secondary_peak: float

    @property
    def usable(self) -> bool:
        return self.label == "ok"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "r_lag2": self.r_lag2,
            "noise_level": self.noise_level,
            "secondary_peak": self.secondary_peak,
        }


def diagnose_regime(
    x: np.ndarray,
    dx: float = 1.0,
    *,
    white_sigmas: float = 2.5,
    periodic_level: float = 0.55,
    decay_factor: float = 3.0,
) -> Regime:
    """Classify a record as ``ok`` / ``white_noise`` / ``periodic``.

    *White noise*: the lag-2 autocorrelation is below ``white_sigmas`` times its
    sampling noise ``1/sqrt(N)`` -- there is no measurable memory left.
    *Periodic*: at lags **beyond** ``decay_factor`` e-folding lengths (where a
    non-periodic record must already be decorrelated) the ACF climbs back above
    ``periodic_level``.  Lags inside the correlation region are excluded on
    purpose: ``R(2 dx) ~ 0.9`` for a smooth correlated field is *not* a period.
    """
    _, R = autocorrelation(x, dx, unbiased=True)
    n = R.size
    noise = 1.0 / np.sqrt(max(n, 1))
    r2 = float(R[min(2, n - 1)])
    lam_e = lambda_efold(x, dx)
    scale = lam_e if np.isfinite(lam_e) and lam_e > 0 else max(2.0 * dx, n * dx / 8.0)
    k_lo = max(2, int(np.ceil(decay_factor * scale / dx)))
    hi = max(k_lo, n // 2)
    seg = R[k_lo : hi + 1]
    peak = float(seg.max()) if seg.size else 0.0
    if peak >= periodic_level:
        label = "periodic"
    elif r2 < white_sigmas * noise:
        label = "white_noise"
    else:
        label = "ok"
    return Regime(label=label, r_lag2=r2, noise_level=noise, secondary_peak=peak)


def estimate_all(
    x: np.ndarray, dx: float = 1.0, *, methods: tuple[str, ...] = METHODS
) -> dict:
    """All estimators on one record."""
    out: dict[str, float] = {}
    for m in methods:
        if m == "integral":
            out[m] = lambda_integral(x, dx)
        elif m == "expfit":
            out[m] = lambda_expfit(x, dx)
        elif m == "spectral":
            out[m] = lambda_spectral(x, dx)
        elif m == "efold":
            out[m] = lambda_efold(x, dx)
        elif m == "lagslope":
            out[m] = lambda_lagslope(x, dx)
        elif m == "crossing":
            out[m] = lambda_crossing(x, dx)
        else:
            raise ValueError(f"unknown method {m!r}")
    return out