"""AE statistical analysis: event rate, b-value, and bootstrap confidence intervals.

Event rate λ(t):
    The instantaneous event rate is estimated by binning events in time windows
    and computing the count per unit time. For a monotonic loading test at
    constant rate, λ(t) should increase as damage accumulates.

    The rate scaling with loading rate v is:
        λ ∝ v^α
    where α is a material-dependent exponent. For three loading rates, we
    fit log(λ) vs log(v) and require the slope CI half-width < 20%.

b-value (Gutenberg-Richter / Aki maximum likelihood):
    The frequency-magnitude distribution follows:
        log₁₀(N) = a - b · M
    where M is the magnitude (here, log₁₀ of amplitude or energy).

    The Aki (1965) maximum likelihood estimator for b is:
        b = ln(10) / (⟨M⟩ - M_min)
    or equivalently for energy:
        b = ln(10) / (⟨log₁₀(E)⟩ - log₁₀(E_min))

    For a Weibull distribution of bond strengths with shape parameter m:
        b_theoretical = m / ln(10) ≈ 0.434 · m

    We require |b_measured - b_theoretical| / b_theoretical < 10%.

Bootstrap confidence intervals:
    Non-parametric bootstrap (B=1000 resamples) of the b-value to estimate
    the 95% confidence interval. The CI half-width should be < 20% of the
    point estimate for well-constrained data.
"""
from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np


def event_rate(times_s: Sequence[float],
               window_size_s: float | None = None,
               n_bins: int = 10) -> tuple[list[float], list[float]]:
    """Compute AE event rate λ(t) by time binning.

    Args:
        times_s: Event times in seconds (must be sorted).
        window_size_s: Bin width in seconds. If None, uses (t_max - t_min) / n_bins.
        n_bins: Number of bins (used only if window_size_s is None).

    Returns:
        (bin_centers, rates): Time bin centers (s) and event rates (events/s).
    """
    if len(times_s) < 2:
        return [], []

    t_min, t_max = min(times_s), max(times_s)
    if t_max <= t_min:
        return [t_min], [float(len(times_s))]

    if window_size_s is None:
        window_size_s = (t_max - t_min) / n_bins

    n_windows = max(1, int(math.ceil((t_max - t_min) / window_size_s)))
    counts = [0] * n_windows

    for t in times_s:
        idx = min(int((t - t_min) / window_size_s), n_windows - 1)
        counts[idx] += 1

    rates = [c / window_size_s for c in counts]
    bin_centers = [t_min + (i + 0.5) * window_size_s for i in range(n_windows)]

    return bin_centers, rates


def event_rate_mean(times_s: Sequence[float]) -> float:
    """Mean event rate (events/s) over the entire duration."""
    if len(times_s) < 2:
        return 0.0
    duration = max(times_s) - min(times_s)
    if duration <= 0:
        return float(len(times_s))
    return len(times_s) / duration


def aki_bvalue_mle(magnitudes: Sequence[float],
                   magnitude_min: float | None = None) -> float:
    """Aki (1965) maximum likelihood estimator for the b-value.

    For a Gutenberg-Richter distribution log₁₀(N) = a - b·M:
        b_MLE = ln(10) / (⟨M⟩ - M_min)

    Args:
        magnitudes: Event magnitudes (e.g., log₁₀(amplitude) or log₁₀(energy)).
        magnitude_min: Minimum magnitude cutoff. If None, uses min(magnitudes).

    Returns:
        b-value estimate.
    """
    if len(magnitudes) < 2:
        return float('nan')

    mags = list(magnitudes)
    if magnitude_min is None:
        magnitude_min = min(mags)

    # Filter to magnitudes >= magnitude_min
    filtered = [m for m in mags if m >= magnitude_min]
    if len(filtered) < 2:
        return float('nan')

    mean_mag = sum(filtered) / len(filtered)
    delta = mean_mag - magnitude_min

    if delta <= 0:
        return float('inf')

    b = math.log(10) / delta
    return b


def aki_bvalue_from_energies(energies_J: Sequence[float],
                             energy_min_J: float | None = None) -> float:
    """Compute b-value from energy release values.

    Uses magnitude = log₁₀(E) and applies Aki MLE.
    """
    if len(energies_J) < 2:
        return float('nan')

    # Filter positive energies
    pos_energies = [e for e in energies_J if e > 0]
    if len(pos_energies) < 2:
        return float('nan')

    log_energies = [math.log10(e) for e in pos_energies]

    if energy_min_J is None:
        energy_min_J = min(pos_energies)
    log_min = math.log10(energy_min_J)

    return aki_bvalue_mle(log_energies, log_min)


def bootstrap_bvalue_ci(magnitudes: Sequence[float],
                        magnitude_min: float | None = None,
                        n_bootstrap: int = 1000,
                        confidence: float = 0.95,
                        seed: int | None = None) -> dict[str, float]:
    """Bootstrap confidence interval for the b-value.

    Args:
        magnitudes: Event magnitudes.
        magnitude_min: Minimum magnitude cutoff.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: b_mean, b_median, ci_lower, ci_upper, ci_half_width,
        ci_half_width_relative.
    """
    if len(magnitudes) < 2:
        return {
            "b_mean": float('nan'), "b_median": float('nan'),
            "ci_lower": float('nan'), "ci_upper": float('nan'),
            "ci_half_width": float('nan'), "ci_half_width_relative": float('nan'),
        }

    rng = random.Random(seed)
    mags = list(magnitudes)
    n = len(mags)

    b_samples = []
    for _ in range(n_bootstrap):
        resample = [rng.choice(mags) for _ in range(n)]
        b = aki_bvalue_mle(resample, magnitude_min)
        if math.isfinite(b):
            b_samples.append(b)

    if len(b_samples) < 10:
        return {
            "b_mean": float('nan'), "b_median": float('nan'),
            "ci_lower": float('nan'), "ci_upper": float('nan'),
            "ci_half_width": float('nan'), "ci_half_width_relative": float('nan'),
        }

    b_samples.sort()
    alpha = 1.0 - confidence
    lo_idx = int(math.floor(alpha / 2 * len(b_samples)))
    hi_idx = int(math.ceil((1 - alpha / 2) * len(b_samples))) - 1

    b_mean = sum(b_samples) / len(b_samples)
    b_median = b_samples[len(b_samples) // 2]
    ci_lower = b_samples[lo_idx]
    ci_upper = b_samples[hi_idx]
    ci_half_width = (ci_upper - ci_lower) / 2.0
    ci_half_width_relative = ci_half_width / abs(b_mean) if abs(b_mean) > 0 else float('inf')

    return {
        "b_mean": b_mean,
        "b_median": b_median,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_half_width": ci_half_width,
        "ci_half_width_relative": ci_half_width_relative,
        "n_valid_samples": len(b_samples),
    }


def rate_scaling_exponent(rates: Sequence[float],
                          velocities: Sequence[float]) -> dict[str, float]:
    """Fit the rate-velocity scaling exponent α where λ ∝ v^α.

    Uses log-log linear regression: log(λ) = α · log(v) + const.

    Returns:
        Dict with alpha, alpha_se, r_squared.
    """
    if len(rates) != len(velocities) or len(rates) < 2:
        return {"alpha": float('nan'), "alpha_se": float('nan'), "r_squared": float('nan')}

    # Filter positive values
    pairs = [(r, v) for r, v in zip(rates, velocities) if r > 0 and v > 0]
    if len(pairs) < 2:
        return {"alpha": float('nan'), "alpha_se": float('nan'), "r_squared": float('nan')}

    log_r = [math.log(p[0]) for p in pairs]
    log_v = [math.log(p[1]) for p in pairs]

    n = len(pairs)
    mean_log_v = sum(log_v) / n
    mean_log_r = sum(log_r) / n

    ss_vv = sum((x - mean_log_v) ** 2 for x in log_v)
    ss_vr = sum((log_v[i] - mean_log_v) * (log_r[i] - mean_log_r) for i in range(n))
    ss_rr = sum((x - mean_log_r) ** 2 for x in log_r)

    if ss_vv <= 0:
        return {"alpha": float('nan'), "alpha_se": float('nan'), "r_squared": float('nan')}

    alpha = ss_vr / ss_vv
    r_squared = (ss_vr ** 2) / (ss_vv * ss_rr) if ss_rr > 0 else 0.0

    # Standard error of alpha
    if n > 2:
        residuals = [log_r[i] - (alpha * log_v[i] + mean_log_r - alpha * mean_log_v) for i in range(n)]
        sse = sum(r ** 2 for r in residuals)
        mse = sse / (n - 2)
        alpha_se = math.sqrt(mse / ss_vv)
    else:
        alpha_se = 0.0

    return {"alpha": alpha, "alpha_se": alpha_se, "r_squared": r_squared}


def weibel_bvalue_theoretical(weibull_m: float) -> float:
    """Theoretical b-value for a Weibull strength distribution with shape m.

    For Weibull P(σ) ∝ σ^(m-1) · exp(-(σ/σ₀)^m), the equivalent GR b-value is:
        b = m / ln(10) ≈ 0.4343 · m
    """
    return weibull_m / math.log(10)


__all__ = [
    "event_rate", "event_rate_mean",
    "aki_bvalue_mle", "aki_bvalue_from_energies",
    "bootstrap_bvalue_ci", "rate_scaling_exponent",
    "weibel_bvalue_theoretical",
]
