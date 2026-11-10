"""Weibull weakest-link strength calibration for granular particle breakage.

Why this module
---------------
The repository already owns a Weibull single-particle strength law
(:mod:`tensordem.single_particle_crushing`, :mod:`tensordem.crushing`) and a
floe-area population balance (:mod:`tensordem.fsd_evolution.pbe`).  What the
DEM -> PBM (population balance) verification face needs on top of those is the
*statistical calibration chain* used by the population balance:

1. **Sample fitting** -- two-parameter Weibull MLE plus the CDF least-squares
   fit whose coefficient of determination is the acceptance metric of the
   crushing-force distribution.
2. **Size effect** -- the volume scaling ``sigma_c(V) = sigma_0 (V_0/V)**(1/m)``
   (weakest link) and its measured exponent.
3. **Hazard rate** -- the *breakage rate* ``S(V, t)`` that the population
   balance consumes.  Under monotone loading the survivor probability of a
   particle of volume ``V`` under the effective stress ``sigma_eff(t)`` is
   ``exp(-(sigma_eff/sigma_c(V))**m)``, so

   .. math::
      S(V,t) = -\\frac{d}{dt}\\ln P_{\\rm surv}
             = \\frac{m\\,\\sigma_{\\rm eff}(t)^{m-1}\\,\\dot\\sigma_{\\rm eff}(t)}
                    {\\sigma_c(V)^{m}} .

4. **Effective stress under a heterogeneous stress field** -- Weibull's
   volume/stress integral for a random field of particle stresses,
   ``sigma_eff = (E[sigma_p**m])**(1/m)``.  That single scalar is what a
   DEM-informed PBM needs in order to carry the force-chain heterogeneity of a
   compressed packing into the rate.

All quantities are SI (m, kg, s, Pa).  Nothing here is fitted to the target
that the PBM is later compared against.
"""
from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np
from scipy import optimize
from scipy.special import logsumexp

__all__ = [
    "weibull_cdf",
    "weibull_pdf",
    "weibull_survival",
    "WeibullFit",
    "fit_weibull_mle",
    "fit_weibull_cdf",
    "weibull_cdf_r2",
    "particle_stress",
    "effective_stress",
    "size_effect_strength",
    "fit_size_effect_exponent",
    "WeibullStrength",
]


# ------------------------------------------------------------------ densities
def weibull_cdf(sigma, m: float, sigma_0: float) -> np.ndarray:
    """Two-parameter Weibull CDF ``1 - exp(-(sigma/sigma_0)**m)``."""
    s = np.clip(np.asarray(sigma, dtype=np.float64), 0.0, None)
    return 1.0 - np.exp(-((s / float(sigma_0)) ** float(m)))


def weibull_survival(sigma, m: float, sigma_0: float) -> np.ndarray:
    """Survivor probability ``exp(-(sigma/sigma_0)**m)``."""
    s = np.clip(np.asarray(sigma, dtype=np.float64), 0.0, None)
    return np.exp(-((s / float(sigma_0)) ** float(m)))


def weibull_pdf(sigma, m: float, sigma_0: float) -> np.ndarray:
    """Two-parameter Weibull PDF."""
    s = np.clip(np.asarray(sigma, dtype=np.float64), 1e-300, None)
    z = s / float(sigma_0)
    return (float(m) / float(sigma_0)) * z ** (float(m) - 1.0) * np.exp(-(z ** float(m)))


# ------------------------------------------------------------------ fitting
@dataclass(frozen=True)
class WeibullFit:
    """A fitted two-parameter Weibull distribution with its diagnostics."""

    m: float
    sigma_0: float
    n: int
    r2_cdf: float
    r2_linear: float
    ks_statistic: float

    def as_dict(self) -> dict:
        return {
            "m": self.m,
            "sigma_0": self.sigma_0,
            "n": self.n,
            "r2_cdf": self.r2_cdf,
            "r2_linear": self.r2_linear,
            "ks_statistic": self.ks_statistic,
        }


def _as_positive_samples(samples) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 3:
        raise ValueError("need at least 3 finite samples")
    if np.any(x <= 0.0):
        raise ValueError("Weibull samples must be strictly positive")
    return np.sort(x)


def fit_weibull_mle(samples) -> tuple[float, float]:
    """Maximum-likelihood ``(m, sigma_0)`` with the profiled shape modulus."""
    x = _as_positive_samples(samples)
    n = x.size
    ln_x = np.log(x)

    def neg_profile_loglik(m: float) -> float:
        m = float(m)
        # sigma_0^m = mean(x^m) at the MLE of the scale; computed in log space so
        # the profile stays finite for large moduli
        ln_mean_xm = float(logsumexp(m * ln_x) - math.log(n))
        # -n*m*ln(sigma_0) = -n*ln(mean(x**m)) with the profiled scale
        loglik = (n * math.log(m) - n * ln_mean_xm
                  + (m - 1.0) * float(np.sum(ln_x)) - n)
        return -loglik

    result = optimize.minimize_scalar(
        neg_profile_loglik, bounds=(0.1, 30.0), method="bounded",
        options={"xatol": 1e-8},
    )
    m = float(result.x)
    sigma_0 = float(np.mean(x ** m) ** (1.0 / m))
    return m, sigma_0


def weibull_cdf_r2(samples, m: float, sigma_0: float) -> float:
    """Coefficient of determination of the fitted CDF against the empirical CDF.

    ``R2 = 1 - SS_res/SS_tot`` on the CDF scale, with the unbiased plotting
    position ``F_i = (i - 0.3)/(n + 0.4)`` (Blom).  This is the acceptance
    metric of the crushing-force distribution (``R2 > 0.95``).
    """
    x = _as_positive_samples(samples)
    n = x.size
    f_emp = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
    f_fit = weibull_cdf(x, m, sigma_0)
    ss_res = float(np.sum((f_emp - f_fit) ** 2))
    ss_tot = float(np.sum((f_emp - f_emp.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def fit_weibull_cdf(samples) -> WeibullFit:
    """CDF least-squares fit (linearised) plus MLE-free diagnostics.

    The linearised fit ``ln(-ln(1-F)) = m ln(sigma) - m ln(sigma_0)`` is the
    classical graphical estimator; the returned ``r2_cdf`` is computed on the
    CDF scale (not on the linearised scale) because that is the quantity the
    acceptance criterion quotes.
    """
    x = _as_positive_samples(samples)
    n = x.size
    f_emp = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
    y = np.log(-np.log(1.0 - f_emp))
    design = np.vstack([np.log(x), np.ones(n)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, y, rcond=None)
    m = float(slope)
    sigma_0 = float(np.exp(-intercept / slope))

    f_fit_lin = 1.0 - np.exp(-np.exp(slope * np.log(x) + intercept))
    ss_res = float(np.sum((f_emp - f_fit_lin) ** 2))
    ss_tot = float(np.sum((f_emp - f_emp.mean()) ** 2))
    r2_linear = 1.0 - ss_res / ss_tot

    r2_cdf = weibull_cdf_r2(x, m, sigma_0)
    f_fit = weibull_cdf(x, m, sigma_0)
    ks = float(np.max(np.abs(f_emp - f_fit)))
    return WeibullFit(m=m, sigma_0=sigma_0, n=n, r2_cdf=r2_cdf,
                      r2_linear=r2_linear, ks_statistic=ks)


# ------------------------------------------------------------------ mechanics
def particle_stress(force, diameter) -> np.ndarray:
    """Nominal particle crushing stress ``sigma = F / d**2``.

    The platen-force-over-projected-area convention of single-particle
    crushing tests; the same convention is used for the DEM bond-stress
    detection so that calibration and application cannot drift apart.
    """
    f = np.asarray(force, dtype=np.float64)
    d = np.asarray(diameter, dtype=np.float64)
    return f / (d * d)


def effective_stress(particle_stresses, m: float) -> float:
    """Weibull effective stress ``(E[sigma**m])**(1/m)`` of a stress field.

    Exact volume/stress integral of the weakest-link failure probability for a
    population whose stress is not uniform: ``P_surv = E[exp(-(s/s_c)**m)]`` is
    *not* ``exp(-(E[s]/s_c)**m)``, but ``exp(-(sigma_eff/s_c)**m)`` with the
    moment definition above is the leading (small-probability) closure.
    """
    s = np.asarray(particle_stresses, dtype=np.float64).ravel()
    s = s[np.isfinite(s) & (s > 0.0)]
    if s.size == 0:
        return 0.0
    return float(np.mean(s ** float(m)) ** (1.0 / float(m)))


def size_effect_strength(sigma_0: float, v_0: float, volume, m: float) -> np.ndarray:
    """Characteristic strength of a particle of the given volume."""
    v = np.clip(np.asarray(volume, dtype=np.float64), 1e-30, None)
    return float(sigma_0) * (float(v_0) / v) ** (1.0 / float(m))


def fit_size_effect_exponent(volumes, median_strengths) -> dict:
    """Fit ``ln sigma_med = ln A - (1/m) ln V``; return the modulus ``m = -1/slope``."""
    v = np.asarray(volumes, dtype=np.float64).ravel()
    s = np.asarray(median_strengths, dtype=np.float64).ravel()
    if v.size != s.size or v.size < 3:
        raise ValueError("need >= 3 (volume, median strength) pairs")
    slope, intercept = np.polyfit(np.log(v), np.log(s), 1)
    pred = slope * np.log(v) + intercept
    resid = np.log(s) - pred
    ss_tot = float(np.sum((np.log(s) - np.log(s).mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0
    return {"slope": float(slope), "m": float(-1.0 / slope), "r2": r2,
            "sigma_at_v0": float(np.exp(intercept))}


@dataclass(frozen=True)
class WeibullStrength:
    """Weakest-link strength law of a particle population.

    ``m`` is the shape modulus, ``sigma_0`` the characteristic strength at the
    reference volume ``v_0``; the characteristic strength of a particle of
    volume ``V`` is ``sigma_0 (v_0/V)**(1/m)``.
    """

    m: float
    sigma_0: float
    v_0: float

    def characteristic_stress(self, volume) -> np.ndarray:
        return size_effect_strength(self.sigma_0, self.v_0, volume, self.m)

    def cdf(self, stress, volume) -> np.ndarray:
        return weibull_cdf(stress, self.m, self.characteristic_stress(volume))

    def survival(self, stress, volume) -> np.ndarray:
        return weibull_survival(stress, self.m, self.characteristic_stress(volume))

    def hazard_rate(self, stress, stress_rate, volume) -> np.ndarray:
        """Breakage rate ``S(V,t)`` for a monotone loading history."""
        sigma = np.clip(np.asarray(stress, dtype=np.float64), 0.0, None)
        rate = np.asarray(stress_rate, dtype=np.float64)
        sigma_c = self.characteristic_stress(volume)
        prefactor = self.m * rate / (sigma_c ** self.m)
        return prefactor * sigma ** (self.m - 1.0)

    def sample_strength(self, volume, rng: np.random.Generator, size=None) -> np.ndarray:
        """Draw strengths: ``sigma_c(V) * (-ln U)**(1/m)``."""
        u = rng.random(size) if size is not None else rng.random()
        sigma_c = self.characteristic_stress(volume)
        return sigma_c * (-np.log(np.clip(u, 1e-300, 1.0))) ** (1.0 / self.m)