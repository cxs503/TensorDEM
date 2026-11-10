"""Weibull particle strength with the volume size effect, and its R^2 fit.

The strength *law* is not re-implemented here: it is
:class:`tensordem.granule_crush.WeibullStrength`, reused unchanged so the
crushing, DEM and erosion surfaces cannot drift apart:

    P_f(sigma, V) = 1 - exp[ -(sigma/sigma_0)**m (V/V_0) ]

What this module adds is the part the erosion/PBM fusion needs:

* **Inverse-CDF strength sampling** ``sigma_i = sigma_0 (V_0/V_i)**(1/m)
  (-ln U_i)**(1/m)`` -- one frozen strength per particle, which is what makes a
  DEM ensemble different from a mean-field model.
* **Impact-stress proxy** ``sigma_imp = rho_p (u sin a)**2``: the normal
  component of the dynamic pressure of a particle hitting a wall.  A particle
  breaks when ``sigma_imp`` exceeds its sampled strength.  This is the
  documented, deliberately simple closure of the impact-DEM used by
  :mod:`tensordem.erosion_pbm.dem`; it is *not* a Hertzian contact-stress
  solution, and the limitation is listed in the report.
* **The Weibull probability plot fit** used by criterion C1:

      ln ln(1/(1 - F_i)) = m ln sigma_i - m ln sigma_0

  with ``F_i`` the median-rank estimator ``(i - 0.3)/(N + 0.4)``.  ``R^2`` of
  that straight line is the goodness-of-fit number reported for C1 (threshold
  0.95).  ``m`` and ``sigma_0`` are also available from maximum likelihood
  (:func:`tensordem.bpm_calibration.weibull_mle`) for cross-checking.

Units: SI (Pa, m).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..granule_crush import WeibullStrength, sphere_volume

#: ``pi/6``: physical volume of a sphere from its diameter, ``V = (pi/6) d**3``.
SPHERE_FACTOR = math.pi / 6.0

__all__ = [
    "GranularStrength",
    "sample_strengths",
    "impact_stress",
    "breakage_probability",
    "weibull_plot_fit",
    "median_rank",
]


@dataclass(frozen=True)
class GranularStrength:
    """Weibull strength law + reference diameter (``V_0 = pi/6 d_0**3``)."""

    sigma_0: float          # Pa, at reference volume V_0
    m: float                # Weibull modulus
    d_0: float              # m, reference diameter for V_0

    @property
    def V_0(self) -> float:
        return sphere_volume(0.5 * self.d_0)

    @property
    def law(self) -> WeibullStrength:
        return WeibullStrength(sigma_0=self.sigma_0, m=self.m, V_0=self.V_0)

    def characteristic_strength(self, diameter):
        """Size-effect strength ``sigma_0 (V_0/V)**(1/m)`` [Pa].

        ``V = (pi/6) d**3`` and ``V_0 = (pi/6) d_0**3``, so the law reduces to
        ``sigma_0 (d_0/d)**(3/m)`` and is exactly ``sigma_0`` at ``d = d_0``.
        """
        d = np.asarray(diameter, dtype=float)
        return self.sigma_0 * (self.d_0 / d) ** (3.0 / self.m)

    def with_sigma0(self, sigma_0: float) -> "GranularStrength":
        """Copy with a different characteristic strength (negative controls)."""
        return GranularStrength(sigma_0=float(sigma_0), m=self.m, d_0=self.d_0)


def sample_strengths(model: GranularStrength, diameters, rng) -> np.ndarray:
    """Frozen per-particle strength ``sigma_i`` by inverse-CDF sampling [Pa]."""
    d = np.asarray(diameters, dtype=float)
    u = rng.random(d.shape)
    u = np.clip(u, 1e-300, 1.0)
    # -ln U ~ Exp(1); the size effect enters through the volume ratio.
    return model.characteristic_strength(d) * (-np.log(u)) ** (1.0 / model.m)


def impact_stress(u, alpha_deg, rho_p: float):
    """Impact-stress proxy ``rho_p (u sin a)**2`` [Pa] (normal dynamic pressure)."""
    u = np.asarray(u, dtype=float)
    s = np.sin(np.radians(np.asarray(alpha_deg, dtype=float)))
    return rho_p * (u * s) ** 2


def breakage_probability(model: GranularStrength, stress, diameter):
    """Weibull failure probability at ``stress`` for a particle of ``diameter``."""
    d = np.asarray(diameter, dtype=float)
    V = SPHERE_FACTOR * d ** 3
    x = (np.asarray(stress, dtype=float) / model.sigma_0) ** model.m * (V / model.V_0)
    return -np.expm1(-x)


def median_rank(n: int) -> np.ndarray:
    """Median-rank plotting positions ``(i - 0.3)/(n + 0.4)``."""
    i = np.arange(1, n + 1, dtype=float)
    return (i - 0.3) / (n + 0.4)


def weibull_plot_fit(sigma) -> dict:
    """Fit ``ln ln 1/(1-F)`` against ``ln sigma``; return slope, ``m``, ``R^2``.

    Returns ``{"m", "sigma_0", "r2", "intercept", "n_points"}``.  ``R^2`` is the
    coefficient of determination of the Weibull straight line, i.e. the C1
    goodness-of-fit statistic.
    """
    s = np.sort(np.asarray(sigma, dtype=float))
    if s.size < 5:
        raise ValueError("need at least 5 strength samples")
    if np.any(s <= 0.0):
        raise ValueError("strengths must be strictly positive")
    F = median_rank(s.size)
    y = np.log(np.log(1.0 / (1.0 - F)))
    x = np.log(s)
    A = np.column_stack([x, np.ones_like(x)])
    (m, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = m * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    sigma_0 = float(math.exp(-intercept / m))
    return {"m": float(m), "sigma_0": sigma_0, "r2": float(r2),
            "intercept": float(intercept), "n_points": int(s.size)}