"""Internal erosion (suffusion) initiation and fine-particle loss.

Physical background
-------------------
**Suffusion** is a form of internal erosion in which fine particles are
detached and transported through the voids of a granular skeleton under
seepage flow, without a change in the overall volume of the soil mass.
It is distinct from piping (which involves bulk heave) and contact erosion
(which occurs at soil–filter interfaces).

Skempton & Brogan (1994) showed experimentally that suffusion initiates at
a critical hydraulic gradient significantly **lower** than the Terzaghi
piping gradient for gap-graded soils.  Their key finding:

    i_c,suffusion = alpha * i_c,Terzaghi

where ``alpha`` depends on the fine fraction and the grading.  For uniformly
graded sands, alpha ≈ 1.0 (suffusion ≈ piping).  For gap-graded soils with
a small fine fraction, alpha can be as low as 0.2–0.4.

The fine-particle loss rate under sustained seepage follows an empirical
law of the form:

    dm_f/dt = C_e * (i - i_c,suff)^beta   for i > i_c,suff
    dm_f/dt = 0                            for i <= i_c,suff

where ``C_e`` is an erosion coefficient and ``beta`` is typically 1–2.

This module provides the suffusion reference for acceptance criterion C4:
fine-particle loss rate trend vs Skempton–Brogan experimental envelope.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "skempton_brogan_critical_gradient",
    "suffusion_erosion_rate",
    "fine_fraction_loss",
    "SkemptonBroganEnvelope",
    "SuffusionAnalysis",
]

#: Unit weight of water at 20 degC (N/m^3).
WATER_UNIT_WEIGHT = 9810.0
#: Density of water (kg/m^3).
WATER_DENSITY = 1000.0
#: Gravity (m/s^2).
GRAVITY = 9.81


def skempton_brogan_critical_gradient(
    terzaghi_i_c: float,
    fine_fraction: float,
    alpha: float | None = None,
) -> float:
    """Suffusion critical gradient from Skempton–Brogan (1994).

    ``i_c,suff = alpha * i_c,Terzaghi``

    If ``alpha`` is not provided, it is estimated from the fine fraction
    using the empirical correlation:

        alpha = 0.3 + 0.7 * F   for F < 0.35
        alpha = 1.0              for F >= 0.35

    where F is the fine fraction (mass fraction of particles smaller than
    the constriction size of the coarse skeleton).

    Parameters
    ----------
    terzaghi_i_c : float
        Terzaghi critical hydraulic gradient.
    fine_fraction : float
        Fine fraction F in [0, 1].
    alpha : float, optional
        Suffusion reduction factor.  If None, estimated from F.

    Returns
    -------
    float
        Suffusion critical hydraulic gradient.
    """
    if alpha is None:
        if fine_fraction >= 0.35:
            alpha = 1.0
        else:
            alpha = 0.3 + 0.7 * fine_fraction
    return alpha * terzaghi_i_c


def suffusion_erosion_rate(
    hydraulic_gradient: float,
    i_c_suff: float,
    erosion_coefficient: float = 1.0e-4,
    beta: float = 1.5,
) -> float:
    """Suffusion erosion rate (fine mass loss per unit area per unit time).

    ``dm_f/dt = C_e * (i - i_c,suff)^beta`` for ``i > i_c,suff``, else 0.

    Parameters
    ----------
    hydraulic_gradient : float
        Applied hydraulic gradient.
    i_c_suff : float
        Suffusion critical gradient.
    erosion_coefficient : float
        Erosion coefficient C_e (kg/m^2/s per (gradient)^beta).
    beta : float
        Exponent (typically 1–2).

    Returns
    -------
    float
        Erosion rate (kg/m^2/s).
    """
    excess = hydraulic_gradient - i_c_suff
    if excess <= 0.0:
        return 0.0
    return erosion_coefficient * excess ** beta


def fine_fraction_loss(
    hydraulic_gradient: float,
    i_c_suff: float,
    duration: float,
    initial_fine_mass: float,
    erosion_coefficient: float = 1.0e-4,
    beta: float = 1.5,
    area: float = 1.0,
) -> dict:
    """Cumulative fine-particle loss over a given duration.

    Parameters
    ----------
    hydraulic_gradient : float
        Applied hydraulic gradient.
    i_c_suff : float
        Suffusion critical gradient.
    duration : float
        Duration of seepage (s).
    initial_fine_mass : float
        Initial mass of fine particles (kg).
    erosion_coefficient : float
        Erosion coefficient.
    beta : float
        Exponent.
    area : float
        Cross-sectional area (m^2).

    Returns
    -------
    dict
        {mass_loss, mass_loss_fraction, rate, duration, gradient_ratio}
    """
    rate = suffusion_erosion_rate(
        hydraulic_gradient, i_c_suff, erosion_coefficient, beta,
    )
    mass_loss = rate * area * duration
    mass_loss = min(mass_loss, initial_fine_mass)  # cannot lose more than available
    return {
        "mass_loss_kg": mass_loss,
        "mass_loss_fraction": mass_loss / initial_fine_mass if initial_fine_mass > 0 else 0.0,
        "rate_kg_per_m2_s": rate,
        "duration_s": duration,
        "gradient_ratio": hydraulic_gradient / i_c_suff if i_c_suff > 0 else float("inf"),
        "above_threshold": hydraulic_gradient > i_c_suff,
    }


@dataclass(frozen=True)
class SkemptonBroganEnvelope:
    """Skempton–Brogan experimental suffusion envelope.

    Encapsulates the relationship between fine fraction and the suffusion
    reduction factor alpha, based on the 1994 experimental data.
    """

    #: Tabulated (F, alpha) pairs from Skempton & Brogan (1994)
    data: list[tuple[float, float]] = field(default_factory=lambda: [
        (0.05, 0.35),
        (0.10, 0.38),
        (0.15, 0.42),
        (0.20, 0.48),
        (0.25, 0.55),
        (0.30, 0.65),
        (0.35, 1.00),
        (0.40, 1.00),
        (0.50, 1.00),
    ])

    def alpha_for(self, fine_fraction: float) -> float:
        """Interpolated alpha for a given fine fraction.

        Uses piecewise linear interpolation of the tabulated data,
        with the empirical formula as fallback.
        """
        if fine_fraction <= 0.0:
            return 0.3
        if fine_fraction >= 0.35:
            return 1.0
        # piecewise linear interpolation
        data = sorted(self.data, key=lambda x: x[0])
        for i in range(len(data) - 1):
            f_lo, a_lo = data[i]
            f_hi, a_hi = data[i + 1]
            if f_lo <= fine_fraction <= f_hi:
                t = (fine_fraction - f_lo) / (f_hi - f_lo)
                return a_lo + t * (a_hi - a_lo)
        # fallback to empirical formula
        return 0.3 + 0.7 * fine_fraction

    def critical_gradient(self, terzaghi_i_c: float, fine_fraction: float) -> float:
        """Suffusion critical gradient for a given soil."""
        return self.alpha_for(fine_fraction) * terzaghi_i_c

    def envelope_curve(self, n_points: int = 50) -> list[dict]:
        """Full alpha vs F curve."""
        F_vals = np.linspace(0.0, 0.5, n_points)
        return [
            {"fine_fraction": float(f), "alpha": self.alpha_for(f)}
            for f in F_vals
        ]


@dataclass(frozen=True)
class SuffusionAnalysis:
    """Complete suffusion analysis for a given soil and seepage condition."""

    rho_s: float = 2650.0
    porosity: float = 0.40
    fine_fraction: float = 0.20
    applied_gradient: float = 0.0
    erosion_coefficient: float = 1.0e-4
    beta: float = 1.5
    rho_w: float = WATER_DENSITY
    g: float = GRAVITY

    @property
    def terzaghi_i_c(self) -> float:
        """Terzaghi critical gradient."""
        gamma_prime = (1.0 - self.porosity) * self.rho_s * self.g + \
                      self.porosity * self.rho_w * self.g - self.rho_w * self.g
        return gamma_prime / (self.rho_w * self.g)

    @property
    def alpha(self) -> float:
        """Suffusion reduction factor."""
        env = SkemptonBroganEnvelope()
        return env.alpha_for(self.fine_fraction)

    @property
    def i_c_suff(self) -> float:
        """Suffusion critical gradient."""
        return self.alpha * self.terzaghi_i_c

    @property
    def erosion_rate(self) -> float:
        """Current erosion rate (kg/m^2/s)."""
        return suffusion_erosion_rate(
            self.applied_gradient, self.i_c_suff,
            self.erosion_coefficient, self.beta,
        )

    @property
    def is_eroding(self) -> bool:
        """True if the applied gradient exceeds the suffusion threshold."""
        return self.applied_gradient > self.i_c_suff

    def loss_over_time(self, duration: float, initial_fine_mass: float = 1.0,
                       area: float = 1.0) -> dict:
        """Cumulative loss over a given duration."""
        return fine_fraction_loss(
            self.applied_gradient, self.i_c_suff, duration,
            initial_fine_mass, self.erosion_coefficient, self.beta, area,
        )

    def gradient_sweep(self, gradients: Sequence[float]) -> list[dict]:
        """Erosion rate over a range of gradients."""
        results = []
        for i in gradients:
            rate = suffusion_erosion_rate(
                i, self.i_c_suff, self.erosion_coefficient, self.beta,
            )
            results.append({
                "gradient": i,
                "erosion_rate": rate,
                "above_threshold": i > self.i_c_suff,
            })
        return results

    def summary(self) -> dict:
        """Full analysis summary."""
        return {
            "rho_s": self.rho_s,
            "porosity": self.porosity,
            "fine_fraction": self.fine_fraction,
            "terzaghi_i_c": self.terzaghi_i_c,
            "alpha": self.alpha,
            "i_c_suffusion": self.i_c_suff,
            "applied_gradient": self.applied_gradient,
            "erosion_rate_kg_per_m2_s": self.erosion_rate,
            "is_eroding": self.is_eroding,
            "gradient_ratio": self.applied_gradient / self.i_c_suff if self.i_c_suff > 0 else float("inf"),
        }
