"""Rubble particle generation for 3D ice ridge keel.

Generates ice particles (spheres) satisfying:
- Prescribed porosity n (void fraction)
- Power-law size distribution N(D) ~ D^(-alpha)
- Particle count closed-form for given porosity and distribution

Analytical anchors:
- Particle volume: V_particles = (1 - n) * V_keel  (criterion b, error < 2%)
- Particle count closed-form from power-law integration
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "PowerLawPSD3D",
    "generate_rubble_particles",
    "particle_count_closed_form",
    "measure_porosity",
    "total_particle_volume",
]


def _finite(name: str, value, *, positive: bool = False, nonnegative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class PowerLawPSD3D:
    """Power-law particle size distribution in 3D: N(D) = C * D^(-alpha).

    The number of particles in diameter range [D, D+dD] is:
        dN = C * D^(-alpha) dD

    For alpha != 1, the cumulative count from D_min to D_max is:
        N_total = C/(1-alpha) * (D_max^(1-alpha) - D_min^(1-alpha))

    The total volume of particles is:
        V_total = C * pi/6 * integral(D^(3-alpha), D_min, D_max)
                = C * pi/6 * (D_max^(4-alpha) - D_min^(4-alpha)) / (4 - alpha)

    Parameters
    ----------
    d_min : float
        Minimum particle diameter (m), typical 0.1-0.5 m.
    d_max : float
        Maximum particle diameter (m), typical 1.0-5.0 m.
    alpha : float
        Power-law exponent, typical 2.0-3.5 for ridge rubble.
    """

    d_min: float = 0.2
    d_max: float = 2.0
    alpha: float = 2.5

    def __post_init__(self) -> None:
        _finite("d_min", self.d_min, positive=True)
        _finite("d_max", self.d_max, positive=True)
        _finite("alpha", self.alpha)
        if self.d_min >= self.d_max:
            raise ValueError("d_min must be less than d_max")
        if abs(self.alpha - 1.0) < 1e-12:
            raise ValueError("alpha must not equal 1 (logarithmic divergence)")
        if abs(self.alpha - 4.0) < 1e-12:
            raise ValueError("alpha must not equal 4 (volume integral divergence)")

    def normalization_constant(self, n_total: float) -> float:
        """Compute C such that total particle count = n_total."""
        p = self.alpha
        lo, hi = self.d_min ** (1.0 - p), self.d_max ** (1.0 - p)
        return n_total * (1.0 - p) / (hi - lo)

    def total_volume_for_count(self, n_total: float) -> float:
        """Total volume of n_total particles following this PSD."""
        c = self.normalization_constant(n_total)
        p = self.alpha
        vol_integral = (self.d_max ** (4.0 - p) - self.d_min ** (4.0 - p)) / (4.0 - p)
        return c * math.pi / 6.0 * vol_integral

    def count_for_volume(self, target_volume: float) -> float:
        """Number of particles needed to achieve target_volume."""
        p = self.alpha
        vol_integral = (self.d_max ** (4.0 - p) - self.d_min ** (4.0 - p)) / (4.0 - p)
        c_factor = math.pi / 6.0 * vol_integral
        # N_total = target_volume / (C_factor * C_per_particle)
        # where C_per_particle = (1-alpha)/(D_max^(1-alpha) - D_min^(1-alpha))
        count_factor = (1.0 - p) / (self.d_max ** (1.0 - p) - self.d_min ** (1.0 - p))
        return target_volume / (c_factor * count_factor)

    def mean_diameter(self) -> float:
        """Mean particle diameter <D>."""
        p = self.alpha
        if abs(p - 2.0) < 1e-12:
            num = math.log(self.d_max / self.d_min)
        else:
            num = (self.d_max ** (2.0 - p) - self.d_min ** (2.0 - p)) / (2.0 - p)
        den = (self.d_max ** (1.0 - p) - self.d_min ** (1.0 - p)) / (1.0 - p)
        return num / den

    def mean_volume(self) -> float:
        """Mean particle volume <V>."""
        p = self.alpha
        num = (self.d_max ** (4.0 - p) - self.d_min ** (4.0 - p)) / (4.0 - p)
        den = (self.d_max ** (1.0 - p) - self.d_min ** (1.0 - p)) / (1.0 - p)
        return math.pi / 6.0 * num / den

    def sample(self, count: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample `count` particle diameters from the power-law distribution."""
        if count <= 0:
            raise ValueError("count must be positive")
        if rng is None:
            rng = np.random.default_rng(42)
        p = self.alpha
        lo = self.d_min ** (1.0 - p)
        hi = self.d_max ** (1.0 - p)
        u = rng.random(count)
        return (lo + u * (hi - lo)) ** (1.0 / (1.0 - p))

    def sample_scaled(self, count: int, target_volume: float,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample diameters and scale them to achieve target total volume.

        This preserves the power-law shape while matching the exact volume.
        """
        raw = self.sample(count, rng)
        raw_vol = total_particle_volume(raw)
        if raw_vol <= 0:
            return raw
        # Scale factor: V ~ D^3, so scale = (target/raw)^(1/3)
        scale = (target_volume / raw_vol) ** (1.0 / 3.0)
        return raw * scale


def particle_count_closed_form(keel_volume: float, porosity: float,
                                psd: PowerLawPSD3D) -> float:
    """Closed-form particle count for given keel volume, porosity, and PSD.

    V_solid = (1 - n) * V_keel
    N = V_solid / <V_particle>

    Analytical anchor for acceptance criterion (b).
    """
    _finite("keel_volume", keel_volume, positive=True)
    _finite("porosity", porosity, nonnegative=True)
    if porosity >= 1.0:
        raise ValueError("porosity must be < 1.0")
    v_solid = (1.0 - porosity) * keel_volume
    mean_vol = psd.mean_volume()
    return v_solid / mean_vol


def total_particle_volume(diameters: np.ndarray) -> float:
    """Total volume of spheres with given diameters."""
    return float(np.sum(math.pi / 6.0 * diameters ** 3))


def measure_porosity(particle_volume: float, envelope_volume: float) -> float:
    """Measured porosity from particle and envelope volumes."""
    if envelope_volume <= 0.0:
        raise ValueError("envelope_volume must be positive")
    return max(0.0, min(1.0, 1.0 - particle_volume / envelope_volume))


def generate_rubble_particles(keel_volume: float, porosity: float,
                               psd: PowerLawPSD3D,
                               rng: Optional[np.random.Generator] = None,
                               target_count: Optional[int] = None) -> dict:
    """Generate rubble particle diameters satisfying porosity and PSD.

    Parameters
    ----------
    keel_volume : float
        Keel envelope volume (m³).
    porosity : float
        Target porosity (void fraction).
    psd : PowerLawPSD3D
        Particle size distribution.
    rng : np.random.Generator, optional
        Random number generator.
    target_count : int, optional
        Override the closed-form count with a specific number.

    Returns
    -------
    dict with keys:
        - diameters: np.ndarray of particle diameters (m)
        - count: number of particles
        - total_volume: total particle volume (m³)
        - target_solid_volume: (1-n)*V_keel (m³)
        - volume_error: relative error vs target solid volume
        - measured_porosity: actual porosity from generated particles
    """
    _finite("keel_volume", keel_volume, positive=True)
    _finite("porosity", porosity, nonnegative=True)
    if porosity >= 1.0:
        raise ValueError("porosity must be < 1.0")

    target_solid_volume = (1.0 - porosity) * keel_volume

    if target_count is not None:
        count = target_count
    else:
        count = int(round(particle_count_closed_form(keel_volume, porosity, psd)))

    if count <= 0:
        raise ValueError("computed particle count must be positive")

    # Use scaled sampling to match target volume exactly
    diameters = psd.sample_scaled(count, target_solid_volume, rng)
    actual_volume = total_particle_volume(diameters)
    measured_por = measure_porosity(actual_volume, keel_volume)
    volume_error = abs(actual_volume - target_solid_volume) / target_solid_volume

    return {
        "diameters": diameters,
        "count": count,
        "total_volume": actual_volume,
        "target_solid_volume": target_solid_volume,
        "volume_error": volume_error,
        "measured_porosity": measured_por,
    }
