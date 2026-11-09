"""Rotary dryer bed kinematics: regime classification based on Froude number.

The bed motion regime is determined by the Froude number:
    Fr = ω²R/g

Regime boundaries (standard industrial practice):
- Slipping:    Fr < 0.05
- Rolling:     0.05 ≤ Fr < 0.4
- Cataracting: 0.4 ≤ Fr < 1.0
- Centrifuging: Fr ≥ 1.0

References
----------
- Henein, H., et al., 1983. A model study of the kiln feed motion.
  Metallurgical and Materials Transactions B, 14(4), 679-688.
- Kelly, E.G., O'Donnell, P.W., 1977. Introduction to Mineral Dressing.
  Elsevier, Amsterdam.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

__all__ = [
    "RotaryDryerConfig",
    "BedRegime",
    "classify_regime",
    "froude_number",
    "regime_boundaries",
    "omega_for_froude",
]


class BedRegime(str, Enum):
    """Bed motion regime classification."""
    SLIPPING = "slipping"
    ROLLING = "rolling"
    CATARACTING = "cataracting"
    CENTRIFUGING = "centrifuging"


# Regime boundaries (Froude number thresholds)
REGIME_BOUNDS = {
    BedRegime.SLIPPING: (0.0, 0.05),
    BedRegime.ROLLING: (0.05, 0.4),
    BedRegime.CATARACTING: (0.4, 1.0),
    BedRegime.CENTRIFUGING: (1.0, float("inf")),
}


def regime_boundaries() -> dict:
    """Return the Froude number boundaries for each regime."""
    return {r.value: list(bounds) for r, bounds in REGIME_BOUNDS.items()}


def froude_number(omega: float, radius: float, gravity: float = 9.81) -> float:
    """Calculate the Froude number.

    Parameters
    ----------
    omega : float
        Angular velocity of the drum (rad/s).
    radius : float
        Drum internal radius (m).
    gravity : float, optional
        Gravitational acceleration (m/s²). Default 9.81.

    Returns
    -------
    float
        Froude number Fr = ω²R/g.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    if gravity <= 0:
        raise ValueError("gravity must be positive")
    return (omega ** 2) * radius / gravity


def omega_for_froude(fr: float, radius: float, gravity: float = 9.81) -> float:
    """Calculate angular velocity for a target Froude number.

    Parameters
    ----------
    fr : float
        Target Froude number.
    radius : float
        Drum internal radius (m).
    gravity : float, optional
        Gravitational acceleration (m/s²). Default 9.81.

    Returns
    -------
    float
        Angular velocity (rad/s).
    """
    if fr < 0:
        raise ValueError("Froude number must be non-negative")
    if radius <= 0:
        raise ValueError("radius must be positive")
    return math.sqrt(fr * gravity / radius)


def classify_regime(fr: float) -> BedRegime:
    """Classify the bed regime based on Froude number.

    Parameters
    ----------
    fr : float
        Froude number.

    Returns
    -------
    BedRegime
        The classified bed regime.
    """
    if not math.isfinite(fr):
        raise ValueError("Froude number must be finite")
    if fr < 0:
        raise ValueError("Froude number must be non-negative")

    for regime, (lo, hi) in REGIME_BOUNDS.items():
        if lo <= fr < hi:
            return regime
    return BedRegime.CENTRIFUGING


@dataclass
class RotaryDryerConfig:
    """Configuration for a rotary dryer simulation.

    Parameters
    ----------
    drum_diameter : float
        Internal drum diameter D (m).
    drum_length : float
        Drum length L (m).
    omega : float
        Angular velocity (rad/s).
    inclination : float
        Drum inclination angle β (radians, positive = discharge end lower).
    particle_diameter : float
        Mean particle diameter d_p (m).
    particle_density : float
        Particle density ρ_p (kg/m³).
    particle_heat_capacity : float
        Specific heat capacity c_p (J/(kg·K)).
    particle_conductivity : float
        Particle thermal conductivity k_p (W/(m·K)).
    wall_temperature : float
        Drum wall temperature T_w (K).
    particle_initial_temperature : float
        Initial particle temperature T_p0 (K).
    gas_conductivity : float
        Gas (air) thermal conductivity k_g (W/(m·K)).
    gas_temperature : float
        Gas temperature T_g (K).
    particle_emissivity : float
        Particle surface emissivity ε_p (-).
    wall_emissivity : float
        Wall surface emissivity ε_w (-).
    fill_fraction : float
        Volumetric fill fraction (-).
    restitution : float
        Coefficient of restitution (-).
    friction_coefficient : float
        Particle-wall friction coefficient μ (-).
    """
    drum_diameter: float = 1.5
    drum_length: float = 10.0
    omega: float = 0.5  # rad/s
    inclination: float = 0.05  # radians (~3 degrees)
    particle_diameter: float = 0.005  # 5 mm
    particle_density: float = 2500.0  # kg/m³
    particle_heat_capacity: float = 800.0  # J/(kg·K)
    particle_conductivity: float = 1.5  # W/(m·K)
    wall_temperature: float = 473.15  # 200°C in K
    particle_initial_temperature: float = 293.15  # 20°C in K
    gas_conductivity: float = 0.026  # W/(m·K) for air
    gas_temperature: float = 373.15  # 100°C in K
    particle_emissivity: float = 0.9
    wall_emissivity: float = 0.8
    fill_fraction: float = 0.15
    restitution: float = 0.5
    friction_coefficient: float = 0.4

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        for name in ("drum_diameter", "drum_length", "particle_diameter",
                     "particle_density", "particle_heat_capacity",
                     "particle_conductivity", "gas_conductivity"):
            val = getattr(self, name)
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")

        if self.omega < 0:
            raise ValueError(f"omega must be non-negative, got {self.omega}")

        if not 0 <= self.inclination < math.pi / 2:
            raise ValueError(f"inclination must be in [0, π/2), got {self.inclination}")

        if not 0 < self.fill_fraction < 1:
            raise ValueError(f"fill_fraction must be in (0, 1), got {self.fill_fraction}")

        if not 0 < self.restitution <= 1:
            raise ValueError(f"restitution must be in (0, 1], got {self.restitution}")

        if self.friction_coefficient < 0:
            raise ValueError(f"friction_coefficient must be non-negative")

        if self.wall_temperature <= 0 or self.particle_initial_temperature <= 0:
            raise ValueError("temperatures must be positive (Kelvin)")

    @property
    def radius(self) -> float:
        """Drum internal radius (m)."""
        return self.drum_diameter / 2.0

    @property
    def particle_radius(self) -> float:
        """Particle radius (m)."""
        return self.particle_diameter / 2.0

    @property
    def froude(self) -> float:
        """Froude number for current configuration."""
        return froude_number(self.omega, self.radius)

    @property
    def regime(self) -> BedRegime:
        """Bed regime for current configuration."""
        return classify_regime(self.froude)

    @property
    def rpm(self) -> float:
        """Rotational speed in revolutions per minute."""
        return self.omega * 60.0 / (2.0 * math.pi)

    @property
    def inclination_deg(self) -> float:
        """Inclination angle in degrees."""
        return math.degrees(self.inclination)

    def particle_mass(self) -> float:
        """Mass of a single particle (kg)."""
        return self.particle_density * (4.0 / 3.0) * math.pi * self.particle_radius ** 3

    def total_particle_count(self) -> int:
        """Estimated total number of particles in the drum."""
        drum_volume = math.pi * self.radius ** 2 * self.drum_length
        solid_volume = drum_volume * self.fill_fraction
        particle_volume = (4.0 / 3.0) * math.pi * self.particle_radius ** 3
        return max(1, int(solid_volume / particle_volume))
