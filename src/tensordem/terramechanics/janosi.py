"""Janosi shear–traction curve for soft sediment terramechanics.

Physics
-------
The Janosi equation relates the shear stress ``tau`` at the soil–track interface
to the shear displacement ``j``:

    tau = tau_max * (1 - exp(-j / K))

where:
- ``tau_max`` : maximum shear stress (Pa), given by Mohr–Coulomb:
  ``tau_max = c + p * tan(phi)``
- ``c``       : soil cohesion (Pa)
- ``p``       : normal pressure (Pa)
- ``phi``     : internal friction angle (rad)
- ``K``       : shear deformation modulus (m), typically 0.01–0.06 m
- ``j``       : shear displacement (m)

The shear displacement for a tracked vehicle is related to the slip ratio ``s``
and the position ``x`` along the track contact length ``L``:

    j(x) = s * x    (for driving, s > 0)
    j(x) = s * (L - x)    (for braking, s < 0)

The total tractive force is obtained by integrating ``tau`` over the contact
area.

References
----------
- Janosi, Z. & Hanamoto, M. (1961) "The analytical determination of drawbar
  pull as a function of slip for tracked vehicles in a deformable medium."
- Wong, J.Y. (2001) *Theory of Ground Vehicles*, 3rd ed., Wiley, §2.3.

Units are SI (m, Pa, rad, -).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "JanosiModel",
    "janosi_shear_stress",
    "janosi_max_shear",
    "mohr_coulomb_tau_max",
]


def _finite(name: str, value: float, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def mohr_coulomb_tau_max(
    cohesion: float, pressure: float, friction_angle_rad: float
) -> float:
    """Mohr–Coulomb maximum shear stress ``tau_max = c + p * tan(phi)`` (Pa).

    Parameters
    ----------
    cohesion : float
        Soil cohesion ``c`` (Pa), must be >= 0.
    pressure : float
        Normal pressure ``p`` (Pa), must be >= 0.
    friction_angle_rad : float
        Internal friction angle ``phi`` (rad), must be in [0, pi/2).

    Returns
    -------
    float
        Maximum shear stress ``tau_max`` (Pa).
    """
    c = _finite("cohesion", cohesion, nonnegative=True)
    p = _finite("pressure", pressure, nonnegative=True)
    phi = _finite("friction_angle_rad", friction_angle_rad)
    if phi < 0.0 or phi >= math.pi / 2.0:
        raise ValueError("friction_angle_rad must be in [0, pi/2)")
    return c + p * math.tan(phi)


def janosi_shear_stress(
    shear_displacement: float, tau_max: float, K: float
) -> float:
    """Janosi shear stress ``tau = tau_max * (1 - exp(-j/K))`` (Pa).

    Parameters
    ----------
    shear_displacement : float
        Shear displacement ``j`` (m), must be >= 0.
    tau_max : float
        Maximum shear stress (Pa), must be >= 0.
    K : float
        Shear deformation modulus (m), must be > 0.

    Returns
    -------
    float
        Shear stress ``tau`` (Pa).
    """
    j = _finite("shear_displacement", shear_displacement, nonnegative=True)
    tmax = _finite("tau_max", tau_max, nonnegative=True)
    kk = _finite("K", K, positive=True)
    return tmax * (1.0 - math.exp(-j / kk))


def janosi_max_shear(
    cohesion: float, pressure: float, friction_angle_rad: float,
    shear_displacement: float, K: float
) -> float:
    """Combined Mohr–Coulomb + Janosi: ``tau(j)`` in one call (Pa)."""
    tmax = mohr_coulomb_tau_max(cohesion, pressure, friction_angle_rad)
    return janosi_shear_stress(shear_displacement, tmax, K)


@dataclass(frozen=True)
class JanosiModel:
    """Complete Janosi shear–traction model.

    Typical parameter ranges (Wong 2001):

    ============ ============ ============ ============
    Soil         c (Pa)       phi (deg)    K (m)
    ============ ============ ============ ============
    Clay         5000–25000   5–15         0.01–0.03
    Sandy loam   2000–10000   20–35        0.02–0.04
    Dry sand     0–2000       28–40        0.02–0.06
    ============ ============ ============ ============
    """

    cohesion: float = 0.0           # Pa
    friction_angle_deg: float = 30.0  # degrees
    K: float = 0.03                 # m (shear deformation modulus)

    def __post_init__(self) -> None:
        _finite("cohesion", self.cohesion, nonnegative=True)
        _finite("friction_angle_deg", self.friction_angle_deg)
        if not 0.0 <= self.friction_angle_deg < 90.0:
            raise ValueError("friction_angle_deg must be in [0, 90)")
        _finite("K", self.K, positive=True)

    @property
    def friction_angle_rad(self) -> float:
        return math.radians(self.friction_angle_deg)

    def tau_max(self, pressure: float) -> float:
        """Maximum shear stress at given normal pressure (Pa)."""
        return mohr_coulomb_tau_max(
            self.cohesion, pressure, self.friction_angle_rad
        )

    def shear_stress(self, shear_displacement: float, pressure: float) -> float:
        """Shear stress at given displacement and pressure (Pa)."""
        tmax = self.tau_max(pressure)
        return janosi_shear_stress(shear_displacement, tmax, self.K)

    def shear_curve(
        self, pressure: float, j_max: float = 0.2, n_points: int = 200
    ) -> dict[str, np.ndarray]:
        """Full shear stress vs displacement curve.

        Returns dict with keys 'j' (m) and 'tau' (Pa).
        """
        j = np.linspace(0.0, j_max, n_points)
        tmax = self.tau_max(pressure)
        tau = tmax * (1.0 - np.exp(-j / self.K))
        return {"j": j, "tau": tau}

    @classmethod
    def clay(cls) -> "JanosiModel":
        """Typical clay parameters."""
        return cls(cohesion=15000.0, friction_angle_deg=10.0, K=0.02)

    @classmethod
    def sandy_loam(cls) -> "JanosiModel":
        """Typical sandy loam parameters."""
        return cls(cohesion=5000.0, friction_angle_deg=28.0, K=0.03)

    @classmethod
    def dry_sand(cls) -> "JanosiModel":
        """Typical dry sand parameters."""
        return cls(cohesion=0.0, friction_angle_deg=35.0, K=0.04)
