"""Bekker pressure–sinkage relation for soft sediment terramechanics.

Physics
-------
The Bekker equation relates the normal pressure ``p`` under a loaded plate or
track to the sinkage ``z``:

    p = (k_c / b + k_phi) * z^n

where:
- ``k_c`` : cohesive modulus of soil deformation (N/m^{n+1})
- ``k_phi`` : frictional modulus of soil deformation (N/m^{n+2})
- ``b``     : plate width or track width (m)
- ``n``     : exponent of soil deformation (dimensionless, typically 0.5–1.8)
- ``z``     : sinkage (m)

The term ``k_c / b`` captures the cohesive contribution (dominant for clay),
while ``k_phi`` captures the frictional contribution (dominant for sand).
The exponent ``n`` controls the nonlinearity: ``n = 1`` gives a linear spring,
``n < 1`` gives a stiffening response, ``n > 1`` gives a softening response.

The stiffness (tangent modulus) is:

    dp/dz = n * (k_c / b + k_phi) * z^{n-1}

and the inverse (sinkage from pressure):

    z = [p / (k_c / b + k_phi)]^{1/n}

References
----------
- Bekker, M.G. (1956) *Theory of Land Locomotion*, Univ. of Michigan Press.
- Wong, J.Y. (2001) *Theory of Ground Vehicles*, 3rd ed., Wiley, §2.2.

Units are SI (m, Pa, N, -).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "BekkerModel",
    "bekker_pressure",
    "bekker_sinkage_from_pressure",
    "bekker_stiffness",
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


def bekker_pressure(
    sinkage: float, width: float, k_c: float, k_phi: float, n: float
) -> float:
    """Bekker pressure ``p = (k_c/b + k_phi) * z^n`` (Pa).

    Parameters
    ----------
    sinkage : float
        Vertical sinkage ``z`` (m), must be >= 0.
    width : float
        Plate or track width ``b`` (m), must be > 0.
    k_c : float
        Cohesive modulus (N/m^{n+1}), must be >= 0.
    k_phi : float
        Frictional modulus (N/m^{n+2}), must be >= 0.
    n : float
        Deformation exponent, must be > 0.

    Returns
    -------
    float
        Normal pressure ``p`` (Pa).
    """
    z = _finite("sinkage", sinkage, nonnegative=True)
    b = _finite("width", width, positive=True)
    kc = _finite("k_c", k_c, nonnegative=True)
    kph = _finite("k_phi", k_phi, nonnegative=True)
    nn = _finite("n", n, positive=True)
    if kc == 0.0 and kph == 0.0:
        raise ValueError("at least one of k_c or k_phi must be positive")
    stiffness = kc / b + kph
    return stiffness * z ** nn


def bekker_sinkage_from_pressure(
    pressure: float, width: float, k_c: float, k_phi: float, n: float
) -> float:
    """Inverse Bekker: sinkage from pressure ``z = [p / (k_c/b + k_phi)]^{1/n}``.

    Parameters
    ----------
    pressure : float
        Normal pressure ``p`` (Pa), must be >= 0.
    width, k_c, k_phi, n : float
        As in :func:`bekker_pressure`.

    Returns
    -------
    float
        Sinkage ``z`` (m).
    """
    p = _finite("pressure", pressure, nonnegative=True)
    b = _finite("width", width, positive=True)
    kc = _finite("k_c", k_c, nonnegative=True)
    kph = _finite("k_phi", k_phi, nonnegative=True)
    nn = _finite("n", n, positive=True)
    if kc == 0.0 and kph == 0.0:
        raise ValueError("at least one of k_c or k_phi must be positive")
    stiffness = kc / b + kph
    if p == 0.0:
        return 0.0
    return (p / stiffness) ** (1.0 / nn)


def bekker_stiffness(
    sinkage: float, width: float, k_c: float, k_phi: float, n: float
) -> float:
    """Tangent stiffness ``dp/dz = n * (k_c/b + k_phi) * z^{n-1}`` (Pa/m).

    At ``z = 0`` and ``n > 1`` the stiffness is zero; at ``z = 0`` and
    ``n < 1`` it diverges (returned as ``inf``).
    """
    z = _finite("sinkage", sinkage, nonnegative=True)
    b = _finite("width", width, positive=True)
    kc = _finite("k_c", k_c, nonnegative=True)
    kph = _finite("k_phi", k_phi, nonnegative=True)
    nn = _finite("n", n, positive=True)
    if kc == 0.0 and kph == 0.0:
        raise ValueError("at least one of k_c or k_phi must be positive")
    stiffness_base = kc / b + kph
    if z == 0.0:
        if nn < 1.0:
            return float("inf")
        elif nn == 1.0:
            return nn * stiffness_base
        else:
            return 0.0
    return nn * stiffness_base * z ** (nn - 1.0)


@dataclass(frozen=True)
class BekkerModel:
    """Complete Bekker pressure–sinkage model with typical soil parameters.

    Typical parameter ranges (Wong 2001, Table 2.1):

    ============ ============ ============= ============ ===========
    Soil         k_c (N/m^{n+1})  k_phi (N/m^{n+2})  n
    ============ ============ ============= ============ ===========
    Clay         100–600      0–5000         0.5–1.0
    Sandy loam   200–1000     5000–20000     0.6–1.1
    Dry sand     0–500        10000–50000    0.5–1.0
    ============ ============ ============= ============ ===========
    """

    k_c: float = 0.0        # N/m^{n+1}
    k_phi: float = 15000.0  # N/m^{n+2}
    n: float = 1.1
    width: float = 0.3      # m (default track width)

    def __post_init__(self) -> None:
        _finite("k_c", self.k_c, nonnegative=True)
        _finite("k_phi", self.k_phi, nonnegative=True)
        _finite("n", self.n, positive=True)
        _finite("width", self.width, positive=True)
        if self.k_c == 0.0 and self.k_phi == 0.0:
            raise ValueError("at least one of k_c or k_phi must be positive")

    def pressure(self, sinkage: float) -> float:
        """Normal pressure at given sinkage (Pa)."""
        return bekker_pressure(sinkage, self.width, self.k_c, self.k_phi, self.n)

    def sinkage(self, pressure: float) -> float:
        """Sinkage at given normal pressure (m)."""
        return bekker_sinkage_from_pressure(
            pressure, self.width, self.k_c, self.k_phi, self.n
        )

    def stiffness(self, sinkage: float) -> float:
        """Tangent stiffness dp/dz at given sinkage (Pa/m)."""
        return bekker_stiffness(sinkage, self.width, self.k_c, self.k_phi, self.n)

    @classmethod
    def clay(cls, width: float = 0.3) -> "BekkerModel":
        """Typical clay parameters."""
        return cls(k_c=400.0, k_phi=2000.0, n=0.7, width=width)

    @classmethod
    def sandy_loam(cls, width: float = 0.3) -> "BekkerModel":
        """Typical sandy loam parameters."""
        return cls(k_c=500.0, k_phi=12000.0, n=0.8, width=width)

    @classmethod
    def dry_sand(cls, width: float = 0.3) -> "BekkerModel":
        """Typical dry sand parameters."""
        return cls(k_c=0.0, k_phi=20000.0, n=0.6, width=width)
