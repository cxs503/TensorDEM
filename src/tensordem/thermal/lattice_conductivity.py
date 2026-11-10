"""Lattice effective thermal conductivity for SC/BCC/FCC sphere arrangements.

Physics
-------
For a periodic lattice of identical spheres with Hertz contact thermal
resistance between neighbours, the effective thermal conductivity can be
computed exactly from the lattice geometry and the contact conductance.

The effective conductivity of a lattice is:

    k_eff = (Z / 6) * (a_c / R) * k_s * f_contact

where:
- ``Z`` : coordination number (SC=6, BCC=8, FCC=12)
- ``a_c`` : Hertz contact radius
- ``R`` : sphere radius
- ``k_s`` : solid thermal conductivity
- ``f_contact`` : contact efficiency factor (accounts for constriction)

More precisely, for a lattice with lattice constant ``a`` and sphere radius
``R``, the effective conductivity along a principal direction is:

    k_eff = (1/V_cell) * sum_contacts G_c * (dx)^2

where ``G_c = 4 k_s a_c`` is the contact conductance and ``dx`` is the
projection of the contact vector onto the heat flow direction.

For the three cubic lattices:
- SC:  Z=6,  a=2R,    phi = 1 - pi/6 ≈ 0.4764
- BCC: Z=8,  a=4R/sqrt(3), phi = 1 - pi*sqrt(3)/8 ≈ 0.3198
- FCC: Z=12, a=2*sqrt(2)*R, phi = 1 - pi/(3*sqrt(2)) ≈ 0.2595

The Zehner-Schlünder model provides a correlation for packed beds that can
be compared against these lattice results.

References
----------
- Zehner, P. & Schlünder, E.U. (1970) "Wärmeleitfähigkeit von Schüttungen
  bei mäßigen Temperaturen." Chemie Ingenieur Technik 42(16), 933–941.
- Kunii, D. & Smith, J.M. (1960) "Heat transfer characteristics of porous
  rocks." AIChE Journal 6(1), 71–78.

Units are SI (m, W/m/K, -).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = [
    "LatticeType",
    "LatticeThermalConductivity",
    "lattice_porosity",
    "lattice_coordination",
    "lattice_constant",
    "lattice_effective_conductivity",
    "lattice_vs_zehner_schluender",
]


LatticeType = Literal["SC", "BCC", "FCC"]


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


def lattice_porosity(lattice_type: LatticeType) -> float:
    """Porosity of a perfect sphere lattice.

    SC:  phi = 1 - pi/6 ≈ 0.4764
    BCC: phi = 1 - pi*sqrt(3)/8 ≈ 0.3198
    FCC: phi = 1 - pi/(3*sqrt(2)) ≈ 0.2595
    """
    if lattice_type == "SC":
        return 1.0 - math.pi / 6.0
    elif lattice_type == "BCC":
        return 1.0 - math.pi * math.sqrt(3.0) / 8.0
    elif lattice_type == "FCC":
        return 1.0 - math.pi / (3.0 * math.sqrt(2.0))
    else:
        raise ValueError(f"unknown lattice type: {lattice_type}")


def lattice_coordination(lattice_type: LatticeType) -> int:
    """Coordination number of a perfect sphere lattice.

    SC=6, BCC=8, FCC=12.
    """
    return {"SC": 6, "BCC": 8, "FCC": 12}[lattice_type]


def lattice_constant(lattice_type: LatticeType, radius: float) -> float:
    """Lattice constant (unit cell edge length) for touching spheres.

    SC:  a = 2R
    BCC: a = 4R/sqrt(3)  (body diagonal = 4R)
    FCC: a = 2*sqrt(2)*R  (face diagonal = 4R)
    """
    r = _finite("radius", radius, positive=True)
    if lattice_type == "SC":
        return 2.0 * r
    elif lattice_type == "BCC":
        return 4.0 * r / math.sqrt(3.0)
    elif lattice_type == "FCC":
        return 2.0 * math.sqrt(2.0) * r
    else:
        raise ValueError(f"unknown lattice type: {lattice_type}")


def _hertz_contact_radius_from_overlap(radius: float, overlap: float) -> float:
    """Hertz contact radius from geometric overlap: a_c = sqrt(R_eff * delta)."""
    r = _finite("radius", radius, positive=True)
    d = _finite("overlap", overlap, positive=True)
    return math.sqrt(0.5 * r * d)


def lattice_effective_conductivity(
    lattice_type: LatticeType,
    radius: float,
    contact_radius: float,
    k_solid: float,
    k_fluid: float = 0.0,
) -> float:
    """Effective thermal conductivity of a perfect sphere lattice.

    Uses the contact network approach:

        k_eff = (1/V_cell) * sum_contacts G_c * (dx)^2

    where G_c = 4 * k_solid * a_c is the contact conductance and dx is the
    projection of the contact vector onto the heat flow direction (z-axis).

    For isotropic lattices (SC, BCC, FCC), the result is the same in all
    principal directions.

    Parameters
    ----------
    lattice_type : str
        'SC', 'BCC', or 'FCC'.
    radius : float
        Sphere radius (m).
    contact_radius : float
        Hertz contact radius a_c (m).
    k_solid : float
        Solid thermal conductivity (W/m/K).
    k_fluid : float
        Fluid (pore) thermal conductivity (W/m/K). Default 0 (vacuum).

    Returns
    -------
    float
        Effective thermal conductivity (W/m/K).
    """
    r = _finite("radius", radius, positive=True)
    a_c = _finite("contact_radius", contact_radius, positive=True)
    ks = _finite("k_solid", k_solid, positive=True)
    kf = _finite("k_fluid", k_fluid, nonnegative=True)

    Z = lattice_coordination(lattice_type)
    a = lattice_constant(lattice_type, r)
    V_cell = a ** 3

    # Contact conductance
    G_c = 4.0 * ks * a_c

    # Sum of (dx)^2 over all contacts in the unit cell
    # For isotropic cubic lattices: sum(dx^2) = Z * a^2 / 3
    # (each contact contributes a^2/3 on average due to isotropy)
    sum_dx2 = Z * a * a / 3.0

    # Solid contribution
    k_eff_solid = G_c * sum_dx2 / V_cell

    # Fluid contribution (parallel path through pores)
    phi = lattice_porosity(lattice_type)
    k_eff_fluid = kf * phi

    return k_eff_solid + k_eff_fluid


@dataclass(frozen=True)
class LatticeThermalConductivity:
    """Complete lattice thermal conductivity calculation.

    Parameters
    ----------
    lattice_type : str
        'SC', 'BCC', or 'FCC'.
    radius : float
        Sphere radius (m).
    overlap : float
        Geometric overlap between touching spheres (m).
    k_solid : float
        Solid thermal conductivity (W/m/K).
    k_fluid : float
        Fluid thermal conductivity (W/m/K).
    youngs_modulus : float
        Young's modulus for Hertz contact (Pa).
    poisson_ratio : float
        Poisson's ratio.
    normal_force : float
        Normal force per contact (N).
    """

    lattice_type: LatticeType
    radius: float
    overlap: float
    k_solid: float
    k_fluid: float = 0.0
    youngs_modulus: float = 9.0e9
    poisson_ratio: float = 0.33
    normal_force: float = 10.0

    def __post_init__(self) -> None:
        _finite("radius", self.radius, positive=True)
        _finite("overlap", self.overlap, positive=True)
        _finite("k_solid", self.k_solid, positive=True)
        _finite("k_fluid", self.k_fluid, nonnegative=True)
        _finite("youngs_modulus", self.youngs_modulus, positive=True)
        _finite("poisson_ratio", self.poisson_ratio)
        _finite("normal_force", self.normal_force, positive=True)

    @property
    def contact_radius(self) -> float:
        """Hertz contact radius from normal force."""
        r = self.radius
        force = self.normal_force
        e_star = self.youngs_modulus / (2.0 * (1.0 - self.poisson_ratio ** 2))
        r_eff = 0.5 * r
        return (3.0 * force * r_eff / (4.0 * e_star)) ** (1.0 / 3.0)

    @property
    def porosity(self) -> float:
        return lattice_porosity(self.lattice_type)

    @property
    def coordination(self) -> int:
        return lattice_coordination(self.lattice_type)

    @property
    def effective_conductivity(self) -> float:
        return lattice_effective_conductivity(
            self.lattice_type, self.radius, self.contact_radius,
            self.k_solid, self.k_fluid,
        )

    def zehner_schluender_conductivity(self) -> float:
        """Zehner-Schlünder correlation for comparison."""
        from tensordem.thermal.packed_bed import zehner_schluender_conductivity
        return zehner_schluender_conductivity(
            self.porosity, self.k_solid, self.k_fluid if self.k_fluid > 0 else 0.026
        )


def lattice_vs_zehner_schluender(
    lattice_type: LatticeType,
    radius: float,
    contact_radius: float,
    k_solid: float,
    k_fluid: float = 0.026,
) -> dict[str, float]:
    """Compare lattice effective conductivity with Zehner-Schlünder.

    Returns dict with keys: k_eff_lattice, k_eff_zs, relative_error, porosity.
    """
    k_lat = lattice_effective_conductivity(
        lattice_type, radius, contact_radius, k_solid, k_fluid
    )
    phi = lattice_porosity(lattice_type)

    # ZS requires k_fluid > 0
    kf = max(k_fluid, 0.026)
    from tensordem.thermal.packed_bed import zehner_schluender_conductivity
    k_zs = zehner_schluender_conductivity(phi, k_solid, kf)

    rel_error = abs(k_lat - k_zs) / k_zs if k_zs > 0 else float("inf")

    return {
        "k_eff_lattice": k_lat,
        "k_eff_zs": k_zs,
        "relative_error": rel_error,
        "porosity": phi,
        "lattice_type": lattice_type,
    }
