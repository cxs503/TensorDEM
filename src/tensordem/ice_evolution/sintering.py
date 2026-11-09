"""Frenkel viscous sintering of ice-grain necks (campaign IN155-N09).

Physical model
--------------
Two ice grains of radius ``R`` that touch after the ship has broken the cover
form a neck.  In the viscous-flow regime the Frenkel solution for the neck
radius is

    a(t) = (3 gamma t / (2 eta))**(1/3)          (a << R)

with ``gamma`` the ice surface energy and ``eta`` the (strongly
temperature-dependent) viscosity of ice.  The neck cross-section carries
load, so the sintered contact is modelled in the DEM as a linear bond:

* normal stiffness   ``k_bond = E_ice pi a**2 / R`` (cylinder)
* failure force      ``F_bond = sigma_ice * pi a**2``

(the neck area times the uniaxial ice strength).  Both scale with the neck
radius, so the *growth law* of the bond is inherited directly from Frenkel.

References
----------
* Frenkel, J. (1948). Viscous sintering: ``a**2/R**2 ~ t**(2/3)``.
* Timco & Frederking (1996), *The mechanical properties of sea ice*,
  Cold Regions Science and Technology 24: 143-171 -- surface energy
  ``gamma ~ 0.033 J/m^2`` and strength anchors.
* Ice viscosity: Arrhenius with ``E_visc ~ 140 kJ/mol`` (Mellor & Cole,
  1980 order), anchored at ``eta(-10 C) ~ 1e13 Pa s``.

Units are SI (m, s, Pa, J, K).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ----------------------------------------------------------------- constants
GAS_CONSTANT = 8.314462618          # J/(mol K)
ICE_SURFACE_ENERGY = 0.033          # J/m^2, surface energy of ice near 0 C
ICE_VISCOSITY_REF = 1.0e13          # Pa s at ICE_VISCOSITY_T_REF
ICE_VISCOSITY_T_REF = 263.15        # K  (-10 C)
ICE_VISCOSITY_EA = 140.0e3          # J/mol, activation energy, viscous flow
ICE_YOUNG = 9.0e9                   # Pa, Young's modulus of ice
ICE_STRENGTH_REF = 1.5e6            # Pa, uniaxial compressive strength, -10 C
ICE_STRENGTH_T_REF = 263.15         # K


def ice_viscosity(temperature_k: float, *,
                  eta_ref: float = ICE_VISCOSITY_REF,
                  t_ref: float = ICE_VISCOSITY_T_REF,
                  ea: float = ICE_VISCOSITY_EA) -> float:
    """Arrhenius viscosity of ice, anchored at ``eta_ref`` at ``t_ref``."""
    if not math.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("temperature_k must be a positive finite value (K)")
    return eta_ref * math.exp(ea / GAS_CONSTANT * (1.0 / temperature_k - 1.0 / t_ref))


def ice_strength(temperature_k: float, *,
                 sigma_ref: float = ICE_STRENGTH_REF,
                 t_ref: float = ICE_STRENGTH_T_REF,
                 ea: float = 60.0e3) -> float:
    """Uniaxial compressive strength of solid ice, Arrhenius in temperature.

    Colder ice is stronger; the anchor is ``sigma_ref`` at ``t_ref`` with an
    activation energy of 60 kJ/mol (Timco & Frederking 1996 trend).
    """
    if not math.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("temperature_k must be a positive finite value (K)")
    return sigma_ref * math.exp(ea / GAS_CONSTANT * (1.0 / temperature_k - 1.0 / t_ref))


def frenkel_neck_radius(t_s: float, temperature_k: float, *,
                        surface_energy: float = ICE_SURFACE_ENERGY,
                        viscosity: float | None = None,
                        particle_radius: float | None = None) -> float:
    """Frenkel neck radius ``a(t) = (3 gamma t / (2 eta))**(1/3)``.

    ``viscosity`` may be supplied explicitly (unit tests); otherwise it is
    evaluated from ``temperature_k``.  The result is capped at
    ``particle_radius`` when given (a neck cannot exceed the grain radius).
    """
    if not math.isfinite(t_s) or t_s < 0.0:
        raise ValueError("t_s must be finite and >= 0")
    if viscosity is None:
        viscosity = ice_viscosity(temperature_k)
    if not math.isfinite(viscosity) or viscosity <= 0.0:
        raise ValueError("viscosity must be finite and positive")
    a = (3.0 * surface_energy * t_s / (2.0 * viscosity)) ** (1.0 / 3.0)
    if particle_radius is not None:
        a = min(a, particle_radius)
    return a


@dataclass(frozen=True)
class SinterState:
    """Sintered-bond state of one grain contact at time ``t_s``."""
    t_s: float
    neck_radius_m: float
    neck_ratio: float          # a / R
    bond_stiffness: float      # N/m
    bond_strength: float       # N
    surface_area_reduced_m2: float   # per contact, 2*pi*a**2 (two hemispheres)


def sinter_bond(t_s: float, particle_radius_m: float, temperature_k: float, *,
                 surface_energy: float = ICE_SURFACE_ENERGY,
                 viscosity: float | None = None,
                 young: float = ICE_YOUNG,
                 strength: float | None = None) -> SinterState:
    """Bond state of a grain contact sintering for ``t_s`` seconds.

    ``strength`` defaults to the temperature-dependent solid-ice strength.
    """
    if particle_radius_m <= 0.0:
        raise ValueError("particle_radius_m must be positive")
    if strength is None:
        strength = ice_strength(temperature_k)
    a = frenkel_neck_radius(t_s, temperature_k,
                            surface_energy=surface_energy,
                            viscosity=viscosity,
                            particle_radius=particle_radius_m)
    ratio = a / particle_radius_m
    # axial stiffness of a cylindrical neck (radius a, length ~ R)
    k_bond = young * math.pi * a * a / particle_radius_m
    f_bond = strength * math.pi * a * a
    # two hemispherical patches of area pi a^2 disappear from the free surface
    area_reduced = 2.0 * math.pi * a * a
    return SinterState(t_s=t_s, neck_radius_m=a, neck_ratio=ratio,
                        bond_stiffness=k_bond, bond_strength=f_bond,
                        surface_area_reduced_m2=area_reduced)


def neck_growth_exponent(t_samples, a_samples) -> tuple[float, float]:
    """Log-log least-squares fit ``a = C t**p``; returns ``(p, C)``.

    Used by the verification chain to check ``p`` against the Frenkel
    exponent ``1/3`` (criterion a).
    """
    t = [float(v) for v in t_samples]
    a = [float(v) for v in a_samples]
    if len(t) < 3 or any(v <= 0.0 for v in t) or any(v <= 0.0 for v in a):
        raise ValueError("need >= 3 positive (t, a) samples")
    lt = [math.log(v) for v in t]
    la = [math.log(v) for v in a]
    n = len(lt)
    mt = sum(lt) / n
    ma = sum(la) / n
    sxx = sum((x - mt) ** 2 for x in lt)
    sxy = sum((x - mt) * (y - ma) for x, y in zip(lt, la))
    p = sxy / sxx
    c = math.exp(ma - p * mt)
    return p, c
