"""Thermophysical constants for the ice-structure friction heat problem.

Single source of truth for the numbers that the whole
:mod:`tensordem.ice_friction_thermo` package uses.  Nothing here is fitted;
every value is a textbook material property with its citation, so the derived
quantities (:attr:`PhaseProperties.alpha`, :attr:`PhaseProperties.effusivity`,
the Clausius-Clapeyron slope) are reproducible from the citation alone.

The interface effusivity ``e = sqrt(k rho c_p)`` is the quantity that decides how
the friction heat splits between ice and steel:
``gamma_ice = e_ice / (e_ice + e_steel)`` ~ 0.14 for ice on steel, i.e. **86 % of
the frictional heat goes into the steel**, so an ice-only thermal model
over-predicts the contact temperature by a factor ~7.  That partition is the
reason the coupled problem is not a single-material one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DEFAULT_FILM_REFERENCE_M",
    "DEFAULT_SPOT_RADIUS_M",
    "PhaseProperties",
    "ICE",
    "STEEL",
    "WATER_DENSITY_KG_M3",
    "ICE_DENSITY_KG_M3",
    "LATENT_HEAT_J_KG",
    "MELT_TEMPERATURE_C",
    "MELT_TEMPERATURE_K",
    "blok_partition",
]

#: Reference (0 degC, 1 atm) densities.  The water/ice density contrast is what
#: makes the Clausius-Clapeyron slope negative, so it is kept explicit.
ICE_DENSITY_KG_M3: float = 917.0
WATER_DENSITY_KG_M3: float = 1000.0
#: Latent heat of fusion of pure ice at 0 degC (Petrenko & Whitworth 1999).
LATENT_HEAT_J_KG: float = 3.34e5
MELT_TEMPERATURE_C: float = 0.0
MELT_TEMPERATURE_K: float = 273.15

#: Radius of one real (asperity) contact spot [m].  Sets the spot residence time
#: ``t_c = a/v`` and therefore both the flash penetration depth and the melt
#: film carried per pass.  A *geometric* input (surfaces, roughness), not a
#: fitted one; scanned in the sensitivity table (20 um .. 500 um).
DEFAULT_SPOT_RADIUS_M: float = 1.0e-4
#: Melt film thickness at which the interface counts as fully water-separated
#: [m].  The second phenomenological parameter of the model; the default sits in
#: the 0.2 .. 5 um band quoted for ice-steel lubricating films and the
#: sensitivity table reports what the choice costs.
DEFAULT_FILM_REFERENCE_M: float = 1.0e-6


@dataclass(frozen=True)
class PhaseProperties:
    """Isotropic thermophysical properties of one phase of the contact pair."""

    name: str
    k_w_m_k: float          # thermal conductivity [W/(m K)]
    rho_kg_m3: float        # density [kg/m^3]
    cp_j_kg_k: float        # specific heat [J/(kg K)]

    def __post_init__(self) -> None:
        for field_name in ("k_w_m_k", "rho_kg_m3", "cp_j_kg_k"):
            value = getattr(self, field_name)
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{self.name}.{field_name} must be finite and positive")
        if not str(self.name):
            raise ValueError("name must be a non-empty string")

    @property
    def alpha_m2_s(self) -> float:
        """Thermal diffusivity ``k / (rho c_p)`` [m^2/s]."""
        return self.k_w_m_k / (self.rho_kg_m3 * self.cp_j_kg_k)

    @property
    def effusivity_j_m2_k_sqrt_s(self) -> float:
        """Thermal effusivity ``sqrt(k rho c_p)`` [J/(m^2 K s^0.5)]."""
        return math.sqrt(self.k_w_m_k * self.rho_kg_m3 * self.cp_j_kg_k)

    @property
    def capacity_j_m3_k(self) -> float:
        """Volumetric heat capacity ``rho c_p`` [J/(m^3 K)]."""
        return self.rho_kg_m3 * self.cp_j_kg_k


#: Ice at -5 degC (Petrenko & Whitworth 1999; k and c_p vary by ~5 % over
#: -30..0 degC, taken constant here and flagged in the report's limitations).
ICE = PhaseProperties("ice", 2.2, ICE_DENSITY_KG_M3, 2100.0)
#: Structural steel (typical ship hull steel, 0.2 % C).
STEEL = PhaseProperties("steel", 45.0, 7850.0, 470.0)


def blok_partition(ice: PhaseProperties = ICE, steel: PhaseProperties = STEEL) -> float:
    """Fraction of the frictional flux that enters the ice (Blok partition).

    Two semi-infinite bodies sharing a plane heat source at a common interface
    temperature carry fluxes in the ratio of their effusivities, so the ice
    receives ``gamma = e_i / (e_i + e_s)``.
    """
    e_ice = ice.effusivity_j_m2_k_sqrt_s
    e_steel = steel.effusivity_j_m2_k_sqrt_s
    return e_ice / (e_ice + e_steel)