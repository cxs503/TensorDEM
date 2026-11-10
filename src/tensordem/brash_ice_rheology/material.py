"""Brash-ice material state: bulk density, submerged thickness, floe statistics.

The brash-ice layer of a channel is a three-phase mixture: solid ice, water in
the voids, and (above the waterline) air.  For the resistance model only the
*water-filled* state matters, so the bulk density is

.. math:: \\rho_b = (1-\\phi)\\,\\rho_{ice} + \\phi\\,\\rho_w ,

i.e. voids are water-filled, and the layer floats with the submerged thickness

.. math:: h_{sub} = H\\,\\rho_b/\\rho_w .

With ``rho_ice = 917`` and ``rho_w = 1025`` the bulk density runs
938.6 kg/m^3 (phi = 0.2) to 960.2 kg/m^3 (phi = 0.4) and the layer is 91.6-93.7 %
submerged: the buoyant density deficit ``rho_w - rho_b`` that sets the static
confinement is 65-86 kg/m^3, small compared with water - which is exactly why
the *dynamic* compaction term dominates the shear-band pressure at transit
speed (see :mod:`tensordem.brash_ice_rheology.model`).

Floe-size statistics are carried (mean diameter, lognormal spread) because they
enter the mu(I) law through the grain diameter ``d`` and the shear-band
thickness ``delta = 2 d``, not through a calibrated constant.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["BrashIceState", "STANDARD_BRASH"]

from .coefficients import (GRAVITY, ICE_SOLID_DENSITY, WATER_DENSITY)


@dataclass(frozen=True)
class BrashIceState:
    """State of the brash layer in a channel (SI units).

    Parameters
    ----------
    layer_thickness_m
        Equivalent consolidated brash thickness ``H`` (campaign range 0.5-2 m;
        standard case 1 m).
    floe_diameter_m
        Mean floe diameter ``d`` (campaign range 0.2-2 m; standard case 0.5 m).
    porosity
        Void fraction ``phi`` of the layer (campaign range 0.2-0.4; standard
        case 0.3).
    solid_density_kg_m3, water_density_kg_m3, gravity_m_s2
        Material constants; defaults are the graded ledger values.
    """

    layer_thickness_m: float = 1.0
    floe_diameter_m: float = 0.5
    porosity: float = 0.3
    solid_density_kg_m3: float = ICE_SOLID_DENSITY
    water_density_kg_m3: float = WATER_DENSITY
    gravity_m_s2: float = GRAVITY

    def __post_init__(self) -> None:
        if not self.layer_thickness_m > 0.0:
            raise ValueError("layer_thickness_m must be positive")
        if not self.floe_diameter_m > 0.0:
            raise ValueError("floe_diameter_m must be positive")
        if not 0.0 <= self.porosity < 1.0:
            raise ValueError("porosity must be in [0, 1)")
        if not self.solid_density_kg_m3 > 0.0:
            raise ValueError("solid_density_kg_m3 must be positive")
        if not 0.0 < self.solid_density_kg_m3 < self.water_density_kg_m3:
            raise ValueError("brash must be buoyant: solid density < water density")

    # -- derived quantities -------------------------------------------------
    @property
    def bulk_density_kg_m3(self) -> float:
        """Water-filled bulk density ``rho_b = (1-phi) rho_ice + phi rho_w``."""
        return ((1.0 - self.porosity) * self.solid_density_kg_m3
                + self.porosity * self.water_density_kg_m3)

    @property
    def density_deficit_kg_m3(self) -> float:
        """Buoyant density deficit ``rho_w - rho_b`` (> 0, the layer floats)."""
        return self.water_density_kg_m3 - self.bulk_density_kg_m3

    @property
    def submerged_thickness_m(self) -> float:
        """Submerged (draft) thickness of the floating brash layer."""
        return self.layer_thickness_m * self.bulk_density_kg_m3 / self.water_density_kg_m3

    @property
    def emerged_thickness_m(self) -> float:
        """Freeboard of the brash layer."""
        return self.layer_thickness_m - self.submerged_thickness_m

    @property
    def solid_fraction(self) -> float:
        """``1 - phi``."""
        return 1.0 - self.porosity

    @property
    def buoyant_unit_weight_n_m3(self) -> float:
        """Static buoyant confinement gradient ``(rho_w - rho_b) g``, N/m^3."""
        return self.density_deficit_kg_m3 * self.gravity_m_s2

    @property
    def static_confinement_pa(self) -> float:
        """Buoyant confinement at the layer base, ``(rho_w-rho_b) g h_sub``, Pa."""
        return self.buoyant_unit_weight_n_m3 * self.submerged_thickness_m

    @property
    def shear_band_thickness_m(self) -> float:
        """``delta = c_delta * d`` with ``c_delta`` from the ledger."""
        from .coefficients import check_coefficient
        c_delta = check_coefficient("shear_band_coefficient")[0]
        return c_delta * self.floe_diameter_m

    def as_dict(self) -> dict:
        """JSON-ready snapshot of every derived quantity used by the model."""
        return {
            "layer_thickness_m": self.layer_thickness_m,
            "floe_diameter_m": self.floe_diameter_m,
            "porosity": self.porosity,
            "solid_density_kg_m3": self.solid_density_kg_m3,
            "water_density_kg_m3": self.water_density_kg_m3,
            "gravity_m_s2": self.gravity_m_s2,
            "bulk_density_kg_m3": self.bulk_density_kg_m3,
            "density_deficit_kg_m3": self.density_deficit_kg_m3,
            "submerged_thickness_m": self.submerged_thickness_m,
            "emerged_thickness_m": self.emerged_thickness_m,
            "solid_fraction": self.solid_fraction,
            "static_confinement_pa": self.static_confinement_pa,
            "shear_band_thickness_m": self.shear_band_thickness_m,
        }

    def with_porosity(self, porosity: float) -> "BrashIceState":
        """Copy with a different porosity (campaign sensitivity sweep)."""
        return BrashIceState(
            layer_thickness_m=self.layer_thickness_m,
            floe_diameter_m=self.floe_diameter_m,
            porosity=porosity,
            solid_density_kg_m3=self.solid_density_kg_m3,
            water_density_kg_m3=self.water_density_kg_m3,
            gravity_m_s2=self.gravity_m_s2,
        )


#: Campaign standard case: H = 1 m, d = 0.5 m, phi = 0.3.
STANDARD_BRASH = BrashIceState(layer_thickness_m=1.0, floe_diameter_m=0.5, porosity=0.3)
