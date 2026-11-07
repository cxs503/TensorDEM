"""Thermal physics for DEM granular ice / packed beds.

Modules
-------
``packed_bed``
    Static granular-bed effective thermal conductivity: Wiener bounds,
    Krupiczka / Zehner-Schluender correlations, and two independent numerical
    operators (contact-resistance network summation, volume-averaged flux).
``stefan_adapter``
    Adapter layer that couples the static-bed conductivity to the discrete
    Stefan melting front implemented in ``tensordem.threefield.thermal``.
"""
from __future__ import annotations

from .packed_bed import (
    CONDUCTIVITY_RATIO_SCAN,
    POROSITY_SCAN,
    ContactModel,
    StaticPackedBed,
    ZehnerSchluenderConstants,
    conductivity_contact_network,
    conductivity_flux_average,
    contact_radius,
    effective_conductivity,
    generate_static_bed,
    hertz_contact_radius,
    krupiczka_conductivity,
    rescale_porosity,
    wiener_bounds,
    zehner_schluender_conductivity,
)
from .stefan_adapter import (
    BedStefanState,
    melt_time_from_conductivity,
    melt_time_sensitivity,
)

__all__ = [
    "CONDUCTIVITY_RATIO_SCAN",
    "POROSITY_SCAN",
    "ContactModel",
    "StaticPackedBed",
    "ZehnerSchluenderConstants",
    "BedStefanState",
    "conductivity_contact_network",
    "conductivity_flux_average",
    "contact_radius",
    "effective_conductivity",
    "generate_static_bed",
    "hertz_contact_radius",
    "krupiczka_conductivity",
    "rescale_porosity",
    "melt_time_from_conductivity",
    "melt_time_sensitivity",
    "wiener_bounds",
    "zehner_schluender_conductivity",
]