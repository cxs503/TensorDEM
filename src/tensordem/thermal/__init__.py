"""Thermal physics for DEM granular ice / packed beds.

Modules
-------
``packed_bed``
    Static granular-bed effective thermal conductivity: Wiener bounds,
    Krupiczka / Zehner-Schluender correlations, and two independent numerical
    operators (contact-resistance network summation, volume-averaged flux).
``contact_chain``
    Hertz contact constriction resistance ``1/(4 k a_c)``, the 1D/2D particle
    chain networks with their closed-form series solution, the implicit-Euler
    transient, and the ice-bond thermal-mechanical gate.
``stefan_adapter``
    Adapter layer that couples the static-bed conductivity to the discrete
    Stefan melting front implemented in ``tensordem.threefield.thermal``.
``lattice_conductivity``
    Effective thermal conductivity of SC/BCC/FCC sphere lattices with Hertz
    contact thermal resistance; comparison with Zehner-Schluender correlation.
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
from .contact_chain import (
    BondGateResult,
    ChainCase,
    ThermalChain,
    TransientChain,
    analytic_chain_temperature,
    bond_failure_gate,
    bond_temperature_analytic,
    chain_effective_conductivity,
    chain_resistance,
    contact_conductance,
    contact_constriction_resistance,
    hertz_contact_radius,
    ice_bond_tensile_strength,
    lattice_network_temperatures,
    sphere_bulk_resistance,
    square_lattice_resistance,
)
from .stefan_adapter import (
    BedStefanState,
    melt_time_from_conductivity,
    melt_time_sensitivity,
)
from .lattice_conductivity import (
    LatticeType,
    LatticeThermalConductivity,
    lattice_porosity,
    lattice_coordination,
    lattice_constant,
    lattice_effective_conductivity,
    lattice_vs_zehner_schluender,
)

__all__ = [
    "BondGateResult",
    "ChainCase",
    "ThermalChain",
    "TransientChain",
    "analytic_chain_temperature",
    "bond_failure_gate",
    "bond_temperature_analytic",
    "chain_effective_conductivity",
    "chain_resistance",
    "contact_conductance",
    "contact_constriction_resistance",
    "hertz_contact_radius",
    "ice_bond_tensile_strength",
    "lattice_network_temperatures",
    "sphere_bulk_resistance",
    "square_lattice_resistance",
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
    "LatticeType",
    "LatticeThermalConductivity",
    "lattice_porosity",
    "lattice_coordination",
    "lattice_constant",
    "lattice_effective_conductivity",
    "lattice_vs_zehner_schluender",
]
# p74-n11 additions
from .packed_bed import (
    solve_contact_network_sor,
    isothermal_sphere_conductance,
    particle_fluid_effective_conductivity,
    critical_thickness_delta_crit,
    kozeny_carman_permeability,
    thermal_formation_factor,
    archie_cementation_exponent,
)

__all__.extend([
    "solve_contact_network_sor",
    "isothermal_sphere_conductance",
    "particle_fluid_effective_conductivity",
    "critical_thickness_delta_crit",
    "kozeny_carman_permeability",
    "thermal_formation_factor",
    "archie_cementation_exponent",
])
