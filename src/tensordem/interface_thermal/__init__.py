"""Ice-structure contact interface: phase-change thermo-mechanical coupling.

Sub-modules
-----------
``conduction``   Fourier conduction: semi-infinite references, FVM transient
                 conductor, contact thermal resistance, DEM particle network
``stefan``       Stefan condition: Neumann similarity solution and a fixed-grid
                 enthalpy solver for the melt-layer thickness
``friction_heat``frictional flux, Blok partition, Jaeger moving band source
``lubrication``  melt-fraction friction law ``mu(T)`` and film stiffness
``coupling``     the two-way DEM <-> FVM coupled marching solver

Everything is SI internally and temperatures are degC (only differences and
derivatives are used).  See ``docs/interface_thermal_contact_coupling.md`` for
the verification record and ``benchmarks/interface_thermal_coupling.py`` for the
driver that regenerates ``artifacts/interface_thermal_coupling.json``.
"""

from .conduction import (
    Conduction1D,
    Particle,
    ParticleThermalNetwork,
    contact_heat_transfer_coefficient,
    contact_thermal_resistance,
    maxwell_constriction_resistance,
    semi_infinite_surface_temperature,
    semi_infinite_temperature_profile,
    thermal_diffusivity,
)
from .coupling import CoupledInterface, IceSteelContact, hertz_contact_radius
from .friction_heat import (
    blok_partition_coefficient,
    friction_heat_flux,
    friction_power,
    jaeger_band_max_temperature,
    jaeger_band_surface_temperature,
    jaeger_high_speed_limit,
    jaeger_low_speed_limit,
    moving_band_fvm_solution,
)
from .lubrication import (
    MeltLubricationModel,
    film_normal_stiffness,
    load_bearing_area_fraction,
    series_normal_stiffness,
    stefan_melt_rate,
)
from .stefan import (
    EnthalpyStefan1D,
    neumann_interface_position,
    neumann_interface_velocity,
    neumann_lambda,
    neumann_stefan_number,
    neumann_temperature,
)

__all__ = [
    "Conduction1D", "Particle", "ParticleThermalNetwork",
    "contact_heat_transfer_coefficient", "contact_thermal_resistance",
    "maxwell_constriction_resistance", "semi_infinite_surface_temperature",
    "semi_infinite_temperature_profile", "thermal_diffusivity",
    "CoupledInterface", "IceSteelContact", "hertz_contact_radius",
    "blok_partition_coefficient", "friction_heat_flux", "friction_power",
    "jaeger_band_max_temperature", "jaeger_band_surface_temperature",
    "jaeger_high_speed_limit", "jaeger_low_speed_limit",
    "moving_band_fvm_solution",
    "MeltLubricationModel", "film_normal_stiffness", "load_bearing_area_fraction",
    "series_normal_stiffness", "stefan_melt_rate",
    "EnthalpyStefan1D", "neumann_interface_position", "neumann_interface_velocity",
    "neumann_lambda", "neumann_stefan_number", "neumann_temperature",
]