"""3D ice ridge–ship interaction: geometry, rubble generation, DEM contact with
interlocking, ridge failure load, and hull FEM response.

Modules
-------
``geometry``      parametric ridge sail/keel geometry (3D volume)
``rubble``        rubble particle generation with porosity & power-law PSD
``contact``       3D DEM contact mechanics (Hertz + Coulomb + rolling + interlock)
``interaction``   ridge–ship interaction simulation driver
``fem_response``  hull panel FEM response to ridge load
``verification``  analytical anchors and acceptance harness

Capability described in ``docs/ice_ridge_3d_interaction.md``.
"""
from __future__ import annotations

from .geometry import (
    RidgeGeometry,
    ridge_volume,
    keel_half_width,
    sail_half_width,
)
from .rubble import (
    PowerLawPSD3D,
    generate_rubble_particles,
    particle_count_closed_form,
    measure_porosity,
)
from .contact import (
    hertz_normal_force_3d,
    coulomb_friction_3d,
    rolling_resistance_3d,
    interlocking_shear_strength,
)
from .interaction import (
    RidgeShipInteraction,
    RidgeShipConfig,
    simulate_ridge_ship,
)
from .fem_response import (
    plate_stress_simply_supported,
    plate_deflection_simply_supported,
    hull_panel_response,
)
from .verification import (
    run_acceptance,
    AcceptanceConfig,
    AcceptanceResult,
)

__all__ = [
    "RidgeGeometry",
    "ridge_volume",
    "keel_half_width",
    "sail_half_width",
    "PowerLawPSD3D",
    "generate_rubble_particles",
    "particle_count_closed_form",
    "measure_porosity",
    "hertz_normal_force_3d",
    "coulomb_friction_3d",
    "rolling_resistance_3d",
    "interlocking_shear_strength",
    "RidgeShipInteraction",
    "RidgeShipConfig",
    "simulate_ridge_ship",
    "plate_stress_simply_supported",
    "plate_deflection_simply_supported",
    "hull_panel_response",
    "run_acceptance",
    "AcceptanceConfig",
    "AcceptanceResult",
]
