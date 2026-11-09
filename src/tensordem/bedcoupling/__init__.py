"""Unresolved (point-particle) DEM--CFD coupling for packed and fluidised beds.

Sub-modules
-----------
``closures``  Gidaspow blend (Ergun / Wen--Yu), buoyancy and pressure-gradient
              force, Schiller--Naumann terminal velocity, Richardson--Zaki
              correlations.
``packing``   random close packings on a periodic cell, kernel-based local
              voidage with exact sphere-sphere overlap volumes.
``bed``       bed-scale force assembly, fixed-bed pressure drop, static-balance
              audit, semi-coupled batch-settling dynamics, power-law fitting.
``study``     the IN143-N09 acceptance study (six machine-checkable criteria).

Nothing here contains a domain-specific ship or ice constant: the module is a
generic granular--fluid coupling layer for TensorDEM.
"""

from .closures import (
    BRANCHES,
    GIDASPOW_SWITCH,
    GRAVITY,
    ErgunDrop,
    buoyant_weight,
    drag_coefficient,
    drag_force,
    ergun_beta,
    ergun_pressure_gradient,
    gidaspow_branch_jump,
    hindered_slip_velocity,
    hydrostatic_pressure_gradient,
    pressure_gradient_force,
    rz_exponent,
    rz_slip_velocity,
    rz_superficial_velocity,
    sphere_projected_area,
    sphere_volume,
    terminal_velocity_sn,
    terminal_velocity_stokes,
    wen_yu_cd,
)
from .packing import (
    PackedBed,
    local_porosity,
    make_jittered_cubic_array,
    make_random_close_packing,
    porosity_statistics,
    sphere_intersection_volume,
)
from .bed import (
    FluidProperties,
    bed_force_assembly,
    bed_pressure_drop,
    body_force_per_particle,
    fit_power_law,
    relaxation_time,
    settling_sweep,
    simulate_settling,
    static_bed_force_audit,
)
from .study import run_acceptance_study, write_results

__all__ = [
    "GRAVITY",
    "GIDASPOW_SWITCH",
    "BRANCHES",
    "ErgunDrop",
    "sphere_volume",
    "sphere_projected_area",
    "ergun_pressure_gradient",
    "ergun_beta",
    "wen_yu_cd",
    "drag_coefficient",
    "drag_force",
    "gidaspow_branch_jump",
    "hydrostatic_pressure_gradient",
    "pressure_gradient_force",
    "buoyant_weight",
    "terminal_velocity_sn",
    "terminal_velocity_stokes",
    "hindered_slip_velocity",
    "rz_exponent",
    "rz_slip_velocity",
    "rz_superficial_velocity",
    "PackedBed",
    "make_random_close_packing",
    "make_jittered_cubic_array",
    "local_porosity",
    "porosity_statistics",
    "sphere_intersection_volume",
    "FluidProperties",
    "bed_force_assembly",
    "bed_pressure_drop",
    "body_force_per_particle",
    "static_bed_force_audit",
    "relaxation_time",
    "simulate_settling",
    "fit_power_law",
    "settling_sweep",
    "run_acceptance_study",
    "write_results",
]
