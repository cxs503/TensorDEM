"""Lagrangian (point-particle) DEM-fluid bed coupling: minimum provable unit.

Module scope
------------
This package is the *Lagrangian* implementation of the packed/fluidised bed
drag closure in TensorDEM:

* :mod:`~tensordem.lagrangian_bed.closures` -- Gidaspow (Ergun dense branch +
  Wen-Yu dilute branch) exchange coefficient, Schiller-Naumann drag, buoyancy,
  pressure-gradient force, Richardson-Zaki exponent correlation and local
  voidage estimators;
* :mod:`~tensordem.lagrangian_bed.mechanics` -- soft-sphere DEM contacts,
  gravity and time integration (deterministic, seeded packing);
* :mod:`~tensordem.lagrangian_bed.column` -- 1-D fluid column momentum balance
  (steady semi-coupled, fluid-at-rest one-way, unsteady 0-D) with a momentum
  flux ledger;
* :mod:`~tensordem.lagrangian_bed.bed` -- packing/deposition, porosity
  statistics, static force audit, hindered-settling probe, batch settling and
  the GCI (Roache) estimator;
* :mod:`~tensordem.lagrangian_bed.study` -- the six-criterion acceptance study
  of node IN145-N05.

Relationship to the other TensorDEM closures
--------------------------------------------
``tensordem.bedcoupling`` (node IN143-N09) audits the *analytic* closures
directly; ``tensordem.cfd_dem_dense`` (node P50W2-N03) carries the closure on an
Eulerian column.  This package applies the same closure to *individual
Lagrangian particles* that carry contact mechanics, and adds the momentum-flux
ledger and a GCI time-step certificate; the acceptance study cross-checks the
three implementations so that the shared closure has a single source of truth.
"""
from . import bed, closures, column, mechanics, study
from .bed import (
    BedConfig,
    compact,
    deposit,
    gci,
    group_settling,
    hindered_settling_probe,
    porosity,
    porosity_statistics,
    single_particle_settling,
    static_force_audit,
    uniform_suspension_settling,
)
from .study import StudyOptions, run_acceptance_study, write_results
from .closures import (
    GIDASPOW_SWITCH,
    GRAVITY,
    buoyancy_force,
    cell_particle_velocity,
    cell_voidage,
    ergun_beta,
    fit_rz_exponent,
    gidaspow_beta,
    gidaspow_branch_jump,
    local_voidage,
    particle_drag_force,
    pressure_gradient_force,
    rz_exponent,
    rz_hindrance_beta,
    rz_hindrance_drag,
    rz_slip_velocity,
    schiller_naumann_cd,
    schiller_naumann_drag,
    submerged_weight,
    terminal_velocity,
    wen_yu_beta,
)
from .column import (
    ColumnProfile,
    FluidProperties,
    at_rest_pressure,
    column_state,
    momentum_ledger,
    steady_flow,
)
from .mechanics import Box, ContactParameters, ParticleState, make_particles, step

__all__ = [
    "GRAVITY",
    "GIDASPOW_SWITCH",
    "BedConfig",
    "Box",
    "ColumnProfile",
    "ContactParameters",
    "FluidProperties",
    "ParticleState",
    "at_rest_pressure",
    "bed",
    "buoyancy_force",
    "cell_particle_velocity",
    "cell_voidage",
    "closures",
    "column",
    "column_state",
    "compact",
    "deposit",
    "ergun_beta",
    "fit_rz_exponent",
    "gci",
    "gidaspow_beta",
    "gidaspow_branch_jump",
    "group_settling",
    "hindered_settling_probe",
    "local_voidage",
    "make_particles",
    "mechanics",
    "momentum_ledger",
    "particle_drag_force",
    "porosity",
    "porosity_statistics",
    "pressure_gradient_force",
    "rz_exponent",
    "rz_hindrance_beta",
    "rz_hindrance_drag",
    "rz_slip_velocity",
    "schiller_naumann_cd",
    "schiller_naumann_drag",
    "single_particle_settling",
    "static_force_audit",
    "study",
    "uniform_suspension_settling",
    "steady_flow",
    "step",
    "submerged_weight",
    "terminal_velocity",
    "wen_yu_beta",
    "StudyOptions",
    "run_acceptance_study",
    "write_results",
]