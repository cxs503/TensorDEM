"""Dense unresolved (point-particle) DEM--CFD coupling closures.

Sub-modules
-----------
``forces``  Inter-phase closures as switchable channels: Schiller--Naumann,
            Wen--Yu voidage correction, Di Felice, Ergun/Gidaspow dense branch,
            pressure-gradient (Froude--Krylov) force and Maxey--Riley added
            mass.  Also the action--reaction audit used by criterion 1.
``column``  A periodic bed element of particles coupled to a quasi-steady
            fluid: dynamic inversion of the Richardson--Zaki relation, minimum
            fluidisation force balance, fixed/graded bed pressure drop, grid
            convergence, mass advection and the energy ledger.
``study``   The seven machine-checkable acceptance criteria of the campaign
            node ``p50w2-arb7250-n03-dense-dem-cfd`` plus the five figures.

Relation to :mod:`tensordem.bedcoupling`
----------------------------------------
``bedcoupling`` (node IN143-N09) owns a Gidaspow/Ergun packing-level study with
kernel voidage and a settling batch.  This package is the *dense closure*
verification surface: it adds the Di Felice branch, the Maxey--Riley added-mass
force, switchable pressure-gradient force and -- crucially -- criteria that
compare against analytic anchors with pre-declared tolerances (Richardson--Zaki,
Ergun, ``u_mf``, grid order, conservation identities, three negative controls).

Nothing here contains ship, ice or other domain constants.
"""

from .forces import (
    DEFAULT_DENSE,
    DRAG_MODELS,
    GRAVITY,
    GIDASPOW_SWITCH,
    ForceModel,
    added_mass_force,
    beta,
    buoyant_weight,
    di_felice_chi,
    drag_density,
    equilibrium_superficial_velocity,
    ergun_gradient,
    ergun_minimum_fluidization_velocity,
    fluid_pressure_gradient,
    linear_deposit_weights,
    momentum_exchange_audit,
    newton_cd,
    per_particle_drag,
    pressure_gradient_force,
    schiller_naumann_cd,
    sphere_projected_area,
    sphere_volume,
    terminal_velocity,
    voidage_exponent,
)
from .column import (
    ColumnConfig,
    dynamic_equilibrium_velocity,
    fit_richardson_zaki,
    fixed_bed_pressure_drop,
    fluid_state,
    graded_bed_pressure_drop,
    grid_convergence_study,
    jittered_lattice_positions,
    minimum_fluidization_study,
    particle_force_channels,
    relaxation_energy_ledger,
    relax_column,
    richardson_zaki_sweep,
    rz_correlation_exponent,
    solid_mass_advection_test,
    steady_particle_velocity,
)
from .study import (
    NODE_ID,
    THRESHOLDS,
    StudyOptions,
    make_figures,
    run_acceptance_study,
    write_results,
)

__all__ = [
    "NODE_ID",
    "THRESHOLDS",
    "StudyOptions",
    "DEFAULT_DENSE",
    "DRAG_MODELS",
    "GRAVITY",
    "GIDASPOW_SWITCH",
    "ForceModel",
    "ColumnConfig",
    "added_mass_force",
    "beta",
    "buoyant_weight",
    "di_felice_chi",
    "drag_density",
    "dynamic_equilibrium_velocity",
    "equilibrium_superficial_velocity",
    "ergun_gradient",
    "ergun_minimum_fluidization_velocity",
    "fit_richardson_zaki",
    "fixed_bed_pressure_drop",
    "fluid_pressure_gradient",
    "fluid_state",
    "graded_bed_pressure_drop",
    "grid_convergence_study",
    "jittered_lattice_positions",
    "linear_deposit_weights",
    "make_figures",
    "minimum_fluidization_study",
    "momentum_exchange_audit",
    "newton_cd",
    "particle_force_channels",
    "per_particle_drag",
    "pressure_gradient_force",
    "relax_column",
    "relaxation_energy_ledger",
    "richardson_zaki_sweep",
    "run_acceptance_study",
    "rz_correlation_exponent",
    "schiller_naumann_cd",
    "solid_mass_advection_test",
    "sphere_projected_area",
    "sphere_volume",
    "steady_particle_velocity",
    "terminal_velocity",
    "voidage_exponent",
    "write_results",
]