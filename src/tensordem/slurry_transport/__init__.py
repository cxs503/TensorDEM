"""Dredged-slurry pipeline transport: suspended particle -> deposition velocity
-> concentration/pressure drop -> pump operating point -> production rate.

Chain of closures (each face independently testable)
----------------------------------------------------
``closures``            single-particle force balance: Schiller-Naumann drag,
                        numerical terminal velocity, regime-wise closed forms,
                        Rubey cross-check, Richardson-Zaki hindrance
``critical_velocity``   Durand (F_L chart table) vs Cleanaway-type deposition
                        velocity, sqrt(D) scaling audit
``pressure_drop``       laminar/Colebrook friction (implicit solve + Swamee-Jain
                        and Haaland cross-checks), water gradient, derived
                        suspension excess gradient, Durand-form reduction
``pump_pipe``           system head, pump head curve, operating point, shaft
                        power identity, NPSH margin, production rate
``pipe_dem``            L/D = 5 periodic slab DEM-CFD: concentration profile,
                        particle velocity profile, measured excess gradient
``verification``        the seven acceptance faces A1..A7 of campaign node
                        p246w2-n03, returning machine-checkable numbers
"""

from .closures import (
    G,
    MU_W,
    NU_W,
    RHO_W,
    archimedes_number,
    cd_schiller_naumann,
    cd_stokes,
    drag_force,
    hindrance_richardson_zaki,
    particle_reynolds,
    richardson_zaki_exponent,
    sphere_area,
    sphere_mass,
    sphere_volume,
    submerged_weight,
    terminal_velocity_numeric,
    terminal_velocity_regime,
    terminal_velocity_rubey,
    terminal_velocity_stokes,
)
from .critical_velocity import (
    CLEANAWAY_TABLE,
    DURAND_FL_TABLE,
    cleanaway_deposition_velocity,
    deposition_envelope_deviation,
    deposition_velocity_pair,
    deposition_velocity_scaling_exponent,
    dimensional_audit,
    durand_deposition_velocity,
    durand_fl,
    set_durand_fl_table,
)
from .pressure_drop import (
    colebrook_friction,
    darcy_head_loss,
    einstein_viscosity,
    excess_gradient_durand_form,
    excess_gradient_suspension,
    fit_durand_form,
    friction_factor,
    friction_factor_laminar,
    haaland_friction,
    mixture_gradient,
    moody_reference_points,
    pressure_drop,
    swamee_jain_friction,
    water_gradient,
)
from .pump_pipe import (
    PipeSpec,
    PumpSpec,
    SlurrySpec,
    npsh_available,
    npsh_margin,
    npsh_required,
    operating_point,
    production_rate,
    pump_efficiency,
    pump_head,
    shaft_power,
    system_head,
)
from .pipe_dem import PipeSectionDEMCFD, dem_transport_case, skewness

__all__ = [
    "G", "RHO_W", "MU_W", "NU_W",
    "sphere_volume", "sphere_area", "sphere_mass", "submerged_weight",
    "particle_reynolds", "cd_stokes", "cd_schiller_naumann", "drag_force",
    "archimedes_number", "terminal_velocity_numeric",
    "terminal_velocity_regime", "terminal_velocity_stokes",
    "terminal_velocity_rubey", "richardson_zaki_exponent",
    "hindrance_richardson_zaki",
    "DURAND_FL_TABLE", "CLEANAWAY_TABLE", "durand_fl",
    "durand_deposition_velocity", "cleanaway_deposition_velocity",
    "deposition_velocity_pair", "deposition_velocity_scaling_exponent",
    "dimensional_audit", "set_durand_fl_table",
    "deposition_envelope_deviation",
    "friction_factor_laminar", "colebrook_friction", "swamee_jain_friction",
    "haaland_friction", "friction_factor", "moody_reference_points",
    "darcy_head_loss", "water_gradient", "einstein_viscosity",
    "excess_gradient_suspension", "mixture_gradient", "fit_durand_form",
    "excess_gradient_durand_form", "pressure_drop",
    "PipeSpec", "PumpSpec", "SlurrySpec", "system_head", "pump_head",
    "pump_efficiency", "operating_point", "shaft_power", "production_rate",
    "npsh_available", "npsh_required", "npsh_margin",
    "PipeSectionDEMCFD", "dem_transport_case", "skewness",
]