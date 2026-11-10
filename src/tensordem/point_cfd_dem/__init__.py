"""Point-particle (unresolved) two-way CFD-DEM coupling.

Distinct from :mod:`tensordem.lbmdem` (which resolves the particle surface with
an immersed boundary on a lattice-Boltzmann fluid): here the particles are
smaller than a fluid cell, the hydrodynamic interaction is carried by analytical
closures (Stokes / Schiller-Naumann drag, Richardson-Zaki hindrance,
pressure-gradient and added-mass forces) and the reaction is fed back to the
fluid as a momentum source.

Sub-modules
-----------
``closures``     analytical drag / terminal-velocity / RZ / added-mass relations
``fluid``        triply periodic pseudo-spectral incompressible solver with
                 adjoint particle-grid transfer operators
``particles``    Lagrangian point particles and the force budget
``coupling``     two-way driver, container-frame handling, momentum budgets
``verification`` five closed-form verification faces (see module docstring)
"""

from .closures import (
    added_mass_coefficient,
    added_mass_force,
    drag_correction,
    drag_force,
    particle_reynolds,
    relaxation_time_stokes,
    richardson_zaki_exponent,
    richardson_zaki_hindrance,
    schiller_naumann_correction,
    sphere_mass,
    sphere_volume,
    stokes_drag_coefficient,
    stokes_drag_force,
    terminal_velocity_stokes,
)
from .fluid import SpectralPeriodicFluid, trilinear_weights
from .particles import ParticleSet, PointParticleModel, evaluate_forces
from .coupling import PointParticleCFDDEM
from .verification import (
    added_mass_oscillation_case,
    momentum_exchange_case,
    pressure_gradient_case,
    richardson_zaki_case,
    run_all,
    stokes_terminal_velocity_case,
)

__all__ = [
    "ParticleSet",
    "PointParticleModel",
    "PointParticleCFDDEM",
    "SpectralPeriodicFluid",
    "trilinear_weights",
    "evaluate_forces",
    "terminal_velocity_stokes",
    "richardson_zaki_exponent",
    "richardson_zaki_hindrance",
    "stokes_drag_force",
    "drag_force",
    "particle_reynolds",
    "drag_correction",
    "schiller_naumann_correction",
    "added_mass_force",
    "added_mass_coefficient",
    "sphere_volume",
    "sphere_mass",
    "relaxation_time_stokes",
    "stokes_drag_coefficient",
    "stokes_terminal_velocity_case",
    "added_mass_oscillation_case",
    "momentum_exchange_case",
    "richardson_zaki_case",
    "pressure_gradient_case",
    "run_all",
]