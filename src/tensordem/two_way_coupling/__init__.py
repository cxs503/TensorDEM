"""Point-particle Lagrangian two-way coupling verification surface.

This package implements four verification surfaces for the DEM-CFD two-way
coupling in the dilute-to-dense regime:

1. **Stokes terminal velocity** (:mod:`.stokes_terminal`): single particle
   settling in quiescent fluid, compared to analytical Stokes and
   Schiller-Naumann terminal velocities.

2. **Added mass oscillation** (:mod:`.added_mass_oscillation`): particle
   response to oscillating flow, amplitude ratio and phase lag compared
   to the analytical Stokes-limit transfer function.

3. **Momentum exchange ledger** (:mod:`.momentum_ledger`): Newton's third
   law audit of the two-way coupling, tracking per-channel momentum
   exchange and verifying conservation to machine precision.

4. **Dense limit Ergun** (:mod:`.ergun_sweep`): volume fraction sweep from
   dilute (φ=1e-4) to dense packing (φ=0.45), comparing the DEM drag
   closure to the Ergun porous-media relation.

The combined acceptance study (:mod:`.study`) runs all four surfaces and
checks six machine-checkable criteria.

Relation to :mod:`tensordem.cfd_dem_dense`
-------------------------------------------
``cfd_dem_dense`` focuses on the *dense bed* verification (Richardson-Zaki,
minimum fluidization, grid convergence).  This package complements it with
the *Lagrangian point-particle* perspective: single-particle dynamics,
time-resolved oscillatory response, and explicit momentum conservation
accounting.  Both packages share the same force primitives from
``cfd_dem_dense.forces``.

Nothing here carries ship, ice or other domain constants.
"""

from .particle_ode import (
    ParticleState,
    FluidState,
    MomentumLedger,
    ParticleODE,
    integrate_particle,
)
from .stokes_terminal import (
    stokes_terminal_velocity,
    schiller_naumann_terminal_velocity,
    relaxation_time,
    run_stokes_verification,
)
from .settling_transient import (
    linear_settling_solution,
    fit_relaxation_time,
    euler_tau_bias_prediction,
    integrate_settling_exponential,
    transient_convergence_study,
)
from .added_mass_oscillation import (
    transfer_function,
    amplitude_ratio,
    phase_lag,
    run_oscillation_verification,
)
from .momentum_ledger import (
    single_particle_ledger,
    multi_particle_ledger,
    oscillating_flow_ledger,
    channel_toggle_test,
    run_momentum_ledger_verification,
)
from .ergun_sweep import (
    ergun_pressure_gradient_classic,
    ergun_pressure_gradient_model_a,
    dem_drag_per_volume,
    run_ergun_sweep,
)
from .study import (
    NODE_ID,
    THRESHOLDS,
    StudyOptions,
    run_acceptance_study,
    write_results,
    make_figures,
)

__all__ = [
    # particle_ode
    "ParticleState",
    "FluidState",
    "MomentumLedger",
    "ParticleODE",
    "integrate_particle",
    # stokes_terminal
    "stokes_terminal_velocity",
    "schiller_naumann_terminal_velocity",
    "relaxation_time",
    "run_stokes_verification",
    # settling_transient
    "linear_settling_solution",
    "fit_relaxation_time",
    "euler_tau_bias_prediction",
    "integrate_settling_exponential",
    "transient_convergence_study",
    # added_mass_oscillation
    "transfer_function",
    "amplitude_ratio",
    "phase_lag",
    "run_oscillation_verification",
    # momentum_ledger
    "single_particle_ledger",
    "multi_particle_ledger",
    "oscillating_flow_ledger",
    "channel_toggle_test",
    "run_momentum_ledger_verification",
    # ergun_sweep
    "ergun_pressure_gradient_classic",
    "ergun_pressure_gradient_model_a",
    "dem_drag_per_volume",
    "run_ergun_sweep",
    # study
    "NODE_ID",
    "THRESHOLDS",
    "StudyOptions",
    "run_acceptance_study",
    "write_results",
    "make_figures",
]
