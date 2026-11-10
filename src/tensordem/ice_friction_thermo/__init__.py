"""Thermally self-consistent ice-structure friction: the phase-change contact.

Why this package exists
-----------------------
A DEM ice-structure contact normally takes one number, ``mu``, and keeps it: the
existing ice-resistance chain uses ``mu = 0.10`` and the calibrated law of
:mod:`tensordem.ice_steel_friction` is still a *prescribed* function of
``(v, T, p)``.  Neither knows that the contact is heating itself by its own
friction, that the melting point of ice **falls with pressure** by
Clausius-Clapeyron, or that the water produced when it does melt removes the
friction that produced it.

This package makes the coefficient a solved state of the contact:

:: 

    mu --(Coulomb)--> q = mu p_real v_slip          friction heat
     |                       |
     |                 1D conduction -> T_contact     thermal diffusion
     |                       |
     |            T_m(p_real) Clausius-Clapeyron       pressure melting
     |                       |
     |                film h -> melt fraction s        latent heat
     +--- mu = mu_ref exp(-beta (T - T_ref)) [1 - s (1 - r_wet)]

Sub-modules
-----------
``properties``          material constants, Blok effusivity partition
``clausius_clapeyron``  ``dT_m/dp = T dv/L``, ``T_m(p)``, pressure-melting
                        threshold, melt fraction, Stefan melt rate
``friction_law``        exponential ``mu(T)``, melting softening, closed-form fit
``conduction1d``        two-column implicit finite-volume conductor, affine in
                        the interface flux; Blok and steady-state references
``selfconsistent``      the Picard fixed point (+ an independent bisection
                        solver) that returns ``mu``, ``T_contact`` and ``Q_f``
``sliding_case``        the time-marched ice-slab-on-steel case with its energy
                        ledger
``validation``          acceptance evidence, literature holdout, SHA-256 digest

Typical use
-----------
>>> from tensordem.ice_friction_thermo import (
...     ContactConditions, ContactFrictionModel, solve_contact_state)
>>> conditions = ContactConditions(normal_force_n=1.0e5, nominal_area_m2=0.1,
...                                slip_velocity_m_s=1.0, bulk_temperature_c=-5.0)
>>> state = solve_contact_state(conditions, ContactFrictionModel())
>>> round(state.mu, 4) < ContactFrictionModel().mu_dry(-5.0)
True

Everything is SI.  Temperatures are degC (only differences and derivatives are
used), pressures Pa, lengths m.  See ``docs/P272W2_N11_ice_thermal_friction.md``
for the verification record and the acceptance numbers.
"""
from __future__ import annotations

from .clausius_clapeyron import (
    CC_SLOPE_K_PER_MPA,
    CC_SLOPE_K_PER_PA,
    cc_slope_k_per_pa,
    clapeyron_slope_check,
    film_latent_energy_j_m2,
    melt_fraction,
    melting_temperature_c,
    pressure_melting_threshold_pa,
    real_contact_pressure,
    stefan_melt_rate_m_per_s,
)
from .conduction1d import (
    TwoBodyContactConductor,
    blok_interface_rise_k,
    steady_state_resistance_m2_k_w,
)
from .friction_law import (
    BETA_RANGE_PER_K,
    MU_FLOOR,
    ContactFrictionModel,
    fit_exponential_temperature_law,
    mu_exponential,
)
from .properties import (
    ICE,
    ICE_DENSITY_KG_M3,
    LATENT_HEAT_J_KG,
    MELT_TEMPERATURE_C,
    MELT_TEMPERATURE_K,
    STEEL,
    WATER_DENSITY_KG_M3,
    PhaseProperties,
    blok_partition,
)
from .selfconsistent import (
    DEFAULT_TOLERANCE,
    MAX_ITERATIONS,
    ContactConditions,
    ContactThermoState,
    interface_gain_k_per_mu,
    solve_contact_state,
    solve_contact_state_bisection,
)
from .sliding_case import SlidingCase, SlidingResult, run_sliding_case
from .validation import (
    ERROR_GATE,
    PROVENANCE,
    TOLERANCE_GATE,
    acceptance_report,
    baseline_table,
    calibrate_from_barnes_sweep,
    canonical_digest,
    conductor_evidence,
    convergence_table,
    friction_heat_identity_check,
    predict_anchor,
    pressure_melting_map,
    sensitivity_table,
    sliding_case_evidence,
    temperature_monotonicity_check,
    validation_table,
    verify_evidence,
    write_evidence,
)

__all__ = [
    # clausius-clapeyron
    "CC_SLOPE_K_PER_MPA", "CC_SLOPE_K_PER_PA", "cc_slope_k_per_pa",
    "clapeyron_slope_check", "film_latent_energy_j_m2", "melt_fraction",
    "melting_temperature_c", "pressure_melting_threshold_pa",
    "real_contact_pressure", "stefan_melt_rate_m_per_s",
    # conduction
    "TwoBodyContactConductor", "blok_interface_rise_k",
    "steady_state_resistance_m2_k_w",
    # friction law
    "BETA_RANGE_PER_K", "MU_FLOOR", "ContactFrictionModel",
    "fit_exponential_temperature_law", "mu_exponential",
    # properties
    "ICE", "ICE_DENSITY_KG_M3", "LATENT_HEAT_J_KG", "MELT_TEMPERATURE_C",
    "MELT_TEMPERATURE_K", "STEEL", "WATER_DENSITY_KG_M3", "PhaseProperties", "blok_partition",
    # self-consistent solve
    "DEFAULT_TOLERANCE", "MAX_ITERATIONS", "ContactConditions",
    "ContactThermoState", "interface_gain_k_per_mu", "solve_contact_state",
    "solve_contact_state_bisection",
    # sliding case
    "SlidingCase", "SlidingResult", "run_sliding_case",
    # validation / evidence
    "ERROR_GATE", "PROVENANCE", "TOLERANCE_GATE", "acceptance_report",
    "baseline_table", "calibrate_from_barnes_sweep", "canonical_digest",
    "conductor_evidence", "convergence_table", "friction_heat_identity_check",
    "predict_anchor", "pressure_melting_map", "sensitivity_table",
    "sliding_case_evidence",
    "temperature_monotonicity_check", "validation_table", "verify_evidence",
    "write_evidence",
]