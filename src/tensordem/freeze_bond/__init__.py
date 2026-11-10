"""Freeze bonds and ice arches: from a single bond to a self-sustaining arch.

The package has four layers, each with its own machine-checkable face:

``constitutive``
    :class:`FreezeBondLaw` -- the closed-form strength law
    ``sigma_fb = A dT^alpha (1 + beta S)^-1 t^gamma`` with the shear ratio and the
    fracture energy, in SI.
``cohesive``
    :class:`CohesiveBondSet` -- the vectorised mixed-mode cohesive bond element:
    damage history variable, linear softening that spends exactly ``G_f``,
    incremental work/dissipation ledger, and re-freezing (a reversible strength
    history that keeps the dissipated energy).
``assembly``
    bonded disk lattices (channel of ice rubble between frictional walls;
    confined biaxial cell), Cundall-Strack contacts and walls, internal forces.
``solver``
    quasi-static displacement-controlled loading with a closed energy ledger.
``force_network`` / ``arch``
    strong-force loops and the arch observables (cyclomatic number, spanning
    loops, load transmission to the downstream load cell).
``experiments``
    the campaign drivers: single-bond self-consistency, multi-bond energy
    closure, biaxial and channel tests as a function of ``kappa = sigma_fb /
    sigma_bulk``, the ``kappa_c`` detection and the self-sustain protocol.

``thermal_profile``
    1-D enthalpy-method freezing of the bond layer (implicit finite volume,
    phase change with the full latent heat, energy closure ~1e-12) with
    sub-cell freezing-front tracking anchored on the Stefan/Neumann solutions,
    the Cox & Weeks / Timco-Frederking saline reduction and the
    ``tau_eff``-versus-``tau_uniform`` (m7439) deviation caliper
    (P316-W1-N03, ``tests/test_thermal_profile.py``).

Evidence grades are stated per module.  The freeze-bond *law* is a declared
hypothesis (self-consistent, not physically validated); the energy ledger and the
DEM bookkeeping are verified by ``tests/test_freeze_bond_*.py``.
"""
from __future__ import annotations

from .constitutive import FREEZE_BOND_LAW_REFERENCE, FreezeBondLaw
from .cohesive import BondEnvironment, CohesiveBondSet
from .assembly import (
    BondTopology,
    BondedSpecimen,
    FrictionState,
    SpecimenConfig,
    Wall,
    build_box_specimen,
    build_channel_specimen,
    bond_separations,
    commit_friction,
    contact_pairs,
    internal_forces,
)
from .solver import LoadingResult, StepRecord, load_steps, newton_solve, tangent_matrix
from .force_network import ForceNetwork, analyze_network, strong_force_subgraph

from .strength import (
    ALPHA_DEFAULT,
    BETA_DEFAULT,
    BETA_ICE_DEFAULT,
    FREEZE_BOND_EVIDENCE,
    GAMMA_DEFAULT,
    SASTRY_METZGER_1986_DATA,
    SIGMA_0_DEFAULT,
    SIGMA_ICE_0_DEFAULT,
    T_FREEZE_FRESH,
    T_FREEZE_SEA,
    T_GAMMA_DEFAULT,
    T_MIN_C,
    T_REF_DEFAULT,
    T_REF_SCALE,
    FreezeBondConfig,
    FreezeBondResult,
    failure_mode,
    freeze_bond_strength,
    peak_load_jump,
    sastry_metzger_error,
    sastry_metzger_prediction,
    sigma_bond_temperature,
    sigma_bond_time,
    sigma_ice_temperature,
    transition_temperature,
)

# contact-level frozen bonds (note: the contact-level peak_load_jump lives in
# .contact and is intentionally NOT re-exported here, because the strength
# layer already exports a different peak_load_jump(config, ...))
from .contact import (
    ContactBondConfig,
    ContactBondState,
    FrozenBondAssembly,
    FrozenContactBond,
    REPETTO_ADHESION_WINDOW_MPA,
    bond_strengths,
    break_event,
    energy_ledger,
    failure_mode_at,
    freezing_front_ledger,
    leave_one_out_validation,
    mechanical_energy_ledger,
    mode_transition_scan,
    multi_contact_slide,
    peak_contact_force,
    sastry_metzger_contact_table,
    single_contact_pulloff,
    spearman_rho,
    transition_temperature_analytic,
    transition_temperature_numeric,
)

__all__ = [
    "FREEZE_BOND_LAW_REFERENCE",
    "FreezeBondLaw",
    "BondEnvironment",
    "CohesiveBondSet",
    "BondTopology",
    "BondedSpecimen",
    "SpecimenConfig",
    "FrictionState",
    "Wall",
    "build_box_specimen",
    "build_channel_specimen",
    "bond_separations",
    "commit_friction",
    "contact_pairs",
    "internal_forces",
    "LoadingResult",
    "StepRecord",
    "load_steps",
    "newton_solve",
    "tangent_matrix",
    "ForceNetwork",
    "analyze_network",
    "strong_force_subgraph",
    "ALPHA_DEFAULT",
    "BETA_DEFAULT",
    "BETA_ICE_DEFAULT",
    "FREEZE_BOND_EVIDENCE",
    "GAMMA_DEFAULT",
    "SASTRY_METZGER_1986_DATA",
    "SIGMA_0_DEFAULT",
    "SIGMA_ICE_0_DEFAULT",
    "T_FREEZE_FRESH",
    "T_FREEZE_SEA",
    "T_GAMMA_DEFAULT",
    "T_MIN_C",
    "T_REF_DEFAULT",
    "T_REF_SCALE",
    "FreezeBondConfig",
    "FreezeBondResult",
    "failure_mode",
    "freeze_bond_strength",
    "peak_load_jump",
    "sastry_metzger_error",
    "sastry_metzger_prediction",
    "sigma_bond_temperature",
    "sigma_bond_time",
    "sigma_ice_temperature",
    "transition_temperature",
    "ContactBondConfig",
    "ContactBondState",
    "FrozenBondAssembly",
    "FrozenContactBond",
    "REPETTO_ADHESION_WINDOW_MPA",
    "bond_strengths",
    "break_event",
    "energy_ledger",
    "failure_mode_at",
    "freezing_front_ledger",
    "leave_one_out_validation",
    "mechanical_energy_ledger",
    "mode_transition_scan",
    "multi_contact_slide",
    "peak_contact_force",
    "sastry_metzger_contact_table",
    "single_contact_pulloff",
    "spearman_rho",
    "transition_temperature_analytic",
    "transition_temperature_numeric",
]

# Failure-mode transition + laboratory anchor validation (node P46W3-N07)
from .mode_transition import (  # noqa: E402
    ACCEPTANCE,
    BondInterfaceParams,
    ForceDisplacement,
    bond_area_fraction,
    bond_strength,
    bond_strength_temperature,
    failure_mode,
    failure_mode_map,
    failure_stress,
    fiber_bundle_pull_off,
    ice_tensile_strength,
    load_anchor_set,
    melt_ramp,
    peak_force,
    peak_jump_ratio,
    peak_slope_kink,
    pull_off_curve,
    r_squared,
    run_validation_report,
    traction_separation,
    transition_temperature,
    validate_against_anchor,
)
