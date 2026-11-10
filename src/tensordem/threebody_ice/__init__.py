"""Three-body hull-floe-floe collision and broken-ice resistance decomposition.

Modules
-------
``model``       plan-view DEM core: kinematic hull outline, Cundall-Strack
                contacts with Coulomb friction, quadratic water drag, and a
                fully measured work/energy ledger
``threebody``   the tandem two-floe impact scenario, its analytic references,
                restitution calibration and the spacing sweep
``field``       broken-ice field transit and the three-component resistance
                decomposition with an auditable closure residual
``validation``  predeclared acceptance criteria (F1-F7), manifest and artifact

Entry points
------------
``python scripts/threebody_ice_validation.py --all``     full validation
``python scripts/threebody_ice_validation.py --shard 0/14``  one shard
``python scripts/threebody_ice_validation.py --merge``   criteria + figures
"""
from .model import (ContactLaw, FloeState, Ledger, MultiBodyModel,
                    PARAMETER_LEDGER, damping_ratio_from_restitution,
                    parameter_ledger_sha256, restitution_from_damping_ratio)
from .threebody import (ScenarioSpec, calibrate_law, elastic_reference,
                        measure_restitution, run_three_body,
                        single_collision_fractions, sweep_spacing)
from .field import (FieldSpec, build_field, concentration_sweep,
                    power_law_exponent, run_field, velocity_sweep)

__all__ = [
    "ContactLaw", "FloeState", "Ledger", "MultiBodyModel", "PARAMETER_LEDGER",
    "damping_ratio_from_restitution", "restitution_from_damping_ratio",
    "parameter_ledger_sha256",
    "ScenarioSpec", "calibrate_law", "elastic_reference", "measure_restitution",
    "run_three_body", "single_collision_fractions", "sweep_spacing",
    "FieldSpec", "build_field", "run_field", "velocity_sweep",
    "concentration_sweep", "power_law_exponent",
]