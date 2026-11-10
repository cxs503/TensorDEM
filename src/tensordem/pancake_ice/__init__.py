"""Pancake ice mechanics: resistance, pile-up, and maneuverability.

This package provides a physics-based model for pancake ice resistance
during vessel navigation, including:

- Edge collision resistance (floe-hull impact)
- Slip friction resistance (floe-hull sliding)
- Pile-up resistance (ice accumulation ahead of vessel)
- Maneuverability forces (lateral force and yaw moment)
- Degradation limits (level ice and brash ice)

The model is validated against Antarctic observations (Dierking 2013)
and wave-ice interaction experiments (Herman et al. 2017).

Main entry points
-----------------
- :func:`pancake_resistance`: Total resistance with component breakdown
- :func:`pile_up_geometry`: Pile-up height and length estimation
- :func:`maneuverability_forces`: Lateral forces and yaw moment
- :func:`generate_field`: Random pancake ice field generation
- :func:`run_all_validations`: Validation against experimental data

Example
-------
>>> from tensordem.pancake_ice import (
...     PancakeIceConfig, HullGeometry, pancake_resistance
... )
>>> config = PancakeIceConfig(diameter=1.0, thickness=0.10, concentration=0.7)
>>> hull = HullGeometry(length=80.0, beam=16.0, draught=5.0)
>>> R = pancake_resistance(config, hull, speed=3.0)
>>> print(f"Total resistance: {R.R_total/1e3:.1f} kN")

References
----------
- Dierking, W. (2013). Antarctic pancake ice field observations.
- Herman, A., et al. (2017). Wave-ice interaction in pancake fields.
- Squire, V.A., et al. (2001). Marginal ice zone dynamics.
"""

from .model import (
    PancakeIceConfig,
    HullGeometry,
    ResistanceComponents,
    PileUpGeometry,
    ManeuverabilityForces,
    STANDARD_CASE,
    STANDARD_HULL,
    edge_collision_force,
    slip_friction_force,
    pile_up_resistance,
    pile_up_geometry,
    pancake_resistance,
    maneuverability_forces,
    degradation_level_ice,
    degradation_brash_ice,
    concentration_sweep,
    diameter_sweep,
)

from .field import (
    FloeProperties,
    PancakeIceField,
    generate_field,
    compute_field_statistics,
    simple_dem_step,
    estimate_pile_up_dem,
)

from .validation import (
    DierkingReference,
    HermanReference,
    ValidationResult,
    validate_resistance,
    validate_pile_up,
    validate_edge_scaling,
    validate_degradation_level,
    validate_degradation_brash,
    validate_negative_control,
    validate_monotonicity,
    run_all_validations,
)

__all__ = [
    # Model
    "PancakeIceConfig",
    "HullGeometry",
    "ResistanceComponents",
    "PileUpGeometry",
    "ManeuverabilityForces",
    "STANDARD_CASE",
    "STANDARD_HULL",
    "edge_collision_force",
    "slip_friction_force",
    "pile_up_resistance",
    "pile_up_geometry",
    "pancake_resistance",
    "maneuverability_forces",
    "degradation_level_ice",
    "degradation_brash_ice",
    "concentration_sweep",
    "diameter_sweep",
    # Field
    "FloeProperties",
    "PancakeIceField",
    "generate_field",
    "compute_field_statistics",
    "simple_dem_step",
    "estimate_pile_up_dem",
    # Validation
    "DierkingReference",
    "HermanReference",
    "ValidationResult",
    "validate_resistance",
    "validate_pile_up",
    "validate_edge_scaling",
    "validate_degradation_level",
    "validate_degradation_brash",
    "validate_negative_control",
    "validate_monotonicity",
    "run_all_validations",
]
