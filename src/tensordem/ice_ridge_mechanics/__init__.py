"""Ice-ridge mechanics with a 2D rotating-disk DEM (TensorDEM).

Modules
-------
``engine2d``    rigid-disk kernel: contacts, freeze bonds, buoyancy, rigid
                segments and a closed work/dissipation ledger;
``geometry``    ridge sail/keel statistics and the Hopky (1983) empirical anchor;
``formation``   ridge built by squeezing two floating ice sheets;
``penetration`` ship crossing, failure-mode identification and energy split;
``strength``    confined-compression measurement of the rubble strength;
``verification``the F1-F7 campaign acceptance harness.

The capability is described in ``docs/ICE_RIDGE_MECHANICS_DEM.md``.
"""
from __future__ import annotations

from .engine2d import GRAVITY, Assembly, IceRubbleMaterial, WallSegment, bond_components
from .geometry import (
    envelope_area,
    union_area,
    hopky_keel_depth,
    hopky_sail_height,
    ridge_geometry,
    ridge_profile_geometry,
    waterline_width,
)
from .formation import (
    RidgeSqueezeConfig,
    build_rubble_field,
    simulate_ridge_squeeze,
    RidgeDepositionConfig,
    RidgeDepositionResult,
    RidgeFormationConfig,
    RidgeFormationResult,
    build_deposition,
    build_sheets,
    deposit_ridge,
    extract_ridge,
    simulate_ridge_formation,
)

from .penetration import (
    ShipCrossingConfig,
    ShipCrossingResult,
    build_hull,
    identify_modes,
    simulate_ship_crossing,
)
from .strength import RubbleStrengthConfig, measure_rubble_strength
from .verification import AcceptanceConfig, negative_control_zero_squeeze, run_acceptance

__all__ = [
    "GRAVITY",
    "Assembly",
    "IceRubbleMaterial",
    "WallSegment",
    "bond_components",
    "envelope_area",
    "union_area",
    "hopky_keel_depth",
    "hopky_sail_height",
    "ridge_geometry",
    "ridge_profile_geometry",
    "waterline_width",
    "ShipCrossingConfig",
    "ShipCrossingResult",
    "build_hull",
    "identify_modes",
    "simulate_ship_crossing",
    "RubbleStrengthConfig",
    "measure_rubble_strength",
    "AcceptanceConfig",
    "negative_control_zero_squeeze",
    "run_acceptance",
    "RidgeSqueezeConfig",
    "build_rubble_field",
    "simulate_ridge_squeeze",
    "RidgeDepositionConfig",
    "RidgeDepositionResult",
    "build_deposition",
    "deposit_ridge",
    "RidgeFormationConfig",
    "RidgeFormationResult",
    "build_sheets",
    "extract_ridge",
    "simulate_ridge_formation",
]