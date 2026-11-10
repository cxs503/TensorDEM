"""3-D DEM contact-force -> macroscopic ice-resistance mapping closure.

Public surface:

* :mod:`tensordem.resistance_closure.mechanisms` -- engine-agnostic
  crushing/bending/submersion decomposition with a measured closure residual.
* :mod:`tensordem.resistance_closure.dem` -- capped, frictional 3-D wedge-bow DEM
  (plastic crushing cap, Coulomb hull friction, buoyancy/gravity, bond-graph
  detachment) and the per-case driver.
* :mod:`tensordem.resistance_closure.reference` -- Lindqvist three-component
  reference reduced to the same per-unit-width basis as the DEM.
* :mod:`tensordem.resistance_closure.campaign` -- thickness sweep, particle-size
  refinement, negative control and the machine-checkable verdicts.
"""
from .mechanisms import (
    BENDING,
    CRUSHING,
    MECHANISMS,
    SUBMERSION,
    HullContactSet,
    MechanismDecomposition,
    average_decompositions,
    decompose,
    flexural_demand_ratio,
    flexural_moment_capacity,
    plate_characteristic_length,
    unit_heading,
)
from .reference import (
    LindqvistReference,
    build_reference,
    compare_components,
)
from .dem import CappedBowDEM, ClosureCase, build_simulation, run_case

__all__ = [
    "BENDING",
    "CRUSHING",
    "MECHANISMS",
    "SUBMERSION",
    "CappedBowDEM",
    "ClosureCase",
    "HullContactSet",
    "LindqvistReference",
    "MechanismDecomposition",
    "average_decompositions",
    "build_reference",
    "build_simulation",
    "compare_components",
    "decompose",
    "flexural_demand_ratio",
    "flexural_moment_capacity",
    "plate_characteristic_length",
    "run_case",
    "unit_heading",
]