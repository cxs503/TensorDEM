"""FEM -> DEM fracture handoff for level ice (campaign p302w1-n03).

The chain in one paragraph: a level ice sheet on a Winkler foundation is solved
by a Kirchhoff plate FEM (:mod:`.plate_fem`); the flexural stress
``sigma = 6|M|/t**2`` reaching ``sigma_flexure`` declares fracture; a mode-I
lattice crack tracer turns the moment field into crack polylines and a floe
partition (:mod:`.crack_network`); the fracture state is written as
``failure_faces.json`` and re-read by the DEM side, which rebuilds the mass /
area / boundary ledger from the file alone before emitting body initial
conditions (:mod:`.handoff`); :mod:`.study` runs the production case, the closed
form verification, the mesh-convergence study and the sensitivity sweeps.

Physical scope and limits (repeated from the submodules because they matter):

* thin-plate (Kirchhoff) kinematics -- valid for ``a/t >> 1`` (the case is
  ``a/t = 20``); a thick or heavily loaded floe is outside the model;
* the crack tracer is a *frozen-field* pattern generator: no crack-tip stress
  redistribution, no dynamics, no cohesive zone; the floe census is therefore a
  geometric datum for the DEM, not a strength prediction;
* the crack paths are edge-aligned (staircase) so that the floe partition is
  exact; the azimuth error is O(cell size) and is quantified by the mesh study.
"""
from __future__ import annotations

from .plate_fem import (
    GRAVITY,
    WATER_DENSITY,
    PlateMaterial,
    PlateMesh,
    PlateSolution,
    PressurePatch,
    clamped_square_deflection,
    cylindrical_bending_deflection,
    cylindrical_bending_moment,
    hetenyi_point_load_deflection,
    navier_ssss_square_deflection,
    navier_ssss_square_moment,
    solve_plate,
)
from .crack_network import (
    Crack,
    CrackNetwork,
    CrackTracerConfig,
    Floe,
    SplitState,
    find_splitting_state,
    point_in_polygon,
    polygon_area,
    polygon_centroid,
    polygon_perimeter,
    trace_crack_network,
)
from .handoff import (
    SCHEMA,
    ConservationLedger,
    CrackRecord,
    DemInitialConditions,
    FailureFaces,
    FloeRecord,
    build_failure_faces,
    check_conservation,
    dem_initial_conditions,
    read_failure_faces,
)
from .study import (
    DEFAULT_PATCHES,
    FOUNDATION,
    IcePlateCase,
    fem_benchmark_table,
    mesh_convergence,
    run_handoff,
    sensitivity_sweep,
    solve_case_field,
    write_figures,
)

__all__ = [
    "GRAVITY",
    "WATER_DENSITY",
    "PlateMaterial",
    "PlateMesh",
    "PlateSolution",
    "PressurePatch",
    "solve_plate",
    "clamped_square_deflection",
    "cylindrical_bending_deflection",
    "cylindrical_bending_moment",
    "hetenyi_point_load_deflection",
    "navier_ssss_square_deflection",
    "navier_ssss_square_moment",
    "Crack",
    "CrackNetwork",
    "CrackTracerConfig",
    "Floe",
    "SplitState",
    "find_splitting_state",
    "trace_crack_network",
    "polygon_area",
    "polygon_centroid",
    "polygon_perimeter",
    "point_in_polygon",
    "SCHEMA",
    "ConservationLedger",
    "CrackRecord",
    "DemInitialConditions",
    "FailureFaces",
    "FloeRecord",
    "build_failure_faces",
    "check_conservation",
    "dem_initial_conditions",
    "read_failure_faces",
    "DEFAULT_PATCHES",
    "FOUNDATION",
    "IcePlateCase",
    "solve_case_field",
    "run_handoff",
    "fem_benchmark_table",
    "mesh_convergence",
    "sensitivity_sweep",
    "write_figures",
]