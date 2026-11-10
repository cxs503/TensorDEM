"""Vibration-induced granular convection and size/density segregation.

Node p223w2-n03 (TensorDEM): Brazil-nut effect (BNE) and reverse Brazil-nut
effect (RBNE) verification face.

What is here
------------
``bed``          : a fast, deterministic 2-D rigid-disk engine with *per-particle
                   density* (the existing ``tensordem.granular2d.engine`` engine
                   carries a single global density, so reverse Brazil-nut by
                   density contrast cannot be expressed with it).
``brazil_nut``   : the experiment drivers -- vertical shaking in the shaker frame,
                   rise/sink observables, segregation timescale, phase-diagram scan.
``anchors``      : literature anchors (with citation strings and uncertainties)
                   plus the error bookkeeping used by the acceptance report.

Physics conventions
-------------------
* Gravity acts along ``-y``; the free surface of the bed is ``+y``.
* The cell is shaked *in the shaker frame*: the walls are static and the vertical
  body acceleration is ``-g + A w^2 sin(w t)`` (mechanically equivalent to
  shaking the box in the lab frame, and it avoids moving-wall bookkeeping).
* ``Gamma = A w^2 / g`` is the dimensionless shaking intensity.
* Brazil-nut: large species rises (``dz = z_large - z_small > 0``).
  Reverse Brazil-nut: large species sinks (``dz < 0``).
"""
from .bed import BedConfig, BedSystem, build_bed, segregate_direction
from .brazil_nut import (
    PhasePoint,
    ShakeCase,
    rise_history,
    run_phase_scan,
    run_shake,
    segregation_timescale,
    timescale_from_history,
)
from .anchors import (
    BNE_ANCHORS,
    ONSET_ANCHOR,
    RBNE_BOUNDARY_ANCHOR,
    anchor_errors,
    interpolated_boundary,
    point_error,
)

__all__ = [
    "BedConfig",
    "BedSystem",
    "build_bed",
    "segregate_direction",
    "ShakeCase",
    "PhasePoint",
    "run_shake",
    "rise_history",
    "segregation_timescale",
    "timescale_from_history",
    "run_phase_scan",
    "BNE_ANCHORS",
    "ONSET_ANCHOR",
    "RBNE_BOUNDARY_ANCHOR",
    "anchor_errors",
    "point_error",
    "interpolated_boundary",
]