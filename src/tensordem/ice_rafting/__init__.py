"""Ice floe rafting and ridging: buoyant rigid-floe DEM (node p223w2-n03).

The package provides

``mechanics``
    hydrostatics of a floating convex polygon, the closed-form rafting
    threshold, the flexural (elastic-foundation) length, a buoyant rigid-floe
    discrete-element kernel with a confined compression cell, and the two
    experiment drivers :func:`run_rafting_pair` (rafting threshold and
    load-displacement curve) and :func:`run_ridge` (ridge sail/keel geometry).
``anchors``
    the field-observation anchors (Keinonen et al. 1996 onset pressures;
    Timco & Burden 1994 ridge geometry) and the comparison bookkeeping.

The physics is discussed in ``docs/P223W2N03_ICE_RAFTING_BRAZIL_NUT.md``.
"""
from .mechanics import (
    FloeSpec,
    FloeWorld,
    IceProperties,
    analytic_rafting_threshold,
    clip_polygon_below,
    flexural_length,
    floe_polygon,
    rafting_resistance,
    ramp_factor,
    ridge_geometry,
    run_rafting_pair,
    run_rafting_ramp,
    run_ridge,
    submerged_properties,
)
from .anchors import (
    RAFTING_ANCHORS,
    RIDGE_GEOMETRY_ANCHORS,
    bracket_verdict,
    compare,
)

__all__ = [
    "IceProperties",
    "FloeSpec",
    "FloeWorld",
    "analytic_rafting_threshold",
    "flexural_length",
    "clip_polygon_below",
    "floe_polygon",
    "submerged_properties",
    "run_rafting_pair",
    "run_rafting_ramp",
    "run_ridge",
    "ridge_geometry",
    "rafting_resistance",
    "ramp_factor",
    "RAFTING_ANCHORS",
    "RIDGE_GEOMETRY_ANCHORS",
    "compare",
    "bracket_verdict",
]