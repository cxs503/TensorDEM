"""Floe attrition chain: Wadell roundness -> Archard corner wear -> mu_eff(R) ->
multi-pass ice resistance decay (node p472w2-N05).

The module turns **rounding** into a DEM state variable with an exact geometry
(:mod:`~tensordem.floe_attrition.geometry`), a wear law that updates that
geometry (:mod:`~tensordem.floe_attrition.wear`), a friction closure anchored on
the m9375 disk shear cell (:mod:`~tensordem.floe_attrition.friction`), a design
of experiments that separates roundness from concentration
(:mod:`~tensordem.floe_attrition.design`) and the resulting multi-pass
resistance decay with the rounding/consolidation split closed to 100 %
(:mod:`~tensordem.floe_attrition.resistance`).  The acceptance criteria C1-C7
live in :mod:`~tensordem.floe_attrition.verification`.

Typical use::

    from tensordem.floe_attrition import (
        RoundedPolygon, ArchardWear, integrate_polygon_attrition, load_sphere_baseline)

    floe = RoundedPolygon(4, 0.6, 0.012)          # 0.6 m floe, 12 mm corner radius
    wear = ArchardWear(5e-3, 3e7)                  # ice-ice Archard pair
    hist = integrate_polygon_attrition(floe, wear, pressure=2e5,
                                       slip_per_step=2.5, n_steps=400)
    hist.summary()                                 # roundness 0.799 -> 0.871
"""
from __future__ import annotations

from .design import DesignCell, build_design_matrix, two_way_anova
from .friction import (DEFAULT_BASELINE, InterlockingFriction, SphereBaseline,
                       load_sphere_baseline, mu_effective_cube,
                       mu_effective_from_index, mu_effective_polygon)
from .geometry import (RoundedCube, RoundedPolygon, fillet_radius_for_roundness,
                       polygon_metrics_boundary_integral, rounded_polygon_vertices,
                       sphericity_3d, wadell_roundness_2d, wadell_roundness_3d)
from .resistance import TransitScenario, decay_shares, simulate_multi_pass
from .wear import (ArchardWear, AttritionHistory, integrate_cube_attrition,
                   integrate_polygon_attrition, wear_volume_closed_form)
from .verification import PRODUCTION, criteria

__all__ = [
    "RoundedPolygon", "RoundedCube", "fillet_radius_for_roundness",
    "rounded_polygon_vertices", "polygon_metrics_boundary_integral",
    "wadell_roundness_2d", "wadell_roundness_3d", "sphericity_3d",
    "ArchardWear", "AttritionHistory", "integrate_polygon_attrition",
    "integrate_cube_attrition", "wear_volume_closed_form",
    "SphereBaseline", "InterlockingFriction", "load_sphere_baseline",
    "DEFAULT_BASELINE", "mu_effective_from_index", "mu_effective_polygon",
    "mu_effective_cube",
    "DesignCell", "build_design_matrix", "two_way_anova",
    "TransitScenario", "simulate_multi_pass", "decay_shares",
    "PRODUCTION", "criteria",
]