"""Cross-solver interface operators for 2D/3D non-matching interfaces.

This package lifts the 1D interface-operator contract of TensorPhysics mission
m5207 (conservation / consistency / symmetry of mortar, RBF and momentum-exchange
transfers) to interface *curves in 2D* and interface *surfaces in 3D*, and fixes
the measure-normalisation trap recorded in that mission.

Ownership
---------
The operators operate on *interface discretisations* only (interface geometry,
overlap measures, projections); they contain no solver kernels and no workflow
primitives.  Solvers keep their own physics; this package is the numerical device
that decides how a quantity crosses a non-matching interface.

Modules
-------
``geometry``   exact clipping (segment/interval, convex polygon), curved
               sphere/box surface area, adaptive quadrature
``mortar``     interface sides, mortar projection 1D/2D/3D, conservative backfill
``operators``  RBF/MLS and momentum-exchange companions for the comparison table
``acceptance`` the machine-checkable C1..C6 harness (also used by the tests)
``clip3d``     triangle-to-triangle clipping on *non-coplanar* facets (curved
               interfaces: two different polyhedral approximations of one patch),
               homogeneous Sutherland-Hodgman clipping, surface mortar operator
``acceptance3d`` the machine-checkable D1..D6 harness of campaign node IN52-N01
``curved_patch_mortar``  patch-local parametrisation + Newton-projected true
               intersection + per-patch local conservation with gap/penetration
               volume terms (campaign node P248-N01)
"""

from __future__ import annotations

from .geometry import (cell_overlap, clip_convex_polygon, interval_clip,
                       sphere_box_area, polygon_area)
from .mortar import (InterfaceSide, MortarOperator, cell_averages, from_polygons,
                     gradient_matrix, grid_side, integral, l2_error,
                     measure_proportional_backfill, measure_weighted_symmetry,
                     overlap_entries, reflect_side, rotate_side, segment_side,
                     tri_patch_side)
from .operators import (MomentumExchangeOperator, RBFOperator,
                        characteristic_length)
from .clip3d import (SurfaceMortarOperator, TriSide, clip_homogeneous,
                     clip_tris_by_box, facet_gradients, ico_sphere_tris,
                     measure_backfill, octant_reference, surface_cell_averages,
                     surface_coverage, surface_integral, surface_l2_error,
                     surface_overlap_entries, tri_area, tri_pair_overlap,
                     tri_patch_side, uv_sphere_tris)
# the curved-patch names are prefixed where they would shadow the planar API
# (``grid_side`` exists in ``mortar`` for the coplanar construction)
from .curved_patch_mortar import (CurvedMortarOperator, CurvedSide, GraphPatch,
                                  ParamPatch, SpherePatch, adjoint_apply,
                                  adjoint_diagnostics, coverage_arrangement,
                                  dual_moment_defect, first_order_coupling,
                                  intersection_curve, pair_measures,
                                  per_patch_balance, plane_projection_defect,
                                  side_from_ambient)
from .curved_patch_mortar import grid_side as curved_grid_side
from .curved_patch_mortar import run_acceptance as run_curved_acceptance

__all__ = [
    "cell_overlap",
    "clip_convex_polygon",
    "interval_clip",
    "sphere_box_area",
    "polygon_area",
    "InterfaceSide",
    "MortarOperator",
    "cell_averages",
    "from_polygons",
    "gradient_matrix",
    "grid_side",
    "integral",
    "l2_error",
    "measure_proportional_backfill",
    "measure_weighted_symmetry",
    "overlap_entries",
    "reflect_side",
    "rotate_side",
    "segment_side",
    "tri_patch_side",
    "MomentumExchangeOperator",
    "RBFOperator",
    "characteristic_length",
    "SurfaceMortarOperator",
    "TriSide",
    "clip_homogeneous",
    "clip_tris_by_box",
    "facet_gradients",
    "ico_sphere_tris",
    "measure_backfill",
    "octant_reference",
    "surface_cell_averages",
    "surface_coverage",
    "surface_integral",
    "surface_l2_error",
    "surface_overlap_entries",
    "tri_area",
    "tri_pair_overlap",
    "tri_patch_side",
    "uv_sphere_tris",
    "ParamPatch",
    "SpherePatch",
    "GraphPatch",
    "CurvedSide",
    "curved_grid_side",
    "side_from_ambient",
    "pair_measures",
    "per_patch_balance",
    "coverage_arrangement",
    "intersection_curve",
    "CurvedMortarOperator",
    "plane_projection_defect",
    "first_order_coupling",
    "adjoint_apply",
    "adjoint_diagnostics",
    "dual_moment_defect",
    "run_curved_acceptance",
]