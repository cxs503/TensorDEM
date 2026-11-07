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
]