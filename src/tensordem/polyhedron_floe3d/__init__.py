"""Three-dimensional polyhedral floe DEM with Weibull weakest-link fracture.

This package lifts the 2-D Laguerre/polygon floe field (m7768) to three
dimensions:

* :mod:`~tensordem.polyhedron_floe3d.geometry` -- convex polyhedra from a 3-D
  Voronoi tessellation or by extruding a 2-D floe polygon, with exact volume,
  centroid and inertia tensor;
* :mod:`~tensordem.polyhedron_floe3d.weibull` -- Weibull weakest-link strength
  statistics: maximum-likelihood calibration of ``(m, sigma_0)``, probability
  plot, size effect and Monte-Carlo fragmentation;
* :mod:`~tensordem.polyhedron_floe3d.contact` -- separating-axis convex contact
  (gap, normal, contact patch), wall contact and the contact law;
* :mod:`~tensordem.polyhedron_floe3d.dynamics` -- rigid-body DEM with out-of-
  plane rotation, exact buoyancy of a partially submerged polyhedron and
  auditable mass / energy / momentum ledgers;
* :mod:`~tensordem.polyhedron_floe3d.field` -- 3-D brash-ice fields with a
  thickness distribution and a measured porosity;
* :mod:`~tensordem.polyhedron_floe3d.study` -- the three validation cases
  (impact fragmentation, brash-ice channel clearing, rubble repose & force
  chains) and their machine-checkable verdict.
"""
from __future__ import annotations

from .geometry import (
    ConvexPolyhedron,
    body_frames,
    disc_equivalent_polygon,
    equivalent_disc_prism,
    extrude_polygon,
    halfspace_vertices,
    polygon_area,
    prism_from_polygon,
    regular_polygon,
    voronoi_cell_polyhedron,
    voronoi_tessellate,
)
from .weibull import (
    FragmentationResult,
    WeibullFit,
    fit_mle,
    fit_probability_plot,
    fragment_sizes_from_mask,
    fragmentation_monte_carlo,
    kolmogorov_smirnov,
    probability_plot_points,
    sample_strengths,
    size_effect_study,
    weakest_link_scale,
    weakest_link_survival,
    weibull_cdf,
    weibull_pdf,
    weibull_sf,
)

__all__ = [
    "ConvexPolyhedron",
    "body_frames",
    "disc_equivalent_polygon",
    "equivalent_disc_prism",
    "extrude_polygon",
    "halfspace_vertices",
    "polygon_area",
    "prism_from_polygon",
    "regular_polygon",
    "voronoi_cell_polyhedron",
    "voronoi_tessellate",
    "FragmentationResult",
    "WeibullFit",
    "fit_mle",
    "fit_probability_plot",
    "fragment_sizes_from_mask",
    "fragmentation_monte_carlo",
    "kolmogorov_smirnov",
    "probability_plot_points",
    "sample_strengths",
    "size_effect_study",
    "weakest_link_scale",
    "weakest_link_survival",
    "weibull_cdf",
    "weibull_pdf",
    "weibull_sf",
]