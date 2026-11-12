"""Three-dimensional granular packings.

Sub-package layout: :mod:`tensordem.granular3d.jamming` implements the static
harmonic contact potential for spheres on a triply periodic cubic cell, two
minimisers (FIRE, Polak-Ribiere+ CG), a quasi-static inflation protocol and the
zero-pressure / finite-size estimators of the jamming volume fraction.  It is
the 3-D counterpart of :mod:`tensordem.granular2d.jamming_protocol` and the
measurement surface behind the RCP anchor ``CASE-DEM-RCP-PACKING-FRACTION``.

Units are reduced (``kn = 1``, mean radius 1): a volume fraction and the shape of
``P(phi)`` are dimensionless.
"""

from .jamming import (
    ContactModel,
    MinimizerResult,
    PLACEMENT_PHI,
    SpherePacking,
    finite_size_scale,
    fire_minimize,
    fit_jamming_density,
    growth_protocol,
    make_packing,
    periodic_rsa,
    pg_minimize,
    pressure_reduced,
    pressure_resolution,
)

__all__ = [
    "ContactModel",
    "MinimizerResult",
    "PLACEMENT_PHI",
    "SpherePacking",
    "finite_size_scale",
    "fire_minimize",
    "fit_jamming_density",
    "growth_protocol",
    "make_packing",
    "periodic_rsa",
    "pg_minimize",
    "pressure_reduced",
    "pressure_resolution",
]