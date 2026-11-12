"""Three-dimensional granular packings.

Sub-package layout: :mod:`tensordem.granular3d.jamming` implements the static
harmonic contact potential for spheres on a triply periodic cubic cell, two
minimisers (FIRE, Polak-Ribiere+ CG), a quasi-static inflation protocol and the
zero-pressure / finite-size estimators of the jamming volume fraction.  It is
the 3-D counterpart of :mod:`tensordem.granular2d.jamming_protocol` and the
measurement surface behind the RCP anchor ``CASE-DEM-RCP-PACKING-FRACTION``.

Units are reduced (``kn = 1``, mean radius 1): a volume fraction and the shape of
``P(phi)`` are dimensionless.

:mod:`tensordem.granular3d.stress` is the 3-D counterpart of
:mod:`tensordem.granular2d.stress`: per-contact Virial tensors, plane fluxes,
wall tractions, the discrete moment identity and the region force balance, all
verified against exact statics in ``docs/granular3d_stress_verification.md``.
"""

from . import stress
from .stress import (
    contact_outer,
    deviatoric,
    from_pair_list,
    kinetic_stress,
    moment_residual,
    plane_profile,
    plane_stress,
    plane_traction,
    pressure,
    region_balance,
    stress_tensor,
    wall_stress,
)

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
    "stress",
    "contact_outer",
    "deviatoric",
    "from_pair_list",
    "kinetic_stress",
    "moment_residual",
    "plane_profile",
    "plane_stress",
    "plane_traction",
    "pressure",
    "region_balance",
    "stress_tensor",
    "wall_stress",
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
