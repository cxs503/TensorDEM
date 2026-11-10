"""Terramechanics: vehicle–soft-sediment interaction models.

Modules
-------
``bekker``       Bekker pressure–sinkage relation  p = (k_c/b + k_phi) * z^n
``janosi``       Janosi shear–traction curve       tau = tau_max * (1 - exp(-j/K))
``coupling``     Slip–sinkage coupling for tracked vehicles on soft sediment
``inversion``    Parameter inversion from experimental data

References
----------
- Bekker, M.G. (1956) *Theory of Land Locomotion*, Univ. of Michigan Press.
- Janosi, Z. & Hanamoto, M. (1961) "The analytical determination of drawbar
  pull as a function of slip for tracked vehicles in a deformable medium."
- Wong, J.Y. (2001) *Theory of Ground Vehicles*, 3rd ed., Wiley.

Units are SI throughout (m, Pa, N, -).
"""
from __future__ import annotations

from .bekker import (
    BekkerModel,
    bekker_pressure,
    bekker_sinkage_from_pressure,
    bekker_stiffness,
)
from .janosi import (
    JanosiModel,
    janosi_shear_stress,
    janosi_max_shear,
    mohr_coulomb_tau_max,
)
from .coupling import (
    SlipSinkageState,
    track_slip_sinkage,
    track_drawbar_pull,
    track_motion_resistance,
)
from .inversion import (
    BekkerInversionResult,
    invert_bekker_parameters,
    invert_janosi_parameters,
)

__all__ = [
    "BekkerModel",
    "bekker_pressure",
    "bekker_sinkage_from_pressure",
    "bekker_stiffness",
    "JanosiModel",
    "janosi_shear_stress",
    "janosi_max_shear",
    "mohr_coulomb_tau_max",
    "SlipSinkageState",
    "track_slip_sinkage",
    "track_drawbar_pull",
    "track_motion_resistance",
    "BekkerInversionResult",
    "invert_bekker_parameters",
    "invert_janosi_parameters",
]
