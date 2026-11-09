"""Silo / hopper granular-flow verification surface for TensorDEM.

Four-stage verification chain for bulk-solids storage and discharge:

1. :mod:`tensordem.silo_flow.beverloo` -- flat-bottom orifice discharge and the
   Beverloo ``W = C rho_b sqrt(g) (D - k d)^(5/2)`` law (exponent 5/2);
2. :mod:`tensordem.silo_flow.janssen` -- wall/base pressure saturation in a deep
   filled silo and Janssen's ``sigma_v = P_inf (1 - exp(-z/z0))``;
3. :mod:`tensordem.silo_flow.flow_pattern` -- mass-flow / funnel-flow transition
   in a conical hopper and the critical half angle vs Jenike's chart;
4. :mod:`tensordem.silo_flow.kernel` -- the DEM kernel itself (Cundall--Strack
   wall friction, rolling resistance, closed per-timestep energy ledger).

The kernel is self-contained (numpy + scipy cKDTree): no torch dependency, so the
campaign can be sharded over cores/nodes.
"""
from .kernel import Material, SiloGeometry, SiloFlow, contact_time, damping_ratio
from .packing import build_bed, deposit_bed, lattice_fill, relax_overlaps
from .beverloo import (
    BEVERLOO_C, BEVERLOO_K, DischargeCase, beverloo_prediction, fit_law, run_discharge,
)
from .janssen import JanssenCase, fit_janssen, janssen_profile, run_janssen
from .flow_pattern import (
    HopperCase, classify_flow, find_critical_angle, jenike_mass_flow_limit_deg, run_hopper,
)

__all__ = [
    "Material", "SiloGeometry", "SiloFlow", "contact_time", "damping_ratio",
    "build_bed", "deposit_bed", "lattice_fill", "relax_overlaps",
    "BEVERLOO_C", "BEVERLOO_K", "DischargeCase", "beverloo_prediction", "fit_law",
    "run_discharge",
    "JanssenCase", "fit_janssen", "janssen_profile", "run_janssen",
    "HopperCase", "classify_flow", "find_critical_angle",
    "jenike_mass_flow_limit_deg", "run_hopper",
]