"""Rotary dryer: thermal-mechanical coupling, RTD, and heat transfer validation.

This module implements:
1. Bed kinematics regime classification (slipping/rolling/cataracting/centrifuging)
2. Particle-wall heat transfer (contact conduction + radiation)
3. Residence Time Distribution (RTD) via tracer particle method
4. Nusselt number validation against literature correlations (Tscheng 1979, Descoin 1990)
5. Moist-particle drying: psychrometrics, Ranz-Marshall heat/mass transfer,
   wet-bulb pinning, thin-layer (Lewis/Page) kinetics, and the coupled
   gas-solid bed with water and energy ledgers
   (``psychrometrics``, ``moist_transfer``, ``moist_bed``)

References
----------
- Tscheng, S.H., Watkinson, A.P., 1979. Convective heat transfer in a rotary dryer.
  Can. J. Chem. Eng. 57, 433-437.
- Descoin, C., 1990. Etude du transfert thermique dans un sechoir rotatif.
  PhD thesis, ENSMA Poitiers.
- Li, X.-Z., et al., 2005. Experimental study on the residence time distribution
  of particles in a rotary dryer. Drying Technology 23, 1779-1791.
"""

from .kinematics import (
    RotaryDryerConfig,
    BedRegime,
    classify_regime,
    froude_number,
    omega_for_froude,
    regime_boundaries,
)
from .heat_transfer import (
    contact_heat_transfer_coeff,
    radiation_heat_flux,
    total_heat_flux,
    nusselt_number,
    overall_heat_transfer_coeff,
)
from .rtd import (
    rtd_mean_residence_time,
    rtd_variance,
    rtd_curve,
    empirical_residence_time,
)
from .literature import (
    tscheng_correlation,
    descoin_correlation,
    nusselt_literature,
)
from .api import (
    calculate_rotary_dryer,
    calculate_rotary_dryer_with_drying,
)
from . import psychrometrics
from .moist_transfer import (
    nusselt_ranz_marshall,
    sherwood_ranz_marshall,
    sherwood_whitaker,
    transfer_coefficients,
    evaporation_rate,
    sorption_activity,
    wet_bulb_from_closures,
    simulate_single_particle,
    fit_thin_layer,
    page_model,
    lewis_model,
)
from .moist_drying_case import (
    MAIN_CASE,
    run_acceptance_suite,
)
from .moist_bed import (
    DryerBedConfig,
    simulate_moist_drying,
    simulate_dry_heating,
    solve_steady_profile,
    bed_ledger,
)

__all__ = [
    "RotaryDryerConfig",
    "BedRegime",
    "classify_regime",
    "froude_number",
    "omega_for_froude",
    "regime_boundaries",
    "contact_heat_transfer_coeff",
    "radiation_heat_flux",
    "total_heat_flux",
    "nusselt_number",
    "overall_heat_transfer_coeff",
    "rtd_mean_residence_time",
    "rtd_variance",
    "rtd_curve",
    "empirical_residence_time",
    "tscheng_correlation",
    "descoin_correlation",
    "nusselt_literature",
    "calculate_rotary_dryer",
    "calculate_rotary_dryer_with_drying",
    # moist drying
    "psychrometrics",
    "nusselt_ranz_marshall",
    "sherwood_ranz_marshall",
    "sherwood_whitaker",
    "transfer_coefficients",
    "evaporation_rate",
    "sorption_activity",
    "wet_bulb_from_closures",
    "simulate_single_particle",
    "fit_thin_layer",
    "page_model",
    "lewis_model",
    "DryerBedConfig",
    "simulate_moist_drying",
    "simulate_dry_heating",
    "solve_steady_profile",
    "bed_ledger",
    "MAIN_CASE",
    "run_acceptance_suite",
]
