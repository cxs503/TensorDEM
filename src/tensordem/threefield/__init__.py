"""Ice-hull-propeller three-field coupling (DEM + LBM + fragment thermodynamics).

Sub-modules
-----------
``resistance``    level-ice resistance baselines (Lindqvist / Riska style)
``wake_lbm``      D3Q19 lattice-Boltzmann propeller race with Smagorinsky SGS
``fragmentation`` turbulent floe breakup cascade and Weibull statistics
``thermal``       Ranz-Marshall convection + Stefan phase-change melting
``hull_ice``      DEM hull-ice contact, contact-pressure field, fragment census
``coupling``      the coupled pass (fragmentation -> melting -> mass ledger)
"""

from .resistance import (
    Environment,
    HullGeometry,
    IceProperties,
    lindqvist_resistance,
    relative_error_pct,
    resistance_band,
    riska_resistance,
)
from .wake_lbm import LBMConfig, WakeLBM, gaussian_race_profile, profile_error
from .fragmentation import (
    BreakupProperties,
    cdf_agreement_r2,
    critical_floe_length,
    fit_weibull,
    fit_weibull_cdf,
    fragment,
    weibull_cdf,
)
from .thermal import (
    ThermalProperties,
    analytic_diameter_ransom_marshall,
    analytic_ransom_marshall_time,
    diameter_rate,
    heat_transfer_coefficient,
    integrate_melting,
    nusselt_ransom_marshall,
)
from .hull_ice import HullIceCase, run_hull_ice_case, run_thickness_sweep
from .coupling import (
    CouplingConfig,
    follower_ship_impact,
    mass_ledger,
    run_coupled_pass,
    fragmentation_stage,
    thermal_stage,
)

__all__ = [
    "Environment", "HullGeometry", "IceProperties",
    "lindqvist_resistance", "riska_resistance", "resistance_band", "relative_error_pct",
    "LBMConfig", "WakeLBM", "gaussian_race_profile", "profile_error",
    "BreakupProperties", "critical_floe_length", "fragment", "fit_weibull", "fit_weibull_cdf",
    "weibull_cdf", "cdf_agreement_r2",
    "ThermalProperties", "nusselt_ransom_marshall", "heat_transfer_coefficient",
    "diameter_rate", "integrate_melting", "analytic_ransom_marshall_time",
    "analytic_diameter_ransom_marshall",
    "HullIceCase", "run_hull_ice_case", "run_thickness_sweep",
    "CouplingConfig", "run_coupled_pass", "fragmentation_stage", "thermal_stage",
    "mass_ledger", "follower_ship_impact",
]