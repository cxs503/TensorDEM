"""Ice broken field dynamics: Hertz/Mindlin contact, fracture energy, channel clearing, MT Uikku validation.

This module implements the four-chain verification surface for ice broken field
dynamics maturity (GAP-DEM-G3 closure):

M1. Floe-ship Hertz/Mindlin contact
    - Normal: Hertz F_n = (4/3) E* sqrt(R*) delta^{3/2}
    - Tangential: Mindlin with stick/slip transition
    - Ice-steel friction mu_ice-steel = 0.05-0.15 (temperature dependent)
    - Verification: single floe-ship collision vs Hertz analytical solution (≤5%)
    - Negative control: delta=0 → F_n=0

M2. Ice-ice collision fracture energy ledger
    - Fracture energy G_c from K_IC ≈ 0.1 MPa·m^{1/2}
    - Energy balance: E_k = E_elastic + E_friction + E_fracture + E_kinetic_residual
    - Verification: E_fracture/E_k vs Timco 2008 experimental values (≤20%)
    - Negative control: v=0 → E_fracture=0

M3. Broken ice channel clearing rate
    - Ice field: concentration C_i = 20-80%, thickness h_i = 0.5-2m
    - Output: clearing rate eta_clear = L_clear / L_total
    - Ice resistance: R_ice = sum(F_contact)
    - Verification: R_ice vs Lindqvist empirical formula (≤30%)
    - Negative control: C_i=0 → R_ice=0

M4. MT Uikku full-scale experiment comparison
    - 10 test cases: speed 3-10 kn, thickness 0.6-1.2m, concentration 60-90%
    - Output: F_ice(t) time history
    - Verification: F_peak ≤25%, F_mean ≤20%, correlation rho ≥ 0.7
    - Negative control: h_i=0 → F_ice=0

All quantities SI (m, kg, s, Pa, N). Energy ledger closes to ≤5%.
"""
from .hertz_mindlin_floe_ship import (
    hertz_normal_force,
    mindlin_tangential_force,
    hertz_analytical_solution,
    single_floe_ship_collision,
    verify_hertz_contact,
)
from .ice_ice_fracture import (
    fracture_energy_from_kic,
    ice_ice_collision_energy_balance,
    verify_fracture_energy,
)
from .channel_clearing import (
    broken_ice_field_simulation,
    channel_clearing_rate,
    ice_resistance_lindqvist_comparison,
    verify_channel_clearing,
)
from .mt_uikku_validation import (
    MT_UIKKU_CASES,
    run_mt_uikku_case,
    run_all_mt_uikku_cases,
    verify_mt_uikku,
)

__all__ = [
    # M1: Hertz/Mindlin floe-ship contact
    "hertz_normal_force",
    "mindlin_tangential_force",
    "hertz_analytical_solution",
    "single_floe_ship_collision",
    "verify_hertz_contact",
    # M2: Ice-ice fracture energy
    "fracture_energy_from_kic",
    "ice_ice_collision_energy_balance",
    "verify_fracture_energy",
    # M3: Channel clearing
    "broken_ice_field_simulation",
    "channel_clearing_rate",
    "ice_resistance_lindqvist_comparison",
    "verify_channel_clearing",
    # M4: MT Uikku validation
    "MT_UIKKU_CASES",
    "run_mt_uikku_case",
    "run_all_mt_uikku_cases",
    "verify_mt_uikku",
]
