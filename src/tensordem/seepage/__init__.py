"""Seepage–piping initiation criterion closed-loop verification (campaign p299w4-n02).

Four physically independent links, each with a machine-judgeable acceptance gate:

========================  =====================================================
module                    role
========================  =====================================================
``effective_stress``      Terzaghi effective stress: pore-pressure profile vs
                          analytic hydrostatic + self-weight solution (C1)
``critical_gradient``     Piping onset: critical hydraulic gradient i_c vs
                          Terzaghi formula i_c = (G_s - 1)/(1 + e) (C2)
``kozeny_carman``         Permeability inversion: k vs K-C formula at multiple
                          porosities (C3)
``suffusion``             Internal erosion: fine-particle loss rate vs
                          Skempton–Brogan experimental envelope (C4)
``chain``                 Closed-chain runner: all gates + mass conservation
                          audit (C5)
========================  =====================================================

All quantities are SI.  Sign convention: compression positive for stress,
upward positive for seepage velocity.
"""
from __future__ import annotations

from .effective_stress import (
    SaturatedColumn,
    effective_stress_profile,
    hydrostatic_pore_pressure,
    self_weight_pore_pressure,
    terzaghi_effective_stress,
)
from .critical_gradient import (
    PipingAnalysis,
    buoyant_unit_weight,
    piping_factor_of_safety,
    seepage_force_per_volume,
    terzaghi_critical_gradient,
)
from .kozeny_carman import (
    KozenyCarmanModel,
    kozeny_carman_conductivity,
    kozeny_carman_intrinsic,
    kozeny_carman_porosity_sweep,
    specific_surface_sphere,
)
from .suffusion import (
    SkemptonBroganEnvelope,
    SuffusionAnalysis,
    fine_fraction_loss,
    skempton_brogan_critical_gradient,
    suffusion_erosion_rate,
)
from .chain import (
    SeepagePrototype,
    mass_conservation_audit,
    run_seepage_chain,
)
from .dem_column import ColumnDEMConfig, SeepageColumn
from .dem_measure import (
    PIPING_GRADIENT_RAMP,
    measured_critical_gradient,
    measured_effective_stress_profile,
    measured_fluid_ledger,
    measured_mass_conservation,
    run_measured_links,
)
from .dem_chain import run_dem_seepage_chain

__all__ = [
    "SaturatedColumn",
    "effective_stress_profile",
    "hydrostatic_pore_pressure",
    "self_weight_pore_pressure",
    "terzaghi_effective_stress",
    "PipingAnalysis",
    "buoyant_unit_weight",
    "piping_factor_of_safety",
    "seepage_force_per_volume",
    "terzaghi_critical_gradient",
    "KozenyCarmanModel",
    "kozeny_carman_conductivity",
    "kozeny_carman_intrinsic",
    "kozeny_carman_porosity_sweep",
    "specific_surface_sphere",
    "SkemptonBroganEnvelope",
    "SuffusionAnalysis",
    "fine_fraction_loss",
    "skempton_brogan_critical_gradient",
    "suffusion_erosion_rate",
    "SeepagePrototype",
    "mass_conservation_audit",
    "run_seepage_chain",
    "ColumnDEMConfig",
    "SeepageColumn",
    "PIPING_GRADIENT_RAMP",
    "measured_effective_stress_profile",
    "measured_critical_gradient",
    "measured_mass_conservation",
    "measured_fluid_ledger",
    "run_measured_links",
    "run_dem_seepage_chain",
]
