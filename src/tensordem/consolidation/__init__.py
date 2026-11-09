"""One-dimensional consolidation of a saturated granular bed (campaign IN99-N08).

The package turns ``r_u(t)`` and the settlement-time curve from "a shape someone
plotted" into a deterministic, analytic-anchored quantity that the liquefaction
chain can consume:

======================  =====================================================
module                  role
======================  =====================================================
``terzaghi``            independent Fourier/short-time anchor ``U(T_v)``
``poroelastic``         layered two-field column; ``C_v`` is *assembled*
``dem_oedometer``       DEM piston test: ``E_oed``, ``n``, ``Z``, ``k_hyd``
``inversion``           ``t50``/``t90``/full-curve back-analysis + CIs
``ru_table``            versioned ``r_u(T_v)`` JSON hand-off
``applicability``       fail-closed domain gates (Darcy, saturation, ice thaw)
``chain``               the closed chain and every acceptance gate (C1..C3)
======================  =====================================================
"""
from __future__ import annotations

from .applicability import (
    ApplicabilityLimits,
    ApplicabilityReport,
    ConsolidationDomainError,
    check_applicability,
)
from .chain import (
    DEFAULT_TV_POINTS,
    PrototypeBed,
    column_refinement_scan,
    prototype_pore_reynolds,
    run_chain,
    u_drift_pct,
)
from .dem_oedometer import (
    OedometerConfig,
    OedometerCell,
    carman_kozeny_conductivity,
    pore_reynolds,
    run_oedometer,
)
from .inversion import (
    ci95,
    cv_from_time_factor,
    invert_cv,
    invert_settlement,
    propagate_cv_uncertainty,
    time_to_degree,
)
from .poroelastic import ColumnParameters, PoroelasticColumn
from .ru_table import (
    SCHEMA,
    build_ru_table,
    r_u_at,
    r_u_profile_at,
    read_ru_table,
    table_sha256,
    write_ru_table,
)
from .terzaghi import (
    TV_AT_U50,
    TV_AT_U90,
    Tv_of_U,
    U_of_Tv,
    U_series,
    U_sqrt_approx,
    U_two_branch,
    r_u_profile,
    u_profile,
)

__all__ = [
    "ApplicabilityLimits",
    "ApplicabilityReport",
    "ConsolidationDomainError",
    "check_applicability",
    "DEFAULT_TV_POINTS",
    "PrototypeBed",
    "column_refinement_scan",
    "prototype_pore_reynolds",
    "run_chain",
    "u_drift_pct",
    "OedometerConfig",
    "OedometerCell",
    "carman_kozeny_conductivity",
    "pore_reynolds",
    "run_oedometer",
    "ci95",
    "cv_from_time_factor",
    "invert_cv",
    "invert_settlement",
    "propagate_cv_uncertainty",
    "time_to_degree",
    "ColumnParameters",
    "PoroelasticColumn",
    "SCHEMA",
    "build_ru_table",
    "r_u_at",
    "r_u_profile_at",
    "read_ru_table",
    "table_sha256",
    "write_ru_table",
    "TV_AT_U50",
    "TV_AT_U90",
    "Tv_of_U",
    "U_of_Tv",
    "U_series",
    "U_sqrt_approx",
    "U_two_branch",
    "r_u_profile",
    "u_profile",
]