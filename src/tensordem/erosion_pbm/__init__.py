"""DEM particle breakage (PBM / Weibull) fused with erosion (Finnie / Oka / O501).

Campaign node ``p176w2-n08`` verification surface.  The two particle-dynamics
verification surfaces that are usually built separately are fused here because
they share one state variable -- the particle size distribution:

* **Population balance** (:mod:`.pbm`): finite-volume sectional solution of
  ``dn/dt = -S(v) n + int S(v') b(v|v') n(v') dv'`` with the selection function
  derived from a Weibull strength law (:mod:`.strength`) rather than fitted.
* **Erosion** (:mod:`.erosion`): Finnie (1960) and Oka (2005) angular functions,
  the velocity-exponent calibration, and the DNV-RP-O501 rate structure.
* **Coupling** (:mod:`.dem`): an impact-DEM ensemble that breaks particles on a
  Weibull-strength criterion, removes target mass with the erosion law, and
  closes a mass-loss ledger per size section.
* **Acceptance** (:mod:`.study`): every criterion of the node with its numbers
  and artifacts.
* **Mesh metrology** (:mod:`.mesh_gci`, :mod:`.mesh_study`): discretisation-error
  measurement for the sectional mesh -- log-log variable grids with a fixed
  overall grading, analytic bin-integral moments, an exact-in-time
  uniformisation propagator, Richardson/Roache GCI on the PSD moments, the fines
  floor as an explicit criterion, and the grid share of an erosion rate
  (campaign node ``p488w1-n09``).

Quick start::

    python -m tensordem.erosion_pbm.study --out artifacts/p176w2_n08
    python -m tensordem.erosion_pbm.mesh_study --out artifacts/p488w1_n09
"""
from __future__ import annotations

from .strength import (GranularStrength, breakage_probability, impact_stress,
                      sample_strengths, weibull_plot_fit)
from .pbm import (SectionalGrid, bin_counts, breakage_rhs, daughter_matrix,
                  mass_moment, normalised_psd, number_moment, solve_breakage,
                  two_pivot_weights, wasserstein_1d)
from .erosion import (CARBON_STEEL, FINNIE_CRIT_ANGLE_DEG, FINNIE_PEAK_ANGLE_DEG,
                      DNVO501Material, OkaErosionLaw, dnv_o501_erosion_ratio,
                      dnv_o501_penetration_rate_mm_per_year,
                      finnie_angle_function, fit_coefficient_loglog,
                      fit_power_exponent, max_relative_deviation,
                      oka_angle_function, oka_hv_exponents, oka_peak_angle_deg,
                      oka_size_factor, relative_deviations)
from .dem import (ImpactDEMConfig, impact_ensemble_erosion, ledger_closure,
                  meanfield_selection, run_impact_dem)
from .study import CASE, build_grid, dem_config, erosion_law, run_study, strength_model
from .mesh_gci import (LogLogGrid, cascade_error, floor_threshold, mass_ledger,
                       propagate, richardson_gci)
from .mesh_study import run_study as run_mesh_study

__all__ = [
    "GranularStrength", "breakage_probability", "impact_stress",
    "sample_strengths", "weibull_plot_fit",
    "SectionalGrid", "bin_counts", "breakage_rhs", "daughter_matrix",
    "mass_moment", "normalised_psd", "number_moment", "solve_breakage",
    "two_pivot_weights", "wasserstein_1d",
    "CARBON_STEEL", "FINNIE_CRIT_ANGLE_DEG", "FINNIE_PEAK_ANGLE_DEG",
    "DNVO501Material", "OkaErosionLaw", "dnv_o501_erosion_ratio",
    "dnv_o501_penetration_rate_mm_per_year", "finnie_angle_function",
    "fit_coefficient_loglog", "fit_power_exponent", "max_relative_deviation",
    "oka_angle_function", "oka_hv_exponents", "oka_peak_angle_deg",
    "oka_size_factor", "relative_deviations",
    "ImpactDEMConfig", "impact_ensemble_erosion", "ledger_closure",
    "meanfield_selection", "run_impact_dem",
    "CASE", "build_grid", "dem_config", "erosion_law", "run_study",
    "strength_model",
    "LogLogGrid", "cascade_error", "floor_threshold", "mass_ledger",
    "propagate", "richardson_gci", "run_mesh_study",
]