"""DEM <-> population-balance couple verification face for particle breakage.

Sub-modules
-----------
``weibull_breakage``
    Weibull fitting (MLE + CDF least squares), size effect, hazard rate
    ``S(V,t)`` and the Weibull effective stress of a heterogeneous stress field.
``daughter_distribution``
    Mass-conserving daughter laws ``b(v,v')`` (uniform and normalised
    lognormal), KS / Wasserstein comparison utilities.
``dpbm``
    Sectional (fixed-pivot) population balance in the particle-volume
    coordinate, its closed-form reference for the uniform kernel, and the
    PSD/Wasserstein metrics.
``bonded_cluster_crushing``
    Batched bonded-sphere DEM single-particle crushing test (force statistics
    and fragment-size statistics).
``assembly_compression``
    Load-controlled uniaxial compression of a 10^4-particle breakable assembly:
    the DEM side of the DEM-PBM cross validation.
``study``
    The end-to-end verification surface (criteria C1-C6) and its artefacts.

Boundary: this package owns the *verification surface* only.  The production
crushing DEM lives in :mod:`tensordem.crushing` / :mod:`tensordem.dem3d`; the
ice-floe population balance (area coordinate) lives in
:mod:`tensordem.fsd_evolution`.
"""
from __future__ import annotations

from .weibull_breakage import (
    WeibullFit,
    WeibullStrength,
    effective_stress,
    fit_size_effect_exponent,
    fit_weibull_cdf,
    fit_weibull_mle,
    particle_stress,
    size_effect_strength,
    weibull_cdf,
    weibull_cdf_r2,
    weibull_pdf,
    weibull_survival,
)
from .daughter_distribution import (
    LognormalDaughter,
    UniformDaughter,
    daughter_mass_defect,
    daughter_volume_fractions,
    empirical_cdf,
    ks_two_sample,
    ks_two_sample_bootstrap_pvalue,
    ks_uniform,
    wasserstein_1d,
)
from .dpbm import (
    VolumeGrid,
    analytic_cell_density,
    analytic_delta_solution,
    breakage_rhs,
    coarsen_to_grid,
    cumulative_volume_fraction,
    daughter_matrix,
    equivalent_diameter,
    mass_defect_matrix,
    mass_density,
    mass_moment,
    number_moment,
    particle_volume,
    psd_from_sections,
    solve_breakage,
    two_pivot_weights,
    wasserstein_from_cdf,
)
from .bonded_cluster_crushing import (
    ClusterGeometry,
    CrushingParams,
    cluster_strengths,
    crush_batch,
    fragment_sizes,
    quantize_fractions,
    size_scan,
)

__all__ = [
    "WeibullFit",
    "WeibullStrength",
    "effective_stress",
    "fit_size_effect_exponent",
    "fit_weibull_cdf",
    "fit_weibull_mle",
    "particle_stress",
    "size_effect_strength",
    "weibull_cdf",
    "weibull_cdf_r2",
    "weibull_pdf",
    "weibull_survival",
    "LognormalDaughter",
    "UniformDaughter",
    "daughter_mass_defect",
    "daughter_volume_fractions",
    "empirical_cdf",
    "ks_two_sample",
    "ks_two_sample_bootstrap_pvalue",
    "ks_uniform",
    "wasserstein_1d",
    "VolumeGrid",
    "analytic_cell_density",
    "analytic_delta_solution",
    "breakage_rhs",
    "coarsen_to_grid",
    "cumulative_volume_fraction",
    "daughter_matrix",
    "equivalent_diameter",
    "mass_defect_matrix",
    "mass_density",
    "mass_moment",
    "number_moment",
    "particle_volume",
    "psd_from_sections",
    "solve_breakage",
    "two_pivot_weights",
    "wasserstein_from_cdf",
    "ClusterGeometry",
    "CrushingParams",
    "cluster_strengths",
    "crush_batch",
    "fragment_sizes",
    "quantize_fractions",
    "size_scan",
]