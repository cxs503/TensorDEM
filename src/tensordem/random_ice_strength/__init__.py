"""Random ice-plate bending strength field and its breaking-force ensemble.

Campaign node ``P295W1-N07``.  The chain, in order:

:mod:`~tensordem.random_ice_strength.distributions`
    two-parameter Weibull marginal of the flexural strength with MLE / QQ-plot
    calibration, KS distance and a lognormal negative control;
:mod:`~tensordem.random_ice_strength.field`
    exact (grid-)Karhunen-Loeve Gaussian field with a prescribed correlation
    length, and the Gaussian-copula transform that turns it into a Weibull
    strength field with the *same* rank correlation structure;
:mod:`~tensordem.random_ice_strength.beam`
    unit-width bending beam strips of a floating ice sheet: the strip capacity
    ``0.25 sigma_f h**2 / L_c`` from :mod:`tensordem.ice_channel3d.slab` with a
    random ``sigma_f``, and the bundle sum over the contact-line width;
:mod:`~tensordem.random_ice_strength.ensemble`
    breaking-force ensemble statistics (mean convergence, CV, 5 % quantile) and
    the width sweep with nested realisations that exposes the size effect;
:mod:`~tensordem.random_ice_strength.study`
    the acceptance study that produces every number in the report.

The public contract is ``(marginal, correlation length, sheet geometry)`` in and
``(force ensemble, width curve)`` out; nothing downstream needs to know how the
field was generated.
"""

from .distributions import (  # noqa: F401
    WeibullMarginal,
    fit_weibull_mle,
    gaussian_copula_uniforms,
    ks_distance,
    negative_control_lognormal,
    qq_plot_r2,
    qq_quantiles,
    spearman_from_gaussian_correlation,
    weibull_moments,
)
from .field import (  # noqa: F401
    KERNELS,
    KlField1D,
    correlated_weibull_field,
    correlation_kernel,
    correlation_l2_error,
    correlation_length_1e,
    correlation_matrix,
    empirical_correlation,
    empirical_rank_correlation,
    gaussian_field_1d,
    spearman_target,
)
from .beam import (  # noqa: F401
    BENDING_EFFICIENCY,
    IceSheetStrengthModel,
    breaking_force,
    deterministic_breaking_force,
    strip_force_per_width,
    strip_forces,
)
from .ensemble import (  # noqa: F401
    bootstrap_quantile_ci,
    breaking_force_ensemble,
    ensemble_statistics,
    fit_power_law,
    monotone_decreasing,
    running_mean_convergence,
    size_effect_sweep,
)
from .study import (  # noqa: F401
    CALIBRATION,
    CRITERIA,
    run_study,
)

__all__ = [
    "WeibullMarginal", "fit_weibull_mle", "qq_plot_r2", "qq_quantiles",
    "ks_distance", "negative_control_lognormal", "weibull_moments",
    "gaussian_copula_uniforms", "spearman_from_gaussian_correlation",
    "KERNELS", "KlField1D", "gaussian_field_1d", "correlation_matrix",
    "correlation_kernel", "correlation_length_1e", "correlation_l2_error",
    "empirical_correlation", "empirical_rank_correlation",
    "correlated_weibull_field", "spearman_target",
    "IceSheetStrengthModel", "strip_force_per_width", "strip_forces",
    "breaking_force", "deterministic_breaking_force", "BENDING_EFFICIENCY",
    "ensemble_statistics", "running_mean_convergence", "bootstrap_quantile_ci",
    "breaking_force_ensemble", "size_effect_sweep", "fit_power_law",
    "monotone_decreasing", "CALIBRATION", "CRITERIA", "run_study",
]