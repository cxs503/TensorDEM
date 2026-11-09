"""Stochastic ice-pressure field generation and structural fatigue spectral analysis.

Chain implemented here (campaign node AG27W2-N09):

:mod:`~tensordem.stochastic_pressure_field.dem_field`
    DEM of a crushed-ice layer between a rigid panel and an advancing indenter;
    emits the panel pressure field ``p(x, y, t)`` in model scale.
:mod:`~tensordem.stochastic_pressure_field.correlation`
    spatial correlation function of the DEM field and its fitted
    exponential / Matern model (correlation lengths ``Lx``, ``Ly``);
:mod:`~tensordem.stochastic_pressure_field.kl`
    Karhunen-Loeve expansion of the fitted (separable) covariance, truncated at
    a prescribed energy fraction;
:mod:`~tensordem.stochastic_pressure_field.plate`
    simply-supported plating panel: modal projection of the random pressure
    field and the hot-spot bending stress history (analytic Navier modes);
:mod:`~tensordem.stochastic_pressure_field.fatigue`
    rainflow -> DNV-RP-C203 S-N -> Miner damage -> fatigue life, with the Monte
    Carlo over random-field realisations.

The public statistical contract is the pair ``(variance, correlation model)``
plus the pressure level; everything downstream consumes only those.
"""

from .dem_field import (  # noqa: F401
    ICE_DENSITY,
    PressureSnapshots,
    hertz_contact_stiffness,
    simulate_crushed_layer,
    single_grain_hertz_verification,
)
from .correlation import (  # noqa: F401
    CorrelationFit,
    autocorrelation_2d,
    fit_correlation_model,
    fitted_2d_model,
    matern_1d,
    separable_covariance,
)
from .kl import (  # noqa: F401
    KLExpansion2D,
    kl_expansion_separable,
)
from .plate import (  # noqa: F401
    PlatePanel,
    hot_spot_stress,
    modal_forces,
    modal_response_ramp,
    navier_static_point_load,
    navier_static_uniform,
    static_point_load_stress,
    stress_history_from_modal_force,
)
from .fatigue import (  # noqa: F401
    FatigueMonteCarlo,
    analytic_miner_anchor,
    ice_pressure_load_envelope,
    lifetime_from_damage,
    monte_carlo_fatigue,
    rainflow_miner_life,
)

__all__ = [
    "ICE_DENSITY",
    "PressureSnapshots",
    "hertz_contact_stiffness",
    "simulate_crushed_layer",
    "single_grain_hertz_verification",
    "CorrelationFit",
    "autocorrelation_2d",
    "fit_correlation_model",
    "fitted_2d_model",
    "matern_1d",
    "separable_covariance",
    "KLExpansion2D",
    "kl_expansion_separable",
    "PlatePanel",
    "hot_spot_stress",
    "modal_forces",
    "modal_response_ramp",
    "navier_static_point_load",
    "navier_static_uniform",
    "static_point_load_stress",
    "stress_history_from_modal_force",
    "FatigueMonteCarlo",
    "analytic_miner_anchor",
    "ice_pressure_load_envelope",
    "lifetime_from_damage",
    "monte_carlo_fatigue",
    "rainflow_miner_life",
]