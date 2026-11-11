"""Pressure-band correlation length ``lam_x``: calibrated, auditable estimator.

Campaign node **P204-N08** (TensorDEM).  The literature quotes the along-band
correlation length of the ice--structure pressure field as ``lam_x / h = 1.5-3``
(``h`` = ice thickness) and that number is normally *asserted*, not measured.
This package turns it into an estimator with a measured finite-window bias, a
record-length sufficiency threshold, an uncertainty interval and a versioned
output contract.

Modules
-------
``fields``
    synthetic 1-D fields (spectral and AR(1)) with an analytically known
    ``R(r) = exp(-|r|/lam)`` -- the ground truth for the bias calibration.
``estimators``
    three independent estimators (ACF integral scale, exponential ACF fit,
    Lorentzian spectral fit) plus a regime diagnostic that *rejects* white-noise
    and strictly periodic records instead of silently reporting a number.
``calibration``
    Monte-Carlo finite-window bias ``rho(K) = 1 + a/K + b/K**2`` with
    ``K = L/lam``, the self-consistent correction ``lam_corr = lam_hat /
    rho(L/lam_corr)``, a leave-one-out threshold ``K*`` for ``|bias| < 5 %``, and
    the variance-scaling tests ``sd ~ K**-0.5`` and ``sd ~ lam**1.5`` at fixed
    ``L``.
``dem_band``
    long-domain ice--structure crushing DEM pressure band, its time-averaged
    profile, the corrected correlation length, a moving-block bootstrap interval
    and a shuffle control.
``contract``
    ``lambda_x_estimate.v1`` payload plus the flat prop-1104 consumer view.

Evidence grade: REAL for the simulated band (computed by the DEM, not
hard-coded) and CALIBRATED for the corrected length on a finite record (the
correction is fitted to synthetic fields with a known answer and validated
out-of-sample).
"""

from .calibration import (
    BiasModel,
    MODEL_FORMS,
    select_bias_model,
    sustained_threshold,
    BiasStudy,
    correct_length,
    correct_length_batch,
    fit_bias_model,
    k_sufficient,
    loglog_fit,
    run_bias_study,
    run_variance_family,
    variance_scaling_grid,
    scan_threshold_out_of_sample,
    validate_threshold,
)
from .contract import (
    LITERATURE_BAND,
    SCHEMA_VERSION,
    build_estimate,
    consensus,
    to_prop_1104,
    validate,
)
from .dem_band import (
    BandConfig,
    block_coarsen_profile,
    resolution_ladder,
    BandRun,
    bootstrap_lambda,
    d50_ladder,
    dem_band_cell,
    estimate_profile_lambda,
    run_band,
    shuffle_control,
    snapshot_profiles,
    time_averaged_profile,
)
from .estimators import (
    METHODS,
    PRIMARY_METHODS,
    Regime,
    autocorrelation,
    diagnose_regime,
    dominant_period,
    estimate_all,
    report_length_scale,
    lambda_efold,
    lambda_expfit,
    lambda_crossing,
    lambda_lagslope,
    lambda_integral,
    lambda_spectral,
)
from .fields import ar1_field, exponential_acf, spectral_field, windowed_field

__all__ = [
    "ar1_field",
    "spectral_field",
    "exponential_acf",
    "windowed_field",
    "autocorrelation",
    "lambda_efold",
    "lambda_crossing",
    "lambda_lagslope",
    "lambda_integral",
    "lambda_expfit",
    "lambda_spectral",
    "estimate_all",
    "diagnose_regime",
    "dominant_period",
    "report_length_scale",
    "Regime",
    "METHODS",
    "PRIMARY_METHODS",
    "BiasModel",
    "BiasStudy",
    "fit_bias_model",
    "select_bias_model",
    "sustained_threshold",
    "MODEL_FORMS",
    "correct_length",
    "correct_length_batch",
    "k_sufficient",
    "run_bias_study",
    "run_variance_family",
    "variance_scaling_grid",
    "loglog_fit",
    "d50_ladder",
    "validate_threshold",
    "scan_threshold_out_of_sample",
    "BandConfig",
    "BandRun",
    "run_band",
    "snapshot_profiles",
    "time_averaged_profile",
    "estimate_profile_lambda",
    "bootstrap_lambda",
    "dem_band_cell",
    "block_coarsen_profile",
    "resolution_ladder",
    "shuffle_control",
    "SCHEMA_VERSION",
    "LITERATURE_BAND",
    "build_estimate",
    "to_prop_1104",
    "consensus",
    "validate",
]
# Mission p360w1-n04: ensemble estimation
from .ensemble import (
    ACFfitResult,
    SingleRecordEstimate,
    EnsembleEstimate,
    BootstrapResult,
    ConvergenceResult,
    fit_acf_exponential,
    fit_acf_gaussian,
    acf_fit_model,
    estimate_single_record,
    ensemble_estimate,
    bootstrap_ensemble,
    convergence_study,
    generate_records,
)

__all__.extend([
    "ACFfitResult",
    "SingleRecordEstimate",
    "EnsembleEstimate",
    "BootstrapResult",
    "ConvergenceResult",
    "fit_acf_exponential",
    "fit_acf_gaussian",
    "acf_fit_model",
    "estimate_single_record",
    "ensemble_estimate",
    "bootstrap_ensemble",
    "convergence_study",
    "generate_records",
])
