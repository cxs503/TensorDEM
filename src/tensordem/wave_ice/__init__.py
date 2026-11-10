"""Wave-ice coupling verification surface for TensorDEM.

This package provides verification modules for:
1. Single floe RAO (Response Amplitude Operator) in regular waves
2. Multi-floe array transmission/scattering coefficients
3. Wave-induced ice breaking threshold

References:
- WAMIT/AQWA potential flow theory for RAO validation
- Kohout et al. (2014) for transmission coefficient experiments
- Elastic plate bending theory for breaking threshold
"""
from .floe_rao import (
    FloeRAOConfig,
    FloeRAOResult,
    compute_floe_rao_analytic,
    compute_floe_rao_dem,
    rao_peak_error,
    rao_amplitude_error,
)
from .transmission import (
    FloeArrayConfig,
    TransmissionResult,
    compute_transmission_coefficient_dem,
    compute_transmission_analytic_meylan,
    transmission_error,
)
from .breaking import (
    IceBreakingConfig,
    BreakingResult,
    compute_critical_wave_height,
    compute_critical_wave_height_theory,
    breaking_threshold_error,
)

__all__ = [
    "FloeRAOConfig",
    "FloeRAOResult",
    "compute_floe_rao_analytic",
    "compute_floe_rao_dem",
    "rao_peak_error",
    "rao_amplitude_error",
    "FloeArrayConfig",
    "TransmissionResult",
    "compute_transmission_coefficient_dem",
    "compute_transmission_analytic_meylan",
    "transmission_error",
    "IceBreakingConfig",
    "BreakingResult",
    "compute_critical_wave_height",
    "compute_critical_wave_height_theory",
    "breaking_threshold_error",
]
