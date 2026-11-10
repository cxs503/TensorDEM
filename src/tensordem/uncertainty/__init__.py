"""Uncertainty quantification for DEM ice-structure contact parameters.

This module provides tools for:
- Probability distribution specification for friction coefficient mu
- Monte Carlo propagation of mu uncertainty to ice resistance
- Sobol sensitivity analysis
- Joint uncertainty propagation (mu + compressive strength)
"""
from .distributions import FrictionDistribution, sample_mu
from .surrogate import ResistanceSurrogate, calibrate_surrogate
from .propagation import MonteCarloPropagation, propagate_uncertainty
from .sensitivity import SobolAnalysis, compute_sobol_indices
from .joint import JointUncertainty, propagate_joint_uncertainty

__all__ = [
    "FrictionDistribution",
    "sample_mu",
    "ResistanceSurrogate",
    "calibrate_surrogate",
    "MonteCarloPropagation",
    "propagate_uncertainty",
    "SobolAnalysis",
    "compute_sobol_indices",
    "JointUncertainty",
    "propagate_joint_uncertainty",
]
