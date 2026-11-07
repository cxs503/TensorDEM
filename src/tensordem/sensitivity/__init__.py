"""Global-sensitivity, identifiability and uncertainty-propagation toolkit.

The package gathers the uncertainty-quantification (UQ) machinery used by the
ice-resistance study documented in ``docs/dem_uq_ice_resistance.md``:

``design``
    Parameter-space description and space-filling designs (Latin hypercube,
    scrambled Sobol' sequence, Saltelli ``A/B/AB`` replication designs).
``sobol``
    Saltelli/Jansen Monte-Carlo Sobol' estimators, bootstrap confidence
    intervals and the analytic Ishigami reference problem used to verify them.
``surrogate``
    Polynomial-chaos (Legendre, co-spectral least squares) and Gaussian-process
    surrogates with leave-one-out and held-out scores.
``identifiability``
    Finite-difference Jacobians, Fisher information, condition numbers,
    parameter correlation/collinearity diagnostics, profile likelihood and
    subset-reduction search.
``propagation``
    Laplace posterior from a linearised likelihood, posterior sampling,
    predictive intervals and interval-coverage tests.
``ice_chain``
    The DEM ice-field -> macro ice-resistance evaluation chain (3-D bonded
    wedge indentation plus a heap repose-angle case) that the UQ study drives.

The legacy resolution-sensitivity helpers of the 2-D indentation prototype are
re-exported from :mod:`tensordem.sensitivity.resolution` so that
``from tensordem.sensitivity import build_config`` keeps working.
"""

from .resolution import (  # noqa: F401  (legacy public API)
    build_config,
    compare_histories,
    simulate_case,
)

__all__ = ["build_config", "compare_histories", "simulate_case"]