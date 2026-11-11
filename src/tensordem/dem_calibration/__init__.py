"""Automatic DEM contact-parameter calibration (TensorDEM p284w3b-n05).

The package implements the closed inverse loop

    forward DEM repose test  ->  macro responses (theta_r, phi_solid)
        ->  Gaussian-process surrogate + expected-improvement BayesOpt
        ->  posterior over (mu, e, k_n/k_s)  ->  validation on independent cases

See :mod:`tensordem.dem_calibration.heap` (physics), ``.surrogate`` (GP/EI),
``.inverse`` (BayesOpt + MCMC) and ``.pipeline`` (objective, verdicts).
"""
from .heap import HeapCase, HeapError, build_pile, repose_angle, run_heap, hash_positions
from .surrogate import GaussianProcess, latin_hypercube, expected_improvement
from .inverse import calibrate, posterior_sample

__all__ = [
    "HeapCase", "HeapError", "build_pile", "repose_angle", "run_heap",
    "hash_positions", "GaussianProcess", "latin_hypercube",
    "expected_improvement", "calibrate", "posterior_sample",
]
