"""Variance-based global-sensitivity indices (Saltelli/Jansen estimators).

The estimators are the standard pick-freeze forms

``S_i   = 1 - 0.5 * E[(f_B - f_AB_i)^2] / V``
``S_Ti  =      0.5 * E[(f_A - f_AB_i)^2] / V``

(Saltelli et al. 2010, "Variance based sensitivity analysis of model output",
Comput. Phys. Commun. 181, 259-270; Jansen 1999 for the first-index form).
``V`` is the sample variance of the pooled ``f_A``/``f_B`` evaluations.

The implementation is verified against the analytic Ishigami function, whose
indices are known in closed form (:func:`ishigami_exact_indices`).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "saltelli_indices",
    "bootstrap_indices",
    "ishigami",
    "ishigami_exact_indices",
]


def saltelli_indices(evaluate, A, B, AB) -> dict:
    """Estimate first-order and total Sobol' indices.

    Parameters
    ----------
    evaluate:
        Callable mapping an ``(n, d)`` unit-cube array to an ``(n,)`` array.
    A, B, AB:
        Saltelli design matrices from :meth:`ParameterSpace.saltelli_design`.

    Returns
    -------
    dict with ``first``, ``total`` (arrays of length ``d``), ``variance`` and
    the raw evaluations ``f_A``, ``f_B``.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    AB = np.asarray(AB, dtype=float)
    if A.ndim != 2 or A.shape != B.shape:
        raise ValueError("A and B must be equally shaped 2-D arrays")
    d = A.shape[1]
    if AB.shape != (d,) + A.shape:
        raise ValueError(f"AB must have shape {(d,) + A.shape}")

    f_A = np.asarray(evaluate(A), dtype=float).ravel()
    f_B = np.asarray(evaluate(B), dtype=float).ravel()
    if f_A.size != len(A) or f_B.size != len(B):
        raise ValueError("evaluate must return one value per sample")
    pooled = np.concatenate([f_A, f_B])
    variance = float(np.var(pooled, ddof=1))
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("response variance must be finite and positive")

    first = np.empty(d)
    total = np.empty(d)
    for i in range(d):
        f_AB = np.asarray(evaluate(AB[i]), dtype=float).ravel()
        if f_AB.size != len(A):
            raise ValueError("evaluate must return one value per sample")
        first[i] = 1.0 - 0.5 * np.mean((f_B - f_AB) ** 2) / variance
        total[i] = 0.5 * np.mean((f_A - f_AB) ** 2) / variance
    return {"first": first, "total": total, "variance": variance, "f_A": f_A, "f_B": f_B}


def bootstrap_indices(evaluate, A, B, AB, *, replicates: int = 64, seed: int = 0) -> dict:
    """Percentile bootstrap confidence intervals of the indices.

    Samples are resampled jointly (with replacement) over the ``n`` base rows so
    that the correlated ``f_A``/``f_B``/``f_AB_i`` values stay paired.  Returns
    ``{"first_ci": (2, d), "total_ci": (2, d)}`` with 2.5 %/97.5 % quantiles.
    """
    if replicates < 1:
        raise ValueError("replicates must be a positive integer")
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    AB = np.asarray(AB, dtype=float)
    d = A.shape[1]
    f_A = np.asarray(evaluate(A), dtype=float).ravel()
    f_B = np.asarray(evaluate(B), dtype=float).ravel()
    f_AB = np.stack([np.asarray(evaluate(AB[i]), dtype=float).ravel() for i in range(d)])
    rng = np.random.default_rng(seed)
    n = len(f_A)
    first = np.empty((replicates, d))
    total = np.empty((replicates, d))
    for r in range(replicates):
        idx = rng.integers(0, n, n)
        pooled = np.concatenate([f_A[idx], f_B[idx]])
        variance = float(np.var(pooled, ddof=1))
        if variance <= 0.0:
            first[r] = np.nan
            total[r] = np.nan
            continue
        first[r] = 1.0 - 0.5 * np.mean((f_B[idx][:, None] - f_AB[:, idx].T) ** 2, axis=0) / variance
        total[r] = 0.5 * np.mean((f_A[idx][:, None] - f_AB[:, idx].T) ** 2, axis=0) / variance
    return {
        "first_ci": np.nanpercentile(first, [2.5, 97.5], axis=0),
        "total_ci": np.nanpercentile(total, [2.5, 97.5], axis=0),
        "replicates": replicates,
    }


def ishigami(x) -> np.ndarray:
    """Ishigami function ``sin(x1) + 7 sin^2(x2) + 0.1 x3^4 sin(x1)`` in radians."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] != 3:
        raise ValueError("Ishigami function takes three inputs")
    return np.sin(x[:, 0]) + 7.0 * np.sin(x[:, 1]) ** 2 + 0.1 * x[:, 2] ** 4 * np.sin(x[:, 0])


def ishigami_exact_indices() -> dict:
    """Closed-form indices of the Ishigami function in ``[-pi, pi]^3``."""
    a, b = 7.0, 0.1
    pi = np.pi
    v1 = 0.5 * (1.0 + b * pi**4 / 5.0) ** 2
    v2 = a**2 / 8.0
    v13 = b**2 * pi**8 * (1.0 / 18.0 - 1.0 / 50.0)
    v = v1 + v2 + v13
    return {
        "first": np.array([v1 / v, v2 / v, 0.0]),
        "total": np.array([(v1 + v13) / v, v2 / v, v13 / v]),
        "variance": v,
    }