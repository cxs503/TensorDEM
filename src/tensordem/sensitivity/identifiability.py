"""Identifiability diagnostics for one-dimensional calibration curves.

Given a model that maps parameters ``u`` (unit cube) to a vector of observations
``y`` (e.g. the ``m`` samples of one force-displacement calibration trace), the
local information content is the Fisher information matrix of the linearised
Gaussian likelihood

``FIM = J^T Sigma^-1 J``,  ``Sigma = diag(sigma_j^2)``,  ``J = dy/du``

Because the parameters are always handled in *unit-cube* coordinates the
condition number of the FIM is scale free.  The inverse FIM is the local
posterior covariance, so the parameter correlation matrix and the near-null
eigenvector of the FIM both describe which parameter combinations a single
calibration curve cannot separate.

Everything here is a *local* (linearised) statement: a large condition number
means the curve cannot resolve that direction at the stated noise level, not
that no global solution exists.  :func:`profile_likelihood` adds the global
scan for the parameters that matter.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import least_squares

__all__ = [
    "finite_difference_jacobian",
    "fisher_information",
    "condition_number",
    "correlation_matrix",
    "collinear_pairs",
    "null_direction",
    "profile_likelihood",
    "best_identifiable_subset",
]


def finite_difference_jacobian(model, u0, *, rel_step: float = 1e-5) -> np.ndarray:
    """Central-difference Jacobian of ``model`` at the unit-cube point ``u0``.

    The step is shrunk near the box faces so every evaluation stays inside the
    unit cube (``model`` is only defined there).
    """
    u0 = np.asarray(u0, dtype=float).ravel()
    if np.any(u0 < 0.0) or np.any(u0 > 1.0):
        raise ValueError("u0 must lie in the unit cube")
    if rel_step <= 0.0:
        raise ValueError("rel_step must be positive")
    base = np.asarray(model(u0), dtype=float).ravel()
    J = np.empty((base.size, u0.size))
    for i in range(u0.size):
        step = rel_step * max(u0[i], 0.05)
        step = min(step, 0.5 * (1.0 - u0[i]), 0.5 * u0[i]) or rel_step * 0.05
        plus = u0.copy()
        minus = u0.copy()
        plus[i] += step
        minus[i] -= step
        J[:, i] = (np.asarray(model(plus), dtype=float).ravel()
                   - np.asarray(model(minus), dtype=float).ravel()) / (2.0 * step)
    return J


def fisher_information(J, sigma) -> np.ndarray:
    """``J^T diag(1/sigma^2) J`` with per-observation standard deviations."""
    J = np.asarray(J, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (J.shape[0],))
    if np.any(sigma <= 0.0) or not np.all(np.isfinite(sigma)):
        raise ValueError("sigma must be finite and positive")
    return J.T @ (J / sigma[:, None] ** 2)


def condition_number(FIM) -> float:
    """Spectral condition number of the (symmetric) FIM."""
    FIM = np.asarray(FIM, dtype=float)
    eigenvalues = np.linalg.eigvalsh(FIM)
    if eigenvalues[0] <= 0.0:
        return float("inf")
    return float(eigenvalues[-1] / eigenvalues[0])


def correlation_matrix(FIM) -> np.ndarray:
    """Parameter correlation matrix implied by ``Cov = FIM^-1``."""
    FIM = np.asarray(FIM, dtype=float)
    try:
        covariance = np.linalg.inv(FIM)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(FIM)
    scale = np.maximum(np.sqrt(np.clip(np.diag(covariance), 0.0, None)), 1e-150)
    corr = np.clip(covariance / np.outer(scale, scale), -1.0, 1.0)
    # A perfectly unidentifiable direction has zero variance; report it as a
    # zero row/column rather than as 0/0.
    return np.nan_to_num(corr, nan=0.0)


def collinear_pairs(corr, names=None, *, threshold: float = 0.9) -> list[dict]:
    """Pairs ``|rho_ij| >= threshold`` sorted by decreasing |rho|."""
    corr = np.asarray(corr, dtype=float)
    d = corr.shape[0]
    names = list(names) if names is not None else [f"p{i}" for i in range(d)]
    pairs = []
    for i, j in itertools.combinations(range(d), 2):
        rho = float(corr[i, j])
        if abs(rho) >= threshold:
            pairs.append({"pair": [names[i], names[j]], "index": [i, j],
                          "correlation": rho, "abs_correlation": abs(rho)})
    return sorted(pairs, key=lambda item: -item["abs_correlation"])


def null_direction(FIM, names=None) -> dict:
    """Smallest-eigenvalue direction: the least identifiable parameter combination.

    Returns the eigenvalue, its gap to the next eigenvalue, and the normalised
    eigenvector components (largest-magnitude entries first).
    """
    FIM = np.asarray(FIM, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(FIM)
    names = list(names) if names is not None else [f"p{i}" for i in range(FIM.shape[0])]
    v = eigenvectors[:, 0]
    sign = np.sign(v[np.argmax(np.abs(v))]) or 1.0
    v = v * sign
    order = np.argsort(-np.abs(v))
    return {
        "smallest_eigenvalue": float(eigenvalues[0]),
        "next_eigenvalue": float(eigenvalues[1]) if len(eigenvalues) > 1 else None,
        "eigenvalue_ratio": float(eigenvalues[1] / eigenvalues[0]) if eigenvalues[0] > 0 else float("inf"),
        "components": {names[i]: float(v[i]) for i in order},
    }


def profile_likelihood(model, y_obs, sigma, u0, *, grid=None, names=None) -> list[dict]:
    """Profile-likelihood scan of every parameter on a unit-cube grid.

    For each grid value of parameter ``i`` the remaining parameters are
    re-optimised (bounded least squares) and the resulting chi-square is
    recorded.  The 95 % confidence interval is the connected set where
    ``chi2 <= chi2_min + 3.841`` (one degree of freedom).  An interval that
    touches a box face is reported as ``bounded_by_prior`` -- the classic
    signature of a parameter that the calibration curve cannot identify.
    """
    u0 = np.asarray(u0, dtype=float).ravel()
    y_obs = np.asarray(y_obs, dtype=float).ravel()
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), y_obs.shape)
    d = u0.size
    names = list(names) if names is not None else [f"p{i}" for i in range(d)]
    grid = np.linspace(0.02, 0.98, 9) if grid is None else np.asarray(grid, dtype=float)

    def chi2(u):
        residual = (np.asarray(model(u), dtype=float).ravel() - y_obs) / sigma
        return float(residual @ residual)

    free_index = lambda k: [i for i in range(d) if i != k]  # noqa: E731

    def full_residual(z):
        return (np.asarray(model(z), dtype=float).ravel() - y_obs) / sigma

    base = least_squares(full_residual, u0, bounds=(0.0, 1.0))
    base_chi2 = float(base.fun @ base.fun)

    profiles = []
    for k in range(d):
        others = free_index(k)
        chi2_scan = []
        start = base.x[others]
        for value in grid:
            def residual(z, value=value, others=others):
                u = u0.copy()
                u[k] = value
                u[others] = z
                return (np.asarray(model(u), dtype=float).ravel() - y_obs) / sigma

            # Warm start from the previous grid point: the profile is smooth, so
            # this cuts the optimiser work by roughly an order of magnitude.
            fit = least_squares(residual, start, bounds=(0.0, 1.0), max_nfev=120)
            start = fit.x
            chi2_scan.append(float(fit.fun @ fit.fun))
        chi2_scan = np.asarray(chi2_scan)
        feasible = chi2_scan <= base_chi2 + 3.841
        if feasible.any():
            lo = float(grid[feasible].min())
            hi = float(grid[feasible].max())
        else:
            lo = hi = float(grid[int(np.argmin(chi2_scan))])
        touches = bool(feasible[0] or feasible[-1])
        profiles.append({
            "parameter": names[k],
            "grid": grid.tolist(),
            "chi2": chi2_scan.tolist(),
            "best_fit_value": float(base.x[k]),
            "interval95": [lo, hi],
            "width": float(hi - lo),
            "bounded_by_prior": touches,
        })
    return profiles


def best_identifiable_subset(FIM, names=None, *, sizes=(1, 2, 3, 4, 5)) -> dict:
    """Search parameter subsets by the condition number of the FIM submatrix.

    Because ``FIM_S`` for a subset ``S`` is just the corresponding principal
    submatrix, the search is exhaustive and cheap for ``d <= 8``.  The returned
    per-size winners let the study recommend a calibration-plan reduction with a
    quantified conditioning gain.
    """
    FIM = np.asarray(FIM, dtype=float)
    d = FIM.shape[0]
    names = list(names) if names is not None else [f"p{i}" for i in range(d)]
    results = []
    for size in sizes:
        if size > d:
            continue
        best = None
        for subset in itertools.combinations(range(d), size):
            sub = FIM[np.ix_(subset, subset)]
            cond = condition_number(sub)
            if best is None or cond < best["condition_number"]:
                best = {"size": size,
                        "parameters": [names[i] for i in subset],
                        "index": list(subset),
                        "condition_number": cond}
        if best is not None:
            results.append(best)
    full_cond = condition_number(FIM)
    for entry in results:
        entry["condition_gain_vs_full"] = full_cond / entry["condition_number"]
    return {"full_condition_number": full_cond, "per_size": results}
