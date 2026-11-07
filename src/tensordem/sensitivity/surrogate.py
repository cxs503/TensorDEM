"""Surrogate models used as cheap stand-ins for the DEM ice-resistance chain.

Two surrogates are implemented because they carry different assumptions and can
be cross-checked against each other:

* :class:`PolynomialChaos` -- total-degree Legendre chaos on the unit cube with
  orthonormal (``E[phi_i phi_j] = delta_ij`` for the uniform measure) basis
  functions.  Truncated-chaos coefficients give the Sobol' decomposition
  directly, so the indices are a *post-processing* of a single least-squares
  fit.
* :class:`GaussianProcess` -- ARD radial-basis kernel with the log marginal
  likelihood optimised over length scales, signal variance and nugget.  It
  supplies both a mean prediction and a predictive variance, which the
  propagation stage uses for model-form uncertainty.

Both expose ``predict`` and :class:`FitScore` (:meth:`r2` on held-out data), and
:meth:`PolynomialChaos.sobol_indices` returns analytic (coefficient-based)
indices.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from numpy.polynomial import legendre
from scipy.optimize import minimize

__all__ = ["PolynomialChaos", "GaussianProcess", "r2_score"]


def r2_score(truth, prediction) -> float:
    """Coefficient of determination ``1 - SS_res / SS_tot``."""
    truth = np.asarray(truth, dtype=float).ravel()
    prediction = np.asarray(prediction, dtype=float).ravel()
    if truth.shape != prediction.shape:
        raise ValueError("truth and prediction must have the same shape")
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    if ss_tot <= 0.0:
        raise ValueError("truth has zero variance; R^2 is undefined")
    return 1.0 - float(np.sum((truth - prediction) ** 2)) / ss_tot


class PolynomialChaos:
    """Total-degree Legendre chaos fitted by least squares on unit-cube samples.

    Parameters
    ----------
    dimension:
        Number of inputs.
    degree:
        Maximum total degree of the retained multi-indices.
    ridge:
        Optional Tikhonov regularisation (relative to the mean diagonal of the
        Gram matrix).  ``0`` gives the ordinary least-squares fit.
    """

    def __init__(self, dimension: int, degree: int = 2, *, ridge: float = 0.0):
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        if degree < 1:
            raise ValueError("degree must be >= 1")
        if ridge < 0.0:
            raise ValueError("ridge must be nonnegative")
        self.dimension = int(dimension)
        self.degree = int(degree)
        self.ridge = float(ridge)
        self.multi_index = _total_degree_indices(self.dimension, self.degree)
        self.coefficients: np.ndarray | None = None
        self.y_mean = 0.0

    # -- basis -------------------------------------------------------------
    def basis(self, U) -> np.ndarray:
        U = np.atleast_2d(np.asarray(U, dtype=float))
        if U.shape[1] != self.dimension:
            raise ValueError(f"expected {self.dimension} columns")
        if np.any(U < 0.0) or np.any(U > 1.0):
            raise ValueError("PCE basis is defined on the unit cube")
        shifted = 2.0 * U - 1.0
        # Univariate orthonormal Legendre polynomials sqrt(2 n + 1) P_n(t) are
        # cached because the same per-dimension columns are reused by every
        # multi-index.
        univariate = [
            [_legendre_orthonormal(shifted[:, dim], power)
             for power in range(self.degree + 1)]
            for dim in range(self.dimension)
        ]
        matrix = np.ones((U.shape[0], len(self.multi_index)))
        for k, alpha in enumerate(self.multi_index):
            column = np.ones(U.shape[0])
            for dim, power in enumerate(alpha):
                if power:
                    column = column * univariate[dim][power]
            matrix[:, k] = column
        return matrix

    # -- fit / predict -----------------------------------------------------
    def fit(self, U, y) -> "PolynomialChaos":
        U = np.atleast_2d(np.asarray(U, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if len(U) != len(y):
            raise ValueError("U and y must have the same number of rows")
        if len(y) < len(self.multi_index):
            raise ValueError("number of samples smaller than the chaos basis size")
        design = self.basis(U)
        self.y_mean = float(y.mean())
        target = y - self.y_mean
        gram = design.T @ design
        if self.ridge > 0.0:
            scale = float(np.mean(np.diag(gram)))
            gram = gram + self.ridge * scale * np.eye(gram.shape[0])
        self.coefficients = np.linalg.solve(gram, design.T @ target)
        self._design = design
        self._target = target
        return self

    def predict(self, U) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("fit() must be called before predict()")
        return self.y_mean + self.basis(U) @ self.coefficients

    def loo_error(self) -> dict:
        """Leave-one-out score from the hat matrix (no refitting needed)."""
        if self.coefficients is None:
            raise RuntimeError("fit() must be called first")
        design = self._design
        residual = self._target - design @ self.coefficients
        gram_inv = np.linalg.pinv(design.T @ design)
        leverage = np.einsum("ij,jk,ik->i", design, gram_inv, design)
        denom = np.clip(1.0 - leverage, 1e-12, None)
        loo = residual / denom
        ss_tot = float(np.sum((self._target - self._target.mean()) ** 2))
        return {
            "loo_rmse": float(np.sqrt(np.mean(loo**2))),
            "loo_r2": 1.0 - float(np.sum(loo**2)) / ss_tot,
            "max_leverage": float(leverage.max()),
        }

    # -- indices -----------------------------------------------------------
    def sobol_indices(self) -> dict:
        """Coefficient-based Sobol' indices of the truncated chaos expansion."""
        if self.coefficients is None:
            raise RuntimeError("fit() must be called first")
        squared = self.coefficients**2
        total_variance = float(squared.sum())
        if total_variance <= 0.0:
            raise ValueError("chaos expansion has zero variance")
        first = np.zeros(self.dimension)
        total = np.zeros(self.dimension)
        for k, alpha in enumerate(self.multi_index):
            support = [i for i, power in enumerate(alpha) if power > 0]
            if not support:
                continue
            for i in support:
                total[i] += squared[k]
            if len(support) == 1:
                first[support[0]] += squared[k]
        return {"first": first / total_variance, "total": total / total_variance,
                "variance": total_variance,
                "coefficient_of_variation": math.sqrt(total_variance) / abs(self.y_mean)
                if self.y_mean else float("nan")}


class GaussianProcess:
    """Zero-mean (after standardisation) ARD-RBF Gaussian process.

    The kernel is ``sigma_f^2 * exp(-0.5 * sum_i ((u_i - v_i)/l_i)^2) + nugget``.
    Hyper-parameters are fitted by maximising the log marginal likelihood with
    box constraints on the length scales, which keeps the optimiser away from
    the degenerate ``l -> 0`` interpolation regime.
    """

    def __init__(self, dimension: int, *, length_scale_bounds=(0.02, 10.0),
                 nugget_bounds=(1e-10, 1e-2), max_iter: int = 120):
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        self.dimension = int(dimension)
        self.length_scale_bounds = tuple(length_scale_bounds)
        self.nugget_bounds = tuple(nugget_bounds)
        self.max_iter = int(max_iter)
        self.length_scales = np.full(self.dimension, 0.3)
        self.sigma_f = 1.0
        self.nugget = 1e-8
        self._X = None
        self._alpha = None
        self._chol = None
        self.y_mean = 0.0
        self.y_std = 1.0

    # -- kernel ------------------------------------------------------------
    def _kernel(self, A, B, length_scales, sigma_f, nugget_diag=0.0):
        diff = (A[:, None, :] - B[None, :, :]) / length_scales
        dist2 = np.sum(diff**2, axis=-1)
        K = sigma_f**2 * np.exp(-0.5 * dist2)
        if nugget_diag:
            K = K + nugget_diag * np.eye(len(A))
        return K

    def _negative_log_likelihood(self, params, X, y):
        length_scales = np.exp(params[: self.dimension])
        sigma_f = float(np.exp(params[self.dimension]))
        nugget = float(np.exp(params[self.dimension + 1]))
        K = self._kernel(X, X, length_scales, sigma_f, nugget)
        try:
            # NumPy's LAPACK entry point is used deliberately: the SciPy
            # ``scipy.linalg.cholesky`` wrapper aborted the interpreter
            # (SIGBUS) on this control node for large kernel matrices.
            chol = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = _solve_cholesky(chol, y)
        return float(0.5 * y @ alpha + np.sum(np.log(np.diag(chol)))
                     + 0.5 * len(y) * math.log(2.0 * math.pi))

    def fit(self, U, y) -> "GaussianProcess":
        X = np.atleast_2d(np.asarray(U, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[1] != self.dimension:
            raise ValueError(f"expected {self.dimension} columns")
        if len(X) != len(y):
            raise ValueError("U and y must have the same number of rows")
        self.y_mean = float(y.mean())
        self.y_std = float(y.std())
        if not np.isfinite(self.y_std) or self.y_std <= 0.0:
            raise ValueError("response has zero variance")
        target = (y - self.y_mean) / self.y_std

        lo_l = math.log(self.length_scale_bounds[0])
        hi_l = math.log(self.length_scale_bounds[1])
        lo_n = math.log(self.nugget_bounds[0])
        hi_n = math.log(self.nugget_bounds[1])
        bounds = [(lo_l, hi_l)] * self.dimension + [(math.log(1e-3), math.log(1e3)),
                                                    (lo_n, hi_n)]
        x0 = np.concatenate([np.full(self.dimension, math.log(0.3)), [0.0],
                             [math.log(max(self.nugget_bounds[0] * 10, 1e-8))]])
        result = minimize(
            self._negative_log_likelihood, x0, args=(X, target), method="L-BFGS-B",
            bounds=bounds, options={"maxiter": self.max_iter, "ftol": 1e-9},
        )
        params = result.x if np.isfinite(result.fun) else x0
        self.length_scales = np.exp(params[: self.dimension])
        self.sigma_f = float(np.exp(params[self.dimension]))
        self.nugget = float(np.exp(params[self.dimension + 1]))
        self._X = X
        K = self._kernel(X, X, self.length_scales, self.sigma_f, self.nugget)
        self._chol = np.linalg.cholesky(K)
        self._alpha = _solve_cholesky(self._chol, target)
        self.log_marginal_likelihood = float(-result.fun)
        return self

    def predict(self, U, *, return_std: bool = False):
        if self._X is None:
            raise RuntimeError("fit() must be called before predict()")
        X = np.atleast_2d(np.asarray(U, dtype=float))
        if X.shape[1] != self.dimension:
            raise ValueError(f"expected {self.dimension} columns")
        Ks = self._kernel(X, self._X, self.length_scales, self.sigma_f)
        mean = Ks @ self._alpha
        if not return_std:
            return self.y_mean + self.y_std * mean
        v = _solve_cholesky(self._chol, Ks.T)
        var = self.sigma_f**2 - np.sum(Ks.T * v, axis=0)
        var = np.clip(var, 1e-14, None)
        return self.y_mean + self.y_std * mean, self.y_std * np.sqrt(var)


def _solve_cholesky(chol, rhs):
    """Solve ``K x = rhs`` from a lower Cholesky factor (NumPy only)."""
    return np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))


def _total_degree_indices(dimension: int, degree: int) -> list[tuple[int, ...]]:
    """All multi-indices with ``sum(alpha) <= degree`` (graded ordering)."""
    indices = []
    for total in range(degree + 1):
        for combo in itertools.product(range(total + 1), repeat=dimension):
            if sum(combo) == total:
                indices.append(tuple(int(c) for c in combo))
    return indices


def _legendre_orthonormal(t: np.ndarray, order: int) -> np.ndarray:
    """``sqrt(2 n + 1) P_n(t)`` for ``t`` given on ``[-1, 1]``."""
    coeffs = np.zeros(order + 1)
    coeffs[order] = 1.0
    return math.sqrt(2.0 * order + 1.0) * legendre.legval(t, coeffs)