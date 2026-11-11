"""Gaussian-process surrogate and expected-improvement acquisition.

Small, dependency-light (numpy/scipy only) implementation of the pieces the
calibration loop needs:

* :func:`latin_hypercube` -- space-filling design in the unit cube;
* :class:`GaussianProcess` -- zero-mean GP with an anisotropic Matern-5/2
  kernel, hyperparameters (length scales, signal variance, noise) fitted by
  maximising the log marginal likelihood on a small grid + coordinate search;
* :func:`expected_improvement` -- closed-form EI for a minimisation problem.

Deliberately *not* a wrapper around scikit-learn: the repository's DEM stack is
numpy/scipy only (see ``pyproject.toml``), and the calibration loop must stay
shippable on the same dependency set.  Every routine is checked in
``tests/test_dem_calibration.py`` against a smooth analytic test function whose
optimum is known.

Conventions
-----------
Inputs are ``(n, d)`` arrays in the *unit cube* ``[0, 1]^d``; the caller maps
them to physical bounds.  Responses are ``(n,)`` and are minimised.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "cholesky",
    "cho_solve_np",
    "latin_hypercube",
    "matern52",
    "GaussianProcess",
    "expected_improvement",
    "lower_confidence_bound",
]


# ---------------------------------------------------------------------------
# Dense linear algebra without LAPACK
# ---------------------------------------------------------------------------
# The GP systems here are tiny (n <= a few hundred) but they are solved
# thousands of times inside the hyperparameter search.  On the loongarch64
# cluster nodes the reference scipy/LAPACK build has been observed to crash the
# interpreter ("Bus error" / SIGSEGV inside ```_cholesky```) when the same
# routine is hammered from a heavily loaded host; a pure-NumPy Cholesky is both
# reproducible and immune to that failure mode, at no measurable cost at these
# sizes.

def cholesky(a: np.ndarray) -> np.ndarray:
    """Lower-triangular Cholesky factor, pure NumPy (raises on non-PD input)."""
    a = np.asarray(a, dtype=np.float64)
    m = a.shape[0]
    if a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square")
    if not np.all(np.isfinite(a)):
        raise ValueError("matrix must be finite")
    lower = np.zeros_like(a)
    for i in range(m):
        row = lower[i, :i]
        diag = a[i, i] - float(row @ row)
        if diag <= 0.0 or not math.isfinite(diag):
            raise ValueError("matrix is not positive definite")
        lower[i, i] = math.sqrt(diag)
        if i + 1 < m:
            lower[i + 1:, i] = (a[i + 1:, i] - lower[i + 1:, :i] @ row) / lower[i, i]
    return lower


def cho_solve_np(lower: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``L L^T x = b`` given the lower Cholesky factor ``L``."""
    b = np.asarray(b, dtype=np.float64)
    m = lower.shape[0]
    y = np.zeros_like(b)
    for i in range(m):
        y[i] = (b[i] - lower[i, :i] @ y[:i]) / lower[i, i]
    x = np.zeros_like(y)
    for i in range(m - 1, -1, -1):
        x[i] = (y[i] - lower[i + 1:, i] @ x[i + 1:]) / lower[i, i]
    return x


def latin_hypercube(n: int, d: int, *, seed: int = 0) -> np.ndarray:
    """Latin-hypercube design in ``[0, 1]^d`` (deterministic given ``seed``)."""
    if n < 1 or d < 1:
        raise ValueError("n and d must be positive")
    rng = np.random.default_rng(seed)
    cut = (np.arange(n)[:, None] + rng.random((n, d))) / n
    out = np.empty((n, d))
    for j in range(d):
        out[:, j] = cut[rng.permutation(n), j]
    return out


def _pairwise(a: np.ndarray, b: np.ndarray, lengthscales: np.ndarray) -> np.ndarray:
    diff = (a[:, None, :] - b[None, :, :]) / lengthscales[None, None, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def matern52(r: np.ndarray) -> np.ndarray:
    """Matern-5/2 correlation as a function of the scaled distance."""
    s = math.sqrt(5.0) * r
    return (1.0 + s + s * s / 3.0) * np.exp(-s)


@dataclass
class GaussianProcess:
    """Zero-mean GP with anisotropic Matern-5/2 kernel (minimisation)."""

    lengthscales: np.ndarray | None = None
    signal_var: float = 1.0
    noise_var: float = 1e-8
    x: np.ndarray = field(default_factory=lambda: np.zeros((0, 1)))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y_mean: float = 0.0
    _chol: tuple | None = None

    # -- kernel -------------------------------------------------------
    def kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ls = self.lengthscales if self.lengthscales is not None else np.ones(a.shape[1])
        return self.signal_var * matern52(_pairwise(a, b, ls))

    # -- fitting ------------------------------------------------------
    def fit(self, x: np.ndarray, y: np.ndarray, *, optimise: bool = True,
            n_restarts: int = 3, seed: int = 0) -> "GaussianProcess":
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).ravel()
        if x.shape[0] != y.size:
            raise ValueError("x and y must have the same number of rows")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            # LAPACK (Cholesky) does not raise on NaN input, it crashes the
            # interpreter; a failed forward run must never reach this point.
            raise ValueError("GP training data must be finite")
        self.x, self.y = x, y
        self.y_mean = float(y.mean()) if y.size else 0.0
        d = x.shape[1]
        if self.lengthscales is None:
            self.lengthscales = np.full(d, 0.35)
        if not optimise or x.shape[0] < 4:
            self._recompute()
            return self
        best = self._neg_log_marginal(self.lengthscales, self.signal_var, self.noise_var)
        rng = np.random.default_rng(seed)
        scale = float(np.var(y)) if np.var(y) > 1e-30 else 1.0
        # grid + random multi-start coordinate search on log-hyperparameters
        cands = [self.lengthscales.copy()]
        for _ in range(n_restarts):
            cands.append(np.exp(rng.uniform(math.log(0.08), math.log(2.0), d)))
        for ls0 in cands:
            ls = ls0.copy()
            s2 = scale * float(np.exp(rng.uniform(-1.0, 1.0))) if ls0 is not cands[0] else self.signal_var
            nv = self.noise_var
            cur = self._neg_log_marginal(ls, s2, nv)
            for step in (0.5, 0.25, 0.125):
                improved = True
                while improved:
                    improved = False
                    for j in range(d):
                        for mult in (math.exp(step), math.exp(-step)):
                            trial = ls.copy()
                            trial[j] = min(max(trial[j] * mult, 1e-2), 50.0)
                            val = self._neg_log_marginal(trial, s2, nv)
                            if val < cur - 1e-9:
                                ls, cur, improved = trial, val, True
                    for mult in (math.exp(step), math.exp(-step)):
                        trial_s2 = min(max(s2 * mult, 1e-8 * scale), 1e8 * scale)
                        if trial_s2 == s2:
                            continue
                        val = self._neg_log_marginal(ls, trial_s2, nv)
                        if val < cur - 1e-9:
                            s2, cur, improved = trial_s2, val, True
                if cur < best - 1e-9:
                    best, self.lengthscales, self.signal_var = cur, ls.copy(), s2
        self._recompute()
        return self

    def _neg_log_marginal(self, ls: np.ndarray, s2: float, nv: float) -> float:
        if not (np.all(np.isfinite(ls)) and np.isfinite(s2) and np.isfinite(nv)):
            return 1e12
        if s2 <= 1e-12 or nv <= 0.0:
            return 1e12
        self.lengthscales, self.signal_var, self.noise_var = ls, s2, nv
        try:
            self._recompute()
            resid = self.y - self.y_mean
            alpha = cho_solve_np(self._chol, resid)
            logdet = 2.0 * float(np.sum(np.log(np.diag(self._chol))))
        except Exception:
            return 1e12
        n = self.y.size
        return 0.5 * float(resid @ alpha) + 0.5 * logdet + 0.5 * n * math.log(2 * math.pi)

    def _recompute(self) -> None:
        if self.x.shape[0] == 0:
            self._chol = None
            return
        k = self.kernel(self.x, self.x)
        if not np.all(np.isfinite(k)):
            raise np.linalg.LinAlgError("non-finite covariance (NaN hyperparameters)")
        k = 0.5 * (k + k.T)
        k = k + (max(self.noise_var, 1e-12) + 1e-10) * np.eye(self.x.shape[0])
        for jitter in (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
            try:
                self._chol = cholesky(k + jitter * np.eye(k.shape[0]))
                return
            except ValueError:
                continue
        raise np.linalg.LinAlgError("GP covariance is not positive definite")

    # -- prediction ---------------------------------------------------
    def predict(self, xs: np.ndarray, *, return_std: bool = True):
        xs = np.atleast_2d(np.asarray(xs, dtype=np.float64))
        if self.x.shape[0] == 0:
            mean = np.full(xs.shape[0], self.y_mean)
            std = np.full(xs.shape[0], math.sqrt(self.signal_var))
            return (mean, std) if return_std else mean
        k_s = self.kernel(xs, self.x)
        mean = self.y_mean + k_s @ cho_solve_np(self._chol, self.y - self.y_mean)
        if not return_std:
            return mean
        v = cho_solve_np(self._chol, k_s.T)
        var = self.signal_var - np.einsum("ij,ji->i", k_s, v)
        std = np.sqrt(np.maximum(var, 1e-18))
        return mean, std

    def sample_posterior(self, xs: np.ndarray, n_samples: int, *,
                         seed: int = 0) -> np.ndarray:
        """Joint posterior draws at ``xs`` (used by the MCMC surrogate)."""
        xs = np.atleast_2d(np.asarray(xs, dtype=np.float64))
        mean, _ = self.predict(xs)
        cov = self.kernel(xs, xs) + 1e-10 * np.eye(xs.shape[0])
        lower = cholesky(0.5 * (cov + cov.T))
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n_samples, xs.shape[0]))
        return mean[None, :] + z @ lower.T


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr
    return ndtr(z)


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def expected_improvement(mean: np.ndarray, std: np.ndarray, best: float,
                         *, xi: float = 0.01) -> np.ndarray:
    """EI for minimisation, closed form (Jones et al. 1998)."""
    std = np.maximum(std, 1e-12)
    imp = best - mean - xi
    z = imp / std
    return imp * _norm_cdf(z) + std * _norm_pdf(z)


def lower_confidence_bound(mean: np.ndarray, std: np.ndarray, *, kappa: float = 2.0):
    """LCB for minimisation; the deterministic tie-break used by the loop."""
    return mean - kappa * std