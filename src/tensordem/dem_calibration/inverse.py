"""Inverse problem: Bayesian-optimisation calibration + posterior sampling.

Two stages, both on the unit cube ``[0, 1]^d`` internally:

1. **Search** -- :func:`bayesian_optimize`: a GP surrogate (:mod:`surrogate`)
   trained on the visited points, expected improvement acquisition maximised on
   a dense candidate set, batch size 1.  The initial design is a
   Latin-hypercube sweep, so the loop is a genuine sequential design and not a
   grid disguised as BayesOpt.
2. **Uncertainty quantification** -- :func:`posterior_sample`: random-walk
   Metropolis-Hastings over the *response-surface* posterior.  For each
   macroscopic response ``d_i`` with observation noise ``sigma_i`` the emulator
   gives ``m_i(θ)`` with standard deviation ``s_i(θ)``, so

       log L(θ) = -1/2 Σ_i (m_i(θ) - d_i)^2 / (sigma_i^2 + s_i(θ)^2)

   and the prior is uniform on the box.  Emulator error therefore *widens* the
   posterior instead of being ignored, which is the honest treatment for a
   surrogate-based calibration (Kennedy & O'Hagan 2001).

Everything is deterministic given ``seed`` (NumPy ``default_rng``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .surrogate import (
    GaussianProcess,
    expected_improvement,
    latin_hypercube,
    lower_confidence_bound,
)

__all__ = [
    "ParameterSpace",
    "bayesian_optimize",
    "posterior_sample",
    "calibrate",
]


@dataclass(frozen=True)
class ParameterSpace:
    """Box bounds for the three calibrated contact parameters."""

    names: tuple[str, ...] = ("mu", "restitution", "stiffness_ratio")
    lower: tuple[float, ...] = (0.10, 0.10, 1.0)
    upper: tuple[float, ...] = (0.80, 0.90, 3.0)

    def __post_init__(self) -> None:
        if len(self.names) != len(self.lower) or len(self.names) != len(self.upper):
            raise ValueError("names/lower/upper must have the same length")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("lower bounds must be strictly below upper bounds")

    @property
    def dim(self) -> int:
        return len(self.names)

    def to_unit(self, params: Sequence[float]) -> np.ndarray:
        lo = np.asarray(self.lower, dtype=float)
        hi = np.asarray(self.upper, dtype=float)
        return (np.asarray(params, dtype=float) - lo) / (hi - lo)

    def to_physical(self, u: Sequence[float]) -> dict[str, float]:
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        lo = np.asarray(self.lower, dtype=float)
        hi = np.asarray(self.upper, dtype=float)
        phys = lo + u * (hi - lo)
        return {name: float(value) for name, value in zip(self.names, phys)}


def _acquisition_candidates(space: ParameterSpace, n: int, seed: int,
                            x_seen: np.ndarray) -> np.ndarray:
    """Random candidates plus a jittered resample around the incumbent."""
    rng = np.random.default_rng(seed)
    cand = latin_hypercube(n, space.dim, seed=seed)
    if x_seen.shape[0]:
        best = x_seen[int(np.argmin(x_seen[:, -1]))][:-1] if x_seen.shape[1] == space.dim + 1 \
            else x_seen[0]
        local = best[None, :] + rng.standard_normal((n // 4, space.dim)) * 0.05
        cand = np.vstack([cand, np.clip(local, 0.0, 1.0)])
    return cand


def bayesian_optimize(objective: Callable[[dict[str, float]], float],
                      space: ParameterSpace, *, n_init: int = 8, n_iter: int = 32,
                      seed: int = 0, n_candidates: int = 512,
                      y_stop: float | None = None,
                      log: list[dict] | None = None) -> dict:
    """Sequential model-based minimisation of ``objective`` on ``space``.

    ``objective`` maps a physical-parameter dict to a scalar cost (lower is
    better).  Returns the visited design/response table, the fitted GPs and a
    stop report (``converged_iteration`` is the first iteration at which the
    incumbent dropped below ``y_stop``, ``None`` if never).
    """
    rng = np.random.default_rng(seed)
    design = latin_hypercube(n_init, space.dim, seed=seed)
    xs = np.zeros((0, space.dim))
    ys = np.zeros(0)
    table: list[dict] = []
    converged_iteration: int | None = None
    converged_wall = 0

    def evaluate(u: np.ndarray) -> float:
        params = space.to_physical(u)
        value = float(objective(params))
        table.append({"iteration": len(table), "unit": [float(v) for v in u],
                      "params": params, "objective": value})
        if log is not None:
            log.append(table[-1])
        return value

    for k in range(n_init):
        xs = np.vstack([xs, design[k]])
        ys = np.append(ys, evaluate(design[k]))
        if y_stop is not None and ys[-1] <= y_stop and converged_iteration is None:
            converged_iteration = 0

    gp = GaussianProcess().fit(xs, ys, optimise=False, seed=seed)
    for it in range(1, n_iter + 1):
        gp = GaussianProcess().fit(xs, ys, optimise=True, seed=seed + it)
        cand = _acquisition_candidates(space, n_candidates, seed + 1000 * it,
                                       np.column_stack([xs, ys]))
        mean, std = gp.predict(cand)
        best = float(ys.min())
        ei = expected_improvement(mean, std, best)
        # EI is the primary rank, LCB the deterministic tie-break: EI can be
        # numerically flat once the surrogate is confident, and a flat EI would
        # otherwise be resolved by candidate order (an accidental grid search).
        score = ei - 1e-6 * lower_confidence_bound(mean, std)
        pick = int(np.argmax(score))
        x_new = np.clip(cand[pick], 0.0, 1.0)
        xs = np.vstack([xs, x_new])
        ys = np.append(ys, evaluate(x_new))
        if y_stop is not None and ys[-1] <= y_stop:
            converged_wall += 1
            if converged_iteration is None:
                converged_iteration = it
        else:
            converged_wall = 0

    best = int(np.argmin(ys))
    return {
        "x": xs,
        "y": ys,
        "best_index": best,
        "best_params": space.to_physical(xs[best]),
        "best_objective": float(ys[best]),
        "n_evaluations": int(ys.size),
        "converged_iteration": converged_iteration,
        "table": table,
        "gp": gp,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Posterior sampling on the response surface
# ---------------------------------------------------------------------------

def posterior_sample(x_unit: np.ndarray, responses: dict[str, np.ndarray],
                     targets: dict[str, float], sigmas: dict[str, float],
                     space: ParameterSpace, *, n_samples: int = 4000,
                     burn_in: int = 1000, thin: int = 4, proposal_scale: float = 0.08,
                     seed: int = 0) -> dict:
    """Random-walk Metropolis over the surrogate response surface.

    ``responses`` maps a response name to its value at every row of ``x_unit``;
    every response gets its own GP emulator.  Returns the flattened posterior
    mean/median/95 % credible interval per parameter plus the acceptance rate.
    """
    names = list(responses)
    x_unit = np.atleast_2d(np.asarray(x_unit, dtype=float))
    keep = np.all([np.isfinite(responses[n]) for n in names], axis=0)
    if int(keep.sum()) < 6:
        raise ValueError("fewer than 6 usable design points for the posterior")
    x_unit = x_unit[keep]
    responses = {n: np.asarray(responses[n], dtype=float)[keep] for n in names}
    gps = {}
    for name in names:
        gps[name] = GaussianProcess().fit(x_unit, responses[name], optimise=True, seed=seed)

    def log_like(u: np.ndarray) -> float:
        u = np.atleast_2d(u)
        total = 0.0
        for name in names:
            mean, std = gps[name].predict(u)
            var = sigmas[name] ** 2 + std ** 2
            total -= 0.5 * float(np.sum((mean - targets[name]) ** 2 / var))
        return total

    rng = np.random.default_rng(seed)
    current = np.clip(x_unit[int(np.argmin(np.sum(
        [((responses[n] - targets[n]) / max(targets[n], 1e-12)) ** 2 for n in names], axis=0)))],
        0.02, 0.98)
    lp = log_like(current)
    draws: list[np.ndarray] = []
    accepted = 0
    total_draws = burn_in + n_samples * thin
    for step in range(total_draws):
        prop = np.clip(current + rng.standard_normal(space.dim) * proposal_scale, 0.0, 1.0)
        lp_prop = log_like(prop)
        if math.log(rng.random()) < lp_prop - lp:
            current, lp = prop, lp_prop
            if step >= burn_in:
                accepted += 1
        if step >= burn_in and (step - burn_in) % thin == 0:
            draws.append(current.copy())
    if not draws:
        raise RuntimeError("posterior sampler produced no draws")
    chain = np.asarray(draws)
    physical = np.array([[space.to_physical(u)[n] for n in space.names] for u in chain])
    summary = {}
    for j, name in enumerate(space.names):
        col = physical[:, j]
        summary[name] = {
            "mean": float(col.mean()),
            "median": float(np.median(col)),
            "lo95": float(np.percentile(col, 2.5)),
            "hi95": float(np.percentile(col, 97.5)),
            "std": float(col.std(ddof=1)),
        }
    return {
        "samples_unit": chain,
        "samples": physical,
        "summary": summary,
        # proposals are made at every step after burn-in but only every
        # ``thin``-th state is kept, so the rate is per proposal, not per draw
        "acceptance_rate": accepted / max(n_samples * thin, 1),
        "n_samples": int(chain.shape[0]),
    }


def calibrate(objective: Callable[[dict[str, float]], float],
              space: ParameterSpace, *, n_init: int = 8, n_iter: int = 32,
              y_stop: float | None = None, seed: int = 0) -> dict:
    """BayesOpt search followed by nothing (UQ needs the response table)."""
    return bayesian_optimize(objective, space, n_init=n_init, n_iter=n_iter,
                             seed=seed, y_stop=y_stop)