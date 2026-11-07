"""Parameter-space description and space-filling experiment designs.

All designs live in the **unit cube** ``[0, 1]^d``; :class:`ParameterSpace`
maps them onto the physical box declared by the :class:`Parameter` list.  Every
design generator is deterministic given its seed, so a campaign can be
re-generated and re-run from the JSON description written by the driver script.

Why scrambled Sobol' sequences: the variance-based indices are estimated with
``n = 2**k`` base samples; a digital sequence keeps the quadrature error of the
inner Monte-Carlo averages close to ``O(1/n)`` with a much smaller constant than
pseudo-random sampling, and the scrambling makes the bootstrap replicate design
defensible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import qmc

__all__ = ["Parameter", "ParameterSpace"]


@dataclass(frozen=True)
class Parameter:
    """One scalar input of the model, bounded and optionally log-scaled."""

    name: str
    low: float
    high: float
    unit: str = ""
    description: str = ""
    log: bool = field(default=False)

    def __post_init__(self) -> None:
        for label, value in (("low", self.low), ("high", self.high)):
            if not np.isfinite(value):
                raise ValueError(f"{self.name}: {label} must be finite")
        if self.high <= self.low:
            raise ValueError(f"{self.name}: high must exceed low")
        if self.log and self.low <= 0.0:
            raise ValueError(f"{self.name}: log-scaled parameters need low > 0")

    @property
    def span(self) -> float:
        return self.high - self.low

    def to_physical(self, u):
        """Map unit-cube values onto the physical range."""
        u = np.asarray(u, dtype=float)
        if np.any(u < 0.0) or np.any(u > 1.0):
            raise ValueError(f"{self.name}: unit values must lie in [0, 1]")
        if self.log:
            return self.low * (self.high / self.low) ** u
        return self.low + self.span * u

    def to_unit(self, x):
        """Inverse of :meth:`to_physical`."""
        x = np.asarray(x, dtype=float)
        if self.log:
            return np.log(x / self.low) / np.log(self.high / self.low)
        return (x - self.low) / self.span


class ParameterSpace:
    """A ``d``-dimensional bounded parameter box with sampling helpers."""

    def __init__(self, parameters):
        parameters = tuple(parameters)
        if not parameters:
            raise ValueError("at least one parameter is required")
        names = [p.name for p in parameters]
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        self.parameters = parameters
        self.names = tuple(names)

    # -- shape / mapping ---------------------------------------------------
    def __len__(self) -> int:
        return len(self.parameters)

    @property
    def bounds(self) -> np.ndarray:
        return np.array([[p.low, p.high] for p in self.parameters], dtype=float)

    def to_physical(self, U) -> np.ndarray:
        U = np.atleast_2d(np.asarray(U, dtype=float))
        if U.ndim != 2 or U.shape[1] != len(self):
            raise ValueError(f"unit sample matrix must have {len(self)} columns")
        return np.column_stack([p.to_physical(U[:, i]) for i, p in enumerate(self.parameters)])

    def to_unit(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.ndim != 2 or X.shape[1] != len(self):
            raise ValueError(f"physical sample matrix must have {len(self)} columns")
        return np.column_stack([p.to_unit(X[:, i]) for i, p in enumerate(self.parameters)])

    def subset(self, names) -> "ParameterSpace":
        index = [self.names.index(n) for n in names]
        return ParameterSpace([self.parameters[i] for i in index])

    def describe(self) -> list[dict]:
        return [
            {
                "name": p.name,
                "low": p.low,
                "high": p.high,
                "unit": p.unit,
                "log": bool(p.log),
                "description": p.description,
            }
            for p in self.parameters
        ]

    # -- designs -----------------------------------------------------------
    def sobol_design(self, n: int, *, seed: int = 0) -> np.ndarray:
        """``n = 2**k`` scrambled Sobol' points in the unit cube."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 2:
            raise ValueError("n must be an integer >= 2")
        engine = qmc.Sobol(d=len(self), scramble=True, seed=seed)
        return engine.random(n)

    def latin_hypercube(self, n: int, *, seed: int = 0, candidates: int = 8) -> np.ndarray:
        """Centered LHS with a maximin (max-min Euclidean) tie-break.

        ``candidates`` independent LHS draws are generated and the one with the
        largest minimum pairwise distance is kept; this is a cheap maximin
        heuristic that removes the worst clumping without a global optimiser.
        """
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        if candidates < 1:
            raise ValueError("candidates must be >= 1")
        best, best_score = None, -np.inf
        for k in range(candidates):
            engine = qmc.LatinHypercube(d=len(self), seed=seed + 7919 * k)
            U = engine.random(n)
            score = _min_pair_distance(U) if n > 1 else np.inf
            if score > best_score:
                best, best_score = U, score
        return np.asarray(best)

    def saltelli_design(self, n: int, *, seed: int = 0, kind: str = "latin"):
        """Saltelli ``A``, ``B`` and ``AB_i`` matrices (unit cube).

        ``AB_i`` copies ``A`` and replaces its ``i``-th column with the ``i``-th
        column of ``B``.  Returns ``(A, B, AB)`` with ``AB.shape == (d, n, d)``.

        ``kind='latin'`` (default) draws ``A`` and ``B`` as two *independent*
        Latin hypercubes.  Independent randomisation matters: the pick-freeze
        estimators assume ``A`` and ``B`` are independent sample sets, and a
        single scrambled Sobol' stream split in half violates that.  On the
        Ishigami reference problem the split-stream variant biases the indices
        by up to 0.15 while the independent Latin-hypercube pairs agree with
        the analytic values to 0.01 (see ``tests/test_ice_uq_sensitivity.py``).
        ``kind='sobol'`` keeps the (biased) split-stream variant for
        comparison only.
        """
        if kind not in ("latin", "random", "sobol"):
            raise ValueError("kind must be 'latin', 'random' or 'sobol'")
        if kind == "sobol":
            engine = qmc.Sobol(d=len(self), scramble=True, seed=seed)
            A = engine.random(n)
            B = engine.random(n)
        elif kind == "random":
            rng = np.random.default_rng(seed)
            A = rng.random((n, len(self)))
            B = rng.random((n, len(self)))
        else:
            A = self.latin_hypercube(n, seed=seed)
            B = self.latin_hypercube(n, seed=seed + 104729)
        AB = np.repeat(A[None, :, :], len(self), axis=0).copy()
        for i in range(len(self)):
            AB[i, :, i] = B[:, i]
        return A, B, AB


def _min_pair_distance(U: np.ndarray, *, max_points: int = 512) -> float:
    """Minimum pairwise Euclidean distance (dense cost, so subsampled).

    A full ``n x n x d`` difference array is prohibitive for the Saltelli
    matrices (``n = 4096`` needs ~0.5 GB per candidate), so the maximin
    tie-break is scored on a deterministic stride subsample of at most
    ``max_points`` rows.  The ranking is unaffected for stratified designs.
    """
    if len(U) > max_points:
        stride = int(np.ceil(len(U) / max_points))
        U = U[::stride][:max_points]
    diff = U[:, None, :] - U[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0