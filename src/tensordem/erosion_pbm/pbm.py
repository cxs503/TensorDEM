"""Finite-volume sectional solver for the breakage population balance.

Equation
--------
With ``n(v, t)`` the number density per unit volume of particle volume ``v``
(``v = d**3``, the equivalent-volume cube, times ``pi/6`` when a *physical*
volume is needed), breakage only:

    dn(v,t)/dt = -S(v) n(v,t) + integral_{v}^{inf} S(v') b(v | v') n(v',t) dv'

discretised by the **fixed-pivot sectional method** (Kumar & Ramkrishna 1996)
on the geometric grid ``v_k = v_min * 2**(k/q)`` -- ``q`` sections per octave
of volume.  The operator keeps three things exactly, which is why every
invariant below is a testable identity rather than a fitted property:

* **Mass.**  A daughter whose volume falls between pivots ``k`` and ``k+1`` is
  split with weights ``w = (v_{k+1} - v_d)/(v_{k+1} - v_k)`` and ``1 - w``, so
  ``v_k w + v_{k+1}(1 - w) = v_d`` exactly and ``sum_k v_k n_k`` is conserved to
  machine precision.
* **Fragment count.**  Every daughter contributes weight ``w + (1 - w) = 1``, so
  ``sum_k M[k, j] = n_fragments`` and the total number follows the analytic law
  ``N(t) = N0 exp((n_fragments - 1) S t)`` for a size-independent ``S``.
* **The binary cascade.**  For ``S = const`` and equal binary breakage,
  ``v/2`` lands exactly on the grid (``v_{k-q} = v_k / 2``), so the discrete
  system is ``dN_k/dt = -S N_k + 2 S N_{k-1}`` with the closed-form solution

      N_k(t) = e**(-S t) (2 S t)**k / k!   (monodisperse start in bin 0),

  a genuinely time-dependent, position-dependent analytic spectrum used as the
  primary accuracy reference (criterion C2).

Daughter matrix
---------------
``M[:, j]`` is the expected number of fragments from one breakage event of a
parent in bin ``j`` that land in each section.  Two filler rules are provided:

* ``equal``  -- ``n_fragments`` equal-volume children (used by the analytic
  cascade control; exact on the grid).
* ``gaudin`` -- children whose diameters follow a Gaudin-Schuhmann spectrum
  (reused from :mod:`tensordem.granule_crush`) and are uniformly rescaled so
  ``sum d_i**3 = d_parent**3``.  ``M`` is the *ensemble mean* of that rule,
  computed by quasi-Monte-Carlo with a fixed seed.  Because the discrete
  operator is then exactly the mean-field limit of the DEM fragment rule in
  :mod:`tensordem.erosion_pbm.dem`, the DEM/PBM comparison (criterion C3) tests
  the discretisation and the statistics -- not two different physics.

Time integration is ``scipy.integrate.solve_ivp`` (DOP853, tight tolerances,
analytic Jacobian not needed: the operator is linear and the systems are small).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import solve_ivp

__all__ = [
    "SectionalGrid",
    "two_pivot_weights",
    "daughter_matrix",
    "breakage_rhs",
    "solve_breakage",
    "number_moment",
    "mass_moment",
    "normalised_psd",
    "wasserstein_1d",
]

#: ``v = d**3`` is the state variable; ``pi/6`` is applied when a physical
#: volume is required, so the grid stays a pure geometric ladder.
SPHERE_FACTOR = math.pi / 6.0


@dataclass(frozen=True)
class SectionalGrid:
    """Geometric volume grid ``v_k = v_min * 2**(k/q)``, ``q`` sections/octave."""

    v_min: float
    n_bins: int
    sections_per_octave: int = 6

    def __post_init__(self) -> None:
        if self.v_min <= 0.0:
            raise ValueError(f"v_min must be > 0, got {self.v_min}")
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {self.n_bins}")
        if self.sections_per_octave < 1:
            raise ValueError("sections_per_octave must be >= 1")

    @property
    def ratio(self) -> float:
        return 2.0 ** (1.0 / self.sections_per_octave)

    @property
    def volumes(self) -> np.ndarray:
        return self.v_min * 2.0 ** (np.arange(self.n_bins) / self.sections_per_octave)

    @property
    def diameters(self) -> np.ndarray:
        return self.volumes ** (1.0 / 3.0)

    @property
    def v_max(self) -> float:
        return float(self.volumes[-1])

    def index_of_diameter(self, d) -> np.ndarray:
        """Section index of equivalent-volume diameter ``d`` (clipped)."""
        v = np.asarray(d, dtype=float) ** 3
        idx = np.searchsorted(self.volumes, v, side="right") - 1
        return np.clip(idx, 0, self.n_bins - 1)


def two_pivot_weights(frag_volume: float, volumes: np.ndarray) -> tuple[int, int, float]:
    """Kumar-Ramkrishna two-pivot split ``(k, k+1, w)`` for one fragment.

    ``w`` goes to section ``k`` and ``1 - w`` to section ``k + 1``, chosen so
    that ``v_k w + v_{k+1}(1 - w) = frag_volume``.  Fragments below ``v_min``
    are placed in the floor section with unit weight (documented mass deficit
    measured by :func:`mass_moment` versus the exact ledger).
    """
    v = volumes
    if frag_volume <= v[0]:
        return 0, 0, 1.0
    if frag_volume >= v[-1]:
        return int(len(v) - 1), int(len(v) - 1), 1.0
    i = int(np.searchsorted(v, frag_volume, side="right"))
    k = i - 1
    w = (v[k + 1] - frag_volume) / (v[k + 1] - v[k])
    return k, k + 1, float(w)


def _accumulate(weights_row: np.ndarray, k: int, w: float) -> None:
    weights_row[k] += w


def daughter_matrix(
    grid: SectionalGrid,
    *,
    n_fragments: int = 3,
    rule: str = "gaudin",
    gaudin_alpha: float = 2.5,
    n_samples: int = 4096,
    seed: int = 20260101,
) -> np.ndarray:
    """Expected fragment-count matrix ``M[k, j]`` (see module docstring)."""
    if rule not in ("equal", "gaudin"):
        raise ValueError(f"unknown fragment rule {rule!r}")
    if n_fragments < 1:
        raise ValueError("n_fragments must be >= 1")
    v = grid.volumes
    K = grid.n_bins
    M = np.zeros((K, K), dtype=float)
    rng = np.random.default_rng(seed)

    if rule == "equal":
        for j in range(K):
            frag = v[j] / n_fragments
            k0, k1, w = two_pivot_weights(frag, v)
            M[k0, j] += n_fragments * w
            if k1 != k0:
                M[k1, j] += n_fragments * (1.0 - w)
        return M

    # Gaudin-Schuhmann children, rescaled to conserve the parent volume.
    # Vectorised two-pivot accumulation: every fragment contributes weight w to
    # section k and (1 - w) to k + 1, so mass is exact by construction.
    for j in range(K):
        d_parent = v[j] ** (1.0 / 3.0)
        u = rng.random((n_samples, n_fragments))
        rel = u ** (1.0 / gaudin_alpha)
        scale = np.sum(rel ** 3, axis=1) ** (1.0 / 3.0)
        d_child = d_parent * rel / scale[:, None]
        frag = (d_child ** 3).ravel()
        i = np.searchsorted(v, frag, side="right")
        floor = i <= 0
        top = i >= K
        k0 = np.clip(i - 1, 0, K - 2)
        w = (v[k0 + 1] - frag) / (v[k0 + 1] - v[k0])
        w = np.where(floor | top, 1.0, np.clip(w, 0.0, 1.0))
        k0 = np.where(floor, 0, np.where(top, K - 1, k0))
        k1 = np.where(floor | top, k0, k0 + 1)
        coef1 = np.where(floor | top, 0.0, 1.0 - w) / n_samples
        M[:, j] = (np.bincount(k0, weights=w / n_samples, minlength=K)
                   + np.bincount(k1, weights=coef1, minlength=K))
    return M


def breakage_rhs(n: np.ndarray, selection: np.ndarray, M: np.ndarray) -> np.ndarray:
    """``dn/dt = -S n + M (S n)`` -- loss plus daughter gain."""
    s_n = np.asarray(selection, dtype=float) * np.asarray(n, dtype=float)
    return -s_n + M @ s_n


def solve_breakage(
    grid: SectionalGrid,
    n0: np.ndarray,
    selection,
    M: np.ndarray,
    *,
    t_end: float,
    t_eval=None,
    rtol: float = 1e-10,
    atol: float = 1e-30,
) -> dict:
    """Integrate the discrete breakage PBE from ``n0`` over ``[0, t_end]``.

    ``selection`` is either a constant (size-independent ``S``) or an array of
    per-section rates; anything callable ``S(v)`` is accepted through
    :func:`selection_from_callable`.
    """
    n0 = np.asarray(n0, dtype=float)
    if n0.shape != (grid.n_bins,):
        raise ValueError("n0 must have one entry per section")
    S = np.asarray(selection, dtype=float)
    if S.ndim == 0:
        S = np.full(grid.n_bins, float(S))
    if S.shape != (grid.n_bins,):
        raise ValueError("selection must be scalar or one entry per section")
    t_eval = np.linspace(0.0, t_end, 201) if t_eval is None else np.asarray(t_eval, dtype=float)

    def rhs(_t, y):
        return breakage_rhs(y, S, M)

    sol = solve_ivp(rhs, (0.0, float(t_end)), n0, method="DOP853",
                    t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"PBM integration failed: {sol.message}")
    return {"t": sol.t, "n": sol.y, "selection": S, "matrix": M}


def selection_from_callable(grid: SectionalGrid, rate) -> np.ndarray:
    """Sample a callable ``rate(v)`` on the section representatives."""
    return np.asarray([float(rate(v)) for v in grid.volumes], dtype=float)


def number_moment(n: np.ndarray) -> float:
    """Total number ``sum_k n_k``."""
    return float(np.sum(n))


def mass_moment(n: np.ndarray, grid: SectionalGrid) -> float:
    """Total volume measure ``sum_k v_k n_k`` (``* rho`` for mass)."""
    return float(np.sum(grid.volumes * np.asarray(n, dtype=float)))


def normalised_psd(n: np.ndarray) -> np.ndarray:
    """Number-weighted probability per section (``sum = 1``)."""
    n = np.asarray(n, dtype=float)
    total = float(n.sum())
    if total <= 0.0:
        raise ValueError("empty distribution")
    return n / total


def wasserstein_1d(grid: SectionalGrid, n_a: np.ndarray, n_b: np.ndarray,
                   normalise_by: float | None = None) -> float:
    """1-D Wasserstein-1 distance between two section histograms [m].

    ``W1 = sum_k |F_a(d_k) - F_b(d_k)| * (d_{k+1} - d_k)`` on the common
    diameter grid -- the exact W1 for distributions on a common 1-D support.
    ``normalise_by`` (default: the mean diameter of ``n_a``) makes the number
    dimensionless, which is the quantity criterion C3 bounds by 0.1.
    """
    pa = normalised_psd(n_a)
    pb = normalised_psd(n_b)
    d = grid.diameters
    edges = np.empty(grid.n_bins + 1)
    r = grid.ratio ** (1.0 / 3.0)
    edges[0] = d[0] / math.sqrt(r)
    edges[-1] = d[-1] * math.sqrt(r)
    edges[1:-1] = np.sqrt(d[:-1] * d[1:])
    widths = np.diff(edges)
    Fa = np.cumsum(pa)
    Fb = np.cumsum(pb)
    w1 = float(np.sum(np.abs(Fa - Fb) * widths))
    ref = float(np.sum(pa * d)) if normalise_by is None else float(normalise_by)
    return w1 / ref


def bin_counts(grid: SectionalGrid, diameters, weights=None) -> np.ndarray:
    """Histogram equivalent-volume diameters onto the section grid."""
    d = np.asarray(diameters, dtype=float).ravel()
    idx = grid.index_of_diameter(d)
    if weights is None:
        return np.bincount(idx, minlength=grid.n_bins).astype(float)
    w = np.asarray(weights, dtype=float).ravel()
    return np.bincount(idx, weights=w, minlength=grid.n_bins).astype(float)