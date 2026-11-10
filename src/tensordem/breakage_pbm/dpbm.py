"""Discrete (sectional / fixed-pivot) population balance for particle breakage.

Equation
--------
In the *volume* coordinate ``v`` (mass is ``rho v``, spheres therefore have
``m ~ d**3``; contrast :mod:`tensordem.fsd_evolution.pbe`, which is written in
floe **area** for constant-thickness ice floes and must not be confused with
this one), pure breakage reads

.. math::
   \\frac{\\partial n(v,t)}{\\partial t} = -S(v,t)\\,n(v,t)
       + \\int_v^\\infty S(v',t)\\,b(v\\,|\\,v')\\,n(v',t)\\,dv' .

Number density ``n`` is per unit volume per unit volume (SI: m^-6).

Discretisation
--------------
``VolumeGrid`` is geometric in volume with ``sections_per_octave`` pivots per
octave (two by default), so the diameter grid is geometric as well.  Daughters
landing between pivots ``k`` and ``k+1`` are split with the two-pivot weights
``(m_{k+1}-m)/(m_{k+1}-m_k)`` and ``1-that`` (Kumar & Ramkrishna 1996), which
makes the discrete operator conserve ``sum_k m_k n_k`` to machine precision --
verified in the tests rather than assumed.  The time-dependent rate ``S(v,t)``
is what a DEM-calibrated Weibull strength law produces
(:mod:`tensordem.breakage_pbm.weibull_breakage`).

Analytic reference
------------------
For ``S = const`` and the uniform daughter distribution ``b(v,v') = 2/v'`` the
pure-breakage balance has the closed form (generation series, ``tau = S t``) for
a unit parent of volume ``v0``::

    n_delta(v,t) = e^-tau * delta(v-v0)
                 + e^-tau * sum_{k>=1} (2 tau)^k/k! * (1/v0)
                                        * (ln(v0/v))^(k-1)/(k-1)!

which satisfies ``int n dv = e^tau`` and ``int v n dv = v0`` exactly.  Because
the balance is linear, the solution for an arbitrary initial condition is the
superposition :func:`analytic_solution`, which is what
:func:`analytic_cell_density` returns on the grid (used as the <3% reference in
the verification surface).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import solve_ivp

__all__ = [
    "VolumeGrid",
    "two_pivot_weights",
    "daughter_matrix",
    "mass_defect_matrix",
    "breakage_rhs",
    "solve_breakage",
    "mass_moment",
    "number_moment",
    "mass_density",
    "cumulative_volume_fraction",
    "coarsen_to_grid",
    "analytic_delta_solution",
    "analytic_cell_density",
    "wasserstein_from_cdf",
    "psd_from_sections",
    "equivalent_diameter",
    "particle_volume",
]


def equivalent_diameter(volume) -> np.ndarray:
    """Diameter of the sphere of the given volume."""
    v = np.clip(np.asarray(volume, dtype=np.float64), 1e-300, None)
    return (6.0 * v / math.pi) ** (1.0 / 3.0)


def particle_volume(diameter) -> np.ndarray:
    """Volume of the sphere of the given diameter."""
    d = np.asarray(diameter, dtype=np.float64)
    return math.pi * d ** 3 / 6.0


# ---------------------------------------------------------------------- grid
@dataclass(frozen=True)
class VolumeGrid:
    """Geometric sectional grid in particle volume."""

    v_min: float
    v_max: float
    n_sections: int
    sections_per_octave: float = 2.0

    def __post_init__(self) -> None:
        if not (self.v_min > 0.0 and self.v_max > self.v_min):
            raise ValueError("require 0 < v_min < v_max")
        if int(self.n_sections) < 4:
            raise ValueError("n_sections must be >= 4")
        if self.sections_per_octave <= 0.0:
            raise ValueError("sections_per_octave must be positive")

    @classmethod
    def from_diameters(cls, d_min: float, d_max: float, n_sections: int,
                       sections_per_octave: float = 2.0) -> "VolumeGrid":
        return cls(v_min=float(particle_volume(d_min)),
                   v_max=float(particle_volume(d_max)),
                   n_sections=int(n_sections),
                   sections_per_octave=float(sections_per_octave))

    @property
    def ratio(self) -> float:
        """Pivot ratio placing the last pivot exactly on ``v_max``."""
        return (self.v_max / self.v_min) ** (1.0 / (self.n_sections - 1))

    @property
    def pivots(self) -> np.ndarray:
        return self.v_min * self.ratio ** np.arange(self.n_sections)

    @property
    def diameters(self) -> np.ndarray:
        return equivalent_diameter(self.pivots)

    @property
    def widths(self) -> np.ndarray:
        """Cell widths for the piecewise-constant representation (ends included)."""
        pivots = self.pivots
        r = self.ratio
        lower = np.empty(self.n_sections)
        upper = np.empty(self.n_sections)
        lower[0] = pivots[0] * (1.0 - 1.0 / r)
        lower[1:] = pivots[:-1]
        upper[:-1] = pivots[1:]
        upper[-1] = pivots[-1] * (r - 1.0)
        return upper - lower

    @property
    def cell_centres(self) -> np.ndarray:
        pivots = self.pivots
        r = self.ratio
        lower = np.empty(self.n_sections)
        upper = np.empty(self.n_sections)
        lower[0] = pivots[0] / r
        lower[1:] = pivots[:-1]
        upper[:-1] = pivots[1:]
        upper[-1] = pivots[-1] * r
        return 0.5 * (lower + upper)


# ------------------------------------------------------------------- pivots
def two_pivot_weights(grid: VolumeGrid, volume: float):
    """Split a daughter of volume ``volume`` between the bracketing pivots.

    Returns ``(lower_index, upper_index, lower_weight, upper_weight)`` where the
    weights act on *counts*, so ``w_lower*v_lower + w_upper*v_upper = volume``.
    Values below the floor go to section 0 with the count weight that carries
    their mass exactly (the floor never leaks mass).
    """
    pivots = grid.pivots
    v = float(volume)
    if v <= pivots[0]:
        return 0, 0, v / pivots[0], 0.0
    if v >= pivots[-1]:
        return pivots.size - 1, pivots.size - 1, v / pivots[-1], 0.0
    upper = int(np.searchsorted(pivots, v))
    lower = upper - 1
    span = pivots[upper] - pivots[lower]
    weight_upper = (v - pivots[lower]) / span
    return lower, upper, 1.0 - weight_upper, weight_upper


def daughter_matrix(grid: VolumeGrid, rule, *, n_samples: int = 4000,
                    seed: int = 0) -> np.ndarray:
    """Expected daughter counts ``N[i, j]`` for one parent in section ``j``.

    Monte Carlo over the daughter rule, placed with the two-pivot scheme, so
    ``sum_i N[i, j] v_i = v_j`` holds to machine precision for every ``j``
    (the rule itself conserves mass exactly).
    """
    rng = np.random.default_rng(seed)
    pivots = grid.pivots
    n = pivots.size
    matrix = np.zeros((n, n))
    counts = np.zeros(n)
    for j, parent in enumerate(pivots):
        counts[:] = 0.0
        for _ in range(int(n_samples)):
            daughters = np.asarray(rule.sample(float(parent), rng), dtype=np.float64)
            for size in daughters:
                lower, upper, w_low, w_up = two_pivot_weights(grid, float(size))
                counts[lower] += w_low
                if w_up:
                    counts[upper] += w_up
        matrix[:, j] = counts / float(n_samples)
    return matrix


def mass_defect_matrix(grid: VolumeGrid, matrix: np.ndarray) -> float:
    """Relative mass defect ``max_j |sum_i N[i,j] v_i - v_j| / v_j``."""
    mass = grid.pivots
    residual = np.asarray(matrix).T @ mass - mass
    return float(np.max(np.abs(residual) / mass))


# ------------------------------------------------------------------ operator
def breakage_rhs(t: float, n_density, *, rate, daughter) -> np.ndarray:
    """Right-hand side ``-S n + sum N S n`` for a (possibly time-dependent) rate.

    ``rate`` is either an array of section rates (constant in time) or a
    callable ``rate(t) -> array``.
    """
    n = np.clip(np.asarray(n_density, dtype=np.float64), 0.0, None)
    rates = np.asarray(rate(t), dtype=np.float64) if callable(rate) else np.asarray(rate, float)
    events = rates * n
    return -events + daughter @ events


def solve_breakage(grid: VolumeGrid, initial, *, rate, daughter,
                   t_span=(0.0, 8.0), t_eval=None, rtol: float = 1e-10,
                   atol: float = 1e-14) -> dict:
    """Integrate the sectional balance with the stiff implicit BDF solver."""
    initial = np.asarray(initial, dtype=np.float64)
    if initial.shape != (grid.n_sections,):
        raise ValueError("initial must have one entry per section")
    daughter = np.asarray(daughter, dtype=np.float64)
    if daughter.shape != (grid.n_sections, grid.n_sections):
        raise ValueError("daughter matrix must be (n_sections, n_sections)")
    t_eval = np.linspace(t_span[0], t_span[1], 41) if t_eval is None else np.asarray(t_eval)
    rhs = lambda tt, nn: breakage_rhs(tt, nn, rate=rate, daughter=daughter)  # noqa: E731
    solution = solve_ivp(rhs, t_span, initial, method="BDF", t_eval=t_eval,
                         rtol=rtol, atol=atol)
    if not solution.success:  # pragma: no cover - defensive
        raise RuntimeError(f"population balance integration failed: {solution.message}")
    return {"t": solution.t, "n": solution.y.T, "grid": grid}


# ------------------------------------------------------------------- moments
def mass_moment(grid: VolumeGrid, n_density) -> float:
    """Total particle volume per unit system volume ``sum_k v_k n_k``."""
    return float(np.dot(grid.pivots, np.asarray(n_density, dtype=np.float64)))


def number_moment(n_density) -> float:
    """Total number density ``sum_k n_k``."""
    return float(np.sum(np.asarray(n_density, dtype=np.float64)))


def mass_density(grid: VolumeGrid, n_density) -> np.ndarray:
    """Volume-weighted density ``v_k n_k`` (the mass-per-decade view)."""
    return grid.pivots * np.asarray(n_density, dtype=np.float64)


def cumulative_volume_fraction(grid: VolumeGrid, n_density) -> np.ndarray:
    """Cumulative volume fraction finer than each pivot (monotone, 0..1)."""
    w = mass_density(grid, n_density)
    total = float(np.sum(w))
    if total <= 0.0:
        return np.zeros_like(w)
    return np.cumsum(w) / total


def coarsen_to_grid(grid: VolumeGrid, volumes, weights=None) -> np.ndarray:
    """Bin particle volumes into the sectional grid; returns counts per section."""
    v = np.asarray(volumes, dtype=np.float64).ravel()
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if v.size != w.size:
        raise ValueError("volumes and weights must have the same length")
    counts = np.zeros(grid.n_sections)
    if v.size == 0:
        return counts
    idx = np.searchsorted(grid.pivots, v)
    idx = np.clip(idx, 0, grid.n_sections - 1)
    np.add.at(counts, idx, w)
    return counts


def psd_from_sections(grid: VolumeGrid, n_density, n_sample: int = 4000):
    """Deterministic quantile sample of the volume-weighted diameter PSD.

    Returns ``(diameters, weights)`` of ``n_sample`` equal-mass quantiles of the
    volume-weighted size distribution, which is the reproducible way to compare
    a sectional solution against a DEM particle sample.
    """
    w = mass_density(grid, n_density)
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("empty distribution")
    frac = np.cumsum(w) / total
    targets = (np.arange(n_sample) + 0.5) / n_sample
    idx = np.clip(np.searchsorted(frac, targets), 0, grid.n_sections - 1)
    return grid.diameters[idx], np.full(n_sample, total / n_sample)


# ---------------------------------------------------------------- analytic
def analytic_delta_solution(v_eval, t: float, rate: float, v0: float,
                            n_terms: int = 120) -> np.ndarray:
    """Closed-form number density for one parent of volume ``v0`` (delta IC).

    The ``k = 0`` (surviving parent) delta is *not* included -- it is infinite at
    ``v = v0`` and is handled analytically by the caller (see
    :func:`analytic_cell_density`).
    """
    v = np.asarray(v_eval, dtype=np.float64)
    tau = float(rate) * float(t)
    out = np.zeros_like(v)
    mask = (v > 0.0) & (v < float(v0))
    if not np.any(mask) or tau <= 0.0:
        return out
    logfactor = np.log(float(v0) / v[mask])
    term = 2.0 * tau  # k = 1 term of (2 tau)^k/k! * L^(k-1)/(k-1)!
    acc = np.full(logfactor.shape, term)
    k = 1
    while k < int(n_terms):
        k += 1
        term = term * (2.0 * tau / k) * (logfactor / (k - 1))
        acc = acc + term
        if np.max(np.abs(term)) <= 1e-18 * max(1.0, float(np.max(np.abs(acc)))):
            break
    out[mask] = np.exp(-tau) * acc / float(v0)
    return out


def analytic_cell_density(grid: VolumeGrid, initial, t: float, rate: float,
                          n_terms: int = 120) -> np.ndarray:
    """Closed-form solution of the uniform-kernel balance on the sectional grid.

    Superposition of :func:`analytic_delta_solution` over the initial state
    ``initial`` (counts per section, the fixed-pivot state variable), integrated
    over each cell so the result is directly comparable with the sectional
    solution.  The surviving-parent delta contributes ``exp(-tau) * initial``
    exactly.
    """
    n0 = np.asarray(initial, dtype=np.float64)
    if n0.shape != (grid.n_sections,):
        raise ValueError("initial must have one entry per section")
    tau = float(rate) * float(t)
    centres = grid.cell_centres
    out = np.exp(-tau) * n0.copy()
    if tau <= 0.0 or not np.any(n0 > 0.0):
        return out
    sources = grid.pivots
    counts = n0                       # counts per section (fixed-pivot state)
    widths = grid.widths              # cell volume, so the density integrates to a count
    for j, (v0, c) in enumerate(zip(sources, counts)):
        if c <= 0.0:
            continue
        out += c * analytic_delta_solution(centres, t, rate, float(v0),
                                          n_terms=n_terms) * widths
    return out


# ------------------------------------------------------- distribution metric
def wasserstein_from_cdf(grid_pts, f_a, f_b) -> float:
    """Exact ``W1`` between two right-continuous CDFs sampled on a common grid.

    ``W1 = int |F_a(x) - F_b(x)| dx``; the grid must be dense and must cover
    both supports (the caller supplies log-spaced points with the tails
    clipped), which is a deterministic alternative to sampling.
    """
    x = np.asarray(grid_pts, dtype=np.float64)
    fa = np.asarray(f_a, dtype=np.float64)
    fb = np.asarray(f_b, dtype=np.float64)
    return float(np.trapezoid(np.abs(fa - fb), x))