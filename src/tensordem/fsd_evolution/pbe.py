"""Mass-conserving fixed-pivot population balance driven by a breakage kernel.

Equation
--------
With ``n(d, t)`` the number density per unit *diameter*, breakage only:

.. math::
   \\frac{\\partial n(d,t)}{\\partial t} = -\\nu(d)\\,n(d,t)
       + \\int_d^\\infty \\nu(d')\\,\\beta(d\\,|\\,d')\\,n(d',t)\\,dd'

Two conventions are fixed here and used everywhere in the package:

* **Conserved measure.**  Floes are area-equivalent disks of constant
  thickness, so the conserved quantity is the floe area ``m ~ d**2`` (not the
  ``d**3`` of a sphere).  The sectional grid is therefore geometric in area,
  ``m_k = m_0 2**k``, i.e. two sections per octave of area, ``D_k = D_0
  2**(k/2)``.  ``MASS_EXPONENT = 2`` is imported from
  :mod:`tensordem.fsd_evolution.kernel` so the kernel and the balance cannot
  drift apart.
* **Mass conservation by construction.**  A daughter of area ``m`` that falls
  between pivots ``k`` and ``k+1`` is split between them with weights
  ``(m_{k+1} - m)/(m_{k+1} - m_k)`` and ``1 - that`` (Kumar & Ramkrishna
  1996).  Because each sampled breakage event conserves area exactly (the
  daughter rule renormalises ``sum x_i**2 = 1``), the discrete operator
  conserves ``sum_k m_k n_k`` to machine precision -- verified in the tests,
  not assumed.  Daughters that fall below the grid floor are placed in the
  floor section with the count weight needed to carry their mass, so the floor
  never leaks mass.

The operator is validated against two analytic controls in
``tests/test_fsd_evolution.py``:

* equal-size binary breakage at constant rate: ``N(t) = N_0 exp(nu t)``,
* the no-breakage negative control: ``N(t)`` constant and the initial
  distribution untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import solve_ivp

from .kernel import MASS_EXPONENT

__all__ = [
    "SectionGrid",
    "two_pivot_weights",
    "daughter_matrix",
    "breakage_rhs",
    "solve_breakage",
    "mass_moment",
    "number_moment",
]


# --------------------------------------------------------------------- grid
@dataclass(frozen=True)
class SectionGrid:
    """Geometric sectional grid in the conserved measure (floe area)."""

    d_min: float
    d_max: float
    n_sections: int
    mass_exponent: int = MASS_EXPONENT

    def __post_init__(self) -> None:
        if not (self.d_min > 0.0 and self.d_max > self.d_min):
            raise ValueError("require 0 < d_min < d_max")
        if int(self.n_sections) < 4:
            raise ValueError("n_sections must be >= 4")
        if int(self.mass_exponent) < 1:
            raise ValueError("mass_exponent must be >= 1")

    @property
    def ratio(self) -> float:
        """Pivot ratio that lands the last pivot exactly on ``d_max``."""
        return (self.d_max / self.d_min) ** (1.0 / (self.n_sections - 1))

    @property
    def diameters(self) -> np.ndarray:
        return self.d_min * self.ratio ** np.arange(self.n_sections)

    @property
    def measure(self) -> np.ndarray:
        return self.diameters**self.mass_exponent


# ------------------------------------------------------------------ pivots
def two_pivot_weights(grid: SectionGrid, measure: float) -> tuple[int, int, float, float]:
    """Split a daughter of measure ``m`` between the two bracketing pivots.

    Returns ``(lower_index, upper_index, lower_weight, upper_weight)`` with the
    weights on *counts* (so that ``w_lower*m_lower + w_upper*m_upper = m``).
    Values below the floor are assigned to section 0 with the count weight that
    carries their mass exactly; values above the last pivot are assigned to the
    last section (an overflow guard that the study never triggers).
    """
    pivots = grid.measure
    if measure <= pivots[0]:
        return 0, 0, measure / pivots[0], 0.0
    if measure >= pivots[-1]:
        return pivots.size - 1, pivots.size - 1, measure / pivots[-1], 0.0
    upper = int(np.searchsorted(pivots, measure))
    lower = upper - 1
    span = pivots[upper] - pivots[lower]
    weight_upper = (measure - pivots[lower]) / span
    return lower, upper, 1.0 - weight_upper, weight_upper


def daughter_matrix(grid: SectionGrid, rule, *, n_samples: int = 200_000,
                    seed: int = 0) -> np.ndarray:
    """Expected daughter counts ``N[i, j]`` from a parent in section ``j``.

    ``N[i, j]`` is the *mean* number of daughters landing in section ``i`` when
    one floe in section ``j`` breaks.  It is estimated by Monte Carlo over the
    measured daughter rule and placed with the two-pivot scheme, so
    ``sum_i N[i, j] * m_i = m_j`` holds to machine precision.
    """
    rng = np.random.default_rng(seed)
    diameters = grid.diameters
    counts = np.zeros(diameters.size, dtype=np.float64)
    matrix = np.zeros((diameters.size, diameters.size), dtype=np.float64)
    for j, parent in enumerate(diameters):
        counts[:] = 0.0
        for _ in range(int(n_samples)):
            daughters = rule.sample(parent)
            for size in daughters:
                lower, upper, w_lower, w_upper = two_pivot_weights(
                    grid, float(size) ** grid.mass_exponent
                )
                counts[lower] += w_lower
                if w_upper:
                    counts[upper] += w_upper
        matrix[:, j] = counts / float(n_samples)
    del rng
    return matrix


def daughter_mass_defect(grid: SectionGrid, matrix: np.ndarray) -> float:
    """Relative mass defect ``max_j |sum_i N[i,j] m_i - m_j| / m_j``."""
    mass = grid.measure
    residual = matrix.T @ mass - mass
    return float(np.max(np.abs(residual) / mass))


# ------------------------------------------------------------------ operator
def breakage_rhs(n_density, *, rates, daughter, source=None, sink=None) -> np.ndarray:
    """Right-hand side of the breakage population balance.

    ``source`` (per section, per unit time) renews the largest floes -- the
    level-ice reservoir that a floe field continually calves from -- and
    ``sink`` is a first-order removal rate (lateral melt / export) that closes
    the mass budget at the small-size end.  Both default to zero, which
    recovers the pure breakage balance.
    """
    n = np.clip(np.asarray(n_density, dtype=np.float64), 0.0, None)
    events = rates * n
    dn = -events + daughter @ events
    if source is not None:
        dn = dn + np.asarray(source, dtype=np.float64)
    if sink is not None:
        dn = dn - np.asarray(sink, dtype=np.float64) * n
    return dn


def solve_breakage(
    grid: SectionGrid,
    initial,
    *,
    rates,
    daughter,
    source=None,
    sink=None,
    t_span=(0.0, 1.0),
    t_eval=None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> dict:
    """Integrate the sectional balance with an implicit stiff solver."""
    initial = np.asarray(initial, dtype=np.float64)
    if initial.shape != (grid.n_sections,):
        raise ValueError("initial must have one entry per section")
    rates = np.asarray(rates, dtype=np.float64)
    if rates.shape != (grid.n_sections,):
        raise ValueError("rates must have one entry per section")
    daughter = np.asarray(daughter, dtype=np.float64)
    if daughter.shape != (grid.n_sections, grid.n_sections):
        raise ValueError("daughter matrix must be (n_sections, n_sections)")
    t_eval = np.linspace(t_span[0], t_span[1], 41) if t_eval is None else np.asarray(t_eval)
    rhs = lambda _t, n: breakage_rhs(  # noqa: E731
        n, rates=rates, daughter=daughter, source=source, sink=sink
    )
    solution = solve_ivp(
        rhs, t_span, initial, method="BDF", t_eval=t_eval, rtol=rtol, atol=atol
    )
    if not solution.success:  # pragma: no cover - defensive
        raise RuntimeError(f"population balance integration failed: {solution.message}")
    return {"t": solution.t, "n": solution.y.T, "grid": grid}


def mass_moment(grid: SectionGrid, n_density) -> float:
    """Total conserved measure ``sum_k m_k n_k`` (floe area per unit volume)."""
    return float(np.dot(grid.measure, np.asarray(n_density, dtype=np.float64)))


def number_moment(n_density) -> float:
    """Total number density ``sum_k n_k``."""
    return float(np.sum(np.asarray(n_density, dtype=np.float64)))


def mass_defect(grid: SectionGrid, n0, n1) -> float:
    """Relative drift of the conserved measure between two states."""
    a = mass_moment(grid, n0)
    b = mass_moment(grid, n1)
    scale = abs(a) if abs(a) > 0.0 else 1.0
    return float(abs(b - a) / scale)