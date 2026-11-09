"""Fixed-pivot population balance for the floe size distribution.

Equation
--------
With :math:`n(D,t)` the number density of floes per unit diameter and unit
volume, the campaign's equation is

.. math::
   \\frac{\\partial n}{\\partial t} =
   \\underbrace{\\tfrac12\\int_0^D K(D',D'')\\,n(D')n(D'')\\,\\frac{D'^2}{(D'^3+D''^3)^{2/3}}dD'}_{coalescence\\ source}
   - \\underbrace{n(D)\\int_0^\\infty K(D,D')\\,n(D')\\,dD'}_{coalescence\\ loss}
   - \\underbrace{\\nu(D)\\,n(D)}_{breakage\\ loss}
   + \\underbrace{\\int_D^\\infty \\nu(D')\\,\\beta(D|D')\\,N(D')\\,n(D')\\,dD'}_{breakage\\ source}
   + S,

discretised with the fixed-pivot sectional method of Kumar & Ramkrishna (1996)
on a geometric volume grid ``v_k = v_0 * 2**k`` (two sections per octave of
volume, i.e. ``D_k = D_0 * 2**(k/2)``).  The two-material split of the fixed
pivot is what keeps **both** the volume and the number of bodies exactly
conserved, which is why the solver is checked against

* pure coalescence with a constant kernel, whose analytic solution is
  ``n(t) = n_0/(1 + n_0 K t/2)**2`` in the scaling variable, and
* pure breakage of equal fragments, whose number density grows geometrically.

Breakage is *multiple*: one collision event of a floe in section ``i`` produces
``B(D_i, E)`` fragments of equal volume (the DEM-measured fragment count), so the
breakage frequency and the daughter count are the two measured numbers of
:mod:`tensordem.floe_population.kernels` and nothing else.  ``B = 1`` reproduces
"no breakage" exactly, which is what the negative control uses.

Coalescence is kept in the operator set (it is part of the campaign's equation
and it is validated against the analytic Smoluchowski solution) but it is driven
by a sticking probability that the DEM *measures*; rigid ice floes without
refreezing give ``p_stick = 0``, so the forecast is breakage driven and the
coalescence channel is reported separately rather than smuggled into a fit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import solve_ivp

__all__ = [
    "SectionGrid",
    "effective_fragments",
    "pbe_operators",
    "solve_pbe",
    "census_to_sections",
    "ForecastComparison",
    "compare_trajectories",
    "smoluchowski_constant_kernel",
]


@dataclass(frozen=True)
class SectionGrid:
    """Geometric volume grid; ``n`` sections spanning ``d_min`` to ``d_max``."""

    d_min: float
    d_max: float
    n_sections: int

    def __post_init__(self) -> None:
        if not (self.d_max > self.d_min > 0.0):
            raise ValueError("require 0 < d_min < d_max")
        if self.n_sections < 3:
            raise ValueError("at least three sections are required")

    @property
    def volumes(self) -> np.ndarray:
        ratio = (self.d_max / self.d_min) ** 3
        exponent = np.arange(self.n_sections) / (self.n_sections - 1)
        return self.d_min**3 * ratio**exponent

    @property
    def diameters(self) -> np.ndarray:
        return self.volumes ** (1.0 / 3.0)

    def section_of(self, diameter) -> np.ndarray:
        """Index of the section whose pivot is closest in volume."""
        volumes = np.asarray(diameter, dtype=np.float64) ** 3
        grid = self.volumes
        index = np.searchsorted(grid, volumes)
        index = np.clip(index, 1, grid.size - 1)
        left, right = grid[index - 1], grid[index]
        choose_right = np.abs(volumes - right) < np.abs(volumes - left)
        return np.where(choose_right, index, index - 1)


def _two_pivot_weights(grid: SectionGrid, volume: float) -> tuple[int, int, float, float]:
    """Split ``volume`` between the neighbouring pivots (mass and number exact)."""
    pivots = grid.volumes
    if volume <= pivots[0]:
        return 0, 0, 1.0, 0.0
    if volume >= pivots[-1]:
        return pivots.size - 1, pivots.size - 1, 1.0, 0.0
    right = int(np.searchsorted(pivots, volume))
    left = right - 1
    span = pivots[right] - pivots[left]
    weight_left = (pivots[right] - volume) / span
    return left, right, float(weight_left), float(1.0 - weight_left)


def effective_fragments(grid: SectionGrid, fragments) -> tuple[np.ndarray, list[int]]:
    """Fragment counts reduced to one where a daughter would leave the grid.

    A fragment smaller than the smallest pivot cannot be represented, and simply
    clipping it to the first pivot would destroy mass.  The grid floor is
    therefore treated as the resolution limit of the model: floes whose breakage
    products would fall below it do not break.  The affected sections are
    returned so that the report can state how much of the spectrum is a floor
    artefact (for the campaign grid it is the two or three bottom sections, which
    carry no floes).
    """
    volumes = grid.volumes
    fragments = np.clip(np.asarray(fragments, dtype=np.float64), 1.0, None).copy()
    clipped: list[int] = []
    for i, count in enumerate(fragments):
        if count > 1.0 and volumes[i] / count < volumes[0] * (1.0 - 1e-12):
            fragments[i] = 1.0
            clipped.append(i)
    return fragments, clipped


def pbe_operators(
    grid: SectionGrid,
    *,
    collision_kernel,
    fragments,
    coalescence_kernel=None,
    sticking_probability: float = 0.0,
):
    """Breakage and coalescence operators on the section grid.

    Parameters
    ----------
    collision_kernel
        ``(n, n)`` matrix ``K_ij`` (volume/time) of the DEM-measured kernel
        evaluated on the section pivots.
    fragments
        ``(n,)`` fragment count per breakage event, ``B(D_i, E)``; values below
        one are clipped to one (an event that does not break the floe returns it
        unchanged) and floes whose daughters would fall off the grid floor are
        frozen, see :func:`effective_fragments`.
    coalescence_kernel
        ``(n, n)`` matrix used for the merging channel; defaults to
        ``collision_kernel * sticking_probability``.
    """
    volumes = grid.volumes
    n_sections = volumes.size
    collision_kernel = np.asarray(collision_kernel, dtype=np.float64)
    fragments = np.clip(np.asarray(fragments, dtype=np.float64), 1.0, None)
    if collision_kernel.shape != (n_sections, n_sections):
        raise ValueError("collision_kernel must be (n_sections, n_sections)")
    if fragments.shape != (n_sections,):
        raise ValueError("fragments must be (n_sections,)")

    fragments, _ = effective_fragments(grid, fragments)

    # ---- coalescence: assignment of the merged volume to two pivots
    merge = np.zeros((n_sections, n_sections, 2), dtype=np.int64)
    merge_weight = np.zeros((n_sections, n_sections, 2))
    for i in range(n_sections):
        for j in range(n_sections):
            left, right, w_left, w_right = _two_pivot_weights(grid, volumes[i] + volumes[j])
            merge[i, j] = (left, right)
            merge_weight[i, j] = (w_left, w_right)

    if coalescence_kernel is None:
        coalescence_kernel = collision_kernel * float(sticking_probability)
    coalescence_kernel = np.asarray(coalescence_kernel, dtype=np.float64)

    def derivative(_t, n_density):
        n = np.clip(np.asarray(n_density, dtype=np.float64), 0.0, None)
        frequency = collision_kernel @ n            # per-floe collision rate nu_i
        events = frequency * n                      # events per unit volume and time
        dn = -events                                # the parent is always consumed
        for i in range(n_sections):
            if events[i] <= 0.0:
                continue
            count = fragments[i]
            left, right, w_left, w_right = _two_pivot_weights(grid, volumes[i] / count)
            # count == 1 lands back on the parent pivot with weight one, so the
            # "no breakage" case cancels the loss term exactly
            dn[left] += events[i] * count * w_left
            dn[right] += events[i] * count * w_right
        # ---- coalescence (Smoluchowski): loss and mass-conserving gain
        dn -= n * (coalescence_kernel @ n)
        for i in range(n_sections):
            if n[i] <= 0.0:
                continue
            for j in range(i, n_sections):
                if n[j] <= 0.0:
                    continue
                k_ij = coalescence_kernel[i, j]
                if k_ij <= 0.0:
                    continue
                factor = 0.5 if i == j else 1.0
                merged = factor * k_ij * n[i] * n[j]
                dn[merge[i, j, 0]] += merged * merge_weight[i, j, 0]
                dn[merge[i, j, 1]] += merged * merge_weight[i, j, 1]
        return dn

    return derivative


def solve_pbe(
    grid: SectionGrid,
    initial,
    *,
    collision_kernel,
    fragments,
    coalescence_kernel=None,
    sticking_probability: float = 0.0,
    t_span=(0.0, 1.0),
    t_eval=None,
    rtol: float = 1e-8,
    atol: float = 1e-14,
) -> dict:
    """Integrate the sectional population balance with an implicit stiff solver."""
    initial = np.asarray(initial, dtype=np.float64)
    if initial.shape != (grid.n_sections,):
        raise ValueError("initial must have one entry per section")
    rhs = pbe_operators(
        grid,
        collision_kernel=collision_kernel,
        fragments=fragments,
        coalescence_kernel=coalescence_kernel,
        sticking_probability=sticking_probability,
    )
    t_eval = np.linspace(t_span[0], t_span[1], 41) if t_eval is None else np.asarray(t_eval)
    solution = solve_ivp(
        rhs,
        t_span,
        initial,
        method="BDF",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"population balance integration failed: {solution.message}")
    return {"t": solution.t, "n": solution.y.T, "grid": grid}


def census_to_sections(sizes, grid: SectionGrid, *, volume: float) -> np.ndarray:
    """Number density per section from a DEM census of floe diameters."""
    sizes = np.asarray(sizes, dtype=np.float64).reshape(-1)
    index = grid.section_of(sizes)
    counts = np.bincount(index, minlength=grid.n_sections).astype(np.float64)
    return counts / float(volume)


@dataclass
class ForecastComparison:
    """Comparison of a population-balance forecast with the DEM census history."""

    times: np.ndarray
    mean_model: np.ndarray
    mean_dem: np.ndarray
    density_model: np.ndarray
    density_dem: np.ndarray

    @property
    def mean_relative_error(self) -> float:
        """Relative error of the mean floe size over the forecast window (C3)."""
        return float(np.mean(np.abs(self.mean_model - self.mean_dem) / self.mean_dem))

    @property
    def density_relative_error(self) -> float:
        model = self.density_model.sum(axis=1)
        dem = self.density_dem.sum(axis=1)
        return float(np.mean(np.abs(model - dem) / np.maximum(dem, 1e-30)))

    def as_dict(self) -> dict:
        return dict(
            times=[float(v) for v in self.times],
            mean_model=[float(v) for v in self.mean_model],
            mean_dem=[float(v) for v in self.mean_dem],
            density_model=[[float(v) for v in row] for row in self.density_model],
            density_dem=[[float(v) for v in row] for row in self.density_dem],
            mean_relative_error=float(self.mean_relative_error),
            density_relative_error=float(self.density_relative_error),
        )


def compare_trajectories(
    grid: SectionGrid,
    times,
    model_density,
    dem_density,
) -> ForecastComparison:
    """Mean floe size of both sides, computed with the same section moments."""
    volumes = grid.volumes

    def mean_size(density):
        # volume-equivalent diameter with the campaign convention v = D**3
        number = density.sum(axis=1)
        mass = density @ volumes
        return np.where(number > 0.0, mass / np.maximum(number, 1e-300), np.nan) ** (1.0 / 3.0)

    return ForecastComparison(
        times=np.asarray(times, dtype=np.float64),
        mean_model=mean_size(np.asarray(model_density, dtype=np.float64)),
        mean_dem=mean_size(np.asarray(dem_density, dtype=np.float64)),
        density_model=np.asarray(model_density, dtype=np.float64),
        density_dem=np.asarray(dem_density, dtype=np.float64),
    )


def smoluchowski_constant_kernel(t, *, n0: float = 1.0, rate: float = 1.0,
                                 n_monomers: int = 400) -> np.ndarray:
    """Analytic constant-kernel solution, used to validate the coalescence channel.

    ``N(t) = n0 / (1 + n0 K t / 2)`` for the total number density (in units of
    the initial monomer count).  Returned as the total number of clusters per
    unit volume.
    """
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    return n0 * n_monomers / (1.0 + 0.5 * n0 * rate * t)