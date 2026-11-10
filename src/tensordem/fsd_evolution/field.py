"""Floe-field FSD evolution driven by the DEM-calibrated breakage kernel.

The field model is a mass-conserving sectional population balance (the operator of
:mod:`tensordem.fsd_evolution.pbe`, validated against analytic controls there)
driven by the *measured* kernel of
:mod:`tensordem.fsd_evolution.dynamic_kernel`:

.. math::

   \\frac{\\partial n(d,t)}{\\partial t} =
      -\\nu(d) n(d,t) - \\mu(d) n(d,t)
      + \\int_d^\\infty \\nu(d')\\beta(d|d')\\,n(d',t)\\,dd' + S(d)

with

* ``nu(d) = nu_ref (d/d_ref)**q`` -- breakage frequency measured on the dynamic
  DEM instrument (rate, 1/s),
* ``beta`` -- the measured daughter catalogue,
* ``S(d)`` -- level-ice calving, injected into the two largest sections,
* ``mu(d) = 2 v_lat / d`` -- lateral melt.  This is *geometry, not a fit*: a floe
  whose diameter shrinks at a constant lateral melt velocity ``v_lat`` disappears
  in ``d / (2 v_lat)``, so the first-order removal rate is ``2 v_lat / d``
  (exponent ``p = 1``).  ``v_lat = 1e-5 m/s`` is the upper end of the measured
  lateral melt band (Maykut & Perovich 1987: 1e-6 .. 1e-5 m/s).

Two conventions are reported for the steady-state exponent, and neither is
switched after the fact:

* ``alpha_differential`` -- slope of ``log10 (n_k / dD_k)`` against ``log10 D_k``,
  this is the exponent the campaign window ``[2, 3]`` refers to;
* ``alpha_cumulative`` -- slope of ``log10 N(>D)`` = ``alpha_differential - 1``,
  which is the convention observations of floe-size distributions usually quote
  as ``-2 .. -3``.

The mass budget is closed by construction (two-pivot daughter placement) and
checked against the integrated source/melt mismatch, so the residual is a real
test of the operator and the ODE solver, not an identity on the initial state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .pbe import SectionGrid, daughter_matrix, mass_moment, solve_breakage

__all__ = [
    "FieldConfig",
    "breadth_exponent",
    "FieldState",
    "build_field",
    "evolve_field",
    "fit_power_law",
    "mass_budget_residual",
]

#: Lateral melt velocity (m/s): upper end of the Maykut & Perovich (1987) band.
LATERAL_MELT_VELOCITY = 1.0e-5


@dataclass(frozen=True)
class FieldConfig:
    """Sectional grid, forcing and integration controls of the floe field."""

    d_min: float = 0.02
    d_max: float = 100.0
    n_sections: int = 26
    initial_diameter: float = 1.0
    initial_area_fraction: float = 0.05
    source_diameter: float = 100.0
    source_time_scale: float = 500.0
    lateral_melt_velocity: float = LATERAL_MELT_VELOCITY
    t_end: float = 1000.0
    n_samples: int = 4000
    seed: int = 0

    def __post_init__(self) -> None:
        if not (0.0 < self.d_min < self.d_max):
            raise ValueError("require 0 < d_min < d_max")
        if self.n_sections < 8:
            raise ValueError("n_sections must be >= 8")
        if self.t_end <= 0.0:
            raise ValueError("t_end must be positive")
        if self.n_samples < 500:
            raise ValueError("n_samples must be >= 500")

    @property
    def grid(self) -> SectionGrid:
        return SectionGrid(d_min=self.d_min, d_max=self.d_max,
                           n_sections=self.n_sections)


@dataclass
class FieldState:
    """A frozen FSD snapshot (per section, per unit area of floe field)."""

    t: float
    density: np.ndarray
    diameters: np.ndarray
    widths: np.ndarray
    areas: np.ndarray

    @property
    def differential(self) -> np.ndarray:
        return self.density / self.widths

    @property
    def cumulative(self) -> np.ndarray:
        return np.cumsum(self.density[::-1])[::-1]

    def scaling_mask(self, *, drop_floor: int = 2, drop_top: int = 3) -> np.ndarray:
        keep = np.zeros(self.density.size, dtype=bool)
        keep[drop_floor:self.density.size - drop_top] = True
        return keep & (self.density > 0.0)


def build_field(kernel, config: FieldConfig | None = None) -> dict:
    """Assemble the grid, rates, daughter matrix, source and melt sink."""
    cfg = config or FieldConfig()
    grid = cfg.grid
    d = grid.diameters
    rates = kernel.rate(d)
    rule = kernel.daughter_rule(seed=cfg.seed)
    matrix = daughter_matrix(grid, rule, n_samples=cfg.n_samples, seed=cfg.seed)
    # -- initial condition: monodisperse field of d = initial_diameter -------
    idx0 = int(np.argmin(np.abs(d - cfg.initial_diameter)))
    area = grid.measure
    n0 = np.zeros(d.size, dtype=np.float64)
    total_area = float(cfg.initial_area_fraction)          # m^2 of floe per m^2
    n0[idx0] = total_area / area[idx0]
    # -- level-ice calving source: the two largest sections -----------------
    source = np.zeros(d.size, dtype=np.float64)
    top = np.argsort(d)[-2:]
    source_mass_rate = total_area / cfg.source_time_scale
    source[top] = (source_mass_rate / 2.0) / area[top]
    # -- lateral melt sink --------------------------------------------------
    sink = 2.0 * cfg.lateral_melt_velocity / d
    return dict(
        config=cfg,
        grid=grid,
        diameters=d,
        widths=np.gradient(d),
        rates=rates,
        daughter=matrix,
        source=source,
        sink=sink,
        n0=n0,
        source_mass_rate=float(source_mass_rate),
        daughter_mass_defect=_daughter_mass_defect(grid, matrix),
    )


def _daughter_mass_defect(grid: SectionGrid, matrix: np.ndarray) -> float:
    mass = grid.measure
    residual = matrix.T @ mass - mass
    return float(np.max(np.abs(residual) / mass))


def evolve_field(built: dict, *, t_eval=None) -> dict:
    """Integrate the field balance and return the snapshots."""
    cfg = built["config"]
    if t_eval is None:
        # Snapshots must resolve the initial transient (a monodisperse field at
        # d = 1 m shatters within ~1/nu) as well as the 1000 s evolution,
        # otherwise the trapezoidal source/melt budget integral is limited by
        # the snapshot spacing rather than by the operator.
        early = np.logspace(math.log10(cfg.t_end) - 7.0, math.log10(cfg.t_end), 160)
        t_eval = np.unique(np.concatenate(([0.0], early)))
        t_eval = np.clip(t_eval, 0.0, cfg.t_end)
    solution = solve_breakage(
        built["grid"],
        built["n0"],
        rates=built["rates"],
        daughter=built["daughter"],
        source=built["source"],
        sink=built["sink"],
        t_span=(0.0, cfg.t_end),
        t_eval=t_eval,
    )
    states = [
        FieldState(
            t=float(t),
            density=np.asarray(n, dtype=np.float64),
            diameters=built["diameters"],
            widths=built["widths"],
            areas=built["grid"].measure,
        )
        for t, n in zip(solution["t"], solution["n"])
    ]
    return dict(solution=solution, states=states, built=built)


def fit_power_law(state: FieldState, mask: np.ndarray) -> dict:
    """Fit the differential and cumulative FSD exponents in the scaling range."""
    d = state.diameters[mask]
    diff = state.differential[mask]
    cum = state.cumulative[mask]
    out = {}
    for name, values in (("differential", diff), ("cumulative", cum)):
        keep = values > 0.0
        lx = np.log10(d[keep])
        ly = np.log10(values[keep])
        design = np.vstack([lx, np.ones(lx.size)]).T
        solution, *_ = np.linalg.lstsq(design, ly, rcond=None)
        residual = ly - design @ solution
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((ly - ly.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
        out[name] = dict(
            alpha=float(-solution[0]),
            r2=float(r2),
            n_points=int(lx.size),
            rms_log10=float(math.sqrt(ss_res / lx.size)),
            d_min=float(d[keep].min()),
            d_max=float(d[keep].max()),
        )
    return out


def breadth_exponent(built: dict, *, x_min: float = 0.15, x_max: float = 0.9) -> dict:
    """Exponent ``r`` of the cumulative daughter count of the *discrete operator*.

    ``r`` enters the flux-balance prediction ``alpha = 1 + q + r`` and must
    therefore be measured on exactly the daughter operator that produced the
    FSD -- the sectional matrix, not the raw event catalogue (the catalogue is a
    set of delta spectra; the operator smears them onto the grid).
    ``C(x) = sum_{i: d_i < x d_j} N[i, j]`` is evaluated for every parent section
    ``j`` in the fit range and averaged before the log-log fit.
    """
    grid = built["grid"]
    matrix = built["daughter"]
    d = grid.diameters
    grid_x = np.logspace(math.log10(x_min), math.log10(x_max), 12)
    curves = []
    for j, parent in enumerate(d):
        column = matrix[:, j]
        if column.sum() <= 0.0:
            continue
        if not (d[0] * 2.0 <= parent <= d[-1] / 2.0):
            continue
        curves.append(np.asarray([float(column[d < g * parent].sum()) for g in grid_x]))
    if not curves:
        return dict(r=0.0, r2=float("nan"), n_parents=0)
    mean_curve = np.mean(np.asarray(curves), axis=0)
    keep = mean_curve > 0.0
    if keep.sum() < 3:
        return dict(r=0.0, r2=float("nan"), n_parents=len(curves))
    lx = np.log10(grid_x[keep])
    ly = np.log10(mean_curve[keep])
    design = np.vstack([lx, np.ones(lx.size)]).T
    solution, *_ = np.linalg.lstsq(design, ly, rcond=None)
    residual = ly - design @ solution
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return dict(r=float(max(solution[0], 0.0)), r2=float(r2),
                n_parents=int(len(curves)), x_min=float(x_min), x_max=float(x_max))


def operator_mass_defect(built: dict, density) -> float:
    """Relative defect of the *operator's* area conservation.

    This is the mass-conservation statement the campaign asks for: the discrete
    breakage operator must neither create nor destroy floe area, i.e.
    ``sum_k m_k dn_k = source_mass - sum_k m_k mu_k n_k`` up to round-off.  It is
    evaluated on the state the integrator returned, so it tests the operator and
    the daughter placement, not the ODE solver's dense output.
    """
    from .pbe import breakage_rhs

    n = np.asarray(density, dtype=np.float64)
    dn = breakage_rhs(n, rates=built["rates"], daughter=built["daughter"],
                      source=built["source"], sink=built["sink"])
    mass = built["grid"].measure
    gained = float(np.dot(mass, dn))
    source_mass = float(np.dot(mass, built["source"]))
    sink_mass = float(np.dot(mass, built["sink"] * n))
    scale = max(abs(source_mass), abs(sink_mass), 1e-30)
    return float(abs(gained - (source_mass - sink_mass)) / scale)


def mass_budget_residual(built: dict, states) -> float:
    """Relative mismatch between the area moment and the integrated source/melt.

    ``dM/dt = source_mass - sum_k mu_k m_k n_k``; the residual below is the
    drift of ``M(t) - M(0) - integral``, i.e. a test of the operator and of the
    ODE solve, not of the initial condition.
    """
    grid = built["grid"]
    source_mass = float(np.dot(grid.measure, built["source"]))
    mass = np.asarray([mass_moment(grid, s.density) for s in states], dtype=np.float64)
    times = np.asarray([s.t for s in states], dtype=np.float64)
    sink_rate = np.asarray(
        [float(np.dot(grid.measure, built["sink"] * s.density)) for s in states],
        dtype=np.float64,
    )
    net = source_mass - sink_rate
    integral = np.concatenate(
        ([0.0], np.cumsum(0.5 * (net[1:] + net[:-1]) * np.diff(times)))
    )
    expected = mass[0] + integral
    scale = max(abs(mass[0]), 1e-30)
    return float(np.max(np.abs(mass - expected)) / scale)