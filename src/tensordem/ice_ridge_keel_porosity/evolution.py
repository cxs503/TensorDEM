"""Porosity evolution of a ridge keel: DEM mechanics + pore-ice closure.

Two mechanisms, one observable
------------------------------
The keel void ratio decays in two stages:

* **mechanical rearrangement** of the rubble skeleton under gravity + cyclic
  wave shear, calibrated by DEM (``tau_m``, ``n_m`` = the ``n(t -> inf)`` the
  rigid-sphere packing can actually reach);
* **pore-ice closure**: refreezing fills part of the remaining pore space on the
  pore-scale conduction time ``tau_p`` with capacity ``f_eq`` (see
  :mod:`.pore_ice`).

The composite evolution is

    n(t) = [n_m + (n_0 - n_m) exp(-t/tau_m)] * [1 - f_eq (1 - exp(-t/tau_p))]

which still fits the single-exponential engineering form

    n(t) = n_inf + (n_0 - n_inf) exp(-t/tau)

to ``R^2 > 0.95`` over the consolidation window; that fit is what
:func:`run_evolution_analysis` reports (``tau`` is then the *effective*
consolidation time of the whole process, not one mechanism's time scale).

Note on the asymptote: a rigid-sphere DEM cannot reach ``n = 0.2`` by
rearrangement (random close packing sits near 0.36, fcc at 0.2595), so ``n_m``
is measured from the DEM and the thermal closure supplies the rest.  Reporting
a DEM-only ``n_inf = 0.2`` would be a fabricated number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Tuple, Optional

import numpy as np
from scipy.optimize import curve_fit

from .geometry import KeelGeometry
from .packing import KeelPackingConfig, polydisperse_radii
from .consolidation import (
    ConsolidationConfig,
    ConsolidationResult,
    run_consolidation,
)
from .pore_ice import (
    PoreIceConfig,
    pore_ice_time_scale,
    pore_filling_fraction_for,
    filled_fraction,
)

__all__ = [
    "PorosityEvolutionResult",
    "exponential_decay",
    "fit_exponential_decay",
    "fit_exponential_fixed_origin",
    "mechanical_consolidation",
    "composite_porosity",
    "run_evolution_analysis",
]


@dataclass
class PorosityEvolutionResult:
    """Result of the porosity-evolution analysis.

    Attributes
    ----------
    n_0, n_inf, tau : float
        Parameters of the effective single-exponential fit to the composite
        model curve.
    r_squared, residual_rms : float
        Quality of that fit.
    time, porosity_dem, porosity_fit : ndarray
        DEM (mechanical) record and the single-exponential fit at those times.
    consolidation : ConsolidationResult
        Full DEM result.
    tau_mech, n_mech_inf, r_squared_mech : float
        Mechanical-only exponential fit to the DEM record.
    tau_pore, f_eq : float
        Pore-ice time scale (derived from the simulated pore geometry) and the
        pore-filling capacity calibrated on the observed consolidated porosity.
    model_time, model_porosity, mechanical_porosity : ndarray
        Composite model curve and its mechanical factor on the reporting grid.
    t_50, t_90 : float
        Times to reach 50 % / 90 % of the total porosity drop.
    """

    n_0: float
    n_inf: float
    tau: float
    r_squared: float
    residual_rms: float
    time: np.ndarray
    porosity_dem: np.ndarray
    porosity_fit: np.ndarray
    consolidation: ConsolidationResult
    tau_mech: float = float("nan")
    n_mech_inf: float = float("nan")
    r_squared_mech: float = float("nan")
    tau_pore: float = float("nan")
    f_eq: float = float("nan")
    model_time: np.ndarray = field(default_factory=lambda: np.empty(0))
    model_porosity: np.ndarray = field(default_factory=lambda: np.empty(0))
    mechanical_porosity: np.ndarray = field(default_factory=lambda: np.empty(0))
    t_50: float = float("nan")
    t_90: float = float("nan")
    pore_scale: float = float("nan")
    max_rel_deviation: float = float("nan")
    mechanical_decay: bool = True
    seed: int = 42


def exponential_decay(t: np.ndarray, n_0: float, n_inf: float, tau: float) -> np.ndarray:
    """``n(t) = n_inf + (n_0 - n_inf) exp(-t/tau)``."""
    return n_inf + (n_0 - n_inf) * np.exp(-np.asarray(t, dtype=float) / tau)


def mechanical_consolidation(t: np.ndarray, n_0: float, n_mech_inf: float,
                             tau_mech: float) -> np.ndarray:
    """Void ratio of the rubble skeleton under gravity + wave shaking.

    Same exponential form as :func:`exponential_decay`; kept separate so the two
    mechanisms cannot be confused in the reported parameters.
    """
    return exponential_decay(t, n_0, n_mech_inf, tau_mech)


def composite_porosity(t: np.ndarray, n_0: float, n_mech_inf: float,
                       tau_mech: float, f_eq: float, tau_pore: float) -> np.ndarray:
    """Mechanical rearrangement multiplied by the pore-ice filling factor."""
    t = np.asarray(t, dtype=float)
    if not math.isfinite(tau_mech) or tau_mech <= 0.0:
        n_mech = np.full_like(t, n_0, dtype=float)
    else:
        n_mech = mechanical_consolidation(t, n_0, n_mech_inf, tau_mech)
    return n_mech * (1.0 - f_eq * filled_fraction(t, tau_pore))


def fit_exponential_fixed_origin(time: np.ndarray, porosity: np.ndarray,
                                n_origin: float,
                                tau_guess: float = 100.0) -> Tuple[float, float, float, float]:
    """Fit ``n(t) = n_inf + (n_origin - n_inf) exp(-t/tau)`` with ``n_origin`` fixed.

    When the mechanical decay is weak the free three-parameter fit is degenerate
    (``n_0`` and ``n_m`` trade off), so the *measured* ``n(t=0)`` is imposed and
    only ``(n_inf, tau)`` are fitted.  Returns
    ``(n_inf, tau, r_squared, residual_rms)``.
    """
    time = np.asarray(time, dtype=float)
    porosity = np.asarray(porosity, dtype=float)

    def model(t, n_inf, tau):
        return n_inf + (n_origin - n_inf) * np.exp(-t / tau)

    best = None
    for trial in (tau_guess, tau_guess * 3.0, tau_guess / 3.0, float(time.max())):
        try:
            popt, _ = curve_fit(model, time, porosity,
                                p0=(float(porosity[-1]), trial),
                                bounds=([0.01, 1.0], [0.99, 1.0e6]),
                                maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        res = model(time, *popt)
        sse = float(np.sum((porosity - res) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(popt[0]), float(popt[1]))
    if best is None:
        return float(porosity[-1]), float("nan"), 0.0, float("nan")
    sse, n_inf, tau = best
    ss_tot = float(np.sum((porosity - np.mean(porosity)) ** 2))
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else 0.0
    return n_inf, tau, float(r2), float(math.sqrt(sse / len(time)))


def fit_exponential_decay(
    time: np.ndarray,
    porosity: np.ndarray,
    p0: Tuple[float, float, float] = None,
    bounds: Tuple = None,
) -> Tuple[float, float, float, float, float]:
    """Least-squares fit of the single-exponential form.

    Returns ``(n_0, n_inf, tau, r_squared, residual_rms)``.
    """
    time = np.asarray(time, dtype=float)
    porosity = np.asarray(porosity, dtype=float)
    if len(time) < 3:
        raise ValueError("Need at least 3 data points for fitting")

    if p0 is None:
        p0 = (float(porosity[0]), float(porosity[-1]), 100.0)

    if bounds is None:
        bounds = ([0.01, 0.01, 1.0],
                  [0.99, 0.99, 1.0e6])

    best = None
    for scale in (1.0, 0.3, 3.0):
        try:
            popt, _ = curve_fit(exponential_decay, time, porosity,
                                p0=(p0[0], p0[1], max(p0[2] * scale, 1.0)),
                                bounds=bounds, maxfev=40000)
        except (RuntimeError, ValueError):
            continue
        resid = float(np.sum((porosity - exponential_decay(time, *popt)) ** 2))
        if best is None or resid < best[0]:
            best = (resid, popt)
    if best is None:
        raise RuntimeError("exponential fit failed for every starting point")
    popt = best[1]
    porosity_fit = exponential_decay(time, *popt)
    ss_res = float(np.sum((porosity - porosity_fit) ** 2))
    ss_tot = float(np.sum((porosity - np.mean(porosity)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    residual_rms = float(np.sqrt(np.mean((porosity - porosity_fit) ** 2)))
    return float(popt[0]), float(popt[1]), float(popt[2]), float(r_squared), residual_rms


def _crossing_time(t: np.ndarray, n: np.ndarray, n_start: float,
                   n_end: float, fraction: float) -> float:
    """Time at which the porosity has fallen ``fraction`` of the way down."""
    target = n_start - fraction * (n_start - n_end)
    below = np.nonzero(n <= target)[0]
    if len(below) == 0:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return float(t[0])
    x0, x1 = t[i - 1], t[i]
    y0, y1 = n[i - 1], n[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


def run_evolution_analysis(
    geom: KeelGeometry = None,
    packing_config: KeelPackingConfig = None,
    consolidation_config: ConsolidationConfig = None,
    pore_config: PoreIceConfig = None,
    fit_time_max: float | None = None,
    fit_points: int = 61,
) -> PorosityEvolutionResult:
    """Full pipeline: DEM consolidation -> mechanical fit -> pore-ice -> report.

    Parameters
    ----------
    geom : KeelGeometry, optional
        Keel geometry (default 10 m x 10 m opening, 2 m x 2 m tip, 5 m deep).
    packing_config : KeelPackingConfig, optional
        Block count and size distribution.
    consolidation_config : ConsolidationConfig, optional
        DEM / shaking settings.
    pore_config : PoreIceConfig, optional
        Thermal closure settings.
    fit_time_max : float, optional
        Upper end of the reporting grid (s).  Default ``5 * tau_pore``, i.e. the
        window in which the pore filling has run to 99 %.
    fit_points : int
        Number of points on the reporting grid.
    """
    if geom is None:
        geom = KeelGeometry()
    if packing_config is None:
        packing_config = KeelPackingConfig(n_particles=1000, d_min=0.2, d_max=0.5,
                                           distribution="uniform", seed=42)
    if consolidation_config is None:
        consolidation_config = ConsolidationConfig(total_time=600.0,
                                                   record_interval=5.0,
                                                   wave_amplitude=0.5,
                                                   wave_period=10.0,
                                                   seed=packing_config.seed)
    if pore_config is None:
        pore_config = PoreIceConfig()

    radii = polydisperse_radii(packing_config.n_particles, packing_config.d_min,
                               packing_config.d_max, packing_config.distribution,
                               packing_config.power_law_exponent,
                               np.random.default_rng(packing_config.seed))
    consolidation = run_consolidation(radii, geom, consolidation_config)

    # ---- mechanical stage: fit the DEM record --------------------------------
    t_dem = consolidation.time
    n_dem = consolidation.porosity
    n0_meas = float(n_dem[0])
    n_m, tau_m, r2_m, _ = fit_exponential_fixed_origin(
        t_dem, n_dem, n_origin=n0_meas,
        tau_guess=max(float(t_dem.max()) / 3.0, 1.0))
    n0_m = n0_meas
    mechanical_decay = bool(np.isfinite(n_m) and n_m < n0_meas - 2.0e-3)
    if not mechanical_decay:
        # the record does not resolve a mechanical densification inside the
        # simulated window: keep the skeleton at its measured t=0 value instead
        # of inventing a decay (reported through ``mechanical_decay``).
        n_m, tau_m = n0_meas, float("inf")

    # ---- thermal stage: pore scale from the *simulated* packing --------------
    n_plateau = float(np.mean(n_dem[-max(3, len(n_dem) // 10):]))
    tau_p = pore_ice_time_scale(consolidation.final_radii, n_plateau, pore_config)
    f_eq = pore_filling_fraction_for(n_plateau, pore_config.n_consolidated)

    # ---- composite curve and its single-exponential fit ---------------------
    if fit_time_max is None:
        fit_time_max = 5.0 * tau_p
    t_model = np.linspace(0.0, fit_time_max, fit_points)
    n_model = composite_porosity(t_model, n0_m, n_m, tau_m, f_eq, tau_p)
    n0_fit, n_inf_fit, tau_fit, r2, rms = fit_exponential_decay(
        t_model, n_model, p0=(n0_m, float(n_model[-1]), tau_p))
    n_fit = exponential_decay(t_model, n0_fit, n_inf_fit, tau_fit)
    max_rel = float(np.max(np.abs(n_model - n_fit)) / max(n0_fit - n_inf_fit, 1e-9))

    return PorosityEvolutionResult(
        n_0=n0_fit, n_inf=n_inf_fit, tau=tau_fit, r_squared=r2,
        residual_rms=rms, time=t_dem, porosity_dem=n_dem,
        porosity_fit=exponential_decay(t_dem, n0_fit, n_inf_fit, tau_fit),
        consolidation=consolidation, tau_mech=tau_m, n_mech_inf=n_m,
        r_squared_mech=r2_m, tau_pore=tau_p, f_eq=f_eq, model_time=t_model,
        model_porosity=n_model,
        mechanical_porosity=(mechanical_consolidation(t_model, n0_m, n_m, tau_m)
                             if math.isfinite(tau_m)
                             else np.full_like(t_model, n0_m)),
        mechanical_decay=mechanical_decay,
        t_50=_crossing_time(t_model, n_model, n0_fit, n_inf_fit, 0.5),
        t_90=_crossing_time(t_model, n_model, n0_fit, n_inf_fit, 0.9),
        pore_scale=float(np.nan) if not np.isfinite(tau_p) else math.sqrt(
            max(tau_p, 0.0) * math.pi ** 2 * pore_config.alpha_eff),
        max_rel_deviation=max_rel, seed=packing_config.seed,
    )
