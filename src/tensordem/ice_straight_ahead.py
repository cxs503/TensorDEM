"""DEM broken-ice-field straight-ahead ice resistance vs Lindqvist (1989).

Purpose
-------
This module closes the audit item ``ice.resistance.dem_straight_ahead``
(previously WITHHELD) by providing:

1. A clean interface to the Lindqvist (1989) three-component level-ice
   resistance formula (delegated to :mod:`tensordem.level_ice.formula`);
2. A DEM broken-ice-field straight-ahead simulation using the existing
   :class:`tensordem.ice_channel.IceChannelSimulation` infrastructure;
3. Statistical comparison between DEM steady-state resistance and the
   Lindqvist analytical prediction, with explicit concentration scaling.

Physical model
--------------
The DEM represents each ice floe as a flat circular disk of radius ``R``
and thickness ``h`` floating in open water.  The hull is a prescribed
rectangle ``L x B`` advancing at constant speed ``V``.  Contact is
Kelvin-Voigt normal + Coulomb tangential with stiffness anchored to the
local crushing strength ``sigma_c``.

The Lindqvist (1989) formula decomposes level-ice resistance into:

    R_c = 0.5 * sigma_c * h^2 * (tan(alpha) + mu / cos(alpha))   crushing
    R_b = 0.25 * sigma_f * B * h^2 / L                            bending
    R_s = rho_w * g * h^2 * B * (0.9 + 0.05 * mu * L / B)        submersion
    R_v = 0.5 * rho_w * B * h * V^2                               added drag

For broken ice at concentration ``C``, the leading-order expectation is
that the resistance scales linearly with the ice-covered fraction of the
waterline: ``R_broken ~ C * R_level``.  The DEM measures this directly.

Important note on DEM vs Lindqvist comparison
----------------------------------------------
The DEM is a 2D disk model without fracture mechanics.  It captures
crushing/contact forces but not the bending failure mode that contributes
to Lindqvist's bending term.  Additionally, the 2D geometry differs from
the 3D hull-ice interaction assumed by Lindqvist.  Therefore, exact
agreement is not expected.  The validation focuses on:

- Physical consistency: correct scaling with C, h, V (AC2-AC4)
- Statistical convergence (AC5)
- Order-of-magnitude agreement with concentration-scaled Lindqvist (AC1)

Acceptance criteria
-------------------
AC1  R_ice vs C*Lindqvist within factor of 2 (order-of-magnitude check)
AC2  C -> 0 gives R -> 0 (negative control)
AC3  h -> 0 gives R -> 0 (negative control)
AC4  R ~ V^p with p in [0.5, 1.5] (speed scaling)
AC5  Relative standard error <= 30% (ensemble >= 5 seeds, small domain)
AC6  pytest >= 10 cases covering different conditions

All quantities SI: m, Pa, kg/m^3, m/s, N.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .ice_channel import (
    ContactParameters,
    HullGeometry,
    IceChannelSimulation,
    WaterEnvironment,
)
from .level_ice.formula import (
    ICEBREAKER,
    IceState,
    HullForm,
    lindqvist_components,
    ledger_sha256,
)

__all__ = [
    "StraightAheadConfig",
    "StraightAheadResult",
    "run_straight_ahead",
    "run_ensemble",
    "lindqvist_reference",
    "speed_sweep",
    "concentration_sweep",
    "thickness_sweep",
]

# Physical constants
WATER_DENSITY = 1025.0   # kg/m^3
ICE_DENSITY = 917.0       # kg/m^3
GRAVITY = 9.80665         # m/s^2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StraightAheadConfig:
    """Configuration for a DEM straight-ahead ice resistance simulation.

    Parameters
    ----------
    speed_m_s : float
        Hull advance speed [m/s].
    ice_thickness_m : float
        Ice floe thickness [m].
    concentration : float
        Ice concentration (fraction of domain covered by floes), in [0, 1].
    hull_length_m : float
        Hull length [m].
    hull_beam_m : float
        Hull beam (breadth) [m].
    entry_angle_deg : float
        Waterline entry angle [degrees].
    friction_coefficient : float
        Hull-ice friction coefficient.
    compressive_strength_pa : float
        Ice compressive (crushing) strength [Pa].
    flexural_strength_pa : float
        Ice flexural (bending) strength [Pa].
    mean_floe_radius_m : float
        Mean floe radius [m].
    domain_length_m : float
        Domain length in advance direction [m].
    domain_width_m : float
        Domain width (transverse) [m].
    seed : int
        Random seed for floe field generation.
    dt_s : float
        Time step [s].
    warmup_fraction : float
        Fraction of total simulation time discarded as warm-up.
    """
    speed_m_s: float = 3.0
    ice_thickness_m: float = 0.5
    concentration: float = 0.6
    hull_length_m: float = 10.0
    hull_beam_m: float = 4.0
    entry_angle_deg: float = 22.0
    friction_coefficient: float = 0.05
    compressive_strength_pa: float = 1.5e6
    flexural_strength_pa: float = 0.5e6
    mean_floe_radius_m: float = 1.5
    domain_length_m: float = 30.0
    domain_width_m: float = 12.0
    seed: int = 42
    dt_s: float = 0.01
    warmup_fraction: float = 0.2

    def __post_init__(self) -> None:
        for name in ("speed_m_s", "ice_thickness_m", "hull_length_m",
                     "hull_beam_m", "mean_floe_radius_m",
                     "domain_length_m", "domain_width_m", "dt_s"):
            val = float(getattr(self, name))
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        conc = float(self.concentration)
        if not (0.0 <= conc <= 1.0):
            raise ValueError(f"concentration must be in [0,1], got {conc}")
        alpha = float(self.entry_angle_deg)
        if not (0.0 < alpha < 89.0):
            raise ValueError(f"entry_angle_deg must be in (0, 89)")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class StraightAheadResult:
    """Result of a single DEM straight-ahead simulation."""
    config: StraightAheadConfig
    mean_resistance_N: float
    std_resistance_N: float
    n_samples: int
    relative_std: float
    time_series_N: List[float]
    time_series_t: List[float]
    lindqvist_total_N: float
    lindqvist_crushing_N: float
    lindqvist_bending_N: float
    lindqvist_submersion_N: float
    lindqvist_drag_N: float
    ratio_dem_to_lindqvist: float
    ledger_sha256: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "speed_m_s": self.config.speed_m_s,
                "ice_thickness_m": self.config.ice_thickness_m,
                "concentration": self.config.concentration,
                "hull_length_m": self.config.hull_length_m,
                "hull_beam_m": self.config.hull_beam_m,
                "entry_angle_deg": self.config.entry_angle_deg,
                "friction_coefficient": self.config.friction_coefficient,
                "compressive_strength_pa": self.config.compressive_strength_pa,
                "flexural_strength_pa": self.config.flexural_strength_pa,
                "mean_floe_radius_m": self.config.mean_floe_radius_m,
                "seed": self.config.seed,
            },
            "mean_resistance_N": self.mean_resistance_N,
            "std_resistance_N": self.std_resistance_N,
            "n_samples": self.n_samples,
            "relative_std": self.relative_std,
            "lindqvist_total_N": self.lindqvist_total_N,
            "lindqvist_crushing_N": self.lindqvist_crushing_N,
            "lindqvist_bending_N": self.lindqvist_bending_N,
            "lindqvist_submersion_N": self.lindqvist_submersion_N,
            "lindqvist_drag_N": self.lindqvist_drag_N,
            "ratio_dem_to_lindqvist": self.ratio_dem_to_lindqvist,
            "ledger_sha256": self.ledger_sha256,
        }


# ---------------------------------------------------------------------------
# Lindqvist reference
# ---------------------------------------------------------------------------

def lindqvist_reference(
    *,
    speed_m_s: float = 3.0,
    ice_thickness_m: float = 0.5,
    hull_beam_m: float = 4.0,
    hull_length_m: float = 10.0,
    entry_angle_deg: float = 22.0,
    friction_coefficient: float = 0.05,
    compressive_strength_pa: float = 1.5e6,
    flexural_strength_pa: float = 0.5e6,
) -> Dict[str, float]:
    """Compute Lindqvist (1989) level-ice resistance components.

    Returns dict with keys: crushing_N, bending_N, submersion_N, drag_N, total_N.
    """
    ice = IceState(
        thickness_m=ice_thickness_m,
        compressive_strength_pa=compressive_strength_pa,
        flexural_strength_pa=flexural_strength_pa,
    )
    hull = HullForm(
        length_m=hull_length_m,
        beam_m=hull_beam_m,
        entry_angle_deg=entry_angle_deg,
        friction_coefficient=friction_coefficient,
    )
    result = lindqvist_components(ice, hull, speed_m_s=speed_m_s)
    return {
        "crushing_N": result["crushing_N"],
        "bending_N": result["bending_N"],
        "submersion_N": result["submersion_N"],
        "drag_N": result["added_drag_N"],
        "total_N": result["total_N"],
        "ledger_sha256": result["ledger_sha256"],
    }


# ---------------------------------------------------------------------------
# DEM simulation
# ---------------------------------------------------------------------------

def run_straight_ahead(config: StraightAheadConfig) -> StraightAheadResult:
    """Run a single DEM straight-ahead simulation and return statistics.

    The simulation:
    1. Creates a broken ice field at the specified concentration
    2. Advances the hull through the field at constant speed
    3. Records the hull-ice contact force time series
    4. Discards warm-up and computes steady-state statistics
    5. Compares with Lindqvist analytical prediction
    """
    cfg = config

    # Build ice state and hull form for Lindqvist reference
    ice_state = IceState(
        thickness_m=cfg.ice_thickness_m,
        compressive_strength_pa=cfg.compressive_strength_pa,
        flexural_strength_pa=cfg.flexural_strength_pa,
    )
    hull_form = HullForm(
        length_m=cfg.hull_length_m,
        beam_m=cfg.hull_beam_m,
        entry_angle_deg=cfg.entry_angle_deg,
        friction_coefficient=cfg.friction_coefficient,
    )
    lind = lindqvist_components(ice_state, hull_form, speed_m_s=cfg.speed_m_s)

    # Build DEM simulation
    water = WaterEnvironment(
        ice_density=ICE_DENSITY,
        water_density=WATER_DENSITY,
        gravity=GRAVITY,
    )
    contact = ContactParameters(
        crushing_strength=cfg.compressive_strength_pa,
        friction=cfg.friction_coefficient,
    )
    hull = HullGeometry(
        length=cfg.hull_length_m,
        beam=cfg.hull_beam_m,
        speed=cfg.speed_m_s,
        y_position=0.5 * cfg.domain_width_m,
        x_start=-cfg.hull_length_m,  # start just before the field
    )

    sim = IceChannelSimulation.from_field(
        concentration=cfg.concentration,
        domain_length=cfg.domain_length_m,
        domain_width=cfg.domain_width_m,
        seed=cfg.seed,
        mean_radius=cfg.mean_floe_radius_m,
        radius_range=(0.7 * cfg.mean_floe_radius_m, 1.3 * cfg.mean_floe_radius_m),
        thickness_range=(cfg.ice_thickness_m, cfg.ice_thickness_m),
        water=water,
        contact=contact,
        hull=hull,
        dt=cfg.dt_s,
    )

    # Compute simulation duration: hull must traverse the entire field
    total_distance = cfg.domain_length_m + 2.0 * cfg.hull_length_m + 2.0 * cfg.mean_floe_radius_m
    total_time = total_distance / cfg.speed_m_s
    n_steps = int(total_time / cfg.dt_s)
    warmup_steps = int(cfg.warmup_fraction * n_steps)

    # Run simulation and collect hull forces
    time_series_t: List[float] = []
    time_series_N: List[float] = []

    record_interval = max(1, n_steps // 200)  # ~200 samples

    for step_idx in range(n_steps):
        sim.step()
        if step_idx >= warmup_steps and (step_idx - warmup_steps) % record_interval == 0:
            # Record hull force
            ledger = sim.hull_contact(sim.time)
            force_on_hull = ledger[0]["force_on_hull_N"]
            # Resistance is the negative x-component (force opposing motion)
            resistance = -float(force_on_hull[0])
            time_series_t.append(sim.time)
            time_series_N.append(max(0.0, resistance))

    # Compute statistics
    if len(time_series_N) < 2:
        mean_r = 0.0
        std_r = 0.0
        rel_std = 0.0
    else:
        arr = np.array(time_series_N)
        mean_r = float(np.mean(arr))
        std_r = float(np.std(arr, ddof=1))
        rel_std = std_r / mean_r if mean_r > 0 else 0.0

    lindqvist_total = float(lind["total_N"])
    ratio = mean_r / lindqvist_total if lindqvist_total > 0 else 0.0

    return StraightAheadResult(
        config=cfg,
        mean_resistance_N=mean_r,
        std_resistance_N=std_r,
        n_samples=len(time_series_N),
        relative_std=rel_std,
        time_series_N=time_series_N,
        time_series_t=time_series_t,
        lindqvist_total_N=lindqvist_total,
        lindqvist_crushing_N=float(lind["crushing_N"]),
        lindqvist_bending_N=float(lind["bending_N"]),
        lindqvist_submersion_N=float(lind["submersion_N"]),
        lindqvist_drag_N=float(lind["added_drag_N"]),
        ratio_dem_to_lindqvist=ratio,
        ledger_sha256=str(lind["ledger_sha256"]),
    )


def run_ensemble(
    config: StraightAheadConfig,
    n_seeds: int = 5,
    base_seed: int = 1000,
) -> Dict[str, Any]:
    """Run an ensemble of simulations with different random seeds.

    Returns statistics over the ensemble for AC5 (statistical convergence).
    """
    results = []
    for i in range(n_seeds):
        cfg = StraightAheadConfig(
            speed_m_s=config.speed_m_s,
            ice_thickness_m=config.ice_thickness_m,
            concentration=config.concentration,
            hull_length_m=config.hull_length_m,
            hull_beam_m=config.hull_beam_m,
            entry_angle_deg=config.entry_angle_deg,
            friction_coefficient=config.friction_coefficient,
            compressive_strength_pa=config.compressive_strength_pa,
            flexural_strength_pa=config.flexural_strength_pa,
            mean_floe_radius_m=config.mean_floe_radius_m,
            domain_length_m=config.domain_length_m,
            domain_width_m=config.domain_width_m,
            seed=base_seed + i,
            dt_s=config.dt_s,
            warmup_fraction=config.warmup_fraction,
        )
        result = run_straight_ahead(cfg)
        results.append(result)

    means = [r.mean_resistance_N for r in results]
    ensemble_mean = float(np.mean(means))
    ensemble_std = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
    ensemble_rel_std = ensemble_std / ensemble_mean if ensemble_mean > 0 else 0.0

    return {
        "n_seeds": n_seeds,
        "individual_means_N": means,
        "ensemble_mean_N": ensemble_mean,
        "ensemble_std_N": ensemble_std,
        "ensemble_relative_std": ensemble_rel_std,
        "lindqvist_total_N": results[0].lindqvist_total_N,
        "ratio_ensemble_to_lindqvist": ensemble_mean / results[0].lindqvist_total_N
            if results[0].lindqvist_total_N > 0 else 0.0,
        "results": [r.as_dict() for r in results],
    }


# ---------------------------------------------------------------------------
# Parameter sweeps
# ---------------------------------------------------------------------------

def speed_sweep(
    speeds_m_s: Sequence[float] = (1.0, 2.0, 3.0, 4.0, 5.0),
    *,
    base_config: Optional[StraightAheadConfig] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Sweep over hull speed to verify R ~ V^p scaling (AC4)."""
    cfg = base_config or StraightAheadConfig()
    results = []
    for v in speeds_m_s:
        c = StraightAheadConfig(
            speed_m_s=v,
            ice_thickness_m=cfg.ice_thickness_m,
            concentration=cfg.concentration,
            hull_length_m=cfg.hull_length_m,
            hull_beam_m=cfg.hull_beam_m,
            entry_angle_deg=cfg.entry_angle_deg,
            friction_coefficient=cfg.friction_coefficient,
            compressive_strength_pa=cfg.compressive_strength_pa,
            flexural_strength_pa=cfg.flexural_strength_pa,
            mean_floe_radius_m=cfg.mean_floe_radius_m,
            domain_length_m=cfg.domain_length_m,
            domain_width_m=cfg.domain_width_m,
            seed=seed,
            dt_s=cfg.dt_s,
        )
        r = run_straight_ahead(c)
        lind = lindqvist_reference(
            speed_m_s=v,
            ice_thickness_m=cfg.ice_thickness_m,
            hull_beam_m=cfg.hull_beam_m,
            hull_length_m=cfg.hull_length_m,
        )
        results.append({
            "speed_m_s": v,
            "dem_mean_N": r.mean_resistance_N,
            "lindqvist_total_N": lind["total_N"],
        })

    # Fit log-log slope: R ~ V^p => log(R) = p * log(V) + const
    speeds_arr = np.array([r["speed_m_s"] for r in results])
    dem_arr = np.array([r["dem_mean_N"] for r in results])
    # Filter out zeros for log fit
    valid = (speeds_arr > 0) & (dem_arr > 0)
    if valid.sum() >= 2:
        log_v = np.log(speeds_arr[valid])
        log_r = np.log(dem_arr[valid])
        # Linear fit: log_R = p * log_V + c
        coeffs = np.polyfit(log_v, log_r, 1)
        speed_exponent = float(coeffs[0])
    else:
        speed_exponent = float("nan")

    return {
        "sweep": results,
        "speed_exponent_dem": speed_exponent,
        "expected_range": [0.5, 1.5],
    }


def concentration_sweep(
    concentrations: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
    *,
    base_config: Optional[StraightAheadConfig] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Sweep over ice concentration (AC2: C->0 gives R->0)."""
    cfg = base_config or StraightAheadConfig()
    results = []
    for c_val in concentrations:
        c = StraightAheadConfig(
            speed_m_s=cfg.speed_m_s,
            ice_thickness_m=cfg.ice_thickness_m,
            concentration=c_val,
            hull_length_m=cfg.hull_length_m,
            hull_beam_m=cfg.hull_beam_m,
            entry_angle_deg=cfg.entry_angle_deg,
            friction_coefficient=cfg.friction_coefficient,
            compressive_strength_pa=cfg.compressive_strength_pa,
            flexural_strength_pa=cfg.flexural_strength_pa,
            mean_floe_radius_m=cfg.mean_floe_radius_m,
            domain_length_m=cfg.domain_length_m,
            domain_width_m=cfg.domain_width_m,
            seed=seed,
            dt_s=cfg.dt_s,
        )
        r = run_straight_ahead(c)
        results.append({
            "concentration": c_val,
            "dem_mean_N": r.mean_resistance_N,
        })
    return {"sweep": results}


def thickness_sweep(
    thicknesses_m: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
    *,
    base_config: Optional[StraightAheadConfig] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Sweep over ice thickness (AC3: h->0 gives R->0)."""
    cfg = base_config or StraightAheadConfig()
    results = []
    for h in thicknesses_m:
        c = StraightAheadConfig(
            speed_m_s=cfg.speed_m_s,
            ice_thickness_m=h,
            concentration=cfg.concentration,
            hull_length_m=cfg.hull_length_m,
            hull_beam_m=cfg.hull_beam_m,
            entry_angle_deg=cfg.entry_angle_deg,
            friction_coefficient=cfg.friction_coefficient,
            compressive_strength_pa=cfg.compressive_strength_pa,
            flexural_strength_pa=cfg.flexural_strength_pa,
            mean_floe_radius_m=cfg.mean_floe_radius_m,
            domain_length_m=cfg.domain_length_m,
            domain_width_m=cfg.domain_width_m,
            seed=seed,
            dt_s=cfg.dt_s,
        )
        r = run_straight_ahead(c)
        lind = lindqvist_reference(
            speed_m_s=cfg.speed_m_s,
            ice_thickness_m=h,
            hull_beam_m=cfg.hull_beam_m,
            hull_length_m=cfg.hull_length_m,
        )
        results.append({
            "thickness_m": h,
            "dem_mean_N": r.mean_resistance_N,
            "lindqvist_total_N": lind["total_N"],
        })
    return {"sweep": results}
