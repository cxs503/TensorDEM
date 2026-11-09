"""Beverloo orifice-discharge law: DEM measurement + correlation fit (stage 1).

The correlation (Beverloo, Leniger & van de Velde, 1961) for the mass flow rate
of a cohesionless granular material out of a flat-bottomed silo is

    W = C · rho_b · sqrt(g) · (D - k·d)^(5/2)

with ``D`` the orifice diameter, ``d`` the particle diameter, ``rho_b`` the bulk
density and ``C ≈ 0.58``, ``k ≈ 1.4`` for glass beads (``C`` in 0.55-0.60,
``k`` in 1.2-1.6).  The *measured, falsifiable* content of the law is the
exponent 5/2: the flow rate is set by the free-falling jet through the aperture
and is independent of the filling height.

This module measures ``W`` from a real DEM run (mass balance over discharge
windows, never the correlation value re-labelled as a measurement) and then
tests the law in two ways:

* **fixed-k fit** ``log W = log A + p log(D - k d)`` with ``k = 1.4``: the
  slope ``p`` is the quantity to compare with 2.5 (the acceptance is
  ``|p - 2.5| <= 0.2`` with ``R^2 >= 0.95``);
* **free-k fit**: ``k`` is scanned over ``[0, 3]`` and the value maximising
  ``R^2`` is reported, so a systematic ``k`` offset (the classical finite-size
  correction) can not masquerade as an exponent error.

``C`` follows from the intercept and the measured bulk density, and the
``g``-scaling is checked separately (``W ∝ sqrt(g)``) in the study driver.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .kernel import Material, SiloGeometry, SiloFlow
from .packing import build_bed

__all__ = [
    "BEVERLOO_C",
    "BEVERLOO_K",
    "DischargeCase",
    "run_discharge",
    "beverloo_prediction",
    "fit_power_law",
    "fit_law",
]

BEVERLOO_C = 0.58
BEVERLOO_K = 1.4


@dataclass(frozen=True)
class DischargeCase:
    """One flat-bottom discharge run (SI)."""

    orifice_diameter: float
    particle_diameter: float = 0.0045
    silo_radius: float = 0.045
    fill_height: float = 0.10
    wall_mu: float = 0.3
    seed: int = 1
    settle_time: float = 0.08
    discard_time: float = 0.15
    window_time: float = 0.20
    n_windows: int = 6
    kn: float = 1.0e4
    mu: float = 0.4
    mu_rolling: float = 0.05

    def __post_init__(self) -> None:
        for name in ("orifice_diameter", "particle_diameter", "silo_radius", "fill_height"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.orifice_diameter >= 2.0 * self.silo_radius:
            raise ValueError("orifice must be smaller than the silo diameter")
        if self.fill_height <= self.orifice_diameter:
            raise ValueError("fill_height must exceed the orifice diameter")
        if self.n_windows < 2:
            raise ValueError("at least two discharge windows are required")


def run_discharge(case: DischargeCase, *, dt: float | None = None, flow: SiloFlow | None = None) -> dict:
    """Run one flat-bottom discharge case and return the measured flow rate."""
    material = Material(
        radius=0.5 * case.particle_diameter,
        mu=case.mu,
        mu_rolling=case.mu_rolling,
        kn=case.kn,
    )
    geometry = SiloGeometry(
        cylinder_radius=case.silo_radius,
        cylinder_height=case.fill_height + 0.02,
        outlet_radius=0.5 * case.orifice_diameter,
    )
    t_wall = time.time()
    if flow is None:
        flow = build_bed(
            geometry, material, case.fill_height,
            wall_mu=case.wall_mu, seed=case.seed, settle_time=case.settle_time, dt=dt,
        )
    rho_b = flow.bulk_density(case.fill_height)
    bed_mass = flow.bed_mass
    n_bed = flow.n
    flow.plugged = False
    n_discard = int(round(case.discard_time / flow.dt))
    for _ in range(n_discard):
        flow.step()
    n_win = int(round(case.window_time / flow.dt))
    windows = []
    for _ in range(case.n_windows):
        m0 = flow.mass_exited
        n0 = flow.n_exited
        for _ in range(n_win):
            flow.step()
        windows.append({
            "mass_kg": flow.mass_exited - m0,
            "particles": flow.n_exited - n0,
            "time_s": n_win * flow.dt,
        })
    rates = np.array([w["mass_kg"] / w["time_s"] for w in windows])
    # steady state = the last 50 % of the measurement window series
    n_steady = max(1, len(rates) // 2)
    steady = rates[-n_steady:]
    return {
        "case": {
            "orifice_diameter": case.orifice_diameter,
            "particle_diameter": case.particle_diameter,
            "orifice_over_particle": case.orifice_diameter / case.particle_diameter,
            "silo_radius": case.silo_radius,
            "fill_height": case.fill_height,
            "wall_mu": case.wall_mu,
            "seed": case.seed,
            "kn": case.kn,
            "mu": case.mu,
            "mu_rolling": case.mu_rolling,
        },
        "dt": flow.dt,
        "n_bed": n_bed,
        "bed_mass_kg": bed_mass,
        "rho_bulk": rho_b,
        "n_windows": case.n_windows,
        "window_time_s": case.window_time,
        "discard_time_s": case.discard_time,
        "flow_rate_windows_kg_s": [float(r) for r in rates],
        "flow_rate_kg_s": float(np.mean(steady)),
        "flow_rate_std_kg_s": float(np.std(steady, ddof=1)) if len(steady) > 1 else 0.0,
        "flow_rate_steady_windows": n_steady,
        "particles_exited": int(sum(w["particles"] for w in windows)),
        "n_final": flow.n,
        "energy": flow.balance(),
        "wall_clock_s": time.time() - t_wall,
        "steps": flow.steps,
    }


def beverloo_prediction(orifice_diameter: float, particle_diameter: float, rho_bulk: float,
                        *, c: float = BEVERLOO_C, k: float = BEVERLOO_K,
                        gravity: float = 9.81) -> float:
    """Beverloo correlation ``W = C rho_b sqrt(g) (D - k d)^(5/2)`` (kg/s)."""
    eff = orifice_diameter - k * particle_diameter
    if eff <= 0.0:
        return 0.0
    return c * rho_bulk * math.sqrt(gravity) * eff ** 2.5


def fit_power_law(effective: np.ndarray, rate: np.ndarray) -> dict:
    """Least-squares ``log W = log A + p log(D_eff)`` with ``R^2``."""
    x = np.log(np.asarray(effective, dtype=np.float64))
    y = np.log(np.asarray(rate, dtype=np.float64))
    if x.size < 2:
        raise ValueError("at least two points are required")
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return {
        "exponent": float(slope),
        "amplitude": float(math.exp(intercept)),
        "intercept": float(intercept),
        "r2": r2,
        "n_points": int(x.size),
        "residual_rms_log": float(math.sqrt(ss_res / max(x.size - 2, 1))),
    }


def fit_law(orifice: np.ndarray, particle_diameter: float, rate: np.ndarray,
            rho_bulk: float, *, k_grid: int = 301) -> dict:
    """Fit the Beverloo law: exponent at ``k = 1.4`` and a free-``k`` variant."""
    orifice = np.asarray(orifice, dtype=np.float64)
    rate = np.asarray(rate, dtype=np.float64)
    fixed = fit_power_law(orifice - BEVERLOO_K * particle_diameter, rate)
    fixed["k"] = BEVERLOO_K
    fixed["c"] = fixed["amplitude"] / (rho_bulk * math.sqrt(9.81))
    best = None
    for k in np.linspace(0.0, 3.0, k_grid):
        eff = orifice - k * particle_diameter
        if np.any(eff <= 0.0):
            continue
        f = fit_power_law(eff, rate)
        if best is None or f["r2"] > best["r2"]:
            best = dict(f, k=float(k), c=f["amplitude"] / (rho_bulk * math.sqrt(9.81)))
    return {"fixed_k": fixed, "free_k": best}