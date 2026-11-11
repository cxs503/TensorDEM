"""Janssen wall-stress saturation in a filled silo (stage 2).

Janssen's classical slab analysis of a deep silo balances the weight of a slice
against the wall friction and gives, with ``z`` measured *down* from the free
surface,

    sigma_v(z) = P_inf (1 - exp(-z / z0)),   P_inf = rho_b g R / (2 mu_w K),
                                             z0    = R / (2 mu_w K)

where ``R`` is the silo radius, ``mu_w`` the wall friction coefficient and ``K``
the ratio of horizontal to vertical stress.  The two parameters are not
independent: ``P_inf / z0 = rho_b g`` (a specific weight), so the *shape* of the profile tests the
momentum balance without any assumption about ``K`` -- that identity is the
parameter-free acceptance used here.

Two independent measurements are made on a settled, plugged bed:

* the **wall stress** ``sigma_h(z)`` = wall normal force per unit wall area of a
  horizontal layer (this is Janssen's ``K sigma_v``);
* the **vertical stress** ``sigma_v(z)`` from the force balance of the slice
  above the plane, ``sigma_v = (W_above - wall friction above) / A``, which needs
  no stress-tensor assumption.

Stage acceptance: the fitted saturation amplitude agrees with Janssen's
``P_inf`` within 10 %, and the ``P_inf / z0 = rho_b g`` identity holds within 10 %.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .kernel import Material, SiloGeometry, SiloFlow
from .packing import build_bed

__all__ = ["JanssenCase", "run_janssen", "janssen_profile", "fit_janssen"]


@dataclass(frozen=True)
class JanssenCase:
    """Deep-silo static-pressure case (SI)."""

    silo_radius: float = 0.025
    fill_height: float = 0.30
    outlet_radius: float = 0.008
    wall_mu: float = 0.40
    particle_diameter: float = 0.0045
    n_layers: int = 20
    settle_time: float = 0.10
    seed: int = 11
    kn: float = 1.0e4
    mu: float = 0.40

    def __post_init__(self) -> None:
        for name in ("silo_radius", "fill_height", "outlet_radius", "particle_diameter"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.outlet_radius >= self.silo_radius:
            raise ValueError("outlet must be smaller than the silo")
        if self.n_layers < 4:
            raise ValueError("at least four layers are required")


def janssen_profile(z: np.ndarray, p_inf: float, z0: float) -> np.ndarray:
    """``sigma(z) = P_inf (1 - exp(-z / z0))``."""
    return p_inf * (1.0 - np.exp(-np.asarray(z, dtype=np.float64) / z0))


def fit_janssen(depth: np.ndarray, sigma: np.ndarray) -> dict:
    """Least-squares fit of the Janssen form (``scipy.optimize.curve_fit``)."""
    from scipy.optimize import curve_fit

    d = np.asarray(depth, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    keep = np.isfinite(d) & np.isfinite(s) & (d > 0.0)
    d, s = d[keep], s[keep]
    if d.size < 4:
        raise ValueError("need at least four usable layers")
    p0 = [max(float(np.max(s)), 1e-6), max(float(np.max(d)) / 2.0, 1e-6)]
    try:
        popt, _ = curve_fit(janssen_profile, d, s, p0=p0, maxfev=20000,
                            bounds=([0.0, 1e-6], [np.inf, np.inf]))
    except Exception:
        return {"ok": False}
    resid = s - janssen_profile(d, *popt)
    ss_tot = float(np.sum((s - np.mean(s)) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0.0 else float("nan")
    return {
        "ok": True,
        "p_inf": float(popt[0]),
        "z0": float(popt[1]),
        "r2": r2,
        "rms": float(np.sqrt(np.mean(resid ** 2))),
        "n_layers": int(d.size),
        "p_inf_over_z0": float(popt[0] / popt[1]),
        "P_inf": float(popt[0]),
    }


def run_janssen(case: JanssenCase, *, dt: float | None = None) -> dict:
    """Fill a deep silo, let it settle and measure the wall/base pressures."""
    t_wall = time.time()
    material = Material(radius=0.5 * case.particle_diameter, mu=case.mu, kn=case.kn)
    geometry = SiloGeometry(
        cylinder_radius=case.silo_radius,
        cylinder_height=case.fill_height + 0.02,
        outlet_radius=case.outlet_radius,
    )
    flow = build_bed(
        geometry, material, case.fill_height, wall_mu=case.wall_mu,
        seed=case.seed, settle_time=case.settle_time, plug_outlet=True, dt=dt,
    )
    for _ in range(int(round(0.05 / flow.dt))):
        flow.step()

    free = float(np.max(flow.pos[:, 2]))
    rho_b = flow.bulk_density(free)
    n_lay = case.n_layers
    z_edges = np.linspace(0.0, max(free, 0.02), n_lay + 1)
    wall = flow.wall_stress_profile(z_edges)
    vert = flow.vertical_stress_profile(z_edges)

    depth = np.maximum(free - wall["z_centers"], 1e-9)
    order = np.argsort(depth)
    wall_fit = fit_janssen(depth[order], wall["sigma_w"][order])
    depth_v = np.maximum(free - vert["z_upper"], 1e-9)
    order_v = np.argsort(depth_v)
    vert_fit = fit_janssen(depth_v[order_v], np.maximum(vert["sigma_v"][order_v], 0.0))

    p_base = float(np.sum(flow.plate_normal_force)) / (
        math.pi * (geometry.cylinder_radius ** 2 - geometry.outlet_radius ** 2))
    # Janssen theory with the *measured* stress ratio
    k_meas = None
    if wall_fit.get("ok") and vert_fit.get("ok"):
        sel = vert["sigma_v"] > 1e-6
        if np.any(sel):
            k_meas = float(np.mean(wall["sigma_w"][sel] / vert["sigma_v"][sel]))
    # specific weight [N/m^3]: P_inf/z0 = rho_b g is the parameter-free
    # Janssen identity (a *stress per metre*, not a pressure)
    rho_g = rho_b * material.gravity
    result = {
        "case": {
            "silo_radius": case.silo_radius,
            "fill_height": case.fill_height,
            "outlet_radius": case.outlet_radius,
            "wall_mu": case.wall_mu,
            "particle_diameter": case.particle_diameter,
            "mu": case.mu,
            "kn": case.kn,
            "seed": case.seed,
            "n_layers": n_lay,
        },
        "dt": flow.dt,
        "n_particles": flow.n,
        "free_surface": free,
        "rho_bulk": rho_b,
        "rho_g": rho_g,
        "rho_g_R": rho_g,  # legacy key name, value = rho_b g (was rho_b g R)
        "base_pressure_pa": p_base,
        "base_pressure_janssen_pa": (rho_g * geometry.cylinder_radius
                                     / (2.0 * case.wall_mu * k_meas)
                                     if k_meas else None),
        "k_measured": k_meas,
        "wall_profile": {
            "z_centers": wall["z_centers"].tolist(),
            "depth": depth.tolist(),
            "sigma_w": wall["sigma_w"].tolist(),
            "count": wall["count"].tolist(),
        },
        "vertical_profile": {
            "z_upper": vert["z_upper"].tolist(),
            "depth": depth_v.tolist(),
            "sigma_v": vert["sigma_v"].tolist(),
        },
        "wall_fit": wall_fit,
        "vertical_fit": vert_fit,
        "energy": flow.balance(),
        "wall_clock_s": time.time() - t_wall,
    }
    # parameter-free Janssen identity  P_inf / z0 = rho_b g  (specific weight)
    if wall_fit.get("ok"):
        ratio = wall_fit["p_inf_over_z0"]
        result["identity_ratio"] = ratio
        result["identity_error"] = abs(ratio - rho_g) / rho_g
    # saturation amplitude vs Janssen with the measured stress ratio
    if wall_fit.get("ok") and k_meas:
        p_inf_th = rho_g * geometry.cylinder_radius / (2.0 * case.wall_mu * k_meas)
        z0_th = case.silo_radius / (2.0 * case.wall_mu * k_meas)
        result["janssen_theory"] = {"p_inf": p_inf_th, "z0": z0_th, "k": k_meas}
        result["p_inf_rel_error"] = abs(wall_fit["p_inf"] - p_inf_th) / p_inf_th
        result["z0_rel_error"] = abs(wall_fit["z0"] - z0_th) / z0_th
    return result