"""Beverloo silo-discharge benchmark: REAL 3-D DEM (m5481).

Physical background
-------------------
A cylindrical silo with a *flat* bottom plate and a centred circular orifice
drains under gravity.  The steady mass flow rate ``W`` follows the Beverloo
correlation (Beverloo et al., 1961):

    W = C · rho_b · sqrt(g) · (D - k·d)^(5/2)

where ``D`` is the orifice diameter, ``d`` the particle diameter, ``rho_b`` the
bulk density, and ``C``, ``k`` empirical constants (literature: C ≈ 0.58,
k ≈ 1.4 for glass beads; C ∈ [0.55, 0.60], k ∈ [1.2, 1.6]).

This module runs a genuine 3-D DEM simulation (no formula shortcut): the
particles are integrated with the ``granular3d_core`` kernel, the orifice is a
flat plate of thickness ``t`` (0, d, or 2 d) with a circular hole, and the
discharged mass is counted as particles pass fully through the plate.  The
steady flow rate is the mean over >= 5 time windows with a +/- 1 sigma band.

All quantities are SI (m, kg, s).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .granular3d_core import (
    Material3D, ContactState, build_pairs, contact_forces,
    hcp_positions_in_cylinder,
)


class SiloError(ValueError):
    pass


@dataclass(frozen=True)
class SiloCase:
    """Flat-bottom circular-orifice silo case (SI)."""
    # material
    radius: float = 0.005
    density: float = 2500.0
    mu: float = 0.4
    mu_rolling: float = 0.02
    restitution: float = 0.3
    kn: float = 1.0e5
    kt_ratio: float = 0.5
    gravity: float = 9.81
    # geometry
    silo_radius: float = 0.12
    fill_height: float = 0.08
    orifice_diameter: float = 0.04
    plate_thickness: float = 0.005   # 0, d, or 2 d
    wall_mu: float = 0.0             # smooth side wall (glass-on-glass)
    # numerics
    packing_gap: float = 0.0
    jitter: float = 0.002
    seed: int = 5481
    dt: Optional[float] = None
    settle_time: float = 0.15        # pre-discharge relaxation
    discharge_time: float = 1.2      # total discharge duration
    window: float = 0.2              # flow-rate window length
    n_windows: int = 5               # >= 5 windows for the steady mean

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def orifice_radius(self) -> float:
        return 0.5 * self.orifice_diameter

    @property
    def material(self) -> Material3D:
        return Material3D(radius=self.radius, density=self.density, mu=self.mu,
                          mu_rolling=self.mu_rolling, restitution=self.restitution,
                          kn=self.kn, kt_ratio=self.kt_ratio, gravity=self.gravity)

    @property
    def timestep(self) -> float:
        if self.dt is not None:
            return self.dt
        return self.material.timestep

    def __post_init__(self) -> None:
        for name in ("radius", "density", "kn", "kt_ratio", "silo_radius",
                     "fill_height", "orifice_diameter", "settle_time",
                     "discharge_time", "window"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and v > 0.0):
                raise SiloError(f"{name} must be finite and positive")
        for name in ("mu", "mu_rolling", "restitution", "packing_gap", "jitter",
                     "wall_mu", "plate_thickness"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and v >= 0.0):
                raise SiloError(f"{name} must be finite and nonnegative")
        if not 0.0 <= self.restitution < 1.0:
            raise SiloError("restitution must lie in [0, 1)")
        if self.orifice_diameter >= 2.0 * self.silo_radius:
            raise SiloError("orifice must be smaller than the silo diameter")
        if self.orifice_diameter < 3.0 * self.diameter:
            raise SiloError("orifice must be at least 3 particle diameters")
        if self.n_windows < 5:
            raise SiloError("need at least 5 flow-rate windows")


def _initial_packing(case: SiloCase) -> np.ndarray:
    return hcp_positions_in_cylinder(
        case.radius, case.silo_radius, case.fill_height,
        packing_gap=case.packing_gap, jitter=case.jitter, seed=case.seed)


def _plate_forces(pos, vel, mat, case, f):
    """Forces from the flat bottom plate with a centred circular hole.

    The plate is solid where ``radial > a`` and ``-t < z < 0`` (a = orifice
    radius, t = plate thickness).  Three contact surfaces:

    * top face    (z = 0,  radial > a): pushes +z;
    * bottom face (z = -t, radial < a): pushes -z (only for a particle whose
      centre is already below the plate bottom, i.e. it is emerging from the
      hole);
    * edge        (cylinder of radius a, -t < z < 0): pushes the particle
      away from the solid, i.e. -radial inside the hole and +radial outside.

    For the benchmark the hole is at least 3 particle diameters wide
    (a >= 1.5 r), so a particle in the middle of the hole never touches the
    edge; only particles at the rim do.
    """
    r = case.radius
    a = case.orifice_radius
    t = case.plate_thickness
    kn = mat.kn
    gamma_w = mat.gamma_wall
    mu_w = case.wall_mu

    radial = np.hypot(pos[:, 0], pos[:, 1])
    nz = pos[:, 2]

    # ---- top face (z = 0, outside the hole) --------------------------
    top_pen = r - nz
    top_hit = (top_pen > 0.0) & (radial > a + 0.3 * r)
    if top_hit.any():
        idx = np.nonzero(top_hit)[0]
        fn = np.clip(kn * top_pen[idx] - gamma_w * vel[idx, 2], 0.0, None)
        f[idx, 2] += fn
        if mu_w > 0.0:
            vt = np.stack([vel[idx, 0], vel[idx, 1]], axis=1)
            vtn = np.linalg.norm(vt, axis=1)
            ft = np.where(vtn > 0.0, mu_w * fn[:, None] * vt / np.maximum(vtn[:, None], 1e-30), 0.0)
            f[idx, 0] += ft[:, 0]
            f[idx, 1] += ft[:, 1]

    # ---- bottom face (z = -t, inside the hole) -----------------------
    # Only a particle whose centre is below the plate bottom can press on the
    # underside; a particle above the plate is on the other side of it.
    if t > 0.0:
        bot_pen = nz + r + t
        bot_hit = (bot_pen > 0.0) & (radial < a - 0.3 * r) & (nz < -t)
        if bot_hit.any():
            idx = np.nonzero(bot_hit)[0]
            fn = np.clip(kn * bot_pen[idx] - gamma_w * vel[idx, 2], 0.0, None)
            f[idx, 2] -= fn

    # ---- edge (cylinder of radius a spanning -t < z < 0) -------------
    edge_hit = (np.abs(radial - a) < r) & (nz > -t - r) & (nz < r)
    if edge_hit.any():
        idx = np.nonzero(edge_hit)[0]
        pen = r - np.abs(radial[idx] - a)
        fn = np.clip(kn * pen, 0.0, None)
        rr = np.maximum(radial[idx], 1e-30)
        ux = pos[idx, 0] / rr
        uy = pos[idx, 1] / rr
        sign = np.where(radial[idx] < a, -1.0, 1.0)
        f[idx, 0] += sign * fn * ux
        f[idx, 1] += sign * fn * uy
        if mu_w > 0.0:
            vrad = vel[idx, 0] * ux + vel[idx, 1] * uy
            vt = np.stack([vel[idx, 0] - vrad * ux, vel[idx, 1] - vrad * uy], axis=1)
            vtn = np.linalg.norm(vt, axis=1)
            ft = np.where(vtn > 0.0, mu_w * fn[:, None] * vt / np.maximum(vtn[:, None], 1e-30), 0.0)
            f[idx, 0] += ft[:, 0]
            f[idx, 1] += ft[:, 1]


def _side_wall(pos, vel, mat, case, f):
    """Cylindrical side wall at radial = silo_radius (smooth or rough)."""
    r = case.radius
    R = case.silo_radius
    radial = np.hypot(pos[:, 0], pos[:, 1])
    pen = radial + r - R
    hit = pen > 0.0
    if hit.any():
        idx = np.nonzero(hit)[0]
        fn = np.clip(mat.kn * pen[idx] - mat.gamma_wall *
                      (vel[idx, 0] * pos[idx, 0] / np.maximum(radial[idx], 1e-30)
                       + vel[idx, 1] * pos[idx, 1] / np.maximum(radial[idx], 1e-30)),
                      0.0, None)
        ux = pos[idx, 0] / np.maximum(radial[idx], 1e-30)
        uy = pos[idx, 1] / np.maximum(radial[idx], 1e-30)
        # normal points inward: -u
        f[idx, 0] -= fn * ux
        f[idx, 1] -= fn * uy
        if case.wall_mu > 0.0:
            vrad = vel[idx, 0] * ux + vel[idx, 1] * uy
            vt = np.stack([vel[idx, 0] - vrad * ux, vel[idx, 1] - vrad * uy], axis=1)
            vtn = np.linalg.norm(vt, axis=1)
            ft = np.where(vtn > 0.0, case.wall_mu * fn[:, None] * vt / np.maximum(vtn[:, None], 1e-30), 0.0)
            f[idx, 0] += ft[:, 0]
            f[idx, 1] += ft[:, 1]


def simulate_silo(case: SiloCase, *, return_history: bool = False) -> dict:
    """Run the silo discharge and measure the steady mass flow rate.

    Returns a dict with the per-window flow rates, the steady mean +/- 1 sigma,
    the bulk density, the particle count, and the discharge bookkeeping.
    """
    mat = case.material
    r = case.radius
    mass = mat.mass
    dt = case.timestep
    pos = _initial_packing(case)
    n0 = len(pos)
    vel = np.zeros((n0, 3))
    omega = np.zeros((n0, 3))
    state = ContactState()

    # alive mask (particles removed once fully discharged)
    alive = np.ones(n0, dtype=bool)
    discharged_mass = 0.0
    discharge_z = -(case.plate_thickness + r)   # centre below this => discharged

    # time bookkeeping
    t_settle = case.settle_time
    t_total = t_settle + case.discharge_time
    n_steps = int(math.ceil(t_total / dt))

    # flow-rate windows (measured during the discharge phase)
    win = case.window
    win_edges = [t_settle + i * win for i in range(case.n_windows + 1)]
    win_edges = np.array(win_edges)
    # cumulative discharged mass sampled at each window edge
    win_mass = [0.0]
    win_step = [int(round(e / dt)) for e in win_edges]

    step_of_win = 0
    for step in range(n_steps):
        # active = alive particles still in the domain (the discharge stream
        # below the plate is excluded: it does not feed back into the silo)
        active = alive & (pos[:, 2] > discharge_z - 4.0 * r)
        idx_a = np.nonzero(active)[0]
        pa = pos[idx_a]
        va = vel[idx_a]
        oa = omega[idx_a]

        iu, ju = build_pairs(pa, r)
        gi, gj = idx_a[iu], idx_a[ju]
        f, tq = contact_forces(pa, va, oa, iu, ju, state, mat, dt,
                              state_keys=gi * n0 + gj)

        _plate_forces(pa, va, mat, case, f)
        _side_wall(pa, va, mat, case, f)
        f[:, 2] -= mass * case.gravity

        va = va + f * (dt / mass)
        oa = oa + tq * (dt / mat.inertia)
        pa = pa + va * dt

        pos[idx_a] = pa
        vel[idx_a] = va
        omega[idx_a] = oa

        # discharge: particles whose centre is below the plate bottom
        below = alive & (pos[:, 2] < discharge_z)
        if below.any():
            discharged_mass += mass * int(below.sum())
            alive[below] = False

        # sample the window edges
        if step_of_win < len(win_step) and step >= win_step[step_of_win]:
            win_mass.append(discharged_mass)
            step_of_win += 1

    # flow rates per window
    rates = []
    for i in range(case.n_windows):
        dm = win_mass[i + 1] - win_mass[i]
        rates.append(dm / win)

    rates = np.array(rates)
    W_mean = float(rates.mean())
    W_std = float(rates.std(ddof=1)) if len(rates) > 1 else 0.0

    # bulk density from the initial packing
    bulk_volume = math.pi * case.silo_radius ** 2 * case.fill_height
    rho_b = mass * n0 / bulk_volume

    out = {
        "n_initial": n0,
        "n_discharged": int((~alive).sum()),
        "discharged_mass_kg": discharged_mass,
        "flow_rate_kg_s": W_mean,
        "flow_rate_std_kg_s": W_std,
        "flow_rate_windows_kg_s": [float(x) for x in rates],
        "bulk_density_kg_m3": rho_b,
        "packing_fraction": rho_b / case.density,
        "dt": dt,
        "steps": n_steps,
        "orifice_diameter": case.orifice_diameter,
        "particle_diameter": case.diameter,
        "D_over_d": case.orifice_diameter / case.diameter,
        "plate_thickness": case.plate_thickness,
        "silo_radius": case.silo_radius,
        "fill_height": case.fill_height,
        "mu": case.mu,
        "mu_rolling": case.mu_rolling,
        "restitution": case.restitution,
        "gravity": case.gravity,
    }
    if return_history:
        out["positions"] = pos
        out["alive"] = alive
    return out


def beverloo_fit(W: np.ndarray, D: np.ndarray, d: float, rho_b: float, g: float,
                 *, k_range=(1.0, 2.0), n_k: int = 201) -> dict:
    """Fit W = C rho_b sqrt(g) (D - k d)^(5/2) for C and k.

    For a fixed k the model is linear in C, so C(k) is closed form; we minimise
    the log-residual over a 1-D grid of k.
    """
    W = np.asarray(W, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    logW = np.log(W)
    log_pref = np.log(rho_b * math.sqrt(g))
    best = None
    for k in np.linspace(k_range[0], k_range[1], n_k):
        Deff = D - k * d
        if (Deff <= 0).any():
            continue
        logC = logW - 2.5 * np.log(Deff) - log_pref
        C = float(np.exp(logC.mean()))
        resid = logW - (np.log(C) + log_pref + 2.5 * np.log(Deff))
        ssr = float(np.sum(resid ** 2))
        if best is None or ssr < best["ssr"]:
            best = {"C": C, "k": float(k), "ssr": ssr}
    if best is None:
        raise SiloError("no valid k in range (D - k d <= 0)")
    # per-point relative error at the fitted (C, k)
    Deff = D - best["k"] * d
    W_pred = best["C"] * rho_b * math.sqrt(g) * Deff ** 2.5
    rel_err = (W_pred - W) / W
    best["W_pred_kg_s"] = [float(x) for x in W_pred]
    best["rel_err"] = [float(x) for x in rel_err]
    best["max_rel_err"] = float(np.max(np.abs(rel_err)))
    return best


def beverloo_prediction(W_C: float, k: float, D: float, d: float,
                       rho_b: float, g: float) -> float:
    Deff = D - k * d
    if Deff <= 0:
        return 0.0
    return W_C * rho_b * math.sqrt(g) * Deff ** 2.5


__all__ = [
    "SiloError", "SiloCase", "simulate_silo", "beverloo_fit",
    "beverloo_prediction",
]
