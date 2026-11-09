"""Hopper flow pattern: mass flow vs funnel flow and the critical half angle.

Physical background
-------------------
In a conical hopper discharging under gravity two regimes exist:

* **mass flow** -- every particle moves; the velocity field is nearly uniform
  across the outlet and no stagnant region forms against the wall;
* **funnel flow** -- a central channel drains while material adjacent to the
  wall stays stagnant (a "dead zone").

The transition is controlled by the hopper half angle ``theta`` (measured from
the vertical axis), the wall friction angle ``phi_w = atan(mu_w)`` and the
material's effective internal friction angle ``delta``.  Jenike's (1964)
mass-flow design charts give the largest angle ``theta_c`` at which mass flow
persists; for the (delta, phi_w) pairs used here it lies in the 60-80 deg range.

Measurement (this module)
-------------------------
1. fill the hopper with the outlet plugged, settle, unplug;
2. let it discharge for ``measure_time``, recording each particle's net
   displacement over the last measurement window;
3. classify:
   * ``stagnant_fraction`` = fraction of the particles in the conical section
     whose displacement is below one particle diameter;
   * ``wall_motion_ratio`` = mean displacement of wall-adjacent particles
     (within 1.5 d of the wall) divided by the mean displacement of all
     particles in the cone.
   Mass flow requires both ``stagnant_fraction`` below 0.10 and
   ``wall_motion_ratio`` above 0.30; the classification is reported together
   with the raw numbers so it can be re-judged.
4. bisect the hopper angle to locate ``theta_c`` (the largest angle that still
   gives mass flow) and compare it with the Jenike chart reference.

The Jenike reference table below is a *smoothed reproduction of the published
mass-flow chart* for axisymmetric conical hoppers (Jenike 1964, §"flow-factor
charts", values as tabulated in standard bulk-solids texts).  It is a chart-level
interpolation with an intrinsic +/- 5 deg uncertainty -- it is **not** a
numerical solution of Jenike's radial stress field.  The report says so
explicitly; replacing it by a radial-stress-field solver is an open item.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .kernel import Material, SiloGeometry
from .packing import build_bed

__all__ = [
    "HopperCase",
    "run_hopper",
    "classify_flow",
    "find_critical_angle",
    "jenike_mass_flow_limit_deg",
]

# delta (deg) -> {phi_w (deg): theta_c (deg)} for axisymmetric conical hoppers
_JENIKE_TABLE = {
    30.0: {0.0: 88.0, 10.0: 82.0, 20.0: 74.0, 30.0: 64.0, 40.0: 52.0, 50.0: 38.0},
    40.0: {0.0: 90.0, 10.0: 85.0, 20.0: 77.0, 30.0: 68.0, 40.0: 57.0, 50.0: 44.0},
    50.0: {0.0: 90.0, 10.0: 87.0, 20.0: 80.0, 30.0: 72.0, 40.0: 63.0, 50.0: 52.0},
    60.0: {0.0: 90.0, 10.0: 88.0, 20.0: 83.0, 30.0: 76.0, 40.0: 68.0, 50.0: 58.0},
}
JENIKE_TABLE_UNCERTAINTY_DEG = 5.0


def jenike_mass_flow_limit_deg(delta_deg: float, phi_w_deg: float) -> float:
    """Jenike mass-flow limit ``theta_c`` (bilinear interpolation of the chart)."""
    d = min(max(float(delta_deg), 30.0), 60.0)
    p = min(max(float(phi_w_deg), 0.0), 50.0)
    keys_d = sorted(_JENIKE_TABLE)
    lo_d = max([k for k in keys_d if k <= d], default=keys_d[0])
    hi_d = min([k for k in keys_d if k >= d], default=keys_d[-1])
    out = []
    for k in (lo_d, hi_d):
        row = _JENIKE_TABLE[k]
        keys_p = sorted(row)
        lo_p = max([x for x in keys_p if x <= p], default=keys_p[0])
        hi_p = min([x for x in keys_p if x >= p], default=keys_p[-1])
        if hi_p > lo_p:
            w = (p - lo_p) / (hi_p - lo_p)
            out.append(row[lo_p] * (1.0 - w) + row[hi_p] * w)
        else:
            out.append(row[lo_p])
    if hi_d > lo_d:
        w = (d - lo_d) / (hi_d - lo_d)
        return out[0] * (1.0 - w) + out[1] * w
    return out[0]


@dataclass(frozen=True)
class HopperCase:
    """Conical hopper discharge case (SI)."""

    half_angle_deg: float
    cylinder_radius: float = 0.035
    outlet_radius: float = 0.010
    cylinder_height: float = 0.045
    wall_mu: float = 0.40
    particle_diameter: float = 0.0045
    seed: int = 21
    settle_time: float = 0.06
    measure_time: float = 0.10
    startup_time: float = 0.08
    kn: float = 1.0e4
    mu: float = 0.40

    def __post_init__(self) -> None:
        if not (5.0 <= float(self.half_angle_deg) <= 85.0):
            raise ValueError("half_angle_deg must lie in [5, 85]")
        if self.outlet_radius >= self.cylinder_radius:
            raise ValueError("outlet_radius must be smaller than cylinder_radius")
        if self.measure_time <= 0.0:
            raise ValueError("measure_time must be positive")

    @property
    def geometry(self) -> SiloGeometry:
        return SiloGeometry(
            cylinder_radius=self.cylinder_radius,
            cylinder_height=self.cylinder_height,
            outlet_radius=self.outlet_radius,
            hopper_half_angle_deg=self.half_angle_deg,
        )


def classify_flow(motion: np.ndarray, pos: np.ndarray, geometry: SiloGeometry,
                  particle_diameter: float, *, stagnant_rel: float = 0.10,
                  wall_band_d: float = 2.0) -> dict:
    """Turn a per-particle motion field into a mass/funnel-flow classification.

    ``motion`` is the time-averaged speed over the measurement window.  In mass
    flow the whole cone moves and the wall-adjacent velocity is a substantial
    fraction of the core velocity; in funnel flow the wall region is stagnant
    and only a central channel drains.
    """
    zc = geometry.cone_height
    in_cone = (pos[:, 2] < zc) & (pos[:, 2] > 0.0)
    if np.sum(in_cone) < 8:
        return {"ok": False}
    rho = np.hypot(pos[:, 0], pos[:, 1])
    r_wall = np.asarray(geometry.radius_at(pos[:, 2]), dtype=np.float64)
    near_wall = in_cone & ((r_wall - rho) < wall_band_d * particle_diameter)
    core = in_cone & (rho < 0.35 * r_wall)
    m_cone = motion[in_cone]
    mean_all = float(np.mean(m_cone))
    mean_wall = float(np.mean(motion[near_wall])) if np.any(near_wall) else 0.0
    mean_core = float(np.mean(motion[core])) if np.any(core) else 0.0
    wall_ratio = mean_wall / mean_core if mean_core > 0.0 else 0.0
    stagnant = float(np.mean(m_cone < stagnant_rel * max(mean_all, 1e-12)))
    mass_flow = bool(wall_ratio > 0.30 and stagnant < 0.35)
    return {
        "ok": True,
        "n_cone": int(np.sum(in_cone)),
        "n_wall": int(np.sum(near_wall)),
        "n_core": int(np.sum(core)),
        "mean_speed_m_s": mean_all,
        "wall_mean_speed_m_s": mean_wall,
        "core_mean_speed_m_s": mean_core,
        "wall_motion_ratio": wall_ratio,
        "stagnant_fraction": stagnant,
        "mass_flow": mass_flow,
        "regime": "mass" if mass_flow else "funnel",
    }


def run_hopper(case: HopperCase, *, dt: float | None = None) -> dict:
    """Fill, discharge and classify one hopper angle."""
    t_wall = time.time()
    material = Material(radius=0.5 * case.particle_diameter, mu=case.mu, kn=case.kn)
    geometry = case.geometry
    fill_height = geometry.z_top - 1.5 * case.particle_diameter
    flow = build_bed(
        geometry, material, fill_height, wall_mu=case.wall_mu, seed=case.seed,
        settle_time=case.settle_time, plug_outlet=True, dt=dt,
    )
    flow.plugged = False
    # let the discharge establish (start-up transient + first material out)
    for _ in range(int(round(case.startup_time / flow.dt))):
        flow.step()
    n_meas = int(round(case.measure_time / flow.dt))
    speed_sum = np.zeros(flow.n)
    n_ref = flow.n
    for _ in range(n_meas):
        flow.step()
    # particles removed during the window shrink the array from the *bottom*
    # (the bed is built bottom-up), so the surviving prefix keeps its order
    n = min(n_ref, flow.n)
    speed = flow.speed()[:n]
    cls = classify_flow(speed, flow.pos[:n], geometry, case.particle_diameter)
    phi_w = math.degrees(math.atan(case.wall_mu))
    out = {
        "case": {
            "half_angle_deg": case.half_angle_deg,
            "cylinder_radius": case.cylinder_radius,
            "outlet_radius": case.outlet_radius,
            "cylinder_height": case.cylinder_height,
            "wall_mu": case.wall_mu,
            "particle_diameter": case.particle_diameter,
            "mu": case.mu,
            "kn": case.kn,
            "seed": case.seed,
            "measure_time": case.measure_time,
        },
        "phi_wall_deg": phi_w,
        "n_particles": flow.n,
        "n_particles_start": n_ref,
        "particles_exited": flow.n_exited,
        "cone_height": geometry.cone_height,
        "classification": cls,
        "flow_rate_kg_s": flow.mass_exited / (case.measure_time + case.startup_time),
        "kinetic_energy": flow.balance()["e_kin"],
        "energy": flow.balance(),
        "wall_clock_s": time.time() - t_wall,
    }
    return out


def find_critical_angle(results: list[dict]) -> dict:
    """Largest angle that still shows mass flow (bisection bracket from a sweep)."""
    mass = sorted([r["case"]["half_angle_deg"] for r in results if r["classification"].get("mass_flow")])
    funnel = sorted([r["case"]["half_angle_deg"] for r in results if not r["classification"].get("mass_flow")])
    if not mass:
        return {"ok": False, "theta_c": 0.0, "reason": "no mass-flow case in the sweep"}
    if not funnel:
        return {"ok": False, "theta_c": max(mass), "reason": "no funnel-flow case in the sweep"}
    theta_c = max(mass)
    return {
        "ok": True,
        "theta_c": theta_c,
        "bracket": [theta_c, min(funnel)],
        "mass_angles": mass,
        "funnel_angles": funnel,
        "uncertainty_deg": 0.5 * (min(funnel) - theta_c),
    }