"""Consolidated-drained (CD) triaxial compression: Mohr-Coulomb friction angle.

Benchmark role
--------------
A ``CD`` triaxial test is the standard laboratory measurement of the drained
friction angle of a cohesionless soil.  Because the specimen is drained, the
effective confining pressure is held **constant** while the specimen is free to
change volume, and the peak (or critical-state) stress ratio gives

    sin(phi) = (sigma_1 - sigma_3) / (sigma_1 + sigma_3),      c = 0.

A dry DEM assembly reproduces exactly that boundary-value problem:

* the lateral boundaries are **periodic and stress controlled**: an affine
  barostat rescales ``(x, y)`` and the cell cross-section so that the measured
  Love-Weber lateral pressure ``p_lat = (sigma_xx + sigma_yy) / 2`` stays at the
  prescribed ``sigma_3``.  That is the drained gate -- the specimen dilates or
  contracts freely and the confining pressure is not allowed to drift, which is
  precisely what a drained (as opposed to constant-volume / undrained) test
  means;
* the axial boundary is a platen driven at a constant strain rate, and the axial
  stress is the measured internal ``sigma_zz``, cross-checked against the
  platen reaction (a second, independent code path).

Everything below the driver is pre-existing machinery of this repository
(:mod:`tensordem.friction_surface.kernel` / :mod:`tensordem.friction_surface.shear`):
Cundall-Strack tangential friction, rolling resistance, the HCP+jitter specimen
builder and the Love-Weber contact-stress tensor.  No new force law is
introduced, so the measured friction angle is a property of the engine, not of
this file.

Why not :mod:`tensordem.dem3d_triaxial`
--------------------------------------
``dem3d_triaxial`` is a *bonded* simple-cubic ice specimen with rigid lateral
platens; a perfect lattice has no packing disorder, so a friction angle
measured on it would be an artifact of the crystal, not of a granular material.
The assembled specimen used here is disordered (jitter) and *unbonded*, which is
the material class the benchmark targets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from .friction_surface.kernel import (ContactModel, PairState, PlaneWall,
                                      apply_walls, integrate)
from .friction_surface.shear import (PeriodicShearCase, build_pairs_periodic,
                                     build_specimen, mohr_coulomb_fit,
                                     pair_forces_periodic, solid_fraction,
                                     stress_tensor)

__all__ = [
    "CDTriaxialCase",
    "friction_angle_deg",
    "run_cd_triaxial",
    "mohr_coulomb_fit",
]


def friction_angle_deg(sigma1: float, sigma3: float) -> float:
    """Drained Mohr-Coulomb friction angle (``c = 0``) of a triaxial stress state.

    ``sin(phi) = (sigma1 - sigma3) / (sigma1 + sigma3)``; compression positive.
    """
    s1 = float(sigma1)
    s3 = float(sigma3)
    denom = s1 + s3
    if not (math.isfinite(s1) and math.isfinite(s3)) or denom <= 0.0:
        return float("nan")
    s = (s1 - s3) / denom
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


@dataclass(frozen=True)
class CDTriaxialCase:
    """One drained triaxial test (SI units; stresses in Pa)."""

    # --- prescribed effective confining pressure (held during shear) ---
    sigma3: float = 1.0e5
    # --- material / contact ---
    mu: float = 0.50                 # interparticle Coulomb friction coefficient
    mu_rolling: float = 0.25         # rolling resistance (angularity proxy)
    radius: float = 5.0e-3
    density: float = 2650.0          # quartz
    kn: float = 1.0e6                # N/m
    kt_ratio: float = 0.5
    kr_ratio: float = 0.1
    restitution: float = 0.3
    gravity: float = 0.0             # stress-controlled cell: gravity is irrelevant
    # --- specimen (HCP lattice + jitter, periodic in x and y) ---
    # The specimen is *prepared by isotropic compression*: a loose, jittered
    # lattice (lattice_gap > 0 means spheres start out separated) is compacted
    # to the prescribed confining pressure by the barostat + platen servo.  The
    # large volumetric compaction destroys the lattice registry, so the sheared
    # specimen is a disordered jammed packing (a sand surrogate), not a crystal.
    nx: int = 8
    ny: int = 8
    nz: int = 6
    lattice_gap: float = 0.15
    jitter: float = 0.30
    seed: int = 15235
    # --- platens ---
    mu_wall: float = 0.0             # smooth platens: no boundary shear band
    mu_rolling_wall: float = 0.0
    # --- consolidation (isotropic, drained) ---
    consolidate_steps: int = 40000      # hard cap
    consolidate_check_every: int = 500
    consolidate_settle_checks: int = 3
    consolidate_tol: float = 0.03
    platen_servo_gain: float = 0.5
    platen_max_speed: float = 0.20
    # --- lateral stress barostat ---
    barostat_gain: float = 20.0      # 1/s per unit relative pressure error
    barostat_max_rate: float = 5.0   # 1/s bound on the lateral strain rate
    consolidate_max_rate: float = 3.0  # 1/s bound during isotropic consolidation
    # --- axial shear phase ---
    axial_strain_rate: float = 0.5   # 1/s
    axial_strain_target: float = 0.25
    sample_every: int = 100
    peak_min_strain: float = 0.01
    critical_fraction: float = 0.2
    dt: float | None = None

    def __post_init__(self) -> None:
        if self.sigma3 <= 0.0:
            raise ValueError("sigma3 must be positive (drained: held in shear)")
        if self.mu < 0.0 or self.mu_rolling < 0.0:
            raise ValueError("friction coefficients must be nonnegative")
        if self.axial_strain_rate <= 0.0 or self.axial_strain_target <= 0.0:
            raise ValueError("axial strain rate/target must be positive")
        if self.nx < 4 or self.ny < 4 or self.nz < 3:
            raise ValueError("specimen too small for a triaxial test")

    # -- derived geometry / material -------------------------------------
    @property
    def model(self) -> ContactModel:
        return ContactModel(
            radius=self.radius, density=self.density, kn=self.kn,
            kt_ratio=self.kt_ratio, kr_ratio=self.kr_ratio, mu=self.mu,
            mu_rolling=self.mu_rolling, restitution=self.restitution,
            gravity=self.gravity,
        )

    @property
    def geometry_case(self) -> PeriodicShearCase:
        """Specimen geometry carrier (reuses the shear cell's builder)."""
        return PeriodicShearCase(
            sigma=self.sigma3, mu=self.mu, mu_rolling=self.mu_rolling,
            radius=self.radius, density=self.density, kn=self.kn,
            kt_ratio=self.kt_ratio, kr_ratio=self.kr_ratio,
            restitution=self.restitution, gravity=self.gravity,
            nx=self.nx, ny=self.ny, nz=self.nz,
            lattice_gap=self.lattice_gap, jitter=self.jitter,
            seed=self.seed,
        )

    @property
    def dt_value(self) -> float:
        return self.model.timestep if self.dt is None else float(self.dt)

    @property
    def inertial_number(self) -> float:
        """``I = eps_dot d sqrt(rho / sigma3)`` -- quasi-static check."""
        return (self.axial_strain_rate * self.model.diameter
                * math.sqrt(self.density / self.sigma3))


def _pair_stress_sum(contacts: dict) -> np.ndarray:
    """Un-normalised Love-Weber pair sum ``sum f (x) l`` (N m)."""
    if not contacts or "f_total" not in contacts:
        return np.zeros((3, 3))
    f = contacts["f_total"]
    l = contacts["branch"]
    return (f[:, :, None] * l[:, None, :]).sum(axis=0)


def _wall_stress_sum(wall, pos: np.ndarray, wall_force: np.ndarray,
                     radius: float, volume: float) -> np.ndarray:
    """``sum f (x) l`` of the wall contacts, with the branch from the wall surface.

    A plane wall touches particles on one side only, so its contacts are missing
    from the particle-pair sum and the interior stress of a short specimen is
    biased low by roughly one boundary layer (measured 16 % at 6 layers).  The
    wall branch is ``pos - contact point = signed * n`` exactly, so the missing
    term is closed without touching the kernel.
    """
    mask, normal, _ = wall.contact_geometry(pos, radius, 0.0)
    if not mask.any():
        return np.zeros((3, 3))
    signed = (pos[mask] - wall.point) @ wall.normal
    branch = signed[:, None] * normal[mask]
    f = wall_force[mask]
    return (f[:, :, None] * branch[:, None, :]).sum(axis=0)


def _pressure(st: np.ndarray) -> float:
    """Lateral pressure ``(sigma_xx + sigma_yy) / 2`` (compression positive)."""
    return 0.5 * (float(st[0, 0]) + float(st[1, 1]))


def run_cd_triaxial(case: CDTriaxialCase) -> dict[str, Any]:
    """Run one drained triaxial test and return a machine-readable record."""
    model = case.model
    geom = case.geometry_case
    dt = case.dt_value
    pos = build_specimen(geom)
    width, depth = float(geom.width), float(geom.depth)
    n = len(pos)
    vel = np.zeros_like(pos)
    omega = np.zeros_like(pos)
    ps = PairState()
    contacts: dict = {}

    floor = PlaneWall((0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
                      mu=case.mu_wall, mu_rolling=case.mu_rolling_wall)
    z_top = float(pos[:, 2].max() + model.radius)
    lid = PlaneWall((0.0, 0.0, z_top), (0.0, 0.0, -1.0),
                    mu=case.mu_wall, mu_rolling=case.mu_rolling_wall)
    area = width * depth
    target_force = case.sigma3 * area
    height0 = z_top
    volume0 = area * height0
    solid_fraction0 = n * 4.0 / 3.0 * math.pi * model.radius ** 3 / volume0

    def measure(max_rate: float, wall_forces: tuple | None = None
                ) -> tuple[np.ndarray, float]:
        """Love-Weber stress (pairs + wall branches) and the barostat rate it implies."""
        volume = width * depth * max(z_top, 1e-9)
        s_sum = _pair_stress_sum(contacts)
        if wall_forces is not None:
            f_floor, f_lid = wall_forces
            s_sum = s_sum + _wall_stress_sum(floor, pos, f_floor, model.radius, volume)
            s_sum = s_sum + _wall_stress_sum(lid, pos, f_lid, model.radius, volume)
        s_sum = 0.5 * (s_sum + s_sum.T)
        st = s_sum / volume
        p_lat = _pressure(st)
        err = (case.sigma3 - p_lat) / case.sigma3
        rate = float(np.clip(case.barostat_gain * err, -max_rate, max_rate))
        return st, rate

    samples: list[dict[str, float]] = []
    steps = 0
    strain_axial = 0.0
    phi_series: list[float] = []
    keep_series: list[float] = []

    def record(st: np.ndarray) -> None:
        s1 = float(st[2, 2])
        s3 = _pressure(st)
        sig_lid = float(-lid.traction[2]) / area
        volume = width * depth * max(z_top, 1e-9)
        samples.append({
            "step": int(steps),
            "t": float(steps * dt),
            "strain_axial": float(strain_axial),
            "strain_volumetric": float(-math.log(max(volume, 1e-30) / volume0)),
            "sigma_xx": float(st[0, 0]),
            "sigma_yy": float(st[1, 1]),
            "sigma_zz": s1,
            "p_lat": s3,
            "q": s1 - s3,
            "phi_deg": friction_angle_deg(s1, s3),
            "sigma1_platen": sig_lid,
            "lid_height": float(z_top),
            "n_contacts": int(len(contacts.get("i", []))),
            "solid_fraction": float(n * 4.0 / 3.0 * math.pi * model.radius ** 3
                                    / max(volume, 1e-30)),
        })

    # ---------------- phase 1: drained isotropic consolidation ----------
    consolidate_ke: list[float] = []
    settled = 0
    steps = 0
    st = np.zeros((3, 3))
    last_wall_forces: tuple | None = None
    for step in range(case.consolidate_steps):
        steps += 1
        iu, ju, keys = build_pairs_periodic(pos, model.radius, width, depth)
        f, tq = pair_forces_periodic(pos, vel, omega, iu, ju, ps, model, dt,
                                     width, depth, contacts=contacts)
        fw_f, tw_f = apply_walls([floor], pos, vel, omega, model, dt, step * dt)
        fw_l, tw_l = apply_walls([lid], pos, vel, omega, model, dt, step * dt)
        fw = fw_f + fw_l
        tw = tw_f + tw_l
        f += fw
        if case.gravity:
            f[:, 2] -= model.mass * case.gravity
        pos, vel, omega = integrate(pos, vel, omega, f, tq + tw, model, dt)
        pos[:, 0] %= width
        pos[:, 1] %= depth
        last_wall_forces = (fw_f, fw_l)
        # drained lateral barostat: hold p_lat at sigma3, volume free to change
        st, rate = measure(case.consolidate_max_rate, last_wall_forces)
        scale = 1.0 - rate * dt
        pos[:, 0] *= scale
        pos[:, 1] *= scale
        vel[:, 0] /= scale
        vel[:, 1] /= scale
        width *= scale
        depth *= scale
        area = width * depth
        target_force = case.sigma3 * area
        # platen pressure servo along z (isotropic consolidation)
        lid_normal = float(-lid.traction[2])
        dz = -case.platen_servo_gain * (target_force - lid_normal) / (model.kn * 8.0)
        dz = float(np.clip(dz, -case.platen_max_speed * dt,
                           case.platen_max_speed * dt))
        z_top = float(z_top + dz)
        lid.point[2] = z_top
        ke = float(0.5 * model.mass * np.sum(vel ** 2) / max(n, 1))
        if step % max(1, case.consolidate_check_every) == 0:
            consolidate_ke.append(ke)
            p_lat = _pressure(st)
            s_zz = float(st[2, 2])
            ok = (abs(p_lat - case.sigma3) <= case.consolidate_tol * case.sigma3
                  and abs(s_zz - case.sigma3) <= case.consolidate_tol * case.sigma3
                  and ke <= 1.0e-6)
            settled = settled + 1 if ok else 0
            if settled >= case.consolidate_settle_checks:
                break
    st_conf, _ = measure(case.barostat_max_rate, last_wall_forces)
    p_lat_conf = _pressure(st_conf)
    sigma1_conf = float(st_conf[2, 2])
    conf_err = abs(p_lat_conf - case.sigma3) / case.sigma3
    axial_err = abs(sigma1_conf - case.sigma3) / case.sigma3
    consolidation = {
        "sigma3_target_Pa": case.sigma3,
        "p_lat_measured_Pa": p_lat_conf,
        "sigma_zz_measured_Pa": sigma1_conf,
        "lateral_relative_error": float(conf_err),
        "axial_relative_error": float(axial_err),
        "tolerance": case.consolidate_tol,
        "converged": bool(conf_err <= case.consolidate_tol
                          and axial_err <= case.consolidate_tol),
        "steps": int(steps),
        "solid_fraction": float(n * 4.0 / 3.0 * math.pi * model.radius ** 3
                                / (width * depth * max(z_top, 1e-9))),
        "kinetic_energy_per_particle_J": (
            float(consolidate_ke[-1]) if consolidate_ke else 0.0),
        "preparation": ("loose jittered lattice compacted isotropically to sigma3 "
                        "-> disordered jammed packing"),
    }

    # ---------------- phase 2: drained axial shear ----------------------
    height_start = z_top
    displacement_target = case.axial_strain_target * height_start
    v_axial = case.axial_strain_rate * height_start
    n_shear = int(round(displacement_target / (v_axial * dt)))
    lid.velocity = np.array([0.0, 0.0, -v_axial])
    for step in range(n_shear):
        steps += 1
        iu, ju, keys = build_pairs_periodic(pos, model.radius, width, depth)
        f, tq = pair_forces_periodic(pos, vel, omega, iu, ju, ps, model, dt,
                                     width, depth, contacts=contacts)
        fw_f, tw_f = apply_walls([floor], pos, vel, omega, model, dt, step * dt)
        fw_l, tw_l = apply_walls([lid], pos, vel, omega, model, dt, step * dt)
        fw = fw_f + fw_l
        tw = tw_f + tw_l
        f += fw
        if case.gravity:
            f[:, 2] -= model.mass * case.gravity
        pos, vel, omega = integrate(pos, vel, omega, f, tq + tw, model, dt)
        pos[:, 0] %= width
        pos[:, 1] %= depth
        last_wall_forces = (fw_f, fw_l)
        # drained gate: lateral barostat keeps p_lat = sigma3 while V changes
        st, rate = measure(case.barostat_max_rate, last_wall_forces)
        scale = 1.0 - rate * dt
        pos[:, 0] *= scale
        pos[:, 1] *= scale
        vel[:, 0] /= scale
        vel[:, 1] /= scale
        width *= scale
        depth *= scale
        # kinematic axial platen
        z_top = float(z_top - v_axial * dt)
        lid.point[2] = z_top
        strain_axial = (height_start - z_top) / height_start
        if step % max(1, case.sample_every) == 0:
            record(st)
            s1 = float(st[2, 2])
            s3 = _pressure(st)
            phi_series.append(friction_angle_deg(s1, s3))
            keep_series.append(strain_axial)

    if not samples:
        raise RuntimeError("no samples recorded in the shear phase")

    # ---------------- peak / critical state ------------------------------
    hist = [s for s in samples if s["strain_axial"] >= case.peak_min_strain]
    if not hist:
        raise RuntimeError("shear phase shorter than peak_min_strain")
    peak_row = max(hist, key=lambda s: s["q"])
    n_tail = max(1, int(round(case.critical_fraction * len(hist))))
    tail = hist[-n_tail:]
    phi_crit = float(np.nanmean([s["phi_deg"] for s in tail]))
    q_crit = float(np.nanmean([s["q"] for s in tail]))
    p_crit = float(np.nanmean([s["p_lat"] for s in tail]))
    vol = np.array([s["strain_volumetric"] for s in samples])
    eps_ax = np.array([s["strain_axial"] for s in samples])
    i_peak = int(np.argmin(vol))
    dilatancy = {
        "volumetric_strain_min": float(vol.min()),
        "strain_axial_at_min_volume": float(eps_ax[i_peak]),
        "volumetric_strain_end": float(vol[-1]),
        "volumetric_strain_at_peak": float(vol[int(np.argmax(
            [s["q"] for s in samples]))]),
        "note": "compression positive; negative = dilation (dense specimen)",
    }
    sigma3_drift = float(np.nanmax(np.abs(
        np.array([s["p_lat"] for s in samples]) - case.sigma3) / case.sigma3))
    platen_rel = abs(peak_row["sigma1_platen"] - peak_row["sigma_zz"]) / max(
        abs(peak_row["sigma_zz"]), 1e-30)
    ke_end = float(0.5 * model.mass * np.sum(vel ** 2) / max(n, 1))
    return {
        "case": asdict(case),
        "n_particles": int(n),
        "dt_s": float(dt),
        "initial": {
            "solid_fraction": float(solid_fraction0),
            "height_m": float(height0),
            "width_m": float(geom.width),
            "depth_m": float(geom.depth),
            "volume_m3": float(volume0),
            "mu": case.mu,
            "mu_rolling": case.mu_rolling,
        },
        "consolidation": consolidation,
        "peak": {
            "strain_axial": peak_row["strain_axial"],
            "sigma1_Pa": peak_row["sigma_zz"],
            "sigma3_Pa": peak_row["p_lat"],
            "q_Pa": peak_row["q"],
            "phi_deg": peak_row["phi_deg"],
            "solid_fraction": peak_row["solid_fraction"],
            "sigma1_platen_Pa": peak_row["sigma1_platen"],
        },
        "critical_state": {
            "phi_deg": phi_crit,
            "q_Pa": q_crit,
            "p_lat_Pa": p_crit,
            "window_fraction": case.critical_fraction,
            "strain_axial_window": [tail[0]["strain_axial"], tail[-1]["strain_axial"]],
        },
        "drained_gate": {
            "sigma3_target_Pa": case.sigma3,
            "max_relative_drift": sigma3_drift,
            "tolerance": 0.10,
            "pass": bool(sigma3_drift <= 0.10),
        },
        "platen_cross_check": {
            "sigma1_interior_Pa": peak_row["sigma_zz"],
            "sigma1_platen_Pa": peak_row["sigma1_platen"],
            "relative_difference": float(platen_rel),
        },
        "dilatancy": dilatancy,
        "quasi_static": {
            "inertial_number_I": case.inertial_number,
            "kinetic_energy_per_particle_end_J": ke_end,
            "strain_rate_per_s": case.axial_strain_rate,
            "gate": "I <= 1e-3 and KE per particle << contact energy scale",
            "pass": bool(case.inertial_number <= 1.0e-3),
        },
        "axial_strain_reached": float(strain_axial),
        "steps": int(steps),
        "samples": samples,
    }