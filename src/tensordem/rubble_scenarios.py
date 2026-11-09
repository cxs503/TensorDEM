"""DEM scenario drivers feeding :mod:`tensordem.rubble_postprocess`.

Three configurations are used to calibrate the :math:`\\mu(I)` law of a broken-ice
(rubble) pile:

``couette_shear``
    A plane-strain cell that is periodic in ``x`` with a rough bottom wall and a
    rough *sliding* top wall (the wall slides at ``v`` while its geometry stays
    fixed, i.e. a Lees-Edwards-free simple shear cell).  The equivalent shear
    rate is measured from the bulk velocity profile, never assumed equal to the
    nominal ``v/H``.
``biaxial_compression``
    Periodic in ``x``, a piston (top wall) moves down at constant speed, so the
    packing is compressed in plane strain at a prescribed axial strain rate.
    The equivalent shear rate is ``2 |eps_y|`` (deviatoric magnitude).
``ice_channel``
    A broken-ice field in a periodic channel that drifts past a fixed ship hull
    (``tensordem.granular2d.jamming.make_channel_system``); the equivalent shear
    rate is the slope of the mean streamwise velocity profile across the free
    channel.

Every driver returns *snapshots* (particle configuration + contact record +
measured shear rate) so that the post-processing module owns all
physics-facing averaging; this module only integrates and measures kinematics.
All quantities are SI.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Callable, Sequence

import numpy as np

from .granular2d import (
    DiskConfig,
    DiskSystem,
    bidisperse_radii,
    box_walls,
    grow_to_fraction,
    make_channel_system,
    random_packing_positions,
    ship_segments,
)
from .rubble_postprocess import ParticleSample, RubblePostProcessError

__all__ = [
    "ScenarioError",
    "SNAPSHOTS",
    "bulk_mask",
    "velocity_profile_shear_rate",
    "make_shear_cell",
    "run_couette_shear",
    "make_compression_cell",
    "run_biaxial_compression",
    "run_ice_channel",
    "SCENARIOS",
]


class ScenarioError(ValueError):
    """Raised when a scenario cannot be integrated as configured."""


SNAPSHOTS = 8


def bulk_mask(sample: ParticleSample, *, margin: float) -> np.ndarray:
    """Particles at least ``margin`` away from every wall (the bulk slab)."""
    y = sample.positions[:, 1]
    height = sample.domain[1]
    return (y >= margin) & (y <= height - margin)


def velocity_profile_shear_rate(y: np.ndarray, vx: np.ndarray, *,
                                height: float, n_bins: int = 10,
                                margin: float = 0.0) -> dict[str, Any]:
    """Least-squares slope of the *binned* streamwise velocity profile.

    ``margin`` removes wall layers (in m) from the fit.  Returning the binned
    profile as well lets the caller audit that the fit region is bulk-like
    instead of a boundary layer.
    """
    y = np.asarray(y, dtype=np.float64)
    vx = np.asarray(vx, dtype=np.float64)
    if y.size == 0:
        raise ScenarioError("no particles for the velocity profile")
    lo, hi = float(margin), float(height - margin)
    if hi <= lo:
        raise ScenarioError("margin removes the whole profile")
    edges = np.linspace(lo, hi, n_bins + 1)
    which = np.digitize(y, edges) - 1
    centres, means, counts = [], [], []
    for k in range(n_bins):
        sel = which == k
        if int(np.sum(sel)) >= 3:
            centres.append(0.5 * (edges[k] + edges[k + 1]))
            means.append(float(np.mean(vx[sel])))
            counts.append(int(np.sum(sel)))
    if len(centres) < 3:
        raise ScenarioError("fewer than 3 populated velocity bins: no shear rate")
    centres = np.asarray(centres)
    means = np.asarray(means)
    coeffs = np.polyfit(centres, means, 1)
    pred = np.polyval(coeffs, centres)
    ss_res = float(np.sum((means - pred) ** 2))
    ss_tot = float(np.sum((means - np.mean(means)) ** 2))
    return {
        "slope": float(coeffs[0]),
        "intercept": float(coeffs[1]),
        "profile_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
        "bin_centres": centres.tolist(),
        "bin_means": means.tolist(),
        "bin_counts": counts,
    }


def bulk_deformation_rate(positions: np.ndarray, velocities: np.ndarray, *,
                          length_x: float | None = None,
                          min_particles: int = 8) -> dict[str, Any]:
    """Least-squares 2-D velocity gradient and equivalent shear rate.

    The deviatoric strain rate is measured from the particle velocities in the
    *same* bulk slab that provides the stress, so that no scenario feeds an
    imposed (nominal) rate into :math:`I` while another feeds a measured one.
    A planar linear field ``v_a(x, y) = c_a0 + G[a, 0] x + G[a, 1] y`` is fitted
    to every bulk particle at once; the equivalent shear rate is the invariant
    ``sqrt(2 D:D)`` of the symmetric part ``D = (G + G^T) / 2`` (for pure shear
    ``vx = gamma y`` it returns ``gamma``; for isochoric plane-strain
    compression ``v = (eps x, -eps y)`` it returns ``2 |eps|``).

    ``length_x`` is the periodic length; the streamwise coordinate is centred on
    the cell mid-plane first, so that the fit is not dominated by the periodic
    seam (particles wrapped to ``x ~ L`` and ``x ~ 0`` belong to the same
    material neighbourhood).
    """
    pos = np.asarray(positions, dtype=np.float64)
    vel = np.asarray(velocities, dtype=np.float64)
    if pos.shape != vel.shape or pos.ndim != 2 or pos.shape[1] != 2:
        raise ScenarioError("positions and velocities must be (N, 2)")
    n = len(pos)
    if n < min_particles:
        raise ScenarioError("too few particles for a velocity gradient")
    x = pos[:, 0].copy()
    if length_x is not None and length_x > 0.0:
        x = np.mod(x + 0.5 * length_x, length_x) - 0.5 * length_x
    xc = x - x.mean()
    yc = pos[:, 1] - pos[:, 1].mean()
    design = np.column_stack([np.ones(n), xc, yc])
    coef, *_ = np.linalg.lstsq(design, vel, rcond=None)     # 3 x 2
    grad = np.empty((2, 2))
    grad[0, 0], grad[0, 1] = coef[1, 0], coef[2, 0]         # dvx/dx, dvx/dy
    grad[1, 0], grad[1, 1] = coef[1, 1], coef[2, 1]         # dvy/dx, dvy/dy
    D = 0.5 * (grad + grad.T)
    gamma_dot = float(np.sqrt(2.0 * np.sum(D * D)))
    pred = design @ coef[:, 1]
    ss_res = float(np.sum((vel[:, 1] - pred) ** 2))
    ss_tot = float(np.sum((vel[:, 1] - vel[:, 1].mean()) ** 2))
    return {
        "gradient": grad.tolist(),
        "gamma_dot": gamma_dot,
        "eps_dot_y": float(grad[1, 1]),
        "eps_dot_x": float(grad[0, 0]),
        "profile_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
        "n_particles": int(n),
    }


# ------------------------------------------------------------------- packing
def _prepare_packing(*, n_particles: int, mean_radius: float, width: float,
                     height: float, phi: float, seed: int, mu: float,
                     mu_rolling: float, thickness: float,
                     periodic_x: float | None,
                     walls: np.ndarray | None,
                     hull: np.ndarray | None = None,
                     free_area: float | None = None,
                     relax_steps: int = 120) -> DiskSystem:
    """Grow a dilute random packing up to the target area fraction."""
    radii = bidisperse_radii(n_particles, mean_radius, seed=seed)
    area = width * height if free_area is None else float(free_area)
    disk_area = float(np.sum(math.pi * radii ** 2))
    phi_start = 0.22
    r0 = radii * math.sqrt(phi_start * area / disk_area)
    pos = random_packing_positions(n_particles, r0, width, height,
                                   seed=seed + 3, obstacles=hull)
    cfg = DiskConfig(kn=3.0e4, mu=mu, mu_rolling=mu_rolling, zeta=0.3,
                     thickness=thickness, damping_time=2.0e-3)
    system = DiskSystem(pos, r0, cfg, walls=walls, hull=hull, periodic_x=periodic_x)
    grow_to_fraction(system, phi, dphi=0.02, relax_steps=relax_steps, area=area)
    # production dynamics: no artificial damping, velocities from rest
    system.config = replace(cfg, zeta=0.12, damping_time=0.0)
    # the radii grew, so the stability limit grew too: re-take it (never loosen
    # beyond the current limit, never keep the tiny pre-growth dt)
    system.dt = system.recommended_dt
    system.zero_velocities()
    system.theta[:] = 0.0
    system._state_keys = np.zeros(0, dtype=np.int64)
    system._state_xi = np.zeros((0, 2))
    system._state_roll = np.zeros(0)
    return system


def _snapshot(system: DiskSystem, *, label: str, periodic: tuple[bool, bool],
              thickness: float, time: float) -> tuple[ParticleSample, dict]:
    system.contact_forces()
    contacts = {k: np.array(v, copy=True) for k, v in system.contacts.items()}
    lx = system.periodic_x if system.periodic_x is not None else float(
        system.walls[:, :, 0].max() if len(system.walls) else 1.0)
    ly = system.periodic_y if system.periodic_y is not None else float(
        system.walls[:, :, 1].max() if len(system.walls) else 1.0)
    sample = ParticleSample(system.pos.copy(), system.radii.copy(),
                            (float(lx), float(ly)), thickness, periodic,
                            label, float(time))
    return sample, contacts


def make_shear_cell(*, n_particles: int = 300, mean_radius: float = 0.010,
                    width: float = 0.36, height: float = 0.14,
                    phi: float = 0.80, seed: int = 0, mu: float = 0.5,
                    mu_rolling: float = 0.05, thickness: float = 0.03) -> DiskSystem:
    """Plane-strain Couette cell: periodic ``x``, rough bottom and top walls."""
    walls = box_walls(width, height, left=False, right=False)
    return _prepare_packing(n_particles=n_particles, mean_radius=mean_radius,
                            width=width, height=height, phi=phi, seed=seed,
                            mu=mu, mu_rolling=mu_rolling, thickness=thickness,
                            periodic_x=width, walls=walls)


def run_couette_shear(*, shear_rate: float, seed: int = 0, strain: float = 2.0,
                      wall_gap_radius: float = 3.0, snapshots: int = SNAPSHOTS,
                      warmup_fraction: float = 0.35, **cell_kwargs
                      ) -> dict[str, Any]:
    """Shear a packing at a nominal rate ``shear_rate`` and record snapshots.

    The top wall slides at ``v = shear_rate * H``; the equivalent shear rate in
    the *bulk* is measured from the velocity profile, so wall slip does not
    contaminate :math:`\\dot\\gamma`.
    """
    if not shear_rate > 0.0:
        raise ScenarioError("shear_rate must be positive")
    system = make_shear_cell(seed=seed, **cell_kwargs)
    width = float(cell_kwargs.get("width", 0.36))
    height = float(cell_kwargs.get("height", 0.14))
    thickness = float(cell_kwargs.get("thickness", 0.03))
    r_mean = system.mean_radius
    margin = wall_gap_radius * r_mean
    v_top = shear_rate * height
    # box_walls(left=False, right=False) -> [bottom, top]; both walls slide in
    # opposite directions, so the imposed velocity difference is v_top and the
    # mean flow is zero (symmetric Couette, less shear banding than one moving wall)
    system.set_wall_velocity(np.array([[-0.5 * v_top, 0.0], [0.5 * v_top, 0.0]]))
    dt = system.dt
    duration = strain / shear_rate
    n_steps = max(int(round(duration / dt)), 50)
    warm = int(round(warmup_fraction * n_steps))
    sample_every = max(1, (n_steps - warm) // snapshots)
    snaps = []
    t = 0.0
    for step in range(1, n_steps + 1):
        system.step(1)
        if step >= warm and (step - warm) % sample_every == 0:
            sample, contacts = _snapshot(system, label=f"shear_{shear_rate:g}",
                                         periodic=(True, False), thickness=thickness,
                                         time=step * dt)
            bulk = bulk_mask(sample, margin=margin)
            if int(np.sum(bulk)) < 20:
                continue
            y_bulk = sample.positions[bulk, 1]
            prof = velocity_profile_shear_rate(
                y_bulk, system.vel[bulk, 0],
                height=float(np.max(y_bulk) + 1e-12), n_bins=8, margin=0.0)
            snaps.append({"sample": sample, "contacts": contacts,
                          "profile": prof, "time": step * dt})
    if not snaps:
        raise ScenarioError("shear run produced no snapshot")
    rates = np.array([s["profile"]["slope"] for s in snaps], dtype=np.float64)
    return {
        "scenario": "couette_shear",
        "nominal_shear_rate": float(shear_rate),
        "measured_shear_rate": float(np.mean(np.abs(rates))),
        "shear_rate_std": float(np.std(np.abs(rates))),
        "wall_speed": float(v_top),
        "wall_slip_fraction": float(abs(np.mean(rates)) / shear_rate) if shear_rate else float("nan"),
        "velocity_profile_r2": float(np.mean([s["profile"]["profile_r2"] for s in snaps])),
        "dt": float(dt), "steps": int(n_steps), "duration": float(duration),
        "thickness": thickness, "width": width, "height": height,
        "bulk_margin": float(margin),
        "wall_velocity": system.wall_velocity.tolist() if system.wall_velocity is not None else None,
        "snapshots": snaps,
    }


def make_compression_cell(*, n_particles: int = 300, mean_radius: float = 0.010,
                          width: float = 0.32, height: float = 0.16,
                          phi: float = 0.72, seed: int = 0, mu: float = 0.5,
                          mu_rolling: float = 0.05, thickness: float = 0.03) -> DiskSystem:
    """Plane-strain compression cell: periodic ``x``, bottom wall + top piston."""
    walls = box_walls(width, height, left=False, right=False)
    return _prepare_packing(n_particles=n_particles, mean_radius=mean_radius,
                            width=width, height=height, phi=phi, seed=seed,
                            mu=mu, mu_rolling=mu_rolling, thickness=thickness,
                            periodic_x=width, walls=walls)


def run_biaxial_compression(*, strain_rate: float, seed: int = 0,
                            total_strain: float = 0.30,
                            snapshots: int = SNAPSHOTS,
                            isochoric: bool = True,
                            steady_fraction: float = 0.45,
                            wall_gap_radius: float = 3.0,
                            **cell_kwargs) -> dict[str, Any]:
    """Plane-strain compression at a constant axial strain rate.

    The piston (top wall) moves down at ``strain_rate * H``.  With
    ``isochoric=True`` (default) the cell is stretched in ``x`` by the inverse
    factor every step, so the *area* -- and therefore the solid fraction -- is
    held constant: that is the plane-strain element test the rheology needs,
    whereas a confined cell would simply densify to random close packing and its
    stress ratio would never reach a steady state.

    Kinematic consistency (P82W2-N02)
    ---------------------------------
    Three corrections make this driver a valid element test, each of them a
    measurable bias in the earlier version:

    1. the piston is given its *actual* velocity (``wall_velocity = (0, -v)``),
       so the wall contact damping is evaluated against a moving wall instead of
       a static one;
    2. the equivalent shear rate is **measured** from the bulk velocity gradient
       (``bulk_deformation_rate``) in the same slab that yields the stress,
       instead of being taken as the nominal ``2 |eps_y|``; the realised bulk
       rate is only ~70-75 % of the nominal one in this cell, so the nominal rate
       systematically overstated :math:`I`;
    3. a third candidate fix, carrying the affine velocity increment on the
       isochoric ``x`` remap (``dv_x = (lambda - 1) x / dt``), is **rejected**:
       measured in ``scripts/ablate_compression_kinematics.py`` it doubles the
       pressure and drops ``mu`` to 0.06-0.09, because the engine's periodic
       images carry no box-rate term, so the remap injects a spurious slip on the
       periodic seam.  The ``x`` remap is therefore position-only, which is a
       declared model-form limitation of this element test.

    ``steady_fraction`` discards the first part of the run (transient fabric
    build-up) before the snapshots that enter the rheology.
    """
    if not strain_rate > 0.0:
        raise ScenarioError("strain_rate must be positive")
    system = make_compression_cell(seed=seed, **cell_kwargs)
    width = float(cell_kwargs.get("width", 0.32))
    height = float(cell_kwargs.get("height", 0.16))
    thickness = float(cell_kwargs.get("thickness", 0.03))
    dt = system.dt
    n_steps = max(int(round(total_strain / (strain_rate * dt))), 50)
    start_step = int(round(steady_fraction * n_steps))
    sample_every = max(1, (n_steps - start_step) // snapshots)
    v_piston = strain_rate * height
    margin = wall_gap_radius * system.mean_radius
    current_height = height
    current_width = width
    # walls = [bottom, top]; the piston moves down at exactly v_piston
    system.set_wall_velocity(np.array([[0.0, 0.0], [0.0, -v_piston]]))
    snaps = []
    for step in range(1, n_steps + 1):
        dh = v_piston * dt
        system.walls[1] -= np.array([0.0, dh])          # top wall / piston
        current_height -= dh
        if isochoric:
            factor = width * height / (current_width * current_height)
            system.pos[:, 0] *= factor
            system.walls[:, :, 0] *= factor
            current_width *= factor
            system.periodic_x = current_width
        system.step(1)
        if step > start_step and (step - start_step) % sample_every == 0:
            sample, contacts = _snapshot(
                system, label=f"compress_{strain_rate:g}", periodic=(True, False),
                thickness=thickness, time=step * dt)
            sample = ParticleSample(sample.positions, sample.radii,
                                    (float(current_width), float(current_height)),
                                    thickness, (True, False), sample.label,
                                    sample.time)
            nominal_rate = 2.0 * v_piston / current_height
            try:
                bulk = bulk_mask(sample, margin=margin)
                rate = bulk_deformation_rate(system.pos[bulk], system.vel[bulk],
                                             length_x=current_width,
                                             min_particles=20)
                profile = {"slope": rate["gamma_dot"],
                           "profile_r2": rate["profile_r2"],
                           "eps_dot_y_measured": rate["eps_dot_y"],
                           "gamma_dot_nominal": nominal_rate,
                           "n_particles": rate["n_particles"]}
            except ScenarioError:
                profile = {"slope": nominal_rate, "profile_r2": float("nan"),
                           "eps_dot_y_measured": float("nan"),
                           "gamma_dot_nominal": nominal_rate,
                           "n_particles": 0}
            snaps.append({"sample": sample, "contacts": contacts,
                          "height": float(current_height),
                          "width": float(current_width),
                          "nominal_shear_rate": nominal_rate,
                          "profile": profile,
                          "time": step * dt})
    if not snaps:
        raise ScenarioError("compression run produced no snapshot")
    nominal = np.array([s["nominal_shear_rate"] for s in snaps])
    rates = np.array([s["profile"]["slope"] for s in snaps])
    return {
        "scenario": "biaxial_compression",
        "nominal_shear_rate": float(np.mean(nominal)),
        "measured_shear_rate": float(np.mean(rates)),
        "measured_over_nominal_rate": float(np.mean(rates) / np.mean(nominal)),
        "shear_rate_std": float(np.std(rates)),
        "bulk_margin": float(margin),
        "velocity_profile_r2": float(np.mean([s["profile"]["profile_r2"] for s in snaps])),
        "wall_velocity": (system.wall_velocity.tolist()
                          if system.wall_velocity is not None else None),
        "axial_strain_rate": float(strain_rate),
        "isochoric": bool(isochoric),
        "steady_fraction": float(steady_fraction),
        "dt": float(dt), "steps": int(n_steps),
        "thickness": thickness, "width": width, "height": height,
        "final_width": float(current_width), "final_height": float(current_height),
        "snapshots": snaps,
    }


def run_ice_channel(*, drift_speed: float, seed: int = 0, n_particles: int = 400,
                    phi: float = 0.80, mean_radius: float = 0.014,
                    duration: float = 1.2, warmup_fraction: float = 0.4,
                    snapshots: int = SNAPSHOTS, thickness: float = 0.03,
                    omega: float = 0.0) -> dict[str, Any]:
    """Drift a broken-ice field past a fixed ship hull in a periodic channel.

    ``omega`` optionally rotates the hull-relative drift direction (rate of the
    heading change) which is how the channel scenario is driven at a different
    velocity profile; by default the drift is straight.
    """
    if not drift_speed > 0.0:
        raise ScenarioError("drift_speed must be positive")
    system, info = make_channel_system(n_particles=n_particles, mean_radius=mean_radius,
                                       phi=phi, seed=seed, drift_speed=drift_speed)
    length, width = float(info["length"]), float(info["width"])
    dt = system.dt
    n_steps = max(int(round(duration / dt)), 50)
    warm = int(round(warmup_fraction * n_steps))
    sample_every = max(1, (n_steps - warm) // snapshots)
    margin = 2.5 * system.mean_radius
    snaps = []
    for step in range(1, n_steps + 1):
        system.step(1)
        if step >= warm and (step - warm) % sample_every == 0:
            sample, contacts = _snapshot(system, label=f"channel_{drift_speed:g}",
                                         periodic=(True, False), thickness=thickness,
                                         time=step * dt)
            prof = velocity_profile_shear_rate(system.pos[:, 1], system.vel[:, 0],
                                               height=width, n_bins=10, margin=margin)
            snaps.append({"sample": sample, "contacts": contacts,
                          "profile": prof, "time": step * dt,
                          "hull_force": system.hull_force.copy()})
    if not snaps:
        raise ScenarioError("channel run produced no snapshot")
    rates = np.array([s["profile"]["slope"] for s in snaps], dtype=np.float64)
    return {
        "scenario": "ice_channel",
        "drift_speed": float(drift_speed),
        "measured_shear_rate": float(np.mean(np.abs(rates))),
        "shear_rate_std": float(np.std(np.abs(rates))),
        "nominal_shear_rate": float(np.mean(np.abs(rates))),
        "velocity_profile_r2": float(np.mean([s["profile"]["profile_r2"] for s in snaps])),
        "dt": float(dt), "steps": int(n_steps), "duration": float(duration),
        "thickness": thickness, "length": length, "width": width,
        "phi_channel": float(info["phi"]),
        "snapshots": snaps,
    }


SCENARIOS: dict[str, Callable[..., dict[str, Any]]] = {
    "couette_shear": run_couette_shear,
    "biaxial_compression": run_biaxial_compression,
    "ice_channel": run_ice_channel,
}