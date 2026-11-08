"""Three parallel repose channels (sphere / clump / true superquadric) on one integrator.

Campaign node P27W2-N07, criteria C2, C3, C4.

All three channels use the *same* rigid-body DEM integrator
(:mod:`tensordem.superquadric_dem`) and the same contact law
(:mod:`tensordem.superquadric_kernel`), so any difference in the measured angle
of repose is attributable to the shape representation alone -- not to the
solver.  The clump channel is the volume-matched multi-sphere fit
(:func:`tensordem.clump_geometry.fit_clump`) used elsewhere in the repository.

Masses are matched across channels (the clump is density-rescaled), because a
repose experiment is gravity driven and an unmatched mass would confound the
comparison.

C3 (static-pile spurious dissipation) uses :func:`static_pile_dissipation`: a
settled pile is run on with the gates inactive and the mechanical energy is
logged; the criterion is ``|dE/E| / step <= 1e-6``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..clump_geometry import fit_clump, geometric_error
from ..flat_punch import reduced_modulus
from ..superquadric_dem import (
    SQBody,
    SQSystem,
    SQSystemConfig,
    Wall,
    make_clump_body,
    make_superquadric_body,
    repose_angle_from_cloud,
    run_column_collapse,
)
from ..superquadric_kernel import ContactMaterial


# ---------------------------------------------------------------------------
# Geometry / material setup
# ---------------------------------------------------------------------------

def box_walls(box_half: float, gate_half: float, mu: float,
              mu_rolling: Optional[float] = None) -> List[Wall]:
    """Floor + four side walls + four gate walls (indices 5..8 are the gates)."""
    kw = dict(mu=mu, mu_rolling=mu_rolling)
    return [
        Wall(np.array([0.0, 0.0, 1.0]), 0.0, label="floor", **kw),
        Wall(np.array([1.0, 0.0, 0.0]), -box_half, **kw),
        Wall(np.array([-1.0, 0.0, 0.0]), -box_half, **kw),
        Wall(np.array([0.0, 1.0, 0.0]), -box_half, **kw),
        Wall(np.array([0.0, -1.0, 0.0]), -box_half, **kw),
        Wall(np.array([1.0, 0.0, 0.0]), -gate_half, label="gate_xp", **kw),
        Wall(np.array([-1.0, 0.0, 0.0]), -gate_half, label="gate_xm", **kw),
        Wall(np.array([0.0, 1.0, 0.0]), -gate_half, label="gate_yp", **kw),
        Wall(np.array([0.0, -1.0, 0.0]), -gate_half, label="gate_ym", **kw),
    ]


GATE_WALLS = (5, 6, 7, 8)


def column_positions(n: int, gate_half: float, radius: float,
                     spacing_factor: float = 2.6, seed: int = 7) -> np.ndarray:
    """Deterministic loose column packing inside the gate (centred, no overlap)."""
    rng = np.random.default_rng(seed)
    spacing = spacing_factor * radius
    per = max(int(math.floor(2.0 * gate_half / spacing)), 1)
    layers = int(math.ceil(n / (per * per)))
    off = -0.5 * (per - 1) * spacing
    pts = []
    for kz in range(layers):
        for ky in range(per):
            for kx in range(per):
                pts.append([off + kx * spacing, off + ky * spacing, spacing + kz * spacing])
    return np.array(pts[:n], dtype=float) + 0.25 * radius * rng.standard_normal((n, 3))


def material(mu: float, mu_rolling: float, e_star: float = 1.0e9,
             restitution: float = 0.4) -> ContactMaterial:
    """Contact material; ``e_star`` is fixed so stiffness does not confound the study."""
    return ContactMaterial(e_star=e_star, g_star=e_star / (2.0 * 2.6), mu=mu,
                           mu_rolling=mu_rolling, restitution=restitution)


@dataclass
class ChannelSpec:
    """Particle representation for one channel."""
    kind: str                      # "sphere" | "clump" | "true"
    shape_exponent: float = 2.0    # superellipsoid exponent (2 = ellipsoid)
    aspect: float = 1.0            # b/a
    n_spheres: int = 8
    radius: float = 4.0e-3
    seed: int = 0


def build_bodies(spec: ChannelSpec, positions: np.ndarray, density: float
                 ) -> Tuple[List[SQBody], Dict[str, object]]:
    """Instantiate the particle bodies for a channel; returns bodies + geometry info."""
    a = spec.radius
    b = spec.radius * spec.aspect
    c = spec.radius
    info: Dict[str, object] = {"kind": spec.kind, "shape_exponent": spec.shape_exponent,
                               "aspect": spec.aspect, "n_spheres": 0}
    if spec.kind == "sphere":
        bodies = [make_clump_body(np.zeros((1, 3)), np.array([spec.radius]), p, density)
                  for p in positions]
        info["n_spheres"] = 1
        return bodies, info
    if spec.kind == "clump":
        fit = fit_clump(a, b, c, spec.shape_exponent, spec.shape_exponent,
                        n_spheres=spec.n_spheres, seed=spec.seed)
        err = geometric_error(a, b, c, spec.shape_exponent, spec.shape_exponent,
                              fit, n_samples=300)
        # density rescaling: the inscribed clump under-fills the body, so the
        # sphere density is raised to match the true particle mass
        true_volume = _superquadric_volume(a, b, c, spec.shape_exponent)
        clump_volume = float(np.sum(4.0 / 3.0 * math.pi * fit.radii ** 3))
        rho_clump = density * true_volume / max(clump_volume, 1e-30)
        bodies = [make_clump_body(fit.offsets, fit.radii, p, rho_clump)
                  for p in positions]
        info.update({
            "n_spheres": int(fit.n_spheres),
            "two_sided_max_m": float(fit.two_sided_max),
            "max_distance_ratio": float(err.max_distance_ratio),
            "mean_normal_deviation_deg": float(err.mean_normal_deviation_deg),
            "volume_ratio": float(fit.volume_ratio),
            "density_scale": float(rho_clump / density),
            "sphere_radius_mean_m": float(np.mean(fit.radii)),
        })
        return bodies, info
    if spec.kind == "true":
        bodies = [make_superquadric_body(a, b, c, spec.shape_exponent,
                                         spec.shape_exponent, p, density)
                  for p in positions]
        return bodies, info
    raise ValueError(f"unknown channel kind {spec.kind!r}")


def _superquadric_volume(a: float, b: float, c: float, m: float, n: int = 60) -> float:
    """Volume of ``|x/a|^m+|y/b|^m+|z/c|^m <= 1`` by midpoint quadrature."""
    x = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    X, Y, Z = np.meshgrid(x * a, x * b, x * c, indexing="ij")
    inside = (np.abs(X / a) ** m + np.abs(Y / b) ** m + np.abs(Z / c) ** m) <= 1.0
    cell = (2 * a / n) * (2 * b / n) * (2 * c / n)
    return float(np.sum(inside)) * cell


# ---------------------------------------------------------------------------
# Repose experiment
# ---------------------------------------------------------------------------

def run_repose(spec: ChannelSpec, *, mu: float = 0.5, mu_rolling: float = 0.5,
               n_particles: int = 110, box_half: float = 0.09,
               gate_half: float = 0.016, hold_steps: int = 300,
               settle_steps: int = 4000, dt: Optional[float] = 2.5e-4,
               density: float = 2500.0, e_star: float = 1.0e9,
               restitution: float = 0.4, seed: int = 7,
               settle_probe_steps: int = 0, damping: float = 0.0) -> Dict[str, object]:
    """Column-collapse repose experiment for one channel.

    The column is held by four gate walls for ``hold_steps`` and then released;
    the repose angle is read from the free surface of the final heap
    (:func:`tensordem.superquadric_dem.repose_angle_from_cloud`).  Stiffness,
    friction, restitution, particle count and geometry are identical across
    channels -- only the particle representation changes.
    """
    t0 = time.perf_counter()
    positions = column_positions(n_particles, gate_half, spec.radius, seed=seed)
    bodies, info = build_bodies(spec, positions, density)
    rng = np.random.default_rng(seed + 1)
    for body in bodies:
        body.velocity = np.array([0.06 * rng.standard_normal(),
                                  0.06 * rng.standard_normal(), 0.0])
    mat = material(mu, mu_rolling, e_star=e_star, restitution=restitution)
    cfg = SQSystemConfig(material=mat, dt=dt, damping_global=damping)
    walls = box_walls(box_half, gate_half, mu, mu_rolling)
    res = run_column_collapse(bodies, walls, cfg, hold_steps=hold_steps,
                              settle_steps=settle_steps, gate_walls=GATE_WALLS)
    out: Dict[str, object] = {
        "channel": spec.kind,
        "shape_exponent": float(spec.shape_exponent),
        "aspect": float(spec.aspect),
        "n_particles": int(n_particles),
        "mu": float(mu),
        "mu_rolling": float(mu_rolling),
        "dt_s": float(res.dt),
        "steps": int(res.steps),
        "repose_angle_deg": float(res.repose_angle_deg),
        "steady_penetration_ratio": float(res.steady_penetration_ratio),
        "max_penetration_ratio": float(res.max_penetration_ratio),
        "energy_drift_per_1000": float(res.energy_drift_per_1000),
        "n_contacts": int(res.n_contacts),
        "force_chain_angle_deg": float(res.force_chain_angle_deg),
        "surface_profile": res.surface_profile,
        "wall_clock_s": float(res.wall_clock_s),
        "geometry": info,
    }
    if settle_probe_steps:
        out["static_dissipation"] = static_pile_dissipation(
            res.final_positions, spec, mu=mu, mu_rolling=mu_rolling,
            steps=settle_probe_steps, dt=dt, density=density, e_star=e_star,
            restitution=restitution, box_half=box_half)
    out["total_wall_clock_s"] = time.perf_counter() - t0
    return out


def static_pile_dissipation(positions: np.ndarray, spec: ChannelSpec, *,
                            mu: float, mu_rolling: float, steps: int = 4000,
                            dt: Optional[float] = 2.5e-4, density: float = 2500.0,
                            e_star: float = 1.0e9, restitution: float = 0.4,
                            box_half: float = 0.09,
                            pre_settle_steps: int = 2000) -> Dict[str, float]:
    """C3: spurious energy loss rate of a *static* pile (per step, relative to E0).

    The settled heap of :func:`run_repose` is re-instantiated at rest with the
    gates open; the only energy source is gravity and the only sink should be
    the physical contact damping of genuinely moving particles.  A representation
    that keeps the surface bumpy generates micro-collisions and therefore a
    non-zero ``dE/E0/step``; the criterion is ``<= 1e-6``.
    """
    bodies, _ = build_bodies(spec, np.asarray(positions, dtype=float), density)
    for body in bodies:
        body.velocity = np.zeros(3)
        body.omega = np.zeros(3)
    mat = material(mu, mu_rolling, e_star=e_star, restitution=restitution)
    cfg = SQSystemConfig(material=mat, dt=dt)
    walls = box_walls(box_half, 0.0, mu, mu_rolling)
    for i in GATE_WALLS:
        walls[i].active = False
    sysm = SQSystem(bodies, walls, cfg)
    sysm.run(pre_settle_steps)          # relax the artificially built bed first
    for body in bodies:
        body.velocity[:] = 0.0
        body.omega[:] = 0.0
    sysm.run(20)
    e0 = float(sysm.energy())
    log = []
    for _ in range(steps):
        sysm.step()
        log.append(float(sysm.energy()))
    e1 = float(log[-1])
    drift = abs(e1 - e0) / max(abs(e0), 1e-30)
    vmax = float(max(np.linalg.norm(b.velocity) for b in bodies))
    return {
        "steps": int(steps),
        "energy_start_j": e0,
        "energy_end_j": e1,
        "relative_drift": drift,
        "drift_per_step": drift / max(steps, 1),
        "max_speed_end_m_per_s": vmax,
        "criterion_per_step": 1e-6,
        "passes": bool(drift / max(steps, 1) <= 1e-6),
    }

def pour_positions(n: int, radius: float, funnel_half: float = 0.011,
                   z_base: float = 0.035, seed: int = 5) -> np.ndarray:
    """Initial positions for a pouring heap: a narrow column above the floor.

    Particles are released from a funnel of half-width ``funnel_half`` stacked
    from ``z_base`` upward; they fall, avalanche and build a cone whose free
    surface is the angle of repose (the standard laboratory pour test).
    """
    rng = np.random.default_rng(seed)
    spacing = 2.4 * radius
    per = max(int(math.floor(2.0 * funnel_half / spacing)), 1)
    layers = int(math.ceil(n / (per * per)))
    off = -0.5 * (per - 1) * spacing
    pts = []
    for kz in range(layers):
        for ky in range(per):
            for kx in range(per):
                pts.append([off + kx * spacing, off + ky * spacing, z_base + kz * spacing])
    return np.array(pts[:n], dtype=float) + 0.2 * radius * rng.standard_normal((n, 3))


def run_pour_heap(spec: ChannelSpec, *, mu: float = 0.5, mu_rolling: float = 0.4,
                  n_particles: int = 320, box_half: float = 0.04,
                  funnel_half: float = 0.011, z_base: float = 0.035,
                  settle_steps: int = 6000, dt: Optional[float] = None,
                  density: float = 2500.0, e_star: float = 1.1e6,
                  restitution: float = 0.4, seed: int = 5,
                  report_every: int = 0) -> Dict[str, object]:
    """Pour-test heap: particles fall from a funnel and settle into a cone.

    Returns the same payload as :func:`run_repose` (repose angle, penetration,
    energy drift, force-chain statistics) so the three channels are directly
    comparable.  The gates are absent here: the heap is unconfined, which is what
    makes the free surface carry the angle-of-repose information.
    """
    t0 = time.perf_counter()
    positions = pour_positions(n_particles, spec.radius, funnel_half=funnel_half, seed=seed)
    bodies, info = build_bodies(spec, positions, density)
    mat = material(mu, mu_rolling, e_star=e_star, restitution=restitution)
    cfg = SQSystemConfig(material=mat, dt=dt)
    walls = box_walls(box_half, 0.0, mu, mu_rolling)
    for i in GATE_WALLS:
        walls[i].active = False
    sysm = SQSystem(bodies, walls, cfg)
    sysm.run(settle_steps)
    pos = sysm.positions()
    angle, profile = repose_angle_from_cloud(pos)
    # settled-ness diagnostics
    vmax = float(max(np.linalg.norm(b.velocity) for b in bodies))
    e_window = sysm.energy_log
    drift = float("nan")
    if len(e_window) > 10:
        e0, e1 = float(e_window[len(e_window) // 2]), float(e_window[-1])
        drift = abs(e1 - e0) / max(abs(e0), 1e-30) * 1000.0 / max(len(e_window) // 2, 1)
    return {
        "channel": spec.kind,
        "method": "pour",
        "shape_exponent": float(spec.shape_exponent),
        "aspect": float(spec.aspect),
        "n_particles": int(n_particles),
        "mu": float(mu),
        "mu_rolling": float(mu_rolling),
        "dt_s": float(sysm.dt),
        "steps": int(settle_steps),
        "repose_angle_deg": float(angle),
        "surface_profile": profile,
        "max_penetration_ratio": float(sysm.max_penetration_ratio),
        "energy_drift_per_1000": drift,
        "max_speed_end_m_per_s": vmax,
        "n_contacts": len(sysm.last_contacts),
        "geometry": info,
        "wall_clock_s": float(time.perf_counter() - t0),
        "final_positions": sysm.positions().tolist() if _want_positions() else None,
    }


def _want_positions() -> bool:
    import os
    return os.environ.get("TENSORDEM_KEEP_POSITIONS", "0") == "1"


def run_trickle_heap(spec: ChannelSpec, *, mu: float = 0.6, mu_rolling: float = 0.5,
                     n_particles: int = 240, batch: int = 12, every: int = 120,
                     box_half: float = 0.045, funnel_half: float = 0.008,
                     drop_height: float = 0.045, relax_steps: int = 4000,
                     dt: Optional[float] = None, density: float = 2500.0,
                     e_star: float = 1.1e6, restitution: float = 0.1,
                     damping: float = 0.0, seed: int = 3) -> Dict[str, object]:
    """Slow pour: particles are injected in batches above the box centre.

    Pouring is the laboratory method that actually produces a cone; a bulk column
    release either stands (friction) or topples as a block, both of which were
    measured and rejected before this driver was written.
    """
    t0 = time.perf_counter()
    walls = box_walls(box_half, 0.0, mu, mu_rolling)
    for i in GATE_WALLS:
        walls[i].active = False
    mat = material(mu, mu_rolling, e_star=e_star, restitution=restitution)
    cfg = SQSystemConfig(material=mat, dt=dt, damping_global=damping)
    positions = pour_positions(n_particles, spec.radius, funnel_half=funnel_half,
                               z_base=drop_height, seed=seed)
    bodies, info = build_bodies(spec, positions[:batch], density)
    sysm = SQSystem(bodies, walls, cfg)
    injected = batch
    while injected < n_particles:
        sysm.run(every)
        new_bodies, _ = build_bodies(spec, positions[injected:injected + batch], density)
        sysm.bodies.extend(new_bodies)
        sysm._clump_cache = None
        bodies.extend(new_bodies)
        injected += batch
    sysm.run(relax_steps)
    pos = sysm.positions()
    angle, profile = repose_angle_from_cloud(pos)
    vmax = float(max(np.linalg.norm(b.velocity) for b in bodies))
    return {
        "channel": spec.kind,
        "method": "trickle",
        "shape_exponent": float(spec.shape_exponent),
        "n_particles": int(n_particles),
        "mu": float(mu),
        "mu_rolling": float(mu_rolling),
        "dt_s": float(sysm.dt),
        "repose_angle_deg": float(angle),
        "surface_profile": profile,
        "max_penetration_ratio": float(sysm.max_penetration_ratio),
        "max_speed_end_m_per_s": vmax,
        "n_contacts": len(sysm.last_contacts),
        "geometry": info,
        "wall_clock_s": float(time.perf_counter() - t0),
    }
