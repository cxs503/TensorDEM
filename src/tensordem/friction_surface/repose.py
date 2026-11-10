"""Angle-of-repose anchor: column collapse, free-surface measurement, adjudication.

Protocol
--------
1. Start from a hexagonal-close-packed column of monodisperse spheres standing on
   a frictional floor (no lateral wall, no side confinement — this is the
   "column collapse" / sand-pile test, the DEM stand-in for a pouring
   repose experiment).
2. Release it; the column collapses and deposits into an axisymmetric heap.
3. Wait until the heap is at rest (kinetic-energy criterion, not a hard-coded
   time), then measure the free-surface slope by fitting the *upper envelope* of
   the particle cloud in radial bins.

Why a column collapse and not a lifting cylinder
------------------------------------------------
The repository already has a lifting-cylinder calibration surface
(:mod:`tensordem.repose_calibration`, mu -> theta_r forward map).  That module
is a *calibration* tool for this solver.  The present module answers a different
question — "does the macroscopic repose angle of this friction law reproduce
*measured* repose angles?" — so it needs (a) an independent geometry, and (b) an
automatic at-rest criterion, because a calibration run that stops at a fixed
time can measure a pile that is still spreading.

Measurement
-----------
``heap_surface_angle`` bins particles radially, takes the 90th percentile height
per bin (the surface, robust to the few particles still rolling), rejects bins
with too few particles or inside the rounded apex, and fits a straight line over
the flank.  The reported angle is ``arctan(-slope)`` of that line, i.e. the same
quantity a protractor measures on a poured heap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .kernel import ContactModel, PairState, PlaneWall, apply_walls, build_pairs, \
    integrate, kinetic_energy, pair_forces

__all__ = ["ReposeCase", "build_column", "simulate_repose", "heap_surface_angle"]


@dataclass(frozen=True)
class ReposeCase:
    """Column-collapse repose case (SI)."""

    mu: float = 0.5
    mu_rolling: float = 0.05
    mu_wall: float | None = None          # None -> mu
    mu_rolling_wall: float | None = None  # None -> mu_rolling
    radius: float = 2.0e-3
    density: float = 2500.0
    kn: float = 2.0e3
    kt_ratio: float = 0.5
    kr_ratio: float = 0.1
    restitution: float = 0.3
    ct_ratio: float = 0.0
    cap_total: bool = True
    gravity: float = 9.81
    column_diameter: float = 0.060
    column_height: float = 0.045
    n_particles: int = 1400
    spawn_batch: int = 4
    spawn_every: int = 40
    source_radius: float = 4.0e-3
    release_speed: float = 0.05
    velocity_rest_threshold: float = 0.01   # m/s, rms speed at rest
    source_extent: float = 0.10
    drop_height: float = 0.020
    settle_time: float = 0.40
    packing_gap: float = 0.05             # relative gap in the initial lattice
    jitter: float = 0.02                  # positional jitter, in radii
    dt: float | None = None
    seed: int = 15204
    max_time: float = 2.0                 # safety cap on simulated seconds
    ke_fraction_stop: float = 2.0e-4      # KE / (N m g R) at rest

    @property
    def model(self) -> ContactModel:
        return ContactModel(
            radius=self.radius, density=self.density, kn=self.kn,
            kt_ratio=self.kt_ratio, kr_ratio=self.kr_ratio, mu=self.mu,
            mu_rolling=self.mu_rolling, restitution=self.restitution,
            ct_ratio=self.ct_ratio, cap_total=self.cap_total,
            gravity=self.gravity,
        )


def build_column(case: ReposeCase) -> np.ndarray:
    """Hexagonal-close-packed cylinder of spheres resting on ``z = 0``.

    In-plane: hexagonal lattice with nearest-neighbour distance
    ``d = 2r(1 + packing_gap)``.  Layers: offset by ``(d/2, d/(2 sqrt 3))`` with
    vertical pitch ``d sqrt(2/3)``, i.e. the B/A stacking that keeps the
    nearest-neighbour distance at ``d`` (an offset of ``d/2`` in *both* in-plane
    directions instead gives ``0.87 d`` and the lattice starts interpenetrating).
    """
    r = case.radius
    d = 2.0 * r * (1.0 + case.packing_gap)
    dz = d * math.sqrt(2.0 / 3.0)
    rng = np.random.default_rng(case.seed)
    pts = []
    n_layers = int(case.column_height / dz) + 1
    span = int(math.ceil(case.column_diameter / d)) + 2
    r_max = 0.5 * case.column_diameter - r
    for layer in range(n_layers):
        z = r + layer * dz
        if z > case.column_height - r:
            break
        shift_x = 0.5 * d * (layer % 2)
        shift_y = d / (2.0 * math.sqrt(3.0)) * (layer % 2)
        for iy in range(-span, span + 1):
            y = iy * d * math.sqrt(3.0) / 2.0 + shift_y
            for ix in range(-span, span + 1):
                x = ix * d + (0.5 * d if iy % 2 else 0.0) + shift_x
                if math.hypot(x, y) <= r_max:
                    pts.append((x, y, z))
    pos = np.asarray(pts, dtype=float)
    if case.jitter:
        pos = pos + rng.standard_normal(pos.shape) * case.jitter * r
    pos[:, 0] += 0.5 * case.column_diameter
    pos[:, 1] += 0.5 * case.column_diameter
    pos[:, 2] = np.maximum(pos[:, 2], r)
    return pos


def heap_surface_angle(pos: np.ndarray, radius: float, *, n_bins: int = 24,
                       surface_percentile: float = 90.0,
                       min_per_bin: int = 6) -> dict:
    """Free-surface angle of a settled heap from radial bins of the upper envelope."""
    center = pos[:, :2].mean(axis=0)
    rho = np.linalg.norm(pos[:, :2] - center, axis=1)
    z = pos[:, 2]
    r_max = float(rho.max())
    edges = np.linspace(0.0, r_max * 1.001, n_bins + 1)
    xs, zs, counts = [], [], []
    for k in range(n_bins):
        m = (rho >= edges[k]) & (rho < edges[k + 1])
        cnt = int(m.sum())
        if cnt < min_per_bin:
            continue
        xs.append(0.5 * (edges[k] + edges[k + 1]))
        zs.append(float(np.percentile(z[m], surface_percentile)))
        counts.append(cnt)
    xs = np.asarray(xs)
    zs = np.asarray(zs)
    if xs.size < 3:
        return {"angle_deg": float("nan"), "slope": float("nan"),
                "bins": xs.tolist(), "heights": zs.tolist(), "n_bins_used": 0}
    # flank = bins whose surface height lies between 15 % and 90 % of the apex
    # height (the rounded apex and the thin toe are excluded); if the heap is
    # flat the window is empty and all inner bins are used instead.
    zmax = float(zs.max())
    sel = np.where((zs > 0.15 * zmax) & (zs < 0.90 * zmax))[0]
    if sel.size < 3:
        sel = np.arange(0, max(xs.size - 1, 1))
    if sel.size < 3:
        return {"angle_deg": float("nan"), "slope": float("nan"),
                "bins": xs.tolist(), "heights": zs.tolist(), "n_bins_used": int(xs.size),
                "reason": "flank too short to fit"}
    A = np.vstack([xs[sel] - xs[sel].mean(), np.ones(sel.size)]).T
    coef, *_ = np.linalg.lstsq(A, zs[sel], rcond=None)
    slope = float(coef[0])
    resid = zs[sel] - A @ coef
    return {
        "angle_deg": math.degrees(math.atan(-slope)),
        "slope": slope,
        "bins": xs.tolist(),
        "heights": zs.tolist(),
        "n_bins_used": int(xs.size),
        "rms_residual_m": float(np.sqrt(np.mean(resid ** 2))),
        "heap_radius_m": r_max,
        "apex_height_m": float(z.max()),
    }


def simulate_repose(case: ReposeCase, *, verbose: bool = False) -> dict:
    """Form a heap by deposition ("pouring") and measure its repose angle.

    Protocol (the laboratory repose test, not a column collapse): particles are
    released in small bursts from a short source disk above the base and are
    allowed to fall, roll and come to rest one burst at a time.  The free surface
    that develops is the *dynamic* repose angle of the material.

    Why not a column collapse: the repository's first attempt (documented in the
    node report) released a 40-diameter-tall column at once.  With spheres the
    released energy is dissipated only by restitution and rolling resistance, so
    the "heap" ended as a single-particle-thick carpet spreading over 0.2-1.1 m
    (measured), i.e. the protocol, not the friction law, set the answer.
    """
    model = case.model
    dt = case.dt or model.timestep
    n_max = case.n_particles
    pos = np.zeros((n_max, 3))
    vel = np.zeros((n_max, 3))
    omega = np.zeros((n_max, 3))
    rng = np.random.default_rng(case.seed)
    floor = PlaneWall((0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
                      mu=case.mu_wall if case.mu_wall is not None else case.mu,
                      mu_rolling=(case.mu_rolling_wall
                                  if case.mu_rolling_wall is not None
                                  else case.mu_rolling))
    ps = PairState()
    active = 0
    cx = cy = 0.5 * case.source_extent
    v_rest = case.velocity_rest_threshold
    spawn_events = int(math.ceil(n_max / case.spawn_batch))
    steps_spawn = spawn_events * case.spawn_every
    steps_total = steps_spawn + int(case.settle_time / dt)
    ke_quiet = 0
    t_rest = None
    for step in range(steps_total):
        if step % case.spawn_every == 0 and active < n_max:
            # overlap-safe release: a spawned sphere that would intersect an
            # already-present one is refused (an overlapped spawn injects a huge
            # force and throws particles metres away - measured in the first try)
            k = 0
            z_now = case.drop_height
            if active > 0:
                z_now = max(case.drop_height, float(pos[:active, 2].max()) + 5.0 * model.radius)
            for _ in range(case.spawn_batch):
                for _attempt in range(200):
                    ang = rng.random() * 2.0 * math.pi
                    rad = case.source_radius * math.sqrt(rng.random())
                    cand = np.array([cx + rad * math.cos(ang),
                                     cy + rad * math.sin(ang),
                                     z_now + rng.random() * 2.0 * model.radius])
                    if active > 0:
                        near = np.linalg.norm(pos[:active] - cand, axis=1)
                        if near.min() < 2.2 * model.radius:
                            continue
                    pos[active] = cand
                    vel[active] = (0.0, 0.0, -case.release_speed)
                    omega[active] = 0.0
                    active += 1
                    k += 1
                    break
        p_act = pos[:active]
        v_act = vel[:active]
        o_act = omega[:active]
        iu, ju, keys = build_pairs(p_act, model.radius, n_keys=n_max)
        f, tq = pair_forces(p_act, v_act, o_act, iu, ju, ps, model, dt, n_keys=n_max)
        fw, tw = apply_walls([floor], p_act, v_act, o_act, model, dt, step * dt)
        f += fw
        f[:, 2] -= model.mass * model.gravity
        p_act, v_act, o_act = integrate(p_act, v_act, o_act, f, tq + tw, model, dt)
        pos[:active] = p_act
        vel[:active] = v_act
        omega[:active] = o_act
        if step > steps_spawn and step % 100 == 0:
            v_rms = float(np.sqrt(np.mean(np.sum(v_act ** 2, axis=1)))) if active else 0.0
            if v_rms < v_rest:
                ke_quiet += 1
            else:
                ke_quiet = 0
            if ke_quiet >= 5:
                t_rest = step * dt
                steps_total = step
                break
        if verbose and step % 5000 == 0:
            print(f"  step {step} active {active} KE={kinetic_energy(v_act, o_act, model):.3e}")
    final = pos[:active]
    measured = heap_surface_angle(final, model.radius)
    return {
        "theta_r_deg": measured["angle_deg"],
        "measurement": measured,
        "n_particles": int(active),
        "mu": case.mu,
        "mu_rolling": case.mu_rolling,
        "dt": dt,
        "steps": steps_total,
        "t_end": steps_total * dt,
        "t_at_rest": t_rest,
        "reached_rest": t_rest is not None,
        "protocol": "deposition from a source disk (pouring)",
        "final_positions": final,
    }