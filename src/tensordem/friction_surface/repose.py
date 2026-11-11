"""Angle-of-repose anchor: pouring deposition, free-surface measurement, adjudication.

Protocol (v2, ``source_mode="fixed"``)
-------------------------------------
1. Monodisperse spheres are released in small bursts from a *nozzle of fixed
   height* above a frictional floor (no lateral wall, no side confinement).
2. A burst is only released when the nozzle region (a disk of radius
   ``source_radius`` and a slab of height ``nozzle_clearance`` radii below
   ``drop_height``) is clear of the burst still draining out of it; when that
   region is occupied by *deposit at rest* the pour closes and is never resumed —
   a laboratory pour cannot grow above its own nozzle.  The pour is therefore
   self-paced by the fall time of one charge (no clock is imposed on it), which
   keeps the burst rate slow enough that in-flight spheres never overlap.
3. Wait until the heap is at rest (RMS-speed criterion, not a hard-coded time),
   then measure the free-surface slope by fitting the *upper envelope* of the
   particle cloud in radial bins.

Why the source height is fixed (v1 defect, measured)
---------------------------------------------------
v1 raised the nozzle to ``max(drop_height, apex + 5 R)`` at every burst.  That is
the protocol rejected in the node report §5.3 ("块状点源动态抬高" -> the source
climbs with the heap): the pour never stops, the "heap" grows into a spire whose
height is set by the pour schedule instead of by friction, and the at-rest
criterion is never met — which is exactly why anchor A/B came back WITHHELD.
``source_mode="rising"`` still reproduces that pathology on purpose (it is used
by ``tests/test_friction_surface_repose_v2.py`` to pin the defect).

Why a pouring test and not a lifting cylinder
---------------------------------------------
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

__all__ = ["ReposeCase", "build_column", "simulate_repose", "heap_surface_angle",
           "nozzle_clear_time"]


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
    #: "fixed" = nozzle stays at ``drop_height`` and the pour closes once the
    #: heap surface comes within ``nozzle_clearance`` radii of it (v2 protocol);
    #: "rising" = legacy v1 behaviour (nozzle follows the apex, heap -> spire).
    source_mode: str = "fixed"
    #: nozzle clearance in radii at which the pour is closed (``fixed`` mode)
    nozzle_clearance: float = 2.0
    velocity_rest_threshold: float = 0.01   # m/s, rms speed at rest
    source_extent: float = 0.10
    drop_height: float = 0.020
    settle_time: float = 0.40
    packing_gap: float = 0.05             # relative gap in the initial lattice
    jitter: float = 0.02                  # positional jitter, in radii
    dt: float | None = None
    seed: int = 15204
    max_time: float = 30.0                # safety cap on simulated seconds
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


def nozzle_clear_time(case: ReposeCase) -> float:
    """Free-fall time of one charge through the nozzle region (s).

    A sphere released at ``drop_height`` (up to 2 R above it) must fall
    ``2 R + nozzle_clearance R`` to leave the slab in which a next burst would
    overlap it.  Solved with the release speed as the initial velocity; the
    bursts of the v2 protocol are paced by this time, which is why in-flight
    spheres never overlap (the v1 fountain mechanism).
    """
    model = case.model
    s_fall = (2.0 + max(case.nozzle_clearance, 2.2)) * model.radius
    v0 = max(case.release_speed, 0.0)
    return (-v0 + math.sqrt(v0 * v0 + 2.0 * model.gravity * s_fall)) / model.gravity


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
    if case.source_mode not in ("fixed", "rising"):
        raise ValueError(f"source_mode must be 'fixed' or 'rising', got {case.source_mode!r}")
    ps = PairState()
    active = 0
    cx = cy = 0.5 * case.source_extent
    v_rest = case.velocity_rest_threshold
    spawn_events = int(math.ceil(n_max / case.spawn_batch))
    # v2: the bursts are paced by the charge fall time, so the pour needs
    # ``spawn_events`` clear-times, not ``spawn_events`` spawn intervals.
    clear_steps_guess = max(int(round(nozzle_clear_time(case) / dt)), case.spawn_every)
    steps_spawn = spawn_events * clear_steps_guess
    steps_total = steps_spawn + int(case.settle_time / dt)
    steps_total = min(steps_total, max(int(case.max_time / dt), case.spawn_every + 1))
    ke_quiet = 0
    t_rest = None
    # ---- v2 pour bookkeeping -------------------------------------------------
    # The nozzle stays at ``drop_height``: ``nozzle_z_max`` is recorded so a test
    # can prove it never climbs with the heap (the v1 defect).
    nozzle_z_max = 0.0
    last_spawn_step = -1
    pour_closed = False
    bursts_released = 0
    bursts_skipped = 0
    spawn_step = np.full(n_max, -1, dtype=np.int64)
    clear_steps = max(int(round(nozzle_clear_time(case) / dt)), case.spawn_every)
    clear_r = max(case.nozzle_clearance, 2.2) * model.radius
    zone_hi = case.drop_height + 2.0 * model.radius + clear_r
    for step in range(steps_total):
        if step % case.spawn_every == 0 and active < n_max and not pour_closed:
            # overlap-safe release: a spawned sphere that would intersect an
            # already-present one is refused (an overlapped spawn injects a huge
            # force and throws particles metres away - measured in the first try)
            z_now = case.drop_height
            if case.source_mode == "rising" and active > 0:
                # legacy v1: the nozzle is dragged up with the heap apex
                z_now = max(case.drop_height,
                            float(pos[:active, 2].max()) + 5.0 * model.radius)
            nozzle_z_max = max(nozzle_z_max, z_now)
            release = True
            if case.source_mode == "fixed" and active > 0:
                in_zone = np.where((pos[:active, 2] > z_now - clear_r)
                                   & (pos[:active, 2] < zone_hi))[0]
                if in_zone.size:
                    rho = np.hypot(pos[in_zone, 0] - cx, pos[in_zone, 1] - cy)
                    in_zone = in_zone[rho < case.source_radius + clear_r]
                if in_zone.size:
                    draining = (step - spawn_step[in_zone]) < clear_steps
                    if draining.any():
                        # the previous charge is still in the nozzle: wait for it
                        release = False
                        bursts_skipped += 1
                    else:
                        # deposit at rest filling the nozzle: the pour is over
                        pour_closed = True
                        release = False
            if release:
                accepted = 0
                for _ in range(case.spawn_batch):
                    # a batch may overflow the array when earlier spawns of the
                    # same burst are refused (v1 IndexError)
                    if active >= n_max:
                        break
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
                        spawn_step[active] = step
                        active += 1
                        last_spawn_step = step
                        accepted += 1
                        break
                if accepted:
                    bursts_released += 1
                elif case.source_mode == "fixed" and active > 0:
                    # nothing could be placed: the nozzle is buried in the deposit
                    pour_closed = True
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
        if step > last_spawn_step + case.spawn_every and step % 100 == 0:
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
        "protocol": ("deposition from a fixed-height nozzle (pouring, v2)"
                     if case.source_mode == "fixed"
                     else "deposition from a rising source (pouring, v1 legacy)"),
        "source_mode": case.source_mode,
        "pour_closed": bool(pour_closed),
        "nozzle_z_max": float(nozzle_z_max),
        "pour_end_step": int(last_spawn_step),
        "bursts_released": int(bursts_released),
        "bursts_skipped": int(bursts_skipped),
        "clear_steps": int(clear_steps),
        "apex_height_m": float(pos[:active, 2].max()) if active else 0.0,
        "final_positions": final,
    }