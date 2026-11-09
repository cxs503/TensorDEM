"""Lagrangian bed utilities: packing, static force audit, hindered-settling probes.

Everything here operates on the soft-sphere DEM of :mod:`tensordem.lagrangian_bed.mechanics`
and the 1-D fluid column of :mod:`tensordem.lagrangian_bed.column`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import closures as cl
from . import column as col
from .mechanics import (
    Box,
    ContactParameters,
    ParticleState,
    contact_forces as contact_forces_for,
    kinetic_energy,
    make_particles,
    step,
    wall_reaction_z,
)


@dataclass
class BedConfig:
    diameter: float = 1.0e-3
    rho_particle: float = 2500.0
    rho_fluid: float = 1000.0
    mu: float = 1.0e-3
    n_particles: int = 512
    size_xy: float = 12.0e-3
    polydispersity: float = 0.08
    seed: int = 20261109


# ---------------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------------
def initial_lattice(
    n_particles: int,
    diameter: float,
    size_xy: float,
    floor_z: float,
    seed: int,
    jitter: float = 0.06,
    density: float = 2500.0,
    polydispersity: float = 0.0,
):
    """Non-overlapping random-start lattice above the floor (deposition state).

    The in-plane spacing is set from the box so that neighbouring grains never
    start overlapping; the layers stacked in z give the requested particle
    count.  Small in-plane jitter breaks the crystalline symmetry.
    """
    rng = np.random.default_rng(seed)
    spacing = 1.12 * diameter
    n_xy = max(int(np.floor(size_xy / spacing)), 1)
    n_z = int(np.ceil(n_particles / float(n_xy**2)))
    grid = np.arange(n_xy) * spacing + 0.5 * spacing
    gx, gy = np.meshgrid(grid, grid, indexing="ij")
    pts = []
    for k in range(n_z):
        for x, y in zip(gx.ravel(), gy.ravel()):
            pts.append([x, y, floor_z + (k + 1.0) * spacing])
    pts = np.asarray(pts[:n_particles], dtype=float)
    pts[:, 0] = np.mod(pts[:, 0] + jitter * spacing * rng.standard_normal(pts.shape[0]), size_xy)
    pts[:, 1] = np.mod(pts[:, 1] + jitter * spacing * rng.standard_normal(pts.shape[0]), size_xy)
    radius = 0.5 * diameter * (1.0 + polydispersity * rng.standard_normal(pts.shape[0]))
    radius = np.clip(radius, 0.40 * diameter, 0.60 * diameter)
    return make_particles(pts, radius, density)


def deposit(
    config: BedConfig,
    *,
    steps: int = 2500,
    dt: float | None = None,
    params: ContactParameters | None = None,
    box_height: float = 60.0e-3,
    ceiling_z: float | None = None,
    gravity: float = cl.GRAVITY,
    periodic_xy: bool = True,
) -> dict:
    """Deposit particles under gravity in a box and return the settled state."""
    params = params or ContactParameters()
    state = initial_lattice(
        config.n_particles,
        config.diameter,
        config.size_xy,
        floor_z=0.0,
        seed=config.seed,
        density=config.rho_particle,
        polydispersity=config.polydispersity,
    )
    box = Box(
        size_xy=config.size_xy,
        height=box_height,
        floor_z=0.0,
        ceil_z=ceiling_z,
        periodic_xy=periodic_xy,
    )
    if dt is None:
        dt = params.contact_time(float(np.mean(state.mass))) / 50.0
    history = []
    for k in range(steps):
        state = step(state, box, params, dt, gravity=gravity, max_speed=1.0)
        if k % 200 == 0:
            history.append({"step": k, "ke": kinetic_energy(state), "top_z": float(np.max(state.pos[:, 2]))})
    return {"state": state, "box": box, "params": params, "dt": dt, "history": history}


def compact(
    deposition: dict,
    *,
    target_solid_fraction: float,
    steps: int = 4000,
    piston_speed: float = 0.02,
    rate: int = 200,
    gravity: float = cl.GRAVITY,
) -> dict:
    """Quasi-statically densify a settled bed with a descending piston (ceiling)."""
    state, box, params, dt = (
        deposition["state"],
        deposition["box"],
        deposition["params"],
        deposition["dt"],
    )
    total_solid = float(np.sum(4.0 / 3.0 * np.pi * state.radius**3))
    target_height = total_solid / target_solid_fraction / box.area
    ceil = float(np.max(state.pos[:, 2] + state.radius))
    # quasi-static piston: cover the whole gap in 90 % of the allocated steps
    piston_speed = max((ceil - target_height) / (0.9 * steps * dt), 1e-6)
    box = Box(
        size_xy=box.size_xy,
        height=box.height,
        floor_z=box.floor_z,
        ceil_z=ceil,
        periodic_xy=box.periodic_xy,
    )
    for k in range(steps):
        if ceil > target_height:
            ceil = max(target_height, ceil - piston_speed * dt)
        box.ceil_z = ceil
        state = step(state, box, params, dt, gravity=gravity, max_speed=0.5)
    released = Box(
        size_xy=box.size_xy,
        height=box.height,
        floor_z=box.floor_z,
        ceil_z=None,
        periodic_xy=box.periodic_xy,
    )
    return {
        **deposition,
        "state": state,
        "box": released,
        "piston_ceiling": ceil,
        "target_height": target_height,
        "piston_speed": piston_speed,
    }


def compress_packing(
    config: BedConfig,
    *,
    target_solid_fraction: float = 0.62,
    steps: int = 4000,
    piston_speed: float = 0.01,
    fluidize_steps: int = 400,
    params: ContactParameters | None = None,
    box_height: float = 40.0e-3,
    jitter: float = 0.12,
    seed: int | None = None,
) -> dict:
    """Homogeneous packing by piston compression of a jittered cloud (no gravity).

    Compressing a random cloud under a uniform piston load avoids the vertical
    density gradient that gravity deposition leaves behind, so the packing is a
    reproducible random close packing with a well-defined porosity statistics.
    Gravity is switched off during the compression (``gravity = 0``); the
    resulting configuration is then used as the static bed.
    """
    params = params or ContactParameters()
    seed = config.seed if seed is None else seed
    state = initial_lattice(
        config.n_particles,
        config.diameter,
        config.size_xy,
        floor_z=0.0,
        seed=seed,
        jitter=jitter,
        density=config.rho_particle,
        polydispersity=config.polydispersity,
    )
    box = Box(size_xy=config.size_xy, height=box_height, floor_z=0.0, periodic_xy=True)
    dt = params.contact_time(float(np.mean(state.mass))) / 50.0
    for _ in range(fluidize_steps):  # randomise the lattice at zero gravity
        state = step(state, box, params, dt, gravity=0.0, max_speed=0.5)
    deposition = {"state": state, "box": box, "params": params, "dt": dt, "history": []}
    return compact(
        deposition,
        target_solid_fraction=target_solid_fraction,
        steps=steps,
        piston_speed=piston_speed,
        gravity=0.0,
    )


def bed_height(state: ParticleState, quantile: float = 0.995) -> float:
    return float(np.quantile(state.pos[:, 2] + state.radius, quantile))


def porosity(state: ParticleState, box: Box, height: float | None = None) -> dict:
    h = bed_height(state) if height is None else float(height)
    solid = float(np.sum(4.0 / 3.0 * np.pi * state.radius**3))
    bed_volume = box.area * h
    eps = 1.0 - solid / bed_volume
    return {"porosity": eps, "solid_fraction": 1.0 - eps, "bed_height": h, "bed_volume": bed_volume}


def interior_mask(state: ParticleState, box: Box, probe_radius: float, height: float | None = None) -> np.ndarray:
    """Grains whose voidage probe lies fully inside the bed (away from free surface/walls)."""
    h = bed_height(state) if height is None else float(height)
    r_mean = float(np.mean(state.radius))
    m = state.pos[:, 2] > 2.0 * r_mean
    m &= state.pos[:, 2] < h - probe_radius
    if not box.periodic_xy:  # only wall-bounded boxes need the lateral margin
        m &= state.pos[:, 0] > probe_radius
        m &= state.pos[:, 0] < box.size_xy - probe_radius
        m &= state.pos[:, 1] > probe_radius
        m &= state.pos[:, 1] < box.size_xy - probe_radius
    return m


def porosity_statistics(state: ParticleState, probe_factor: float = 2.0, box: Box | None = None) -> dict:
    """Local porosity (spherical probe) statistics.

    ``probe_factor`` is the probe *radius* in particle diameters.  If ``box`` is
    given the statistics are restricted to grains whose probe stays inside the
    bed (no free-surface or wall bias).
    """
    eps_i, solid_i = cl.local_voidage(
        state.pos, state.radius, probe_factor,
        period=float(box.size_xy) if (box is not None and box.periodic_xy) else None,
    )
    probe_radius = probe_factor * float(2.0 * np.mean(state.radius))
    if box is not None:
        mask = interior_mask(state, box, probe_radius)
    else:
        mask = np.ones_like(eps_i, dtype=bool)
    sub = eps_i[mask] if np.any(mask) else eps_i
    return {
        "probe_factor": probe_factor,
        "probe_radius_m": probe_radius,
        "n_probed": int(mask.sum()),
        "mean": float(np.mean(sub)),
        "std": float(np.std(sub, ddof=1)),
        "min": float(np.min(sub)),
        "max": float(np.max(sub)),
        "per_particle": eps_i,
        "mask": mask,
    }


# ---------------------------------------------------------------------------
# static force audit / momentum
# ---------------------------------------------------------------------------
def static_force_audit(state: ParticleState, box: Box, params: ContactParameters, fluid, dt: float = 0.0) -> dict:
    """Sum of body + fluid forces on a static bed vs the floor reaction.

    The static bed is at rest and the fluid is at rest, so drag and dynamic
    pressure-gradient forces vanish; the residual of the z-force accounting must
    be at round-off level relative to the total weight.
    """
    mass = state.mass
    weight = float(np.sum(mass * fluid.gravity))
    buoyancy = float(np.sum(fluid.rho * 4.0 / 3.0 * np.pi * state.radius**3 * fluid.gravity))
    reaction = wall_reaction_z(state, box, params)
    net = weight - buoyancy - reaction
    return {
        "total_weight": weight,
        "total_buoyancy": buoyancy,
        "floor_reaction": reaction,
        "net_force": net,
        "residual_over_weight": abs(net) / weight,
    }


# ---------------------------------------------------------------------------
# hindered settling probe (Richardson-Zaki)
# ---------------------------------------------------------------------------
def lattice_background(eps_target: float, diameter: float, n_side: int = 7, density: float = 2500.0):
    """Frozen cubic lattice of spheres with a prescribed nominal voidage."""
    spacing = diameter * (np.pi / 6.0 / (1.0 - eps_target)) ** (1.0 / 3.0)
    g = np.arange(n_side) * spacing
    pts = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    pts[:, 0] += 0.5 * spacing
    pts[:, 1] += 0.5 * spacing
    pts[:, 2] += 0.5 * spacing
    radius = np.full(pts.shape[0], 0.5 * diameter)
    return make_particles(pts, radius, density), spacing


def quench(
    deposition: dict,
    *,
    steps: int = 3000,
    damping: float = 0.2,
    tol: float | None = 1e-6,
    gravity: float = cl.GRAVITY,
) -> dict:
    """Damped-dynamics relaxation of a deposited bed to static equilibrium.

    Velocities are scaled by ``1 - damping`` at every step (a standard quench for
    granular packings).  The loop stops early once the residual of the force
    balance ``|sum F_z| / (sum m g)`` falls below ``tol``.
    """
    state, box, params, dt = (
        deposition["state"],
        deposition["box"],
        deposition["params"],
        deposition["dt"],
    )
    weight = float(np.sum(state.mass * gravity))
    residual = None
    for k in range(steps):
        force = contact_forces_for(state, box, params, dt) + np.column_stack(
            (np.zeros(state.n), np.zeros(state.n), -state.mass * gravity)
        )
        residual = float(abs(np.sum(force[:, 2])) / weight)
        mean_force = float(np.mean(np.linalg.norm(force, axis=1)))
        if mean_force > 0:
            state = ParticleState(
                pos=state.pos + state.vel * dt,
                vel=(state.vel + force / state.mass[:, None] * dt) * (1.0 - damping),
                radius=state.radius,
                density=state.density,
            )
        if tol is not None and residual <= tol:
            break
    return {**deposition, "state": state, "quench_steps": k + 1, "static_residual_over_weight": residual}


def static_grain_equilibrium(
    config: BedConfig,
    *,
    steps: int = 4000,
    damping: float = 0.02,
    params: ContactParameters | None = None,
) -> dict:
    """Static force ledger of a single grain resting on the floor.

    The soft-sphere ODE is relaxed until the contact force reproduces the
    submerged weight of the grain; the residual of the *body/contact force
    models* is then at round-off level.  This isolates the force accounting from
    the marginal stability of a many-grain jammed pile (reported separately).
    """
    params = params or ContactParameters()
    fluid = col.FluidProperties(rho=config.rho_fluid, mu=config.mu)
    box = Box(size_xy=config.size_xy, height=40.0e-3, floor_z=0.0, periodic_xy=True)
    state = make_particles(
        np.array([[0.5 * config.size_xy, 0.5 * config.size_xy, 0.5 * config.diameter]]),
        0.5 * config.diameter,
        config.rho_particle,
    )
    dt = params.contact_time(float(state.mass[0])) / 50.0
    weight = float(state.mass[0] * fluid.gravity)
    buoyancy = float(fluid.rho * cl.sphere_volume(config.diameter) * fluid.gravity)
    submerged = weight - buoyancy
    for k in range(steps):
        contact = contact_forces_for(state, box, params, dt)
        body = np.array([[0.0, 0.0, buoyancy - weight]])
        force = contact + body
        state = ParticleState(
            pos=state.pos + state.vel * dt,
            vel=(state.vel + force / state.mass[:, None] * dt) * (1.0 - damping),
            radius=state.radius,
            density=state.density,
        )
    contact = contact_forces_for(state, box, params, dt)
    net = float(contact[0, 2] + buoyancy - weight)
    return {
        "n_steps": k + 1,
        "penetration": float(state.radius[0] - state.pos[0, 2]),
        "floor_force": float(contact[0, 2]),
        "submerged_weight": submerged,
        "net_force": net,
        "residual_over_weight": abs(net) / weight,
    }


def fire_minimize(
    deposition: dict,
    *,
    steps: int = 6000,
    dt0: float | None = None,
    tol: float = 1e-8,
    gravity: float = cl.GRAVITY,
) -> dict:
    """FIRE energy minimisation (Bitzek et al. 2006) to a static force balance.

    Damped molecular dynamics only creeps towards equilibrium; the FIRE
    (fast inertial relaxation engine) update converges the same soft-sphere
    system to ``max|F_i| / (m g) < tol``, which is what a *static bed* force
    ledger requires.
    """
    state, box, params = deposition["state"], deposition["box"], deposition["params"]
    dt = float(dt0 if dt0 is not None else deposition["dt"])
    dt_base = dt
    dt_max = 10.0 * dt
    alpha, alpha0 = 0.1, 0.1
    v = state.vel.copy()
    weight = float(np.sum(state.mass * gravity))
    residual_sum = None
    max_force_rel = None
    for k in range(steps):
        force = contact_forces_for(state, box, params, dt) + np.column_stack(
            (np.zeros(state.n), np.zeros(state.n), -state.mass * gravity)
        )
        residual_sum = abs(float(np.sum(force[:, 2]))) / weight
        max_force_rel = float(np.max(np.linalg.norm(force, axis=1))) / weight
        power = float(np.sum(force * v))
        speed = float(np.linalg.norm(v))
        if speed > 0.0:
            v = (1.0 - alpha) * v + alpha * (speed) * force / np.linalg.norm(force, axis=1, keepdims=True).clip(1e-300)
        v = v + force / state.mass[:, None] * dt
        if power > 0.0:
            dt = min(dt * 1.05, dt_max)
            alpha *= 0.99
        else:
            v[:] = 0.0
            dt = max(dt * 0.5, dt_base * 1e-3)
            alpha = alpha0
        pos = state.pos + v * dt
        state = ParticleState(pos=pos, vel=v, radius=state.radius, density=state.density)
        if max_force_rel is not None and max_force_rel < tol:
            break
    return {
        **deposition,
        "state": state,
        "fire_steps": k + 1,
        "static_residual_over_weight": residual_sum,
        "max_force_over_weight": max_force_rel,
    }


def single_particle_settling(
    config: BedConfig,
    *,
    n_steps: int = 20000,
    dt_factor: float = 0.01,
) -> dict:
    """Free settling of one sphere with Schiller-Naumann drag (isolated limit)."""
    fluid = col.FluidProperties(rho=config.rho_fluid, mu=config.mu)
    tau = config.rho_particle * config.diameter**2 / (18.0 * fluid.mu)
    dt = dt_factor * tau
    u = 0.0
    mass = config.rho_particle * float(cl.sphere_volume(config.diameter))
    hist = []
    for k in range(n_steps):
        drag = float(cl.schiller_naumann_drag(u, config.diameter, fluid.rho, fluid.mu))
        weight = -float(cl.submerged_weight(config.diameter, config.rho_particle, fluid.rho))
        acc = (weight - drag) / mass
        u += acc * dt
        if k % 200 == 0:
            hist.append(u)
        if k > 20 and abs(acc * dt) <= 1e-12 * max(abs(u), 1e-12):
            break
    return {"u_settle": float(u), "n_steps": k + 1, "dt": dt, "time": (k + 1) * dt, "history": hist}


def hindered_settling_probe(
    eps_target: float,
    config: BedConfig,
    *,
    n_steps: int = 6000,
    dt_factor: float = 0.02,
    tol: float = 1e-10,
    closure: str = "gidaspow",
    u_terminal: float | None = None,
) -> dict:
    """Measure the steady settling velocity of one particle in a frozen bed.

    The mobile particle sits at the centre of the lattice; the background grains
    are frozen (one-way coupling, fluid at rest).  The steady velocity is
    compared with ``u_t * eps^(n-1)`` to extract the empirical exponent ``n``.
    """
    background, spacing = lattice_background(eps_target, config.diameter, density=config.rho_particle)
    centre = np.array([[0.5 * (background.pos[:, 0].min() + background.pos[:, 0].max()),
                        0.5 * (background.pos[:, 1].min() + background.pos[:, 1].max()),
                        0.5 * (background.pos[:, 2].min() + background.pos[:, 2].max())]])
    pos = np.vstack([centre, background.pos])
    radius = np.concatenate([[0.5 * config.diameter], background.radius])
    # only particle 0 is integrated (the background grains are frozen); the
    # ODE below updates its position/velocity directly.
    state = make_particles(pos, radius, config.rho_particle)
    box = Box(
        size_xy=float(background.pos[:, 0].max() + 2 * config.diameter),
        height=float(background.pos[:, 2].max() + 2 * config.diameter),
        periodic_xy=False,
    )
    eps_i, _ = cl.local_voidage(background.pos, background.radius, 1.0)
    eps_mobile = float(np.mean(eps_i))
    fluid = col.FluidProperties(rho=config.rho_fluid, mu=config.mu)
    tau = config.rho_particle * config.diameter**2 / (18.0 * config.mu)
    dt = dt_factor * tau
    u_prev = 0.0
    hist = []
    for k in range(n_steps):
        v = state.vel[0]
        if closure == "rz":
            n_exp = float(cl.rz_exponent(fluid.rho * (u_terminal or 1.0) * config.diameter / fluid.mu))
            beta = cl.rz_hindrance_beta(
                np.array([eps_mobile]), config.diameter, fluid.rho, fluid.mu,
                np.array([-v[2]]), np.array([n_exp]),
            )
            drag = -beta * v[2] * float(cl.sphere_volume(config.diameter)) / (1.0 - eps_mobile)
        else:
            drag, beta, _ = cl.particle_drag_force(
                np.array([eps_mobile]), config.diameter, fluid.rho, fluid.mu,
                np.zeros((1, 3)), v[None, :],
            )
        buoy = cl.buoyancy_force(config.diameter, fluid.rho)
        force = np.atleast_2d(drag)[0] + buoy
        force[2] -= config.rho_particle * float(cl.sphere_volume(config.diameter)) * fluid.gravity
        acc = force / (config.rho_particle * float(cl.sphere_volume(config.diameter)))
        vel = v + acc * dt
        state.vel[0] = vel
        state.pos[0] = state.pos[0] + vel * dt
        if k % 50 == 0:
            hist.append(float(-vel[2]))
        if k > 200 and abs(vel[2] - u_prev) <= tol * max(abs(vel[2]), 1e-12):
            break
        u_prev = float(vel[2])
    u_settle = float(-state.vel[0, 2])
    return {
        "closure": closure,
        "eps_nominal": eps_target,
        "eps_local": float(eps_mobile),
        "spacing_over_d": float(spacing / config.diameter),
        "u_settle": u_settle,
        "beta": float(np.atleast_1d(beta)[0]),
        "n_steps": k + 1,
        "dt": dt,
        "history": hist,
    }


def uniform_suspension_settling(
    eps_target: float,
    config: BedConfig,
    *,
    n_side: int = 4,
    closure: str = "rz",
    n_steps: int = 4000,
    dt_factor: float = 0.01,
    tol: float = 1e-10,
) -> dict:
    """Batch settling of a *homogeneous* suspension (exact voidage, periodic).

    A cubic lattice of ``n_side^3`` grains with spacing set from the target
    voidage is released in a fluid at rest; the voidage seen by every grain is
    exactly ``eps_target`` (no walls, no contacts), so the steady settling speed
    is the Richardson-Zaki slip velocity for that voidage.
    """
    d = config.diameter
    spacing = d * (np.pi / 6.0 / (1.0 - eps_target)) ** (1.0 / 3.0)
    n = n_side**3
    grid = np.arange(n_side) * spacing
    pts = np.array(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1).T
    box_len = n_side * spacing
    eps = 1.0 - n * float(cl.sphere_volume(d)) / box_len**3
    fluid = col.FluidProperties(rho=config.rho_fluid, mu=config.mu)
    ut = cl.terminal_velocity(d, config.rho_particle, fluid.rho, fluid.mu)
    n_exp = float(cl.rz_exponent(ut["re_terminal"]))
    tau = config.rho_particle * d**2 / (18.0 * fluid.mu)
    dt = dt_factor * tau
    mass = config.rho_particle * float(cl.sphere_volume(d))
    vel = np.zeros(n)  # upward-positive particle velocity
    u_prev = 0.0
    v_particle = float(cl.sphere_volume(d))
    weight = -(config.rho_particle - fluid.rho) * v_particle * fluid.gravity
    k = 0
    for k in range(n_steps):
        if closure == "rz":
            beta = cl.rz_hindrance_beta(eps, d, fluid.rho, fluid.mu, np.abs(vel), n_exp)
        else:
            beta, _ = cl.gidaspow_beta(eps, d, fluid.rho, fluid.mu, np.abs(vel))
        drag = -beta * vel * v_particle / (1.0 - eps)  # opposes the motion
        acc = (weight + drag) / mass
        vel = vel + acc * dt
        u = float(-np.mean(vel))
        if k > 20 and abs(u - u_prev) <= tol * max(abs(u), 1e-12):
            break
        u_prev = u
    return {
        "closure": closure,
        "eps": float(eps),
        "eps_target": eps_target,
        "u_settle": float(-np.mean(vel)),
        "u_over_ut": float(-np.mean(vel) / ut["u_terminal"]),
        "n_closure": n_exp,
        "u_terminal": ut["u_terminal"],
        "re_terminal": ut["re_terminal"],
        "n_steps": k + 1,
        "dt": dt,
    }


def group_settling(
    config: BedConfig,
    *,
    initial_porosity: float = 0.90,
    steps: int = 1200,
    dt: float | None = None,
    params: ContactParameters | None = None,
    sample_every: int = 20,
    two_way: bool = True,
) -> dict:
    """Batch settling of a particle cloud with one-way fluid coupling.

    The fluid is at rest; particles feel drag + buoyancy and collide with each
    other.  The pressure gradient needed to hold the fluid at rest is
    reconstructed cell-by-cell (see :func:`column.at_rest_pressure`).
    """
    params = params or ContactParameters()
    state = initial_lattice(
        config.n_particles,
        config.diameter,
        config.size_xy,
        floor_z=0.0,
        seed=config.seed,
        density=config.rho_particle,
        polydispersity=config.polydispersity,
    )
    # stretch the lattice vertically to the requested initial porosity
    solid = float(np.sum(4.0 / 3.0 * np.pi * state.radius**3))
    h_target = solid / (1.0 - initial_porosity) / config.size_xy**2
    z_lo, z_hi = float(state.pos[:, 2].min()), float(state.pos[:, 2].max())
    state.pos[:, 2] = z_lo + (state.pos[:, 2] - z_lo) * (h_target / max(z_hi - z_lo, 1e-9))
    box = Box(size_xy=config.size_xy, height=max(3.0 * h_target, 40.0e-3), floor_z=0.0, periodic_xy=True)
    if dt is None:
        dt = params.contact_time(float(np.mean(state.mass))) / 50.0
    fluid = col.FluidProperties(rho=config.rho_fluid, mu=config.mu)
    profile = col.ColumnProfile(
        z_lo=0.0, z_hi=box.height, n_cells=40, area=box.area
    )
    rec = {"time": [], "dp_drag": [], "u_mean": [], "eps_mean": [], "ke": []}
    for k in range(steps):
        cs = col.column_state(profile, state.pos, state.vel, state.radius, config.diameter)
        eps = cs["eps"]
        vp = cs["v_p"]
        drag_cells, beta, _ = cl.particle_drag_force(
            eps, config.diameter, fluid.rho, fluid.mu, np.zeros_like(vp), vp
        )
        # per-cell drag force -> distributed to particles by a cell-share scaling
        # (each particle sees the closure at its own cell voidage)
        particle_eps = np.interp(state.pos[:, 2], profile.centres(), eps)
        particle_drag, _, _ = cl.particle_drag_force(
            particle_eps, config.diameter, fluid.rho, fluid.mu,
            np.zeros_like(state.pos), state.vel,
        )
        buoy = np.zeros_like(state.pos)
        buoy[:, 2] = fluid.rho * cl.sphere_volume(2.0 * state.radius) * fluid.gravity
        state = step(state, box, params, dt, external_force=particle_drag + buoy, gravity=cl.GRAVITY, max_speed=2.0)
        if k % sample_every == 0:
            cs = col.column_state(profile, state.pos, state.vel, state.radius, config.diameter)
            pr = col.at_rest_pressure(profile, cs["eps"], cs["v_p"], config.diameter, fluid)
            rec["time"].append((k + 1) * dt)
            rec["dp_drag"].append(pr["dp_total"])
            rec["u_mean"].append(float(-np.mean(state.vel[:, 2])))
            rec["eps_mean"].append(float(1.0 - np.sum(cs["solid_volume"]) / (profile.dz * box.area)))
            rec["ke"].append(kinetic_energy(state))
    out = {key: np.asarray(val, dtype=float) for key, val in rec.items()}
    out["dt"] = dt
    out["final_state"] = state
    out["box"] = box
    return out


def gci(levels: list[float], refinement: float = 2.0) -> dict:
    """Grid Convergence Index from three (fine -> coarse) values.

    ``levels`` is ordered fine, medium, coarse.  Returns the Richardson-based
    observed order, the extrapolated value and the GCI of the fine level with a
    safety factor of 1.25 (Roache 1994).
    """
    f1, f2, f3 = [float(v) for v in levels]
    eps21 = abs(f2 - f1)
    eps32 = abs(f3 - f2)
    if eps32 <= 0.0 or eps21 <= 0.0:
        return {"p": float("nan"), "extrapolated": f1, "gci_fine": float("nan"), "asymptotic": False}
    p = float(np.log(eps32 / eps21) / np.log(refinement))
    r = refinement**p
    f_ext = f1 + (f1 - f2) / max(r - 1.0, 1e-30)
    gci_fine = 1.25 * abs((f1 - f2) / f1) / max(r - 1.0, 1e-30)
    return {
        "p": p,
        "r": float(r),
        "extrapolated": float(f_ext),
        "gci_fine": float(gci_fine),
        "asymptotic": bool(eps21 < eps32),
    }