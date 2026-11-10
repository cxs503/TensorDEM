"""Settling scenarios for the coupling chain, driven by the real DEM kernel.

Motivation
----------
The TensorPhysics coupling scene ``coupling_energy/scenarios/settling_lbm_dem.py``
fed its DEM leg with an *analytic reference model* (a one-line Stokes
relaxation), not with TensorDEM.  This module supplies the missing piece: the
same settling case solved by the real contact-resolved DEM kernel, plus the
three-field energy ledger and the DEM -> flow momentum/energy exchange hook the
coupling scene needs.

Three-field ledger split (see ``THREE_FIELD_CHANNELS``):

* ``dem_bulk``  : kinetic + gravity potential + elastic contact storage,
* ``contact``   : particle-particle and particle-wall dashpot dissipation,
* ``fluid``     : drag dissipation (momentum handed to the flow leg).

Two validation targets are exposed:

* :func:`relaxation_reference` — exact solution of the linear-drag single
  particle problem, used as an *integrator* benchmark (the DEM trajectory must
  reproduce it to well below 1%),
* :func:`stokes_terminal_velocity` / :func:`oseen_terminal_velocity` — analytic
  drag-law benchmarks for the *physics* (reported together with the measured
  particle Reynolds number and the Oseen correction, since Stokes is only valid
  for Re -> 0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .kernel import DEMKernel, DEMKernelConfig

__all__ = [
    "THREE_FIELD_CHANNELS",
    "SettlingCase",
    "dilute_single_particle_case",
    "dense_bed_case",
    "stokes_terminal_velocity",
    "oseen_terminal_velocity",
    "relaxation_reference",
    "run_settling",
    "three_field_summary",
    "momentum_exchange",
]

THREE_FIELD_CHANNELS = {
    "dem_bulk": ("kinetic_J", "gravity_potential_J", "elastic_J"),
    "contact": ("dem_damping_dissipated_J", "wall_damping_dissipated_J"),
    "fluid": ("drag_dissipated_J",),
}


@dataclass(frozen=True)
class SettlingCase:
    name: str
    config: DEMKernelConfig
    positions: np.ndarray
    velocities: np.ndarray

    @property
    def particle_reynolds_reference(self) -> float:
        cfg = self.config
        if cfg.fluid_viscosity <= 0.0:
            return 0.0
        v = stokes_terminal_velocity(cfg)
        return cfg.fluid_density * 2.0 * cfg.radius * v / cfg.fluid_viscosity


# -- analytic references -----------------------------------------------------
def stokes_terminal_velocity(cfg: DEMKernelConfig) -> float:
    """``v_t = 2 R^2 (rho_p - rho_f) g / (9 mu)`` for a single sphere."""

    if cfg.fluid_viscosity <= 0.0:
        raise ValueError("Stokes reference requires a positive fluid viscosity")
    return (
        2.0
        * cfg.radius**2
        * (cfg.density - cfg.fluid_density)
        * abs(cfg.gravity[2])
        / (9.0 * cfg.fluid_viscosity)
    )


def oseen_terminal_velocity(cfg: DEMKernelConfig, iterations: int = 50) -> float:
    """Terminal velocity with the first-order Oseen correction ``1 + 3/8 Re``."""

    v = stokes_terminal_velocity(cfg)
    diameter = 2.0 * cfg.radius
    for _ in range(iterations):
        re = cfg.fluid_density * diameter * v / cfg.fluid_viscosity
        v = stokes_terminal_velocity(cfg) / (1.0 + 0.375 * re)
    return v


def relaxation_reference(t, tau: float, v_terminal: float):
    """Exact solution of ``dv/dt = g_eff - v/tau`` starting from rest."""

    t = np.asarray(t, dtype=float)
    return v_terminal * (1.0 - np.exp(-t / tau))


def drag_relaxation_time(cfg: DEMKernelConfig) -> float:
    return cfg.mass / (cfg.stokes_coefficient + 1e-300)


# -- case builders -----------------------------------------------------------
def dilute_single_particle_case(
    radius: float = 1.0e-5,
    particle_density: float = 1100.0,
    fluid_density: float = 1000.0,
    fluid_viscosity: float = 1.0e-3,
    dt: float = 1.0e-6,
    box: tuple[float, float, float] = (1.0e-3, 1.0e-3, 1.0e-3),
    young_modulus: float = 1.0e7,
    restitution: float = 0.3,
) -> SettlingCase:
    """Single sphere in a wide box: integrator + drag-law benchmark (Re << 1)."""

    cfg = DEMKernelConfig(
        radius=radius,
        density=particle_density,
        dt=dt,
        young_modulus=young_modulus,
        restitution=restitution,
        gravity=(0.0, 0.0, -9.81),
        fluid_density=fluid_density,
        fluid_viscosity=fluid_viscosity,
        drag_model="stokes",
        box_lo=(0.0, 0.0, 0.0),
        box_hi=box,
    )
    centre = 0.5 * np.asarray(box, dtype=float)
    return SettlingCase(
        name="dilute_single_particle",
        config=cfg,
        positions=centre.reshape(1, 3),
        velocities=np.zeros((1, 3)),
    )


def dense_bed_case(
    n_particles: int = 512,
    radius: float = 5.0e-5,
    volume_fraction: float = 0.15,
    particle_density: float = 2000.0,
    fluid_density: float = 1000.0,
    fluid_viscosity: float = 1.0e-3,
    dt: float = 5.0e-7,
    young_modulus: float = 1.0e7,
    restitution: float = 0.1,
    drag_model: str = "schiller_naumann",
    seed: int = 20261110,
    jitter_fraction: float = 0.05,
) -> SettlingCase:
    """Random-lattice settling bed with a prescribed solid volume fraction.

    The domain is a cube whose side follows from the target solid fraction,
    ``Lz = n V_p / (phi Lx Ly)``; the lateral size is an integer number of mean
    lattice spacings so the lattice is commensurate with the box.
    """

    particle_volume = 4.0 / 3.0 * math.pi * radius**3
    spacing = (particle_volume / volume_fraction) ** (1.0 / 3.0)
    lat = int(math.ceil(n_particles ** (1.0 / 3.0)))
    while lat**3 < n_particles:
        lat += 1
    lx = lat * spacing
    ly = lx
    lz = n_particles * particle_volume / (volume_fraction * lx * ly)

    rng = np.random.default_rng(seed)
    axes = (np.arange(lat) + 0.5) * spacing
    grid = np.stack(np.meshgrid(axes, axes, axes, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = grid[:n_particles]
    pos = grid + rng.uniform(-1.0, 1.0, size=grid.shape) * jitter_fraction * spacing
    # centre the lattice inside the domain and keep one radius of clearance
    pos += np.array([0.5 * (lx - lat * spacing), 0.5 * (ly - lat * spacing), 0.5 * (lz - lat * spacing)])
    lo = np.array([1.05 * radius] * 3)
    hi = np.array([lx, ly, lz]) - 1.05 * radius
    pos = np.clip(pos, lo, hi)

    cfg = DEMKernelConfig(
        radius=radius,
        density=particle_density,
        dt=dt,
        young_modulus=young_modulus,
        restitution=restitution,
        gravity=(0.0, 0.0, -9.81),
        fluid_density=fluid_density,
        fluid_viscosity=fluid_viscosity,
        drag_model=drag_model,
        box_lo=(0.0, 0.0, 0.0),
        box_hi=(lx, ly, lz),
    )
    return SettlingCase(
        name="dense_bed",
        config=cfg,
        positions=pos,
        velocities=np.zeros_like(pos),
    )


# -- run harness -------------------------------------------------------------
def run_settling(
    case: SettlingCase,
    steps: int,
    backend: str = "numpy",
    record_every: int = 0,
) -> dict:
    """Run the case and return trajectory diagnostics + the energy ledger."""

    if record_every <= 0:
        record_every = max(1, steps // 200)
    kernel = DEMKernel(
        case.config,
        case.positions,
        case.velocities,
        backend=backend,
        record_energy_every=record_every,
        name=case.name,
    )
    vz_history = [float(np.mean(kernel.velocities[:, 2]))]
    sample_steps = list(range(record_every, steps + 1, record_every))
    for target in sample_steps:
        kernel.step(target - kernel.step_count)
        vz_history.append(float(np.mean(kernel.velocities[:, 2])))
    kernel.record()
    result = {
        "case": case.name,
        "backend": kernel.be.name,
        "steps": kernel.step_count,
        "time_s": kernel.time,
        "sample_steps": [0] + sample_steps,
        "mean_vz_m_s": vz_history,
        "ledger": kernel.ledger,
        "kernel": kernel,
        "three_field": three_field_summary(kernel.ledger),
        "momentum_exchange": momentum_exchange(kernel),
    }
    if case.name == "dilute_single_particle":
        cfg = case.config
        v = kernel.velocities[0, 2]
        v_t = abs(stokes_terminal_velocity(cfg))
        tau = drag_relaxation_time(cfg)
        exact = float(relaxation_reference(kernel.time, tau, v_t))
        result.update(
            terminal_velocity_measured_m_s=float(-v),
            terminal_velocity_stokes_m_s=v_t,
            terminal_velocity_oseen_m_s=oseen_terminal_velocity(cfg),
            relaxation_time_s=tau,
            exact_at_end_m_s=exact,
            integrator_relative_error=abs(-v - exact) / max(abs(exact), 1e-300),
            stokes_relative_error=abs(abs(v) - v_t) / v_t,
        )
    return result


def three_field_summary(ledger) -> dict:
    """Split the final ledger row into the three coupling-chain fields."""

    last = ledger.latest()
    out = {}
    for field, keys in THREE_FIELD_CHANNELS.items():
        out[field] = {k: last[k] for k in keys}
        out[field]["total_J"] = float(sum(last[k] for k in keys))
    return out


def momentum_exchange(kernel: DEMKernel) -> dict:
    """DEM -> flow coupling hook: mean drag force and power handed to the fluid."""

    cfg = kernel.config
    vel = kernel.velocities
    if cfg.drag_model == "none":
        return {"drag_force_N": np.zeros(3), "drag_power_W": 0.0}
    rel = vel - np.asarray(cfg.fluid_velocity, dtype=float)
    speed = np.sqrt(np.sum(rel * rel, axis=1))
    c = cfg.stokes_coefficient
    if cfg.drag_model == "schiller_naumann":
        re = cfg.fluid_density * 2.0 * cfg.radius * speed / max(cfg.fluid_viscosity, 1e-300)
        c_eff = c * (1.0 + 0.15 * np.power(re, 0.687))
    else:
        c_eff = np.full(kernel.n, c)
    drag_force = -np.sum(c_eff[:, None] * rel, axis=0)
    drag_power = float(-np.sum(c_eff * speed * speed))
    return {"drag_force_N": drag_force, "drag_power_W": drag_power}

def contact_rich_pack_case(
    n_particles: int = 512,
    radius: float = 5.0e-5,
    particle_density: float = 2000.0,
    fluid_density: float = 1000.0,
    fluid_viscosity: float = 1.0e-3,
    dt: float = 5.0e-7,
    young_modulus: float = 1.0e7,
    restitution: float = 0.1,
    drag_model: str = "schiller_naumann",
    spacing_factor: float = 1.9,
    seed: int = 20261110,
) -> SettlingCase:
    """Lattice packing with ``spacing_factor * radius`` spacing (< 2R -> contact).

    Used as the backend-equivalence stress state: every particle starts in
    contact with its six neighbours, so pair forces, dashpots and the ledger
    reductions are all active from step 0.
    """

    particle_volume = 4.0 / 3.0 * math.pi * radius**3
    spacing = spacing_factor * radius
    lat = int(math.ceil(n_particles ** (1.0 / 3.0)))
    while lat**3 < n_particles:
        lat += 1
    lx = lat * spacing
    ly = lx
    lz = lat * spacing
    rng = np.random.default_rng(seed)
    axes = (np.arange(lat) + 0.5) * spacing
    grid = np.stack(np.meshgrid(axes, axes, axes, indexing="ij"), axis=-1).reshape(-1, 3)
    pos = grid[:n_particles] + rng.uniform(-1.0, 1.0, size=(n_particles, 3)) * 0.01 * spacing
    pos = np.clip(pos, 1.05 * radius, np.array([lx, ly, lz]) - 1.05 * radius)
    cfg = DEMKernelConfig(
        radius=radius,
        density=particle_density,
        dt=dt,
        young_modulus=young_modulus,
        restitution=restitution,
        gravity=(0.0, 0.0, -9.81),
        fluid_density=fluid_density,
        fluid_viscosity=fluid_viscosity,
        drag_model=drag_model,
        box_lo=(0.0, 0.0, 0.0),
        box_hi=(lx, ly, lz),
    )
    return SettlingCase(
        name="contact_rich_pack",
        config=cfg,
        positions=pos,
        velocities=np.zeros_like(pos),
    )


def elastic_gas_case(
    n_particles: int = 64,
    radius: float = 1.0e-4,
    particle_density: float = 1000.0,
    dt: float = 5.0e-7,
    young_modulus: float = 1.0e7,
    seed: int = 20261110,
    box: tuple[float, float, float] = (1.6e-3, 1.6e-3, 1.6e-3),
    speed: float = 0.2,
) -> SettlingCase:
    """Frictionless elastic-particle gas: the strict energy-conservation test.

    ``restitution = 1`` (zero dashpot) and no drag/gravity, so the discrete
    mechanical energy ``K + U_c`` must be conserved; the measured drift is the
    integrator's own error, not a model artefact.
    """

    rng = np.random.default_rng(seed)
    lo = np.array([1.05 * radius] * 3)
    hi = np.array(box) - 1.05 * radius
    lat = int(math.ceil(n_particles ** (1.0 / 3.0)))
    while lat**3 < n_particles:
        lat += 1
    axes = [lo[d] + (np.arange(lat) + 0.5) * (hi[d] - lo[d]) / lat for d in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)[:n_particles]
    spacing = min((hi[d] - lo[d]) / lat for d in range(3))
    if spacing <= 2.0 * radius:
        raise ValueError("elastic_gas box too small for a non-overlapping lattice")
    jitter = 0.25 * (spacing - 2.0 * radius)
    pos = grid + rng.uniform(-jitter, jitter, size=grid.shape)
    vel = rng.normal(0.0, speed, size=(n_particles, 3))
    cfg = DEMKernelConfig(
        radius=radius,
        density=particle_density,
        dt=dt,
        young_modulus=young_modulus,
        restitution=1.0,
        gravity=(0.0, 0.0, 0.0),
        fluid_density=0.0,
        fluid_viscosity=0.0,
        drag_model="none",
        box_lo=(0.0, 0.0, 0.0),
        box_hi=box,
    )
    return SettlingCase(
        name="elastic_gas",
        config=cfg,
        positions=pos,
        velocities=vel,
    )
