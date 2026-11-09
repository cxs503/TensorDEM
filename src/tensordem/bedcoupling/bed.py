"""Bed-scale force assembly, static-balance audit and batch-settling dynamics.

This is the *coupling* layer: it takes a packing
(:class:`tensordem.bedcoupling.packing.PackedBed`), a closure
(:mod:`tensordem.bedcoupling.closures`) and builds the force balance that an
unresolved DEM--CFD run would see on every particle.  Three things live here:

``bed_pressure_drop``
    Fixed bed, prescribed superficial velocity: ``dp/dz = sum(F_drag) / V_bed``.
    With the mean voidage this is *identically* the Ergun relation (the tests
    assert it to machine precision); with the per-particle probe voidage it is
    the heterogeneous-bed value, which differs by the voidage-variance bias
    ``~(1/2) Var(eps) d^2 beta/d eps^2``.  That difference is the physically
    informative part of the packed-bed criterion, not the trivial part.

``static_bed_force_audit``
    Fluid at rest: the only fluid force is the hydrostatic pressure gradient,
    ``-V_p grad p = V_p rho_f g`` (buoyancy).  The audit checks that the
    assembled net force equals the effective submerged weight per particle and
    in aggregate, expressed as a residual relative to the total gravity
    ``sum(m_i g)``.  It is a bookkeeping guard: a spurious extra buoyancy term
    or a sign error shows up as an O(1) residual.

``simulate_batch_settling``
    Semi-coupled (single-fluid, uniform-voidage) batch settling in a closed
    periodic cell: the fluid has one degree of freedom, its superficial
    velocity, fixed by zero net volume flux,
    ``u_f = -(1-eps)/eps * <u_p>``.  Each particle feels gravity + buoyancy +
    closure drag, integrated explicitly.  At steady state the slip velocity is
    the hindered-settling slip, so the settling curve ``u_p(eps)`` carries the
    Richardson--Zaki index -- the quantity fitted by :func:`fit_power_law`.

Limitations are stated in the module docstring of the node report; the main
one is that the fluid is a single uniform velocity, so there is no resolved
flow structure (no wakes, no clustering feedback).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .closures import (
    GRAVITY,
    buoyant_weight,
    drag_coefficient,
    drag_force,
    hydrostatic_pressure_gradient,
    pressure_gradient_force,
    sphere_volume,
)
from .packing import PackedBed

__all__ = [
    "FluidProperties",
    "bed_force_assembly",
    "bed_pressure_drop",
    "body_force_per_particle",
    "static_bed_force_audit",
    "relaxation_time",
    "simulate_settling",
    "fit_power_law",
    "settling_sweep",
]


@dataclass(frozen=True)
class FluidProperties:
    """Fluid and particle material data (SI)."""

    rho_fluid: float
    rho_particle: float
    mu: float
    gravity: float = GRAVITY

    @property
    def density_contrast(self) -> float:
        return self.rho_particle - self.rho_fluid


# ---------------------------------------------------------------------------
# fixed bed: force assembly and pressure drop
# ---------------------------------------------------------------------------
def bed_force_assembly(
    bed: PackedBed,
    u_superficial: float,
    fluid: FluidProperties,
    *,
    eps_local: np.ndarray | None = None,
    branch: str = "gidaspow",
    axis: int = 2,
) -> dict:
    """Drag force on every particle of a bed held at rest in a uniform flow.

    ``u_superficial`` is the empty-column velocity along ``axis`` (sign is the
    flow direction).  Returns per-particle forces plus the bed-scale pressure
    drop ``sum(F_drag) / V_bed`` (positive when the flow is driven along
    ``+axis``).
    """
    n = bed.n_particles
    eps = bed.porosity if eps_local is None else np.asarray(eps_local, dtype=float)
    slip = np.zeros((n, 3))
    slip[:, axis] = u_superficial
    force = drag_force(
        slip,
        diameter=bed.diameter,
        porosity=eps,
        rho_fluid=fluid.rho_fluid,
        mu=fluid.mu,
        branch=branch,
    )
    total = force.sum(axis=0)
    return {
        "forces": force,
        "total_force": total,
        "pressure_drop_per_length": total[axis] / bed.bed_volume,
        "porosity_used": eps,
        "branch": branch,
        "u_superficial": u_superficial,
    }


def bed_pressure_drop(
    bed: PackedBed,
    u_superficial: float,
    fluid: FluidProperties,
    *,
    eps_local: np.ndarray | None = None,
    branch: str = "gidaspow",
) -> float:
    """``dp/dz`` of a fixed bed [Pa/m]."""
    return float(
        bed_force_assembly(
            bed, u_superficial, fluid, eps_local=eps_local, branch=branch
        )["pressure_drop_per_length"]
    )


# ---------------------------------------------------------------------------
# static bed: bookkeeping audit
# ---------------------------------------------------------------------------
def static_bed_force_audit(bed: PackedBed, fluid: FluidProperties) -> dict:
    """Net-force residual of a bed at rest, relative to its total weight.

    Buoyancy is *not* added separately: it is the hydrostatic pressure
    gradient force ``-V_p grad p``, so any extra Archimedes term would double
    count it and show up in ``residual_over_total_gravity`` as an O(1) error.
    """
    volumes = bed.volumes
    masses = volumes * fluid.rho_particle
    g_vec = np.array([0.0, 0.0, -fluid.gravity])
    grad_p = hydrostatic_pressure_gradient(fluid.rho_fluid, fluid.gravity)

    f_gravity = masses[:, None] * g_vec
    f_pressure = pressure_gradient_force(volumes, grad_p)
    f_drag = drag_force(
        np.zeros((bed.n_particles, 3)),
        diameter=bed.diameter,
        porosity=bed.porosity,
        rho_fluid=fluid.rho_fluid,
        mu=fluid.mu,
    )
    f_net = f_gravity + f_pressure + f_drag

    expected_particle = volumes[:, None] * fluid.density_contrast * g_vec
    total_gravity = float(np.abs(f_gravity).sum())
    aggregate_residual = float(np.linalg.norm(f_net.sum(axis=0) - expected_particle.sum(axis=0)))
    per_particle = np.linalg.norm(f_net - expected_particle, axis=1)
    weight_per_particle = masses * fluid.gravity
    return {
        "n_particles": bed.n_particles,
        "total_gravity_N": total_gravity,
        "buoyancy_N": float(np.linalg.norm(f_pressure.sum(axis=0))),
        "expected_buoyancy_N": float(fluid.rho_fluid * fluid.gravity * volumes.sum()),
        "aggregate_residual_N": aggregate_residual,
        "residual_over_total_gravity": aggregate_residual / total_gravity,
        "max_per_particle_residual_over_weight": float((per_particle / weight_per_particle).max()),
        "drag_at_rest_N": float(np.abs(f_drag).max()),
        "effective_weight_N": float(np.linalg.norm(expected_particle.sum(axis=0))),
        "effective_weight_analytic_N": buoyant_weight(
            bed.diameter,
            rho_particle=fluid.rho_particle,
            rho_fluid=fluid.rho_fluid,
            gravity=fluid.gravity,
        )
        * bed.n_particles,
    }


# ---------------------------------------------------------------------------
# settling dynamics
# ---------------------------------------------------------------------------
def body_force_per_particle(
    bed: PackedBed, fluid: FluidProperties, *, buoyancy_model: str = "mixture"
) -> np.ndarray:
    """Gravity + pressure-gradient force per particle [N] (z up).

    ``buoyancy_model='hydrostatic'``
        ``grad p = -rho_f g``: the pure-fluid hydrostatic field, i.e. the
        textbook Archimedes buoyancy.  Net body force ``V_p (rho_p - rho_f) g``.
    ``buoyancy_model='mixture'``
        ``grad p = -(eps rho_f + (1-eps) rho_p) g``: the pressure field that
        actually supports a homogeneous suspension (mixture force balance).
        Net body force ``V_p eps (rho_p - rho_f) g``.

    The two differ by exactly one power of the voidage.  The mixture form is
    the one consistent with the Anderson--Jackson (Gidaspow) inter-phase
    momentum coupling: with it, the Wen--Yu branch reproduces the
    Richardson--Zaki Stokes index ``n = 4.65`` exactly, whereas the hydrostatic
    form loses one power of ``eps`` and returns ``n = 3.65``.  That shift is
    the classic one-exponent discrepancy between Ergun-type and
    Richardson--Zaki-type correlations, and it is reported rather than tuned
    away.
    """
    volumes = bed.volumes
    if buoyancy_model == "hydrostatic":
        grad_p = hydrostatic_pressure_gradient(fluid.rho_fluid, fluid.gravity)
    elif buoyancy_model == "mixture":
        eps = bed.porosity
        grad_p = np.array(
            [0.0, 0.0, -(eps * fluid.rho_fluid + (1.0 - eps) * fluid.rho_particle) * fluid.gravity]
        )
    else:
        raise ValueError("buoyancy_model must be 'mixture' or 'hydrostatic'")
    mass = volumes * fluid.rho_particle
    return mass[:, None] * np.array([0.0, 0.0, -fluid.gravity]) + pressure_gradient_force(volumes, grad_p)


def relaxation_time(
    bed: PackedBed, fluid: FluidProperties, *, slip_velocity: float, branch: str = "gidaspow"
) -> float:
    """Drag relaxation time ``tau = m / c(u)`` at a reference slip velocity [s]."""
    coeff = float(
        drag_coefficient(
            slip_velocity,
            diameter=bed.diameter,
            porosity=bed.porosity,
            rho_fluid=fluid.rho_fluid,
            mu=fluid.mu,
            branch=branch,
        )
    )
    mass = fluid.rho_particle * bed.particle_volume
    return mass / coeff


def simulate_settling(
    bed: PackedBed,
    fluid: FluidProperties,
    *,
    n_steps: int = 4000,
    dt: float | None = None,
    dt_safety: float = 0.1,
    fluid_mode: str = "closed_box",
    branch: str = "gidaspow",
    buoyancy_model: str = "mixture",
    seed: int = 3,
    store_every: int = 1,
) -> dict:
    """Settle a particle assembly with the closure drag (explicit Euler).

    ``fluid_mode='closed_box'`` enforces zero net volume flux through the
    periodic cell, ``u_f = -(1-eps)/eps <u_p>`` (batch sedimentation);
    ``fluid_mode='at_rest'`` holds the fluid at rest (unbounded single-particle
    settling).  Particles do not interact with each other, so the run is
    contact-free by construction; :func:`run_contact_audit` verifies that the
    configuration stays contact-free.

    Returns time series, steady-state averages and the drag pressure drop
    ``sum(F_drag,z)/V_bed`` [Pa/m].
    """
    if fluid_mode not in ("closed_box", "at_rest"):
        raise ValueError("fluid_mode must be 'closed_box' or 'at_rest'")
    if buoyancy_model not in ("mixture", "hydrostatic"):
        raise ValueError("buoyancy_model must be 'mixture' or 'hydrostatic'")

    n = bed.n_particles
    vp = bed.particle_volume
    mass = fluid.rho_particle * vp
    eps = bed.porosity
    volumes = bed.volumes
    g_vec = np.array([0.0, 0.0, -fluid.gravity])
    f_body = body_force_per_particle(bed, fluid, buoyancy_model=buoyancy_model)

    pos = np.array(bed.positions, dtype=float)
    rng = np.random.default_rng(seed)
    vel = rng.normal(0.0, 1e-4, size=(n, 3))

    if dt is None:
        slip_ref = float(
            np.abs(
                fluid.density_contrast
                * fluid.gravity
                * bed.diameter**2
                / (18.0 * fluid.mu)
            )
            * max(eps, 1e-6) ** -2.65
        )
        tau = relaxation_time(bed, fluid, slip_velocity=max(slip_ref, 1e-9), branch=branch)
        dt = dt_safety * tau
    else:
        tau = relaxation_time(
            bed,
            fluid,
            slip_velocity=max(fluid.density_contrast * fluid.gravity * bed.diameter**2 / (18.0 * fluid.mu), 1e-9),
            branch=branch,
        )

    time = np.empty(n_steps // store_every + 1)
    mean_vz = np.empty_like(time)
    dpdz = np.empty_like(time)
    fluid_velocity = np.empty_like(time)

    def record(k: int) -> None:
        time[k] = _step_index * dt
        mean_vz[k] = vel[:, 2].mean()
        if fluid_mode == "closed_box":
            u_f = -(1.0 - eps) / eps * vel[:, 2].mean()
        else:
            u_f = 0.0
        fluid_velocity[k] = u_f
        slip = np.zeros((n, 3))
        slip[:, 2] = u_f - vel[:, 2]
        f_drag = drag_force(
            slip,
            diameter=bed.diameter,
            porosity=eps,
            rho_fluid=fluid.rho_fluid,
            mu=fluid.mu,
            branch=branch,
        )
        dpdz[k] = f_drag[:, 2].sum() / bed.bed_volume

    _step_index = 0
    record(0)
    k = 1
    for step in range(1, n_steps + 1):
        _step_index = step
        if fluid_mode == "closed_box":
            u_f = -(1.0 - eps) / eps * vel[:, 2].mean()
        else:
            u_f = 0.0
        slip = np.zeros((n, 3))
        slip[:, 2] = u_f - vel[:, 2]
        f_drag = drag_force(
            slip,
            diameter=bed.diameter,
            porosity=eps,
            rho_fluid=fluid.rho_fluid,
            mu=fluid.mu,
            branch=branch,
        )
        vel += dt * (f_body + f_drag) / mass
        pos = (pos + vel * dt) % bed.box_length
        if step % store_every == 0 and k < time.size:
            record(k)
            k += 1

    n_kept = k
    time, mean_vz, dpdz, fluid_velocity = (
        time[:n_kept],
        mean_vz[:n_kept],
        dpdz[:n_kept],
        fluid_velocity[:n_kept],
    )
    window = max(1, int(0.3 * n_kept))
    steady = slice(n_kept - window, n_kept)
    u_steady = float(mean_vz[steady].mean())
    return {
        "dt": dt,
        "n_steps": n_steps,
        "tau": tau,
        "dt_over_tau": dt / tau,
        "porosity": eps,
        "fluid_mode": fluid_mode,
        "buoyancy_model": buoyancy_model,
        "branch": branch,
        "time": time,
        "mean_vz": mean_vz,
        "dpdz": dpdz,
        "fluid_velocity": fluid_velocity,
        "steady_mean_vz": u_steady,
        "steady_std_vz": float(mean_vz[steady].std()),
        "steady_mean_dpdz": float(dpdz[steady].mean()),
        "steady_std_dpdz": float(dpdz[steady].std()),
        "steady_window_steps": int(window),
        "positions": pos,
        "velocities": vel,
    }


def fit_power_law(x: np.ndarray, y: np.ndarray) -> dict:
    """Least-squares fit of ``y = c x^n`` in log-log space."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("log-log fit needs positive data")
    lx, ly = np.log(x), np.log(y)
    n, intercept = np.polyfit(lx, ly, 1)
    pred = intercept + n * lx
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    return {
        "exponent": float(n),
        "prefactor": float(math.exp(intercept)),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        "max_abs_log_residual": float(np.abs(ly - pred).max()),
        "x": x.tolist(),
        "y": y.tolist(),
    }


def settling_sweep(
    packings: dict[float, PackedBed],
    fluid: FluidProperties,
    *,
    n_steps: int = 4000,
    dt: float | None = None,
    fluid_mode: str = "closed_box",
    branch: str = "gidaspow",
    buoyancy_model: str = "mixture",
    dt_safety: float = 0.1,
) -> dict:
    """Run the settling simulation over a set of packings and fit ``u_p(eps)``.

    In a closed cell the particle (interface) velocity obeys
    ``u_p = u_t eps^n`` with the Richardson--Zaki index ``n``; the fit returns
    that index from the simulation, not from the correlation.
    """
    eps_values = sorted(packings)
    results = []
    for eps in eps_values:
        run = simulate_settling(
            packings[eps],
            fluid,
            n_steps=n_steps,
            dt=dt,
            dt_safety=dt_safety,
            fluid_mode=fluid_mode,
            branch=branch,
            buoyancy_model=buoyancy_model,
        )
        results.append(
            {
                "porosity": eps,
                "u_particle": -run["steady_mean_vz"],  # settling speed, positive down
                "u_particle_std": run["steady_std_vz"],
                "dpdz": run["steady_mean_dpdz"],
                "dt": run["dt_over_tau"],
            }
        )
        if dt is None:
            dt = run["dt"]  # reuse the first stable step for the whole sweep
    fit = fit_power_law(
        np.array([r["porosity"] for r in results]), np.array([r["u_particle"] for r in results])
    )
    return {
        "runs": results,
        "fit": fit,
        "fluid_mode": fluid_mode,
        "branch": branch,
        "buoyancy_model": buoyancy_model,
    }
