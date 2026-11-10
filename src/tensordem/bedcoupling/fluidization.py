"""Fluidisation onset, per-step three-force ledger and resolution convergence.

Complement to :mod:`tensordem.bedcoupling.bed`: that module takes the closures
through a *static* bed audit and a batch-settling sweep.  This module adds the
three things a dense particle--fluid *minimum verifiable unit* still needs:

``wen_yu_minimum_fluidization`` / ``ergun_minimum_fluidization`` /
``simulated_minimum_fluidization``
    The onset of fluidisation: the superficial velocity at which the fluid
    pressure gradient just supports the bed, ``dp/dz = (1-eps)(rho_p-rho_f) g``.
    Two analytic yardsticks (Wen--Yu 1966 Ar-correlation, Ergun quadratic) and
    the DEM closure's own root provide three independent numbers for the same
    physical quantity; the node compares them with each other.

``column_dynamics``
    Velocity-Verlet integration of a monodisperse point-particle column driven
    by the same closures, recording **every step** the three force sums
    (drag, pressure gradient, gravity), the applied-force subset, the fluid
    velocity and the total particle momentum.  That record is the *ledger*.

``ledger_residuals``
    Two residuals on that record:

    * ``bookkeeping`` -- ``|(p_{n+1}-p_n)/dt - ½(A_n + A_{n+½})| / W_bed``,
      which is zero to round-off when the ledger records exactly the force
      subset that was applied and grows to O(1) as soon as a term that is
      actually acting is missing from the record (or vice versa);
    * ``equilibrium`` -- ``|F_drag+F_pressure+F_gravity| / W_bed`` in the steady
      window, i.e. the physical force balance of a particle column that has
      reached its terminal state.

Both are relative to the bed weight ``W_bed = sum(m g)``, so "1 %" means 1 % of
gravity -- a scale, not a tunable.

Conventions follow :mod:`tensordem.bedcoupling.closures`: SI units, ``+z`` up,
forces are forces **on the particle**, and the pressure-gradient force is the
buoyancy force (never added twice).

Nothing in here knows about ships or ice: it is generic granular--fluid physics.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .bed import FluidProperties, body_force_per_particle, relaxation_time
from .closures import GRAVITY, drag_force
from .packing import PackedBed, local_porosity, make_random_close_packing

__all__ = [
    "WEN_YU_COEFFICIENTS",
    "archimedes_number",
    "bed_weight_gradient",
    "ergun_minimum_fluidization",
    "wen_yu_minimum_fluidization",
    "simulated_minimum_fluidization",
    "column_dynamics",
    "ledger_residuals",
    "resolution_convergence",
]

#: Wen & Yu (1966) onset-of-fluidisation coefficients, ``Ar = c1 Re_mf + c2 Re_mf^2``.
#: They are the empirical pair from which the widely quoted
#: ``Re_mf = sqrt(33.7^2 + 0.0408 Ar) - 33.7`` follows; both forms are returned.
WEN_YU_COEFFICIENTS = {"laminar": 1650.0, "inertial": 24.5}


# ---------------------------------------------------------------------------
# onset of fluidisation
# ---------------------------------------------------------------------------
def bed_weight_gradient(
    porosity: float,
    *,
    rho_particle: float,
    rho_fluid: float,
    gravity: float = GRAVITY,
) -> float:
    """Pressure gradient that just supports the bed, ``(1-eps)(rho_p-rho_f) g`` [Pa/m].

    Obtained by dividing the submerged bed weight per unit volume by the bed
    height: ``W/(A L) = (1-eps)(rho_p-rho_f) g``.  Buoyancy is already inside
    the density contrast, so the fluid's own weight must not be subtracted
    again.
    """
    eps = float(porosity)
    if not 0.0 < eps < 1.0:
        raise ValueError(f"porosity must be in (0, 1), got {eps}")
    return (1.0 - eps) * (rho_particle - rho_fluid) * gravity


def archimedes_number(
    diameter: float,
    *,
    rho_particle: float,
    rho_fluid: float,
    mu: float,
    gravity: float = GRAVITY,
) -> float:
    """Archimedes number ``Ar = rho_f d^3 (rho_p-rho_f) g / mu^2``."""
    if diameter <= 0.0 or mu <= 0.0:
        raise ValueError("diameter and mu must be positive")
    return rho_fluid * diameter**3 * (rho_particle - rho_fluid) * gravity / mu**2


def wen_yu_minimum_fluidization(
    diameter: float,
    *,
    rho_particle: float,
    rho_fluid: float,
    mu: float,
    gravity: float = GRAVITY,
    laminar_coefficient: float = WEN_YU_COEFFICIENTS["laminar"],
    inertial_coefficient: float = WEN_YU_COEFFICIENTS["inertial"],
) -> dict:
    """Wen & Yu (1966) onset velocity from the Archimedes number.

    Solves ``Ar = c1 Re_mf + c2 Re_mf^2`` for the positive root and converts to
    a superficial velocity ``U_mf = Re_mf mu / (rho_f d)``.  The correlations
    ``Re_mf = sqrt(33.7^2 + 0.0408 Ar) - 33.7`` and the coefficient pair
    ``(1650, 24.5)`` are the same equation, so both are reported.  The laminar
    and inertial asymptotes are returned to show which term dominates.
    """
    ar = archimedes_number(
        diameter, rho_particle=rho_particle, rho_fluid=rho_fluid, mu=mu, gravity=gravity
    )
    a, b = float(laminar_coefficient), float(inertial_coefficient)
    re_mf = (-a + math.sqrt(a * a + 4.0 * b * ar)) / (2.0 * b)
    re_from_table_form = math.sqrt(33.7**2 + 0.0408 * ar) - 33.7
    u_mf = re_mf * mu / (rho_fluid * diameter)
    return {
        "ar": ar,
        "re_mf": re_mf,
        "re_mf_published_form": re_from_table_form,
        "re_form_mismatch": re_from_table_form / re_mf - 1.0,
        "u_mf": u_mf,
        "u_mf_laminar_limit": ar / a * mu / (rho_fluid * diameter),
        "u_mf_newton_limit": math.sqrt(ar / b) * mu / (rho_fluid * diameter),
        "coefficients": {"laminar": a, "inertial": b},
    }


def ergun_minimum_fluidization(
    porosity: float,
    *,
    diameter: float,
    rho_particle: float,
    rho_fluid: float,
    mu: float,
    gravity: float = GRAVITY,
) -> dict:
    """Onset velocity implied by the Ergun (1952) relation at fixed ``eps``.

    Closed-form root of ``A U + B U^2 = (1-eps)(rho_p-rho_f) g`` with
    ``A = 150 mu (1-eps)^2/(eps^3 d^2)`` and
    ``B = 1.75 rho_f (1-eps)/(eps^3 d)``, i.e. the same quadratic that the
    per-particle Ergun drag of :mod:`tensordem.bedcoupling.closures` integrates
    to -- so this is the *analytic* partner of
    :func:`simulated_minimum_fluidization`.
    """
    eps = float(porosity)
    if not 0.0 < eps < 1.0:
        raise ValueError(f"porosity must be in (0, 1), got {eps}")
    target = bed_weight_gradient(
        eps, rho_particle=rho_particle, rho_fluid=rho_fluid, gravity=gravity
    )
    a = 150.0 * mu * (1.0 - eps) ** 2 / (eps**3 * diameter**2)
    b = 1.75 * rho_fluid * (1.0 - eps) / (eps**3 * diameter)
    u_mf = (-a + math.sqrt(a * a + 4.0 * b * target)) / (2.0 * b)
    viscous = a * u_mf
    inertial = b * u_mf * u_mf
    return {
        "u_mf": u_mf,
        "re_mf": rho_fluid * u_mf * diameter / mu,
        "weight_gradient": target,
        "viscous_part": viscous,
        "inertial_part": inertial,
        "inertial_fraction": inertial / target,
        "u_mf_laminar_limit": target / a,
        "u_mf_newton_limit": math.sqrt(target / b),
        "ergun_coefficients": {"A": a, "B": b},
    }


def simulated_minimum_fluidization(
    bed: PackedBed,
    fluid: FluidProperties,
    *,
    eps_local: np.ndarray | None = None,
    branch: str = "gidaspow",
    u_lo: float | None = None,
    u_hi: float | None = None,
    description: str = "closure",
    max_iter: int = 200,
) -> dict:
    """Onset velocity found from the *drag closure itself*.

    The DEM closure gives ``dp/dz(U)`` for a bed held at rest; the onset is the
    ``U`` where that equals the bed weight gradient.  Because ``dp/dz`` is
    monotone in ``U`` (a positive viscous slope plus a positive inertial
    curvature) a bracketing bisection converges unconditionally.

    With ``eps_local`` given, the per-particle voidage field enters the drag
    assembly, so the returned onset carries the heterogeneity of the packing --
    that is what makes it differ from :func:`ergun_minimum_fluidization`, which
    assumes a uniform ``eps``.
    """
    from .bed import bed_pressure_drop

    target = bed_weight_gradient(
        bed.porosity, rho_particle=fluid.rho_particle, rho_fluid=fluid.rho_fluid, gravity=fluid.gravity
    )

    def gap(u: float) -> float:
        return float(bed_pressure_drop(bed, u, fluid, eps_local=eps_local, branch=branch)) - target

    if u_lo is None or u_hi is None:
        # start from the analytic Ergun estimate and bracket around it
        guess = ergun_minimum_fluidization(
            bed.porosity,
            diameter=bed.diameter,
            rho_particle=fluid.rho_particle,
            rho_fluid=fluid.rho_fluid,
            mu=fluid.mu,
            gravity=fluid.gravity,
        )["u_mf"]
        u_lo = guess * 1e-3
        u_hi = guess * 3.0
        for _ in range(200):
            if gap(u_lo) < 0.0 < gap(u_hi):
                break
            if gap(u_lo) >= 0.0:
                u_lo *= 0.1
            if gap(u_hi) <= 0.0:
                u_hi *= 2.0
    else:
        if not (gap(u_lo) < 0.0 < gap(u_hi)):
            raise ValueError("u_lo/u_hi do not bracket the onset velocity")

    iterations = 0
    while iterations < max_iter:
        mid = 0.5 * (u_lo + u_hi)
        if gap(mid) < 0.0:
            u_lo = mid
        else:
            u_hi = mid
        iterations += 1
        if u_hi - u_lo <= 1e-13 * max(u_hi, 1e-30):
            break
    u_mf = 0.5 * (u_lo + u_hi)
    dpdz = float(bed_pressure_drop(bed, u_mf, fluid, eps_local=eps_local, branch=branch))
    return {
        "description": description,
        "u_mf": u_mf,
        "dpdz_at_onset": dpdz,
        "weight_gradient": target,
        "re_mf": fluid.rho_fluid * u_mf * bed.diameter / fluid.mu,
        "iterations": iterations,
        "bracket": [u_lo, u_hi],
        "porosity_used": "local" if eps_local is not None else "mean",
        "branch": branch,
        "max_local_voidage_gap": float(np.abs(dpdz - target) / target),
    }


# ---------------------------------------------------------------------------
# three-force ledger: per-step momentum accounting of an integrated column
# ---------------------------------------------------------------------------
_LEDGER_TERMS = ("drag", "pressure", "gravity")


def _column_forces(
    velocities: np.ndarray,
    *,
    bed: PackedBed,
    fluid: FluidProperties,
    porosity: np.ndarray,
    u_fluid: float,
    branch: str,
    buoyancy_model: str,
) -> dict[str, np.ndarray]:
    """Per-particle drag / pressure-gradient / gravity forces at one state.

    ``body_force_per_particle`` already contains gravity plus the
    pressure-gradient (buoyancy) force; it is split back into its two ledger
    entries here so that each of the three terms can be reported separately
    (and, for the negative control, omitted separately).
    """
    slip = np.zeros_like(velocities)
    slip[:, 2] = u_fluid - velocities[:, 2]
    f_drag = drag_force(
        slip,
        diameter=bed.diameter,
        porosity=porosity,
        rho_fluid=fluid.rho_fluid,
        mu=fluid.mu,
        branch=branch,
    )
    masses = bed.volumes * fluid.rho_particle
    f_gravity = masses[:, np.newaxis] * np.array([0.0, 0.0, -fluid.gravity])
    body = body_force_per_particle(bed, fluid, buoyancy_model=buoyancy_model)
    f_pressure = body - f_gravity
    return {"drag": f_drag, "pressure": f_pressure, "gravity": f_gravity}


def column_dynamics(
    bed: PackedBed,
    fluid: FluidProperties,
    *,
    u_superficial: float = 0.0,
    n_steps: int = 2000,
    dt: float | None = None,
    dt_safety: float = 0.1,
    fluid_mode: str = "closed_box",
    branch: str = "gidaspow",
    buoyancy_model: str = "mixture",
    eps_local: np.ndarray | None = None,
    applied_terms: Sequence[str] = _LEDGER_TERMS,
    seed: int = 11,
) -> dict:
    """Velocity-Verlet column with a per-step three-force ledger.

    The fluid is one-way (prescribed): either a fixed superficial velocity
    (``fluid_mode='imposed'``, ``u_fluid = u_superficial/eps``) or the
    volume-conserving back-flow of a closed periodic cell
    (``fluid_mode='closed_box'``, ``u_fluid = -(1-eps)/eps * <v_p>``).  No
    contacts are evaluated: the packings used here are contact-free, which is
    checked by the caller through ``bed.minimum_separation()``.

    ``applied_terms`` selects which of the three forces actually *push* the
    particles, while the ledger always records all three.  That asymmetry is the
    negative-control hook of :func:`ledger_residuals`: drop ``'drag'`` and the
    bookkeeping residual must jump to O(1).

    Returns the time series of the three force sums (N), the applied sum (N),
    the mean slip velocity, the fluid velocity and the total particle momentum
    (kg m/s), plus the per-step residuals of :func:`ledger_residuals`.
    """
    for term in applied_terms:
        if term not in _LEDGER_TERMS:
            raise ValueError(f"unknown applied term {term!r}")
    if fluid_mode not in ("imposed", "closed_box"):
        raise ValueError("fluid_mode must be 'imposed' or 'closed_box'")

    n = bed.n_particles
    eps_mean = bed.porosity
    porosity = eps_mean if eps_local is None else np.asarray(eps_local, dtype=float)
    masses = bed.volumes * fluid.rho_particle
    weight_total = float(masses.sum() * fluid.gravity)

    if dt is None:
        slip_ref = abs(fluid.density_contrast * fluid.gravity * bed.diameter**2 / (18.0 * fluid.mu)) * max(
            eps_mean, 1e-6
        ) ** -2.65
        tau = relaxation_time(bed, fluid, slip_velocity=max(slip_ref, 1e-12), branch=branch)
        dt = dt_safety * tau
    else:
        tau = relaxation_time(
            bed,
            fluid,
            slip_velocity=max(
                abs(fluid.density_contrast * fluid.gravity * bed.diameter**2 / (18.0 * fluid.mu)), 1e-12
            ),
            branch=branch,
        )

    # the body force (gravity + pressure gradient) of this column does not
    # depend on the state, so it is evaluated once and split into its two
    # ledger entries; the drag term is recomputed from the state every step.
    masses = bed.volumes * fluid.rho_particle
    f_gravity_per_particle = masses[:, np.newaxis] * np.array([0.0, 0.0, -fluid.gravity])
    f_pressure_per_particle = (
        body_force_per_particle(bed, fluid, buoyancy_model=buoyancy_model) - f_gravity_per_particle
    )

    rng = np.random.default_rng(seed)
    vel = rng.normal(0.0, 1e-4, size=(n, 3))
    pos = np.array(bed.positions, dtype=float)

    time = np.empty(n_steps + 1)
    f_drag_hist = np.empty_like(time)
    f_pressure_hist = np.empty_like(time)
    f_gravity_hist = np.empty_like(time)
    applied_hist = np.empty_like(time)
    momentum_hist = np.empty_like(time)
    mean_vz_hist = np.empty_like(time)
    u_fluid_hist = np.empty_like(time)
    f_drag_mid = np.empty_like(time)
    f_pressure_mid = np.empty_like(time)
    f_gravity_mid = np.empty_like(time)
    mid_applied = np.empty_like(time)  # applied force at t_{n+1/2} for the trapezoid identity

    def fluid_velocity(vz_mean: float) -> float:
        if fluid_mode == "closed_box":
            return -(1.0 - eps_mean) / eps_mean * vz_mean
        return u_superficial / max(eps_mean, 1e-12)

    def ledger(v: np.ndarray, u_f: float) -> tuple[dict[str, np.ndarray], float]:
        forces = _column_forces(
            v,
            bed=bed,
            fluid=fluid,
            porosity=porosity,
            u_fluid=u_f,
            branch=branch,
            buoyancy_model=buoyancy_model,
        )
        applied = sum((forces[t] for t in applied_terms), np.zeros((n, 3)))
        return forces, float(applied[:, 2].sum())

    def applied_force(forces: dict[str, np.ndarray]) -> np.ndarray:
        return sum((forces[t] for t in applied_terms), np.zeros((n, 3)))

    def forces_at(v: np.ndarray, u_f: float) -> dict[str, np.ndarray]:
        slip = np.zeros_like(v)
        slip[:, 2] = u_f - v[:, 2]
        return {
            "drag": drag_force(
                slip,
                diameter=bed.diameter,
                porosity=porosity,
                rho_fluid=fluid.rho_fluid,
                mu=fluid.mu,
                branch=branch,
            ),
            "pressure": f_pressure_per_particle,
            "gravity": f_gravity_per_particle,
        }

    forces_0 = forces_at(vel, fluid_velocity(float(vel[:, 2].mean())))
    time[0] = 0.0
    f_drag_hist[0] = forces_0["drag"][:, 2].sum()
    f_pressure_hist[0] = forces_0["pressure"][:, 2].sum()
    f_gravity_hist[0] = forces_0["gravity"][:, 2].sum()
    applied_hist[0] = applied_force(forces_0)[:, 2].sum()
    momentum_hist[0] = float((masses * vel[:, 2]).sum())
    mean_vz_hist[0] = float(vel[:, 2].mean())
    u_fluid_hist[0] = fluid_velocity(float(vel[:, 2].mean()))
    f_drag_mid[0] = f_drag_hist[0]
    f_pressure_mid[0] = f_pressure_hist[0]
    f_gravity_mid[0] = f_gravity_hist[0]
    mid_applied[0] = applied_hist[0]

    for step in range(1, n_steps + 1):
        # kick-drift-kick (velocity Verlet).  The drift is harmless here
        # because the closures depend on velocity only, but keeping it makes
        # the state record honest.
        u_f = fluid_velocity(float(vel[:, 2].mean()))
        forces_n = forces_at(vel, u_f)
        vel_half = vel + 0.5 * dt * applied_force(forces_n) / masses[:, None]
        pos = (pos + dt * vel_half) % bed.box_length

        u_f_half = fluid_velocity(float(vel_half[:, 2].mean()))
        forces_h = forces_at(vel_half, u_f_half)
        vel = vel_half + 0.5 * dt * applied_force(forces_h) / masses[:, None]

        u_f_end = fluid_velocity(float(vel[:, 2].mean()))
        forces_end = forces_at(vel, u_f_end)
        time[step] = step * dt
        f_drag_hist[step] = forces_end["drag"][:, 2].sum()
        f_pressure_hist[step] = forces_end["pressure"][:, 2].sum()
        f_gravity_hist[step] = forces_end["gravity"][:, 2].sum()
        applied_hist[step] = applied_force(forces_end)[:, 2].sum()
        f_drag_mid[step] = forces_h["drag"][:, 2].sum()
        f_pressure_mid[step] = forces_h["pressure"][:, 2].sum()
        f_gravity_mid[step] = forces_h["gravity"][:, 2].sum()
        mid_applied[step] = applied_force(forces_h)[:, 2].sum()
        momentum_hist[step] = float((masses * vel[:, 2]).sum())
        mean_vz_hist[step] = float(vel[:, 2].mean())
        u_fluid_hist[step] = u_f_end

    out = {
        "dt": dt,
        "tau": tau,
        "dt_over_tau": dt / tau,
        "n_steps": n_steps,
        "n_particles": n,
        "porosity": eps_mean,
        "fluid_mode": fluid_mode,
        "branch": branch,
        "buoyancy_model": buoyancy_model,
        "applied_terms": tuple(applied_terms),
        "u_superficial": u_superficial,
        "weight_total": weight_total,
        "time": time,
        "f_drag": f_drag_hist,
        "f_pressure": f_pressure_hist,
        "f_gravity": f_gravity_hist,
        "f_applied": applied_hist,
        "f_drag_mid": f_drag_mid,
        "f_pressure_mid": f_pressure_mid,
        "f_gravity_mid": f_gravity_mid,
        "f_applied_mid": mid_applied,
        "momentum": momentum_hist,
        "mean_vz": mean_vz_hist,
        "u_fluid": u_fluid_hist,
        "final_velocities": vel,
        "finite": bool(np.all(np.isfinite(vel)) and np.all(np.isfinite(momentum_hist))),
    }
    out.update(ledger_residuals(out))
    return out


def ledger_residuals(run: dict, *, steady_fraction: float = 0.25) -> dict:
    """Residuals of the three-force ledger of a :func:`column_dynamics` run.

    ``bookkeeping``
        ``max_n |(p_{n+1}-p_n)/dt - ½(L_n + L_{n+½})| / W_bed`` with
        ``L = F_drag + F_pressure + F_gravity`` the *recorded* ledger.  Velocity
        Verlet satisfies ``p_{n+1}-p_n = dt/2 (A_n + A_{n+½})`` identically for
        the *applied* force ``A``, so this residual is round-off sized **iff**
        the recorded ledger reproduces exactly the forces that acted.  It is a
        completeness check of the ledger, not a physics check -- the physics
        check is ``equilibrium``.
    ``equilibrium``
        ``|L| / W_bed`` averaged over the steady window of the *last*
        ``steady_fraction`` of the steps: at terminal state the three forces
        must cancel.
    ``discretisation``
        ``max_n |L_n - (p_{n+1}-p_{n-1})/(2 dt)| / W_bed``: how far the
        instantaneous force is from the centred momentum rate.  This one does
        depend on ``dt`` and is expected to shrink when the step is refined
        (negative control N4).
    """
    time = np.asarray(run["time"], dtype=float)
    momentum = np.asarray(run["momentum"], dtype=float)
    ledger = (
        np.asarray(run["f_drag"], dtype=float)
        + np.asarray(run["f_pressure"], dtype=float)
        + np.asarray(run["f_gravity"], dtype=float)
    )
    weight = float(run["weight_total"])
    dt = float(run["dt"])
    n_steps = momentum.size - 1

    ledger_mid = (
        np.asarray(run.get("f_drag_mid", run["f_drag"]), dtype=float)
        + np.asarray(run.get("f_pressure_mid", run["f_pressure"]), dtype=float)
        + np.asarray(run.get("f_gravity_mid", run["f_gravity"]), dtype=float)
    )

    # bookkeeping: the velocity-Verlet trapezoid p_{n+1}-p_n = dt/2 (L_n + L_{n+1/2})
    # against the recorded three-force ledger.  Round-off sized iff the ledger
    # contains exactly the forces that were applied.
    identity = 0.5 * (ledger[:-1] + ledger_mid[1:])
    momentum_rate = np.diff(momentum) / dt
    book = np.abs(momentum_rate - identity) / weight

    # applied force vs ledger at the same state points (a pure record check)
    applied_mid = np.asarray(run.get("f_applied_mid", run["f_applied"]), dtype=float)
    applied_vs_ledger = np.abs(
        0.5 * ((np.asarray(run["f_applied"], dtype=float)[:-1] - ledger[:-1])
               + (applied_mid[1:] - ledger_mid[1:]))
    ) / weight

    # instantaneous force vs centred momentum rate: the dt-dependent residual
    if momentum.size >= 3:
        centred = (momentum[2:] - momentum[:-2]) / (2.0 * dt)
        discrete = np.abs(ledger[1:-1] - centred) / weight
    else:  # pragma: no cover - tiny runs
        discrete = np.zeros(1)

    steady_start = max(0, int((1.0 - steady_fraction) * (n_steps + 1)))
    steady_slice = slice(steady_start, None)
    steady_mean_ledger = float(ledger[steady_slice].mean())
    steady_std_ledger = float(ledger[steady_slice].std())
    return {
        "bookkeeping_residual": float(book.max()),
        "bookkeeping_residual_median": float(np.median(book)),
        "discretisation_residual": float(discrete.max()),
        "equilibrium_residual": abs(steady_mean_ledger) / weight,
        "equilibrium_residual_fluctuation": steady_std_ledger / weight,
        "applied_minus_ledger_residual": float(applied_vs_ledger.max()),
        "steady_mean_ledger_N": steady_mean_ledger,
        "steady_mean_drag_N": float(np.asarray(run["f_drag"])[steady_slice].mean()),
        "steady_mean_pressure_N": float(np.asarray(run["f_pressure"])[steady_slice].mean()),
        "steady_mean_gravity_N": float(np.asarray(run["f_gravity"])[steady_slice].mean()),
        "steady_mean_vz": float(np.asarray(run["mean_vz"])[steady_slice].mean()),
        "steady_std_vz": float(np.asarray(run["mean_vz"])[steady_slice].std()),
        "steady_window_start_time": float(time[steady_start]),
        "weight_total_N": weight,
    }


# ---------------------------------------------------------------------------
# resolution convergence
# ---------------------------------------------------------------------------
def resolution_convergence(
    fluid: FluidProperties,
    *,
    particle_counts: Iterable[int] = (600, 1200, 2400),
    diameter: float = 1.0e-3,
    target_porosity: float = 0.38,
    probe_radius_over_d: float = 1.5,
    sweep_velocities: Iterable[float] = (2.0e-3, 5.0e-2, 5.0e-1),
    settling_n_side: Iterable[int] = (8, 10, 13),
    settling_porosity: float = 0.80,
    settling_steps: int = 1200,
    packing_seed: int = 20261110,
    dynamics_seed: int = 11,
) -> dict:
    """Same physics at three resolutions: ``dp/dz``, ``U_mf`` and ``u_settle``.

    Resolutions are three random close packings with ``N = 600/1200/2400``
    particles at the *same* target porosity and the *same* local-voidage probe
    radius (``1.5 d``), plus three jittered cubic arrays (``n_side =
    8/10/13``) for the settling cell.  The spread of each quantity across the
    three resolutions is ``(max - min) / mean``: the finite-cell uncertainty of
    the point-particle closure, which is what a CFD--DEM user needs to know.
    """
    from .bed import bed_pressure_drop
    from .packing import make_jittered_cubic_array

    u_list = [float(u) for u in sweep_velocities]
    rows = []
    for n in particle_counts:
        bed = make_random_close_packing(n, diameter, target_porosity, seed=packing_seed)
        probe = probe_radius_over_d * diameter
        if probe + diameter >= 0.5 * bed.box_length:
            raise ValueError(
                f"probe radius {probe:.3g} too large for a {n}-particle cell "
                f"(box {bed.box_length:.3g})"
            )
        eps_local = local_porosity(bed.positions, bed.diameter, bed.box_length, probe)
        dpdz = [
            float(bed_pressure_drop(bed, u, fluid, eps_local=eps_local)) for u in u_list
        ]
        dpdz_mean_voidage = [float(bed_pressure_drop(bed, u, fluid)) for u in u_list]
        onset = simulated_minimum_fluidization(
            bed, fluid, eps_local=eps_local, description=f"N={n} local voidage"
        )
        rows.append(
            {
                "n_particles": n,
                "box_length_over_d": bed.box_length / diameter,
                "porosity_volume_balance": bed.porosity,
                "porosity_local_mean": float(eps_local.mean()),
                "porosity_local_std": float(eps_local.std()),
                "max_overlap_over_d": bed.max_overlap / diameter,
                "dpdz_local_voidage": dpdz,
                "dpdz_mean_voidage": dpdz_mean_voidage,
                "u_mf_simulated": onset["u_mf"],
                "u_mf_re": onset["re_mf"],
                "probe_radius_over_d": probe_radius_over_d,
            }
        )

    settling_rows = []
    for n_side in settling_n_side:
        cell = make_jittered_cubic_array(n_side, diameter, porosity=settling_porosity, seed=7)
        run = column_dynamics(
            cell,
            fluid,
            n_steps=settling_steps,
            fluid_mode="closed_box",
            buoyancy_model="mixture",
            seed=dynamics_seed,
        )
        settling_rows.append(
            {
                "n_side": n_side,
                "n_particles": cell.n_particles,
                "porosity": cell.porosity,
                "u_settle": -run["steady_mean_vz"],
                "u_settle_std": run["steady_std_vz"],
                "dt_over_tau": run["dt_over_tau"],
                "equilibrium_residual": run["equilibrium_residual"],
            }
        )

    def spread(values: Sequence[float]) -> dict:
        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean())
        return {
            "values": [float(v) for v in arr],
            "mean": mean,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "spread": float((arr.max() - arr.min()) / abs(mean)) if mean != 0.0 else float("inf"),
        }

    quantities = {}
    for idx, u in enumerate(u_list):
        re = fluid.rho_fluid * u * diameter / fluid.mu
        quantities[f"dpdz_local_u{u:.3g}"] = {**spread([r["dpdz_local_voidage"][idx] for r in rows]), "Re": re}
        quantities[f"dpdz_mean_u{u:.3g}"] = {**spread([r["dpdz_mean_voidage"][idx] for r in rows]), "Re": re}
    quantities["u_mf_simulated"] = spread([r["u_mf_simulated"] for r in rows])
    quantities["u_settle"] = spread([r["u_settle"] for r in settling_rows])
    worst = max(v["spread"] for v in quantities.values())
    return {
        "resolutions": rows,
        "settling_resolutions": settling_rows,
        "sweep_velocities": u_list,
        "quantities": quantities,
        "worst_spread": worst,
        "fluid": {
            "rho_fluid": fluid.rho_fluid,
            "rho_particle": fluid.rho_particle,
            "mu": fluid.mu,
            "gravity": fluid.gravity,
        },
    }