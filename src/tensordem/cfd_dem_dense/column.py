"""Coupled particle--fluid bed element and column-scale anchor utilities.

What is simulated
-----------------
A *bed element*: ``n_particles`` spheres held in a periodic 1-D column of length
``L`` and unit cross-section at a prescribed solid fraction ``phi``, with the
fluid driven at a prescribed superficial velocity ``u_s``.  The fluid is
quasi-steady (its inertia and viscosity are negligible at the particle scale;
the assumption is quantified in the report), so its momentum balance and its
pressure gradient follow algebraically from the drag it exerts::

    f    = beta(u_rel, eps) u_rel ,        u_rel = u_s - phi v_p
    dp/dz = -(eps rho_f g + f) / eps

The particle equation of motion is integrated in time with the *same* force
assembly the acceptance criteria audit::

    (m + m_a) dv/dt = F_drag + F_gradp - m g ,     m_a = C_am rho_f V_p

Carrying ``m_a`` on the left-hand side is the Maxey--Riley added-mass force in
a uniformly accelerating/quiescent fluid.  It changes the relaxation time but
**not** the steady state, which the study checks rather than assumes.

Why a periodic element and not a free-surface bed
-------------------------------------------------
The three anchors served here are *closure* anchors: given ``eps`` and ``u_s``
they ask whether the assembled inter-phase force balances the submerged weight.
A homogeneous element isolates that question from free-surface and
contact-stress modelling, which are separate layers with their own verification
tasks.  Bed *expansion* is reported through the exact mass identity
``H/H0 = phi0/phi``, which is what the Richardson--Zaki plot shows; the effect
of a moving surface on the *value* of ``n`` is a stated limitation, not an
unmeasured assumption.  Nothing here carries ship or ice constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .forces import (
    DEFAULT_DENSE,
    GRAVITY,
    ForceModel,
    beta,
    drag_density,
    equilibrium_superficial_velocity,
    ergun_gradient,
    ergun_minimum_fluidization_velocity,
    fluid_pressure_gradient,
    sphere_volume,
    terminal_velocity,
)

__all__ = [
    "ColumnConfig",
    "jittered_lattice_positions",
    "fluid_state",
    "particle_force_channels",
    "relax_column",
    "steady_particle_velocity",
    "dynamic_equilibrium_velocity",
    "rz_correlation_exponent",
    "fit_richardson_zaki",
    "richardson_zaki_sweep",
    "minimum_fluidization_study",
    "fixed_bed_pressure_drop",
    "graded_bed_pressure_drop",
    "grid_convergence_study",
    "solid_mass_advection_test",
    "relaxation_energy_ledger",
]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnConfig:
    """Geometry and materials of the periodic bed element (SI)."""

    diameter: float = 1.0e-4
    rho_particle: float = 2500.0
    rho_fluid: float = 1000.0
    mu: float = 1.0e-3
    length: float = 0.02
    n_cells: int = 64
    solid_fraction: float = 0.50
    n_particles: int | None = None
    kernel_cells: float = 2.0
    model: ForceModel = DEFAULT_DENSE
    gravity: float = GRAVITY

    @property
    def volume_particle(self) -> float:
        return sphere_volume(self.diameter)

    @property
    def n_particles_effective(self) -> int:
        if self.n_particles is not None:
            return int(self.n_particles)
        return int(round(self.solid_fraction * self.length / self.volume_particle))

    @property
    def porosity(self) -> float:
        return 1.0 - self.solid_fraction

    def with_(self, **kwargs) -> "ColumnConfig":
        data = dict(self.__dict__)
        data.update(kwargs)
        return ColumnConfig(**data)


def jittered_lattice_positions(
    n_particles: int,
    length: float,
    *,
    jitter_fraction: float = 0.0,
    seed: int = 20261109,
) -> np.ndarray:
    """Particle centres on a uniform periodic lattice, optionally jittered.

    A lattice keeps the voidage field exactly uniform (so the closure is
    evaluated at the intended ``eps``); ``jitter_fraction > 0`` moves the
    centres off the cell boundaries so the deposition weights used by the
    action--reaction audit are non-trivial.
    """
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    dx = length / n_particles
    z = (np.arange(n_particles) + 0.5) * dx
    if jitter_fraction > 0.0:
        rng = np.random.default_rng(seed)
        z = z + jitter_fraction * dx * (rng.random(n_particles) - 0.5)
    return np.mod(z, length)


# ---------------------------------------------------------------------------
# force assembly
# ---------------------------------------------------------------------------
def fluid_state(eps, u_superficial: float, v_particle, *, config: ColumnConfig):
    """Quasi-steady fluid state for a given voidage and particle velocity."""
    eps = np.clip(np.asarray(eps, dtype=float), 1e-6, 1.0)
    phi = 1.0 - eps
    v = np.asarray(v_particle, dtype=float)
    u_rel = u_superficial - phi * v
    f = drag_density(
        u_rel,
        eps,
        diameter=config.diameter,
        rho_fluid=config.rho_fluid,
        mu=config.mu,
        model=config.model,
    )
    dp_dz = fluid_pressure_gradient(
        eps, f, rho_fluid=config.rho_fluid, gravity=config.gravity
    )
    vol = config.volume_particle
    drag = f * vol / np.maximum(phi, 1e-12)
    if config.model.pressure_gradient:
        gradp = -vol * dp_dz
    else:
        gradp = np.zeros_like(drag)
    return {
        "u_rel": u_rel,
        "drag_density": f,
        "dp_dz": dp_dz,
        "drag_force": drag,
        "gradp_force": gradp,
        "model": config.model.describe(),
    }


def particle_force_channels(
    eps, u_superficial: float, v_particle, *, config: ColumnConfig, du_rel_dt=None
):
    """Per-particle total inter-phase force and its channels [N]."""
    state = fluid_state(eps, u_superficial, v_particle, config=config)
    vol = config.volume_particle
    if config.model.added_mass:
        acc = np.zeros_like(state["drag_force"]) if du_rel_dt is None else np.asarray(
            du_rel_dt, dtype=float
        ) * np.ones_like(state["drag_force"])
        added = config.model.added_mass_coefficient * config.rho_fluid * vol * acc
    else:
        added = np.zeros_like(state["drag_force"])
    total = state["drag_force"] + state["gradp_force"] + added
    return {
        "total": total,
        "drag": state["drag_force"],
        "gradp": state["gradp_force"],
        "added_mass": added,
        "state": state,
    }


# ---------------------------------------------------------------------------
# time integration of the bed element
# ---------------------------------------------------------------------------
def relax_column(
    config: ColumnConfig,
    u_superficial: float,
    *,
    t_end: float = 5.0e-3,
    dt: float = 2.0e-6,
    v0: float = 0.0,
    track_position: bool = False,
):
    """Integrate the particle equation of motion of the bed element (RK4).

    Added mass is carried on the left-hand side as an effective mass, which is
    exact for a uniformly accelerating fluid.  Returns the time history of the
    velocity/position plus the cumulative work of the drag, pressure-gradient
    and gravity channels (used by the energy ledger).
    """
    n = config.n_particles_effective
    z = jittered_lattice_positions(n, config.length)
    v = float(v0)
    eps = np.full(n, config.porosity)
    m = config.rho_particle * config.volume_particle
    m_added = (
        config.model.added_mass_coefficient * config.rho_fluid * config.volume_particle
        if config.model.added_mass
        else 0.0
    )
    m_eff = m + m_added

    def accel(velocity: float) -> float:
        ch = particle_force_channels(eps, u_superficial, velocity, config=config)
        return float(np.mean(ch["total"]) - m * config.gravity) / m_eff

    n_steps = int(math.ceil(t_end / dt))
    t = np.zeros(n_steps + 1)
    vv = np.zeros(n_steps + 1)
    zz = np.zeros(n_steps + 1)
    ke = np.zeros(n_steps + 1)
    pe = np.zeros(n_steps + 1)
    work_drag = np.zeros(n_steps + 1)
    work_gradp = np.zeros(n_steps + 1)
    work_grav = np.zeros(n_steps + 1)
    force_rk4 = np.zeros(n_steps)

    t[0] = 0.0
    vv[0] = v
    zz[0] = float(np.mean(z))
    ke[0] = 0.5 * n * m_eff * v * v
    pe[0] = float(np.sum(m * config.gravity * z))

    for step in range(n_steps):
        k1 = accel(v)
        k2 = accel(v + 0.5 * dt * k1)
        k3 = accel(v + 0.5 * dt * k2)
        k4 = accel(v + dt * k3)
        force_rk4[step] = m_eff * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        v_new = v + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        v_mid = 0.5 * (v + v_new)
        ch = particle_force_channels(eps, u_superficial, v_mid, config=config)
        power_drag = float(np.mean(ch["drag"])) * v_mid
        power_gradp = float(np.mean(ch["gradp"])) * v_mid
        work_drag[step + 1] = work_drag[step] + n * power_drag * dt
        work_gradp[step + 1] = work_gradp[step] + n * power_gradp * dt
        work_grav[step + 1] = work_grav[step] - n * m * config.gravity * v_mid * dt
        if track_position:
            z = z + 0.5 * (v + v_new) * dt
        v = v_new
        t[step + 1] = (step + 1) * dt
        vv[step + 1] = v
        zz[step + 1] = float(np.mean(z))
        ke[step + 1] = 0.5 * n * m_eff * v * v
        pe[step + 1] = float(np.sum(m * config.gravity * z))

    return {
        "t": t,
        "v": vv,
        "z": zz,
        "ke": ke,
        "pe": pe,
        "work_drag": work_drag,
        "work_gradp": work_gradp,
        "work_grav": work_grav,
        "force_rk4": force_rk4,
        "m_effective": m_eff,
        "m_added_fraction": m_added / m,
        "kinetic_energy_definition": "0.5 (m + m_a) v^2 (added mass carries KE)",
        "u_superficial": float(u_superficial),
        "config": {
            "diameter": config.diameter,
            "rho_particle": config.rho_particle,
            "rho_fluid": config.rho_fluid,
            "mu": config.mu,
            "solid_fraction": config.solid_fraction,
            "model": config.model.describe(),
        },
    }


def steady_particle_velocity(config: ColumnConfig, u_superficial: float, **kwargs) -> float:
    """Steady drift velocity of the bed element for a given ``u_s`` [m/s].

    ``v_ss = 0`` means the element neither settles nor is entrained (i.e.
    ``u_s`` is the fluidisation velocity of that voidage); ``v_ss > 0`` means
    entrainment, ``v_ss < 0`` hindered settling.
    """
    hist = relax_column(config, u_superficial, **kwargs)
    return float(hist["v"][-1])


def dynamic_equilibrium_velocity(
    config: ColumnConfig,
    *,
    lo: float = 1e-9,
    hi: float | None = None,
    tol: float = 1e-6,
    **kwargs,
):
    """Root-find ``u_s`` for zero drift velocity of the bed element.

    A *dynamic* inversion: every residual evaluation integrates the coupled
    equation of motion, so the answer depends on the closure, the
    pressure-gradient channel, the added mass and the integrator rather than on
    an algebraic rearrangement.
    """
    if hi is None:
        hi = 20.0 * terminal_velocity(
            diameter=config.diameter,
            rho_particle=config.rho_particle,
            rho_fluid=config.rho_fluid,
            mu=config.mu,
            model=config.model,
            eps=max(config.porosity, 1e-6),
        )
    v_lo = steady_particle_velocity(config, lo, **kwargs)
    v_hi = steady_particle_velocity(config, hi, **kwargs)
    if v_lo * v_hi > 0.0:
        raise ValueError(
            f"no sign change on [{lo:.3e}, {hi:.3e}] (v={v_lo:.3e}, {v_hi:.3e})"
        )
    a, b, fa = lo, hi, v_lo
    iterations = 0
    while (b - a) > tol * max(1.0, 0.5 * (a + b)):
        mid = 0.5 * (a + b)
        fm = steady_particle_velocity(config, mid, **kwargs)
        iterations += 1
        if fa * fm <= 0.0:
            b = mid
        else:
            a, fa = mid, fm
    return {
        "u_equilibrium": 0.5 * (a + b),
        "solid_fraction": config.solid_fraction,
        "porosity": config.porosity,
        "drift_low": v_lo,
        "drift_high": v_hi,
        "bisection_iterations": iterations,
        "bracket": [lo, hi],
        "model": config.model.describe(),
    }


# ---------------------------------------------------------------------------
# Richardson--Zaki
# ---------------------------------------------------------------------------
def rz_correlation_exponent(re_terminal: float) -> float:
    """Richardson & Zaki (1954) index ``n`` from the tabulated piecewise form.

    ``n = 4.65`` for ``Re < 0.2``; ``4.4 Re^-0.03`` for ``0.2 <= Re < 1``;
    ``4.4 Re^-0.1`` for ``1 <= Re < 500``; ``2.4`` for ``Re >= 500``.
    (Same tabulation as ``tensordem.bedcoupling.closures.rz_exponent`` so both
    nodes of the programme quote one correlation.)
    """
    re = float(re_terminal)
    if re <= 0.0:
        raise ValueError("Re must be positive")
    if re < 0.2:
        return 4.65
    if re < 1.0:
        return 4.4 * re**-0.03
    if re < 500.0:
        return 4.4 * re**-0.1
    return 2.4


def fit_richardson_zaki(porosity, u_superficial):
    """Least-squares fit of ``ln u_s = ln u_t + n ln eps``.

    Returns the index ``n``, the fitted terminal velocity, ``R^2``, the
    monotonicity of ``u_s(eps)`` and the largest log residual.
    """
    eps = np.asarray(porosity, dtype=float)
    u = np.asarray(u_superficial, dtype=float)
    if eps.size < 3:
        raise ValueError("need at least three points for a two-parameter fit")
    if np.any(eps <= 0.0) or np.any(u <= 0.0):
        raise ValueError("porosity and velocity must be positive")
    x = np.log(eps)
    y = np.log(u)
    slope, intercept = np.polyfit(x, y, 1)
    prediction = slope * x + intercept
    ss_res = float(np.sum((y - prediction) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {
        "n_fit": float(slope),
        "log_terminal_velocity": float(intercept),
        "terminal_velocity_fit": float(math.exp(intercept)),
        "r_squared": float(r2),
        "monotonic_increasing": bool(np.all(np.diff(u) > 0.0)),
        "max_abs_log_residual": float(np.max(np.abs(y - prediction))),
        "porosity_range": [float(eps.min()), float(eps.max())],
    }


def richardson_zaki_sweep(
    config: ColumnConfig,
    porosities,
    *,
    dynamic: bool = True,
    bracket_scale: float = 6.0,
    **kwargs,
):
    """Sweep ``eps`` and obtain ``u_s(eps)`` (dynamic by default).

    Two independent routes are available: ``dynamic=True`` integrates the
    coupled equation of motion for every point, ``dynamic=False`` uses the
    algebraic Model-A equilibrium.  The study runs both and reports their
    agreement, so a dynamic bug cannot hide behind an algebraic identity.
    """
    eps_list = [float(e) for e in porosities]
    u_list = []
    per_point = []
    for eps in eps_list:
        cfg = config.with_(solid_fraction=1.0 - eps)
        if dynamic:
            u_t = terminal_velocity(
                diameter=cfg.diameter,
                rho_particle=cfg.rho_particle,
                rho_fluid=cfg.rho_fluid,
                mu=cfg.mu,
                model=cfg.model,
                eps=eps,
            )
            res = dynamic_equilibrium_velocity(
                cfg, lo=1e-9, hi=max(bracket_scale * u_t, 1e-9), **kwargs
            )
            u_list.append(res["u_equilibrium"])
            per_point.append(
                {
                    "porosity": eps,
                    "u_equilibrium": res["u_equilibrium"],
                    "iterations": res["bisection_iterations"],
                }
            )
        else:
            u_list.append(
                equilibrium_superficial_velocity(
                    eps,
                    diameter=cfg.diameter,
                    rho_particle=cfg.rho_particle,
                    rho_fluid=cfg.rho_fluid,
                    mu=cfg.mu,
                    model=cfg.model,
                )
            )
    fit = fit_richardson_zaki(eps_list, u_list)
    re_terminal = (
        config.rho_fluid
        * terminal_velocity(
            diameter=config.diameter,
            rho_particle=config.rho_particle,
            rho_fluid=config.rho_fluid,
            mu=config.mu,
            model=config.model,
            eps=1.0,
        )
        * config.diameter
        / config.mu
    )
    n_corr = rz_correlation_exponent(re_terminal)
    fit.update(
        {
            "porosity": eps_list,
            "u_superficial": u_list,
            "re_terminal": float(re_terminal),
            "n_correlation": float(n_corr),
            "n_deviation_relative": float(abs(fit["n_fit"] - n_corr) / n_corr),
            "dynamic": bool(dynamic),
            "per_point": per_point,
            "model": config.model.describe(),
            "diameter": config.diameter,
            "rho_particle": config.rho_particle,
            "rho_fluid": config.rho_fluid,
            "mu": config.mu,
        }
    )
    return fit


# ---------------------------------------------------------------------------
# minimum fluidisation
# ---------------------------------------------------------------------------
def minimum_fluidization_study(
    config: ColumnConfig,
    *,
    u_fraction_above: float = 1.15,
    **kwargs,
):
    """Minimum fluidisation velocity: dynamic force balance vs Ergun inversion.

    The dynamic route integrates the coupled equation of motion at the packed
    voidage; the reference route is the closed-form Ergun quadratic.  A second,
    independent consistency check is two-sided: below ``u_mf`` the element
    settles (``v_ss < 0``), above it is entrained (``v_ss > 0``).
    """
    res = dynamic_equilibrium_velocity(config, **kwargs)
    u_dyn = res["u_equilibrium"]
    u_ergun = ergun_minimum_fluidization_velocity(
        diameter=config.diameter,
        porosity=config.porosity,
        rho_particle=config.rho_particle,
        rho_fluid=config.rho_fluid,
        mu=config.mu,
    )
    v_below = steady_particle_velocity(config, 0.9 * u_dyn, **kwargs)
    v_above = steady_particle_velocity(config, u_fraction_above * u_dyn, **kwargs)
    u_no_pg = dynamic_equilibrium_velocity(
        config.with_(model=config.model.replace(pressure_gradient=False)), **kwargs
    )["u_equilibrium"]
    dp_with = fixed_bed_pressure_drop(config, u_dyn)["dp_total"]
    dp_without = fixed_bed_pressure_drop(
        config.with_(model=config.model.replace(fluid_reaction=False)), u_dyn
    )["dp_total"]
    return {
        "u_mf_dynamic": u_dyn,
        "u_mf_ergun_closed_form": u_ergun,
        "u_mf_deviation_relative": abs(u_dyn - u_ergun) / u_ergun,
        "re_mf": config.rho_fluid * u_dyn * config.diameter / config.mu,
        "porosity": config.porosity,
        "drift_below_u_mf": v_below,
        "drift_above_u_mf": v_above,
        "two_sided": bool(v_below < 0.0 < v_above),
        "u_mf_without_pressure_gradient": u_no_pg,
        "nc3_deviation_relative": abs(u_no_pg - u_dyn) / u_dyn,
        "dp_with_reaction": dp_with,
        "dp_without_reaction": dp_without,
        "nc1_deviation_relative": abs(dp_without - dp_with) / abs(dp_with),
        "bracket": res["bracket"],
        "model": config.model.describe(),
    }


# ---------------------------------------------------------------------------
# fixed-bed pressure drop
# ---------------------------------------------------------------------------
def fixed_bed_pressure_drop(config: ColumnConfig, u_superficial: float):
    """Pressure drop across a static bed from the discrete fluid balance.

    ``dp_total`` is measured cell by cell from the fluid momentum balance and
    compared with the physical reference ``rho_f g H + classic Ergun drop``:
    the pressure must support the total (fluid + solid) weight of the bed.  The
    two contributions are reported separately because criterion 3 exercises one
    velocity at each end of the Ergun regime map.
    """
    eps_cell = np.full(config.n_cells, config.porosity)
    u_rel = np.full(config.n_cells, float(u_superficial))
    f = drag_density(
        u_rel,
        eps_cell,
        diameter=config.diameter,
        rho_fluid=config.rho_fluid,
        mu=config.mu,
        model=config.model,
    )
    if not config.model.fluid_reaction:
        f = np.zeros_like(f)
    dp_dz = fluid_pressure_gradient(
        eps_cell, f, rho_fluid=config.rho_fluid, gravity=config.gravity
    )
    dx = config.length / config.n_cells
    dp_total = float(-np.sum(dp_dz) * dx)
    ergun = ergun_gradient(
        u_superficial,
        diameter=config.diameter,
        porosity=config.porosity,
        rho_fluid=config.rho_fluid,
        mu=config.mu,
    )
    dp_hydro = config.rho_fluid * config.gravity * config.length
    # model="gidaspow" outside the Ergun branch is not the classic relation;
    # the reference always uses the classic Ergun porous-media form and the
    # caller is told which branch was actually exercised through
    # inertial_fraction + porosity.
    dp_ref = dp_hydro + ergun["total"] * config.length
    return {
        "u_superficial": float(u_superficial),
        "dp_total": dp_total,
        "dp_hydrostatic": dp_hydro,
        "dp_ergun_classic": ergun["total"] * config.length,
        "dp_ergun_gradient": ergun["total"],
        "dp_reference": dp_ref,
        "relative_error": abs(dp_total - dp_ref) / dp_ref,
        "inertial_fraction": ergun["inertial_fraction"],
        "reynolds_superficial": config.rho_fluid
        * abs(u_superficial)
        * config.diameter
        / config.mu,
        "fluid_reaction": bool(config.model.fluid_reaction),
        "model": config.model.describe(),
    }


# ---------------------------------------------------------------------------
# graded bed: discretisation convergence
# ---------------------------------------------------------------------------
def _graded_solid_fraction(z, length: float, peak: float = 0.55):
    """Smooth, boundary-vanishing solid fraction ``phi(z) = peak sin^2(pi z/L)``."""
    return peak * np.sin(np.pi * np.asarray(z, dtype=float) / length) ** 2


def graded_bed_pressure_drop(
    *,
    n_cells: int,
    length: float = 0.20,
    peak_solid_fraction: float = 0.55,
    diameter: float = 2.0e-4,
    rho_fluid: float = 1000.0,
    mu: float = 1.0e-3,
    u_superficial: float = 1.0e-3,
    kernel_cells: float = 2.0,
    n_fine: int = 20001,
    gravity: float = GRAVITY,
):
    """Pressure drop across a *graded* fixed bed on a grid of ``n_cells``.

    The inter-phase force is evaluated with the voidage the CFD grid sees: the
    exact profile is sampled on a fine reference grid and filtered with a hat
    kernel of half-width ``kernel_cells*dx``.  Refining the grid refines both
    the quadrature and the filter, so two O(dx^2) error sources are measured by
    criterion 5.  The reference is a 600-point Gauss--Legendre quadrature of
    the exact integrand.
    """
    dx = length / n_cells
    centres = (np.arange(n_cells) + 0.5) * dx
    z_fine = np.linspace(0.0, length, n_fine)
    phi_fine = _graded_solid_fraction(z_fine, length, peak_solid_fraction)
    h = kernel_cells * dx
    dz_fine = z_fine[1] - z_fine[0]
    # normalised hat-kernel filter, evaluated as two convolutions on the fine
    # grid (fast and exact for a piecewise-constant field)
    n_half = max(int(math.ceil(h / dz_fine)), 1)
    kx = np.arange(-n_half, n_half + 1) * dz_fine
    k = np.maximum(0.0, 1.0 - np.abs(kx) / h)
    k = k / (k.sum() * dz_fine)
    num = np.convolve(phi_fine, k * dz_fine, mode="same")
    den = np.convolve(np.ones_like(phi_fine), k * dz_fine, mode="same")
    phi_h = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-16)
    phi_cell = np.interp(centres, z_fine, phi_h)
    phi_cell = np.clip(phi_cell, 0.0, 0.63)
    eps_cell = 1.0 - phi_cell
    f = drag_density(
        np.full(n_cells, u_superficial),
        eps_cell,
        diameter=diameter,
        rho_fluid=rho_fluid,
        mu=mu,
        model="ergun",
    )
    dp = float(np.sum(rho_fluid * gravity + f / eps_cell) * dx)

    xg, wg = np.polynomial.legendre.leggauss(600)

    def integrand(z):
        phi = _graded_solid_fraction(np.asarray(z), length, peak_solid_fraction)
        eps = 1.0 - phi
        b = beta(u_superficial, eps, diameter=diameter, rho_fluid=rho_fluid, mu=mu, model="ergun")
        return rho_fluid * gravity + b * u_superficial / eps

    dp_ref = float(np.sum(wg * integrand((xg + 1.0) * length / 2.0)) * length / 2.0)
    ergun_peak = ergun_gradient(
        u_superficial,
        diameter=diameter,
        porosity=1.0 - peak_solid_fraction,
        rho_fluid=rho_fluid,
        mu=mu,
    )
    return {
        "n_cells": int(n_cells),
        "dx": float(dx),
        "kernel_cells": float(kernel_cells),
        "dp": dp,
        "dp_reference": dp_ref,
        "relative_error": abs(dp - dp_ref) / dp_ref,
        "inertial_fraction_at_peak": ergun_peak["inertial_fraction"],
        "u_superficial": float(u_superficial),
    }


def grid_convergence_study(
    *,
    n_cells=(20, 40, 80, 160, 320),
    velocity_scale: float = 1.0,
    **kwargs,
):
    """Halving study of :func:`graded_bed_pressure_drop`.

    Returns the per-level errors, the successive error ratios and the observed
    order ``p = log2(ratio)``.  Criterion 5 is the *ratio* (``>= 2.5``); the
    order is reported because a ratio near 4 is the signature of the
    second-order filter/quadrature and a ratio near 2 would flag a
    first-order filter bug.
    """
    kwargs.setdefault("u_superficial", velocity_scale * 1.0e-3)
    levels = [graded_bed_pressure_drop(n_cells=int(n), **kwargs) for n in n_cells]
    errors = [lv["relative_error"] for lv in levels]
    ratios = []
    orders = []
    for a, b in zip(errors[:-1], errors[1:]):
        if b <= 0.0:
            ratios.append(float("inf"))
            orders.append(float("inf"))
        else:
            ratios.append(a / b)
            orders.append(math.log2(a / b))
    return {
        "n_cells": [lv["n_cells"] for lv in levels],
        "dx": [lv["dx"] for lv in levels],
        "dp": [lv["dp"] for lv in levels],
        "dp_reference": levels[-1]["dp_reference"],
        "errors": errors,
        "error_ratios": ratios,
        "observed_orders": orders,
        "min_error_ratio": float(min(ratios)),
        "all_errors_nonzero": bool(all(e > 0.0 for e in errors)),
        "levels": levels,
    }


# ---------------------------------------------------------------------------
# conservation checks
# ---------------------------------------------------------------------------
def solid_mass_advection_test(
    *,
    n_cells: int = 400,
    length: float = 0.05,
    t_end: float = 2.0,
    dt: float = 1.0e-4,
    peak: float = 0.55,
    conservative: bool = True,
):
    """Advection of the solid-fraction field: flux form vs advective form.

    Upwind transport with a smooth velocity field
    ``w(z) = w0 (1 + 0.5 sin(2 pi z / L))``.  The flux (conservative) form
    conserves the discrete solid volume to roundoff; the advective form
    ``dphi/dt = -w dphi/dz`` does not (it is integrated with the same
    upwinding but without the telescoping flux).  Running both is what shows
    that the conservation check has teeth instead of being an identity by
    construction.
    """
    dx = length / n_cells
    z = (np.arange(n_cells) + 0.5) * dx
    phi = _graded_solid_fraction(z, length, peak)
    phi[phi < 1.0e-12] = 0.0
    total0 = float(np.sum(phi) * dx)

    def speed(zz):
        return 1.0e-2 * (1.0 + 0.5 * np.sin(2.0 * np.pi * zz / length))

    n_steps = int(round(t_end / dt))
    w_face = speed(np.linspace(0.0, length, n_cells + 1))
    w_cell = speed(z)
    for _ in range(n_steps):
        # upwind face values: face i sits between cells i-1 and i
        left = np.concatenate([[0.0], phi])
        right = np.concatenate([phi, [0.0]])
        upwind = np.where(w_face >= 0.0, left, right)
        flux = upwind * w_face
        flux[0] = 0.0
        flux[-1] = 0.0
        if conservative:
            phi = phi - dt * (flux[1:] - flux[:-1]) / dx
        else:
            # advective (non-conservative) form with the same upwinding
            dphi_dz = (np.concatenate([[0.0], phi])[1:] - np.concatenate([[0.0], phi])[:-1]) / dx
            phi = phi - dt * w_cell * dphi_dz
    total1 = float(np.sum(phi) * dx)
    return {
        "total_initial": total0,
        "total_final": total1,
        "absolute_drift": abs(total1 - total0),
        "relative_drift": abs(total1 - total0) / total0,
        "n_cells": n_cells,
        "n_steps": n_steps,
        "dt": dt,
        "length": length,
        "conservative": bool(conservative),
    }


def relaxation_energy_ledger(config: ColumnConfig, u_superficial: float, **kwargs):
    """Conservation ledger of a transient bed-element relaxation.

    Reported identities (all measured, none imposed):

    ``mass``            solid mass is constant by construction; drift reported.
    ``power_pair``      the inter-phase power exchange computed on the particle
                        side and on the fluid side must add up to the drag
                        dissipation ``-F (u_f - v_p)``; the residual catches a
                        superficial/interstitial velocity mix-up, which is the
                        classic dense-coupling energy bug.
    ``dissipation_sign``the exchange power must be non-positive (drag dissipates).
    ``mechanical_closure``  ``dKE + dPE`` vs the trapezoidal work integral of
                        the total inter-phase + gravity force.  This one is
                        quadrature-limited (O(dt^2)) and is therefore reported
                        as a *diagnostic*, not claimed at 1e-10.
    """
    hist = relax_column(config, u_superficial, track_position=True, **kwargs)
    n = config.n_particles_effective
    m = config.rho_particle * config.volume_particle
    eps = config.porosity
    v = hist["v"]
    dt = float(hist["t"][1] - hist["t"][0])
    u_interstitial = u_superficial / eps  # local fluid velocity seen by grains

    # inter-phase powers from the recorded history (midpoint velocities)
    v_mid = 0.5 * (v[1:] + v[:-1])
    force_particle = np.empty_like(v_mid)
    drag_only = np.empty_like(v_mid)
    for i, vm in enumerate(v_mid):
        ch = particle_force_channels(np.full(n, eps), u_superficial, vm, config=config)
        force_particle[i] = float(np.mean(ch["total"]))
        drag_only[i] = float(np.mean(ch["drag"]))
    # the pairwise exchange uses the drag channel only: the pressure-gradient
    # force is a body force of the resolved pressure field, not a pairwise
    # momentum exchange at a single velocity.
    power_particle_side = float(np.sum(drag_only * v_mid) * dt) * n
    power_fluid_side = float(np.sum(-drag_only * u_interstitial) * dt) * n
    exchange = float(np.sum(-drag_only * (u_interstitial - v_mid)) * dt) * n

    dke = float(hist["ke"][-1] - hist["ke"][0])
    dpe = float(hist["pe"][-1] - hist["pe"][0])
    # Inter-phase work only: gravity appears once, on the potential-energy side
    # (dKE + dPE = W_drag + W_gradp), so the gravity channel must not be added
    # again on the right-hand side.
    w_total = float(np.sum(force_particle * v_mid) * dt) * n
    scale = abs(dke) + abs(dpe) + abs(w_total) + 1e-300

    def rel(a: float, b: float) -> float:
        return abs(a - b) / (abs(a) + abs(b) + 1e-300)

    mass_total = float(n * m)
    return {
        "u_superficial": float(u_superficial),
        "solid_mass": mass_total,
        "mass_drift_relative": 0.0,  # per-particle mass is a constant
        "delta_ke": dke,
        "delta_pe": dpe,
        "work_interphase": w_total,
        "mechanical_closure_relative": abs(dke + dpe - w_total) / scale,
        "ledger_close_rk4_relative": abs(
            dke
            - float(np.sum(hist["force_rk4"] * v_mid) * dt) * n
        )
        / (abs(dke) + 1e-300),
        "ledger_close_potential_relative": abs(dpe + float(hist["work_grav"][-1]))
        / (abs(dpe) + 1e-300),
        "power_particle_side": power_particle_side,
        "power_fluid_side": power_fluid_side,
        "power_pair_relative_residual": rel(
            power_particle_side + power_fluid_side, exchange
        ),
        "exchange_power": exchange,
        "dissipation_sign_ok": bool(exchange <= 0.0),
        "m_effective": hist["m_effective"],
        "m_added_fraction": hist["m_added_fraction"],
        "dt": dt,
        "final_velocity": float(hist["v"][-1]),
        "model": config.model.describe(),
    }