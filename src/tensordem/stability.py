"""Explicit time-step stability limits (CFL-class criteria) for the 3-D DEM.

Why this module exists
----------------------
``IceDEM3D.step`` integrates with *semi-implicit* (symplectic) Euler::

    v_{n+1} = v_n + dt * f(x_n, v_n) / m
    x_{n+1} = x_n + dt * v_{n+1}

with a linear bond spring ``k_b``, a linear contact spring ``k_c``, velocity
proportional contact damping ``c`` and linear drag ``c_d``.  The engine only
carried an ad-hoc ``DEM3DConfig.recommended_dt`` heuristic (a 0.15 safety factor
on ``sqrt(m/k)``); no stability criterion, growth rate, or CFL-type bound
existed anywhere in the repository.  This module supplies

1. the *closed-form* amplification factor of the scheme for the canonical
   single-degree-of-freedom (SDOF) test problem (so the criterion can be
   verified against an exact reference instead of against the solver itself),
2. the exact critical step for that SDOF problem,
3. a Gershgorin (row-sum) bound on the highest eigenfrequency of the assembled
   bonded/contacting particle system, which turns the SDOF result into a
   sufficient condition for the many-body lattice, and
4. :func:`stability_report`, which audits a live model state and reports the
   margin between the step actually used and the limit.

Analytic reference (hand derivation, no solver involved)
--------------------------------------------------------
For ``f = -k x - c v`` the update is linear::

    [x_{n+1}]   [ 1 - d^2        dt (1 - g) ] [x_n]
    [v_{n+1}] = [ -dt w^2          1 - g    ] [v_n]

with ``d = dt * w``, ``w = sqrt(k/m)`` and ``g = dt * c / m``.  Hence

    det A = 1 - g,      tr A = 2 - d^2 - g,

and the spectral radius (per-step growth factor of any perturbation) is

    rho = max(|lambda_+|, |lambda_-|),  lambda_\pm = (tr A +/- sqrt(tr^2 - 4 det A)) / 2,

which is real-rooted when ``tr^2 >= 4 det`` and otherwise ``rho = sqrt(|det A|)``.
The stability region is exactly ``0 <= g <= 2`` **and** ``d <= 2``, i.e.

    dt_crit = min(2 / w, 2 m / c).

Both limits are independent, which is why the SDOF critical step factorises so
cleanly; the equality cases ``d = 2`` (repeated root at ``-1``) and ``g = 2``
are marginally stable (``rho = 1``).

The many-body generalisation uses the symmetric positive semi-definite
translational stiffness matrix ``K`` of the free particles.  Its Gershgorin row
bound is ``lambda_max(K) <= 2 * max_i S_i`` where ``S_i`` is the sum of the
stiffness magnitudes of every alive bond / active contact / platen / tool
coupling attached to particle ``i`` (diagonal ``S_i`` plus off-diagonals
``-k_ij``); therefore

    w_max <= sqrt(2 * max_i S_i / m),     dt_crit <= min(2 / w_max, 2 m / c_max).

For identity-proportional damping (pure drag) the bound is *exact* per mode
because ``C = c I`` commutes with ``K``.  For contact damping acting only along
contact normals ``C`` does not commute with ``K`` and the row-sum bound is a
sufficient (conservative) condition; this is stated as such in
``docs/dem3d_timestep_stability.md``.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from .dem3d import DEM3DConfig, IceDEM3D

__all__ = [
    "elastic_timestep_limit",
    "damping_timestep_limit",
    "critical_timestep",
    "symplectic_euler_growth_factor",
    "symplectic_euler_stable",
    "gershgorin_omega_bound",
    "particle_stiffness_sums",
    "particle_damping_sums",
    "kinematic_timestep_limit",
    "stability_report",
]


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0 or (result == 0.0 and not allow_zero):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return result


def symplectic_euler_growth_factor(
    dt: float,
    omega: float,
    damping_rate: float = 0.0,
) -> float:
    """Exact per-step amplification ``rho`` of semi-implicit Euler on an SDOF mode.

    Parameters
    ----------
    dt:
        Time step in seconds.
    omega:
        Undamped angular frequency ``sqrt(k/m)`` in rad/s (``>= 0``).
    damping_rate:
        Viscous rate ``c/m`` in 1/s (``>= 0``).

    Returns
    -------
    float
        Spectral radius of the two-by-two amplification matrix.  ``rho <= 1``
        means perturbations do not grow; ``rho == 1`` is marginal.
    """
    dt = _positive(dt, "dt")
    omega = _positive(omega, "omega", allow_zero=True)
    damping_rate = _positive(damping_rate, "damping_rate", allow_zero=True)

    d2 = (dt * omega) ** 2
    gamma = dt * damping_rate
    det = 1.0 - gamma
    trace = 2.0 - d2 - gamma
    discriminant = trace * trace - 4.0 * det
    if discriminant >= 0.0:
        root = math.sqrt(discriminant)
        return max(abs(0.5 * (trace + root)), abs(0.5 * (trace - root)))
    return math.sqrt(abs(det))


def symplectic_euler_stable(
    dt: float,
    omega: float,
    damping_rate: float = 0.0,
    *,
    tolerance: float = 0.0,
) -> bool:
    """True when the SDOF amplification factor satisfies ``rho <= 1 + tolerance``."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    return symplectic_euler_growth_factor(dt, omega, damping_rate) <= 1.0 + tolerance


def elastic_timestep_limit(omega: float) -> float:
    """``dt <= 2/omega`` for the SDOF oscillator; ``inf`` when ``omega == 0``."""
    omega = _positive(omega, "omega", allow_zero=True)
    return math.inf if omega == 0.0 else 2.0 / omega


def damping_timestep_limit(damping_rate: float) -> float:
    """``dt <= 2 * m / c`` (i.e. ``dt * c / m <= 2``); ``inf`` when zero damping."""
    damping_rate = _positive(damping_rate, "damping_rate", allow_zero=True)
    return math.inf if damping_rate == 0.0 else 2.0 / damping_rate


def critical_timestep(omega: float, damping_rate: float = 0.0) -> float:
    """Closed-form critical step of semi-implicit Euler, ``min(2/w, 2m/c)``."""
    return min(elastic_timestep_limit(omega), damping_timestep_limit(damping_rate))


def gershgorin_omega_bound(
    stiffness_sums: torch.Tensor,
    mass: float,
    free_mask: torch.Tensor | None = None,
) -> float:
    """Row-sum bound ``sqrt(2 * max_i S_i / m)`` on the highest mode frequency.

    ``stiffness_sums[i]`` is the sum of all stiffness magnitudes attached to
    particle ``i``.  Because the assembled stiffness matrix has diagonal
    ``S_i`` and off-diagonals ``-k_ij``, its Gershgorin radius on row ``i`` is
    ``2 S_i``, giving ``lambda_max <= 2 max_i S_i``.  Fixed (Dirichlet)
    particles carry no free degree of freedom and are excluded through
    ``free_mask``; couplings to fixed neighbours still enter ``S_i``, which
    keeps the bound valid (and slightly conservative).
    """
    mass = _positive(mass, "mass")
    sums = torch.as_tensor(stiffness_sums, dtype=torch.float64).flatten()
    if sums.numel() == 0:
        return 0.0
    if bool((sums < -1e-30).any()):
        raise ValueError("stiffness_sums must be nonnegative")
    if free_mask is not None:
        mask = torch.as_tensor(free_mask).flatten().to(torch.bool)
        if mask.shape != sums.shape:
            raise ValueError("free_mask must match stiffness_sums shape")
        if not bool(mask.any()):
            return 0.0
        sums = sums[mask]
    return math.sqrt(2.0 * float(sums.max().item()) / mass)


@torch.no_grad()
def particle_stiffness_sums(model: "IceDEM3D") -> torch.Tensor:
    """Per-particle sum of tangent stiffness magnitudes for the current state.

    Mirrors the force evaluation in :meth:`tensordem.dem3d.IceDEM3D.forces`:
    alive bonds (``bond_stiffness``), active unbonded contacts and the indenter
    (``contact_stiffness``, only where the linear branch is engaged) and the
    optional plane platens (``platen_stiffness``).  Broken bonds contribute
    nothing; one-sided contacts are linearised as active springs, which is the
    upper bound relevant for stability.
    """
    cfg = model.config
    n = len(model.positions)
    sums = torch.zeros(n, dtype=model.dtype, device=model.device)

    if len(model.pairs):
        i, j = model.pairs.T
        alive = model.alive
        stiffness = cfg.bond_stiffness
        sums.index_add_(0, i[alive], torch.full_like(i[alive], stiffness, dtype=model.dtype))
        sums.index_add_(0, j[alive], torch.full_like(j[alive], stiffness, dtype=model.dtype))

    contact_pairs = model._contact_candidate_pairs()
    if contact_pairs.numel():
        ci, cj = contact_pairs.T
        distance = torch.linalg.vector_norm(model.positions[cj] - model.positions[ci], dim=1)
        active = distance < 2.0 * cfg.radius
        active &= ~model._alive_bond_for_pairs(contact_pairs)
        if bool(active.any()):
            stiffness = cfg.contact_stiffness
            ci_active = ci[active]
            cj_active = cj[active]
            sums.index_add_(0, ci_active, torch.full_like(ci_active, stiffness, dtype=model.dtype))
            sums.index_add_(0, cj_active, torch.full_like(cj_active, stiffness, dtype=model.dtype))

    tool_distance = torch.linalg.vector_norm(model.positions - model.tool_position, dim=1)
    tool_active = tool_distance < cfg.radius + cfg.tool_radius
    if bool(tool_active.any()):
        sums[tool_active] += cfg.contact_stiffness

    if cfg.top_platen_enabled:
        top_z = model.top_platen_start_z + model.time * cfg.top_platen_velocity
        sums[(model.positions[:, 2] + cfg.radius > top_z)] += cfg.platen_stiffness
    if cfg.bottom_platen_enabled:
        bottom_z = model.bottom_platen_start_z + model.time * cfg.bottom_platen_velocity
        sums[(model.positions[:, 2] - cfg.radius < bottom_z)] += cfg.platen_stiffness
    return sums


@torch.no_grad()
def particle_damping_sums(model: "IceDEM3D") -> torch.Tensor:
    """Per-particle viscous coefficient (N s/m) acting on the velocity.

    Drag acts on every particle (it is proportional to the identity and
    therefore commutes with the stiffness matrix); contact and tool damping act
    on particles whose contact is currently active; the platens add their own
    damping where engaged.
    """
    cfg = model.config
    n = len(model.positions)
    damping = torch.full((n,), cfg.drag, dtype=model.dtype, device=model.device)

    contact_pairs = model._contact_candidate_pairs()
    if contact_pairs.numel():
        ci, cj = contact_pairs.T
        distance = torch.linalg.vector_norm(model.positions[cj] - model.positions[ci], dim=1)
        active = distance < 2.0 * cfg.radius
        active &= ~model._alive_bond_for_pairs(contact_pairs)
        if bool(active.any()):
            ci_active = ci[active]
            cj_active = cj[active]
            damping.index_add_(0, ci_active, torch.full_like(ci_active, cfg.contact_damping, dtype=model.dtype))
            damping.index_add_(0, cj_active, torch.full_like(cj_active, cfg.contact_damping, dtype=model.dtype))

    tool_distance = torch.linalg.vector_norm(model.positions - model.tool_position, dim=1)
    damping[tool_distance < cfg.radius + cfg.tool_radius] += cfg.contact_damping

    if cfg.top_platen_enabled:
        top_z = model.top_platen_start_z + model.time * cfg.top_platen_velocity
        damping[(model.positions[:, 2] + cfg.radius > top_z)] += cfg.platen_damping
    if cfg.bottom_platen_enabled:
        bottom_z = model.bottom_platen_start_z + model.time * cfg.bottom_platen_velocity
        damping[(model.positions[:, 2] - cfg.radius < bottom_z)] += cfg.platen_damping
    return damping


def kinematic_timestep_limit(model: "IceDEM3D", fraction: float = 0.1) -> float:
    """Heuristic travel bound ``fraction * radius / v_boundary`` (not stability).

    Bounds how far the prescribed indenter or a driven platen may travel in one
    step relative to the particle radius.  It is a discretisation-quality
    heuristic, deliberately reported separately from the stability limits.
    """
    if not (0.0 < fraction):
        raise ValueError("fraction must be positive")
    cfg = model.config
    speeds = [abs(cfg.tool_speed)]
    if cfg.top_platen_enabled:
        speeds.append(abs(cfg.top_platen_velocity))
    if cfg.bottom_platen_enabled:
        speeds.append(abs(cfg.bottom_platen_velocity))
    speed = max(speeds)
    if speed == 0.0:
        return math.inf
    return fraction * cfg.radius / speed


def stability_report(model: "IceDEM3D") -> dict[str, Any]:
    """Audit the explicit step of a live model against the CFL-class limits.

    Returns a JSON-serialisable dictionary with the per-particle row sums, the
    resulting elastic/damping limits, the governing (binding) limit, the margin
    of the step actually in use and the engine's own ``recommended_dt`` guard
    for comparison.  Read-only: the model state is not modified.
    """
    cfg = model.config
    stiffness_sums = particle_stiffness_sums(model)
    damping_sums = particle_damping_sums(model)
    free = ~model.fixed

    omega_upper = gershgorin_omega_bound(stiffness_sums, cfg.mass, free_mask=free)
    damping_max = float(damping_sums[free].max().item()) if bool(free.any()) else 0.0
    damping_rate_upper = damping_max / cfg.mass

    elastic_limit = elastic_timestep_limit(omega_upper)
    damping_limit = damping_timestep_limit(damping_rate_upper)
    critical = critical_timestep(omega_upper, damping_rate_upper)
    if elastic_limit <= damping_limit:
        binding = "elastic" if math.isfinite(elastic_limit) else "none"
    else:
        binding = "damping"

    return {
        "dt_s": float(model.dt),
        "n_particles": int(len(model.positions)),
        "n_free_particles": int(free.sum().item()),
        "n_alive_bonds": int(model.alive.sum().item()),
        "mass_kg": float(cfg.mass),
        "max_stiffness_sum_N_per_m": float(stiffness_sums[free].max().item()) if bool(free.any()) else 0.0,
        "max_damping_Ns_per_m": damping_max,
        "omega_upper_rad_s": omega_upper,
        "damping_rate_upper_per_s": damping_rate_upper,
        "elastic_dt_limit_s": elastic_limit,
        "damping_dt_limit_s": damping_limit,
        "critical_dt_s": critical,
        "binding_limit": binding,
        "margin_factor": (critical / float(model.dt)) if math.isfinite(critical) else math.inf,
        "is_stable": bool(float(model.dt) < critical if math.isfinite(critical) else True),
        "engine_recommended_dt_s": float(cfg.recommended_dt),
        "engine_dt_over_recommended": float(model.dt) / float(cfg.recommended_dt),
        "kinematic_dt_limit_s": kinematic_timestep_limit(model),
        "growth_factor_at_dt": symplectic_euler_growth_factor(
            float(model.dt), omega_upper, damping_rate_upper
        ),
    }