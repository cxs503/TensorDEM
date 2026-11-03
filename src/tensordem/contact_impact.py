"""Analytic contact-impact benchmark for the DEM3D normal contact model.

Why this module exists
----------------------
``contact_tangential.restitution_to_damping`` maps a *target* coefficient of
restitution ``e`` to a normal damping ``gamma_n`` using the textbook
**unclamped** linear spring-dashpot identity

    e = exp(-pi*zeta / sqrt(1 - zeta**2)),   zeta = gamma_n / (2*sqrt(k_n*m_eff))

That identity is exact only for the free mass-spring-dashpot oscillator, whose
normal force is allowed to become *attractive* (``F < 0``) late in the contact.

The DEM3D solver does not allow that: every normal force branch is truncated by
``clamp_min(0)``::

    F_n = max(0, k_n*overlap - c_n*v_n)        (dem3d.py, pair / tool / platen)

The truncation removes the retarding force over the last part of the separation
phase, so the *realised* restitution of the merged solver is systematically
**higher** than the target fed into the mapping.  For ``e_target = 0.50`` the
bias is about +10 %, i.e. outside the 3 % accuracy band used for engineering
predictions.  This module provides the closed-form reference for the clamped
model, a corrected damping mapping, and a harness that measures the solver
against both references so the bias can never silently regress.

Analytic basis
--------------
Relative coordinate ``delta = 2R - s`` (overlap, ``s`` = centre separation),
effective mass ``m_eff`` (``m/2`` for an equal-mass pair), closing speed ``v0``:

    m_eff*delta'' + c*delta' + k*delta = 0      (while F > 0)

with ``zeta = c/(2*sqrt(k*m_eff))``, ``omega_n = sqrt(k/m_eff)``,
``omega_d = omega_n*sqrt(1-zeta**2)`` and ``delta(0)=0, delta'(0)=v0``:

    delta(t)  = (v0/omega_d) * exp(-zeta*omega_n*t) * sin(omega_d*t)
    delta'(t) = v0*exp(-zeta*omega_n*t)
                * [cos(theta) - (zeta/sqrt(1-zeta**2))*sin(theta)],  theta = omega_d*t

Unclamped (``e_unclamped``): contact lasts one half period ``t_c = pi/omega_d``
and ``e = -delta'(t_c)/v0 = exp(-pi*zeta/sqrt(1-zeta**2))``.

Clamped (``e_clamped``): the force vanishes as soon as
``k*delta + c*delta' = 0``, i.e. ``omega_n*delta + 2*zeta*delta' = 0``.  Dividing
by ``v0*exp(-zeta*omega_n*t)``:

    (1 - 2*zeta**2)*sin(theta*) + 2*zeta*sqrt(1-zeta**2)*cos(theta*) = 0

    => theta* = atan2(2*zeta*sqrt(1-zeta**2), 2*zeta**2 - 1)   in (pi/2, pi] for zeta < 1/sqrt(2)

    => e_clamped = -delta'(t*)/v0,   t* = theta*/omega_d

For ``zeta >= 1/sqrt(2)`` the force vanishes during the *approach* phase, so the
clamped model loses contact before reaching maximum compression; that regime is
integrated numerically (``rk4_clamped_restitution``) instead of using theta*.

All quantities are SI: N, m, s, kg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .dem3d import DEM3DConfig, IceDEM3D

__all__ = [
    "ImpactAnalytic",
    "impact_analytic",
    "unclamped_restitution",
    "clamped_restitution",
    "rk4_clamped_restitution",
    "restitution_to_damping_clamped",
    "PairImpactResult",
    "pair_impact",
]

_ZETA_CLAMP_LIMIT = 1.0 / math.sqrt(2.0)


# ===================================================================
# Analytic references
# ===================================================================

def unclamped_restitution(zeta: float) -> float:
    """Restitution of the free (possibly adhesive) spring-dashpot oscillator.

    ``e = exp(-pi*zeta/sqrt(1-zeta**2))``.  This is the identity implemented by
    ``contact_tangential.restitution_to_damping``; it is *not* what a solver
    with ``max(0, F)`` contact realises.
    """
    if zeta < 0:
        raise ValueError("zeta must be nonnegative")
    if zeta == 0.0:
        return 1.0
    if zeta >= 1.0:
        return 0.0  # critically damped or beyond: no rebound
    return math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta ** 2))


def clamped_restitution(zeta: float) -> float:
    """Restitution of the clamped model ``F = max(0, k*delta + c*delta')``.

    Closed form valid for ``0 <= zeta < 1/sqrt(2)``; above that limit the model
    detaches during the approach phase and ``rk4_clamped_restitution`` is used.
    """
    if zeta < 0:
        raise ValueError("zeta must be nonnegative")
    if zeta == 0.0:
        return 1.0
    if zeta >= 1.0:
        return 0.0
    if zeta >= _ZETA_CLAMP_LIMIT:
        return rk4_clamped_restitution(zeta)

    root = math.sqrt(1.0 - zeta ** 2)
    theta_star = math.atan2(2.0 * zeta * root, 2.0 * zeta ** 2 - 1.0)  # in (pi/2, pi]
    decay = math.exp(-zeta * theta_star / root)  # exp(-zeta*omega_n*t*) with t* = theta*/omega_d
    rebound = decay * (math.cos(theta_star) - (zeta / root) * math.sin(theta_star))
    return -rebound


def rk4_clamped_restitution(zeta: float, *, n_steps: int = 200_000, zeta_limit: float = 0.999) -> float:
    """Numerically integrate the clamped 1-DOF impact, in normalised units.

    State ``(delta, delta')`` normalised as ``x = delta*omega_n/v0`` and
    ``u = delta'/v0`` so that ``x' = u`` and ``u' = -2*zeta*u - x`` (time in
    units of ``1/omega_n``).  The clamp removes the force when
    ``x + 2*zeta*u <= 0``; the contact ends when the gap closes again
    (``x <= 0`` after the first compression).  Reference value independent of
    the theta* derivation.
    """
    if zeta < 0 or zeta > zeta_limit:
        raise ValueError(f"zeta must be in [0, {zeta_limit}]")

    x, u = 0.0, 1.0
    # Contact lasts less than one natural period; resolution is far beyond need.
    dt = 2.0 * math.pi / n_steps

    def acc(x: float, u: float) -> float:
        f = x + 2.0 * zeta * u
        return -x - 2.0 * zeta * u if f > 0.0 else 0.0

    compressed = False
    for _ in range(n_steps * 2):
        k1x, k1u = u, acc(x, u)
        k2x, k2u = u + 0.5 * dt * k1u, acc(x + 0.5 * dt * k1x, u + 0.5 * dt * k1u)
        k3x, k3u = u + 0.5 * dt * k2u, acc(x + 0.5 * dt * k2x, u + 0.5 * dt * k2u)
        k4x, k4u = u + dt * k3u, acc(x + dt * k3x, u + dt * k3u)
        x_new = x + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
        u_new = u + dt * (k1u + 2 * k2u + 2 * k3u + k4u) / 6.0
        if x_new <= 0.0 and compressed:
            # Linear interpolation of the zero crossing for the rebound speed.
            frac = x / (x - x_new) if x != x_new else 0.0
            u_end = u + frac * (u_new - u)
            return max(0.0, -u_end)
        compressed = compressed or x_new > 0.0
        x, u = x_new, u_new
    raise RuntimeError("clamped impact did not separate within one natural period")


@dataclass(frozen=True)
class ImpactAnalytic:
    """Analytic impact metrics for one (k, c, m_eff) combination."""

    zeta: float
    omega_n: float
    omega_d: float
    e_unclamped: float
    e_clamped: float
    contact_time_unclamped: float
    contact_time_force: float
    contact_time_geometric: float
    clamp_active: bool
    theta_star: float | None

    @property
    def e_bias(self) -> float:
        """Relative bias of the unclamped mapping: e_clamped/e_unclamped - 1."""
        return self.e_clamped / self.e_unclamped - 1.0


def impact_analytic(k_n: float, c_n: float, m_eff: float, v0: float = 1.0) -> ImpactAnalytic:
    """Closed-form impact metrics of the clamped and unclamped models.

    ``v0`` does not enter any ratio (the model is linear); it is accepted for
    signature symmetry with :func:`pair_impact`.
    """
    if k_n <= 0 or m_eff <= 0:
        raise ValueError("k_n and m_eff must be positive")
    if c_n < 0:
        raise ValueError("c_n must be nonnegative")
    if v0 <= 0:
        raise ValueError("v0 must be positive")

    omega_n = math.sqrt(k_n / m_eff)
    zeta = c_n / (2.0 * math.sqrt(k_n * m_eff))
    if zeta >= 1.0:
        raise ValueError("zeta >= 1: overdamped contact has no rebound to validate")
    omega_d = omega_n * math.sqrt(1.0 - zeta ** 2)

    e_unc = unclamped_restitution(zeta)
    t_unc = math.pi / omega_d
    if zeta >= _ZETA_CLAMP_LIMIT:
        nan = float("nan")
        return ImpactAnalytic(zeta, omega_n, omega_d, e_unc, rk4_clamped_restitution(zeta),
                              t_unc, nan, nan, True, None)

    root = math.sqrt(1.0 - zeta ** 2)
    theta_star = math.atan2(2.0 * zeta * root, 2.0 * zeta ** 2 - 1.0)
    e_clamped = clamped_restitution(zeta)
    decay = math.exp(-zeta * theta_star / root)
    t_force = theta_star / omega_d
    # After the force clamps to zero the pair still overlaps and separates at the
    # constant rebound speed, so the geometric overlap lasts longer:
    #   delta(t*) = (v0/omega_d)*decay*sin(theta*),  |delta'(t*)| = e_clamped*v0
    overlap_at_detach = decay * math.sin(theta_star) / omega_d  # per unit v0
    t_geometric = t_force + overlap_at_detach / e_clamped
    return ImpactAnalytic(
        zeta=zeta,
        omega_n=omega_n,
        omega_d=omega_d,
        e_unclamped=e_unc,
        e_clamped=e_clamped,
        contact_time_unclamped=t_unc,
        contact_time_force=t_force,
        contact_time_geometric=t_geometric,
        clamp_active=True,
        theta_star=theta_star,
    )


def restitution_to_damping_clamped(
    e_target: float,
    k_n: float,
    m_eff: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """Damping coefficient that realises ``e_target`` for the **clamped** model.

    Uses bisection on ``zeta`` (``e_clamped(zeta)`` is strictly decreasing in
    ``zeta``), then ``c = 2*zeta*sqrt(k_n*m_eff)``.  Use this instead of
    ``contact_tangential.restitution_to_damping`` when the produced damping will
    be consumed by a solver whose normal force is truncated at zero.
    """
    if not (0.0 < e_target <= 1.0):
        raise ValueError(f"e_target must be in (0, 1], got {e_target}")
    if k_n <= 0 or m_eff <= 0:
        raise ValueError("k_n and m_eff must be positive")
    if e_target == 1.0:
        return 0.0

    lo, hi = 0.0, 0.999
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if clamped_restitution(mid) > e_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    zeta = 0.5 * (lo + hi)
    return 2.0 * zeta * math.sqrt(k_n * m_eff)


# ===================================================================
# Solver harness: equal-mass head-on pair collision
# ===================================================================

@dataclass(frozen=True)
class PairImpactResult:
    """Solver measurements for one head-on pair impact."""

    e_measured: float
    impact_speed: float
    rebound_speed: float
    contact_steps: int
    contact_time: float
    force_steps: int
    contact_time_force: float
    max_overlap: float
    momentum_error: float
    dt: float
    steps_per_contact: float
    force_law_residual: float
    e_unclamped: float
    e_clamped: float
    contact_time_analytic: float
    contact_time_force_analytic: float

    @property
    def e_error_vs_clamped(self) -> float:
        return self.e_measured / self.e_clamped - 1.0

    @property
    def e_error_vs_unclamped(self) -> float:
        return self.e_measured / self.e_unclamped - 1.0

    @property
    def contact_time_error(self) -> float:
        """Error of the geometric (overlap) contact duration."""
        return self.contact_time / self.contact_time_analytic - 1.0

    @property
    def contact_time_force_error(self) -> float:
        """Error of the force-contact duration, i.e. the time the normal force acts."""
        return self.contact_time_force / self.contact_time_force_analytic - 1.0


def _build_pair(k_n: float, c_n: float, radius: float, density: float, dt_scale: float) -> tuple[IceDEM3D, int, int]:
    """Two free particles facing each other; every other lattice site parked and fixed."""
    cfg = DEM3DConfig(
        nx=2, ny=2, nz=2,
        radius=radius, density=density,
        contact_stiffness=k_n, contact_damping=c_n,
        # The harness injects the dashpot absolutely, i.e. it integrates
        # F_n = max(0, k*delta - c*v_n).  Since the engine's default pair branch
        # derives gamma_n from restitution_coefficient and ignores
        # contact_damping, the convention must be requested explicitly (m2626).
        contact_damping_model="absolute",
        drag=0.0,
        bond_stiffness=1.0, breaking_strain=1e-12, shear_breaking_strain=1e-12,
        fix_x_edges=False, fix_bottom=False,
        tool_gap=1.0e3, tool_speed=1.0e-9,
    )
    dem = IceDEM3D(cfg)
    n = len(dem.positions)
    pos = torch.zeros_like(dem.positions)
    for i in range(n):
        pos[i] = torch.tensor([10.0 * i, 500.0, 500.0], dtype=dem.dtype)
    # 1 um initial clearance (each centre one radius from the mid-plane) keeps the
    # free-flight phase negligible in step count while starting outside contact.
    pos[0] = torch.tensor([0.0, 0.0, -(radius + 1.0e-6)], dtype=dem.dtype)
    pos[1] = torch.tensor([0.0, 0.0, radius + 1.0e-6], dtype=dem.dtype)
    dem.positions = pos.clone()
    dem.initial_positions = pos.clone()
    mask = torch.ones(n, dtype=torch.bool)
    mask[0] = False
    mask[1] = False
    dem.fixed = mask
    # Break every lattice bond before the impact so the pair is free-flying.
    dem.forces()
    if int(dem.alive.sum().item()) != 0:
        raise RuntimeError("bonded lattice did not fully break; harness assumption violated")
    dem.dt = dem.dt * dt_scale
    return dem, 0, 1


def pair_impact(
    *,
    e_target: float = 0.5,
    v0: float = 0.5,
    k_n: float = 5.0e4,
    radius: float = 0.05,
    density: float = 900.0,
    dt_scale: float = 1.0,
    damping: str = "unclamped_mapping",
) -> PairImpactResult:
    """Run an equal-mass head-on impact and measure the realised restitution.

    Parameters
    ----------
    e_target : target coefficient of restitution used to build the damping.
    damping  : ``"unclamped_mapping"`` uses
               ``contact_tangential.restitution_to_damping`` (the mapping under
               review); ``"clamped_mapping"`` uses
               :func:`restitution_to_damping_clamped` (the corrected mapping).
    dt_scale : multiplier on the solver's ``recommended_dt``.

    Returns
    -------
    PairImpactResult with the measured restitution, contact duration, momentum
    drift and the residual of the documented force law.
    """
    from .contact_tangential import restitution_to_damping

    m = density * (4.0 / 3.0) * math.pi * radius ** 3
    m_eff = 0.5 * m
    if damping == "unclamped_mapping":
        c_n = restitution_to_damping(e_target, k_n, m_eff)
    elif damping == "clamped_mapping":
        c_n = restitution_to_damping_clamped(e_target, k_n, m_eff)
    else:
        raise ValueError("damping must be 'unclamped_mapping' or 'clamped_mapping'")

    dem, i, j = _build_pair(k_n, c_n, radius, density, dt_scale)
    dem.velocities = torch.zeros_like(dem.positions)
    dem.velocities[i] = torch.tensor([0.0, 0.0, v0], dtype=dem.dtype)
    dem.velocities[j] = torch.tensor([0.0, 0.0, -v0], dtype=dem.dtype)
    dem.time = 0.0
    dem.step_count = 0

    # Force-law conformance at a prescribed overlap, compared with max(0, k*d - c*vn).
    probe = dem.positions.clone()
    probe[i] = torch.tensor([0.0, 0.0, -(radius - 1.0e-4)], dtype=dem.dtype)
    probe[j] = torch.tensor([0.0, 0.0, radius - 1.0e-4], dtype=dem.dtype)
    saved = dem.positions.clone()
    dem.positions = probe
    f_probe, _ = dem.forces(update_fracture=False)
    overlap = 2.0 * radius - (probe[j, 2] - probe[i, 2])
    vn = float(dem.velocities[j, 2] - dem.velocities[i, 2])
    expected = max(0.0, k_n * float(overlap) - c_n * vn)
    force_law_residual = abs(float(f_probe[i, 2]) + expected) / max(expected, 1e-30)
    dem.positions = saved

    momentum_before = float(dem.velocities[:, 2].sum())
    contact_steps = 0
    force_steps = 0
    max_overlap = 0.0
    for _ in range(400_000):
        v_prev = float(dem.velocities[i, 2])
        dem.step()
        # In free flight the normal force is exactly zero, so the velocity is
        # bit-for-bit unchanged; any change means the contact force was active.
        if float(dem.velocities[i, 2]) != v_prev:
            force_steps += 1
        sep = float(dem.positions[j, 2] - dem.positions[i, 2])
        overlap_now = 2.0 * radius - sep
        if overlap_now > 0.0:
            contact_steps += 1
            max_overlap = max(max_overlap, overlap_now)
        elif contact_steps:
            break
    else:  # pragma: no cover - guarded by the physics
        raise RuntimeError("pair did not separate within the step budget")

    rebound = -float(dem.velocities[i, 2])
    analytic = impact_analytic(k_n, c_n, m_eff, v0)
    return PairImpactResult(
        e_measured=rebound / v0,
        impact_speed=v0,
        rebound_speed=rebound,
        contact_steps=contact_steps,
        contact_time=contact_steps * dem.dt,
        force_steps=force_steps,
        contact_time_force=force_steps * dem.dt,
        max_overlap=max_overlap,
        momentum_error=float(dem.velocities[:, 2].sum()) - momentum_before,
        dt=dem.dt,
        steps_per_contact=contact_steps,
        force_law_residual=force_law_residual,
        e_unclamped=analytic.e_unclamped,
        e_clamped=analytic.e_clamped,
        contact_time_analytic=analytic.contact_time_geometric,
        contact_time_force_analytic=analytic.contact_time_force,
    )