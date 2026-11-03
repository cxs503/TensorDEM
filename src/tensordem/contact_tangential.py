"""Tangential contact friction and restitution-coefficient damping mapping.

Provides:
1. Coulomb-friction tangential spring with saturation  |F_t| <= mu * |F_n| .
   The tangential spring is incremented each step by k_t * delta_t * dt and
   then projected onto the friction cone.  This is the classic Cundall-Strack
   incremental tangential-displacement model (no rolling resistance in this
   module; rolling resistance lives in :mod:`tensordem.rolling_resistance`).

2. Analytical mapping from coefficient of restitution e to normal damping
   coefficient gamma_n for a mass-spring-dashpot collision model:
       gamma_n = 2 * zeta * sqrt(k_n * m_eff)
       zeta = |ln(e)| / sqrt(pi^2 + ln(e)^2)
   where m_eff is the effective mass of the colliding pair and k_n is the
   normal stiffness.  The inverse is documented with unit tests.

3. The *projected* tangential state machine
   (:func:`advance_tangential_contact`).  Three properties distinguish it from
   the stateless stepper (:func:`integrate_tangential_displacement` +
   :func:`compute_tangential_force`):

   * the stored state is the *clamped* displacement ``F_t / k_t``, so sliding
     cannot accumulate an unbounded elastic history (|delta_t| <= delta_s
     always, ``delta_s = mu * |F_n| / k_t``);
   * the state is projected onto the current tangent plane, so a rotating
     contact normal cannot convert tangential history into a normal-direction
     force;
   * the state of a pair whose contact has vanished (``|F_n| = 0``) is reset to
     zero, so re-contact always starts from a virgin contact.

All quantities use SI units.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


# ===================================================================
# Part 1 — Coefficient of restitution -> normal damping mapping
# ===================================================================

def restitution_to_damping(
    e: float,
    k_n: float,
    m_eff: float,
) -> float:
    """Map coefficient of restitution *e* to normal damping coefficient gamma_n.

    Derivation (standard mass-spring-dashpot collision model):
    ---------------------------------------
    For a linear spring-dashpot  m_eff * x_ddot + gamma_n * x_dot + k_n * x = 0
    with initial velocity v0 and x(0)=0, the rebound velocity v1 after one
    contact cycle gives e = |v1| / v0.

    The analytical relation is:
        e = exp(-pi * zeta / sqrt(1 - zeta^2))
    where zeta = gamma_n / (2 * sqrt(k_n * m_eff)) is the damping ratio.

    Solving for zeta:
        ln(e) = -pi * zeta / sqrt(1 - zeta^2)
        Let L = ln(e) (negative for e in (0,1)).
        L^2 = pi^2 * zeta^2 / (1 - zeta^2)
        zeta^2 * (L^2 + pi^2) = L^2
        zeta = |L| / sqrt(pi^2 + L^2)

    Therefore:
        gamma_n = 2 * zeta * sqrt(k_n * m_eff)
                = 2 * |ln(e)| / sqrt(pi^2 + ln(e)^2) * sqrt(k_n * m_eff)

    Special cases:
    - e = 1.0  =>  ln(e) = 0  =>  gamma_n = 0  (undamped, elastic)
    - e -> 0+  =>  ln(e) -> -inf  =>  gamma_n -> 2*sqrt(k_n*m_eff)
                                         (critical damping limit)

    Unit test anchor:
    ---------------
    For m_eff=1.0, k_n=1000.0:
      e=0.5 => gamma_n ≈ 13.626
      e=0.8 => gamma_n ≈ 4.481
      e=1.0 => gamma_n = 0.0

    Args:
        e:     Coefficient of restitution, must be in (0, 1].
        k_n:   Normal stiffness, N/m (positive).
        m_eff: Effective mass of the colliding pair, kg (positive).

    Returns:
        gamma_n: Normal damping coefficient, N·s/m (>= 0).
    """
    if not (0.0 < e <= 1.0):
        raise ValueError(f"e must be in (0, 1], got {e}")
    if k_n <= 0 or m_eff <= 0:
        raise ValueError("k_n and m_eff must be positive")

    if e == 1.0:
        return 0.0

    ln_e = math.log(e)  # negative
    abs_ln_e = -ln_e
    zeta = abs_ln_e / math.sqrt(math.pi ** 2 + abs_ln_e ** 2)
    gamma_n = 2.0 * zeta * math.sqrt(k_n * m_eff)
    return gamma_n


def damping_to_restitution(
    gamma_n: float,
    k_n: float,
    m_eff: float,
) -> float:
    """Inverse of restitution_to_damping: map gamma_n back to e.

    This is useful for verification: round-trip e -> gamma_n -> e should
    recover the original e within floating-point precision.

    Args:
        gamma_n: Normal damping coefficient, N·s/m (>= 0).
        k_n:     Normal stiffness, N/m (positive).
        m_eff:   Effective mass, kg (positive).

    Returns:
        e: Coefficient of restitution in (0, 1].
    """
    if gamma_n < 0:
        raise ValueError("gamma_n must be nonnegative")
    if k_n <= 0 or m_eff <= 0:
        raise ValueError("k_n and m_eff must be positive")

    omega_n = math.sqrt(k_n / m_eff)
    critical = 2.0 * m_eff * omega_n
    zeta = gamma_n / critical  # damping ratio

    if zeta >= 1.0:
        return 0.0  # critically damped or overdamped: no rebound

    e = math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta ** 2))
    return e


# ===================================================================
# Part 2 — Tangential Coulomb-friction spring (Cundall-Strack)
# ===================================================================

@dataclass(frozen=True)
class TangentialContactConfig:
    """Tangential contact model parameters."""
    mu: float = 0.0           # Coulomb friction coefficient
    k_t: Optional[float] = None  # tangential stiffness (defaults to k_n if None)
    rolling_mu: float = 0.0   # legacy placeholder; use tensordem.rolling_resistance


def _validate_tangential_parameters(k_t: float, mu: float) -> None:
    """Reject non-physical stiffness / friction coefficients."""
    if not math.isfinite(k_t) or k_t <= 0.0:
        raise ValueError(f"k_t must be finite and positive, got {k_t}")
    if not math.isfinite(mu) or mu < 0.0:
        raise ValueError(f"mu must be finite and nonnegative, got {mu}")


def saturation_displacement(
    normal_force_mag: torch.Tensor,
    k_t: float,
    mu: float,
) -> torch.Tensor:
    """Cone radius in displacement space, ``delta_s = mu * |F_n| / k_t`` (m).

    ``|delta_t| <= delta_s`` is exactly equivalent to ``|F_t| <= mu*|F_n|`` for
    the linear tangential spring, so this is the only state bound the
    Cundall-Strack integrator has to enforce.
    """
    _validate_tangential_parameters(k_t, mu)
    return mu * normal_force_mag.abs() / k_t


def project_tangential_displacement(
    tangential_disp: torch.Tensor,
    normal_force_mag: torch.Tensor,
    k_t: float,
    mu: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Radially clamp ``tangential_disp`` into the Coulomb cone.

    Returns ``(clamped_disp, sliding)`` where ``clamped_disp`` is the
    physically admissible state (``|clamped_disp| <= delta_s``) and ``sliding``
    flags the rows that had to be clamped.  The clamp is rate-independent and
    preserves the direction of the trial displacement (radial projection), so
    the tangential force stays opposed to the accumulated slip.
    """
    # Work in the displacement dtype: a float32 normal force must not degrade
    # the (float64) state update to float32 precision.
    normal_force_mag = normal_force_mag.to(dtype=tangential_disp.dtype)
    limit = saturation_displacement(normal_force_mag, k_t, mu)
    magnitude = torch.linalg.vector_norm(tangential_disp, dim=1)
    sliding = magnitude > limit
    scale = torch.where(
        sliding,
        limit / magnitude.clamp_min(torch.finfo(tangential_disp.dtype).eps),
        torch.ones_like(magnitude),
    )
    clamped = torch.where(sliding[:, None], tangential_disp * scale[:, None], tangential_disp)
    return clamped, sliding


def reset_tangential_displacement(
    tangential_disp: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Zero the rows of ``tangential_disp`` whose contact is not (or no longer) active."""
    mask = active.to(dtype=tangential_disp.dtype)[:, None]
    return tangential_disp * mask


def tangent_plane_projection(
    vector: torch.Tensor,
    normal: torch.Tensor,
) -> torch.Tensor:
    """Remove the component along the (unit) contact normal."""
    along = (vector * normal).sum(dim=1, keepdim=True)
    return vector - along * normal


def advance_tangential_contact(
    tangential_disp: torch.Tensor,
    relative_velocity: torch.Tensor,
    normal: torch.Tensor,
    normal_force_mag: torch.Tensor,
    k_t: float,
    mu: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the Cundall-Strack tangential state by one time step.

    Projected state machine (all rows vectorised):

    1. **contact alive?** ``|F_n| = 0`` means the contact is gone; its stored
       history is discarded (``delta_t = 0``) so a later re-contact starts from
       a virgin contact instead of inheriting stale slip.
    2. **tangent plane**: the stored displacement is projected onto the current
       tangent plane, ``delta_t -= (delta_t . n) n``.  A contact whose normal
       rotates (curved surface, sliding around an edge) therefore cannot turn
       old tangential history into a spurious normal-direction force.
    3. **increment**: ``delta_t += v_t * dt`` with ``v_t = v - (v . n) n``.
    4. **cone clamp**: ``|delta_t| <= delta_s = mu*|F_n|/k_t``.  The *clamped*
       value is the state that must be stored; keeping the raw trial value is
       the classic bug that makes sliding accumulate unbounded elastic memory.
    5. **force**: ``F_t = k_t * delta_t`` (on particle ``i``; particle ``j``
       receives ``-F_t``).

    Args:
        tangential_disp:   (N, D) stored tangential displacement at the start of
                           the step.
        relative_velocity: (N, D) ``v_j - v_i``.
        normal:            (N, D) unit contact normal from ``i`` to ``j``.
        normal_force_mag:  (N,) magnitude of the normal contact force.
        k_t:               Tangential stiffness, N/m.
        mu:                Coulomb friction coefficient.
        dt:                Time step, s.

    Returns:
        ``(force, disp_new, sliding)`` with shapes ``(N, D)``, ``(N, D)`` and
        ``(N,)``.
    """
    _validate_tangential_parameters(k_t, mu)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    if tangential_disp.shape != relative_velocity.shape:
        raise ValueError("tangential_disp and relative_velocity must have the same shape")

    normal_force_mag = normal_force_mag.to(dtype=tangential_disp.dtype)
    relative_velocity = relative_velocity.to(dtype=tangential_disp.dtype)
    normal = normal.to(dtype=tangential_disp.dtype)
    active = normal_force_mag > 0.0
    disp = reset_tangential_displacement(tangential_disp, active)
    disp = tangent_plane_projection(disp, normal)

    velocity_tangent = tangent_plane_projection(relative_velocity, normal)
    trial = tangent_plane_projection(disp + velocity_tangent * dt, normal)

    disp_new, sliding = project_tangential_displacement(trial, normal_force_mag, k_t, mu)
    force = k_t * disp_new
    force = torch.where(active[:, None], force, torch.zeros_like(force))
    return force, disp_new, sliding & active


def tangential_cycle_dissipated_work(
    mu: float,
    normal_force: float,
    k_t: float,
    amplitude: float,
) -> float:
    """Work dissipated by one closed stick-slip cycle (J).

    For a contact under a constant normal force ``F_n`` whose tangential
    displacement is driven harmonically with amplitude ``a``, the Cundall-Strack
    loop (two stick branches of length ``2*delta_s`` and two sliding branches)
    encloses

        W = 4 * mu * F_n * (a - delta_s),   delta_s = mu * F_n / k_t,

    and the contact stays elastic (``W = 0``) for ``a <= delta_s``.  This is the
    closed-form reference used by the friction-cone verification campaign.
    """
    _validate_tangential_parameters(k_t, mu)
    if not math.isfinite(normal_force) or normal_force < 0.0:
        raise ValueError("normal_force must be finite and nonnegative")
    if not math.isfinite(amplitude) or amplitude < 0.0:
        raise ValueError("amplitude must be finite and nonnegative")
    limit = mu * normal_force / k_t
    if amplitude <= limit:
        return 0.0
    return 4.0 * mu * normal_force * (amplitude - limit)


def compute_tangential_force(
    normal: torch.Tensor,
    tangential_disp: torch.Tensor,
    normal_force_mag: torch.Tensor,
    k_t: float,
    mu: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stateless Coulomb projection of a trial tangential displacement.

    This is the single-step kernel without history handling; it is used as the
    *negative control* in the friction-cone verification campaign (feeding the
    raw, unclamped displacement back in is exactly the defect that
    :func:`advance_tangential_contact` prevents):

        F_t_raw = k_t * delta_t
        |F_t| = min(|F_t_raw|, mu * |F_n|)

    Args:
        normal:            (N, D) contact normal vectors (unit).
        tangential_disp:   (N, D) accumulated tangential displacement.
        normal_force_mag:  (N,) magnitude of normal repulsion force.
        k_t:               Tangential stiffness, N/m.
        mu:                Coulomb friction coefficient.

    Returns:
        ``(tangential_force, updated_tangential_disp)`` each (N, D), with
        ``updated_tangential_disp = F_t / k_t`` the clamped state (so
        ``|updated_tangential_disp| <= mu*|F_n|/k_t`` and
        ``tangential_force == k_t * updated_tangential_disp``).
    """
    _validate_tangential_parameters(k_t, mu)
    # Raw tangential force from spring
    F_t_raw = k_t * tangential_disp  # (N, D)

    # Coulomb limit per contact
    F_limit = mu * normal_force_mag  # (N,)

    # Magnitude of raw tangential force
    F_t_mag = torch.linalg.vector_norm(F_t_raw, dim=1)  # (N,)

    # Saturate
    scale = torch.where(F_t_mag > F_limit, F_limit / F_t_mag.clamp_min(1e-15), torch.ones_like(F_t_mag))
    F_t = F_t_raw * scale[:, None]

    return F_t, F_t / k_t  # clamped displacement is the state that must be stored


def integrate_tangential_displacement(
    tangential_disp: torch.Tensor,
    relative_velocity: torch.Tensor,
    normal: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Increment tangential displacement by the tangential component of
    relative velocity over one time step.

        delta_t_new = delta_t + (v_rel - (v_rel·n)*n) * dt

    No cone clamp and no contact-loss reset is applied here; use
    :func:`advance_tangential_contact` for a physically admissible update.

    Args:
        tangential_disp:   (N, D) current accumulated tangential displacement.
        relative_velocity: (N, D) velocity of particle j minus particle i.
        normal:            (N, D) contact normal (unit, from i to j).
        dt:                Time step, s.

    Returns:
        (N, D) updated tangential displacement.
    """
    # Project relative velocity onto tangential plane
    vn = (relative_velocity * normal).sum(dim=1, keepdim=True)  # (N, 1)
    v_tangent = relative_velocity - vn * normal  # tangential component
    return tangential_disp + v_tangent * dt