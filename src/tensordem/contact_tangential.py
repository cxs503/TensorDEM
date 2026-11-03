"""Tangential contact friction and restitution-coefficient damping mapping.

Provides:
1. Coulomb-friction tangential spring with saturation  |F_t| <= mu * |F_n| .
   The tangential spring is incremented each step by k_t * delta_t * dt and
   then projected onto the friction cone.  This is the classic Cundall-Strack
   incremental tangential-displacement model (no rolling resistance in this
   first version; rolling can be added later as SI-3).

2. Analytical mapping from coefficient of restitution e to normal damping
   coefficient gamma_n for a mass-spring-dashpot collision model:
       gamma_n = 2 * zeta * sqrt(k_n * m_eff)
       zeta = |ln(e)| / sqrt(pi^2 + ln(e)^2)
   where m_eff is the effective mass of the colliding pair and k_n is the
   normal stiffness.  The inverse is documented with unit tests.

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


def compute_tangential_force(
    normal: torch.Tensor,
    tangential_disp: torch.Tensor,
    normal_force_mag: torch.Tensor,
    k_t: float,
    mu: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute tangential (frictional) force with Coulomb saturation.

    This implements the incremental Cundall-Strack tangential spring:
        F_t_raw = k_t * delta_t
        |F_t| = min(|F_t_raw|, mu * |F_n|)

    State projection (required for hysteresis):
    ------------------------------------------
    The sliding part of the tangential history must be forgotten, i.e. the
    stored displacement has to be projected back onto the friction cone
    (``delta_t <- F_t / k_t``) after every increment.  That projection is what
    makes the contact dissipative: reusing the raw displacement as state makes
    the force a single-valued function of the driver, the steady-state loop has
    zero area, and the Coulomb model dissipates exactly zero energy per cycle
    (closed-form anchor ``4 * mu * F_n * (a - mu * F_n / k_t)``, see
    :func:`tangential_cycle_dissipated_work` and
    ``tests/test_tangential_friction_verification.py``).
    This stateless helper does not project: it returns the raw spring force as
    its second value.  For time stepping use :func:`advance_tangential_contact`,
    which projects the stored state (see
    ``docs/dem3d_coulomb_friction_cone_verification.md``).

    Args:
        normal:            (N, D) contact normal vectors (unit).
        tangential_disp:   (N, D) accumulated tangential displacement.
        normal_force_mag:  (N,) magnitude of normal repulsion force.
        k_t:               Tangential stiffness, N/m (positive).
        mu:                Coulomb friction coefficient (>= 0).

    Returns:
        (tangential_force, tangential_disp_next): the saturated force and the
        projected displacement state ``F_t / k_t`` to feed back in on the next
        step, each (N, D), D = 2 or 3.

    Note:
        The second return value is the *projected* state, not the raw spring
        displacement ``k_t * delta_t``.  Feeding an unprojected ``delta_t`` back
        into the next step lets the stored displacement grow without bound while
        the pair keeps sliding, so a later velocity reversal needs an
        unphysically long backtrack before the friction force changes sign and
        the hysteresis loop collapses to zero area.  For full time stepping
        (velocity integration included) use :func:`advance_tangential_contact`,
        which applies the same projection (see
        docs/dem3d_coulomb_friction_cone_verification.md).
    """
    if k_t <= 0:
        raise ValueError(f"k_t must be positive, got {k_t}")
    if mu < 0:
        raise ValueError(f"mu must be nonnegative, got {mu}")

    # Raw tangential force from spring
    F_t_raw = k_t * tangential_disp  # (N, 2)

    # Coulomb limit per contact
    F_limit = mu * normal_force_mag  # (N,)

    # Magnitude of raw tangential force
    F_t_mag = torch.linalg.vector_norm(F_t_raw, dim=1)  # (N,)

    # Saturate
    scale = torch.where(F_t_mag > F_limit, F_limit / F_t_mag.clamp_min(1e-15), torch.ones_like(F_t_mag))
    F_t = F_t_raw * scale[:, None]

    # Project the displacement state onto the friction cone (Cundall-Strack).
    return F_t, F_t / k_t


def tangential_cycle_dissipated_work(
    mu: float,
    normal_force_mag: float,
    k_t: float,
    amplitude: float,
) -> float:
    """Closed-form frictional work dissipated per steady stick-slip cycle.

    For a contact under a constant normal force ``F_n`` whose tangential
    displacement is driven harmonically with amplitude ``a`` (full peak-to-peak
    2a), the Cundall-Strack / Coulomb loop is a parallelogram in the
    (delta, F_t) plane:

      - sticking branches of slope ``k_t`` up to ``|F_t| = mu * F_n``,
      - friction plateaus at ``+-mu * F_n`` while sliding.

    The hysteresis loop encloses the area

        W = 4 * mu * F_n * (a - mu * F_n / k_t)      for  a >= mu * F_n / k_t

    and ``W = 0`` below the shakedown (pure-stick) amplitude
    ``a_stick = mu * F_n / k_t``.  This is the analytic anchor used by
    ``tests/test_tangential_friction_verification.py``.

    Args:
        mu:                Coulomb friction coefficient (>= 0).
        normal_force_mag:  Normal force magnitude, N (>= 0).
        k_t:               Tangential stiffness, N/m (positive).
        amplitude:         Tangential displacement amplitude, m (>= 0).

    Returns:
        Dissipated work per cycle, J (>= 0).
    """
    if k_t <= 0:
        raise ValueError(f"k_t must be positive, got {k_t}")
    if mu < 0 or normal_force_mag < 0 or amplitude < 0:
        raise ValueError("mu, normal_force_mag and amplitude must be nonnegative")

    a_stick = mu * normal_force_mag / k_t
    if amplitude <= a_stick:
        return 0.0
    return 4.0 * mu * normal_force_mag * (amplitude - a_stick)


def integrate_tangential_displacement(
    tangential_disp: torch.Tensor,
    relative_velocity: torch.Tensor,
    normal: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Increment tangential displacement by the tangential component of
    relative velocity over one time step.

        delta_t_new = delta_t + (v_rel - (v_rel·n)*n) * dt

    Args:
        tangential_disp:   (N, 2) current accumulated tangential displacement.
        relative_velocity: (N, 2) velocity of particle j minus particle i.
        normal:            (N, 2) contact normal (unit, from i to j).
        dt:                Time step, s.

    Returns:
        (N, 2) updated tangential displacement.
    """
    # Project relative velocity onto tangential plane
    vn = (relative_velocity * normal).sum(dim=1, keepdim=True)  # (N, 1)
    v_tangent = relative_velocity - vn * normal  # tangential component
    return tangential_disp + v_tangent * dt


# ===================================================================
# Part 3 — Coulomb friction cone with persistent (projected) state
# ===================================================================
#
# The stateless functions in Part 2 clamp the *force* but keep the raw spring
# displacement.  Correct Cundall-Strack (1979) sliding requires the *state* to be
# projected onto the Coulomb cone after each increment:
#
#     delta_t <- delta_t + v_t * dt
#     |delta_t| <= delta_s,      delta_s = mu * |F_n| / k_t        (cone radius)
#     F_t = k_t * delta_t
#
# The projected state makes sliding history-consistent: while a contact slides,
# the elastic part of the tangential displacement stays at delta_s, so reversing
# the relative velocity re-sticks the contact after the tangential travel
#
#     s_zero = delta_s            (force passes through zero)
#     s_neg  = 2 * delta_s        (force reaches -mu*|F_n|)
#
# These two lengths are the analytic anchors used by
# scripts/verify_coulomb_friction_cone.py.

def saturation_displacement(
    normal_force_mag: torch.Tensor,
    k_t: float,
    mu: float,
) -> torch.Tensor:
    """Coulomb cone radius in displacement space: ``delta_s = mu * |F_n| / k_t``.

    Args:
        normal_force_mag: (N,) magnitude of the normal contact force.
        k_t:              Tangential stiffness, N/m (positive).
        mu:               Coulomb friction coefficient (>= 0).

    Returns:
        (N,) tangential displacement at sliding onset, m.
    """
    if k_t <= 0:
        raise ValueError(f"k_t must be positive, got {k_t}")
    if mu < 0:
        raise ValueError(f"mu must be nonnegative, got {mu}")
    return mu * normal_force_mag / k_t


def project_tangential_displacement(
    tangential_disp: torch.Tensor,
    normal_force_mag: torch.Tensor,
    k_t: float,
    mu: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project stored tangential displacement onto the Coulomb cone.

    Args:
        tangential_disp:  (N, D) accumulated tangential displacement (D = 2 or 3).
        normal_force_mag: (N,) normal force magnitude (>= 0).
        k_t:              Tangential stiffness, N/m.
        mu:               Coulomb friction coefficient.

    Returns:
        (projected_disp, sliding_mask): (N, D) and boolean (N,).
        ``sliding_mask`` is True where the increment had to be truncated.
    """
    disp = torch.as_tensor(tangential_disp)
    limit = saturation_displacement(torch.as_tensor(normal_force_mag).to(disp.dtype), k_t, mu)
    mag = torch.linalg.vector_norm(disp, dim=1)
    sliding = mag > limit
    tiny = torch.finfo(disp.dtype).tiny
    scale = torch.where(sliding, limit / mag.clamp_min(tiny), torch.ones_like(mag))
    return disp * scale[:, None], sliding


def reset_tangential_displacement(
    tangential_disp: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Zero the stored tangential displacement for inactive (new or lost) contacts.

    A freshly formed contact must start with ``delta_t = 0``; carrying the old
    history over is a silent source of spurious tangential forces.

    Args:
        tangential_disp: (N, D) stored tangential displacement.
        active:          (N,) boolean mask, True where the contact exists.

    Returns:
        (N, D) displacement with inactive rows zeroed.
    """
    disp = torch.as_tensor(tangential_disp)
    active = torch.as_tensor(active).to(torch.bool)
    return torch.where(active[:, None], disp, torch.zeros_like(disp))


def advance_tangential_contact(
    tangential_disp: torch.Tensor,
    relative_velocity: torch.Tensor,
    normal: torch.Tensor,
    normal_force_mag: torch.Tensor,
    k_t: float,
    mu: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One friction-cone-correct Cundall-Strack time increment.

    Performs ``delta_t += v_t * dt`` and then projects the stored state onto
    ``|delta_t| <= mu * |F_n| / k_t``.  Contacts with zero normal force are
    treated as inactive: their stored displacement is reset to zero and no
    tangential force is reported.

    Args:
        tangential_disp:  (N, D) stored tangential displacement from the previous step.
        relative_velocity: (N, D) relative velocity of the pair.
        normal:           (N, D) unit contact normal.
        normal_force_mag: (N,) normal force magnitude (>= 0).
        k_t:              Tangential stiffness, N/m.
        mu:               Coulomb friction coefficient.
        dt:               Time step, s.

    Returns:
        (tangential_force, updated_tangential_disp, sliding_mask)
        with shapes (N, D), (N, D), (N,).
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    disp = torch.as_tensor(tangential_disp)
    f_n = torch.as_tensor(normal_force_mag).to(disp.dtype)
    active = f_n > 0

    disp_active = reset_tangential_displacement(disp, active)
    disp_raw = integrate_tangential_displacement(disp_active, relative_velocity, normal, dt)
    disp_new, sliding = project_tangential_displacement(disp_raw, f_n, k_t, mu)
    disp_new = reset_tangential_displacement(disp_new, active)
    sliding = sliding & active
    return k_t * disp_new, disp_new, sliding
