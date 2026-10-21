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
    rolling_mu: float = 0.0   # rolling-resistance coefficient (placeholder, not yet used)


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

    Args:
        normal:            (N, 2) contact normal vectors (unit).
        tangential_disp:   (N, 2) accumulated tangential displacement.
        normal_force_mag:  (N,) magnitude of normal repulsion force.
        k_t:               Tangential stiffness, N/m.
        mu:                Coulomb friction coefficient.

    Returns:
        (tangential_force, updated_tangential_disp) each (N, 2).
    """
    # Raw tangential force from spring
    F_t_raw = k_t * tangential_disp  # (N, 2)

    # Coulomb limit per contact
    F_limit = mu * normal_force_mag  # (N,)

    # Magnitude of raw tangential force
    F_t_mag = torch.linalg.vector_norm(F_t_raw, dim=1)  # (N,)

    # Saturate
    scale = torch.where(F_t_mag > F_limit, F_limit / F_t_mag.clamp_min(1e-15), torch.ones_like(F_t_mag))
    F_t = F_t_raw * scale[:, None]

    return F_t, F_t_raw  # return raw as the "updated" disp proxy (displacement = F_t_raw / k_t)


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
