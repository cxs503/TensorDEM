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
# ===================================================================
# Part 4 — Mindlin spring-damper-slip history model (loading/unloading)
# ===================================================================
#
# Why a second tangential model
# -----------------------------
# Cundall-Strack (Part 3) uses a *constant* tangential stiffness and a Coulomb
# cap on the stored displacement.  For a granular assembly used as a macroscopic
# anchor (angle of repose, packing fraction) that is not enough: the contact
# needs (i) the Mindlin elastic-plastic tangential backbone, so that the
# incremental stiffness softens as the contact approaches gross slip,
# (ii) an explicit unloading branch, so that a load reversal unloads elastically
# with the no-slip stiffness instead of ratcheting, (iii) a tangential dashpot,
# and (iv) per-contact memory of the tangential displacement increment.
#
# Model (incremental, vectorial, closed-loop)
# ------------------------------------------
# State per contact: tangential force ``F``, elastic anchor ``(delta_r, F_r)``
# and the accumulated tangential displacement ``delta``.
#
#   delta     <- delta + v_t * dt                       (memory increment)
#   elastic predictor  F* = F_r + k_t0 (delta - delta_r)
#   envelope           env(|delta|) = min( mu_s |F_n| , backbone(|delta|) )
#   if |F*| > env : F = env * F*/|F*|,  delta_r = delta, F_r = F   (plastic)
#   else          : F = F*
#   output force       F_out = F - gamma_t * v_t          (Mindlin dashpot)
#
# backbone "linear"   : backbone(x) = k_t0 x            (exactly Cundall-Strack)
# backbone "mindlin"  : backbone(x) = mu_s |F_n| [1 - (1 - min(x/d*,1))^{3/2}]
#                       d* = 3 mu_s |F_n| / (2 k_t0)     (Mindlin-Deresiewicz
#                       virgin loading curve, see Johnson, Contact Mechanics,
#                       §4.2; the initial slope is k_t0 and the asymptote is
#                       the Coulomb cap)
#
# Properties that are machine-checked (tests/test_contact_mindlin_history.py):
#   * "linear" + gamma_t = 0 reproduces advance_tangential_contact exactly;
#   * virgin loading follows the closed-form backbone;
#   * unloading from any state has slope exactly k_t0 (no ratcheting);
#   * the steady stick-slip loop is CLOSED and dissipates the analytic
#     Cundall-Strack value 4 mu_s F_n (a - mu_s F_n / k_t0) per cycle --
#     the same anchor already used by tests/test_tangential_friction_verification;
#   * the cumulative work over a closed cycle is negative semidefinite;
#   * gamma_t > 0 dissipates exactly gamma_t * integral|v_t|^2 dt.

BACKBONES = ("linear", "mindlin")


@dataclass(frozen=True)
class MindlinTangentialConfig:
    """Mindlin-type tangential contact parameters (all SI).

    Attributes:
        k_t0:             No-slip tangential stiffness at contact onset, N/m.
        mu_s:             Coulomb (sliding) friction coefficient.
        backbone:         ``"linear"`` or ``"mindlin"`` (virgin backbone shape).
        gamma_t:          Tangential dashpot coefficient, N*s/m (0 = off).
        unloading_ratio:  Elastic unloading stiffness multiplier.  ``1.0`` is
                          the physically conservative Mindlin elastic branch;
                          values > 1 select a Masing-type stiffening unloading
                          branch and are flagged experimental.
    """

    k_t0: float
    mu_s: float = 0.5
    backbone: str = "mindlin"
    gamma_t: float = 0.0
    unloading_ratio: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.k_t0) or self.k_t0 <= 0.0:
            raise ValueError(f"k_t0 must be finite and positive, got {self.k_t0}")
        if not math.isfinite(self.mu_s) or self.mu_s < 0.0:
            raise ValueError(f"mu_s must be finite and nonnegative, got {self.mu_s}")
        if self.backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {BACKBONES}, got {self.backbone!r}")
        if not math.isfinite(self.gamma_t) or self.gamma_t < 0.0:
            raise ValueError(f"gamma_t must be finite and nonnegative, got {self.gamma_t}")
        if not math.isfinite(self.unloading_ratio) or self.unloading_ratio < 1.0:
            raise ValueError(
                f"unloading_ratio must be finite and >= 1, got {self.unloading_ratio}")

    @property
    def enabled(self) -> bool:
        """False when friction is switched off (mu_s = 0): no tangential force."""
        return self.mu_s > 0.0


def mindlin_backbone(
    delta_mag: torch.Tensor,
    normal_force_mag: torch.Tensor,
    config: MindlinTangentialConfig,
) -> torch.Tensor:
    """Elastic-plastic tangential envelope ``env(|delta|)`` (magnitude, N).

    ``"linear"``: ``k_t0 |delta|`` capped by ``mu_s |F_n|``.
    ``"mindlin"``: ``mu_s |F_n| [1 - (1 - x)^{3/2}]`` with ``x = |delta|/d*``
    clamped to unity and ``d* = 3 mu_s |F_n| / (2 k_t0)``.

    Args:
        delta_mag:        (N,) magnitude of the accumulated tangential displacement.
        normal_force_mag: (N,) normal force magnitude (>= 0).
        config:           :class:`MindlinTangentialConfig`.

    Returns:
        (N,) envelope magnitude.
    """
    delta_mag = torch.as_tensor(delta_mag)
    f_n = torch.as_tensor(normal_force_mag).to(delta_mag.dtype)
    cap = config.mu_s * f_n
    if config.backbone == "linear":
        return torch.minimum(config.k_t0 * delta_mag, cap)
    tiny = torch.finfo(delta_mag.dtype).tiny
    d_star = 3.0 * cap / (2.0 * config.k_t0)
    x = (delta_mag / d_star.clamp_min(tiny)).clamp(0.0, 1.0)
    return cap * (1.0 - (1.0 - x).clamp_min(0.0).pow(1.5))


def mindlin_tangential_step(
    tangential_force: torch.Tensor,
    tangential_disp: torch.Tensor,
    anchor_force: torch.Tensor,
    anchor_disp: torch.Tensor,
    tangential_velocity: torch.Tensor,
    normal_force_mag: torch.Tensor,
    config: MindlinTangentialConfig,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One Mindlin tangential increment with elastic unloading and slip limit.

    Args:
        tangential_force:  (N, D) stored tangential force at the last committed step.
        tangential_disp:   (N, D) accumulated tangential displacement (memory).
        anchor_force:      (N, D) elastic anchor force ``F_r``.
        anchor_disp:       (N, D) elastic anchor displacement ``delta_r``.
        tangential_velocity: (N, D) contact-point slip velocity (normal part removed).
        normal_force_mag:  (N,) normal force magnitude (>= 0).  Contacts with
                           ``F_n == 0`` are inactive: state is zeroed.
        config:            :class:`MindlinTangentialConfig`.
        dt:                Time step, s.

    Returns:
        ``(force_out, force_state, disp_state, anchor_force, anchor_disp,
        sliding_mask)`` where ``force_out = force_state - gamma_t * v_t`` is the
        force to apply, the following four are the state to carry forward and
        ``sliding_mask`` flags gross slip (envelope saturation).
    """
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    F = torch.as_tensor(tangential_force)
    delta = torch.as_tensor(tangential_disp)
    F_r = torch.as_tensor(anchor_force)
    d_r = torch.as_tensor(anchor_disp)
    v_t = torch.as_tensor(tangential_velocity)
    f_n = torch.as_tensor(normal_force_mag).to(F.dtype)

    if F.shape != delta.shape or F.shape != F_r.shape or F.shape != d_r.shape:
        raise ValueError("force and displacement state tensors must share one shape")
    if v_t.shape != F.shape:
        raise ValueError("tangential_velocity must match the state shape")
    if f_n.shape != (F.shape[0],):
        raise ValueError("normal_force_mag must have shape (N,)")

    active = f_n > 0.0
    keep = active.to(F.dtype)[:, None]
    # A separated contact forgets its history entirely.
    F = F * keep
    delta = delta * keep
    F_r = F_r * keep
    d_r = d_r * keep

    delta_new = delta + v_t * dt
    k_unload = config.k_t0 * config.unloading_ratio
    # Elastic predictor from the anchor: the tangential force OPPOSES the
    # tangential displacement increment (F = F_r - k (delta - delta_r)), which
    # is the Cundall-Strack sign convention (F = -k delta).
    trial = F_r - k_unload * (delta_new - d_r)

    delta_mag = torch.linalg.vector_norm(delta_new, dim=1)
    env = mindlin_backbone(delta_mag, f_n, config) if config.enabled else torch.zeros_like(delta_mag)
    trial_mag = torch.linalg.vector_norm(trial, dim=1)
    tiny = torch.finfo(F.dtype).tiny
    sliding = (trial_mag > env) & active
    scale = torch.where(sliding, env / trial_mag.clamp_min(tiny), torch.ones_like(trial_mag))
    F_new = trial * scale[:, None]
    # Plastic flow moves the anchor onto the yield surface (kinematic hardening).
    F_r_new = torch.where(sliding[:, None], F_new, F_r)
    d_r_new = torch.where(sliding[:, None], delta_new, d_r)

    # Inactive contacts have no force and no memory.
    F_new = F_new * keep
    F_r_new = F_r_new * keep
    d_r_new = d_r_new * keep
    delta_new = delta_new * keep

    force_out = F_new - config.gamma_t * v_t
    force_out = force_out * keep
    return force_out, F_new, delta_new, F_r_new, d_r_new, sliding & active


# ===================================================================
# Part 5 — SI-3 rolling-resistance mount
# ===================================================================
#
# ``rolling_resistance`` is the single implementation of the rolling laws
# (cdt / viscous / epsd).  This mount is the SI-3 attachment point on the
# tangential contact: callers that already own the tangential contact only need
# to hand in the relative spin, the normal and |F_n|.

def rolling_resistance_si3(
    omega_rel: torch.Tensor,
    normal: torch.Tensor,
    normal_force_mag: torch.Tensor,
    r_eff: torch.Tensor,
    *,
    mu_r: float,
    model: str = "epsd",
    k_r: float = 0.0,
    dt: float | None = None,
    torque_state: torch.Tensor | None = None,
    omega_tol: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rolling-resistance torque from the shared ``rolling_resistance`` module.

    Implements ``M_r <= mu_r r_eff |F_n|`` (Ai et al. 2011 convention) and
    routes to :func:`tensordem.rolling_resistance.rolling_resistance_torque` so
    there is exactly one rolling model in the code base.

    Args:
        omega_rel:        (N, 3) relative spin ``omega_j - omega_i``.
        normal:           (N, 3) unit contact normal (i -> j).
        normal_force_mag: (N,) normal force magnitude.
        r_eff:            (N,) effective radius ``r_i r_j / (r_i + r_j)``.
        mu_r:             Rolling-resistance coefficient (Luding prior 0.05).
        model:            ``"epsd"`` (default, history-bearing), ``"cdt"`` or
                          ``"viscous"``.
        k_r:              Rolling spring stiffness, N*m/rad (``"epsd"`` only).
        dt:               Time step, s (``"epsd"`` only).
        torque_state:     (N, 3) EPSD elastic torque state from the last step.
        omega_tol:        Dead-band for ``"cdt"``.

    Returns:
        ``(torque_on_j, new_torque_state, dissipated_power)``.  The torque on
        ``i`` is ``-torque_on_j``; the dissipated power (<= 0) is
        ``torque_on_j . omega_rel``.
    """
    from . import rolling_resistance as _rr

    config = _rr.RollingResistanceConfig(
        model=model, mu_r=mu_r, k_r=k_r, omega_tol=omega_tol)
    torque_j, state = _rr.rolling_resistance_torque(
        omega_rel, normal_force_mag, r_eff, config,
        normal=normal, torque_state=torque_state, dt=dt)
    power = (torque_j * omega_rel).sum(dim=1)
    return torque_j, state, power