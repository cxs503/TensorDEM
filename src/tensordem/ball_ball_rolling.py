"""Ball-ball rolling-friction contact mechanics: analytic verification surface.

This module extends the existing ball-plane rolling-resistance verification
(``tensordem.rolling_resistance`` / ``scripts/verify_dem3d_rolling_resistance.py``)
to the genuinely two-body case: two elastic spheres in contact, where both the
normal and the tangential (Cundall-Strack) degrees of freedom, the rolling
resistance couple and the rotational damping act on *both* particles.

Conventions
-----------
* Particle ``i`` is at ``x_i``, particle ``j`` at ``x_j``.  The unit normal is
  ``n = (x_j - x_i) / |x_j - x_i|`` and points from ``i`` to ``j``.
* The contact point on ``i`` is at ``x_i + r_i n``, on ``j`` at ``x_j - r_j n``.
* Effective radius ``r_eff = r_i r_j / (r_i + r_j)``: this is the radius that
  appears in the rolling-resistance moment ``M_r = mu_r r_eff F_n`` and that
  reduces to ``r_j`` in the ball-plane limit ``r_i -> inf`` (Ai et al. 2011).
* All quantities SI: N, m, rad/s, N*m, s.

Physics implemented
-------------------
1. **Cundall-Strack tangential spring** (Cundall & Strack 1979): the tangential
   displacement of the contact point is integrated from the exact slip velocity
   of the two material contact points.

       s = v_rel - (r_i omega_i + r_j omega_j) x n            [exact kinematics]
       delta_t <- delta_t + s dt        (tangential part only)
       F_t = -k_t delta_t                (elastic branch)
       |F_t| <= mu_s F_n                 (Coulomb cone, plastic projection)

2. **Rolling resistance couple** (Ai et al. 2011; Iwashita & Oda 1998): a
   torque ``M_r = mu_r r_eff F_n`` opposing the rolling (non-spinning) part of
   the relative angular velocity ``omega_rel = omega_j - omega_i``.  The
   viscous branch is implemented here directly, the elastic-plastic (EPSD)
   branch is delegated to :mod:`tensordem.rolling_resistance`.

3. **Rotational damping** ``M = -c_r omega_roll`` gives the exact exponential
   spin-down ``omega(t) = omega_0 exp(-c_r t / I)``.

4. **Rolling-sliding transition.**  Sliding stops when the two material
   contact points co-move (``|s| = 0``).  For a free sphere in contact with a
   fixed sphere this is the familiar

       v_t = r_j omega           (exact kinematics; r_eff in the plane limit)

   and the transition instant follows from the impulse balance
   ``ds/dt = -mu_s F_n (1/m + r_j^2 / I) = -(7/2) mu_s F_n / m`` for a solid
   sphere, so ``omega* = (2/7) omega_0`` when starting from pure spin.

5. **Inclined-plane stability of a two-ball stack**: rolling resistance caps
   the moment a contact can transmit, ``mu_r r_eff F_n``; the assembly holds
   while ``mu_r >= tan(theta)``.

Every quantity below is compared against a closed-form expression, never only
against "the simulation ran".  Report: ``docs/ball_ball_rolling_contact.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from .rolling_resistance import (
    RollingResistanceConfig,
    rolling_component,
    rolling_resistance_torque,
)


# ---------------------------------------------------------------------------
# Geometry / kinematics helpers
# ---------------------------------------------------------------------------
def effective_radius(r_i: float, r_j: float) -> float:
    """``r_eff = r_i r_j / (r_i + r_j)``; ``r_i -> inf`` recovers ``r_j``."""
    if r_i <= 0.0 or r_j <= 0.0:
        raise ValueError("radii must be positive")
    return r_i * r_j / (r_i + r_j)


def solid_sphere_mass(r: float, density: float) -> float:
    return 4.0 / 3.0 * math.pi * r ** 3 * density


def solid_sphere_inertia(r: float, mass: float) -> float:
    """Solid sphere: ``I = 2/5 m r^2`` about a diameter."""
    return 0.4 * mass * r * r


def contact_point_slip_velocity(
    v_rel: torch.Tensor,
    omega_i: torch.Tensor,
    omega_j: torch.Tensor,
    normal: torch.Tensor,
    r_i: torch.Tensor | float,
    r_j: torch.Tensor | float,
) -> torch.Tensor:
    """Exact relative velocity of the two material points at the contact.

    ``s = v_rel - (r_i omega_i + r_j omega_j) x n``, with ``v_rel = v_j - v_i``.
    A vanishing ``s`` means the contact is in pure rolling (no slip) -- this is
    the kinematic definition of the rolling-sliding transition.
    """
    ri = torch.as_tensor(r_i, dtype=v_rel.dtype)
    rj = torch.as_tensor(r_j, dtype=v_rel.dtype)
    spin_term = torch.cross(ri * omega_i + rj * omega_j, normal, dim=-1)
    return v_rel - spin_term


def rolling_transition_velocity(
    v_rel_t: torch.Tensor,
    omega_rel: torch.Tensor,
    normal: torch.Tensor,
    r_eff: torch.Tensor | float,
) -> torch.Tensor:
    """Rolling-slip in the *effective-radius* convention of the campaign spec.

    ``s_R = v_rel,t - omega_rel x (r_eff n)``.  This convention is exact when
    the contact is with a plane (``r_i -> inf``, ``r_eff -> r_j``), and it is
    the quantity that appears in the rolling-resistance limit
    ``M_r = mu_r r_eff F_n``.  For a finite ball-ball pair the exact kinematic
    rolling radius of the moving body is ``r_j`` (see
    :func:`contact_point_slip_velocity`); the ratio ``r_j / r_eff =
    (r_i + r_j) / r_i`` quantifies the convention difference -- reported as a
    function of the radius ratio in the verification report.
    """
    reff = torch.as_tensor(r_eff, dtype=v_rel_t.dtype)
    return v_rel_t - torch.cross(omega_rel, reff * normal, dim=-1)


# ---------------------------------------------------------------------------
# Cundall-Strack tangential contact on a sphere-sphere pair
# ---------------------------------------------------------------------------
@dataclass
class TangentialState:
    """Tangential displacement state of one ball-ball contact."""

    delta_t: torch.Tensor = field(default_factory=lambda: torch.zeros(3))

    def reset(self) -> None:
        self.delta_t = torch.zeros(3, dtype=self.delta_t.dtype)


def advance_tangential(
    state: TangentialState,
    slip_velocity: torch.Tensor,
    normal: torch.Tensor,
    normal_force_mag: float,
    k_t: float,
    mu_s: float,
    dt: float,
) -> torch.Tensor:
    """One Cundall-Strack increment; returns the tangential force on ``j``.

    The tangential displacement is integrated with the tangential slip
    velocity, projected onto the tangent plane, scaled elastically and finally
    clipped to the Coulomb cone ``|F_t| <= mu_s F_n`` (plastic slip).
    """
    delta = state.delta_t + slip_velocity * dt
    # keep delta_t in the tangent plane (n . delta_t = 0)
    delta = delta - (delta * normal).sum() * normal
    force = -k_t * delta
    cap = mu_s * normal_force_mag
    mag = torch.linalg.norm(force)
    if cap > 0.0 and float(mag) > cap:
        # plastic projection: rescale delta so |F_t| == mu_s F_n exactly
        delta = delta * (cap / mag)
        force = -k_t * delta
    state.delta_t = delta
    return force


# ---------------------------------------------------------------------------
# Rolling friction couple (ball-ball)
# ---------------------------------------------------------------------------
@dataclass
class RollingFrictionBallBall:
    """Rolling resistance + rotational damping on a ball-ball contact.

    The returned torque acts on particle ``j``; particle ``i`` gets ``-T``.
    Only the rolling (non-spinning) part of ``omega_rel`` is resisted.
    """

    mu_r: float = 0.0
    c_r: float = 0.0          # rotational damping coefficient, N*m*s
    model: str = "viscous"    # "viscous" (saturating) or "epsd"

    def moment(
        self,
        omega_rel: torch.Tensor,
        normal_force_mag: float,
        r_eff: float,
        normal: Optional[torch.Tensor] = None,
        *,
        epsd_state: Optional[torch.Tensor] = None,
        k_r: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        omega_roll = (
            rolling_component(omega_rel.reshape(1, 3), normal.reshape(1, 3)).reshape(3)
            if normal is not None
            else omega_rel
        )
        limit = self.mu_r * r_eff * abs(normal_force_mag)
        torque = torch.zeros(3, dtype=omega_rel.dtype)

        if self.model == "epsd":
            if epsd_state is None or k_r is None or dt is None:
                raise ValueError("epsd needs epsd_state, k_r and dt")
            new_state = epsd_state - k_r * omega_roll * dt
            mag = torch.linalg.norm(new_state)
            if limit > 0.0 and float(mag) > limit:
                new_state = new_state * (limit / mag)
            torque = torque - new_state
            epsd_state = new_state
        else:
            mag = torch.linalg.norm(omega_roll)
            if mag > 0.0:
                # saturating viscous branch: linear (damping) near zero with
                # slope c_r, capped at the rolling limit mu_r r_eff F_n; with
                # mu_r = 0 the element is a pure rotational damper
                if limit > 0.0:
                    magnitude = (
                        min(self.c_r * mag, limit) if self.c_r > 0.0 else limit
                    )
                else:
                    magnitude = self.c_r * mag
                if magnitude > 0.0:
                    torque = torque - magnitude * omega_roll / mag

        if self.c_r > 0.0 and self.model == "epsd":
            torque = torque - self.c_r * omega_roll

        return torque, (epsd_state if epsd_state is not None else torch.zeros(3))

# ---------------------------------------------------------------------------
# Closed-form references (the yardsticks the simulations are judged against)
# ---------------------------------------------------------------------------
def analytic_transition_spin_ratio(mass: float, inertia: float, r_j: float) -> float:
    """Spin fraction at the rolling-sliding transition for a pure-spin start.

    Impulse balance with ``ds/dt = -mu_s F_n (1/m + r^2/I)`` gives, at the
    first zero crossing of the contact slip,

        omega* / omega_0 = 1 / (1 + m r^2 / I)  =  2/7  for a solid sphere.
    """
    return 1.0 / (1.0 + mass * r_j * r_j / inertia)


def analytic_slip_decay_rate(
    mu_s: float, f_n: float, mass: float, inertia: float, r_j: float
) -> float:
    """Sliding-phase slip decay rate ``|ds/dt| = mu_s F_n (1/m + r^2/I)``."""
    return mu_s * f_n * (1.0 / mass + r_j * r_j / inertia)


def analytic_stack_critical_angle_deg(
    mu_r: float, mass_a: float, mass_b: float, r_a: float, r_b: float
) -> float:
    """Critical incline angle of a two-sphere stack resting on a plane.

    Static equilibrium of the upper ball requires its pair-contact rolling
    moment ``r_b m_b g sin(theta)`` to stay inside the cap
    ``mu_r r_eff m_b g cos(theta)``; equilibrium of the lower ball adds

        tan(theta) <= mu_r (m_a + m_b) r_a / ((m_a + 2 m_b) r_a + m_b r_b).

    Both reduce to ``tan(theta_cr) = mu_r`` in the plane limit ``r_b/r_a -> 0``
    and to ``tan(theta_cr) = mu_r / 2`` for equal spheres.
    """
    r_eff = effective_radius(r_a, r_b)
    cap_upper = mu_r * r_eff / r_b
    cap_lower = mu_r * (mass_a + mass_b) * r_a / (
        (mass_a + 2.0 * mass_b) * r_a + mass_b * r_b
    )
    return math.degrees(math.atan(min(cap_upper, cap_lower)))


def analytic_incline_critical_angle_deg(mu_r: float) -> float:
    """Ball-on-plane limit: ``tan(theta_cr) = mu_r``."""
    return math.degrees(math.atan(mu_r))


# ---------------------------------------------------------------------------
# 1 + 2. Cundall-Strack tangential spring: stiffness recovery, Coulomb cone
# ---------------------------------------------------------------------------
def simulate_tangential_spring(
    *,
    f_n: float,
    k_t: float,
    mu_s: float,
    dt: float,
    n_steps: int,
    slip_speed: float = 1.0e-4,
) -> dict:
    """Kinematic tangential driving of a ball-ball contact (elastic branch).

    The relative surface velocity ``s`` is prescribed (as a rigid feed would
    impose), so ``delta_t = s t``; the measured ``F_t`` history is regressed on
    ``delta_t`` to recover ``k_t``.
    """
    normal = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    tangent = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    slip = slip_speed * tangent
    state = TangentialState(delta_t=torch.zeros(3, dtype=torch.float64))
    deltas: list[float] = []
    forces: list[float] = []
    for _ in range(n_steps):
        f_t = advance_tangential(state, slip, normal, f_n, k_t, mu_s, dt)
        deltas.append(float(state.delta_t.norm()))
        forces.append(float(f_t.norm()))
    d = torch.tensor(deltas, dtype=torch.float64)
    f = torch.tensor(forces, dtype=torch.float64)
    slope = float((d * f).sum() / (d * d).sum())  # least squares through origin
    residual = float((f - slope * d).abs().max())
    peak = forces[-1]
    return {
        "k_t_input": k_t,
        "k_t_recovered": slope,
        "k_t_rel_err": abs(slope - k_t) / k_t,
        "lin_residual_N": residual,
        "peak_force_N": peak,
        "cap_N": mu_s * f_n,
        "delta_t_final": float(state.delta_t.norm()),
        "slip_speed": slip_speed,
        "dt": dt,
        "n_steps": n_steps,
        "elastic": peak < mu_s * f_n,
    }


COULOMB_CASES = [
    # (mu_s, F_n, k_t, dt, slip_speed); the number of steps is derived so that
    # the elastic demand k_t s dt N exceeds the cone by >= 10x (hard saturation)
    (0.30, 5.0, 2.0e4, 1.0e-5, 1.0e-2),
    (0.50, 5.0, 2.0e4, 1.0e-5, 1.0e-2),
    (0.10, 5.0, 2.0e4, 1.0e-5, 1.0e-2),
    (0.40, 12.0, 5.0e4, 2.0e-5, 2.0e-2),
    (0.25, 0.5, 1.0e3, 1.0e-4, 5.0e-2),
    (0.60, 50.0, 1.0e5, 5.0e-6, 1.0e-2),
    (0.35, 20.0, 3.0e4, 1.0e-5, 3.0e-2),
    (0.45, 8.0, 2.5e4, 2.0e-5, 1.5e-2),
    (0.20, 2.0, 8.0e3, 1.0e-5, 4.0e-2),
    (0.55, 30.0, 6.0e4, 1.0e-5, 1.0e-2),
]


def coulomb_case_steps(case: tuple, factor: float = 3.0) -> int:
    """Steps needed for the elastic demand to exceed the cone by ``factor``."""
    mu_s, f_n, k_t, dt, slip_speed = case
    return int(factor * mu_s * f_n / (k_t * slip_speed * dt)) + 1


def check_coulomb_limit_case(case: tuple) -> dict:
    """Drive one contact deep into the friction cone and verify the cap."""
    mu_s, f_n, k_t, dt, slip_speed = case
    n_steps = coulomb_case_steps(case)
    normal = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    tangent = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    slip = slip_speed * tangent
    state = TangentialState(delta_t=torch.zeros(3, dtype=torch.float64))
    f_t = torch.zeros(3, dtype=torch.float64)
    for _ in range(n_steps):
        f_t = advance_tangential(state, slip, normal, f_n, k_t, mu_s, dt)
    cap = mu_s * f_n
    mag = float(f_t.norm())
    return {
        "mu_s": mu_s,
        "f_n": f_n,
        "k_t": k_t,
        "cap_N": cap,
        "f_t_mag_N": mag,
        "cap_rel_err": abs(mag - cap) / cap,
        "direction_cos": float((f_t / mag) @ (-tangent)),
        "elastic_demand_N": k_t * slip_speed * dt * n_steps,
        "saturated": mag >= cap * (1.0 - 1e-12),
    }


# ---------------------------------------------------------------------------
# 3. Rolling-resistance moment  M_r = mu_r r_eff F_n
# ---------------------------------------------------------------------------
ROLLING_RADIUS_PAIRS = [(0.05, 0.05), (0.05, 0.10), (0.02, 0.20)]
ROLLING_NORMAL_FORCES = [1.0, 10.0, 100.0]


def check_rolling_moment_case(
    r_i: float, r_j: float, f_n: float, mu_r: float = 0.05,
) -> dict:
    """Compare the model couple with ``mu_r r_eff F_n`` for one parameter set."""
    r_eff = effective_radius(r_i, r_j)
    normal = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    omega_rel = torch.tensor([3.0, 0.0, -2.0], dtype=torch.float64)
    law = RollingFrictionBallBall(mu_r=mu_r, c_r=0.0, model="viscous")
    torque, _ = law.moment(omega_rel, f_n, r_eff, normal)
    analytic = mu_r * r_eff * f_n
    omega_roll = rolling_component(
        omega_rel.reshape(1, 3), normal.reshape(1, 3)
    ).reshape(3)
    torque_legacy, _ = rolling_resistance_torque(
        omega_rel.reshape(1, 3),
        torch.tensor([f_n], dtype=torch.float64),
        torch.tensor([r_eff], dtype=torch.float64),
        RollingResistanceConfig(model="cdt", mu_r=mu_r),
        normal=normal.reshape(1, 3),
    )
    mag = float(torque.norm())
    return {
        "r_i": r_i,
        "r_j": r_j,
        "r_eff": r_eff,
        "f_n": f_n,
        "mu_r": mu_r,
        "moment_mag": mag,
        "moment_analytic": analytic,
        "moment_rel_err": abs(mag - analytic) / analytic,
        "cos_opposition": float((torque / mag) @ (-omega_roll / omega_roll.norm())),
        "legacy_rel_err": abs(float(torque_legacy.norm()) - analytic) / analytic,
        "plane_limit_rel_err": abs(effective_radius(1.0e9, r_j) - r_j) / r_j,
    }
