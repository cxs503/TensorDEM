"""Restitution-coefficient mapping for TensorDEM's *unilateral* contact law.

Why this module exists
----------------------
The contact law actually implemented by ``tensordem.dem3d`` (sphere-sphere
contact, tool contact and platen contact all share one form) is

    F_n = max(0, k_n * delta + c_n * delta_dot)          (delta > 0 = overlap)

i.e. a linear spring in parallel with a linear dashpot whose total force is
**clipped at zero**.  The clip makes the law *unilateral*: the dashpot cannot
pull the two bodies together, so the contact opens as soon as the spring force
is overcome by the (tensile) dashpot term.

``tensordem.contact_tangential.restitution_to_damping`` maps a target
coefficient of restitution ``e`` to ``c_n`` using the classical *bilateral*
mass-spring-dashpot result (Cundall & Strack 1979; Johnson, *Contact
Mechanics*, 1985):

    zeta = c_n / (2 sqrt(k_n m_eff)),   e = exp(-pi zeta / sqrt(1 - zeta^2))

That relation assumes the dashpot keeps acting in tension until the linear
oscillator returns to ``delta = 0``.  It is therefore **not** the response of
the clipped law implemented by this solver: for the same ``c_n`` the clipped
law rebounds with a substantially higher velocity (and it still rebounds at
all where the bilateral model predicts full sticking).  The two branches are
kept here under explicit names so that the difference can be audited instead
of being silently mixed up.

Contents
--------
* :func:`effective_mass` -- two-body effective mass (sphere/sphere) or the
  single-particle mass for a contact against an infinitely massive partner
  (tool, platen).
* :func:`unilateral_restitution` / :func:`unilateral_contact_detail` --
  high-accuracy reference solution of the clipped-law impact problem
  (RK4 with event localisation).  This is the *reference* the engine is
  verified against.
* :func:`damping_for_restitution` -- inverse of the reference solution:
  choose ``c_n`` so that the engine, not the bilateral formula, reproduces a
  target ``e``.
* :func:`classical_restitution` / :func:`classical_damping` -- the bilateral
  textbook relation, re-exported for comparison (delegates to
  ``contact_tangential`` so there is a single implementation).

Scope
-----
Both bodies are treated as rigid spheres with a linear (Hookean) contact;
there is no Hertzian non-linearity, no adhesion, no tangential/rotational
branch and no rate-dependent friction.  Restitution is computed for a
single isolated impact; multiple simultaneous contacts (as in a bonded
lattice or a granular assembly) change the effective mass and are out of
scope.  The result is a statement about *implementation fidelity* against the
continuum contact law, not about the physical restitution of ice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .contact_tangential import damping_to_restitution, restitution_to_damping

__all__ = [
    "ImpactSolution",
    "effective_mass",
    "unilateral_contact_detail",
    "unilateral_restitution",
    "damping_for_restitution",
    "classical_restitution",
    "classical_damping",
]

# Samples per contact period used by the reference integrator.  RK4 error is
# O(h^4); 4000 samples per period gives e to ~1e-9 (checked against a doubled
# resolution in the test suite).
_DEFAULT_SAMPLES = 4000


def effective_mass(mass_a: float, mass_b: Optional[float] = None) -> float:
    """Two-body effective mass used in the contact oscillator.

    ``mass_b = None`` (or ``inf``) means the partner is infinitely massive
    (prescribed tool, rigid platen, wall), in which case only ``mass_a``
    accelerates and ``m_eff = mass_a``.

    For two equal spheres ``m_eff = m / 2``: the pair force acts with opposite
    signs on both bodies, so the relative coordinate obeys a single oscillator
    with the reduced mass.
    """
    if not math.isfinite(mass_a) or mass_a <= 0:
        raise ValueError(f"mass_a must be finite and positive, got {mass_a}")
    if mass_b is None or math.isinf(mass_b):
        return mass_a
    if not math.isfinite(mass_b) or mass_b <= 0:
        raise ValueError(f"mass_b must be finite and positive, got {mass_b}")
    return mass_a * mass_b / (mass_a + mass_b)


@dataclass(frozen=True)
class ImpactSolution:
    """Reference solution of one isolated impact of the clipped contact law.

    Attributes
    ----------
    restitution:
        ``e = |v_out| / |v_in|`` across the contact episode.
    duration:
        Time the two bodies spend in contact (s).
    peak_overlap:
        Maximum penetration ``delta`` reached (m).
    detached:
        ``True`` when the contact opened because the clipped dashpot force
        reached zero while ``delta > 0`` (the characteristic unilateral
        event), ``False`` when the overlap returned to zero while the force
        was still positive.
    """

    restitution: float
    duration: float
    peak_overlap: float
    detached: bool


def _period(stiffness: float, m_eff: float) -> float:
    return 2.0 * math.pi * math.sqrt(m_eff / stiffness)


def unilateral_contact_detail(
    damping: float,
    stiffness: float,
    m_eff: float,
    impact_speed: float = 1.0,
    samples_per_period: int = _DEFAULT_SAMPLES,
) -> ImpactSolution:
    """Integrate one impact of ``F = max(0, k*delta + c*delta_dot)`` accurately.

    State variables are the overlap ``delta >= 0`` and its rate
    ``u = d(delta)/dt`` (``u > 0`` while the bodies approach).  The contact
    equation of motion for the relative coordinate is

        d(delta)/dt = u
        du/dt       = -max(0, k*delta + c*u) / m_eff

    Integration is classical RK4 with fixed step ``T/samples_per_period``
    (``T = 2*pi*sqrt(m_eff/k)``); the end of contact is located by linear
    interpolation inside the last step.  Two end conditions are handled:

    * overlap returns to zero (force still positive) -- bilateral-like exit;
    * the clipped force would become negative while separating -- the bodies
      detach at ``delta > 0`` and then travel freely.

    Because the law is linear and the clip is a sign condition, the result is
    independent of ``impact_speed``; the argument only sets the velocity scale
    used by the stopping condition.
    """
    if not math.isfinite(damping) or damping < 0:
        raise ValueError(f"damping must be finite and nonnegative, got {damping}")
    if not math.isfinite(stiffness) or stiffness <= 0:
        raise ValueError(f"stiffness must be finite and positive, got {stiffness}")
    if not math.isfinite(m_eff) or m_eff <= 0:
        raise ValueError(f"effective mass must be finite and positive, got {m_eff}")
    if not math.isfinite(impact_speed) or impact_speed <= 0:
        raise ValueError(f"impact_speed must be finite and positive, got {impact_speed}")
    if samples_per_period < 100:
        raise ValueError("samples_per_period must be >= 100")

    period = _period(stiffness, m_eff)
    h = period / samples_per_period
    # Generous step budget: the clipped law opens earlier than the bilateral
    # oscillator, but stays well inside two periods for any finite damping.
    max_steps = 4 * samples_per_period

    def derivative(overlap: float, rate: float) -> tuple[float, float]:
        force = stiffness * overlap + damping * rate
        if force <= 0.0:  # clipped dashpot: no tensile force
            return rate, 0.0
        return rate, -force / m_eff

    overlap = 0.0
    rate = impact_speed
    peak = 0.0
    for _ in range(max_steps):
        k1_o, k1_r = derivative(overlap, rate)
        k2_o, k2_r = derivative(overlap + 0.5 * h * k1_o, rate + 0.5 * h * k1_r)
        k3_o, k3_r = derivative(overlap + 0.5 * h * k2_o, rate + 0.5 * h * k2_r)
        k4_o, k4_r = derivative(overlap + h * k3_o, rate + h * k3_r)
        new_overlap = overlap + h / 6.0 * (k1_o + 2 * k2_o + 2 * k3_o + k4_o)
        new_rate = rate + h / 6.0 * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)

        # Detachment: separating and the clipped force has reached zero.
        if new_rate < 0.0 and (stiffness * new_overlap + damping * new_rate) <= 0.0:
            return ImpactSolution(
                restitution=abs(new_rate) / impact_speed,
                duration=(_ + 1) * h,
                peak_overlap=peak,
                detached=True,
            )
        # Geometric separation.
        if new_overlap <= 0.0 and new_rate < 0.0:
            # Linear interpolation on the overlap to land exactly at zero.
            frac = overlap / (overlap - new_overlap) if overlap > new_overlap else 1.0
            exit_rate = rate + frac * (new_rate - rate)
            return ImpactSolution(
                restitution=abs(exit_rate) / impact_speed,
                duration=(_ + frac) * h,
                peak_overlap=peak,
                detached=False,
            )
        overlap, rate = new_overlap, new_rate
        peak = max(peak, overlap)
    raise RuntimeError("contact did not open within the step budget")  # pragma: no cover


def unilateral_restitution(
    damping: float,
    stiffness: float,
    m_eff: float,
    impact_speed: float = 1.0,
    samples_per_period: int = _DEFAULT_SAMPLES,
) -> float:
    """Reference restitution ``e`` of the clipped (unilateral) contact law."""
    return unilateral_contact_detail(
        damping, stiffness, m_eff, impact_speed, samples_per_period
    ).restitution


def damping_for_restitution(
    e_target: float,
    stiffness: float,
    m_eff: float,
    impact_speed: float = 1.0,
    damping_upper: float = 1e6,
    rel_tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    """Invert the *engine* contact law: choose ``c_n`` for a target ``e``.

    ``unilateral_restitution`` is monotonically decreasing in ``c_n`` on
    ``[0, inf)`` (more dashpot always removes more energy), so a bisection on
    ``c_n`` converges robustly.  Use this instead of
    ``contact_tangential.restitution_to_damping`` whenever the damping is fed
    to ``DEM3DConfig.contact_damping`` / ``platen_damping`` or to the 2-D
    ``DEMConfig.contact_damping``.

    Returns the damping coefficient ``c_n`` in N*s/m.  Raises ``ValueError`` if
    ``e_target`` is outside ``(0, 1]`` or unreachable within
    ``damping_upper``.
    """
    if not (0.0 < e_target <= 1.0):
        raise ValueError(f"e_target must be in (0, 1], got {e_target}")
    if not math.isfinite(stiffness) or stiffness <= 0:
        raise ValueError(f"stiffness must be finite and positive, got {stiffness}")
    if not math.isfinite(m_eff) or m_eff <= 0:
        raise ValueError(f"effective mass must be finite and positive, got {m_eff}")
    if damping_upper <= 0:
        raise ValueError("damping_upper must be positive")

    def residual(c: float) -> float:
        return unilateral_restitution(c, stiffness, m_eff, impact_speed) - e_target

    # e(c -> 0) -> 1 from below; grow the bracket until it brackets the target.
    lo, hi = 0.0, None
    c = max(1e-12, 0.01 * math.sqrt(stiffness * m_eff))
    while c <= damping_upper:
        if residual(c) <= 0.0:
            hi = c
            break
        lo = c
        c *= 2.0
    if hi is None:
        raise ValueError(
            f"e_target={e_target} not reachable with damping <= {damping_upper:g}"
        )

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= rel_tol * max(hi, 1e-30):
            break
    return 0.5 * (lo + hi)


def classical_restitution(damping: float, stiffness: float, m_eff: float) -> float:
    """Bilateral (textbook) restitution -- *not* the engine's contact law.

    Kept as the comparison branch of the audit; see the module docstring.
    """
    return damping_to_restitution(damping, stiffness, m_eff)


def classical_damping(e_target: float, stiffness: float, m_eff: float) -> float:
    """Bilateral (textbook) damping mapping -- *not* the engine's contact law."""
    return restitution_to_damping(e_target, stiffness, m_eff)