"""2-D engine (``tensordem.dem``) contact-damping -> restitution closure.

Why this module exists
----------------------
``DEMConfig.contact_damping`` is a raw dashpot coefficient in N*s/m.  Until this
module existed the 2-D solver had no measurement of what coefficient of
restitution a given dashpot actually produces: the 3-D path owns that closure
(``tensordem.restitution``), and its docstring already claimed the mapping also
applies to ``DEMConfig.contact_damping`` -- a transfer that was never executed.

The 2-D law implemented by ``tensordem.dem`` is the same tensionless linear
spring-dashpot used in 3-D::

    F_n = max(0, k_n * delta - c_n * v_n)          (delta > 0 = overlap)

with ``v_n`` the normal *separation* rate (positive when the bodies separate),
integrated with semi-implicit Euler.  The clip makes the law unilateral: the
dashpot cannot pull, so the contact opens as soon as the spring force is
overcome and the rebound is faster than the bilateral textbook
mass-spring-dashpot result ``e = exp(-pi zeta / sqrt(1 - zeta^2))``.

Because the 2-D solver has no rotational degrees of freedom and its counterface
(the circular tool) is kinematic, a single particle hitting the tool is exactly
one degree of freedom against an infinite-mass wall: the reduced mass equals the
particle mass and ``tensordem.restitution``'s closed forms apply without any
correction.  This module makes that correspondence measurable:

* :func:`measure_restitution` -- drive a one-particle impact through the real
  ``IceDEM`` contact branch and measure the outgoing/incoming normal speed ratio;
* :func:`recommended_damping` -- invert the *clipped* engine law for a requested
  restitution (thin wrapper over ``restitution.damping_for_restitution``, which
  is a bisection on the unilateral law rather than on the textbook formula);
* :func:`discretization_bias` -- the residual error at the solver's own
  recommended time step, i.e. the part of the discrepancy that is semi-implicit
  Euler, not physics.

The restitution itself is *not* re-derived here: ``tensordem.restitution`` stays
the single source of truth for the continuous forms.  ``tensordem.dem2d_restitution``
only owns the 2-D engine probe and the comparison table.

Scope / limits
--------------
A bonded multi-particle block is not one degree of freedom (bond compliance and
staggered contact onset add their own dynamics), so the probe deliberately
isolates a single free particle and parks the rest of the lattice.  The reported
restitution is therefore the *contact-law* restitution; the effective
restitution of a whole ice block is a different, configuration-dependent
quantity and is not covered here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .dem import DEMConfig, IceDEM
from .restitution import (
    classical_damping,
    classical_restitution,
    damping_for_restitution,
    unilateral_restitution,
)

__all__ = [
    "Dem2DImpact",
    "particle_mass",
    "measure_restitution",
    "recommended_damping",
    "recommended_classical_damping",
    "discretization_bias",
]

# The tool is driven by a prescribed velocity in the 2-D solver and the config
# rejects a non-positive speed, so a vanishing speed stands for "static
# counterface".  The tool travel during one contact is O(1e-9 * 0.1) m, i.e. a
# factor 1e-7 below the particle radius.
_STATIC_TOOL_SPEED = 1.0e-9


def particle_mass(*, radius: float = 0.025, thickness: float = 0.2,
                  density: float = 917.0) -> float:
    """Return the 2-D disk mass ``rho * pi * r^2 * t`` used by ``DEMConfig``."""
    for name, value in (("radius", radius), ("thickness", thickness), ("density", density)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value}")
    return density * math.pi * radius**2 * thickness


@dataclass(frozen=True)
class Dem2DImpact:
    """One measured 2-D tool impact plus its two closed-form references."""

    stiffness: float
    damping: float
    particle_mass: float
    impact_speed: float
    dt: float
    steps: int
    measured_restitution: float
    unilateral_prediction: float
    classical_prediction: float

    @property
    def damping_ratio(self) -> float:
        """``zeta = c / (2 sqrt(k m))`` -- the textbook damping ratio."""
        return self.damping / (2.0 * math.sqrt(self.stiffness * self.particle_mass))

    @property
    def error_vs_unilateral(self) -> float:
        """``|measured - unilateral| / unilateral``; the transfer error of interest."""
        reference = self.unilateral_prediction
        if reference <= 0.0:
            return abs(self.measured_restitution)
        return abs(self.measured_restitution - reference) / reference

    @property
    def error_vs_classical(self) -> float:
        """``|measured - classical| / classical``; the bilateral text-book error."""
        reference = self.classical_prediction
        if reference <= 0.0:
            return abs(self.measured_restitution)
        return abs(self.measured_restitution - reference) / reference

    def as_dict(self) -> dict[str, float | int]:
        return {
            "stiffness_N_per_m": self.stiffness,
            "damping_N_s_per_m": self.damping,
            "damping_ratio": self.damping_ratio,
            "particle_mass_kg": self.particle_mass,
            "impact_speed_m_per_s": self.impact_speed,
            "dt_s": self.dt,
            "steps": self.steps,
            "measured_restitution": self.measured_restitution,
            "unilateral_prediction": self.unilateral_prediction,
            "classical_prediction": self.classical_prediction,
            "relative_error_vs_unilateral": self.error_vs_unilateral,
            "relative_error_vs_classical": self.error_vs_classical,
        }


def measure_restitution(
    *,
    stiffness: float = 2_000.0,
    damping: float = 0.0,
    impact_speed: float = 1.0,
    radius: float = 0.025,
    thickness: float = 0.2,
    density: float = 917.0,
    tool_radius: float = 0.1,
    tool_gap: float = 0.01,
    bond_stiffness: float = 1.0e-9,
    dt: float | None = None,
    max_steps: int = 100_000,
    probe: int = 0,
) -> Dem2DImpact:
    """Measure the 2-D engine restitution for ``DEMConfig.contact_damping``.

    The lattice is built with the production constructor (``nx = ny = 2``, the
    smallest legal bonded patch) so that the solver state has its normal layout.
    Every particle except ``probe`` is then parked far from the tool and clamped,
    and the probe is released with the normal velocity ``impact_speed`` towards
    the circular tool.  The bonds are broken by evaluating the force field once
    (their strain after parking is astronomical), which leaves the probe as a
    free particle: a pure one-degree-of-freedom normal impact.

    ``bond_stiffness`` defaults to a value seven decades below the contact
    stiffness so that a *surviving* bond could not perturb the impact either;
    :func:`measure_restitution` asserts that all bonds are in fact broken.

    Returns
    -------
    Dem2DImpact
        Measured restitution with the continuous unilateral (clipped law) and
        classical (bilateral spring-dashpot) predictions for the same ``(k, c, m)``.
    """
    if not math.isfinite(impact_speed) or impact_speed <= 0.0:
        raise ValueError("impact_speed must be finite and positive")
    for name, value in (("stiffness", stiffness), ("bond_stiffness", bond_stiffness),
                        ("tool_radius", tool_radius)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(damping) or damping < 0.0:
        raise ValueError("damping must be finite and nonnegative")

    config = DEMConfig(
        nx=2,
        ny=2,
        radius=radius,
        thickness=thickness,
        density=density,
        bond_stiffness=bond_stiffness,
        contact_stiffness=stiffness,
        contact_damping=damping,
        drag=0.0,
        tool_radius=tool_radius,
        tool_speed=_STATIC_TOOL_SPEED,
        tool_gap=tool_gap,
        fix_edges=False,
        dt=dt,
    )
    sim = IceDEM(config)
    tool_position = sim.tool_start.clone()
    particle_count = int(sim.positions.shape[0])
    if not 0 <= probe < particle_count:
        raise ValueError("probe index out of range")

    parked = torch.zeros_like(sim.positions)
    for index in range(particle_count):
        parked[index] = torch.tensor(
            [50.0 * index, -500.0], dtype=sim.positions.dtype, device=sim.device
        )
    touch = float(tool_position[1]) - (radius + tool_radius)
    parked[probe] = torch.tensor(
        [float(tool_position[0]), touch - 1.0e-6],
        dtype=sim.positions.dtype,
        device=sim.device,
    )
    sim.positions = parked.clone()
    sim.initial_positions = parked.clone()
    fixed = torch.ones(particle_count, dtype=torch.bool, device=sim.device)
    fixed[probe] = False
    sim.fixed = fixed
    sim.velocities = torch.zeros_like(sim.positions)
    sim.velocities[probe, 1] = impact_speed
    sim.forces()  # break the lattice bonds so no residual bond force acts
    if int(sim.alive.sum().item()) != 0:
        raise RuntimeError("bonds did not break; single-degree-of-freedom probe violated")

    contact_radius = radius + tool_radius
    touched = False
    for step in range(1, max_steps + 1):
        sim.step()
        separation = float((sim.positions[probe] - tool_position).norm())
        normal_velocity = float(sim.velocities[probe, 1])
        if separation <= contact_radius:
            touched = True
        elif touched and normal_velocity < 0.0:
            # Separated after the impact and moving away from the tool (the tool
            # sits above the probe, so the rebound velocity is negative).
            return Dem2DImpact(
                stiffness=stiffness,
                damping=damping,
                particle_mass=config.mass,
                impact_speed=impact_speed,
                dt=sim.dt,
                steps=step,
                measured_restitution=abs(normal_velocity) / impact_speed,
                unilateral_prediction=unilateral_restitution(
                    damping, stiffness, config.mass, impact_speed
                ),
                classical_prediction=classical_restitution(
                    damping, stiffness, config.mass
                ),
            )
    raise RuntimeError(
        f"probe did not separate from the tool within {max_steps} steps "
        f"(overdamped contact or unstable dt)"
    )


def recommended_damping(
    target_restitution: float,
    *,
    stiffness: float = 2_000.0,
    mass: float | None = None,
    impact_speed: float = 1.0,
) -> float:
    """Invert the clipped engine law: damping for a requested restitution.

    Delegates to :func:`tensordem.restitution.damping_for_restitution`, which
    bisects the continuous *unilateral* solution.  Use this instead of
    :func:`recommended_classical_damping` when the result is fed to
    ``DEMConfig.contact_damping``.
    """
    m_eff = particle_mass() if mass is None else mass
    return damping_for_restitution(target_restitution, stiffness, m_eff, impact_speed)


def recommended_classical_damping(
    target_restitution: float,
    *,
    stiffness: float = 2_000.0,
    mass: float | None = None,
) -> float:
    """Bilateral textbook inverse (``classical_damping``), kept for the audit."""
    m_eff = particle_mass() if mass is None else mass
    return classical_damping(target_restitution, stiffness, m_eff)


def discretization_bias(
    *,
    stiffness: float = 2_000.0,
    damping: float = 5.0,
    impact_speed: float = 1.0,
    refinement: int = 4,
    levels: int = 3,
) -> list[Dem2DImpact]:
    """Measure the time-step error: recommended ``dt`` refined by ``refinement``.

    Returns one :class:`Dem2DImpact` per level, coarsest first.  The config's own
    ``recommended_dt`` is the upper limit accepted by ``IceDEM``; because that
    limit depends on ``damping``, the refined grid is anchored at it and each
    level divides it by ``refinement``.

    A large ``refinement`` on a large lattice is *not* free: cost grows linearly
    with ``levels * refinement``.  The default (3 levels of 4x) is a
    seconds-scale probe on 2x2 lattices.
    """
    if refinement < 2:
        raise ValueError("refinement must be >= 2")
    base = DEMConfig(
        nx=2, ny=2, bond_stiffness=1.0e-9,
        contact_stiffness=stiffness, contact_damping=damping,
        fix_edges=False, drag=0.0, tool_speed=_STATIC_TOOL_SPEED,
    ).recommended_dt
    results: list[Dem2DImpact] = []
    for level in range(levels):
        dt = base / float(refinement**level)
        results.append(
            measure_restitution(
                stiffness=stiffness, damping=damping, impact_speed=impact_speed, dt=dt
            )
        )
    return results