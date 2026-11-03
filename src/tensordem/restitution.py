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


__all__ = [
    "ImpactSolution",
    "effective_mass",
    "unilateral_contact_detail",
    "unilateral_restitution",
    "damping_for_restitution",
    "damping_for_restitution_analytic",
    "classical_restitution",
    "classical_damping",
    "RestitutionMeasurement",
    "damping_ratio",
    "restitution_from_damping",
    "restitution_from_damping_clamped",
    "damping_for_clamped_restitution",
    "contact_restitution",
    "clamped_contact_restitution",
    "platen_restitution",
    "measure_platen_restitution",
]

"""Coefficient-of-restitution analytics and calibration for DEM contacts.

The 2-D and 3-D engines in this package use a *linear spring-dashpot*
(Kelvin-Voigt) normal contact law

    F_n = k * delta - c * v_n         (clamped at zero, unilateral)

instead of a restitution coefficient input.  Until this module existed the
damping coefficient ``c`` was a free knob with no documented link to the
physical coefficient of restitution ``e``, which made the engine impossible to
map onto the ice-ice / ice-steel restitution values quoted in the
ice-mechanics literature.

Both engines clamp the normal force at zero, i.e. the contact can push but not
pull.  Two closed forms matter:

* **Unclamped (textbook) Kelvin-Voigt** -- the classic single-degree-of-freedom
  result with the damping ratio ``zeta = c / (2 sqrt(k m))``::

      e_unclamped = exp(-zeta * pi / sqrt(1 - zeta^2))    zeta < 1
      e_unclamped = 0                                     zeta >= 1

  This solution keeps a *tensile* tail after the particle starts to rebound,
  which the engine does not allow, so it under-predicts the restitution.

* **Clamped (unilateral) Kelvin-Voigt** -- the form the engine actually
  integrates.  Starting from zero overlap with approach speed ``v0`` the
  compression is ``delta(t) = (v0/w_d) exp(-zeta w_n t) sin(w_d t)`` with
  ``w_n = sqrt(k/m)``, ``w_d = w_n sqrt(1-zeta^2)``.  The contact ends when the
  total normal force ``k delta + c delta_dot`` first vanishes, which happens at
  the phase::

      phi = atan2(2 zeta sqrt(1 - zeta^2), -(1 - 2 zeta^2))  in (0, pi)

  and the rebound ratio collapses to the compact form::

      e_clamped = exp(-(zeta / sqrt(1 - zeta^2)) * phi)      zeta < 1
      e_clamped = 0                                         zeta >= 1

  At ``zeta -> 0`` both forms tend to one, and ``e_clamped > e_unclamped`` for
  every ``0 < zeta < 1`` because the tensile tail is removed.

Both classes of attainable restitution are ``(0, 1]``; the damping coefficient
that realises a target ``e`` is

    zeta(e) = -ln(e) / sqrt(pi^2 + ln(e)^2),
    c(e)    = 2 * zeta(e) * sqrt(k * m)

for the unclamped form, and is obtained by monotone bisection for the clamped
form (:func:`damping_for_clamped_restitution`).  For ``zeta < 0.05`` the two
agree to better than 1 %; the difference grows to 16 % at ``zeta ~ 0.29``
(see ``docs/dem3d_restitution_calibration.md``), so the clamped inverse is the
correct calibration target for this engine.

Because the law is linear, ``e`` is *independent of impact velocity*.  That is
the standard limitation of a linear dashpot: real ice restitution varies with
impact speed and temperature (see ``docs/dem3d_restitution_calibration.md``).

The helpers below also provide :func:`measure_platen_restitution`, a
discrete-element experiment that bounces a bonded particle block on a rigid
platen of the same spring-dashpot form and measures the realised restitution.
It is used as the verification probe for the analytic relation.
"""

from dataclasses import dataclass
import math

import torch

from .dem3d import DEM3DConfig, IceDEM3D


def damping_ratio(stiffness: float, damping: float, mass: float) -> float:
    """Return ``zeta = c / (2 sqrt(k m))`` with argument validation."""
    for name, value in (("stiffness", stiffness), ("mass", mass)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(damping) or damping < 0.0:
        raise ValueError("damping must be finite and nonnegative")
    return damping / (2.0 * math.sqrt(stiffness * mass))


def restitution_from_damping(stiffness: float, damping: float, mass: float) -> float:
    """Closed-form Kelvin-Voigt restitution ratio for ``(k, c, m)``.

    Returns ``1.0`` for a frictionless spring, values in ``(0, 1)`` for an
    underdamped contact and exactly ``0.0`` for a critically/over-damped
    contact (the model then decays without separating).
    """
    zeta = damping_ratio(stiffness, damping, mass)
    if zeta == 0.0:
        return 1.0
    if zeta >= 1.0:
        return 0.0
    return math.exp(-zeta * math.pi / math.sqrt(1.0 - zeta * zeta))


def restitution_from_damping_clamped(
    stiffness: float, damping: float, mass: float
) -> float:
    """Restitution of a *unilateral* (force-clamped) Kelvin-Voigt contact.

    This is the closed form that matches the normal contact law implemented by
    :class:`tensordem.dem3d.IceDEM3D`, whose dashpot contribution is clipped at
    zero so the contact cannot pull.
    """
    zeta = damping_ratio(stiffness, damping, mass)
    if zeta == 0.0:
        return 1.0
    if zeta >= 1.0:
        return 0.0
    root = math.sqrt(1.0 - zeta * zeta)
    phase = math.atan2(2.0 * zeta * root, -(1.0 - 2.0 * zeta * zeta))
    return math.exp(-(zeta / root) * phase)


def damping_for_clamped_restitution(
    stiffness: float, restitution: float, mass: float, *, iterations: int = 200
) -> float:
    """Damping coefficient that realises ``restitution`` for the clamped law.

    ``restitution_from_damping_clamped`` is strictly decreasing in the damping
    coefficient, so a bracketed bisection converges to machine precision in a
    few dozen iterations.  This is the calibration entry point for the engine:
    the textbook (unclamped) inverse under-shoots the target by up to 16 % once
    the damping ratio exceeds ~0.1.
    """
    if not math.isfinite(restitution) or not 0.0 < restitution < 1.0:
        raise ValueError("restitution must lie in (0, 1)")
    for name, value in (("stiffness", stiffness), ("mass", mass)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    low = 0.0
    high = 2.0 * math.sqrt(stiffness * mass)
    while restitution_from_damping_clamped(stiffness, high, mass) > restitution:
        high *= 2.0
        if high > 1.0e12:
            raise ValueError("requested restitution requires an ill-conditioned dashpot")
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        if restitution_from_damping_clamped(stiffness, mid, mass) > restitution:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def clamped_contact_restitution(config: DEM3DConfig) -> float:
    """Clamped particle-particle restitution implied by ``config``."""
    return restitution_from_damping_clamped(
        config.contact_stiffness, config.contact_damping, config.mass
    )


def damping_for_restitution_analytic(stiffness: float, restitution: float, mass: float) -> float:
    """Inverse of :func:`restitution_from_damping` (analytic, unclamped law).

    ``restitution == 1`` maps to zero damping; ``restitution == 0`` is only
    reached in the limit of infinite damping and raises ``ValueError`` so the
    caller cannot silently install a non-physical dashpot.
    """
    if not math.isfinite(restitution) or not 0.0 < restitution <= 1.0:
        raise ValueError("restitution must lie in (0, 1]")
    for name, value in (("stiffness", stiffness), ("mass", mass)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if restitution == 1.0:
        return 0.0
    log_e = math.log(restitution)
    zeta = -log_e / math.sqrt(math.pi**2 + log_e * log_e)
    return 2.0 * zeta * math.sqrt(stiffness * mass)


def contact_restitution(config: DEM3DConfig) -> float:
    """Particle-particle restitution implied by ``config`` (linear law)."""
    return restitution_from_damping(
        config.contact_stiffness, config.contact_damping, config.mass
    )


def platen_restitution(config: DEM3DConfig) -> float:
    """Particle-platen restitution implied by ``config`` (linear law)."""
    return restitution_from_damping(
        config.platen_stiffness, config.platen_damping, config.mass
    )


@dataclass(frozen=True)
class RestitutionMeasurement:
    """Outcome of the platen bounce verification experiment."""

    stiffness: float
    damping: float
    particle_mass: float
    impact_speed: float
    particle_count: int
    dt: float
    steps: int
    measured_restitution: float
    analytic_restitution: float
    analytic_restitution_clamped: float

    @property
    def relative_error(self) -> float:
        """``|measured - analytic| / analytic`` against the unclamped form."""
        if self.analytic_restitution == 0.0:
            return abs(self.measured_restitution)
        return abs(self.measured_restitution - self.analytic_restitution) / self.analytic_restitution

    @property
    def relative_error_clamped(self) -> float:
        """``|measured - analytic| / analytic`` against the clamped form."""
        if self.analytic_restitution_clamped == 0.0:
            return abs(self.measured_restitution)
        return abs(
            self.measured_restitution - self.analytic_restitution_clamped
        ) / self.analytic_restitution_clamped

    def as_dict(self) -> dict[str, float | int]:
        payload: dict[str, float | int] = {
            "stiffness_N_per_m": self.stiffness,
            "damping_N_s_per_m": self.damping,
            "damping_ratio": damping_ratio(self.stiffness, self.damping, self.particle_mass),
            "particle_mass_kg": self.particle_mass,
            "impact_speed_m_per_s": self.impact_speed,
            "particle_count": self.particle_count,
            "dt_s": self.dt,
            "steps": self.steps,
            "measured_restitution": self.measured_restitution,
            "analytic_restitution": self.analytic_restitution,
            "analytic_restitution_clamped": self.analytic_restitution_clamped,
            "relative_error": self.relative_error,
            "relative_error_clamped": self.relative_error_clamped,
        }
        return payload


def measure_platen_restitution(
    *,
    stiffness: float = 5_000.0,
    damping: float = 0.0,
    impact_speed: float = 0.5,
    radius: float = 0.025,
    density: float = 917.0,
    bond_stiffness: float = 2.0e5,
    nx: int = 2,
    ny: int = 2,
    nz: int = 2,
    dt: float | None = None,
    max_steps: int = 500_000,
) -> RestitutionMeasurement:
    """Bounce a single free particle on a rigid platen and measure ``e``.

    The closed forms compared against here describe *one* degree of freedom with
    mass ``m`` and contact ``(k, c)`` against an infinite-mass wall, i.e. a
    reduced mass equal to the particle mass.  The probe therefore isolates
    exactly that configuration: the lattice is generated as usual (so that the
    solver state has its production layout), every particle except one is parked
    far away and fixed, and the bonds are broken before the impact.  A
    multi-particle bonded block is *not* a single degree of freedom -- the bond
    compliance and the layer-by-layer contact onset add their own dynamics, and
    the measured restitution then deviates from the single-DOF closed form by
    more than 10 % at moderate damping -- so the block configuration is
    deliberately avoided here.

    The free particle is released with normal velocity ``-impact_speed``
    towards the platen; the experiment stops when it has separated (zero platen
    overlap, positive normal velocity).  The measured restitution is the
    outgoing normal velocity divided by the impact speed.
    """
    if not math.isfinite(impact_speed) or impact_speed <= 0.0:
        raise ValueError("impact_speed must be finite and positive")
    for name, value in (("stiffness", stiffness), ("bond_stiffness", bond_stiffness)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(damping) or damping < 0.0:
        raise ValueError("damping must be finite and nonnegative")

    config = DEM3DConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        radius=radius,
        density=density,
        bond_stiffness=bond_stiffness,
        drag=0.0,
        fix_x_edges=False,
        fix_bottom=False,
        bottom_platen_enabled=True,
        platen_stiffness=stiffness,
        platen_damping=damping,
        dt=dt,
    )
    sim = IceDEM3D(config)
    particle_count = int(sim.positions.shape[0])

    # Isolate one free particle; park the rest far away and clamp them so that
    # the impact probes a single mass-spring-dashpot contact against the platen.
    probe = 0
    parked = torch.zeros_like(sim.positions)
    for index in range(particle_count):
        parked[index] = torch.tensor(
            [10.0 * index, 500.0, 500.0], dtype=sim.dtype, device=sim.device
        )
    parked[probe] = torch.tensor(
        [0.0, 0.0, sim.bottom_platen_start_z + config.radius + 1.0e-6],
        dtype=sim.dtype,
        device=sim.device,
    )
    sim.positions = parked.clone()
    sim.initial_positions = parked.clone()
    fixed = torch.ones(particle_count, dtype=torch.bool, device=sim.device)
    fixed[probe] = False
    sim.fixed = fixed
    sim.velocities = torch.zeros_like(sim.positions)
    sim.velocities[probe, 2] = -impact_speed
    sim.forces()  # break the lattice bonds so no residual bond force acts
    if int(sim.alive.sum().item()) != 0:
        raise RuntimeError("bonds did not break; probe assumption violated")

    for step in range(1, max_steps + 1):
        sim.step()
        overlap = (
            sim.bottom_platen_start_z + config.radius - sim.positions[probe, 2]
        ).clamp_min(0.0)
        normal_velocity = sim.velocities[probe, 2]
        if float(overlap) == 0.0 and float(normal_velocity) > 0.0:
            outgoing = float(normal_velocity)
            return RestitutionMeasurement(
                stiffness=stiffness,
                damping=damping,
                particle_mass=config.mass,
                impact_speed=impact_speed,
                particle_count=particle_count,
                dt=sim.dt,
                steps=step,
                measured_restitution=outgoing / impact_speed,
                analytic_restitution=restitution_from_damping(stiffness, damping, config.mass),
                analytic_restitution_clamped=restitution_from_damping_clamped(
                    stiffness, damping, config.mass
                ),
            )
    raise RuntimeError(
        f"probe particle did not separate from the platen within {max_steps} steps "
        f"(overdamped contact or unstable dt)"
    )
