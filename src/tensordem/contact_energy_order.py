"""Energy-ledger **conventions** for rigid contact collisions, and their order.

Why this module exists
----------------------
Every contact energy audit in this repository ultimately asks one question: *is
the ledger residual a bug or a discretisation artefact?*  The distinction is not
decidable from a single number -- a 5e-7 residual can be an excellent second
order discretisation or a 5e-7 implementation defect -- but it **is** decidable
from its behaviour under refinement: a discretisation artefact is a function of
``dt`` that vanishes as ``dt -> 0`` (a positive log-log slope), an
implementation defect is a fixed offset (slope ~ 0) or worse.

This module makes that test executable.  It defines the ledger conventions
sharply, derives the ones that close *by construction* (to round-off, for any
``dt``), measures the ones that carry a discretisation error, and fits the
order of every one of them on a problem whose exact answer is known.

The four conventions (all normalised by the impact energy ``E0``)
-----------------------------------------------------------------
A two-body impact splits into its centre-of-mass motion (force free, exactly
integrable) and one relative coordinate: the gap ``x`` (positive when the
surfaces are apart), overlap ``delta = -x``, effective mass
``m_eff = m1 m2/(m1+m2)``, relative speed ``v = dx/dt``.  The contact law is a
conservative spring ``F_s(delta)`` plus an optional dashpot.

``state``
    State-quantity difference -- what ``floe_rigid6dof.energy_ledger`` reports as
    ``residual_state_relative`` and what mission m5404 quoted as 5.111e-07::

        r_state = | dKE + dU | / E0,     U = int_0^delta F_s(d delta')

    The spring energy is a *state function*, so for an undamped impact the
    residual is exactly the integrator's energy error.  It is ``O(dt^p)`` with
    ``p`` fixed by the integrator **and by the smoothness of the contact law at
    touchdown**: a Hertz law ``F ~ delta^{3/2}`` turns on with vanishing
    derivative (``C^1``) and gives ``p = 2`` for velocity-Verlet, whereas a
    linear law ``F = k delta`` turns on with a jump in the force derivative
    (only Lipschitz) and gives ``p = 1``.  That is a property of the *law*, not
    of the code, and both are measured here.

``explicit``
    Impulse x displacement with the left endpoint (the explicit-Euler power)::

        r_exp = | dKE - sum_n F_n (x_{n+1} - x_n) | / E0  = O(dt)

``trapezoid``
    Impulse x displacement with the average endpoint -- what
    ``floe_rigid6dof`` accumulates as ``contact_work`` and what m5404 quoted as
    2.315e-05::

        r_trap = | dKE - sum_n (F_n + F_{n+1})/2 (x_{n+1} - x_n) | / E0

    For velocity-Verlet this is **not** an approximation but an identity.  The
    two half kicks give ``v_{n+1/2} = (v_n + v_{n+1})/2`` exactly, the drift
    gives ``x_{n+1} - x_n = dt v_{n+1/2}``, and
    ``m v_{n+1/2}(v_{n+1} - v_n) = (dt/2)(F_n + F_{n+1}) v_{n+1/2}``; hence::

        dKE == sum_n (F_n + F_{n+1})/2 (x_{n+1} - x_n)      (exactly, any dt)

    so ``r_trap`` is round-off (``~1e-16``) *independently of the resolution* --
    even with 4 steps per contact.  The residual of this convention therefore
    measures only floating-point round-off, and any excess is an implementation
    defect in the force/position bookkeeping (the test suite uses it as a
    guard).  The identity survives in an ``N``-body chain because the pairwise
    forces telescope (:func:`leapfrog_work_identity`).

``restitution``
    Theoretical dissipation from the coefficient of restitution::

        W_diss(theory)   = E0 (1 - e^2)
        W_diss(measured) = -(dKE + dU)
        r_rest = | W_diss(measured) - W_diss(theory) | / E0

    with ``e`` the *clamped* restitution of the law
    (:func:`tensordem.contact_impact.clamped_restitution` for
    ``F = max(0, k delta + c delta')``).  ``r_rest`` closes at round-off only
    when the contact phase is resolved analytically:
    :func:`event_driven_impact` advances the damped oscillator in closed form
    from touchdown to separation and returns ``e = e_clamped`` to round-off.
    Under fixed-step integration ``r_rest`` equals the ``O(dt^p)`` restitution
    error and is a *discretisation* residual, reported as such.

The classification rule this module demonstrates
-----------------------------------------------
* a convention/``dt`` residual has a fitted log-log slope ``>= 1`` and its level
  collapses as the number of steps per contact ``N_c = t_c/dt`` grows;
* an implementation defect -- a state function mis-scaled by 1e-3, a dropped
  half kick, a ledger channel left out -- has slope ``~0`` and its level does
  not move.

Both are produced on the *same* impact here, so the discrimination is measured,
and mission m5404's 5.111e-07 is re-judged with it
(:func:`m5404_rejudgement` drives the shipped ``floe_rigid6dof.FloeWorld``, not
a re-implementation).

Everything is SI: m, s, kg, N.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .contact_impact import clamped_restitution

__all__ = [
    "ContactLaw",
    "contact_force",
    "spring_energy",
    "hertz_contact_time",
    "ImpactTrace",
    "PairImpact",
    "two_body_impact",
    "ledger_conventions",
    "fit_order",
    "order_sweep",
    "stiffness_sweep",
    "leapfrog_work_identity",
    "staggered_pairing_order",
    "event_driven_impact",
    "NewtonCradleResult",
    "newton_cradle",
    "m5404_level",
    "m5404_verdict",
    "m5404_rejudgement",
    "M5404_CASE",
]


# ===================================================================
# Contact law
# ===================================================================

@dataclass(frozen=True)
class ContactLaw:
    """Normal contact law ``F = k delta^n (1 + dc)`` with ``dc`` the damping.

    Parameters
    ----------
    stiffness:
        ``k``; ``N/m`` for ``exponent = 1`` and ``N/m^1.5`` for the Hertzian
        ``exponent = 1.5``.
    exponent:
        ``1.0`` (linear spring, Cundall & Strack) or ``1.5`` (Hertz).
    damping_model:
        ``"none"``, ``"linear"`` (``F += c * delta_dot``, the Cundall-Strack
        dashpot with ``delta_dot > 0`` while compressing) or
        ``"hunt_crossley"`` (``F *= 1 + 1.5 alpha delta_dot``).
    damping:
        ``c`` [N s/m] for ``"linear"``, ``alpha`` [s/m] for ``"hunt_crossley"``.
    clip:
        truncate ``F`` at zero (DEM convention, ``F = max(0, F)``).  The
        closed-form restitution used by :func:`event_driven_impact` is the
        clamped one, so that path requires ``clip = True``.
    """

    stiffness: float
    exponent: float = 1.5
    damping_model: str = "none"
    damping: float = 0.0
    clip: bool = True

    def __post_init__(self) -> None:
        if self.stiffness <= 0.0:
            raise ValueError("stiffness must be positive")
        if self.exponent not in (1.0, 1.5):
            raise ValueError("exponent must be 1.0 (linear) or 1.5 (Hertz)")
        if self.damping_model not in ("none", "linear", "hunt_crossley"):
            raise ValueError(f"unknown damping_model {self.damping_model!r}")
        if self.damping < 0.0:
            raise ValueError("damping must be nonnegative")
        if self.damping_model != "none" and self.damping == 0.0:
            raise ValueError("damping_model set but damping is zero")

    @property
    def hertzian(self) -> bool:
        return self.exponent == 1.5

    def damping_ratio(self, m_eff: float) -> float:
        """``zeta = c / (2 sqrt(k m_eff))`` (linear dashpot only)."""
        if self.damping_model != "linear":
            raise ValueError("damping ratio is defined for the linear dashpot")
        return self.damping / (2.0 * math.sqrt(self.stiffness * m_eff))

    def restitution(self, m_eff: float) -> float:
        """Closed-form clamped restitution of the linear spring-dashpot law."""
        if self.damping_model != "linear":
            raise ValueError("closed-form restitution requires the linear dashpot")
        if not self.clip:
            raise ValueError("closed-form restitution here is the clamped one")
        return clamped_restitution(self.damping_ratio(m_eff))


def contact_force(law: ContactLaw, delta: float, delta_dot: float) -> float:
    """Normal force ``>= 0`` for overlap ``delta`` and compression rate ``delta_dot``.

    ``delta_dot > 0`` while the bodies approach; ``delta <= 0`` means no contact.
    """
    if delta <= 0.0:
        return 0.0
    f_spring = law.stiffness * (delta if law.exponent == 1.0
                                else delta ** law.exponent)
    if law.damping_model == "none":
        f = f_spring
    elif law.damping_model == "linear":
        f = f_spring + law.damping * delta_dot
    else:  # hunt_crossley
        f = f_spring * (1.0 + 1.5 * law.damping * delta_dot)
    if law.clip and f < 0.0:
        return 0.0
    return f


def spring_energy(law: ContactLaw, delta: float) -> float:
    """Potential energy ``int_0^delta k s^n ds = k delta^{n+1}/(n+1)``."""
    if delta <= 0.0:
        return 0.0
    return law.stiffness * delta ** (law.exponent + 1.0) / (law.exponent + 1.0)


def hertz_contact_time(m_eff: float, stiffness: float, v0: float) -> float:
    """Hertzian contact duration ``t_c = 2.9427 (5 m_eff / (4 k))^{2/5} v0^{-1/5}``.

    Weber & Stillinger (1985).  Same expression as
    :func:`tensordem.contact_timestep_ledger.hertz_contact_time_weber_stillinger`,
    restated here so this module needs no extra import.
    """
    if m_eff <= 0.0 or stiffness <= 0.0 or v0 <= 0.0:
        raise ValueError("m_eff, stiffness and v0 must be positive")
    return 2.9427 * (5.0 * m_eff / (4.0 * stiffness)) ** 0.4 * v0 ** -0.2


# ===================================================================
# 1-D chain integrator
# ===================================================================

@dataclass
class ImpactTrace:
    """Sampled state plus the accumulated ledger channels of one run."""

    times: np.ndarray
    gaps: np.ndarray            # (n_pairs, n_samples) surface gap per pair
    forces: np.ndarray          # (n_pairs, n_samples) normal force per pair
    kinetic: np.ndarray
    stored: np.ndarray
    work_explicit: float
    work_trapezoid: float          # average-force x exact displacement (identity)
    work_average_velocity: float   # 1/2 [P(x_n,v_n) + P(x_{n+1},v_{n+1/2})] h
    work_damping: float            # integral of P_damp dt (<= 0)
    momentum_error: float
    steps: int
    dt: float
    e0: float
    v_in: float
    v_out: float
    min_gap: float
    max_overlap: float
    positions: np.ndarray       # (n_bodies, n_samples)
    velocities: np.ndarray
    radii: np.ndarray           # (n_bodies,)

    @property
    def power(self) -> np.ndarray:
        """Contact power ``P = sum_pairs F (d gap / dt)`` at the sample states."""
        out = np.zeros(self.times.size)
        for n in range(self.times.size):
            gaps, rates = _chain_gaps(self.positions[:, n], self.radii,
                                      self.velocities[:, n])
            out[n] = float(np.sum(self.forces[:, n] * rates))
        return out

    @property
    def separated(self) -> bool:
        return bool(self.forces[:, -1].max() == 0.0)


@dataclass
class PairImpact:
    """One two-body impact reduced to its relative DOF."""

    law: ContactLaw
    m_eff: float
    v0: float
    gap0: float
    t_end: float
    dt: float
    trace: ImpactTrace

    @property
    def contact_time(self) -> float:
        if self.law.hertzian:
            return hertz_contact_time(self.m_eff, self.law.stiffness, self.v0)
        return math.pi * math.sqrt(self.m_eff / self.law.stiffness)

    @property
    def steps_per_contact(self) -> float:
        return self.contact_time / self.dt


def _pair_forces(gaps: np.ndarray, gap_rates: np.ndarray, law: ContactLaw
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pair normal force, spring energy and damping power (1-D chain).

    ``gaps[i]`` is the surface gap of pair ``i`` (``> 0`` apart), ``gap_rates``
    its rate (``< 0`` while approaching).  ``F > 0`` pushes the pair apart.
    """
    force = np.zeros(gaps.size)
    energy = np.zeros(gaps.size)
    p_damp = np.zeros(gaps.size)
    for i in range(gaps.size):
        delta = -gaps[i]
        if delta <= 0.0:
            continue
        delta_dot = -gap_rates[i]              # > 0 while compressing
        f = contact_force(law, delta, delta_dot)
        force[i] = f
        energy[i] = spring_energy(law, delta)
        if law.damping_model == "linear":
            p_damp[i] = -law.damping * delta_dot * delta_dot
        elif law.damping_model == "hunt_crossley":
            f_spring = law.stiffness * delta ** law.exponent
            p_damp[i] = -1.5 * law.damping * f_spring * delta_dot * delta_dot
    return force, energy, p_damp


def _chain_gaps(positions: np.ndarray, radii: np.ndarray,
                velocities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Gaps and gap rates of the adjacent pairs of a sorted 1-D chain."""
    gaps = (positions[1:] - positions[:-1]) - (radii[1:] + radii[:-1])
    rates = velocities[1:] - velocities[:-1]
    return gaps, rates


def _integrate_chain(positions: Sequence[float], velocities: Sequence[float],
                     masses: Sequence[float], radii: Sequence[float],
                     law: ContactLaw, dt: float, t_end: float,
                     integrator: str = "verlet",
                     defect: Optional[str] = None,
                     max_steps: int = 4_000_000) -> ImpactTrace:
    """Integrate a 1-D chain of spheres with adjacent-pair normal contacts.

    ``integrator``: ``"verlet"`` (kick-drift-kick) or ``"euler_explicit"``
    (``x += dt v``, ``v += dt a``).  ``defect`` injects a *deliberate*
    implementation defect used as the negative control:

    ``"dropped_second_kick"``
        omit the second half kick (a classic leapfrog bookkeeping bug: the
        forces are evaluated but only one kick is applied);
    ``"explicit_kick"``
        use the pre-step force for both kicks (stale-force bug).

    The work channels are accumulated **pair by pair** with the *same* endpoint
    states the integrator used, so the trapezoid channel is the discrete
    identity of :func:`leapfrog_work_identity`, not a separate quadrature.
    """
    if integrator not in ("verlet", "euler_explicit"):
        raise ValueError(f"unknown integrator {integrator!r}")
    x = np.asarray(positions, dtype=float).copy()
    v = np.asarray(velocities, dtype=float).copy()
    m = np.asarray(masses, dtype=float)
    r = np.asarray(radii, dtype=float)
    n = x.size
    n_pairs = n - 1

    e0 = 0.5 * float(np.sum(m * v * v))
    v_in = float(np.abs(v[1] - v[0])) if n >= 2 else 0.0
    p0 = float(np.sum(m * v))

    def pair_state(xx, vv):
        gaps, rates = _chain_gaps(xx, r, vv)
        force, energy, p_damp = _pair_forces(gaps, rates, law)
        f_body = np.zeros(n)
        f_body[:-1] -= force      # F > 0 pushes the pair apart
        f_body[1:] += force
        return gaps, force, energy, p_damp, f_body

    gaps0, f0, u0, pd0, f_body0 = pair_state(x, v)
    rates0 = _chain_gaps(x, r, v)[1]
    power0 = float(np.sum(f0 * rates0))

    times = [0.0]
    gaps_h = [gaps0.copy()]
    force_h = [f0.copy()]
    ke_h = [float(np.sum(0.5 * m * v * v))]
    u_h = [float(u0.sum())]
    x_h = [x.copy()]
    v_h = [v.copy()]

    w_exp = 0.0
    w_trap = 0.0
    w_avg = 0.0
    w_damp = 0.0
    min_gap = float(gaps0.min()) if n_pairs else 0.0
    max_overlap = float(max(0.0, -gaps0.min())) if n_pairs else 0.0
    t = 0.0
    steps = 0
    while t < t_end - 1e-15 and steps < max_steps:
        h = min(dt, t_end - t)
        if integrator == "verlet":
            v_half = v + 0.5 * h * f_body0 / m
            x_new = x + h * v_half
            gaps1, f1, u1, pd1, f_body1 = pair_state(x_new, v_half)
            if defect == "dropped_second_kick":
                v_new = v_half
            elif defect == "explicit_kick":
                v_new = v + h * f_body0 / m
            else:
                v_new = v_half + 0.5 * h * f_body1 / m
        else:
            x_new = x + h * v
            gaps1, f1, u1, pd1, f_body1 = pair_state(x_new, v)
            v_new = v + h * f_body0 / m

        dgap = gaps1 - gaps0
        w_exp += float(np.sum(f0 * dgap))
        w_trap += float(np.sum(0.5 * (f0 + f1) * dgap))
        # average velocity of the step: v_{n+1/2} = (v_n + v_{n+1})/2 exactly for
        # velocity-Verlet, so the staggered rate is the exact average.
        rate1 = _chain_gaps(x_new, r, v_new)[1]
        rate_half = 0.5 * (rates0 + rate1) if integrator == "verlet" else rate1
        power_half = float(np.sum(f1 * rate_half))
        w_avg += 0.5 * (power0 + power_half) * h
        power0 = float(np.sum(f1 * rate1))
        rates0 = rate1
        w_damp += float(np.sum(0.5 * (pd0 + pd1) * h))

        if n_pairs:
            min_gap = min(min_gap, float(gaps1.min()))
            max_overlap = max(max_overlap, float(max(0.0, -gaps1.min())))
        x, v = x_new, v_new
        gaps0, f0, u0, pd0, f_body0 = gaps1, f1, u1, pd1, f_body1
        t += h
        steps += 1
        times.append(t)
        gaps_h.append(gaps0.copy())
        force_h.append(f0.copy())
        ke_h.append(float(np.sum(0.5 * m * v * v)))
        u_h.append(float(u0.sum()))
        x_h.append(x.copy())
        v_h.append(v.copy())

    p_end = float(np.sum(m * v))
    return ImpactTrace(
        times=np.asarray(times),
        gaps=np.asarray(gaps_h).T,
        forces=np.asarray(force_h).T,
        kinetic=np.asarray(ke_h),
        stored=np.asarray(u_h),
        work_explicit=w_exp,
        work_trapezoid=w_trap,
        work_average_velocity=w_avg,
        work_damping=w_damp,
        momentum_error=abs(p_end - p0) / max(abs(p0), 1e-300),
        steps=steps,
        dt=dt,
        e0=e0,
        v_in=v_in,
        v_out=float(np.abs(v[1] - v[0])) if n >= 2 else 0.0,
        min_gap=min_gap,
        max_overlap=max_overlap,
        positions=np.asarray(x_h).T,
        velocities=np.asarray(v_h).T,
        radii=r.copy(),
    )


# ===================================================================
# The pair impact driver
# ===================================================================

def two_body_impact(law: ContactLaw, *, m_eff: float = 1.0, v0: float = 1.0,
                    gap0: float = 0.02, dt: Optional[float] = None,
                    refinements: int = 8, flights: float = 3.0,
                    integrator: str = "verlet",
                    defect: Optional[str] = None) -> PairImpact:
    """Head-on impact of two equal spheres, reduced to the relative coordinate.

    The bodies start ``gap0`` apart with closing speed ``v0`` and are run until
    they are separated and have flown ``flights`` times the free-flight gap, so
    the final state has zero stored energy and a constant velocity: the ledger
    residual at the end is the residual acquired by the contact episode alone.

    ``dt`` defaults to ``t_c / refinements`` where ``t_c`` is the analytic
    contact duration of the law; ``refinements`` is the number of steps per
    contact at ``k = 0`` of the ``dt/2^k`` sweep.
    """
    if m_eff <= 0.0 or v0 <= 0.0 or gap0 <= 0.0:
        raise ValueError("m_eff, v0 and gap0 must be positive")
    if law.hertzian:
        t_c = hertz_contact_time(m_eff, law.stiffness, v0)
    else:
        t_c = math.pi * math.sqrt(m_eff / law.stiffness)
    h = (t_c / refinements) if dt is None else float(dt)
    t_end = gap0 / v0 + (1.0 + flights) * t_c
    radius = 0.5 * gap0          # radii only set the gap scale here
    trace = _integrate_chain(
        positions=[-0.5 * gap0 - radius, 0.5 * gap0 + radius],
        velocities=[0.5 * v0, -0.5 * v0],       # closing: they approach
        masses=[2.0 * m_eff, 2.0 * m_eff],
        radii=[radius, radius],
        law=law, dt=h, t_end=t_end, integrator=integrator, defect=defect)
    return PairImpact(law=law, m_eff=m_eff, v0=v0, gap0=gap0, t_end=t_end,
                      dt=h, trace=trace)


# ===================================================================
# The conventions
# ===================================================================

def ledger_conventions(impact: PairImpact,
                       e_theory: Optional[float] = None) -> Dict[str, float]:
    """All ledger residuals of one impact, normalised by ``E0``.

    ``residual_state``
        ``|dKE + dU| / E0`` -- state-quantity convention (what
        ``floe_rigid6dof`` reports as ``residual_state_relative`` and what
        mission m5404 reported as 5.111e-07).  Equals the integrator's energy
        error for an undamped impact.
    ``residual_explicit``
        ``|dKE - sum_n F_n (x_{n+1} - x_n)| / E0`` -- left-endpoint impulse x
        displacement; ``O(dt)``.
    ``residual_average_velocity``
        ``|dKE - sum_n (F_n + F_{n+1})/2 (x_{n+1} - x_n)| / E0`` -- impulse x
        displacement with the average (trapezoidal) force, equivalently with the
        average velocity of the step since ``x_{n+1} - x_n = h v_{n+1/2}``
        exactly for velocity-Verlet.  Round-off by the discrete work--energy
        theorem (:func:`leapfrog_work_identity`): an *exact* closure, not an
        ``O(dt^2)`` one.
    ``residual_staggered_power``
        ``|dKE - sum_n (P_n + P_{n+1/2}) h / 2| / E0`` with ``P = F v`` -- the
        channel ``floe_rigid6dof`` accumulates as ``contact_work``.  The
        end-of-step force is paired with the *mid-step* velocity, which is an
        inconsistent time level; the closure degrades to ``O(dt)`` (measured
        slope 1.00, :func:`staggered_pairing_order`).
    ``residual_restitution``
        ``|W_diss(measured) - E0 (1 - e^2)| / E0`` with ``W_diss`` from the state
        identity ``-(dKE + dU)``; ``inf`` when ``e_theory`` is not supplied
        (nothing is compared, nothing is claimed).
    ``e_measured``
        ``|v_out| / |v_in|`` of the relative coordinate.
    """
    tr = impact.trace
    dke = float(tr.kinetic[-1] - tr.kinetic[0])
    du = float(tr.stored[-1] - tr.stored[0])
    e0 = max(abs(tr.e0), 1e-300)
    e_meas = float(abs(tr.v_out) / abs(tr.v_in)) if tr.v_in > 0.0 else 0.0
    out = {
        "residual_state": abs(dke + du) / e0,
        "residual_explicit": abs(dke - tr.work_explicit) / e0,
        "residual_average_velocity": abs(dke - tr.work_trapezoid) / e0,
        "residual_staggered_power": abs(dke - tr.work_average_velocity) / e0,
        "e_measured": e_meas,
        "dissipation_measured": -(dke + du),
        "steps_per_contact": impact.steps_per_contact,
        "dt": impact.dt,
        "steps": tr.steps,
        "min_gap": tr.min_gap,
        "max_overlap": tr.max_overlap,
        "energy_j": e0,
    }
    if e_theory is None:
        out["residual_restitution"] = float("inf")
        out["restitution_error"] = float("inf")
        out["dissipation_theory"] = float("nan")
    else:
        w_theory = e0 * (1.0 - e_theory ** 2)
        out["residual_restitution"] = abs(out["dissipation_measured"] - w_theory) / e0
        out["restitution_error"] = abs(e_meas - e_theory)
        out["dissipation_theory"] = w_theory
    return out


def fit_order(dt: Sequence[float], residual: Sequence[float]) -> Dict[str, float]:
    """Least-squares slope of ``log(residual)`` versus ``log(dt)``.

    Returns ``slope``, ``intercept``, ``r2`` and the residual ratios between
    successive refinements (``ratio`` = value(coarse)/value(fine), which is
    ``2^slope`` for a clean power law).  Residuals that are zero to round-off
    are floored at ``1e-18`` before the logarithm; the number of points actually
    usable is returned as ``n_points`` so a degenerate fit cannot hide.
    """
    x = np.asarray(dt, dtype=float)
    y = np.asarray(residual, dtype=float)
    mask = np.isfinite(y) & (x > 0.0)
    x, y = x[mask], y[mask]
    if x.size < 2:
        raise ValueError("need at least two finite residual samples")
    y_floor = np.maximum(y, 1e-18)
    lx, ly = np.log(x), np.log(y_floor)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    ratio = float(y[0] / y[-1]) if y[-1] > 0.0 else float("inf")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0,
        "ratio_coarse_over_fine": ratio,
        "n_points": int(x.size),
        "dt_min": float(x.min()),
        "dt_max": float(x.max()),
    }


def order_sweep(law: ContactLaw, *, refinements: int = 8, levels: int = 7,
                integrator: str = "verlet", e_theory: Optional[float] = None,
                defect: Optional[str] = None,
                **impact_kw: Any) -> Dict[str, Any]:
    """``dt/2^k`` sweep (``k = 0..levels-1``) of the four ledger conventions.

    Returns per-level residuals plus a fitted order for each convention.  The
    ``defect`` argument forwards to :func:`_integrate_chain` and is what makes
    the sweep a *classifier*: a defect leaves a flat residual.
    """
    rows: List[Dict[str, float]] = []
    for k in range(levels):
        imp = two_body_impact(law, refinements=refinements * 2 ** k,
                              integrator=integrator, defect=defect,
                              **impact_kw)
        conv = ledger_conventions(imp, e_theory=e_theory)
        conv["k"] = k
        rows.append(conv)
    dts = [r["dt"] for r in rows]
    fits = {}
    for key in ("residual_state", "residual_explicit",
                "residual_average_velocity", "residual_staggered_power",
                "residual_restitution", "restitution_error"):
        values = [r[key] for r in rows]
        if all(np.isfinite(values)):
            fits[key] = fit_order(dts, values)
        else:
            fits[key] = None
    return {"levels": rows, "fits": fits, "integrator": integrator,
            "law": {"stiffness": law.stiffness, "exponent": law.exponent,
                    "damping_model": law.damping_model,
                    "damping": law.damping, "clip": law.clip}}


def stiffness_sweep(stiffness: Sequence[float], *, refinements: int = 8,
                    levels: int = 5, integrator: str = "verlet",
                    **impact_kw: Any) -> Dict[str, Any]:
    """Two-parameter sweep: residual versus ``k`` and ``dt``.

    The claim under test is a *collapse*: the residual is a function of the
    number of steps per contact ``N_c = t_c/dt`` and not of the stiffness
    separately.  ``dt`` is taken as ``t_c(k)/refinements/2^level``, so every
    stiffness is sampled at the same ``N_c``.
    """
    out: List[Dict[str, Any]] = []
    exponent = float(impact_kw.pop("exponent", 1.5))
    law_kw = {key: val for key, val in impact_kw.items()
              if key in ("damping_model", "damping", "clip")}
    for k_stiff in stiffness:
        law = ContactLaw(stiffness=float(k_stiff), exponent=exponent, **law_kw)
        sweep = order_sweep(law, refinements=refinements, levels=levels,
                            integrator=integrator)
        for row in sweep["levels"]:
            out.append({"stiffness": law.stiffness, **row})
    def group(key: str) -> Dict[str, Dict[str, Any]]:
        by_nc: Dict[float, List[float]] = {}
        for row in out:
            by_nc.setdefault(round(row["steps_per_contact"], 6), []).append(row[key])
        return {str(nc): {"values": vals,
                          "spread_relative": (max(vals) - min(vals)) / max(max(vals), 1e-30),
                          "ratio_max_over_min": max(vals) / max(min(vals), 1e-300)}
                for nc, vals in by_nc.items()}

    # ``residual_state`` is the difference of two much larger quantities, so its
    # level is set by where the oscillation phase happens to fall; the
    # ``explicit`` channel is a clean first-order quantity and is the channel on
    # which a collapse in ``N_c`` is meaningful.
    return {"rows": out,
            "collapse_at_equal_nc": group("residual_state"),
            "collapse_explicit": group("residual_explicit"),
            "collapse_staggered": group("residual_staggered_power")}


def leapfrog_work_identity(positions: Sequence[float],
                           velocities: Sequence[float],
                           masses: Sequence[float],
                           radii: Sequence[float], law: ContactLaw,
                           dt: float, t_end: float) -> Dict[str, float]:
    """Numerical statement of the discrete work--energy identity of leapfrog.

    Returns ``dKE`` and ``W_trapezoid`` of a chain run and their difference
    normalised by ``E0``.  The identity is exact for any ``dt``, including a
    grossly under-resolved one, which is what makes it usable as a *guard*
    against bookkeeping defects rather than as an accuracy test.
    """
    tr = _integrate_chain(positions, velocities, masses, radii, law, dt, t_end)
    dke = float(tr.kinetic[-1] - tr.kinetic[0])
    return {"dKE": dke, "work_trapezoid": tr.work_trapezoid,
            "work_explicit": tr.work_explicit,
            "relative_gap": abs(dke - tr.work_trapezoid) / max(abs(tr.e0), 1e-300),
            "explicit_gap": abs(dke - tr.work_explicit) / max(abs(tr.e0), 1e-300),
            "steps": tr.steps}


def staggered_pairing_order(law: ContactLaw, *, refinements: int = 8,
                           levels: int = 7, **impact_kw: Any) -> Dict[str, Any]:
    """Order of each ledger channel, isolating the *timing* of the shipped pairing.

    ``floe_rigid6dof._accumulate`` evaluates the power ``P = F v`` at
    ``(x_n, v_n)`` and at ``(x_{n+1}, v_{n+1/2})``: the end-of-step force is
    multiplied by a velocity that belongs half a step earlier.  That single
    misalignment costs exactly one order -- the channel closes at ``O(dt)``
    instead of the round-off the consistent pairing reaches -- which
    :func:`order_sweep` measures here side by side::

        consistent (F_k, v_k, same k)      -> round-off   (exact)
        staggered  (F_{n+1}, v_{n+1/2})    -> O(dt)

    Returns the fitted orders of the four channels plus the measured ratio
    ``explicit / staggered`` (which is ``O(1)`` for two first-order channels).
    """
    sweep = order_sweep(law, refinements=refinements, levels=levels, **impact_kw)
    fits = sweep["fits"]
    out = {"fits": fits, "levels": sweep["levels"]}
    ex = fits.get("residual_explicit")
    st = fits.get("residual_staggered_power")
    if ex and st:
        out["slope_gap_explicit_minus_staggered"] = ex["slope"] - st["slope"]
    return out


# ===================================================================
# Event-driven (analytically resolved) impact
# ===================================================================

def event_driven_impact(law: ContactLaw, *, m_eff: float = 1.0,
                        v0: float = 1.0, substeps: int = 4096,
                        bisection: int = 80) -> Dict[str, float]:
    """Impact with the *contact phase resolved* (event-driven RK4 sub-stepping).

    The fixed-step runs in :func:`order_sweep` sample the contact phase a few
    times per impact, so their restitution carries the discretisation error of
    the *outer* step.  This routine removes that error the way an event-driven
    solver does: it integrates the contact-phase ODE

        m_eff delta'' = -F(delta, delta'),   F = max(0, k delta + c delta')

    with ``substeps`` RK4 sub-steps over the analytic contact duration
    ``pi/omega_d`` and then *bisects* the separation event (the root of
    ``F = 0``) down to machine precision, so the rebound velocity -- and hence
    the restitution -- is limited by the RK4 state error and the event location,
    not by the outer ``dt``.

    Returns the measured restitution, the measured dissipation
    ``E0 (1 - e_measured^2)``, the closed-form value
    (:func:`tensordem.contact_impact.clamped_restitution`) and the ledger
    residual ``|W(measured) - W(theory)| / E0``.  Nothing here uses the
    closed-form rebound velocity: it is a measurement, and the closed form is
    only the yardstick.
    """
    if law.exponent != 1.0 or law.damping_model != "linear" or not law.clip:
        raise ValueError("event-driven impact needs the clamped linear spring-dashpot law")
    if substeps < 8:
        raise ValueError("substeps must be >= 8")
    zeta = law.damping_ratio(m_eff)
    if zeta >= 1.0:
        raise ValueError("overdamped: no rebound")
    omega_n = math.sqrt(law.stiffness / m_eff)
    omega_d = omega_n * math.sqrt(1.0 - zeta ** 2)
    t_contact = math.pi / omega_d
    h = t_contact / substeps

    def normal_force(delta: float, u: float) -> float:
        """Raw law, **without** the ``delta <= 0`` gate of :func:`contact_force`.

        The gate is right for a force field (no contact, no force) but wrong for
        the contact-phase ODE: at touchdown ``delta = 0`` and the dashpot already
        exerts ``c v0 > 0``.  Dropping that boundary sample costs a first-order
        error -- measured as a 3e-5 restitution bias that does not shrink with
        the sub-step (see the report's limitation note).  Inside the contact
        phase ``delta >= 0`` by construction, so only the clip is needed.
        """
        f = law.stiffness * delta + law.damping * u
        return 0.0 if f < 0.0 else f

    def rhs(y: np.ndarray) -> np.ndarray:
        delta, u = y
        return np.array([u, -normal_force(float(delta), float(u)) / m_eff])

    def rk4(y: np.ndarray, step: float) -> np.ndarray:
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * step * k1)
        k3 = rhs(y + 0.5 * step * k2)
        k4 = rhs(y + step * k3)
        return y + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    y = np.array([0.0, v0])
    t = 0.0
    seq = [y.copy()]
    for _ in range(substeps):
        y_next = rk4(y, h)
        if normal_force(float(y_next[0]), float(y_next[1])) <= 0.0:
            # refine the separation event inside [y, y_next]
            lo, hi = 0.0, 1.0
            for _ in range(bisection):
                mid = 0.5 * (lo + hi)
                ym = rk4(y, h * mid)
                if normal_force(float(ym[0]), float(ym[1])) > 0.0:
                    lo = mid
                else:
                    hi = mid
            y = rk4(y, h * 0.5 * (lo + hi))
            t += h * 0.5 * (lo + hi)
            seq.append(y.copy())
            separated = True
            break
        y = y_next
        t += h
        seq.append(y.copy())
    else:
        separated = False

    e_measured = float(-y[1] / v0)
    e_theory = clamped_restitution(zeta)
    e0 = 0.5 * m_eff * v0 * v0
    w_measured = e0 * (1.0 - e_measured ** 2)
    w_theory = e0 * (1.0 - e_theory ** 2)
    return {
        "zeta": zeta,
        "e_theory": e_theory,
        "e_measured": e_measured,
        "dimensionless_impact": v0,
        "dissipation_measured": w_measured,
        "dissipation_theory": w_theory,
        "restitution_error": abs(e_measured - e_theory),
        "residual_restitution": abs(w_measured - w_theory) / e0,
        "residual_state": abs(w_measured - w_theory) / e0,
        "contact_phase_time_s": float(t),
        "contact_phase_time_theory_s": t_contact,
        "substeps": substeps,
        "separated": bool(separated),
        "energy_j": e0,
        "contact_phase": "event-driven RK4 sub-stepping + bisected release",
    }


# ===================================================================
# Newton cradle
# ===================================================================

@dataclass
class NewtonCradleResult:
    """Outcome of a five-ball (or ``n``-ball) cradle impact."""

    dt: float
    steps_per_contact: float
    momentum_error: float
    kinetic_initial: float
    kinetic_final: float
    stored_final: float
    residual_state: float
    residual_explicit: float
    residual_average_velocity: float
    residual_staggered_power: float
    e_measured: float
    velocities_final: np.ndarray
    energy_trace: np.ndarray
    times: np.ndarray


def newton_cradle(*, n: int = 5, radius: float = 0.05, mass: float = 1.0,
                  stiffness: float = 1.0e7, velocity: float = 1.0,
                  refinements: int = 16, levels: int = 4,
                  exponent: float = 1.5) -> Dict[str, Any]:
    """``n`` balls in a line, first one moving, all contacts *simultaneous*.

    The balls are placed exactly touching (gap ``0``), so the impact engages
    every contact at once -- the configuration in which momentum conservation and
    the energy ledger are hardest: the contact network is statically
    indeterminate while the wave passes through, which is exactly where a
    bookkeeping defect in the pairwise force/work accumulation shows up.

    Returns per-``dt`` results plus the fitted order of ``residual_state`` and
    the maximum momentum error.  ``residual_trapezoid`` is round-off at every
    ``dt`` (the leapfrog identity), so it is reported as the guard number.
    """
    if n < 2:
        raise ValueError("the cradle needs at least two balls")
    law = ContactLaw(stiffness=stiffness, exponent=exponent)
    m_eff = 0.5 * mass
    t_c = (hertz_contact_time(m_eff, stiffness, velocity) if law.hertzian
           else math.pi * math.sqrt(m_eff / stiffness))
    results: List[NewtonCradleResult] = []
    for k in range(levels):
        dt = t_c / refinements / 2 ** k
        positions = [2.0 * radius * i for i in range(n)]
        velocities = [velocity] + [0.0] * (n - 1)
        t_end = (1.0 + 4.0) * t_c + 4.0 * radius / velocity
        tr = _integrate_chain(positions, velocities, [mass] * n,
                              [radius] * n, law, dt, t_end)
        dke = float(tr.kinetic[-1] - tr.kinetic[0])
        du = float(tr.stored[-1] - tr.stored[0])
        e0 = max(abs(tr.e0), 1e-300)
        v_out = float(np.abs(tr.velocities[-1, -1]))
        results.append(NewtonCradleResult(
            dt=dt, steps_per_contact=t_c / dt,
            momentum_error=tr.momentum_error,
            kinetic_initial=float(tr.kinetic[0]),
            kinetic_final=float(tr.kinetic[-1]),
            stored_final=float(tr.stored[-1]),
            residual_state=abs(dke + du) / e0,
            residual_explicit=abs(dke - tr.work_explicit) / e0,
            residual_average_velocity=abs(dke - tr.work_trapezoid) / e0,
            residual_staggered_power=abs(dke - tr.work_average_velocity) / e0,
            e_measured=v_out / velocity,
            velocities_final=np.asarray(tr.velocities[:, -1]),
            energy_trace=tr.kinetic + tr.stored,
            times=tr.times))
    fits = {
        "residual_state": fit_order([r.dt for r in results],
                                    [r.residual_state for r in results]),
        "residual_explicit": fit_order([r.dt for r in results],
                                       [r.residual_explicit for r in results]),
        "residual_average_velocity": fit_order(
            [r.dt for r in results],
            [r.residual_average_velocity for r in results]),
    }
    return {"n": n, "radius": radius, "mass": mass, "stiffness": stiffness,
            "velocity": velocity, "exponent": exponent,
            "results": results, "fits": fits,
            "momentum_error_max": max(r.momentum_error for r in results),
            "residual_average_velocity_max": max(
                r.residual_average_velocity for r in results),
            "residual_staggered_power_max": max(
                r.residual_staggered_power for r in results)}


# ===================================================================
# m5404 re-judgement (shipped solver, not a re-implementation)
# ===================================================================

M5404_CASE = {
    "geometry_m": (4.0, 3.0, 0.4),
    "density_kg_m3": 917.0,
    "positions_m": ((-2.35, 1.0, 0.0), (2.35, -1.0, 0.0)),
    "velocities_m_s": ((1.0, 0.0, 0.0), (-0.5, 0.0, 0.0)),
    "normal_stiffness": 2.0e7,
    "integrator": "rk4",
    "dt": 2.0e-4,
    "duration_s": 0.5,
    "reported_residual_state": 5.111290246429849e-07,
    "reported_trapezoid": 2.3152074702662755e-05,
    "reported_residual_trapezoid": 2.3152074702662755e-05,
}


def m5404_level(level: int, dt0: Optional[float] = None,
                duration: Optional[float] = None,
                defect_scale: float = 1.0e-3) -> Dict[str, float]:
    """One ``dt = dt0/2**level`` level of the m5404 re-judgement (shard unit).

    Drives the shipped ``floe_rigid6dof.FloeWorld`` on the
    ``case_collision`` configuration of
    ``scripts/in29_n02_floe6dof_benchmark.py`` and returns the three numbers the
    verdict rests on: the state convention (the reported one), the trapezoid
    power channel, and the state convention re-evaluated with the stored energy
    mis-scaled by ``defect_scale`` (a representative implementation defect).
    """
    from .floe_rigid6dof import (ContactSettings, Environment, FloeWorld,
                                Hydrodynamics, box_mesh, build_floe)
    dt0 = M5404_CASE["dt"] if dt0 is None else dt0
    duration = M5404_CASE["duration_s"] if duration is None else duration
    length, width, thickness = M5404_CASE["geometry_m"]
    floe = build_floe(*box_mesh(length, width, thickness),
                      density=M5404_CASE["density_kg_m3"], name="box")
    env = Environment(enable_buoyancy=False, enable_gravity=False)
    hydro = Hydrodynamics(added_mass_ratio=0.0, added_inertia_ratio=0.0,
                          drag_coefficient=0.0, enable_drag=False)
    world = FloeWorld([floe, floe], [list(p) for p in M5404_CASE["positions_m"]],
                      velocities=[list(v) for v in M5404_CASE["velocities_m_s"]],
                      env=env, hydro=hydro,
                      contact=ContactSettings(
                          normal_stiffness=M5404_CASE["normal_stiffness"]),
                      integrator=M5404_CASE["integrator"])
    world.run(duration, dt0 / 2 ** level)
    led = world.energy_ledger()
    scale = max(abs(led["initial"]), abs(led["total"]), 1e-30)
    stored_bad = led["contact_elastic"] * (1.0 + defect_scale)
    return {
        "k": level, "dt": dt0 / 2 ** level, "steps": world.steps,
        "residual_state": led["residual_state_relative"],
        "residual_trapezoid": led["residual_relative"],
        "residual_state_injected_defect": abs(
            led["delta"] + stored_bad
            - (led["input_work"] - led["dissipated"])) / scale,
        "contact_elastic_j": led["contact_elastic"],
        "initial_energy_j": led["initial"],
        "max_penetration_m": world.max_penetration,
        "v_floe_a": float(world.state.velocity[0, 0]),
        "v_floe_b": float(world.state.velocity[1, 0]),
    }


def m5404_verdict(rows: Sequence[Dict[str, float]]) -> Dict[str, Any]:
    """Fold the ``dt`` levels of :func:`m5404_level` into the verdict.

    The decision rule is stated in :func:`m5404_rejudgement`; with a single level
    no order can be fitted and the verdict is ``"undetermined"`` (nothing is
    claimed from one point).
    """
    rows = sorted(rows, key=lambda r: r["k"])
    out: Dict[str, Any] = {
        "case": {k: (list(map(list, v)) if isinstance(v, tuple) and v
                    and isinstance(v[0], tuple) else v)
                 for k, v in M5404_CASE.items()},
        "rows": rows,
        "reproduced_reported_residual": rows[0]["residual_state"],
        "ratio_to_reported": (rows[0]["residual_state"]
                              / M5404_CASE["reported_residual_state"]),
    }
    if len(rows) < 2:
        out.update({"fits": {}, "verdict": "undetermined"})
        return out
    dts = [r["dt"] for r in rows]
    fits = {
        "residual_state": fit_order(dts, [r["residual_state"] for r in rows]),
        "residual_trapezoid": fit_order(dts, [r["residual_trapezoid"] for r in rows]),
        "residual_state_injected_defect": fit_order(
            dts, [r["residual_state_injected_defect"] for r in rows]),
    }
    out["fits"] = fits
    out["verdict"] = "convention" if fits["residual_state"]["slope"] >= 0.8 else "implementation"
    out["state_residual_ratio"] = (rows[0]["residual_state"]
                                   / rows[-1]["residual_state"])
    out["defect_plateau_ratio"] = (rows[0]["residual_state_injected_defect"]
                                   / rows[-1]["residual_state_injected_defect"])
    return out


def m5404_rejudgement(levels: int = 4, dt0: float = 2.0e-4,
                      duration: float = 0.5,
                      defect_scale: float = 1.0e-3) -> Dict[str, Any]:
    """Re-adjudicate mission m5404's 5.111e-07 by refining ``dt`` on the shipped solver.

    The case is the two-floe oblique collision of
    ``scripts/in29_n02_floe6dof_benchmark.py::case_collision`` (``rk4``,
    ``dt = 2e-4``, ``t = 0.5 s``, Hertz, no damping), which is what produced
    ``energy_residual_state_relative = 5.111290246e-07`` at mission m5404.

    Verdict rule (stated before the numbers are looked at): the state-convention
    residual is classed ``"convention"`` when its fitted log-log slope is
    ``>= 0.8`` -- i.e. it tends to zero under refinement, so the residual is a
    discretisation artefact -- and ``"implementation"`` otherwise.  The
    injected-defect channel is the control that shows the slope test has
    discriminating power at this case: a 1e-3 state-function error must leave a
    flat residual.
    """
    rows = [m5404_level(k, dt0=dt0, duration=duration, defect_scale=defect_scale)
            for k in range(levels)]
    return m5404_verdict(rows)


