"""Closed-loop verification of the restitution <-> contact-damping mapping.

``tensordem.contact_tangential.restitution_to_damping`` maps a target
coefficient of restitution *e* onto a normal damping coefficient ``gamma_n``
using the closed-form result for an *unclamped* linear spring-dashpot
oscillator::

    e_classic = exp(-pi * zeta / sqrt(1 - zeta^2)),
    zeta      = gamma_n / (2 * sqrt(k_n * m_eff)).

The solvers in this repository (``tensordem.dem``, ``tensordem.dem3d``) do not
integrate an unclamped oscillator: they evaluate a *compressive-only* normal
force, ``F_n = max(k_n * overlap - gamma_n * v_n, 0)``.  Once the damping term
exceeds the elastic term the contact force is switched off while the spheres are
still overlapping, so contact terminates early and the coefficient of
restitution observed in a simulation is **larger** than ``e_classic``.  The
mismatch is negligible for small damping ratios and becomes large
(> 15 %) around ``zeta ~ 0.3``.

This module supplies three things so that the gap can be measured instead of
assumed:

1. :func:`clamped_restitution_ratio` -- a high-accuracy reference integration of
   the *clamped* one-dimensional impact model (no sphere/neighbour bookkeeping),
   which is the continuum limit the solver is supposed to reproduce.
2. :func:`damping_for_target_restitution` -- inverse mapping in the clamped
   model, i.e. the damping coefficient that actually yields a requested rebound
   coefficient in this solver's contact law (bisection, no fitting).
3. :func:`measure_dem3d_impact` -- a closed-loop measurement with the real
   :class:`tensordem.dem3d.IceDEM3D` solver, so that the mapping can be validated
   against the code that is actually shipped rather than against itself.

All quantities are SI (N, m, s, kg). The reference model is drag free; the
solvers additionally apply a global viscous drag ``-drag * v`` to every
particle, which is a separate dissipation channel that must be switched off to
isolate the contact law (see the ``drag`` argument of
:func:`measure_dem3d_impact`).
"""
from __future__ import annotations

import math
from typing import Any

import torch

from .contact_tangential import damping_to_restitution, restitution_to_damping

__all__ = [
    "classic_restitution_ratio",
    "clamped_restitution_ratio",
    "damping_for_target_restitution",
    "measure_dem3d_impact",
]


def classic_restitution_ratio(zeta: float) -> float:
    """Unclamped linear spring-dashpot restitution for damping ratio ``zeta``.

    ``e = exp(-pi * zeta / sqrt(1 - zeta^2))``; returns 0.0 at and beyond
    critical damping (the oscillator no longer returns).
    """
    if not math.isfinite(zeta) or zeta < 0:
        raise ValueError(f"zeta must be finite and nonnegative, got {zeta}")
    if zeta >= 1.0:
        return 0.0
    return math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta * zeta))


def clamped_restitution_ratio(
    gamma_n: float,
    k_n: float,
    m_eff: float,
    *,
    v_in: float = 1.0,
    steps: int = 4_000,
) -> dict[str, Any]:
    """Reference restitution of the compressive-only spring-dashpot impact.

    Integrates, with classical RK4 and a fixed step count per contact,

        m_eff * d2(delta) = -max(k_n * delta + gamma_n * d(delta), 0)

    from ``delta(0) = 0``, ``d(delta)(0) = v_in`` until the overlap returns to
    zero. Here ``delta`` is the overlap and ``v_in`` the closing speed; the
    returned restitution is ``|d(delta)| / v_in`` at separation.

    The clamping is the only difference from :func:`classic_restitution_ratio`;
    it makes the contact force vanish while ``delta > 0`` still holds, which
    leaves elastic energy unspent and raises the observed restitution.

    Returns a dict with the restitution, the damping ratio, the contact
    duration, and the residual overlap at which the force was switched off.
    """
    for name, value in (("k_n", k_n), ("m_eff", m_eff), ("v_in", v_in)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(gamma_n) or gamma_n < 0:
        raise ValueError(f"gamma_n must be finite and nonnegative, got {gamma_n}")
    if steps < 200:
        raise ValueError("steps must be >= 1000")
    if gamma_n == 0.0:
        return {
            "e_reference": 1.0,
            "zeta": 0.0,
            "contact_time_s": math.pi * math.sqrt(m_eff / k_n),
            "release_overlap_m": 0.0,
            "released_early": False,
        }

    zeta = gamma_n / (2.0 * math.sqrt(k_n * m_eff))
    # Step from the undamped period so that the integration stays stable (and
    # the step count meaningful) also for critical and over-critical damping.
    contact_guess = math.pi * math.sqrt(m_eff / k_n)
    dt = contact_guess / steps

    def accel(delta: float, rate: float) -> float:
        force = max(k_n * delta + gamma_n * rate, 0.0)
        if force <= 0.0 and rate < 0.0:
            return 0.0
        return -force / m_eff

    delta, rate = 0.0, v_in
    t = 0.0
    released = False
    release_delta = 0.0
    max_delta = 0.0
    max_steps = 40 * steps
    for _ in range(max_steps):
        if not released and k_n * delta + gamma_n * rate <= 0.0:
            # Contact force switched off while still overlapping: the particle
            # now coasts at constant velocity (clamped-law separation).
            released = True
            release_delta = delta
        a1 = accel(delta, rate)
        d2, r2 = delta + 0.5 * dt * rate, rate + 0.5 * dt * a1
        a2 = accel(d2, r2)
        d3, r3 = delta + 0.5 * dt * r2, rate + 0.5 * dt * a2
        a3 = accel(d3, r3)
        d4, r4 = delta + dt * r3, rate + dt * a3
        a4 = accel(d4, r4)
        new_delta = delta + dt * (rate + 2.0 * r2 + 2.0 * r3 + r4) / 6.0
        new_rate = rate + dt * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0
        t += dt
        max_delta = max(max_delta, new_delta)
        if new_delta <= 0.0 and rate < 0.0 and t > dt:
            # Locate the delta = 0 crossing linearly inside the final step.
            frac = delta / (delta - new_delta) if delta != new_delta else 1.0
            rate_out = rate + frac * (new_rate - rate)
            return {
                "e_reference": abs(rate_out / v_in),
                "zeta": zeta,
                "contact_time_s": t - dt + frac * dt,
                "release_overlap_m": release_delta,
                "released_early": released,
                "max_overlap_m": max_delta,
            }
        delta, rate = new_delta, new_rate
    raise RuntimeError("clamped reference impact did not separate; check inputs")


def damping_for_target_restitution(
    e_target: float,
    k_n: float,
    m_eff: float,
    *,
    bracket: tuple[float, float] = (0.0, 2.0),
    iterations: int = 60,
) -> dict[str, Any]:
    """Damping coefficient that yields ``e_target`` under the clamped law.

    Bisection on ``gamma_n / (2 sqrt(k_n m_eff))`` in ``bracket``, evaluating the
    clamped reference model at every trial. This is the mapping a user should
    apply when the requested coefficient of restitution must be reproduced by
    this solver's compressive-only contact; ``restitution_to_damping`` is only
    accurate for small damping ratios (see the module docstring).
    """
    if not (0.0 < e_target <= 1.0):
        raise ValueError(f"e_target must be in (0, 1], got {e_target}")
    if e_target == 1.0:
        return {
            "gamma_n": 0.0,
            "zeta": 0.0,
            "e_target": 1.0,
            "e_classic_match": 0.0,
        }
    lo, hi = bracket
    if not (0.0 <= lo < hi):
        raise ValueError("bracket must satisfy 0 <= lo < hi")
    root = math.sqrt(k_n * m_eff)
    e_lo = clamped_restitution_ratio(2.0 * lo * root, k_n, m_eff, steps=4_000)["e_reference"]
    e_hi = clamped_restitution_ratio(2.0 * hi * root, k_n, m_eff, steps=4_000)["e_reference"]
    if not (e_hi <= e_target <= e_lo):
        raise ValueError(
            f"e_target={e_target} not bracketed by zeta in [{lo}, {hi}] "
            f"(e spans [{e_hi:.4f}, {e_lo:.4f}])"
        )
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        e_mid = clamped_restitution_ratio(2.0 * mid * root, k_n, m_eff)["e_reference"]
        if e_mid > e_target:
            lo = mid
        else:
            hi = mid
    zeta = 0.5 * (lo + hi)
    gamma_n = 2.0 * zeta * root
    e_check = clamped_restitution_ratio(gamma_n, k_n, m_eff)["e_reference"]
    # Distance from the classic closed form, which the shipped solvers do not obey.
    e_classic = damping_to_restitution(gamma_n, k_n, m_eff)
    return {
        "gamma_n": gamma_n,
        "zeta": zeta,
        "e_target": e_target,
        "e_reference": e_check,
        "e_classic_of_same_gamma": e_classic,
        "classic_gap_relative": (e_classic - e_target) / e_target,
        "gamma_classic_mapping": restitution_to_damping(e_target, k_n, m_eff),
    }


def measure_dem3d_impact(
    gamma_n: float,
    *,
    k_n: float = 5_000.0,
    radius: float = 0.025,
    density: float = 917.0,
    drag: float = 0.0,
    dt_scale: float = 1.0,
    v_in: float = 0.5,
    gap: float = 0.01,
    both_free: bool = False,
) -> dict[str, Any]:
    """Closed-loop restitution measurement with the real 3-D solver.

    A two-sphere impact is carved out of a ``2x2x2`` :class:`IceDEM3D` box: all
    bonds are severed and all particles except the pair are frozen, so the pair
    interacts through the ordinary contact branch only (the tool and platens are
    parked far away). With ``both_free=False`` the second sphere is frozen and
    behaves as an infinite-mass wall (``m_eff = m``); with ``both_free=True`` two
    equal masses approach each other (``m_eff = m / 2``) and both are released.

    The coefficient of restitution is measured from the relative normal velocity
    on the first step with overlap and on the first step after the spheres
    separate again, i.e. it is the quantity a user would extract from a real run.
    """
    from .dem3d import DEM3DConfig, IceDEM3D

    if not math.isfinite(gamma_n) or gamma_n < 0:
        raise ValueError(f"gamma_n must be finite and nonnegative, got {gamma_n}")
    if not (0.0 < dt_scale <= 1.0):
        raise ValueError("dt_scale must be in (0, 1]")
    if not math.isfinite(gap) or gap <= 0:
        raise ValueError("gap must be finite and positive")

    torch.set_num_threads(1)
    config = DEM3DConfig(
        nx=2, ny=2, nz=2,
        radius=radius, density=density,
        contact_stiffness=k_n, contact_damping=gamma_n,
        # DEM3DConfig defaults to the legacy "restitution" dashpot convention,
        # in which ``contact_damping`` does not enter the pair law at all. This
        # harness measures the *absolute* dashpot coefficient, so the pair law
        # must be the clipped F = max(k delta - gamma_n v_n, 0) the reference
        # integration in this module solves.
        contact_damping_model="absolute",
        drag=drag, fix_x_edges=False,
        tool_speed=1e-3, tool_gap=1.0,
    )
    sim = IceDEM3D(config)
    # Carve the two-body impact out of the bonded lattice.
    sim.bonded[:] = False
    sim.alive[:] = False
    sim.fixed[:] = True
    left, right = 0, 1
    sim.fixed[left] = False
    if both_free:
        sim.fixed[right] = False
    sim.positions[left, 0] -= gap
    sim.initial_positions[left, 0] -= gap
    sim.velocities[left, 0] = v_in
    if both_free:
        sim.positions[right, 0] += gap
        sim.initial_positions[right, 0] += gap
        sim.velocities[right, 0] = -v_in
    sim.dt = config.recommended_dt * dt_scale

    contact_distance = 2.0 * radius
    m_eff = config.mass if not both_free else 0.5 * config.mass
    in_contact = False
    v_in_measured = 0.0
    contact_steps = 0
    max_overlap = 0.0
    max_steps = 400_000
    for _ in range(max_steps):
        distance = float(torch.linalg.vector_norm(sim.positions[left] - sim.positions[right]))
        relative = float(sim.velocities[left, 0] - sim.velocities[right, 0])
        if not in_contact and distance < contact_distance:
            in_contact = True
            v_in_measured = relative
        if in_contact:
            contact_steps += 1
            max_overlap = max(max_overlap, contact_distance - distance)
            if distance >= contact_distance and relative < 0.0:
                return {
                    "e_measured": abs(relative / v_in_measured),
                    "dt": sim.dt,
                    "recommended_dt": config.recommended_dt,
                    "dt_scale": dt_scale,
                    "v_in": v_in_measured,
                    "v_out": relative,
                    "contact_time_s": contact_steps * sim.dt,
                    "max_overlap_m": max_overlap,
                    "m_eff": m_eff,
                    "mass": config.mass,
                    "zeta": gamma_n / (2.0 * math.sqrt(k_n * m_eff)),
                    "steps": sim.step_count,
                    "drag": drag,
                }
        sim.step()
    raise RuntimeError("3-D impact did not separate; check gap/v_in/dt")