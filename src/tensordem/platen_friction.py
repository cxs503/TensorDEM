"""Coulomb friction at the infinite platen boundaries, with closed-form anchors.

Why this module exists
----------------------
``tensordem.dem3d`` modelled both infinite planes (``top_platen``/``bottom_platen``)
as perfectly smooth anvils: they could only push along ``z``.  Every platen
benchmark in this repository (uniaxial compression, direct shear, the platen
contact-mask regressions) therefore measured a specimen against a **frictionless**
boundary, and no configuration could represent a rough anvil, a shear box with
gripping faces, or the tangential traction that a real steel platen carries.
The tangential traction at ``z``-planes was pinned to zero by construction, not
by physics.

``DEM3DConfig.platen_friction_coefficient`` (``mu_p``) closes that: the planes now
carry a rotationless Cundall-Strack tangential spring whose force
``F_t = -k_t * delta_t`` is saturated at ``mu_p * N``, where ``delta_t`` integrates
the particle's tangential travel relative to the (tangentially stationary) plane
and ``N`` is the normal force of that same plane contact.  Default ``mu_p = 0``
keeps the legacy frictionless branch bit-for-bit.

What this module verifies
-------------------------
Anchors that are exact for the *discrete* system, not merely asymptotic, so a
sub-1 % agreement cannot be an artefact of small time steps:

* **Settled normal load.** A raft of particles loaded with a constant vertical
  force ``F_n`` per particle on a platen of stiffness ``k_p`` sinks by
  ``delta_z = -F_n / k_p`` and carries ``N = F_n``.
* **Static (stick) branch.** A tangential force ``F_h < mu_p * N`` leaves the raft
  at rest with a purely elastic offset ``delta_x = F_h / k_t`` -- the Cundall-
  Strack static friction solution.
* **Onset bracket.** ``F_h = (1 -/+ eps) * mu_p * N`` sticks / slips, so the
  Coulomb threshold is located to within ``eps``.
* **Kinetic (slip) branch.** Once saturated, the net tangential force is the
  constant ``F_h - mu_p * N``, so the raft slides at exactly
  ``a = (F_h - mu_p * N) / m`` regardless of ``k_t``: a linear-in-time velocity
  whose slope is a closed-form number.

Everything is SI.  No gravity in this solver, so ``N`` is imposed by an external
vertical load rather than by weight; that keeps the anchor free of hydrostatic
or buoyancy modelling.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch

from .dem3d import DEM3DConfig, IceDEM3D

__all__ = [
    "ACCURACY_GATE",
    "platen_friction_limit",
    "platen_static_offset",
    "platen_slip_acceleration",
    "settle_normal_load",
    "platen_friction_case",
    "platen_friction_sweep",
]

#: Platform accuracy gate for a measured-vs-analytic comparison.
ACCURACY_GATE = 0.03

_TOOL_PARK = 1.0e6


def platen_friction_limit(mu_p: float, normal_force: float) -> float:
    """Coulomb limit ``mu_p * N`` at a plane contact (N)."""
    return mu_p * abs(normal_force)


def platen_static_offset(tangential_force: float, stiffness: float) -> float:
    """Static (stick) elastic offset of the friction spring, ``F_h / k_t`` (m)."""
    if stiffness <= 0.0:
        raise ValueError("stiffness must be positive")
    return tangential_force / stiffness


def platen_slip_acceleration(tangential_force: float, mu_p: float,
                             normal_force: float, mass: float) -> float:
    """Sliding acceleration once the friction spring is saturated, m/s^2.

    The friction force is ``-mu_p * N`` against the motion, so the net tangential
    force is the constant ``F_h - mu_p * N``; drag must be zero for this to be the
    whole story (the driver below runs the slip phase with ``drag = 0``).
    """
    if mass <= 0.0:
        raise ValueError("mass must be positive")
    return (abs(tangential_force) - mu_p * abs(normal_force)) / mass


def _raft(
    *, mu_p: float, platen_stiffness: float, friction_stiffness: float | None,
    platen_damping: float, drag: float, dt: float | None, platen: str,
) -> tuple[IceDEM3D, torch.Tensor]:
    """Build an 8-particle cell whose lower (or upper) 4 particles are the raft.

    The other layer is frozen (``fixed``) so it cannot act as an overburden: the
    specimen transmits exactly the external load to the plane, which is what makes
    ``N`` a closed-form number.  The spherical tool is parked far away.
    """
    if platen not in ("bottom", "top"):
        raise ValueError("platen must be 'bottom' or 'top'")
    cfg = DEM3DConfig(
        nx=2, ny=2, nz=2, radius=0.025, density=917.0,
        bond_stiffness=2000.0, contact_stiffness=5000.0,
        platen_stiffness=platen_stiffness, platen_damping=platen_damping,
        top_platen_enabled=platen == "top",
        bottom_platen_enabled=platen == "bottom",
        top_platen_gap=0.0, bottom_platen_gap=0.0,
        fix_x_edges=False, drag=drag,
        platen_friction_coefficient=mu_p,
        platen_friction_stiffness=friction_stiffness,
        dt=dt,
    )
    sim = IceDEM3D(cfg)
    sim.tool_start[:] = torch.tensor([_TOOL_PARK] * 3, dtype=sim.dtype)
    sim.tool_velocity.zero_()
    z = sim.positions[:, 2]
    half = 0.5 * cfg.radius
    raft = ((z <= z.min() + half) if platen == "bottom"
            else (z >= z.max() - half))
    sim.fixed |= ~raft
    return sim, raft


def settle_normal_load(
    sim: IceDEM3D, *, normal_force: float, raft: torch.Tensor,
    platen: str = "bottom", steps: int = 4000,
    velocity_tolerance: float = 1.0e-8,
) -> dict[str, float]:
    """Load the raft vertically with ``normal_force`` per particle and let it settle.

    Returns the settled plane reaction (the analytic ``N`` input of every friction
    anchor) and the vertical sink, which must equal ``+/- F_n / k_p``.  The raft is
    integrated until the raft is at rest (velocity tolerance 1e-8 m/s, which makes
    the dashpot contribution to the reaction ~ 5e-9 N).  The settling budget is the
    reason the heavy anchor cases are not part of the fast unit-test selection.
    """
    n = len(sim.positions)
    sign = -1.0 if platen == "bottom" else 1.0
    loads = torch.zeros((n, 3), dtype=sim.dtype)
    loads[raft, 2] = sign * abs(normal_force)
    z0 = sim.positions[raft, 2].clone()
    for _ in range(max(1, steps)):
        sim.step(loads)
        if float(sim.velocities[raft].abs().max()) < velocity_tolerance:
            break
    reaction = (float(sim.last_bottom_platen_reaction_z) if platen == "bottom"
                else float(sim.last_top_platen_reaction_z))
    return {
        "reaction_N": abs(reaction),
        "per_particle_N": abs(reaction) / float(raft.sum()),
        "sink_m": float((sim.positions[raft, 2] - z0).mean()),
        "settle_steps": float(sim.step_count),
    }


def platen_friction_case(
    *, phase: str, mu_p: float, tangential_force: float, normal_force: float,
    platen: str = "bottom", stiffness: float = 5000.0,
    friction_stiffness: float | None = None, platen_damping: float = 0.5,
    drag: float = 2.0, steps: int = 1800, dt: float | None = None,
    settle_steps: int = 8000, dt_scale: float = 0.05, ramp_steps: int = 600,
    fit_from_fraction: float = 0.5,
) -> dict[str, Any]:
    """Run one stick or slip case and compare the trajectory with the closed form.

    ``phase="stick"`` needs ``drag > 0``: the damped tangential spring then settles
    at the *static* offset ``delta_x = F_h / k_t`` (the anchor).  The default
    default ``drag = 2`` N*s/m gives a tangential relaxation time ``m/drag`` ~ 30 ms,
    so a ``steps - ramp_steps = 1200`` hold leaves a residual transient far below
    the offset itself (a shorter hold measures its own incomplete settling).  ``phase="slip"``
    needs ``drag == 0``: the saturated friction force is then the only tangential
    term besides the load, so the velocity slope is exactly
    ``(F_h - mu_p N) / m``.  The two requirements are mutually exclusive by
    construction, which is why the phase is an explicit argument instead of being
    inferred from whichever number happens to look good.
    """
    if phase not in ("stick", "slip"):
        raise ValueError("phase must be 'stick' or 'slip'")
    if phase == "stick" and drag <= 0.0:
        raise ValueError("the stick anchor needs drag > 0 to settle the tangential spring")
    if phase == "slip" and drag != 0.0:
        raise ValueError("the slip anchor needs drag = 0 so the net force stays constant")
    if dt is None:
        # The friction spring is an extra oscillator; the solver's recommended dt
        # (which knows only the normal/platen stiffness set) resolves it with far
        # too few steps per period, so pin dt to a fixed fraction of 1/omega_t.
        probe, _ = _raft(mu_p=mu_p, platen_stiffness=stiffness,
                         friction_stiffness=friction_stiffness,
                         platen_damping=platen_damping, drag=drag, dt=None,
                         platen=platen)
        dt = min(dt_scale * (probe.config.mass / probe.platen_friction_k) ** 0.5,
                 probe.config.recommended_dt)
    sim, raft = _raft(mu_p=mu_p, platen_stiffness=stiffness,
                      friction_stiffness=friction_stiffness,
                      platen_damping=platen_damping, drag=drag, dt=dt,
                      platen=platen)
    settled = settle_normal_load(sim, normal_force=normal_force, raft=raft,
                                 platen=platen, steps=settle_steps)
    normal = settled["per_particle_N"]
    n = len(sim.positions)
    loads = torch.zeros((n, 3), dtype=sim.dtype)
    # The normal preload must stay on during the tangential phase: without it the
    # platen spring lifts the raft off the plane, the contact mask closes and the
    # friction history is (correctly) discarded, so there would be nothing to test.
    loads[raft, 2] = (-1.0 if platen == "bottom" else 1.0) * abs(normal_force)
    x0 = sim.positions[raft, 0].clone()
    times: list[float] = [sim.time]
    speeds: list[float] = [float(sim.velocities[raft, 0].mean())]
    peak_ratio = 0.0
    limit = platen_friction_limit(mu_p, normal)
    ramp = max(1, int(ramp_steps))
    for k in range(steps):
        # Ramp the tangential load: a step load on a light, under-damped
        # tangential spring overshoots its static offset by 1 + exp(-pi*zeta/..)
        # and can cross the Coulomb limit, which is a legitimate stick-slip
        # transient but not the branch this case is meant to measure.
        loads[raft, 0] = tangential_force * min(1.0, (k + 1) / ramp)
        sim.step(loads)
        times.append(sim.time)
        speeds.append(float(sim.velocities[raft, 0].mean()))
        friction = (sim.last_bottom_platen_friction if platen == "bottom"
                    else sim.last_top_platen_friction)
        magnitude = float(torch.linalg.vector_norm(friction[raft], dim=1).max())
        peak_ratio = max(peak_ratio, magnitude / limit if limit > 0.0 else 0.0)

    offset = float((sim.positions[raft, 0] - x0).mean())
    start = int(len(times) * fit_from_fraction)
    t = torch.tensor(times[start:], dtype=torch.float64)
    v = torch.tensor(speeds[start:], dtype=torch.float64)
    tc = t - t.mean()
    slope = float((tc * (v - v.mean())).sum() / (tc * tc).sum())
    mass = sim.config.mass
    sink_sign = -1.0 if platen == "bottom" else 1.0
    predicted_sink = sink_sign * normal_force / stiffness
    predicted_offset = platen_static_offset(tangential_force, sim.platen_friction_k)
    predicted_slope = platen_slip_acceleration(tangential_force, mu_p, normal, mass)

    def _rel(measured: float, predicted: float) -> float:
        if predicted == 0.0:
            return abs(measured)
        return abs(measured - predicted) / abs(predicted)

    measured = {
        "plane_reaction_N": settled["reaction_N"],
        "sink_m": settled["sink_m"],
        "tangential_offset_m": offset,
        "velocity_slope_m_per_s2": slope,
        "peak_friction_ratio": peak_ratio,
        "final_speed_m_per_s": speeds[-1],
        "elapsed_s": sim.time,
    }
    predicted = {
        "sink_m": predicted_sink,
        "tangential_offset_m": predicted_offset,
        "velocity_slope_m_per_s2": predicted_slope,
    }
    anchor = "tangential_offset_m" if phase == "stick" else "velocity_slope_m_per_s2"
    return {
        "phase": phase,
        "mu_p": mu_p,
        "platen": platen,
        "tangential_force_N": tangential_force,
        "normal_force_N": normal,
        "friction_limit_N": limit,
        "stiffness_N_per_m": sim.platen_friction_k,
        "settled": settled,
        "measured": measured,
        "predicted": predicted,
        "relative_error": {
            "sink": _rel(settled["sink_m"], predicted_sink),
            anchor: _rel(measured[anchor], predicted[anchor]),
        },
        "anchor": anchor,
    }


def platen_friction_sweep(
    mu_values: Sequence[float], *, normal_force: float = 10.0,
    load_ratio: float = 0.6, platen: str = "bottom",
    stiffness: float = 5000.0, slip_steps: int = 400,
) -> list[dict[str, Any]]:
    """Run the stick branch over a set of friction coefficients.

    ``F_h = load_ratio * mu_p * N`` is scaled with ``mu_p``, so every member of the
    sweep is a *stick* case and must reproduce ``delta_x = F_h / k_t``: a solution
    whose offset depended on ``mu_p`` (i.e. one that saturates too early) fails
    the sweep even though each single case could look acceptable.
    """
    rows: list[dict[str, Any]] = []
    for mu_p in mu_values:
        case = platen_friction_case(
            phase="stick", mu_p=mu_p, normal_force=normal_force,
            tangential_force=load_ratio * mu_p * normal_force,
            platen=platen, stiffness=stiffness, steps=slip_steps,
        )
        rows.append(case)
    return rows