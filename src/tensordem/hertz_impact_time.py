"""Time-domain (impact *duration*) verification of the DEM3D Hertz contact law.

Why this module exists
----------------------
``docs/dem3d_hertz_contact_verification.md`` and ``tests/test_dem3d_hertz_contact.py``
pin the Hertz law as a *static* relation: ``F = (4/3) E* sqrt(R*) delta^(3/2)``,
its potential, the secant stiffness, and (for the solver path) a single
free-fall test that recovers the **peak** force of one elastic impact.

None of those anchors constrains the *time* axis.  A contact law can reproduce
the peak force of an impact and still have a wrong contact duration -- the two
quantities depend on different combinations of the same parameters -- so the
capability was verified only up to ``F(delta)``, never up to the trajectory that
the solver actually produces.  This module closes that gap with a classical
closed-form anchor that involves time explicitly.

The literature anchor (Johnson, *Contact Mechanics*, 1985, Sec. 4.2; the same
result is in Goldsmith 1960 and in every DEM textbook): for an elastic Hertzian
impact of effective mass ``m`` on a stiffness

    k_h = (4/3) E* sqrt(R*)        [N m^(-3/2)],   F = k_h delta^(3/2),

energy conservation gives the peak overlap ``delta_max = (5 m v0^2 / (4 k_h))^(2/5)``
and the full contact duration

    t_c = 2 * I * delta_max / v0,   I = int_0^1 du / sqrt(1 - u^(5/2))

    => t_c = C * (5 m / (4 k_h))^(2/5) * v0^(-1/5),     C = 2 I = 2.9427 ,

where ``I`` is evaluated here from its Beta-function form
``I = (2/5) B(2/5, 1/2)`` -> ``C = 2.94273...`` (the "2.94" of the textbooks).

Two independent signatures follow, and both are tested:

* the **prefactor** ``C = 2.94`` (an absolute numeric anchor), and
* the **exponent** ``t_c ~ v0^(-1/5)`` (a shape anchor that no linear-spring
  code can fake, see the negative control below).

Rigid-plane convention
----------------------
Against a rigid platen only the particle deforms, so ``1/E* = (1 - nu^2)/E``
(**not** the two-body ``2(1 - nu^2)/E``) and ``R* = R``.  Getting this factor
wrong shifts ``t_c`` by ``2^(-2/5) = 0.758`` and ``delta_max`` by ``2^(-2/5)``
as well; the ``rigid_plane_hertz_stiffness`` helper below re-derives it from the
textbook relation instead of importing ``dem3d.hertz_reduced_modulus``, so the
comparison stays independent of the code under test.

Measurement and its bias
-------------------------
The measured contact duration is the number of integration steps for which the
particle-platen overlap is strictly positive, times ``dt``.  The true entry and
exit instants fall inside the two boundary steps with opposite fractional parts,
so ``n * dt`` is an unbiased O(dt) estimate; the residual error at
``dt = t_c/600`` is a fraction of a percent (measured: +0.17 %), and it is
*not* tuned -- every reported number comes from the shipped solver.

All quantities are SI.  The problems are single-contact and tiny: the whole
campaign fits in a few thousand steps of an 8-particle box.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Iterable, Sequence

import torch

from .dem3d import DEM3DConfig, IceDEM3D

__all__ = [
    "JOHNSON_PREFACTOR",
    "hertz_contact_time_integral",
    "johnson_prefactor",
    "rigid_plane_hertz_stiffness",
    "hertz_peak_overlap",
    "hertz_contact_time",
    "linear_contact_time",
    "linear_peak_overlap",
    "measure_platen_impact",
    "fit_loglog_exponent",
    "sweep_speeds",
]


# ---------------------------------------------------------------------------
# Closed-form anchors (re-derived here, independent of tensordem.dem3d)
# ---------------------------------------------------------------------------

def hertz_contact_time_integral() -> float:
    """``I = int_0^1 (1 - u^(5/2))^(-1/2) du = (2/5) B(2/5, 1/2)``.

    The substitution ``s = u^(5/2)`` turns the integral into the Beta function
    ``(2/5) B(2/5, 1/2)``; evaluated through ``lgamma`` so that the constant is
    not transcribed by hand.
    """
    return 0.4 * math.exp(
        math.lgamma(0.4) + math.lgamma(0.5) - math.lgamma(0.9)
    )


def johnson_prefactor() -> float:
    """``C = 2 I = 2.9427...``, the textbook 2.94 of the Hertz contact time."""
    return 2.0 * hertz_contact_time_integral()


#: Convenience constant, exact value from :func:`johnson_prefactor`.
JOHNSON_PREFACTOR = johnson_prefactor()


def rigid_plane_hertz_stiffness(
    young_modulus: float, poisson_ratio: float, radius: float
) -> float:
    """``k_h = (4/3) E* sqrt(R*)`` for a sphere of ``radius`` on a rigid plane.

    Only the sphere deforms: ``1/E* = (1 - nu^2)/E`` and ``R* = R``.
    """
    for name, value in (("young_modulus", young_modulus), ("radius", radius)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not (math.isfinite(poisson_ratio) and -1.0 < poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must be in (-1, 0.5)")
    e_star = young_modulus / (1.0 - poisson_ratio * poisson_ratio)
    return (4.0 / 3.0) * e_star * math.sqrt(radius)


def hertz_peak_overlap(v_in: float, mass: float, k_h: float) -> float:
    """``delta_max = (5 m v0^2 / (4 k_h))^(2/5)``."""
    if not (math.isfinite(v_in) and v_in > 0.0):
        raise ValueError("v_in must be finite and positive")
    if not (math.isfinite(mass) and mass > 0.0):
        raise ValueError("mass must be finite and positive")
    if not (math.isfinite(k_h) and k_h > 0.0):
        raise ValueError("k_h must be finite and positive")
    return (5.0 * mass * v_in * v_in / (4.0 * k_h)) ** 0.4


def hertz_contact_time(v_in: float, mass: float, k_h: float) -> float:
    """Johnson contact duration ``t_c = C (5 m / (4 k_h))^(2/5) v0^(-1/5)``."""
    delta_max = hertz_peak_overlap(v_in, mass, k_h)
    return 2.0 * hertz_contact_time_integral() * delta_max / v_in


def linear_contact_time(mass: float, stiffness: float) -> float:
    """Contact duration of a *linear* unilateral spring: ``pi sqrt(m / k)``.

    Independent of the impact speed -- this is the negative control: a code that
    measures ``t_c`` correctly *and* uses a linear contact law must recover this
    constant with slope zero in ``log t_c`` vs ``log v0``.
    """
    if not (math.isfinite(mass) and mass > 0.0):
        raise ValueError("mass must be finite and positive")
    if not (math.isfinite(stiffness) and stiffness > 0.0):
        raise ValueError("stiffness must be finite and positive")
    return math.pi * math.sqrt(mass / stiffness)


def linear_peak_overlap(v_in: float, mass: float, stiffness: float) -> float:
    """``delta_max = v0 sqrt(m / k)`` for a linear unilateral spring."""
    return v_in * math.sqrt(mass / stiffness)


# ---------------------------------------------------------------------------
# Measurement against the shipped solver
# ---------------------------------------------------------------------------

def _impact_base_config(
    *,
    radius: float,
    density: float,
    contact_model: str,
    young_modulus: float,
    poisson_ratio: float,
    platen_stiffness: float,
) -> DEM3DConfig:
    """Minimal single-contact rig: one driven particle, frictionless platens."""
    return DEM3DConfig(
        nx=2, ny=2, nz=2,
        radius=radius, density=density,
        # Bond stiffness 1e-4 N/m: the driven particle is bonded to its
        # neighbours by a spring four orders of magnitude softer than anything
        # in the contact, so the impact is the free-body problem the closed form
        # describes (same device as test_dem3d_hertz_contact.py).
        bond_stiffness=1.0e-4,
        contact_model=contact_model,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        contact_damping=0.0,
        drag=0.0,
        platen_stiffness=platen_stiffness,
        platen_damping=0.0,
        bottom_platen_enabled=True,
        bottom_platen_gap=0.0,
        fix_x_edges=False,
    )


def measure_platen_impact(
    *,
    v_in: float,
    dt: float,
    contact_model: str = "hertz",
    radius: float = 0.025,
    density: float = 917.0,
    young_modulus: float = 1.0e9,
    poisson_ratio: float = 0.3,
    platen_stiffness: float = 5_000.0,
    steps_per_contact: int = 600,
    max_contacts: int = 3,
    record_history: bool = False,
) -> dict[str, Any]:
    """Drive one particle into the bottom platen and time the contact.

    The lowest, innermost particle is given a downward velocity ``v_in`` and the
    simulation is advanced until the overlap first becomes positive, then
    negative again.  Nothing is fitted: ``dt`` is set by the caller (the script
    and the tests pass ``t_c_reference / steps_per_contact``) and the measured
    duration is ``n_contact_steps * dt``.

    Returns the measured duration, the peak overlap, the outgoing/incoming speed
    ratio, ``dt``, the step count, the residual platen reaction left *after*
    separation (recorded because it is a diagnostic of the reaction read-out, not
    of the contact law), and the grid / stability metadata.
    """
    if not (math.isfinite(v_in) and v_in > 0.0):
        raise ValueError("v_in must be finite and positive")
    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError("dt must be finite and positive")
    if contact_model not in ("linear", "hertz"):
        raise ValueError("contact_model must be 'linear' or 'hertz'")

    base = _impact_base_config(
        radius=radius, density=density, contact_model=contact_model,
        young_modulus=young_modulus, poisson_ratio=poisson_ratio,
        platen_stiffness=platen_stiffness,
    )
    if dt > base.recommended_dt:
        raise ValueError(
            f"dt={dt:.6g} s exceeds the solver stability bound "
            f"{base.recommended_dt:.6g} s for this configuration"
        )
    sim = IceDEM3D(replace(base, dt=dt))

    xy = torch.linalg.vector_norm(sim.positions[:, :2], dim=1)
    depth = torch.isclose(sim.positions[:, 2], sim.positions[:, 2].min())
    index = int(torch.nonzero(depth, as_tuple=False).flatten()[torch.argmin(xy[depth])])
    plane_z = float(sim.bottom_platen_start_z)
    radius_eff = base.radius
    sim.velocities[index, 2] = -v_in

    contact_steps = 0
    contacts = 0
    peak_overlap = 0.0
    v_out = 0.0
    residual_reaction = 0.0
    in_contact = False
    history: list[tuple[float, float, float]] = []
    max_steps = int(max_contacts * steps_per_contact * 4)
    for _ in range(max_steps):
        sim.step()
        overlap = plane_z - (float(sim.positions[index, 2]) - radius_eff)
        if record_history:
            history.append(
                (float(sim.time), overlap, float(sim.velocities[index, 2]))
            )
        if overlap > 0.0:
            if not in_contact:
                in_contact = True
                contacts += 1
            contact_steps += 1
            peak_overlap = max(peak_overlap, overlap)
        elif in_contact:
            in_contact = False
            v_out = float(sim.velocities[index, 2])
            if contacts >= 1:
                break
    else:  # pragma: no cover - guarded by max_steps
        raise RuntimeError("particle did not separate from the platen")

    # Reaction read-out after separation (diagnostic only; see module docstring).
    for _ in range(50):
        sim.step()
        residual_reaction = max(
            residual_reaction, abs(float(sim.last_bottom_platen_reaction_z))
        )

    payload = {
        "contact_model": contact_model,
        "v_in": v_in,
        "dt": dt,
        "contact_steps": contact_steps,
        "measured_duration_s": contact_steps * dt,
        "peak_overlap_m": peak_overlap,
        "restitution_measured": v_out / v_in,
        "post_separation_reaction_N": residual_reaction,
        "recommended_dt": base.recommended_dt,
        "particle_mass": base.mass,
    }
    if record_history:
        payload["history"] = history
    return payload


def fit_loglog_exponent(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Least-squares slope of ``log y`` against ``log x`` (two-column fit)."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need at least two matched samples")
    lx = [math.log(v) for v in xs]
    ly = [math.log(v) for v in ys]
    n = float(len(lx))
    mx = sum(lx) / n
    my = sum(ly) / n
    sxx = sum((a - mx) ** 2 for a in lx)
    if sxx == 0.0:
        raise ValueError("xs must not be constant")
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    return sxy / sxx


def sweep_speeds(
    speeds: Iterable[float],
    *,
    contact_model: str = "hertz",
    steps_per_contact: int = 600,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Measure one impact per speed and attach the matching closed form."""
    rows: list[dict[str, Any]] = []
    for v_in in speeds:
        radius = kwargs.get("radius", 0.025)
        density = kwargs.get("density", 917.0)
        young = kwargs.get("young_modulus", 1.0e9)
        nu = kwargs.get("poisson_ratio", 0.3)
        stiffness = kwargs.get("platen_stiffness", 5_000.0)
        mass = density * (4.0 / 3.0) * math.pi * radius ** 3
        if contact_model == "hertz":
            k_h = rigid_plane_hertz_stiffness(young, nu, radius)
            duration_ref = hertz_contact_time(v_in, mass, k_h)
            overlap_ref = hertz_peak_overlap(v_in, mass, k_h)
        else:
            duration_ref = linear_contact_time(mass, stiffness)
            overlap_ref = linear_peak_overlap(v_in, mass, stiffness)
        dt = duration_ref / steps_per_contact
        row = measure_platen_impact(
            v_in=v_in, dt=dt, contact_model=contact_model,
            steps_per_contact=steps_per_contact, **kwargs,
        )
        row["duration_reference_s"] = duration_ref
        row["peak_overlap_reference_m"] = overlap_ref
        row["duration_rel_error"] = row["measured_duration_s"] / duration_ref - 1.0
        row["peak_overlap_rel_error"] = row["peak_overlap_m"] / overlap_ref - 1.0
        rows.append(row)
    return rows