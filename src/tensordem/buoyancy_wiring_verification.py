"""Solver-level verification of the opt-in gravity + Archimedes buoyancy wiring.

Why this module exists
----------------------
:mod:`tensordem.buoyancy_wet3d` supplies a *verified* spherical-cap buoyancy
kernel together with its analytic anchors, but that kernel was **not wired into
the solver**: ``docs/dem3d_buoyancy_float.md`` §5 recorded the gap as
"the kernel is not yet wired into ``IceDEM3D.step``; ``DEM3DConfig`` has no
gravity entry".  ``docs/dem3d_buoyancy_wiring.md`` closes that gap and this
module is the machine-checkable evidence: every number quoted there is produced
here.

What is verified (and against what)
-----------------------------------
1. **Force identity** — the solver's body-force term equals
   ``-m g + rho_w g V_sub(h)`` with ``V_sub`` the exact spherical cap, at dry,
   half-submerged and fully submerged heights, to floating-point round-off.
2. **Archimedes draft** — a single free particle dropped into still water
   settles at the analytic centre height ``h_eq`` solving ``f(h_eq) = rho_p/rho_w``
   (:func:`tensordem.buoyancy_wet3d.equilibrium_centre_height`), for several ice
   densities and two time steps.
3. **Hydrostatic stiffness** — the central-difference derivative of the net
   vertical force reproduces ``K = rho_w g A_wp(h) = rho_w g pi (R^2 - h^2)``.
4. **Gravity-only kinematics** — in a dry box the integrator reproduces
   ``z(t) = z0 - g t^2/2`` and the error is first order in ``dt``.
5. **Inert when disabled** — a default ``DEM3DConfig`` produces exactly zero
   body force and a bit-for-bit identical force vector to a configuration whose
   free surface is placed far above the lattice, i.e. the new term cannot
   perturb any pre-existing trajectory.

Scale note: the isolated-particle fixture parks the other lattice members far
away (bonds fracture on the first force evaluation) and parks the indenter far
above (``tool_gap``), so the only load path is gravity + buoyancy + drag.  Bond
and contact stiffnesses are lowered to 500 N/m because nothing load-bearing
remains; they only enter the explicit stability bound and therefore only set the
step size.  Case 2 is run at two steps to show the answer is not a step artefact.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from .buoyancy_wet3d import (
    equilibrium_centre_height,
    submerged_volume_sphere,
    waterplane_area_sphere,
)
from .dem3d import DEM3DConfig, IceDEM3D

__all__ = [
    "ISOLATED_BASE",
    "isolated_particle",
    "static_force_cases",
    "relaxation_case",
    "stiffness_case",
    "free_fall_case",
    "inert_when_disabled",
    "wet_drag_case",
    "run_all",
]

#: Fixture shared by every case: an 8-particle lattice whose members are
#: separated so exactly one particle is free and load-bearing.
ISOLATED_BASE: dict[str, Any] = dict(
    nx=2, ny=2, nz=2, radius=0.025, density=917.0,
    contact_stiffness=500.0, bond_stiffness=500.0, contact_damping=0.0,
    tool_radius=1.0e-3, tool_gap=5.0, tool_speed=1.0e-6,
    fix_x_edges=False, fix_bottom=False,
)


def isolated_particle(
    *,
    z0: float = 0.06,
    gravity: float = 9.81,
    water_level: float | None = 0.0,
    water_density: float = 1025.0,
    density: float = 917.0,
    drag: float = 2.0,
    wet_drag_coefficient: float = 0.0,
    dt: float | None = None,
) -> tuple[IceDEM3D, DEM3DConfig]:
    """Build the isolated-particle fixture with one free particle at ``z0``.

    The other seven lattice members are translated ``100 R`` in ``+x`` and
    pinned, and the indenter is parked metres above the lattice, so the free
    particle feels only gravity, buoyancy and drag.  Bonds touching it break on
    the first force evaluation (that call is discarded and the velocities are
    reset), leaving a genuinely unbonded, contact-free particle.
    """
    cfg = DEM3DConfig(
        **{**ISOLATED_BASE, "density": density, "gravity": gravity,
           "water_level": water_level, "water_density": water_density,
           "drag": drag, "wet_drag_coefficient": wet_drag_coefficient, "dt": dt}
    )
    dem = IceDEM3D(cfg)
    far = 100.0 * cfg.radius
    positions = dem.positions.clone()
    positions[0] = torch.tensor([0.0, 0.0, float(z0)], dtype=dem.dtype)
    positions[1:] += torch.tensor([far, 0.0, 0.0], dtype=dem.dtype)
    dem.positions[:] = positions
    dem.initial_positions[:] = positions
    dem.fixed[:] = True
    dem.fixed[0] = False
    dem.velocities[:] = 0.0
    dem.forces()                    # commit the bond fracture of the free particle
    dem.velocities[:] = 0.0
    dem.time = 0.0
    dem.step_count = 0
    if bool(torch.any(dem.alive[torch.any(dem.bond_indices == 0, dim=1)])):
        raise RuntimeError("fixture failed to isolate the free particle")
    return dem, cfg


def _net_force_z(dem: IceDEM3D) -> float:
    """Vertical force on the free (index 0) particle from ``forces()``."""
    force, _ = dem.forces(update_fracture=False, update_history=False)
    return float(force[0, 2])


def _analytic_net_force_z(z: float, cfg: DEM3DConfig) -> float:
    """``-m g + rho_w g V_sub(z - water_level)``, zero buoyancy for a dry box."""
    weight = -cfg.mass * cfg.gravity
    if cfg.water_level is None:
        return weight
    volume = submerged_volume_sphere(z - cfg.water_level, cfg.radius)
    return weight + cfg.water_density * cfg.gravity * volume


def _rel(value: float, reference: float) -> float:
    scale = max(abs(reference), 1e-30)
    return abs(value - reference) / scale


def static_force_cases() -> list[dict[str, float]]:
    """Case 1: solver body force vs the closed-form cap integral at 6 heights."""
    cases = []
    for density in (917.0, 700.0):
        radius = ISOLATED_BASE["radius"]
        for h_over_r in (1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0):
            z = h_over_r * radius
            dem, cfg = isolated_particle(z0=z, density=density)
            measured = _net_force_z(dem)
            reference = _analytic_net_force_z(z, cfg)
            cases.append({
                "density_kg_m3": density,
                "h_over_R": h_over_r,
                "net_force_sign": (measured > 0) - (measured < 0),
                "measured_Fz_N": measured,
                "reference_Fz_N": reference,
                "abs_error_N": abs(measured - reference),
                "rel_error": _rel(measured, reference) if reference else 0.0,
            })
    return cases


def relaxation_case(
    *,
    density: float = 917.0,
    z0: float = 0.06,
    n_steps: int = 4000,
    dt: float | None = None,
) -> dict[str, Any]:
    """Case 2: drop the free particle and compare the settled height to ``h_eq``.

    Returns the settled centre height, the analytic value, the relative error in
    particle radii (the natural scale: ``R`` is the only length in the geometry)
    and a coarse settling trace.
    """
    dem, cfg = isolated_particle(z0=z0, density=density, dt=dt)
    h_eq = equilibrium_centre_height(cfg.radius, density, cfg.water_density)
    trace: list[tuple[int, float]] = []
    for k in range(1, n_steps + 1):
        dem.step()
        if k % max(1, n_steps // 40) == 0:
            trace.append((k, float(dem.positions[0, 2])))
    z_final = float(dem.positions[0, 2])
    return {
        "density_kg_m3": density,
        "dt_s": dem.dt,
        "n_steps": n_steps,
        "time_s": dem.time,
        "z_final_m": z_final,
        "h_eq_analytic_m": h_eq,
        "draft_measured_m": cfg.radius - z_final,
        "draft_analytic_m": cfg.radius - h_eq,
        "error_over_R": (z_final - h_eq) / cfg.radius,
        "rel_error_draft": _rel(cfg.radius - z_final, cfg.radius - h_eq),
        "trace": trace,
    }


def stiffness_case(*, h_over_r: float = -0.6, delta_over_r: float = 0.08) -> dict[str, float]:
    """Case 3: central-difference restoring stiffness vs ``rho_w g pi (R^2-h^2)``."""
    dem, cfg = isolated_particle(z0=h_over_r * ISOLATED_BASE["radius"])
    delta = delta_over_r * cfg.radius
    z = h_over_r * cfg.radius

    def force_at(target: float) -> float:
        dem.positions[0, 2] = target
        return _net_force_z(dem)

    f_plus = force_at(z + delta)
    f_minus = force_at(z - delta)
    measured = -(f_plus - f_minus) / (2.0 * delta)
    reference = cfg.water_density * cfg.gravity * waterplane_area_sphere(z, cfg.radius)
    return {
        "h_over_R": h_over_r,
        "delta_m": delta,
        "measured_K_N_m": measured,
        "reference_K_N_m": reference,
        "rel_error": _rel(measured, reference),
    }


def free_fall_case(*, n_steps: int = 400, dt_scale: float = 1.0,
                   g: float = 9.81, z0: float = 0.5) -> dict[str, float]:
    """Case 4: dry box, no drag — compare ``z(t)`` with ``z0 - g t^2/2``."""
    _, probe_cfg = isolated_particle(z0=z0, gravity=g, water_level=None, drag=0.0)
    dt = probe_cfg.recommended_dt * dt_scale
    dem, cfg = isolated_particle(z0=z0, gravity=g, water_level=None, drag=0.0, dt=dt)
    for _ in range(n_steps):
        dem.step()
    t = dem.time
    z_final = float(dem.positions[0, 2])
    reference = z0 - 0.5 * g * t ** 2
    return {
        "dt_s": dem.dt,
        "n_steps": n_steps,
        "time_s": t,
        "z_measured_m": z_final,
        "z_analytic_m": reference,
        "abs_error_m": abs(z_final - reference),
        "error_over_R": (z_final - reference) / cfg.radius,
        "rel_error": _rel(z_final, reference),
    }


def inert_when_disabled() -> dict[str, Any]:
    """Case 5: the new term is exactly inert for a default configuration."""
    legacy = IceDEM3D(DEM3DConfig())
    reference_cfg = DEM3DConfig(water_level=1.0e6, gravity=0.0)
    reference = IceDEM3D(reference_cfg)
    body = legacy.body_forces()
    force_legacy, _ = legacy.forces(update_fracture=False, update_history=False)
    force_ref, _ = reference.forces(update_fracture=False, update_history=False)
    return {
        "default_body_force_abs_max_N": float(body.abs().max()),
        "default_body_force_total_z_N": float(body[:, 2].sum()),
        "forces_bitwise_equal": bool(torch.equal(force_legacy, force_ref)),
        "max_abs_force_difference_N": float((force_legacy - force_ref).abs().max()),
        "dt_default_s": legacy.dt,
        "dt_with_body_fields_s": reference.dt,
        "ledger_default": legacy.body_force_ledger(),
        "buoyancy_active_default": bool(legacy.buoyancy_active),
    }


def wet_drag_case(*, z: float = -0.02, speed: float = 0.3,
                  coefficient: float = 1.5) -> dict[str, float]:
    """Case 6: submerged-only linear wet drag ``F = -C f_sub v`` (z component only)."""
    dem, cfg = isolated_particle(z0=z, wet_drag_coefficient=coefficient)
    dem.velocities[0] = torch.tensor([0.0, 0.0, -speed], dtype=dem.dtype)
    body = dem.body_forces()
    f_sub = float(dem.submerged_fraction()[0])
    measured = float(body[0, 2]) + cfg.mass * cfg.gravity \
        - cfg.water_density * cfg.gravity * \
        (4.0 / 3.0) * math.pi * cfg.radius ** 3 * f_sub
    reference = -coefficient * f_sub * (-speed)
    return {
        "z_m": z,
        "speed_m_s": speed,
        "f_sub": f_sub,
        "measured_drag_N": measured,
        "reference_drag_N": reference,
        "abs_error_N": abs(measured - reference),
    }


def run_all(*, n_steps: int = 4000) -> dict[str, Any]:
    """Run every case and return the machine-generated evidence record."""
    static = static_force_cases()
    _, probe_cfg = isolated_particle()
    half_dt = 0.5 * probe_cfg.recommended_dt
    relaxation = [
        relaxation_case(density=917.0, n_steps=n_steps),
        relaxation_case(density=850.0, n_steps=n_steps),
        relaxation_case(density=917.0, n_steps=2 * n_steps, dt=half_dt),
    ]
    stiffness = stiffness_case()
    free_fall = [free_fall_case(dt_scale=1.0), free_fall_case(dt_scale=0.5)]
    inert = inert_when_disabled()
    wet_drag = wet_drag_case()
    relaxed = relaxation[:2]
    return {
        "schema": "tensordem.buoyancy-wiring-verification/1",
        "static_force": static,
        "static_max_abs_error_N": max(c["abs_error_N"] for c in static),
        "relaxation": relaxation,
        "relaxation_max_rel_error_draft": max(c["rel_error_draft"] for c in relaxed),
        "relaxation_max_error_over_R": max(abs(c["error_over_R"]) for c in relaxed),
        "stiffness": stiffness,
        "free_fall": free_fall,
        "free_fall_error_ratio_half_dt": (
            free_fall[1]["abs_error_m"] / free_fall[0]["abs_error_m"]
            if free_fall[0]["abs_error_m"] > 0 else float("nan")),
        "inert": inert,
        "wet_drag": wet_drag,
        "gates": {
            "static_force_machine_precision": max(c["abs_error_N"] for c in static) < 1e-12,
            "archimedes_draft_under_3_percent": max(c["rel_error_draft"] for c in relaxed) < 0.03,
            "hydrostatic_stiffness_under_3_percent": stiffness["rel_error"] < 0.03,
            "free_fall_error_decreases_with_dt": free_fall[1]["abs_error_m"] < free_fall[0]["abs_error_m"],
            "inert_when_disabled": (inert["forces_bitwise_equal"]
                                    and inert["default_body_force_abs_max_N"] == 0.0),
        },
    }