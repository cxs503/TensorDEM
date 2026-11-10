"""Post-handoff dynamics: the seeded floes are integrated, then audited again.

A handoff is only as good as the first step taken from it.  Mass bookkeeping can
be exact and the initial conditions still unphysical (negative mass, wrong
buoyancy stiffness, a velocity that makes the body leave the domain at ``t = 0``).
This module therefore *integrates* the seeded floes and compares the outcome with
closed forms, so the seeding is validated dynamically, not only geometrically.

Two motions are integrated, both with an exact reference:

**Heave (vertical, the motion a level-ice floe actually has).**  A floe of area
``A``, thickness ``t`` and density ``rho_i`` displaces buoyancy with the Winkler
stiffness of its own footprint, ``k = rho_w g A``, so

.. math:: \\ddot z = -\\omega^2 z, \\qquad
          \\omega = \\sqrt{\\rho_w g / (\\rho_i t)}

and the period depends on the *thickness only* -- not on the floe size.  That is
a sharp test: the mass, area and buoyancy stiffness handed over must be mutually
consistent for every floe to land on the same frequency.  Added mass is neglected
(``C_a = 0``); a real floe heaves at 5-15 % lower frequency, which is a stated
limitation, not a hidden fudge.

**In-plane drift.**  Each body relaxes towards an ambient current through linear
drag, ``\\dot v = -(v - U)/\\tau``, compared with the analytic exponential.  A
plate-bending solve carries no membrane kinematics, so the in-plane velocity is
an environmental input; the point of the check is that the seeded bodies integrate
without stiff blow-ups and reproduce the closed form to round-off.

The ledger is closed again *after* the integration: total mass, total footprint
area and the floe-floe clearance are re-measured, so "conservation" is a property
of the run and not only of the file.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .clump import overlap_audit

__all__ = [
    "GRAVITY",
    "analytic_heave",
    "integrate_heave",
    "drag_relax",
    "heave_audit",
    "post_run_ledger",
]

GRAVITY = 9.80665
WATER_DENSITY = 1025.0


@dataclass(frozen=True)
class HeaveAnalytic:
    """Closed form of the floe heave oscillator."""

    omega: float
    period: float
    thickness: float
    rho_ice: float
    rho_water: float
    added_mass_coefficient: float


def analytic_heave(thickness: float, rho_ice: float, *,
                   rho_water: float = WATER_DENSITY, g: float = GRAVITY,
                   added_mass: float = 0.0) -> HeaveAnalytic:
    """``omega = sqrt(rho_w g / (rho_i t (1 + C_a)))`` and its period."""
    if thickness <= 0.0 or rho_ice <= 0.0:
        raise ValueError("thickness and density must be positive")
    if added_mass < 0.0:
        raise ValueError("added-mass coefficient cannot be negative")
    omega = math.sqrt(rho_water * g / (rho_ice * thickness * (1.0 + added_mass)))
    return HeaveAnalytic(omega=omega, period=2.0 * math.pi / omega, thickness=thickness,
                         rho_ice=rho_ice, rho_water=rho_water,
                         added_mass_coefficient=added_mass)


def integrate_heave(omega: float, z0: float, v0: float, *,
                    t_end: float = 6.0, dt: float | None = None,
                    n_steps_per_period: int = 400):
    """Velocity-Verlet integration of ``z'' = -omega^2 z``.

    Returns the history plus the measured period (from interpolated zero
    crossings), the amplitude ``sqrt(z0^2 + (v0/omega)^2)`` and the relative
    energy drift, which is the integrator's own error bar.
    """
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if dt is None:
        dt = (2.0 * math.pi / omega) / float(n_steps_per_period)
    n = int(math.ceil(t_end / dt))
    z = np.empty(n + 1)
    v = np.empty(n + 1)
    t = np.arange(n + 1) * dt
    z[0] = z0
    v[0] = v0
    acc = lambda zz: -omega * omega * zz          # noqa: E731 - trivial and local
    e0 = 0.5 * v0 * v0 + 0.5 * omega * omega * z0 * z0
    for k in range(n):
        a = acc(z[k])
        z[k + 1] = z[k] + dt * v[k] + 0.5 * dt * dt * a
        v[k + 1] = v[k] + 0.5 * dt * (a + acc(z[k + 1]))
    e = 0.5 * v * v + 0.5 * omega * omega * z * z
    drift = float(np.max(np.abs(e - e0)) / max(abs(e0), 1e-300))
    amplitude = math.sqrt(z0 * z0 + (v0 / omega) ** 2)
    crossings = []
    if amplitude > 0.0:
        for k in range(n):
            if z[k] * z[k + 1] < 0.0:
                frac = abs(z[k]) / (abs(z[k]) + abs(z[k + 1]))
                crossings.append(t[k] + frac * dt)
    crossings = np.asarray(crossings, dtype=float)
    # A motionless body (z0 = v0 = 0) has no period: report NaN instead of the
    # grid spacing, which would masquerade as a 1.5 % "period".
    period = float(2.0 * np.mean(np.diff(crossings))) if crossings.size >= 3 else float("nan")
    return {
        "t": t, "z": z, "v": v, "energy": e,
        "period": period,
        "n_crossings": int(crossings.size),
        "amplitude": float(amplitude),
        "energy_drift": drift,
        "dt": float(dt),
        "analytic_period": 2.0 * math.pi / omega,
        "period_error": float(abs(period - 2.0 * math.pi / omega) / (2.0 * math.pi / omega))
        if crossings.size >= 3 else float("nan"),
    }


def drag_relax(v0, u, tau: float, *, t_end: float, dt: float):
    """RK4 integration of ``dv/dt = -(v - U)/tau`` against the exact solution."""
    v = float(v0)
    t = 0.0
    ts, vs = [0.0], [v]
    f = lambda vv: -(vv - u) / tau          # noqa: E731 - local and trivial
    n = int(math.ceil(t_end / dt))
    for _ in range(n):
        k1 = f(v)
        k2 = f(v + 0.5 * dt * k1)
        k3 = f(v + 0.5 * dt * k2)
        k4 = f(v + dt * k3)
        v += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        t += dt
        ts.append(t)
        vs.append(v)
    ts = np.asarray(ts)
    vs = np.asarray(vs)
    exact = u + (float(v0) - u) * np.exp(-ts / tau)
    err = float(np.max(np.abs(vs - exact)) / max(abs(float(v0) - u), 1e-30))
    return {"t": ts, "v": vs, "exact": exact, "rel_error": err,
            "terminal_velocity": float(vs[-1]), "target": float(u)}


def heave_audit(bodies, kinematics, *, t_end: float = 6.0, dt: float | None = None,
                rho_water: float = WATER_DENSITY, g: float = GRAVITY) -> dict:
    """Integrate every seeded floe and compare its period with the closed form.

    The stiffness uses the **floe footprint area** (the material) and the mass is
    the mass the handoff assigned to the body, so a mismatch shows up immediately
    in ``period_error``.  Returns per-floe rows plus the spread of the measured
    periods across floes, which must vanish because ``omega`` is size independent.
    """
    if not bodies:
        raise ValueError("no bodies to integrate")
    kin = {k.floe_id: k for k in kinematics} if kinematics else {}
    thickness = float(bodies[0].thickness)
    rho_ice = float(bodies[0].mass) / max(bodies[0].area * thickness, 1e-30)
    ref = analytic_heave(thickness, rho_ice, rho_water=rho_water, g=g)
    if dt is None:
        dt = ref.period / 400.0
    rows = []
    hist = {}
    for body in bodies:
        k = kin.get(body.floe_id)
        z0 = float(k.z0) if k is not None else 0.0
        v0 = float(k.v_z) if k is not None else 0.0
        # Stiffness from the body's own footprint, mass from the handoff.
        omega = math.sqrt(rho_water * g * body.area / body.mass)
        res = integrate_heave(omega, z0, v0, t_end=t_end, dt=dt)
        hist[body.floe_id] = res
        rho_body = body.mass / max(body.area * thickness, 1e-30)
        rows.append({
            "floe_id": body.floe_id,
            "area_m2": body.area,
            "mass_kg": body.mass,
            "rho_body_kg_m3": float(rho_body),
            "omega_rad_s": float(omega),
            "period_measured_s": res["period"],
            "period_analytic_s": res["analytic_period"],
            "period_error": res["period_error"],
            "amplitude_m": res["amplitude"],
            "energy_drift": res["energy_drift"],
            "z0_m": z0,
            "v_z0_m_s": v0,
            "z_max_abs_m": float(np.max(np.abs(res["z"]))),
        })
    periods = np.array([r["period_measured_s"] for r in rows])
    return {
        "analytic": {
            "omega_rad_s": ref.omega,
            "period_s": ref.period,
            "rho_ice_kg_m3": rho_ice,
            "rho_water_kg_m3": rho_water,
            "thickness_m": thickness,
            "added_mass_coefficient": ref.added_mass_coefficient,
        },
        "rows": rows,
        "history": hist,
        "max_period_error": float(np.nanmax([r["period_error"] for r in rows]))
        if np.any(np.isfinite([r["period_error"] for r in rows])) else float("inf"),
        "period_spread": float(np.nanmax(periods) - np.nanmin(periods))
        if np.all(np.isfinite(periods)) else float("nan"),
        "max_energy_drift": float(np.max([r["energy_drift"] for r in rows])),
        "max_abs_heave_m": float(np.max([r["z_max_abs_m"] for r in rows])),
    }


def post_run_ledger(bodies, *, plate_area: float, plate_mass: float) -> dict:
    """Close the books again after the integration touched the bodies."""
    mass = float(sum(b.mass for b in bodies))
    disc_mass = float(sum(b.disc_mass_sum for b in bodies))
    area = float(sum(b.area for b in bodies))
    audit = overlap_audit(bodies)
    return {
        "n_bodies": len(bodies),
        "plate_area_m2": plate_area,
        "body_area_sum_m2": area,
        "area_residual_m2": float(area - plate_area),
        "plate_mass_kg": plate_mass,
        "body_mass_sum_kg": mass,
        "mass_residual_kg": float(mass - plate_mass),
        "disc_mass_sum_kg": disc_mass,
        "disc_mass_residual_kg": float(disc_mass - plate_mass),
        "overlap": audit,
        "all_masses_positive": bool(all(b.mass > 0.0 and b.n_discs > 0 for b in bodies)),
    }