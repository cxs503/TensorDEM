"""Orchestration of the dynamic FEM -> DEM fracture handoff (wave 2, p302w2-n03).

The static chain lives in :mod:`tensordem.fem_dem_fracture.study` (wave 1).  This
module adds the dynamic half and the discrete-side intake, and writes one
auditable bundle:

``failure_faces.json``              static census (wave 1, schema 1)
``failure_faces_dynamic.json``      schema 2: floe kinematics + cell velocity field
``dem_initial_conditions.json``     static bodies (wave 1)
``dem_initial_conditions_dynamic.json``  rigid bodies with velocity and tilt
``dem_intake.json``                 clump seeding + heave integration (DEM side)
``benchmark.json``                  reference vs computed table
``summary_dynamic.json``            criteria D1..D7 + numbers
``manifest.json``                   artefact SHA-256, environment, reproduction command
``fig_*.png``                       field / energy / heave figures

The DEM intake is a *real* call into the repository's discrete side
(:mod:`tensordem.fem_dem_seeding`): the artefact is re-read, the floe polygons are
re-sampled into DEM bodies and every body is integrated in heave against the
closed-form period ``omega = sqrt(rho_w g / (rho_i t))`` while carrying the
velocity inherited from the FEM.  What it is not is a contact-resolved DEM run of
the whole ice sheet: the repository's production lattice solvers
(:class:`tensordem.dem3d.IceDEM3D`) take their geometry from a fixed lattice, so
arbitrary floe insertion is out of scope here and is stated as a limitation
rather than faked.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

from .dynamics import (
    SCHEMA_DEM_IC,
    SCHEMA_DYNAMIC,
    build_dynamic_failure_faces,
    cell_centroid_unit_rates,
    dem_initial_conditions_dynamic,
    kinetic_energy,
    linearity_check,
    momentum_ledger,
    project_rigid_body_motion,
    quasi_static_ledger,
    read_dynamic_failure_faces,
    velocity_field,
    write_dynamic_failure_faces,
)
from .handoff import read_failure_faces
from .study import (
    IcePlateCase,
    fem_benchmark_table,
    run_handoff,
    solve_case_field,
)

__all__ = [
    "run_dynamic_handoff",
    "dem_intake",
    "dynamic_criteria",
    "write_figures",
    "main",
]

DEFAULT_PRESSURE_RATE_PA_S = 1000.0


# --------------------------------------------------------------------------
# DEM side: seed bodies from the artefact and integrate them
# --------------------------------------------------------------------------
def dem_intake(dynamic, *, heave_periods: float = 4.0) -> dict:
    """Insert the schema-2 bodies into the repository's DEM seeding path.

    Steps (all real calls, no re-implementation):

    1. re-read ``failure_faces_dynamic.json`` -> floes + kinematics;
    2. build a DEM raster on the plate and seed one equivalent-area body per
       floe with :func:`tensordem.fem_dem_seeding.clump.seed_floe_bodies`;
    3. close the mass/area ledger of the seeded bodies
       (:func:`...clump.body_ledger`) and audit the disc overlaps;
    4. integrate each body in heave from the inherited vertical velocity and
       compare the measured period with the closed form
       (:func:`...dynamics.heave_audit`);
    5. re-close the ledger after the integration
       (:func:`...dynamics.post_run_ledger`).
    """
    from ..fem_dem_seeding.clump import body_ledger, overlap_audit, seed_floe_bodies
    from ..fem_dem_seeding.dynamics import analytic_heave, heave_audit, post_run_ledger
    from ..fem_dem_seeding.raster import Raster

    faces = dynamic.faces
    meta = faces.meta
    length_x = float(meta["plate"]["length_x_m"])
    length_y = float(meta["plate"]["length_y_m"])
    thickness = float(meta["plate"]["thickness_m"])
    density = float(meta["material"]["density_kg_m3"])
    plate_area = length_x * length_y
    plate_mass = plate_area * thickness * density

    h = float(np.sqrt(meta["mesh"]["cell_area_m2"]))
    raster = Raster(0.0, 0.0, length_x, length_y, h)
    seeded = seed_floe_bodies(faces, raster, modes=("equivalent_disc",))
    bodies = seeded["equivalent_disc"]

    reference = analytic_heave(thickness, density, added_mass=0.0)
    t_end = heave_periods * reference.period
    dt = reference.period / 400.0
    audit = heave_audit(bodies, dynamic.kinematics, t_end=t_end, dt=dt)
    ledger = body_ledger(bodies, plate_area=plate_area, plate_mass=plate_mass)
    final = post_run_ledger(bodies, plate_area=plate_area, plate_mass=plate_mass)
    v_inherited = np.array([k.v_z for k in dynamic.kinematics], dtype=float)
    return {
        "raster": {"x0": raster.x0, "y0": raster.y0, "x1": raster.x1, "y1": raster.y1,
                   "h": raster.h, "nx": raster.nx, "ny": raster.ny},
        "n_bodies": len(bodies),
        "seeded_ledger": ledger,
        "post_run_ledger": final,
        "overlap_audit": overlap_audit(bodies),
        "heave": {
            "t_end_s": t_end,
            "dt_s": dt,
            "analytic": audit["analytic"],
            "max_period_error": audit["max_period_error"],
            "period_spread_s": audit["period_spread"],
            "max_energy_drift": audit["max_energy_drift"],
            "max_abs_heave_m": audit["max_abs_heave_m"],
            "rows": audit["rows"],
        },
        "inherited_vertical_velocity_m_s": {
            "min": float(np.min(v_inherited)),
            "max": float(np.max(v_inherited)),
            "mean": float(np.mean(v_inherited)),
        },
        "velocity_field_momentum_kg_m_s": float(
            np.sum([b.mass * v for b, v in zip(bodies, v_inherited)])),
    }


# --------------------------------------------------------------------------
# criteria
# --------------------------------------------------------------------------
def dynamic_criteria(dynamic, intake: dict, *, linearity: dict,
                     quasi_static: dict, static_summary: dict) -> dict:
    """Machine-checkable dynamic criteria (D1..D7) with the numbers behind them."""
    led = dynamic.ledger
    p_scale = max(abs(led.momentum_field), 1e-300)
    l_scale = max(float(np.max(np.abs(led.angular_field))), 1e-300)
    ke_scale = max(abs(led.ke_field), 1e-300)
    ke_self = max(sum(abs(k.fit.ke_total) for k in dynamic.kinematics), 1e-300)
    fractions = [k.rigid_fraction for k in dynamic.kinematics]
    mass_rel = abs(led.mass_residual) / max(abs(led.mass_field), 1e-300)
    heave = intake["heave"]
    seeded = intake["seeded_ledger"]
    body_mass_rel = abs(seeded["mass_residual_kg"]) / max(
        abs(seeded["plate_mass_kg"]), 1e-300)
    return {
        "D1_velocity_field_nonzero": {
            "cells": int(dynamic.field.n_cells),
            "max_abs_rate_m_s": dynamic.field.max_abs_rate,
            "pass": bool(dynamic.field.max_abs_rate > 0.0),
        },
        "D1b_loading_path_linearity": {
            **linearity,
            "tolerance": 1e-9,
            "pass": bool(linearity["nodal_rel_residual"] < 1e-9
                         and linearity["cell_rel_residual"] < 1e-9),
        },
        "D2_linear_momentum": {
            "field_kg_m_s": led.momentum_field,
            "bodies_kg_m_s": led.momentum_bodies,
            "residual_kg_m_s": led.momentum_residual,
            "relative_residual": abs(led.momentum_residual) / p_scale,
            "tolerance": 1e-10,
            "pass": bool(abs(led.momentum_residual) / p_scale < 1e-10),
        },
        "D3_angular_momentum": {
            "field_kg_m2_s": [float(v) for v in led.angular_field],
            "bodies_kg_m2_s": [float(v) for v in led.angular_bodies],
            "residual_kg_m2_s": [float(v) for v in led.angular_residual],
            "relative_residual": float(np.max(np.abs(led.angular_residual)) / l_scale),
            "tolerance": 1e-10,
            "pass": bool(float(np.max(np.abs(led.angular_residual)) / l_scale) < 1e-10),
        },
        "D4_kinetic_energy_split": {
            "ke_field_j": led.ke_field,
            "ke_rigid_j": led.ke_rigid,
            "ke_deform_j": led.ke_deform,
            "sum_residual_j": led.ke_sum_residual,
            "relative_residual": abs(led.ke_sum_residual) / ke_scale,
            "rigid_fraction_plate": led.rigid_fraction,
            "tolerance": 1e-10,
            "pass": bool(abs(led.ke_sum_residual) / ke_scale < 1e-10),
        },
        "D5_rigid_fraction_bounds": {
            "min": float(min(fractions)),
            "max": float(max(fractions)),
            "n_floes_with_fraction_in_unit_interval": int(
                sum(1 for f in fractions if -1e-12 <= f <= 1.0 + 1e-12)),
            "n_floes": len(fractions),
            "pass": bool(all(-1e-12 <= f <= 1.0 + 1e-12 for f in fractions)),
        },
        "D6_mass_conservation_after_transfer": {
            "field_kg": led.mass_field,
            "bodies_kg": led.mass_bodies,
            "residual_kg": led.mass_residual,
            "relative_residual": mass_rel,
            "tolerance_kg": 1e-6,
            "pass": bool(abs(led.mass_residual) < 1e-6),
        },
        "D7_dem_intake": {
            "n_bodies": intake["n_bodies"],
            "body_mass_residual_kg": seeded["mass_residual_kg"],
            "body_mass_relative_residual": body_mass_rel,
            "body_area_residual_m2": seeded["area_residual_m2"],
            "heave_max_period_error": heave["max_period_error"],
            "heave_period_target": 1e-3,
            "heave_energy_drift": heave["max_energy_drift"],
            "pass": bool(body_mass_rel < 1e-12
                         and abs(seeded["area_residual_m2"]) < 1e-9
                         and heave["max_period_error"] < 1e-3),
        },
        "D8_quasi_static_validity": {
            **quasi_static,
            "pass": bool(quasi_static["quasi_static"]),
        },
        "C_static_chain": {
            "C1_fracture_declared": static_summary["criteria"]["C1_fracture_declared"],
            "C2_mass_residual_kg": static_summary["criteria"]["C2_mass_residual_kg"],
            "C2_pass": static_summary["criteria"]["C2_pass"],
            "C3_area_residual_m2": static_summary["criteria"]["C3_area_residual_m2"],
            "C3_pass": static_summary["criteria"]["C3_pass"],
            "C4_n_floe": static_summary["criteria"]["C4_n_floe"],
            "C4_pass": static_summary["criteria"]["C4_pass"],
            "C5_pass": static_summary["criteria"]["C5_pass"],
        },
    }


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def write_figures(dynamic, intake: dict, out_dir: str) -> list[str]:
    """Three publication figures; returns the written paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    faces = dynamic.faces
    mesh = dynamic.field.mesh
    meta = faces.meta
    length_x = float(meta["plate"]["length_x_m"])
    length_y = float(meta["plate"]["length_y_m"])

    # (1) inherited velocity field + floe outlines
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=150)
    grid = dynamic.field.rates.reshape(mesh.ny, mesh.nx)
    im = ax.imshow(grid * 1e3, origin="lower", extent=(0, length_x, 0, length_y),
                   cmap="viridis")
    fig.colorbar(im, ax=ax, label="inherited $v_z$ [mm/s]")
    for floe in faces.floes:
        ring = np.asarray(floe.polygons[0], dtype=float)
        ax.plot(np.append(ring[:, 0], ring[0, 0]), np.append(ring[:, 1], ring[0, 1]),
                color="w", lw=0.9)
        ax.text(floe.centroid[0], floe.centroid[1], f"{floe.floe_id}",
                color="w", fontsize=7, ha="center", va="center")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Inherited deflection rate at $\\dot q$ = "
                 f"{dynamic.field.pressure_rate_pa_s:g} Pa/s\n"
                 f"(floe ids, {len(faces.floes)} floes, {mesh.nx}x{mesh.ny} FEM cells)")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_velocity_field.png")
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    # (2) energy split per floe
    kin = sorted(dynamic.kinematics, key=lambda k: k.equivalent_diameter, reverse=True)
    ids = [k.floe_id for k in kin]
    rigid = np.array([k.fit.ke_rigid for k in kin])
    deform = np.array([k.fit.ke_deform for k in kin])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 7.2), dpi=150, sharex=True)
    x = np.arange(len(ids))
    ax1.bar(x, rigid, label="rigid (transl.+tilt)", color="#1f77b4")
    ax1.bar(x, deform, bottom=rigid, label="deformational (lost to rigidity)",
            color="#ff7f0e")
    ax1.set_yscale("log")
    ax1.set_ylabel("kinetic energy [J]")
    ax1.legend(fontsize=8)
    ax1.set_title("Rigid-body transfer per floe: energy split")
    ax2.plot(x, [k.rigid_fraction for k in kin], "o-", color="#2ca02c")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("rigid fraction $KE_{rigid}/KE_{total}$")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{i}\n{d:.1f} m" for i, d in
                         zip(ids, [k.equivalent_diameter for k in kin])], fontsize=7)
    ax2.set_xlabel("floe id / equivalent diameter")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_rigid_fraction.png")
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    # (3) heave trace of the largest floe carrying the inherited velocity
    rows = sorted(intake["heave"]["rows"], key=lambda r: r["area_m2"], reverse=True)
    if rows:
        from ..fem_dem_seeding.dynamics import analytic_heave, integrate_heave

        top = rows[0]
        period = analytic_heave(float(dynamic.faces.meta["plate"]["thickness_m"]),
                                float(top["rho_body_kg_m3"])).period
        omega = top["omega_rad_s"]
        trace = integrate_heave(omega, float(top["z0_m"]), float(top["v_z0_m_s"]),
                                t_end=3.0 * period, dt=period / 400.0)
        t = np.arange(trace["z"].size) * (period / 400.0)
        fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=150)
        ax.plot(t, np.asarray(trace["z"]) * 1e3, label="floe heave (DEM-side ODE)")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("heave [mm]")
        ax.set_title(f"Floe {top['floe_id']} heave from the inherited $v_z$ = "
                     f"{top['v_z0_m_s']:.3e} m/s; period measured "
                     f"{top['period_measured_s']:.4f} s vs analytic "
                     f"{top['period_analytic_s']:.4f} s")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, "fig_heave.png")
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


# --------------------------------------------------------------------------
# benchmark table (reference vs computed)
# --------------------------------------------------------------------------
def benchmark_table(case: IcePlateCase, dynamic, intake: dict,
                    *, linearity: dict, quasi_static: dict) -> dict:
    """Reference vs computed rows for the dynamic handoff (L4 evidence)."""
    from .dynamics import project_point_masses

    rows: list[dict] = []
    fem = fem_benchmark_table()
    for row in fem["rows"]:
        rows.append({
            "benchmark": row["benchmark"],
            "quantity": row["quantity"],
            "reference": float(row["reference"]),
            "got": float(row["value"]),
            "rel_error": float(row["rel_error"]),
            "tolerance": 0.03,
            "source": "closed form (Navier series / beam limit / Hetenyi)",
        })

    # synthetic exact references for the rigid projection
    n = 12
    length = 1.0
    coords = (np.arange(n) + 0.5) * (length / n)
    XX, YY = np.meshgrid(coords, coords, indexing="xy")
    x = XX.ravel() - length / 2
    y = YY.ravel() - length / 2
    m = np.ones(x.size)
    # (i) a rigid field: translation + tilt -> the fit must recover it exactly
    v, wx, wy = 1.0e-3, 2.0e-4, -3.0e-4
    u_rigid = v + wx * y - wy * x
    fit = project_point_masses(x, y, u_rigid, m)
    rows.append({"benchmark": "rigid projection, exactly rigid field",
                 "quantity": "KE_rigid / KE_total", "reference": 1.0,
                 "got": fit.rigid_fraction, "rel_error": abs(fit.rigid_fraction - 1.0),
                 "tolerance": 1e-12,
                 "source": "analytic: a rigid field is reproduced by the rigid fit"})
    rows.append({"benchmark": "rigid projection, exactly rigid field",
                 "quantity": "max |v - v_rigid| [m/s]", "reference": 0.0,
                 "got": fit.max_abs_deform, "rel_error": fit.max_abs_deform,
                 "tolerance": 1e-12, "source": "analytic: residual is identically zero"})
    rows.append({"benchmark": "rigid projection, exactly rigid field",
                 "quantity": "omega_x recovery [rad/s]", "reference": wx,
                 "got": fit.omega_x,
                 "rel_error": abs(fit.omega_x - wx) / abs(wx), "tolerance": 1e-10,
                 "source": "analytic"})
    # (ii) pure isotropic bending w = kappa/2 (x^2 + y^2) on a symmetric square:
    # the tilt moments vanish by symmetry, so the rigid fit must collapse to the
    # mean (pure translation).  Two exact consequences are checked:
    #   * the tilt coefficients are zero to round-off (ref 0);
    #   * KE_rigid/KE_total = mean(u)^2 / mean(u^2) (ref: closed form of the
    #     translation-only fit -- note the rigid KE keeps the mean motion, so this
    #     ratio is NOT the variance-explained R^2 of the fit, it is an energy ratio).
    kappa = 1.0e-3
    u_bend = 0.5 * kappa * (x ** 2 + y ** 2)
    fit_b = project_point_masses(x, y, u_bend, m)
    ref_fraction = float(np.mean(u_bend) ** 2 / np.mean(u_bend ** 2))
    rows.append({"benchmark": "rigid projection, isotropic bending field",
                 "quantity": "KE_rigid / KE_total", "reference": ref_fraction,
                 "got": fit_b.rigid_fraction,
                 "rel_error": abs(fit_b.rigid_fraction - ref_fraction) / ref_fraction,
                 "tolerance": 1e-12,
                 "source": ("analytic: symmetry kills the tilt, so the rigid fit is the "
                            "mean and the fraction is mean(u)^2/mean(u^2)")})
    rows.append({"benchmark": "rigid projection, isotropic bending field",
                 "quantity": "max |tilt rate| [rad/s]", "reference": 0.0,
                 "got": float(max(abs(fit_b.omega_x), abs(fit_b.omega_y))),
                 "rel_error": float(max(abs(fit_b.omega_x), abs(fit_b.omega_y))),
                 "tolerance": 1e-12,
                 "source": "analytic: the first moments of (x^2+y^2) vanish on a "
                           "symmetric square"})

    led = dynamic.ledger
    rows.append({"benchmark": "momentum transfer", "quantity": "residual [kg m/s]",
                 "reference": 0.0, "got": abs(led.momentum_residual),
                 "rel_error": abs(led.momentum_residual),
                 "tolerance": 1e-12 * max(1.0, abs(led.momentum_field)),
                 "source": "exact identity of the mass-weighted fit"})
    rows.append({"benchmark": "angular momentum transfer",
                 "quantity": "max residual [kg m^2/s]", "reference": 0.0,
                 "got": float(np.max(np.abs(led.angular_residual))),
                 "rel_error": float(np.max(np.abs(led.angular_residual))),
                 "tolerance": 1e-12 * max(1.0, float(np.max(np.abs(led.angular_field)))),
                 "source": "exact identity (fit in principal axes)"})
    rows.append({"benchmark": "kinetic energy split",
                 "quantity": "KE_rigid + KE_deform - KE_field [J]", "reference": 0.0,
                 "got": abs(led.ke_sum_residual), "rel_error": abs(led.ke_sum_residual),
                 "tolerance": 1e-12 * max(1.0, led.ke_field),
                 "source": "exact identity (cross terms vanish)"})

    heave = intake["heave"]
    rows.append({"benchmark": "floe heave (DEM-side integration)",
                 "quantity": "period [s]",
                 "reference": heave["analytic"]["period_s"],
                 "got": float(np.mean([r["period_measured_s"] for r in heave["rows"]])),
                 "rel_error": heave["max_period_error"], "tolerance": 1e-3,
                 "source": "omega = sqrt(rho_w g / (rho_i t)), no added mass"})

    rows.append({"benchmark": "loading path linearity", "quantity":
                 "relative finite-difference residual", "reference": 0.0,
                 "got": linearity["nodal_rel_residual"],
                 "rel_error": linearity["nodal_rel_residual"], "tolerance": 1e-9,
                 "source": "K w = q b_hat is linear in q"})
    if np.isfinite(quasi_static.get("critical_rate_pa_s", float("inf"))):
        rows.append({"benchmark": "quasi-static validity",
                     "quantity": "KE/U at the run rate", "reference": 0.1,
                     "got": quasi_static["ke_over_elastic"],
                     "rel_error": max(0.0, quasi_static["ke_over_elastic"] - 0.1) / 0.1,
                     "tolerance": 1e9,
                     "source": ("reporting row: the run is quasi-static while "
                                "KE/U <= 0.1 (critical rate "
                                f"{quasi_static['critical_rate_pa_s']:.3g} Pa/s)")})

    failing = [r for r in rows
               if r["rel_error"] > r["tolerance"] and r["benchmark"] != "quasi-static validity"]
    return {
        "rows": rows,
        "n_rows": len(rows),
        "n_failing": len(failing),
        "max_rel_error_analytic_closed_forms": max(
            r["rel_error"] for r in rows
            if r["benchmark"].split(" ")[0] in ("rigid", "momentum", "angular",
                                                "kinetic", "floe", "loading", "strip",
                                                "SSSS", "clamped", "Winkler")),
        "all_within_tolerance": len(failing) == 0,
    }


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
def _sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def run_dynamic_handoff(case: IcePlateCase, out_dir: str, *,
                        pressure_rate_pa_s: float = DEFAULT_PRESSURE_RATE_PA_S,
                        tag: str = "dynamic",
                        make_figures: bool = True,
                        heave_periods: float = 4.0) -> dict:
    """Run the static chain, then the dynamic handoff, then the DEM intake."""
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    static = run_handoff(case, out_dir, tag=f"{tag}-static")
    faces = read_failure_faces(os.path.join(out_dir, "failure_faces.json"))

    unit = solve_case_field(case)
    fracture_pressure = float(static["criteria"]["splitting_pressure_pa"])
    field = velocity_field(unit, pressure_rate_pa_s=pressure_rate_pa_s,
                           fracture_pressure_pa=fracture_pressure)
    kinematics = project_rigid_body_motion(faces, field)
    ledger = momentum_ledger(faces, field, kinematics)
    quasi = quasi_static_ledger(unit, list(case.patches), field,
                                density=case.material.density,
                                thickness=case.material.thickness)
    linearity = linearity_check(unit, list(case.patches))

    payload = build_dynamic_failure_faces(
        faces, field, kinematics, ledger, quasi_static=quasi,
        extra_meta={"static_summary_tag": static["tag"]})
    dynamic_path = os.path.join(out_dir, "failure_faces_dynamic.json")
    digest = write_dynamic_failure_faces(payload, dynamic_path)

    # read the artefact back: everything below is computed from the file alone
    dynamic = read_dynamic_failure_faces(dynamic_path)
    ic = dem_initial_conditions_dynamic(dynamic)
    ic_path = os.path.join(out_dir, "dem_initial_conditions_dynamic.json")
    with open(ic_path, "w", encoding="utf-8") as handle:
        json.dump(ic.as_payload(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    intake = dem_intake(dynamic, heave_periods=heave_periods)
    intake_path = os.path.join(out_dir, "dem_intake.json")
    with open(intake_path, "w", encoding="utf-8") as handle:
        json.dump(intake, handle, indent=2, sort_keys=True)
        handle.write("\n")

    criteria = dynamic_criteria(dynamic, intake, linearity=linearity,
                                quasi_static=quasi, static_summary=static)
    benchmark = benchmark_table(case, dynamic, intake,
                               linearity=linearity, quasi_static=quasi)
    bench_path = os.path.join(out_dir, "benchmark.json")
    with open(bench_path, "w", encoding="utf-8") as handle:
        json.dump(benchmark, handle, indent=2, sort_keys=True)
        handle.write("\n")

    figures = write_figures(dynamic, intake, out_dir) if make_figures else []

    summary = {
        "tag": tag,
        "case": static["case"],
        "pressure_rate_pa_s": pressure_rate_pa_s,
        "fracture_pressure_pa": fracture_pressure,
        "criteria": criteria,
        "ledger": ledger.as_dict(),
        "quasi_static": quasi,
        "linearity": linearity,
        "per_floe": [k.as_dict() for k in dynamic.kinematics],
        "benchmark_max_rel_error": benchmark["max_rel_error_analytic_closed_forms"],
        "benchmark_all_within_tolerance": benchmark["all_within_tolerance"],
        "artifacts": {
            "failure_faces": os.path.join(out_dir, "failure_faces.json"),
            "failure_faces_dynamic": dynamic_path,
            "dem_initial_conditions_static": os.path.join(
                out_dir, "dem_initial_conditions.json"),
            "dem_initial_conditions_dynamic": ic_path,
            "dem_intake": intake_path,
            "benchmark": bench_path,
            "figures": figures,
        },
        "schema_dynamic": SCHEMA_DYNAMIC,
        "schema_dem_ic": SCHEMA_DEM_IC,
        "payload_sha256": digest,
        "wall_time_s": round(time.time() - t0, 2),
    }
    summary_path = os.path.join(out_dir, "summary_dynamic.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    artifacts = {name: path for name, path in summary["artifacts"].items()
                 if isinstance(path, str) and os.path.exists(path)}
    manifest = {
        "mission": "p302w2-n03-fem-dem-fracture-handoff",
        "wave": 2,
        "tag": tag,
        "schema": {"failure_faces_static": "tensordem.failure_faces/1",
                   "failure_faces_dynamic": SCHEMA_DYNAMIC,
                   "dem_initial_conditions_dynamic": SCHEMA_DEM_IC},
        "impl_files": [
            "src/tensordem/fem_dem_fracture/dynamics.py",
            "src/tensordem/fem_dem_fracture/study_dynamic.py",
            "src/tensordem/fem_dem_fracture/__main__.py",
            "src/tensordem/fem_dem_fracture/handoff.py",
            "src/tensordem/fem_dem_fracture/__init__.py",
        ],
        "entry": "python -m tensordem.fem_dem_fracture",
        "integration_callers": [
            "tensordem.fem_dem_seeding.clump.seed_floe_bodies",
            "tensordem.fem_dem_seeding.clump.body_ledger",
            "tensordem.fem_dem_seeding.dynamics.heave_audit",
            "tensordem.fem_dem_seeding.dynamics.post_run_ledger",
            "tensordem.fem_dem_seeding.raster.Raster",
            "examples/fem_dem_fracture_dynamic/run.py",
        ],
        "tests": ["tests/test_fem_dem_fracture_dynamic.py"],
        "benchmark": "benchmarks/fem_dem_fracture_dynamic/benchmark.py",
        "repro_cmd": (
            "python -m tensordem.fem_dem_fracture --out runs/p302w2_n03 "
            f"--pressure-rate {pressure_rate_pa_s:g}"),
        "artifacts": {name: {"path": path, "sha256": _sha256(path)}
                      for name, path in artifacts.items()},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "criteria_all_pass": bool(all(
            (value.get("pass") if isinstance(value, dict) else False)
            for key, value in criteria.items() if key.startswith("D"))),
        "wall_time_s": summary["wall_time_s"],
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    summary["artifacts"]["manifest"] = manifest_path
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m tensordem.fem_dem_fracture",
        description=("FEM -> DEM fracture handoff: static census plus the dynamic "
                     "velocity inheritance and the DEM-side intake (campaign "
                     "p302w2-n03)."))
    parser.add_argument("--out", default="runs/p302w2_n03",
                        help="output directory for the artefact bundle")
    parser.add_argument("--pressure-rate", type=float,
                        default=DEFAULT_PRESSURE_RATE_PA_S,
                        help="contact-pressure ramp rate qdot [Pa/s]")
    parser.add_argument("--nx", type=int, default=40, help="FEM cells in x")
    parser.add_argument("--ny", type=int, default=40, help="FEM cells in y")
    parser.add_argument("--length", type=float, default=10.0, help="plate side [m]")
    parser.add_argument("--thickness", type=float, default=None,
                        help="ice thickness [m] (default: plate material default)")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--tag", default="dynamic")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    from dataclasses import replace

    case = IcePlateCase(nx=args.nx, ny=args.ny,
                        length_x=args.length, length_y=args.length)
    if args.thickness is not None:
        case = replace(case, material=replace(case.material, thickness=args.thickness))
    summary = run_dynamic_handoff(
        case, args.out, pressure_rate_pa_s=args.pressure_rate, tag=args.tag,
        make_figures=not args.no_figures)
    if not args.quiet:
        criteria = summary["criteria"]
        print(json.dumps({
            "out": args.out,
            "pressure_rate_pa_s": summary["pressure_rate_pa_s"],
            "n_floes": len(summary["per_floe"]),
            "criteria_pass": {k: v["pass"] for k, v in criteria.items()
                              if isinstance(v, dict) and "pass" in v},
            "max_abs_cell_rate_m_s": criteria["D1_velocity_field_nonzero"]["max_abs_rate_m_s"],
            "momentum_residual": criteria["D2_linear_momentum"]["residual_kg_m_s"],
            "angular_residual": criteria["D3_angular_momentum"]["residual_kg_m2_s"],
            "ke_split_residual_j": criteria["D4_kinetic_energy_split"]["sum_residual_j"],
            "rigid_fraction_plate": criteria["D4_kinetic_energy_split"]["rigid_fraction_plate"],
            "heave_max_period_error": criteria["D7_dem_intake"]["heave_max_period_error"],
            "ke_over_elastic": criteria["D8_quasi_static_validity"]["ke_over_elastic"],
            "benchmark_all_within_tolerance": summary["benchmark_all_within_tolerance"],
            "wall_time_s": summary["wall_time_s"],
        }, indent=2, sort_keys=True))
    failures = [k for k, v in summary["criteria"].items()
                if isinstance(v, dict) and "pass" in v and not v["pass"]]
    if failures:
        print(f"FAILED criteria: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())