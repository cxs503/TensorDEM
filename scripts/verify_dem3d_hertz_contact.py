"""Analytic verification of the DEM3D Hertz normal-contact law.

The 3-D bonded-sphere solver historically used an uncalibrated linear penalty
contact.  ``DEM3DConfig(contact_model="hertz")`` adds the classical Hertz law

    F(delta) = (4/3) E* sqrt(R*) delta^(3/2)

with the reduced modulus ``1/E* = sum((1 - nu_i^2)/E_i)`` and the reduced radius
``1/R* = sum(1/R_i)``.  This script verifies that law against an *independently*
coded closed form (general two-body Hertz relations, with the counterpart set to
a numerically rigid body), against its own force integral, and against the
force/bounce response of the actual integration loop.

Every reported number is written to ``outputs/dem3d_hertz_verification/``; the
markdown table of this file is generated from the JSON, never hand-copied.

Run:  python scripts/verify_dem3d_hertz_contact.py --output outputs/dem3d_hertz_verification
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import torch

from tensordem.dem3d import (
    DEM3DConfig,
    IceDEM3D,
    hertz_contact_potential,
    hertz_effective_radius,
    hertz_equivalent_linear_stiffness,
    hertz_normal_force,
    hertz_reduced_modulus,
    hertz_secant_stiffness,
)

PROTOCOL = "tensordem-dem3d-hertz-verification-v1"
# Declared acceptance threshold for analytic (closed-form) comparisons.  Force
# laws and their integrals are exact identities, so they must close to machine
# precision; the dynamic bounce carries integration error and gets an
# engineering tolerance.
LAW_TOLERANCE = 1e-12
DYNAMIC_TOLERANCE = 0.03
IMPULSE_TOLERANCE = 0.01

YOUNG_MODULUS = 1.0e9
POISSON_RATIO = 0.3
RADIUS = 0.025
DENSITY = 917.0


# --------------------------------------------------------------------------- #
# Reference implementations (deliberately independent of tensordem formulas)
# --------------------------------------------------------------------------- #
def reference_hertz(overlap: float, e1: float, nu1: float, r1: float,
                    e2: float | None = None, nu2: float = 0.3,
                    r2: float | None = None) -> float:
    """Hertz force from the *general* two-body relations.

    ``e2=None`` and ``r2=None`` mean a rigid counterpart, represented by a
    vanishing compliance and an infinite radius.
    """
    compliance = (1.0 - nu1**2) / e1
    if e2 is not None:
        compliance += (1.0 - nu2**2) / e2
    e_eff = 1.0 / compliance
    if r2 is None:
        r_eff = r1
    else:
        r_eff = 1.0 / (1.0 / r1 + 1.0 / r2)
    return (4.0 / 3.0) * e_eff * math.sqrt(r_eff) * overlap**1.5


def reference_impact_max_overlap(mass: float, speed: float, e_eff: float, r_eff: float) -> float:
    """Elastic Hertz impact of a sphere on a rigid plane: delta_max (m)."""
    return (15.0 * mass * speed**2 / (16.0 * e_eff * math.sqrt(r_eff))) ** 0.4


def rel_error(measured: float, reference: float) -> float:
    if reference == 0.0:
        return abs(measured - reference)
    return abs(measured - reference) / abs(reference)


def case(name: str, *, measured: float, reference: float, tolerance: float,
         note: str) -> dict:
    error = rel_error(measured, reference)
    return {
        "case": name,
        "measured": measured,
        "reference": reference,
        "relative_error": error,
        "tolerance": tolerance,
        "passed": bool(error <= tolerance),
        "note": note,
    }


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
def case_force_law() -> dict:
    overlaps = [1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2]
    rows = []
    worst = 0.0
    for delta in overlaps:
        measured = float(hertz_normal_force(
            torch.tensor(delta, dtype=torch.float64),
            hertz_reduced_modulus(YOUNG_MODULUS, POISSON_RATIO),
            hertz_effective_radius(RADIUS)))
        reference = reference_hertz(delta, YOUNG_MODULUS, POISSON_RATIO, RADIUS,
                                   e2=YOUNG_MODULUS, nu2=POISSON_RATIO, r2=RADIUS)
        error = rel_error(measured, reference)
        worst = max(worst, error)
        rows.append({"overlap_m": delta, "force_N": measured,
                     "reference_force_N": reference, "relative_error": error})
    return {
        "case": "hertz_force_vs_independent_closed_form",
        "metric": "worst relative error vs closed form",
        "geometry": "equal spheres, E=1 GPa, nu=0.3, R=25 mm",
        "measured": rows[-1]["force_N"],
        "reference": rows[-1]["reference_force_N"],
        "relative_error": worst,
        "worst_relative_error": worst,
        "tolerance": LAW_TOLERANCE,
        "passed": bool(worst <= LAW_TOLERANCE),
        "samples": rows,
        "note": "worst relative error over the overlap sweep; exact identity expected",
    }


def case_reduced_modulus() -> dict:
    measured_sphere = hertz_reduced_modulus(YOUNG_MODULUS, POISSON_RATIO)
    measured_plane = hertz_reduced_modulus(YOUNG_MODULUS, POISSON_RATIO,
                                           rigid_counterpart=True)
    ref_sphere = 1.0 / (2.0 * (1.0 - POISSON_RATIO**2) / YOUNG_MODULUS)
    ref_plane = 1.0 / ((1.0 - POISSON_RATIO**2) / YOUNG_MODULUS)
    error = max(rel_error(measured_sphere, ref_sphere), rel_error(measured_plane, ref_plane))
    return {
        "case": "reduced_modulus_mapping",
        "metric": "relative error of E* mapping",
        "measured": measured_sphere,
        "reference": ref_sphere,
        "sphere_sphere_E_star_Pa": measured_sphere,
        "sphere_rigid_E_star_Pa": measured_plane,
        "worst_relative_error": error,
        "tolerance": LAW_TOLERANCE,
        "passed": bool(error <= LAW_TOLERANCE),
        "note": "sphere-sphere E* = E/(2(1-nu^2)); rigid plane E* = E/(1-nu^2)",
    }


def case_exponent() -> dict:
    deltas = torch.logspace(-6, -2, 41, dtype=torch.float64)
    forces = hertz_normal_force(
        deltas,
        hertz_reduced_modulus(YOUNG_MODULUS, POISSON_RATIO),
        hertz_effective_radius(RADIUS))
    x = torch.log(deltas)
    y = torch.log(forces)
    xm, ym = x.mean(), y.mean()
    slope = float(((x - xm) * (y - ym)).sum() / ((x - xm).square()).sum())
    error = abs(slope - 1.5)
    return {
        "case": "hertz_exponent_three_halves",
        "metric": "absolute error of log-log slope vs 1.5",
        "measured": slope,
        "reference": 1.5,
        "relative_error": error,
        "tolerance": 1e-10,
        "passed": bool(error <= 1e-10),
        "note": "log-log slope of implemented F(delta)",
    }


def case_potential_integral() -> dict:
    e_star = hertz_reduced_modulus(YOUNG_MODULUS, POISSON_RATIO)
    r_eff = hertz_effective_radius(RADIUS)
    delta = 5e-3
    intervals = 100000
    step = delta / intervals
    total = 0.0
    for i in range(intervals):
        f0 = float(hertz_normal_force(i * step, e_star, r_eff))
        f1 = float(hertz_normal_force((i + 1) * step, e_star, r_eff))
        total += 0.5 * (f0 + f1) * step
    measured = float(hertz_contact_potential(delta, e_star, r_eff))
    error = rel_error(measured, total)
    return {
        "case": "contact_potential_matches_force_integral",
        "metric": "relative error, potential vs integral",
        "measured": measured,
        "reference": total,
        "relative_error": error,
        "tolerance": 1e-10,
        "passed": bool(error <= 1e-10),
        "note": "U = (8/15) E* sqrt(R*) delta^(5/2) vs trapezoidal int F d(delta)",
    }


def case_secant_stiffness() -> dict:
    e_star = hertz_reduced_modulus(YOUNG_MODULUS, POISSON_RATIO)
    r_eff = hertz_effective_radius(RADIUS)
    delta = 1e-3
    measured = float(hertz_secant_stiffness(delta, e_star, r_eff))
    force = float(hertz_normal_force(delta, e_star, r_eff))
    reference = force / delta
    error = rel_error(measured, reference)
    return {
        "case": "secant_stiffness_identity",
        "metric": "relative error of secant stiffness identity",
        "measured": measured,
        "reference": reference,
        "relative_error": error,
        "tolerance": LAW_TOLERANCE,
        "passed": bool(error <= LAW_TOLERANCE),
        "note": "secant stiffness equals F(delta)/delta; used by the linear-equivalent calibration",
    }


def _static_indentation_base() -> DEM3DConfig:
    return DEM3DConfig(
        nx=3, ny=3, nz=3, radius=RADIUS, density=DENSITY,
        contact_model="hertz", young_modulus=YOUNG_MODULUS, poisson_ratio=POISSON_RATIO,
        contact_damping=0.0, drag=0.0, tool_radius=0.1, tool_speed=0.01, tool_gap=0.0,
        fix_x_edges=True,
    )


def case_solver_static_indentation() -> dict:
    """Force-evaluation path: frozen lattice under a prescribed Hertz indenter.

    Overlap grows only through indenter travel, so the analytic overlap is
    ``tool_speed * time``.  Overlaps stay below the multi-particle contact
    threshold, so a single particle is indented.
    """
    base = _static_indentation_base()
    sim = IceDEM3D(replace(base, dt=0.5 * base.recommended_dt))
    sim.fixed[:] = True
    e_plane = hertz_reduced_modulus(YOUNG_MODULUS, POISSON_RATIO, rigid_counterpart=True)
    rows = []
    worst = 0.0
    contacts = 0
    for delta in (1e-5, 1e-4, 1e-3, 5e-3, 1e-2):
        sim.time = delta / base.tool_speed
        _, reaction = sim.forces(update_fracture=False)
        measured = abs(float(reaction[2]))
        reference = reference_hertz(delta, YOUNG_MODULUS, POISSON_RATIO, RADIUS)
        error = rel_error(measured, reference)
        worst = max(worst, error)
        contacts = max(contacts, int((sim.config.radius + sim.config.tool_radius
                                     - torch.linalg.vector_norm(
                                         sim.positions - sim.tool_position, dim=1) > 0).sum()))
        rows.append({"overlap_m": delta, "reaction_N": measured,
                     "reference_force_N": reference, "relative_error": error,
                     "E_star_rigid_Pa": e_plane})
    return {
        "case": "solver_static_hertz_indentation",
        "metric": "worst relative error, solver vs closed form",
        "measured": rows[-1]["reaction_N"],
        "reference": rows[-1]["reference_force_N"],
        "relative_error": worst,
        "worst_relative_error": worst,
        "tolerance": 1e-9,
        "passed": bool(worst <= 1e-9 and contacts == 1),
        "max_simultaneous_contacts": contacts,
        "samples": rows,
        "note": "solver force path vs rigid-plane closed form; single contact required",
    }


def case_solver_dynamic_impact() -> dict:
    """Explicit integration: particle dropped onto a Hertzian rigid platen.

    Elastic impact of mass ``m`` at speed ``v`` on a Hertz plane has
    ``delta_max = (15 m v^2 / (16 E* sqrt(R)))^(2/5)``; the impulse equals
    ``2 m v`` for a purely elastic (undamped) contact.
    """
    speed = 0.5
    base = DEM3DConfig(
        nx=2, ny=2, nz=2, radius=RADIUS, density=DENSITY,
        bond_stiffness=1e-4, contact_stiffness=5_000.0,
        contact_model="hertz", young_modulus=YOUNG_MODULUS, poisson_ratio=POISSON_RATIO,
        contact_damping=0.0, drag=0.0, platen_damping=0.0,
        bottom_platen_enabled=True, bottom_platen_gap=0.0,
        fix_x_edges=False, tool_speed=1.0,
    )
    cfg = replace(base, dt=0.4 * base.recommended_dt)
    sim = IceDEM3D(cfg)
    xy = torch.linalg.vector_norm(sim.positions[:, :2], dim=1)
    lowest = float(sim.positions[:, 2].min())
    candidates = torch.nonzero(torch.isclose(sim.positions[:, 2],
                                             torch.tensor(lowest, dtype=sim.dtype)), as_tuple=False).flatten()
    index = int(candidates[torch.argmin(xy[candidates])])
    sim.velocities[index, 2] = -speed

    e_plane = hertz_reduced_modulus(YOUNG_MODULUS, POISSON_RATIO, rigid_counterpart=True)
    delta_max = reference_impact_max_overlap(cfg.mass, speed, e_plane, RADIUS)
    force_max_ref = (4.0 / 3.0) * e_plane * math.sqrt(RADIUS) * delta_max**1.5
    impulse_ref = 2.0 * cfg.mass * speed

    duration = 1.5e-3
    steps = int(math.ceil(duration / cfg.dt))
    times, forces = [], []
    for _ in range(steps):
        t_now = sim.time
        sim.step()
        times.append(t_now)
        forces.append(abs(float(sim.last_bottom_platen_reaction_z)))
    impulse = 0.0
    for i in range(len(times) - 1):
        impulse += 0.5 * (forces[i] + forces[i + 1]) * (times[i + 1] - times[i])
    force_max = max(forces)
    result = case("solver_dynamic_hertz_impact_force",
                  measured=force_max, reference=force_max_ref,
                  tolerance=DYNAMIC_TOLERANCE,
                  note=f"elastic impact v={speed} m/s, dt={cfg.dt:.3e} s, {steps} steps")
    result["metric"] = "relative error of peak force vs closed form"
    result.update({
        "impulse_Ns": impulse,
        "impulse_reference_Ns": impulse_ref,
        "impulse_relative_error": rel_error(impulse, impulse_ref),
        "impulse_passed": bool(rel_error(impulse, impulse_ref) <= IMPULSE_TOLERANCE),
        "peak_overlap_m": delta_max,
        "dt_s": cfg.dt,
        "recommended_dt_s": base.recommended_dt,
        "steps": steps,
        "impact_speed_m_per_s": speed,
        "history": [{"time_s": t, "reaction_N": f} for t, f in zip(times, forces)],
    })
    return result


def case_legacy_linear_error() -> dict:
    """Quantify the verification debt of the legacy linear penalty contact.

    For an ice-like modulus the legacy stiffness (5000 N/m default) is compared
    with Hertz over an overlap sweep, and the linear-equivalent calibration is
    tabulated.
    """
    e_star_plane = hertz_reduced_modulus(YOUNG_MODULUS, POISSON_RATIO, rigid_counterpart=True)
    legacy_k = 5_000.0
    rows = []
    for delta in [1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2]:
        hertz = reference_hertz(delta, YOUNG_MODULUS, POISSON_RATIO, RADIUS)
        linear = legacy_k * delta
        rows.append({
            "overlap_m": delta,
            "hertz_force_N": hertz,
            "linear_force_N": linear,
            "linear_relative_error": rel_error(linear, hertz),
            "equivalent_stiffness_N_per_m": hertz / delta,
        })
    # Overlap at which the legacy stiffness reproduces the Hertz force exactly.
    match = (legacy_k / ((4.0 / 3.0) * e_star_plane * math.sqrt(RADIUS))) ** 2
    table = {}
    for e_pa in (1.0e9, 5.0e9, 1.0e10):
        table[f"E_{e_pa:.0e}"] = {
            f"delta_{d:.0e}": hertz_equivalent_linear_stiffness(
                e_pa, POISSON_RATIO, RADIUS, d, rigid_counterpart=True)
            for d in (1e-4, 1e-3, 1e-2)
        }
    worst = max(row["linear_relative_error"] for row in rows)
    return {
        "case": "legacy_linear_penalty_vs_hertz",
        "metric": "max relative error of the legacy linear law vs Hertz",
        "relative_error": worst,
        "measured": worst,
        "reference": 0.0,
        "max_relative_error": max(row["linear_relative_error"] for row in rows),
        "tolerance": DYNAMIC_TOLERANCE,
        "passed": bool(max(row["linear_relative_error"] for row in rows) > DYNAMIC_TOLERANCE),
        "legacy_stiffness_N_per_m": legacy_k,
        "matching_overlap_m": match,
        "samples": rows,
        "equivalent_stiffness_table": table,
        "note": ("the case passes when the legacy linear law deviates by more than the "
                 "engineering tolerance, i.e. the debt is real and bounded"),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _figures(results: dict, out: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures = []
    e_star = hertz_reduced_modulus(YOUNG_MODULUS, POISSON_RATIO)
    r_eff = hertz_effective_radius(RADIUS)
    deltas = np.logspace(-6, -1.5, 120)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.loglog(deltas, [float(hertz_normal_force(float(d), e_star, r_eff)) for d in deltas],
              "k-", label="Hertz sphere-sphere (implemented)")
    ax.loglog(deltas, [reference_hertz(float(d), YOUNG_MODULUS, POISSON_RATIO, RADIUS)
                       for d in deltas], "C3--", lw=1.0,
              label="independent closed form")
    static = results["solver_static_hertz_indentation"]["samples"]
    ax.loglog([row["overlap_m"] for row in static], [row["reaction_N"] for row in static],
              "o", ms=6, mfc="none", mec="C0", label="solver reaction (rigid indenter)")
    legacy_rows = results["legacy_linear_penalty_vs_hertz"]["samples"]
    ax.loglog([row["overlap_m"] for row in legacy_rows], [row["linear_force_N"] for row in legacy_rows],
              "s-", color="0.5", ms=5, label="legacy linear penalty (5000 N/m)")
    ax.set_xlabel("normal overlap $\\delta$ (m)")
    ax.set_ylabel("normal contact force (N)")
    ax.set_title("DEM3D Hertz contact law vs closed form")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = out / "fig1_hertz_force_law.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    figures.append({"path": path, "caption": "Implemented Hertz law, independent closed form, "
                                            "solver reaction under a static rigid indenter, and "
                                            "the legacy linear penalty law."})

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    errs = [rel_error(row["linear_force_N"], row["hertz_force_N"]) * 100 for row in legacy_rows]
    ax.semilogx([row["overlap_m"] for row in legacy_rows], errs, "s-", color="C1")
    ax.axhline(3.0, color="C3", ls="--", lw=1.0, label="3% tolerance")
    ax.set_xlabel("normal overlap $\\delta$ (m)")
    ax.set_ylabel("linear-penalty relative error (%)")
    ax.set_title("Verification debt: legacy linear penalty vs Hertz")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = out / "fig2_linear_penalty_error.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    figures.append({"path": path, "caption": "Relative error of the legacy linear penalty contact "
                                            "(5000 N/m) against the Hertz law versus overlap."})

    hist = results["solver_dynamic_hertz_impact_force"]["history"]
    t = np.array([row["time_s"] for row in hist])
    f = np.array([row["reaction_N"] for row in hist])
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(t * 1e3, f, "-", lw=1.2, color="C0", label="DEM3D reaction")
    ax.axhline(results["solver_dynamic_hertz_impact_force"]["reference"],
               color="C3", ls="--", lw=1.0, label="Hertz impact closed form $F_{max}$")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("platen reaction (N)")
    ax.set_title("Hertz impact on a rigid platen (elastic bounce)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    path = out / "fig3_hertz_impact_history.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    figures.append({"path": path, "caption": "Explicit-integration impact history against the "
                                            "analytic Hertz impact peak force."})
    return figures


def _render_tables(results: dict, cases: list[dict]) -> str:
    import math as _math

    def fmt(value: float) -> str:
        return "n/a" if value is None or (isinstance(value, float) and _math.isnan(value)) else f"{value:.6e}"

    lines = ["| case | error metric | error | tolerance | verdict |", "|---|---|---|---|---|"]
    for c in cases:
        error = c["relative_error"] if "relative_error" in c else c.get("worst_relative_error")
        lines.append("| `{case}` | {metric} | {e} | {t:.1e} | {v} |".format(
            case=c["case"], metric=c["metric"], e=fmt(error),
            t=c["tolerance"], v="PASS" if c["passed"] else "FAIL"))

    lines += ["", "### Force-law sweep (equal spheres, E = 1 GPa, nu = 0.3, R = 25 mm)", "",
              "| overlap (m) | implemented F (N) | independent closed form F (N) | relative error |",
              "|---|---|---|---|"]
    for row in results["hertz_force_vs_independent_closed_form"]["samples"]:
        lines.append("| {:.0e} | {:.6e} | {:.6e} | {:.2e} |".format(
            row["overlap_m"], row["force_N"], row["reference_force_N"], row["relative_error"]))

    static = results["solver_static_hertz_indentation"]
    lines += ["", "### Solver force path under a prescribed rigid indenter", "",
              "| overlap (m) | solver reaction (N) | closed form F (N) | relative error |",
              "|---|---|---|---|"]
    for row in static["samples"]:
        lines.append("| {:.0e} | {:.6e} | {:.6e} | {:.2e} |".format(
            row["overlap_m"], row["reaction_N"], row["reference_force_N"], row["relative_error"]))
    lines.append("")
    lines.append("Simultaneous indenter contacts during the sweep: {}".format(
        static["max_simultaneous_contacts"]))

    dyn = results["solver_dynamic_hertz_impact_force"]
    lines += ["", "### Elastic impact on a rigid Hertzian platen", "",
              "| quantity | DEM3D | closed form | relative error |",
              "|---|---|---|---|",
              "| peak force (N) | {:.6e} | {:.6e} | {:.3e} |".format(
                  dyn["measured"], dyn["reference"], dyn["relative_error"]),
              "| impulse (N s) | {:.6e} | {:.6e} | {:.3e} |".format(
                  dyn["impulse_Ns"], dyn["impulse_reference_Ns"], dyn["impulse_relative_error"]),
              "", "dt = {:.3e} s (0.4 x recommended {:.3e} s), {} steps, impact speed {} m/s.".format(
                  dyn["dt_s"], dyn["recommended_dt_s"], dyn["steps"], dyn["impact_speed_m_per_s"])]

    legacy = results["legacy_linear_penalty_vs_hertz"]
    lines += ["", "### Legacy linear penalty vs Hertz (verification debt, k = 5000 N/m)", "",
              "| overlap (m) | Hertz F (N) | linear F (N) | relative error | Hertz secant k (N/m) |",
              "|---|---|---|---|---|"]
    for row in legacy["samples"]:
        lines.append("| {:.0e} | {:.6e} | {:.6e} | {:.3f} | {:.6e} |".format(
            row["overlap_m"], row["hertz_force_N"], row["linear_force_N"],
            row["linear_relative_error"], row["equivalent_stiffness_N_per_m"]))
    lines.append("")
    lines.append("Overlap at which 5000 N/m reproduces the Hertz force exactly: {:.3e} m.".format(
        legacy["matching_overlap_m"]))
    lines += ["", "| ice modulus | k_eq at 1e-4 m | k_eq at 1e-3 m | k_eq at 1e-2 m |",
              "|---|---|---|---|"]
    for key, row in legacy["equivalent_stiffness_table"].items():
        lines.append("| {} | {:.4e} | {:.4e} | {:.4e} |".format(
            key, row["delta_1e-04"], row["delta_1e-03"], row["delta_1e-02"]))
    return "\n".join(lines) + "\n"


def _inject_docs(path: Path, table: str) -> None:
    begin, end = "<!-- AUTOGEN:RESULT_TABLES -->", "<!-- /AUTOGEN -->"
    text = path.read_text()
    if begin not in text or end not in text:
        raise SystemExit(f"{path} lacks the {begin} / {end} markers")
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    path.write_text(head + begin + "\n" + table + end + tail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/dem3d_hertz_verification")
    parser.add_argument("--docs", default="docs/dem3d_hertz_contact_verification.md",
                        help="markdown file whose AUTOGEN block receives the tables")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    results = {
        "protocol": PROTOCOL,
        "parameters": {
            "young_modulus_Pa": YOUNG_MODULUS,
            "poisson_ratio": POISSON_RATIO,
            "particle_radius_m": RADIUS,
            "density_kg_per_m3": DENSITY,
            "reduced_modulus_sphere_sphere_Pa": hertz_reduced_modulus(YOUNG_MODULUS, POISSON_RATIO),
            "reduced_modulus_rigid_Pa": hertz_reduced_modulus(YOUNG_MODULUS, POISSON_RATIO,
                                                              rigid_counterpart=True),
        },
        "tolerances": {
            "law": LAW_TOLERANCE,
            "dynamic": DYNAMIC_TOLERANCE,
            "impulse": IMPULSE_TOLERANCE,
        },
    }
    results["hertz_force_vs_independent_closed_form"] = case_force_law()
    results["reduced_modulus_mapping"] = case_reduced_modulus()
    results["hertz_exponent_three_halves"] = case_exponent()
    results["contact_potential_matches_force_integral"] = case_potential_integral()
    results["secant_stiffness_identity"] = case_secant_stiffness()
    results["solver_static_hertz_indentation"] = case_solver_static_indentation()
    results["solver_dynamic_hertz_impact_force"] = case_solver_dynamic_impact()
    results["legacy_linear_penalty_vs_hertz"] = case_legacy_linear_error()

    cases = [v for v in results.values() if isinstance(v, dict) and "case" in v]
    results["summary"] = {
        "cases": len(cases),
        "passed": sum(1 for c in cases if c["passed"]),
        "failed": [c["case"] for c in cases if not c["passed"]],
        "all_passed": all(c["passed"] for c in cases),
    }
    figures = _figures(results, out)
    results["figures"] = [{"file": f["path"].name, "caption": f["caption"]} for f in figures]

    (out / "hertz_verification_report.json").write_text(json.dumps(results, indent=2))

    table = _render_tables(results, cases)
    (out / "hertz_verification_table.md").write_text(table)

    csv = ["case,metric,error,tolerance,passed"]
    for c in cases:
        csv.append("{case},{metric},{e:.12e},{t:.3e},{p}".format(
            case=c["case"], metric=c["metric"],
            e=c["relative_error"] if "relative_error" in c else c.get("worst_relative_error", float("nan")),
            t=c["tolerance"], p=c["passed"]))
    (out / "hertz_verification_summary.csv").write_text("\n".join(csv) + "\n")

    if args.docs:
        _inject_docs(Path(args.docs), table)

    print(json.dumps(results["summary"], indent=2))
    for c in cases:
        print(f"{c['case']:45s} {'PASS' if c['passed'] else 'FAIL'}")
    return 0 if results["summary"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())