"""Verify the two-dimensional bonded-disk lattice against its exact elasticity.

Fills the verification debt described in ``tensordem.lattice_elasticity2d``:
``IceDEM``'s eight-neighbour (square + face-diagonal) central-force lattice had
no reference modulus, so ``bond_stiffness`` was not tied to any Young's modulus
and the specimen-direction bias was only qualitative.

Four independent routes are compared:

1. closed form ``C11 = k_a + k_d``, ``C12 = C66 = k_d``;
2. direct Cauchy-Born (Voigt) summation over the ideal bond set;
3. the solver's *own* bond topology (``IceDEM.bond_indices``), interior average;
4. the solver's *own* bond energy (``IceDEM.mechanical_energy``) under affine
   strain, with the exact free-surface bond deficit added back.

It then measures the practical consequence: the same indentation case run with
the engine default lattice (``k_d = k_a``) and with the exactly isotropic
calibration (``k_d = k_a / 2``), reached through ``CalibratedIceDEM`` (<- per-bond
stiffness, no engine change).

Run:
    python scripts/validate_lattice_elasticity2d.py
    python scripts/validate_lattice_elasticity2d.py --steps 300 --output /tmp/x

The runner exits non-zero if an acceptance check fails.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tensordem.calibrated_dem import CalibratedIceDEM  # noqa: E402
from tensordem.dem import DEMConfig, IceDEM  # noqa: E402
from tensordem.lattice_elasticity2d import (  # noqa: E402
    ISOTROPIC_DIAGONAL_SCALE,
    affine_stiffness_tensor,
    anisotropy_report,
    bulk_stiffness_from_sim,
    directional_moduli,
    isotropic_bond_stiffness,
    stiffness_components,
    stiffness_from_strain_energy,
    tensor_error,
    voigt_sum_stiffness,
    canonical_bonds,
)

BOND_STIFFNESS = 2000.0
RADIUS = 0.025
THICKNESS = 0.2
DENSITY = 917.0
CONTACT_STIFFNESS = 2000.0
TOOL_RADIUS = 0.1
TOOL_SPEED = 1.0


def lattice_config(nx: int, ny: int, *, fix_edges: bool) -> DEMConfig:
    return DEMConfig(
        nx=nx,
        ny=ny,
        radius=RADIUS,
        thickness=THICKNESS,
        density=DENSITY,
        bond_stiffness=BOND_STIFFNESS,
        contact_stiffness=CONTACT_STIFFNESS,
        tool_radius=TOOL_RADIUS,
        tool_speed=TOOL_SPEED,
        fix_edges=fix_edges,
    )


def four_route_check(nx: int = 21) -> dict:
    """Compare closed form, Voigt summation, solver topology and solver energy."""
    config = lattice_config(nx, nx, fix_edges=False)
    spacing = 2.0 * RADIUS
    closed = stiffness_components(BOND_STIFFNESS)
    closed_tensor = affine_stiffness_tensor(BOND_STIFFNESS, spacing=spacing)
    normals, lengths, stiffness = canonical_bonds(
        spacing, bond_stiffness=BOND_STIFFNESS
    )
    voigt_tensor = voigt_sum_stiffness(
        normals, lengths, stiffness, spacing * spacing, weight=0.5
    )
    sim = IceDEM(config)
    topology = bulk_stiffness_from_sim(sim)
    started = time.perf_counter()
    energy_tensor = stiffness_from_strain_energy(sim)
    energy_seconds = time.perf_counter() - started
    return {
        "nx": nx,
        "closed_form": closed,
        "voigt_sum_error": tensor_error(voigt_tensor, closed_tensor),
        "solver_topology_error": tensor_error(topology["stiffness"], closed_tensor),
        "solver_energy_error": tensor_error(energy_tensor, closed_tensor),
        "solver_energy_seconds": energy_seconds,
        "interior_particles": topology["interior_particles"],
        "coordination_max": topology["coordination_max"],
        "closed_tensor": closed_tensor,
        "topology_tensor": topology["stiffness"],
        "energy_tensor": energy_tensor,
    }


def isotropic_check(nx: int = 21) -> dict:
    """Same four routes for the exactly isotropic calibration (k_d = k_a/2)."""
    spacing = 2.0 * RADIUS
    config = lattice_config(nx, nx, fix_edges=False)
    reference = IceDEM(config)
    coefficients = isotropic_bond_stiffness(reference)
    calibrated = CalibratedIceDEM(config, bond_stiffness=coefficients)
    closed_tensor = affine_stiffness_tensor(
        BOND_STIFFNESS, ISOTROPIC_DIAGONAL_SCALE, spacing=spacing
    )
    topology = bulk_stiffness_from_sim(calibrated)
    energy_tensor = stiffness_from_strain_energy(calibrated)
    return {
        "diagonal_stiffness_N_per_m": float(coefficients.min()),
        "axis_stiffness_N_per_m": float(coefficients.max()),
        "solver_topology_error": tensor_error(topology["stiffness"], closed_tensor),
        "solver_energy_error": tensor_error(energy_tensor, closed_tensor),
        "closed_tensor": closed_tensor,
        "topology_tensor": topology["stiffness"],
        "energy_tensor": energy_tensor,
    }


def directional_table() -> list[dict]:
    rows = []
    for label, scale in (
        ("engine_default", 1.0),
        ("isotropic_calibration", ISOTROPIC_DIAGONAL_SCALE),
    ):
        for row in directional_moduli(
            [0.0, 7.5, 15.0, 22.5, 30.0, 37.5, 45.0, 52.5, 60.0, 67.5, 75.0, 82.5, 90.0],
            BOND_STIFFNESS,
            scale,
            spacing=2.0 * RADIUS,
        ):
            rows.append(
                {
                    "lattice": label,
                    "diagonal_scale": scale,
                    "theta_deg": row["theta_deg"],
                    "E_over_ka": row["youngs_modulus_N_per_m"] / BOND_STIFFNESS,
                    "poisson_ratio": row["poisson_ratio"],
                    "coupling_c16_over_ka": row["coupling_c16"] / BOND_STIFFNESS,
                }
            )
    return rows


def indentation_case(
    *, label: str, isotropic: bool, steps: int, nx: int = 21, ny: int = 5
) -> dict:
    config = lattice_config(nx, ny, fix_edges=True)
    base = IceDEM(config)
    if isotropic:
        sim = CalibratedIceDEM(config, bond_stiffness=isotropic_bond_stiffness(base))
    else:
        sim = base
    reaction: list[float] = []
    times: list[float] = []
    started = time.perf_counter()
    for _ in range(steps):
        _, tool_reaction = sim.forces()
        reaction.append(float(torch.linalg.vector_norm(tool_reaction)))
        times.append(sim.time)
        sim.step()
    elapsed = time.perf_counter() - started
    energy = sim.mechanical_energy()
    finite = all(math.isfinite(value) for value in reaction)
    peak_index = max(range(len(reaction)), key=lambda index: reaction[index])
    return {
        "label": label,
        "isotropic_calibration": isotropic,
        "nx": nx,
        "ny": ny,
        "steps": steps,
        "dt_s": sim.dt,
        "end_time_s": sim.time,
        "tool_penetration_m": TOOL_SPEED * sim.time,
        "peak_tool_reaction_N": reaction[peak_index],
        "peak_time_s": times[peak_index],
        "mean_tool_reaction_N": sum(reaction) / len(reaction),
        "final_tool_reaction_N": reaction[-1],
        "final_bond_strain_energy_J": energy["bond_J"],
        "final_kinetic_energy_J": energy["kinetic_J"],
        "broken_bonds": sim.broken_bonds,
        "all_finite": finite,
        "wall_seconds": elapsed,
        "history_N": reaction,
        "history_t_s": times,
    }


def make_figures(
    report: dict, table: list[dict], cases: list[dict], output: Path, doc: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: list[Path] = []
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 4.2))
    ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
    ax_nu = fig.add_subplot(1, 2, 2)
    colours = {"engine_default": "#b2182b", "isotropic_calibration": "#2166ac"}
    labels = {
        "engine_default": r"engine default $k_d = k_a$",
        "isotropic_calibration": r"isotropic calibration $k_d = k_a/2$",
    }
    for lattice in ("engine_default", "isotropic_calibration"):
        rows = [row for row in table if row["lattice"] == lattice]
        theta = [math.radians(row["theta_deg"]) for row in rows]
        modulus = [row["E_over_ka"] for row in rows]
        ax_polar.plot(theta, modulus, marker="o", ms=3, color=colours[lattice],
                      label=labels[lattice])
        ax_nu.plot([row["theta_deg"] for row in rows],
                   [row["poisson_ratio"] for row in rows],
                   marker="o", ms=3, color=colours[lattice], label=labels[lattice])
    ax_polar.set_title("plane-stress $E(\\theta)/k_a$ from the exact\nCauchy-Born tensor", fontsize=10)
    ax_polar.set_theta_zero_location("E")
    ax_polar.legend(loc="lower right", bbox_to_anchor=(1.25, -0.12), fontsize=7.5)
    ax_nu.set_title("plane-stress Poisson ratio $\\nu(\\theta)$", fontsize=10)
    ax_nu.set_xlabel("loading direction $\\theta$ (deg)")
    ax_nu.set_ylabel("$\\nu$")
    ax_nu.grid(alpha=0.3)
    ax_nu.legend(fontsize=7.5)
    fig.suptitle(
        "2-D bonded-disk lattice: four-fold symmetry is not isotropy "
        "(E(45 deg)/E(0 deg) = 1.6)", fontsize=10
    )
    fig.tight_layout()
    path = figure_dir / "directional_modulus.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for case in cases:
        colour = "#2166ac" if case["isotropic_calibration"] else "#b2182b"
        axes[0].plot(case["history_t_s"], case["history_N"], color=colour,
                     label=f"{case['label']} (peak {case['peak_tool_reaction_N']:.3g} N)")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("|ice-on-tool reaction| (N)")
    axes[0].set_title("indentation case, %.0f steps, dt = %.2e s" % (cases[0]["steps"], cases[0]["dt_s"]), fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    names = [case["label"] for case in cases]
    peaks = [case["peak_tool_reaction_N"] for case in cases]
    axes[1].bar(names, peaks, color=["#b2182b", "#2166ac"][: len(cases)])
    for index, value in enumerate(peaks):
        axes[1].text(index, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylabel("peak |reaction| (N)")
    axes[1].set_title("peak reaction %+.1f %% ; final bond energy %+.1f %%"
                      % (100.0 * (peaks[-1] / peaks[0] - 1.0),
                         100.0 * (cases[-1]["final_bond_strain_energy_J"]
                                  / cases[0]["final_bond_strain_energy_J"] - 1.0)), fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = figure_dir / "isotropic_calibration_impact.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(path)

    for variation in figures:
        relative = variation.relative_to(output)
        print(f"figure: {doc.parent / doc.stem.replace('_', '_')}" if False else f"figure: {relative}")
    return figures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/lattice_elasticity2d")
    parser.add_argument("--doc", default="docs/dem2d_lattice_elasticity.md")
    parser.add_argument("--nx", type=int, default=11, help="square patch size for the modulus routes")
    parser.add_argument("--steps", type=int, default=900, help="indentation steps per case")
    parser.add_argument("--skip-impact", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print("route 1-4 (engine default lattice)")
    default_routes = four_route_check(args.nx)
    print("   closed form C11/C12/C66 = %.6f / %.6f / %.6f N/m"
          % (default_routes["closed_form"]["c11"], default_routes["closed_form"]["c12"],
             default_routes["closed_form"]["c66"]))
    print("   voigt error %.2e, solver topology %.2e, solver energy %.2e"
          % (default_routes["voigt_sum_error"], default_routes["solver_topology_error"],
             default_routes["solver_energy_error"]))

    print("route 1-4 (isotropic calibration)")
    iso_routes = isotropic_check(args.nx)
    print("   solver topology %.2e, solver energy %.2e"
          % (iso_routes["solver_topology_error"], iso_routes["solver_energy_error"]))

    default_report = anisotropy_report(BOND_STIFFNESS, 1.0, spacing=2.0 * RADIUS, thickness=THICKNESS)
    iso_report = anisotropy_report(
        BOND_STIFFNESS, ISOTROPIC_DIAGONAL_SCALE, spacing=2.0 * RADIUS, thickness=THICKNESS
    )
    print("default: E45/E0 = %.4f, nu in [%.3f, %.3f], zener = %.3f"
          % (default_report["modulus_ratio_45_over_0"], default_report["poisson_ratio_min"],
             default_report["poisson_ratio_max"], default_report["zener_ratio_2d"]))
    print("isotropic: E45/E0 = %.4f, nu = %.4f, zener = %.3f, E = %.4f k_a"
          % (iso_report["modulus_ratio_45_over_0"], iso_report["poisson_ratio_max"],
             iso_report["zener_ratio_2d"],
             iso_report["youngs_modulus_0deg_N_per_m"] / BOND_STIFFNESS))

    table = directional_table()
    with (output / "directional_moduli.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    cases: list[dict] = []
    if not args.skip_impact:
        print("impact case: engine default lattice")
        cases.append(indentation_case(label="engine_default", isotropic=False, steps=args.steps))
        print("impact case: isotropic calibration")
        cases.append(indentation_case(label="isotropic_calibration", isotropic=True, steps=args.steps))
        with (output / "indentation_impact.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["case", "time_s", "reaction_N"])
            for case in cases:
                for time_s, value in zip(case["history_t_s"], case["history_N"]):
                    writer.writerow([case["label"], f"{time_s:.9e}", f"{value:.9e}"])
        peak_change = cases[1]["peak_tool_reaction_N"] / cases[0]["peak_tool_reaction_N"] - 1.0
        print("   peak reaction change %+.2f %%, final bond energy change %+.2f %%, "
              "broken bonds %d -> %d"
              % (100.0 * peak_change,
                 100.0 * (cases[1]["final_bond_strain_energy_J"]
                          / cases[0]["final_bond_strain_energy_J"] - 1.0),
                 cases[0]["broken_bonds"], cases[1]["broken_bonds"]))

    checks = {
        "closed_form_matches_voigt_sum": default_routes["voigt_sum_error"] < 1e-12,
        "closed_form_matches_solver_topology": default_routes["solver_topology_error"] < 1e-12,
        "closed_form_matches_solver_energy": default_routes["solver_energy_error"] < 1e-5,
        "isotropic_calibration_matches_solver_topology": iso_routes["solver_topology_error"] < 1e-12,
        "isotropic_calibration_matches_solver_energy": iso_routes["solver_energy_error"] < 1e-5,
        "engine_default_is_anisotropic": default_report["modulus_ratio_45_over_0"] > 1.15,
        "isotropic_calibration_is_isotropic": abs(iso_report["modulus_ratio_45_over_0"] - 1.0) < 1e-9,
        "isotropic_poisson_ratio_third": abs(iso_report["poisson_ratio_max"] - 1.0 / 3.0) < 1e-9,
        "isotropic_zener_unity": abs(iso_report["zener_ratio_2d"] - 1.0) < 1e-12,
        "engine_default_zener_two": abs(default_report["zener_ratio_2d"] - 2.0) < 1e-12,
    }
    if cases:
        checks["impact_case_finite"] = all(case["all_finite"] for case in cases)
        checks["isotropic_case_softer"] = (
            cases[1]["peak_tool_reaction_N"] < cases[0]["peak_tool_reaction_N"]
        )
        checks["isotropic_case_stores_less_bond_energy"] = (
            cases[1]["final_bond_strain_energy_J"] < cases[0]["final_bond_strain_energy_J"]
        )
    status = all(checks.values())

    payload = {
        "schema": "tensordem.lattice_elasticity2d/1",
        "inputs": {
            "bond_stiffness_N_per_m": BOND_STIFFNESS,
            "radius_m": RADIUS,
            "thickness_m": THICKNESS,
            "density_kg_per_m3": DENSITY,
            "contact_stiffness_N_per_m": CONTACT_STIFFNESS,
            "tool_radius_m": TOOL_RADIUS,
            "tool_speed_m_per_s": TOOL_SPEED,
            "torch_version": torch.__version__,
            "python": sys.version.split()[0],
        },
        "routes": {
            "engine_default": {
                key: value
                for key, value in default_routes.items()
                if key
                not in {"closed_tensor", "topology_tensor", "energy_tensor"}
            },
            "isotropic_calibration": {
                key: value
                for key, value in iso_routes.items()
                if key not in {"closed_tensor", "topology_tensor", "energy_tensor"}
            },
        },
        "anisotropy_engine_default": default_report,
        "anisotropy_isotropic": iso_report,
        "impact": [
            {key: value for key, value in case.items() if not key.startswith("history_")}
            for case in cases
        ],
        "checks": checks,
        "status": "PASS" if status else "FAIL",
    }
    (output / "lattice_elasticity2d_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    make_figures(payload, table, cases, output, Path(args.doc))
    print("status:", payload["status"])
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    return 0 if status else 1


if __name__ == "__main__":
    raise SystemExit(main())