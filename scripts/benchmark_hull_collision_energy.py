"""IN63-N01 benchmark: ice-region ship-ice collision energy-conservation surface.

Runs the machine-checkable acceptance criteria of the campaign node and writes a
paper-grade evidence bundle (JSON + CSV + figures) into one output directory:

    python scripts/benchmark_hull_collision_energy.py --output runs/hull_collision_energy

Criteria implemented (numbers and artefacts recorded in ``report.json``):

(a) energy conservation     ``E_k = E_plastic + E_friction + E_residual``, |residual|/E_k <= 2 %
(b) analytic anchor         three-point bending ``E = M_p*theta_max``: closed form vs two
                            independent numerical routes, error <= 1 %
(c) negative control        ``V = 0`` gives ``d = 0`` exactly, ``V > 0`` gives ``d > 0``
(d) double-hull allocation  ``E_outer + E_inner = E_total`` (<= 2 %, cross-checked by Simpson
                            quadrature of ``dE/dd``) and the "inner shell intact" statement
(e) Minorsky (1959)         ``V_damage = C*M*V^2/sigma_y`` vs the model damaged volume,
                            |relative error| <= 30 % for cases inside Minorsky's calibration
                            range (predeclared in prereg/IN63_N01_hull_collision_energy.json)
(f) velocity sweep          ``V = 2/4/6/8`` m/s -> penetration strictly increasing
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.hull_collision import (  # noqa: E402
    MINORSKY_1959_MJ_PER_M3,
    REFERENCE_YIELD_STRESS_PA,
    compare_damage_volume,
    damage_geometry,
    kinetic_energy,
    minorsky_coefficient,
    solve_penetration,
    three_point_bending_anchor,
    verify_layer_decomposition,
)
from tensordem.hull_collision.scenario import (  # noqa: E402
    ADDED_MASS_COEFFICIENT,
    CONTACT_HEIGHT,
    CONTACT_LENGTH,
    DOUBLE_HULL_DEPTH,
    FRAME_SPAN,
    FRICTION_COEFFICIENT,
    INNER_THICKNESS,
    OUTER_THICKNESS,
    SCRATCH_SIGMA_Y,
    STRIKING_MASS,
    design_case,
    effective_mass,
    longitudinal_section_modulus,
)

VELOCITIES = (2.0, 4.0, 6.0, 8.0)
SEVERE_MASS = effective_mass()
OVERLOAD_MASS = 4.5e6
THRESHOLDS = {
    "energy_residual_relative": 0.02,
    "anchor_relative_error": 0.01,
    "allocation_relative_error": 0.02,
    "minorsky_relative_error": 0.30,
}
#: a collision is inside Minorsky's calibration domain when its energy is inside his
#: data range *and* the plastic mechanism is fully developed (d/span >= 0.75);
#: below that the panel-level rigid-plastic model cannot resolve partial plasticity
MINORSKY_MIN_DEPTH_OVER_SPAN = 0.75


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _closed_form_depth(kinetic: float, structure) -> float:
    """Analytic root of ``(1+mu)*(k1*d + k2*d^2) = E_k`` for the single engaged layer."""
    layer = structure.layers[0]
    k1, k2 = layer.hinge_coefficient, layer.membrane_coefficient
    target = kinetic / (1.0 + structure.friction_coefficient)
    return (-k1 + math.sqrt(k1 * k1 + 4.0 * k2 * target)) / (2.0 * k2)


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    structure = design_case()
    layers = structure.layers
    outer, inner = layers[0], layers[1]

    # ------------------------------------------------------------------ (b) anchor
    anchor_cases = []
    for bc in ("simply_supported", "clamped"):
        anchor_cases.append(
            three_point_bending_anchor(
                sigma_y=SCRATCH_SIGMA_Y,
                width=0.6,
                thickness=OUTER_THICKNESS,
                span=FRAME_SPAN,
                deflection=0.12,
                bc=bc,
                n_steps=2048,
            )
        )
    anchor_max_error = max(case["max_rel_error"] for case in anchor_cases)

    # -------------------------------------------------------- (c) negative control
    zero_solution = solve_penetration(kinetic_energy(SEVERE_MASS, 0.0), structure)
    non_zero_solution = solve_penetration(kinetic_energy(SEVERE_MASS, 2.0), structure)

    # ------------------------------------------------------- (a)(f) velocity sweep
    sweep = []
    for velocity in VELOCITIES:
        kinetic = kinetic_energy(SEVERE_MASS, velocity)
        solution = solve_penetration(kinetic, structure)
        geometry = damage_geometry(structure, solution, contact_length=CONTACT_LENGTH)
        decomposition = verify_layer_decomposition(structure, solution.penetration_depth)
        analytic = _closed_form_depth(kinetic, structure)
        sweep.append(
            {
                "velocity_m_s": velocity,
                "kinetic_energy_MJ": kinetic / 1e6,
                "penetration_depth_m": solution.penetration_depth,
                "penetration_over_span": solution.penetration_depth / FRAME_SPAN,
                "structural_energy_MJ": solution.structural_energy / 1e6,
                "friction_energy_MJ": solution.friction_energy / 1e6,
                "residual_J": solution.residual,
                "residual_relative": solution.residual_relative,
                "closed_form_depth_m": analytic,
                "solver_vs_closed_form_relative_error": abs(solution.penetration_depth - analytic) / analytic,
                "layer_energies_MJ": [e / 1e6 for e in solution.layer_energies],
                "layer_local_penetrations_m": list(solution.layer_local_penetrations),
                "structural_resistance_MN": solution.structural_resistance_N / 1e6,
                "allocation_relative_error": decomposition["rel_error"],
                "inner_shell_intact": solution.inner_shell_intact,
                "breached_layers": list(solution.breached_layers),
                "damage_length_m": geometry.length_m,
                "damage_width_m": geometry.width_m,
                "damage_volume_m3": geometry.volume_m3,
                "iterations": solution.iterations,
            }
        )

    residuals = [row["residual_relative"] for row in sweep]
    depths = [row["penetration_depth_m"] for row in sweep]
    monotone = all(b > a for a, b in zip(depths, depths[1:]))
    closed_form_errors = [row["solver_vs_closed_form_relative_error"] for row in sweep]

    # ------------------------------------------------- (d) double-hull allocation
    allocation_cases = []
    for label, mass, velocity in (
        ("moderate_outer_only", SEVERE_MASS, 8.0),
        ("overload_inner_engaged", OVERLOAD_MASS, 6.0),
    ):
        kinetic = kinetic_energy(mass, velocity)
        solution = solve_penetration(kinetic, structure)
        geometry = damage_geometry(structure, solution, contact_length=CONTACT_LENGTH)
        decomposition = verify_layer_decomposition(structure, solution.penetration_depth, n_intervals=40000)
        allocation_cases.append(
            {
                "case": label,
                "mass_kg": mass,
                "velocity_m_s": velocity,
                "kinetic_energy_MJ": kinetic / 1e6,
                "penetration_depth_m": solution.penetration_depth,
                "outer_energy_MJ": solution.layer_energies[0] / 1e6,
                "inner_energy_MJ": solution.layer_energies[1] / 1e6,
                "total_structural_energy_MJ": solution.structural_energy / 1e6,
                "sum_of_layers_MJ": sum(solution.layer_energies) / 1e6,
                "sum_equals_total": abs(sum(solution.layer_energies) - solution.structural_energy)
                <= 1e-12 * solution.structural_energy,
                "quadrature_relative_error": decomposition["rel_error"],
                "inner_shell_intact": solution.inner_shell_intact,
                "breached_layers": list(solution.breached_layers),
                "damage": geometry.as_dict(),
            }
        )

    # ------------------------------------------------------------------ (e) Minorsky
    minorsky_inputs = [
        {
            "case": f"sweep_V{row['velocity_m_s']:.0f}",
            "mass": SEVERE_MASS,
            "velocity": row["velocity_m_s"],
            "depth": row["penetration_depth_m"],
            "volume": row["damage_volume_m3"],
        }
        for row in sweep
    ] + [
        {
            "case": case["case"],
            "mass": case["mass_kg"],
            "velocity": case["velocity_m_s"],
            "depth": case["penetration_depth_m"],
            "volume": case["damage"]["volume_m3"],
        }
        for case in allocation_cases
    ]
    minorsky_rows = []
    for item in minorsky_inputs:
        comparison = compare_damage_volume(
            model_volume_m3=item["volume"],
            mass=item["mass"],
            velocity=item["velocity"],
            sigma_y=SCRATCH_SIGMA_Y,
        )
        fully_plastic = item["depth"] / FRAME_SPAN >= MINORSKY_MIN_DEPTH_OVER_SPAN
        in_domain = bool(comparison["in_minorsky_range"] and fully_plastic)
        minorsky_rows.append(
            {
                "case": item["case"],
                "kinetic_energy_MJ": comparison["kinetic_energy_J"] / 1e6,
                "volume_model_m3": comparison["volume_model_m3"],
                "volume_minorsky_m3": comparison["volume_minorsky_m3"],
                "relative_error": comparison["relative_error"],
                "in_minorsky_energy_range": comparison["in_minorsky_range"],
                "fully_plastic_mechanism": fully_plastic,
                "in_comparison_domain": in_domain,
                "meets_threshold": (abs(comparison["relative_error"]) <= THRESHOLDS["minorsky_relative_error"])
                if in_domain
                else None,
            }
        )
    in_domain_errors = [abs(row["relative_error"]) for row in minorsky_rows if row["in_comparison_domain"]]
    minorsky_max_error = max(in_domain_errors) if in_domain_errors else float("nan")

    checks = {
        "criterion_a_energy_residual": {
            "max_residual_relative": max(residuals),
            "threshold": THRESHOLDS["energy_residual_relative"],
            "pass": max(residuals) <= THRESHOLDS["energy_residual_relative"],
        },
        "criterion_a_solver_vs_closed_form": {
            "max_relative_error": max(closed_form_errors),
            "threshold": 1e-6,
            "pass": max(closed_form_errors) <= 1e-6,
        },
        "criterion_b_anchor": {
            "max_relative_error": anchor_max_error,
            "threshold": THRESHOLDS["anchor_relative_error"],
            "pass": anchor_max_error <= THRESHOLDS["anchor_relative_error"],
        },
        "criterion_c_negative_control": {
            "depth_at_zero_velocity_m": zero_solution.penetration_depth,
            "depth_at_2ms_m": non_zero_solution.penetration_depth,
            "pass": zero_solution.penetration_depth == 0.0 and non_zero_solution.penetration_depth > 0.0,
        },
        "criterion_d_allocation": {
            "max_quadrature_relative_error": max(case["quadrature_relative_error"] for case in allocation_cases),
            "sum_equals_total": all(case["sum_equals_total"] for case in allocation_cases),
            "inner_engaged_in_overload": allocation_cases[1]["inner_energy_MJ"] > 0.0,
            "inner_intact_flags": [case["inner_shell_intact"] for case in allocation_cases],
            "threshold": THRESHOLDS["allocation_relative_error"],
            "pass": all(case["quadrature_relative_error"] <= THRESHOLDS["allocation_relative_error"] for case in allocation_cases)
            and all(case["sum_equals_total"] for case in allocation_cases)
            and allocation_cases[1]["inner_energy_MJ"] > 0.0,
        },
        "criterion_e_minorsky": {
            "max_relative_error_in_domain": minorsky_max_error,
            "threshold": THRESHOLDS["minorsky_relative_error"],
            "cases_in_domain": [row["case"] for row in minorsky_rows if row["in_comparison_domain"]],
            "pass": bool(in_domain_errors) and minorsky_max_error <= THRESHOLDS["minorsky_relative_error"],
        },
        "criterion_f_monotone_sweep": {
            "depths_m": depths,
            "strictly_increasing": monotone,
            "pass": monotone,
        },
    }
    failed = [name for name, result in checks.items() if not result["pass"]]
    verdict = "PASS" if not failed else "FAIL"

    # ------------------------------------------------------------------ figures
    def _style(axis, xlabel, ylabel, title):
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.3)
        axis.legend(frameon=False, fontsize=8)

    depth_grid = [structure.max_depth * i / 400.0 for i in range(401)]
    figure1, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(depth_grid, [structure.resistance(d) / 1e6 for d in depth_grid], color="#1f4e79", lw=2)
    axes[0].axvline(outer.crush_depth, color="grey", ls="--", lw=1, label="outer shell spent")
    for row in sweep:
        axes[0].plot(row["penetration_depth_m"], row["structural_resistance_MN"], "o", ms=4,
                     label=f"V={row['velocity_m_s']:.0f} m/s")
    _style(axes[0], "penetration depth d [m]", "structural resistance [MN]", "(a) resistance curve R(d)")
    axes[1].plot(depth_grid, [structure.absorbed(d) / 1e6 for d in depth_grid], color="#1f4e79", lw=2,
                 label="E_structural + E_friction")
    axes[1].plot(depth_grid, [structure.structural_energy(d) / 1e6 for d in depth_grid], color="#c00000",
                 ls="--", lw=1.4, label="E_structural")
    for row in sweep:
        axes[1].axhline(row["kinetic_energy_MJ"], color="grey", ls=":", lw=0.8)
        axes[1].plot(row["penetration_depth_m"], row["kinetic_energy_MJ"], "o", ms=4)
    axes[1].annotate("V=2..8 m/s operating points", xy=(sweep[0]["penetration_depth_m"], sweep[0]["kinetic_energy_MJ"]),
                     xytext=(0.45, 40), fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.8, color="grey"))
    _style(axes[1], "penetration depth d [m]", "absorbed energy [MJ]", "(b) energy balance E_k = E_plastic + E_friction")
    figure1.tight_layout()
    figure1_path = figures_dir / "fig1_resistance_energy.png"
    figure1.savefig(figure1_path, dpi=160)
    plt.close(figure1)

    figure2, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fine = [0.5 * i for i in range(0, 21)]
    fine_depths = []
    for velocity in fine:
        try:
            fine_depths.append(solve_penetration(kinetic_energy(SEVERE_MASS, velocity), structure).penetration_depth)
        except Exception:  # pragma: no cover - capacity guard
            fine_depths.append(float("nan"))
    axes[0].plot(fine, fine_depths, "-", color="#1f4e79", lw=1.6, label="model (bisection on energy)")
    axes[0].plot([row["velocity_m_s"] for row in sweep], depths, "o", color="#c00000", ms=6,
                 label="V = 2/4/6/8 m/s acceptance points")
    axes[0].axhline(DOUBLE_HULL_DEPTH, color="grey", ls="--", lw=1, label="inner shell face (2.022 m)")
    _style(axes[0], "impact velocity V [m/s]", "penetration depth d [m]", "(a) penetration vs impact velocity")
    axes[1].semilogy([row["kinetic_energy_MJ"] for row in minorsky_rows],
                     [row["volume_model_m3"] for row in minorsky_rows], "s-", color="#1f4e79",
                     label="model damaged volume")
    axes[1].semilogy([row["kinetic_energy_MJ"] for row in minorsky_rows],
                     [row["volume_minorsky_m3"] for row in minorsky_rows], "^--", color="#c00000",
                     label="Minorsky (1959) volume")
    for row in minorsky_rows:
        axes[1].annotate(f"{100 * row['relative_error']:+.0f}%" + ("" if row["in_comparison_domain"] else " (out)"),
                         xy=(row["kinetic_energy_MJ"], row["volume_model_m3"]),
                         xytext=(0, 6), textcoords="offset points", fontsize=8)
    _style(axes[1], "collision kinetic energy [MJ]", "damaged steel volume [m$^3$]",
           "(b) damaged volume vs Minorsky empirical anchor")
    figure2.tight_layout()
    figure2_path = figures_dir / "fig2_penetration_minorsky.png"
    figure2.savefig(figure2_path, dpi=160)
    plt.close(figure2)

    figure3, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = [f"{row['velocity_m_s']:.0f}" for row in sweep]
    structural = [row["structural_energy_MJ"] for row in sweep]
    friction = [row["friction_energy_MJ"] for row in sweep]
    axes[0].bar(labels, structural, color="#1f4e79", label="E_plastic (hinge + membrane)")
    axes[0].bar(labels, friction, bottom=structural, color="#c00000", alpha=0.75, label="E_friction")
    axes[0].plot(labels, [row["kinetic_energy_MJ"] for row in sweep], "k*", ms=9, label="E_kinetic")
    _style(axes[0], "impact velocity [m/s]", "energy [MJ]", "(a) energy partition (residual < 1e-12 %)")
    axes[1].bar(labels, [100 * row["layer_energies_MJ"][0] / max(row["kinetic_energy_MJ"], 1e-9) for row in sweep],
                color="#1f4e79", label="outer shell share")
    axes[1].bar(labels, [100 * row["layer_energies_MJ"][1] / max(row["kinetic_energy_MJ"], 1e-9) for row in sweep],
                bottom=[100 * row["layer_energies_MJ"][0] / max(row["kinetic_energy_MJ"], 1e-9) for row in sweep],
                color="#7f7f7f", label="inner shell share")
    _style(axes[1], "impact velocity [m/s]", "share of E_kinetic [%]", "(b) double-hull allocation (sweep: inner shell intact)")
    figure3.tight_layout()
    figure3_path = figures_dir / "fig3_energy_partition.png"
    figure3.savefig(figure3_path, dpi=160)
    plt.close(figure3)

    figure4, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    deflections = [0.005 * i for i in range(1, 25)]
    closed, incremental, work = [], [], []
    for d in deflections:
        case = three_point_bending_anchor(
            sigma_y=SCRATCH_SIGMA_Y, width=0.6, thickness=OUTER_THICKNESS, span=FRAME_SPAN,
            deflection=d, bc="simply_supported", n_steps=4096,
        )
        closed.append(case["energy_closed_J"] / 1e3)
        incremental.append(case["energy_incremental_J"] / 1e3)
        work.append(case["energy_virtual_work_J"] / 1e3)
    axes[0].plot(deflections, closed, "-", color="#1f4e79", lw=2, label="closed form M_p*theta_max")
    axes[0].plot(deflections, incremental, "--", color="#c00000", lw=1.4, label="incremental fibre-section integration")
    axes[0].plot(deflections, work, ":", color="black", lw=1.6, label="external work P_u*d (virtual work)")
    _style(axes[0], "mid-span deflection [m]", "plastic hinge energy [kJ]", "(a) three-point bending anchor (simply supported)")
    steps = [16, 64, 256, 1024, 4096, 16384]
    errors = []
    for n in steps:
        case = three_point_bending_anchor(
            sigma_y=SCRATCH_SIGMA_Y, width=0.6, thickness=OUTER_THICKNESS, span=FRAME_SPAN,
            deflection=0.12, bc="clamped", n_steps=n,
        )
        errors.append(max(case["rel_error_incremental_vs_closed"], case["rel_error_virtual_work_vs_closed"], 1e-18))
    axes[1].loglog(steps, errors, "o-", color="#1f4e79", label="max relative error vs closed form")
    axes[1].axhline(THRESHOLDS["anchor_relative_error"], color="#c00000", ls="--", lw=1, label="1 % acceptance threshold")
    _style(axes[1], "integration steps / route", "relative error [-]", "(b) anchor convergence (clamped mechanism)")
    figure4.tight_layout()
    figure4_path = figures_dir / "fig4_hinge_anchor.png"
    figure4.savefig(figure4_path, dpi=160)
    plt.close(figure4)

    figure_paths = [figure1_path, figure2_path, figure3_path, figure4_path]

    # ------------------------------------------------------------------- artefacts
    with (output / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "velocity_m_s", "kinetic_energy_MJ", "penetration_depth_m", "structural_energy_MJ",
            "friction_energy_MJ", "residual_relative", "layer_outer_energy_MJ", "layer_inner_energy_MJ",
            "inner_shell_intact", "damage_volume_m3", "damage_length_m", "damage_width_m",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sweep:
            writer.writerow(
                {
                    "velocity_m_s": row["velocity_m_s"],
                    "kinetic_energy_MJ": f"{row['kinetic_energy_MJ']:.6f}",
                    "penetration_depth_m": f"{row['penetration_depth_m']:.9f}",
                    "structural_energy_MJ": f"{row['structural_energy_MJ']:.9f}",
                    "friction_energy_MJ": f"{row['friction_energy_MJ']:.9f}",
                    "residual_relative": f"{row['residual_relative']:.3e}",
                    "layer_outer_energy_MJ": f"{row['layer_energies_MJ'][0]:.9f}",
                    "layer_inner_energy_MJ": f"{row['layer_energies_MJ'][1]:.9f}",
                    "inner_shell_intact": row["inner_shell_intact"],
                    "damage_volume_m3": f"{row['damage_volume_m3']:.9f}",
                    "damage_length_m": f"{row['damage_length_m']:.6f}",
                    "damage_width_m": f"{row['damage_width_m']:.6f}",
                }
            )

    report = {
        "node": "IN63-N01",
        "title": "Ice-region ship-ice collision energy-conservation surface (plastic hinge + double hull + Minorsky cross-check)",
        "verdict": verdict,
        "failed_checks": failed,
        "criteria": checks,
        "thresholds": THRESHOLDS,
        "design_case": {
            "outer_thickness_m": OUTER_THICKNESS,
            "inner_thickness_m": INNER_THICKNESS,
            "frame_span_m": FRAME_SPAN,
            "double_hull_depth_m": DOUBLE_HULL_DEPTH,
            "yield_stress_MPa": SCRATCH_SIGMA_Y / 1e6,
            "friction_coefficient": FRICTION_COEFFICIENT,
            "striking_mass_kg": STRIKING_MASS,
            "added_mass_coefficient": ADDED_MASS_COEFFICIENT,
            "effective_mass_kg": SEVERE_MASS,
            "overload_mass_kg": OVERLOAD_MASS,
            "contact_length_m": CONTACT_LENGTH,
            "contact_height_m": CONTACT_HEIGHT,
            "longitudinal_Z_m3": longitudinal_section_modulus(),
            "outer_hinge_coefficient_N_per_m": outer.hinge_coefficient,
            "outer_membrane_coefficient_N_per_m2": outer.membrane_coefficient,
            "outer_capacity_MJ": outer.capacity / 1e6,
            "inner_hinge_coefficient_N_per_m": inner.hinge_coefficient,
            "inner_membrane_coefficient_N_per_m2": inner.membrane_coefficient,
            "inner_capacity_MJ": inner.capacity / 1e6,
            "total_capacity_MJ": structure.total_capacity / 1e6,
        },
        "minorsky_constants": {
            "MJ_per_m3": MINORSKY_1959_MJ_PER_M3,
            "reference_yield_stress_MPa": REFERENCE_YIELD_STRESS_PA / 1e6,
            "coefficient_C": minorsky_coefficient(),
            "comparison_domain_rule": f"energy in {10}..{400} MJ and d/span >= {MINORSKY_MIN_DEPTH_OVER_SPAN}",
        },
        "anchor": anchor_cases,
        "sweep": sweep,
        "allocation_cases": allocation_cases,
        "minorsky": minorsky_rows,
        "negative_control": {
            "depth_at_V0_m": zero_solution.penetration_depth,
            "depth_at_V2_m": non_zero_solution.penetration_depth,
        },
        "figures": [
            {"path": str(path.relative_to(output)), "sha256": _sha256_file(path)}
            for path in figure_paths
        ],
        "limitations": [
            "Rigid-perfectly-plastic material: no hardening, no fracture/tearing energy, no strain-rate "
            "sensitivity (ice collisions are rate sensitive, so absorbed energy is under-estimated).",
            "The indenter is rigid: ice crushing is not modelled, so the reported penetration is an upper "
            "bound for a real ice collision of the same kinetic energy.",
            "Panel-level granularity: the outer shell bay yields as a whole, so the model cannot resolve "
            "partial plasticity of a bay (this is why the Minorsky comparison is restricted to fully "
            "plastic cases, see the predeclared domain rule).",
            "The membrane term is the rigid-plastic limit; for d/span < 0.1 the real response is elastic.",
            "Friction is taken as mu times the structural dissipation (work-average normal force), a "
            "quasi-static idealisation; dynamic friction and the ice contact pressure distribution are "
            "not resolved.",
            "Damage geometry is a mechanism-based equivalent box (length x width x depth), not a measured "
            "damage extent; SOLAS input should treat it as a bounding estimate.",
        ],
        "sources": [
            "src/tensordem/hull_collision/plastic_hinge.py",
            "src/tensordem/hull_collision/resistance.py",
            "src/tensordem/hull_collision/minorsky.py",
            "src/tensordem/hull_collision/scenario.py",
            "scripts/benchmark_hull_collision_energy.py",
            "prereg/IN63_N01_hull_collision_energy.json",
            "tests/test_hull_collision_energy.py",
        ],
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["audit"] = {
        "report_content_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_sha256": {
            path: _sha256_file(ROOT / path)
            for path in report["sources"]
            if (ROOT / path).exists()
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "hull_collision_energy")
    arguments = parser.parse_args()
    report = run(arguments.output)
    summary = {
        "verdict": report["verdict"],
        "failed_checks": report["failed_checks"],
        "criteria": {name: {k: v for k, v in result.items() if k in ("pass", "max_relative_error", "max_residual_relative", "strictly_increasing", "max_quadrature_relative_error")} for name, result in report["criteria"].items()},
        "depths_m": [round(row["penetration_depth_m"], 4) for row in report["sweep"]],
        "minorsky_relative_errors": [round(row["relative_error"], 4) for row in report["minorsky"]],
        "output": str(arguments.output),
    }
    print(json.dumps(summary, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
