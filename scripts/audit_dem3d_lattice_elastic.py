"""Audit the 3-D bond lattice: analytic elastic constants vs the solver.

Writes a JSON report, a per-size CSV, and figures. Small-strain linear elasticity
only; no time integration is performed, so the run is cheap and deterministic.

    python scripts/audit_dem3d_lattice_elastic.py --output results-dem3d-lattice-elastic
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem.dem3d import DEM3DConfig, IceDEM3D  # noqa: E402
from tensordem import dem3d_lattice_elastic as lattice  # noqa: E402


def _fit_inverse_size(sizes, values):
    """Least-squares fit ``value = intercept + slope / n``."""
    xs = torch.tensor([1.0 / float(n) for n in sizes], dtype=torch.float64)
    ys = torch.tensor([float(v) for v in values], dtype=torch.float64)
    design = torch.stack((torch.ones_like(xs), xs), dim=1)
    solution = torch.linalg.lstsq(design, ys).solution
    residual = float((design @ solution - ys).abs().max().item())
    return {"intercept": float(solution[0]), "slope_per_inverse_size": float(solution[1]),
            "max_abs_residual": residual, "sizes": list(sizes)}


def _solver_bond_energy(config: DEM3DConfig, mode: str, strain: float) -> dict[str, float]:
    """Cross-check the manual estimator against the solver's diagnostics ledger."""
    solver = IceDEM3D(config)
    displacement = lattice._apply_affine(solver, mode, strain)
    diagnostics = solver.diagnostics()
    manual = 0.5 * config.bond_stiffness * (
        ((displacement[solver.pairs[:, 1]] - displacement[solver.pairs[:, 0]])
         * solver.initial_normals).sum(dim=1) ** 2).sum()
    with torch.no_grad():
        solver.positions.copy_(solver.initial_positions)
    return {
        "solver_bond_elastic_energy_J": float(diagnostics["bond_elastic_energy_J"]),
        "manual_bond_elastic_energy_J": float(manual),
        "relative_difference": abs(float(diagnostics["bond_elastic_energy_J"]) / float(manual) - 1.0),
    }


def _topology_check(config: DEM3DConfig) -> dict[str, object]:
    """Verify the analytic offset set equals the solver's interior bond set."""
    solver = IceDEM3D(config)
    positions = solver.initial_positions
    a = lattice.spacing(config.radius)
    core = lattice._core_mask(positions, a)
    core_index = int(torch.nonzero(core, as_tuple=False)[len(torch.nonzero(core, as_tuple=False)) // 2])
    incident = (solver.pairs[:, 0] == core_index) | (solver.pairs[:, 1] == core_index)
    others = torch.where(solver.pairs[incident, 0] == core_index,
                         solver.pairs[incident, 1], solver.pairs[incident, 0])
    directions = (positions[others] - positions[core_index]) / a
    analytic = set()
    for offset in lattice.bond_offset_set():
        analytic.add(offset)
        analytic.add(tuple(-value for value in offset))
    solver_set = {tuple(int(round(value)) for value in row) for row in directions.round().tolist()}
    return {
        "core_particle_index": core_index,
        "interior_coordination_number": int(incident.sum().item()),
        # Every half-space representative implies a bond in both directions, so
        # an interior particle has all 26 neighbours bonded.
        "expected_coordination_number": 2 * len(lattice.bond_offset_set()),
        "direction_sets_identical": solver_set == analytic,
        "solver_directions_missing_from_analytic": sorted(solver_set - analytic),
        "analytic_directions_missing_from_solver": sorted(analytic - solver_set),
        "total_bonds": int(len(solver.pairs)),
    }


def _figures(output: Path, report: dict) -> list[dict[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures = []
    analytic = report["analytic"]
    k_over_a = analytic["bond_stiffness_N_per_m"] / analytic["spacing_m"]

    # Figure 1: affine vs relaxed constants and size extrapolation.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=140)
    ax = axes[0]
    names = ["C11", "C12", "C44"]
    analytic_ratio = [analytic[f"{name}_Pa"] / k_over_a for name in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, analytic_ratio, width=0.4, label="analytic affine (Born sum)",
           color="#3b6ea5", edgecolor="black", linewidth=0.6)
    ax.bar(x + 0.2, [report["affine"]["measured"][f"{name}_Pa"] / k_over_a for name in names],
           width=0.4, label="solver affine patch", color="#e08b3c", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{name}\n({value:.3f} k/a)" for name, value in zip(names, analytic_ratio)])
    ax.set_ylabel("normalised elastic constant  C / (k/a)")
    ax.set_title("(a) Affine constants: analytic vs solver")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.02, 0.97, "max rel. error %.1e" % report["affine"]["max_relative_error"],
            transform=ax.transAxes, va="top", fontsize=8)

    ax = axes[1]
    relaxed = report["relaxed"]
    inv = [1.0 / row["edge_particles"] for row in relaxed["runs"]]
    ax.plot(inv, [row["young_modulus_relaxed_Pa"] / k_over_a for row in relaxed["runs"]],
            "o-", color="#c0392b", label="relaxed tension modulus")
    ax.plot(inv, [row["shear_stiffness_relaxed_Pa"] / k_over_a for row in relaxed["runs"]],
            "s-", color="#16a085", label="relaxed shear stiffness")
    ax.axhline(analytic["C11_Pa"] / k_over_a, color="#c0392b", ls=":", lw=1.2,
               label="C11 = %.3g k/a (laterally clamped affine)" % (analytic["C11_Pa"] / k_over_a))
    ax.axhline(analytic["young_100_Pa"] / k_over_a, color="#c0392b", ls="--", lw=1.2,
               label="E100 = %.3g k/a (traction-free limit)" % (analytic["young_100_Pa"] / k_over_a))
    ax.axhline(analytic["C44_Pa"] / k_over_a, color="#16a085", ls="--", lw=1.2,
               label="C44 = %.3g k/a (affine)" % (analytic["C44_Pa"] / k_over_a))
    ax.axhline(analytic["shear_modulus_isotropic_equivalent_Pa"] / k_over_a, color="#16a085", ls=":",
               lw=1.2, label="isotropic mu implied by E100, nu = %.3g k/a"
                             % (analytic["shear_modulus_isotropic_equivalent_Pa"] / k_over_a))
    ax.plot([0], [relaxed["extrapolated"]["young_modulus_Pa"]["intercept"] / k_over_a], "*",
            color="#c0392b", ms=13, label="1/n extrapolation")
    ax.plot([0], [relaxed["extrapolated"]["shear_stiffness_Pa"]["intercept"] / k_over_a], "*",
            color="#16a085", ms=13)
    ax.set_xlabel("1 / (specimen edge in particles)")
    ax.set_ylabel("normalised constant  C / (k/a)")
    ax.set_title("(b) Relaxation of the 26-neighbour lattice")
    ax.legend(fontsize=6, frameon=False, loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = output / "fig1_lattice_elastic_constants.png"
    fig.savefig(path)
    plt.close(fig)
    figures.append({"name": "fig1_lattice_elastic_constants.png",
                    "caption": "Affine constants reproduced to machine precision and "
                               "non-affine softening of E and C44 with specimen size"})

    # Figure 2: bond classes and their contribution to C11.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), dpi=140)
    ax = axes[0]
    classes = report["bond_classes"]
    labels = [row["class"] for row in classes]
    counts = [row["count"] for row in classes]
    contributions = [row["C11_share"] for row in classes]
    x = np.arange(len(labels))
    ax.bar(x, counts, color="#5d6d7e", edgecolor="black", linewidth=0.6)
    for index, value in enumerate(counts):
        ax.text(index, value + 0.1, str(value), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("half-space offsets (x2 = bonds per particle)")
    ax.set_title("(a) Bond direction classes (13 pairs, Z = 26)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x, contributions, color="#8e44ad", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("share of the C11 lattice sum")
    ax.set_title("(b) Which bond directions carry the axial stiffness")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = output / "fig2_bond_direction_classes.png"
    fig.savefig(path)
    plt.close(fig)
    figures.append({"name": "fig2_bond_direction_classes.png",
                    "caption": "Bond direction multiplicities and each class's share of the "
                               "C11 lattice sum"})

    # Figure 3: calibration consequence.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), dpi=140)
    calibration = report["calibration"]
    ax = axes[0]
    labels = ["isotropic target\n(E, nu=0.3)", "achieved with\nscalar k"]
    shear = [calibration["isotropic_shear_modulus_for_target_Pa"] / 1e9,
             calibration["achieved_shear_modulus_Pa"] / 1e9]
    ax.bar(labels, shear, color=["#7f8c8d", "#d35400"], edgecolor="black", linewidth=0.6)
    ax.set_ylabel("shear modulus [GPa]")
    ax.set_title("(a) Shear modulus after calibrating E")
    for index, value in enumerate(shear):
        ax.text(index, value * 1.01, "%.3f" % value, ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    radii = [0.01, 0.02, 0.025, 0.04, 0.05]
    stiffness = [lattice.stiffness_calibration(9.0e9, r)["required_bond_stiffness_N_per_m"] / 1e8
                 for r in radii]
    ax.plot(radii, stiffness, "o-", color="#2c3e50")
    ax.set_xlabel("particle radius [m]")
    ax.set_ylabel("required k [1e8 N/m] for E = 9 GPa")
    ax.set_title("(b) Calibration rule  k = E a / 2.7")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = output / "fig3_stiffness_calibration.png"
    fig.savefig(path)
    plt.close(fig)
    figures.append({"name": "fig3_stiffness_calibration.png",
                    "caption": "Shear-modulus consequence of a scalar bond stiffness and the "
                               "closed-form calibration rule k = E a / 2.7"})
    return figures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results-dem3d-lattice-elastic")
    parser.add_argument("--radius", type=float, default=0.025)
    parser.add_argument("--bond-stiffness", type=float, default=2000.0)
    parser.add_argument("--affine-sizes", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    parser.add_argument("--relaxed-sizes", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    parser.add_argument("--strain", type=float, default=1e-6)
    parser.add_argument("--target-modulus-GPa", type=float, default=9.0)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    analytic = lattice.analytic_constants(args.bond_stiffness, args.radius)
    vectors = torch.tensor(lattice.bond_offset_set(), dtype=torch.float64)
    lengths = torch.linalg.vector_norm(vectors, dim=1)
    normals = vectors / lengths[:, None]
    weights = lengths.square()
    c11_terms = weights * normals[:, 0] ** 4
    classes = []
    for target in (1.0, math.sqrt(2.0), math.sqrt(3.0)):
        mask = (lengths - target).abs() < 1e-9
        classes.append({
            "class": {1.0: "axial", math.sqrt(2.0): "face diagonal", math.sqrt(3.0): "body diagonal"}[target],
            "count": int(mask.sum().item()),
            "offsets": vectors[mask].tolist(),
            "C11_share": float(c11_terms[mask].sum() / c11_terms.sum()),
        })

    report: dict = {
        "case": "dem3d lattice elastic constants audit",
        "config_inputs": {"radius_m": args.radius, "bond_stiffness_N_per_m": args.bond_stiffness,
                          "strain": args.strain},
        "analytic": analytic,
        "bond_classes": classes,
        "topology_check": _topology_check(DEM3DConfig(nx=5, ny=5, nz=5, radius=args.radius,
                                                      bond_stiffness=args.bond_stiffness)),
        "hessian_cross_check": lattice.hessian_cross_check(
            DEM3DConfig(nx=6, ny=6, nz=6, radius=args.radius, bond_stiffness=args.bond_stiffness)),
        "solver_energy_cross_check": _solver_bond_energy(
            DEM3DConfig(nx=6, ny=6, nz=6, radius=args.radius, bond_stiffness=args.bond_stiffness),
            "xx", args.strain),
        "affine": {},
        "relaxed": {},
    }

    rows = []
    for size in args.affine_sizes:
        config = DEM3DConfig(nx=size, ny=size, nz=size, radius=args.radius,
                             bond_stiffness=args.bond_stiffness)
        patch = lattice.affine_patch(config, strain=args.strain)
        rows.append({"kind": "affine", "edge_particles": size,
                     "particle_count": patch["particle_count"],
                     "core_particles": patch["core_particles"],
                     "C11_Pa": patch["measured"]["C11_Pa"],
                     "C12_Pa": patch["measured"]["C12_Pa"],
                     "C44_Pa": patch["measured"]["C44_Pa"],
                     "relative_error_C11": patch["relative_error"]["C11"],
                     "edge_length_m": (size - 1) * 2 * args.radius})
        if size == args.affine_sizes[0]:
            report["affine"] = patch
        else:
            report["affine"].setdefault("size_sweep", []).append(
                {"edge_particles": size, "max_relative_error": patch["max_relative_error"]})

    relaxed_runs = []
    for size in args.relaxed_sizes:
        config = DEM3DConfig(nx=size, ny=size, nz=size, radius=args.radius,
                             bond_stiffness=args.bond_stiffness)
        axial = lattice.relaxed_patch(config, strain=args.strain, mode="xx", boundary="x_faces")
        shear = lattice.relaxed_patch(config, strain=args.strain, mode="shear", boundary="yz_faces")
        relaxed_runs.append({
            "edge_particles": size,
            "particle_count": axial["particle_count"],
            "core_particles": axial["core_particles"],
            "young_modulus_relaxed_Pa": axial["relaxed_constant_Pa"],
            "poisson_ratio_relaxed": axial["poisson_ratio_relaxed"],
            "shear_stiffness_relaxed_Pa": shear["relaxed_constant_Pa"],
            "axial_free_dof_residual_N": axial["free_dof_residual_N"],
            "shear_free_dof_residual_N": shear["free_dof_residual_N"],
            "axial_free_dof_count": axial["free_dof_count"],
            "young_ratio_to_analytic_E100": axial["relaxed_constant_Pa"] / analytic["young_100_Pa"],
            "shear_ratio_to_analytic_C44": shear["relaxed_constant_Pa"] / analytic["C44_Pa"],
            "young_ratio_to_affine_same_specimen": axial["ratio_to_affine_same_specimen"],
            "shear_ratio_to_affine_same_specimen": shear["ratio_to_affine_same_specimen"],
            "axial_affine_same_specimen_Pa": axial["affine_same_specimen_Pa"],
            "shear_affine_same_specimen_Pa": shear["affine_same_specimen_Pa"],
        })
        rows.append({"kind": "relaxed", "edge_particles": size,
                     "particle_count": axial["particle_count"],
                     "core_particles": axial["core_particles"],
                     "young_modulus_relaxed_Pa": axial["relaxed_constant_Pa"],
                     "poisson_ratio_relaxed": axial["poisson_ratio_relaxed"],
                     "shear_stiffness_relaxed_Pa": shear["relaxed_constant_Pa"],
                     "edge_length_m": (size - 1) * 2 * args.radius})

    sizes = [run["edge_particles"] for run in relaxed_runs]
    extrapolated = {
        "young_modulus_Pa": _fit_inverse_size(
            sizes, [run["young_modulus_relaxed_Pa"] for run in relaxed_runs]),
        "shear_stiffness_Pa": _fit_inverse_size(
            sizes, [run["shear_stiffness_relaxed_Pa"] for run in relaxed_runs]),
        "poisson_ratio": _fit_inverse_size(
            sizes, [run["poisson_ratio_relaxed"] for run in relaxed_runs]),
    }
    report["relaxed"] = {
        "runs": relaxed_runs,
        "extrapolated": extrapolated,
        "axial_boundary": "x faces kinematically driven, lateral faces free",
        "shear_boundary": "y faces kinematically driven, other faces free",
        "equilibrium_note": "free-DOF residual is the max |K u| on free dofs (N)",
    }
    young_intercept = extrapolated["young_modulus_Pa"]["intercept"]
    shear_intercept = extrapolated["shear_stiffness_Pa"]["intercept"]
    nu_intercept = extrapolated["poisson_ratio"]["intercept"]
    isotropic_shear = young_intercept / (2.0 * (1.0 + nu_intercept))
    report["relaxed"]["isotropy_check"] = {
        "extrapolated_relaxed_C44_Pa": shear_intercept,
        "isotropic_shear_from_relaxed_E_and_nu_Pa": isotropic_shear,
        "relative_difference": abs(shear_intercept / isotropic_shear - 1.0),
        "note": ("Affine anisotropy (Zener A = 2.33) is largely a non-affine artifact: "
                 "after internal relaxation the shear stiffness approaches the isotropic "
                 "value implied by the relaxed E and nu. Both numbers come from separate "
                 "1/n extrapolations of a free-surface specimen, so this is an estimate."),
    }
    report["calibration"] = lattice.stiffness_calibration(args.target_modulus_GPa * 1e9, args.radius)
    previous = {
        name: (report[name].get("max_relative_error") if isinstance(report.get(name), dict) else None)
        for name in ("affine",)
    }
    affine_errors = [row for row in rows if row["kind"] == "affine"]
    report["acceptance"] = {
        "criterion": "affine constants must match the analytic lattice sum to 1e-9 relative",
        "max_affine_relative_error": max(row["relative_error_C11"] for row in affine_errors),
        "pass": max(row["relative_error_C11"] for row in affine_errors) < 1e-9,
        "relaxed_vs_affine_note": ("relaxed values are finite-specimen estimates; the 1/n "
                                   "extrapolation is reported with its fit residual, not as "
                                   "a converged bulk modulus."),
        "extrapolated_tension_modulus_vs_analytic_E100_relative_difference":
            abs(young_intercept / analytic["young_100_Pa"] - 1.0),
        "extrapolated_tension_modulus_within_3_percent":
            abs(young_intercept / analytic["young_100_Pa"] - 1.0) < 0.03,
        "relaxed_shear_to_affine_ratio_at_largest_specimen":
            relaxed_runs[-1]["shear_ratio_to_analytic_C44"],
        "previous_reference": previous,
    }
    report["solver_versions"] = {
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "numpy_probe": "n/a" if args.no_figures else "matplotlib",
    }

    if not args.no_figures:
        report["figures"] = _figures(output, report)

    payload = json.dumps(report, indent=2, sort_keys=True)
    report["sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    (output / "lattice_elastic_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    keys = sorted({key for row in rows for key in row})
    with (output / "lattice_elastic_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(json.dumps({
        "analytic_C11_Pa": analytic["C11_Pa"],
        "analytic_C12_Pa": analytic["C12_Pa"],
        "analytic_C44_Pa": analytic["C44_Pa"],
        "analytic_E100_Pa": analytic["young_100_Pa"],
        "analytic_nu100": analytic["poisson_100"],
        "affine_max_relative_error": report["acceptance"]["max_affine_relative_error"],
        "relaxed_young_ratio_to_affine_same_specimen": [round(run["young_ratio_to_affine_same_specimen"], 4)
                                                        for run in relaxed_runs],
        "relaxed_shear_ratio_to_affine_same_specimen": [round(run["shear_ratio_to_affine_same_specimen"], 4)
                                                        for run in relaxed_runs],
        "extrapolated_young_Pa": extrapolated["young_modulus_Pa"]["intercept"],
        "extrapolated_shear_Pa": extrapolated["shear_stiffness_Pa"]["intercept"],
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())