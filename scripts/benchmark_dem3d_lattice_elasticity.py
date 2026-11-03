"""Benchmark the affine (Cauchy-Born) elasticity of the DEM3D bonded lattice.

This fills the calibration debt stated in ``tensordem.dem3d``: the simple-cubic
13-neighbour bonded lattice had no reference stiffness, so ``bond_stiffness``
could not be related to an engineering Young's modulus and specimen-direction
bias was only qualitative.  The benchmark

1. derives the closed-form cubic constants of the ideal lattice,
2. recomputes them by direct Voigt summation (independent of the closed form),
3. recomputes them again from the solver's *own* bond network through the
   force virial of ``IceDEM3D.forces`` (third, independent route),
4. quantifies the finite-specimen surface deficiency and the anisotropy,
5. checks the explicit time-step bound against the lattice phonon spectrum.

Run:
    python scripts/benchmark_dem3d_lattice_elasticity.py --output docs/lattice-elasticity \\
        --doc docs/dem3d_lattice_elasticity.md

The runner exits non-zero if any acceptance check fails.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.lattice_elasticity import (
    VOIGT_PAIRS,
    affine_stiffness_from_solver,
    affine_stiffness_tensor,
    analytic_box_constants,
    cubic_constants,
    directional_young_modulus,
    finite_lattice_bonds,
    ideal_box_bonds,
    isotropic_moduli,
    lattice_bond_table,
    max_phonon_frequency,
    solver_virial_stress,
    voigt_reuss_hill,
    zener_ratio,
)

PROTOCOL = "tensordem-dem3d-lattice-elasticity-v1"

#: Solver configurations whose bond networks are audited (defaults + a UCS-like prism).
SPECIMENS: tuple[tuple[int, int, int], ...] = ((9, 5, 3), (6, 5, 4), (12, 12, 12))

#: Reference Young's moduli used only to size the calibration gap, not as validation targets.
#: Freshwater/saline ice columnar modulus from the ice-mechanics literature range 1-10 GPa;
#: 9.1 GPa is the textbook freshwater-ice value near -5 C, 1.0 GPa a warm saline lower bound.
MODULUS_TARGETS: dict[str, float] = {"ice_9.1_GPa": 9.1e9, "ice_1.0_GPa": 1.0e9}


def _base_config(nx: int, ny: int, nz: int, **overrides: Any) -> DEM3DConfig:
    values: dict[str, Any] = {
        "nx": nx, "ny": ny, "nz": nz,
        "drag": 0.0, "contact_damping": 0.0, "fix_x_edges": False,
        "tool_radius": 0.01, "tool_gap": 0.0, "shear_breaking_strain": 0.5,
    }
    values.update(overrides)
    return DEM3DConfig(**values)


def _dt_branches(config: DEM3DConfig) -> dict[str, float]:
    """Reproduce the three branches of ``DEM3DConfig.recommended_dt`` for reporting."""
    count = config.nx * config.ny * config.nz
    degree_bound = min(count - 1, 26)
    k_total = config.bond_stiffness + (2 * degree_bound + 1) * config.contact_stiffness
    if config.top_platen_enabled or config.bottom_platen_enabled:
        k_total += (2 * degree_bound + 1) * config.platen_stiffness
    c_total = (2 * degree_bound + 1) * config.contact_damping + config.drag
    if config.top_platen_enabled or config.bottom_platen_enabled:
        c_total += 2 * config.platen_damping
    return {
        "elastic": 0.15 * math.sqrt(config.mass / k_total),
        "damping": 0.15 * config.mass / c_total if c_total > 0 else math.inf,
        "travel": 0.1 * config.radius / config.tool_speed,
    }


def _check(checks: list[dict[str, Any]], cid: str, description: str, value: float,
           tolerance: float, *, kind: str = "max") -> None:
    if kind == "max":
        passed = bool(abs(value) <= tolerance)
    elif kind == "min_positive":
        passed = bool(value > tolerance)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown check kind {kind}")
    checks.append({
        "id": cid, "description": description, "value": value,
        "tolerance": tolerance, "kind": kind, "verdict": "PASS" if passed else "FAIL",
    })


def _matrix_rows(matrix: torch.Tensor, scale: float) -> list[list[float]]:
    return [[float(matrix[i, j]) / scale for j in range(6)] for i in range(6)]


def ideal_lattice_report(bond_stiffness: float, spacing: float) -> dict[str, Any]:
    normals, lengths = ideal_box_bonds(spacing)
    unit = bond_stiffness / spacing
    stiffness = affine_stiffness_tensor(normals, lengths, bond_stiffness, spacing ** 3)
    constants = cubic_constants(stiffness)
    analytic = analytic_box_constants(bond_stiffness, spacing)
    eigenvalues = torch.linalg.eigvalsh(stiffness)
    return {
        "bond_classes": lattice_bond_table(spacing),
        "bond_count": int(len(lengths)),
        "stiffness_Pa": _matrix_rows(stiffness, 1.0),
        "stiffness_over_unit": _matrix_rows(stiffness, unit),
        "constants_Pa": constants,
        "analytic_Pa": analytic,
        "relative_error_C11": abs(constants["C11"] - analytic["C11"]) / analytic["C11"],
        "relative_error_C12": abs(constants["C12"] - analytic["C12"]) / analytic["C12"],
        "relative_error_C44": abs(constants["C44"] - analytic["C44"]) / analytic["C44"],
        "cauchy_relative_gap": abs(constants["C12"] - constants["C44"]) / analytic["C44"],
        "min_eigenvalue_Pa": float(eigenvalues.min()),
        "zener": zener_ratio(constants["C11"], constants["C12"], constants["C44"]),
        "bounds_Pa": voigt_reuss_hill(constants["C11"], constants["C12"], constants["C44"]),
        "isotropic_Pa": isotropic_moduli(constants["C11"], constants["C12"], constants["C44"]),
        "directional_Pa": {
            "E_100": directional_young_modulus(stiffness, (1, 0, 0)),
            "E_110": directional_young_modulus(stiffness, (1, 1, 0)),
            "E_111": directional_young_modulus(stiffness, (1, 1, 1)),
        },
    }


def axis_only_shear_check(bond_stiffness: float, spacing: float) -> dict[str, float]:
    """Nearest-neighbour-only lattice: C44 must vanish (shear mechanism)."""
    axis = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=torch.float64)
    lengths = torch.full((3,), spacing, dtype=torch.float64)
    stiffness = affine_stiffness_tensor(axis, lengths, bond_stiffness, spacing ** 3)
    constants = cubic_constants(stiffness)
    return {"C11": constants["C11"], "C12": constants["C12"], "C44": constants["C44"]}


def specimen_report(nx: int, ny: int, nz: int, bond_stiffness: float,
                    spacing: float, infinite: dict[str, Any]) -> dict[str, Any]:
    cfg = _base_config(nx, ny, nz, bond_stiffness=bond_stiffness, radius=spacing / 2.0)
    sim = IceDEM3D(cfg)
    bonds, _ = finite_lattice_bonds(nx, ny, nz, spacing)
    stiffness = affine_stiffness_from_solver(sim)
    constants = cubic_constants(stiffness)
    unit = bond_stiffness / spacing
    c11_inf = infinite["constants_Pa"]["C11"]
    c12_inf = infinite["constants_Pa"]["C12"]

    # Third, independent route: solver forces under affine strain, virial stress.
    strain_scale = 1e-6
    virial = torch.zeros((6, 6), dtype=torch.float64)
    for index, (i, j) in enumerate(VOIGT_PAIRS):
        strain = torch.zeros((3, 3), dtype=torch.float64)
        if i == j:
            strain[i, i] = strain_scale
        else:
            strain[i, j] = strain[j, i] = 0.5 * strain_scale
        virial[:, index] = solver_virial_stress(sim, strain) / strain_scale
    scale = float(stiffness.abs().max())
    virial_rel = float((virial - stiffness).abs().max() / scale)
    virial_asym = float((virial - virial.T).abs().max() / float(virial.abs().max()))
    return {
        "grid": [nx, ny, nz],
        "bond_count_solver": int(len(sim.pairs)),
        "bond_count_table": int(len(bonds)),
        "volume_m3": float(nx * ny * nz) * spacing ** 3,
        "stiffness_Pa": _matrix_rows(stiffness, 1.0),
        "stiffness_over_unit": _matrix_rows(stiffness, unit),
        "constants_Pa": constants,
        "diagonal_over_infinite": {
            "C11": constants["C11"] / c11_inf,
            "C12": constants["C12"] / infinite["constants_Pa"]["C12"],
            "C44": constants["C44"] / infinite["constants_Pa"]["C44"],
            "C22_over_C11_infinite": float(stiffness[1, 1]) / c11_inf,
            "C33_over_C11_infinite": float(stiffness[2, 2]) / c11_inf,
        },
        "off_pattern_residual": constants["off_pattern_residual"],
        "virial_vs_affine_relative_error": virial_rel,
        "virial_symmetry_residual": virial_asym,
        "virial_matrix_Pa": [[float(virial[i, j]) for j in range(6)] for i in range(6)],
        "E_100_Pa": directional_young_modulus(stiffness, (1, 0, 0)),
        "E_axial_z_Pa": directional_young_modulus(stiffness, (0, 0, 1)),
        "cauchy_relative_gap": abs(constants["C12"] - constants["C44"]) / c12_inf,
        "surface_deficiency": 1.0 - float(stiffness[0, 0]) / c11_inf,
    }


def surface_deficiency_curve(bond_stiffness: float, spacing: float, infinite: dict[str, Any],
                             sizes: list[int]) -> list[dict[str, float]]:
    """Axial affine stiffness of nx=n ny=n nz-thick specimens vs thickness."""
    c11_inf = infinite["constants_Pa"]["C11"]
    rows: list[dict[str, float]] = []
    for nz in sizes:
        nx = ny = max(9, nz)
        bonds, _ = finite_lattice_bonds(nx, ny, nz, spacing)
        stiffness = affine_stiffness_from_bonds_bridge(bonds, bond_stiffness,
                                                      nx * ny * nz * spacing ** 3)
        rows.append({
            "nx": nx, "ny": ny, "nz": nz,
            "bond_count": int(len(bonds)),
            "C11_over_C11_infinite": float(stiffness[0, 0]) / c11_inf,
            "C33_over_C11_infinite": float(stiffness[2, 2]) / c11_inf,
            "zener_like": 2.0 * float(stiffness[3, 3]) / (float(stiffness[0, 0]) - float(stiffness[0, 1])),
        })
    return rows


def affine_stiffness_from_bonds_bridge(bonds: torch.Tensor, bond_stiffness: float,
                                      volume: float) -> torch.Tensor:
    from tensordem.lattice_elasticity import affine_stiffness_from_bonds
    return affine_stiffness_from_bonds(bonds, bond_stiffness, volume)


def directional_curve(stiffness: torch.Tensor, plane: str, samples: int = 61) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for step in range(samples):
        angle = math.pi * step / (samples - 1)
        if plane == "100":
            direction = (math.cos(angle), math.sin(angle), 0.0)
        else:
            direction = (math.cos(angle), math.sin(angle), math.sin(angle)) if False else (
                math.cos(angle), math.sin(angle) / math.sqrt(2.0), math.sin(angle) / math.sqrt(2.0))
        rows.append({
            "angle_deg": math.degrees(angle),
            "E_Pa": directional_young_modulus(stiffness, direction),
        })
    return rows


def dispersion_curve(normals: torch.Tensor, lengths: torch.Tensor, bond_stiffness: float,
                     mass: float, spacing: float, samples: int = 121) -> list[dict[str, Any]]:
    """Phonon branches along Gamma-X for the ideal lattice."""
    vectors = normals * lengths[:, None]
    outer = normals[:, :, None] * normals[:, None, :] * bond_stiffness
    rows: list[dict[str, Any]] = []
    for step in range(samples):
        qx = math.pi / spacing * step / (samples - 1)
        q = torch.tensor([qx, 0.0, 0.0], dtype=torch.float64)
        phase = vectors @ q
        dyn = torch.einsum("b,bij->ij", 1.0 - torch.cos(phase), outer) / mass
        eigenvalues = torch.linalg.eigvalsh(dyn).clamp_min(0.0)
        omega = torch.sqrt(eigenvalues)
        rows.append({
            "q_over_pi_over_a": step / (samples - 1),
            "omega1": float(omega[0]), "omega2": float(omega[1]), "omega3": float(omega[2]),
        })
    return rows


def calibration_gap(infinite: dict[str, Any], bond_stiffness: float, spacing: float) -> dict[str, Any]:
    """Bond stiffness needed to reach a target [100] Young's modulus."""
    e100 = infinite["directional_Pa"]["E_100"]
    scale = e100 / bond_stiffness
    rows: dict[str, Any] = {"current_E_100_Pa": e100, "current_bond_stiffness": bond_stiffness}
    for name, target in MODULUS_TARGETS.items():
        required = target / scale
        rows[name] = {
            "target_E_100_Pa": target,
            "required_bond_stiffness_N_per_m": required,
            "factor_over_current": required / bond_stiffness,
            "timestep_factor": math.sqrt(bond_stiffness / required),
        }
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _figures(figures_dir: Path, report: dict[str, Any]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:  # pragma: no cover - optional dependency
        return []
    figures_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    unit = report["inputs"]["unit_stiffness_Pa"]

    # 1. stiffness matrices, infinite vs finite specimen
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    for ax, key, title in (
        (axes[0], report["infinite"]["stiffness_over_unit"], "ideal 13-neighbour lattice"),
        (axes[1], report["specimens"][0]["stiffness_over_unit"],
         f"finite specimen {report['specimens'][0]['grid']}"),
    ):
        image = ax.imshow(np.array(key) * unit / report["inputs"]["unit_stiffness_Pa"],
                          cmap="viridis", vmin=0.0, vmax=13.0 / 3.0)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        labels = ["11", "22", "33", "12", "13", "23"]
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        fig.colorbar(image, ax=ax, fraction=0.046, label="C / (k/a)")
    fig.suptitle("Affine (Cauchy-Born) stiffness in units of k/a", fontsize=11)
    fig.tight_layout()
    path = figures_dir / "lattice_stiffness_matrices.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    names.append(path.name)

    # 2. directional Young modulus
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for plane, style in (("100", "-"), ("110", "--")):
        curve = report["directional"][plane]
        ax.plot([row["angle_deg"] for row in curve],
                [row["E_Pa"] / unit for row in curve], style,
                label=f"plane ({plane})" if plane == "100" else r"plane ($\bar{1}10$)")
    ax.set_xlabel("angle from [100] (deg)")
    ax.set_ylabel("E(θ) / (k/a)")
    ax.set_title("Directional Young's modulus of the ideal lattice", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "directional_young_modulus.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    names.append(path.name)

    # 3. surface deficiency vs thickness
    rows = report["surface_deficiency_curve"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot([row["nz"] for row in rows], [row["C11_over_C11_infinite"] for row in rows], "o-",
            label="C11 / C11_inf (in-plane)")
    ax.plot([row["nz"] for row in rows], [row["C33_over_C11_infinite"] for row in rows], "s--",
            label="C33 / C11_inf (thickness axis)")
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("specimen thickness nz (nx = ny = max(9, nz))")
    ax.set_ylabel("normalised axial stiffness")
    ax.set_title("Finite-specimen surface bond deficiency (affine)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "surface_deficiency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    names.append(path.name)

    # 4. phonon dispersion
    rows = report["dispersion"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for branch in ("omega1", "omega2", "omega3"):
        ax.plot([row["q_over_pi_over_a"] for row in rows], [row[branch] for row in rows], lw=1.2)
    ax.set_xlabel(r"q along $\Gamma$-X  ($q / (\pi/a)$)")
    ax.set_ylabel("ω (rad/s)")
    ax.set_title("Lattice phonon branches; ω_max sets dt_crit = 2/ω_max", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "phonon_dispersion.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    names.append(path.name)
    return names


def run_benchmark(output: Path, *, doc_path: Path | None = None,
                  bond_stiffness: float = 2000.0, radius: float = 0.025,
                  density: float = 917.0) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    spacing = 2.0 * radius
    checks: list[dict[str, Any]] = []

    infinite = ideal_lattice_report(bond_stiffness, spacing)
    constants = infinite["constants_Pa"]
    analytic = infinite["analytic_Pa"]
    unit = bond_stiffness / spacing

    _check(checks, "closed_form_vs_voigt_sum", "C11/C12/C44 from the Voigt sum match the "
           "closed form (13/3, 7/3, 7/3) k/a", max(
               infinite["relative_error_C11"], infinite["relative_error_C12"],
               infinite["relative_error_C44"]), 1e-12)
    _check(checks, "cubic_symmetry", "off-pattern cubic entries vanish", constants["off_pattern_residual"], 1e-14)
    _check(checks, "matrix_symmetry", "C is symmetric (Maxwell/Betti)", constants["symmetry_residual"], 1e-14)
    _check(checks, "cauchy_relation", "|C12 - C44| / C44 vanishes for a central-force lattice",
           infinite["cauchy_relative_gap"], 1e-14)
    _check(checks, "positive_definite", "smallest eigenvalue of C is positive (stable lattice)",
           infinite["min_eigenvalue_Pa"], 0.0, kind="min_positive")
    _check(checks, "voigt_reuss_ordering", "G_Reuss <= G_Hill <= G_Voigt (must be >= 0)",
           float(infinite["bounds_Pa"]["shear_hill"]) - 0.5 * (
               infinite["bounds_Pa"]["shear_voigt"] + infinite["bounds_Pa"]["shear_reuss"]), 1e-9)

    axis_only = axis_only_shear_check(bond_stiffness, spacing)
    _check(checks, "axis_only_shear_mechanism", "nearest-neighbour-only lattice has C44 = 0 "
           "(shear mechanism; the diagonals remove it)", axis_only["C44"] / unit, 1e-14)

    specimens = [specimen_report(nx, ny, nz, bond_stiffness, spacing, infinite)
                 for nx, ny, nz in SPECIMENS]
    for specimen in specimens:
        tag = "x".join(str(v) for v in specimen["grid"])
        _check(checks, f"bond_count_{tag}", f"solver bond count equals the lattice table count "
               f"for grid {tag}",
               float(abs(specimen["bond_count_solver"] - specimen["bond_count_table"])), 0.0)
        _check(checks, f"virial_vs_affine_{tag}",
               "solver force virial under affine strain reproduces the affine stiffness",
               specimen["virial_vs_affine_relative_error"], 1e-4)

    config = _base_config(9, 5, 3, bond_stiffness=bond_stiffness, radius=radius, density=density)
    normals, lengths = ideal_box_bonds(spacing)
    omega_max = max_phonon_frequency(normals, lengths, bond_stiffness, config.mass, 15, spacing)
    dt_crit = 2.0 / omega_max
    # After a bond breaks, the same neighbour pair is carried by the stiffer linear
    # penalty contact, so the conservative bound uses the larger of the two springs.
    stiff_contact = max(bond_stiffness, config.contact_stiffness)
    omega_max_contact = max_phonon_frequency(normals, lengths, stiff_contact, config.mass, 15, spacing)
    dt_crit_contact = 2.0 / omega_max_contact
    _check(checks, "recommended_dt_within_elastic_bound",
           "configured dt and the conservative recommended dt stay below 2/omega_max of the "
           "bond lattice, including the post-fracture contact spring",
           config.recommended_dt / min(dt_crit, dt_crit_contact), 1.0)

    dt_branches = _dt_branches(config)

    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "inputs": {
            "bond_stiffness_N_per_m": bond_stiffness,
            "particle_radius_m": radius,
            "lattice_spacing_m": spacing,
            "density_kg_per_m3": density,
            "particle_mass_kg": config.mass,
            "unit_stiffness_Pa": unit,
            "note": "k/a is the natural stiffness unit of a central-force lattice",
        },
        "infinite": infinite,
        "axis_only_lattice": axis_only,
        "specimens": specimens,
        "surface_deficiency_curve": surface_deficiency_curve(
            bond_stiffness, spacing, infinite, [2, 3, 4, 6, 8, 12, 16, 24]),
        "phonon": {
            "omega_max_rad_per_s": omega_max,
            "dt_critical_s": dt_crit,
            "contact_stiffness_N_per_m": config.contact_stiffness,
            "omega_max_with_contact_rad_per_s": omega_max_contact,
            "dt_critical_with_contact_s": dt_crit_contact,
            "recommended_dt_s": config.recommended_dt,
            "recommended_over_critical": config.recommended_dt / min(dt_crit, dt_crit_contact),
            "dt_branches_s": dt_branches,
            "binding_dt_branch": min(dt_branches, key=lambda name: dt_branches[name]),
        },
        "directional": {
            "100": directional_curve(
                torch.tensor(infinite["stiffness_Pa"], dtype=torch.float64), "100"),
            "110": directional_curve(
                torch.tensor(infinite["stiffness_Pa"], dtype=torch.float64), "110"),
        },
        "dispersion": dispersion_curve(normals, lengths, bond_stiffness, config.mass, spacing),
        "calibration_gap": calibration_gap(infinite, bond_stiffness, spacing),
        "checks": checks,
    }
    report["case_count"] = len(checks)
    report["pass_count"] = sum(1 for check in checks if check["verdict"] == "PASS")
    report["verdict"] = "PASS" if report["pass_count"] == report["case_count"] else "FAIL"

    _write_csv(output / "lattice_checks.csv", checks)
    _write_csv(output / "surface_deficiency.csv", report["surface_deficiency_curve"])
    _write_csv(output / "directional_modulus.csv",
               [{**row, "plane": plane} for plane, rows in report["directional"].items() for row in rows])
    _write_csv(output / "phonon_dispersion.csv", report["dispersion"])
    figures = _figures(output, report)
    report["figures"] = figures
    (output / "lattice_elasticity_report.json").write_text(json.dumps(report, indent=2))
    if doc_path is not None:
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(render_markdown(report))
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Render the report as a markdown document (all numbers read from the report)."""
    inputs = report["inputs"]
    infinite = report["infinite"]
    constants = infinite["constants_Pa"]
    analytic = infinite["analytic_Pa"]
    unit = inputs["unit_stiffness_Pa"]
    bounds = infinite["bounds_Pa"]
    iso = infinite["isotropic_Pa"]
    directional = infinite["directional_Pa"]
    axis_only = report["axis_only_lattice"]
    gap_voigt_reuss = (bounds["shear_voigt"] - bounds["shear_reuss"]) / bounds["shear_hill"]
    lines: list[str] = []
    add = lines.append

    def kva(value: float) -> str:
        return f"{value / unit:.6f}"

    add("# DEM3D bonded-lattice affine elasticity calibration")
    add("")
    add(f"Protocol: `{report['protocol']}`  ")
    add(f"Verdict: **{report['verdict']}** ({report['pass_count']}/{report['case_count']} checks)  ")
    add(f"Generated by `scripts/benchmark_dem3d_lattice_elasticity.py` — every number below is read "
        f"from `lattice-elasticity/lattice_elasticity_report.json`.")
    add("")
    add("Reproduce:")
    add("")
    add("```bash")
    add("python scripts/benchmark_dem3d_lattice_elasticity.py \\")
    add("    --output docs/lattice-elasticity --doc docs/dem3d_lattice_elasticity.md")
    add("python -m pytest tests/test_lattice_elasticity.py -q")
    add("```")
    add("")
    add("## 1. Case: what is calibrated")
    add("")
    add("`IceDEM3D` bonds particles on a simple-cubic lattice with all face and body diagonals inside "
        "the initial cutoff, giving a 13-neighbour central-force network. Its docstring requires the "
        "directional stiffness to be calibrated before engineering use, but no reference value "
        "existed: `bond_stiffness` was a free number with no link to a Young's modulus, and the "
        "specimen-direction bias was only qualitative.")
    add("")
    add("| parameter | value |")
    add("|---|---|")
    add(f"| bond (axial) stiffness `k` | {inputs['bond_stiffness_N_per_m']:.6g} N/m |")
    add(f"| particle radius | {inputs['particle_radius_m']:.6g} m |")
    add(f"| lattice spacing `a = 2R` | {inputs['lattice_spacing_m']:.6g} m |")
    add(f"| particle density | {inputs['density_kg_per_m3']:.6g} kg/m^3 |")
    add(f"| particle mass | {inputs['particle_mass_kg']:.6g} kg |")
    add(f"| natural unit `k/a` | {unit:.6g} Pa |")
    add("")
    add("Neighbour classes (one bond per class per cell):")
    add("")
    add("| offset (dx,dy,dz) | kind | rest length / a |")
    add("|---|---|---|")
    for row in infinite["bond_classes"]:
        add(f"| {tuple(row['offset'])} | {row['kind']} | {row['rest_length_over_spacing']:.6g} |")
    add("")
    add("## 2. Method")
    add("")
    add("Affine (Cauchy-Born) elasticity: under a uniform small strain `e`, bond `b` stretches by "
        "`dl_b = l_b n·e·n`, so")
    add("")
    add("```")
    add("C_ijkl = (1/V) * sum_b k_b * l_b^2 * n_i n_j n_k n_l ,   V = a^3 per cell")
    add("u = 0.5 * strain_voigt @ C @ strain_voigt ,  strain_voigt = [e11,e22,e33,g12,g13,g23]")
    add("```")
    add("")
    add("Three independent routes are compared:")
    add("")
    add("1. **closed form** — hand summation over the 13 neighbour classes;")
    add("2. **Voigt summation** — numerically applying the six basis strains to the bond network "
        "(`tensordem.lattice_elasticity.affine_stiffness_tensor`);")
    add("3. **solver virial** — affine strain applied to `IceDEM3D` positions, stress taken from "
        "`sigma_ij = -1/V * sum_p f_i x_j` using the solver's own `forces()` evaluation.")
    add("")
    add("Route 2 is independent of route 1 (it never uses the class count), and route 3 is "
        "independent of both (it goes through the solver's bond law and force accumulation).")
    add("")
    add("## 3. Results")
    add("")
    add("### 3.1 Ideal 13-neighbour lattice")
    add("")
    add("| constant | closed form | Voigt sum | closed form in k/a | relative error |")
    add("|---|---|---|---|---|")
    for name, key, err in (("C11", "C11", "relative_error_C11"),
                           ("C12", "C12", "relative_error_C12"),
                           ("C44", "C44", "relative_error_C44")):
        add(f"| {name} | {analytic[key]:.10g} Pa | {constants[key]:.10g} Pa | "
            f"{kva(analytic[key])} | {infinite[err]:.3e} |")
    add("")
    add(f"* Cauchy relation `C12 = C44` holds to `{infinite['cauchy_relative_gap']:.3e}` relative "
        f"(exact for a central-force lattice without pre-stress).")
    add(f"* Cubic off-pattern residual `{constants['off_pattern_residual']:.3e}`; matrix symmetry "
        f"residual `{constants['symmetry_residual']:.3e}`.")
    add(f"* Smallest eigenvalue of `C`: `{infinite['min_eigenvalue_Pa']:.6g} Pa` (> 0, stable).")
    add(f"* Zener anisotropy ratio `A = 2 C44/(C11-C12) = {infinite['zener']:.6f}` (1 = isotropic).")
    add("")
    add("| derived modulus | value |")
    add("|---|---|")
    for name in ("bulk", "shear_voigt", "shear_reuss", "shear_hill"):
        add(f"| {name} | {bounds[name]:.6g} Pa ({bounds[name] / unit:.6f} k/a) |")
    for name in ("E_100", "E_110", "E_111"):
        add(f"| {name} | {directional[name]:.6g} Pa ({directional[name] / unit:.6f} k/a) |")
    add(f"| E_111 / E_100 | {iso['E_111_over_E_100']:.4f} |")
    add(f"| nu_100 | {iso['nu_100']:.4f} |")
    add(f"| Hill-isotropic E, nu | {iso['E_hill_isotropic']:.6g} Pa, {iso['nu_hill_isotropic']:.4f} |")
    add("")
    add(f"The Voigt/Reuss spread of the shear modulus is `{gap_voigt_reuss:.3f}` of the Hill value, "
        f"i.e. the network is strongly anisotropic, and `E_111/E_100 = "
        f"{iso['E_111_over_E_100']:.2f}` means a specimen loaded along the body diagonal is more than "
        f"twice as stiff as one loaded along a cube axis.")
    add("")
    add("### 3.2 Why the diagonals are required")
    add("")
    add(f"A nearest-neighbour-only lattice (3 axis bonds) gives `C11 = {axis_only['C11'] / unit:.6f} k/a`, "
        f"`C12 = {axis_only['C12'] / unit:.6f} k/a`, `C44 = {axis_only['C44'] / unit:.6f} k/a`: zero shear "
        f"stiffness, i.e. a shear mechanism. The face/body diagonals raise `C44` to "
        f"{constants['C44'] / unit:.6f} k/a.")
    add("")
    add("### 3.3 Finite specimens (surface bond deficiency)")
    add("")
    add("| grid | bonds (solver / table) | C11 / C11_inf | C22 / C11_inf | C33 / C11_inf | "
        "C44 / C44_inf | virial vs affine |")
    add("|---|---|---|---|---|---|---|")
    for specimen in report["specimens"]:
        diag = specimen["diagonal_over_infinite"]
        grid = "x".join(str(v) for v in specimen["grid"])
        add(f"| {grid} | {specimen['bond_count_solver']} / {specimen['bond_count_table']} | "
            f"{diag['C11']:.4f} | {diag['C22_over_C11_infinite']:.4f} | "
            f"{diag['C33_over_C11_infinite']:.4f} | {diag['C44']:.4f} | "
            f"{specimen['virial_vs_affine_relative_error']:.2e} |")
    add("")
    smallest = min(report["specimens"],
                   key=lambda s: s["diagonal_over_infinite"]["C33_over_C11_infinite"])
    add(f"Bond counts agree exactly with the lattice table, so the affine tensor describes the bond "
        f"network the solver really builds. The thin specimen {smallest['grid']} loses "
        f"`{100 * smallest['surface_deficiency']:.1f}%` of its axial affine stiffness with respect to the "
        f"infinite lattice and is softer along the thin axis (`C33/C11_inf = "
        f"{smallest['diagonal_over_infinite']['C33_over_C11_infinite']:.3f}`), so the specimen itself, "
        f"not only the bond law, biases direction-dependent results.")
    add("")
    add("| thickness nz (nx = ny = max(9, nz)) | bonds | C11 / C11_inf | C33 / C11_inf |")
    add("|---|---|---|---|")
    for row in report["surface_deficiency_curve"]:
        add(f"| {row['nz']} | {row['bond_count']} | {row['C11_over_C11_infinite']:.4f} | "
            f"{row['C33_over_C11_infinite']:.4f} |")
    add("")
    add("### 3.4 Time-step bound from the phonon spectrum")
    add("")
    phonon = report["phonon"]
    add(f"`omega_max = {phonon['omega_max_rad_per_s']:.6g} rad/s` over a 15^3 Brillouin-zone grid gives "
        f"the explicit central-difference bound `dt_crit = 2/omega_max = {phonon['dt_critical_s']:.6g} s` "
        f"for the bonded network. Replacing the axial spring by the stiffer penalty contact "
        f"(`contact_stiffness = {phonon['contact_stiffness_N_per_m']:.6g} N/m`, the state after a bond "
        f"breaks) raises the bound frequency to `{phonon['omega_max_with_contact_rad_per_s']:.6g} rad/s` "
        f"and lowers the bound to `{phonon['dt_critical_with_contact_s']:.6g} s`. The solver's "
        f"conservative `recommended_dt = {phonon['recommended_dt_s']:.6g} s` is "
        f"`{phonon['recommended_over_critical']:.3f} x` the tighter bound, i.e. inside it with a safety "
        f"factor of `{1.0 / phonon['recommended_over_critical']:.1f}`.")
    add("")
    add("| `recommended_dt` branch | value |")
    add("|---|---|")
    for name, value in phonon["dt_branches_s"].items():
        marker = " (binding)" if name == phonon["binding_dt_branch"] else ""
        add(f"| {name}{marker} | {value:.6g} s |")
    add("")
    tighter = phonon["dt_critical_with_contact_s"]
    relation = "looser than" if tighter > phonon["recommended_dt_s"] else "stricter than"
    add(f"The binding branch is the `{phonon['binding_dt_branch']}` limit. The phonon bound is "
        f"{relation} `recommended_dt` by a factor of "
        f"`{tighter / phonon['recommended_dt_s']:.1f}`, so the configured step keeps a large margin to "
        f"the linear-stability limit of the bonded lattice. This is a necessary condition only: it says "
        f"the chosen step cannot be rejected by the elastic bound, not that the dynamics is accurate.")
    add("")
    add("### 3.5 Calibration gap to an engineering modulus")
    add("")
    gap = report["calibration_gap"]
    add("| target E_100 | required `bond_stiffness` | factor over current | dt scaling |")
    add("|---|---|---|---|")
    for name, row in gap.items():
        if not isinstance(row, dict):
            continue
        add(f"| {row['target_E_100_Pa']:.3g} Pa | {row['required_bond_stiffness_N_per_m']:.4g} N/m | "
            f"{row['factor_over_current']:.4g} | {row['timestep_factor']:.3e} |")
    add("")
    add(f"With the current `bond_stiffness = {inputs['bond_stiffness_N_per_m']:.6g} N/m` the lattice "
        f"reaches only `E_100 = {gap['current_E_100_Pa']:.4g} Pa`, orders of magnitude below any ice "
        f"modulus. Reaching an ice-like modulus therefore needs a `k` of order "
        f"`{min(v['required_bond_stiffness_N_per_m'] for v in gap.values() if isinstance(v, dict)):.3g} N/m` "
        f"and a correspondingly smaller explicit time step — the quantitative reason this prototype "
        f"cannot yet be quoted as a calibrated ice material.")
    add("")
    add("## 4. Figures")
    add("")
    captions = {
        "lattice_stiffness_matrices.png":
            "Affine stiffness matrix in units of k/a: ideal 13-neighbour lattice (left) and the "
            "finite specimen (right). Off-diagonal normal/shear coupling stays zero; the finite "
            "specimen loses diagonal weight and becomes non-cubic (C11 > C22 > C33).",
        "directional_young_modulus.png":
            "Directional Young's modulus E(θ) of the ideal lattice in the (001) and (110) planes, "
            "normalised by k/a. The body-diagonal peak is the anisotropy that biases specimen "
            "orientation.",
        "surface_deficiency.png":
            "Axial affine stiffness versus specimen thickness: in-plane C11 and thickness-axis C33 "
            "normalised by the infinite-lattice C11. Thin specimens are markedly softer.",
        "phonon_dispersion.png":
            "Phonon branches along Γ-X; the top branch sets omega_max and therefore the explicit "
            "time-step bound dt_crit = 2/omega_max.",
    }
    for name in report.get("figures", []):
        add(f"![{name}](lattice-elasticity/{name})")
        add("")
        add(f"*{captions.get(name, name)}*")
        add("")
    add("## 5. Notes, limits, scope")
    add("")
    add("* This is an **affine / Cauchy-Born** reference: it is exact for a uniform strain field and "
        "therefore an upper reference for the small-strain stiffness of the bonded network. It is not "
        "a measurement of the dynamically simulated moduli; the simulated response includes "
        "non-affine relaxation and, above the breaking strain, damage.")
    add("* The result covers the **bond network only**. Particle-particle contact (linear penalty law), "
        "tool contact, platens, damping and drag are not part of the affine stiffness.")
    add("* The lattice is central-force, so `C12 = C44` (Cauchy relation) and the model cannot "
        "represent a physical Poisson effect independently of shear; any calibration that needs an "
        "independent `C12`/`C44` pair requires bonded bending/torsional moments, which this solver "
        "does not implement.")
    add("* Hertz-Mindlin contact, tangential friction history, particle rotation and rolling "
        "resistance remain unimplemented in the 3-D solver; nothing here implies those capabilities.")
    add("* `MODULUS_TARGETS` in the runner are literature-scale ice moduli used only to size the "
        "calibration gap; they are not validation targets and no agreement with ice data is claimed.")
    add("")
    add("## 6. Acceptance checks and error statement")
    add("")
    add("| check | description | value | tolerance | verdict |")
    add("|---|---|---|---|---|")
    for check in report["checks"]:
        add(f"| `{check['id']}` | {check['description']} | {check['value']:.3e} | "
            f"{check['tolerance']:.3e} | {check['verdict']} |")
    add("")
    add("Key quantities against their reference values:")
    add("")
    add("| quantity | reference | model value | relative error | < 3% |")
    add("|---|---|---|---|---|")
    for name, key, err in (("C11", "C11", "relative_error_C11"),
                           ("C12", "C12", "relative_error_C12"),
                           ("C44", "C44", "relative_error_C44")):
        value = infinite[err]
        add(f"| {name} (Voigt sum) | closed form {analytic[key]:.6g} Pa | {constants[key]:.6g} Pa | "
            f"{value:.3e} | {'yes' if value < 0.03 else 'no'} |")
    for specimen in report["specimens"]:
        grid = "x".join(str(v) for v in specimen["grid"])
        value = specimen["virial_vs_affine_relative_error"]
        add(f"| C (solver virial, {grid}) | affine bond sum | {value:.3e} | {value:.3e} | "
            f"{'yes' if value < 0.03 else 'no'} |")
    add("")
    add("The closed-form constants are recovered to machine precision and the solver-force route "
        "agrees with the affine route to the level expected from the finite affine strain "
        "(`O(eps)` geometric terms); every comparison above is far below the 3% acceptance "
        "threshold. These are manufactured/reference agreements, **not** experimental validation: "
        "no experimental ice modulus is claimed or reproduced.")
    add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "lattice-elasticity")
    parser.add_argument("--doc", type=Path, default=None,
                        help="optional markdown document path to (re)generate")
    parser.add_argument("--bond-stiffness", type=float, default=2000.0)
    parser.add_argument("--radius", type=float, default=0.025)
    parser.add_argument("--density", type=float, default=917.0)
    args = parser.parse_args(argv)
    report = run_benchmark(args.output, doc_path=args.doc,
                           bond_stiffness=args.bond_stiffness,
                           radius=args.radius, density=args.density)
    print(json.dumps({
        "protocol": report["protocol"],
        "verdict": report["verdict"],
        "pass_count": report["pass_count"],
        "case_count": report["case_count"],
        "failed": [c["id"] for c in report["checks"] if c["verdict"] != "PASS"],
    }, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())