"""Audit the DEM3D bonded-particle fracture-energy parameter set.

Closes three gaps listed in ``docs/dem3d_material_calibration_roadmap.md``:

* bond energy before failure / released energy at failure bookkeeping under a
  controlled tensile test;
* a calibrated fracture-energy law instead of equating breaking strain with
  fracture energy;
* sensitivity of the parameter set to particle resolution.

The script never changes the solver; it only drives it. It writes a JSON report,
two CSV tables and a matplotlib figure. Everything is reproducible on CPU with
small specimens (a few hundred particles).

Run::

    python scripts/benchmark_dem3d_bond_fracture_energy.py \
        --output artifacts/bond_fracture_energy
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from tensordem.bond_fracture_energy import (
    analytic_fracture_energy_J_per_m2,
    audit_config,
    fracture_energy_from_config,
    ice_fracture_energy_J_per_m2,
    required_breaking_strain,
)
from tensordem.dem3d import DEM3DConfig, IceDEM3D

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "bond_fracture_energy"

#: Sea-ice fracture toughness range used as the physical reference scale.
K_IC_RANGE_MPA_SQRT_M = (0.05, 0.15)
YOUNGS_RANGE_GPA = (5.0, 10.0)
POISSON_RATIO = 0.3


def physical_band() -> tuple[float, float]:
    """Return the fracture-energy band implied by the reference toughness range."""
    values = [
        ice_fracture_energy_J_per_m2(k, e, POISSON_RATIO)
        for k in K_IC_RANGE_MPA_SQRT_M
        for e in YOUNGS_RANGE_GPA
    ]
    return min(values), max(values)


def size_series(sizes: tuple[int, ...] = (5, 7, 9, 13, 17, 21)) -> list[dict[str, Any]]:
    """Finite-specimen convergence of the measured fracture energy to the closed form."""
    rows: list[dict[str, Any]] = []
    for n in sizes:
        cfg = DEM3DConfig(nx=n, ny=n, nz=n)
        measured = fracture_energy_from_config(cfg)
        spacing = measured["lattice_spacing_m"]
        footprint = (n - 1) ** 2 * spacing ** 2
        population = n ** 2 * spacing ** 2
        rows.append({
            "size": n,
            "particle_count": measured["particle_count"],
            "bond_count": measured["bond_count"],
            "crossing_bond_count": measured["crossing_bond_count"],
            "analytic_J_per_m2": measured["analytic_fracture_energy_J_per_m2"],
            "measured_footprint_J_per_m2": measured["measured_fracture_energy_J_per_m2"],
            "measured_population_J_per_m2": measured["released_energy_J"] / population,
            "footprint_relative_error": measured["relative_error"],
            "population_relative_error": abs(
                measured["released_energy_J"] / population
                - measured["analytic_fracture_energy_J_per_m2"]
            ) / measured["analytic_fracture_energy_J_per_m2"],
            "footprint_m2": footprint,
        })
    return rows


def axis_symmetry(size: int = 11) -> list[dict[str, Any]]:
    """Equivalent fracture energy must not depend on which cube axis the crack cuts."""
    rows: list[dict[str, Any]] = []
    for axis in (0, 1, 2):
        measured = fracture_energy_from_config(DEM3DConfig(nx=size, ny=size, nz=size), axis=axis)
        rows.append({
            "axis": axis,
            "measured_J_per_m2": measured["measured_fracture_energy_J_per_m2"],
            "analytic_J_per_m2": measured["analytic_fracture_energy_J_per_m2"],
        })
    return rows


def resolution_series(radii: tuple[float, ...] = (0.01, 0.015, 0.02, 0.025, 0.05),
                      size: int = 11) -> list[dict[str, Any]]:
    """The strain-based criterion makes the equivalent fracture energy size independent."""
    rows: list[dict[str, Any]] = []
    for radius in radii:
        cfg = DEM3DConfig(nx=size, ny=size, nz=size, radius=radius)
        measured = fracture_energy_from_config(cfg)
        rows.append({
            "radius_m": radius,
            "lattice_spacing_m": measured["lattice_spacing_m"],
            "particle_count": measured["particle_count"],
            "measured_J_per_m2": measured["measured_fracture_energy_J_per_m2"],
            "analytic_J_per_m2": measured["analytic_fracture_energy_J_per_m2"],
            "mass_kg": cfg.mass,
        })
    return rows


def strain_sweep(strains: tuple[float, ...] = (0.004, 0.006, 0.008, 0.010, 0.012,
                                               0.015, 0.018, 0.022, 0.026, 0.030),
                 bond_stiffness: float = 2000.0) -> list[dict[str, Any]]:
    """Equivalent fracture energy versus breaking strain for a fixed bond stiffness."""
    band = physical_band()
    rows: list[dict[str, Any]] = []
    for strain in strains:
        analytic = analytic_fracture_energy_J_per_m2(bond_stiffness, strain)
        rows.append({
            "bond_stiffness_N_per_m": bond_stiffness,
            "breaking_strain": strain,
            "fracture_energy_J_per_m2": analytic,
            "inside_physical_band": band[0] <= analytic <= band[1],
            "ratio_to_band_midpoint": analytic / (0.5 * sum(band)),
        })
    return rows


def controlled_tensile_test(size: int = 3, breaking_strain: float = 0.01,
                           increment: float = 5e-4,
                           bond_stiffness: float = 2000.0,
                           final_strain: float | None = None) -> dict[str, Any]:
    """Impose affine axial stretch and audit every failure record against analytics.

    Positions are prescribed, so there is no time-integration error: the test
    isolates the failure detection and the release-energy bookkeeping. Each
    broken bond is checked against
    ``0.5 * k * (strain_at_detection * rest_length) ** 2`` recomputed from the
    solver's recorded extension, and against the analytic failure point
    ``strain = breaking_strain`` with an overshoot bounded by one strain
    increment.
    """
    cfg = DEM3DConfig(nx=size, ny=size, nz=size, breaking_strain=breaking_strain,
                      bond_stiffness=bond_stiffness, drag=0.0, contact_damping=0.0,
                      shear_breaking_strain=1.0, fix_x_edges=False)
    sim = IceDEM3D(cfg)
    initial = sim.initial_positions.clone()
    bulk = ~sim.fixed
    strain = 0.0
    target = 2.0 * breaking_strain if final_strain is None else final_strain
    if not math.isfinite(target) or target <= 0:
        raise ValueError("final_strain must be finite and positive")
    steps = int(math.ceil(target / increment))
    for _ in range(steps):
        strain += increment
        positions = initial.clone()
        positions[bulk, 0] = initial[bulk, 0] * (1.0 + strain)
        sim.positions = positions
        sim.forces(update_fracture=True)

    broken = sim.failure_energy_J > 0
    broken_count = int(broken.sum())
    extensions = sim.failure_extension_m[broken]
    rest = sim.rest_lengths[broken]
    recorded = sim.failure_energy_J[broken]
    diagnostics = sim.diagnostics()
    ledger = float(diagnostics["released_bond_energy_J"])
    total_recorded = float(recorded.sum())
    result: dict[str, Any] = {
        "size": size,
        "bond_stiffness_N_per_m": bond_stiffness,
        "breaking_strain": breaking_strain,
        "strain_increment": increment,
        "target_strain": target,
        "applied_strain_final": strain,
        "bond_count": int(len(sim.pairs)),
        "broken_bond_count": broken_count,
        "intact_bond_count": int((~broken).sum()),
        "recorded_release_energy_J": total_recorded,
        "ledger_release_energy_J": ledger,
    }
    if broken_count == 0:
        result.update({
            "ledger_relative_discrepancy": 0.0,
            "max_relative_energy_recompute_error": 0.0,
            "max_relative_energy_error_vs_analytic_failure": 0.0,
            "broken_bonds_are_stretched_ones": True,
            "min_detected_strain": None,
            "max_detected_strain": None,
            "max_strain_overshoot": None,
            "overshoot_within_one_increment": True,
        })
        return result

    recomputed = 0.5 * bond_stiffness * extensions ** 2
    detected_strain = extensions / rest
    analytic_failure = 0.5 * bond_stiffness * (breaking_strain * rest) ** 2
    pairs = sim.pairs[broken]
    result.update({
        "ledger_relative_discrepancy": abs(ledger - total_recorded) / total_recorded,
        "max_relative_energy_recompute_error": float(
            ((recorded - recomputed).abs() / recomputed).max()
        ),
        "max_relative_energy_error_vs_analytic_failure": float(
            ((recorded - analytic_failure).abs() / analytic_failure).max()
        ),
        "broken_bonds_are_stretched_ones": bool(
            (initial[pairs[:, 0], 0] - initial[pairs[:, 1], 0]).abs().min() > 0
        ),
        "min_detected_strain": float(detected_strain.min()),
        "max_detected_strain": float(detected_strain.max()),
        "max_strain_overshoot": float((detected_strain - breaking_strain).max()),
        "overshoot_within_one_increment": bool(
            float((detected_strain - breaking_strain).max()) <= increment + 1e-12
        ),
    })
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(report: dict[str, Any], output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    band = report["physical_band_J_per_m2"]
    strains = [row["breaking_strain"] for row in report["strain_sweep"]]
    energies = [row["fracture_energy_J_per_m2"] for row in report["strain_sweep"]]
    axes[0].axhspan(band[0], band[1], color="tab:green", alpha=0.2,
                    label="sea-ice band K$_{Ic}^2(1-\\nu^2)/E$")
    axes[0].plot(strains, energies, "o-", color="tab:blue",
                 label=f"$G_f=10.5\\,k\\,\\varepsilon_b^2$, k={report['default_bond_stiffness']:.0f} N/m")
    axes[0].axvline(report["default_parameters"]["breaking_strain"], color="tab:red", ls="--",
                    label=f"default $\\varepsilon_b$={report['default_parameters']['breaking_strain']}")
    axes[0].set_xlabel("breaking strain $\\varepsilon_b$ [-]")
    axes[0].set_ylabel("equivalent $G_f$ [J/m$^2$]")
    axes[0].set_title("fracture-energy calibration law")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    sizes = [row["size"] for row in report["size_series"]]
    axes[1].axhline(report["analytic_fracture_energy_J_per_m2"], color="black", ls=":",
                    label="closed form $10.5 k \\varepsilon_b^2$")
    axes[1].plot(sizes, [row["measured_footprint_J_per_m2"] for row in report["size_series"]],
                 "o-", label="footprint area $(n-1)^2a^2$")
    axes[1].plot(sizes, [row["measured_population_J_per_m2"] for row in report["size_series"]],
                 "s--", label="population area $n^2a^2$")
    axes[1].set_xlabel("specimen size n (n$\\times$n$\\times$n particles)")
    axes[1].set_ylabel("measured $G_f$ [J/m$^2$]")
    axes[1].set_title("finite-specimen convergence, $O(1/n)$")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    test = report["controlled_tensile_test"]
    radii = [f"{row['radius_m']*1000:.0f}" for row in report["resolution_series"]]
    axes[2].plot(radii, [row["measured_J_per_m2"] for row in report["resolution_series"]],
                 "o-", label="measured $G_f$")
    axes[2].axhline(report["analytic_fracture_energy_J_per_m2"], color="black", ls=":",
                    label="closed form")
    axes[2].set_xlabel("particle radius [mm] (lattice spacing = 2r)")
    axes[2].set_ylabel("$G_f$ [J/m$^2$]")
    axes[2].set_title(f"resolution independence; tensile audit "
                      f"{test['broken_bond_count']}/{test['bond_count']} bonds")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    path = output_dir / "bond_fracture_energy.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def run_audit(output_dir: Path = DEFAULT_OUTPUT, make_figure: bool = True) -> dict[str, Any]:
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    default = DEM3DConfig()
    audit = audit_config(default)
    band = physical_band()
    sizes = size_series()
    tensile = controlled_tensile_test()
    report = {
        "protocol": "tensordem-dem3d-bond-fracture-energy-v1",
        "units": "SI",
        "default_parameters": {
            "bond_stiffness": default.bond_stiffness,
            "breaking_strain": default.breaking_strain,
            "shear_breaking_strain": default.shear_breaking_strain,
            "radius": default.radius,
        },
        "default_bond_stiffness": default.bond_stiffness,
        "analytic_fracture_energy_J_per_m2": audit.analytic_J_per_m2,
        "physical_band_J_per_m2": list(band),
        "physical_band_basis": {
            "K_Ic_MPa_sqrt_m": list(K_IC_RANGE_MPA_SQRT_M),
            "E_GPa": list(YOUNGS_RANGE_GPA),
            "poisson_ratio": POISSON_RATIO,
            "formula": "G = K_Ic^2 (1 - nu^2) / E",
        },
        "default_audit": audit.as_dict(),
        "ratio_default_to_band_midpoint": audit.relative_to_band(),
        "size_series": sizes,
        "axis_symmetry": axis_symmetry(),
        "resolution_series": resolution_series(),
        "strain_sweep": strain_sweep(),
        "controlled_tensile_test": tensile,
        "calibration_targets": [
            {
                "target_J_per_m2": target,
                "required_breaking_strain": required_breaking_strain(default.bond_stiffness, target),
            }
            for target in (band[0], 0.5 * sum(band), band[1])
        ],
        "interpretation": {
            "law": "For the 26-neighbour simple-cubic lattice a crack sheet normal to a "
                   "cube axis releases G_f = 10.5 * k * eps_b^2 J/m^2; the spacing cancels "
                   "because bonds per area ~ 1/a^2 and stored energy per bond ~ a^2.",
            "limits": "Axis-normal crack sheets only. Real cracks are mixed-mode and the "
                      "model has no explicit mode-II energy; the shear criterion breaks "
                      "bonds at a strain, not at an energy release rate.",
            "not_validation": "Matching a plausible G_f band does not validate the DEM. It "
                              "only shows which physical fracture energy a parameter set "
                              "represents.",
        },
    }
    if make_figure:
        _plot(report, output_dir)
    (output_dir / "bond_fracture_energy.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "bond_fracture_energy_size.csv", sizes)
    _write_csv(output_dir / "bond_fracture_energy_strain.csv", report["strain_sweep"])
    _write_csv(output_dir / "bond_fracture_energy_resolution.csv", report["resolution_series"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    report = run_audit(args.output, make_figure=not args.no_figure)
    print(json.dumps({
        "analytic_G_f_J_per_m2": report["analytic_fracture_energy_J_per_m2"],
        "physical_band_J_per_m2": report["physical_band_J_per_m2"],
        "ratio_to_band_midpoint": report["ratio_default_to_band_midpoint"],
        "size_series_max_error": max(row["footprint_relative_error"] for row in report["size_series"]),
        "tensile_broken_bonds": report["controlled_tensile_test"]["broken_bond_count"],
        "tensile_max_relative_energy_error": report["controlled_tensile_test"]["max_relative_energy_recompute_error"],
    }, indent=2))


if __name__ == "__main__":
    main()