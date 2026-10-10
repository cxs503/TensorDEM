"""Run analytic and invariant mechanics checks against the actual DEM3D solver.

This protocol intentionally tests the laws currently implemented by TensorDEM:
linear central-force bonds and linear penalty normal contact. It does not claim
Hertz-Mindlin, calibrated rock/ice strength, or experimental validation.
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


def _base_config(**overrides: Any) -> DEM3DConfig:
    values: dict[str, Any] = {
        "nx": 3, "ny": 3, "nz": 2, "drag": 0.0,
        "contact_damping": 0.0, "fix_x_edges": False,
        "tool_radius": 0.01, "tool_gap": 0.0,
        "breaking_strain": 0.015, "shear_breaking_strain": 0.5,
    }
    values.update(overrides)
    return DEM3DConfig(**values)


def _isolated_pair(*, bond_active: bool, extension_m: float = 0.0,
                   overlap_m: float = 0.0, bond_stiffness: float = 2000.0,
                   contact_stiffness: float = 5000.0) -> tuple[IceDEM3D, int, int]:
    """Move every non-target particle away and isolate one axial lattice pair."""
    sim = IceDEM3D(_base_config(
        bond_stiffness=bond_stiffness, contact_stiffness=contact_stiffness
    ))
    target = int(torch.argmin(torch.abs(sim.rest_lengths - 2.0 * sim.config.radius)))
    i, j = (int(v) for v in sim.pairs[target].tolist())
    rest = float(sim.rest_lengths[target])
    sim.positions[:, 0] = 1000.0 + 10.0 * torch.arange(
        len(sim.positions), dtype=sim.dtype, device=sim.device
    )
    sim.positions[:, 1:] = 0.0
    sim.positions[i] = torch.tensor([0.0, 0.0, 0.0], dtype=sim.dtype)
    distance = rest + extension_m if bond_active else 2.0 * sim.config.radius - overlap_m
    sim.positions[j] = torch.tensor([distance, 0.0, 0.0], dtype=sim.dtype)
    sim.alive[:] = False
    sim.alive[target] = bond_active
    sim.tool_start[:] = torch.tensor([1.0e6, 1.0e6, 1.0e6], dtype=sim.dtype)
    sim.tool_velocity.zero_()
    sim.velocities.zero_()
    return sim, i, j


def _record(case_id: str, observed: Any, expected: Any, passed: bool,
            units: str, detail: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "status": "PASS" if passed else "FAIL",
        "observed": observed,
        "expected": expected,
        "units": units,
        "detail": detail,
    }


def run_mechanics_benchmarks(output: Path) -> dict[str, Any]:
    """Execute the full analytic/invariant benchmark pack and write audit files."""
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    manifest_path = ROOT / "benchmarks" / "dem3d" / "mechanics_cases.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "tensordem-dem3d-mechanics-v1":
        raise ValueError(f"unsupported mechanics benchmark protocol in {manifest_path}")
    definitions = {case["id"]: case for case in manifest.get("cases", [])}
    if len(definitions) != len(manifest.get("cases", [])):
        raise ValueError("mechanics benchmark manifest contains duplicate case IDs")

    # 1. Isolated central-force bond: compare a real solver force with F=k*delta_l.
    bond_case = definitions["bond_axial_spring"]
    extension = float(bond_case["parameters"]["extension_m"])
    bond_stiffness = float(bond_case["parameters"]["bond_stiffness_N_per_m"])
    sim, i, j = _isolated_pair(
        bond_active=True, extension_m=extension, bond_stiffness=bond_stiffness
    )
    force, _ = sim.forces(update_fracture=False)
    expected_force = sim.config.bond_stiffness * extension
    observed_force = float(force[i, 0])
    abs_error = abs(observed_force - expected_force)
    rel_error = abs_error / max(abs(expected_force), 1.0e-30)
    bond_acceptance = bond_case["acceptance"]
    records.append(_record(
        "bond_axial_spring", observed_force, expected_force,
        abs_error <= float(bond_acceptance["absolute_force_error_N"])
        and rel_error <= float(bond_acceptance["relative_force_error"]), "N",
        f"absolute_error_N={abs_error:.6g}; relative_error={rel_error:.6g}; pair=({i},{j})",
    ))

    # 2. Isolated unbonded pair: current law is linear penalty, not Hertz contact.
    contact_case = definitions["linear_contact_penalty"]
    overlap = float(contact_case["parameters"]["overlap_m"])
    contact_stiffness = float(contact_case["parameters"]["contact_stiffness_N_per_m"])
    sim, i, j = _isolated_pair(
        bond_active=False, overlap_m=overlap, contact_stiffness=contact_stiffness
    )
    force, _ = sim.forces(update_fracture=False)
    expected_contact = sim.config.contact_stiffness * overlap
    observed_contact = abs(float(force[i, 0]))
    contact_error = abs(observed_contact - expected_contact)
    contact_rel = contact_error / max(abs(expected_contact), 1.0e-30)
    contact_acceptance = contact_case["acceptance"]
    records.append(_record(
        "linear_contact_penalty", observed_contact, expected_contact,
        contact_error <= float(contact_acceptance["absolute_force_error_N"])
        and contact_rel <= float(contact_acceptance["relative_force_error"]), "N",
        f"absolute_error_N={contact_error:.6g}; relative_error={contact_rel:.6g}; "
        "zero normal velocity; no tangential history",
    ))

    # 3. Action-reaction consistency with a displaced indenter and a deformed lattice.
    sim = IceDEM3D(_base_config(tool_radius=0.1, tool_gap=0.0))
    top = sim.initial_positions[:, 2].max()
    top_axis = torch.linalg.vector_norm(sim.initial_positions[:, :2], dim=1)
    candidates = torch.where(sim.initial_positions[:, 2] == top, top_axis,
                             torch.full_like(top_axis, float("inf")))
    loaded = int(torch.argmin(candidates))
    sim.positions[loaded, 2] += 0.015
    sim.velocities[loaded, 0] = 0.02
    force, reaction = sim.forces(update_fracture=False)
    residual = float(torch.linalg.vector_norm(force.sum(dim=0) + reaction))
    contact_count = int((torch.linalg.vector_norm(
        sim.positions - sim.tool_position, dim=1
    ) < sim.config.radius + sim.config.tool_radius).sum())
    force_limit = float(definitions["force_balance"]["acceptance"]["net_force_residual_N"])
    records.append(_record(
        "force_balance", {"residual_N": residual, "tool_contact_particles": contact_count}, 0.0,
        math.isfinite(residual) and residual <= force_limit and contact_count >= 1, "N",
        "Checks action-reaction with an explicitly loaded top-axis particle in tool contact; "
        "the contact count must be nonzero.",
    ))

    # 4. Objective shear failure with tensile threshold intentionally out of range.
    shear_case = definitions["single_bond_shear_failure"]
    shear_params = shear_case["parameters"]
    shear_cfg = _base_config(
        breaking_strain=float(shear_params["breaking_strain"]),
        shear_breaking_strain=float(shear_params["shear_breaking_strain"]),
    )
    sim = IceDEM3D(shear_cfg)
    gamma = float(shear_params["engineering_shear_strain"])
    deformation = torch.tensor([
        [1.0, gamma, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=sim.dtype)
    sim.positions = sim.initial_positions @ deformation.T
    sim.forces()
    shear_broken = sim.failure_mode[~sim.alive]
    required_mode = int(shear_case["acceptance"]["failure_mode_code"])
    shear_passed = (
        sim.broken_bonds >= int(shear_case["acceptance"]["minimum_broken_bonds"])
        and shear_broken.numel() > 0
        and bool((shear_broken == required_mode).all())
    )
    records.append(_record(
        "single_bond_shear_failure",
        {"broken_bonds": sim.broken_bonds,
         "failure_modes": sorted(set(int(v) for v in shear_broken.tolist()))},
        {"minimum_broken_bonds": int(shear_case["acceptance"]["minimum_broken_bonds"]),
         "failure_mode_code": required_mode},
        shear_passed, "bonds",
        "Affine simple shear is applied while tensile failure is disabled by a high threshold.",
    ))

    # 5. Tensile failure is committed by the force update and remains irreversible.
    sim = IceDEM3D(_base_config(breaking_strain=0.015, shear_breaking_strain=0.5))
    target = int(torch.argmin(torch.abs(sim.rest_lengths - 2.0 * sim.config.radius)))
    i, j = (int(v) for v in sim.pairs[target].tolist())
    sim.positions[j] = sim.positions[i] + 1.03 * (
        sim.initial_positions[j] - sim.initial_positions[i]
    )
    sim.forces()
    broken_after_load = sim.broken_bonds
    sim.positions.copy_(sim.initial_positions)
    sim.forces()
    broken_after_unload = sim.broken_bonds
    healing = broken_after_load - broken_after_unload
    minimum_broken = int(definitions["irreversible_tensile_failure"]["acceptance"]["minimum_broken_bonds"])
    passed = broken_after_load >= minimum_broken and healing == 0
    records.append(_record(
        "irreversible_tensile_failure", {"after_load": broken_after_load,
                                         "after_unload": broken_after_unload},
        {"minimum_after_load": 1, "healing_events": 0}, passed, "bonds",
        "Bond damage must be monotone under unloading; only a real force update commits failure.",
    ))

    # 6. Objectivity: rigid rotation must not create strain damage.
    sim = IceDEM3D(_base_config())
    angle = 0.37
    rotation = torch.tensor([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=sim.dtype)
    sim.positions = sim.initial_positions @ rotation.T + torch.tensor(
        [0.4, -0.2, 0.7], dtype=sim.dtype
    )
    force, _ = sim.forces(update_fracture=False)
    max_force = float(force.abs().max())
    rotation_limit = float(definitions["rigid_rotation_objectivity"]["acceptance"]["maximum_internal_force_N"])
    required_broken = int(definitions["rigid_rotation_objectivity"]["acceptance"]["broken_bonds"])
    records.append(_record(
        "rigid_rotation_objectivity", {"max_abs_force_N": max_force,
                                       "broken_bonds": sim.broken_bonds},
        {"max_abs_force_N": rotation_limit, "broken_bonds": required_broken},
        max_force <= rotation_limit and sim.broken_bonds == required_broken, "N",
        "A rigid-body rotation and translation should not generate internal force or shear failure.",
    ))

    # 7. The configured explicit step must not exceed the model's conservative guard.
    base = _base_config()
    dt_factor = float(definitions["timestep_guard"]["parameters"]["dt_factor"])
    oversized_rejected = False
    try:
        IceDEM3D(_base_config(dt=base.recommended_dt * dt_factor))
    except ValueError:
        oversized_rejected = True
    records.append(_record(
        "timestep_guard", oversized_rejected, True, oversized_rejected, "boolean",
        f"recommended_dt_s={base.recommended_dt:.12g}; attempted_dt_s={base.recommended_dt * dt_factor:.12g}",
    ))

    failed = [row for row in records if row["status"] != "PASS"]
    report = {
        "protocol": "tensordem-dem3d-mechanics-v1",
        "model_scope": "linear central-force bonds and linear penalty normal contact",
        "interpretation": (
            "PASS means the implemented mechanics satisfy these analytic/invariant checks. "
            "It is not experimental validation, constitutive calibration, convergence proof, "
            "or qualification for full-scale icebreaking."
        ),
        "verdict": "FAIL" if failed else "PASS",
        "case_count": len(records),
        "pass_count": len(records) - len(failed),
        "fail_count": len(failed),
        "cases": records,
        "not_yet_supported": manifest.get("not_yet_supported", []),
    }
    (output / "mechanics_benchmark_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "mechanics_benchmark_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["case_id", "status", "observed", "expected", "units", "detail"]
        )
        writer.writeheader()
        writer.writerows(records)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run analytic and invariant mechanics benchmarks for TensorDEM 3-D"
    )
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-mechanics"))
    args = parser.parse_args(argv)
    report = run_mechanics_benchmarks(args.output)
    print(json.dumps({
        "protocol": report["protocol"], "verdict": report["verdict"],
        "case_count": report["case_count"], "pass_count": report["pass_count"],
        "fail_count": report["fail_count"],
        "json": str(args.output / "mechanics_benchmark_report.json"),
        "csv": str(args.output / "mechanics_benchmark_report.csv"),
    }, indent=2, sort_keys=True))
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
