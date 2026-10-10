"""Deterministic regression probes for DEM3D fracture objectivity and energy release.

This is a numerical verification harness, not an ice-material calibration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.dem3d_energy import EnergyAuditedIceDEM3D


def _config(**kwargs: Any) -> DEM3DConfig:
    values: dict[str, Any] = {
        "nx": 3, "ny": 3, "nz": 2, "drag": 0.0,
        "fix_x_edges": False, "tool_gap": 2.0,
        "breaking_strain": 0.5, "shear_breaking_strain": 0.5,
    }
    values.update(kwargs)
    return DEM3DConfig(**values)


def _rotation() -> torch.Tensor:
    a, b, c = 0.31, -0.22, 0.17
    rx = torch.tensor([[1., 0., 0.], [0., math.cos(a), -math.sin(a)],
                       [0., math.sin(a), math.cos(a)]], dtype=torch.float64)
    ry = torch.tensor([[math.cos(b), 0., math.sin(b)], [0., 1., 0.],
                       [-math.sin(b), 0., math.cos(b)]], dtype=torch.float64)
    rz = torch.tensor([[math.cos(c), -math.sin(c), 0.],
                       [math.sin(c), math.cos(c), 0.], [0., 0., 1.]],
                      dtype=torch.float64)
    return rz @ ry @ rx


def check_rigid_rotation_objectivity() -> dict[str, Any]:
    sim = IceDEM3D(_config())
    sim.positions[4, 0] += 1.0e-4
    force_before, _ = sim.forces(update_fracture=False)
    rotation = _rotation()
    translation = torch.tensor([1.7, -2.1, 3.4], dtype=torch.float64)
    sim.positions = sim.positions @ rotation.T + translation
    force_after, _ = sim.forces(update_fracture=False)
    expected = force_before @ rotation.T
    residual = float(torch.linalg.vector_norm(force_after - expected))
    scale = max(float(torch.linalg.vector_norm(expected)), 1.0e-30)
    relative = residual / scale
    passed = math.isfinite(relative) and relative < 1.0e-8 and sim.broken_bonds == 0
    return {
        "name": "rigid_rotation_objectivity",
        "passed": passed,
        "relative_force_rotation_error": relative,
        "broken_bonds": sim.broken_bonds,
        "tolerance": 1.0e-8,
    }


def check_single_bond_release_energy() -> dict[str, Any]:
    sim = IceDEM3D(_config(breaking_strain=0.01, shear_breaking_strain=100.0))
    sim.alive[:] = False
    bond_id = 0
    sim.alive[bond_id] = True
    i, j = (int(v) for v in sim.pairs[bond_id].tolist())
    reference = sim.initial_positions[j] - sim.initial_positions[i]
    normal = reference / torch.linalg.vector_norm(reference)
    sim.positions[j] += normal * (0.02 * sim.rest_lengths[bond_id])
    extension = torch.linalg.vector_norm(sim.positions[j] - sim.positions[i]) - sim.rest_lengths[bond_id]
    expected = 0.5 * sim.config.bond_stiffness * float(extension.square())
    sim.forces()
    observed = float(sim.failure_energy_J[bond_id])
    error = abs(observed - expected)
    passed = (not bool(sim.alive[bond_id])) and math.isfinite(observed) and error <= max(1.0e-14, expected * 1.0e-12)
    return {
        "name": "single_bond_release_energy",
        "passed": passed,
        "bond_id": bond_id,
        "extension_m": float(extension),
        "expected_release_J": expected,
        "observed_release_J": observed,
        "absolute_error_J": error,
        "broken_bonds": sim.broken_bonds,
    }


def check_mixed_mode_failure_and_ledger() -> dict[str, Any]:
    cfg = _config(breaking_strain=0.02, shear_breaking_strain=0.001)
    sim = EnergyAuditedIceDEM3D(cfg)
    deformation = torch.tensor([[1.04, 0.08, 0.0], [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]], dtype=torch.float64)
    sim.positions = sim.initial_positions @ deformation.T
    sim.forces()
    modes = sim.failure_mode[~sim.alive]
    mixed = int((modes == 3).sum())
    release = float(sim.failure_energy_J.sum())
    row = sim.diagnostics()
    ledger = float(row["fracture_release_cumulative_J"])
    passed = (
        sim.broken_bonds > 0 and mixed > 0
        and math.isfinite(release) and release >= 0.0
        and abs(ledger - release) <= max(1.0e-14, release * 1.0e-12)
    )
    return {
        "name": "mixed_mode_failure_and_release_ledger",
        "passed": passed,
        "broken_bonds": sim.broken_bonds,
        "mixed_mode_bonds": mixed,
        "sum_per_bond_release_J": release,
        "cumulative_ledger_release_J": ledger,
        "ledger_error_J": abs(ledger - release),
    }


def _fracture_event_ledger() -> list[dict[str, Any]]:
    """Return a deterministic per-bond ledger from the controlled mixed-mode probe."""
    sim = EnergyAuditedIceDEM3D(
        _config(breaking_strain=0.02, shear_breaking_strain=0.001)
    )
    deformation = torch.tensor(
        [[1.04, 0.08, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    sim.positions = sim.initial_positions @ deformation.T
    sim.forces()
    pairs = sim.pairs.detach().cpu().tolist()
    alive = sim.alive.detach().cpu().tolist()
    modes = sim.failure_mode.detach().cpu().tolist()
    times = sim.failure_time_s.detach().cpu().tolist()
    extensions = sim.failure_extension_m.detach().cpu().tolist()
    energies = sim.failure_energy_J.detach().cpu().tolist()
    lengths = sim.rest_lengths.detach().cpu().tolist()
    events: list[dict[str, Any]] = []
    for bond_id, pair in enumerate(pairs):
        if alive[bond_id]:
            continue
        extension = float(extensions[bond_id])
        length = float(lengths[bond_id])
        events.append({
            "bond_id": bond_id,
            "particle_i": int(pair[0]),
            "particle_j": int(pair[1]),
            "failure_mode_code": int(modes[bond_id]),
            "failure_time_s": float(times[bond_id]),
            "failure_extension_m": extension,
            "reference_length_m": length,
            "failure_strain": extension / max(length, 1.0e-30),
            "released_energy_J": float(energies[bond_id]),
        })
    return events


def run_validation(output: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    checks = [
        check_rigid_rotation_objectivity(),
        check_single_bond_release_energy(),
        check_mixed_mode_failure_and_ledger(),
    ]
    events = _fracture_event_ledger()
    event_ids = [int(event["bond_id"]) for event in events]
    energies = [float(event["released_energy_J"]) for event in events]
    event_finite = all(
        math.isfinite(float(value))
        for event in events
        for value in event.values()
    )
    expected_energy = float(checks[2]["sum_per_bond_release_J"])
    event_checks = {
        "unique_bond_ids": len(event_ids) == len(set(event_ids)),
        "stable_bond_id_order": event_ids == sorted(event_ids),
        "finite_event_values": event_finite,
        "nonnegative_release_energy": all(value >= 0.0 for value in energies),
        "event_count_matches_fracture_count": len(events) == int(checks[2]["broken_bonds"]),
        "event_energy_matches_ledger": abs(sum(energies) - expected_energy)
        <= max(1.0e-14, abs(expected_energy) * 1.0e-12),
    }
    ledger_digest = hashlib.sha256(
        json.dumps(events, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    all_passed = all(item["passed"] for item in checks) and all(event_checks.values())
    report = {
        "protocol": "tensordem-dem3d-fracture-objectivity-energy-v2",
        "verdict": "PASS" if all_passed else "FAIL",
        "check_count": len(checks) + len(event_checks),
        "checks": checks,
        "fracture_event_count": len(events),
        "fracture_event_checks": event_checks,
        "fracture_release_energy_sum_J": sum(energies),
        "fracture_event_signature_sha256": ledger_digest,
        "interpretation": (
            "PASS means controlled numerical invariants and event-ledger integrity checks met their "
            "tolerances. The ledger is a deterministic audit record of the controlled mixed-mode "
            "probe, not a full time-resolved crack-propagation history and not physical ice "
            "fracture-energy calibration."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dem3d_fracture_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
    )
    with (output / "dem3d_fracture_events.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "bond_id", "particle_i", "particle_j", "failure_mode_code",
            "failure_time_s", "failure_extension_m", "reference_length_m",
            "failure_strain", "released_energy_J",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)
    if not all_passed:
        raise RuntimeError("DEM3D fracture validation or event-ledger integrity check failed")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-fracture-validation"))
    args = parser.parse_args(argv)
    report = run_validation(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
