"""Kinematic specimen-response patch tests for the 3-D bonded-particle DEM.

This is a small-strain affine-compression patch test, not a standard UCS fixture.
It measures a reproducible apparent secant modulus and checks near-linearity.
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

import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tensordem.dem3d import DEM3DConfig, IceDEM3D


def _run_once(definition: dict[str, Any]) -> tuple[list[dict[str, float | int]], str]:
    cfg = definition["config"]
    sim = IceDEM3D(DEM3DConfig(
        nx=int(cfg["nx"]), ny=int(cfg["ny"]), nz=int(cfg["nz"]),
        radius=float(cfg["radius_m"]), density=float(cfg["density_kg_m3"]),
        bond_stiffness=float(cfg["bond_stiffness_N_m"]),
        drag=0.0, contact_damping=0.0, fix_x_edges=False, fix_bottom=False,
        tool_radius=float(cfg["radius_m"]) * 0.1, tool_gap=0.0,
    ))
    # Keep the spherical indenter entirely outside the specimen; this protocol
    # isolates the bond-network response under prescribed affine kinematics.
    sim.tool_start[:] = torch.tensor([1.0e6, 1.0e6, 1.0e6], dtype=sim.dtype)
    sim.tool_velocity.zero_()
    original = sim.initial_positions.clone()
    z_bottom = float(original[:, 2].min())
    z_top = float(original[:, 2].max())
    top_mask = torch.isclose(original[:, 2], torch.tensor(z_top, dtype=sim.dtype))
    area = (int(cfg["nx"]) - 1) * (2.0 * float(cfg["radius_m"])) * (
        (int(cfg["ny"]) - 1) * (2.0 * float(cfg["radius_m"]))
    )
    rows: list[dict[str, float | int]] = []
    digest = hashlib.sha256()
    for strain in definition["strain_levels"]:
        strain = float(strain)
        sim.positions.copy_(original)
        sim.positions[:, 2] = z_bottom + (original[:, 2] - z_bottom) * (1.0 - strain)
        force, _ = sim.forces(update_fracture=False)
        reaction_n = abs(float(force[top_mask, 2].sum()))
        stress_pa = reaction_n / area
        modulus_pa = stress_pa / strain
        if not all(math.isfinite(v) for v in (strain, reaction_n, stress_pa, modulus_pa)):
            raise FloatingPointError(f"non-finite specimen result at strain={strain}")
        row = {
            "strain": strain,
            "reaction_force_N": reaction_n,
            "engineering_stress_Pa": stress_pa,
            "secant_modulus_Pa": modulus_pa,
            "broken_bonds": sim.broken_bonds,
            "particle_count": len(sim.positions),
            "bond_count": len(sim.pairs),
        }
        rows.append(row)
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
    return rows, digest.hexdigest()


def run_specimen_response_benchmarks(output: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "benchmarks" / "dem3d" / "specimen_response_cases.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "tensordem-dem3d-specimen-response-v1":
        raise ValueError("unsupported specimen-response benchmark protocol")
    case = next((item for item in manifest["cases"]
                 if item["id"] == "affine_compression_patch"), None)
    if case is None:
        raise ValueError("manifest is missing affine_compression_patch")
    rows, signature = _run_once(case)
    repeat_rows, repeat_signature = _run_once(case)
    acceptance = case["acceptance"]
    moduli = [float(row["secant_modulus_Pa"]) for row in rows]
    mean_modulus = sum(moduli) / len(moduli)
    spread = (max(moduli) - min(moduli)) / max(abs(mean_modulus), 1.0e-30)
    max_row_delta = max(
        abs(float(a["engineering_stress_Pa"]) - float(b["engineering_stress_Pa"]))
        / max(abs(float(a["engineering_stress_Pa"])), 1.0e-30)
        for a, b in zip(rows, repeat_rows)
    )
    damage = max(int(row["broken_bonds"]) for row in rows)
    checks = {
        "positive_stress": all(float(row["engineering_stress_Pa"]) > float(acceptance["minimum_stress_Pa"]) for row in rows),
        "near_linear_response": spread <= float(acceptance["maximum_secant_modulus_relative_spread"]),
        "no_damage": damage <= int(acceptance["maximum_broken_bonds"]),
        "repeatable": signature == repeat_signature and max_row_delta <= float(acceptance["repeat_relative_tolerance"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "protocol": manifest["protocol"],
        "case_id": case["id"],
        "scope": manifest["scope"],
        "verdict": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "mean_secant_modulus_Pa": mean_modulus,
        "relative_modulus_spread": spread,
        "repeat_relative_stress_delta": max_row_delta,
        "signature_sha256": signature,
        "repeat_signature_sha256": repeat_signature,
        "rows": rows,
        "not_yet_supported": manifest.get("not_yet_supported", []),
        "interpretation": (
            "PASS verifies this prescribed-kinematics regression only. The apparent modulus "
            "is not a calibrated material property; this is not standard UCS, Brazilian, "
            "three-point-bending, experimental, or ship-scale validation."
        ),
    }
    (output / "specimen_response_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "specimen_stress_strain.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-specimen-response"))
    args = parser.parse_args(argv)
    report = run_specimen_response_benchmarks(args.output)
    print(json.dumps({
        "case_id": report["case_id"], "verdict": report["verdict"],
        "checks": report["checks"], "mean_secant_modulus_Pa": report["mean_secant_modulus_Pa"],
        "json": str(args.output / "specimen_response_report.json"),
        "csv": str(args.output / "specimen_stress_strain.csv"),
    }, indent=2, sort_keys=True))
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
