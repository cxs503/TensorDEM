"""Integration smoke benchmark for the DEM3D moving plane-platen contact.

This validates contact activation, signed plate reactions, finite integration
state and repeatable outputs. It is not a standard UCS or material calibration.
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


def run_platen_compression(output: Path, *, steps: int | None = None) -> dict[str, Any]:
    torch.set_num_threads(1)
    manifest_path = ROOT / "benchmarks" / "dem3d" / "platen_compression_cases.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "tensordem-dem3d-platen-compression-v1":
        raise ValueError("unsupported platen compression benchmark protocol")
    case = manifest["cases"][0]
    p = case["config"]
    nsteps = int(p["steps"] if steps is None else steps)
    if nsteps < 1:
        raise ValueError("steps must be positive")
    speed = float(p["speed_m_s"])
    sim = IceDEM3D(DEM3DConfig(
        nx=int(p["nx"]), ny=int(p["ny"]), nz=int(p["nz"]),
        radius=float(p["radius_m"]), density=float(p["density_kg_m3"]),
        platen_stiffness=float(p["platen_stiffness_N_m"]),
        platen_damping=float(p["platen_damping_Ns_m"]),
        top_platen_enabled=True, bottom_platen_enabled=True,
        top_platen_velocity=-speed, bottom_platen_velocity=speed,
        drag=0.1, fix_x_edges=False, fix_bottom=False,
    ))
    # The sphere indenter is intentionally disabled for this fixture.
    sim.tool_start[:] = torch.tensor([1.0e6, 1.0e6, 1.0e6], dtype=sim.dtype)
    sim.tool_velocity.zero_()
    rows: list[dict[str, float | int]] = []
    digest = hashlib.sha256()
    for step in range(nsteps + 1):
        if step:
            sim.step()
        d = sim.diagnostics()
        row = {
            "step": step,
            "time_s": float(d["time"]),
            "top_platen_displacement_m": -speed * float(d["time"]),
            "bottom_platen_displacement_m": speed * float(d["time"]),
            "top_platen_reaction_z_N": float(d["top_platen_reaction_z_N"]),
            "bottom_platen_reaction_z_N": float(d["bottom_platen_reaction_z_N"]),
            "broken_bonds": int(d["broken_bonds"]),
            "kinetic_energy_J": float(d["kinetic_energy_J"]),
            "bond_elastic_energy_J": float(d["bond_elastic_energy_J"]),
            "platen_contact_energy_J": float(d["platen_contact_energy_J"]),
            "mechanical_energy_J": float(d["mechanical_energy_J"]),
        }
        if not all(math.isfinite(float(v)) for v in row.values()):
            raise FloatingPointError(f"non-finite platen benchmark output at step {step}")
        rows.append(row)
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
    peak_top = max(float(row["top_platen_reaction_z_N"]) for row in rows)
    peak_bottom_abs = max(abs(float(row["bottom_platen_reaction_z_N"])) for row in rows)
    acceptance = case["acceptance"]
    checks = {
        "top_reaction_activated": peak_top > float(acceptance["minimum_peak_top_reaction_N"]),
        "bottom_reaction_activated": peak_bottom_abs > float(acceptance["minimum_peak_bottom_reaction_abs_N"]),
        "finite_state": all(math.isfinite(float(row["mechanical_energy_J"])) for row in rows),
    }
    failed = [name for name, ok in checks.items() if not ok]
    report = {
        "protocol": manifest["protocol"], "case_id": case["id"],
        "verdict": "PASS" if not failed else "FAIL",
        "checks": checks, "failed_checks": failed,
        "steps": nsteps, "dt_s": sim.dt, "particle_count": len(sim.positions),
        "bond_count": len(sim.pairs), "broken_bonds": sim.broken_bonds,
        "peak_top_platen_reaction_N": peak_top,
        "peak_bottom_platen_reaction_abs_N": peak_bottom_abs,
        "signature_sha256": digest.hexdigest(), "rows": rows,
        "interpretation": (
            "PASS means the moving plane-wall contact and reaction history ran numerically. "
            "It does not establish standard UCS strength, physical material calibration, "
            "experimental validity, or ship-scale accuracy."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "platen_compression_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "platen_compression_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-platen-compression"))
    args = parser.parse_args(argv)
    report = run_platen_compression(args.output, steps=args.steps)
    print(json.dumps({
        "case_id": report["case_id"], "verdict": report["verdict"],
        "checks": report["checks"], "steps": report["steps"],
        "peak_top_platen_reaction_N": report["peak_top_platen_reaction_N"],
        "peak_bottom_platen_reaction_abs_N": report["peak_bottom_platen_reaction_abs_N"],
        "json": str(args.output / "platen_compression_report.json"),
        "csv": str(args.output / "platen_compression_history.csv"),
    }, indent=2, sort_keys=True))
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
