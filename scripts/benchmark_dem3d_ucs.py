"""Dynamic uniaxial compression benchmark for the 3-D bonded-particle DEM.

The case uses opposed moving plane platens and leaves lateral particle motion
unconstrained. It is a numerical integration/regression test, not a calibrated
laboratory UCS test or a physical ice-strength prediction.
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


def _run_once(case: dict[str, Any], steps: int) -> tuple[list[dict[str, float | int]], str, dict[str, Any]]:
    p = case["config"]
    radius = float(p["radius_m"])
    speed = float(p["loading_speed_m_s"])
    nx, ny, nz = int(p["nx"]), int(p["ny"]), int(p["nz"])
    sim = IceDEM3D(DEM3DConfig(
        nx=nx, ny=ny, nz=nz,
        radius=radius,
        density=float(p["density_kg_m3"]),
        bond_stiffness=float(p["bond_stiffness_N_m"]),
        contact_stiffness=float(p["contact_stiffness_N_m"]),
        breaking_strain=float(p["breaking_strain"]),
        platen_stiffness=float(p["platen_stiffness_N_m"]),
        platen_damping=float(p["platen_damping_Ns_m"]),
        top_platen_enabled=True,
        bottom_platen_enabled=True,
        top_platen_velocity=-speed,
        bottom_platen_velocity=speed,
        drag=0.1,
        contact_damping=0.5,
        fix_x_edges=False,
        fix_bottom=False,
    ))
    # This protocol isolates plane-platen compression; disable the spherical tool.
    sim.tool_start[:] = torch.tensor([1.0e6, 1.0e6, 1.0e6], dtype=sim.dtype)
    sim.tool_velocity.zero_()

    initial_height = nz * 2.0 * radius
    gross_area = (nx * 2.0 * radius) * (ny * 2.0 * radius)
    initial_x_span = float(sim.positions[:, 0].max() - sim.positions[:, 0].min()) + 2 * radius
    initial_y_span = float(sim.positions[:, 1].max() - sim.positions[:, 1].min()) + 2 * radius
    rows: list[dict[str, float | int]] = []
    digest = hashlib.sha256()
    for step in range(steps + 1):
        if step:
            sim.step()
        d = sim.diagnostics()
        top_load = abs(float(d["top_platen_reaction_z_N"]))
        bottom_load = abs(float(d["bottom_platen_reaction_z_N"]))
        load = 0.5 * (top_load + bottom_load)
        closure = 2.0 * speed * float(d["time"])
        strain = closure / initial_height
        row: dict[str, float | int] = {
            "step": step,
            "time_s": float(d["time"]),
            "platen_closure_m": closure,
            "axial_engineering_strain": strain,
            "top_reaction_abs_N": top_load,
            "bottom_reaction_abs_N": bottom_load,
            "mean_compressive_load_N": load,
            "engineering_stress_Pa": load / gross_area,
            "broken_bonds": int(d["broken_bonds"]),
            "kinetic_energy_J": float(d["kinetic_energy_J"]),
            "bond_elastic_energy_J": float(d["bond_elastic_energy_J"]),
            "platen_contact_energy_J": float(d["platen_contact_energy_J"]),
            "mechanical_energy_J": float(d["mechanical_energy_J"]),
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"non-finite UCS history at step {step}")
        rows.append(row)
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    peak = max(rows, key=lambda row: float(row["engineering_stress_Pa"]))
    lateral_x_span = float(sim.positions[:, 0].max() - sim.positions[:, 0].min()) + 2 * radius
    lateral_y_span = float(sim.positions[:, 1].max() - sim.positions[:, 1].min()) + 2 * radius
    metadata = {
        "particle_count": len(sim.positions),
        "bond_count": len(sim.pairs),
        "initial_height_m": initial_height,
        "initial_gross_area_m2": gross_area,
        "initial_x_span_m": initial_x_span,
        "initial_y_span_m": initial_y_span,
        "final_x_span_m": lateral_x_span,
        "final_y_span_m": lateral_y_span,
        "lateral_x_engineering_strain": (lateral_x_span - initial_x_span) / initial_x_span,
        "lateral_y_engineering_strain": (lateral_y_span - initial_y_span) / initial_y_span,
        "peak_load_N": float(peak["mean_compressive_load_N"]),
        "peak_engineering_stress_Pa": float(peak["engineering_stress_Pa"]),
        "strain_at_peak_stress": float(peak["axial_engineering_strain"]),
        "final_broken_bonds": int(sim.broken_bonds),
        "signature_sha256": digest.hexdigest(),
    }
    return rows, digest.hexdigest(), metadata


def run_ucs_benchmark(output: Path, *, steps: int | None = None) -> dict[str, Any]:
    """Run the named UCS-style case twice and persist JSON/CSV histories."""
    torch.set_num_threads(1)
    manifest_path = ROOT / "benchmarks" / "dem3d" / "ucs_cases.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "tensordem-dem3d-ucs-v1":
        raise ValueError("unsupported UCS benchmark protocol")
    case = next((item for item in manifest["cases"] if item["id"] == "ucs_dynamic_smoke"), None)
    if case is None:
        raise ValueError("manifest is missing ucs_dynamic_smoke")
    nsteps = int(case["config"]["steps"] if steps is None else steps)
    if nsteps < 1:
        raise ValueError("steps must be positive")
    rows, signature, metrics = _run_once(case, nsteps)
    repeat_rows, repeat_signature, repeat_metrics = _run_once(case, nsteps)
    acceptance = case["acceptance"]
    checks = {
        "top_and_bottom_reactions_activated": (
            max(float(row["top_reaction_abs_N"]) for row in rows) > float(acceptance["minimum_peak_load_N"])
            and max(float(row["bottom_reaction_abs_N"]) for row in rows) > float(acceptance["minimum_peak_load_N"])
        ),
        "positive_compressive_stress": metrics["peak_engineering_stress_Pa"] > float(acceptance["minimum_peak_stress_Pa"]),
        "finite_history": all(math.isfinite(float(v)) for row in rows for v in row.values()),
        "repeatable_history": (
            signature == repeat_signature
            and metrics["peak_engineering_stress_Pa"] == repeat_metrics["peak_engineering_stress_Pa"]
            and metrics["final_broken_bonds"] == repeat_metrics["final_broken_bonds"]
        ),
        "free_lateral_motion_configured": not bool(case["config"].get("fix_x_edges", False)),
    }
    failed = [name for name, passed in checks.items() if not passed]
    report: dict[str, Any] = {
        "protocol": manifest["protocol"],
        "case_id": case["id"],
        "scope": manifest["scope"],
        "verdict": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "steps": nsteps,
        "dt_s": DEM3DConfig(
            nx=int(case["config"]["nx"]), ny=int(case["config"]["ny"]), nz=int(case["config"]["nz"]),
            radius=float(case["config"]["radius_m"]),
            density=float(case["config"]["density_kg_m3"]),
            bond_stiffness=float(case["config"]["bond_stiffness_N_m"]),
            contact_stiffness=float(case["config"]["contact_stiffness_N_m"]),
            breaking_strain=float(case["config"]["breaking_strain"]),
            platen_stiffness=float(case["config"]["platen_stiffness_N_m"]),
            platen_damping=float(case["config"]["platen_damping_Ns_m"]),
            top_platen_enabled=True, bottom_platen_enabled=True,
            top_platen_velocity=-float(case["config"]["loading_speed_m_s"]),
            bottom_platen_velocity=float(case["config"]["loading_speed_m_s"]),
        ).recommended_dt,
        **metrics,
        "repeat_signature_sha256": repeat_signature,
        "rows": rows,
        "interpretation_limits": manifest["interpretation_limits"],
        "interpretation": (
            "PASS confirms finite, repeatable dynamic platen-compression histories and active reactions. "
            "The peak stress is a numerical output for this discrete configuration, not calibrated ice strength."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "ucs_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "ucs_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-ucs"))
    args = parser.parse_args(argv)
    report = run_ucs_benchmark(args.output, steps=args.steps)
    print(json.dumps({
        "case_id": report["case_id"],
        "verdict": report["verdict"],
        "checks": report["checks"],
        "steps": report["steps"],
        "peak_load_N": report["peak_load_N"],
        "peak_engineering_stress_Pa": report["peak_engineering_stress_Pa"],
        "strain_at_peak_stress": report["strain_at_peak_stress"],
        "final_broken_bonds": report["final_broken_bonds"],
        "json": str(args.output / "ucs_report.json"),
        "csv": str(args.output / "ucs_history.csv"),
    }, indent=2, sort_keys=True))
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
