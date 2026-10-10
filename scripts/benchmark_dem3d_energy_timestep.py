"""Timestep sensitivity campaign for DEM3D discrete energy accounting.

The residual is a diagnostic, not an exact-conservation assertion.
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

from tensordem.dem3d import DEM3DConfig
from tensordem.dem3d_energy import EnergyAuditedIceDEM3D


def _signature(sim: EnergyAuditedIceDEM3D, row: dict[str, Any]) -> str:
    payload = {
        "step_count": sim.step_count,
        "positions": [round(float(v), 12) for v in sim.positions.flatten().cpu()],
        "velocities": [round(float(v), 12) for v in sim.velocities.flatten().cpu()],
        "metrics": {k: round(float(row[k]), 12) for k in (
            "tool_work_cumulative_J", "drag_dissipation_cumulative_J",
            "particle_damping_cumulative_J", "tool_damping_cumulative_J",
            "fracture_release_cumulative_J", "energy_balance_residual_J",
        )},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _run_case(factor: float, base_steps: int) -> dict[str, Any]:
    template = DEM3DConfig(
        nx=3, ny=3, nz=2, drag=0.0, fix_x_edges=False,
        tool_gap=0.0, breaking_strain=0.5, shear_breaking_strain=0.5,
    )
    dt = template.recommended_dt * factor
    config = DEM3DConfig(
        nx=template.nx, ny=template.ny, nz=template.nz,
        radius=template.radius, density=template.density,
        bond_stiffness=template.bond_stiffness,
        contact_stiffness=template.contact_stiffness,
        breaking_strain=template.breaking_strain,
        shear_breaking_strain=template.shear_breaking_strain,
        contact_damping=template.contact_damping, drag=template.drag,
        tool_radius=template.tool_radius, tool_speed=template.tool_speed,
        tool_gap=template.tool_gap, dt=dt, device="cpu",
        fix_x_edges=template.fix_x_edges, fix_bottom=template.fix_bottom,
    )
    steps = int(math.ceil(base_steps / factor))
    sim = EnergyAuditedIceDEM3D(config)
    history: list[dict[str, float | int]] = []
    for step in range(steps):
        sim.step()
        if step == 0 or (step + 1) % max(1, steps // 20) == 0 or step + 1 == steps:
            row = sim.diagnostics()
            row["step"] = step + 1
            history.append(row)
    final = sim.diagnostics()
    signature = _signature(sim, final)
    repeat = EnergyAuditedIceDEM3D(config)
    for _ in range(steps):
        repeat.step()
    repeat_row = repeat.diagnostics()
    repeat_signature = _signature(repeat, repeat_row)
    denom = max(
        abs(sim.reference_mechanical_energy_J)
        + abs(sim.cumulative_tool_work_J)
        + abs(sim.cumulative_external_work_J)
        + sim.cumulative_fracture_release_J,
        1.0e-30,
    )
    values = {
        "dt_factor": factor,
        "dt_s": dt,
        "steps": steps,
        "physical_duration_s": steps * dt,
        "particle_count": len(sim.positions),
        "bond_count": len(sim.pairs),
        "broken_bonds": sim.broken_bonds,
        "tool_work_J": sim.cumulative_tool_work_J,
        "drag_dissipation_J": sim.cumulative_drag_dissipation_J,
        "particle_damping_J": sim.cumulative_particle_damping_J,
        "tool_damping_J": sim.cumulative_tool_damping_J,
        "fracture_release_J": sim.cumulative_fracture_release_J,
        "mechanical_energy_final_J": sim._mechanical_energy(),
        "energy_balance_residual_J": sim.energy_balance_residual_J,
        "energy_balance_residual_relative": sim.energy_balance_residual_J / denom,
        "signature_sha256": signature,
        "repeat_signature_sha256": repeat_signature,
        "repeat_verified": signature == repeat_signature,
        "finite": all(math.isfinite(float(v)) for k, v in values_safe(final).items()),
    }
    values["finite"] = values["finite"] and all(
        math.isfinite(float(values[k])) for k in (
            "tool_work_J", "drag_dissipation_J", "particle_damping_J",
            "tool_damping_J", "fracture_release_J", "mechanical_energy_final_J",
            "energy_balance_residual_J", "energy_balance_residual_relative",
        )
    )
    values["passed"] = bool(values["finite"] and values["repeat_verified"])
    return {"summary": values, "history": history}


def values_safe(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if isinstance(v, (float, int))}


def run_campaign(output: Path, factors: tuple[float, ...] = (1.0, 0.5, 0.25),
                 base_steps: int = 40) -> dict[str, Any]:
    if not factors or any(not math.isfinite(v) or v <= 0 or v > 1 for v in factors):
        raise ValueError("dt factors must be finite and in (0, 1]")
    if len(set(factors)) != len(factors):
        raise ValueError("dt factors must be unique")
    if isinstance(base_steps, bool) or base_steps < 1:
        raise ValueError("base_steps must be a positive integer")
    torch.set_num_threads(1)
    cases = [_run_case(float(f), base_steps) for f in factors]
    summary = [case["summary"] for case in cases]
    report = {
        "protocol": "tensordem-dem3d-energy-timestep-v1",
        "verdict": "PASS" if all(row["passed"] for row in summary) else "FAIL",
        "acceptance": [
            "all monitored energy terms and residuals are finite",
            "each timestep case repeats with an identical rounded state/metric signature",
        ],
        "cases": summary,
        "interpretation": (
            "This campaign checks finite bookkeeping and deterministic repeatability "
            "across timestep choices. It does not impose a universal residual threshold "
            "or claim exact energy conservation."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dem3d_energy_timestep_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "dem3d_energy_timestep_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    with (output / "dem3d_energy_timestep_history.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["dt_factor", "sample_index", *cases[0]["history"][0].keys()]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            for index, row in enumerate(case["history"]):
                writer.writerow({"dt_factor": case["summary"]["dt_factor"], "sample_index": index, **row})
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-energy-timestep"))
    parser.add_argument("--base-steps", type=int, default=40)
    args = parser.parse_args(argv)
    report = run_campaign(args.output, base_steps=args.base_steps)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
