"""DEM3D three-point-bending particle-resolution sensitivity campaign.

This is a discretization-sensitivity diagnostic, not experimental validation.
The physical envelope is approximately held fixed by refining the lattice and
reducing particle radius together. Bond/contact parameters are intentionally
not recalibrated; their resolution dependence is therefore reported, not hidden.
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


def _relative_delta(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-30)


def _run_level(base: dict[str, Any], factor: int, base_steps: int) -> dict[str, Any]:
    radius = float(base["radius_m"]) / factor
    nx = (int(base["nx"]) - 1) * factor + 1
    ny = (int(base["ny"]) - 1) * factor + 1
    nz = (int(base["nz"]) - 1) * factor + 1
    cfg = DEM3DConfig(
        nx=nx, ny=ny, nz=nz, radius=radius,
        density=float(base["density_kg_m3"]),
        bond_stiffness=float(base["bond_stiffness_N_m"]),
        contact_stiffness=float(base["contact_stiffness_N_m"]),
        breaking_strain=float(base["breaking_strain"]),
        shear_breaking_strain=float(base["shear_breaking_strain"]),
        contact_damping=float(base["contact_damping_Ns_m"]),
        drag=float(base["drag"]), fix_x_edges=False, fix_bottom=False,
        tool_gap=1.0, tool_radius=0.01, tool_speed=0.01,
    )
    # Preserve approximately the same physical ramp duration as the lattice
    # is refined. This is not a timestep-convergence study.
    steps = int(math.ceil(base_steps * factor ** 1.5))
    sim = IceDEM3D(cfg)
    sim.tool_start[:] = torch.tensor([1e6, 1e6, 1e6], dtype=sim.dtype)
    sim.tool_velocity.zero_()
    x = sim.initial_positions[:, 0]
    z = sim.initial_positions[:, 2]
    xv = torch.unique(x)
    zb, zt = z.min(), z.max()
    lx = xv[max(1, len(xv) // 4)]
    rx = xv[min(len(xv) - 2, (3 * len(xv)) // 4)]
    supports = (z == zb) & ((x == lx) | (x == rx))
    loaded = (z == zt) & (x == xv[len(xv) // 2])
    if int(supports.sum()) < 2 or int(loaded.sum()) < 1:
        raise ValueError(f"invalid support/loading geometry at factor {factor}")
    sim.fixed[:] = supports
    sim.positions[sim.fixed] = sim.initial_positions[sim.fixed]
    peak_load = float(base["peak_load_N"])
    digest = hashlib.sha256()
    peak_reaction = 0.0
    peak_deflection = 0.0
    rows: list[dict[str, Any]] = []
    for step in range(steps + 1):
        applied = peak_load * step / steps
        loads = torch.zeros_like(sim.positions)
        loads[loaded, 2] = -applied / int(loaded.sum())
        if step:
            sim.step(external_forces=loads)
        force, _ = sim.forces(loads, update_fracture=False)
        reaction = -force[sim.fixed].sum(dim=0)
        deflection = float((sim.positions[loaded, 2] - sim.initial_positions[loaded, 2]).mean())
        diag = sim.diagnostics()
        row = {
            "step": step, "time_s": float(diag["time"]),
            "applied_load_N": applied, "support_reaction_z_N": float(reaction[2]),
            "midspan_deflection_m": deflection,
            "broken_bonds": int(diag["broken_bonds"]),
            "mechanical_energy_J": float(diag["mechanical_energy_J"]),
        }
        if not all(math.isfinite(float(v)) for v in row.values()):
            raise FloatingPointError(f"non-finite output at factor={factor}, step={step}")
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        rows.append(row)
        peak_reaction = max(peak_reaction, abs(float(reaction[2])))
        peak_deflection = max(peak_deflection, abs(deflection))
    return {
        "resolution_factor": factor, "nx": nx, "ny": ny, "nz": nz,
        "particle_count": len(sim.positions), "bond_count": len(sim.pairs),
        "steps": steps, "dt_s": sim.dt, "duration_s": float(rows[-1]["time_s"]),
        "peak_support_reaction_abs_N": peak_reaction,
        "peak_midspan_deflection_abs_m": peak_deflection,
        "broken_bonds": int(sim.broken_bonds),
        "finite": True, "signature_sha256": digest.hexdigest(), "history": rows,
    }


def run_resolution_campaign(
    output: Path, *, base_steps: int = 60,
    factors: tuple[int, ...] = (1, 2),
) -> dict[str, Any]:
    if isinstance(base_steps, bool) or not isinstance(base_steps, int) or base_steps < 1:
        raise ValueError("base_steps must be a positive integer")
    if not factors or any(isinstance(f, bool) or not isinstance(f, int) or f < 1 for f in factors):
        raise ValueError("factors must be positive integers")
    if len(set(factors)) != len(factors):
        raise ValueError("factors must be unique")
    factors = tuple(sorted(factors))
    torch.set_num_threads(1)
    manifest = json.loads((ROOT / "benchmarks/dem3d/three_point_bending_cases.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != "tensordem-dem3d-three-point-bending-v1":
        raise ValueError("unsupported three-point-bending protocol")
    case = manifest["cases"][0]
    records = [_run_level(case["config"], factor, base_steps) for factor in factors]
    reference = records[-1]
    for record in records:
        record["peak_reaction_relative_delta_vs_finest"] = _relative_delta(
            record["peak_support_reaction_abs_N"], reference["peak_support_reaction_abs_N"])
        record["peak_deflection_relative_delta_vs_finest"] = _relative_delta(
            record["peak_midspan_deflection_abs_m"], reference["peak_midspan_deflection_abs_m"])
        record["fracture_count_changed_vs_finest"] = record["broken_bonds"] != reference["broken_bonds"]
    hard_pass = all(r["finite"] and r["peak_support_reaction_abs_N"] > 0
                    and r["peak_midspan_deflection_abs_m"] > 0 for r in records)
    summary = [{k: v for k, v in r.items() if k != "history"} for r in records]
    report = {
        "protocol": "tensordem-dem3d-bending-resolution-sensitivity-v1",
        "verdict": "PASS" if hard_pass else "FAIL",
        "base_steps_at_factor_1": base_steps,
        "reference": "finest resolution in this campaign",
        "cases": summary,
        "interpretation": (
            "PASS means each tested discretization produced finite nonzero numerical load transfer. "
            "Relative deltas are sensitivity indicators, not experimental errors or universal acceptance "
            "criteria. Particle radius and lattice counts are refined together to approximately preserve "
            "the specimen envelope, but bond/contact stiffness is not recalibrated. Changes in fracture "
            "count may reflect discrete crack-path changes. This campaign does not establish material "
            "calibration or physical convergence."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "bending_resolution_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "bending_resolution_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (output / "bending_resolution_history.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["resolution_factor"] + list(records[0]["history"][0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for row in record["history"]:
                writer.writerow({"resolution_factor": record["resolution_factor"], **row})
    if not hard_pass:
        raise RuntimeError("bending resolution campaign failed its finite/load-transfer gates")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-bending-resolution"))
    parser.add_argument("--base-steps", type=int, default=60)
    args = parser.parse_args(argv)
    report = run_resolution_campaign(args.output, base_steps=args.base_steps)
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
