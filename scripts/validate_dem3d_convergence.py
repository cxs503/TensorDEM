"""Run a same-physical-time DEM3D time-step sensitivity study.

Example:
  python scripts/validate_dem3d_convergence.py --steps 1000 --output results-dem3d-convergence

The report is a numerical sensitivity study, not material calibration or proof of
asymptotic convergence. All runs use the same physical configuration and final
time; only dt and the corresponding number of steps change.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.dem3d import DEM3DConfig
from tensordem.dem3d_energy import EnergyAuditedIceDEM3D


def run_convergence_study(
    config: DEM3DConfig,
    *,
    base_steps: int = 1000,
    refinement_factors: Sequence[int] = (1, 2, 4),
) -> dict[str, Any]:
    """Run integer-refinement levels to the same final physical time."""
    if isinstance(base_steps, bool) or base_steps < 1:
        raise ValueError("base_steps must be a positive integer")
    factors = tuple(refinement_factors)
    if len(factors) < 2 or any(
        isinstance(f, bool) or not isinstance(f, int) or f < 1 for f in factors
    ):
        raise ValueError("provide at least two positive integer refinement factors")
    if len(set(factors)) != len(factors):
        raise ValueError("refinement factors must be unique")
    factors = tuple(sorted(factors))
    if config.device != "cpu":
        raise ValueError("the reproducibility study currently requires device='cpu'")

    torch.set_num_threads(1)
    base_dt = config.dt if config.dt is not None else config.recommended_dt
    if base_dt > config.recommended_dt:
        raise ValueError("base dt exceeds the solver's recommended limit")
    final_time = base_steps * base_dt
    levels: list[dict[str, Any]] = []

    for factor in factors:
        dt = base_dt / factor
        level_config = replace(config, dt=dt)
        steps = int(round(final_time / dt))
        sim = EnergyAuditedIceDEM3D(level_config)
        peak_reaction = 0.0
        peak_reaction_signed = 0.0
        max_energy_residual_abs = 0.0
        for _ in range(steps):
            sim.step()
            row = sim.diagnostics()
            rz = float(row["reaction_z"])
            if not math.isfinite(rz):
                raise AssertionError(f"non-finite reaction at factor {factor}")
            if abs(rz) > peak_reaction:
                peak_reaction = abs(rz)
                peak_reaction_signed = rz
            residual = abs(float(row["energy_balance_residual_J"]))
            max_energy_residual_abs = max(max_energy_residual_abs, residual)
        final = sim.diagnostics()
        levels.append({
            "refinement_factor": factor,
            "dt_s": dt,
            "steps": steps,
            "final_time_s": float(sim.time),
            "peak_abs_reaction_z_N": peak_reaction,
            "peak_reaction_z_signed_N": peak_reaction_signed,
            "final_reaction_z_N": float(final["reaction_z"]),
            "broken_bonds": int(final["broken_bonds"]),
            "final_mechanical_energy_J": float(final["mechanical_energy_J"]),
            "final_energy_balance_residual_J": float(final["energy_balance_residual_J"]),
            "max_abs_energy_balance_residual_J": max_energy_residual_abs,
        })

    finest = levels[-1]
    force_reference = max(abs(float(finest["peak_abs_reaction_z_N"])), 1e-12)
    for level in levels:
        level["peak_force_abs_delta_vs_finest_N"] = abs(
            float(level["peak_abs_reaction_z_N"]) - float(finest["peak_abs_reaction_z_N"])
        )
        level["peak_force_relative_delta_vs_finest"] = (
            level["peak_force_abs_delta_vs_finest_N"] / force_reference
        )
        level["broken_bond_delta_vs_finest"] = (
            int(level["broken_bonds"]) - int(finest["broken_bonds"])
        )

    return {
        "protocol": "tensordem-dem3d-timestep-sensitivity-v1",
        "interpretation": (
            "Sensitivity report only. Compare force histories, damage and energy residuals "
            "across refinements; agreement is not material calibration or proof of convergence."
        ),
        "config": {**config.__dict__, "dt": base_dt},
        "base_steps": base_steps,
        "base_dt_s": base_dt,
        "target_final_time_s": final_time,
        "levels": levels,
        "finest_refinement_factor": factors[-1],
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "convergence_3d.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    levels = report["levels"]
    with (output / "convergence_3d.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(levels[0]))
        writer.writeheader()
        writer.writerows(levels)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DEM3D time-step sensitivity study")
    parser.add_argument("--nx", type=int, default=5)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--nz", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000,
                        help="number of steps at the coarsest refinement level")
    parser.add_argument("--factors", type=int, nargs="+", default=[1, 2, 4],
                        help="integer dt refinement factors, e.g. 1 2 4")
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--radius", type=float, default=0.025)
    parser.add_argument("--tool-radius", type=float, default=0.1)
    parser.add_argument("--breaking-strain", type=float, default=0.015)
    parser.add_argument("--shear-breaking-strain", type=float, default=0.03)
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-convergence"))
    args = parser.parse_args(argv)
    try:
        config = DEM3DConfig(
            nx=args.nx, ny=args.ny, nz=args.nz, tool_speed=args.speed,
            radius=args.radius, tool_radius=args.tool_radius,
            breaking_strain=args.breaking_strain,
            shear_breaking_strain=args.shear_breaking_strain,
        )
        report = run_convergence_study(
            config, base_steps=args.steps, refinement_factors=args.factors
        )
        write_report(report, args.output)
    except (ValueError, RuntimeError, AssertionError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
