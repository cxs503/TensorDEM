"""Run a physical-domain-preserving DEM3D particle-resolution sensitivity study.

Refinement factor m preserves the initial block dimensions by using
n'=(n-1)*m+1 and particle radius r'=r/m. Bond/contact stiffness and viscous
damping are scaled to maintain first-order continuum-like stiffness/damping
scales. This is a sensitivity protocol, not a substitute for calibration.
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


def _validate_factors(factors: Sequence[int]) -> tuple[int, ...]:
    values = tuple(factors)
    if len(values) < 2:
        raise ValueError("provide at least two resolution factors")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
        raise ValueError("resolution factors must be positive integers")
    if len(set(values)) != len(values):
        raise ValueError("resolution factors must be unique")
    return tuple(sorted(values))


def refined_config(config: DEM3DConfig, factor: int, base_dt: float) -> DEM3DConfig:
    """Create a finer lattice while preserving block extents and tool geometry."""
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("factor must be a positive integer")
    return replace(
        config,
        nx=(config.nx - 1) * factor + 1,
        ny=(config.ny - 1) * factor + 1,
        nz=(config.nz - 1) * factor + 1,
        radius=config.radius / factor,
        bond_stiffness=config.bond_stiffness / factor,
        contact_stiffness=config.contact_stiffness / factor,
        contact_damping=config.contact_damping / (factor * factor),
        drag=config.drag / (factor * factor),
        dt=base_dt / factor,
    )


def run_resolution_study(
    config: DEM3DConfig,
    *,
    base_steps: int = 200,
    resolution_factors: Sequence[int] = (1, 2),
    sample_every: int = 10,
) -> dict[str, Any]:
    """Compare mesh/particle resolutions at the same physical final time."""
    if isinstance(base_steps, bool) or not isinstance(base_steps, int) or base_steps < 1:
        raise ValueError("base_steps must be a positive integer")
    if isinstance(sample_every, bool) or not isinstance(sample_every, int) or sample_every < 1:
        raise ValueError("sample_every must be a positive integer")
    factors = _validate_factors(resolution_factors)
    if config.device != "cpu":
        raise ValueError("the reproducibility study currently requires device='cpu'")

    torch.set_num_threads(1)
    base_dt = config.dt if config.dt is not None else config.recommended_dt
    if base_dt > config.recommended_dt:
        raise ValueError("base dt exceeds the solver's recommended limit")
    target_time = base_steps * base_dt
    levels: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []

    for factor in factors:
        level_config = refined_config(config, factor, base_dt)
        steps = base_steps * factor
        sim = EnergyAuditedIceDEM3D(level_config)
        peak_abs_reaction = 0.0
        peak_signed_reaction = 0.0
        max_abs_energy_residual = 0.0
        initial_positions = sim.positions.clone()
        initial_diag = sim.diagnostics()

        def record(step: int, row: dict[str, Any]) -> None:
            values = [float(row[k]) for k in (
                "time", "reaction_x", "reaction_y", "reaction_z",
                "mechanical_energy_J", "energy_balance_residual_J",
            )]
            if not all(math.isfinite(value) for value in values):
                raise AssertionError(f"non-finite diagnostic at factor {factor}, step {step}")
            history.append({
                "resolution_factor": factor,
                "step": step,
                "time_s": float(row["time"]),
                "dt_s": level_config.dt,
                "particle_count": int(row["particle_count"]),
                "reaction_x_N": float(row["reaction_x"]),
                "reaction_y_N": float(row["reaction_y"]),
                "reaction_z_N": float(row["reaction_z"]),
                "reaction_magnitude_N": math.sqrt(
                    float(row["reaction_x"]) ** 2 + float(row["reaction_y"]) ** 2
                    + float(row["reaction_z"]) ** 2
                ),
                "broken_bonds": int(row["broken_bonds"]),
                "mechanical_energy_J": float(row["mechanical_energy_J"]),
                "energy_balance_residual_J": float(row["energy_balance_residual_J"]),
            })

        record(0, initial_diag)
        for step in range(1, steps + 1):
            sim.step()
            row = sim.diagnostics()
            rz = float(row["reaction_z"])
            if abs(rz) > peak_abs_reaction:
                peak_abs_reaction, peak_signed_reaction = abs(rz), rz
            max_abs_energy_residual = max(
                max_abs_energy_residual, abs(float(row["energy_balance_residual_J"]))
            )
            if step % sample_every == 0 or step == steps:
                record(step, row)

        final = sim.diagnostics()
        extent = [
            (level_config.nx - 1) * 2 * level_config.radius,
            (level_config.ny - 1) * 2 * level_config.radius,
            (level_config.nz - 1) * 2 * level_config.radius,
        ]
        levels.append({
            "resolution_factor": factor,
            "nx": level_config.nx,
            "ny": level_config.ny,
            "nz": level_config.nz,
            "particle_count": int(final["particle_count"]),
            "bond_count": int(sim.pairs.shape[0]),
            "particle_radius_m": level_config.radius,
            "block_extent_x_m": extent[0],
            "block_extent_y_m": extent[1],
            "block_extent_z_m": extent[2],
            "dt_s": level_config.dt,
            "steps": steps,
            "final_time_s": float(sim.time),
            "peak_abs_reaction_z_N": peak_abs_reaction,
            "peak_reaction_z_signed_N": peak_signed_reaction,
            "final_reaction_z_N": float(final["reaction_z"]),
            "broken_bonds": int(final["broken_bonds"]),
            "final_mechanical_energy_J": float(final["mechanical_energy_J"]),
            "final_energy_balance_residual_J": float(final["energy_balance_residual_J"]),
            "max_abs_energy_balance_residual_J": max_abs_energy_residual,
            "initial_state_unchanged": bool(torch.equal(
                initial_positions, sim.initial_positions
            )),
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
        level["final_reaction_abs_delta_vs_finest_N"] = abs(
            float(level["final_reaction_z_N"]) - float(finest["final_reaction_z_N"])
        )

    final_times = [float(row["final_time_s"]) for row in levels]
    return {
        "protocol": "tensordem-dem3d-resolution-sensitivity-v1",
        "interpretation": (
            "Resolution sensitivity only. Refinement changes particle count and the "
            "discrete fracture network; stiffness/damping scaling is an explicit "
            "continuum-like assumption requiring calibration. Agreement is not proof "
            "of convergence or experimental validation."
        ),
        "base_config": {**config.__dict__, "dt": base_dt},
        "base_steps": base_steps,
        "sample_every": sample_every,
        "target_final_time_s": target_time,
        "same_final_time_within_tolerance": all(
            math.isclose(value, target_time, rel_tol=1e-10, abs_tol=1e-14)
            for value in final_times
        ),
        "levels": levels,
        "history": history,
        "finest_resolution_factor": factors[-1],
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in report.items() if key != "history"}
    (output / "resolution_sensitivity_3d.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for filename, key in (
        ("resolution_levels_3d.csv", "levels"),
        ("resolution_history_3d.csv", "history"),
    ):
        rows = report.get(key, [])
        if not rows:
            continue
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DEM3D particle-resolution sensitivity study")
    parser.add_argument("--nx", type=int, default=5)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--nz", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--factors", type=int, nargs="+", default=[1, 2],
                        help="resolution factors, e.g. 1 2 (particle count grows cubically)")
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--radius", type=float, default=0.025)
    parser.add_argument("--tool-radius", type=float, default=0.1)
    parser.add_argument("--breaking-strain", type=float, default=0.015)
    parser.add_argument("--shear-breaking-strain", type=float, default=0.03)
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-resolution"))
    args = parser.parse_args(argv)
    try:
        config = DEM3DConfig(
            nx=args.nx, ny=args.ny, nz=args.nz, tool_speed=args.speed,
            radius=args.radius, tool_radius=args.tool_radius,
            breaking_strain=args.breaking_strain,
            shear_breaking_strain=args.shear_breaking_strain,
        )
        report = run_resolution_study(
            config, base_steps=args.steps, resolution_factors=args.factors,
            sample_every=args.sample_every,
        )
        write_report(report, args.output)
    except (ValueError, RuntimeError, AssertionError) as exc:
        parser.error(str(exc))
    printable = {key: value for key, value in report.items() if key != "history"}
    printable["history_rows_written"] = len(report["history"])
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
