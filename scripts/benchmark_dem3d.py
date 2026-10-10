"""Reproducible 3-D spherical-indenter DEM verification campaign.

Example:
  python scripts/benchmark_dem3d.py --steps 400 --output results-dem3d-benchmark
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

# Permit direct execution from a source checkout without an editable install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.dem3d import DEM3DConfig
from tensordem.dem3d_energy import EnergyAuditedIceDEM3D


def _tensor_digest(digest: Any, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    # Avoid Tensor.numpy(), so the benchmark also runs when PyTorch's optional
    # NumPy bridge is unavailable in a minimal CI environment.
    digest.update(bytes(value.view(torch.uint8).reshape(-1).tolist()))


def run_benchmark(
    output: Path,
    config: DEM3DConfig,
    steps: int = 400,
    sample_every: int = 10,
    verify_repeat: bool = False,
) -> dict[str, Any]:
    if steps < 1 or sample_every < 1:
        raise ValueError("steps and sample_every must be positive")
    if config.device != "cpu":
        raise ValueError("the reproducibility protocol currently requires device='cpu'")
    torch.set_num_threads(1)

    def execute(write_history: bool) -> tuple[list[dict[str, Any]], str, int]:
        sim = EnergyAuditedIceDEM3D(config)
        rows: list[dict[str, Any]] = []
        digest = hashlib.sha256()
        max_broken = 0
        for step in range(steps + 1):
            if step:
                sim.step()
            max_broken = max(max_broken, sim.broken_bonds)
            if step % sample_every == 0 or step == steps:
                row = sim.diagnostics()
                if row["broken_bonds"] < max_broken:
                    raise AssertionError("bond damage decreased during the benchmark")
                if not all(
                    isinstance(value, (int, float)) and
                    (isinstance(value, int) or torch.isfinite(torch.tensor(value)).item())
                    for value in row.values()
                ):
                    raise AssertionError(f"non-finite diagnostic at step {step}")
                rows.append({"step": step, **row})
                digest.update(json.dumps(rows[-1], sort_keys=True, separators=(",", ":")).encode())
        for tensor in (sim.positions, sim.velocities, sim.alive, sim.failure_mode,
                       sim.failure_time_s, sim.failure_extension_m, sim.failure_energy_J):
            _tensor_digest(digest, tensor)
        if write_history:
            output.mkdir(parents=True, exist_ok=True)
            with (output / "history_3d.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            torch.save({"format": "tensordem-dem3d-energy-benchmark-v1",
                        "config": config.__dict__, "state": sim.state_dict()},
                       output / "final_checkpoint_3d.pt")
        return rows, digest.hexdigest(), sim.broken_bonds

    rows, signature, broken = execute(write_history=True)
    repeat_signature = None
    if verify_repeat:
        _, repeat_signature, repeat_broken = execute(write_history=False)
        if signature != repeat_signature or broken != repeat_broken:
            raise AssertionError("identical CPU runs produced different signatures")

    final = rows[-1]
    summary = {
        "protocol": "tensordem-dem3d-indenter-v1",
        "interpretation": (
            "Regression benchmark only; not calibrated material validation or ship-scale prediction. "
            "Energy residual includes time-discretization and ledger approximations."
        ),
        "config": config.__dict__,
        "steps": steps,
        "sample_every": sample_every,
        "dt_s": config.dt if config.dt is not None else config.recommended_dt,
        "final_time_s": final["time"],
        "particle_count": final["particle_count"],
        "broken_bonds": final["broken_bonds"],
        "peak_abs_reaction_z_N": max(abs(float(row["reaction_z"])) for row in rows),
        "final_mechanical_energy_J": final["mechanical_energy_J"],
        "tool_work_cumulative_J": final["tool_work_cumulative_J"],
        "external_work_cumulative_J": final["external_work_cumulative_J"],
        "drag_dissipation_cumulative_J": final["drag_dissipation_cumulative_J"],
        "particle_damping_cumulative_J": final["particle_damping_cumulative_J"],
        "tool_damping_cumulative_J": final["tool_damping_cumulative_J"],
        "fracture_release_cumulative_J": final["fracture_release_cumulative_J"],
        "energy_balance_residual_J": final["energy_balance_residual_J"],
        "energy_balance_residual_relative": final["energy_balance_residual_relative"],
        "signature_sha256": signature,
        "repeat_signature_sha256": repeat_signature,
        "repeat_verified": verify_repeat,
    }
    (output / "summary_3d.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reproducible 3-D DEM indenter benchmark")
    parser.add_argument("--nx", type=int, default=7)
    parser.add_argument("--ny", type=int, default=5)
    parser.add_argument("--nz", type=int, default=3)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--radius", type=float, default=0.025)
    parser.add_argument("--tool-radius", type=float, default=0.1)
    parser.add_argument("--breaking-strain", type=float, default=0.015)
    parser.add_argument("--shear-breaking-strain", type=float, default=0.03)
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-benchmark"))
    parser.add_argument("--verify-repeat", action="store_true",
                        help="run the same case twice and require matching signatures")
    args = parser.parse_args(argv)
    try:
        config = DEM3DConfig(
            nx=args.nx, ny=args.ny, nz=args.nz, tool_speed=args.speed,
            radius=args.radius, tool_radius=args.tool_radius,
            breaking_strain=args.breaking_strain,
            shear_breaking_strain=args.shear_breaking_strain,
        )
        result = run_benchmark(args.output, config, args.steps, args.sample_every,
                               args.verify_repeat)
    except (ValueError, RuntimeError, AssertionError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
