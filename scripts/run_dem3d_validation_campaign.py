"""Run the DEM3D numerical verification pipeline end to end.

Stages: time-step sensitivity -> particle-resolution sensitivity ->
cross-resolution force/damage comparison -> integrated numerical screening.
This produces a screening report; it does not calibrate or validate material physics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))

from tensordem.dem3d import DEM3DConfig
from scripts.validate_dem3d_convergence import run_convergence_study, write_report as write_timestep_report
from scripts.validate_dem3d_resolution import run_resolution_study, write_report as write_resolution_report
from scripts.compare_dem3d_resolutions import read_resolution_history, compare_resolution_histories, write_report as write_cross_report
from scripts.report_dem3d_engineering_screen import build_acceptance_report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_validation_campaign(
    config: DEM3DConfig,
    *,
    output: Path,
    base_steps: int = 100,
    timestep_factors: Sequence[int] = (1, 2, 4),
    resolution_factors: Sequence[int] = (1, 2),
    sample_every: int = 5,
    screening_limits: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Execute all verification stages and write a reproducibility manifest."""
    if isinstance(base_steps, bool) or not isinstance(base_steps, int) or base_steps < 1:
        raise ValueError("base_steps must be a positive integer")
    if isinstance(sample_every, bool) or not isinstance(sample_every, int) or sample_every < 1:
        raise ValueError("sample_every must be a positive integer")
    if config.device != "cpu":
        raise ValueError("the campaign runner currently requires device='cpu'")
    time_factors = tuple(timestep_factors)
    space_factors = tuple(resolution_factors)
    for name, values in (("timestep_factors", time_factors), ("resolution_factors", space_factors)):
        if len(values) < 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values
        ) or len(set(values)) != len(values):
            raise ValueError(f"{name} must contain at least two unique positive integers")

    output.mkdir(parents=True, exist_ok=True)
    timestep_dir = output / "timestep"
    resolution_dir = output / "resolution"
    cross_dir = output / "cross_resolution"
    screen_dir = output / "screening"

    timestep = run_convergence_study(
        config, base_steps=base_steps, refinement_factors=time_factors
    )
    write_timestep_report(timestep, timestep_dir)

    resolution = run_resolution_study(
        config, base_steps=base_steps, resolution_factors=space_factors,
        sample_every=sample_every,
    )
    write_resolution_report(resolution, resolution_dir)

    histories = read_resolution_history(resolution_dir / "resolution_history_3d.csv")
    cross = compare_resolution_histories(histories)
    write_cross_report(cross, cross_dir)

    limits = screening_limits or {}
    screen = build_acceptance_report(
        timestep_dir / "convergence_3d.json",
        resolution_dir / "resolution_sensitivity_3d.json",
        cross_dir / "cross_resolution_summary_3d.json",
        **limits,
    )
    screen_dir.mkdir(parents=True, exist_ok=True)
    (screen_dir / "engineering_screen_3d.json").write_text(
        json.dumps(screen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    import csv
    with (screen_dir / "engineering_screen_3d.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["check", "status", "observed", "threshold", "detail"]
        )
        writer.writeheader()
        writer.writerows(screen["checks"])

    generated_files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "protocol": "tensordem-dem3d-validation-campaign-v1",
        "configuration": asdict(config),
        "campaign": {
            "base_steps": base_steps,
            "timestep_factors": list(time_factors),
            "resolution_factors": list(space_factors),
            "sample_every": sample_every,
            "device": config.device,
        },
        "stages": {
            "timestep_sensitivity": str(timestep_dir / "convergence_3d.json"),
            "resolution_sensitivity": str(resolution_dir / "resolution_sensitivity_3d.json"),
            "cross_resolution_comparison": str(cross_dir / "cross_resolution_summary_3d.json"),
            "engineering_screen": str(screen_dir / "engineering_screen_3d.json"),
        },
        "screening_verdict": screen["verdict"],
        "screening_counts": {
            "checks": screen["check_count"], "pass": screen["pass_count"],
            "warn": screen["warn_count"], "fail": screen["fail_count"],
        },
        "artifacts": [
            {"path": str(path.relative_to(output)), "sha256": _sha256(path)}
            for path in generated_files
        ],
        "interpretation": (
            "This manifest makes numerical screening artifacts auditable. A screening "
            "verdict does not establish material calibration, experimental validity, "
            "asymptotic convergence, or full-scale engineering qualification."
        ),
    }
    (output / "validation_campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="End-to-end DEM3D numerical validation campaign")
    parser.add_argument("--nx", type=int, default=5)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--nz", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100,
                        help="base physical duration in steps at the coarsest time resolution")
    parser.add_argument("--timestep-factors", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--resolution-factors", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--radius", type=float, default=0.025)
    parser.add_argument("--tool-radius", type=float, default=0.1)
    parser.add_argument("--breaking-strain", type=float, default=0.015)
    parser.add_argument("--shear-breaking-strain", type=float, default=0.03)
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-validation-campaign"))
    parser.add_argument("--max-timestep-peak-force-delta", type=float, default=0.05)
    parser.add_argument("--max-resolution-peak-force-delta", type=float, default=0.10)
    parser.add_argument("--max-cross-resolution-force-nrmse", type=float, default=0.10)
    parser.add_argument("--max-energy-residual-ratio", type=float, default=0.05)
    parser.add_argument("--final-time-relative-tolerance", type=float, default=1e-8)
    args = parser.parse_args(argv)
    try:
        config = DEM3DConfig(
            nx=args.nx, ny=args.ny, nz=args.nz, tool_speed=args.speed,
            radius=args.radius, tool_radius=args.tool_radius,
            breaking_strain=args.breaking_strain,
            shear_breaking_strain=args.shear_breaking_strain,
        )
        manifest = run_validation_campaign(
            config, output=args.output, base_steps=args.steps,
            timestep_factors=args.timestep_factors,
            resolution_factors=args.resolution_factors,
            sample_every=args.sample_every,
            screening_limits={
                "max_timestep_peak_force_delta": args.max_timestep_peak_force_delta,
                "max_resolution_peak_force_delta": args.max_resolution_peak_force_delta,
                "max_cross_resolution_force_nrmse": args.max_cross_resolution_force_nrmse,
                "max_energy_residual_ratio": args.max_energy_residual_ratio,
                "final_time_relative_tolerance": args.final_time_relative_tolerance,
            },
        )
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "protocol": manifest["protocol"],
        "screening_verdict": manifest["screening_verdict"],
        "screening_counts": manifest["screening_counts"],
        "manifest": str(args.output / "validation_campaign_manifest.json"),
        "artifacts": len(manifest["artifacts"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
