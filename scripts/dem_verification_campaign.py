"""Run a reproducible TensorDEM resolution-verification campaign."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tensordem.verification import (
    CampaignSpec,
    CampaignValidationError,
    make_resolution_specs,
    report_metadata,
    run_resolution_campaign,
    write_report,
    write_summary_csv,
)


def parse_levels(value: str) -> list[int]:
    """Parse a comma-separated sequence of unique integer resolutions."""
    try:
        levels = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("levels must be comma-separated integers") from exc
    if not levels:
        raise argparse.ArgumentTypeError("at least one resolution is required")
    if len(set(levels)) != len(levels):
        raise argparse.ArgumentTypeError("resolution levels must be unique")
    if any(level < 3 for level in levels):
        raise argparse.ArgumentTypeError("each resolution must be >= 3")
    return levels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled 2-D bonded-particle DEM sensitivity campaign. "
            "Outputs are numerical evidence, not physical validation."
        )
    )
    parser.add_argument("--nx-levels", type=parse_levels, default=parse_levels("5,7,9"))
    parser.add_argument("--width", type=float, default=0.3, help="target ice width (m)")
    parser.add_argument("--height", type=float, default=0.2, help="target ice height (m)")
    parser.add_argument("--thickness", type=float, default=0.2, help="extrusion thickness (m)")
    parser.add_argument("--duration", type=float, default=0.002, help="simulation horizon (s)")
    parser.add_argument("--speed", type=float, default=0.2, help="prescribed indenter speed (m/s)")
    parser.add_argument("--save-every", type=int, default=1, help="record every N steps")
    parser.add_argument("--max-steps", type=int, default=2_000_000, help="per-case safety cap")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--reference-index", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("verification-results"))
    parser.add_argument("--bond-stiffness", type=float, default=2_000.0)
    parser.add_argument("--contact-stiffness", type=float, default=2_000.0)
    parser.add_argument("--breaking-strain", type=float, default=0.015)
    parser.add_argument("--shear-breaking-strain", type=float, default=0.03)
    parser.add_argument("--dt", type=float, default=None, help="optional timestep (s), must satisfy solver bound")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        specs = make_resolution_specs(
            args.nx_levels,
            target_width_m=args.width,
            target_height_m=args.height,
            duration_s=args.duration,
            speed_m_s=args.speed,
            thickness_m=args.thickness,
            save_every=args.save_every,
            bond_stiffness_N_m=args.bond_stiffness,
            contact_stiffness_N_m=args.contact_stiffness,
            breaking_strain=args.breaking_strain,
            shear_breaking_strain=args.shear_breaking_strain,
            dt_s=args.dt,
            device=args.device,
        )
        report = run_resolution_campaign(
            specs,
            reference_index=args.reference_index,
            max_steps=args.max_steps,
        )
        report_path = write_report(report, args.output / "campaign.json")
        csv_path = write_summary_csv(report, args.output / "summary.csv")
    except (CampaignValidationError, ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    metadata = report_metadata(report)
    print(f"Cases: {metadata['case_count']}")
    print(f"Reference case index: {metadata['reference_index']}")
    print(f"Campaign digest: {metadata['campaign_sha256']}")
    print("Physical validation: NOT CLAIMED")
    for comparison in report["comparisons"]:
        print(
            "nx={nx}: relative force RMS={rms:.6g}, peak difference={peak:.6g}, "
            "impulse difference={impulse:.6g}".format(
                nx=comparison["case_nx"],
                rms=comparison["force_relative_rms_difference"],
                peak=comparison["force_peak_absolute_relative_difference"],
                impulse=comparison["signed_impulse_relative_difference"],
            )
        )
    print(f"JSON report: {report_path}")
    print(f"CSV summary: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
