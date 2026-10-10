"""Step-refinement campaign for the DEM3D three-point-bending smoke fixture.

The campaign reports numerical refinement trends. It is not experimental validation,
a flexural-strength calibration, or a guarantee that the underlying model is physical.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.benchmark_dem3d_three_point_bending import run_three_point_bending


def _relative_change(previous: float, current: float) -> float:
    scale = max(abs(previous), abs(current), 1.0e-30)
    return abs(current - previous) / scale


def run_bending_load_convergence(
    output: Path, *, base_steps: int = 30, levels: int = 3,
    peak_load_N: float = 0.02,
) -> dict[str, Any]:
    """Run geometrically refined load ramps and summarize response changes.

    steps controls the number of load increments over the same nominal ramp;
    it is a load-path discretization study, not a physical-time timestep study.
    """
    if isinstance(base_steps, bool) or not isinstance(base_steps, int) or base_steps < 2:
        raise ValueError("base_steps must be an integer >= 2")
    if isinstance(levels, bool) or not isinstance(levels, int) or not 2 <= levels <= 6:
        raise ValueError("levels must be an integer from 2 through 6")
    if not math.isfinite(peak_load_N) or peak_load_N <= 0:
        raise ValueError("peak_load_N must be finite and positive")

    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for level in range(levels):
        nsteps = base_steps * (2 ** level)
        report = run_three_point_bending(
            output / f"level-{level + 1:02d}-steps-{nsteps}",
            steps=nsteps, peak_load_N=peak_load_N,
        )
        rows.append({
            "level": level + 1,
            "steps": nsteps,
            "peak_load_N": peak_load_N,
            "peak_support_reaction_abs_N": float(report["peak_support_reaction_abs_N"]),
            "peak_midspan_deflection_abs_m": float(report["peak_midspan_deflection_abs_m"]),
            "broken_bonds": int(report["broken_bonds"]),
            "signature_sha256": report["signature_sha256"],
            "verdict": report["verdict"],
        })

    comparisons: list[dict[str, Any]] = []
    for previous, current in zip(rows, rows[1:]):
        comparisons.append({
            "from_steps": previous["steps"],
            "to_steps": current["steps"],
            "support_reaction_relative_change": _relative_change(
                previous["peak_support_reaction_abs_N"],
                current["peak_support_reaction_abs_N"],
            ),
            "midspan_deflection_relative_change": _relative_change(
                previous["peak_midspan_deflection_abs_m"],
                current["peak_midspan_deflection_abs_m"],
            ),
            "broken_bonds_changed": previous["broken_bonds"] != current["broken_bonds"],
        })

    report = {
        "protocol": "tensordem-dem3d-bending-load-convergence-v1",
        "verdict": "PASS" if all(row["verdict"] == "PASS" for row in rows) else "FAIL",
        "base_steps": base_steps,
        "levels": levels,
        "step_counts": [row["steps"] for row in rows],
        "peak_load_N": peak_load_N,
        "runs": rows,
        "comparisons": comparisons,
        "interpretation": (
            "This is a load-increment refinement study with the same nominal load ramp. "
            "Changes quantify numerical path sensitivity only; they do not demonstrate "
            "time-step convergence, experimental agreement, or calibrated ice flexural strength. "
            "Bond-break count changes indicate a potentially discontinuous fracture response."
        ),
    }
    with (output / "bending_load_convergence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "bending_load_convergence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if report["verdict"] != "PASS":
        raise RuntimeError("one or more three-point-bending load-refinement runs failed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-bending-load-convergence"))
    parser.add_argument("--base-steps", type=int, default=30)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--peak-load-N", type=float, default=0.02)
    args = parser.parse_args()
    report = run_bending_load_convergence(
        args.output, base_steps=args.base_steps, levels=args.levels,
        peak_load_N=args.peak_load_N,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
