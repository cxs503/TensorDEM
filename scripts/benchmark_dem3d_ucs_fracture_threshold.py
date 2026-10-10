"""Bond-break threshold screening for the 3-D UCS-style DEM benchmark.

This is a numerical parameter-identifiability diagnostic, not a material
calibration: no experimental target strength or fracture energy is assumed.
"""
from __future__ import annotations

import argparse
import csv
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

from scripts.benchmark_dem3d_ucs import _run_once
from scripts.benchmark_dem3d_ucs_sensitivity import _base_config, _steps_for_strain


def run_fracture_threshold_screen(
    output: Path, *, target_strain: float = 0.015,
    thresholds: tuple[float, ...] = (0.01, 0.015, 0.02),
) -> dict[str, Any]:
    """Run repeatable UCS histories over a declared bond tensile threshold set."""
    if not math.isfinite(target_strain) or not 0 < target_strain <= 0.08:
        raise ValueError("target_strain must be finite and in (0, 0.08]")
    if len(thresholds) < 2 or any(not math.isfinite(x) or x <= 0 for x in thresholds):
        raise ValueError("thresholds must contain at least two finite positive values")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("thresholds must be unique")
    torch.set_num_threads(1)
    base = _base_config()
    summaries: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []

    for threshold in sorted(thresholds):
        cfg = dict(base)
        cfg["breaking_strain"] = float(threshold)
        case = {"id": f"break_strain_{threshold:g}", "config": cfg}
        steps = _steps_for_strain(cfg, target_strain)
        rows, signature, metrics = _run_once(case, steps)
        repeat_rows, repeat_signature, repeat_metrics = _run_once(case, steps)
        finite = all(math.isfinite(float(v)) for row in rows for v in row.values())
        repeatable = (
            signature == repeat_signature
            and metrics["peak_engineering_stress_Pa"] == repeat_metrics["peak_engineering_stress_Pa"]
            and metrics["final_broken_bonds"] == repeat_metrics["final_broken_bonds"]
        )
        summaries.append({
            "case_id": case["id"],
            "breaking_strain": float(threshold),
            "target_axial_strain": target_strain,
            "achieved_axial_strain": float(rows[-1]["axial_engineering_strain"]),
            "steps": steps,
            "dt_s": float(rows[1]["time_s"] - rows[0]["time_s"]),
            "particle_count": int(metrics["particle_count"]),
            "bond_count": int(metrics["bond_count"]),
            "peak_load_N": float(metrics["peak_load_N"]),
            "peak_engineering_stress_Pa": float(metrics["peak_engineering_stress_Pa"]),
            "strain_at_peak_stress": float(metrics["strain_at_peak_stress"]),
            "final_broken_bonds": int(metrics["final_broken_bonds"]),
            "broken_bond_fraction": (
                float(metrics["final_broken_bonds"]) / int(metrics["bond_count"])
                if int(metrics["bond_count"]) else 0.0
            ),
            "final_bond_elastic_energy_J": float(rows[-1]["bond_elastic_energy_J"]),
            "final_kinetic_energy_J": float(rows[-1]["kinetic_energy_J"]),
            "signature_sha256": signature,
            "repeat_signature_sha256": repeat_signature,
            "finite_history": finite,
            "repeatable": repeatable,
        })
        histories.extend({"case_id": case["id"], **row} for row in rows)

    reference = next(x for x in summaries if x["breaking_strain"] == 0.015) if 0.015 in thresholds else summaries[len(summaries) // 2]
    for item in summaries:
        item["reference_case_id"] = reference["case_id"]
        item["peak_stress_delta_percent_vs_reference"] = (
            100.0 * (item["peak_engineering_stress_Pa"] / reference["peak_engineering_stress_Pa"] - 1.0)
            if reference["peak_engineering_stress_Pa"] else None
        )

    checks = {
        "all_histories_finite": all(x["finite_history"] for x in summaries),
        "all_cases_repeatable": all(x["repeatable"] for x in summaries),
        "positive_peak_stress": all(x["peak_engineering_stress_Pa"] > 0 for x in summaries),
        "target_strain_reached": all(
            abs(x["achieved_axial_strain"] - target_strain) <= target_strain * 0.05 + 1e-12
            for x in summaries
        ),
        "broken_bond_counts_bounded": all(
            0 <= x["final_broken_bonds"] <= x["bond_count"] for x in summaries
        ),
        "nonnegative_reported_energies": all(
            x["final_bond_elastic_energy_J"] >= -1e-12 and x["final_kinetic_energy_J"] >= -1e-12
            for x in summaries
        ),
    }
    report = {
        "protocol": "tensordem-dem3d-ucs-fracture-threshold-screen-v1",
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "target_axial_strain": target_strain,
        "case_count": len(summaries),
        "reference_case_id": reference["case_id"],
        "cases": summaries,
        "interpretation": (
            "PASS verifies finite and repeatable numerical screening only. "
            "A breaking-strain threshold is not a fracture-energy parameter, and this screen "
            "does not identify a physical value without measured stress-strain and fracture data."
        ),
        "limitations": [
            "Simple-cubic bonded-sphere prism with infinite frictionless prescribed-velocity platens.",
            "Only the tensile bond breaking-strain threshold is varied; shear threshold and other parameters are fixed.",
            "Peak stress and broken-bond fraction are discrete-model outputs, not measured ice properties.",
            "No experimental data, inverse fit, fracture-energy calibration, or uncertainty quantification is included.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "ucs_fracture_threshold_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "ucs_fracture_threshold_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (output / "ucs_fracture_threshold_histories.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(histories[0]))
        writer.writeheader()
        writer.writerows(histories)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-strain", type=float, default=0.015)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.015, 0.02])
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-ucs-fracture-threshold"))
    args = parser.parse_args(argv)
    report = run_fracture_threshold_screen(
        args.output, target_strain=args.target_strain, thresholds=tuple(args.thresholds)
    )
    print(json.dumps({
        "protocol": report["protocol"], "verdict": report["verdict"],
        "checks": report["checks"], "case_count": report["case_count"],
        "cases": [{
            "case_id": x["case_id"],
            "breaking_strain": x["breaking_strain"],
            "peak_stress_Pa": x["peak_engineering_stress_Pa"],
            "broken_bond_fraction": x["broken_bond_fraction"],
            "stress_delta_percent": x["peak_stress_delta_percent_vs_reference"],
            "repeatable": x["repeatable"],
        } for x in report["cases"]],
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
