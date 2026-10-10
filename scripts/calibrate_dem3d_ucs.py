"""Experimental-curve-driven parameter screening for the DEM3D UCS model.

Input CSV columns: axial_strain, compressive_stress_Pa. The tool ranks
candidate parameter combinations against measured data; it does not claim that
a numerical fit is physically valid without independent validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch

from scripts.benchmark_dem3d_ucs import _run_once
from scripts.benchmark_dem3d_ucs_sensitivity import _base_config, _steps_for_strain


def load_experimental_curve(path: Path) -> list[dict[str, float]]:
    """Load and validate a measured engineering stress-strain CSV."""
    points: list[dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"axial_strain", "compressive_stress_Pa"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "experimental CSV must contain axial_strain and compressive_stress_Pa columns"
            )
        for line, row in enumerate(reader, start=2):
            try:
                strain = float(row["axial_strain"])
                stress = float(row["compressive_stress_Pa"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid numeric data on CSV line {line}") from exc
            if not math.isfinite(strain) or not math.isfinite(stress):
                raise ValueError(f"non-finite data on CSV line {line}")
            if strain < 0 or stress < 0:
                raise ValueError(f"strain and compressive stress must be nonnegative (line {line})")
            if points and strain <= points[-1]["axial_strain"]:
                raise ValueError("axial_strain values must be strictly increasing")
            points.append({"axial_strain": strain, "compressive_stress_Pa": stress})
    if len(points) < 3:
        raise ValueError("experimental curve must contain at least three data points")
    if max(p["compressive_stress_Pa"] for p in points) <= 0:
        raise ValueError("experimental curve must contain a positive compressive stress")
    if points[0]["axial_strain"] > 1e-12:
        raise ValueError("experimental curve must start at zero strain (within 1e-12)")
    if points[-1]["axial_strain"] <= 0 or points[-1]["axial_strain"] > 0.08:
        raise ValueError("maximum experimental strain must be in (0, 0.08]")
    return points


def _interpolate(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    if x < xs[0] - 1e-12 or x > xs[-1] + 1e-12:
        raise ValueError("requested strain lies outside the simulation curve")
    if x <= xs[0]:
        return float(ys[0])
    for i in range(1, len(xs)):
        if x <= xs[i]:
            dx = xs[i] - xs[i - 1]
            if dx <= 0:
                raise ValueError("simulation strain history must be strictly increasing")
            fraction = (x - xs[i - 1]) / dx
            return float(ys[i - 1] + fraction * (ys[i] - ys[i - 1]))
    return float(ys[-1])


def _score_curve(
    experimental: list[dict[str, float]],
    simulated_rows: list[dict[str, float | int]],
) -> dict[str, float]:
    exp_strain = [p["axial_strain"] for p in experimental]
    exp_stress = [p["compressive_stress_Pa"] for p in experimental]
    sim_strain = [float(r["axial_engineering_strain"]) for r in simulated_rows]
    sim_stress = [float(r["engineering_stress_Pa"]) for r in simulated_rows]
    if any(b <= a for a, b in zip(sim_strain, sim_strain[1:])):
        raise ValueError("simulation strain history must be strictly increasing")
    if sim_strain[-1] + 1e-12 < exp_strain[-1]:
        raise ValueError("simulation does not cover the full experimental strain range")

    predicted = [_interpolate(x, sim_strain, sim_stress) for x in exp_strain]
    exp_peak_index = max(range(len(exp_stress)), key=exp_stress.__getitem__)
    sim_peak_index = max(range(len(sim_stress)), key=sim_stress.__getitem__)
    exp_peak = exp_stress[exp_peak_index]
    sim_peak = sim_stress[sim_peak_index]
    scale = max(exp_peak, 1e-12)
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(predicted, exp_stress)) / len(exp_stress))
    normalized_rmse = rmse / scale
    peak_stress_relative_error = abs(sim_peak - exp_peak) / scale
    strain_scale = max(exp_strain[-1], 1e-12)
    peak_strain_error = abs(sim_strain[sim_peak_index] - exp_strain[exp_peak_index]) / strain_scale
    # Explicit, interpretable weights; users should declare alternative weights
    # before fitting rather than selecting weights to force a desired result.
    objective = normalized_rmse + 0.25 * peak_stress_relative_error + 0.25 * peak_strain_error
    return {
        "rmse_Pa": rmse,
        "normalized_rmse": normalized_rmse,
        "peak_stress_relative_error": peak_stress_relative_error,
        "peak_strain_relative_error": peak_strain_error,
        "objective": objective,
        "experimental_peak_stress_Pa": exp_peak,
        "simulated_peak_stress_Pa": sim_peak,
        "experimental_strain_at_peak": exp_strain[exp_peak_index],
        "simulated_strain_at_peak": sim_strain[sim_peak_index],
    }


def run_calibration_campaign(
    experimental_csv: Path,
    output: Path,
    *,
    bond_stiffness_scales: tuple[float, ...] = (0.5, 1.0, 2.0),
    breaking_strains: tuple[float, ...] = (0.01, 0.015, 0.02),
    verify_repeat: bool = True,
) -> dict[str, Any]:
    """Run a transparent grid screen and rank parameter combinations."""
    if not bond_stiffness_scales or any(not math.isfinite(x) or x <= 0 for x in bond_stiffness_scales):
        raise ValueError("bond stiffness scales must be finite and positive")
    if len(set(bond_stiffness_scales)) != len(bond_stiffness_scales):
        raise ValueError("bond stiffness scales must be unique")
    if not breaking_strains or any(not math.isfinite(x) or x <= 0 or x >= 1 for x in breaking_strains):
        raise ValueError("breaking strains must be finite and in (0, 1)")
    if len(set(breaking_strains)) != len(breaking_strains):
        raise ValueError("breaking strains must be unique")

    experimental = load_experimental_curve(experimental_csv)
    target_strain = experimental[-1]["axial_strain"]
    torch.set_num_threads(1)
    base = _base_config()
    cases: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []

    for stiffness_scale in sorted(bond_stiffness_scales):
        for threshold in sorted(breaking_strains):
            cfg = dict(base)
            cfg["bond_stiffness_N_m"] = float(base["bond_stiffness_N_m"]) * stiffness_scale
            cfg["breaking_strain"] = float(threshold)
            case_id = f"bond_k_{stiffness_scale:g}_break_{threshold:g}"
            case = {"id": case_id, "config": cfg}
            steps = _steps_for_strain(cfg, target_strain)
            rows, signature, metrics = _run_once(case, steps)
            repeatable = True
            repeat_signature = signature
            if verify_repeat:
                _, repeat_signature, repeat_metrics = _run_once(case, steps)
                repeatable = (
                    signature == repeat_signature
                    and metrics["peak_engineering_stress_Pa"] == repeat_metrics["peak_engineering_stress_Pa"]
                    and metrics["final_broken_bonds"] == repeat_metrics["final_broken_bonds"]
                )
            finite = all(math.isfinite(float(v)) for row in rows for v in row.values())
            scores = _score_curve(experimental, rows)
            item: dict[str, Any] = {
                "case_id": case_id,
                "bond_stiffness_N_m": float(cfg["bond_stiffness_N_m"]),
                "bond_stiffness_scale": stiffness_scale,
                "breaking_strain": threshold,
                "target_axial_strain": target_strain,
                "achieved_axial_strain": float(rows[-1]["axial_engineering_strain"]),
                "steps": steps,
                "dt_s": float(rows[1]["time_s"] - rows[0]["time_s"]),
                "particle_count": int(metrics["particle_count"]),
                "bond_count": int(metrics["bond_count"]),
                "final_broken_bonds": int(metrics["final_broken_bonds"]),
                "broken_bond_fraction": (
                    float(metrics["final_broken_bonds"]) / int(metrics["bond_count"])
                    if int(metrics["bond_count"]) else 0.0
                ),
                "final_bond_elastic_energy_J": float(rows[-1]["bond_elastic_energy_J"]),
                "final_kinetic_energy_J": float(rows[-1]["kinetic_energy_J"]),
                "signature_sha256": signature,
                "repeat_signature_sha256": repeat_signature,
                "repeatable": repeatable,
                "finite_history": finite,
                **scores,
            }
            cases.append(item)
            histories.extend({"case_id": case_id, **row} for row in rows)

    cases.sort(key=lambda x: (float(x["objective"]), str(x["case_id"])))
    for rank, item in enumerate(cases, start=1):
        item["rank"] = rank
    checks = {
        "all_histories_finite": all(x["finite_history"] for x in cases),
        "all_cases_repeatable": all(x["repeatable"] for x in cases),
        "all_simulations_cover_experiment": all(
            x["achieved_axial_strain"] + 1e-12 >= target_strain for x in cases
        ),
        "nonnegative_reported_energies": all(
            x["final_bond_elastic_energy_J"] >= -1e-12 and x["final_kinetic_energy_J"] >= -1e-12
            for x in cases
        ),
    }
    report = {
        "protocol": "tensordem-dem3d-ucs-curve-calibration-screen-v1",
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "experimental_file": experimental_csv.name,
        "experimental_point_count": len(experimental),
        "target_axial_strain": target_strain,
        "candidate_count": len(cases),
        "objective_definition": (
            "normalized stress-curve RMSE + 0.25 * relative peak-stress error "
            "+ 0.25 * peak-strain error normalized by maximum experimental strain"
        ),
        "best_candidate": cases[0] if cases else None,
        "candidates": cases,
        "limitations": [
            "Grid screening varies only tensile bond stiffness and tensile breaking strain; other parameters remain fixed.",
            "The objective and weights are a transparent baseline and should be preregistered for a scientific study.",
            "A low score is not proof of identifiability, physical validity, or predictive capability.",
            "Use independent validation curves and report uncertainty, temperature, salinity, rate, specimen geometry and boundary conditions.",
            "Breaking strain is not fracture energy; this workflow does not calibrate fracture energy or mixed-mode failure.",
            "The simple-cubic bonded-sphere specimen and frictionless platens are not a laboratory-equivalent specimen.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "ucs_calibration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "ucs_calibration_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = ["rank"] + [k for k in cases[0] if k != "rank"] if cases else ["rank"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(cases)
    with (output / "ucs_calibration_histories.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", *histories[0].keys()][1:] if histories else ["case_id"])
        writer.writeheader()
        writer.writerows(histories)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimental-csv", type=Path, required=True)
    parser.add_argument("--bond-stiffness-scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--breaking-strains", type=float, nargs="+", default=[0.01, 0.015, 0.02])
    parser.add_argument("--no-repeat-check", action="store_true", help="skip duplicate runs (faster, weaker verification)")
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-ucs-calibration"))
    args = parser.parse_args(argv)
    report = run_calibration_campaign(
        args.experimental_csv, args.output,
        bond_stiffness_scales=tuple(args.bond_stiffness_scales),
        breaking_strains=tuple(args.breaking_strains),
        verify_repeat=not args.no_repeat_check,
    )
    print(json.dumps({
        "protocol": report["protocol"],
        "verdict": report["verdict"],
        "checks": report["checks"],
        "candidate_count": report["candidate_count"],
        "best_candidate": {
            "case_id": report["best_candidate"]["case_id"],
            "objective": report["best_candidate"]["objective"],
            "peak_stress_Pa": report["best_candidate"]["simulated_peak_stress_Pa"],
        } if report["best_candidate"] else None,
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
