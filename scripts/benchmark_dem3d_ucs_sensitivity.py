"""UCS loading-rate and particle-resolution sensitivity campaign.

This is a numerical sensitivity study for the current DEM3D bonded-sphere model.
It does not establish calibrated material properties or physical convergence.
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

from tensordem.dem3d import DEM3DConfig
from scripts.benchmark_dem3d_ucs import _run_once


def _base_config() -> dict[str, Any]:
    manifest = json.loads((ROOT / "benchmarks/dem3d/ucs_cases.json").read_text(encoding="utf-8"))
    return dict(manifest["cases"][0]["config"])


def _steps_for_strain(config: dict[str, Any], target_strain: float) -> int:
    radius = float(config["radius_m"])
    speed = float(config["loading_speed_m_s"])
    nz = int(config["nz"])
    sim_config = DEM3DConfig(
        nx=int(config["nx"]), ny=int(config["ny"]), nz=nz,
        radius=radius,
        density=float(config["density_kg_m3"]),
        bond_stiffness=float(config["bond_stiffness_N_m"]),
        contact_stiffness=float(config["contact_stiffness_N_m"]),
        breaking_strain=float(config["breaking_strain"]),
        platen_stiffness=float(config["platen_stiffness_N_m"]),
        platen_damping=float(config["platen_damping_Ns_m"]),
        top_platen_enabled=True, bottom_platen_enabled=True,
        top_platen_velocity=-speed, bottom_platen_velocity=speed,
        drag=0.1, contact_damping=0.5,
        fix_x_edges=False, fix_bottom=False,
    )
    dt = sim_config.recommended_dt
    initial_height = nz * 2.0 * radius
    return max(1, math.ceil(target_strain * initial_height / (2.0 * speed * dt)))


def _case(case_id: str, config: dict[str, Any]) -> dict[str, Any]:
    return {"id": case_id, "config": config}


def run_sensitivity_campaign(output: Path, *, target_strain: float = 0.015) -> dict[str, Any]:
    if not math.isfinite(target_strain) or target_strain <= 0 or target_strain > 0.08:
        raise ValueError("target_strain must be finite and in (0, 0.08]")
    torch.set_num_threads(1)
    base = _base_config()
    scenarios: list[tuple[str, dict[str, Any], str, float]] = []

    # Rate sweep: hold geometry and all material/numerical parameters fixed.
    for label, speed in (("slow", 0.025), ("reference", 0.05), ("fast", 0.1)):
        cfg = dict(base)
        cfg["loading_speed_m_s"] = speed
        scenarios.append((f"rate_{label}", cfg, "loading_rate", speed))

    # Resolution sweep: keep the gross specimen dimensions approximately fixed.
    # Spring stiffness is scaled linearly with particle radius as a first-order
    # network scaling assumption; this is not a substitute for calibration.
    base_radius = float(base["radius_m"])
    for label, nx, ny, nz, radius in (
        ("coarse", 4, 4, 3, 0.025),
        ("medium", 5, 5, 4, 0.01875),
        ("fine", 6, 6, 5, 0.015),
    ):
        cfg = dict(base)
        cfg.update(nx=nx, ny=ny, nz=nz, radius_m=radius)
        scale = radius / base_radius
        for key in ("bond_stiffness_N_m", "contact_stiffness_N_m", "platen_stiffness_N_m"):
            cfg[key] = float(base[key]) * scale
        scenarios.append((f"resolution_{label}", cfg, "resolution", float(nx * ny * nz)))

    summaries: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    for case_id, cfg, family, level in scenarios:
        nsteps = _steps_for_strain(cfg, target_strain)
        case = _case(case_id, cfg)
        rows, signature, metrics = _run_once(case, nsteps)
        _, repeat_signature, repeat_metrics = _run_once(case, nsteps)
        final_strain = float(rows[-1]["axial_engineering_strain"])
        item = {
            "case_id": case_id,
            "family": family,
            "level": level,
            "loading_speed_m_s": float(cfg["loading_speed_m_s"]),
            "nx": int(cfg["nx"]), "ny": int(cfg["ny"]), "nz": int(cfg["nz"]),
            "radius_m": float(cfg["radius_m"]),
            "particle_count": int(metrics["particle_count"]),
            "bond_count": int(metrics["bond_count"]),
            "steps": nsteps,
            "dt_s": float(rows[1]["time_s"] - rows[0]["time_s"]),
            "target_axial_strain": target_strain,
            "achieved_axial_strain": final_strain,
            "peak_load_N": float(metrics["peak_load_N"]),
            "peak_engineering_stress_Pa": float(metrics["peak_engineering_stress_Pa"]),
            "strain_at_peak_stress": float(metrics["strain_at_peak_stress"]),
            "final_broken_bonds": int(metrics["final_broken_bonds"]),
            "lateral_x_engineering_strain": float(metrics["lateral_x_engineering_strain"]),
            "lateral_y_engineering_strain": float(metrics["lateral_y_engineering_strain"]),
            "signature_sha256": signature,
            "repeat_signature_sha256": repeat_signature,
            "repeatable": signature == repeat_signature
                and metrics["peak_engineering_stress_Pa"] == repeat_metrics["peak_engineering_stress_Pa"]
                and metrics["final_broken_bonds"] == repeat_metrics["final_broken_bonds"],
            "finite_history": all(math.isfinite(float(v)) for row in rows for v in row.values()),
        }
        summaries.append(item)
        for row in rows:
            histories.append({"case_id": case_id, **row})

    rate_reference = next(x for x in summaries if x["case_id"] == "rate_reference")
    resolution_reference = next(x for x in summaries if x["case_id"] == "resolution_coarse")
    for item in summaries:
        if item["family"] == "loading_rate":
            reference = rate_reference
            item["reference_case_id"] = reference["case_id"]
            item["peak_stress_delta_percent_vs_reference"] = (
                100.0 * (item["peak_engineering_stress_Pa"] / reference["peak_engineering_stress_Pa"] - 1.0)
                if reference["peak_engineering_stress_Pa"] else None
            )
        else:
            reference = resolution_reference
            item["reference_case_id"] = reference["case_id"]
            item["peak_stress_delta_percent_vs_reference"] = (
                100.0 * (item["peak_engineering_stress_Pa"] / reference["peak_engineering_stress_Pa"] - 1.0)
                if reference["peak_engineering_stress_Pa"] else None
            )

    checks = {
        "all_histories_finite": all(x["finite_history"] for x in summaries),
        "all_cases_repeatable": all(x["repeatable"] for x in summaries),
        "positive_peak_stress": all(x["peak_engineering_stress_Pa"] > 0 for x in summaries),
        "target_strain_reached_within_one_step": all(
            abs(x["achieved_axial_strain"] - target_strain)
            <= target_strain * 0.05 + 1e-12 for x in summaries
        ),
    }
    report = {
        "protocol": "tensordem-dem3d-ucs-sensitivity-v1",
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "target_axial_strain": target_strain,
        "case_count": len(summaries),
        "cases": summaries,
        "interpretation": (
            "PASS means finite, repeatable runs reached the common target strain. "
            "Sensitivity deltas describe this discrete model only; they do not prove physical convergence "
            "or calibrated ice properties."
        ),
        "limitations": [
            "Resolution variants approximate rather than exactly match specimen dimensions.",
            "Bond/contact/platen spring stiffness is scaled linearly with particle radius as a first-order assumption.",
            "The specimen remains a simple-cubic bonded-sphere prism and platen boundaries remain infinite, frictionless and velocity-controlled.",
            "No experimental reference data or material calibration is included.",
            "The campaign is a sensitivity diagnostic, not a mesh-independent or rate-independent convergence claim.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "ucs_sensitivity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "ucs_sensitivity_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (output / "ucs_sensitivity_histories.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(histories[0]))
        writer.writeheader()
        writer.writerows(histories)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-strain", type=float, default=0.015)
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-ucs-sensitivity"))
    args = parser.parse_args(argv)
    report = run_sensitivity_campaign(args.output, target_strain=args.target_strain)
    print(json.dumps({
        "protocol": report["protocol"], "verdict": report["verdict"],
        "checks": report["checks"], "case_count": report["case_count"],
        "output": str(args.output),
        "cases": [{
            "case_id": x["case_id"], "steps": x["steps"],
            "peak_stress_Pa": x["peak_engineering_stress_Pa"],
            "delta_percent": x["peak_stress_delta_percent_vs_reference"],
            "repeatable": x["repeatable"],
        } for x in report["cases"]],
    }, indent=2, sort_keys=True))
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
