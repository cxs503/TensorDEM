"""Run a reproducible moving-wedge bow / bonded-ice DEM3D smoke campaign."""
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

from tensordem.dem3d import DEM3DConfig
from tensordem.dem3d_wedge import MovingWedgeBowIceDEM3D


def _run_once(steps: int, *, bow_speed: float, wedge_angle_deg: float) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    radius = 0.02
    config = DEM3DConfig(
        nx=7, ny=3, nz=4,
        radius=radius,
        density=917.0,
        bond_stiffness=1200.0,
        contact_stiffness=3500.0,
        breaking_strain=0.008,
        shear_breaking_strain=0.025,
        contact_damping=0.5,
        drag=0.1,
        tool_speed=bow_speed,
        fix_x_edges=False,
        fix_bottom=False,
    )
    sim = MovingWedgeBowIceDEM3D(
        config, bow_speed=bow_speed, wedge_angle_deg=wedge_angle_deg,
        initial_gap=0.0, tip_height_fraction=0.5,
    )
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    resistance_work = 0.0
    previous_resistance = 0.0
    previous_time = 0.0
    first_fracture_time = None
    for step in range(steps + 1):
        if step:
            sim.step()
        d = sim.diagnostics()
        resistance = max(0.0, -float(d["bow_reaction_x_N"]))
        if step:
            resistance_work += 0.5 * (previous_resistance + resistance) * bow_speed * (float(d["time"]) - previous_time)
        if first_fracture_time is None and int(d["broken_bonds"]) > 0:
            first_fracture_time = float(d["time"])
        row = {
            "step": step,
            "time_s": float(d["time"]),
            "bow_tip_x_m": float(d["bow_tip_x_m"]),
            "bow_reaction_x_N": float(d["bow_reaction_x_N"]),
            "bow_reaction_z_N": float(d["bow_reaction_z_N"]),
            "ice_resistance_N": resistance,
            "broken_bonds": int(d["broken_bonds"]),
            "kinetic_energy_J": float(d["kinetic_energy_J"]),
            "bond_elastic_energy_J": float(d["bond_elastic_energy_J"]),
            "bow_contact_energy_J": float(d["bow_contact_energy_J"]),
        }
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError(f"non-finite wedge history at step {step}")
        rows.append(row)
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        previous_resistance, previous_time = resistance, float(d["time"])
    resistance_values = [float(row["ice_resistance_N"]) for row in rows]
    metrics = {
        "particle_count": int(len(sim.positions)),
        "initial_bond_count": int(len(sim.pairs)),
        "final_broken_bonds": int(sim.broken_bonds),
        "broken_bond_fraction": float(sim.broken_bonds / max(len(sim.pairs), 1)),
        "peak_ice_resistance_N": max(resistance_values),
        "mean_ice_resistance_N": sum(resistance_values) / len(resistance_values),
        "integrated_resistance_work_J": resistance_work,
        "first_fracture_time_s": first_fracture_time,
        "bow_travel_m": bow_speed * float(rows[-1]["time_s"]),
        "signature_sha256": digest.hexdigest(),
    }
    return rows, digest.hexdigest(), metrics


def run_campaign(output: Path, *, steps: int = 300, bow_speed: float = 0.1, wedge_angle_deg: float = 45.0) -> dict[str, Any]:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not math.isfinite(bow_speed) or bow_speed <= 0:
        raise ValueError("bow_speed must be finite and positive")
    if not math.isfinite(wedge_angle_deg) or not 5 <= wedge_angle_deg <= 85:
        raise ValueError("wedge_angle_deg must be in [5, 85]")
    torch.set_num_threads(1)
    rows, signature, metrics = _run_once(steps, bow_speed=bow_speed, wedge_angle_deg=wedge_angle_deg)
    repeat_rows, repeat_signature, repeat_metrics = _run_once(steps, bow_speed=bow_speed, wedge_angle_deg=wedge_angle_deg)
    checks = {
        "finite_history": all(math.isfinite(float(value)) for row in rows for value in row.values()),
        "repeatable_history": signature == repeat_signature,
        "bow_moves_forward": float(rows[-1]["bow_tip_x_m"]) > float(rows[0]["bow_tip_x_m"]),
        "nonnegative_resistance_work": metrics["integrated_resistance_work_J"] >= 0.0,
        "metrics_repeatable": metrics == repeat_metrics,
    }
    report = {
        "protocol": "tensordem-dem3d-moving-wedge-bow-v1",
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "steps": steps,
        "bow_speed_m_s": bow_speed,
        "wedge_angle_deg": wedge_angle_deg,
        "checks": checks,
        "metrics": metrics,
        "interpretation": (
            "Ice resistance is the positive opposing x reaction of the prescribed wedge. "
            "Integrated resistance work is trapezoidal force-speed-time quadrature. "
            "This is a dry, quasi-static-style bonded-particle contact model without fluid dynamics, "
            "hydrostatic pressure, buoyancy, ship heave/pitch, or calibrated hull/ice parameters."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "moving_wedge_bow_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "moving_wedge_bow_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "moving_wedge_bow_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-moving-wedge-bow"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--bow-speed", type=float, default=0.1)
    parser.add_argument("--wedge-angle-deg", type=float, default=45.0)
    args = parser.parse_args(argv)
    report = run_campaign(args.output, steps=args.steps, bow_speed=args.bow_speed, wedge_angle_deg=args.wedge_angle_deg)
    print(json.dumps(report, indent=2))
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
