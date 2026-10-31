"""DEM3D large-scale particle benchmark: 10k / 50k / 100k wall-clock scaling study.

Measures per-step wall-clock time, energy drift, and reproducibility across
three particle-count regimes.  The ideal linear-scaling baseline assumes
O(N) work per step; the acceptance criterion requires the 100k wall-clock
to be < 2× the ideal extrapolation from the 10k measurement.

Usage:
    python scripts/benchmark_dem3d_scale.py --output results-dem3d-scale
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.dem3d import DEM3DConfig
from tensordem.dem3d_energy import EnergyAuditedIceDEM3D

# ---------------------------------------------------------------------------
# Scale definitions  (nx≈ny≈nz cube → N≈nx³ particles)
# ---------------------------------------------------------------------------
SCALE_CASES = [
    {"id": "scale_10k",  "target_n": 10_000, "nx": 22, "ny": 22, "nz": 22},
    {"id": "scale_50k",  "target_n": 50_000, "nx": 37, "ny": 37, "nz": 37},
    {"id": "scale_100k", "target_n": 100_000, "nx": 47, "ny": 47, "nz": 47},
]

# Benchmark protocol constants
STEPS = 10
SAMPLE_EVERY = 5
TOOL_SPEED = 0.01  # m/s – small to limit bond breakage during scaling study
DT_FRACTION = 0.10  # fraction of recommended_dt
WARMUP_STEPS = 2  # minimal warmup to avoid timeout on large cases


def _build_config(nx: int, ny: int, nz: int, device: str = "cpu") -> DEM3DConfig:
    """Build a DEM3DConfig for a given grid size with benchmark-friendly defaults."""
    base = DEM3DConfig(
        nx=nx, ny=ny, nz=nz,
        radius=0.025,
        density=917.0,
        bond_stiffness=2000.0,
        contact_stiffness=5000.0,
        contact_damping=5.0,
        drag=2.0,
        tool_radius=0.1,
        tool_speed=TOOL_SPEED,
        tool_gap=0.01,
        breaking_strain=0.015,
        shear_breaking_strain=0.03,
        fix_x_edges=False,
        fix_bottom=False,
        device=device,
    )
    cfg_dict = dict(base.__dict__)
    cfg_dict["dt"] = DT_FRACTION * base.recommended_dt
    return DEM3DConfig(**cfg_dict)


def _state_digest(sim: EnergyAuditedIceDEM3D) -> str:
    """Deterministic SHA-256 of solver state for reproducibility check."""
    h = hashlib.sha256()
    for tensor in (sim.positions, sim.velocities, sim.alive, sim.failure_mode,
                   sim.failure_time_s, sim.failure_extension_m, sim.failure_energy_J):
        value = tensor.detach().cpu().contiguous()
        h.update(str(value.dtype).encode("ascii"))
        h.update(json.dumps(list(value.shape)).encode("ascii"))
        h.update(bytes(value.view(torch.uint8).reshape(-1).tolist()))
    return h.hexdigest()


def run_scale_case(
    case_id: str,
    nx: int, ny: int, nz: int,
    steps: int = STEPS,
    sample_every: int = SAMPLE_EVERY,
    device: str = "cpu",
    output: Path | None = None,
) -> dict[str, Any]:
    """Run a single scale case and return metrics."""
    torch.set_num_threads(1)

    config = _build_config(nx, ny, nz, device=device)
    n_particles_target = nx * ny * nz

    # --- First run (timed + sampled) ---
    t_init_start = time.perf_counter()
    sim = EnergyAuditedIceDEM3D(config)
    t_init_end = time.perf_counter()
    n_particles = len(sim.positions)

    # Warm-up: minimal steps
    for _ in range(WARMUP_STEPS):
        sim.step()

    # Timed run
    rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    for step in range(1, steps + 1):
        sim.step()
        if step % sample_every == 0 or step == steps:
            row = sim.diagnostics()
            rows.append({"step": step, **row})
    t_end = time.perf_counter()
    wallclock_s = t_end - t_start
    digest_1 = _state_digest(sim)

    # --- Second run (reproducibility check, no sampling, no diagnostics) ---
    sim2 = EnergyAuditedIceDEM3D(config)
    for _ in range(WARMUP_STEPS):
        sim2.step()
    for _ in range(steps):
        sim2.step()
    digest_2 = _state_digest(sim2)
    reproducible = digest_1 == digest_2

    # --- Compute metrics ---
    final = rows[-1]
    energy_drift_per_step = abs(final["energy_balance_residual_J"]) / max(steps, 1)
    energy_scale = max(
        abs(final.get("mechanical_energy_J", 0)),
        abs(final.get("tool_work_cumulative_J", 0)),
        1e-30,
    )
    energy_drift_ratio = energy_drift_per_step / energy_scale

    result = {
        "case_id": case_id,
        "target_n": n_particles_target,
        "actual_n": n_particles,
        "nx": nx, "ny": ny, "nz": nz,
        "steps": steps,
        "dt_s": config.dt,
        "init_wallclock_s": t_init_end - t_init_start,
        "wallclock_s": wallclock_s,
        "wallclock_per_step_ms": wallclock_s / steps * 1000,
        "energy_drift_per_step_J": energy_drift_per_step,
        "energy_drift_ratio": energy_drift_ratio,
        "reproducible_hash": reproducible,
        "digest": digest_1,
        "broken_bonds": final["broken_bonds"],
        "peak_abs_reaction_z_N": max(abs(float(r["reaction_z"])) for r in rows),
        "final_mechanical_energy_J": final["mechanical_energy_J"],
        "energy_balance_residual_J": final["energy_balance_residual_J"],
    }

    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        case_dir = output / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with (case_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    return result


def compute_scaling_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute ideal linear scaling comparison and acceptance verdict."""
    r10k = next((r for r in results if "10k" in r["case_id"]), None)
    r50k = next((r for r in results if "50k" in r["case_id"]), None)
    r100k = next((r for r in results if "100k" in r["case_id"]), None)

    report: dict[str, Any] = {"cases": results}

    if r10k and r100k:
        n10k = r10k["actual_n"]
        n100k = r100k["actual_n"]
        t10k = r10k["wallclock_s"]
        t100k = r100k["wallclock_s"]

        ideal_t100k = t10k * (n100k / n10k)
        ratio = t100k / ideal_t100k if ideal_t100k > 0 else float("inf")

        report["wallclock_100k_ratio_vs_ideal"] = ratio
        report["wallclock_10k_s"] = t10k
        report["wallclock_50k_s"] = r50k["wallclock_s"] if r50k else None
        report["wallclock_100k_s"] = t100k
        report["ideal_wallclock_100k_s"] = ideal_t100k
    else:
        report["wallclock_100k_ratio_vs_ideal"] = None

    ref = r100k or r10k
    if ref:
        report["energy_drift_per_step"] = ref["energy_drift_ratio"]
    else:
        report["energy_drift_per_step"] = None

    report["reproducible_hash"] = all(r["reproducible_hash"] for r in results) if results else False

    criteria = {
        "wallclock_100k_ratio_vs_ideal": {
            "value": report.get("wallclock_100k_ratio_vs_ideal"),
            "threshold": 2.0,
            "unit": "ratio",
            "status": "PASS" if (report.get("wallclock_100k_ratio_vs_ideal") is not None
                                 and report["wallclock_100k_ratio_vs_ideal"] < 2.0) else "FAIL",
        },
        "energy_drift_per_step": {
            "value": report.get("energy_drift_per_step"),
            "threshold": 1e-06,
            "unit": "ratio",
            "status": "PASS" if (report.get("energy_drift_per_step") is not None
                                 and report["energy_drift_per_step"] < 1e-06) else "FAIL",
        },
        "reproducible_hash": {
            "value": report.get("reproducible_hash"),
            "expected": True,
            "unit": "bool",
            "status": "PASS" if report.get("reproducible_hash") else "FAIL",
        },
    }
    report["acceptance_criteria"] = criteria
    report["all_criteria_met"] = all(c["status"] == "PASS" for c in criteria.values())

    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DEM3D large-scale particle benchmark")
    parser.add_argument("--output", type=Path, default=Path("results-dem3d-scale"))
    parser.add_argument("--case", action="append", dest="cases",
                        help="case id to run (scale_10k, scale_50k, scale_100k); default runs all")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    requested = set(args.cases or [])
    cases_to_run = [c for c in SCALE_CASES if not requested or c["id"] in requested]
    if not cases_to_run:
        parser.error(f"no matching cases; available: {[c['id'] for c in SCALE_CASES]}")

    results = []
    for case in cases_to_run:
        print(f"\n{'='*60}")
        print(f"Running {case['id']} (target ~{case['target_n']:,} particles)")
        print(f"{'='*60}")
        result = run_scale_case(
            case["id"], case["nx"], case["ny"], case["nz"],
            steps=args.steps, device=args.device, output=args.output,
        )
        print(f"  Particles: {result['actual_n']:,}")
        print(f"  Init wall-clock: {result['init_wallclock_s']:.2f}s")
        print(f"  Step wall-clock: {result['wallclock_s']:.2f}s ({result['wallclock_per_step_ms']:.1f}ms/step)")
        print(f"  Energy drift/step: {result['energy_drift_per_step_J']:.2e} J (ratio: {result['energy_drift_ratio']:.2e})")
        print(f"  Reproducible: {result['reproducible_hash']}")
        results.append(result)

    report = compute_scaling_report(results)
    (args.output / "scaling_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print("SCALING REPORT")
    print(f"{'='*60}")
    print(json.dumps({
        "wallclock_100k_ratio_vs_ideal": report.get("wallclock_100k_ratio_vs_ideal"),
        "energy_drift_per_step": report.get("energy_drift_per_step"),
        "reproducible_hash": report.get("reproducible_hash"),
        "all_criteria_met": report.get("all_criteria_met"),
    }, indent=2))

    if not report.get("all_criteria_met"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
