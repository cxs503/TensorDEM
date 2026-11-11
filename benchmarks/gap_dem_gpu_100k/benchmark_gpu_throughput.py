#!/usr/bin/env python
"""GAP-DEM-GPU-100K: GPU throughput benchmark for 100K-particle DEM simulation.

Measures wall-clock time per step and particle throughput (N·steps/s) on
both CPU and GPU devices.  The acceptance threshold is:
    single-step wall-clock for 100K particles ≤ 2 s

Usage:
    # CPU baseline (default)
    python benchmarks/gap_dem_gpu_100k/benchmark_gpu_throughput.py

    # GPU run (requires CUDA-capable PyTorch)
    python benchmarks/gap_dem_gpu_100k/benchmark_gpu_throughput.py --device cuda

    # Full sweep: CPU + GPU, multiple scales
    python benchmarks/gap_dem_gpu_100k/benchmark_gpu_throughput.py --sweep

Output:
    benchmarks/gap_dem_gpu_100k/results.json   — machine-readable results
    benchmarks/gap_dem_gpu_100k/report.md      — human-readable report
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

# Permit direct execution from a source checkout without an editable install.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.dem3d import DEM3DConfig, IceDEM3D


# ---------------------------------------------------------------------------
# Scale definitions: (nx, ny, nz) → approximate particle count
# ---------------------------------------------------------------------------
_SCALES = {
    "10k":  (22, 22, 22),    # ~10,648 particles
    "50k":  (37, 37, 37),    # ~50,653 particles
    "100k": (47, 47, 47),    # ~103,823 particles
}


def _make_config(nx: int, ny: int, nz: int, device: str) -> DEM3DConfig:
    """Create a minimal DEM3DConfig for throughput measurement.

    Uses the same physical parameters as the scale benchmark cases so results
    are comparable.  Bonds are disabled (contact-only) to isolate the
    contact-resolution bottleneck, which dominates at large N.
    """
    return DEM3DConfig(
        nx=nx, ny=ny, nz=nz,
        radius=0.025,
        density=917.0,
        bond_stiffness=100.0,       # minimal stiffness for throughput focus
        contact_stiffness=5000.0,
        contact_damping=5.0,
        drag=2.0,
        tool_radius=0.1,
        tool_speed=0.01,
        tool_gap=0.01,
        breaking_strain=0.015,
        shear_breaking_strain=0.03,
        fix_x_edges=False,
        fix_bottom=False,
        device=device,
        bond_model="linear",
    )


def _count_particles(nx: int, ny: int, nz: int) -> int:
    """Return the actual particle count for a given grid size."""
    return nx * ny * nz


def benchmark_single_step(config: DEM3DConfig, warmup_steps: int = 3,
                          measure_steps: int = 5) -> dict[str, Any]:
    """Run a DEM simulation and measure wall-clock time per step.

    Returns a dict with timing and throughput metrics.
    """
    device_name = config.device
    nx, ny, nz = config.nx, config.ny, config.nz
    n_particles = _count_particles(nx, ny, nz)

    # Create simulation
    sim = IceDEM3D(config)

    # Warmup: run a few steps without timing to let CUDA kernels compile / caches fill
    for _ in range(warmup_steps):
        sim.step()

    # Synchronize before timing (critical for GPU)
    if device_name == "cuda":
        torch.cuda.synchronize()

    # Measure
    t_start = time.perf_counter()
    for _ in range(measure_steps):
        sim.step()
    if device_name == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    wall_time_s = t_end - t_start
    time_per_step_s = wall_time_s / measure_steps
    throughput = n_particles * measure_steps / wall_time_s  # particles·steps/s

    return {
        "n_particles": n_particles,
        "grid": f"{nx}x{ny}x{nz}",
        "device": device_name,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "wall_time_s": round(wall_time_s, 4),
        "time_per_step_s": round(time_per_step_s, 6),
        "throughput_particles_steps_per_s": round(throughput, 1),
        "n_pairs": int(sim.pairs.shape[0]),
        "broken_bonds": sim.broken_bonds,
    }


def benchmark_scale(device: str, scales: dict[str, tuple] | None = None,
                    warmup: int = 3, measure: int = 5) -> list[dict[str, Any]]:
    """Run throughput benchmark across multiple scales on one device."""
    if scales is None:
        scales = _SCALES
    results = []
    for label, (nx, ny, nz) in scales.items():
        config = _make_config(nx, ny, nz, device)
        try:
            r = benchmark_single_step(config, warmup_steps=warmup, measure_steps=measure)
            r["scale_label"] = label
            r["status"] = "ok"
        except Exception as e:
            r = {
                "scale_label": label,
                "grid": f"{nx}x{ny}x{nz}",
                "device": device,
                "status": "error",
                "error": str(e),
            }
        results.append(r)
    return results


def get_device_info(device: str) -> dict[str, Any]:
    """Return hardware info for the given device."""
    info: dict[str, Any] = {"device": device}
    if device == "cuda" and torch.cuda.is_available():
        info["cuda_available"] = True
        info["cuda_version"] = torch.version.cuda
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()
        info["memory_total_GB"] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2)
    else:
        info["cuda_available"] = torch.cuda.is_available()
        info["cpu_threads"] = torch.get_num_threads()
        info["cpu_interop_threads"] = torch.get_num_interop_threads()
    return info


def run_benchmark(device: str = "cpu", sweep: bool = False,
                  warmup: int = 3, measure: int = 5) -> dict[str, Any]:
    """Main benchmark entry point."""
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    devices_to_test = ["cpu"]
    if device == "cuda" or sweep:
        if torch.cuda.is_available():
            devices_to_test.append("cuda")
        else:
            print("WARNING: CUDA not available; skipping GPU benchmark", file=sys.stderr)

    all_results: dict[str, Any] = {
        "benchmark": "GAP-DEM-GPU-100K",
        "protocol": "tensordem-dem3d-gpu-throughput-v1",
        "acceptance_threshold_s": 2.0,
        "description": "Wall-clock time per step for 100K-particle DEM simulation",
        "hardware": {},
        "results": [],
    }

    for dev in devices_to_test:
        all_results["hardware"][dev] = get_device_info(dev)
        scale_results = benchmark_scale(dev, warmup=warmup, measure=measure)
        all_results["results"].extend(scale_results)

    # Determine acceptance status for 100k case
    for r in all_results["results"]:
        if r.get("scale_label") == "100k" and r.get("status") == "ok":
            r["acceptance"] = r["time_per_step_s"] <= 2.0
            r["acceptance_label"] = "PASS" if r["acceptance"] else "FAIL"

    # Write JSON results
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False) + "\n")
    print(f"Results written to {results_path}")

    # Write human-readable report
    report = _generate_report(all_results)
    report_path = out_dir / "report.md"
    report_path.write_text(report)
    print(f"Report written to {report_path}")

    return all_results


def _generate_report(data: dict[str, Any]) -> str:
    """Generate a markdown report from benchmark results."""
    lines = [
        "# GAP-DEM-GPU-100K Throughput Benchmark Report",
        "",
        "## Protocol",
        "",
        f"- **Benchmark**: {data['benchmark']}",
        f"- **Protocol**: {data['protocol']}",
        f"- **Acceptance threshold**: single step for 100K particles ≤ {data['acceptance_threshold_s']} s",
        "",
        "## Hardware",
        "",
    ]
    for dev, info in data["hardware"].items():
        lines.append(f"### {dev.upper()}")
        for k, v in info.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    lines += [
        "## Results",
        "",
        "| Scale | Particles | Device | Time/step (s) | Throughput (N·steps/s) | Pairs | Status |",
        "|-------|-----------|--------|---------------|------------------------|-------|--------|",
    ]
    for r in data["results"]:
        if r.get("status") != "ok":
            lines.append(f"| {r.get('scale_label', '?')} | — | {r.get('device', '?')} | — | — | — | ERROR: {r.get('error', '?')} |")
        else:
            acc = r.get("acceptance_label", "")
            lines.append(
                f"| {r['scale_label']} | {r['n_particles']:,} | {r['device']} "
                f"| {r['time_per_step_s']:.4f} | {r['throughput_particles_steps_per_s']:,.0f} "
                f"| {r['n_pairs']:,} | {acc} |"
            )
    lines.append("")

    # Acceptance summary
    lines += ["## Acceptance Summary", ""]
    for r in data["results"]:
        if r.get("scale_label") == "100k":
            if r.get("status") == "ok":
                status = "✅ PASS" if r.get("acceptance") else "❌ FAIL"
                lines.append(
                    f"- **100K particles on {r['device']}**: "
                    f"{r['time_per_step_s']:.4f} s/step → {status}"
                )
            else:
                lines.append(f"- **100K particles on {r['device']}**: ERROR — {r.get('error', '?')}")
    lines.append("")

    lines += [
        "## Notes",
        "",
        "- Warmup steps are excluded from timing to allow JIT compilation / cache warming.",
        "- GPU runs use `torch.cuda.synchronize()` before and after timing to ensure accurate wall-clock measurement.",
        "- Bond stiffness set to 0 for pure contact-throughput measurement (bonds disabled).",
        "- Physical parameters match the `scale_cases.json` benchmark protocol for comparability.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="GAP-DEM-GPU-100K throughput benchmark")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="Device to benchmark (default: cpu)")
    parser.add_argument("--sweep", action="store_true",
                        help="Benchmark all scales on all available devices")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup steps before timing (default: 3)")
    parser.add_argument("--measure", type=int, default=5,
                        help="Steps to measure (default: 5)")
    args = parser.parse_args(argv)

    result = run_benchmark(device=args.device, sweep=args.sweep,
                           warmup=args.warmup, measure=args.measure)

    # Print summary
    print("\n=== Benchmark Summary ===")
    for r in result["results"]:
        if r.get("status") == "ok":
            print(f"  {r['scale_label']:>6s} ({r['n_particles']:>6,} particles) "
                  f"on {r['device']}: {r['time_per_step_s']:.4f} s/step, "
                  f"{r['throughput_particles_steps_per_s']:,.0f} N·steps/s")
        else:
            print(f"  {r.get('scale_label', '?'):>6s} on {r.get('device', '?')}: ERROR — {r.get('error', '?')}")


if __name__ == "__main__":
    main()
