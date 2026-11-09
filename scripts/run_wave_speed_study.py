#!/usr/bin/env python
"""Run the full granular wave-speed study and save results.

Usage:
    python scripts/run_wave_speed_study.py [--nx N] [--nz N] [--seed S] [--out PATH]

Produces a JSON report with all acceptance criteria evaluated.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import sys
from pathlib import Path

import numpy as np

from tensordem.granular_wave.engine import GranularWaveConfig, GranularPacking
from tensordem.granular_wave.emt import (
    digby_hertz_mindlin,
    affine_contact_medium,
    loglog_slope,
    poisson_from_velocities,
)
from tensordem.granular_wave.measure import pulse_wave_speed, sweep_wave_speed
from tensordem.granular_wave.study import WaveSpeedStudy, WaveStudyConfig


def run_full_study(nx: int = 5, ny: int = 5, nz: int = 20, seed: int = 42,
                   out_path: str = "runs/wave_speed_study.json") -> dict:
    """Run the complete wave-speed verification study."""
    cfg = GranularWaveConfig(
        nx=nx, ny=ny, nz=nz,
        pressure=1.0e5,
        seed=seed,
        young_modulus=5.0e9,
        poisson_ratio=0.30,
        density=2500.0,
        friction=0.30,
        normal_damping=0.10,
        tangential_damping=0.10,
    )

    # Six solid fractions spanning the pressure range
    solid_fractions = (0.540, 0.555, 0.570, 0.585, 0.600, 0.615)

    study_cfg = WaveStudyConfig(
        base=cfg,
        solid_fractions=solid_fractions,
        prepare_solid_fraction=0.520,
        relax_steps=500,
        shear_relax_steps=200,
        run_dynamic=True,
        run_negative_control=True,
        run_size_test=True,
        pulse_amplitude=0.02,
        sweep_amplitude=0.02,
        fractions=(0.15, 0.85),
    )

    t0 = time.time()
    study = WaveSpeedStudy(study_cfg)
    report = study.run()
    elapsed = time.time() - t0
    report["elapsed_s"] = elapsed
    report["n_particles"] = nx * ny * nz

    # Save
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Study saved to {out} ({elapsed:.1f}s, N={nx*ny*nz})")
    return report


def run_damping_sweep(nx: int = 4, ny: int = 4, nz: int = 16, seed: int = 42) -> dict:
    """Sweep contact damping Z and measure Q⁻¹ to verify monotonic relationship."""
    dampings = [0.02, 0.05, 0.10, 0.20, 0.40, 0.60]
    results = []

    for zeta in dampings:
        cfg = GranularWaveConfig(
            nx=nx, ny=ny, nz=nz,
            pressure=1.0e5,
            seed=seed,
            normal_damping=zeta,
            tangential_damping=zeta,
        )
        sim = GranularPacking(cfg)
        sim.prepare(solid_fraction=0.520)
        sim.set_solid_fraction(0.580, relax_steps=400)

        state = sim.state()
        pred = digby_hertz_mindlin(
            coordination_number=state["coordination_number"],
            porosity=state["porosity"],
            grain_shear_modulus=sim.cfg.shear_modulus,
            grain_poisson_ratio=sim.cfg.poisson_ratio,
            pressure=state["pressure_mean_Pa"],
            bulk_density=state["bulk_density_kg_m3"],
        )

        pulse = pulse_wave_speed(
            sim, mode="P", amplitude=0.02,
            fractions=(0.15, 0.85), sponge_start=0.93,
            v_ref=pred["v_p_m_s"],
        )

        results.append({
            "damping_zeta": zeta,
            "q_inv_peak": pulse.q_inv_peak,
            "q_inv_spectral": pulse.q_inv_spectral,
            "speed_m_s": pulse.speed_m_s,
            "valid": pulse.valid,
        })
        print(f"  zeta={zeta:.2f}: Q⁻¹(peak)={pulse.q_inv_peak:.4f}, "
              f"Q⁻¹(spec)={pulse.q_inv_spectral:.4f}, v_p={pulse.speed_m_s:.1f} m/s")

    # Check monotonicity
    q_vals = [r["q_inv_peak"] for r in results if math.isfinite(r["q_inv_peak"])]
    monotonic = all(q_vals[i] <= q_vals[i + 1] + 1e-9 for i in range(len(q_vals) - 1))

    return {"damping_sweep": results, "monotonic": monotonic}


def run_servo_convergence_test(nx: int = 4, ny: int = 4, nz: int = 16,
                                seed: int = 42) -> dict:
    """Test servo convergence: Z displacement change < 0.01% in 100 steps."""
    cfg = GranularWaveConfig(
        nx=nx, ny=ny, nz=nz,
        pressure=1.0e5,
        seed=seed,
    )
    sim = GranularPacking(cfg)
    sim.prepare(solid_fraction=0.520)

    # Set to a target solid fraction and measure convergence
    target_sf = 0.580
    sim.set_solid_fraction(target_sf, relax_steps=100)

    # Record Z position history over 100 steps
    z_history = []
    for _ in range(100):
        sim.servo(cfg.pressure)
        sim.step()
        z_history.append(float(sim.wall_pos[5]))

    z_init = z_history[0]
    z_final = z_history[-1]
    z_range = max(z_history) - min(z_history)
    rel_change = z_range / z_init if z_init != 0 else float("inf")

    return {
        "z_initial": z_init,
        "z_final": z_final,
        "z_range": z_range,
        "relative_change": rel_change,
        "converged": rel_change < 1e-4,  # 0.01%
        "history_last_10": z_history[-10:],
    }


def run_zero_stiffness_control(nx: int = 4, ny: int = 4, nz: int = 16,
                                seed: int = 42) -> dict:
    """Negative control: k_n → 0 (via soft_scale) should give v_p → 0."""
    cfg = GranularWaveConfig(
        nx=nx, ny=ny, nz=nz,
        pressure=1.0e5,
        seed=seed,
        young_modulus=5.0e9,
    )

    results = []
    for soft in [1.0, 0.1, 0.01, 0.001]:
        sim = GranularPacking(cfg)
        sim.soft_scale = soft
        sim.prepare(solid_fraction=0.520)
        sim.set_solid_fraction(0.580, relax_steps=400)

        state = sim.state()
        pred = digby_hertz_mindlin(
            coordination_number=state["coordination_number"],
            porosity=state["porosity"],
            grain_shear_modulus=sim.cfg.shear_modulus * soft,
            grain_poisson_ratio=sim.cfg.poisson_ratio,
            pressure=state["pressure_mean_Pa"],
            bulk_density=state["bulk_density_kg_m3"],
        )

        try:
            pulse = pulse_wave_speed(
                sim, mode="P", amplitude=0.02,
                fractions=(0.15, 0.85), sponge_start=0.93,
                v_ref=max(pred["v_p_m_s"], 10.0),
            )
            results.append({
                "soft_scale": soft,
                "speed_m_s": pulse.speed_m_s,
                "valid": pulse.valid,
            })
        except Exception as e:
            results.append({
                "soft_scale": soft,
                "speed_m_s": float("nan"),
                "valid": False,
                "error": str(e),
            })

    return {"zero_stiffness_control": results}


def run_size_convergence(seed: int = 42) -> dict:
    """Particle number convergence: N ∈ {500, 1000, 2000, 4000}, v_p change < 2%."""
    configs = [
        (8, 8, 8),    # 512
        (10, 10, 10), # 1000
        (12, 12, 12), # 1728
        (16, 16, 16), # 4096
    ]

    results = []
    for nx, ny, nz in configs:
        cfg = GranularWaveConfig(
            nx=nx, ny=ny, nz=nz,
            pressure=1.0e5,
            seed=seed,
        )
        sim = GranularPacking(cfg)
        sim.prepare(solid_fraction=0.520)
        sim.set_solid_fraction(0.580, relax_steps=400)

        state = sim.state()
        pred = digby_hertz_mindlin(
            coordination_number=state["coordination_number"],
            porosity=state["porosity"],
            grain_shear_modulus=sim.cfg.shear_modulus,
            grain_poisson_ratio=sim.cfg.poisson_ratio,
            pressure=state["pressure_mean_Pa"],
            bulk_density=state["bulk_density_kg_m3"],
        )

        pulse = pulse_wave_speed(
            sim, mode="P", amplitude=0.02,
            fractions=(0.15, 0.85), sponge_start=0.93,
            v_ref=pred["v_p_m_s"],
        )

        results.append({
            "n_particles": nx * ny * nz,
            "nx": nx, "ny": ny, "nz": nz,
            "speed_m_s": pulse.speed_m_s,
            "coordination_number": state["coordination_number"],
            "pressure_Pa": state["pressure_mean_Pa"],
            "valid": pulse.valid,
        })
        print(f"  N={nx*ny*nz}: v_p={pulse.speed_m_s:.1f} m/s, Z={state['coordination_number']:.2f}")

    # Check convergence
    speeds = [r["speed_m_s"] for r in results if math.isfinite(r["speed_m_s"])]
    if len(speeds) >= 2:
        max_dev = max(abs(s - speeds[-1]) / speeds[-1] for s in speeds[:-1])
    else:
        max_dev = float("nan")

    return {"size_convergence": results, "max_deviation": max_dev, "converged": max_dev < 0.02}


def main():
    parser = argparse.ArgumentParser(description="Run granular wave-speed study")
    parser.add_argument("--nx", type=int, default=5)
    parser.add_argument("--ny", type=int, default=5)
    parser.add_argument("--nz", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="runs/wave_speed_study.json")
    parser.add_argument("--damping-sweep", action="store_true")
    parser.add_argument("--servo-test", action="store_true")
    parser.add_argument("--zero-stiffness", action="store_true")
    parser.add_argument("--size-convergence", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    report = {}

    if args.all or not (args.damping_sweep or args.servo_test or args.zero_stiffness or args.size_convergence):
        print("=== Running full wave-speed study ===")
        report["study"] = run_full_study(args.nx, args.ny, args.nz, args.seed, args.out)

    if args.all or args.damping_sweep:
        print("=== Running damping sweep ===")
        report["damping_sweep"] = run_damping_sweep()

    if args.all or args.servo_test:
        print("=== Running servo convergence test ===")
        report["servo_convergence"] = run_servo_convergence_test()

    if args.all or args.zero_stiffness:
        print("=== Running zero-stiffness control ===")
        report["zero_stiffness"] = run_zero_stiffness_control()

    if args.all or args.size_convergence:
        print("=== Running size convergence test ===")
        report["size_convergence"] = run_size_convergence()

    # Save combined report
    combined_path = Path(args.out).parent / "wave_speed_full_report.json"
    with open(combined_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to {combined_path}")


if __name__ == "__main__":
    main()
