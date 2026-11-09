#!/usr/bin/env python3
"""BPM calibration runner: quasi-static gate + size effect + UCS comparison.

This script runs the complete BPM calibration pipeline:
(a) Quasi-static gate: β∈{100,400,1600,3600}, KE/external_work ≤ 1e-3
(b) Size effect: t=0.1/0.2/0.4m, n≥8 specimens, Weibull bootstrap CI
(c) UCS vs 3 literature sources cross-validation
(d) Dual-coefficient friction regularization

Usage:
    python scripts/bpm_calibration_runner.py --output results/bpm_calibration

The script uses the existing DEM3D engine (tensordem.dem3d) for UCS and
Brazilian test simulations, and the new bpm_calibration module for
statistical analysis.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.bpm_calibration import (
    check_quasi_static,
    quasi_static_convergence,
    weibull_mle,
    weibull_bootstrap_ci,
    size_effect_weibull_fit,
    DualFrictionConfig,
    LITERATURE_UCS_RANGES,
    generate_synthetic_strengths,
    generate_size_effect_data,
)


def run_ucs_at_beta(
    beta: float,
    *,
    nx: int = 7, ny: int = 7, nz: int = 5,
    radius: float = 0.020,
    bond_stiffness: float = 5000.0,
    breaking_strain: float = 0.008,
    contact_stiffness: float = 10000.0,
    platen_stiffness: float = 20000.0,
    density: float = 917.0,
    target_strain: float = 0.005,
    static_friction: float = 0.0,
    dynamic_friction: float = 0.0,
) -> dict[str, Any]:
    """Run UCS simulation at a given loading rate parameter β.

    β = t_wave / t_loading where t_wave = H/c_s (wave transit time)
    and t_loading = H / (2*v_platen) (time to reach target strain).

    Higher β = slower loading = more quasi-static.
    """
    # Compute wave speed (approximate for bonded lattice)
    # c_s ≈ sqrt(k/(m/a)) where k=bond_stiffness, m=particle_mass, a=2*radius
    particle_mass = density * (4/3) * math.pi * radius**3
    a = 2 * radius
    c_s = math.sqrt(bond_stiffness * a / particle_mass)

    H = nz * 2 * radius  # specimen height
    t_wave = H / c_s

    # Loading velocity to achieve β
    t_loading = t_wave * beta
    v_platen = H * target_strain / (2 * t_loading)

    # Compute steps
    config = DEM3DConfig(
        nx=nx, ny=ny, nz=nz,
        radius=radius,
        density=density,
        bond_stiffness=bond_stiffness,
        contact_stiffness=contact_stiffness,
        breaking_strain=breaking_strain,
        platen_stiffness=platen_stiffness,
        platen_damping=0.5,
        top_platen_enabled=True,
        bottom_platen_enabled=True,
        top_platen_velocity=-v_platen,
        bottom_platen_velocity=v_platen,
        drag=0.1,
        contact_damping=0.5,
        fix_x_edges=False,
        fix_bottom=False,
        static_friction_coefficient=static_friction,
        dynamic_friction_coefficient=dynamic_friction,
        friction_transition_velocity=1e-3,
    )
    dt = config.recommended_dt * 0.5
    steps = max(100, int(target_strain * H / (2 * v_platen * dt)))

    sim = IceDEM3D(DEM3DConfig(**{**config.__dict__, "dt": dt}))
    sim.tool_start[:] = torch.tensor([1e6, 1e6, 1e6], dtype=sim.dtype)
    sim.tool_velocity.zero_()

    gross_area = (nx * 2 * radius) * (ny * 2 * radius)
    initial_height = nz * 2 * radius

    history = []
    max_ke = 0.0
    total_work = 0.0
    peak_stress = 0.0
    strain_at_peak = 0.0

    for step in range(steps + 1):
        if step > 0:
            sim.step()
        d = sim.diagnostics()
        top_load = abs(float(d["top_platen_reaction_z_N"]))
        bottom_load = abs(float(d["bottom_platen_reaction_z_N"]))
        load = 0.5 * (top_load + bottom_load)
        closure = 2 * v_platen * float(d["time"])
        strain = closure / initial_height
        stress = load / gross_area
        ke = float(d["kinetic_energy_J"])

        # External work: integral of load * d(closure)
        if step > 0:
            d_closure = 2 * v_platen * dt
            total_work += load * d_closure

        max_ke = max(max_ke, ke)
        if stress > peak_stress:
            peak_stress = stress
            strain_at_peak = strain

        if step % max(1, steps // 50) == 0 or step == steps:
            history.append({
                "step": step,
                "time_s": float(d["time"]),
                "kinetic_energy_J": ke,
                "external_work_J": total_work,
                "engineering_stress_Pa": stress,
                "axial_engineering_strain": strain,
                "broken_bonds": int(d["broken_bonds"]),
                "bond_elastic_energy_J": float(d["bond_elastic_energy_J"]),
            })

    return {
        "beta": beta,
        "v_platen_m_s": v_platen,
        "steps": steps,
        "dt": dt,
        "history": history,
        "peak_stress_Pa": peak_stress,
        "strain_at_peak": strain_at_peak,
        "max_ke_J": max_ke,
        "total_work_J": total_work,
        "ke_ratio": max_ke / max(total_work, 1e-30),
        "final_broken_bonds": int(sim.broken_bonds),
    }


def run_brazilian_specimen(
    *,
    nx: int = 11, ny: int = 3, nz: int = 11,
    radius: float = 0.015,
    bond_stiffness: float = 5000.0,
    breaking_strain: float = 0.005,
    contact_stiffness: float = 10000.0,
    platen_stiffness: float = 20000.0,
    density: float = 917.0,
    v_platen: float = 0.001,
    max_steps: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run a single Brazilian (indirect tensile) test specimen.

    Returns the splitting stress σ_t = 2P/(πDt) at failure.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = DEM3DConfig(
        nx=nx, ny=ny, nz=nz,
        radius=radius,
        density=density,
        bond_stiffness=bond_stiffness,
        contact_stiffness=contact_stiffness,
        breaking_strain=breaking_strain,
        platen_stiffness=platen_stiffness,
        platen_damping=0.5,
        top_platen_enabled=True,
        bottom_platen_enabled=True,
        top_platen_velocity=-v_platen,
        bottom_platen_velocity=v_platen,
        drag=0.1,
        contact_damping=0.5,
        fix_x_edges=False,
        fix_bottom=False,
        geometry="cylinder_xz",
    )
    dt = config.recommended_dt * 0.5
    sim = IceDEM3D(DEM3DConfig(**{**config.__dict__, "dt": dt}))
    sim.tool_start[:] = torch.tensor([1e6, 1e6, 1e6], dtype=sim.dtype)
    sim.tool_velocity.zero_()

    D = (nx * 2 * radius)  # diameter
    t = (ny * 2 * radius)  # thickness

    peak_load = 0.0
    broken_at_peak = 0

    for step in range(max_steps + 1):
        if step > 0:
            sim.step()
        d = sim.diagnostics()
        top_load = abs(float(d["top_platen_reaction_z_N"]))
        bottom_load = abs(float(d["bottom_platen_reaction_z_N"]))
        load = 0.5 * (top_load + bottom_load)

        if load > peak_load:
            peak_load = load
            broken_at_peak = int(d["broken_bonds"])

        # Stop if significant bond breaking has occurred (failure)
        if int(d["broken_bonds"]) > max(3, 0.05 * len(sim.pairs)):
            break

    # Splitting stress: σ_t = 2P/(πDt)
    splitting_stress = 2 * peak_load / (math.pi * D * t)

    return {
        "peak_load_N": peak_load,
        "splitting_stress_Pa": splitting_stress,
        "diameter_m": D,
        "thickness_m": t,
        "broken_bonds_at_peak": broken_at_peak,
        "final_broken_bonds": int(sim.broken_bonds),
        "nx": nx, "ny": ny, "nz": nz,
        "radius_m": radius,
    }


def run_size_effect_campaign(
    thicknesses: list[float],
    n_per_thickness: int,
    *,
    base_nx: int = 11,
    base_nz: int = 11,
    radius: float = 0.015,
    bond_stiffness: float = 5000.0,
    breaking_strain: float = 0.005,
    seed_base: int = 42,
) -> dict[float, np.ndarray]:
    """Run Brazilian tests at multiple thicknesses for size effect study."""
    strengths = {}
    for t_idx, t_target in enumerate(thicknesses):
        # Compute ny to achieve target thickness
        ny = max(3, round(t_target / (2 * radius)))
        actual_t = ny * 2 * radius

        specimen_strengths = []
        for n in range(n_per_thickness):
            result = run_brazilian_specimen(
                nx=base_nx, ny=ny, nz=base_nz,
                radius=radius,
                bond_stiffness=bond_stiffness,
                breaking_strain=breaking_strain,
                seed=seed_base + t_idx * 100 + n,
            )
            specimen_strengths.append(result["splitting_stress_Pa"])
            print(f"  t={actual_t:.3f}m specimen {n+1}/{n_per_thickness}: "
                  f"σ_t = {result['splitting_stress_Pa']:.0f} Pa")

        strengths[actual_t] = np.array(specimen_strengths)
    return strengths


def run_full_calibration(
    output_dir: Path,
    *,
    beta_values: list[float] | None = None,
    thicknesses: list[float] | None = None,
    n_per_thickness: int = 8,
    n_bootstrap: int = 1000,
    use_synthetic: bool = False,
) -> dict[str, Any]:
    """Run the complete BPM calibration pipeline.

    Parameters
    ----------
    output_dir : directory for output files
    beta_values : loading rate parameters for quasi-static gate
    thicknesses : specimen thicknesses for size effect
    n_per_thickness : specimens per thickness (≥8)
    n_bootstrap : bootstrap samples for Weibull CI
    use_synthetic : if True, use synthetic data instead of DEM runs
    """
    if beta_values is None:
        beta_values = [100.0, 400.0, 1600.0, 3600.0]
    if thicknesses is None:
        thicknesses = [0.090, 0.180, 0.360]  # ny=3,6,12 with r=0.015

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    report: dict[str, Any] = {
        "protocol": "bpm-calibration-v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "beta_values": beta_values,
        "thicknesses_m": thicknesses,
        "n_per_thickness": n_per_thickness,
        "n_bootstrap": n_bootstrap,
    }

    # (a) Quasi-static gate
    print("=" * 60)
    print("(a) Quasi-static gate: β ∈", beta_values)
    print("=" * 60)

    if use_synthetic:
        # Synthetic UCS histories for testing
        ucs_results = {}
        for beta in beta_values:
            ke_ratio = 0.05 * (100 / beta)
            history = []
            for step in range(200):
                strain = 0.005 * (step + 1) / 200
                stress = 3e6 * min(strain / 0.003, 1.0)
                ke = ke_ratio * 1.0
                work = 3e6 * 0.005 * (step + 1) / 200
                history.append({
                    "kinetic_energy_J": ke,
                    "external_work_J": work,
                    "engineering_stress_Pa": stress,
                    "axial_engineering_strain": strain,
                })
            ucs_results[beta] = {
                "beta": beta,
                "history": history,
                "peak_stress_Pa": 3e6,
                "ke_ratio": ke_ratio,
                "total_work_J": 3e6 * 0.005,
                "max_ke_J": ke_ratio * 3e6 * 0.005,
            }
    else:
        ucs_results = {}
        for beta in beta_values:
            print(f"  Running UCS at β={beta}...")
            t0 = time.time()
            result = run_ucs_at_beta(beta)
            elapsed = time.time() - t0
            print(f"    peak stress = {result['peak_stress_Pa']:.0f} Pa, "
                  f"KE/work = {result['ke_ratio']:.2e}, "
                  f"time = {elapsed:.1f}s")
            ucs_results[beta] = result

    # Quasi-static assessment
    qs_results = []
    for beta in beta_values:
        r = ucs_results[beta]
        qs = check_quasi_static(r["history"], beta, threshold=1e-3)
        qs_results.append(qs)
        print(f"  β={beta}: KE/work={qs.ke_ratio:.2e}, "
              f"quasi-static={'YES' if qs.is_quasi_static else 'NO'}")

    qs_assessment = quasi_static_convergence(qs_results)
    report["quasi_static"] = qs_assessment
    report["ucs_results"] = {
        str(beta): {k: v for k, v in r.items() if k != "history"}
        for beta, r in ucs_results.items()
    }

    # (b) Size effect
    print("\n" + "=" * 60)
    print("(b) Size effect: t ∈", thicknesses, f"× {n_per_thickness} specimens")
    print("=" * 60)

    if use_synthetic:
        # Synthetic Brazilian strengths
        m_true = 10.0
        sigma_0_ref = 3e6
        brazilian_strengths = generate_size_effect_data(
            thicknesses, n_per_thickness, sigma_0_ref, m_true, seed_base=42
        )
    else:
        brazilian_strengths = run_size_effect_campaign(
            thicknesses, n_per_thickness,
            radius=0.015,
            bond_stiffness=5000.0,
            breaking_strain=0.005,
        )

    # Weibull analysis per thickness
    weibull_per_thickness = {}
    mean_strengths = []
    actual_thicknesses = []
    for t, strengths in sorted(brazilian_strengths.items()):
        m, sigma_0 = weibull_mle(strengths)
        ci = weibull_bootstrap_ci(strengths, n_bootstrap=n_bootstrap)
        weibull_per_thickness[f"t_{t:.3f}m"] = {
            "thickness_m": t,
            "n_specimens": len(strengths),
            "mean_strength_Pa": float(np.mean(strengths)),
            "std_strength_Pa": float(np.std(strengths)),
            "min_strength_Pa": float(np.min(strengths)),
            "max_strength_Pa": float(np.max(strengths)),
            "m_weibull": m,
            "sigma_0_Pa": sigma_0,
            "bootstrap_ci": ci,
            "strengths_Pa": strengths.tolist(),
        }
        mean_strengths.append(float(np.mean(strengths)))
        actual_thicknesses.append(t)
        print(f"  t={t:.3f}m: mean={np.mean(strengths):.0f} Pa, "
              f"m={m:.2f} [{ci['m_ci_low']:.2f}, {ci['m_ci_high']:.2f}]")

    report["weibull_per_thickness"] = weibull_per_thickness

    # Size effect fit
    if len(actual_thicknesses) >= 2:
        size_effect = size_effect_weibull_fit(
            np.array(actual_thicknesses), np.array(mean_strengths)
        )
        report["size_effect"] = size_effect
        print(f"\n  Size effect fit: m={size_effect['m_weibull']:.2f}, "
              f"σ_ref={size_effect['sigma_ref_Pa']:.0f} Pa, "
              f"R²={size_effect['r_squared']:.4f}")

    # (c) UCS vs literature
    print("\n" + "=" * 60)
    print("(c) UCS vs literature comparison")
    print("=" * 60)

    ucs_comparison = {}
    if qs_assessment.get("converged") and qs_assessment.get("stress_mean_Pa"):
        ucs_dem_pa = qs_assessment["stress_mean_Pa"]
        ucs_dem_mpa = ucs_dem_pa / 1e6

        for key, lit in LITERATURE_UCS_RANGES.items():
            lit_range = lit["ucsi_MPa"]
            within = lit_range[0] <= ucs_dem_mpa <= lit_range[1]
            ucs_comparison[key] = {
                "dem_ucsi_MPa": ucs_dem_mpa,
                "literature_range_MPa": list(lit_range),
                "source": lit["source"],
                "within_range": within,
            }
            status = "✓ WITHIN" if within else "✗ OUTSIDE"
            print(f"  {key}: DEM={ucs_dem_mpa:.2f} MPa, "
                  f"lit=[{lit_range[0]}, {lit_range[1]}] MPa → {status}")

    report["ucs_comparison"] = ucs_comparison

    # (d) Dual friction summary
    print("\n" + "=" * 60)
    print("(d) Dual-coefficient friction model")
    print("=" * 60)
    friction_config = DualFrictionConfig(
        mu_static=0.6, mu_dynamic=0.4, v_characteristic=1e-3
    )
    report["dual_friction"] = friction_config.to_dict()
    print(f"  μ_s = {friction_config.mu_static}")
    print(f"  μ_d = {friction_config.mu_dynamic}")
    print(f"  v_c = {friction_config.v_characteristic} m/s")

    # Save report
    report_path = output_dir / "bpm_calibration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="BPM calibration runner")
    parser.add_argument("--output", type=Path, default=Path("results/bpm_calibration"))
    parser.add_argument("--beta-values", type=float, nargs="+",
                       default=[100.0, 400.0, 1600.0, 3600.0])
    parser.add_argument("--thicknesses", type=float, nargs="+",
                       default=[0.090, 0.180, 0.360])
    parser.add_argument("--n-per-thickness", type=int, default=8)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic data instead of DEM runs")
    args = parser.parse_args()

    run_full_calibration(
        args.output,
        beta_values=args.beta_values,
        thicknesses=args.thicknesses,
        n_per_thickness=args.n_per_thickness,
        n_bootstrap=args.n_bootstrap,
        use_synthetic=args.synthetic,
    )


if __name__ == "__main__":
    main()
