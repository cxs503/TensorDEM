#!/usr/bin/env python
"""Integration caller examples for pid_mdof — demonstrates how the module
is consumed from a structural analysis workflow.

These examples show:
1. How to compute insertion loss for a user-defined structure
2. How to integrate with a structural FEM solver (mock)
3. How to use the module in a design optimisation loop

Usage:
    python examples/pid_mdof_integration.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem import pid_mdof as pm


def example_1_cantilever_design():
    """Example 1: Design a PID for a cantilever beam.

    Given a cantilever beam with known properties, find the optimal
    tuning ratio and the expected insertion loss.
    """
    print("=" * 60)
    print("Example 1: PID design for a cantilever beam")
    print("=" * 60)

    # Define the structure
    s = pm.cantilever_structure(
        length=2.0, width=0.1, thickness=0.005,
        E=210e9, rho=7850.0, n_modes=8,
        x_damper=1.8, x_obs=2.0, x_force=2.0
    )

    print(f"Structure: {s.name}")
    print(f"  First natural frequency: {s.modes[0].frequency_hz:.2f} Hz")
    print(f"  Effective mass at damper: {s.effective_mass_at_damper(0):.4f} kg")

    # Design the PID
    mu = 0.03  # 3% mass ratio
    prov = pm.ZetaEffProvider(filling_ratio=0.3, mass_ratio=mu)

    # Optimise tuning
    opt = pm.optimise_injection_tuning(s, prov)
    print(f"\nOptimal tuning: ν = {opt['nu']:.4f}")
    print(f"Peak insertion loss: {opt['il_peak_db']:.2f} dB")
    print(f"Added damping (target mode): {opt['zeta_added_target_mode']:.6f}")

    # Compute IL curve
    r = np.linspace(0.3, 2.0, 500)
    w = r * s.omega_1
    dam = pm.ParticleDamperModel(filling_ratio=0.3, mass_ratio=mu)
    il = pm.insertion_loss(s, w, dam, nu=opt['nu'])

    print(f"\nIL at resonance (ω/ω₁=1): {il[np.argmin(np.abs(r - 1.0))]:.2f} dB")
    print(f"Max IL: {np.max(il):.2f} dB at ω/ω₁ = {r[np.argmax(il)]:.3f}")
    return opt


def example_2_plate_with_stiffeners():
    """Example 2: PID on a stiffened plate panel.

    Shows how to use the stiffened plate structure model.
    """
    print("\n" + "=" * 60)
    print("Example 2: PID on a stiffened plate panel")
    print("=" * 60)

    plate = pm.stiffened_plate_structure(
        a=1.0, b=0.6, thickness=0.006,
        E=210e9, rho=7850.0, nu_poisson=0.3,
        n_stiffeners=4, stiffener_inertia=3.0e-9,
        n_modes=8
    )

    print(f"Structure: {plate.name}")
    print(f"  First natural frequency: {plate.modes[0].frequency_hz:.2f} Hz")
    print(f"  Bending stiffness D_x: {plate.meta['D_x_Nm']:.2f} N·m")

    mu = 0.02
    prov = pm.ZetaEffProvider(filling_ratio=0.25, mass_ratio=mu)
    opt = pm.optimise_injection_tuning(plate, prov)

    print(f"\nOptimal tuning: ν = {opt['nu']:.4f}")
    print(f"Peak insertion loss: {opt['il_peak_db']:.2f} dB")
    return opt


def example_3_comparison_with_floating_raft():
    """Example 3: Compare PID with floating raft isolation.

    Shows the per-unit-mass comparison table.
    """
    print("\n" + "=" * 60)
    print("Example 3: PID vs floating raft comparison")
    print("=" * 60)

    s = pm.cantilever_structure(n_modes=8)
    mu = 0.03

    # PID on cantilever
    prov = pm.ZetaEffProvider(filling_ratio=0.3, mass_ratio=mu)
    il_pid = pm.optimise_injection_tuning(s, prov)
    mass_pid = mu * s.effective_mass_at_damper(0)

    # Floating raft
    mu_raft = 0.03
    r_iso = 0.3
    il_raft = pm.floating_raft_il_db(1.0, mu_raft=mu_raft, r_iso=r_iso)

    print(f"{'Scheme':<25} {'IL (dB)':>10} {'Mass (kg)':>12} {'IL/kg':>10}")
    print("-" * 60)
    print(f"{'PID on cantilever':<25} {il_pid['il_peak_db']:10.2f} {mass_pid:12.4f} {il_pid['il_peak_db']/mass_pid:10.2f}")
    print(f"{'Floating raft':<25} {il_raft:10.2f} {mu_raft:12.4f} {il_raft/mu_raft:10.2f}")


def example_4_detuning_study():
    """Example 4: Detuning robustness study.

    Shows how the PID performance degrades with frequency mistuning.
    """
    print("\n" + "=" * 60)
    print("Example 4: Detuning robustness study")
    print("=" * 60)

    s = pm.cantilever_structure(n_modes=8)
    mu = 0.03
    prov = pm.ZetaEffProvider(filling_ratio=0.3, mass_ratio=mu)

    # Find optimum
    opt = pm.optimise_injection_tuning(s, prov)
    nu_opt = opt['nu']
    il_peak = opt['il_peak_db']

    print(f"Optimum: ν = {nu_opt:.4f}, IL = {il_peak:.2f} dB\n")
    print(f"{'Detuning':>10} {'ν':>10} {'IL (dB)':>10} {'% of peak':>12}")
    print("-" * 45)

    for pct in (-20, -10, -5, 0, 5, 10, 20):
        nu_det = nu_opt * (1.0 + pct / 100.0)
        res = pm.injection_peak_insertion_loss(s, prov, nu=nu_det)
        pct_of_peak = res['il_peak_db'] / il_peak * 100
        print(f"{pct:+8d}% {nu_det:10.4f} {res['il_peak_db']:10.2f} {pct_of_peak:11.1f}%")


if __name__ == "__main__":
    example_1_cantilever_design()
    example_2_plate_with_stiffeners()
    example_3_comparison_with_floating_raft()
    example_4_detuning_study()
    print("\n" + "=" * 60)
    print("All integration examples completed successfully.")
    print("=" * 60)
