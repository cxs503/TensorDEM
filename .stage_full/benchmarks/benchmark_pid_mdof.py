#!/usr/bin/env python
"""Benchmark for pid_mdof — reference vs got numerical comparison.

This benchmark verifies:
1. Den Hartog closed form vs MDOF solve (N=1 equivalence)
2. IN32-D02 surface regression lock
3. Insertion loss convergence with modal truncation
4. Detuning robustness envelope

Reproducible command:
    python benchmarks/benchmark_pid_mdof.py

Expected output: benchmark_pid_mdof.json with ref vs got comparisons.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem import pid_mdof as pm


def benchmark_den_hartog_equivalence():
    """Benchmark: N=1 MDOF solve must equal Den Hartog closed form."""
    mu = 0.03
    nu = 0.95
    zeta_d = 0.02

    s = pm.cantilever_structure(n_modes=1)
    dam = pm.ParticleDamperModel(filling_ratio=0.3, mass_ratio=mu)
    dam.zeta_d_fixed = zeta_d

    class Fixed(type(dam)):
        def zeta_d(self, nu=1.0):
            return self.zeta_d_fixed

    dam_fixed = Fixed(**{k: getattr(dam, k) for k in
                         ("filling_ratio", "mass_ratio", "gamma", "restitution",
                          "detuning_coupling", "n_particles", "primary_mass",
                          "primary_stiffness", "cavity_length", "gravity")})
    dam_fixed.zeta_d_fixed = zeta_d

    r = np.linspace(0.6, 1.4, 4001)
    x_mdof = np.abs(pm.modal_response(s, r * s.omega_1, dam_fixed, nu=nu))
    static = abs(pm.modal_response(s, np.array([1e-6]), None)[0])
    ref = np.abs(pm.two_dof_response(r, mu, nu, zeta_d, s.zeta[0]))

    dev = np.max(np.abs(x_mdof / static - ref)) / np.max(ref)

    return {
        "name": "den_hartog_equivalence",
        "description": "N=1 MDOF solve vs Den Hartog closed form",
        "ref": "Den Hartog closed form (analytic)",
        "got": f"MDOF solve (N=1, max rel dev)",
        "ref_value": "analytic",
        "got_value": float(dev),
        "tolerance": 1e-10,
        "pass": dev < 1e-10,
    }


def benchmark_den_hartog_optimum():
    """Benchmark: Den Hartog optimum must reproduce classical peak."""
    results = []
    for mu in (0.01, 0.03, 0.05, 0.10, 0.20):
        nu_opt, zeta_opt = pm.den_hartog_optimum(mu)
        r = np.linspace(0.5, 1.5, 100001)
        peak = float(np.max(np.abs(pm.two_dof_response(r, mu, nu_opt, zeta_opt, 0.0))))
        expected = math.sqrt(1.0 + 2.0 / mu)
        rel_err = abs(peak - expected) / expected
        results.append({
            "mu": mu,
            "ref_peak": expected,
            "got_peak": peak,
            "rel_err": rel_err,
            "pass": rel_err < 5e-3,
        })
    return {
        "name": "den_hartog_optimum",
        "description": "Den Hartog optimum reproduces classical sqrt(1+2/mu)",
        "results": results,
        "pass": all(r["pass"] for r in results),
    }


def benchmark_in32_regression():
    """Benchmark: IN32-D02 surface regression lock."""
    from tensordem import pid as pid_mod
    cfg = pid_mod.SweepConfig()
    surface = pid_mod.ZetaSurface.from_results(pid_mod.run_sweep(cfg), cfg)
    gi = surface.gamma_axis.index(2.0)
    fi = surface.freq_ratio_axis.index(1.0)
    pi = surface.filling_ratio_axis.index(0.3)
    ei = surface.restitution_axis.index(0.6)
    expected = surface.zeta_values[gi][fi][pi][ei]
    got = pm.zeta_eff_delivered(0.3, 1.0, gamma=2.0, restitution=0.6,
                                mass_ratio=pm.MU_REF)
    rel_err = abs(got - expected) / expected if expected != 0 else 0.0
    return {
        "name": "in32_d02_regression",
        "description": "IN32-D02 surface regression lock at reference mass ratio",
        "ref": float(expected),
        "got": float(got),
        "rel_err": float(rel_err),
        "tolerance": 1e-12,
        "pass": rel_err < 1e-12,
    }


def benchmark_modal_truncation():
    """Benchmark: IL convergence with modal truncation."""
    results = []
    for n in (6, 8, 12, 16):
        s = pm.cantilever_structure(n_modes=n)
        dam = pm.ParticleDamperModel(filling_ratio=0.3, mass_ratio=0.05)
        il = pm.peak_insertion_loss(s, dam, nu=1.0)["il_peak_db"]
        results.append({"n_modes": n, "il_peak_db": il})

    # Check convergence: N=12 vs N=16
    il_12 = next(r["il_peak_db"] for r in results if r["n_modes"] == 12)
    il_16 = next(r["il_peak_db"] for r in results if r["n_modes"] == 16)
    rel_err = abs(il_12 - il_16) / abs(il_16) if il_16 != 0 else 0.0

    return {
        "name": "modal_truncation_convergence",
        "description": "IL convergence with modal truncation",
        "results": results,
        "ref": f"N=16: {il_16:.4f} dB",
        "got": f"N=12: {il_12:.4f} dB",
        "rel_err": float(rel_err),
        "tolerance": 0.01,
        "pass": rel_err < 0.01,
    }


def benchmark_negative_control():
    """Benchmark: Zero filling gives zero IL."""
    s = pm.cantilever_structure(n_modes=6)
    prov = pm.ZetaEffProvider(filling_ratio=0.0, mass_ratio=0.03)
    il_a = pm.injection_peak_insertion_loss(s, prov, nu=1.0)
    dam = pm.ParticleDamperModel(filling_ratio=0.0, mass_ratio=0.03)
    il_b = pm.peak_insertion_loss(s, dam, nu=1.0)
    max_il = max(abs(il_a["il_peak_db"]), abs(il_b["il_peak_db"]))
    return {
        "name": "negative_control_zero_filling",
        "description": "Zero filling gives zero IL",
        "ref": 0.0,
        "got_model_A": float(il_a["il_peak_db"]),
        "got_model_B": float(il_b["il_peak_db"]),
        "max_abs_il": float(max_il),
        "tolerance": 0.5,
        "pass": max_il <= 0.5,
    }


def benchmark_detuning_robustness():
    """Benchmark: ±20% detuning IL > 50% of peak."""
    mu = 0.03
    s = pm.cantilever_structure(n_modes=8)
    prov = pm.ZetaEffProvider(filling_ratio=0.3, mass_ratio=mu)

    nu_scan = np.linspace(0.6, 1.3, 300)
    il_a_scan = [pm.injection_peak_insertion_loss(s, prov, nu=nu)["il_peak_db"]
                 for nu in nu_scan]
    peak_a = max(il_a_scan)
    nu_a_opt = float(nu_scan[np.argmax(il_a_scan)])

    results = []
    for pct in (-20, -10, -5, 0, 5, 10, 20):
        nu_det = nu_a_opt * (1.0 + pct / 100.0)
        il = pm.injection_peak_insertion_loss(s, prov, nu=nu_det)
        ratio = il["il_peak_db"] / peak_a if peak_a > 0 else 0
        results.append({
            "pct": pct, "nu": nu_det,
            "il_db": il["il_peak_db"],
            "ratio": ratio,
            "pass": il["il_peak_db"] > 0.5 * peak_a,
        })

    return {
        "name": "detuning_robustness",
        "description": "±20% detuning IL > 50% of peak",
        "peak_il_db": float(peak_a),
        "nu_opt": nu_a_opt,
        "results": results,
        "pass": all(r["pass"] for r in results),
    }


def main():
    benchmarks = [
        benchmark_den_hartog_equivalence(),
        benchmark_den_hartog_optimum(),
        benchmark_in32_regression(),
        benchmark_modal_truncation(),
        benchmark_negative_control(),
        benchmark_detuning_robustness(),
    ]

    all_pass = all(b.get("pass", False) for b in benchmarks)

    output = {
        "benchmark": "pid_mdof",
        "mission_id": 14587,
        "campaign_node": "p307w1-n010",
        "repro_cmd": "python benchmarks/benchmark_pid_mdof.py",
        "all_pass": all_pass,
        "benchmarks": benchmarks,
    }

    outpath = Path(__file__).parent / "benchmark_pid_mdof.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Benchmark results written to {outpath}")
    print(f"\n{'Benchmark':<35} {'Status':>8}")
    print("-" * 45)
    for b in benchmarks:
        status = "PASS" if b.get("pass", False) else "FAIL"
        print(f"{b['name']:<35} {status:>8}")
    print(f"\nAll pass: {all_pass}")


if __name__ == "__main__":
    main()
