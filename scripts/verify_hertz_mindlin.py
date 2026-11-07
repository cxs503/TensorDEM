#!/usr/bin/env python3
"""Generate verification figures and report for Hertz-Mindlin contact model.

Produces:
- artifacts/hertz_mindlin_stiffness_ratio.png: k_t/k_n vs Poisson's ratio
- artifacts/hertz_mindlin_force_displacement.png: Force-displacement curves
- artifacts/hertz_mindlin_report.json: Machine-readable verification results
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensordem.dem3d import hertz_reduced_modulus, hertz_effective_radius, hertz_secant_stiffness
from tensordem import hertz_mindlin

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def plot_stiffness_ratio():
    """Plot k_t/k_n vs Poisson's ratio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nu_values = [i / 100.0 for i in range(0, 49)]  # 0.00 to 0.48
    ratios = [hertz_mindlin.mindlin_stiffness_ratio(nu) for nu in nu_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nu_values, ratios, "b-", linewidth=2, label=r"$k_t/k_n = 3(1-\nu)/(2-\nu)$")
    ax.axhline(y=1.5, color="gray", linestyle="--", alpha=0.5, label="ν=0: 1.5")
    ax.axhline(y=1.235, color="red", linestyle="--", alpha=0.5, label="ν=0.3: 1.235")
    ax.set_xlabel("Poisson's ratio ν", fontsize=12)
    ax.set_ylabel("Stiffness ratio $k_t/k_n$", fontsize=12)
    ax.set_title("Hertz-Mindlin Tangential-to-Normal Stiffness Ratio", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(1.0, 2.0)

    path = ARTIFACTS_DIR / "hertz_mindlin_stiffness_ratio.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def plot_force_displacement():
    """Plot tangential force vs displacement for different overlaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    E = 1e9
    nu = 0.3
    R = 0.01
    e_star = hertz_reduced_modulus(E, nu)
    g_star = hertz_mindlin.hertz_reduced_shear_modulus(E, nu)
    r_eff = hertz_effective_radius(R)
    mu = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))

    for delta in [1e-5, 5e-5, 1e-4]:
        k_t = float(hertz_mindlin.hertz_mindlin_tangential_stiffness(delta, e_star, g_star, r_eff))
        F_n = (4.0 / 3.0) * e_star * math.sqrt(r_eff) * delta ** 1.5
        F_limit = mu * F_n

        delta_t = torch.linspace(0, 2 * F_limit / k_t, 100)
        F_t_raw = k_t * delta_t
        F_t = torch.minimum(F_t_raw, torch.full_like(F_t_raw, F_limit))

        ax.plot(delta_t.numpy() * 1e6, F_t.numpy(), label=f"δ={delta*1e3:.1f} mm, k_t={k_t:.0f} N/m")

    ax.set_xlabel("Tangential displacement δ_t (μm)", fontsize=12)
    ax.set_ylabel("Tangential force F_t (N)", fontsize=12)
    ax.set_title("Hertz-Mindlin Tangential Force-Displacement (μ=0.3)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    path = ARTIFACTS_DIR / "hertz_mindlin_force_displacement.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def compute_verification_results():
    """Compute verification results for the report."""
    results = {
        "mindlin_stiffness_ratio": {},
        "reduced_shear_modulus": {},
        "tangential_force": {},
    }

    # Stiffness ratio tests
    for nu in [0.0, 0.25, 0.3]:
        expected = 3.0 * (1.0 - nu) / (2.0 - nu)
        actual = hertz_mindlin.mindlin_stiffness_ratio(nu)
        results["mindlin_stiffness_ratio"][f"nu_{nu}"] = {
            "expected": expected,
            "actual": actual,
            "error": abs(actual - expected),
            "pass": abs(actual - expected) < 1e-10,
        }

    # Reduced shear modulus tests
    E = 1e9
    nu = 0.3
    expected_g = E / (4.0 * (2.0 - nu) * (1.0 + nu))
    actual_g = hertz_mindlin.hertz_reduced_shear_modulus(E, nu)
    results["reduced_shear_modulus"]["equal_spheres"] = {
        "expected": expected_g,
        "actual": actual_g,
        "error": abs(actual_g - expected_g),
        "pass": abs(actual_g - expected_g) < 1e-3,
    }

    expected_g_rigid = E / (2.0 * (2.0 - nu) * (1.0 + nu))
    actual_g_rigid = hertz_mindlin.hertz_reduced_shear_modulus(E, nu, rigid_counterpart=True)
    results["reduced_shear_modulus"]["rigid_counterpart"] = {
        "expected": expected_g_rigid,
        "actual": actual_g_rigid,
        "error": abs(actual_g_rigid - expected_g_rigid),
        "pass": abs(actual_g_rigid - expected_g_rigid) < 1e-3,
    }

    # Tangential force tests
    delta = 1e-4
    e_star = hertz_reduced_modulus(E, nu)
    g_star = hertz_mindlin.hertz_reduced_shear_modulus(E, nu)
    r_eff = hertz_effective_radius(0.01)
    k_t = float(hertz_mindlin.hertz_mindlin_tangential_stiffness(delta, e_star, g_star, r_eff))

    # Stick test
    delta_t = torch.tensor([[0.0, 1e-6, 0.0]], dtype=torch.float64)
    F_n = torch.tensor([100.0], dtype=torch.float64)
    F_t, _ = hertz_mindlin.hertz_mindlin_tangential_force(
        delta, delta_t, e_star, g_star, r_eff, 0.5, F_n)
    expected_Ft = k_t * 1e-6
    results["tangential_force"]["stick"] = {
        "expected": expected_Ft,
        "actual": float(F_t[0, 1].item()),
        "error": abs(float(F_t[0, 1].item()) - expected_Ft),
        "pass": abs(float(F_t[0, 1].item()) - expected_Ft) < 1e-5,
    }

    # Slip test
    delta_t = torch.tensor([[0.0, 0.1, 0.0]], dtype=torch.float64)
    F_t, _ = hertz_mindlin.hertz_mindlin_tangential_force(
        delta, delta_t, e_star, g_star, r_eff, 0.3, F_n)
    expected_limit = 0.3 * 100.0
    results["tangential_force"]["slip"] = {
        "expected": expected_limit,
        "actual": float(F_t.norm(dim=1).item()),
        "error": abs(float(F_t.norm(dim=1).item()) - expected_limit),
        "pass": abs(float(F_t.norm(dim=1).item()) - expected_limit) < 1e-5,
    }

    return results


def main():
    print("Generating Hertz-Mindlin verification figures and report...")

    # Generate figures
    fig1 = plot_stiffness_ratio()
    fig2 = plot_force_displacement()

    # Compute verification results
    results = compute_verification_results()

    # Check all tests pass
    all_pass = True
    for category, tests in results.items():
        for name, test in tests.items():
            if not test.get("pass", False):
                all_pass = False
                print(f"FAIL: {category}/{name}: error={test.get('error', 'N/A')}")

    if all_pass:
        print("All verification tests PASSED")
    else:
        print("Some verification tests FAILED")

    # Save report JSON
    report = {
        "title": "Hertz-Mindlin Contact Model Verification",
        "mission_id": 4948,
        "scenario_key": "hertz_mindlin_contact",
        "tags": ["dem", "hertz", "mindlin", "friction", "contact"],
        "verification_results": results,
        "all_tests_pass": all_pass,
        "figures": [
            {"name": "stiffness_ratio", "path": str(fig1)},
            {"name": "force_displacement", "path": str(fig2)},
        ],
    }

    report_path = ARTIFACTS_DIR / "hertz_mindlin_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {report_path}")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
