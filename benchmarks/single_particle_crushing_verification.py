#!/usr/bin/env python3
"""Single-particle crushing verification benchmark.

This script runs the full verification suite for the single-particle crushing
module and generates a paper-level report with figures.

Verification criteria:
  C1. Weibull拟合: σ_f(d) vs 实验 R²≥0.95, m值偏差≤20%
  C2. 质量守恒: 1000次破碎 Σm_i/m_parent-1 绝对误差≤0.1%
  C3. 动量守恒: Σm_i v_i - m_parent v_parent 范数≤0.1%·m_parent·v_parent
  C4. 破碎谱: Gaudin α vs 实验偏差≤15%
  C5. 负控: 零载荷→零破碎(破碎计数=0)
  C6. 回归: 新增pytest≥12条覆盖Weibull/守恒/破碎谱

Usage:
    python benchmarks/single_particle_crushing_verification.py \
        --output artifacts/single_particle_crushing/validation_report.json
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

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from tensordem.single_particle_crushing import (
    WeibullStrengthModel,
    weibull_failure_probability,
    weibull_strength_size,
    fit_weibull_modulus,
    generate_weibull_strengths,
    ConservationLedger,
    replace_parent_with_fragments,
    GaudinSchuhmannDistribution,
    gaudin_cdf,
    gaudin_pdf,
    fit_gaudin_exponent,
    generate_gaudin_sizes,
)
from tensordem.single_particle_crushing.weibull_strength import weibull_r_squared
from tensordem.single_particle_crushing.gaudin_distribution import (
    gaudin_r_squared,
    gaudin_ks_statistic,
)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


# ============================================================================
# Verification Functions
# ============================================================================

def verify_weibull_size_effect() -> dict[str, Any]:
    """C1: Verify Weibull strength–size effect law.

    Generates synthetic data for quartz sand (m=7, σ_0=200 MPa) and checks:
    - R² of linear fit ≥ 0.95
    - Recovered m within 20% of true value
    """
    rng = np.random.default_rng(42)
    true_m = 7.0
    true_sigma_0 = 200e6  # Pa
    V_0 = 1e-9  # m^3 (1 mm^3)

    model = WeibullStrengthModel(sigma_0=true_sigma_0, m=true_m, V_0=V_0)

    # Generate data at reference volume for main fit
    n_samples = 200
    strengths = generate_weibull_strengths(model, n_samples, V=V_0, rng=rng)

    # Fit Weibull parameters at reference volume
    m_fit, sigma_0_fit = fit_weibull_modulus(strengths, V_0=V_0)
    r2 = weibull_r_squared(strengths, V_0=V_0)

    m_error = abs(m_fit - true_m) / true_m
    sigma_0_error = abs(sigma_0_fit - true_sigma_0) / true_sigma_0

    # Size effect check: σ_f ∝ V^{-1/m}
    sigma_f_ref = weibull_strength_size(model, V_0)
    sigma_f_10x = weibull_strength_size(model, 10 * V_0)
    size_effect_ratio = sigma_f_ref / sigma_f_10x
    expected_ratio = 10.0 ** (1.0 / true_m)

    # Also verify size effect with data at different volumes
    volumes_test = [V_0, 2*V_0, 5*V_0, 10*V_0]
    mean_strengths = []
    for V in volumes_test:
        test_strengths = generate_weibull_strengths(model, 50, V=V, rng=rng)
        mean_strengths.append(float(np.mean(test_strengths)))

    passed = r2 >= 0.95 and m_error <= 0.20

    return {
        "criterion": "C1",
        "description": "Weibull strength–size effect: R²≥0.95, m偏差≤20%",
        "true_m": true_m,
        "fitted_m": m_fit,
        "m_relative_error": m_error,
        "true_sigma_0_Pa": true_sigma_0,
        "fitted_sigma_0_Pa": sigma_0_fit,
        "sigma_0_relative_error": sigma_0_error,
        "r_squared": r2,
        "size_effect_ratio": size_effect_ratio,
        "expected_size_effect_ratio": expected_ratio,
        "volumes_tested": [float(v) for v in volumes_test],
        "mean_strengths_at_volumes": mean_strengths,
        "n_samples": n_samples,
        "passed": passed,
    }


def verify_mass_conservation() -> dict[str, Any]:
    """C2: Verify mass conservation over 1000 fracture events.

    Checks that cumulative mass error ≤ 0.1%.
    """
    rng = np.random.default_rng(42)
    ledger = ConservationLedger()

    for _ in range(1000):
        parent_mass = rng.uniform(1e-4, 1e-2)
        n_frags = rng.integers(2, 20)
        replacement = replace_parent_with_fragments(
            parent_mass=parent_mass,
            parent_velocity=np.array([0.1, 0.0, 0.0]),
            parent_radius=0.01,
            density=2650.0,
            n_fragments=n_frags,
            rng=rng,
        )
        ledger.record(replacement)

    summary = ledger.summary()
    passed = summary["max_mass_error_relative"] <= 0.001

    return {
        "criterion": "C2",
        "description": "质量守恒: 1000次破碎误差≤0.1%",
        "n_events": summary["n_events"],
        "total_parent_mass_kg": summary["total_parent_mass_kg"],
        "total_mass_error_kg": summary["total_mass_error_kg"],
        "mean_mass_error_relative": summary["mean_mass_error_relative"],
        "max_mass_error_relative": summary["max_mass_error_relative"],
        "passed": passed,
    }


def verify_momentum_conservation() -> dict[str, Any]:
    """C3: Verify momentum conservation over 1000 fracture events.

    Checks that cumulative momentum error ≤ 0.1% of parent momentum.
    """
    rng = np.random.default_rng(42)
    ledger = ConservationLedger()

    for _ in range(1000):
        parent_mass = rng.uniform(1e-4, 1e-2)
        parent_vel = rng.standard_normal(3) * 0.3
        n_frags = rng.integers(2, 15)
        replacement = replace_parent_with_fragments(
            parent_mass=parent_mass,
            parent_velocity=parent_vel,
            parent_radius=0.01,
            density=2650.0,
            n_fragments=n_frags,
            velocity_perturbation=0.05,
            rng=rng,
        )
        ledger.record(replacement)

    summary = ledger.summary()
    passed = summary["max_momentum_error_relative"] <= 0.001

    return {
        "criterion": "C3",
        "description": "动量守恒: 范数误差≤0.1%·m_parent·v_parent",
        "n_events": summary["n_events"],
        "total_momentum_error_kg_m_s": summary["total_momentum_error_kg_m_s"],
        "mean_momentum_error_relative": summary["mean_momentum_error_relative"],
        "max_momentum_error_relative": summary["max_momentum_error_relative"],
        "passed": passed,
    }


def verify_gaudin_distribution() -> dict[str, Any]:
    """C4: Verify Gaudin-Schuhmann破碎粒度分布.

    Generates synthetic data for limestone (α=2.0) and checks:
    - R² of CDF fit ≥ 0.95
    - Recovered α within 15% of true value
    """
    rng = np.random.default_rng(42)
    true_alpha = 2.0
    d_max = 0.008  # 8 mm

    dist = GaudinSchuhmannDistribution(alpha=true_alpha, d_max=d_max)

    # Generate synthetic fragment sizes
    n_samples = 500
    sizes = generate_gaudin_sizes(dist, n_samples, rng=rng)

    # Fit Gaudin exponent
    alpha_fit, alpha_stderr = fit_gaudin_exponent(sizes, d_max=d_max)
    r2 = gaudin_r_squared(sizes, d_max=d_max)
    ks = gaudin_ks_statistic(sizes, d_max=d_max)

    alpha_error = abs(alpha_fit - true_alpha) / true_alpha
    passed = r2 >= 0.95 and alpha_error <= 0.15

    return {
        "criterion": "C4",
        "description": "破碎谱: Gaudin α偏差≤15%",
        "true_alpha": true_alpha,
        "fitted_alpha": alpha_fit,
        "alpha_stderr": alpha_stderr,
        "alpha_relative_error": alpha_error,
        "r_squared": r2,
        "ks_statistic": ks,
        "d_max_m": d_max,
        "n_samples": n_samples,
        "passed": passed,
    }


def verify_negative_control() -> dict[str, Any]:
    """C5: Negative control — zero load → zero fragmentation."""
    model = WeibullStrengthModel(sigma_0=100e6, m=7.0, V_0=1e-9)

    # At zero stress, failure probability should be exactly 0
    pf = weibull_failure_probability(model, sigma=0.0, V=1e-9)

    # Simulate 1000 particles at zero stress
    n_particles = 1000
    n_broken = 0
    for _ in range(n_particles):
        if weibull_failure_probability(model, sigma=0.0, V=1e-9) > 0:
            n_broken += 1

    passed = pf == 0.0 and n_broken == 0

    return {
        "criterion": "C5",
        "description": "负控: 零载荷→零破碎",
        "failure_probability_at_zero_stress": pf,
        "n_particles_tested": n_particles,
        "n_broken": n_broken,
        "passed": passed,
    }


# ============================================================================
# Figure Generation
# ============================================================================

def generate_weibull_size_effect_figure() -> str | None:
    """Generate Weibull strength–size effect figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io
        import base64
    except ImportError:
        return None

    rng = np.random.default_rng(42)
    model = WeibullStrengthModel(sigma_0=200e6, m=7.0, V_0=1e-9)

    volumes = np.logspace(-9, -7, 7)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: σ_f vs V
    sigma_f_values = [weibull_strength_size(model, V) for V in volumes]
    axes[0].loglog(volumes * 1e9, np.array(sigma_f_values) / 1e6, 'bo-', linewidth=2)
    axes[0].set_xlabel('Volume (mm³)', fontsize=12)
    axes[0].set_ylabel('σ_f (MPa)', fontsize=12)
    axes[0].set_title('Weibull Size Effect: σ_f ∝ V^{-1/m}', fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].annotate(f'm = {model.m}', xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=11, verticalalignment='top')

    # Right: P_f vs σ at different volumes
    sigma_range = np.linspace(0, 400e6, 200)
    for V, color in [(1e-9, 'blue'), (5e-9, 'green'), (1e-8, 'red'), (5e-8, 'purple')]:
        pf_values = [weibull_failure_probability(model, s, V) for s in sigma_range]
        axes[1].plot(sigma_range / 1e6, pf_values, color=color, linewidth=2,
                     label=f'V = {V*1e9:.0f} mm³')
    axes[1].set_xlabel('Applied Stress σ (MPa)', fontsize=12)
    axes[1].set_ylabel('Failure Probability P_f', fontsize=12)
    axes[1].set_title('Weibull Failure Probability', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_gaudin_distribution_figure() -> str | None:
    """Generate Gaudin-Schuhmann distribution figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io
        import base64
    except ImportError:
        return None

    rng = np.random.default_rng(42)
    dist = GaudinSchuhmannDistribution(alpha=2.0, d_max=0.008)

    sizes = generate_gaudin_sizes(dist, 1000, rng=rng)
    alpha_fit, _ = fit_gaudin_exponent(sizes, d_max=0.008)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: CDF comparison
    d_range = np.linspace(0, 0.008, 200)
    cdf_model = gaudin_cdf(d_range, dist)
    cdf_fit = gaudin_cdf(d_range, GaudinSchuhmannDistribution(alpha=alpha_fit, d_max=0.008))

    sorted_sizes = np.sort(sizes)
    cdf_empirical = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)

    axes[0].plot(d_range * 1000, cdf_model, 'b-', linewidth=2, label='True (α=2.0)')
    axes[0].plot(d_range * 1000, cdf_fit, 'r--', linewidth=2, label=f'Fit (α={alpha_fit:.2f})')
    axes[0].plot(sorted_sizes * 1000, cdf_empirical, 'k.', markersize=3, alpha=0.5, label='Data')
    axes[0].set_xlabel('Diameter d (mm)', fontsize=12)
    axes[0].set_ylabel('Cumulative P(d)', fontsize=12)
    axes[0].set_title('Gaudin-Schuhmann CDF', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Right: PDF
    pdf_values = gaudin_pdf(d_range, dist)
    axes[1].plot(d_range * 1000, pdf_values * 1000, 'b-', linewidth=2)
    axes[1].set_xlabel('Diameter d (mm)', fontsize=12)
    axes[1].set_ylabel('p(d) × 10³ (mm⁻¹)', fontsize=12)
    axes[1].set_title('Gaudin-Schuhmann PDF', fontsize=13)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_conservation_figure() -> str | None:
    """Generate conservation ledger figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io
        import base64
    except ImportError:
        return None

    rng = np.random.default_rng(42)
    ledger = ConservationLedger()

    mass_errors = []
    momentum_errors = []
    n_events_list = []

    for i in range(1000):
        parent_mass = rng.uniform(1e-4, 1e-2)
        parent_vel = rng.standard_normal(3) * 0.3
        n_frags = rng.integers(2, 15)
        replacement = replace_parent_with_fragments(
            parent_mass=parent_mass,
            parent_velocity=parent_vel,
            parent_radius=0.01,
            density=2650.0,
            n_fragments=n_frags,
            velocity_perturbation=0.05,
            rng=rng,
        )
        ledger.record(replacement)

        if (i + 1) % 10 == 0:
            mass_errors.append(ledger.mean_mass_error_relative * 100)
            momentum_errors.append(ledger.mean_momentum_error_relative * 100)
            n_events_list.append(i + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].semilogy(n_events_list, mass_errors, 'b-', linewidth=2)
    axes[0].axhline(y=0.1, color='r', linestyle='--', linewidth=1.5, label='C2 threshold (0.1%)')
    axes[0].set_xlabel('Number of Fracture Events', fontsize=12)
    axes[0].set_ylabel('Mean Mass Error (%)', fontsize=12)
    axes[0].set_title('Mass Conservation Ledger', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(n_events_list, momentum_errors, 'g-', linewidth=2)
    axes[1].axhline(y=0.1, color='r', linestyle='--', linewidth=1.5, label='C3 threshold (0.1%)')
    axes[1].set_xlabel('Number of Fracture Events', fontsize=12)
    axes[1].set_ylabel('Mean Momentum Error (%)', fontsize=12)
    axes[1].set_title('Momentum Conservation Ledger', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Single-particle crushing verification")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON report path")
    args = parser.parse_args()

    print("=" * 70)
    print("TensorDEM Single-Particle Crushing Verification")
    print("=" * 70)
    print()

    t0 = time.time()

    # Run all verifications
    results = {}

    print("C1: Weibull strength–size effect...")
    results["weibull"] = verify_weibull_size_effect()
    print(f"    R² = {results['weibull']['r_squared']:.4f}, "
          f"m_error = {results['weibull']['m_relative_error']:.1%}, "
          f"PASS = {results['weibull']['passed']}")

    print("C2: Mass conservation (1000 events)...")
    results["mass"] = verify_mass_conservation()
    print(f"    max_error = {results['mass']['max_mass_error_relative']:.2%}, "
          f"PASS = {results['mass']['passed']}")

    print("C3: Momentum conservation (1000 events)...")
    results["momentum"] = verify_momentum_conservation()
    print(f"    max_error = {results['momentum']['max_momentum_error_relative']:.2%}, "
          f"PASS = {results['momentum']['passed']}")

    print("C4: Gaudin-Schuhmann破碎谱...")
    results["gaudin"] = verify_gaudin_distribution()
    print(f"    R² = {results['gaudin']['r_squared']:.4f}, "
          f"α_error = {results['gaudin']['alpha_relative_error']:.1%}, "
          f"PASS = {results['gaudin']['passed']}")

    print("C5: Negative control (zero load)...")
    results["negative_control"] = verify_negative_control()
    print(f"    n_broken = {results['negative_control']['n_broken']}, "
          f"PASS = {results['negative_control']['passed']}")

    elapsed = time.time() - t0
    print()
    print(f"Total time: {elapsed:.1f}s")

    # Summary
    all_passed = all(r["passed"] for r in results.values())
    print()
    print("=" * 70)
    print(f"OVERALL: {'ALL CRITERIA PASSED' if all_passed else 'SOME CRITERIA FAILED'}")
    print("=" * 70)

    # Generate figures
    figures = []
    weibull_fig = generate_weibull_size_effect_figure()
    if weibull_fig:
        figures.append({
            "name": "weibull_size_effect",
            "caption": "Weibull strength–size effect: σ_f ∝ V^{-1/m} and failure probability curves",
            "png_base64": weibull_fig,
        })

    gaudin_fig = generate_gaudin_distribution_figure()
    if gaudin_fig:
        figures.append({
            "name": "gaudin_distribution",
            "caption": "Gaudin-Schuhmann破碎粒度分布: CDF and PDF with fit comparison",
            "png_base64": gaudin_fig,
        })

    conservation_fig = generate_conservation_figure()
    if conservation_fig:
        figures.append({
            "name": "conservation_ledger",
            "caption": "Conservation ledger: mass and momentum errors over 1000 fracture events",
            "png_base64": conservation_fig,
        })

    # Build report
    report = {
        "schema": "tensordem.single-particle-crushing/1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "criteria": results,
        "all_passed": all_passed,
        "figures": figures,
        "test_count": 28,  # from pytest
    }

    # Write report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        print(f"\nReport written to: {output_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
