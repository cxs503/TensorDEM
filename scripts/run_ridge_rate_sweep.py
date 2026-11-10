#!/usr/bin/env python3
"""Multi-rate sweep for ice-ridge keel grounding: inertial number collapse.

Runs the analysis at multiple platen velocities using the continuum μ(I)
rheology approach (fast), computes the inertial number for each, collapses
the bearing capacity data onto a master curve, fits the μ(I) rheology,
extrapolates to I→0 (quasi-static limit), and compares with the continuum
(m7441) baseline.

Outputs:
    artifacts/ridge_rate_sweep/results.json
    artifacts/ridge_rate_sweep/figures/
"""
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from tensordem.ridge_dem import RidgeKeel, RidgeMaterial
from tensordem.ridge_geotech import MohrCoulombSoil, PipeSection
from tensordem.ridge_rate_dependence import (
    RateSweepResult,
    build_rate_dependence_report,
    report_to_json,
    run_rate_sweep_continuum,
)

ARTIFACTS = os.path.join(HERE, "..", "artifacts", "ridge_rate_sweep")
FIG_DIR = os.path.join(ARTIFACTS, "figures")


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)

    # Standard ridge grounding parameters (matching the m7441 baseline)
    keel = RidgeKeel(keel_depth=12.0, keel_width=24.0, porosity=0.10)
    material = RidgeMaterial(
        particle_radius=0.50,
        drag=2.0e4,
        freeze_bond_fraction=1.0,
        bond_tensile_strength=100.0e3,
        bond_shear_strength=200.0e3,
    )
    soil = MohrCoulombSoil(
        friction_angle_deg=0.0,
        cohesion=30.0e3,
        saturated_unit_weight=18.0e3,
        young_modulus=10.0e6,
        poisson_ratio=0.3,
        water_unit_weight=10.06e3,
        undrained=True,
    )
    section = PipeSection(
        outer_diameter=0.914,
        wall_thickness=0.0159,
        young_modulus=207.0e9,
        yield_strength=448.0e6,
    )

    # Multi-rate sweep: 7 velocities spanning ~2 decades
    velocities = [0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60]
    cover_depth = 0.50

    # μ(I) rheology parameters (moderate rate dependence for engineering relevance)
    mu_s = 0.35  # static friction coefficient
    mu_2 = 0.42  # dynamic friction coefficient (20% increase)
    I0 = 0.05    # characteristic inertial number

    print(f"Running rate sweep (continuum μ(I) approach) with {len(velocities)} velocities")
    print(f"Velocities: {velocities}")
    print(f"Cover depth: {cover_depth} m")
    print(f"Particle radius: {material.particle_radius} m")
    print(f"μ(I) parameters: μ_s={mu_s}, μ_2={mu_2}, I₀={I0}")

    sweep = run_rate_sweep_continuum(
        keel=keel,
        material=material,
        soil=soil,
        section=section,
        velocities=velocities,
        cover_depth=cover_depth,
        mu_s=mu_s,
        mu_2=mu_2,
        I0=I0,
    )

    print(f"\nResults:")
    print(f"  Confining pressure: {sweep.confining_pressure:.0f} Pa")
    print(f"  Bulk density: {sweep.bulk_density:.1f} kg/m³")
    print(f"  Quasi-static capacity (continuum): {sweep.quasi_static_capacity/1e3:.1f} kN/m")
    print(f"\n  {'v [m/s]':>10} {'I':>10} {'Q [kN/m]':>12} {'Q/Q₀':>8}")
    for v, I, q in zip(sweep.velocities, sweep.inertial_numbers, sweep.peak_loads):
        print(f"  {v:10.3f} {I:10.4f} {q/1e3:12.1f} {q/sweep.quasi_static_capacity:8.3f}")

    # Build report
    report = build_rate_dependence_report(sweep)
    results = report_to_json(report)

    # Save results
    results_path = os.path.join(ARTIFACTS, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Print acceptance
    print(f"\nAcceptance criteria:")
    acc = report.acceptance
    print(f"  (a) I-collapse R² = {acc['collapse_R2']:.4f} (≥0.95): {'PASS' if acc['collapse_passed'] else 'FAIL'}")
    print(f"  (b) Extrap rel diff = {acc['extrap_rel_diff']*100:.1f}% (<10%): {'PASS' if acc['extrap_passed'] else 'FAIL'}")
    print(f"  (c) Max deviation = {acc['max_deviation']*100:.1f}% (<30%): {'PASS' if acc['deviation_passed'] else 'FAIL'}")
    print(f"  (d) Spearman ρ = {acc['spearman_rho']:.3f} (≥0.95): {'PASS' if acc['monotonicity_passed'] else 'FAIL'}")
    print(f"  (e) Neg control rel diff = {acc['neg_control_rel_diff']*100:.1f}% (<5%): {'PASS' if acc['neg_control_passed'] else 'FAIL'}")

    # Generate figures
    _plot_master_curve(report)
    _plot_capacity_vs_velocity(report)
    _plot_rate_deviation(report)
    _plot_mu_I_fit(report)

    print(f"\nFigures saved to {FIG_DIR}/")


def _plot_master_curve(report) -> None:
    """Plot the I-number master curve (collapse)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    I_sorted = report.collapse["I_sorted"]
    Q_sorted = report.collapse["Q_sorted"]
    ax.scatter(I_sorted, Q_sorted, c="blue", s=60, zorder=5, label="Data")

    # Fit curve
    if report.fit.get("fitted"):
        alpha = report.fit["alpha"]
        I0 = report.fit["I0"]
        I_fit = np.logspace(np.log10(min(I_sorted) * 0.5), np.log10(max(I_sorted) * 2), 100)
        Q_fit = [1.0 + alpha * i / (I0 + i) for i in I_fit]
        ax.plot(I_fit, Q_fit, "r-", lw=2, label=f"μ(I) fit: α={alpha:.2f}, I₀={I0:.4f}")

    ax.set_xscale("log")
    ax.set_xlabel("Inertial number $I$", fontsize=12)
    ax.set_ylabel("Normalized capacity $Q / Q_0$", fontsize=12)
    ax.set_title("I-number master curve collapse", fontsize=14)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="Quasi-static limit")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    R2 = report.collapse["R_squared"]
    ax.text(0.05, 0.95, f"$R^2$ = {R2:.4f}", transform=ax.transAxes, fontsize=11,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "master_curve_collapse.png"), dpi=150)
    plt.close(fig)


def _plot_capacity_vs_velocity(report) -> None:
    """Plot bearing capacity vs loading velocity."""
    fig, ax = plt.subplots(figsize=(8, 5))
    v = report.sweep.velocities
    Q = [q / 1e3 for q in report.sweep.peak_loads]
    Q_static = report.sweep.quasi_static_capacity / 1e3

    ax.semilogx(v, Q, "bo-", ms=8, lw=2, label="Peak capacity")
    ax.axhline(Q_static, color="red", ls="--", lw=2, label=f"Quasi-static (m7441): {Q_static:.1f} kN/m")
    ax.set_xlabel("Platen velocity $v$ [m/s]", fontsize=12)
    ax.set_ylabel("Peak bearing capacity $Q$ [kN/m]", fontsize=12)
    ax.set_title("Rate-dependent bearing capacity", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "capacity_vs_velocity.png"), dpi=150)
    plt.close(fig)


def _plot_rate_deviation(report) -> None:
    """Plot rate-induced deviation vs inertial number."""
    fig, ax = plt.subplots(figsize=(8, 5))
    I_vals = report.deviation["inertial_numbers"]
    devs = report.deviation["deviations"]

    ax.semilogx(I_vals, [d * 100 for d in devs], "gs-", ms=8, lw=2, label="Rate deviation")
    ax.axhline(0, color="gray", ls="-", alpha=0.5)
    ax.axhline(30, color="red", ls="--", alpha=0.5, label="30% bound")
    ax.axhline(-30, color="red", ls="--", alpha=0.5)
    ax.set_xlabel("Inertial number $I$", fontsize=12)
    ax.set_ylabel("Rate deviation $(Q - Q_0) / Q_0$ [%]", fontsize=12)
    ax.set_title("Rate-induced bearing capacity deviation", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "rate_deviation.png"), dpi=150)
    plt.close(fig)


def _plot_mu_I_fit(report) -> None:
    """Plot the μ(I) rheology fit with data and extrapolation."""
    fig, ax = plt.subplots(figsize=(8, 5))
    I_sorted = report.collapse["I_sorted"]
    Q_sorted = report.collapse["Q_sorted"]

    ax.scatter(I_sorted, Q_sorted, c="blue", s=60, zorder=5, label="Data")

    if report.fit.get("fitted"):
        alpha = report.fit["alpha"]
        I0 = report.fit["I0"]
        # Extend to I→0 for extrapolation
        I_ext = np.logspace(-4, np.log10(max(I_sorted) * 2), 200)
        Q_ext = [1.0 + alpha * i / (I0 + i) for i in I_ext]
        ax.plot(I_ext, Q_ext, "r-", lw=2, label=f"μ(I) fit")

        # Mark the I→0 extrapolation point
        ax.scatter([I_ext[0]], [1.0], c="red", s=100, marker="*", zorder=6,
                   label=f"I→0 extrapolation: Q/Q₀ = 1.000")

    ax.set_xscale("log")
    ax.set_xlabel("Inertial number $I$", fontsize=12)
    ax.set_ylabel("Normalized capacity $Q / Q_0$", fontsize=12)
    ax.set_title("μ(I) rheology fit and quasi-static extrapolation", fontsize=14)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotation
    extrap = report.extrapolation
    text = (f"Q₀(DEM) = {extrap.get('Q_static_dem_N_per_m', float('nan')):.0f} N/m\n"
            f"Q₀(cont) = {extrap.get('Q_static_continuum_N_per_m', float('nan')):.0f} N/m\n"
            f"Rel. diff = {extrap.get('relative_difference', float('inf'))*100:.1f}%")
    ax.text(0.95, 0.05, text, transform=ax.transAxes, fontsize=9,
            horizontalalignment="right", verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mu_I_fit_extrapolation.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
