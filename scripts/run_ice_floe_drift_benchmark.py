#!/usr/bin/env python3
"""Run ice floe drift benchmark and generate results.

This script runs the full DEM-LBM coupled ice floe drift benchmark:
1. Baseline case (D=1m, U=1 m/s)
2. Parameter sweep (9 cases: 3 U_current × 3 D)
3. Negative control (no fluid)
4. Force balance verification
5. Generate plots and JSON results

Usage:
    python scripts/run_ice_floe_drift_benchmark.py
"""
import sys
import json
import math
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tensordem.ice_floe_drift import (
    analytical_drift_velocity,
    IceFloeProperties,
    DEMLBMConfig,
    IceFloeDriftSimulator,
    run_parameter_sweep,
    negative_control_no_fluid,
    verify_force_balance,
)


def run_baseline_case():
    """Run baseline case and return time series."""
    print("=" * 60)
    print("BASELINE CASE: D=1m, U_current=1 m/s")
    print("=" * 60)
    
    floe = IceFloeProperties(diameter=1.0, thickness=0.2)
    cfg = DEMLBMConfig(u_current=1.0, floe=floe, max_time=50.0, dt=0.01)
    sim = IceFloeDriftSimulator(cfg)
    sim.run()
    
    u_analytical = analytical_drift_velocity(1.0)
    u_sim = sim.get_steady_state_velocity()
    error_pct = 100.0 * abs(u_sim - u_analytical) / u_analytical
    
    print(f"Analytical drift velocity: {u_analytical:.4f} m/s")
    print(f"Simulated drift velocity:  {u_sim:.4f} m/s")
    print(f"Error: {error_pct:.2f}%")
    print(f"Converged in {len(sim.time_history)} steps")
    print()
    
    return sim.time_history, sim.velocity_history, sim.force_history


def plot_time_series(time_hist, vel_hist, force_hist, output_dir):
    """Plot time series of velocity and force."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Velocity
    ax1.plot(time_hist, vel_hist, 'b-', linewidth=2, label='DEM-LBM simulation')
    u_analytical = analytical_drift_velocity(1.0)
    ax1.axhline(u_analytical, color='r', linestyle='--', linewidth=2, 
                label=f'Analytical: {u_analytical:.3f} m/s')
    ax1.set_xlabel('Time [s]', fontsize=12)
    ax1.set_ylabel('Floe velocity [m/s]', fontsize=12)
    ax1.set_title('Ice Floe Drift Velocity vs Time', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, max(time_hist)])
    
    # Force
    ax2.plot(time_hist, force_hist, 'g-', linewidth=2)
    ax2.axhline(0, color='k', linestyle='-', linewidth=1)
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('Net force [N]', fontsize=12)
    ax2.set_title('Force Balance (driving - resistive)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, max(time_hist)])
    
    plt.tight_layout()
    fig_path = output_dir / "ice_floe_drift_time_series.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close()


def plot_parameter_sweep(results, output_dir):
    """Plot parameter sweep results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    u_currents = sorted(set(r["u_current"] for r in results))
    diameters = sorted(set(r["diameter"] for r in results))
    
    # Plot analytical vs simulated
    x = np.arange(len(results))
    u_analytical = [r["u_drift_analytical"] for r in results]
    u_sim = [r["u_drift_sim"] for r in results]
    
    width = 0.35
    ax.bar(x - width/2, u_analytical, width, label='Analytical', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, u_sim, width, label='DEM-LBM', color='coral', alpha=0.7)
    
    # Labels
    labels = [f"U={r['u_current']}\nD={r['diameter']}" for r in results]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel('Case (U_current, Diameter)', fontsize=12)
    ax.set_ylabel('Drift velocity [m/s]', fontsize=12)
    ax.set_title('Parameter Sweep: Ice Floe Drift Velocity', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = output_dir / "ice_floe_drift_parameter_sweep.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close()


def plot_error_distribution(results, output_dir):
    """Plot error distribution for parameter sweep."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    errors = [r["error_pct"] for r in results]
    ax.hist(errors, bins=10, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(3.0, color='r', linestyle='--', linewidth=2, label='3% threshold')
    ax.set_xlabel('Error [%]', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Error Distribution: DEM-LBM vs Analytical', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = output_dir / "ice_floe_drift_error_distribution.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close()


def main():
    """Run full benchmark."""
    output_dir = Path(__file__).parent.parent / "docs" / "ice_floe_drift_dem_lbm"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("ICE FLOE DRIFT DEM-LBM BENCHMARK")
    print("=" * 60 + "\n")
    
    # 1. Baseline case
    time_hist, vel_hist, force_hist = run_baseline_case()
    
    # 2. Parameter sweep
    print("=" * 60)
    print("PARAMETER SWEEP: 9 cases (3 U × 3 D)")
    print("=" * 60)
    results = run_parameter_sweep()
    
    print(f"\n{'U_current':<12} {'Diameter':<10} {'U_analytical':<15} {'U_sim':<12} {'Error':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['u_current']:<12.2f} {r['diameter']:<10.2f} "
              f"{r['u_drift_analytical']:<15.4f} {r['u_drift_sim']:<12.4f} "
              f"{r['error_pct']:<10.2f}%")
    
    max_error = max(r["error_pct"] for r in results)
    mean_error = sum(r["error_pct"] for r in results) / len(results)
    print(f"\nMax error: {max_error:.2f}%")
    print(f"Mean error: {mean_error:.2f}%")
    print()
    
    # 3. Negative control
    print("=" * 60)
    print("NEGATIVE CONTROL: No fluid (pure DEM)")
    print("=" * 60)
    u_neg = negative_control_no_fluid()
    print(f"Final velocity: {u_neg:.6f} m/s (expected: 0.0)")
    print(f"Test: {'PASS' if abs(u_neg) < 1e-6 else 'FAIL'}")
    print()
    
    # 4. Force balance
    print("=" * 60)
    print("FORCE BALANCE VERIFICATION")
    print("=" * 60)
    fb = verify_force_balance(u_current=1.0, diameter=1.0)
    print(f"Driving force:  {fb['f_drive']:.4f} N")
    print(f"Resistive force: {fb['f_resist']:.4f} N")
    print(f"Residual: {fb['residual_pct']:.4f}%")
    print(f"Test (<1%): {'PASS' if fb['passed'] else 'FAIL'}")
    print()
    
    # 5. Generate plots
    print("=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    plot_time_series(time_hist, vel_hist, force_hist, output_dir)
    plot_parameter_sweep(results, output_dir)
    plot_error_distribution(results, output_dir)
    
    # 6. Save JSON results
    results_json = {
        "benchmark": "ice_floe_drift_dem_lbm",
        "mission_id": "m5068",
        "baseline": {
            "u_current": 1.0,
            "diameter": 1.0,
            "u_drift_analytical": analytical_drift_velocity(1.0),
            "u_drift_sim": vel_hist[-1],
            "error_pct": 100.0 * abs(vel_hist[-1] - analytical_drift_velocity(1.0)) / analytical_drift_velocity(1.0),
        },
        "parameter_sweep": results,
        "negative_control": {
            "u_final": u_neg,
            "passed": abs(u_neg) < 1e-6,
        },
        "force_balance": fb,
        "summary": {
            "max_error_pct": max_error,
            "mean_error_pct": mean_error,
            "all_cases_pass": all(r["error_pct"] < 3.0 for r in results),
            "force_balance_pass": fb["passed"],
            "negative_control_pass": abs(u_neg) < 1e-6,
        },
    }
    
    json_path = output_dir / "ice_floe_drift_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved: {json_path}")
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"All cases within 3%: {results_json['summary']['all_cases_pass']}")
    print(f"Force balance <1%: {results_json['summary']['force_balance_pass']}")
    print(f"Negative control: {results_json['summary']['negative_control_pass']}")
    print()


if __name__ == "__main__":
    main()
