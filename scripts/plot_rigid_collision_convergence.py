#!/usr/bin/env python
"""Generate convergence plots for the rigid-collision energy-ledger certification.

Produces:
- convergence_curves.png: residual vs dt for all three criteria
- newton_cradle.png: momentum/energy conservation visualization

Run: PYTHONPATH=src python scripts/plot_rigid_collision_convergence.py --output runs/m7444
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensordem.rigid_collision_ledger import (
    CollisionConfig,
    dt_sweep,
    observed_order,
    run_newton_cradle,
    run_single_collision,
)
from tensordem.dem3d import DEM3DConfig


def plot_convergence_curves(output_dir: Path) -> dict:
    """Generate convergence curve plot and return data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run dt sweep with k_max=3 for speed
    rows = dt_sweep(CollisionConfig(impact_speed=1.0), k_max=3)

    dts = np.array([r["dt_s"] for r in rows])
    residual_state = np.array([r["residual_state_diff_j"] for r in rows])
    residual_work_exp = np.array([r["residual_work_explicit_j"] for r in rows])
    residual_work_trap = np.array([r["residual_work_trap_j"] for r in rows])

    # Compute orders
    order_state = observed_order(residual_state.tolist(), dts.tolist())
    order_exp = observed_order(residual_work_exp.tolist(), dts.tolist())
    order_trap = observed_order(residual_work_trap.tolist(), dts.tolist())

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot residuals
    ax.loglog(dts, residual_state, "o-", label=f"State-diff (order={order_state:.2f})",
              color="C0", markersize=8)
    ax.loglog(dts, residual_work_exp, "s-", label=f"Explicit work (order={order_exp:.2f})",
              color="C1", markersize=8)
    ax.loglog(dts, residual_work_trap, "^-", label=f"Trapezoidal work (order={order_trap:.2f})",
              color="C2", markersize=8)

    # Reference lines
    dt_ref = dts[0]
    ref_state = residual_state[0]
    ref_exp = residual_work_exp[0]
    ref_trap = residual_work_trap[0]

    ax.loglog(dts, ref_state * (dts / dt_ref) ** 1, "k--", alpha=0.3, label="O(dt)")
    ax.loglog(dts, ref_trap * (dts / dt_ref) ** 2, "k:", alpha=0.3, label="O(dt^2)")

    ax.set_xlabel("Time step dt (s)", fontsize=12)
    ax.set_ylabel("Energy residual (J)", fontsize=12)
    ax.set_title("Rigid Collision Energy Ledger: Convergence of Three Criteria", fontsize=14)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    png_path = output_dir / "convergence_curves.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # Save data
    data = {
        "dt_sweep": rows,
        "orders": {
            "state_diff": order_state,
            "explicit_work": order_exp,
            "trapezoidal_work": order_trap,
        }
    }
    json_path = output_dir / "convergence_data.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    return {"png": str(png_path), "json": str(json_path), "orders": data["orders"]}


def plot_newton_cradle(output_dir: Path) -> dict:
    """Generate Newton cradle conservation plot."""
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_newton_cradle(n=5, v0=1.0, dt=5e-5, max_steps=8000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Collision velocities
    ax1.set_title("Newton Cradle: Collision Velocities", fontsize=12)
    for col in result.collisions:
        idx = col["index"]
        pair = col["pair"]
        v_before = col["v_before"]
        v_after = col["v_after"]
        ax1.scatter([idx, idx], v_before, c="C0", marker="o", s=80, label="Before" if idx == 0 else "")
        ax1.scatter([idx, idx], v_after, c="C1", marker="x", s=80, label="After" if idx == 0 else "")
        ax1.plot([idx, idx], v_before, "C0-", alpha=0.3)
        ax1.plot([idx, idx], v_after, "C1-", alpha=0.3)
    ax1.set_xlabel("Collision index")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Conservation errors
    ax2.set_title("Conservation Errors", fontsize=12)
    ax2.bar(["Momentum", "Energy"],
            [result.momentum_error_pct, result.energy_error_pct],
            color=["C3", "C4"])
    ax2.set_ylabel("Relative error (%)")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    png_path = output_dir / "newton_cradle.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # Save data
    data = {
        "n_balls": result.n_balls,
        "mass": result.mass,
        "dt": result.dt,
        "v0": result.v0,
        "e_theory": result.e_theory,
        "collisions": result.collisions,
        "momentum_error_pct": result.momentum_error_pct,
        "energy_error_pct": result.energy_error_pct,
        "energy_residual_j": result.energy_residual_j,
    }
    json_path = output_dir / "newton_cradle_data.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    return {"png": str(png_path), "json": str(json_path), "data": data}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/m7444"))
    args = parser.parse_args()

    print("Generating convergence curves...")
    conv = plot_convergence_curves(args.output)
    print(f"  Orders: {conv['orders']}")
    print(f"  PNG: {conv['png']}")

    print("Generating Newton cradle plot...")
    nc = plot_newton_cradle(args.output)
    print(f"  Collisions: {len(nc['data']['collisions'])}")
    print(f"  Momentum error: {nc['data']['momentum_error_pct']:.4f}%")
    print(f"  Energy error: {nc['data']['energy_error_pct']:.4f}%")
    print(f"  PNG: {nc['png']}")

    print("Done.")


if __name__ == "__main__":
    main()
