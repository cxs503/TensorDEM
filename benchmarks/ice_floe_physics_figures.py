"""Run the ice-floe verification anchors and render the report figure.

Writes:
  artifacts/ice_floe_physics/validation_report.json   (full-step configuration)
  artifacts/ice_floe_physics/ice_floe_verification.png
  artifacts/ice_floe_physics/figure_data.json         (numbers plotted, no hand copy)

Usage:  python3 benchmarks/ice_floe_physics_figures.py [--fast]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

torch.set_num_threads(1)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT = _REPO_ROOT / "artifacts" / "ice_floe_physics"


def _load_benchmark():
    path = _REPO_ROOT / "benchmarks" / "ice_floe_physics.py"
    spec = importlib.util.spec_from_file_location("ice_floe_physics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def heave_history(bench, dt=4e-3, periods=2.5):
    radius, thickness = 0.5, 1.0
    rho_i, rho_w = 917.0, 1025.0
    mass, cfg = bench._flotation_case(rho_i, rho_w, radius, thickness, drag=0.0)
    h_eq = bench.analytic_equilibrium_height(rho_i, rho_w, radius)
    period = 2.0 * math.pi * math.sqrt(
        mass / (rho_w * bench.G * thickness * 2.0 * math.sqrt(radius ** 2 - h_eq ** 2)))
    steps = int(round(periods * period / dt))
    pos = torch.tensor([[0.0, h_eq + 4e-3 * radius]], dtype=torch.float64)
    vel = torch.zeros_like(pos)
    ts, ys = [], []
    for n in range(steps):
        f = bench.compute_buoyancy_force(pos, radius, thickness, rho_i, cfg)
        f[:, 1] -= mass * bench.G
        vel = vel + f / mass * dt
        pos = pos + vel * dt
        ts.append(n * dt)
        ys.append(float(pos[0, 1]) - h_eq)
    return ts, ys, period


def settling_history(bench, dt=4e-3, duration=4.0):
    radius, thickness = 0.5, 1.0
    rho_i, rho_w = 917.0, 1025.0
    mass, cfg = bench._flotation_case(rho_i, rho_w, radius, thickness)
    h_eq = bench.analytic_equilibrium_height(rho_i, rho_w, radius)
    pos = torch.tensor([[0.0, 1.2 * radius]], dtype=torch.float64)
    vel = torch.zeros_like(pos)
    ts, ys = [], []
    for n in range(int(round(duration / dt))):
        f = bench.compute_buoyancy_force(pos, radius, thickness, rho_i, cfg)
        f = f + bench.compute_wet_drag_force(pos, vel, radius, cfg)
        f[:, 1] -= mass * bench.G
        vel = vel + f / mass * dt
        pos = pos + vel * dt
        ts.append(n * dt)
        ys.append(float(pos[0, 1]) - h_eq)
    return ts, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    bench = _load_benchmark()
    report = bench.run_all(fast=args.fast)
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")

    by_name = {a["anchor"]: a for a in report["anchors"]}
    t_settle, y_settle = settling_history(bench)
    t_heave, y_heave, period = heave_history(bench)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))

    # (a) buoyancy force / displaced volume vs centre height
    ax = axes[0, 0]
    rows = by_name["A_archimedes_static"]["rows"]
    h = [r["centre_height_m"] for r in rows]
    ax.plot(h, [r["f_sub_closed_form"] for r in rows], "-",
            label="closed-form A_sub/(pi R^2)")
    ax.plot(h, [r["f_sub_code"] for r in rows], "o", ms=4,
            label="compute_buoyancy_force")
    h_eq = by_name["H_hydrostatic_equilibrium"]["h_eq_reference_m"]
    f_eq = by_name["H_hydrostatic_equilibrium"]["f_eq_reference"]
    ax.plot([h_eq], [f_eq], "k*", ms=13,
            label=f"equilibrium h={h_eq:.4f} m, f={f_eq:.4f}")
    ax.set_xlabel("disk centre height above waterline  [m]")
    ax.set_ylabel("submerged fraction  f_sub  [-]")
    ax.set_title("(a) Archimedes: force vs independently derived area")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    # (b) settling trajectory
    ax = axes[0, 1]
    ax.plot(t_settle, [y / 0.5 for y in y_settle], lw=1.2)
    ax.axhline(0.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("time  [s]")
    ax.set_ylabel("(y - h_eq) / R  [-]")
    ax.set_title("(b) release from 1.2R: settles on the Archimedes waterline")
    ax.grid(alpha=0.3)

    # (c) heave oscillation against the linear period
    ax = axes[1, 0]
    ax.plot(t_heave, [y / 0.5 * 1e3 for y in y_heave], lw=1.2,
            label="simulated heave")
    for k in range(0, 4):
        ax.axvline(k * period, color="tab:red", ls="--", lw=0.8,
                   label="analytic T" if k == 0 else None)
    ax.set_xlabel("time  [s]")
    ax.set_ylabel("(y - h_eq) / R  [1e-3]")
    ax.set_title(f"(c) heave period: T_ref = {period:.4f} s")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) friction onset
    ax = axes[1, 1]
    for row in by_name["F_friction_slip"]["rows"]:
        lvl = row["dt_levels"][-1]
        theta = [row["theta_onset_theory_deg"] - 1.0 + 2.0 * i / 20
                 for i in range(21)]
        scale = lvl["stick_scale_m"]
        ax.semilogy(theta, [max(t, 1e-3) / s for t, s in zip(lvl["travel_m"], scale)],
                    "o-", ms=3, lw=1.0, label=f"mu = {row['mu']}")
        ax.axvline(row["theta_onset_theory_deg"], color="grey", ls="--", lw=0.8)
    ax.set_xlabel("incline angle  [deg]")
    ax.set_ylabel("travel / (mu N / k_t)  [-]")
    ax.set_title("(d) Cundall-Strack slip onset at arctan(mu) (grey dashed)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("TensorDEM ice-floe physics verification "
                 f"(schema {report['schema']}, mode {report['mode']})")
    fig.tight_layout()
    png = _OUT / "ice_floe_verification.png"
    fig.savefig(png, dpi=150)

    fig_data = {
        "period_reference_s": period,
        "f_eq_reference": f_eq,
        "h_eq_reference_m": h_eq,
        "anchors": {a["anchor"]: a["passed"] for a in report["anchors"]},
        "overall_pass": report["overall_pass"],
    }
    (_OUT / "figure_data.json").write_text(json.dumps(fig_data, indent=2) + "\n")
    print(json.dumps(fig_data, indent=2))
    print(f"png: {png}")


if __name__ == "__main__":
    main()