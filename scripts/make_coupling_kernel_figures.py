#!/usr/bin/env python
"""Figures for the coupling-kernel maturity report (p223w3-n05).

Consumes the JSON artifacts written by
``scripts/run_coupling_kernel_settling.py`` and emits PNGs under
``artifacts/coupling_kernel/figures/``.  Figures are generated from the
artifacts only (no hand-typed numbers).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "coupling_kernel"
FIG = ART / "figures"


def _load(name: str):
    path = ART / f"{name}.json"
    if not path.exists():
        print(f"skip {name}: missing {path}")
        return None
    return json.loads(path.read_text())


def fig_dilution(data: dict) -> Path | None:
    if not data:
        return None
    tau = data["relaxation_time_s"]
    v_t = data["stokes_terminal_velocity_m_s"]
    steps = data["steps"]
    # reconstruct the exact curve over the simulated window
    t = np.linspace(0.0, data["time_s"], 400)
    exact = v_t * (1.0 - np.exp(-t / tau))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(t * 1e3, exact * 1e3, "-", color="C0", label="exact linear-drag solution")
    ax.axhline(v_t * 1e3, color="C0", ls=":", lw=1.0, label=r"Stokes $v_t$")
    ax.axhline(data["oseen_terminal_velocity_m_s"] * 1e3, color="C3", ls="--", lw=1.0, label="Oseen $v_t$")
    ax.plot([data["time_s"] * 1e3], [data["measured_terminal_velocity_m_s"] * 1e3], "o", color="C1",
            label="DEM kernel (end state)")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel(r"$v_z$ [mm/s]")
    ax.set_title(f"Single-sphere settling vs exact drag solution (Re = {data['particle_reynolds']:.1e})")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    out = FIG / "dilution_relaxation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_conservation(data: dict) -> Path | None:
    if not data:
        return None
    series = data["ledger_series"]
    step = np.asarray(series["step"], dtype=float)
    mech = np.asarray(series["mechanical_J"], dtype=float)
    e0 = data["baseline_mechanical_J"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(step, (mech - e0) / abs(e0), "-", color="C0", lw=1.2, label=r"$(K+U_c)(t)-(K+U_c)(0)$ / $E_0$")
    ax.plot(step, np.asarray(series["kinetic_J"]) / abs(e0), "-", color="C2", lw=0.9, alpha=0.8, label=r"$K/E_0$")
    ax.plot(step, np.asarray(series["elastic_J"]) / abs(e0), "-", color="C3", lw=0.9, alpha=0.8, label=r"$U_c/E_0$")
    ax.set_xlabel("step")
    ax.set_ylabel("normalised energy [-]")
    ax.set_title(
        f"Elastic gas energy budget, no dissipation (drift {data['final_relative_residual']:.2e})"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out = FIG / "conservation_energy.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_settling(data: dict) -> Path | None:
    if not data:
        return None
    series = data["ledger_series"]
    t = np.asarray(series["step"], dtype=float) * data["config"]["dt"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    for key, label, color in (
        ("kinetic_J", "DEM bulk: $K$", "C0"),
        ("elastic_J", "DEM bulk: $U_c$ (contact)", "C2"),
        ("gravity_potential_J", r"DEM bulk: $U_g$ (rel. to initial)", "C4"),
        ("dem_damping_dissipated_J", "contact field: pair damping", "C1"),
        ("wall_damping_dissipated_J", "contact field: wall damping", "C5"),
        ("drag_dissipated_J", "fluid field: drag work", "C3"),
    ):
        y = np.asarray(series[key], dtype=float)
        if key == "gravity_potential_J":
            y = y - y[0]
        ax.plot(t * 1e3, y, label=label, lw=1.2)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("energy channel [J]")
    ax.set_title("Three-field energy ledger, dense bed (real DEM kernel)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax = axes[1]
    row = data["ledger_summary"]
    labels = ["$K$", "$U_g$", "pair dashpot", "wall dashpot", "drag"]
    vals = [
        row["final_kinetic_J"],
        row["final_gravity_potential_J"] - data["ledger_series"]["gravity_potential_J"][0],
        row["dem_damping_dissipated_J"],
        row["wall_damping_dissipated_J"],
        row["drag_dissipated_J"],
    ]
    ax.bar(np.arange(len(vals)), [-v if v < 0 else v for v in vals], color=["C0", "C4", "C1", "C5", "C3"])
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yscale("symlog", linthresh=1e-16)
    ax.set_ylabel("|J| at end of run")
    ax.set_title(f"Closure residual {data['energy_closure_relative_residual']:.1e} (relative)")
    ax.grid(alpha=0.3, axis="y")
    out = FIG / "settling_three_field_ledger.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_convergence(data: dict) -> Path | None:
    if not data:
        return None
    runs = data["runs"]
    dt = np.array([r["dt_s"] for r in runs])
    res = np.array([max(abs(r["final_relative_residual"]), 1e-18) for r in runs])
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.loglog(dt, res, "o-", color="C0", label="ledger drift")
    ref_dt = np.array([dt.min(), dt.max()])
    ax.loglog(ref_dt, res.min() * (ref_dt / ref_dt.min()) ** 2, "k--", lw=1.0, label="slope 2 guide")
    ax.set_xlabel(r"$\Delta t$ [s]")
    ax.set_ylabel("|relative energy drift|")
    ax.set_title(f"Timestep refinement (fitted order {data['observed_order']:.2f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    out = FIG / "convergence_order.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    for name, fn in (
        ("dilution", fig_dilution),
        ("conservation", fig_conservation),
        ("settling", fig_settling),
        ("convergence", fig_convergence),
    ):
        out = fn(_load(name))
        if out:
            made.append(out)
            print("wrote", out)
    print(f"{len(made)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())