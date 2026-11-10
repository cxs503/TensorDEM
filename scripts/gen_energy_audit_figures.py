#!/usr/bin/env python
"""Generate figures for the ice-structure energy audit report (p304w2-n03)."""
from __future__ import annotations

import base64
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.ice_structure_energy_ledger import (
    run_energy_audit_case,
    ContactLaw,
    CohesiveBondLaw,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "energy_audit_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _encode_png(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def fig1_energy_ledger_evolution(sim, title: str, filename: str) -> str:
    """Figure 1: Four-ledger energy evolution over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    t = np.array([r["t"] for r in sim.history]) * 1000  # ms
    ek = [r["E_k"] for r in sim.history]
    es = [r["E_s"] for r in sim.history]
    ef = [r["E_f"] for r in sim.history]
    ev = [r["E_v"] for r in sim.history]
    etot = [r["E_total"] for r in sim.history]

    ax.plot(t, ek, label="E_k (kinetic)", linewidth=1.5)
    ax.plot(t, es, label="E_s (contact strain)", linewidth=1.5)
    ax.plot(t, ef, label="E_f (fracture)", linewidth=1.5)
    ax.plot(t, ev, label="E_v (viscous dissipation)", linewidth=1.5)
    ax.plot(t, etot, label="E_total", linewidth=2.0, color="black", linestyle="--")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Energy (J)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig2_residual_evolution(sim_elastic, sim_fracture, filename: str) -> str:
    """Figure 2: Energy residual evolution (elastic vs fracture)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    t_el = np.array([r["t"] for r in sim_elastic.history]) * 1000
    res_el = [r["residual_relative"] for r in sim_elastic.history]
    t_fr = np.array([r["t"] for r in sim_fracture.history]) * 1000
    res_fr = [r["residual_relative"] for r in sim_fracture.history]

    ax.semilogy(t_el, np.maximum(res_el, 1e-16), label="Elastic (no fracture)", linewidth=1.5)
    ax.semilogy(t_fr, np.maximum(res_fr, 1e-16), label="Fracture (24 bonds broken)", linewidth=1.5)
    ax.axhline(y=0.01, color="red", linestyle="--", alpha=0.7, label="1% threshold")
    ax.axhline(y=0.02, color="orange", linestyle="--", alpha=0.7, label="2% threshold")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Energy residual ε_E (log scale)")
    ax.set_title("Energy Conservation Residual: Elastic vs Fracture Impact")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, which="both")

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig3_ledger_breakdown(sim, title: str, filename: str) -> str:
    """Figure 3: Stacked energy breakdown at peak contact."""
    fig, ax = plt.subplots(figsize=(8, 6))
    # Find the step with maximum E_s (peak contact)
    idx_max_es = max(range(len(sim.history)), key=lambda i: sim.history[i]["E_s"])
    row = sim.history[idx_max_es]

    labels = ["E_k\n(kinetic)", "E_s\n(strain)", "E_f\n(fracture)", "E_v\n(viscous)"]
    values = [row["E_k"], row["E_s"], row["E_f"], row["E_v"]]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 10}
    )
    ax.set_title(f"{title}\n(t={row['t']*1000:.2f} ms, step {idx_max_es})")

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig4_timestep_sensitivity(filename: str) -> str:
    """Figure 4: Timestep sensitivity study."""
    dts = [2.0e-5, 1.0e-5, 5.0e-6, 2.5e-6]
    residuals_elastic = []
    residuals_fracture = []

    for dt in dts:
        sim_el = run_energy_audit_case(
            impact_velocity=(-2.0, 0.0, 0.0),
            contact=ContactLaw(k_normal=1.0e6, restitution=1.0),
            bond_law=CohesiveBondLaw(tensile_strength=4.0e5),
            dt=dt, duration=8.0e-3, gap=2.0e-3,
        )
        sim_fr = run_energy_audit_case(
            impact_velocity=(-2.0, 0.0, 0.0),
            contact=ContactLaw(k_normal=1.0e6, restitution=0.9),
            bond_law=CohesiveBondLaw(tensile_strength=2.0e5),
            dt=dt, duration=8.0e-3, gap=2.0e-3,
        )
        residuals_elastic.append(sim_el.summary()["residual_relative_max"])
        residuals_fracture.append(sim_fr.summary()["residual_relative_max"])

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(dts))
    width = 0.35

    ax.bar(x - width/2, residuals_elastic, width, label="Elastic", color="#4C72B0")
    ax.bar(x + width/2, residuals_fracture, width, label="Fracture", color="#C44E52")
    ax.axhline(y=0.01, color="red", linestyle="--", alpha=0.7, label="1% threshold")
    ax.axhline(y=0.02, color="orange", linestyle="--", alpha=0.7, label="2% threshold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{dt*1e6:.0f} μs" for dt in dts])
    ax.set_xlabel("Time step Δt")
    ax.set_ylabel("Max energy residual ε_E")
    ax.set_title("Timestep Sensitivity: ε_E vs Δt")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    print("Running elastic case...")
    sim_elastic = run_energy_audit_case(
        impact_velocity=(-2.0, 0.0, 0.0),
        contact=ContactLaw(k_normal=1.0e6, restitution=1.0),
        bond_law=CohesiveBondLaw(tensile_strength=4.0e5),
        dt=1.0e-5, duration=8.0e-3, gap=2.0e-3,
    )
    print("Running fracture case...")
    sim_fracture = run_energy_audit_case(
        impact_velocity=(-2.0, 0.0, 0.0),
        contact=ContactLaw(k_normal=1.0e6, restitution=0.9),
        bond_law=CohesiveBondLaw(tensile_strength=2.0e5),
        dt=1.0e-5, duration=8.0e-3, gap=2.0e-3,
    )

    print("Generating Figure 1: Energy ledger evolution (elastic)...")
    fig1_energy_ledger_evolution(
        sim_elastic, "Energy Ledger Evolution: Elastic Impact (no fracture)",
        "fig1_elastic_ledger.png"
    )
    print("Generating Figure 1b: Energy ledger evolution (fracture)...")
    fig1_energy_ledger_evolution(
        sim_fracture, "Energy Ledger Evolution: Fracture Impact (24 bonds broken)",
        "fig1b_fracture_ledger.png"
    )
    print("Generating Figure 2: Residual evolution...")
    fig2_residual_evolution(sim_elastic, sim_fracture, "fig2_residual_evolution.png")
    print("Generating Figure 3: Ledger breakdown (elastic)...")
    fig3_ledger_breakdown(sim_elastic, "Energy Breakdown at Peak Contact (Elastic)", "fig3_elastic_pie.png")
    print("Generating Figure 3b: Ledger breakdown (fracture)...")
    fig3_ledger_breakdown(sim_fracture, "Energy Breakdown at Peak Contact (Fracture)", "fig3b_fracture_pie.png")
    print("Generating Figure 4: Timestep sensitivity...")
    fig4_timestep_sensitivity("fig4_timestep_sensitivity.png")

    # Collect base64 for report
    figures = []
    for fname in ["fig1_elastic_ledger.png", "fig1b_fracture_ledger.png",
                   "fig2_residual_evolution.png", "fig3_elastic_pie.png",
                   "fig3b_fracture_pie.png", "fig4_timestep_sensitivity.png"]:
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        figures.append({"name": fname, "png_base64": b64})

    # Save figures JSON for report
    with open(os.path.join(OUTPUT_DIR, "figures.json"), "w") as f:
        json.dump(figures, f)

    print(f"\nAll figures saved to {OUTPUT_DIR}")
    print(f"Elastic residual max: {sim_elastic.summary()['residual_relative_max']:.6e}")
    print(f"Fracture residual max: {sim_fracture.summary()['residual_relative_max']:.6e}")


if __name__ == "__main__":
    main()
