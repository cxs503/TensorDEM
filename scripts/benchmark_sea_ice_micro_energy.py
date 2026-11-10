"""Benchmark for the sea-ice micro-fracture energy budget (p216w3-n07).

Runs the full three-channel energy budget audit and writes the machine-checkable
criteria ledger to ``artifacts/sea_ice_micro_energy/``.  Also renders the
F-delta curve and the energy-allocation-vs-fragment-count figure.

Usage::

    PYTHONPATH=src python scripts/benchmark_sea_ice_micro_energy.py \
        --output artifacts/sea_ice_micro_energy
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordem.sea_ice_micro_energy import (
    check_f_delta_monotonic,
    check_fragment_convergence,
    n06_break_crush_channel,
    run_energy_budget_audit,
    _run_single_fracture,
)

#: Particle diameter for the convergence study (large enough that the
#: surface-energy demand stays below the released elastic budget).
CONVERGENCE_DIAMETER = 32.0e-3


def render_f_delta_curve(budget: dict, output: Path) -> Path:
    """Render the force-displacement curve with the elastic limit marked."""
    fig, ax = plt.subplots(figsize=(8, 5))
    force = np.asarray(budget["force_history_N"])
    disp = np.asarray(budget["displacement_history_m"])
    ax.plot(disp * 1e3, force * 1e-3, "b-", lw=1.5, label="F($\\delta$)")
    # Mark the elastic limit (first force drop).
    for k in range(1, len(force)):
        if force[k] < force[k - 1]:
            ax.axvline(disp[k] * 1e3, color="r", ls="--", lw=1,
                       label=f"fracture @ $\\delta$={disp[k]*1e3:.2f} mm")
            break
    ax.set_xlabel("Displacement $\\delta$ [mm]")
    ax.set_ylabel("Force $F$ [kN]")
    ax.set_title("Single-particle sea-ice fracture: F-$\\delta$ curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output / "f_delta_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render_allocation_convergence(convergence: dict, output: Path) -> Path:
    """Render the energy-allocation fractions vs fragment count."""
    fig, ax = plt.subplots(figsize=(8, 5))
    n_values = [b["n_frag"] for b in convergence["budgets"]]
    surf = [b["fractions"]["surf"] for b in convergence["budgets"]]
    kin = [b["fractions"]["kin"] for b in convergence["budgets"]]
    diss = [b["fractions"]["diss"] for b in convergence["budgets"]]
    ax.plot(n_values, surf, "o-", label="surface $E_{surf}/E_{in}$")
    ax.plot(n_values, kin, "s-", label="kinetic $E_{kin}/E_{in}$")
    ax.plot(n_values, diss, "^-", label="dissipation $E_{diss}/E_{in}$")
    ax.set_xlabel("Fragment count $n_{frag}$")
    ax.set_ylabel("Energy allocation fraction")
    ax.set_title("Energy allocation vs fragment count (convergence)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output / "allocation_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/sea_ice_micro_energy"))
    parser.add_argument("--diameter", type=float, default=CONVERGENCE_DIAMETER)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4000)
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    # Full audit.
    audit = run_energy_budget_audit(
        diameter=args.diameter, seed=args.seed, platen_velocity=-0.05,
        n_frag=8, steps=args.steps,
        n_frag_values=(4, 8, 16, 32),
    )

    # Render figures.
    f_delta_png = render_f_delta_curve(audit["budget"], output)
    alloc_png = render_allocation_convergence(audit["convergence"], output)

    # Write the criteria ledger.
    ledger = {
        "protocol": "tensordem-sea-ice-micro-energy-v1",
        "diameter_m": args.diameter,
        "seed": args.seed,
        "steps": args.steps,
        "criteria": audit["criteria"],
        "all_passed": audit["all_passed"],
        "budget": {
            "E_in_J": audit["budget"]["E_in_J"],
            "E_surf_J": audit["budget"]["E_surf_J"],
            "E_kin_J": audit["budget"]["E_kin_J"],
            "E_diss_J": audit["budget"]["E_diss_J"],
            "E_out_J": audit["budget"]["E_out_J"],
            "epsilon_s": audit["budget"]["epsilon_s"],
            "fractions": audit["budget"]["fractions"],
            "n_frag": audit["budget"]["n_frag"],
            "dA_m2": audit["budget"]["dA_m2"],
        },
        "f_delta": {
            "n_drops": audit["f_delta"]["n_drops"],
            "max_drop_N": audit["f_delta"]["max_drop_N"],
            "monotonic": audit["f_delta"]["monotonic"],
        },
        "convergence": {
            "n_frag_values": list(audit["convergence"]["n_frag_values"]),
            "max_delta": audit["convergence"]["max_delta"],
            "converged": audit["convergence"]["converged"],
        },
        "n06": {
            "channel": audit["n06"]["channel"],
            "sub_channel": audit["n06"]["sub_channel"],
            "closure_C1": audit["n06"]["closure_C1"],
        },
        "figures": {
            "f_delta_curve": str(f_delta_png),
            "allocation_convergence": str(alloc_png),
        },
    }
    (output / "criteria_ledger.json").write_text(
        json.dumps(ledger, indent=2, default=str))

    # Print the verdict.
    print(f"All criteria passed: {audit['all_passed']}")
    for name, criterion in audit["criteria"].items():
        print(f"  {name}: {criterion['measured']:.6f} -> {'PASS' if criterion['passed'] else 'FAIL'}")
    print(f"Figures: {f_delta_png}, {alloc_png}")


if __name__ == "__main__":
    main()
