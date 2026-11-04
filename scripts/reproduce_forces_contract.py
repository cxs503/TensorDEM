"""Reproduce the m3147 evidence: IceDEM3D.forces() override contract + density ledger.

Outputs (all under ``docs/artifacts/forces_contract/``):

* ``contract_summary.json``   -- per-subclass keyword contract, purity probe, case numbers
* ``bow_history.csv``         -- diagnostic history of one small threefield hull-ice case
* ``density_error.csv``       -- mass error induced by a hard-coded 917 kg/m3 density
* ``tables.md``               -- markdown tables quoted verbatim in the report
* ``fig1_bow_stage.png``      -- ice resistance / bow travel vs time (the stage that was dead)
* ``fig2_density_error.png``  -- induced mass error vs configured ice density + 3 % gate

Run:  PYTHONPATH=src python3 scripts/reproduce_forces_contract.py
"""
from __future__ import annotations

import csv
import importlib
import inspect
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

OUT = Path(__file__).resolve().parents[1] / "docs" / "artifacts" / "forces_contract"
OUT.mkdir(parents=True, exist_ok=True)

BASE_FORCE_KEYWORDS = ("external_forces", "update_fracture", "update_history")
SOLVER_MODULES = ("dem3d", "dem3d_energy", "dem3d_wedge", "rudder_ice_load", "threefield.hull_ice")


def probe_contract() -> list[dict]:
    for module in SOLVER_MODULES:
        importlib.import_module(f"tensordem.{module}")
    from tensordem.dem3d import IceDEM3D

    rows: list[dict] = []
    stack = [IceDEM3D]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            stack.append(sub)
            params = inspect.signature(sub.forces).parameters
            rows.append(
                {
                    "class": sub.__name__,
                    "accepts_update_history": "update_history" in params,
                    "missing_keywords": [k for k in BASE_FORCE_KEYWORDS if k not in params],
                    "own_forces_override": sub.forces is IceDEM3D.forces
                    or sub.forces.__qualname__.split(".")[0] != "IceDEM3D",
                }
            )
    rows.sort(key=lambda row: row["class"])
    return rows


def probe_purity() -> dict:
    """diagnostics() must not perturb the simulated state."""
    from tensordem.threefield.hull_ice import HullIceCase
    from tensordem.threefield.hull_ice import BowContactDEM

    case = HullIceCase(thickness_m=0.15, particle_radius_m=0.025, n_length=5, n_width=2, steps=20)
    sim = BowContactDEM(case.to_dem_config(), bow_speed=case.bow_speed_m_s,
                        wedge_angle_deg=case.wedge_angle_deg, initial_gap=0.0,
                        tip_height_fraction=case.tip_height_fraction, bow_half_width_m=None)
    for _ in range(6):
        sim.step()
    before = {
        "positions": sim.positions.clone(),
        "velocities": sim.velocities.clone(),
        "alive": sim.alive.clone(),
        "rolling_torque": sim._rolling_torque.clone(),
        "last_reaction": sim.last_reaction.clone(),
        "time": float(sim.time),
    }
    diag = sim.diagnostics()
    after = {
        "positions": sim.positions.clone(),
        "velocities": sim.velocities.clone(),
        "alive": sim.alive.clone(),
        "rolling_torque": sim._rolling_torque.clone(),
        "last_reaction": sim.last_reaction.clone(),
        "time": float(sim.time),
    }
    drift = {name: float((after[name] - before[name]).abs().max().item())
             for name in ("positions", "velocities", "rolling_torque", "last_reaction")}
    drift["alive_changes"] = int((after["alive"] != before["alive"]).sum().item())
    drift["time"] = after["time"] - before["time"]
    return {"drift": drift, "bow_reaction_x_N": float(diag["bow_reaction_x_N"])}


def run_case() -> list[dict]:
    from tensordem.threefield.hull_ice import HullIceCase, run_hull_ice_case

    case = HullIceCase(thickness_m=0.15, particle_radius_m=0.025, n_length=5, n_width=2, steps=40)
    result = run_hull_ice_case(case)
    rows = result["history"]
    with (OUT / "bow_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows, result


def density_error() -> list[dict]:
    """Mass error induced by reporting every mass at 917 kg/m3 instead of rho."""
    rows = []
    for rho in (880, 890, 900, 910, 917, 925, 935, 950):
        rows.append({"density_kg_m3": rho, "induced_mass_error_pct": (917.0 - rho) / rho * 100.0})
    with (OUT / "density_error.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def figure_bow(rows: list[dict], purity: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    time = [row["time_s"] for row in rows]
    axes[0].plot(time, [row["ice_resistance_N"] for row in rows], lw=1.8, color="#1f4e79")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("ice resistance [N] (unit width)")
    axes[0].set_title("threefield hull-ice stage after the fix")
    axes[0].grid(alpha=0.3)

    axes[1].plot(time, [row["bow_tip_x_m"] for row in rows], lw=1.8, color="#c0504d",
                 label="bow tip x")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("bow tip position [m]")
    axes[1].set_title("bow advance (prescribed speed)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_bow_stage.png", dpi=150)
    plt.close(fig)


def figure_density(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    densities = [row["density_kg_m3"] for row in rows]
    errors = [row["induced_mass_error_pct"] for row in rows]
    ax.plot(densities, errors, "o-", color="#1f4e79", lw=1.8)
    ax.axhspan(-3.0, 3.0, color="#70ad47", alpha=0.18, label="+/-3 % acceptance band")
    ax.axhline(3.0, color="#70ad47", lw=1.0)
    ax.axhline(-3.0, color="#70ad47", lw=1.0)
    ax.axvline(917.0, color="#a6a6a6", ls="--", lw=1.0, label="hard-coded 917 kg/m3")
    for row in rows:
        if row["density_kg_m3"] in (890, 917, 935):
            ax.annotate(f"{row['induced_mass_error_pct']:+.2f} %",
                        (row["density_kg_m3"], row["induced_mass_error_pct"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("configured ice density rho_i [kg/m3]")
    ax.set_ylabel("mass error of the hard-coded ledger [%]")
    ax.set_title("fragmentation mass ledger: 917 kg/m3 hard code")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_density_error.png", dpi=150)
    plt.close(fig)


def write_tables(contract, purity, rows, density_rows) -> None:
    lines = ["### Per-subclass `forces()` keyword contract (live introspection)", "",
             "| subclass override | `update_history` accepted | missing base keywords |",
             "|---|---|---|"]
    for row in contract:
        lines.append(
            f"| `{row['class']}` | {'yes' if row['accepts_update_history'] else '**NO**'} | "
            f"{', '.join(row['missing_keywords']) if row['missing_keywords'] else 'none'} |"
        )
    lines += ["", "### `diagnostics()` purity probe on `BowContactDEM` (6 steps, then probe)", "",
              "| state component | max |delta| after probe |", "|---|---|"]
    for name, value in purity["drift"].items():
        lines.append(f"| `{name}` | {value:.3e} |")
    lines += ["", "### Bowl-ice stage (thickness 0.15 m, r = 0.025 m, 40 steps)", "",
              "| quantity | value |", "|---|---|"]
    lines.append(f"| simulated time [s] | {rows[-1]['time_s']:.4f} |")
    lines.append(f"| bow travel [m] | {rows[-1]['bow_tip_x_m'] - rows[0]['bow_tip_x_m']:.4f} |")
    lines.append(f"| peak ice resistance [N] | {max(r['ice_resistance_N'] for r in rows):.3f} |")
    lines.append(f"| mean ice resistance [N] | "
                 f"{sum(r['ice_resistance_N'] for r in rows) / len(rows):.3f} |")
    lines.append(f"| max contact pressure [Pa] | {max(r['max_contact_pressure_pa'] for r in rows):.3e} |")
    lines.append(f"| broken bonds at end | {int(rows[-1]['broken_bonds'])} |")
    lines += ["", "### Mass error induced by the 917 kg/m3 hard code", "",
              "| rho_i [kg/m3] | induced mass error [%] | inside +/-3 % gate |", "|---|---|---|"]
    for row in density_rows:
        ok = "yes" if abs(row["induced_mass_error_pct"]) <= 3.0 else "**no**"
        lines.append(f"| {row['density_kg_m3']} | {row['induced_mass_error_pct']:+.3f} | {ok} |")
    (OUT / "tables.md").write_text("\n".join(lines) + "\n")


def main() -> dict:
    contract = probe_contract()
    purity = probe_purity()
    rows, result = run_case()
    density_rows = density_error()
    figure_bow(rows, purity)
    figure_density(density_rows)
    write_tables(contract, purity, rows, density_rows)
    summary = {
        "contract": contract,
        "purity": purity,
        "case": {
            "case": result["case"],
            "peak_ice_resistance_N": result["peak_ice_resistance_N"],
            "mean_ice_resistance_N": result["mean_ice_resistance_N"],
            "fragment_total_mass_kg": result["fragments"]["total_mass_kg"],
            "broken_bonds": result["fragments"]["broken_bonds"],
            "max_contact_pressure_pa": result["max_contact_pressure_pa"],
        },
        "density_error": density_rows,
    }
    (OUT / "contract_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["purity"], indent=2))
    print((OUT / "tables.md").read_text())
    return summary


if __name__ == "__main__":
    main()