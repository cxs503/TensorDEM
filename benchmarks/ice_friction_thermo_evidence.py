"""Regenerate the acceptance artifact and the report figures (P272W2-N11).

    python benchmarks/ice_friction_thermo_evidence.py [--out DIR]

Writes
------
``artifacts/ice_friction_thermo/acceptance.json``   every number the campaign
    asks for, plus the SHA-256 of the payload (criterion C6);
``artifacts/ice_friction_thermo/fig_*.png``         the figures embedded in the
    report.

No number in this script is typed: everything is read back out of
:mod:`tensordem.ice_friction_thermo`, and the artifact carries its digest so a
later edit is detectable (:func:`tensordem.ice_friction_thermo.verify_evidence`).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensordem.ice_friction_thermo import (  # noqa: E402
    CC_SLOPE_K_PER_MPA,
    ContactConditions,
    ContactFrictionModel,
    ICE,
    STEEL,
    acceptance_report,
    blok_interface_rise_k,
    melt_fraction,
    melting_temperature_c,
    pressure_melting_threshold_pa,
    solve_contact_state,
    TwoBodyContactConductor,
    write_evidence,
)
from tensordem.ice_steel_friction import (  # noqa: E402
    CALIBRATION_ANCHORS,
    VALIDATION_ANCHORS,
)

DEFAULT_OUT = Path("artifacts/ice_friction_thermo")


def _fig_temperature(report: dict, path: Path) -> None:
    """mu(T): the model in three regimes against the compiled literature rows."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    temperatures = np.linspace(-30.0, -0.2, 300)
    model = ContactFrictionModel(**{k: v for k, v in report["model"].items()
                                    if k in ContactFrictionModel.__dataclass_fields__})
    ax.plot(temperatures, [model.mu(float(t), 2.0e7) for t in temperatures],
            "k-", lw=2, label=r"quasi-static $\mu(T,p_\mathrm{real}=20\,$MPa$)$")
    ax.plot(temperatures, [model.mu_dry(float(t)) for t in temperatures],
            "k--", lw=1.2, label=r"dry exponential law $\mu_0e^{-\beta(T-T_0)}$")
    ax.plot(temperatures, [-CC_SLOPE_K_PER_MPA * 0.0 + melting_temperature_c(2.0e7)
                           for _ in temperatures], ":", color="tab:red", lw=1.2,
            label=r"$T_m(p_\mathrm{real})$ = %.2f degC" % melting_temperature_c(2.0e7))

    calibration = [row for row in CALIBRATION_ANCHORS]
    ax.plot([row.temperature_c for row in calibration],
            [row.mu for row in calibration], "o", ms=4, color="tab:blue",
            label="calibration anchors (Barnes / Timco / Oksanen)")
    holdout = report["criteria"]["C4_validation_against_literature"]["rows"]
    ax.plot([row["temperature_c"] for row in holdout],
            [row["mu_literature"] for row in holdout], "s", ms=6, mfc="none",
            color="tab:green", label="holdout literature (Riska / Poirier)")
    ax.plot([row["temperature_c"] for row in holdout],
            [row["mu_predicted"] for row in holdout], "+", ms=9,
            color="tab:red", label="predicted by the coupled model")
    ax.set_xlabel("contact / bulk temperature [degC]")
    ax.set_ylabel(r"friction coefficient $\mu$ [-]")
    ax.set_title("Ice-steel friction: temperature law and melting point")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fig_sliding(report: dict, path: Path) -> None:
    """The time march: mu(t) with and without the phase-change feedback."""
    coupled = report["sliding_case"]["coupled"]
    control = report["sliding_case"]["constant_friction_control"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    ax1.plot(coupled["time_s"], coupled["mu_history"], "-", color="tab:red",
             label="coupled (phase change feeds back)")
    ax1.plot(control["time_s"], control["mu_history"], "--", color="tab:gray",
             label="control (melt ignored by friction)")
    ax1.set_ylabel(r"$\mu$ [-]")
    ax1.set_title("Sliding ice slab: self-consistent coefficient of friction")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(coupled["time_s"], coupled["interface_temperature_c_history"], "-",
             color="tab:blue", label=r"$T_\mathrm{contact}$ (coupled)")
    ax2.plot(coupled["time_s"], [coupled["melting_temperature_c"]] * len(coupled["time_s"]),
             ":", color="tab:red",
             label=r"$T_m(p_\mathrm{real})$ = %.2f degC" % coupled["melting_temperature_c"])
    ax2.set_ylabel("temperature [degC]")
    ax2.set_xlabel("time [s]")
    ax2b = ax2.twinx()
    ax2b.plot(coupled["time_s"], coupled["melt_fraction_history"], "-",
              color="tab:green", lw=1.0, alpha=0.8, label="melt fraction $s$")
    ax2b.set_ylabel("melt fraction [-]")
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, fontsize=8, loc="center right")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fig_pressure_melting(report: dict, path: Path) -> None:
    """Where Clausius-Clapeyron switches the friction off."""
    grid = report["pressure_melting"]["grid"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    hardnesses = sorted({row["hardness_mpa"] for row in grid})
    for hardness in hardnesses:
        rows = [row for row in grid if row["hardness_mpa"] == hardness]
        rows.sort(key=lambda row: row["temperature_c"])
        ax.plot([row["temperature_c"] for row in rows],
                [row["mu"] for row in rows], "o-", ms=4,
                label=f"$H$ = {hardness:g} MPa ($p_\\mathrm{{real}}$ = {rows[0]['real_pressure_mpa']:g} MPa)")
    temperatures = np.linspace(-20.0, -0.2, 200)
    ax2 = ax.twinx()
    ax2.plot(temperatures,
             [pressure_melting_threshold_pa(float(t)) / 1.0e6 for t in temperatures],
             "k:", lw=1.2, label="pressure needed to melt at T")
    ax2.set_yscale("log")
    ax2.set_ylabel("melting pressure $|T|/|dT_m/dp|$ [MPa] (log)")
    ax.set_xlabel("bulk ice temperature [degC]")
    ax.set_ylabel(r"$\mu$ [-]")
    ax.set_title("Pressure melting: the Clausius-Clapeyron threshold")
    ax.grid(alpha=0.3)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fig_convergence(report: dict, path: Path) -> None:
    """Residual of the fixed point, iteration by iteration."""
    rows = report["criteria"]["C3_fixed_point_convergence"]["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for row in rows:
        history = row["residual_history"]
        ax.semilogy(range(1, len(history) + 1), history, "o-", ms=5,
                    label=(f"v={row['v']:g} m/s, p={row['pressure_mpa']:g} MPa, "
                           f"T={row['temperature_c']:g} degC ({row['iterations']} it)"))
    ax.axhline(1.0e-2, color="k", ls="--", lw=1.0, label="criterion 1 %")
    ax.set_xlabel("Picard iteration")
    ax.set_ylabel(r"$|\Delta\mu|/\mu$")
    ax.set_title("Self-consistent loop: convergence of the friction coefficient")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fig_conductor(report: dict, path: Path) -> None:
    """The 1D conductor against the Blok semi-infinite reference."""
    early = report["conductor"]["early_time"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    depths = [row["dx_m"] for row in early]
    ax1.plot([row["time_s"] * 1e3 for row in early],
             [row["numerical_rise_k"] for row in early], "o-", label="finite volume")
    ax1.plot([row["time_s"] * 1e3 for row in early],
             [row["blok_rise_k"] for row in early], "s--", label="Blok semi-infinite")
    ax1.set_xlabel("time [ms]")
    ax1.set_ylabel(r"$T_\mathrm{contact}-T_\mathrm{bulk}$ [K]")
    ax1.set_title("Flash temperature at $q = %.0f$ W/m$^2$"
                  % report["conductor"]["flux_w_m2"])
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    ax2.semilogy([row["dt_s"] for row in early],
                 [abs(row["rel_error"]) for row in early], "o-")
    ax2.set_xlabel("time step [s]")
    ax2.set_ylabel("relative deviation from Blok")
    ax2.set_title("First-order convergence in $\\Delta t$")
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fig_sensitivity(report: dict, path: Path) -> None:
    """What the un-measured parameters cost, in the wet regime."""
    rows = report["sensitivity"]["rows"]
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["parameter"], []).append(row)
    fig, axes = plt.subplots(1, len(groups), figsize=(11.0, 3.8))
    for ax, (name, entries) in zip(axes, groups.items()):
        entries = sorted(entries, key=lambda row: row["value"])
        ax.plot([entry["value"] for entry in entries],
                [entry["wet_case_mu"] for entry in entries], "o-")
        ax.set_xscale("log")
        ax.set_xlabel(name.replace("_", " "))
        ax.set_ylabel(r"$\mu$ of the wet case")
        ax.set_title(name.replace("_", " "), fontsize=9)
        ax.grid(alpha=0.3, which="both")
    wet = report["sensitivity"]["wet_case"]
    fig.suptitle("Lubricated coefficient vs the un-fitted parameters "
                 "(wet case: p=%.1f MPa, T=%.1f degC, v=%.2f m/s)"
                 % (wet["nominal_pressure_mpa"], wet["temperature_c"], wet["velocity_m_s"]),
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = write_evidence(out / "acceptance.json")
    print(f"acceptance.json written, digest={report['C6_provenance']['digest']}")

    _fig_temperature(report, out / "fig_mu_vs_temperature.png")
    _fig_sliding(report, out / "fig_sliding_case.png")
    _fig_pressure_melting(report, out / "fig_pressure_melting.png")
    _fig_convergence(report, out / "fig_convergence.png")
    _fig_conductor(report, out / "fig_conductor_blok.png")
    _fig_sensitivity(report, out / "fig_sensitivity.png")

    summary = {
        "digest": report["C6_provenance"]["digest"],
        "criterion_pass": {name: block["pass"]
                           for name, block in report["criteria"].items()},
        "c3_max_iterations": report["criteria"]["C3_fixed_point_convergence"]["max_iterations"],
        "c4_within_gate": report["criteria"]["C4_validation_against_literature"]["stats"]["n_within_gate"],
        "c4_baseline_within_gate": report["criteria"]["C4_validation_against_literature"]["baseline_stats"]["n_within_gate"],
        "c5_max_rel_error": report["criteria"]["C5_friction_heat_identity"]["max_rel_error"],
        "sliding_mu_initial": report["sliding_case"]["coupled"]["mu_initial"],
        "sliding_mu_final": report["sliding_case"]["coupled"]["mu_final"],
        "sliding_energy_closure": report["sliding_case"]["coupled"]["energy_closure_error"],
        "figures": sorted(p.name for p in out.glob("fig_*.png")),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()