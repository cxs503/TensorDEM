"""Calibrate and verify the DEM3D normal-contact coefficient of restitution.

Runs two experiments with the public engine API only:

1. **Damping sweep** -- bounce a bonded particle block on the rigid bottom
   platen for a range of dashpot coefficients at the engine's default time
   step, and compare the measured restitution with the closed-form
   Kelvin-Voigt value (:mod:`tensordem.restitution`).
2. **Time-step convergence** -- repeat the most heavily damped case at dt/4 and
   dt/8 to separate model error from explicit-integration error.

Outputs (CSV, JSON, matplotlib figure and a generated markdown table) land in
``outputs/dem3d_restitution`` unless ``--out-dir`` says otherwise.  Nothing in
this script is used by the engine itself; it is a verification/calibration
harness.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from tensordem.dem3d import DEM3DConfig
from tensordem.restitution import (
    clamped_contact_restitution,
    contact_restitution,
    damping_for_clamped_restitution,
    damping_ratio,
    measure_platen_restitution,
    restitution_from_damping,
    restitution_from_damping_clamped,
)

SWEEP = (0.0, 1.0, 2.0, 5.0, 10.0)
CONVERGENCE_DAMPING = 10.0
CONVERGENCE_FACTORS = (1, 4, 8)


def _sweep_rows(radius: float, density: float, impact_speed: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for damping in SWEEP:
        measurement = measure_platen_restitution(
            damping=damping, radius=radius, density=density, impact_speed=impact_speed
        )
        row = measurement.as_dict()
        row["study"] = "damping_sweep"
        row["dt_factor"] = 1
        rows.append(row)
        print(
            f"[sweep] c={damping:5.2f} N.s/m  dt={row['dt_s']:.3e} s  steps={row['steps']:5d}  "
            f"e_sim={row['measured_restitution']:.5f}  e_unclamped={row['analytic_restitution']:.5f} "
            f"({100.0 * row['relative_error']:.2f}%)  e_clamped={row['analytic_restitution_clamped']:.5f} "
            f"({100.0 * row['relative_error_clamped']:.2f}%)",
            flush=True,
        )
    return rows


def _convergence_rows(
    radius: float, density: float, impact_speed: float, base_dt: float
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for factor in CONVERGENCE_FACTORS:
        measurement = measure_platen_restitution(
            damping=CONVERGENCE_DAMPING,
            radius=radius,
            density=density,
            impact_speed=impact_speed,
            dt=None if factor == 1 else base_dt / factor,
        )
        row = measurement.as_dict()
        row["study"] = "timestep_convergence"
        row["dt_factor"] = factor
        rows.append(row)
        print(
            f"[conv ] c={CONVERGENCE_DAMPING:5.2f} dt/1e{factor if factor > 1 else 0}="
            f"{row['dt_s']:.3e} s steps={row['steps']:6d} e_sim={row['measured_restitution']:.5f} "
            f"e_unclamped={row['analytic_restitution']:.5f} ({100.0 * row['relative_error']:.2f}%) "
            f"e_clamped={row['analytic_restitution_clamped']:.5f} "
            f"({100.0 * row['relative_error_clamped']:.2f}%)",
            flush=True,
        )
    return rows


def _calibration_rows() -> list[dict[str, float]]:
    """Damping coefficients that realise literature restitution targets."""
    config = DEM3DConfig()
    rows: list[dict[str, float]] = []
    for target in (0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2):
        damping = damping_for_clamped_restitution(
            config.contact_stiffness, target, config.mass
        )
        rows.append(
            {
                "target_restitution": target,
                "contact_damping_N_s_per_m": damping,
                "damping_ratio": damping_ratio(config.contact_stiffness, damping, config.mass),
                "roundtrip_restitution": restitution_from_damping_clamped(
                    config.contact_stiffness, damping, config.mass
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = [
        "study", "dt_factor", "stiffness_N_per_m", "damping_N_s_per_m", "damping_ratio",
        "particle_mass_kg", "impact_speed_m_per_s", "particle_count", "dt_s", "steps",
        "measured_restitution", "analytic_restitution", "relative_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _markdown_tables(rows: list[dict[str, float]], calibration: list[dict[str, float]]) -> str:
    lines = ["### Damping sweep (default engine time step)", ""]
    lines.append(
        "| c [N·s/m] | ζ | dt [s] | steps | e measured | e unclamped | error | e clamped | error |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        if row["study"] != "damping_sweep":
            continue
        lines.append(
            f"| {row['damping_N_s_per_m']:.2f} | {row['damping_ratio']:.4f} | "
            f"{row['dt_s']:.4e} | {int(row['steps'])} | {row['measured_restitution']:.5f} | "
            f"{row['analytic_restitution']:.5f} | {100.0 * row['relative_error']:.2f}% | "
            f"{row['analytic_restitution_clamped']:.5f} | "
            f"{100.0 * row['relative_error_clamped']:.2f}% |"
        )
    lines += ["", f"### Time-step convergence (c = {CONVERGENCE_DAMPING:.1f} N·s/m)", ""]
    lines.append("| dt [s] | steps | e measured | e unclamped | error | e clamped | error |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        if row["study"] != "timestep_convergence":
            continue
        lines.append(
            f"| {row['dt_s']:.4e} | {int(row['steps'])} | {row['measured_restitution']:.5f} | "
            f"{row['analytic_restitution']:.5f} | {100.0 * row['relative_error']:.2f}% | "
            f"{row['analytic_restitution_clamped']:.5f} | "
            f"{100.0 * row['relative_error_clamped']:.2f}% |"
        )
    lines += ["", "### Restitution-to-damping calibration (default r = 25 mm, ρ = 917 kg/m³)", ""]
    lines.append("| target e | c [N·s/m] | ζ | round-trip e_clamped |")
    lines.append("|---:|---:|---:|---:|")
    for row in calibration:
        lines.append(
            f"| {row['target_restitution']:.2f} | {row['contact_damping_N_s_per_m']:.3f} | "
            f"{row['damping_ratio']:.4f} | {row['roundtrip_restitution']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def _figure(path: Path, rows: list[dict[str, float]], radius: float, density: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = [row for row in rows if row["study"] == "damping_sweep"]
    conv = [row for row in rows if row["study"] == "timestep_convergence"]
    mass = sweep[0]["particle_mass_kg"]

    zeta_curve = [i * 0.6 / 200.0 for i in range(201)]
    e_curve = [restitution_from_damping(5_000.0, z * 2.0 * math.sqrt(5_000.0 * mass), mass)
               for z in zeta_curve]
    e_curve_clamped = [
        restitution_from_damping_clamped(5_000.0, z * 2.0 * math.sqrt(5_000.0 * mass), mass)
        for z in zeta_curve
    ]
    zeta_sim = [row["damping_ratio"] for row in sweep]
    e_sim = [row["measured_restitution"] for row in sweep]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    ax.plot(zeta_curve, e_curve, "--", color="#808080", lw=1.5,
            label="unclamped Kelvin–Voigt $e^{-\\zeta\\pi/\\sqrt{1-\\zeta^2}}$")
    ax.plot(zeta_curve, e_curve_clamped, "-", color="#1f4e79", lw=2,
            label="clamped (unilateral) closed form")
    ax.plot(zeta_sim, e_sim, "o", ms=7, color="#c00000", label="DEM3D platen bounce")
    for row in sweep:
        ax.annotate(f"c={row['damping_N_s_per_m']:.0f}", (row["damping_ratio"], row["measured_restitution"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("damping ratio $\\zeta=c/(2\\sqrt{km})$")
    ax.set_ylabel("coefficient of restitution $e$")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(a) restitution vs damping at default $\\Delta t$")

    ax = axes[1]
    ax.axhline(conv[0]["analytic_restitution"], color="#808080", ls="--", lw=1.4,
               label="unclamped closed form")
    ax.axhline(conv[0]["analytic_restitution_clamped"], color="#1f4e79", ls="-", lw=1.6,
               label="clamped closed form")
    ax.plot([row["dt_s"] for row in conv], [row["measured_restitution"] for row in conv],
            "s-", color="#c00000", label="measured")
    for row in conv:
        ax.annotate(f"{100.0 * row['relative_error_clamped']:.2f}%",
                    (row["dt_s"], row["measured_restitution"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("time step $\\Delta t$ [s]")
    ax.set_ylabel("coefficient of restitution $e$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"(b) $\\Delta t$ convergence, c={CONVERGENCE_DAMPING:.0f} N·s/m")

    fig.suptitle(
        f"DEM3D normal-contact restitution verification "
        f"(r={1000 * radius:.0f} mm, $\\rho$={density:.0f} kg/m³, bonded 2×2×2 block)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/dem3d_restitution"))
    parser.add_argument("--radius", type=float, default=0.025)
    parser.add_argument("--density", type=float, default=917.0)
    parser.add_argument("--impact-speed", type=float, default=0.5)
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = DEM3DConfig()
    print(
        "default DEM3D contact restitution: unclamped",
        round(contact_restitution(config), 5),
        "clamped",
        round(clamped_contact_restitution(config), 5),
        flush=True,
    )
    rows = _sweep_rows(args.radius, args.density, args.impact_speed)
    base_dt = rows[0]["dt_s"]
    rows += _convergence_rows(args.radius, args.density, args.impact_speed, base_dt)
    calibration = _calibration_rows()

    _write_csv(args.out_dir / "dem3d_restitution_calibration.csv", rows)
    payload = {
        "model": "linear spring-dashpot (Kelvin-Voigt), unilateral",
        "analytic_restitution_unclamped": "e = exp(-zeta*pi/sqrt(1-zeta^2)), zeta = c/(2*sqrt(k*m))",
        "analytic_restitution_clamped": (
            "e = exp(-(zeta/sqrt(1-zeta^2))*phi), "
            "phi = atan2(2*zeta*sqrt(1-zeta^2), -(1-2*zeta^2))"
        ),
        "default_contact_restitution_unclamped": contact_restitution(DEM3DConfig()),
        "default_contact_restitution_clamped": clamped_contact_restitution(DEM3DConfig()),
        "default_config": {
            "radius_m": DEM3DConfig().radius,
            "density_kg_per_m3": DEM3DConfig().density,
            "particle_mass_kg": DEM3DConfig().mass,
            "contact_stiffness_N_per_m": DEM3DConfig().contact_stiffness,
            "contact_damping_N_s_per_m": DEM3DConfig().contact_damping,
        },
        "measurements": rows,
        "calibration": calibration,
    }
    (args.out_dir / "dem3d_restitution_calibration.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (args.out_dir / "dem3d_restitution_calibration_tables.md").write_text(
        _markdown_tables(rows, calibration), encoding="utf-8"
    )
    _figure(args.out_dir / "dem3d_restitution_calibration.png", rows, args.radius, args.density)
    print("wrote", args.out_dir, flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())