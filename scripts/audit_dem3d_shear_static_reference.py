"""Audit the DEM3D direct-shear specimen against a rate-free static reference.

Writes docs/dem3d-shear-static-reference/{report.json,summary.md,figures/*.png}.
Run: PYTHONPATH=src python scripts/audit_dem3d_shear_static_reference.py
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.shear_specimen import (
    apparent_shear_modulus_from_history,
    run_direct_shear,
    shear_modulus_closed_form,
)
from tensordem.shear_static import affine_effective_area, bond_census, static_direct_shear

OUT = Path("docs/dem3d-shear-static-reference")
SPECIMENS = [(9, 5, 3), (9, 5, 5), (9, 5, 9), (17, 13, 3), (21, 21, 3)]
RATES = (0.1, 0.05)


def relax_to_static(config, gamma, *, drag, increments=40, settle=4000):
    """Displacement-controlled relaxation: clamp top u_x, let the body settle."""
    cfg = replace(config, fix_bottom=True, fix_x_edges=False, drag=drag,
                  breaking_strain=1.0e9, shear_breaking_strain=1.0e9)
    sim = IceDEM3D(cfg)
    height = (cfg.nz - 1) * 2 * cfg.radius
    z = sim.initial_positions[:, 2]
    top = torch.isclose(z, z.max())
    trace = []

    def clamp(distance):
        sim.positions[top, 0] = sim.initial_positions[top, 0] + distance
        sim.velocities[top, 0] = 0.0

    for k in range(1, increments + 1):
        clamp(gamma * height * k / increments)
        for _ in range(10):
            sim.step()
            clamp(gamma * height * k / increments)
    for step in range(settle):
        sim.step()
        clamp(gamma * height)
        if (step + 1) % 500 == 0:
            forces, _ = sim.forces(update_fracture=False)
            reaction = abs(float(forces[sim.fixed].sum(dim=0)[0]))
            kinetic = 0.5 * cfg.mass * float(sim.velocities.square().sum())
            trace.append({"step": step + 1, "reaction_N": reaction, "kinetic_energy_J": kinetic})
    area_nominal = cfg.nx * 2 * cfg.radius * cfg.ny * 2 * cfg.radius
    area_effective = affine_effective_area(cfg)
    c44 = shear_modulus_closed_form(cfg)
    return {
        "drag_Ns_per_m": drag,
        "area_nominal_m2": area_nominal,
        "area_effective_m2": area_effective,
        "C44_Pa": c44,
        "gamma": gamma,
        "trace": trace,
        "G_nominal_over_C44": abs(trace[-1]["reaction_N"]) / area_nominal / gamma / c44,
        "G_effective_over_C44": abs(trace[-1]["reaction_N"]) / area_effective / gamma / c44,
        "kinetic_energy_J": trace[-1]["kinetic_energy_J"],
        "broken_bonds": int(sim.broken_bonds),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    c44_default = shear_modulus_closed_form(DEM3DConfig())

    # 1. bond census / cross-section definition bias
    census_rows = []
    for nx, ny, nz in SPECIMENS:
        cfg = replace(DEM3DConfig(), nx=nx, ny=ny, nz=nz)
        row = bond_census(cfg)
        row.update({"nx": nx, "ny": ny, "nz": nz,
                    "area_ratio": row["weight"] / (7.0 / 3.0) / row["nominal_area_m2"]})
        census_rows.append(row)

    # 2. static (rate-free) equilibrium for each specimen
    static_rows = []
    profiles = {}
    for nx, ny, nz in SPECIMENS:
        res = static_direct_shear(replace(DEM3DConfig(), nx=nx, ny=ny, nz=nz), gamma=1.0e-3)
        profile = res.pop("profile")
        if nz == 9:
            profiles["static"] = profile
        res.update({"nx": nx, "ny": ny, "nz": nz, "height_over_a": (nz - 1)})
        static_rows.append(res)

    # 2b. linearity in gamma
    linearity = []
    for gamma in (1.0e-5, 1.0e-4, 1.0e-3, 5.0e-3):
        res = static_direct_shear(DEM3DConfig(), gamma=gamma)
        linearity.append({"gamma": gamma,
                          "G_effective_over_C44": res["G_effective_over_C44"],
                          "G_nominal_over_C44": res["G_nominal_over_C44"]})

    # 3. legacy driven protocol: rate / transient scatter
    driven_rows = []
    curves = {}
    for rate in RATES:
        run = run_direct_shear(DEM3DConfig(), velocity=rate, gamma_max=0.025)
        fit = apparent_shear_modulus_from_history(run["history"], 0.008, 0.022)
        area_nominal = run["protocol"]["specimen_area_m2"]
        driven_rows.append({
            "velocity_m_per_s": rate,
            "steps": run["protocol"]["steps"],
            "G_over_C44_nominal": fit["apparent_shear_modulus_Pa"] / run["C44_Pa"],
            "G_over_C44_effective": fit["apparent_shear_modulus_Pa"]
            / (10.0 / 7.0 * run["protocol"]["specimen_area_m2"] / 0.7873) / run["C44_Pa"]
            if False else fit["apparent_shear_modulus_Pa"] * area_nominal
            / (area_nominal * 0.7873) / run["C44_Pa"],
            "fit_relative_residual": fit["relative_residual"],
            "first_failure_gamma": (run["first_failure"] or {}).get("gamma"),
        })
        curves[rate] = [(h["gamma"], h["tau_Pa"]) for h in run["history"]]

    # 4. dynamics verification: relaxed run must reproduce the static reaction
    relax_rows = []
    for drag in (50.0, 300.0):
        res = relax_to_static(DEM3DConfig(), 1.0e-3, drag=drag)
        static_ref = static_direct_shear(DEM3DConfig(), gamma=1.0e-3)
        trace = res.pop("trace")
        res["static_G_effective_over_C44"] = static_ref["G_effective_over_C44"]
        res["deviation_vs_static"] = (res["G_effective_over_C44"]
                                      / static_ref["G_effective_over_C44"] - 1.0)
        res["relative_trace_drift"] = abs(
            trace[-1]["reaction_N"] / trace[len(trace) // 2]["reaction_N"] - 1.0)
        relax_rows.append(res)
        if drag == 50.0:
            profiles["relaxed"] = {
                "trace": trace, "drag": drag,
                "reaction_static_N": static_ref["reaction_N"],
            }

    # ---- figures ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    z_rel = [row["z_rel"] for row in profiles["static"]]
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="affine  $u_x=\\gamma z$")
    ax.plot(z_rel, [row["mean_ux_over_gamma_h"] for row in profiles["static"]],
            "o-", label="static equilibrium (mean)")
    ax.fill_between(z_rel,
                    [row["mean_ux_over_gamma_h"] - row["spread_ux_over_gamma_h"] / 2
                     for row in profiles["static"]],
                    [row["mean_ux_over_gamma_h"] + row["spread_ux_over_gamma_h"] / 2
                     for row in profiles["static"]],
                    alpha=0.25, label="static (layer spread)")
    ax.set_xlabel("$z/h$")
    ax.set_ylabel("$u_x/(\\gamma h)$")
    ax.set_title("9x5x9 specimen: non-affine shear profile")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels = [f"{r['nx']}x{r['ny']}x{r['nz']}" for r in static_rows]
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], [r["G_nominal_over_C44"] for r in static_rows],
           width=0.4, label="nominal area")
    ax.bar([i + 0.2 for i in x], [r["G_effective_over_C44"] for r in static_rows],
           width=0.4, label="bond-census area")
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="$C_{44}=7k/3a$ (affine)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, fontsize=7)
    ax.set_ylabel("$G_{app}/C_{44}$")
    ax.set_title("static shear response vs lattice constant")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    for rate, curve in curves.items():
        ax.plot([g for g, _ in curve], [t for _, t in curve], lw=1.2,
                label=f"driven ramp v={rate} m/s")
    stat = static_direct_shear(DEM3DConfig(), gamma=1.0e-3)
    ax.plot([0, 0.025], [0, stat["tau_effective_Pa"] * 0.025 / 1.0e-3],
            "k--", lw=1.2, label="static limit (census area)")
    ax.set_xlabel("$\\gamma$")
    ax.set_ylabel("$\\tau$ (Pa), census area")
    ax.set_title("driven ramps are not quasi-static")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "figures" / "shear_static_reference.png", dpi=160)
    plt.close(fig)

    mean_case = static_rows[0]
    report = {
        "schema": "tensordem-dem3d-shear-static-reference-v1",
        "C44_default_config_Pa": c44_default,
        "bond_census": census_rows,
        "static_equilibrium": static_rows,
        "linearity_in_gamma": linearity,
        "driven_protocol": driven_rows,
        "relaxation_verification": relax_rows,
        "notes": [
            "G_effective uses the bond-census area A_eff = sum l^2 nx^2 nz^2 / (7a^2/3), "
            "which makes the affine reaction equal C44*gamma exactly.",
            "G_nominal uses nx*a*ny*a, the convention of the legacy driven test.",
        ],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))

    lines = [
        "# DEM3D direct-shear specimen: static reference and rate audit",
        "",
        "## 1. Cross-section definition bias (affine, exact)",
        "",
        "| specimen | bonds on fixed layer | $A_{eff}/A_{nom}$ |",
        "| --- | ---: | ---: |",
    ]
    for row in census_rows:
        lines.append(f"| {row['nx']}x{row['ny']}x{row['nz']} | {int(row['bond_count'])} "
                     f"| {row['area_ratio']:.4f} |")
    lines += [
        "",
        "## 2. Static equilibrium (rate-free limit, sparse solve)",
        "",
        "| specimen | $h/a$ | $G_{nom}/C_{44}$ | $G_{eff}/C_{44}$ | reaction check | free residual |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in static_rows:
        lines.append(f"| {row['nx']}x{row['ny']}x{row['nz']} | {row['height_over_a']} "
                     f"| {row['G_nominal_over_C44']:.4f} | {row['G_effective_over_C44']:.4f} "
                     f"| {row['reaction_relative_error']:.1e} "
                     f"| {row['free_residual_over_bond_force']:.1e} |")
    lines += [
        "",
        f"Default 9x5x3 specimen: $G_{{eff}}/C_{{44}} = {mean_case['G_effective_over_C44']:.4f}$,",
        f"$G_{{nom}}/C_{{44}} = {mean_case['G_nominal_over_C44']:.4f}$.",
        "",
        "## 3. Linearity in $\\gamma$",
        "",
        "| $\\gamma$ | $G_{eff}/C_{44}$ |",
        "| --- | ---: |",
    ]
    for row in linearity:
        lines.append(f"| {row['gamma']:.0e} | {row['G_effective_over_C44']:.4f} |")
    lines += [
        "",
        "## 4. Legacy driven ramp (window $0.008 \\le \\gamma \\le 0.022$)",
        "",
        "| $v$ (m/s) | steps | $G_{nom}/C_{44}$ | fit residual |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in driven_rows:
        lines.append(f"| {row['velocity_m_per_s']} | {row['steps']} "
                     f"| {row['G_over_C44_nominal']:.4f} | {row['fit_relative_residual']:.1e} |")
    lines += [
        "",
        "## 5. Relaxation verification against the static solution",
        "",
        "| drag (N s/m) | $G_{eff}/C_{44}$ | static | deviation | trace drift | broken bonds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in relax_rows:
        lines.append(f"| {row['drag_Ns_per_m']} | {row['G_effective_over_C44']:.4f} "
                     f"| {row['static_G_effective_over_C44']:.4f} "
                     f"| {100 * row['deviation_vs_static']:+.2f} % "
                     f"| {row['relative_trace_drift']:.1e} | {row['broken_bonds']} |")
    lines += ["", "![static reference](figures/shear_static_reference.png)", ""]
    (OUT / "summary.md").write_text("\n".join(lines))

    print(json.dumps({
        "census": [(r["nx"], r["ny"], r["nz"], round(r["area_ratio"], 4)) for r in census_rows],
        "static": [(r["nx"], r["ny"], r["nz"], round(r["G_nominal_over_C44"], 4),
                    round(r["G_effective_over_C44"], 4), f"{r['free_residual_over_bond_force']:.1e}")
                   for r in static_rows],
        "driven": [(r["velocity_m_per_s"], round(r["G_over_C44_nominal"], 4)) for r in driven_rows],
        "relax": [(r["drag_Ns_per_m"], round(r["G_effective_over_C44"], 4),
                   round(100 * r["deviation_vs_static"], 2), r["broken_bonds"]) for r in relax_rows],
    }, indent=1))


if __name__ == "__main__":
    main()