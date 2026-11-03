"""Audit of the Rayleigh-wave step estimate and the measured ``dt_crit``.

Produces, into ``docs/dem3d-rayleigh-timestep/``:

* ``rayleigh_dt_audit.json`` -- every number used by the page, with a SHA-256
  digest of the report body;
* ``generated_tables.md`` -- the markdown tables quoted by the page;
* ``fig1_amplification.png``, ``fig2_rayleigh_ratio.png``, ``fig3_dt_limits.png``.

The script also rewrites the block between the ``BEGIN GENERATED`` /
``END GENERATED`` markers of ``docs/dem3d_rayleigh_timestep_audit.md`` so the
tables in the page are machine-generated and never transcribed by hand.

Run:

    python scripts/audit_dem3d_rayleigh_timestep.py            # full audit
    python scripts/audit_dem3d_rayleigh_timestep.py --quick    # short threshold bisection
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch

torch.set_num_threads(1)

from tensordem.dem3d import DEM3DConfig, IceDEM3D  # noqa: E402
from tensordem.rayleigh_timestep import (  # noqa: E402
    RAYLEIGH_RATIO_LITERATURE,
    divergence_threshold,
    energy_amplification,
    energy_amplification_closed_form,
    exact_bond_omega_max,
    rayleigh_dt_comparison,
    rayleigh_speed_ratio,
    rayleigh_speed_ratio_fit,
)
from tensordem.stability import stability_report  # noqa: E402

OUT = ROOT / "docs" / "dem3d-rayleigh-timestep"
PAGE = ROOT / "docs" / "dem3d_rayleigh_timestep_audit.md"
BEGIN = "<!-- BEGIN GENERATED -->"
END = "<!-- END GENERATED -->"

RADIUS = 0.025
DENSITY = 917.0
BOND_STIFFNESS = 2000.0

CONFIGS = [
    ("box4_k2000", dict(nx=4, ny=4, nz=4, bond_stiffness=BOND_STIFFNESS)),
    ("box9x5x3_k2000", dict(nx=9, ny=5, nz=3, bond_stiffness=BOND_STIFFNESS)),
    ("box6_k20000", dict(nx=6, ny=6, nz=6, bond_stiffness=20_000.0)),
]


def model_factory(**overrides):
    params = dict(
        nx=4, ny=4, nz=4, radius=RADIUS, density=DENSITY,
        bond_stiffness=BOND_STIFFNESS, contact_damping=0.0, drag=0.0,
        tool_radius=1e-6, tool_gap=10.0, tool_speed=0.5, contact_model="linear",
    )
    params.update(overrides)

    def factory() -> IceDEM3D:
        return IceDEM3D(DEM3DConfig(**params))

    return factory


def _fmt(value, digits=6):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "inf" if value > 0 else "-inf"
        if value != 0.0 and (abs(value) >= 1e5 or abs(value) < 1e-4):
            return f"{value:.{max(digits - 3, 2)}e}"
        return f"{value:.{digits}g}"
    return str(value)


def md_table(header, rows, align=None):
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(align or ["---"] * len(header)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="short threshold bisection (fewer steps and iterations)")
    args = parser.parse_args()
    n_steps = 200 if args.quick else 600
    iterations = 10 if args.quick else 18

    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "script": Path(__file__).name,
        "quick": args.quick,
        "radius_m": RADIUS,
        "density_kg_per_m3": DENSITY,
        "bond_stiffness_N_per_m": BOND_STIFFNESS,
        "threshold_settings": {"n_steps": n_steps, "iterations": iterations},
    }

    # 1. Rayleigh secular equation vs literature -----------------------------
    anchors = []
    for nu, reference in RAYLEIGH_RATIO_LITERATURE.items():
        value = rayleigh_speed_ratio(nu)
        anchors.append({
            "poisson": nu,
            "ratio_measured": value,
            "ratio_literature": reference,
            "relative_error": abs(value - reference) / reference,
        })
    report["rayleigh_anchors"] = anchors

    fit_scan = []
    worst = {"relative_deviation": 0.0}
    for step in range(0, 46):
        nu = 0.45 * step / 45.0
        exact = rayleigh_speed_ratio(nu)
        fit = rayleigh_speed_ratio_fit(nu)
        deviation = abs(fit - exact) / exact
        fit_scan.append({"poisson": nu, "ratio_exact": exact, "ratio_fit": fit,
                         "relative_deviation": deviation})
        if deviation > worst["relative_deviation"]:
            worst = fit_scan[-1]
    report["rayleigh_fit_scan"] = fit_scan
    report["rayleigh_fit_worst"] = worst

    # 2. Per-configuration step limits ---------------------------------------
    configs, simulations = [], []
    for name, overrides in CONFIGS:
        factory = model_factory(**overrides)
        model = factory()
        comparison = rayleigh_dt_comparison(model)
        exact = exact_bond_omega_max(model)
        report_stability = stability_report(model)
        configs.append({
            "name": name,
            "n_particles": int(len(model.positions)),
            "n_free_particles": int((~model.fixed).sum()),
            "n_alive_bonds": int(model.alive.sum()),
            "bond_stiffness": overrides["bond_stiffness"],
            "omega_max_exact_rad_s": comparison["omega_max_exact_rad_s"],
            "omega_row_sum_bound_rad_s": report_stability["omega_upper_rad_s"],
            "row_sum_bound_over_exact": (report_stability["omega_upper_rad_s"]
                                         / comparison["omega_max_exact_rad_s"]),
            "dt_crit_exact_s": comparison["dt_crit_exact_s"],
            "dt_rayleigh_s": comparison["dt_rayleigh_s"],
            "dt_rayleigh_over_dt_crit": comparison["dt_rayleigh_over_dt_crit"],
            "engine_recommended_dt_s": comparison["engine_recommended_dt_s"],
            "engine_recommended_over_dt_crit": comparison["engine_recommended_over_dt_crit"],
            "c_s_m_per_s": comparison["rayleigh_c_s_m_per_s"],
            "c_R_m_per_s": comparison["rayleigh_c_R_m_per_s"],
            "c_p_m_per_s": comparison["rayleigh_c_p_m_per_s"],
            "poisson_vrh": comparison["rayleigh_poisson_vrh"],
            "G_vrh_Pa": comparison["rayleigh_G_vrh_Pa"],
            "lattice_density_kg_per_m3": comparison["rayleigh_lattice_density_kg_per_m3"],
            "stability_report_is_stable": report_stability["is_stable"],
            "stability_report_margin_factor": report_stability["margin_factor"],
            "n_free_dof": exact["n_free_dof"],
        })
        simulations.append((name, factory))
    report["configs"] = configs

    # 3. Energy amplification of the top mode (config 1) ---------------------
    ratios = [0.1, 0.25, 0.5, 0.8, 0.9, 0.95, 1.0, 1.02, 1.05]
    amplification = []
    for ratio in ratios:
        trial = energy_amplification(simulations[0][1], dt_over_crit=ratio, n_steps=n_steps)
        closed = energy_amplification_closed_form(ratio)
        trial["closed_form"] = closed
        trial["closed_form_relative_error"] = (
            abs(float(trial["peak_over_initial"]) / closed - 1.0)
            if math.isfinite(closed) and math.isfinite(float(trial["peak_over_initial"]))
            else math.inf
        )
        amplification.append(trial)
    report["amplification"] = amplification
    bounded = [row for row in amplification if row["dt_over_crit"] < 1.0]
    report["amplification_closed_form_worst"] = max(
        row["closed_form_relative_error"] for row in bounded
    )
    report["accuracy_budget"] = {
        "eps_1pct_dt_over_crit": 0.01 / (2.0 * 1.01),
        "eps_3pct_dt_over_crit": 0.03 / (2.0 * 1.03),
    }

    # 4. Measured engine threshold ------------------------------------------
    thresholds = []
    for name, factory in simulations[:2]:
        measurement = divergence_threshold(
            factory, n_steps=n_steps, iterations=iterations
        )
        measurement["name"] = name
        thresholds.append(measurement)
    report["thresholds"] = thresholds

    body = json.dumps(report, sort_keys=True).encode()
    report["sha256_of_body"] = hashlib.sha256(body).hexdigest()
    (OUT / "rayleigh_dt_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    # 5. Markdown tables ----------------------------------------------------
    tables = []
    tables.append("### T1. Rayleigh secular equation vs literature\n\n" + md_table(
        ["nu", "c_R/c_s (secular equation)", "literature", "relative error"],
        [[row["poisson"], row["ratio_measured"], row["ratio_literature"],
          row["relative_error"]] for row in anchors],
        ["---:", "---:", "---:", "---:"],
    ))
    tables.append("### T2. Engineering fit `0.163 nu + 0.877` vs the exact equation\n\n" + md_table(
        ["nu", "c_R/c_s (secular)", "c_R/c_s (fit)", "relative deviation"],
        [[row["poisson"], row["ratio_exact"], row["ratio_fit"], row["relative_deviation"]]
         for row in fit_scan[::5]] + [[worst["poisson"], worst["ratio_exact"],
                                       worst["ratio_fit"], worst["relative_deviation"]]],
        ["---:", "---:", "---:", "---:"],
    ))
    tables.append("### T3. Step limits of the engine configurations\n\n" + md_table(
        ["config", "N", "free DOF", "omega_max (rad/s)", "dt_crit (s)",
         "dt_R = 2r/c_R (s)", "dt_R/dt_crit", "recommended dt (s)",
         "recommended/dt_crit", "row-sum bound / exact"],
        [[row["name"], row["n_particles"], row["n_free_dof"], row["omega_max_exact_rad_s"],
          row["dt_crit_exact_s"], row["dt_rayleigh_s"], row["dt_rayleigh_over_dt_crit"],
          row["engine_recommended_dt_s"], row["engine_recommended_over_dt_crit"],
          row["row_sum_bound_over_exact"]] for row in configs],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
    ))
    tables.append("### T4. Wave speeds and isotropic moduli per configuration\n\n" + md_table(
        ["config", "rho_lattice (kg/m^3)", "G_VRH (Pa)", "nu_VRH", "c_s (m/s)",
         "c_R (m/s)", "c_p (m/s)"],
        [[row["name"], row["lattice_density_kg_per_m3"], row["G_vrh_Pa"], row["poisson_vrh"],
          row["c_s_m_per_s"], row["c_R_m_per_s"], row["c_p_m_per_s"]] for row in configs],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:"]))
    tables.append("### T5. Top-mode energy amplification in the engine (config `box4_k2000`)\n\n" + md_table(
        ["dt/dt_crit", "d = 2 dt/dt_crit", "peak E / E0 (engine)",
         "1 + d/(2-d) (closed form)", "closed-form rel. error", "envelope growth",
         "flagged unstable"],
        [[row["dt_over_crit"], 2.0 * row["dt_over_crit"], row["peak_over_initial"],
          row["closed_form"], row["closed_form_relative_error"], row["envelope_growth"],
          row["diverged"]] for row in amplification],
        ["---:", "---:", "---:", "---:", "---:", "---:", ":---:"],
    ))
    tables.append("### T6. Measured instability threshold of the real engine\n\n" + md_table(
        ["config", "predicted dt/dt_crit", "measured dt/dt_crit", "relative error",
         "measured dt_crit (s)", "steps per trial", "bisection iterations"],
        [[row["name"], row["predicted_threshold_ratio"], row["measured_threshold_ratio"],
          row["relative_error"], row["measured_threshold_dt_s"], row["n_steps"],
          row["iterations"]] for row in thresholds],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:"],
    ))
    tables.append(
        "### T7. Accuracy budget implied by the closed form\n\n"
        + md_table(["tolerated artificial amplification", "required dt/dt_crit"],
                   [["1 %", report["accuracy_budget"]["eps_1pct_dt_over_crit"]],
                    ["3 %", report["accuracy_budget"]["eps_3pct_dt_over_crit"]]],
                   ["---:", "---:"])
    )
    generated = "\n\n".join(tables) + "\n"
    (OUT / "generated_tables.md").write_text(generated)
    if PAGE.exists():
        text = PAGE.read_text()
        if BEGIN in text and END in text:
            head, rest = text.split(BEGIN, 1)
            _, tail = rest.split(END, 1)
            PAGE.write_text(head + BEGIN + "\n" + generated + END + tail)

    # 6. Figures ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    dt_ratio = [row["dt_over_crit"] for row in amplification]
    peak = [row["peak_over_initial"] for row in amplification]
    ax.semilogy(dt_ratio, peak, "o-", label="engine (top-mode run)")
    grid = [0.05 + 0.005 * i for i in range(0, 191)]
    ax.semilogy(grid, [energy_amplification_closed_form(d) for d in grid], "--",
                label=r"$1 + d/(2-d)$, $d = 2\,dt/dt_{crit}$")
    ax.axvline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel(r"$dt / dt_{crit}$")
    ax.set_ylabel(r"peak $E / E_0$")
    ax.set_title("Top-mode energy amplification vs the explicit step")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig1_amplification.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    nus = [row["poisson"] for row in fit_scan]
    ax.plot(nus, [row["ratio_exact"] for row in fit_scan], "-",
            label="Rayleigh secular equation")
    ax.plot(nus, [row["ratio_fit"] for row in fit_scan], "--",
            label=r"engineering fit $0.163\nu + 0.877$")
    anchors_nu = list(RAYLEIGH_RATIO_LITERATURE)
    ax.plot(anchors_nu, [RAYLEIGH_RATIO_LITERATURE[n] for n in anchors_nu], "ks",
            label="literature anchors")
    ax.set_xlabel(r"Poisson ratio $\nu$")
    ax.set_ylabel(r"$c_R / c_s$")
    ax.set_title("Rayleigh speed ratio: exact equation, engineering fit, anchors")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig2_rayleigh_ratio.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels = [row["name"] for row in configs]
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], [row["dt_crit_exact_s"] for row in configs], width=0.4,
           label=r"exact $2/\omega_{max}$")
    ax.bar([i + 0.2 for i in x], [row["dt_rayleigh_s"] for row in configs], width=0.4,
           label=r"Rayleigh $2r/c_R$")
    ax.plot([i for i in x], [row["engine_recommended_dt_s"] for row in configs], "k*",
            label="engine recommended_dt")
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("step (s)")
    ax.set_title("Step limits per configuration (log scale)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig3_dt_limits.png", dpi=160)
    plt.close(fig)

    print(f"wrote {OUT}/rayleigh_dt_audit.json (sha256 {report['sha256_of_body'][:16]}...)")
    print(f"amplification closed-form worst rel. error (dt<dt_crit): "
          f"{report['amplification_closed_form_worst']:.3e}")
    for row in thresholds:
        print(f"threshold {row['name']}: measured {row['measured_threshold_ratio']:.8f} "
              f"(relative error {row['relative_error']:+.3e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())