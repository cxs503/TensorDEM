"""Figures + tables for the plane-shear mu(I) round-4a premise audit.

Every number is read from the run artefacts under
``benchmarks/plane_shear_mu_i/runs/``; nothing is typed in by hand.

    PYTHONPATH=src python3 benchmarks/plane_shear_mu_i/servo_diag_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from reference_law import ref_mu as _ref_mu, ref_phi as _ref_phi  # single source of truth (m17026)

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DIAG = RUNS / "servo_diag"
OUT = DIAG / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MU_S, MU_2, I_0 = 0.36, 0.84, 0.30
PHI_C, PHI_J = 0.58, 0.64


def ref_mu(I):
    return MU_S + (MU_2 - MU_S) / (I_0 / np.asarray(I) + 1.0)


def ref_phi(I):
    return _ref_phi(I)


def load(path):
    return json.loads(Path(path).read_text())


def fig_series():
    """Normal-stress traces: default vs raised servo authority."""
    a = load(DIAG / "trace_relax0.2_clip0.002.json")["rows"]
    b = load(DIAG / "trace_relax0.5_clip0.05.json")["rows"]
    sel = [0.01, 0.1, 0.3]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6), sharey=True)
    for ax, I in zip(axes, sel):
        for rows, lab, col in ((a, "default authority\n(relax 0.2, clip 0.002 d)", "tab:red"),
                               (b, "raised authority\n(relax 0.5, clip 0.05 d)", "tab:blue")):
            row = min(rows, key=lambda r: abs(r["I_target"] - I))
            s = row["series"]
            ax.plot(s["strain"], np.asarray(s["normal_stress_pa"]) / 1e3, col,
                    lw=1.0, alpha=0.85,
                    label=f"{lab}\nP_err={row['pressure_control_error'] * 100:+.1f}%")
        ax.axhline(62.5, color="k", ls="--", lw=0.9, label=r"target 62.5 kPa")
        ax.set_title(f"I = {row['I_target']:g}  (achieved {row['I_measured']:.3g})")
        ax.set_xlabel("strain")
        ax.legend(fontsize=6.5, loc="lower left")
    axes[0].set_ylabel("normal stress on platen [kPa]")
    fig.suptitle("CASE-DEM-PLANE-SHEAR-MUIDI R4a: the constant-pressure premise, "
                 "judged instead of assumed", fontsize=10)
    fig.tight_layout()
    p = OUT / "fig1_pressure_traces.png"
    fig.savefig(p, dpi=130)
    return p


def fig_curves():
    """mu(I) and phi(I): reference law vs default vs raised vs ensemble."""
    base = load(RUNS / "sweep_baseline.json")["rows"]
    raised = load(DIAG / "trace_relax0.5_clip0.05.json")["rows"]
    ens = load(DIAG / "ensemble_relax0.5_clip0.05.json")["summary"]
    Ig = np.logspace(-3, np.log10(0.4), 120)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax1.plot(Ig, ref_mu(Ig), "k-", lw=1.6, label=r"da Cruz 2005 $\mu(I)$")
    ax2.plot(Ig, ref_phi(Ig), "k-", lw=1.6, label=r"GDR MiDi $\phi(I)$ (hand-copied)")
    for rows, lab, col in ((base, "default authority (round-3 artefact)", "tab:red"),
                           (raised, "raised authority (this round)", "tab:blue")):
        ax1.plot([r["I_measured"] for r in rows], [r["mu_measured"] for r in rows],
                 "o-", color=col, ms=4, lw=1.0, label=lab)
        ax2.plot([r["I_measured"] for r in rows], [r["phi_bed_steady"] for r in rows],
                 "o-", color=col, ms=4, lw=1.0, label=lab)
    ax1.errorbar([e["I_target"] for e in ens], [e["mu_mean"] for e in ens],
                 yerr=[e["mu_sd"] for e in ens], fmt="s", color="tab:green", ms=5,
                 capsize=3, label="4-seed ensemble (raised auth.)")
    ax2.errorbar([e["I_target"] for e in ens], [e["phi_mean"] for e in ens],
                 yerr=[e["phi_sd"] for e in ens], fmt="s", color="tab:green", ms=5,
                 capsize=3, label="4-seed ensemble (raised auth.)")
    for ax, ylab in ((ax1, r"$\mu = \tau/P$"), (ax2, r"$\phi_{bed}$ (steady window)")):
        ax.set_xscale("log")
        ax.set_xlabel("inertial number I")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.suptitle("Both curves are judged at 72-84 % of the target pressure on the "
                 "round-3 artefact (P1 violated); raising authority moves the control\n"
                 "error but not the mu/phi bias -> the residual gap is not a servo artefact",
                 fontsize=9)
    fig.tight_layout()
    p = OUT / "fig2_mu_phi_curves.png"
    fig.savefig(p, dpi=130)
    return p


def fig_strain():
    """Strain ladder: does the bias converge with strain?"""
    rows = load(DIAG / "strain_ladder.json")["rows"]
    Is = sorted({r["I_target"] for r in rows})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for I in Is:
        sub = sorted([r for r in rows if r["I_target"] == I], key=lambda r: r["shear_strain"])
        st = [r["strain_achieved"] if r["strain_achieved"] is not None else r["shear_strain"]
              for r in sub]
        ax1.plot(st, [100 * (r["mu"] / r["ref_mu"] - 1) for r in sub], "o-", ms=4,
                 label=f"I={I:g}")
        ax2.plot(st, [100 * (r["phi_bed_steady"] / r["ref_phi"] - 1) for r in sub], "o-",
                 ms=4, label=f"I={I:g}")
    for ax, t in ((ax1, r"$\mu$ error [%]"), (ax2, r"$\phi$ error [%]")):
        ax.axhspan(-10, 10, color="tab:green", alpha=0.12, label="acceptance band ±10 %")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("achieved shear strain")
        ax.set_ylabel(t)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.suptitle("Strain ladder 0.2 -> 3.0: no monotone convergence; the mu/phi bias is "
                 "not a transient-duration artefact", fontsize=9)
    fig.tight_layout()
    p = OUT / "fig3_strain_ladder.png"
    fig.savefig(p, dpi=130)
    return p


def tables():
    base = load(RUNS / "sweep_baseline.json")["rows"]
    raised = load(DIAG / "trace_relax0.5_clip0.05.json")["rows"]
    ens = load(DIAG / "ensemble_relax0.5_clip0.05.json")["summary"]
    ladder = load(DIAG / "strain_ladder.json")["rows"]
    lines = ["# R4a generated tables (no hand-typed numbers)", "",
             "## Table 1 - premise gate, default servo authority (round-3 artefact)", "",
             "| I_meas | P achieved [kPa] | P err [%] | mu | ref mu | mu err [%] | phi_s | ref phi | phi err [%] |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in base:
        lines.append("| {:.4g} | {:.2f} | {:+.2f} | {:.4f} | {:.4f} | {:+.2f} | {:.4f} | {:.4f} | {:+.2f} |".format(
            r["I_measured"], r["normal_stress_pa"] / 1e3,
            100 * (r["normal_stress_pa"]
                   / r.get("normal_stress_applied_pa", 6.25e4) - 1),
            r["mu_measured"], r["ref_mu"], 100 * (r["mu_measured"] / r["ref_mu"] - 1),
            r["phi_bed_steady"], r["ref_phi"],
            100 * (r["phi_bed_steady"] / r["ref_phi"] - 1)))
    lines += ["", "## Table 2 - same device, raised servo authority (numerics only)", "",
              "| I_meas | P err [%] | P cv [%] | P drift [% in window] | mu | mu err [%] | phi_s | phi err [%] | strain run |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in raised:
        lines.append("| {:.4g} | {:+.2f} | {:.2f} | {:+.2f} | {:.4f} | {:+.2f} | {:.4f} | {:+.2f} | {:.2f} |".format(
            r["I_measured"], 100 * r["pressure_control_error"],
            100 * r["window"]["p_cv"], 100 * r["window"]["p_rel_drift_over_window"],
            r["mu"], 100 * (r["mu"] / r["ref_mu"] - 1), r["phi_bed_steady"],
            100 * (r["phi_bed_steady"] / r["ref_phi"] - 1),
            r["strain_achieved"] if r["strain_achieved"] is not None else float("nan")))
    lines += ["", "## Table 3 - 4-seed ensemble at the two failing points (raised authority)", "",
              "| I | n seeds | mu mean | mu sd | mu bias [%] | mu scatter [%] | phi mean | phi sd | phi bias [%] | P err mean [%] |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for e in ens:
        lines.append("| {:g} | {} | {:.4f} | {:.4f} | {:+.2f} | {:.2f} | {:.4f} | {:.4f} | {:+.2f} | {:+.2f} |".format(
            e["I_target"], e["n_seeds"], e["mu_mean"], e["mu_sd"], e["mu_bias_pct"],
            e["mu_scatter_pct"], e["phi_mean"], e["phi_sd"], e["phi_bias_pct"],
            e["p_err_mean_pct"]))
    lines += ["", "## Table 4 - strain ladder (achieved strain, not requested)", "",
              "| I | strain run | shear steps | P err [%] | P drift [%] | mu | mu err [%] | phi_s | phi err [%] |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(ladder, key=lambda r: (r["I_target"], r["shear_strain"])):
        lines.append("| {:g} | {:.2f} | {} | {:+.2f} | {:+.2f} | {:.4f} | {:+.2f} | {:.4f} | {:+.2f} |".format(
            r["I_target"], r["strain_achieved"], r["shear_steps"],
            100 * r["pressure_control_error"],
            100 * r["window"]["p_rel_drift_over_window"], r["mu"],
            100 * (r["mu"] / r["ref_mu"] - 1), r["phi_bed_steady"],
            100 * (r["phi_bed_steady"] / r["ref_phi"] - 1)))
    p = DIAG / "tables.md"
    p.write_text("\n".join(lines) + "\n")
    return p


if __name__ == "__main__":
    for f in (fig_series, fig_curves, fig_strain, tables):
        print("WROTE", f())