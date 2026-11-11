"""Figures + paper tables for the joint DEM calibration (node P284W3-N08).

Reads ``artifacts/dem_joint_calibration/*.json`` (written by
``scripts/run_joint_calibration.py``) and emits

* ``artifacts/dem_joint_calibration/figures/fig1_benchmarks.png``
* ``.../fig2_convergence.png``
* ``.../fig3_posterior.png``
* ``.../fig4_verification.png``
* ``artifacts/dem_joint_calibration/tables.md``  (markdown, zero hand-copy)

Usage:  python scripts/joint_calibration_report.py [--indir ...] [--outdir ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PARAM_LABELS = {
    "kn": r"$k_n$ [N/m]",
    "kt": r"$k_t$ [N/m]",
    "mu": r"$\mu$ [-]",
    "restitution": r"$e$ [-]",
    "sigma_bond": r"$\sigma_{bond}$ [Pa]",
}


def _load(indir: str) -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(indir, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name.endswith("_replicates"):
            continue
        with open(path) as fh:
            out[name] = json.load(fh)
    return out


def _curve(entry: dict, key: str):
    raw = entry.get(key, {})
    return raw


def fig_benchmarks(data: dict, path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for name, res in data.items():
        raw = _curve(res, "target_raw")
        shr = raw.get("shear", {})
        uni = raw.get("uniaxial", {})
        if shr:
            axes[0].plot(shr["strain"], shr["tau_smooth"], label=f"{name} (target)")
        if uni:
            axes[1].plot(uni["strain"], uni["stress_smooth"], label=f"{name} (target)")
    entries = list(data.values())
    labels = [e["material"] for e in entries]
    theta = [e["targets"]["repose.theta"] for e in entries]
    axes[2].bar(labels, theta, color="#4477aa")
    axes[2].set_ylabel("repose angle [deg]")
    for i, v in enumerate(theta):
        axes[2].text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
    axes[0].set_xlabel(r"engineering shear strain $\gamma$ [-]")
    axes[0].set_ylabel(r"shear stress $\tau$ [Pa]")
    axes[0].set_title("direct-shear response (twin targets)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel(r"axial strain $\varepsilon$ [-]")
    axes[1].set_ylabel(r"axial stress $\sigma$ [Pa]")
    axes[1].set_title("bonded uniaxial response (twin targets)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    axes[2].set_title("angle of repose (dense-column collapse)")
    axes[2].tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_convergence(data: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, res in data.items():
        J = [row["objective"] for row in res["design"]]
        ax.semilogy(np.arange(1, len(J) + 1), J, marker="o", ms=3, label=name)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("forward evaluation (50 = budget)")
    ax.set_ylabel(r"joint objective $J(\theta)$")
    ax.set_title("GP + expected-improvement search")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_posterior(data: dict, path: str) -> None:
    params = ("kn", "kt", "mu", "restitution", "sigma_bond")
    fig, axes = plt.subplots(1, len(params), figsize=(15.0, 3.4))
    for name, res in data.items():
        for ax, p in zip(axes, params):
            s = res["posterior"][p]
            truth = res["truth"][p]
            ax.errorbar([list(data).index(name)], [s["median"]],
                        yerr=[[s["median"] - s["lo95"]], [s["hi95"] - s["median"]]],
                        fmt="o", capsize=4, label=name)
            ax.axhline(truth, color="k", ls="--", lw=0.8)
    for ax, p in zip(axes, params):
        ax.set_title(PARAM_LABELS[p], fontsize=9)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(list(data), rotation=20, fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("posterior 95 % credible intervals (dashed = truth)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_verification(data: dict, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for name, res in data.items():
        v = res["verification"]
        axes[0].plot(v["shear_curve_target"], v["shear_curve_sim"], "o-", ms=4,
                     label=name)
        axes[1].bar([name], [v["peak_stress_rel_err"] * 100.0], alpha=0.7)
    lo, hi = 0.0, 1.0
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=0.8)
    axes[0].set_xlabel(r"target $\tau/\sigma_n$ [-]")
    axes[0].set_ylabel(r"calibrated $\tau/\sigma_n$ [-]")
    axes[0].set_title("shear curve: calibrated vs target (independent seed)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].axhline(5.0, color="r", ls="--", lw=0.8, label="5 % criterion")
    axes[1].set_ylabel("peak-stress error [%]")
    axes[1].set_title("uniaxial peak stress error")
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def tables(data: dict, replicates: dict) -> str:
    lines = []
    lines.append("### T1  campaign criteria per material\n")
    lines.append("| material | (a) repose err [deg] < 3 | (b) shear RMSE rel < 5% "
                 "| (c) peak err < 5% | (d) J drop > 90% | (e) coverage >= 90% | all pass |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, res in data.items():
        v = res["verdicts"]
        mark = lambda d: f"{d['value']:.4g} {'PASS' if d['pass'] else 'FAIL'}"  # noqa: E731
        lines.append(f"| {name} | {mark(v['a_repose'])} | {mark(v['b_shear_rmse'])} | "
                     f"{mark(v['c_peak_stress'])} | {mark(v['d_objective_drop'])} | "
                     f"{mark(v['e_coverage'])} | {res['all_pass']} |")
    lines.append("\n### T2  parameter recovery and posterior 95 % intervals\n")
    lines.append("| material | parameter | truth | posterior median | 95 % interval | covered |")
    lines.append("|---|---|---|---|---|---|")
    for name, res in data.items():
        for p in ("kn", "kt", "mu", "restitution", "sigma_bond"):
            s = res["posterior"][p]
            truth = res["truth"][p]
            covered = s["lo95"] <= truth <= s["hi95"]
            lines.append(f"| {name} | {p} | {truth:.4g} | {s['median']:.4g} | "
                         f"[{s['lo95']:.4g}, {s['hi95']:.4g}] | {covered} |")
    lines.append("\n### T3  search budget and cost\n")
    lines.append("| material | evaluations | unique forwards | J initial | J best | drop | BayesOpt [s] | total [s] |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, res in data.items():
        d = res["verdicts"]["d_objective_drop"]
        lines.append(f"| {name} | {res['evaluations']} | {res['n_unique_evaluations']} | "
                     f"{d['J_initial']:.4g} | {d['J_best']:.4g} | {res['objective_drop']:.3f} | "
                     f"{res['wall_seconds']['bayesopt']:.0f} | {res['wall_seconds_total']:.0f} |")
    lines.append("\n### T4  twin-experiment noise floor (seed replicates at the truth)\n")
    if replicates:
        lines.append("| material | theta mean [deg] | theta rel-std | peak mean [Pa] | peak rel-std |")
        lines.append("|---|---|---|---|---|")
        for name, rep in replicates.items():
            f = rep["features"]
            th = f["repose.theta"]
            pk = f["uniaxial.peak"]
            lines.append(f"| {name} | {th['mean']:.3f} | {th['rel_std']:.4f} | "
                         f"{pk['mean']:.1f} | {pk['rel_std']:.4f} |")
    else:
        lines.append("_no replicate artefact found_")
    lines.append("\n### T5  verification on independent seeds\n")
    lines.append("| material | theta sim [deg] | theta target [deg] | shear sigma_n [Pa] | "
                 "intact bonds after UCS | escaped |")
    lines.append("|---|---|---|---|---|---|")
    for name, res in data.items():
        v = res["verification"]
        lines.append(f"| {name} | {v['theta_sim_deg']:.3f} | {v['theta_target_deg']:.3f} | "
                     f"{v['shear_sigma_n']:.0f} | {v['uniaxial_intact_fraction']:.3f} | "
                     f"{v['shear_escaped']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="artifacts/dem_joint_calibration")
    ap.add_argument("--doc", default="docs/P284W3_N08_JOINT_CALIBRATION.md")
    ap.add_argument("--outdir", default="artifacts/dem_joint_calibration")
    args = ap.parse_args()
    data = _load(args.indir)
    if not data:
        raise SystemExit(f"no artefacts in {args.indir}")
    figdir = os.path.join(args.outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    replicates = {}
    for path in glob.glob(os.path.join(args.indir, "*_replicates.json")):
        name = os.path.basename(path).replace("_replicates.json", "")
        with open(path) as fh:
            replicates[name] = json.load(fh)
    fig_benchmarks(data, os.path.join(figdir, "fig1_benchmarks.png"))
    fig_convergence(data, os.path.join(figdir, "fig2_convergence.png"))
    fig_posterior(data, os.path.join(figdir, "fig3_posterior.png"))
    fig_verification(data, os.path.join(figdir, "fig4_verification.png"))
    md = tables(data, replicates)
    with open(os.path.join(args.outdir, "tables.md"), "w") as fh:
        fh.write(md)
    if args.doc and os.path.exists(args.doc):
        with open(args.doc, encoding="utf-8") as fh:
            doc = fh.read()
        if "<!--TABLES-->" in doc:
            doc = doc.replace("<!--TABLES-->", "<!--TABLES-BEGIN-->" + md +
                              "<!--TABLES-END-->")
        elif "<!--TABLES-BEGIN-->" in doc:   # refresh the injected block
            start = doc.index("<!--TABLES-BEGIN-->")
            end = doc.index("<!--TABLES-END-->") + len("<!--TABLES-END-->")
            doc = doc[:start] + "<!--TABLES-BEGIN-->" + md + "<!--TABLES-END-->" + doc[end:]
        with open(args.doc, "w", encoding="utf-8") as fh:
            fh.write(doc)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())