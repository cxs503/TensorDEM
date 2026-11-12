"""CASE-DEM-PLANE-SHEAR-MUIDI round 3 (m16497): root-cause controls for the
bed solid-fraction deficit.

Round 2 (``docs/CASE_DEM_PLANE_SHEAR_MUIDI_R2.md``) measured the bed solid
fraction in the steady state and left the *level* offset (phi_bed_steady
0.520-0.536 vs reference 0.58-0.61) unattributed, with two owed controls:

    D3  contact softness -- the baseline arm runs at P/kn = 1, and the settle
        phase compresses the HCP fill by 18 %; a 10x stiffer contact (P/kn=0.1)
        was never run because the fixed step budget aborted the servo;
    thin bed -- the box is 5d wide and ~6 layers tall (m14960 measured a 19 %
        low packing at 3 layers).

This script adjudicates the four arms produced by ``run_parallel.py``
(baseline / stiff / thick / stiff_thick).  It reports, per arm and per inertial
number: mu vs reference, phi_bed_steady vs reference, the settle compression
``1 - h_settled/fill_height``, and the normalised single-contact overlap
``P*d/kn`` (the actual softness indicator, not the stiffness ratio).

Outputs
    runs/analysis_r3.json          numbers (programmatic, no hand-copy)
    runs/analysis_r3_tables.md     the markdown tables used by the report doc
    figures/fig_r3_mu_vs_I.png
    figures/fig_r3_phi_vs_I.png
    figures/fig_r3_phi_deficit_vs_softness.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARM_ORDER = ("baseline", "stiff", "thick", "stiff_thick")
ARM_STYLE = {
    "baseline": dict(color="C0", marker="o", label="P/kn=1, 5d x 6 layers"),
    "stiff": dict(color="C3", marker="s", label="P/kn=0.1, 5d x 6 layers"),
    "thick": dict(color="C2", marker="^", label="P/kn=1, 8d x 9 layers"),
    "stiff_thick": dict(color="C4", marker="D", label="P/kn=0.1, 8d x 9 layers"),
}


def _load(arm: str):
    path = RUNS / f"sweep_{arm}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    # round-2 artefacts predate the round-3 schema: derive the missing
    # bookkeeping fields instead of dropping the baseline arm
    cfg = data.get("config", {})
    p_applied = float(cfg.get("P_pa", 6.25e4))
    fill = float(cfg.get("box", [0.12, 0.05, 0.06])[2])
    cfg.setdefault("layers", fill / 0.01)
    cfg.setdefault("box_depth_d", float(cfg.get("box", [0.12, 0.05, 0.06])[1]) / 0.01)
    cfg.setdefault("step_scale", float(cfg.get("kn", 6.25e4)) / 6.25e4 and
                   math.sqrt(float(cfg.get("kn", 6.25e4)) / 6.25e4))
    for row in data["rows"]:
        row.setdefault("normal_stress_applied_pa", p_applied)
        if "pressure_control_error" not in row:
            row["pressure_control_error"] = row["normal_stress_pa"] / p_applied - 1.0
        if "settle_compression" not in row:
            row["settle_compression"] = 1.0 - row["plate_height_settled"] / fill
        row.setdefault("n_particles", row.get("n_particles", 0))
    return data


def _slope(xs, ys) -> tuple[float, float]:
    """Least-squares slope of y vs log10(x) and its standard error."""
    x = np.log10(np.asarray(xs, dtype=float))
    y = np.asarray(ys, dtype=float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    se = float(np.sqrt(np.sum(resid ** 2) / max(len(x) - 2, 1))
               / np.sqrt(np.sum((x - x.mean()) ** 2)))
    return float(slope), se


def _softness(row: dict) -> float:
    """Normalised single-contact overlap at the servo pressure, P*d/kn."""
    return row["normal_stress_applied_pa"] * (2.0 * 0.005) / row["kn"]


def main() -> None:
    arms = {name: _load(name) for name in ARM_ORDER}
    base = arms["baseline"]
    if base is None:
        raise SystemExit("runs/sweep_baseline.json missing")
    ref = base["reference"]

    out = {
        "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "round": 3,
        "mission": "m16497",
        "reference": ref,
        "phi_reference_status": "PROVENANCE-UNVERIFIED (hand-copied phi_c/phi_j; "
                                "no in-repo anchor) -- see round 2 note D2",
        "arms": {},
    }

    for name, data in arms.items():
        if data is None:
            out["arms"][name] = {"available": False}
            continue
        rows = data["rows"]
        mu_err = np.array([r["mu_measured"] / r["ref_mu"] - 1.0 for r in rows])
        phi_err = np.array([r["phi_bed_steady"] / r["ref_phi"] - 1.0 for r in rows])
        phi_slope, phi_se = _slope([r["I_measured"] for r in rows],
                                   [r["phi_bed_steady"] for r in rows])
        mu_slope, mu_se = _slope([r["I_measured"] for r in rows],
                                 [r["mu_measured"] for r in rows])
        out["arms"][name] = {
            "available": True,
            "label": data.get("arm_label"),
            "config": data["config"],
            "kn": data["config"].get("kn", 6.25e4),
            "box": data["config"].get("box", [0.12, 0.05, 0.06]),
            "step_scale": data["config"].get("step_scale"),
            "points": len(rows),
            "n_particles": sorted({r["n_particles"] for r in rows}),
            "softness_Pd_over_kn": _softness(rows[0]),
            "settle_compression": {
                "mean": float(np.mean([r["settle_compression"] for r in rows])),
                "min": float(np.min([r["settle_compression"] for r in rows])),
                "max": float(np.max([r["settle_compression"] for r in rows])),
            },
            "pressure_control_error": {
                "mean": float(np.mean([r["pressure_control_error"] for r in rows])),
                "worst": float(np.max(np.abs([r["pressure_control_error"] for r in rows]))),
            },
            "mu_error_rel": {
                "mean": float(np.mean(mu_err)),
                "mean_abs": float(np.mean(np.abs(mu_err))),
                "worst_abs": float(np.max(np.abs(mu_err))),
                "n_within_10pct": int(np.sum(np.abs(mu_err) < 0.10)),
                "n_within_15pct": int(np.sum(np.abs(mu_err) < 0.15)),
            },
            "phi_error_rel": {
                "mean": float(np.mean(phi_err)),
                "mean_abs": float(np.mean(np.abs(phi_err))),
                "worst_abs": float(np.max(np.abs(phi_err))),
                "n_within_10pct": int(np.sum(np.abs(phi_err) < 0.10)),
            },
            "phi_range": [float(np.min([r["phi_bed_steady"] for r in rows])),
                          float(np.max([r["phi_bed_steady"] for r in rows]))],
            "phi_slope_per_decade": {"value": phi_slope, "se": phi_se},
            "mu_slope_per_decade": {"value": mu_slope, "se": mu_se},
            "coordination_mean": float(np.mean([r["coordination"] for r in rows])),
            "wall_clock_total_s": float(np.sum([r["wall_clock_s"] for r in rows])),
            "rows": [
                {
                    "I_target": r["I_target"],
                    "I_measured": r["I_measured"],
                    "mu": r["mu_measured"],
                    "ref_mu": r["ref_mu"],
                    "mu_err": r["mu_measured"] / r["ref_mu"] - 1.0,
                    "phi_steady": r["phi_bed_steady"],
                    "ref_phi": r["ref_phi"],
                    "phi_err": r["phi_bed_steady"] / r["ref_phi"] - 1.0,
                    "dilatancy": r["dilatancy"],
                    "coordination": r["coordination"],
                    "p_control": 1.0 + r["pressure_control_error"],
                    "settle_compression": r["settle_compression"],
                }
                for r in rows
            ],
        }

    # ---------- hypothesis adjudication ----------
    verdicts = {}
    if out["arms"].get("stiff", {}).get("available"):
        d_phi = (out["arms"]["stiff"]["phi_error_rel"]["mean"]
                 - out["arms"]["baseline"]["phi_error_rel"]["mean"])
        verdicts["H1_soft_contact_artefact"] = {
            "statistic": "mean phi error, stiff minus baseline",
            "value": float(d_phi),
            "verdict": ("supported" if d_phi > 0.02 else
                        "not supported" if d_phi > -0.005 else "worsened"),
            "rule": "|d phi| < 2 % => contact softness cannot explain the deficit",
        }
    if out["arms"].get("thick", {}).get("available"):
        d_phi_t = (out["arms"]["thick"]["phi_error_rel"]["mean"]
                   - out["arms"]["baseline"]["phi_error_rel"]["mean"])
        verdicts["H2_thin_bed_artefact"] = {
            "statistic": "mean phi error, thick minus baseline",
            "value": float(d_phi_t),
            "verdict": ("supported" if d_phi_t > 0.02 else
                        "not supported" if d_phi_t > -0.005 else "worsened"),
            "rule": "|d phi| < 2 % => bed thickness cannot explain the deficit",
        }
    if out["arms"].get("stiff_thick", {}).get("available"):
        d_phi_st = (out["arms"]["stiff_thick"]["phi_error_rel"]["mean"]
                    - out["arms"]["baseline"]["phi_error_rel"]["mean"])
        verdicts["H3_interaction"] = {
            "statistic": "mean phi error, stiff_thick minus baseline",
            "value": float(d_phi_st),
            "verdict": ("supported" if d_phi_st > 0.02 else
                        "not supported" if d_phi_st > -0.005 else "worsened"),
        }
    out["verdicts"] = verdicts

    # residual-deficit budget after the controls (what is left unexplained)
    best = min((name for name, entry in out["arms"].items()
                if entry.get("available")),
               key=lambda n: abs(out["arms"][n]["phi_error_rel"]["mean"]))
    out["best_arm_for_phi"] = {
        "arm": best,
        "mean_phi_error": out["arms"][best]["phi_error_rel"]["mean"],
        "residual_after_control": out["arms"][best]["phi_error_rel"]["mean"],
    }

    (RUNS / "analysis_r3.json").write_text(json.dumps(out, indent=1))
    write_tables(out)
    make_figures(arms, out)
    print("WROTE", RUNS / "analysis_r3.json")


# ---------------------------------------------------------------- tables ---
def write_tables(out: dict) -> None:
    lines: list[str] = []
    avail = [n for n in ARM_ORDER if out["arms"].get(n, {}).get("available")]
    lines.append("### Table 1 - arm design\n")
    lines.append("| arm | kn (N/m) | box W x D x H (m) | d_layers | P*d/kn | step scale | N_p | "
                 "settle compression | p/p_target (worst) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name in avail:
        a = out["arms"][name]
        cfg = a["config"]
        lines.append(
            f"| {name} | {a['kn']:.3g} | {cfg['box'][0]:.2f} x {cfg['box'][1]:.2f} x "
            f"{cfg['box'][2]:.2f} | {cfg['layers']:.0f} | {a['softness_Pd_over_kn']:.4f} | "
            f"{a['step_scale']:.3f} | {','.join(str(x) for x in a['n_particles'])} | "
            f"{100*a['settle_compression']['mean']:.1f}% | "
            f"{100*a['pressure_control_error']['worst']:+.1f}% |")

    lines.append("\n### Table 2 - mu(I) vs reference, per arm\n")
    header = "| I_target |" + "".join(f" mu[{n}] | err[{n}] |" for n in avail)
    lines.append(header)
    lines.append("|---|" + "---|" * (2 * len(avail)))
    for idx in range(len(out["arms"][avail[0]]["rows"])):
        cells = []
        for name in avail:
            row = out["arms"][name]["rows"][idx]
            cells.append(f" {row['mu']:.4f} | {100*row['mu_err']:+.1f}% |")
        lines.append(f"| {out['arms'][avail[0]]['rows'][idx]['I_target']:g} |"
                     + "".join(cells))

    lines.append("\n### Table 3 - phi_bed_steady(I) vs reference, per arm\n")
    lines.append("| I_target |" + "".join(f" phi[{n}] | err[{n}] |" for n in avail)
                 + " ref_phi |")
    lines.append("|---|" + "---|" * (2 * len(avail)) + "---|")
    for idx in range(len(out["arms"][avail[0]]["rows"])):
        cells = []
        for name in avail:
            row = out["arms"][name]["rows"][idx]
            cells.append(f" {row['phi_steady']:.4f} | {100*row['phi_err']:+.1f}% |")
        ref_phi = out["arms"][avail[0]]["rows"][idx]["ref_phi"]
        lines.append(f"| {out['arms'][avail[0]]['rows'][idx]['I_target']:g} |"
                     + "".join(cells) + f" {ref_phi:.4f} |")

    lines.append("\n### Table 4 - arm-level aggregates\n")
    lines.append("| arm | phi range | phi slope/decade | mean phi err | worst phi err | "
                 "mean mu err | worst mu err | mu within 10% | phi within 10% | Z | wall clock (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name in avail:
        a = out["arms"][name]
        lines.append(
            f"| {name} | {a['phi_range'][0]:.4f}-{a['phi_range'][1]:.4f} | "
            f"{a['phi_slope_per_decade']['value']:+.3e} "
            f"(+-{a['phi_slope_per_decade']['se']:.1e}) | "
            f"{100*a['phi_error_rel']['mean']:+.1f}% | "
            f"{100*a['phi_error_rel']['worst_abs']:.1f}% | "
            f"{100*a['mu_error_rel']['mean']:+.1f}% | "
            f"{100*a['mu_error_rel']['worst_abs']:.1f}% | "
            f"{a['mu_error_rel']['n_within_10pct']}/{a['points']} | "
            f"{a['phi_error_rel']['n_within_10pct']}/{a['points']} | "
            f"{a['coordination_mean']:.2f} | {a['wall_clock_total_s']:.0f} |")

    lines.append("\n### Table 5 - dilatancy (h_steady/h_settled - 1), per arm\n")
    lines.append("| I_target |" + "".join(f" dilatancy[{n}] |" for n in avail))
    lines.append("|---|" + "---|" * len(avail))
    for idx in range(len(out["arms"][avail[0]]["rows"])):
        cells = "".join(f" {100*out['arms'][n]['rows'][idx]['dilatancy']:+.2f}% |"
                        for n in avail)
        lines.append(f"| {out['arms'][avail[0]]['rows'][idx]['I_target']:g} |" + cells)

    lines.append("\n### Table 6 - hypothesis adjudication\n")
    lines.append("| hypothesis | statistic | value | verdict |")
    lines.append("|---|---|---|---|")
    for key, entry in out["verdicts"].items():
        lines.append(f"| {key} | {entry['statistic']} | {100*entry['value']:+.2f}% | "
                     f"{entry['verdict']} |")
    best = out["best_arm_for_phi"]
    lines.append(f"\nBest arm after the controls: **{best['arm']}**, residual mean phi "
                 f"error {100*best['mean_phi_error']:+.2f}%.\n")
    (RUNS / "analysis_r3_tables.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------- figures ---
def make_figures(arms: dict, out: dict) -> None:
    avail = [n for n in ARM_ORDER if out["arms"].get(n, {}).get("available")]

    # mu(I)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    I_grid = np.logspace(-3.2, 0.0, 200)
    ref = out["reference"]
    mu_ref = ref["mu_s"] + (ref["mu_2"] - ref["mu_s"]) / (ref["I_0"] / I_grid + 1.0)
    ax.semilogx(I_grid, mu_ref, "k--", lw=1.6,
                label="da Cruz 2005 / GDR MiDi 2004 mu(I)")
    for name in avail:
        rows = out["arms"][name]["rows"]
        ax.semilogx([r["I_measured"] for r in rows], [r["mu"] for r in rows],
                    ls="none", ms=7, **ARM_STYLE[name])
    ax.set_xlabel("inertial number I = gdot*d/sqrt(P/rho)")
    ax.set_ylabel("effective friction mu = tau/P")
    ax.set_title("CASE-DEM-PLANE-SHEAR-MUIDI r3: mu(I) with stiffness / bed controls")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_r3_mu_vs_I.png", dpi=150)
    plt.close(fig)

    # phi(I)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    phi_ref = (ref["phi_c"] + (ref["phi_j"] - ref["phi_c"])
               / (ref["I_0"] / I_grid + 1.0))
    ax.semilogx(I_grid, phi_ref, "k--", lw=1.6,
                label="reference phi(I) [PROVENANCE-UNVERIFIED]")
    ax.axhline(0.6366, color="0.5", lw=1.0, ls=":",
               label="RCP 0.6366 (Bernal & Mason 1960, in-repo anchor)")
    for name in avail:
        rows = out["arms"][name]["rows"]
        ax.semilogx([r["I_measured"] for r in rows], [r["phi_steady"] for r in rows],
                    ls="-", lw=1.2, ms=7, markerfacecolor="none",
                    **ARM_STYLE[name])
    ax.set_xlabel("inertial number I")
    ax.set_ylabel("steady bed solid fraction phi_bed_steady")
    ax.set_title("CASE-DEM-PLANE-SHEAR-MUIDI r3: phi(I) under stiffness / bed controls")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_r3_phi_vs_I.png", dpi=150)
    plt.close(fig)

    # deficit vs softness
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name in avail:
        a = out["arms"][name]
        ax.errorbar(a["softness_Pd_over_kn"], 100 * a["phi_error_rel"]["mean"],
                    yerr=100 * (a["phi_error_rel"]["worst_abs"]
                                - abs(a["phi_error_rel"]["mean"])),
                    fmt=ARM_STYLE[name]["marker"], color=ARM_STYLE[name]["color"],
                    ms=9, capsize=4, label=ARM_STYLE[name]["label"])
        ax.annotate(name, (a["softness_Pd_over_kn"], 100 * a["phi_error_rel"]["mean"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.axhline(0.0, color="k", lw=1.0)
    ax.axhspan(-10.0, 10.0, color="0.85", alpha=0.5, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("single-contact overlap P*d/kn (softness indicator)")
    ax.set_ylabel("mean phi error vs reference (%)")
    ax.set_title("phi deficit vs contact softness (grey band = |10 %|)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_r3_phi_deficit_vs_softness.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()