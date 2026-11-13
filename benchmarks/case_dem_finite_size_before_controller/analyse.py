#!/usr/bin/env python
"""Gates, fits and figures for XVAL-DEM-FINITE-SIZE-BEFORE-CONTROLLER.

Every number printed here is read from ``runs/result.json``; the analytic
predictions are re-derived from the pre-declared formulas
(lambda = 1 + 1.7601 phi^(1/3) + phi, drive built for unbounded Stokes drag), so
nothing is typed in by hand.

    python benchmarks/case_dem_finite_size_before_controller/analyse.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FIGS = HERE / "figures"


def gate(name: str, criterion: str, measured, threshold, passed: bool, evidence: str) -> dict:
    return {"gate": name, "criterion": criterion, "measured": measured,
            "threshold": threshold, "passed": bool(passed), "evidence": evidence}


def fit_dilute_coefficient(points: list[tuple[float, float]]) -> dict:
    """Least squares for lambda = c0 (1 + a phi^(1/3) + phi).

    Linear in (c0, c0*a), so it is solved exactly by lstsq; one degree of freedom
    is left over the three phi points and that residual is reported.
    """
    phi = np.array([p[0] for p in points])
    lam = np.array([p[1] for p in points])
    rhs = lam / (1.0 + phi)
    design = np.column_stack([np.ones_like(phi), phi ** (1.0 / 3.0)])
    (c0, c0a), *_ = np.linalg.lstsq(design, rhs, rcond=None)
    a = c0a / c0
    resid = design @ np.array([c0, c0a]) - rhs
    return {"c0": float(c0), "a": float(a), "max_abs_residual": float(np.abs(resid).max()),
            "n_points": int(phi.size), "dof_left": int(phi.size - 2)}


def main() -> int:
    data = json.loads((RUNS / "result.json").read_text())
    cases = {c["label"]: c for c in data["cases"]}
    prereg = data["prereg"]
    tol = prereg["thresholds"]
    FIGS.mkdir(exist_ok=True)
    gates: list[dict] = []

    # ------------------------------------------------------------------ A/B
    def settle(label: str) -> dict:
        return cases[label]

    a_ld4_m0 = settle("settle_ld4_M0_posthoc")
    a_ld4_m1 = settle("settle_ld4_M1_drive_and_posthoc")
    a_ld4_m2 = settle("settle_ld4_M2_drive_only")
    lam4 = a_ld4_m0["array_correction"]
    pred_m0 = 1.0 / lam4 - 1.0
    pred_m1 = lam4 - 1.0

    gates.append(gate(
        "xval_settling_M2_operating_point_ld4",
        "|Re_terminal/Re_target - 1| with the correction folded into the drive only, <= 15 %",
        a_ld4_m2["re_terminal_vs_target"], tol["xval_accept"],
        abs(a_ld4_m2["re_terminal_vs_target"]) <= tol["xval_accept"],
        f"settle_ld4_M2_drive_only: Re_target=1, lambda={lam4:.4f}, "
        f"steadiness={a_ld4_m2['steadiness']:.4f}"))
    gates.append(gate(
        "defect_settling_M0_operating_point_ld4",
        "post-hoc-only path reproduces the predicted 1/lambda - 1 bias to <= 15 % AND the bias exceeds 15 %",
        a_ld4_m0["re_terminal_vs_target"], tol["xval_accept"],
        abs(a_ld4_m0["re_terminal_vs_target"] - pred_m0) <= tol["xval_accept"]
        and abs(a_ld4_m0["re_terminal_vs_target"]) > tol["xval_accept"],
        f"settle_ld4_M0_posthoc: predicted {pred_m0:+.4f}, "
        f"measured {a_ld4_m0['re_terminal_vs_target']:+.4f}"))
    gates.append(gate(
        "defect_settling_M1_double_count_ld4",
        "drive+post-hoc double counting reproduces the predicted +lambda - 1 bias to <= 15 % AND exceeds 15 %",
        a_ld4_m1["u_reported_vs_target"], tol["xval_accept"],
        abs(a_ld4_m1["u_reported_vs_target"] - pred_m1) <= tol["xval_accept"]
        and abs(a_ld4_m1["u_reported_vs_target"]) > tol["xval_accept"],
        f"settle_ld4_M1: predicted {pred_m1:+.4f}, measured {a_ld4_m1['u_reported_vs_target']:+.4f}"))

    b_ld8_m0 = settle("settle_ld8_M0_posthoc")
    b_ld8_m2 = cases.get("settle_ld8_M2_drive_only")
    lam8 = b_ld8_m0["array_correction"]
    gates.append(gate(
        "xval_settling_M2_operating_point_ld8",
        "|Re_terminal/Re_target - 1| with correction in the drive only, L/d=8 (d=4), <= 15 %",
        ("not_run" if b_ld8_m2 is None else b_ld8_m2["re_terminal_vs_target"]), tol["xval_accept"],
        bool(b_ld8_m2 is not None and abs(b_ld8_m2["re_terminal_vs_target"]) <= tol["xval_accept"]),
        f"settle_ld8_M2_drive_only: lambda={lam8:.4f} (busy-case reaped; not_run)"))
    gates.append(gate(
        "defect_settling_M0_operating_point_ld8",
        "post-hoc-only bias at L/d=8 matches 1/lambda - 1 to <= 15 % (magnitude reported, not gated)",
        b_ld8_m0["re_terminal_vs_target"], tol["xval_accept"],
        abs(b_ld8_m0["re_terminal_vs_target"] - (1.0 / lam8 - 1.0)) <= tol["xval_accept"],
        f"predicted {1.0/lam8-1.0:+.4f}, measured {b_ld8_m0['re_terminal_vs_target']:+.4f}"))

    steady_max = max(c["steadiness"] or 0.0 for k, c in cases.items() if k.startswith("settle_"))
    gates.append(gate(
        "settling_steady_state",
        "window-to-window drift of the settled velocity, all settling cases, <= 2 %",
        steady_max, tol["steadiness_accept"], steady_max <= tol["steadiness_accept"],
        "max over " + ", ".join(sorted(k for k in cases if k.startswith("settle_")))
        + f" at {data['cases'][0]['steps'] if 'steps' in data['cases'][0] else ''}"))

    # ------------------------------------------------------- pump-controller
    for gain in ("0.005", "0.02"):
        before = cases[f"fixed_ld4_g{gain}_before"]
        after = cases[f"fixed_ld4_g{gain}_after"]
        sb = before["steps_to_within_2pct"]
        sa = after["steps_to_within_2pct"]
        gates.append(gate(
            f"controller_seed_effect_g{gain}",
            "steps-to-within-2%-of-final-force is lower with the correction before the pump seed "
            "(materiality: ratio >= 2)",
            (None if (sb is None or sa is None or sb == 0) else sa / sb),
            tol["convergence_material_factor"],
            bool(sb is not None and sa is not None and sb > 0
                 and sa / sb >= tol["convergence_material_factor"]),
            f"before={sb} steps (wall {before['wall_seconds']:.0f}s), "
            f"after={sa} steps (wall {after['wall_seconds']:.0f}s), "
            f"converged flags: {before['converged_flag']}/{after['converged_flag']}"))
        gates.append(gate(
            f"controller_converged_ratio_agreement_g{gain}",
            "the converged/end-budget corrected Stokes ratio is timing independent, |diff| <= 15 %",
            abs(before["stokes_ratio_corrected"] - after["stokes_ratio_corrected"]),
            tol["xval_accept"],
            abs(before["stokes_ratio_corrected"] - after["stokes_ratio_corrected"]) <= tol["xval_accept"],
            f"before {before['stokes_ratio_corrected']:.4f} vs after "
            f"{after['stokes_ratio_corrected']:.4f} at {before['steps']} steps"))

    # --------------------------------------------------- Hasimoto XVAL scan
    phi_scan_labels = ["fixed_ld4_g0.02_before", "fixed_phi_ld6_d6", "fixed_phi_ld8_d6"]
    phi_scan = [cases[k] for k in phi_scan_labels]
    worst = max(phi_scan, key=lambda c: abs(c["xval_error_vs_hasimoto"]))
    gates.append(gate(
        "xval_array_ratio_vs_hasimoto_d6",
        "|F/lambda - 3 pi mu d U| / (3 pi mu d U) over phi in {0.0082, 0.0024, 0.0010} (d=6), <= 15 %",
        worst["xval_error_vs_hasimoto"], tol["xval_accept"],
        abs(worst["xval_error_vs_hasimoto"]) <= tol["xval_accept"],
        f"worst point {worst['label']} (phi={worst['solid_fraction']:.5f}, "
        f"lambda={worst['array_correction']:.4f})"))
    res_labels = ["fixed_phi_ld8_d4", "fixed_phi_ld8_d6", "fixed_phi_ld8_d8"]
    res = [cases[k] for k in res_labels if k in cases]
    spread = (max(c["xval_error_vs_hasimoto"] for c in res)
              - min(c["xval_error_vs_hasimoto"] for c in res))
    gates.append(gate(
        "resolution_spread_ld8",
        "spread of the corrected array ratio over d in {4,6,8} at L/d=8, <= 15 %",
        spread, tol["xval_accept"], spread <= tol["xval_accept"],
        "; ".join(f"{c['label']} d={c['diameter']:g}: {c['xval_error_vs_hasimoto']:+.4f}" for c in res)))

    fit = fit_dilute_coefficient([(c["solid_fraction"], c["stokes_ratio"]) for c in phi_scan])
    gates.append(gate(
        "fitted_dilute_coefficient",
        "phi^(1/3) coefficient fitted from the d=6 scan vs Hasimoto 1.7601, within 15 %",
        fit["a"] / 1.7601 - 1.0, tol["xval_accept"],
        abs(fit["a"] / 1.7601 - 1.0) <= tol["xval_accept"],
        f"a_fit={fit['a']:.4f} (c0={fit['c0']:.4f}, max|resid|={fit['max_abs_residual']:.2e}, "
        f"{fit['dof_left']} dof)"))

    # momentum audit on the pump cases (drive consistency)
    mom = max(abs(c["momentum_balance_ratio"] - 1.0) for k, c in cases.items() if k.startswith("fixed"))
    gates.append(gate(
        "pump_momentum_balance",
        "|F_drag/(body*L^3) - 1| over all pump cases, <= 5 %",
        mom, 0.05, mom <= 0.05, "all fixed_* cases"))

    # ---------------------------------------------------------------- figures
    fig1, ax = plt.subplots(figsize=(5.2, 3.4))
    labels = ["M0 post-hoc only\n(current path)", "M1 drive+post-hoc\n(double count)",
              "M2 drive only\n(correction before)"]
    vals = [a_ld4_m0["re_terminal_vs_target"], a_ld4_m1["re_terminal_vs_target"],
            a_ld4_m2["re_terminal_vs_target"]]
    reported = [a_ld4_m0["u_reported_vs_target"], a_ld4_m1["u_reported_vs_target"],
                a_ld4_m2["u_reported_vs_target"]]
    x = np.arange(3)
    ax.axhspan(-0.15, 0.15, color="#e8f5e9", zorder=0, label="±15 % acceptance band")
    ax.bar(x - 0.18, vals, 0.36, label="achieved Re/Re_target − 1", color="#1f77b4")
    ax.bar(x + 0.18, reported, 0.36, label="reported corrected U vs target", color="#ff7f0e")
    ax.axhline(1.0 / lam4 - 1.0, ls="--", lw=1, color="#1f77b4",
               label=f"predicted 1/λ−1 = {1.0/lam4-1.0:+.3f}")
    ax.axhline(lam4 - 1.0, ls="--", lw=1, color="#ff7f0e",
               label=f"predicted λ−1 = {lam4-1.0:+.3f}")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("relative deviation from target")
    ax.set_title(f"Settling operating point, L/d=4, λ={lam4:.4f}")
    ax.legend(fontsize=6, loc="lower left")
    fig1.tight_layout(); fig1.savefig(FIGS / "fig1_settling_operating_point.png", dpi=160)

    fig2, ax = plt.subplots(figsize=(5.2, 3.4))
    phis = np.logspace(math.log10(5e-4), math.log10(2e-2), 100)
    lam_curve = [1.0 + 1.7601 * p ** (1.0 / 3.0) + p for p in phis]
    ax.plot(phis, lam_curve, "-", color="k", lw=1, label="Hasimoto/Sangani–Acrivos")
    styled = {"fixed_ld4_g0.02_before": ("o", "d=6, L/d=4"),
              "fixed_phi_ld6_d6": ("s", "d=6, L/d=6"),
              "fixed_phi_ld8_d6": ("^", "d=6, L/d=8"),
              "fixed_phi_ld8_d4": ("v", "d=4, L/d=8"),
              "fixed_phi_ld8_d8": ("D", "d=8, L/d=8"),
              "fixed_phi_ld16_d4": ("P", "d=4, L/d=16")}
    for label, (marker, text) in styled.items():
        if label not in cases:
            continue
        c = cases[label]
        ax.plot([c["solid_fraction"]], [c["stokes_ratio"]], marker, ms=6,
                label=f"{text}: {c['stokes_ratio']:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("solid fraction φ = (π/6)(d/L)³")
    ax.set_ylabel("measured F/(3πμdU)")
    ax.set_title("Periodic-array drag enhancement vs analytic finite-size ratio")
    ax.legend(fontsize=6)
    fig2.tight_layout(); fig2.savefig(FIGS / "fig2_array_ratio_phi.png", dpi=160)

    fig3, ax = plt.subplots(figsize=(5.2, 3.4))
    for gain, style in (("0.005", "-"), ("0.02", "--")):
        for timing, color in (("before", "#1f77b4"), ("after", "#d62728")):
            c = cases[f"fixed_ld4_g{gain}_{timing}"]
            # rebuild the force trace from the thin history (force + u_mean)
            hist = c["history"]
            f_final = hist[-1]["force"]
            steps = [h["step"] for h in hist]
            dev = [abs(h["force"] - f_final) / abs(f_final) for h in hist]
            ax.semilogy(steps, dev, style, color=color, lw=1,
                        label=f"gain {gain}, seed {timing}")
    ax.axhline(0.02, color="k", lw=0.8, ls=":")
    ax.set_xlabel("LBM step")
    ax.set_ylabel("|F(step) − F(final)| / |F(final)|")
    ax.set_title("Pump-controller transient, L/d=4 (λ=1.36)")
    ax.legend(fontsize=6)
    fig3.tight_layout(); fig3.savefig(FIGS / "fig3_pump_convergence.png", dpi=160)

    out = {
        "mission_id": data["mission_id"],
        "anchor": data["anchor"],
        "n_gates": len(gates),
        "all_pass": all(g["passed"] for g in gates),
        "gates": gates,
        "dilute_fit": fit,
        "environment": data["environment"],
    }
    (HERE / "gates.json").write_text(json.dumps(out, indent=1))
    for g in gates:
        m = g["measured"]
        ms = f"{m:+.4f}" if isinstance(m, float) else str(m)
        print(f"{'PASS' if g['passed'] else 'FAIL'}  {g['gate']:<42} {ms}")
    print(f"\nfit: a={fit['a']:.4f} vs 1.7601 ({fit['a']/1.7601-1:+.1%}), "
          f"c0={fit['c0']:.4f}, max|resid|={fit['max_abs_residual']:.2e}")
    print(f"figures -> {FIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())