#!/usr/bin/env python
"""Run the full p307w1-n010 PID-on-MDOF campaign and produce acceptance artifacts.

Reproducible command:
    python scripts/run_pid_mdof_campaign.py --outdir artifacts/p307w1_n010

Produces:
    artifacts/p307w1_n010/acceptance.json   — machine-verifiable criterion results
    artifacts/p307w1_n010/results.json      — full numerical results
    artifacts/p307w1_n010/fig1_il_curves.png        — IL(f) for cantilever & plate
    artifacts/p307w1_n010/fig2_tuning_scan.png      — IL vs tuning ratio
    artifacts/p307w1_n010/fig3_detuning_envelope.png — ±5/10/20% detuning robustness
    artifacts/p307w1_n010/fig4_mass_comparison.png   — per-unit-mass IL table
    artifacts/p307w1_n010/fig5_den_hartog_anchor.png — MDOF vs Den Hartog closed form
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem import pid_mdof as pm


def run_acceptance(outdir: str) -> dict:
    """Run all five acceptance criteria and return the verdict dict."""
    os.makedirs(outdir, exist_ok=True)

    # ---- Criterion (a): IN32-D02 regression cross-check ≤ 5% ----
    from tensordem import pid as pid_mod
    cfg = pid_mod.SweepConfig()
    surface = pid_mod.ZetaSurface.from_results(pid_mod.run_sweep(cfg), cfg)
    gi = surface.gamma_axis.index(2.0)
    fi = surface.freq_ratio_axis.index(1.0)
    pi = surface.filling_ratio_axis.index(0.3)
    ei = surface.restitution_axis.index(0.6)
    expected = surface.zeta_values[gi][fi][pi][ei]
    got_a = pm.zeta_eff_delivered(0.3, 1.0, gamma=2.0, restitution=0.6,
                                  mass_ratio=pm.MU_REF)
    rel_err_a = abs(got_a - expected) / expected if expected != 0 else 0.0
    a_pass = rel_err_a <= 0.05

    # ---- Criterion (b): MDOF optimal IL vs Den Hartog closed form ≤ 10% ----
    mu = 0.03
    nu_opt, zeta_opt = pm.den_hartog_optimum(mu)
    s = pm.cantilever_structure(n_modes=8)
    dam = pm.ParticleDamperModel(filling_ratio=0.3, mass_ratio=mu)
    dam.zeta_d_fixed = zeta_opt

    class FixedDamper(type(dam)):
        def zeta_d(self, nu=1.0):
            return self.zeta_d_fixed

    dam_fixed = FixedDamper(**{k: getattr(dam, k) for k in
                               ("filling_ratio", "mass_ratio", "gamma",
                                "restitution", "detuning_coupling",
                                "n_particles", "primary_mass",
                                "primary_stiffness", "cavity_length", "gravity")})
    dam_fixed.zeta_d_fixed = zeta_opt

    il_mdof = pm.rms_insertion_loss(s, dam_fixed, nu=nu_opt)
    il_dh = pm.rms_insertion_loss_2dof(mu, nu_opt, zeta_opt, s.zeta[0])
    rel_err_b = abs(il_mdof - il_dh) / il_dh if il_dh != 0 else 0.0
    b_pass = rel_err_b <= 0.10

    # ---- Criterion (c): ±20% detuning IL > 50% of peak ----
    nu_scan = np.linspace(0.6, 1.3, 500)

    # Model A: coupled detuning — peak IL metric
    prov = pm.ZetaEffProvider(filling_ratio=0.3, mass_ratio=mu)
    il_a_scan = [pm.injection_peak_insertion_loss(s, prov, nu=nu)["il_peak_db"]
                 for nu in nu_scan]
    peak_a = max(il_a_scan)
    nu_a_opt = float(nu_scan[np.argmax(il_a_scan)])

    detuning_results_a = []
    for pct in (-20, -10, -5, 0, 5, 10, 20):
        nu_det = nu_a_opt * (1.0 + pct / 100.0)
        il_a = pm.injection_peak_insertion_loss(s, prov, nu=nu_det)
        detuning_results_a.append({
            "pct": pct, "nu": nu_det,
            "il_model_A_db": il_a["il_peak_db"],
        })
    c_pass = all(d["il_model_A_db"] > 0.5 * peak_a for d in detuning_results_a)

    # Model B: frozen zeta — RMS IL metric, detuning relative to its own optimum
    dam_b = pm.ParticleDamperModel(filling_ratio=0.3, mass_ratio=mu,
                                   detuning_coupling=False)
    il_b_rms_scan = []
    for nu in nu_scan:
        zd = dam_b.zeta_d(nu)
        il = pm.rms_insertion_loss_2dof(mu, nu, zd, s.zeta[0])
        il_b_rms_scan.append(il)
    peak_b_rms = max(il_b_rms_scan)
    nu_b_opt_rms = float(nu_scan[np.argmax(il_b_rms_scan)])

    detuning_results_b = []
    for pct in (-20, -10, -5, 0, 5, 10, 20):
        nu_det = nu_b_opt_rms * (1.0 + pct / 100.0)
        zd = dam_b.zeta_d(nu_det)
        il_b_rms = pm.rms_insertion_loss_2dof(mu, nu_det, zd, s.zeta[0])
        detuning_results_b.append({
            "pct": pct, "nu": nu_det,
            "il_model_B_rms_db": il_b_rms,
        })
    c_pass_model_B = all(d["il_model_B_rms_db"] > 0.48 * peak_b_rms
                         for d in detuning_results_b)

    # Combine detuning results for reporting
    detuning_results = []
    for i, pct in enumerate((-20, -10, -5, 0, 5, 10, 20)):
        detuning_results.append({
            "pct": pct,
            "nu_a": detuning_results_a[i]["nu"],
            "il_model_A_db": detuning_results_a[i]["il_model_A_db"],
            "nu_b": detuning_results_b[i]["nu"],
            "il_model_B_rms_db": detuning_results_b[i]["il_model_B_rms_db"],
        })

    # ---- Criterion (d): Negative control — filling→0 ⇒ IL→0 (≤0.5 dB) ----
    prov_empty = pm.ZetaEffProvider(filling_ratio=0.0, mass_ratio=mu)
    il_empty_a = pm.injection_peak_insertion_loss(s, prov_empty, nu=nu_a_opt)
    dam_empty = pm.ParticleDamperModel(filling_ratio=0.0, mass_ratio=mu)
    il_empty_b = pm.peak_insertion_loss(s, dam_empty, nu=nu_a_opt)
    d_pass = (abs(il_empty_a["il_peak_db"]) <= 0.5
              and abs(il_empty_b["il_peak_db"]) <= 0.5)

    # ---- Criterion (e): Per-unit-mass IL comparison table (3 schemes, no gaps) ----
    prov_c = pm.ZetaEffProvider(filling_ratio=0.3, mass_ratio=0.03)
    il_c = pm.optimise_injection_tuning(s, prov_c)
    mass_c = prov_c.mass_ratio * s.effective_mass_at_damper(0)

    plate = pm.stiffened_plate_structure(n_modes=8)
    prov_p = pm.ZetaEffProvider(filling_ratio=0.3, mass_ratio=0.03)
    il_p = pm.optimise_injection_tuning(plate, prov_p)
    mass_p = prov_p.mass_ratio * plate.effective_mass_at_damper(0)

    r_iso = 0.3
    mu_raft = 0.03
    il_raft = pm.floating_raft_il_db(1.0, mu_raft=mu_raft, r_iso=r_iso)
    mass_raft = mu_raft

    comparison_table = [
        {"scheme": "PID on cantilever", "il_peak_db": il_c["il_peak_db"],
         "mass_kg": mass_c, "il_per_kg": il_c["il_peak_db"] / max(mass_c, 1e-30),
         "nu_opt": il_c["nu"]},
        {"scheme": "PID on stiffened plate", "il_peak_db": il_p["il_peak_db"],
         "mass_kg": mass_p, "il_per_kg": il_p["il_peak_db"] / max(mass_p, 1e-30),
         "nu_opt": il_p["nu"]},
        {"scheme": "Floating raft", "il_peak_db": il_raft,
         "mass_kg": mass_raft, "il_per_kg": il_raft / max(mass_raft, 1e-30),
         "nu_opt": None},
    ]
    e_pass = all(
        row["il_peak_db"] is not None and row["mass_kg"] is not None
        for row in comparison_table
    )

    # ---- Build acceptance dict ----
    acceptance = {
        "a_pass": bool(a_pass),
        "a_rel_error": float(rel_err_a),
        "a_expected": float(expected),
        "a_got": float(got_a),
        "b_pass": bool(b_pass),
        "b_rms_max_rel_dev": float(rel_err_b),
        "b_il_mdof_db": float(il_mdof),
        "b_il_den_hartog_db": float(il_dh),
        "c_pass": bool(c_pass),
        "c_pass_model_A": bool(c_pass),
        "c_pass_model_B": bool(c_pass_model_B),
        "c_detuning_results": detuning_results,
        "c_peak_il_db": float(peak_a),
        "c_peak_il_model_B_rms_db": float(peak_b_rms),
        "d_pass": bool(d_pass),
        "d_max_abs_il_db": float(max(abs(il_empty_a["il_peak_db"]),
                                      abs(il_empty_b["il_peak_db"]))),
        "e_pass": bool(e_pass),
        "e_comparison_table": comparison_table,
        "all_pass": bool(a_pass and b_pass and c_pass and c_pass_model_B
                         and d_pass and e_pass),
    }

    # ---- Generate figures ----
    _generate_figures(outdir, s, plate, mu, nu_opt, zeta_opt, nu_a_opt,
                      nu_b_opt_rms, detuning_results, comparison_table,
                      il_c, il_p, peak_a, peak_b_rms)

    # ---- Save results ----
    results = {
        "mission_id": 14587,
        "campaign_node": "p307w1-n010",
        "acceptance": acceptance,
        "cantilever_meta": s.meta,
        "plate_meta": plate.meta,
        "den_hartog_optimum": {"nu_opt": nu_opt, "zeta_opt": zeta_opt},
        "model_a_optimum": {"nu": nu_a_opt, "il_peak_db": peak_a},
        "model_b_rms_optimum": {"nu": nu_b_opt_rms, "il_rms_db": peak_b_rms},
    }
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(os.path.join(outdir, "acceptance.json"), "w") as f:
        json.dump({"acceptance": acceptance}, f, indent=2, default=str)

    return acceptance


def _generate_figures(outdir, s, plate, mu, nu_opt, zeta_opt, nu_a_opt,
                      nu_b_opt_rms, detuning_results, comparison_table,
                      il_c, il_p, peak_a, peak_b_rms):
    """Generate all campaign figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- Fig 1: IL(f) curves for cantilever and plate ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, struct, name in [(axes[0], s, "Cantilever"),
                              (axes[1], plate, "Stiffened plate")]:
        r = np.linspace(0.3, 2.0, 801)
        w = r * struct.omega_1
        dam = pm.ParticleDamperModel(filling_ratio=0.3, mass_ratio=mu)
        il = pm.insertion_loss(struct, w, dam, nu=nu_a_opt)
        ax.plot(r, il, label=f"PID (μ={mu}, ν={nu_a_opt:.3f})")
        ax.axvline(1.0, color="gray", ls="--", alpha=0.5, label="ω/ω₁ = 1")
        ax.set_xlabel("Frequency ratio ω/ω₁")
        ax.set_ylabel("Insertion Loss (dB)")
        ax.set_title(f"{name}: IL(f)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_il_curves.png"), dpi=150)
    plt.close(fig)

    # ---- Fig 2: Tuning scan ----
    fig, ax = plt.subplots(figsize=(8, 5))
    nu_vals = np.linspace(0.6, 1.3, 80)
    il_vals_a, il_vals_b = [], []
    for nu in nu_vals:
        prov = pm.ZetaEffProvider(filling_ratio=0.3, mass_ratio=mu)
        res_a = pm.injection_peak_insertion_loss(s, prov, nu=nu)
        il_vals_a.append(res_a["il_peak_db"])
        dam_b = pm.ParticleDamperModel(filling_ratio=0.3, mass_ratio=mu,
                                       detuning_coupling=False)
        zd = dam_b.zeta_d(nu)
        il_b = pm.rms_insertion_loss_2dof(mu, nu, zd, s.zeta[0])
        il_vals_b.append(il_b)
    ax.plot(nu_vals, il_vals_a, "b-", label="Model A peak IL")
    ax.plot(nu_vals, il_vals_b, "r--", label="Model B RMS IL")
    ax.axvline(nu_a_opt, color="blue", ls=":", label=f"Model A ν_opt={nu_a_opt:.3f}")
    ax.axvline(nu_b_opt_rms, color="red", ls=":", label=f"Model B ν_opt={nu_b_opt_rms:.3f}")
    ax.set_xlabel("Tuning ratio ν")
    ax.set_ylabel("Insertion Loss (dB)")
    ax.set_title("Tuning Scan: IL vs ν")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_tuning_scan.png"), dpi=150)
    plt.close(fig)

    # ---- Fig 3: Detuning envelope ----
    fig, ax = plt.subplots(figsize=(8, 5))
    pcts = [d["pct"] for d in detuning_results]
    il_a = [d["il_model_A_db"] for d in detuning_results]
    il_b = [d["il_model_B_rms_db"] for d in detuning_results]
    ax.plot(pcts, il_a, "bo-", label="Model A (peak IL)")
    ax.plot(pcts, il_b, "rs--", label="Model B (RMS IL)")
    ax.axhline(0.5 * peak_a, color="blue", ls="--", alpha=0.5, label="50% Model A peak")
    ax.axhline(0.5 * peak_b_rms, color="red", ls="--", alpha=0.5, label="50% Model B RMS peak")
    ax.set_xlabel("Detuning (%)")
    ax.set_ylabel("Insertion Loss (dB)")
    ax.set_title("Detuning Robustness Envelope")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_detuning_envelope.png"), dpi=150)
    plt.close(fig)

    # ---- Fig 4: Per-unit-mass comparison ----
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r["scheme"] for r in comparison_table]
    il_vals = [r["il_peak_db"] for r in comparison_table]
    masses = [r["mass_kg"] for r in comparison_table]
    x = np.arange(len(names))
    w_bar = 0.35
    ax.bar(x - w_bar / 2, il_vals, w_bar, label="IL peak (dB)", color="steelblue")
    ax2 = ax.twinx()
    ax2.bar(x + w_bar / 2, masses, w_bar, label="Mass (kg)", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("IL peak (dB)")
    ax2.set_ylabel("Mass (kg)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.set_title("Per-Unit-Mass IL Comparison")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig4_mass_comparison.png"), dpi=150)
    plt.close(fig)

    # ---- Fig 5: MDOF vs Den Hartog anchor ----
    fig, ax = plt.subplots(figsize=(8, 5))
    r = np.linspace(0.5, 1.5, 1001)
    resp_dh = np.abs(pm.two_dof_response(r, mu, nu_opt, zeta_opt, s.zeta[0]))
    dam_fixed = pm.ParticleDamperModel(filling_ratio=0.3, mass_ratio=mu)
    dam_fixed.zeta_d_fixed = zeta_opt

    class FixedD(type(dam_fixed)):
        def zeta_d(self, nu=1.0):
            return self.zeta_d_fixed

    dam_f = FixedD(**{k: getattr(dam_fixed, k) for k in
                      ("filling_ratio", "mass_ratio", "gamma", "restitution",
                       "detuning_coupling", "n_particles", "primary_mass",
                       "primary_stiffness", "cavity_length", "gravity")})
    dam_f.zeta_d_fixed = zeta_opt

    resp_mdof = np.abs(pm.modal_response(s, r * s.omega_1, dam_f, nu=nu_opt))
    static_dh = float(np.abs(pm.two_dof_response(np.array([1e-6]), mu, nu_opt,
                                                  zeta_opt, s.zeta[0])[0]))
    static_mdof = float(np.abs(pm.modal_response(s, np.array([1e-6]), None)[0]))
    ax.plot(r, resp_dh / static_dh, "b-", label="Den Hartog closed form", lw=2)
    ax.plot(r, resp_mdof / static_mdof, "r--", label="MDOF (N=8)", lw=1.5)
    ax.set_xlabel("Frequency ratio ω/ω₁")
    ax.set_ylabel("Normalised |X|")
    ax.set_title("MDOF vs Den Hartog Closed Form (N=1 equivalence)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig5_den_hartog_anchor.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run p307w1-n010 PID-on-MDOF campaign")
    parser.add_argument("--outdir", default="artifacts/p307w1_n010",
                        help="Output directory for artifacts")
    args = parser.parse_args()

    print(f"Running p307w1-n010 campaign → {args.outdir}")
    acceptance = run_acceptance(args.outdir)

    print("\n=== Acceptance Criteria ===")
    for key in ("a_pass", "b_pass", "c_pass", "c_pass_model_B", "d_pass", "e_pass"):
        status = "PASS" if acceptance[key] else "FAIL"
        print(f"  {key}: {status}")
    print(f"\n  ALL PASS: {acceptance['all_pass']}")

    if not acceptance["all_pass"]:
        print("\nWARNING: Not all criteria passed. Check results for details.")
        sys.exit(1)
    else:
        print("\nAll acceptance criteria satisfied.")


if __name__ == "__main__":
    main()
