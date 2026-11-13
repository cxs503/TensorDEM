#!/usr/bin/env python
"""Generate figures for the XVAL report: TensorDEM vs Jiang-2005 EPSD rolling spring."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "xval_jiang2005_rolling_spring"
FIG_DIR = ART / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load summary
summary = json.loads((ART / "summary.json").read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# Figure 1: Stiffness comparison
# ---------------------------------------------------------------------------
def fig_stiffness_comparison():
    mu_r_vals = [0.05, 0.1, 0.2, 0.3, 0.5]
    k_tensordem = [2.25 * mu * 2e4 * 0.01**2 for mu in mu_r_vals]
    k_jiang = [2.25 * mu**2 * 2e4 * 0.01**2 for mu in mu_r_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(mu_r_vals))
    width = 0.35

    bars1 = ax.bar(x - width/2, k_tensordem, width, label="TensorDEM: k_r = 2.25·μ_r·k_n·R²", color="#2196F3")
    bars2 = ax.bar(x + width/2, k_jiang, width, label="Jiang-2005: k_r = 2.25·μ_r²·k_n·R²", color="#FF5722")

    ax.set_xlabel("Rolling friction coefficient μ_r", fontsize=12)
    ax.set_ylabel("Rolling stiffness k_r (N·m/rad)", fontsize=12)
    ax.set_title("Rolling Stiffness: TensorDEM vs Jiang-2005", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{mu:.2f}" for mu in mu_r_vals])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Add ratio labels
    for i, (kt, kj) in enumerate(zip(k_tensordem, k_jiang)):
        ratio = kj / kt if kt > 0 else 0
        ax.text(i, max(kt, kj) * 1.05, f"ratio={ratio:.2f}", ha="center", fontsize=9, color="#666")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_stiffness_comparison.png", dpi=150)
    plt.close()
    print("  Saved fig1_stiffness_comparison.png")


# ---------------------------------------------------------------------------
# Figure 2: Acceleration parity
# ---------------------------------------------------------------------------
def fig_acceleration_parity():
    accel = summary["acceleration_parity"]
    # Filter to rolling regime only
    rolling = [r for r in accel if r["regime"] == "rolling"]

    mu_r_groups = {}
    for r in rolling:
        mu = r["mu_r"]
        if mu not in mu_r_groups:
            mu_r_groups[mu] = {"tensordem": [], "jiang2005": [], "analytic": [], "theta_over": []}
        rule = r["rule"]
        mu_r_groups[mu][rule].append(r["a_simulated_m_s2"])
        mu_r_groups[mu]["theta_over"].append(r["theta_over_theta_c"])
        # Analytic is the same for both rules
        if not mu_r_groups[mu]["analytic"] or r["a_analytic_m_s2"] not in mu_r_groups[mu]["analytic"]:
            mu_r_groups[mu]["analytic"].append(r["a_analytic_m_s2"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, (mu, data) in enumerate(sorted(mu_r_groups.items())):
        ax = axes[idx]
        x_td = data["theta_over"][:len(data["tensordem"])]
        x_jg = data["theta_over"][:len(data["jiang2005"])]
        x_an = data["theta_over"][:len(data["analytic"])]
        ax.plot(x_an, data["analytic"], "k--", label="Analytic", linewidth=2)
        ax.plot(x_td, data["tensordem"], "bo-", label="TensorDEM", markersize=4)
        ax.plot(x_jg, data["jiang2005"], "rs-", label="Jiang-2005", markersize=4)
        ax.set_xlabel("θ / θ_c", fontsize=10)
        ax.set_ylabel("a (m/s²)", fontsize=10)
        ax.set_title(f"μ_r = {mu:.2f}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_visible(False)
    axes[-2].set_visible(False)
    axes[-3].set_visible(False)

    fig.suptitle("Acceleration Parity: Simulated vs Analytic (steady state)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_acceleration_parity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig2_acceleration_parity.png")


# ---------------------------------------------------------------------------
# Figure 3: Transient dynamics - oscillation frequency
# ---------------------------------------------------------------------------
def fig_transient_dynamics():
    transient = summary["transient_dynamics"]

    mu_r_vals = sorted(set(r["mu_r"] for r in transient))
    x = np.arange(len(mu_r_vals))
    width = 0.35

    freq_td = [next(r["freq_analytic_hz"] for r in transient if r["mu_r"] == mu and r["rule"] == "tensordem") for mu in mu_r_vals]
    freq_jg = [next(r["freq_analytic_hz"] for r in transient if r["mu_r"] == mu and r["rule"] == "jiang2005") for mu in mu_r_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x - width/2, freq_td, width, label="TensorDEM", color="#2196F3")
    ax1.bar(x + width/2, freq_jg, width, label="Jiang-2005", color="#FF5722")
    ax1.set_xlabel("μ_r", fontsize=12)
    ax1.set_ylabel("Oscillation frequency (Hz)", fontsize=12)
    ax1.set_title("Analytic Oscillation Frequency", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{mu:.2f}" for mu in mu_r_vals])
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Oscillation amplitude
    amp_td = [next(r["osc_amplitude_m_s"] for r in transient if r["mu_r"] == mu and r["rule"] == "tensordem") for mu in mu_r_vals]
    amp_jg = [next(r["osc_amplitude_m_s"] for r in transient if r["mu_r"] == mu and r["rule"] == "jiang2005") for mu in mu_r_vals]

    ax2.bar(x - width/2, amp_td, width, label="TensorDEM", color="#2196F3")
    ax2.bar(x + width/2, amp_jg, width, label="Jiang-2005", color="#FF5722")
    ax2.set_xlabel("μ_r", fontsize=12)
    ax2.set_ylabel("Oscillation amplitude (m/s)", fontsize=12)
    ax2.set_title("Velocity Oscillation Amplitude", fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{mu:.2f}" for mu in mu_r_vals])
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_transient_dynamics.png", dpi=150)
    plt.close()
    print("  Saved fig3_transient_dynamics.png")


# ---------------------------------------------------------------------------
# Figure 4: Torque evolution
# ---------------------------------------------------------------------------
def fig_torque_evolution():
    torque = summary["torque_evolution"]
    mu_r = 0.2
    limit = next(r["limit_Nm"] for r in torque if r["mu_r"] == mu_r)

    omega_vals = sorted(set(r["omega_rad_s"] for r in torque if r["mu_r"] == mu_r))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(omega_vals))
    width = 0.35

    steps_td = [next(r["steps_to_99pct_limit"] for r in torque if r["mu_r"] == mu_r and r["rule"] == "tensordem" and r["omega_rad_s"] == w) for w in omega_vals]
    steps_jg = [next(r["steps_to_99pct_limit"] for r in torque if r["mu_r"] == mu_r and r["rule"] == "jiang2005" and r["omega_rad_s"] == w) for w in omega_vals]

    ax.bar(x - width/2, steps_td, width, label="TensorDEM", color="#2196F3")
    ax.bar(x + width/2, steps_jg, width, label="Jiang-2005", color="#FF5722")
    ax.set_xlabel("Prescribed spin ω (rad/s)", fontsize=12)
    ax.set_ylabel("Steps to 99% plastic limit", fontsize=12)
    ax.set_title(f"Torque Buildup (μ_r = {mu_r}, limit = {limit:.4f} N·m)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w:.1f}" for w in omega_vals])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add ratio labels
    for i, (st, sj) in enumerate(zip(steps_td, steps_jg)):
        ratio = sj / st if st > 0 else 0
        ax.text(i, max(st, sj) * 1.05, f"×{ratio:.1f}", ha="center", fontsize=10, color="#666")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_torque_evolution.png", dpi=150)
    plt.close()
    print("  Saved fig4_torque_evolution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Generating figures...")
    fig_stiffness_comparison()
    fig_acceleration_parity()
    fig_transient_dynamics()
    fig_torque_evolution()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
