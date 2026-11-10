"""Figures for the P82W5-N07 acceptance gate.

Reads ``artifacts/P82W5_N07/acceptance.json`` (produced by
``benchmarks/mts_speedup_gate.py``) and writes the PNGs next to it.  Every number
is taken from the JSON -- nothing is retyped in this file, so the figures cannot
drift from the report table.

    python scripts/mts_speedup_gate_figures.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MISSION = "P82W5_N07"
ART = ROOT / "artifacts" / MISSION


def _load() -> dict:
    path = ART / "acceptance.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run benchmarks/mts_speedup_gate.py first")
    return json.loads(path.read_text())


def fig_restitution(data: dict) -> Path:
    ladder = data["a_restitution_ladder"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axvspan(0.0, 0.05, color="tab:green", alpha=0.10, label="gate: single step (ratio ≤ 0.05)")
    ax.axvspan(0.05, 0.5, color="tab:orange", alpha=0.08, label="gate: sub-cycling required")
    ax.axvspan(0.5, 0.6, color="tab:red", alpha=0.15, label="gate: inadmissible (≥ 0.5)")
    for e_target, marker in ((0.5, "o"), (0.9, "s")):
        rows = [c for c in ladder["cases"] if c["e_target"] == e_target]
        x = [c["dt_over_dt_c"] for c in rows]
        y = [100.0 * c["restitution_error"] for c in rows]
        ax.plot(x, y, marker=marker, label=f"e = {e_target}")
    ax.axhline(2.0, color="k", ls="--", lw=1.0, label="criterion: 2 %")
    ax.set_xlabel(r"requested $\Delta t / t_c$")
    ax.set_ylabel("restitution error [%]")
    ax.set_xlim(0.0, 0.6)
    ax.set_title(f"criterion (a): worst error {100 * ladder['worst_error']:.2f} %")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    out = ART / "fig1_restitution_ladder.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_ledger(data: dict) -> Path:
    ledger = data["b_energy_ledger"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for mode, style in (("single", "-"), ("mts", "--")):
        hist = ledger["history"][mode]
        t = [row["time"] for row in hist]
        ax = axes[0]
        ax.plot(t, [100.0 * row["energy_drift"] for row in hist], style, label=mode)
        ax2 = axes[1]
        ax2.plot(t, [row["contact_fraction"] for row in hist], style, label=mode)
    axes[0].axhline(1.0, color="k", ls=":", label="criterion 1 %")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("|ΔE| / E$_0$  [%]")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].set_title("criterion (b): energy ledger")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("active-contact fraction")
    axes[1].legend(fontsize=8)
    axes[1].set_title("contact state over the same span")
    fig.tight_layout()
    out = ART / "fig2_energy_ledger.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_speed(data: dict) -> Path:
    speed = data["c1_speed"]
    names = list(speed["variants"])
    wall_single = [speed["variants"][n]["wall_single"] for n in names]
    wall_mts = [speed["variants"][n]["wall_mts"] for n in names]
    gain = [speed["variants"][n]["speedup"] for n in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].bar(x - 0.2, wall_single, 0.4, label="single-rate")
    axes[0].bar(x + 0.2, wall_mts, 0.4, label="multi-rate")
    axes[0].set_xticks(x, names, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("wall clock for the same physical span [s]")
    axes[0].legend(fontsize=8)
    axes[0].set_title("interleaved median, same span")
    axes[1].bar(x, gain, 0.5, color=["tab:grey", "tab:blue", "tab:cyan", "tab:green"])
    axes[1].axhline(2.0, color="k", ls="--", label="criterion 2×")
    axes[1].set_xticks(x, names, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("speed-up")
    for xi, g in zip(x, gain):
        axes[1].text(xi, g + 0.05, f"{g:.2f}×", ha="center", fontsize=8)
    axes[1].legend(fontsize=8)
    axes[1].set_title(f"criterion (c): matured {speed['speedup']:.2f}×, "
                      f"legacy {speed['speedup_legacy']:.2f}×")
    fig.tight_layout()
    out = ART / "fig3_speedup_ablation.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_jamming(data: dict) -> Path:
    jam = data["c2_jamming"]
    anchor = jam.get("w4_p1_servo_anchor", {})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for mode, style in (("single", "o-"), ("mts", "s--")):
        rows = jam["cases"]["hierarchical"][mode]["levels"]
        axes[0].plot([r[0] for r in rows], [r[1] for r in rows], style, label=f"hierarchical {mode}")
    if anchor:
        axes[0].plot([r[0] for r in anchor["levels"]], [r[1] for r in anchor["levels"]],
                     "^-", color="k", label="W4-P1 servo anchor")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("pressure P [N/m]")
    axes[0].set_ylabel(r"area fraction $\varphi$")
    axes[0].legend(fontsize=8)
    axes[0].set_title(r"$\varphi(P)$ decompression ladder")

    labels = ["anchor"]
    values = [anchor["phi_j"]] if anchor else []
    for label in ("hierarchical", "uniform"):
        for mode in ("single", "mts"):
            labels.append(f"{label}\n{mode}")
            values.append(jam["cases"][label][mode]["phi_j"])
    axes[1].bar(np.arange(len(values)), values, 0.6,
                color=["k"] + ["tab:grey", "tab:green"] * 2)
    axes[1].set_xticks(np.arange(len(values)), labels, fontsize=7, rotation=25, ha="right")
    axes[1].set_ylim(min(values) - 0.01, max(values) + 0.005)
    axes[1].set_ylabel(r"$\varphi_J$")
    axes[1].set_title(
        r"$\Delta\varphi_J$ mts vs single = "
        f"{100 * jam['phi_j_rel_diff']:.3f} % ; vs anchor = "
        f"{100 * (jam['phi_j_vs_anchor_rel_diff'] or float('nan')):.2f} %")
    fig.tight_layout()
    out = ART / "fig4_jamming_density.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_cost_model(data: dict) -> Path:
    """Cost per coarse step versus the analytic work model."""
    speed = data["c1_speed"]
    jam = data["c2_jamming"]
    n_sub = jam["cases"]["hierarchical"]["mts"]["gate"]["n_sub"]
    var = speed["variants"]["matured"]
    ratio = var["cost_ratio_per_coarse_step"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    budget = np.linspace(1.0, n_sub, 200)
    ax.plot(budget, n_sub / budget, "k-", label=r"$n_{sub}/\mathrm{cost\ ratio}$")
    ax.axhline(2.0, color="tab:red", ls="--", label="criterion 2×")
    ax.scatter([ratio], [var["speedup"]], s=90, color="tab:green", zorder=5,
               label=f"matured (ratio {ratio:.2f})")
    legacy = speed["variants"]["legacy"]
    ax.scatter([legacy["cost_ratio_per_coarse_step"]], [legacy["speedup"]], s=90,
               color="tab:grey", zorder=5,
               label=f"legacy (ratio {legacy['cost_ratio_per_coarse_step']:.2f})")
    ax.set_xlabel("cost of one coarse step / cost of one single-rate step")
    ax.set_ylabel("speed-up")
    ax.set_title(f"cost contract, $n_{{sub}}$ = {n_sub}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = ART / "fig5_cost_contract.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    data = _load()
    ART.mkdir(parents=True, exist_ok=True)
    made = []
    if "a_restitution_ladder" in data:
        made.append(fig_restitution(data))
    if "b_energy_ledger" in data:
        made.append(fig_ledger(data))
    if "c1_speed" in data:
        made.append(fig_speed(data))
    if "c2_jamming" in data:
        made.append(fig_jamming(data))
    if "c1_speed" in data and "c2_jamming" in data:
        made.append(fig_cost_model(data))
    for path in made:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())