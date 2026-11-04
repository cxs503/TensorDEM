"""Benchmark: platen Coulomb friction vs its closed-form anchors.

Writes ``artifacts/platen_friction/verification_report.json`` (all numbers, so the
report tables are machine-extracted, never hand-copied) and a measured-vs-analytic
parity figure.  Run from the repository root:

    python benchmarks/platen_friction_verification.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.platen_friction import (  # noqa: E402
    ACCURACY_GATE,
    platen_friction_case,
)

OUT = ROOT / "artifacts" / "platen_friction"
MU_STICK = (0.15, 0.30)
MU_SLIP = (0.15, 0.30)
NORMAL = 10.0
STICK_RATIO = 0.6
SLIP_RATIO = 2.0
FAST = dict(settle_steps=400, steps=1800, ramp_steps=600)


def run() -> dict:
    rows = []
    for mu in MU_STICK:
        rows.append(platen_friction_case(
            phase="stick", mu_p=mu, normal_force=NORMAL,
            tangential_force=STICK_RATIO * mu * NORMAL, **FAST))
    slip = []
    for mu in MU_SLIP:
        slip.append(platen_friction_case(
            phase="slip", mu_p=mu, normal_force=NORMAL,
            tangential_force=SLIP_RATIO * mu * NORMAL, drag=0.0,
            ramp_steps=50, steps=600, settle_steps=400,
            platen_damping=5.0))
    top = platen_friction_case(
        phase="stick", mu_p=0.5, normal_force=12.0, tangential_force=5.0,
        platen="top", **FAST)

    summary = {
        "accuracy_gate": ACCURACY_GATE,
        "stick": [{"mu_p": r["mu_p"],
                   "measured_m": r["measured"]["tangential_offset_m"],
                   "predicted_m": r["predicted"]["tangential_offset_m"],
                   "relative_error": r["relative_error"]["tangential_offset_m"],
                   "peak_friction_ratio": r["measured"]["peak_friction_ratio"],
                   "final_speed_m_per_s": r["measured"]["final_speed_m_per_s"],
                   "normal_force_N": r["normal_force_N"]} for r in rows],
        "slip": [{"mu_p": r["mu_p"],
                  "measured_m_per_s2": r["measured"]["velocity_slope_m_per_s2"],
                  "predicted_m_per_s2": r["predicted"]["velocity_slope_m_per_s2"],
                  "relative_error": r["relative_error"]["velocity_slope_m_per_s2"],
                  "peak_friction_ratio": r["measured"]["peak_friction_ratio"],
                  "final_speed_m_per_s": r["measured"]["final_speed_m_per_s"],
                  "normal_force_N": r["normal_force_N"]} for r in slip],
        "top_platen": {"mu_p": top["mu_p"],
                       "measured_m": top["measured"]["tangential_offset_m"],
                       "predicted_m": top["predicted"]["tangential_offset_m"],
                       "measurement_relative_error": top["relative_error"][
                           "tangential_offset_m"]},
        "all_within_gate": all(
            error < ACCURACY_GATE
            for error in [r["relative_error"]["tangential_offset_m"] for r in rows]
            + [r["relative_error"]["velocity_slope_m_per_s2"] for r in slip]),
    }
    return summary


def figure(summary: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    stick_m = [r["measured_m"] for r in summary["stick"]]
    stick_p = [r["predicted_m"] for r in summary["stick"]]
    slip_m = [r["measured_m_per_s2"] for r in summary["slip"]]
    slip_p = [r["predicted_m_per_s2"] for r in summary["slip"]]

    for ax, meas, pred, label, unit in (
        (axes[0], stick_m, stick_p, "stick: $\\delta_x = F_h/k_t$", "offset [m]"),
        (axes[1], slip_m, slip_p, "slip: $a=(F_h-\\mu_p N)/m$", "acceleration [m/s$^2$]"),
    ):
        lo = min(min(meas), min(pred)) * 0.9
        hi = max(max(meas), max(pred)) * 1.1
        ax.plot([lo, hi], [lo, hi], "--", color="0.4", label="analytic = measured")
        ax.plot(pred, meas, "o", ms=8, color="#1f77b4")
        for row in (summary["stick"] if unit.startswith("offset") else summary["slip"]):
            key = "mu_p"
            ax.annotate(f"$\\mu_p$={row[key]}", (row["predicted_m"] if unit.startswith("offset")
                        else row["predicted_m_per_s2"],
                        row["measured_m"] if unit.startswith("offset")
                        else row["measured_m_per_s2"]),
                        textcoords="offset points", xytext=(6, -10), fontsize=8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"analytic {unit}"); ax.set_ylabel(f"measured {unit}")
        ax.set_title(label); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    labels = ([f"stick $\\mu_p$={r['mu_p']}" for r in summary["stick"]]
              + [f"slip $\\mu_p$={r['mu_p']}" for r in summary["slip"]])
    errors = ([r["relative_error"] for r in summary["stick"]]
              + [r["relative_error"] for r in summary["slip"]])
    idx = range(len(labels))
    axes[2].bar(idx, errors, color="#2ca02c")
    axes[2].axhline(summary["accuracy_gate"], color="crimson", ls="--",
                    label=f"{int(summary['accuracy_gate']*100)} % gate")
    axes[2].set_yscale("log"); axes[2].set_xticks(list(idx))
    axes[2].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axes[2].set_ylabel("relative error vs closed form")
    axes[2].set_title("every anchor inside the gate"); axes[2].legend(fontsize=8)
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle("DEM3D platen Coulomb friction: measured vs closed-form anchors", fontsize=12)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "platen_friction_anchors.png"
    fig.savefig(path, dpi=140)
    return path


if __name__ == "__main__":
    summary = run()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verification_report.json").write_text(json.dumps(summary, indent=2))
    png = figure(summary)
    print(json.dumps(summary, indent=2))
    print("figure:", png)