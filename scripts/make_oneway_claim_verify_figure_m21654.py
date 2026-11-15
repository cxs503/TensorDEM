"""Figure for the GAP-ICE-ONEWAY-CLAIM verification (m21654).

Panel (a): deflection gate response as the FEM beam stiffness is detuned -- the
1 % gate is one-sided, so a stiffer beam is caught at ~+0.5 % while a softer beam
down to ~-1.8 % reads *better* than the shipped number.
Panel (b): delivered dt-refinement chain vs the registered acceptance chain.

All numbers are read from ``runs/oneway_claim_verify_m21654/probe.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "runs" / "oneway_claim_verify_m21654" / "probe.json"
OUT = ROOT / "docs" / "figures" / "oneway_claim_verify_m21654.png"


def main() -> Path:
    data = json.loads(PROBE.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # ---- (a) gate response vs stiffness detuning
    rows = data["P3_gate_bite"]["stiffness_scale"]
    xs, ys = [], []
    for key, row in rows.items():
        xs.append(float(key.split("_x")[1]))
        ys.append(100.0 * row["error_vs_anchor"])
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    ax = axes[0]
    ax.axhline(1.0, color="crimson", ls="--", lw=1.2, label="1 % gate")
    ax.plot(xs, ys, "o-", color="#1f4e79", lw=1.4, ms=5,
            label="FEM peak vs modal anchor")
    for x, y in zip(xs, ys):
        if x in (1.0, 1.005, 0.99):
            ax.annotate(f"{y:.3f}%", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("beam stiffness multiplier applied to the FEM only")
    ax.set_ylabel("centre-deflection error [%]")
    ax.set_title("(a) 1 % deflection gate: one-sided response\n"
                 "softer FEM up to -1.8 % reads *better*; +0.5 % already red",
                 fontsize=9.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # ---- (b) dt chain: delivered vs registered
    ax = axes[1]
    delivered = [100.0 * e for e in data["P4_registered_chain_search"]["delivered_pct"]]
    registered = data["P4_registered_chain_search"]["registered_pct"]
    closest = data["P4_registered_chain_search"]["closest_variant"]
    facs = [1, 2, 4, 8]
    ax.plot(facs, delivered, "s-", color="#1f4e79", lw=1.4, ms=6,
            label="delivered (n_modes=9, anchor per level)")
    ax.plot(facs, registered, "o--", color="darkorange", lw=1.2, ms=5,
            label="registered acceptance chain")
    ax.plot(facs, closest["chain_pct"], "^:", color="seagreen", lw=1.1, ms=5,
            label=f"closest variant ({closest['variant']})")
    ax.set_yscale("log")
    ax.set_xticks(facs)
    ax.set_xticklabels([f"dt/{f}" for f in facs])
    ax.set_xlabel("time-step refinement level")
    ax.set_ylabel("peak-deflection error vs modal anchor [%]")
    ax.set_title("(b) dt chain: monotone delivered chain vs the\n"
                 "non-monotone registered numbers (not reproduced)",
                 fontsize=9.5)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")
    return OUT


if __name__ == "__main__":
    main()