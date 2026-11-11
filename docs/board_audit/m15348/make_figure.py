#!/usr/bin/env python3
"""Figure for m15348: m15259 anchor case — filed vs reproducible.

Left  : per-window discharge rate of the only committed m15259 run vs the
        Beverloo band (C = 0.55-0.60) evaluated at the same case parameters.
Right : relative error of (a) the filed headline (hand-transcribed m6457 value),
        (b) the run artifact this mission actually produced, against the 15 %
        anchor threshold.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RUN = Path("/root/tp-agent/work/m15259/runs/beverloo_m15259/local_med.json")
OUT = ROOT / "docs/board_audit/m15348/m15348_m15259_anchor_check.png"

rec = json.loads(RUN.read_text())
D, d, rho_b = rec["D"], rec["d"], rec["rho_b"]
deff = D - 1.4 * d
band = [C * rho_b * math.sqrt(9.81) * deff ** 2.5 for C in (0.55, 0.60)]
filed = json.loads((ROOT / "docs/board_audit/m15348/audit_m15259.json").read_text())
rel_filed = 1.8290 / 1.8356 - 1.0
rel_run = rec["rel_err"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
w = rec["windows"]
ax1.bar(range(1, len(w) + 1), w, color="#4c72b0", label="DEM window rate")
ax1.axhspan(band[0], band[1], color="#dd8452", alpha=0.35,
            label=f"Beverloo C=0.55-0.60 band ({band[0]:.2f}-{band[1]:.2f} kg/s)")
ax1.set_xlabel("flow-rate window #")
ax1.set_ylabel("W (kg/s)")
ax1.set_title(f"m15259 run artifact: D/d={rec['D_over_d']:.1f}, "
              f"mean W={rec['W_meas']:.2f} kg/s")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

labels = ["filed headline\n(m6457 case C,\nhand-transcribed)", "m15259 own run\n(only artifact)"]
vals = [rel_filed * 100, rel_run * 100]
ax2.bar(labels, vals, color=["#55a868", "#c44e52"])
ax2.axhline(15, ls="--", c="k", lw=1, label="+15 % anchor threshold")
ax2.axhline(-15, ls="--", c="k", lw=1)
ax2.set_ylabel("relative error vs Beverloo (%)")
ax2.set_title("anchor criterion: |error| < 15 %")
for i, v in enumerate(vals):
    ax2.annotate(f"{v:+.1f}%", (i, v), ha="center",
                 va="bottom" if v > 0 else "top", fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
fig.suptitle("CASE-HOPPER-BEVERLOO / TensorDEM m15259 — board audit (m15348)", fontsize=11)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140)
print("wrote", OUT, "band", band, "filed", rel_filed, "run", rel_run)