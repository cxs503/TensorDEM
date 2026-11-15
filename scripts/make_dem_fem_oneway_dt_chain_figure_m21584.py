"""Figure for GAP-ICE-ONEWAY-CLAIM: dt refinement chain vs the modal anchor.

Reads ``examples/dem_fem_oneway/results.json`` (no hand-copied numbers) and
writes ``docs/figures/dem_fem_oneway_dt_chain_m21584.png``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
res = json.loads((ROOT / "examples/dem_fem_oneway/results.json").read_text())
chain = res["dt_convergence"]

ratios = np.array([r["dt_over_t1"] for r in chain])
peaks = np.array([r["peak_deflection_m"] for r in chain])
anchors = np.array([r["modal_anchor_peak_m"] for r in chain])
errors = np.array([r["deflection_error_vs_modal_relative"] for r in chain])
v = res["verification"]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

ax = axes[0]
ax.plot(ratios, peaks * 1e8, "o-", label="FEM (Hermite, Newmark)", color="#1f77b4")
ax.plot(ratios, anchors * 1e8, "s--", label="continuum modal superposition", color="#d62728")
ax.axhline(res["fem"]["peak_deflection_m"] * 1e8, ls=":", lw=0.8, color="0.5")
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel(r"$dt/T_1$  (coarser $\rightarrow$)")
ax.set_ylabel(r"peak centre deflection  [$\times 10^{-8}$ m]")
ax.set_title("(a) four-level $dt$ chain vs independent anchor")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)

ax = axes[1]
ax.loglog(ratios, errors * 100, "o-", color="#2ca02c", label="measured error")
ref = errors[0] * (ratios / ratios[0]) ** 2 * 100
ax.loglog(ratios, ref, "k--", lw=1.0, label=r"2nd-order reference $\propto (dt/T_1)^2$")
ax.set_xlabel(r"$dt/T_1$")
ax.set_ylabel("peak-deflection error vs anchor [%]")
ax.set_title("(b) error decays to the %.3f%% floor at $dt/4$" % (100 * errors[2]))
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
for x, y in zip(ratios, errors * 100):
    ax.annotate(f"{y:.3f}%", (x, y), textcoords="offset points", xytext=(4, 5), fontsize=7)

fig.suptitle(
    "DEM-FEM one-way coupling: deflection %s (gate <=1%%), port parity %s [numerical_verification_only]"
    % (f"{100 * v['deflection_error_relative']:.3f}%",
       f"{v['force_history_port_parity_error_relative']:.1e}"),
    fontsize=9,
)
fig.tight_layout()
out = ROOT / "docs" / "figures" / "dem_fem_oneway_dt_chain_m21584.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=140)
print("wrote", out)