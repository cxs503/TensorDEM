#!/usr/bin/env python3
"""Board-audit figure for m17628 (audit of m17503). Numbers come from
verification.json + refs/artifacts/results.json + refs/probe/probe_overlap_out.txt
— no hand-typed digits."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
rep = json.loads((HERE / "refs/artifacts/results.json").read_text())
ver = json.loads((HERE / "verification.json").read_text())
cases = {(c["gamma_dot"], c["arm"]): c for c in rep["cases"]}
gs = sorted({c["gamma_dot"] for c in rep["cases"]})

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

# panel 1: tau/P per arm + the 15 % bar as a band relative to the LE arm
w = 0.36
ax[0].bar([i - w / 2 for i in range(len(gs))],
          [cases[(g, "legacy")]["tau_over_p"] for g in gs], w, label="legacy SLLOD (no tilt, unwrapped field)")
ax[0].bar([i + w / 2 for i in range(len(gs))],
          [cases[(g, "lees_edwards")]["tau_over_p"] for g in gs], w, label="Lees-Edwards (opt-in, m17503)")
for i, g in enumerate(gs):
    ax[0].errorbar(i - w / 2, cases[(g, "legacy")]["tau_over_p"],
                   yerr=cases[(g, "legacy")]["shear_stderr"] / cases[(g, "legacy")]["pressure"],
                   fmt="none", ecolor="k", capsize=3)
    ax[0].errorbar(i + w / 2, cases[(g, "lees_edwards")]["tau_over_p"],
                   yerr=cases[(g, "lees_edwards")]["shear_stderr"] / cases[(g, "lees_edwards")]["pressure"],
                   fmt="none", ecolor="k", capsize=3)
ax[0].axhline(0, color="k", lw=0.8)
ax[0].set_xticks(range(len(gs)), [f"$\\dot\\gamma$={int(g)}" for g in gs])
ax[0].set_ylabel("$\\tau/P$")
ax[0].set_title("(a) macroscopic friction, equal strain window")
ax[0].legend(fontsize=7, loc="upper left")

# panel 2: relative deviation between the arms vs the 15 % criterion
rel = [ver["checks"]["C2_cross_table"]["rows"][i]["rel_diff_pct_recomputed"] for i in range(len(gs))]
bars = ax[1].bar([f"{int(g)}" for g in gs], rel, color="tab:red")
ax[1].axhline(15, color="k", ls="--", lw=1.2)
ax[1].text(0.02, 18, "acceptance bar 15 %", fontsize=8)
for b, v in zip(bars, rel):
    ax[1].text(b.get_x() + b.get_width() / 2, v * 1.03, f"{v:.1f}%", ha="center", fontsize=8)
ax[1].set_yscale("log")
ax[1].set_xlabel("$\\dot\\gamma$ [1/s]")
ax[1].set_ylabel("|$\\tau/P$ legacy - LE| / |LE|  [%]")
ax[1].set_title("(b) XVAL criterion: FAIL at all three rates")

# panel 3: independent probe — max interpenetration per arm + frame-invariance residual
txt = (HERE / "refs/probe/probe_overlap_out.txt").read_text().strip().splitlines()
probe = {}
for ln in txt:
    d = {}
    for tok in ln.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    probe["lees_edwards" if d["lee"] == "True" else "legacy"] = d
ov = [float(probe["legacy"]["ov_max"].rstrip("%rad")), float(probe["lees_edwards"]["ov_max"].rstrip("%rad"))]
ax[2].bar([0, 1], ov, 0.5, color=["tab:blue", "tab:orange"])
for i, v in enumerate(ov):
    ax[2].text(i, v + 0.3, f"{v:.2f}% r", ha="center", fontsize=9)
ax[2].set_xticks([0, 1], ["legacy", "Lees-Edwards"])
ax[2].set_ylabel("max overlap / radius  [%]")
ax[2].set_title("(c) auditor probe: LE arm interpenetrates up to "
                f"{ov[1]:.1f}% r")
fig.suptitle("m17628 board audit — m17503 XVAL periodic-SLLOD vs Lees-Edwards crosscheck", fontsize=11)
fig.tight_layout()
out = HERE / "fig_m17628.png"
fig.savefig(out, dpi=130)
print("wrote", out, out.stat().st_size)