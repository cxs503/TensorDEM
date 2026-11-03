"""Figures for ``docs/dem2d-arc-bow/study.json`` (report figures)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "docs" / "dem2d-arc-bow"
d = json.loads((OUT / "study.json").read_text())

rows = d["geometry"]["rows"]
chords = d["case"]["chords"]
ratios = d["case"]["penetrations_delta_over_rp"]

# --- Figure 1: single-disk force vs penetration, arc vs polyline ------------
fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=110)
ref = {r["delta_over_rp"]: r["closed_form_N"] for r in rows if r["chords"] is None}
ax.plot(list(ref), list(ref.values()), "k-", lw=2.0, label="closed form $k\\delta$")
styles = {None: ("o", "exact arc"), 8: ("s", "N=8"), 32: ("^", "N=32"),
          128: ("v", "N=128")}
for n, (marker, label) in styles.items():
    sel = [r for r in rows if r["chords"] == n]
    ax.plot([r["delta_over_rp"] for r in sel], [r["force_N"] for r in sel],
            marker=marker, ms=4, lw=0.9, label=label)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"arc overlap $\delta/r_p$ ($r_p$ = 25 mm)"); ax.set_ylabel("contact force [N]")
ax.set_title("single disk: arc kernel = closed form, chords under-load")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig1_geometry.png"); plt.close(fig)

# --- Figure 2: dynamic peak-load error vs chord count ----------------------
fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=110)
ax.axhspan(-3, 3, color="0.85", label="±3 % band")
for case, style in (("sheet", ("o", "33-disk sheet")), ("floe", ("s", "4-disk floe"))):
    runs = d[case]["runs"]
    xs = [int(k[1:]) for k in runs if k != "arc"]
    ys = [100 * runs[f"N{x}"]["peak_error_vs_arc"] for x in xs]
    ax.plot(xs, ys, style[0] + "-", ms=4, lw=1.0, label=style[1])
ax.set_xscale("log")
ax.set_xlabel("bow chords $N$ (180° bow, $R/r_p=10$)")
ax.set_ylabel("peak tool-force error vs exact arc [%]")
ax.set_title("indentation peak load: chord discretisation bias")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig2_peak_error.png"); plt.close(fig)

# --- Figure 3: early-time load history, sheet ------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=110)
runs = d["sheet"]["runs"]
dt = runs["arc"]["dt_s"]
t = [i * dt for i in range(len(runs["arc"]["history_N"]))]
seg = [i for i, ti in enumerate(t) if ti <= 0.45]
for key, label in (("arc", "exact arc"), ("N8", "N=8"), ("N32", "N=32"),
                   ("N4096", "N=4096")):
    h = runs[key]["history_N"]
    ax.plot([t[i] for i in seg], [h[i] for i in seg], lw=1.3, label=label)
ax.set_xlabel("time [s]"); ax.set_ylabel("tool reaction $-R_y$ [N]")
ax.set_title("load ramp: chord bias lives in the shallow-contact stage")
ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig3_load_ramp.png"); plt.close(fig)

for name in ("fig1_geometry.png", "fig2_peak_error.png", "fig3_load_ramp.png"):
    print(name, (OUT / name).stat().st_size, "bytes")