#!/usr/bin/env python3
"""Audit figures for m17676 (board audit m17745).

Zero hand-copy: every number is read from the delivered run artifacts that were
snapshotted next to this script (``artifacts/cases/*.json``, ``artifacts/verdict.json``).
Run:  python docs/audit_m17676/make_figures.py
Out:  docs/audit_m17676/figures/*.png
"""
from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

verdict = json.load(open(os.path.join(ART, "verdict.json")))
cases = verdict["cases"]
ref = verdict["reference"]
band = ref["sphere_band_deg"]
centre = ref["sphere_centre_deg"]
tol = verdict["thresholds"]["TOL_REL"]

order = ["as_delivered", "ctrl_mu_r_zero", "ctrl_mu_s_zero", "gentle_anchor",
         "sens_N1000", "sens_N4000", "sens_settle_0p6", "sens_dt_half"]
theta = [cases[k]["theta_deg"] for k in order]
err = [cases[k].get("stderr_deg", 0.0) for k in order]

# ---------------------------------------------------------------- fig 1
fig, ax = plt.subplots(figsize=(8.2, 4.6))
x = np.arange(len(order))
ax.axhspan(band[0], band[1], color="#7fb800", alpha=0.22,
           label=f"Zhou-1999 sphere band {band[0]:.0f}-{band[1]:.0f} deg")
ax.axhline(centre * (1 - tol), ls="--", lw=1.1, color="#b23a48",
           label=f"-15% owner tolerance ({centre * (1 - tol):.2f} deg)")
ax.errorbar(x, theta, yerr=err, fmt="o", ms=6, capsize=4, color="#1f4e79",
            label="TensorDEM sandpile-v1 (this delivery)")
for i, k in enumerate(order):
    ok = abs(theta[i] - centre) / centre <= tol
    ax.annotate(f"{theta[i]:.2f}", (i, theta[i]), textcoords="offset points",
                xytext=(0, 8 if ok else -16), ha="center", fontsize=8,
                color="#1f4e79" if ok else "#b23a48")
ax.set_xticks(x)
ax.set_xticklabels(order, rotation=28, ha="right", fontsize=8)
ax.set_ylabel("angle of repose [deg]")
ax.set_title("m17676 claim vs machine-checkable anchor (D1 = |err|/ref <= 15%)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_theta_vs_anchor.png"), dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- fig 2
anchor = json.load(open(os.path.join(ART, "cases", "gentle_anchor.json")))
prof = anchor["surface_profile"]
fit = anchor["repose_fit"]
r = np.asarray(prof["radii"])
h = np.asarray(prof["heights"])
s = np.asarray(prof.get("sector_std", np.zeros_like(r)))
fig, ax = plt.subplots(figsize=(7.4, 4.8))
ax.errorbar(r * 1e3, h * 1e3, yerr=s * 1e3, fmt="o", ms=4, lw=0.8,
            color="#1f4e79", label="measured free surface h(r) (sector mean +/- std)")
rf = np.linspace(fit["r_lo"], fit["r_hi"], 50)
ax.plot(rf * 1e3, (fit["intercept"] + fit["slope"] * rf) * 1e3, "r-", lw=1.8,
        label=(f"flank fit theta={fit['theta_deg']:.2f}+/-{fit['theta_stderr_deg']:.2f} deg, "
               f"R^2={fit['r2']:.3f}"))
ax.axvline(fit["r_lo"] * 1e3, color="0.5", ls=":", lw=1)
ax.axvline(fit["r_hi"] * 1e3, color="0.5", ls=":", lw=1, label="25-85% toe-radius fit window")
ax.set_xlabel("radius r [mm]")
ax.set_ylabel("surface height h [mm]")
ax.set_title("gentle_anchor (N=2000) surface profile and repose fit")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_profile_fit.png"), dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- fig 3
fig, ax = plt.subplots(figsize=(8.0, 4.2))
rel = [abs(theta[i] - centre) / centre * 100.0 for i in range(len(order))]
bars = ax.bar(np.arange(len(order)), rel, color=["#b23a48" if v > tol * 100 else "#4c9f70"
                                                 for v in rel])
ax.axhline(tol * 100, color="#b23a48", ls="--", lw=1.4,
           label="15% acceptance line (owner standard)")
for b, v in zip(bars, rel):
    ax.annotate(f"{v:.2f}%", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=8)
ax.set_xticks(np.arange(len(order)))
ax.set_xticklabels(order, rotation=28, ha="right", fontsize=8)
ax.set_ylabel("relative error vs band centre [%]")
ax.set_title("m17676: no delivered case reaches the 15% band (honest FAIL)")
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_error_vs_tolerance.png"), dpi=140)
plt.close(fig)

print(json.dumps({"figures": sorted(os.listdir(OUT)),
                  "theta": dict(zip(order, theta)),
                  "rel_err_pct": dict(zip(order, [round(v, 3) for v in rel]))},
                 ensure_ascii=False, indent=1))