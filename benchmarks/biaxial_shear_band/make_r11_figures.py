"""Figures for R11 of CASE-BIAXIAL-SHEAR-BAND (no simulation, JSON in, PNG out).

Reads ``artifacts/biaxial_shear_band/r11_phi_state_sensitivity/
phi_state_sensitivity.json`` and renders:

* left: the Arthur and Mohr-Coulomb absolute-relative errors vs the ``phi``
  state-point sweep (at ``phi_cv = 32``, the R9/R10 headline arm), with the
  3 % precision line, the 15 % tolerance line, the measured peak ``phi``
  (35.45) and the robustness window ``[34.0, 35.5]`` shaded;
* right: the ``inside_robustness_window`` flag across the sweep (the window is
  the set of ``phi`` where Arthur is supported at every declared ``phi_cv``),
  showing the window is narrow (1.5 deg) around the peak.

All numbers are program-injected from the JSON; no hand transcription.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(ROOT, "artifacts", "biaxial_shear_band", "r11_phi_state_sensitivity")
FIG = os.path.join(ART, "figures")
os.makedirs(FIG, exist_ok=True)

d = json.load(open(os.path.join(ART, "phi_state_sensitivity.json")))
sweep = d["sweep"]
phis = [r["phi_deg"] for r in sweep]
# headline arm phi_cv = 32
arthur_err = [r["per_phi_cv"]["32.0"]["errors"]["arthur"]["abs_relative_error"]
              * 100 for r in sweep]
mc_err = [r["per_phi_cv"]["32.0"]["errors"]["mohr_coulomb"]["abs_relative_error"]
          * 100 for r in sweep]
inside = [1.0 if r["inside_robustness_window"] else 0.0 for r in sweep]
peak = d["phi_peak_deg"]
win_low, win_high = d["robustness_window_deg"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# left: errors vs phi
ax1.plot(phis, arthur_err, "o-", color="tab:green", label="Arthur 45+(φ+ψ)/4")
ax1.plot(phis, mc_err, "s--", color="tab:red", label="Mohr-Coulomb 45+φ/2")
ax1.axhline(3.0, color="k", ls=":", lw=1, label="3 % precision")
ax1.axhline(15.0, color="k", ls="--", lw=1, label="15 % tolerance")
ax1.axvline(peak, color="tab:blue", lw=2, label=f"measured peak φ={peak}°")
ax1.axvspan(win_low, win_high, color="tab:green", alpha=0.15,
            label=f"robustness window [{win_low}, {win_high}]")
ax1.set_xlabel("φ state-point (deg)")
ax1.set_ylabel("|θ_meas − θ_law| / θ_law  (%)")
ax1.set_title("R11: Arthur vs Mohr-Coulomb error vs φ state-point\n"
              "(θ=55.00°, φ_cv=32 arm)")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# right: inside-window flag
ax2.fill_between(phis, 0, inside, step="mid", color="tab:green", alpha=0.6,
                 label="inside robustness window")
ax2.axvline(peak, color="tab:blue", lw=2, label=f"peak φ={peak}°")
ax2.set_xlabel("φ state-point (deg)")
ax2.set_ylabel("Arthur supported at every φ_cv (0/1)")
ax2.set_ylim(-0.05, 1.05)
ax2.set_yticks([0, 1])
ax2.set_title("R11: robustness window (narrow, 1.5° wide)\n"
              "peak inside, pre-registered [33,38] NOT met")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIG, "r11_phi_state_sensitivity.png"), dpi=150)
print("wrote", os.path.join(FIG, "r11_phi_state_sensitivity.png"))
