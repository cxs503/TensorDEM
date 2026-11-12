#!/usr/bin/env python
"""Generate figures for the CASE-ROTATING-DRUM mixing/resting benchmark (m15329).

Reads the combined case records (runs/m15329/combined.json) and produces:
  fig1_final_states.png   final particle positions (colored by species) per Fr
  fig2_segregation_growth.png  segregation index vs revolution (per Fr)
  fig3_release_angle.png  measured vs analytic release angle vs Fr
  fig4_excess_vs_fr.png   core-shell excess vs Fr (per run)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("runs/m15329/figures")
OUT.mkdir(parents=True, exist_ok=True)

data = json.load(open("runs/m15329/combined.json"))
cases = data["cases"]


def ref_release_angle_deg(fr):
    return 180.0 if fr >= 1.0 else math.degrees(math.acos(-fr))


# ---------------------------------------------------------------------------
# Fig 1: final states (real positions from the main run, 10 rev N=581)
# ---------------------------------------------------------------------------
main = sorted([c for c in cases if c["run"] == "main_10rev_N581"],
              key=lambda c: c["fr"])
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
fig.suptitle("CASE-ROTATING-DRUM: final bed states after 10 rev (N=581)\n"
             "red = large (d=12 mm), blue = small (d=6 mm)", fontsize=12)
for ax, c in zip(axes, main):
    pos = np.asarray(c["final_pos"])
    isL = np.asarray(c["final_is_large"], bool)
    R = c["radius"]
    ax.scatter(pos[~isL, 0] / R, pos[~isL, 1] / R, s=9, c="steelblue",
               label="small", zorder=3)
    ax.scatter(pos[isL, 0] / R, pos[isL, 1] / R, s=22, c="red",
               label="large", zorder=3)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, lw=2, color="k"))
    ax.set_xlim(-1.12, 1.12); ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"Fr={c['fr']:.1f} ({c['ref_regime']})  "
                 f"excess={c['excess_final']:+.3f}", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig1_final_states.png", dpi=130)
plt.close(fig)
print("wrote fig1")

# ---------------------------------------------------------------------------
# Fig 2: segregation growth S(rev) per Fr (main run)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
for c in main:
    rev = c["series"]["rev"]
    S = c["series"]["segregation"]
    ax.plot(rev, S, marker="o", ms=3, label=f"Fr={c['fr']:.1f}")
ax.set_xlabel("drum revolutions")
ax.set_ylabel("segregation index S")
ax.set_title("Segregation index growth (10 rev, N=581)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig2_segregation_growth.png", dpi=130)
plt.close(fig)
print("wrote fig2")

# ---------------------------------------------------------------------------
# Fig 3: release angle measured vs analytic
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
frs = sorted(set(c["fr"] for c in cases))
meas = []
for fr in frs:
    pref = [c for c in cases if c["fr"] == fr and c["run"] == "main_10rev_N581"]
    src = pref[0] if pref else next(c for c in cases if c["fr"] == fr)
    meas.append(src["measured_release_angle_deg"])
refs = [ref_release_angle_deg(fr) for fr in frs]
ax.plot(frs, refs, "k-", lw=2, label="analytic  $\\theta_d=\\arccos(-Fr)$")
ax.plot(frs, meas, "ro-", ms=6, label="DEM (95th-pct wall-contact angle)")
ax.set_xlabel("Froude number  $Fr=\\omega^2 R/g$")
ax.set_ylabel("release angle $\\theta_d$ [deg]")
ax.set_title("Release (detachment) angle vs Froude number")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig3_release_angle.png", dpi=130)
plt.close(fig)
print("wrote fig3")

# ---------------------------------------------------------------------------
# Fig 4: core-shell excess vs Fr
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
for run_tag, mk in (("sweep_6rev_N209", "o"), ("main_10rev_N581", "s")):
    sub = sorted([c for c in cases if c["run"] == run_tag], key=lambda c: c["fr"])
    ax.plot([c["fr"] for c in sub], [c["excess_final"] for c in sub],
            mk + "-", label=run_tag)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("Froude number  $Fr=\\omega^2 R/g$")
ax.set_ylabel("core-shell excess  $\\phi_L^{out}-\\phi_L^{in}$")
ax.set_title("Segregation (core-shell excess) vs Froude number\n"
             "(+ = coarse at periphery, Brazil-nut)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig4_excess_vs_fr.png", dpi=130)
plt.close(fig)
print("wrote fig4")
print("all figures done")
