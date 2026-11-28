#!/usr/bin/env python3
"""Generate figures for W32 ice-ridge keel grounding convergence study."""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "src")

from tensordem import keel_grounding_coupled as kc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KEEL_HEIGHT = 2.0
W0_FRACTION = 4.0e-3
BEAM_LENGTH = 16.0
PLATEN_WIDTH = 2.0
ICE_SHEET_THICKNESS = 2.0
ICE_SHEET_YOUNG = 5.0e9
ICE_POISSON = 0.3
RAMP_WIDTH = 8.0e-4
PENETRATION_LEVELS = 6
MESH_ELEMENTS = (32, 64, 128)

EI_STRUCTURE = ICE_SHEET_YOUNG * ICE_SHEET_THICKNESS ** 3 / (12.0 * (1.0 - ICE_POISSON ** 2))

DEM_SETTLE = [1.3333e-3, 2.6667e-3, 4.0e-3, 5.3333e-3, 6.6667e-3, 8.0e-3]
DEM_PRESS = [4.0817e6, 6.9547e6, 9.2988e6, 11.3277e6, 13.1406e6, 14.7937e6]

OUT_DIR = os.path.join("papers", "data", "w32-2026-11-16", "ice", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Run the study
# ---------------------------------------------------------------------------
law = kc.KeelPressureLaw(DEM_SETTLE, DEM_PRESS)
w0 = W0_FRACTION * KEEL_HEIGHT

results = []
for ne in MESH_ELEMENTS:
    out = kc.beam_on_unilateral_keel(
        length=BEAM_LENGTH, ei=EI_STRUCTURE, n_elements=ne,
        platen_centre=0.5 * BEAM_LENGTH, platen_width=PLATEN_WIDTH, w0=w0,
        law=law, ramp_width_m=RAMP_WIDTH, samples_per_element=32
    )
    results.append({
        "n_elements": ne,
        "x_m": out["x_m"],
        "w_m": out["w_m"],
        "resistance_N_per_m": out["resistance_N_per_m"],
        "contact_area_m2_per_m": out["contact_area_m2_per_m"],
        "q_max_Pa": out["q_max_Pa"],
        "contact_pressure_profile": out["contact_pressure_profile"],
    })

# ---------------------------------------------------------------------------
# Figure 1: Deflection profiles
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
for r in results:
    x = np.array(r["x_m"])
    w = np.array(r["w_m"])
    ax.plot(x, w, label=f"n={r['n_elements']}")
ax.set_xlabel("x [m]")
ax.set_ylabel("w [m]")
ax.set_title("Beam deflection profiles (platen settlement w0 = 8 mm)")
ax.legend()
ax.grid(True)
fig.tight_layout()
fig_path = os.path.join(OUT_DIR, "deflection_profiles.png")
fig.savefig(fig_path, dpi=150)
plt.close(fig)
print(f"Saved {fig_path}")

# ---------------------------------------------------------------------------
# Figure 2: Contact pressure profiles
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
for r in results:
    x = np.array(r["contact_pressure_profile"]["x_m"])
    p = np.array(r["contact_pressure_profile"]["p_Pa"])
    ax.plot(x, p, label=f"n={r['n_elements']}")
ax.set_xlabel("x [m]")
ax.set_ylabel("p [Pa]")
ax.set_title("Contact pressure profiles (platen settlement w0 = 8 mm)")
ax.legend()
ax.grid(True)
fig.tight_layout()
fig_path = os.path.join(OUT_DIR, "pressure_profiles.png")
fig.savefig(fig_path, dpi=150)
plt.close(fig)
print(f"Saved {fig_path}")

# ---------------------------------------------------------------------------
# Figure 3: Resistance vs penetration
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
settlements = [w0 * k / PENETRATION_LEVELS for k in range(1, PENETRATION_LEVELS + 1)]
for r in results:
    ne = r["n_elements"]
    forces = []
    for w in settlements:
        out = kc.beam_on_unilateral_keel(
            length=BEAM_LENGTH, ei=EI_STRUCTURE, n_elements=ne,
            platen_centre=0.5 * BEAM_LENGTH, platen_width=PLATEN_WIDTH, w0=w,
            law=law, ramp_width_m=RAMP_WIDTH, samples_per_element=32
        )
        forces.append(out["resistance_N_per_m"])
    ax.plot(settlements, forces, label=f"n={ne}")
ax.set_xlabel("Settlement w [m]")
ax.set_ylabel("Resistance R [N/m]")
ax.set_title("Resistance vs penetration (mesh convergence)")
ax.legend()
ax.grid(True)
fig.tight_layout()
fig_path = os.path.join(OUT_DIR, "resistance_vs_penetration.png")
fig.savefig(fig_path, dpi=150)
plt.close(fig)
print(f"Saved {fig_path}")

# ---------------------------------------------------------------------------
# Figure 4: Convergence plot (GCI)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
resistances = [r["resistance_N_per_m"] for r in results]
h = [BEAM_LENGTH / r["n_elements"] for r in results]
ax.loglog(h, resistances, "o-", label="Resistance")
ax.set_xlabel("Element size h [m]")
ax.set_ylabel("Resistance R [N/m]")
ax.set_title("Mesh convergence (log-log)")
ax.legend()
ax.grid(True)
fig.tight_layout()
fig_path = os.path.join(OUT_DIR, "convergence_plot.png")
fig.savefig(fig_path, dpi=150)
plt.close(fig)
print(f"Saved {fig_path}")

print("All figures generated.")
