#!/usr/bin/env python3
"""W32 · ice-ridge keel grounding: coupled beam-on-keel convergence study.

This script runs a mesh-convergence study on the coupled beam-on-keel problem,
measuring resistance, contact area, and peak pressure for different mesh sizes.
It also computes the Spearman rank correlation between penetration and contact
force, and the force peak difference between two mesh levels.

Target artifact: papers/data/w32-2026-11-16/ice/convergence.json
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

# Add src to path
sys.path.insert(0, "src")

from tensordem import keel_grounding_coupled as kc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KEEL_HEIGHT = 2.0            # m
W0_FRACTION = 4.0e-3         # platen settlement / keel height -> w0 = 8 mm
BEAM_LENGTH = 16.0           # m, structure length (8 x keel height)
PLATEN_WIDTH = 2.0           # m, = keel height (W27 validation footing width)
ICE_SHEET_THICKNESS = 2.0    # m
ICE_SHEET_YOUNG = 5.0e9      # Pa
ICE_POISSON = 0.3
RAMP_WIDTH = 8.0e-4          # m, C1 no-tension ramp = 0.1 w0
PENETRATION_LEVELS = 6
MESH_ELEMENTS = (32, 64, 128)  # h = 0.5 / 0.25 / 0.125 m

EI_STRUCTURE = ICE_SHEET_YOUNG * ICE_SHEET_THICKNESS ** 3 / (12.0 * (1.0 - ICE_POISSON ** 2))

# Keel pressure law from DEM confined-compression test (W28 data)
DEM_SETTLE = [1.3333e-3, 2.6667e-3, 4.0e-3, 5.3333e-3, 6.6667e-3, 8.0e-3]
DEM_PRESS = [4.0817e6, 6.9547e6, 9.2988e6, 11.3277e6, 13.1406e6, 14.7937e6]

OUT_DIR = os.path.join("papers", "data", "w32-2026-11-16", "ice")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Run the convergence study
# ---------------------------------------------------------------------------
def run_study():
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
            "element_length_m": BEAM_LENGTH / ne,
            "resistance_N_per_m": out["resistance_N_per_m"],
            "contact_area_m2_per_m": out["contact_area_m2_per_m"],
            "q_max_Pa": out["q_max_Pa"],
            "converged": out["converged"],
            "newton_iterations": out["newton_iterations"],
        })
    
    # Compute Spearman rank correlation between penetration and contact force
    settlements = [w0 * k / PENETRATION_LEVELS for k in range(1, PENETRATION_LEVELS + 1)]
    forces = []
    for ne in MESH_ELEMENTS:
        out = kc.beam_on_unilateral_keel(
            length=BEAM_LENGTH, ei=EI_STRUCTURE, n_elements=ne,
            platen_centre=0.5 * BEAM_LENGTH, platen_width=PLATEN_WIDTH, w0=w0,
            law=law, ramp_width_m=RAMP_WIDTH, samples_per_element=32
        )
        # Get the resistance at each penetration level
        steps = []
        for k in range(1, PENETRATION_LEVELS + 1):
            w = w0 * k / PENETRATION_LEVELS
            step_out = kc.beam_on_unilateral_keel(
                length=BEAM_LENGTH, ei=EI_STRUCTURE, n_elements=ne,
                platen_centre=0.5 * BEAM_LENGTH, platen_width=PLATEN_WIDTH, w0=w,
                law=law, ramp_width_m=RAMP_WIDTH, samples_per_element=32
            )
            steps.append(step_out["resistance_N_per_m"])
        forces.append(steps)
    
    # Spearman correlation for the finest mesh
    spearman_result = kc.spearman(settlements, forces[-1])
    
    # Force peak difference between two mesh levels (fine vs medium)
    force_diff = abs(forces[-1][-1] - forces[-2][-1]) / max(abs(forces[-1][-1]), 1e-300)
    
    # GCI and observed order
    resistances = [r["resistance_N_per_m"] for r in results]
    gci_info = kc.observed_order(resistances[0], resistances[1], resistances[2], r=2.0)
    richardson_info = kc.richardson_limit(resistances[0], resistances[1], resistances[2], r=2.0)
    
    # Build the artifact
    artifact = {
        "mission": "paper-exp-ice-ridge-keel-grounding-6",
        "study": "coupled_beam_on_keel_convergence",
        "date": "2026-11-16",
        "configuration": {
            "keel_height_m": KEEL_HEIGHT,
            "beam_length_m": BEAM_LENGTH,
            "platen_width_m": PLATEN_WIDTH,
            "w0_m": w0,
            "ei_N_m2": EI_STRUCTURE,
            "ramp_width_m": RAMP_WIDTH,
            "penetration_levels": PENETRATION_LEVELS,
            "mesh_elements": list(MESH_ELEMENTS),
            "keel_law": {
                "settlements_m": DEM_SETTLE,
                "pressures_Pa": DEM_PRESS,
            },
        },
        "results": results,
        "penetration_force_data": {
            "settlements_m": settlements,
            "forces_per_mesh": forces,
        },
        "spearman_correlation": spearman_result,
        "force_peak_difference": {
            "fine_vs_medium_rel": force_diff,
            "fine_resistance_N_per_m": forces[-1][-1],
            "medium_resistance_N_per_m": forces[-2][-1],
        },
        "gci_analysis": gci_info,
        "richardson_limit": richardson_info,
        "criteria": {
            "spearman_rho": spearman_result["rho"],
            "spearman_threshold": 1.0,
            "force_peak_diff_threshold": 0.05,
            "force_peak_diff_actual": force_diff,
        },
    }
    
    # Write the artifact
    artifact_path = os.path.join(OUT_DIR, "convergence.json")
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)
    
    print(f"Artifact written to {artifact_path}")
    print(f"Spearman rho: {spearman_result['rho']}")
    print(f"Force peak difference (fine vs medium): {force_diff}")
    print(f"GCI: {gci_info.get('gci_fine_rel')}")
    print(f"Observed order: {gci_info.get('observed_order_p')}")
    
    return artifact

if __name__ == "__main__":
    artifact = run_study()
