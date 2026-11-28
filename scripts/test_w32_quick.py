#!/usr/bin/env python3
"""Quick test: verify imports and basic functionality."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tensordem.keel_foundation_dem import build_keel_sample, confined_compression, footing_test, ICE_YOUNG, REFERENCE_LENGTH
from tensordem.ridge_keel_grounding import KerrFoundation, kerr_footing_resistance, kerr_footing_selfcheck

print("Imports OK")

# Build a small specimen
sample = build_keel_sample(1000, height=2.0, cell_ratio=8.0, solid_fraction=0.62, seed=7, ice_young=ICE_YOUNG, bond_reference_length=REFERENCE_LENGTH)
print(f"Sample: {sample.n_particles} particles, {len(sample.bi)} bonds")

# Confined compression
confined = confined_compression(sample, w_max=0.008, increments=6, steps_per_increment=100)
print(f"Confined: K={confined['k_subgrade_N_per_m3']:.2e} N/m^3")

# Footing test
ft = footing_test(sample, footing_width=1.0, w0=0.008, increments=6, steps_per_increment=100)
print(f"Footing b=1.0: R={ft['resistance_N_per_m'][-1]:.2e} N/m")

# Kerr prediction
K = confined['k_subgrade_N_per_m3']
kerr_fnd = KerrFoundation(k_upper=2*K, k_lower=2*K, shear=2*(ft['resistance_N_per_m'][-1] - K*0.008*1.0)/(2*0.008)**2/K)
R_kerr = kerr_footing_resistance(width=1.0, w0=0.008, foundation=kerr_fnd)
print(f"Kerr b=1.0: R={R_kerr:.2e} N/m")

# Write test result
import json
result = {"test": "ok", "K": K, "R_dem": ft['resistance_N_per_m'][-1], "R_kerr": R_kerr}
with open("papers/data/w32-2026-11-16/ice/test_result.json", "w") as f:
    json.dump(result, f, indent=2)
print("Test result written to papers/data/w32-2026-11-16/ice/test_result.json")
