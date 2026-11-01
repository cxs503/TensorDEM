"""Ice-side (rigid-pipe) keel-depth sensitivity for the ridge grounding campaign.

The coupled run is subgrade-governed, so the *system* capacity cannot reveal the
ice-side dependence on keel depth.  This auxiliary study fixes the pipeline
(``pipe_response='rigid'``) and compares the transferred ice load at identical
platen stroke and rate, which isolates the ice contact response.
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.ridge_dem import RidgeKeel, RidgeMaterial  # noqa: E402
from tensordem.ridge_geotech import MohrCoulombSoil, PipeSection  # noqa: E402
from tensordem.ridge_grounding import GroundingScenario, run_grounding  # noqa: E402

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "artifacts", "ridge_grounding")
SOIL = MohrCoulombSoil(friction_angle_deg=0.0, cohesion=30.0e3, saturated_unit_weight=18.0e3,
                       young_modulus=10.0e6, water_unit_weight=10.06e3, undrained=True)
SECTION = PipeSection(0.914, 0.0159, 207.0e9, 448.0e6)
MATERIAL = RidgeMaterial(particle_radius=0.5, drag=2.0e4, freeze_bond_fraction=1.0,
                         bond_tensile_strength=100.0e3, bond_shear_strength=200.0e3)


def main() -> int:
    torch.set_num_threads(1)
    rows = []
    for depth in (8.0, 12.0, 16.0):
        scenario = GroundingScenario(
            keel=RidgeKeel(depth, 2.0 * depth, 0.10), material=MATERIAL, soil=SOIL,
            section=SECTION, cover_depth=0.5, tip_clearance=0.15, stroke=1.2,
            platen_velocity=0.4, pipe_response="rigid", contact_length=2.0 * depth)
        result = run_grounding(scenario)
        rows.append({"keel_depth": depth, "keel_width": 2.0 * depth,
                     "capacity_per_length": result.capacity,
                     "final_load": result.history["pipe_load"][-1],
                     "broken_bonds": result.info["broken_bonds"],
                     "particles": result.info["particles"], "steps": result.steps})
        print(f"rigid depth {depth}: {result.capacity:.0f} N/m", flush=True)
    with open(os.path.join(ART, "sensitivity_rigid.json"), "w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, indent=2)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
