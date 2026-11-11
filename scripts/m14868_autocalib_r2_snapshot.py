"""Save one pluviation deposit (positions + speeds) for the round-2 figure.

A single forward call at N = 960, sand twin-truth parameters, so that the
report can show *what the deposit actually is* next to the measured slope:
the two things that decide whether the shipped repose response is a material
function are the deposit aspect (how many grain layers) and the speed field
(is it at rest).
"""
from __future__ import annotations

import dataclasses
import json
import os

import numpy as np

from tensordem.dem_calibration.heap import run_pluviation
from tensordem.dem_calibration.materials import spec_by_name

OUT = os.path.join("artifacts", "gap_dem_autocalib_r2", "snapshot_n960.json")


def main() -> None:
    sand = spec_by_name("sand")
    case = dataclasses.replace(sand.case("calibration"), n_grains=960, seed=0)
    res = run_pluviation(case, return_positions=True)
    pos = np.asarray(res["positions"])
    speed = np.asarray(res["speeds"])
    rad = np.hypot(pos[:, 0], pos[:, 1])
    payload = {
        "n_grains": 960, "seed": 0, "radius_m": float(case.radius),
        "theta_r_deg": float(res["theta_r_deg"]),
        "phi_solid": float(res["phi_solid"]), "r2": float(res["r2"]),
        "ke_ratio": float(res["ke_ratio"]),
        "x_m": pos[:, 0].tolist(), "z_m": pos[:, 2].tolist(),
        "speed": speed.tolist(),
        "layers": float(pos[:, 2].max() / (2.0 * case.radius)),
        "footprint_radius_m": float(np.percentile(rad, 99)),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(payload, fh)
    print({k: payload[k] for k in ("theta_r_deg", "phi_solid", "r2",
                                   "ke_ratio", "layers", "footprint_radius_m")})


if __name__ == "__main__":
    main()
