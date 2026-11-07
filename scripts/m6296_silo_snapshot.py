"""m6296: DEM silo field snapshot (positions + speed) for the report figure.

Runs the D4 geometry for a short discharge and dumps the particle state to
``runs/m6296_silo/snapshot.json``; ``scripts/m6296_beverloo_analysis.py``
turns it into ``docs/figures/m6296_silo_field_snapshot.png``.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m6296_silo_beverloo_sweep import build_case  # noqa: E402
from tensordem.silo_discharge import simulate_silo  # noqa: E402

SNAP_AFTER = 0.55   # s of physical time (0.25 s settle + 0.30 s discharge)


def main() -> int:
    case = build_case("D4")
    case = dataclasses.replace(case, settle_time=0.25, discharge_time=0.30,
                              window=0.06, n_windows=5)
    out = simulate_silo(case, return_history=True)
    pos = out["positions"]
    alive = out["alive"]

    speed = np.linalg.norm(out["velocities"], axis=1)
    data = {
        "case": "D4",
        "t": case.settle_time + case.discharge_time,
        "positions": pos.tolist(),
        "speed": [float(s) for s in speed],
        "alive": alive.tolist(),
        "n_discharged": out["n_discharged"],
        "flow_rate_kg_s": out["flow_rate_kg_s"],
        "orifice_diameter": out["orifice_diameter"],
    }
    os.makedirs("runs/m6296_silo", exist_ok=True)
    with open("runs/m6296_silo/snapshot.json", "w", encoding="utf8") as fh:
        json.dump(data, fh)
    print("snapshot written:", data["n_discharged"], "discharged of", len(pos))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())