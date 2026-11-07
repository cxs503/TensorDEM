#!/usr/bin/env python3
"""Lattice-resolution sensitivity of the fitted pressure-area exponent."""
from __future__ import annotations

import json
from pathlib import Path

from tensordem.pressure_area_dem import PunchConfig, run_area_scan

OUT = Path("artifacts/pressure_area_index")
AREAS = [0.5, 1.25, 3.5, 10.0]
BASE = dict(punch_width=1.0, sheet_thickness=0.2, strength_median=1.0e-3,
            strength_cov=0.52, correlation_length=0.10, surface_roughness=0.02,
            max_penetration_fraction=0.03, n_increments=60, newton_iterations=6)


def main() -> int:
    out = {}
    for spacing, rough in ((0.03, 0.06), (0.05, 0.10)):
        cfg = PunchConfig(**{**BASE, "spacing": spacing, "roughness_correlation": rough})
        scan = run_area_scan(AREAS, 12, cfg, label=f"sp{spacing}", n_boot=2000,
                             workers=40)
        out[f"spacing_{spacing}"] = dict(scan.fit, seeds_per_area=12, areas_m2=AREAS)
        print(spacing, scan.fit, flush=True)
    (OUT / "resolution_check.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
