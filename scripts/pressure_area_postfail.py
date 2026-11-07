#!/usr/bin/env python3
"""Post-yield failure statistics: how simultaneous is the failure sequence?

Runs the punch past the first failure (diagnostic only; the campaign proper
reports the elastic limit) for one area with a correlated and with a perfectly
correlated strength field, and records the failure-event sequence.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensordem.pressure_area_dem import PunchConfig, run_punch_indentation

OUT = Path("artifacts/pressure_area_index")
BASE = dict(punch_width=2.0, sheet_thickness=0.2, strength_median=1.0e-3,
            strength_cov=0.52, surface_roughness=0.02,
            roughness_correlation=0.08, max_penetration_fraction=0.03,
            n_increments=40, newton_iterations=6, stop_at_first_failure=False,
            peak_drop_fraction=0.0)


def main() -> int:
    out = {}
    for label, corr in (("correlated", 0.10), ("uniform", math.inf)):
        runs = []
        for seed in range(3):
            cfg = PunchConfig(**{**BASE, "correlation_length": corr, "seed": seed})
            r = run_punch_indentation(cfg)
            events = r.failure_events
            runs.append({
                "seed": seed,
                "failure_events": len(events),
                "bonds_removed_total": int(r.failed_total),
                "largest_single_event": int(max(events) if events else 0),
                "max_force_N_per_m": float(max(r.force)),
                "non_simultaneity": float(r.non_simultaneity),
            })
        out[label] = {
            "per_seed": runs,
            "mean_events": float(np.mean([x["failure_events"] for x in runs])),
            "mean_bonds_removed": float(np.mean([x["bonds_removed_total"] for x in runs])),
            "mean_largest_event": float(np.mean([x["largest_single_event"] for x in runs])),
        }
        print(label, out[label]["mean_events"], out[label]["mean_bonds_removed"],
              out[label]["mean_largest_event"], flush=True)
    (OUT / "postfail.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
