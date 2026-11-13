#!/usr/bin/env python
"""Run ONE arm of ``xval_dem_round2_column_arms.py`` (frozen parameters) per process.

The single-process driver is too long for one shell slice, so the arms are run
separately and stitched by ``xval_dem_round2_column_arms_merge.py``.  Parameters
and the arm table are imported from the frozen driver, never retyped.

Usage: PYTHONPATH=src python benchmarks/xval_dem_round2_column_arms_arm.py legacy_mu0.5
Writes artifacts/xval_dem_round2_method_selection/arm_<name>.json
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.granular_column.axisym import (AxisymCase,  # noqa: E402
                                             lube_2004_emplacement_time,
                                             lube_2004_runout_ratio,
                                             simulate_axisymmetric_column)

_spec = importlib.util.spec_from_file_location(
    "frozen_arms", os.path.join(ROOT, "benchmarks", "xval_dem_round2_column_arms.py"))
_frozen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_frozen)

OUT = os.path.join(ROOT, "artifacts", "xval_dem_round2_method_selection")


def main() -> None:
    name = sys.argv[1]
    arms = dict((a[0], a[1:]) for a in _frozen.ARMS)
    if name not in arms:
        raise SystemExit(f"unknown arm {name!r}; known {sorted(arms)}")
    lever, mu_r = arms[name]
    base = _frozen.BASE
    case = AxisymCase(mu_rolling=mu_r, rolling_lever=lever, **base)
    t0 = time.time()
    out = simulate_axisymmetric_column(case)
    rec = {
        "rolling_lever": lever, "mu_rolling": mu_r,
        "runout_ratio": float(out["runout_ratio_p995"]),
        "runout_ratio_max": float(out["runout_ratio_max"]),
        "emplacement_time_s": float(out["t_reach"]) if "t_reach" in out else None,
        "n_particles": int(out["n_particles"]),
        "elapsed_s": round(time.time() - t0, 1),
        "aspect_measured": float(out.get("aspect_measured", float("nan"))),
    }
    ref_runout = lube_2004_runout_ratio(base["aspect_ratio"])
    rec["runout_err_vs_lube"] = float((rec["runout_ratio"] - ref_runout) / ref_runout)
    os.makedirs(OUT, exist_ok=True)
    payload = {"arm": name, "record": rec, "base": base,
               "reference": {"runout_ratio": ref_runout,
                             "emplacement_time_s": lube_2004_emplacement_time(
                                 base["aspect_ratio"],
                                 base["grains_per_radius"] * 2.0 * base["radius"], 9.81),
                             "source": "Lube et al. 2004 JFM 508:175-199"}}
    with open(os.path.join(OUT, f"arm_{name}.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(json.dumps(rec, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()