"""Integration driver: freezing-front profile inside the keel DEM (P316-W1-N03).

The module under test (``tensordem.freeze_bond.thermal_profile``) is not only a
stand-alone solver: the keel DEM (``tensordem.ice_ridge_keel_freezebond``) now
consumes it through :func:`depth_profile_strength` / ``run_keel_case(...,
depth_profile=...)``.  This driver runs the *same* pile twice -- once with the
m7439 uniform-temperature bond caliber, once with the depth-resolved caliber --
and records the couple (reference, got) that the maturity gate asks for:

* reference: the m7439 uniform bond strength ``tau_adfreeze(dT, t, Ra, S)``
  (independent of the thermal solver) and the exact Neumann front;
* got: the DEM-level bond population and the effective strength produced by the
  depth profile, plus the numerical front of the enthalpy solver.

Run:  python benchmarks/freeze_bond_thermal_profile/run_keel_integration.py
Writes ``benchmarks/freeze_bond_thermal_profile/integration_keel.json``.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from tensordem.ice_ridge_keel_freezebond import (  # noqa: E402
    depth_profile_strength,
    run_keel_case,
    tau_adfreeze,
)

OUT = Path(__file__).resolve().parent
CASE = dict(keel_height_target=4.0, undercooling_c=-15.0, salinity_ppt=0.0,
            consol_steps=800, release_steps=120)
PROFILE = {"t_end_s": 1.1e8, "k": 2.2, "n_cells": 200, "dt": 3600.0}
#: the profile front may differ from the exact Neumann front by less than this
FRONT_TOL = 0.005


def main() -> int:
    t0 = time.time()
    uniform = run_keel_case(**CASE)
    profiled = run_keel_case(**CASE, depth_profile=dict(PROFILE))
    prof = profiled["bond_strength_profile"]

    # independent reference of the *strength*: the m7439 analytic caliber
    tau_uniform_ref = tau_adfreeze(CASE["undercooling_c"], 1000.0, 20e-6,
                                   CASE["salinity_ppt"])
    front_rel = (prof["front_m"] - prof["front_neumann_m"]) / prof["front_neumann_m"]
    rows = [
        dict(id="bond_strength_caliber",
             reference=round(tau_uniform_ref, 9), got=round(prof["tau_eff_mpa"], 9),
             note=("m7439 uniform tau_adfreeze [MPa] vs the depth-averaged "
                   "tau_eff of the freezing-front profile"),
             rel_diff=round((prof["tau_eff_mpa"] - tau_uniform_ref) / tau_uniform_ref, 9)),
        dict(id="front_vs_neumann",
             reference=round(prof["front_neumann_m"], 9), got=round(prof["front_m"], 9),
             tolerance=FRONT_TOL, rel_diff=round(front_rel, 9),
             passed=bool(abs(front_rel) <= FRONT_TOL),
             note="exact one-phase Neumann front [m] vs the numerical enthalpy front"),
        dict(id="energy_closure",
             reference=0.0, got=round(prof["energy_rel_residual"], 12),
             tolerance=1e-4, passed=bool(abs(prof["energy_rel_residual"]) <= 1e-4),
             note="stored enthalpy change vs boundary flux integral (dimensionless)"),
    ]
    out = {
        "node": "P316-W1-N03",
        "integration": "tensordem.ice_ridge_keel_freezebond.run_keel_case(depth_profile=...)",
        "callers": ["tensordem.ice_ridge_keel_freezebond.run_keel_case",
                    "tensordem.ice_ridge_keel_freezebond.depth_profile_strength",
                    "tensordem.ice_ridge_keel_freezebond.create_bonds(tau_by_depth=...)"],
        "generated_by": "benchmarks/freeze_bond_thermal_profile/run_keel_integration.py",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "case": {"keel": CASE, "profile": PROFILE},
        "uniform_caliber": {
            "tau_adfreeze_mpa": uniform["tau_adfreeze_mpa"],
            "n_bonds": uniform["bonds"]["n_bonds"],
            "n_broken": uniform["bonds"]["n_broken"],
            "f_keel_N_per_m": uniform["f_keel"],
            "keel_height_m": uniform["keel_height_m"],
        },
        "profile_caliber": {
            **{k: prof[k] for k in ("tau_uniform_mpa", "tau_eff_mpa", "ratio",
                                    "H_m", "front_m", "front_neumann_m",
                                    "energy_rel_residual", "t_end_s")},
            "n_bonds": profiled["bonds"]["n_bonds"],
            "n_broken": profiled["bonds"]["n_broken"],
            "f_keel_N_per_m": profiled["f_keel"],
            "keel_height_m": profiled["keel_height_m"],
        },
        "benchmark": rows,
        "all_passed": all(r.get("passed", True) for r in rows),
        "repro_cmd": "python benchmarks/freeze_bond_thermal_profile/run_keel_integration.py",
        "runtime_s": round(time.time() - t0, 2),
    }
    (OUT / "integration_keel.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"tau_uniform_mpa": out["uniform_caliber"]["tau_adfreeze_mpa"],
                      "tau_eff_mpa": prof["tau_eff_mpa"], "ratio": prof["ratio"],
                      "bonds_uniform": out["uniform_caliber"]["n_bonds"],
                      "bonds_profile": out["profile_caliber"]["n_bonds"],
                      "all_passed": out["all_passed"],
                      "runtime_s": out["runtime_s"]}, indent=1))
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())