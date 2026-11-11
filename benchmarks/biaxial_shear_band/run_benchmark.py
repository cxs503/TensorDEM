"""Benchmark: shear-band inclination in the plane-strain biaxial cell.

Acceptance criterion (pre-declared, case ``CASE-BIAXIAL-SHEAR-BAND``)
--------------------------------------------------------------------
Reference: shear-band inclination of a dense frictional granular material in a
plane-strain biaxial compression test, measured from the ``sigma_3`` axis.

* ``theta_ref = 60 deg`` -- midpoint of the 55-60 deg range reported for dense
  Hostun sand biaxial tests by Desrues & Viggiani (2004), *Int. J. Numer. Anal.
  Methods Geomech.* 28:279-321 (reviews the Grenoble stereophotogrammetry
  programme); consistent with the Mohr-Coulomb value ``45 + phi/2 = 60 deg``
  for ``phi = 30 deg``.
* tolerance 15 % relative => accept ``51 deg <= theta <= 69 deg``.
* qualitative requirement: the deformation must localise, i.e. the radon
  ``eta2`` of the dominant band must beat the angular null level by
  ``z > 3`` and the band must be resolved as one dominant orientation.

The run prints and stores, for every configuration, the measured band angle,
the measured ``phi`` / ``psi``, the three theoretical inclinations (Roscoe
1970, Arthur et al. 1977, Mohr-Coulomb) and the relative errors.

Usage
-----
``PYTHONPATH=src python benchmarks/biaxial_shear_band/run_benchmark.py``
``--quick``   reduced specimen / strain for a smoke run.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np

THETA_REF_DEG = 60.0
TOLERANCE = 0.15

# The 65 % freeze-bond lattice of RidgeMaterial never softens in this cell
# (it hardens to 10 % strain, see docs/BIAXIAL_SHEAR_BAND.md), so the benchmark
# uses a 35 % bonded, frictional specimen -- the material class that the
# literature shear-band references were measured on.
BOND_FRACTION = 0.35

CONFIGS = [
    {"name": "r16x32_p100", "columns": 16, "rows": 32, "confining_stress": 100.0e3},
    {"name": "r16x32_p200", "columns": 16, "rows": 32, "confining_stress": 200.0e3},
    {"name": "r24x48_p100", "columns": 24, "rows": 48, "confining_stress": 100.0e3},
    {"name": "r24x48_p200", "columns": 24, "rows": 48, "confining_stress": 200.0e3},
]


def _run_one(config: dict) -> dict:
    import torch
    torch.set_num_threads(1)
    from tensordem.biaxial_shear_band import run_biaxial_shear_band

    started = time.time()
    common = dict(confining_stress=config["confining_stress"],
                  columns=config["columns"], rows=config["rows"],
                  strain_rate=0.05, strain_target=0.12,
                  consolidation_steps=2000, snapshot_strain=0.01,
                  servo_gain=3.0e-4, servo_velocity_limit=2.0,
                  material_overrides={"freeze_bond_fraction": BOND_FRACTION})
    if config.get("quick"):
        common.update(strain_target=0.06, consolidation_steps=1200,
                      snapshot_strain=0.02, strain_rate=0.05)
    result = run_biaxial_shear_band(**common)
    result["name"] = config["name"]
    result["wall_time_s"] = time.time() - started
    result["config"] = {k: v for k, v in common.items()}
    stress = result["stress_path"]
    result["servo"] = {
        "target": float(config["confining_stress"]),
        "measured_mean": float(np.mean(stress["sigma_h"])),
        "measured_std": float(np.std(stress["sigma_h"])),
        "relative_offset": float(np.mean(stress["sigma_h"]) / config["confining_stress"] - 1.0),
    }
    result.pop("stress_path")
    return result


def _judge(result: dict) -> dict:
    band = result["band"]
    theta = band["angle_deg"]
    representative = band.get("representative") or {}
    cumulative = representative.get("cumulative") or {}
    rel = abs(theta - THETA_REF_DEG) / THETA_REF_DEG if np.isfinite(theta) else float("nan")
    localised = bool(band.get("resolved", False)) and (
        float(cumulative.get("z_score", 0.0)) > 3.0
        and float(cumulative.get("eta2", 0.0)) > 0.2)
    return {
        "theta_deg": theta,
        "theta_ref_deg": THETA_REF_DEG,
        "relative_error": rel,
        "within_tolerance": bool(np.isfinite(rel) and rel <= TOLERANCE),
        "localised": bool(localised),
        "z_score": float(cumulative.get("z_score", 0.0)),
        "eta2": float(cumulative.get("eta2", 0.0)),
        "angle_std_post_peak_deg": band["angle_std_deg"],
        "late_snapshots": band.get("late_snapshots"),
        "passed": bool(np.isfinite(rel) and rel <= TOLERANCE and localised),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="artifacts/biaxial_shear_band/results.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", default=None,
                        help="comma separated configuration names")
    args = parser.parse_args(argv)

    configs = [c for c in CONFIGS
               if args.only is None or c["name"] in args.only.split(",")]
    if args.quick:
        configs = [dict(configs[0], quick=True, name=configs[0]["name"] + "_quick")]

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if here not in sys.path:
        sys.path.insert(0, os.path.join(here, "src"))

    results = []
    if args.workers > 1 and len(configs) > 1:
        with mp.get_context("fork").Pool(min(args.workers, len(configs))) as pool:
            for result in pool.imap_unordered(_run_one, configs):
                results.append(result)
                print(f"[done] {result['name']} theta={result['band']['angle_deg']:.1f} deg",
                      flush=True)
    else:
        for config in configs:
            try:
                results.append(_run_one(config))
            except Exception as exc:                     # pragma: no cover - reported
                results.append({"name": config["name"], "error": repr(exc),
                                "traceback": traceback.format_exc()})
            print(f"[done] {results[-1].get('name')}", flush=True)

    judged = []
    for result in results:
        if "error" in result:
            judged.append({"name": result["name"], "error": result["error"],
                           "passed": False})
            continue
        verdict = _judge(result)
        judged.append({"name": result["name"], "verdict": verdict,
                       "peak": result["peak"], "theory": result["theory"],
                       "band": {"angle_deg": result["band"]["angle_deg"],
                                "angle_mean_deg": result["band"]["angle_mean_deg"],
                                "angle_std_deg": result["band"]["angle_std_deg"],
                                "resolved": result["band"]["resolved"],
                                "history": result["band"]["history"]},
                       "specimen": result["specimen"], "loading": result["loading"],
                       "servo": result["servo"], "broken_bonds": result["broken_bonds"],
                       "wall_time_s": result.get("wall_time_s")})

    resolved = [j for j in judged if "verdict" in j]
    payload = {
        "case": "CASE-BIAXIAL-SHEAR-BAND",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "external_anchor": {
            "theta_ref_deg": THETA_REF_DEG,
            "tolerance": TOLERANCE,
            "source": ("Desrues & Viggiani (2004) I/JNUMAG 28:279-321, dense Hostun sand "
                       "plane-strain biaxial: shear band 55-60 deg from sigma_3; "
                       "Mohr-Coulomb 45+phi/2 = 60 deg at phi = 30 deg"),
            "theory": {
                "mohr_coulomb": "theta = 45 + phi/2",
                "roscoe_1970": "theta = 45 + psi/2",
                "arthur_1977": "theta = 45 + (phi+psi)/4",
            },
        },
        "runs": judged,
        "summary": {
            "runs": len(judged),
            "resolved_runs": len(resolved),
            "passed": sum(1 for j in resolved if j["verdict"]["passed"]),
            "theta_deg": [round(j["verdict"]["theta_deg"], 2) for j in resolved],
            "relative_error": [round(j["verdict"]["relative_error"], 4) for j in resolved],
            "phi_deg": [round(j["peak"]["friction_angle_deg"], 2) for j in resolved],
            "psi_deg": [round(j["peak"]["dilation_angle_deg"], 2) for j in resolved],
        },
    }
    payload["summary"]["all_passed"] = (len(resolved) == len(judged)
                                        and payload["summary"]["passed"] == len(judged))

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=1, default=float)
    print(json.dumps(payload["summary"], indent=1))
    print(f"[written] {out_path}")
    return 0 if payload["summary"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())