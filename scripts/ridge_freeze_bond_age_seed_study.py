"""Young-age paired seed study for the ridge keel resistance (IN54W1-N05).

The full-deck sweep of ``run_ridge_freeze_bond_age.py`` gave a **non-monotone**
``R_keel(t)``: the 6 h keel came out weaker than the unbonded 0 h keel for one
of its two packings.  Because the DEM peak load is a packing-sensitive
quantity, that dip has to be resolved on *paired* seeds (same packing, only the
bond state differs) before it is reported as physics.

This script therefore

* runs the young ages ``(0, 6, 24) h`` on several seeds and reports the paired
  differences per seed (a paired difference removes the packing variance that
  dominates the ensemble mean);
* repeats the 6 h run with the bond *shear* capacity scaled by
  ``shear_probe_factor`` as a diagnostic for the hypothesis that a weak bond
  fails in shear on accumulated sliding rather than in tension.

Output: ``artifacts/ridge_freeze_bond_age/young_age_seed_study.json``.

Usage::

    python scripts/ridge_freeze_bond_age_seed_study.py [--seeds 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.ridge_dem import RidgeKeel, RidgeMaterial, generate_keel_assembly  # noqa: E402
from tensordem.ridge_freeze_bond_age import (  # noqa: E402
    FreezeBondAgeModel, aged_material, keel_resistance)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "artifacts", "ridge_freeze_bond_age")
AGES = (0.0, 6.0, 24.0)
BASE_SEEDS = (20261031, 20261068)
SPARE_SEEDS = (20261105, 20261142, 20261179, 20261216)
KEEL = RidgeKeel(keel_depth=8.0, keel_width=12.0, porosity=0.25)
MATERIAL = RidgeMaterial(particle_radius=0.5, freeze_bond_fraction=0.65,
                         bond_tensile_strength=100.0e3,
                         bond_shear_strength=200.0e3)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--shear-probe-factor", type=float, default=10.0)
    parser.add_argument("--press-velocity", type=float, default=0.02)
    parser.add_argument("--stroke", type=float, default=0.10)
    args = parser.parse_args(argv)
    started = time.time()
    seeds = (BASE_SEEDS + SPARE_SEEDS)[: max(args.seeds, 2)]
    model = FreezeBondAgeModel.calibrated(thermal_length=MATERIAL.particle_radius)
    runs = {}
    for age in AGES:
        for seed in seeds:
            aged = aged_material(MATERIAL, model, age, maximum_fraction=0.65)
            aged = replace(aged, seed=int(seed))
            out = keel_resistance(KEEL, aged, press_velocity=args.press_velocity,
                                  stroke=args.stroke, bond_contact_mode="parallel")
            runs[f"{age}|{seed}"] = out
            print(f"[pair] t={age:5.1f} h seed={seed} peak={out['peak_n_per_m']/1e3:8.2f} "
                  f"res={out['residual_n_per_m']/1e3:8.2f} bonds={out['bonds']} "
                  f"broken={out['broken_bonds']}", flush=True)

    paired = []
    for seed in seeds:
        row = {"seed": int(seed),
               "peak_n_per_m": {str(age): runs[f"{age}|{seed}"]["peak_n_per_m"]
                                for age in AGES},
               "residual_n_per_m": {str(age): runs[f"{age}|{seed}"]["residual_n_per_m"]
                                    for age in AGES}}
        row["delta_6h_minus_0h_n_per_m"] = (row["peak_n_per_m"]["6.0"]
                                            - row["peak_n_per_m"]["0.0"])
        row["ratio_6h_over_0h"] = (row["peak_n_per_m"]["6.0"]
                                   / row["peak_n_per_m"]["0.0"])
        paired.append(row)

    # shear-capacity diagnostic at 6 h, same packings
    probe = {}
    for seed in seeds:
        aged = aged_material(MATERIAL, model, 6.0, maximum_fraction=0.65)
        stiff = replace(aged, seed=int(seed),
                        bond_shear_strength=args.shear_probe_factor
                        * aged.bond_shear_strength)
        out = keel_resistance(KEEL, stiff, press_velocity=args.press_velocity,
                              stroke=args.stroke, bond_contact_mode="parallel")
        probe[str(seed)] = {"peak_n_per_m": out["peak_n_per_m"],
                            "residual_n_per_m": out["residual_n_per_m"],
                            "broken_bonds": out["broken_bonds"],
                            "bonds": out["bonds"],
                            "shear_capacity_factor": args.shear_probe_factor}
        print(f"[shear x{args.shear_probe_factor:g}] seed={seed} "
              f"peak={out['peak_n_per_m']/1e3:8.2f} broken={out['broken_bonds']}/{out['bonds']}",
              flush=True)

    ratios = [row["ratio_6h_over_0h"] for row in paired]
    dips = sum(1 for value in ratios if value < 1.0)
    payload = {
        "node": "IN54W1-N05",
        "mission_id": 7096,
        "ages_hours": list(AGES),
        "seeds": [int(seed) for seed in seeds],
        "apparatus": {"keel_depth_m": KEEL.keel_depth, "keel_width_m": KEEL.keel_width,
                      "porosity": KEEL.porosity,
                      "particle_radius_m": MATERIAL.particle_radius,
                      "press_velocity_m_per_s": args.press_velocity,
                      "stroke_m": args.stroke,
                      "bond_contact_mode": "parallel"},
        "bonds": int(generate_keel_assembly(KEEL, MATERIAL)["info"]["bonds"]),
        "particles": int(generate_keel_assembly(KEEL, MATERIAL)["info"]["particles"]),
        "paired": paired,
        "ensemble": {
            "peak_mean_n_per_m": {str(age): sum(r["peak_n_per_m"][str(age)]
                                                for r in paired) / len(paired)
                                  for age in AGES},
            "peak_sd_n_per_m": {str(age): (sum((r["peak_n_per_m"][str(age)]
                                                - sum(q["peak_n_per_m"][str(age)]
                                                      for q in paired) / len(paired)) ** 2
                                               for r in paired) / len(paired)) ** 0.5
                                for age in AGES},
            "ratio_6h_over_0h_mean": sum(ratios) / len(ratios),
            "ratio_6h_over_0h_min": min(ratios),
            "ratio_6h_over_0h_max": max(ratios),
            "seeds_with_6h_below_0h": int(dips),
            "verdict": ("6 h keel weaker than the unbonded keel for "
                        f"{dips}/{len(ratios)} paired packings"),
        },
        "shear_capacity_probe": probe,
        "runtime_seconds": time.time() - started,
    }
    path = os.path.join(OUT, "young_age_seed_study.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print(f"[study] {payload['ensemble']['verdict']} -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())