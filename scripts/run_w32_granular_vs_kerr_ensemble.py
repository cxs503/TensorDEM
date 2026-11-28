#!/usr/bin/env python3
"""W32-5: Granular DEM vs Kerr -- multi-seed ensemble for ice-ridge keel grounding.

Closes the gap left by m28180 (report #25068): that run reported the DEM-vs-Kerr
resistance relative error (max 10.6% <= 15%) from a *single* seed (seed=7) and did
not provide the ensemble standard deviation.  This script runs the same
bonded-particle keel pipeline at N >= 3 independent seeds and reports:

  * per-seed footing resistance R_dem [N/m] at b = 0.5Hk, 1.0Hk, 2.0Hk,
  * the ensemble standard deviation of R_dem across seeds (as CV = std/mean %),
  * the per-seed DEM-vs-Kerr relative error (each seed's Kerr parameters are
    identified from that seed's own confined-compression K and narrow-footing
    sqrt(G2*K)).

Machine-judged criteria (both must hold):
  * max |R_kerr - R_dem| / R_dem  <= 15%  (engineering criterion)
  * max ensemble CV of R_dem      <= 10%  (ensemble reproducibility)

If the CV exceeds 10% the artifact records an honest FAIL with attribution.

Output: papers/data/w32-2026-11-16/ice/granular_vs_kerr.json (updated, ensemble
section added; seed=7 block is a deterministic re-run of the m28180 numbers).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tensordem.keel_foundation_dem import (
    build_keel_sample,
    confined_compression,
    footing_test,
    simple_shear,
    ICE_YOUNG,
    REFERENCE_LENGTH,
)
from tensordem.ridge_keel_grounding import (
    KerrFoundation,
    kerr_footing_resistance,
    kerr_footing_selfcheck,
)

# Independent seeds for the granular ensemble (>= 3 required; 5 used).
SEEDS = [7, 8, 9, 10, 11]


def run_one_seed(seed: int, *, n_particles: int, height: float, cell_ratio: float,
                 solid_fraction: float, w0: float, increments: int,
                 steps_per_increment: int) -> dict:
    """Run the full granular-vs-Kerr pipeline for one seed and return its record."""
    t0 = time.time()
    sample = build_keel_sample(
        n_particles,
        height=height,
        cell_ratio=cell_ratio,
        solid_fraction=solid_fraction,
        seed=seed,
        ice_young=ICE_YOUNG,
        bond_reference_length=REFERENCE_LENGTH,  # spacing_consistent
    )
    build_time = time.time() - t0
    n_bonds = len(sample.bi)

    # Confined compression -> K (series modulus)
    t0 = time.time()
    confined = confined_compression(
        sample, w_max=w0, increments=increments,
        steps_per_increment=steps_per_increment,
    )
    confined_time = time.time() - t0

    # Simple shear -> G_s (diagnostic)
    t0 = time.time()
    shear = simple_shear(
        sample, gamma_max=w0 / height, increments=increments,
        steps_per_increment=steps_per_increment,
    )
    shear_time = time.time() - t0

    # Footing tests at b = 0.5Hk, 1.0Hk, 2.0Hk
    footing_widths = [0.5 * height, 1.0 * height, 2.0 * height]
    footing_results = {}
    for b in footing_widths:
        b_ratio = b / height
        t0 = time.time()
        ft = footing_test(
            sample, footing_width=b, w0=w0, increments=increments,
            steps_per_increment=steps_per_increment,
        )
        ft_time = time.time() - t0
        footing_results[f"{b_ratio:.1f}"] = {
            "footing_width_m": b,
            "settlements_m": [float(s) for s in ft["settlements_m"]],
            "resistance_N_per_m": [float(r) for r in ft["resistance_N_per_m"]],
            "secant_stiffness_N_per_m2": ft["secant_stiffness_N_per_m2"],
            "n_footing_particles": ft["n_footing_particles"],
            "wall_time_s": ft_time,
        }

    # Kerr identification: K from confined, sqrt(G2*K) from narrow footing (b=0.5Hk)
    K = confined["k_subgrade_N_per_m3"]
    b_narrow = footing_results["0.5"]["footing_width_m"]
    R_narrow = footing_results["0.5"]["resistance_N_per_m"][-1]
    sqrt_G2K = (R_narrow - K * w0 * b_narrow) / (2.0 * w0)
    G2 = sqrt_G2K ** 2 / K
    k_upper = 2.0 * K
    k_lower = 2.0 * K
    G = 2.0 * G2

    kerr_foundation = KerrFoundation(k_upper=k_upper, k_lower=k_lower, shear=G)
    selfcheck = kerr_footing_selfcheck(width=1.0, w0=w0, foundation=kerr_foundation)

    # Predict resistance at all widths with Kerr
    predictions = {}
    for b_ratio_str, ft_data in footing_results.items():
        b = ft_data["footing_width_m"]
        R_kerr = kerr_footing_resistance(width=b, w0=w0, foundation=kerr_foundation)
        R_dem = ft_data["resistance_N_per_m"][-1]
        rel_error = (R_kerr - R_dem) / R_dem if R_dem != 0 else float("nan")
        predictions[b_ratio_str] = {
            "footing_width_m": b,
            "R_dem_N_per_m": R_dem,
            "R_kerr_N_per_m": R_kerr,
            "rel_error": rel_error,
            "rel_error_pct": rel_error * 100,
        }

    return {
        "seed": seed,
        "dem_specimen": {
            "n_particles": sample.n_particles,
            "n_bonds": n_bonds,
            "coordination": sample.coordination,
            "length_m": sample.length,
            "height_m": sample.height,
            "spacing_m": sample.spacing,
            "mean_radius_m": float(np.mean(sample.r)),
            "solid_fraction": sample.solid_fraction,
            "kPa_per_bond": float(np.mean(sample.kn) / 1e3),
            "seed": sample.rng_seed,
            "bond_law": sample.meta["bond_law"],
            "bond_reference_length_m": sample.meta.get("bond_reference_length_m"),
        },
        "dem_results": {
            "confined_compression": {
                "test": "confined_compression",
                "settlements_m": [float(s) for s in confined["settlements_m"]],
                "pressures_Pa": [float(p) for p in confined["pressures_Pa"]],
                "k_subgrade_small_strain_N_per_m3": confined["k_subgrade_small_strain_N_per_m3"],
                "k_subgrade_N_per_m3": confined["k_subgrade_N_per_m3"],
                "wall_time_s": confined_time,
            },
            "simple_shear": {
                "test": "simple_shear",
                "gammas": [float(g) for g in shear["gammas"]],
                "taus_Pa": [float(t) for t in shear["taus_Pa"]],
                "G_skeleton_Pa": shear["G_skeleton_Pa"],
                "wall_time_s": shear_time,
            },
            "footings": footing_results,
            "total_wall_time_s": build_time + confined_time + shear_time + sum(
                ft["wall_time_s"] for ft in footing_results.values()
            ),
        },
        "kerr_identification": {
            "method": "K from confined compression, sqrt(G2*K) from narrow footing (b=0.5Hk)",
            "parameters": {
                "K_N_per_m3": K,
                "G2_N_per_m": G2,
                "sqrt_G2K_N_per_m1_5": sqrt_G2K,
                "k_upper_N_per_m3": k_upper,
                "k_lower_N_per_m3": k_lower,
                "shear_N_per_m": G,
                "c_m2": G / (k_upper + k_lower),
            },
            "foundation_dict": kerr_foundation.as_dict(),
        },
        "kerr_selfcheck": selfcheck,
        "comparison": {
            "w0_m": w0,
            "predictions": predictions,
            "summary": {
                "max_abs_rel_error": max(abs(p["rel_error"]) for p in predictions.values()),
                "max_abs_rel_error_pct": max(abs(p["rel_error_pct"]) for p in predictions.values()),
                "all_pass_15pct": all(abs(p["rel_error"]) <= 0.15 for p in predictions.values()),
            },
        },
    }


def main() -> None:
    n_particles = 50000
    height = 2.0
    cell_ratio = 8.0
    solid_fraction = 0.62
    w0 = 0.008
    increments = 6
    steps_per_increment = 100

    print(f"=== W32-5 granular ensemble: {len(SEEDS)} seeds ===")
    per_seed = []
    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")
        rec = run_one_seed(
            seed, n_particles=n_particles, height=height, cell_ratio=cell_ratio,
            solid_fraction=solid_fraction, w0=w0, increments=increments,
            steps_per_increment=steps_per_increment,
        )
        per_seed.append(rec)
        for b_ratio_str, pred in rec["comparison"]["predictions"].items():
            print(f"  b={pred['footing_width_m']:.2f}m ({b_ratio_str}*Hk): "
                  f"DEM={pred['R_dem_N_per_m']:.2e} N/m, "
                  f"Kerr={pred['R_kerr_N_per_m']:.2e} N/m, "
                  f"err={pred['rel_error_pct']:.1f}%")

    # ---- Ensemble statistics on R_dem across seeds ----
    widths = ["0.5", "1.0", "2.0"]
    ensemble = {}
    all_rel_errors = []
    for b_ratio_str in widths:
        R_dem_list = [rec["comparison"]["predictions"][b_ratio_str]["R_dem_N_per_m"]
                      for rec in per_seed]
        R_kerr_list = [rec["comparison"]["predictions"][b_ratio_str]["R_kerr_N_per_m"]
                       for rec in per_seed]
        rel_list = [rec["comparison"]["predictions"][b_ratio_str]["rel_error"]
                    for rec in per_seed]
        all_rel_errors.extend(rel_list)
        R_dem_arr = np.array(R_dem_list)
        mean = float(R_dem_arr.mean())
        std = float(R_dem_arr.std(ddof=1))  # sample std (N-1)
        cv = std / mean * 100.0 if mean != 0 else float("nan")
        ensemble[b_ratio_str] = {
            "footing_width_m": per_seed[0]["comparison"]["predictions"][b_ratio_str]["footing_width_m"],
            "R_dem_per_seed_N_per_m": R_dem_list,
            "R_kerr_per_seed_N_per_m": R_kerr_list,
            "rel_error_per_seed": rel_list,
            "mean_R_dem_N_per_m": mean,
            "std_R_dem_N_per_m": std,
            "cv_pct": cv,
            "n_seeds": len(SEEDS),
        }
        print(f"\n  [{b_ratio_str}*Hk] R_dem across seeds: "
              f"mean={mean:.2e} N/m, std={std:.2e} N/m, CV={cv:.2f}%")

    max_abs_rel_error = max(abs(e) for e in all_rel_errors)
    max_cv = max(ensemble[b]["cv_pct"] for b in widths)
    print(f"\n=== Ensemble summary ===")
    print(f"  max |DEM-Kerr| rel error (all seeds, all widths): {max_abs_rel_error*100:.2f}%")
    print(f"  max ensemble CV of R_dem: {max_cv:.2f}%")
    print(f"  Criterion 1 (rel err <= 15%): {'PASS' if max_abs_rel_error <= 0.15 else 'FAIL'}")
    print(f"  Criterion 2 (CV <= 10%):     {'PASS' if max_cv <= 10.0 else 'FAIL'}")

    # ---- Reproducibility: seed=7 must match the m28180 artifact ----
    prev_path = Path(__file__).resolve().parent.parent / "papers" / "data" / "w32-2026-11-16" / "ice" / "granular_vs_kerr.json"
    repro = None
    if prev_path.exists():
        prev = json.loads(prev_path.read_text())
        prev_R = {k: v["R_dem_N_per_m"] for k, v in prev["comparison"]["predictions"].items()}
        cur_R = {k: v["R_dem_N_per_m"] for k, v in per_seed[0]["comparison"]["predictions"].items()}
        repro = {
            "seed": 7,
            "prev_R_dem_N_per_m": prev_R,
            "rerun_R_dem_N_per_m": cur_R,
            "max_abs_rel_dev": max(abs(cur_R[k] - prev_R[k]) / prev_R[k] for k in prev_R),
        }
        print(f"\n  Reproducibility (seed=7 vs m28180): max rel dev = "
              f"{repro['max_abs_rel_dev']*100:.2e}%")

    # ---- Assemble artifact ----
    artifact = {
        "artifact": "papers/data/w32-2026-11-16/ice/granular_vs_kerr.json",
        "mission": "paper-exp-ice-ridge-keel-grounding-5",
        "mission_id": 28264,
        "repo": "TensorDEM",
        "runner": "native",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration": {
            "keel_height_m": height,
            "cell_ratio": cell_ratio,
            "solid_fraction": solid_fraction,
            "w0_m": w0,
            "n_particles": n_particles,
            "seeds": SEEDS,
            "n_seeds": len(SEEDS),
            "increments": increments,
            "steps_per_increment": steps_per_increment,
            "bond_law": "spacing_consistent",
            "bond_reference_length_m": REFERENCE_LENGTH,
            "ice_young_Pa": ICE_YOUNG,
            "footing_widths_m": [0.5 * height, 1.0 * height, 2.0 * height],
        },
        "per_seed": per_seed,
        "ensemble": {
            "quantity": "R_dem_N_per_m (DEM footing resistance at w0=0.008m)",
            "statistic": "sample std (ddof=1) / mean, expressed as CV %",
            "widths": ensemble,
            "summary": {
                "max_abs_rel_error_all_seeds": max_abs_rel_error,
                "max_abs_rel_error_all_seeds_pct": max_abs_rel_error * 100,
                "max_cv_pct": max_cv,
                "criterion_rel_err_15pct": max_abs_rel_error <= 0.15,
                "criterion_cv_10pct": max_cv <= 10.0,
                "verdict": "PASS" if (max_abs_rel_error <= 0.15 and max_cv <= 10.0) else "FAIL",
            },
        },
        "reproducibility_vs_m28180": repro,
        "limitations": [
            "Small-strain only (w0/Hk = 0.004), no bond breakage",
            "Plane strain, periodic lateral boundaries",
            "Kerr layer cannot resolve k_u/k_l split from footing test alone",
            "DEM modulus is a lattice modulus, not a continuum ice modulus",
            "Quasi-static solver: no inertia physics claimed",
            "Ensemble std is over lattice-realisation seeds only (same geometry/loading); "
            "it does not cover loading-path or material-parameter uncertainty",
        ],
    }

    output_dir = Path(__file__).resolve().parent.parent / "papers" / "data" / "w32-2026-11-16" / "ice"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "granular_vs_kerr.json"
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nArtifact written to {output_path}")


if __name__ == "__main__":
    main()
