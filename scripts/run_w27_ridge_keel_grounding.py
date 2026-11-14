#!/usr/bin/env python3
"""W27 · ice-ridge keel grounding: Kerr beam / bonded-keel reconciliation.

Runs the three evidence blocks of
``papers/artifacts/w27/ice-ridge-keel-grounding/ridge_keel_anchor.json``:

  A. Kerr analytic beam vs an independent mixed-Galerkin FEM (deflection profile),
     plus a mesh-refinement GCI on the FE solution.
  B. Bonded-granular keel (plane-strain DEM) footing tests at 25k / 50k / 100k
     particles: subgrade modulus, footing resistance vs footing width, particle
     refinement GCI.
  C. Two-point identification of the Kerr layer from (A) and (B) and a predeclared
     out-of-sample prediction at the validation footing width.

Numbers are written by the script only (zero hand copying).  Usage::

    PYTHONPATH=src python scripts/run_w27_ridge_keel_grounding.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

from tensordem import ridge_keel_grounding as rg
from tensordem import keel_foundation_dem as kd

OUT_DIR = os.path.join("papers", "artifacts", "w27", "ice-ridge-keel-grounding")

# ---------------------------------------------------------------------------
# predeclared configuration (fixed before the results were inspected)
# ---------------------------------------------------------------------------
KEEL_HEIGHT = 2.0          # m
CELL_RATIO = 8.0           # periodic cell width / keel height
SOLID_FRACTION = 0.62      # = 1 - porosity (keel rubble n = 0.38)
W0_FRACTION = 4.0e-3       # footing settlement / keel height
PARTICLE_COUNTS = (25000, 50000, 100000)      # GCI triad: fine/medium/coarse
FOOTING_WIDTHS = (0.5, 1.0, 2.0)              # x keel height: 1.0 = identification
IDENT_WIDTH = 0.5                             # narrow footing: identifies the Kerr layer
VALID_WIDTHS = (1.0, 2.0)                     # 1.0 = predeclared validation point (b = H_k)
INCREMENTS = 6
STEPS_PER_INCREMENT = 300

# three Kerr foundation parameter sets for the beam verification
KERR_SETS = {
    "soft_shear_layer": dict(ei=8.0e7, k_upper=1.0e7, k_lower=4.0e7, shear=1.0e8),
    "balanced": dict(ei=1.0e8, k_upper=2.0e7, k_lower=8.0e7, shear=4.0e8),
    "stiff_thin_layer": dict(ei=2.0e8, k_upper=5.0e7, k_lower=2.0e8, shear=2.0e9),
}


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# A. Kerr analytic vs independent FEM
# ---------------------------------------------------------------------------
def kerr_vs_fem() -> dict:
    a = 2.0
    q0 = 1.0e5
    x = np.linspace(-30.0, 30.0, 1201)
    cases = []
    for name, cfg in KERR_SETS.items():
        fnd = rg.KerrFoundation(k_upper=cfg["k_upper"], k_lower=cfg["k_lower"],
                                shear=cfg["shear"])
        an = rg.kerr_beam_analytic(a, q0, ei=cfg["ei"], foundation=fnd, x=x)
        fem = rg.kerr_beam_fem(np.linspace(-30.0, 30.0, 1201), ei=cfg["ei"],
                               foundation=fnd,
                               load=lambda xx: np.where(np.abs(xx) <= a, q0, 0.0))
        wa = np.interp(x, an.x, an.w)
        wf = np.interp(x, fem.x, fem.w)
        near = np.abs(x) <= 5.0 * a
        denom = float(np.max(np.abs(wa[near])))
        dev = float(np.max(np.abs(wf - wa)[near])) / denom
        pa = np.interp(x, an.x, an.p)
        pf = np.interp(x, fem.x, fem.p)
        pdev = float(np.max(np.abs(pf - pa)[near])) / float(np.max(np.abs(pa[near])))
        cases.append({
            "name": name, "foundation": fnd.as_dict(), "ei_N_m2": cfg["ei"],
            "patch_half_width_m": a, "line_load_N_per_m": q0,
            "w_centre_analytic_m": an.meta["w_centre_m"],
            "w_centre_fem_m": float(np.interp(0.0, fem.x, fem.w)),
            "max_rel_dev_deflection_absx_le_5a": dev,
            "max_rel_dev_reaction_absx_le_5a": pdev,
            "reaction_analytic_N_per_m": an.meta["reaction_quadrature_N_per_m"],
            "reaction_exact_N_per_m": an.meta["reaction_total_N_per_m"],
            "reaction_fem_N_per_m": fem.meta["reaction_total_N_per_m"],
            "applied_total_N_per_m": 2.0 * a * q0,
            "lambda_right": [list(map(float, (z.real, z.imag))) for z in an.meta["lambda_right"]],
        })
    # mesh refinement GCI on the balanced case
    cfg = KERR_SETS["balanced"]
    fnd = rg.KerrFoundation(k_upper=cfg["k_upper"], k_lower=cfg["k_lower"], shear=cfg["shear"])
    mesh = []
    for ne in (75, 150, 300, 600):
        fem = rg.kerr_beam_fem(np.linspace(-30.0, 30.0, ne + 1), ei=cfg["ei"],
                               foundation=fnd,
                               load=lambda xx: np.where(np.abs(xx) <= a, q0, 0.0))
        mesh.append({"n_elements": ne,
                     "w_centre_m": float(np.interp(0.0, fem.x, fem.w)),
                     "max_deflection_m": float(np.max(np.abs(fem.w)))})
    triad = [m["w_centre_m"] for m in mesh[:3]]          # 75/150/300 -> fine->coarse
    gci_mesh = rg.gci_triad(triad, r=2.0)
    return {"cases": cases, "mesh_refinement": mesh, "mesh_gci": gci_mesh,
            "max_rel_dev_all_cases": max(c["max_rel_dev_deflection_absx_le_5a"] for c in cases)}


# ---------------------------------------------------------------------------
# B. bonded-keel DEM footing study
# ---------------------------------------------------------------------------
def dem_resolution(n_particles: int) -> dict:
    t0 = time.time()
    sample = kd.build_keel_sample(n_particles, height=KEEL_HEIGHT, cell_ratio=CELL_RATIO,
                                  solid_fraction=SOLID_FRACTION, seed=7)
    w0 = W0_FRACTION * KEEL_HEIGHT
    confined = kd.confined_compression(sample, w_max=w0, increments=INCREMENTS,
                                       steps_per_increment=STEPS_PER_INCREMENT)
    shear = kd.simple_shear(sample, increments=INCREMENTS,
                            steps_per_increment=STEPS_PER_INCREMENT)
    footings = {}
    for factor in FOOTING_WIDTHS:
        b = factor * KEEL_HEIGHT
        footings[f"{factor:g}"] = kd.footing_test(
            sample, footing_width=b, w0=w0, increments=INCREMENTS,
            steps_per_increment=STEPS_PER_INCREMENT)
    return {"n_particles_requested": n_particles,
            "sample": sample.as_dict(), "w0_m": w0,
            "confined": confined, "shear": shear, "footings": footings,
            "wall_time_s": time.time() - t0}


def dem_relaxation_adequacy() -> dict:
    """Stopping-rule check: does the footing resistance change with more relaxation?"""
    sample = kd.build_keel_sample(PARTICLE_COUNTS[0], height=KEEL_HEIGHT,
                                  cell_ratio=CELL_RATIO, solid_fraction=SOLID_FRACTION, seed=7)
    w0 = W0_FRACTION * KEEL_HEIGHT
    out = {}
    for steps in (150, 300, 600):
        f = kd.footing_test(sample, footing_width=KEEL_HEIGHT, w0=w0,
                            increments=INCREMENTS, steps_per_increment=steps)
        out[str(steps)] = f["resistance_N_per_m"][-1]
    vals = [out[k] for k in ("150", "300", "600")]
    return {"resistance_vs_steps": out,
            "rel_change_300_to_600": abs(vals[2] - vals[1]) / max(abs(vals[2]), 1e-300)}


# ---------------------------------------------------------------------------
# C. reduction: Kerr layer from the DEM, predeclared validation
# ---------------------------------------------------------------------------
def reduce_keel_to_kerr(dem: dict) -> dict:
    w0 = dem["w0_m"]
    p_series = np.array(dem["confined"]["pressures_Pa"])
    ws = np.array(dem["confined"]["settlements_m"])
    k_subgrade = float(p_series[-1] / ws[-1])            # secant at the test settlement
    r_by_width = {}
    for key, foot in dem["footings"].items():
        r_by_width[key] = {"footing_width_m": foot["footing_width_m"],
                           "resistance_N_per_m": float(foot["resistance_N_per_m"][-1]),
                           "per_unit_width_N_per_m2": float(foot["resistance_N_per_m"][-1]
                                                           / foot["footing_width_m"])}
    ident = r_by_width[f"{IDENT_WIDTH:g}"]
    # Kerr: R = K w0 b + 2 w0 sqrt(G2 K)  ->  identify sqrt(G2 K) from one footing
    edge_term = ident["resistance_N_per_m"] - k_subgrade * w0 * ident["footing_width_m"]
    sqrt_g2k = edge_term / (2.0 * w0)
    g2 = sqrt_g2k ** 2 / k_subgrade
    # homogeneous-layer reduction k_u = k_l = 2K (the only split that keeps the
    # three-parameter model consistent with a one-layer specimen)
    # k_u = k_l = 2K keeps the series modulus equal to the measured K; then
    # G2 = k_u G/(k_u+k_l) = G/2 -> G = 2 G2.
    k_layer = 2.0 * k_subgrade
    foundation = rg.KerrFoundation(k_upper=k_layer, k_lower=k_layer, shear=2.0 * g2)
    predictions = {}
    for key, item in r_by_width.items():
        b = item["footing_width_m"]
        pred = rg.kerr_footing_resistance(width=b, w0=w0, foundation=foundation)
        predictions[key] = {
            "footing_width_m": b, "dem_N_per_m": item["resistance_N_per_m"],
            "kerr_N_per_m": pred,
            "rel_dev": abs(pred - item["resistance_N_per_m"]) / item["resistance_N_per_m"],
            "role": ("identification" if abs(b - IDENT_WIDTH * KEEL_HEIGHT) < 1e-9
                     else "validation"),
        }
    # linear structure of R(b) (a property of the reduced model, checked on the DEM)
    widths = np.array([item["footing_width_m"] for item in r_by_width.values()])
    res = np.array([item["resistance_N_per_m"] for item in r_by_width.values()])
    slope, intercept = np.polyfit(widths, res, 1)
    lin_resid = float(np.max(np.abs(np.polyval([slope, intercept], widths) - res)
                             / np.max(np.abs(res))))
    return {"k_subgrade_N_per_m3": k_subgrade,
            "confined_pressure_settlement": {"settlements_m": ws.tolist(),
                                             "pressures_Pa": p_series.tolist()},
            "identified_sqrt_G2K_N_per_m": sqrt_g2k, "identified_G2_N_per_m": g2,
            "kerr_foundation": foundation.as_dict(),
            "foundation_note": ("homogeneous-layer reduction k_u = k_l = 2K, so "
                                "G = 2 G2 and c = G/(k_u+k_l) = G/(4K); the k_u/k_l "
                                "split is NOT identifiable from a footing test because "
                                "R = K w0 b + 2 w0 sqrt(G2 K) does not depend on c"),
            "footing_predictions": predictions,
            "linear_structure": {"slope_N_per_m2": float(slope),
                                 "intercept_N_per_m": float(intercept),
                                 "slope_over_K_w0": float(slope / (k_subgrade * w0)),
                                 "intercept_over_2w0_sqrtG2K": float(intercept / edge_term),
                                 "max_rel_residual_vs_linear": lin_resid,
                                 "n_widths": int(widths.size)},
            "footing_selfcheck": rg.kerr_footing_selfcheck(width=2.0, w0=w0, foundation=foundation)}


def particle_gci(dems: list[dict]) -> dict:
    """Particle-count GCI of the validation-footing resistance (100k/50k/25k)."""
    ordered = sorted(dems, key=lambda d: -d["n_particles_requested"])
    vals = [d["footings"]["1"]["resistance_N_per_m"][-1] for d in ordered]
    out = rg.gci_triad(vals, r=2.0)
    out["levels_particles_fine_to_coarse"] = [d["n_particles_requested"] for d in ordered]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "ridge_keel_anchor.json"))
    args = ap.parse_args()
    t_start = time.time()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    counts = PARTICLE_COUNTS[:2] if args.quick else PARTICLE_COUNTS
    print("[A] Kerr analytic vs independent FEM ...", flush=True)
    block_a = kerr_vs_fem()

    print("[B] bonded-keel DEM at", counts, "...", flush=True)
    cache_dir = os.environ.get("W27_CACHE_DIR", "/dev/shm/w27_dem_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cached = {}
    for n in counts:
        path = os.path.join(cache_dir, f"dem_{n}.json")
        if os.path.exists(path):
            with open(path) as fh:
                cached[n] = json.load(fh)
            print(f"    reuse cached DEM n={n}", flush=True)
    todo = [n for n in counts if n not in cached]
    if todo:
        with Pool(processes=len(todo)) as pool:
            for n, res in zip(todo, pool.map(dem_resolution, todo)):
                with open(os.path.join(cache_dir, f"dem_{n}.json"), "w") as fh:
                    json.dump(res, fh)
                cached[n] = res
    dems = [cached[n] for n in counts]
    adeq_path = os.path.join(os.environ.get("W27_CACHE_DIR", "/dev/shm/w27_dem_cache"), "adequacy.json")
    if os.path.exists(adeq_path):
        with open(adeq_path) as fh:
            adequacy = json.load(fh)
    else:
        adequacy = dem_relaxation_adequacy()
        with open(adeq_path, "w") as fh:
            json.dump(adequacy, fh)

    dem_summary = []
    reductions = []
    for d in dems:
        dem_summary.append({
            "n_particles_requested": d["n_particles_requested"],
            "n_particles": d["sample"]["n_particles"],
            "n_bonds": d["sample"]["n_bonds"],
            "coordination": d["sample"]["coordination"],
            "spacing_m": d["sample"]["spacing_m"],
            "wall_time_s": d["wall_time_s"],
            "k_subgrade_N_per_m3": d["confined"]["k_subgrade_N_per_m3"],
            "k_subgrade_small_strain_N_per_m3":
                d["confined"]["k_subgrade_small_strain_N_per_m3"],
            "G_skeleton_Pa": d["shear"]["G_skeleton_Pa"],
            "resistance_N_per_m": {k: v["resistance_N_per_m"][-1]
                                   for k, v in d["footings"].items()},
            "n_footing_particles": {k: v["n_footing_particles"]
                                    for k, v in d["footings"].items()},
        })
        reductions.append(reduce_keel_to_kerr(d))

    gci_particles = particle_gci(dems) if len(dems) >= 3 else None
    validation = reductions[-1]["footing_predictions"]
    primary_key = f"{VALID_WIDTHS[0]:g}"
    val_dev = validation[primary_key]["rel_dev"]
    secondary = {k: v["rel_dev"] for k, v in validation.items()
                 if v["role"] == "validation" and k != primary_key}
    extra = [validation[k]["rel_dev"] for k in validation if validation[k]["role"] != "validation"]

    op_checks = rg.pressure_area_operator_checks()
    block_a_pass = block_a["max_rel_dev_all_cases"] < 0.03
    block_b_pass = val_dev < 0.15
    block_c_pass = bool(gci_particles and gci_particles["gci_fine_rel"] < 0.05)
    mesh_pass = bool(block_a["mesh_gci"]["gci_fine_rel"] < 0.05)

    criteria = [
        {"id": "C1", "criterion": "verbatim ISO 19906:2019 A.8 clause registered with "
                                  "matching sha256, or honest WITHHELD + operator-layer checks",
         "verdict": "WITHHELD", "pass": None,
         "detail": "primary standard text not present in this environment; operator layer "
                   f"machine-checked all_pass={op_checks['all_pass']} (see anchor file)"},
        {"id": "C2", "criterion": "Kerr analytic deflection vs independent FEM < 3% "
                                  "(max relative deviation, |x| <= 5a)",
         "verdict": "PASS" if block_a_pass else "FAIL", "pass": bool(block_a_pass),
         "value": block_a["max_rel_dev_all_cases"]},
        {"id": "C3", "criterion": "bonded keel vs Kerr analytic resistance < 15% at the "
                                  "predeclared validation footing width b = 1.0 H_k "
                                  "(Kerr layer identified from the confined test + one "
                                  "narrow footing b = 0.5 H_k)",
         "verdict": "PASS" if block_b_pass else "FAIL", "pass": bool(block_b_pass),
         "value": val_dev, "identification_width_m": IDENT_WIDTH * KEEL_HEIGHT,
         "validation_width_m": VALID_WIDTHS[0] * KEEL_HEIGHT,
         "secondary_validation_rel_dev": secondary},
        {"id": "C4", "criterion": "particle-count (50k/100k) and mesh refinement GCI < 5%",
         "verdict": "PASS" if (block_c_pass and mesh_pass) else "FAIL",
         "pass": bool(block_c_pass and mesh_pass),
         "particle_gci": gci_particles, "mesh_gci": block_a["mesh_gci"]},
    ]

    artifact = {
        "artifact": args.out.replace(os.sep, "/"),
        "mission": "paper-exp-ice-ridge-keel-grounding-1-rework1",
        "mission_id": 18963,
        "repo": "TensorDEM",
        "runner": "native",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "generator": "scripts/run_w27_ridge_keel_grounding.py",
        "environment": {"python": sys.version.split()[0],
                        "numpy": np.__version__, "platform": platform.platform(),
                        "nproc": os.cpu_count()},
        "configuration": {"keel_height_m": KEEL_HEIGHT, "cell_ratio": CELL_RATIO,
                          "solid_fraction": SOLID_FRACTION,
                          "w0_fraction_of_height": W0_FRACTION,
                          "w0_m": W0_FRACTION * KEEL_HEIGHT,
                          "particle_counts": list(counts),
                          "footing_width_factors": list(FOOTING_WIDTHS),
                          "increments": INCREMENTS,
                          "steps_per_increment": STEPS_PER_INCREMENT,
                          "kerr_sets": KERR_SETS},
        "A_kerr_beam_verification": block_a,
        "B_bonded_keel_dem": {"summary": dem_summary,
                              "full": [{"n_particles_requested": d["n_particles_requested"],
                                        "confined": d["confined"], "shear": d["shear"],
                                        "footings": d["footings"]} for d in dems],
                              "relaxation_adequacy": adequacy},
        "C_keel_to_kerr_reduction": reductions[-1],
        "particle_gci": gci_particles,
        "criteria": criteria,
        "iso_19906_operator_layer": op_checks,
        "wall_time_s": time.time() - t_start,
        "provenance": {
            "command": "PYTHONPATH=src python scripts/run_w27_ridge_keel_grounding.py",
            "inputs": {"source_modules": {
                "src/tensordem/ridge_keel_grounding.py": _sha256("src/tensordem/ridge_keel_grounding.py"),
                "src/tensordem/keel_foundation_dem.py": _sha256("src/tensordem/keel_foundation_dem.py")}},
            "seeds": {"dem": 7},
        },
        "limitations": [
            "The keel is idealised as a bonded lattice with frozen topology (small-strain "
            "regime, strain ~4e-3); bond breakage, contact creation and block rotation are "
            "not represented.",
            "Plane strain, periodic lateral boundaries, unit thickness; the specimen is a "
            "representative patch, not a full 3-D ridge.",
            "The Kerr reduction fixes k_u = k_l = 2K because a footing test cannot separate "
            "the two Winkler moduli (R depends on K and G2 only).",
            "The bonded-lattice bulk shear modulus (from the simple-shear test) is stiff "
            "relative to the footing edge term; the shear parameter used for the prediction "
            "is the one identified from the narrow footing, not the bulk shear test.",
            "ISO 19906:2019 A.8 values are WITHHELD: no clause text is available in this "
            "environment, so no standard number enters any computation.",
        ],
    }
    with open(args.out, "w") as fh:
        json.dump(artifact, fh, indent=1)
    print(json.dumps({"criteria": criteria, "wall_time_s": artifact["wall_time_s"]},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())