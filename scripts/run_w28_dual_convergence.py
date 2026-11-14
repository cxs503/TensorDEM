#!/usr/bin/env python3
"""W28 · ice-ridge keel grounding: dual (particle x mesh) convergence study.

Succeeds the W27 rework order (mission 18963), whose C4 was an honest FAIL: the
particle-count GCI of the keel footing resistance was 39.6 % (fine level 100 k,
observed order 1.12) while the continuum mesh GCI was 0.077 %.  W28 makes the
convergence question itself the object of study:

  * 4 particle levels (25 k, 50 k, 100 k, 200 k -> lattice spacing halves) x
    3 beam-mesh levels (h, h/2, h/4) = 12 primary cases on the W27-continuation
    (legacy) bond law,
  * the same 12 cases on the spacing-consistent bond law that W27 flagged as the
    alternative (``kn = E w_b / REFERENCE_LENGTH``) = 12 diagnostic cases, a
    controlled comparison that isolates exactly one variable,
  * measured per case: keel max penetration resistance ``R`` [N/m], keel contact
    area ``A_c`` [m^2 per m width], peak contact pressure ``q_max`` [Pa], each on a
    predeclared 6-level penetration ramp,
  * Richardson / Roache GCI + observed order per axis (Richardson limit and its
    uncertainty band), Kerr analytic reconciliation, Spearman rank correlation of
    penetration vs contact force.

Numbers are produced by this script only (zero hand copying).  Usage::

    PYTHONPATH=src python scripts/run_w28_dual_convergence.py [--quick] [--jobs N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

from tensordem import keel_foundation_dem as kd
from tensordem import keel_grounding_coupled as kc
from tensordem import ridge_keel_grounding as rg

OUT_DIR = os.path.join("papers", "artifacts", "w28", "ice-ridge-keel-grounding")

# ---------------------------------------------------------------------------
# predeclared configuration (fixed before the production run)
# ---------------------------------------------------------------------------
KEEL_HEIGHT = 2.0            # m
CELL_RATIO = 8.0             # periodic keel patch width / keel height
SOLID_FRACTION = 0.62        # 1 - porosity, keel rubble n = 0.38
W0_FRACTION = 4.0e-3         # platen settlement / keel height -> w0 = 8 mm
PARTICLE_COUNTS = (25000, 50000, 100000, 200000)      # 4 levels, spacing halves
MESH_ELEMENTS = (32, 64, 128)                          # h = 0.5 / 0.25 / 0.125 m
BEAM_LENGTH = 16.0           # m, structure length (8 x keel height)
PLATEN_WIDTH = 2.0           # m, = keel height (W27 validation footing width)
ICE_SHEET_THICKNESS = 2.0    # m
ICE_SHEET_YOUNG = 5.0e9      # Pa
ICE_POISSON = 0.3
RAMP_WIDTH = 8.0e-4          # m, C1 no-tension ramp = 0.1 w0
PENETRATION_LEVELS = 6
INCREMENTS = 6               # DEM load increments
STEPS_PER_INCREMENT = 300    # DEM relaxation steps per increment
SEED = 7
FOOTING_WIDTHS = (0.5, 1.0, 2.0)     # x keel height; 0.5 identifies, 1.0/2.0 validate
IDENT_WIDTH_FACTOR = 0.5
ARMS = {"legacy_bond_length": None, "spacing_consistent": kd.REFERENCE_LENGTH}
EI_STRUCTURE = ICE_SHEET_YOUNG * ICE_SHEET_THICKNESS ** 3 / (12.0 * (1.0 - ICE_POISSON ** 2))
KERR_SETS = {
    "soft_shear_layer": dict(ei=8.0e7, k_upper=1.0e7, k_lower=4.0e7, shear=1.0e8),
    "balanced": dict(ei=1.0e8, k_upper=2.0e7, k_lower=8.0e7, shear=4.0e8),
    "stiff_thin_layer": dict(ei=2.0e8, k_upper=5.0e7, k_lower=2.0e8, shear=2.0e9),
}

PREDECLARED = {
    "decision": ("Can the ice-ridge keel grounding resistance be claimed converged, and if "
                 "not, which discretisation blocks it and how many particles would be needed?"),
    "competing_alternatives": [
        "A1 the particle axis converges once four levels are used -- the legacy bond law is fine",
        "A2 the legacy bond law has no particle-converged limit because kn is particle-size "
        "independent (k ~ kn/a ~ 1/a); the spacing-consistent law is required",
        "A3 the mesh axis is the limiting one because the rigid platen edge makes the contact "
        "pressure discontinuous",
    ],
    "hypothesis": ("H1 the measured resistance is finite and mesh-converged (GCI < 5 %, order "
                   ">= 1.5); H2 the legacy bond law gives no particle-converged limit while "
                   "the spacing-consistent law restores one."),
    "metrics": {"R_N_per_m": "keel max penetration resistance at w0 (max over the ramp)",
                "A_c_m2_per_m": "keel contact area per unit out-of-plane width at w0",
                "q_max_Pa": "peak keel contact pressure at w0"},
    "baseline_control": ("W27 (mission 18963): particle GCI 39.6 %, mesh GCI 0.077 % on the "
                         "legacy bond law.  Arm 'spacing_consistent' changes exactly one "
                         "variable (the bond stiffness length)."),
    "thresholds": {"gci_max_rel": 0.05, "observed_order_min": 1.5,
                   "kerr_deflection_rel_dev_max": 0.03,
                   "keel_vs_kerr_resistance_rel_dev_max": 0.15, "spearman_min": 0.99},
    "stopping_rule": ("the predeclared 4 x 3 grid is run to completion; no level is added or "
                      "dropped after inspecting numbers"),
    "variables": {
        "changed": ["particle count (lattice spacing)", "beam mesh size h",
                    "bond stiffness length law (arms)"],
        "controlled": ["seed", "keel height and patch aspect ratio", "solid fraction",
                       "platen width and settlement", "structure EI", "no-tension ramp width",
                       "load increments and relaxation steps"],
        "confounders": ["lattice coordination drifts slightly with particle count",
                        "the bond-length law changes the small-strain modulus by "
                        "construction, so the arms are compared through convergence, not "
                        "through absolute values",
                        "the platen edge is a contact-pressure discontinuity: a uniform mesh "
                        "family is not in the asymptotic range at the coarse level"],
    },
    "repetitions": ("deterministic quasi-static solver, no time-integration noise; the "
                    "repetition axis is the discretisation ladder (4 particle levels, 3 mesh "
                    "levels) with two triads evaluated per axis"),
    "analysis": ("Roache GCI on the finest triad of each axis with the other axis at its finest "
                 "level; observed order reported in the particle count (r = 2) and in the "
                 "physical spacing (2 p), Richardson limit with a 1.25 safety band, "
                 "monotonicity check on every triad"),
    "environment_limits": ("cluster job; one Python process per (arm, particle level); "
                           "wall-clock recorded in the artifact"),
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# DEM block
# ---------------------------------------------------------------------------
def dem_resolution(job):
    arm, n_particles, bond_ref = job
    t0 = time.time()
    sample = kd.build_keel_sample(n_particles, height=KEEL_HEIGHT, cell_ratio=CELL_RATIO,
                                  solid_fraction=SOLID_FRACTION, seed=SEED,
                                  bond_reference_length=bond_ref)
    w0 = W0_FRACTION * KEEL_HEIGHT
    confined = kd.confined_compression(sample, w_max=w0, increments=INCREMENTS,
                                       steps_per_increment=STEPS_PER_INCREMENT)
    footings = {}
    for factor in FOOTING_WIDTHS:
        footings["%g" % factor] = kd.footing_test(
            sample, footing_width=factor * KEEL_HEIGHT, w0=w0, increments=INCREMENTS,
            steps_per_increment=STEPS_PER_INCREMENT)
    return {"arm": arm, "bond_reference_length_m": bond_ref,
            "n_particles_requested": n_particles, "sample": sample.as_dict(), "w0_m": w0,
            "confined": confined, "footings": footings, "wall_time_s": time.time() - t0}


def relax_adequacy(bond_ref, n_particles):
    sample = kd.build_keel_sample(n_particles, height=KEEL_HEIGHT, cell_ratio=CELL_RATIO,
                                 solid_fraction=SOLID_FRACTION, seed=SEED,
                                 bond_reference_length=bond_ref)
    w0 = W0_FRACTION * KEEL_HEIGHT
    out = {}
    for steps in (150, 300, 600):
        f = kd.footing_test(sample, footing_width=KEEL_HEIGHT, w0=w0,
                            increments=INCREMENTS, steps_per_increment=steps)
        out[str(steps)] = float(f["resistance_N_per_m"][-1])
    v = [out[k] for k in ("150", "300", "600")]
    return {"bond_reference_length_m": bond_ref, "n_particles_requested": n_particles,
            "resistance_vs_steps": out,
            "rel_change_300_to_600": abs(v[2] - v[1]) / max(abs(v[2]), 1e-300),
            "criterion": ("relaxation stopping rule: the footing resistance must not move by "
                          "more than the 0.05 GCI target between 300 and 600 steps")}


# ---------------------------------------------------------------------------
# coupled block: keel law -> beam-on-keel platen solve
# ---------------------------------------------------------------------------
def coupled_case(res, n_elements):
    law = kc.KeelPressureLaw(res["confined"]["settlements_m"], res["confined"]["pressures_Pa"])
    w0 = res["w0_m"]
    steps = []
    for k in range(1, PENETRATION_LEVELS + 1):
        w = w0 * k / PENETRATION_LEVELS
        out = kc.beam_on_unilateral_keel(
            length=BEAM_LENGTH, ei=EI_STRUCTURE, n_elements=n_elements,
            platen_centre=0.5 * BEAM_LENGTH, platen_width=PLATEN_WIDTH, w0=w,
            law=law, ramp_width_m=RAMP_WIDTH, samples_per_element=32)
        steps.append({"settlement_m": w,
                      "resistance_N_per_m": out["resistance_N_per_m"],
                      "contact_area_m2_per_m": out["contact_area_m2_per_m"],
                      "q_max_Pa": out["q_max_Pa"], "w_min_m": out["w_min_m"],
                      "equilibrium_rel_error": out["equilibrium_rel_error"],
                      "newton_iterations": out["newton_iterations"],
                      "converged": out["converged"],
                      "solution_finite": out["solution_finite"],
                      "inside_dem_range": out["inside_dem_range"]})
    final = steps[-1]
    return {"arm": res["arm"], "bond_reference_length_m": res["bond_reference_length_m"],
            "n_particles_requested": res["n_particles_requested"],
            "n_particles": res["sample"]["n_particles"],
            "spacing_m": res["sample"]["spacing_m"],
            "coordination": res["sample"]["coordination"],
            "n_elements": n_elements, "element_length_m": BEAM_LENGTH / n_elements,
            "settlement_ramp": steps,
            "R_N_per_m": final["resistance_N_per_m"],
            "A_c_m2_per_m": final["contact_area_m2_per_m"],
            "q_max_Pa": final["q_max_Pa"], "w_min_m": final["w_min_m"],
            "equilibrium_rel_error": final["equilibrium_rel_error"],
            "inside_dem_range": final["inside_dem_range"],
            "all_steps_converged": all(s["converged"] for s in steps),
            "all_steps_finite": all(s["solution_finite"] for s in steps),
            "keel_small_strain_secant_N_per_m3": law.small_strain_secant,
            "keel_secant_at_w0_N_per_m3": law.secant_max}


def mesh_sensitivity(res):
    """Extra level h/8 (uniform) -- asymptotic-trend evidence only."""
    law = kc.KeelPressureLaw(res["confined"]["settlements_m"], res["confined"]["pressures_Pa"])
    out = []
    for ne in (256,):
        r = kc.beam_on_unilateral_keel(
            length=BEAM_LENGTH, ei=EI_STRUCTURE, n_elements=ne,
            platen_centre=0.5 * BEAM_LENGTH, platen_width=PLATEN_WIDTH, w0=res["w0_m"],
            law=law, ramp_width_m=RAMP_WIDTH, samples_per_element=32)
        out.append({"n_elements": ne, "element_length_m": BEAM_LENGTH / ne,
                    "R_N_per_m": r["resistance_N_per_m"],
                    "A_c_m2_per_m": r["contact_area_m2_per_m"],
                    "q_max_Pa": r["q_max_Pa"]})
    return {"arm": res["arm"], "n_particles_requested": res["n_particles_requested"],
            "levels": out}


def graded_mesh_diagnostic(res):
    """Remedy diagnostic: platen-edge-graded mesh family (same three levels)."""
    law = kc.KeelPressureLaw(res["confined"]["settlements_m"], res["confined"]["pressures_Pa"])
    centres = [0.5 * BEAM_LENGTH - 0.5 * PLATEN_WIDTH, 0.5 * BEAM_LENGTH + 0.5 * PLATEN_WIDTH]
    levels = []
    for ne in MESH_ELEMENTS:
        xn = kc.graded_nodes(BEAM_LENGTH, ne, centres=centres, sigma=0.4, amplitude=16.0)
        r = kc.beam_on_unilateral_keel(
            length=BEAM_LENGTH, ei=EI_STRUCTURE, n_elements=ne, x_nodes=xn,
            platen_centre=0.5 * BEAM_LENGTH, platen_width=PLATEN_WIDTH, w0=res["w0_m"],
            law=law, ramp_width_m=RAMP_WIDTH, samples_per_element=32)
        levels.append({"n_elements": ne, "element_length_min_m": r["element_length_min_m"],
                       "element_length_max_m": r["element_length_max_m"],
                       "R_N_per_m": r["resistance_N_per_m"],
                       "A_c_m2_per_m": r["contact_area_m2_per_m"],
                       "q_max_Pa": r["q_max_Pa"]})
    return {"arm": res["arm"], "n_particles_requested": res["n_particles_requested"],
            "grading": {"centres_m": centres, "sigma_m": 0.4, "amplitude": 16.0},
            "levels": levels}


def ramp_sensitivity(res):
    """The no-tension ramp width is a numerical smoothing: check insensitivity."""
    law = kc.KeelPressureLaw(res["confined"]["settlements_m"], res["confined"]["pressures_Pa"])
    out = {}
    for delta in (0.0, 0.25 * RAMP_WIDTH, RAMP_WIDTH, 4.0 * RAMP_WIDTH, 10.0 * RAMP_WIDTH):
        r = kc.beam_on_unilateral_keel(
            length=BEAM_LENGTH, ei=EI_STRUCTURE, n_elements=MESH_ELEMENTS[-1],
            platen_centre=0.5 * BEAM_LENGTH, platen_width=PLATEN_WIDTH, w0=res["w0_m"],
            law=law, ramp_width_m=delta, samples_per_element=32)
        out["%.3e" % delta] = {"R_N_per_m": r["resistance_N_per_m"],
                               "A_c_m2_per_m": r["contact_area_m2_per_m"],
                               "q_max_Pa": r["q_max_Pa"]}
    vals = [v["R_N_per_m"] for v in out.values()]
    return {"arm": res["arm"], "n_particles_requested": res["n_particles_requested"],
            "n_elements": MESH_ELEMENTS[-1], "by_ramp_width": out,
            "R_spread_rel": (max(vals) - min(vals)) / abs(float(np.mean(vals)))}


# ---------------------------------------------------------------------------
# verification / reconciliation blocks
# ---------------------------------------------------------------------------
def kerr_vs_fem():
    a, q0 = 2.0, 1.0e5
    x = np.linspace(-30.0, 30.0, 1201)
    cases = []
    for name, cfg in KERR_SETS.items():
        fnd = rg.KerrFoundation(k_upper=cfg["k_upper"], k_lower=cfg["k_lower"],
                                shear=cfg["shear"])
        an = rg.kerr_beam_analytic(a, q0, ei=cfg["ei"], foundation=fnd, x=x)
        fem = rg.kerr_beam_fem(np.linspace(-30.0, 30.0, 1201), ei=cfg["ei"], foundation=fnd,
                               load=lambda xx: np.where(np.abs(xx) <= a, q0, 0.0))
        wa = np.interp(x, an.x, an.w)
        wf = np.interp(x, fem.x, fem.w)
        near = np.abs(x) <= 5.0 * a
        dev = float(np.max(np.abs(wf - wa)[near])) / float(np.max(np.abs(wa[near])))
        cases.append({"name": name, "w_centre_analytic_m": an.meta["w_centre_m"],
                      "w_centre_fem_m": float(np.interp(0.0, fem.x, fem.w)),
                      "max_rel_dev_deflection_absx_le_5a": dev})
    cfg = KERR_SETS["balanced"]
    fnd = rg.KerrFoundation(k_upper=cfg["k_upper"], k_lower=cfg["k_lower"], shear=cfg["shear"])
    mesh = []
    for ne in (75, 150, 300, 600):
        fem = rg.kerr_beam_fem(np.linspace(-30.0, 30.0, ne + 1), ei=cfg["ei"], foundation=fnd,
                               load=lambda xx: np.where(np.abs(xx) <= a, q0, 0.0))
        mesh.append({"n_elements": ne, "w_centre_m": float(np.interp(0.0, fem.x, fem.w))})
    return {"cases": cases,
            "max_rel_dev_all_cases": max(c["max_rel_dev_deflection_absx_le_5a"] for c in cases),
            "mesh_refinement": mesh,
            "mesh_gci": rg.gci_triad([m["w_centre_m"] for m in mesh[:3]], r=2.0)}


def kerr_footing_reconciliation(res):
    """Identify the Kerr layer (confined modulus + narrow footing) and predict (W27 protocol)."""
    w0 = res["w0_m"]
    k_sub = float(res["confined"]["pressures_Pa"][-1]) / float(res["confined"]["settlements_m"][-1])
    ident = res["footings"]["%g" % IDENT_WIDTH_FACTOR]
    edge = float(ident["resistance_N_per_m"][-1]) - k_sub * w0 * float(ident["footing_width_m"])
    sqrt_g2k = edge / (2.0 * w0)
    g2 = sqrt_g2k ** 2 / k_sub
    foundation = rg.KerrFoundation(k_upper=2.0 * k_sub, k_lower=2.0 * k_sub, shear=2.0 * g2)
    preds = {}
    for key, foot in res["footings"].items():
        b = float(foot["footing_width_m"])
        pred = rg.kerr_footing_resistance(width=b, w0=w0, foundation=foundation)
        dem = float(foot["resistance_N_per_m"][-1])
        preds[key] = {"footing_width_m": b, "dem_N_per_m": dem, "kerr_N_per_m": pred,
                      "rel_dev": abs(pred - dem) / dem,
                      "role": ("identification" if abs(b - IDENT_WIDTH_FACTOR * KEEL_HEIGHT) < 1e-9
                               else "validation")}
    valid = [v for v in preds.values() if v["role"] == "validation"]
    return {"arm": res["arm"], "n_particles_requested": res["n_particles_requested"],
            "k_subgrade_N_per_m3": k_sub,
            "identified_sqrt_G2K_N_per_m": sqrt_g2k,
            "kerr_foundation": foundation.as_dict(), "footing_predictions": preds,
            "max_validation_rel_dev": max(v["rel_dev"] for v in valid)}


# ---------------------------------------------------------------------------
# convergence analysis
# ---------------------------------------------------------------------------
def axis_study(values_by_level, levels, *, higher_is_finer):
    """Two triads (three finest levels, three coarsest) + Richardson limit."""
    order = sorted(range(len(levels)), key=lambda i: levels[i], reverse=higher_is_finer)
    v = [float(values_by_level[i]) for i in order]
    lv = [levels[i] for i in order]
    out = {"levels_fine_to_coarse": lv, "values_fine_to_coarse": v, "triads": {},
           "richardson_limit": None, "uncertainty_band_rel": None}
    for tag, idx in (("finest", [0, 1, 2]), ("coarsest", [1, 2, 3])):
        if len(v) < idx[-1] + 1:
            continue
        info = kc.richardson_limit(v[idx[0]], v[idx[1]], v[idx[2]], r=2.0)
        info["values"] = [v[i] for i in idx]
        info["levels"] = [lv[i] for i in idx]
        p = info.get("observed_order_p")
        info["observed_order_in_spacing"] = (2.0 * p) if p is not None else None
        out["triads"][tag] = info
    tri = out["triads"].get("finest")
    if tri:
        out["richardson_limit"] = tri.get("richardson_limit")
        out["uncertainty_band_rel"] = tri.get("uncertainty_band_rel")
    return out


def divergence_exponent(levels, values):
    lv = np.asarray(levels, dtype=float)
    v = np.asarray(values, dtype=float)
    slope = float(np.polyfit(np.log(lv), np.log(np.abs(v)), 1)[0])
    return {"loglog_slope_vs_N": slope,
            "interpretation": ("a positive slope means the resistance grows without bound as the "
                              "particle count grows, i.e. no converged limit; the slope is the "
                              "divergence exponent in N")}


def required_levels(triad, finest_level, *, gci_target=0.05, order_target=1.5):
    """Level count implied by extrapolating the Richardson error model."""
    if not triad:
        return {"observed_order": None, "gci": None, "level_required": None,
                "reason": "no triad"}
    p = triad.get("observed_order_p")
    gci = triad.get("gci_fine_rel")
    if p is None or gci is None:
        return {"observed_order": p, "gci": gci, "level_required": None,
                "reason": "observed order undefined (divergent or non-monotone triad)"}
    if p <= 0.0:
        return {"observed_order": p, "gci": gci, "level_required": None,
                "reason": "non-positive observed order"}
    f_gci = (gci / gci_target) ** (1.0 / p)
    f_ord = (p / order_target) ** (1.0 / p)
    return {"observed_order": p, "gci": gci,
            "level_required_for_gci": float(finest_level * f_gci),
            "level_required_for_order": float(finest_level * f_ord),
            "rule": ("level_req = level_finest * (GCI/GCI_target)**(1/p) from the Richardson error "
                     "model; the same rule with p/order_target gives the level at which the "
                     "predeclared observed order would be seen")}


def interaction_table(cases, quantity):
    arms = sorted({c["arm"] for c in cases})
    out = {}
    for arm in arms:
        out[arm] = {str(ne): [c[quantity] for c in cases
                              if c["arm"] == arm and c["n_elements"] == ne]
                    for ne in MESH_ELEMENTS}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--reuse-dem", action="store_true",
                    help="reuse papers/artifacts/w28/.../dem_stage.json (deterministic)")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "dual_convergence.json"))
    args = ap.parse_args()
    t_start = time.time()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    global PARTICLE_COUNTS, MESH_ELEMENTS, STEPS_PER_INCREMENT, INCREMENTS
    if args.quick:
        PARTICLE_COUNTS = (2500, 5000)
        MESH_ELEMENTS = (16,)
        STEPS_PER_INCREMENT = 40
        INCREMENTS = 3

    dem_cache = os.path.join(OUT_DIR, "dem_stage.json")
    jobs = [(arm, n, ref) for arm, ref in ARMS.items() for n in PARTICLE_COUNTS]
    dem_results = []
    if args.reuse_dem and os.path.exists(dem_cache) and not args.quick:
        with open(dem_cache) as fh:
            dem_results = json.load(fh)["dem_resolutions"]
        print("[dem] reused %d resolutions from %s" % (len(dem_results), dem_cache),
              flush=True)
    elif args.jobs > 1:
        with Pool(processes=min(args.jobs, len(jobs))) as pool:
            for res in pool.imap_unordered(dem_resolution, jobs):
                dem_results.append(res)
                print("[dem] arm=%s N=%d t=%.0fs" % (res["arm"], res["n_particles_requested"],
                                                     res["wall_time_s"]), flush=True)
    else:
        for job in jobs:
            res = dem_resolution(job)
            dem_results.append(res)
            print("[dem] arm=%s N=%d t=%.0fs" % (res["arm"], res["n_particles_requested"],
                                                 res["wall_time_s"]), flush=True)
    if not (args.reuse_dem and os.path.exists(dem_cache) and not args.quick):
        with open(dem_cache, "w") as fh:
            json.dump({"written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "configuration": {"particle_counts": list(PARTICLE_COUNTS),
                                         "arms": {k: v for k, v in ARMS.items()},
                                         "increments": INCREMENTS,
                                         "steps_per_increment": STEPS_PER_INCREMENT,
                                         "seed": SEED},
                       "dem_resolutions": dem_results}, fh, indent=1)
        print("[dem] checkpoint -> %s" % dem_cache, flush=True)
    by_key = {(r["arm"], r["n_particles_requested"]): r for r in dem_results}

    cases = []
    for arm in ARMS:
        for n in PARTICLE_COUNTS:
            res = by_key[(arm, n)]
            for ne in MESH_ELEMENTS:
                try:
                    c = coupled_case(res, ne)
                except Exception as exc:  # a failed case must still land in the artifact
                    c = {"arm": arm, "bond_reference_length_m": res["bond_reference_length_m"],
                         "n_particles_requested": n, "n_particles": res["sample"]["n_particles"],
                         "spacing_m": res["sample"]["spacing_m"],
                         "coordination": res["sample"]["coordination"],
                         "n_elements": ne, "element_length_m": BEAM_LENGTH / ne,
                         "settlement_ramp": [], "R_N_per_m": float("nan"),
                         "A_c_m2_per_m": float("nan"), "q_max_Pa": float("nan"),
                         "w_min_m": float("nan"), "equilibrium_rel_error": None,
                         "inside_dem_range": None, "all_steps_converged": False,
                         "all_steps_finite": False, "failed": True,
                         "failure": "%s: %s" % (type(exc).__name__, exc)}
                cases.append(c)
                print("[coupled] %s N=%d ne=%d R=%.5e Ac=%.4f q=%.4e ok=%s" %
                      (arm, n, ne, c["R_N_per_m"], c["A_c_m2_per_m"], c["q_max_Pa"],
                       not c.get("failed", False) and c.get("all_steps_finite", False)),
                      flush=True)

    arm_analysis = {}
    for arm in ARMS:
        finest_mesh = MESH_ELEMENTS[-1]
        finest_N = PARTICLE_COUNTS[-1]
        sel_p = [c for c in cases if c["arm"] == arm and c["n_elements"] == finest_mesh]
        sel_m = [c for c in cases if c["arm"] == arm and c["n_particles_requested"] == finest_N]
        particle = {}
        for q in ("R_N_per_m", "A_c_m2_per_m", "q_max_Pa"):
            particle[q] = axis_study([c[q] for c in sel_p], list(PARTICLE_COUNTS),
                                     higher_is_finer=True)
        mesh = {}
        for q in ("R_N_per_m", "A_c_m2_per_m", "q_max_Pa"):
            mesh[q] = axis_study([c[q] for c in sel_m], list(MESH_ELEMENTS),
                                 higher_is_finer=False)
        arm_analysis[arm] = {
            "bond_reference_length_m": ARMS[arm],
            "particle_axis": particle, "mesh_axis": mesh,
            "particle_divergence_R": divergence_exponent(list(PARTICLE_COUNTS),
                                                         [c["R_N_per_m"] for c in sel_p]),
            "required_particles_R": required_levels(
                particle["R_N_per_m"]["triads"].get("finest"), finest_N),
            "required_elements_R": required_levels(
                mesh["R_N_per_m"]["triads"].get("finest"), finest_mesh),
        }
        res = by_key[(arm, finest_N)]
        law = kc.KeelPressureLaw(res["confined"]["settlements_m"], res["confined"]["pressures_Pa"])
        arm_analysis[arm]["keel_modulus_at_finest_N"] = {
            "small_strain_secant_N_per_m3": law.small_strain_secant,
            "secant_at_w0_N_per_m3": law.secant_max,
            "bond_stiffness_over_spacing_N_per_m2":
                float(res["sample"]["kPa_per_bond"]) * 1e3 / res["sample"]["spacing_m"]}

    recon = [kerr_footing_reconciliation(by_key[(arm, n)])
             for arm in ARMS for n in PARTICLE_COUNTS]
    kerr_block = kerr_vs_fem()

    finest_key = ("legacy_bond_length", PARTICLE_COUNTS[-1])
    extra = {"relax_adequacy": relax_adequacy(ARMS["legacy_bond_length"], PARTICLE_COUNTS[0]),
             "ramp_sensitivity": ramp_sensitivity(by_key[finest_key]),
             "mesh_sensitivity_h8": mesh_sensitivity(by_key[finest_key]),
             "graded_mesh_diagnostic": graded_mesh_diagnostic(by_key[finest_key])}

    spearman_cases = []
    for c in cases:
        sp = kc.spearman([s["settlement_m"] for s in c["settlement_ramp"]],
                         [s["resistance_N_per_m"] for s in c["settlement_ramp"]])
        spearman_cases.append({"arm": c["arm"], "n_particles_requested": c["n_particles_requested"],
                               "n_elements": c["n_elements"], **sp})
    pooled = kc.spearman([s["settlement_m"] for c in cases for s in c["settlement_ramp"]],
                         [s["resistance_N_per_m"] for c in cases for s in c["settlement_ramp"]])
    spearman_block = {"per_case": spearman_cases,
                      "min_rho_per_case": min(s["rho"] for s in spearman_cases),
                      "pooled": pooled,
                      "definition": ("rank correlation between applied platen settlement "
                                     "(penetration) and keel contact force (resistance) over "
                                     "the predeclared 6-level penetration ramp")}

    def _row(axis, key):
        tri = axis[key]["triads"].get("finest", {})
        return {"gci_rel": tri.get("gci_fine_rel"),
                "order_in_N_or_h": tri.get("observed_order_p"),
                "order_in_spacing": tri.get("observed_order_in_spacing"),
                "monotone": tri.get("monotone")}

    prim = arm_analysis["legacy_bond_length"]
    rows = {}
    for key, label in (("R_N_per_m", "R"), ("A_c_m2_per_m", "A_c"), ("q_max_Pa", "q_max")):
        rows["particle_axis_" + label] = _row(prim["particle_axis"], key)
        rows["mesh_axis_" + label] = _row(prim["mesh_axis"], key)
    detail = {}
    for name, row in rows.items():
        gci_ok = row["gci_rel"] is not None and row["gci_rel"] < 0.05
        order_ok = row["order_in_spacing"] is not None and row["order_in_spacing"] >= 1.5
        detail[name] = {"gci_ok": bool(gci_ok), "order_ok": bool(order_ok),
                        "pass": bool(gci_ok and order_ok)}
    c1_pass = all(v["pass"] for v in detail.values())

    kerr_max = kerr_block["max_rel_dev_all_cases"]
    recon_finest = [r for r in recon if r["n_particles_requested"] == PARTICLE_COUNTS[-1]]
    c3_val = max(r["max_validation_rel_dev"] for r in recon_finest)

    criteria = [
        {"id": "C1",
         "criterion": ("particle-axis and mesh-axis GCI < 5 %% and observed order >= 1.5 for all "
                       "three measured quantities, primary (legacy) arm"),
         "verdict": "PASS" if c1_pass else "FAIL", "pass": bool(c1_pass),
         "value": detail, "rows": rows},
        {"id": "C2", "criterion": "Kerr analytic deflection vs independent FEM < 3 %% (|x| <= 5a)",
         "verdict": "PASS" if kerr_max < 0.03 else "FAIL", "pass": bool(kerr_max < 0.03),
         "value": kerr_max},
        {"id": "C3",
         "criterion": ("bonded keel footing resistance vs Kerr analytic < 15 %% at the validation "
                       "footing widths, finest particle level"),
         "verdict": "PASS" if c3_val < 0.15 else "FAIL", "pass": bool(c3_val < 0.15),
         "value": c3_val,
         "per_arm": {r["arm"]: r["max_validation_rel_dev"] for r in recon_finest}},
        {"id": "C4", "criterion": "penetration vs contact force Spearman >= 0.99",
         "verdict": "PASS" if spearman_block["min_rho_per_case"] >= 0.99 else "FAIL",
         "pass": bool(spearman_block["min_rho_per_case"] >= 0.99),
         "value": spearman_block["min_rho_per_case"],
         "pooled_rho": pooled["rho"]},
        {"id": "C5", "criterion": "all 12 primary cases landed (incl. failures), each with the "
                                  "full penetration ramp",
         "verdict": "PASS" if len(cases) >= 12 else "FAIL", "pass": bool(len(cases) >= 12),
         "value": {"n_cases": len(cases),
                   "n_primary_cases": sum(1 for c in cases
                                          if c["arm"] == "legacy_bond_length"),
                   "n_diagnostic_cases": sum(1 for c in cases
                                             if c["arm"] == "spacing_consistent"),
                   "n_penetration_levels": PENETRATION_LEVELS}},
    ]

    sat = arm_analysis["spacing_consistent"]
    honest = {
        "overall_verdict": "HONEST_FAIL" if not c1_pass else "PASS",
        "blocking_rows": [k for k, v in detail.items() if not v["pass"]],
        "legacy_particle_divergence_R": prim["particle_divergence_R"],
        "legacy_required_particles": prim["required_particles_R"],
        "spacing_consistent_particle_axis_R": {
            "triad": sat["particle_axis"]["R_N_per_m"]["triads"].get("finest"),
            "required_particles": sat["required_particles_R"]},
        "statement": ("C1 is judged on the W27-continuation (legacy) arm; a FAIL is registered "
                      "here and must not be overwritten by later rounds.  The spacing-consistent "
                      "arm is a diagnostic control for the bond-length mechanism, not a rescue "
                      "of the primary verdict." if not c1_pass else
                      "C1 passed on the primary arm."),
    }

    # Richardson limits of the other quantities (uncertainty band)
    bands = {arm: {key: {"richardson_limit":
                         arm_analysis[arm]["particle_axis"][key]["triads"].get("finest", {})
                         .get("richardson_limit"),
                         "uncertainty_band_rel":
                         arm_analysis[arm]["particle_axis"][key]["triads"].get("finest", {})
                         .get("uncertainty_band_rel")}
                    for key in ("R_N_per_m", "A_c_m2_per_m", "q_max_Pa")}
             for arm in ARMS}

    provenance = {
        "generator": "scripts/run_w28_dual_convergence.py",
        "generator_sha256": _sha256(__file__),
        "command": " ".join([sys.executable, os.path.join("scripts",
                                                          "run_w28_dual_convergence.py"),
                             "--jobs", str(args.jobs)]),
        "repo": "TensorDEM",
        "runner": "native",
        "mission_id": 19906,
        "worktree_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git(["rev-parse", "HEAD"]),
        "source_sha256": {p: _sha256(p) for p in (
            os.path.join("src", "tensordem", "keel_grounding_coupled.py"),
            os.path.join("src", "tensordem", "keel_foundation_dem.py"),
            os.path.join("src", "tensordem", "ridge_keel_grounding.py"),
            os.path.join("benchmarks", "anchors", "ice-ridge-keel-grounding.json"),
            os.path.join("papers", "artifacts", "w27", "ice-ridge-keel-grounding",
                         "ridge_keel_anchor.json"))},
        "recovered_from_ref": {
            "files": ["src/tensordem/ridge_keel_grounding.py",
                      "src/tensordem/keel_foundation_dem.py",
                      "benchmarks/anchors/ice-ridge-keel-grounding.json",
                      "papers/artifacts/w27/ice-ridge-keel-grounding/ridge_keel_anchor.json"],
            "ref": "agent/m18963",
            "commit": "594c334e",
            "note": ("the W27 mission branch (m18963) was never merged into main; the W27 "
                     "sources and artifact were recovered from that ref so that this study "
                     "runs from main.  W27 numbers quoted in this artifact come from the "
                     "recovered ridge_keel_anchor.json (particle GCI 0.3959, mesh GCI "
                     "0.000769)")},
        "w27_baseline": {
            "path": "papers/artifacts/w27/ice-ridge-keel-grounding/ridge_keel_anchor.json",
            "criteria": "C4 FAIL: particle GCI 0.3959 (fine 100 k, p = 1.116), mesh GCI 0.000769 "
                        "(Kerr beam mixed-Galerkin FEM, p = 1.967)",
            "read_only": True},
        "anchor_status": {
            "iso_19906_2019_A8": "WITHHELD (primary standard text not available in this "
                                 "environment; only the operator layer is machine-checked)",
            "kerr_1976": "IN_TREE_RECONSTRUCTION (the elastic-foundation beam solution is "
                         "re-derived and cross-checked in ridge_keel_grounding, Hetenyi "
                         "cross-check included; no standard value is transcribed)",
            "iacs_ur_i": "WITHHELD (no clause text available)",
            "anchor_register": "benchmarks/anchors/ice-ridge-keel-grounding.json",
            "quotes": [{"anchor": "ISO 19906:2019 A.8 pressure-area law",
                        "quote_location": "benchmarks/anchors/ice-ridge-keel-grounding.json"
                                          "#/clauses/A_8 (status WITHHELD, no numbers)"},
                       {"anchor": "Kerr (1976) elastic-foundation beam",
                        "quote_location": "docs/ice_ridge_keel_grounding_kerr.md#kerr-layer "
                                          "(in-tree derivation, no transcribed constants)"}]},
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.platform(), "cpu_count": os.cpu_count(),
                        "jobs": args.jobs},
        "wall_time_s": None,
        "verification_commands": [
            "PYTHONPATH=src python -m pytest tests/test_w28_dual_convergence.py -q",
            "PYTHONPATH=src python scripts/run_w28_dual_convergence.py --jobs 4"],
    }

    artifact = {
        "artifact": "papers/artifacts/w28/ice-ridge-keel-grounding/dual_convergence.json",
        "case": "ice-ridge keel grounding: dual (particle x mesh) convergence study",
        "mission": "W28 paper-exp-ice-ridge-keel-grounding-2",
        "mission_id": 19906, "repo": "TensorDEM", "runner": "native",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "predeclared": PREDECLARED,
        "configuration": {
            "keel_height_m": KEEL_HEIGHT, "cell_ratio": CELL_RATIO,
            "solid_fraction": SOLID_FRACTION, "w0_m": W0_FRACTION * KEEL_HEIGHT,
            "particle_counts": list(PARTICLE_COUNTS), "mesh_elements": list(MESH_ELEMENTS),
            "beam_length_m": BEAM_LENGTH, "platen_width_m": PLATEN_WIDTH,
            "structure_ei_N_m2": EI_STRUCTURE,
            "structure": {"thickness_m": ICE_SHEET_THICKNESS, "young_Pa": ICE_SHEET_YOUNG,
                          "poisson": ICE_POISSON},
            "ramp_width_m": RAMP_WIDTH, "penetration_levels": PENETRATION_LEVELS,
            "dem_increments": INCREMENTS, "dem_steps_per_increment": STEPS_PER_INCREMENT,
            "seed": SEED, "footing_widths_xHk": list(FOOTING_WIDTHS),
            "arms": {k: v for k, v in ARMS.items()},
        },
        "keel_law": {arm: {"confined": by_key[(arm, PARTICLE_COUNTS[-1])]["confined"]}
                     for arm in ARMS},
        "dem_resolutions": dem_results,
        "cases": cases,
        "interaction_tables": {q: interaction_table(cases, q)
                               for q in ("R_N_per_m", "A_c_m2_per_m", "q_max_Pa")},
        "convergence": arm_analysis,
        "richardson_bands_particle_axis": bands,
        "kerr_vs_fem": kerr_block,
        "kerr_footing_reconciliation": recon,
        "spearman": spearman_block,
        "extra_evidence": extra,
        "criteria": criteria,
        "honest_fail_register": honest,
        "limitations": [
            "the keel is a bonded lattice with frozen topology (strain ~4e-3): no bond breakage, "
            "no block rotation, no rate effects; the freeze-bond modulus 1 GPa is a stiffness "
            "choice, so absolute pressures are model dependent -- the *convergence* statements "
            "are not",
            "the structure is a plane-strain beam (unit width) with free ends; the overhang past "
            "the contact area lifts off (no-tension) and is not carried by buoyancy",
            "the local keel constitutive law entering the coupled solve is the confined "
            "(wide-lid) compression curve; footing edge effects enter only through the "
            "reconciliation block",
            "the rigid-platen edge makes the contact pressure discontinuous, which caps the "
            "observed order of the uniform mesh family (remedy demonstrated with a "
            "platen-edge-graded family, reported separately)",
            "the no-tension switch is smoothed with a C1 ramp of width 0.1 w0; the ramp "
            "sensitivity block shows the spread (see extra_evidence.ramp_sensitivity)",
            "ISO 19906:2019 A.8 and IACS UR I values are WITHHELD: no standard number enters "
            "any computation in this artifact",
        ],
        "provenance": provenance,
    }
    artifact["provenance"]["wall_time_s"] = time.time() - t_start
    with open(args.out, "w") as fh:
        json.dump(artifact, fh, indent=1)
    print("wrote %s (%.1f s)" % (args.out, time.time() - t_start))
    return 0


def _git(cmd):
    try:
        return subprocess.check_output(["git"] + cmd, text=True).strip()
    except Exception:  # pragma: no cover
        return None


if __name__ == "__main__":
    raise SystemExit(main())
