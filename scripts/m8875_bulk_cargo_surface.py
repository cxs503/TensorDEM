#!/usr/bin/env python
"""LW3-N06 : TensorDEM bulk-cargo hold granular-dynamics verification surface.

Runs the case matrix of :mod:`tensordem.bulk_cargo` (poured-heap repose angle,
Beverloo discharge-rate curve, Janssen wall-pressure profile + two negative
controls), stores every raw case result, evaluates the acceptance criteria
(C1-C5 of the campaign node) and writes

    artifacts/bulk_cargo/case.json      -- the exact case matrix + numerics
    artifacts/bulk_cargo/results.json   -- measurements, criteria verdicts

Usage
-----
    python scripts/m8875_bulk_cargo_surface.py --jobs 8 --out artifacts/bulk_cargo
    python scripts/m8875_bulk_cargo_surface.py --only R1,R4 --jobs 2   # diagnostics

The heavy work is embarrassingly parallel over cases; run it on the cluster with
``--jobs`` = requested cores.  Nothing here fabricates a number: every reported
quantity is copied from a case result that the DEM actually produced.
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
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.bulk_cargo import (  # noqa: E402
    BEVERLOO_C,
    BEVERLOO_K,
    REPOSE_REFERENCE_BAND_DEG,
    REPOSE_REFERENCE_DEG,
    CargoHoldCase,
    PourCase,
    beverloo_prediction,
    fit_power_law,
    numerical_checks,
    simulate_hold_discharge,
    simulate_hold_static,
    simulate_pour,
)

# --------------------------------------------------------------------------- #
# case matrix
# --------------------------------------------------------------------------- #
# One cargo parameter set for the discharge/storage anchors: near-spherical
# rounded cargo (mu = 0.5 as specified by the node, rolling resistance 0.05 =
# rounded grains; the angular/ore class is probed by the repose sweep R2/R3).
HOLD_BASE = dict(
    radius=0.005, density=2500.0, mu=0.5, mu_rolling=0.05, restitution=0.3,
    kn=1.0e4, kt_ratio=0.5, gravity=9.81,
    length=0.14, breadth=0.12, fill_height=0.12,
    wall_mu=0.4, floor_mu=0.5,
    packing_gap=0.03, jitter=0.12, seed=1,
    settle_time=0.30,
    discharge_time=0.80, n_windows=4,
)

POUR_BASE = dict(
    radius=0.005, density=2500.0, mu=0.5, mu_rolling=0.05, restitution=0.3,
    kn=1.0e4, kt_ratio=0.5, gravity=9.81,
    funnel_side=0.080, funnel_bottom=0.090, column_height=0.240,
    wall_mu=0.0, floor_mu=0.5, packing_gap=0.03, jitter=0.10, seed=7,
    settle_time=0.60, max_time=6.0,
)

CASES = [
    # --- poured heap: repose-angle emergence (C1) + negative control (C4a)
    {"id": "R1_beads", "kind": "pour", "role": "primary",
     "params": dict(POUR_BASE, mu_rolling=0.05)},
    {"id": "R2_ore", "kind": "pour", "role": "material-class",
     "params": dict(POUR_BASE, mu_rolling=0.20)},
    {"id": "R3_angular", "kind": "pour", "role": "trend",
     "params": dict(POUR_BASE, mu_rolling=0.35)},
    {"id": "R4_frictionless", "kind": "pour", "role": "negative-control",
     "params": dict(POUR_BASE, mu=0.0, mu_rolling=0.0, seed=11)},
    # --- Beverloo discharge curve (C2) + aperture negative control (C4b)
    {"id": "W_D040", "kind": "discharge", "role": "beverloo",
     "params": dict(HOLD_BASE, outlet_diameter=0.040)},
    {"id": "W_D048", "kind": "discharge", "role": "beverloo",
     "params": dict(HOLD_BASE, outlet_diameter=0.048)},
    {"id": "W_D056", "kind": "discharge", "role": "beverloo",
     "params": dict(HOLD_BASE, outlet_diameter=0.056)},
    {"id": "W_D064", "kind": "discharge", "role": "beverloo",
     "params": dict(HOLD_BASE, outlet_diameter=0.064, discharge_time=0.60,
                    n_windows=3)},
    {"id": "W_D008_blocked", "kind": "discharge", "role": "negative-control",
     "params": dict(HOLD_BASE, outlet_diameter=0.008, discharge_time=0.40,
                    n_windows=4, settle_time=0.20)},
    # --- Janssen wall pressure in the filled hold (C3)
    {"id": "J_mu040", "kind": "static", "role": "primary",
     "params": dict(HOLD_BASE, wall_mu=0.40, settle_time=0.40)},
    {"id": "J_mu020", "kind": "static", "role": "lambda-scaling",
     "params": dict(HOLD_BASE, wall_mu=0.20, settle_time=0.40)},
]


# --------------------------------------------------------------------------- #
# workers
# --------------------------------------------------------------------------- #
def _clean(obj):
    """JSON-safe copy (numpy -> python, drop particle arrays)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()
                if k not in ("positions", "alive")}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run_case(spec: dict) -> dict:
    t0 = time.time()
    kind = spec["kind"]
    if kind == "pour":
        res = simulate_pour(PourCase(**spec["params"]))
        out = {
            "id": spec["id"], "kind": kind, "role": spec["role"],
            "params": spec["params"],
            "repose": res["repose"],
            "n_particles": res["n_particles"],
            "steps_run": res["steps_run"],
            "pour_complete_step": res["pour_complete_step"],
            "numerics": res["numerics"],
            "mu": res["mu"], "mu_rolling": res["mu_rolling"],
        }
    elif kind == "discharge":
        res = simulate_hold_discharge(CargoHoldCase(**spec["params"]))
        out = {k: v for k, v in res.items() if k != "positions"}
        out.update({"id": spec["id"], "role": spec["role"],
                    "params": spec["params"]})
    elif kind == "static":
        res = simulate_hold_static(CargoHoldCase(**spec["params"]))
        out = {k: v for k, v in res.items()}
        out.update({"id": spec["id"], "role": spec["role"],
                    "params": spec["params"]})
    else:                                                # pragma: no cover
        raise SystemExit(f"unknown case kind {kind!r}")
    out["result_kind"] = out.get("kind")
    out["kind"] = kind
    out = _clean(out)
    out["wall_time_s"] = round(time.time() - t0, 2)
    out["host"] = platform.node()
    return out


# --------------------------------------------------------------------------- #
# criteria
# --------------------------------------------------------------------------- #
def _rel_dev(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else float("inf")


def evaluate(results: list[dict]) -> dict:
    by_id = {r["id"]: r for r in results}
    crit = {}

    # ---- C1: repose angle vs literature band / central value
    r1 = by_id["R1_beads"]["repose"]
    theta1 = r1["theta_deg"]
    band = REPOSE_REFERENCE_BAND_DEG
    c1 = {
        "case": "R1_beads",
        "theta_deg": theta1,
        "reference_deg": REPOSE_REFERENCE_DEG,
        "reference_band_deg": list(band),
        "rel_dev_vs_reference": _rel_dev(theta1, REPOSE_REFERENCE_DEG),
        "inside_literature_band": bool(band[0] <= theta1 <= band[1]),
        "fit_r2": r1["r2"],
        "section_theta_deg": r1["section_theta_deg"],
        "criterion": "|theta_meas - theta_ref|/theta_ref < 0.10 and inside band",
    }
    c1["pass"] = bool(c1["rel_dev_vs_reference"] < 0.10 and c1["inside_literature_band"])
    crit["C1_repose_angle"] = c1

    # ---- C2: Beverloo curve
    dis = [r for r in results if r["role"] == "beverloo"]
    dis.sort(key=lambda r: r["outlet_diameter"])
    pts = []
    for r in dis:
        d = r["particle_diameter"]
        rho_b = r["bulk_density_kg_m3"]
        th = float(beverloo_prediction(r["outlet_diameter"], d, rho_b))
        pts.append({
            "case": r["id"],
            "D_m": r["outlet_diameter"],
            "D_over_d": r["outlet_diameter"] / d,
            "W_measured_kg_s": r["flow_rate_kg_s"],
            "W_std_kg_s": r["flow_rate_std_kg_s"],
            "W_theory_kg_s": th,
            "rel_dev": _rel_dev(r["flow_rate_kg_s"], th),
            "windows_kg_s": r["flow_rate_windows_kg_s"],
            "steadiness_r2": r["flow_steadiness_r2"],
            "bulk_density_kg_m3": rho_b,
            "packing_fraction": r["packing_fraction"],
            "n_discharged": r["n_discharged"],
            "n_initial": r["n_initial"],
        })
    fit = fit_power_law([p["D_m"] for p in pts], [p["W_measured_kg_s"] for p in pts],
                        pts[0]["D_over_d"] and pts[0]["D_m"] / pts[0]["D_over_d"])
    worst = max((p["rel_dev"] for p in pts), default=float("inf"))
    c2 = {
        "points": pts,
        "law": "W = C rho_b sqrt(g) (D - k d)^(5/2)",
        "constants": {"C": BEVERLOO_C, "k": BEVERLOO_K},
        "fit_exponent": fit["exponent"],
        "fit_r2": fit["r2"],
        "fit_prefactor": fit["prefactor"],
        "worst_rel_dev": worst,
        "criterion": "max_D |W_meas - W_Beverloo| / W_Beverloo < 0.10",
    }
    c2["pass"] = bool(worst < 0.10)
    crit["C2_beverloo_curve"] = c2

    # ---- C3: Janssen profile
    j = by_id["J_mu040"]
    prof = j["profile"]
    fit_v = j["janssen_fit_sigma_v"]
    k_meas = j["K_measured"]
    rho_b = j["bulk_density_kg_m3"]
    R_h = j["hydraulic_radius_m"]
    # Janssen prediction with the *measured* K (a material property of this bed)
    # and the analytic momentum balance -> no fitted profile parameter.
    # Janssen slab balance for a cross-section of hydraulic radius R_h = A/P:
    #   d(sigma_v)/dz = rho_b g - mu_w K sigma_v / R_h
    #   => lambda = R_h/(mu_w K), p_inf = rho_b g R_h/(mu_w K)   (for a cylinder
    #   of radius R, R_h = R/2 and this reduces to the classical R/(2 mu_w K)).
    mu_w = by_id["J_mu040"]["wall_mu"]
    lam_th = R_h / (mu_w * k_meas)
    p_inf_th = rho_b * 9.81 * R_h / (mu_w * k_meas)
    devs = []
    excluded = []
    for row in prof:
        z = j["bed_top_m"] - row["z_mid"]
        pred = p_inf_th * (1.0 - math.exp(-z / lam_th))
        row_out = dict(row)
        row_out["z_below_surface"] = z
        row_out["sigma_v_theory_Pa"] = pred
        row_out["rel_dev"] = _rel_dev(row["sigma_v_Pa"], pred)
        # A slice thinner than two particle diameters is not a continuum slice of
        # the Janssen balance (it holds 1-2 grain layers and sigma_v reduces to a
        # single-contact statistics), so it is reported but excluded from the
        # criterion -- the exclusion is explicit, not silent.
        if z >= 2.0 * j["params"]["radius"] * 2.0:
            devs.append(row_out["rel_dev"])
            row_out["in_criterion"] = True
        else:
            excluded.append(row_out)
            row_out["in_criterion"] = False
    worst_v = max(devs, default=float("inf"))
    identity_meas = j["identity_p_inf_over_lambda_Pa_per_m"]
    identity_th = j["identity_theory_Pa_per_m"]
    ident_dev = _rel_dev(identity_meas, identity_th)
    c3 = {
        "case": "J_mu040",
        "profile": [{k: v for k, v in r.items() if k != "sigma_v_theory_Pa"}
                    for r in prof],
        "K_measured": k_meas,
        "lambda_measured_m": fit_v["lambda"],
        "p_inf_measured_Pa": fit_v["p_inf"],
        "lambda_theory_m": lam_th,
        "p_inf_theory_Pa": p_inf_th,
        "fit_rms_rel": fit_v["rms_rel"],
        "worst_point_rel_dev": worst_v,
        "excluded_surface_slabs": excluded,
        "criterion_slabs": len(devs),
        "identity_measured": identity_meas,
        "identity_theory": identity_th,
        "identity_rel_dev": ident_dev,
        "rel_dev_from_hydrostatic": j["rel_dev_from_hydrostatic"],
        "criterion": ("max_z |p_meas - p_Janssen| / p_Janssen < 0.15 with "
                      "lambda from the measured K"),
    }
    c3["pass"] = bool(worst_v < 0.15 and ident_dev < 0.15)
    crit["C3_janssen_pressure"] = c3

    # ---- C4: negative controls
    r4 = by_id["R4_frictionless"]["repose"]
    blocked = by_id["W_D008_blocked"]
    c4a = {
        "case": "R4_frictionless",
        "mu": 0.0, "mu_rolling": 0.0,
        "theta_deg": r4["theta_deg"],
        "tan_theta": r4["tan_theta"],
        "criterion": "mu = mu_r = 0 -> theta_repose -> 0 (< 5 deg)",
        "pass": bool(r4["theta_deg"] < 5.0),
    }
    blocked_ratio = (blocked["flow_rate_kg_s"]
                     / max(pts[0]["W_measured_kg_s"], 1e-12)) if pts else float("nan")
    c4b = {
        "case": "W_D008_blocked",
        "outlet_diameter": blocked["outlet_diameter"],
        "outlet_over_d": blocked["outlet_over_d"],
        "flow_rate_kg_s": blocked["flow_rate_kg_s"],
        "n_discharged": blocked["n_discharged"],
        "n_initial": blocked["n_initial"],
        "ratio_to_smallest_flowing_case": blocked_ratio,
        "criterion": "D < 2 d -> W -> 0 (arching; <= 5 % of the D/d=5 rate)",
        "pass": bool(blocked["outlet_over_d"] < 2.0 and blocked_ratio <= 0.05),
    }
    crit["C4a_frictionless_repose"] = c4a
    crit["C4b_blocked_orifice"] = c4b

    # ---- C5: end-to-end payload
    payload = {
        "repose_angle": {
            "primary_case": "R1_beads",
            "theta_deg": c1["theta_deg"],
            "reference_deg": REPOSE_REFERENCE_DEG,
            "rel_dev": c1["rel_dev_vs_reference"],
            "tan_theta": r1["tan_theta"],
            "fit_r2": r1["r2"],
            "cases": {r["id"]: {"mu_rolling": r["mu_rolling"],
                                "theta_deg": r["repose"]["theta_deg"],
                                "r2": r["repose"]["r2"]}
                      for r in results if r["kind"] == "pour"},
        },
        "beverloo_curve": {
            "points": [{k: p[k] for k in ("D_m", "D_over_d", "W_measured_kg_s",
                                          "W_theory_kg_s", "rel_dev")}
                       for p in pts],
            "fit_exponent": c2["fit_exponent"],
            "fit_r2": c2["fit_r2"],
            "blocked_control": c4b,
        },
        "janssen_pressure": {
            "profile": [{k: r[k] for k in ("z_mid", "sigma_h_Pa", "sigma_v_Pa")}
                        for r in prof],
            "K_measured": k_meas,
            "lambda_measured_m": fit_v["lambda"],
            "lambda_theory_m": lam_th,
            "p_inf_measured_Pa": fit_v["p_inf"],
            "p_inf_theory_Pa": p_inf_th,
        },
    }
    c5 = {
        "keys": sorted(payload.keys()),
        "criterion": "JSON with repose_angle + beverloo_curve + janssen_pressure",
        "pass": sorted(payload.keys()) == ["beverloo_curve", "janssen_pressure",
                                           "repose_angle"],
    }
    crit["C5_payload"] = c5
    crit["payload"] = payload
    crit["all_pass"] = bool(all(crit[k]["pass"] for k in
                                ("C1_repose_angle", "C2_beverloo_curve",
                                 "C3_janssen_pressure", "C4a_frictionless_repose",
                                 "C4b_blocked_orifice", "C5_payload")))
    return crit


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:                                    # pragma: no cover
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="artifacts/bulk_cargo")
    ap.add_argument("--assemble-only", action="store_true",
                    help="re-evaluate raw/*.json without running the DEM again")
    ap.add_argument("--only", default=None,
                    help="comma separated case ids (diagnostics)")
    args = ap.parse_args(argv)

    specs = CASES
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        specs = [c for c in CASES if c["id"] in want]
        missing = want - {c["id"] for c in specs}
        if missing:
            raise SystemExit(f"unknown case ids: {sorted(missing)}")

    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    t0 = time.time()
    results = []
    if args.assemble_only:
        import glob
        for path in sorted(glob.glob(os.path.join(args.out, "raw", "*.json"))):
            with open(path) as fh:
                results.append(json.load(fh))
        specs = [r["params"] and {"id": r["id"], "kind": r["kind"],
                                  "role": r["role"], "params": r["params"]}
                 for r in results]
        print(f"[assemble] {len(results)} raw cases", flush=True)
    elif args.jobs <= 1:
        for spec in specs:
            r = run_case(spec)
            results.append(r)
            print(f"[done] {r['id']} {r['wall_time_s']}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_case, spec): spec["id"] for spec in specs}
            for fut in futs:
                r = fut.result()
                results.append(r)
                print(f"[done] {r['id']} {r['wall_time_s']}s", flush=True)
    wall = time.time() - t0

    for r in results:
        with open(os.path.join(args.out, "raw", f"{r['id']}.json"), "w") as fh:
            json.dump(r, fh, indent=1)

    case_doc = {
        "node": "LW3-N06",
        "repo": "TensorDEM",
        "module": "src/tensordem/bulk_cargo.py",
        "commit": _git_commit(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hold_base": HOLD_BASE,
        "pour_base": POUR_BASE,
        "cases": specs,
        "numerics": {
            "hold": _clean(numerical_checks(CargoHoldCase(**HOLD_BASE))),
            "pour": _clean(numerical_checks(PourCase(**POUR_BASE))),
        },
        "wall_time_s": round(wall, 1),
        "jobs": args.jobs,
        "host": platform.node(),
    }
    complete = len(results) == len(CASES)
    if complete:
        crit = evaluate(results)
    else:                                   # --only diagnostic run
        crit = {"partial": True, "payload": None, "all_pass": False}
    results_doc = {
        "node": "LW3-N06",
        "commit": case_doc["commit"],
        "generated_at": case_doc["generated_at"],
        "criteria": {k: v for k, v in crit.items() if k != "payload"},
        "payload": crit["payload"],
        "all_pass": crit["all_pass"],
        "raw": results,
        "wall_time_s": round(wall, 1),
    }
    with open(os.path.join(args.out, "case.json"), "w") as fh:
        json.dump(case_doc, fh, indent=1)
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(results_doc, fh, indent=1)

    if complete:
        print(json.dumps({k: v.get("pass") for k, v in crit.items()
                          if isinstance(v, dict) and "pass" in v}, indent=1))
    print(f"all_pass={crit['all_pass']} wall_time={wall:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())