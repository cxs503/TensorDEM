"""CASE-COLUMN-COLLAPSE — axisymmetric granular-column collapse vs Lube et al. (2004).

算例池 case card (owner 2026-11-12).  Reference experiment
--------------------------------------------------------
Lube, Huppert, Sparks & Freundt (2004), *Axisymmetric collapses of granular
columns*, J. Fluid Mech. **508**, 175-199, doi:10.1017/S0022112004009036.
Targets (paper abstract, quoted verbatim in
:mod:`tensordem.granular_column.axisym`):

* ``r_inf = r_i (1 + 1.24 a)``            for ``a < 1.7``
* ``r_inf = r_i (1 + 1.6 a^(1/2))``       for ``1.7 < a``
* ``t_inf = 3 (h_i/g)^(1/2) = 3 (r_i/g)^(1/2) a^(1/2)``

**evaluated from the formula at run time** (:func:`lube_2004_runout_ratio`,
:func:`lube_2004_emplacement_time`) — no reference number is typed by hand.

Usage
-----
::

    python benchmarks/case_column_collapse_lube2004.py --list
    python benchmarks/case_column_collapse_lube2004.py --wave1        # control + calibration
    python benchmarks/case_column_collapse_lube2004.py --shard 2/6    # wave-2 worker
    python benchmarks/case_column_collapse_lube2004.py --collect      # verdicts

``--wave1`` runs the negative control (``mu_r = 0``, no rolling resistance) and
the one-parameter calibration grid at ``a = 0.5``; ``--shard i/n`` runs one
worker of the wave-2 grid (validation + sensitivity) at a given ``mu_r``
(``--mu-rolling``, read off the calibration by ``--collect``).
``--collect`` evaluates the predeclared criteria (< 15 % relative error on the
runout radius ratio and on the emplacement time) over every stored case and
writes ``summary.json`` + ``cases.csv``.  Nothing is recomputed from a
hard-coded number, so the per-case JSONs are the provenance.

Predeclared protocol (fixed before the runs, see
``docs/case_column_collapse_lube2004.md``)
------------------------------------------
control     ``a = 1.0``, ``mu_rolling = 0`` — demonstrates that sphere DEM
            without rolling resistance never arrests (degenerate deposit);
calibration ``a = 0.5``, ``mu_rolling in {0.2, 0.3, 0.4}`` — **one** grain-shape
            parameter, fitted on the smallest aspect ratio only;
validation  ``a in {0.5, 1.0, 1.5, 2.0, 3.0}`` at the calibrated
            ``mu_rolling`` — out-of-sample for ``a != 0.5``;
sensitivity ``a = 2.0``: ``grains_per_radius = 14``, ``dt/2``, ``mu = 0.3``,
            ``mu = 0.8`` (all at the calibrated ``mu_rolling``).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.granular_column.axisym import (LUBE2004_DOI, LUBE2004_SOURCE,  # noqa: E402
                                             AxisymCase, lube_2004_emplacement_time,
                                             lube_2004_runout_ratio, verdict,
                                             simulate_axisymmetric_column)

OUT_DIR = os.path.join("runs", "case_column_collapse_lube2004")
CASE_DIR = os.path.join(OUT_DIR, "cases")
TOLERANCE = 0.15

BASE = dict(radius=0.03, density=2500.0, mu=0.5, mu_rolling=0.3,
            restitution=0.5, kn=6.0e4, kt_ratio=0.5, gravity=9.81,
            packing_gap=0.02, jitter=0.0, seed=12345,
            dt_scale=1.0, duration_factor=8.0, min_time=0.5,
            sample_stride=20, rest_ke_tol=1.0e-4, rest_hold=200,
            grains_per_radius=8.0)

VALIDATION_ASPECTS = (1.0, 1.5, 2.0, 3.0)
CALIBRATION_ASPECT = 0.5
CALIBRATION_MUR = (0.2, 0.3, 0.4)


def wave1_cases() -> list[dict]:
    """Negative control + the mu_rolling calibration grid (cheap, ~3 min total)."""
    cases = [{"id": "control_mur0_a1.0", "phase": "control",
              "note": "no rolling resistance: spheres cannot arrest (negative control)",
              "params": dict(BASE, aspect_ratio=1.0, mu_rolling=0.0)}]
    for mur in CALIBRATION_MUR:
        cases.append({"id": f"calib_mur{mur:g}_a{CALIBRATION_ASPECT:g}",
                      "phase": "calibration",
                      "note": f"rolling resistance mu_r = {mur:g} at a = "
                              f"{CALIBRATION_ASPECT:g} (single fitted parameter)",
                      "params": dict(BASE, aspect_ratio=CALIBRATION_ASPECT,
                                     mu_rolling=mur)})
    return cases


def wave2_cases(mu_rolling: float) -> list[dict]:
    """Validation scan + sensitivity grid at the calibrated mu_rolling."""
    cases = [{"id": f"valid_mur{mu_rolling:g}_a{a:g}", "phase": "validation",
              "note": f"mu_r = {mu_rolling:g} (calibrated at a = 0.5)",
              "params": dict(BASE, aspect_ratio=a, mu_rolling=mu_rolling)}
             for a in VALIDATION_ASPECTS]
    sens = (
        ("sens_dt0.5_a2.0", dict(aspect_ratio=2.0, dt_scale=0.5),
         "time step halved"),
        ("sens_gpr6_a2.0", dict(aspect_ratio=2.0, grains_per_radius=6.0),
         "particle resolution: 6 grains per column radius (coarser)"),
        ("sens_mu0.3_a2.0", dict(aspect_ratio=2.0, mu=0.3),
         "contact friction mu = 0.3"),
        ("sens_gpr11_a2.0", dict(aspect_ratio=2.0, grains_per_radius=11.0),
         "particle resolution: 11 grains per column radius"),
    )
    for cid, params, note in sens:
        cases.append({"id": cid + ("" if mu_rolling == BASE["mu_rolling"]
                                   else f"_mur{mu_rolling:g}"),
                      "phase": "sensitivity", "note": note,
                      "params": dict(BASE, **params, mu_rolling=mu_rolling)})
    return cases


def case_path(case_id: str) -> str:
    return os.path.join(CASE_DIR, f"{case_id}.json")


def evaluate(res: dict, case: AxisymCase) -> dict:
    """Predeclared criteria for one case (targets evaluated from the formulas)."""
    a = case.aspect_ratio
    ref_ratio = lube_2004_runout_ratio(a)
    ref_time = lube_2004_emplacement_time(a, case.column_radius, case.gravity)
    return {
        "runout_ratio": verdict(ref_ratio, res["runout_ratio_p995"], TOLERANCE),
        "runout_ratio_max_metric": verdict(ref_ratio, res["runout_ratio_max"], TOLERANCE),
        "emplacement_time": verdict(ref_time, res["t_reach"], TOLERANCE),
    }


def run_case(spec: dict, progress: bool = False) -> dict:
    t0 = time.perf_counter()
    case = AxisymCase(**spec["params"], label=spec["id"])
    res = simulate_axisymmetric_column(case, progress=progress)
    res["wall_clock_s"] = time.perf_counter() - t0
    res["phase"] = spec["phase"]
    res["note"] = spec.get("note", "")
    res["criteria"] = evaluate(res, case)
    os.makedirs(CASE_DIR, exist_ok=True)
    with open(case_path(spec["id"]), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def _row(res: dict) -> dict:
    c = res["case"]
    crit = res["criteria"]
    return {
        "case": c["label"],
        "phase": res["phase"],
        "aspect_requested": c["aspect_ratio"],
        "aspect_measured": res["aspect_measured"],
        "n_particles": res["n_particles"],
        "dt": res["dt"],
        "grains_per_radius": c["grains_per_radius"],
        "mu": c["mu"],
        "mu_rolling": c["mu_rolling"],
        "runout_ratio_p995": res["runout_ratio_p995"],
        "runout_ratio_max": res["runout_ratio_max"],
        "ref_runout_ratio": crit["runout_ratio"]["reference"],
        "err_runout_p995": crit["runout_ratio"]["relative_error"],
        "pass_runout_p995": crit["runout_ratio"]["pass"],
        "err_runout_max": crit["runout_ratio_max_metric"]["relative_error"],
        "t_reach": res["t_reach"],
        "t90": res.get("t_reach_by_fraction", {}).get("0.9"),
        "t95": res.get("t_reach_by_fraction", {}).get("0.95"),
        "ref_t_reach": crit["emplacement_time"]["reference"],
        "err_emplacement_time": crit["emplacement_time"]["relative_error"],
        "pass_emplacement_time": crit["emplacement_time"]["pass"],
        "stop_reason": res["stop_reason"],
        "energy_residual": res["energy"]["residual_fraction_of_potential"],
        "wall_clock_s": res.get("wall_clock_s"),
    }


def collect() -> dict:
    rows = []
    for path in sorted(glob.glob(os.path.join(CASE_DIR, "*.json"))):
        with open(path) as fh:
            rows.append(_row(json.load(fh)))

    def group(phase):
        return [r for r in rows if r["phase"] == phase]

    def stats(rs):
        if not rs:
            return {"n": 0}
        return {
            "n": len(rs),
            "pass_runout": sum(1 for r in rs if r["pass_runout_p995"]),
            "pass_time": sum(1 for r in rs if r["pass_emplacement_time"]),
            "max_abs_err_runout": max(abs(r["err_runout_p995"]) for r in rs),
            "max_abs_err_time": max((abs(r["err_emplacement_time"]) for r in rs
                                     if r["err_emplacement_time"] == r["err_emplacement_time"]),
                                    default=None),
        }

    calib = [r for r in group("calibration")
             if r["err_runout_p995"] == r["err_runout_p995"]]
    calibrated = min(calib, key=lambda r: abs(r["err_runout_p995"])) if calib else None
    valid = group("validation")
    out_of_sample = [r for r in valid
                     if abs(r["aspect_requested"] - CALIBRATION_ASPECT) > 1e-9]
    summary = {
        "case": "CASE-COLUMN-COLLAPSE",
        "reference": LUBE2004_SOURCE,
        "doi": LUBE2004_DOI,
        "tolerance": TOLERANCE,
        "calibration": {
            "chosen_mu_rolling": calibrated["mu_rolling"] if calibrated else None,
            "chosen_case": calibrated["case"] if calibrated else None,
            "chosen_err": calibrated["err_runout_p995"] if calibrated else None,
        },
        "phases": {
            "control": stats(group("control")),
            "calibration": stats(group("calibration")),
            "validation": stats(valid),
            "validation_out_of_sample": stats(out_of_sample),
            "sensitivity": stats(group("sensitivity")),
        },
        "rows": rows,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(OUT_DIR, "cases.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    return summary


def _report_row(r: dict) -> str:
    return (f"{r['case']:>24} a={r['aspect_requested']:.2f} "
            f"ratio={r['runout_ratio_p995']:.3f} ref={r['ref_runout_ratio']:.3f} "
            f"err={r['err_runout_p995'] * 100:+.1f}% "
            f"{'PASS' if r['pass_runout_p995'] else 'FAIL'} | "
            f"t99={r['t_reach']:.3f} (t90={r['t90']:.3f}) ref={r['ref_t_reach']:.3f} "
            f"err={r['err_emplacement_time'] * 100:+.1f}% "
            f"{'PASS' if r['pass_emplacement_time'] else 'FAIL'}")


def _run_specs(specs: list[dict], tag: str, progress: bool) -> None:
    for spec in specs:
        print(f"[{tag}] {spec['id']}", flush=True)
        res = run_case(spec, progress=progress)
        print(f"  n={res['n_particles']} ratio={res['runout_ratio_p995']:.3f} "
              f"ref={res['reference_runout_ratio']:.3f} t99={res['t_reach']:.3f} "
              f"stop={res['stop_reason']} wall={res['wall_clock_s']:.1f}s", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run one wave-1 case id")
    ap.add_argument("--wave1", action="store_true", help="control + calibration cases")
    ap.add_argument("--mu-rolling", type=float, default=BASE["mu_rolling"],
                    help="calibrated rolling resistance for the wave-2 grid")
    ap.add_argument("--shard", help="i/n: run the i-th of n interleaved wave-2 cases")
    ap.add_argument("--collect", action="store_true", help="aggregate + verdicts")
    ap.add_argument("--list", action="store_true", help="list wave-1 case ids")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for spec in wave1_cases():
            print(spec["id"], spec["phase"], spec["params"]["aspect_ratio"])
        return 0
    if args.collect:
        s = collect()
        print(json.dumps({k: v for k, v in s.items() if k != "rows"}, indent=1))
        for r in s["rows"]:
            print(_report_row(r))
        return 0
    if args.run:
        spec = next((s for s in wave1_cases() if s["id"] == args.run), None)
        if spec is None:
            print(f"unknown case id {args.run}", file=sys.stderr)
            return 2
        res = run_case(spec, progress=args.progress)
        print(json.dumps({k: res[k] for k in
                          ("n_particles", "dt", "n_steps", "time_end", "stop_reason",
                           "runout_ratio_p995", "reference_runout_ratio", "t_reach",
                           "reference_emplacement_time")}, indent=1))
        return 0
    if args.wave1:
        _run_specs(wave1_cases(), "wave1", args.progress)
        return 0
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        mine = [c for k, c in enumerate(wave2_cases(args.mu_rolling)) if k % n == i]
        _run_specs(mine, f"wave2 {i}/{n}", args.progress)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())