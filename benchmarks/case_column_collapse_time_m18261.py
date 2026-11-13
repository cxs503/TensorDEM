"""CASE-COLUMN-COLLAPSE · grid refinement + pre-registered arrival definitions.

Mission m18261.  Anchor ``CASE-COLUMN-COLLAPSE`` (算例池·颗粒柱坍塌, reference
Lube, Huppert, Sparks & Freundt 2004 JFM **508**:175-199,
doi:10.1017/S0022112004009036, tolerance 15 %).

State of the anchor on entry
----------------------------
Round 5 (``docs/case_column_collapse_round5_m17111.md``) closed the **first**
prediction (runout law) 5/5 inside 15 % with one frozen configuration — the
"closed arm" ``mu = 0.5, mu_rolling = 0.5, mu_wall = 0.8`` — and
``docs/AUDIT_COLUMN_COLLAPSE_TIME_LAW_m17534.md`` re-derived that verdict from
the raw case JSONs (5/5 at nominal and at measured aspect ratio).  The same
audit showed the **second** prediction (emplacement time
``t_inf = 3 sqrt(r_i a / g)``) is *not* closed: 4/5 at the 90 % definition, 1/5
at 99 %, and the verdict moves 21 percentage points across the arrival-
definition family.  Two questions were left explicitly open:

1. **arrival definition** — "the correct thing to do is to declare a velocity-
   threshold definition (needs a re-run)" (audit §5.1);
2. **resolution** — the audit noted the runout closure rests on *one*
   resolution (8 grains per column radius) and that round 1's "6 vs 8 grains"
   and round 5's "12 grains" controls were never shown to be refinements at all.

Both questions are decided here, on new runs, with the protocol frozen below
*before* the first case is integrated.

What is new here (and why it is not a re-run of rounds 1-5)
-----------------------------------------------------------
* **Refinement is done correctly for the first time.**  A grid-refinement study
  must hold the *dimensionless* problem fixed.  In this case that means holding
  the column radius ``r_i`` and the dimensionless contact stiffness (the static
  sag ratio ``delta / d = pi rho g d^2 / (6 k_n)``) fixed while ``d / r_i``
  shrinks.  Therefore ``radius = 0.24 / gpr`` and ``k_n = 6e4 * (d / 0.03)^2``.
  Round 5's ``w5c_gpr12_*`` controls instead kept ``radius = 0.03 m``, which
  turned "12 grains per radius" into a column of ``r_i = 0.718 m`` (+52 % in
  radius, ``a_meas`` 1.078 -> 0.988): a **scale** change, not a refinement.
  That historical defect is re-verified from the archived JSONs (axis ``H6``)
  and never re-used as convergence evidence.
* **Arrival definitions are pre-registered** in
  :mod:`tensordem.granular_column.arrival` (fraction / velocity-threshold /
  length-tolerance families, all members reported, primary members named) and
  evaluated on **4x denser** front histories (``sample_stride`` 20 -> 5) so the
  velocity-threshold member is resolvable.
* **Two confounding axes are separated**: initial lattice gap
  (``packing_gap`` 0.02 vs 0.0 — the 2 % gap is an initial free-fall transient
  the experiment does not have) and time-step (``dt`` vs ``dt/2``).  The audit
  attributed the shallow-end lateness to "creep"; if instead the initial
  free-fall/compaction transient is responsible, ``packing_gap = 0`` should
  move the shallow-end error, and that is a *fixable* initialization artifact.

Pre-registered protocol (frozen before any run; do not edit after the fact)
--------------------------------------------------------------------------
Frozen closed-arm configuration (round 5, unchanged)::

    mu = 0.5   mu_rolling = 0.5   mu_wall = 0.8   restitution = 0.5
    kt_ratio = 0.5   gravity = 9.81   density = 2500 kg/m^3
    packing_gap = 0.02   jitter = 0   seed = 12345   dt_scale = 1.0
    duration_factor = 8.0   min_time = 0.5 s   contact_mode = "reference"
    r_i = 0.48 m (via grains_per_radius * 2 * radius)

Axes (14 cases)::

    R  refinement, a in {0.5, 1.0} x gpr in {8, 12, 16}
       + a in {1.5, 2.0} at gpr = 8  (completes the 5-aspect law of the card)
       + a = 3.0 x gpr in {8, 12}    (deep-end anchor of the sqrt branch)
    T  integration, a in {0.5, 1.0} at gpr = 8, dt_scale = 0.5
    G  initialisation, a in {0.5, 1.0} at gpr = 8, packing_gap = 0.0

Hypotheses and decision rules (fixed now, adjudicated in
``scripts/analyze_column_collapse_time_m18261.py``):

* **H1** (the audit's proposed remedy).  Under the velocity-threshold
  definition ``v_front < 0.01 v_peak``, all five card aspects
  ``a in {0.5, 1, 1.5, 2, 3}`` are inside 15 % of ``t_inf``.
  Supported **iff** 5/5 pass.  Anything less = falsified (the family is still
  reported; "the audit suggested it" is not evidence).
* **H2** (runout closure survives refinement).  At ``a in {0.5, 1.0}`` and every
  ``gpr in {8, 12, 16}``: ``|err_runout| <= 15 %`` **and**
  ``|err(gpr=16) - err(gpr=8)| <= 5 pp``.  If a refined point leaves the band,
  the round-5 closure is resolution-limited and must be reported as such.
* **H3** (the time deficit is model, not discretisation).  At ``a = 0.5`` the
  error of the *primary* time definitions (fraction 0.90, velocity 0.01,
  length 1 d) changes by ``<= 5 pp`` from gpr 8 to gpr 16.  A change
  ``>= 5 pp`` attributes it to discretisation; if gpr 16 also puts every aspect
  inside 15 %, the second prediction closes at converged resolution instead.
* **H4** (initialisation, not creep).  ``packing_gap`` 0.0 vs 0.02 at
  ``a in {0.5, 1.0}``, gpr 8: both ``err_runout`` and ``err_t`` move by
  ``<= 5 pp`` -> the 2 % lattice gap is not the cause (else it is a fixable
  initial-condition artifact, and the fixed-gap result becomes the new
  reference configuration for the time axis only — the runout closure stays on
  the frozen closed arm, which is what the card accepted).
* **H5** (time-step adequacy).  ``dt/2`` moves ``err_runout`` and ``err_t`` by
  ``<= 2 pp`` at ``a in {0.5, 1.0}``.
* **H6** (historical control defect).  Re-derived from the archived round-5
  JSONs (read-only, ``--audit-historical``): those "12 grains per radius"
  controls changed ``r_i``, hence they are not refinements.  Reported as a
  method finding, no new run needed.

Every metric is written per case (JSON + CSV); the analysis module recomputes
nothing by hand and the report is injected from those files.

Usage
-----
::

    python3 benchmarks/case_column_collapse_time_m18261.py --list
    python3 benchmarks/case_column_collapse_time_m18261.py --case R_a0.5_g8
    python3 benchmarks/case_column_collapse_time_m18261.py --shard 0/8
    python3 benchmarks/case_column_collapse_time_m18261.py --collect
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from tensordem.granular_column.arrival import arrival_report  # noqa: E402
from tensordem.granular_column.axisym import (  # noqa: E402
    AxisymCase, lube_2004_emplacement_time, lube_2004_runout_ratio,
    simulate_axisymmetric_column)

ROUND = "m18261"
OUT_DIR = os.path.join("runs", "case_column_collapse_time_m18261")
CASE_DIR = os.path.join(OUT_DIR, "cases")
TOLERANCE = 0.15
#: Column radius held fixed across the refinement axis (m).
COLUMN_RADIUS = 0.48
#: Reference grain radius of the frozen closed arm (m) and its stiffness.
REF_RADIUS = 0.03
REF_KN = 6.0e4
#: Frozen closed-arm configuration (round 5; unchanged).
BASE = dict(density=2500.0, mu=0.5, mu_rolling=0.5, mu_wall=0.8,
            restitution=0.5, kt_ratio=0.5, gravity=9.81,
            packing_gap=0.02, jitter=0.0, seed=12345,
            dt_scale=1.0, duration_factor=8.0, min_time=0.5,
            sample_stride=5, rest_ke_tol=1.0e-4, rest_hold=200,
            contact_mode="reference")
#: Card aspects (all five are judged; the tolerance is the card's).
CARD_ASPECTS = (0.5, 1.0, 1.5, 2.0, 3.0)
#: Pre-registered primary definitions (see arrival module).
PRIMARY = ("fraction_0.9", "velocity_0.01", "length_1d")


def case_radius(gpr: float) -> float:
    """Grain radius that keeps ``r_i`` fixed while ``gpr`` changes."""
    return COLUMN_RADIUS / (2.0 * float(gpr))


def case_stiffness(radius: float) -> float:
    """``k_n`` that keeps the static sag ratio ``delta/d`` of the closed arm."""
    return REF_KN * (float(radius) / REF_RADIUS) ** 2


def build_cases() -> list[dict]:
    """The 14 pre-registered cases (axes R/T/G)."""
    cases = []

    def add(cid, axis, aspect, gpr, **over):
        cases.append({"case_id": cid, "axis": axis, "aspect_ratio": aspect,
                      "grains_per_radius": gpr, **over})

    for aspect in (0.5, 1.0):
        for gpr in (8.0, 12.0, 16.0):
            add(f"R_a{aspect:g}_g{gpr:g}", "R", aspect, gpr)
    for aspect in (1.5, 2.0):
        add(f"R_a{aspect:g}_g8", "R", aspect, 8.0)
    for gpr in (8.0, 12.0):
        add(f"R_a3_g{gpr:g}", "R", 3.0, gpr)
    for aspect in (0.5, 1.0):
        add(f"T_a{aspect:g}_g8_dt0.5", "T", aspect, 8.0, dt_scale=0.5)
    for aspect in (0.5, 1.0):
        add(f"G_a{aspect:g}_g8_gap0", "G", aspect, 8.0, packing_gap=0.0)
    return cases


CASES = build_cases()
CASE_BY_ID = {c["case_id"]: c for c in CASES}


def case_path(cid: str) -> str:
    return os.path.join(CASE_DIR, f"{cid}.json")


def make_case(spec: dict) -> AxisymCase:
    gpr = float(spec["grains_per_radius"])
    radius = case_radius(gpr)
    kwargs = dict(BASE)
    kwargs.update({k: v for k, v in spec.items()
                   if k not in ("case_id", "axis", "aspect_ratio",
                                "grains_per_radius")})
    return AxisymCase(aspect_ratio=float(spec["aspect_ratio"]),
                      grains_per_radius=gpr, radius=radius,
                      kn=case_stiffness(radius), label=spec["case_id"],
                      **kwargs)


def protocol_signature(spec: dict) -> dict:
    """Canonical, hashable description of one case (provenance)."""
    case = make_case(spec)
    sig = case.signature()
    sig.pop("label", None)
    sig["case_id"] = spec["case_id"]
    sig["axis"] = spec["axis"]
    sig["column_radius_target"] = COLUMN_RADIUS
    sig["sag_ratio"] = (math.pi * case.density * case.gravity * case.radius ** 2
                        / (6.0 * case.kn))
    sig["round"] = ROUND
    return sig


def signature_sha256(sig: dict) -> str:
    return hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()


def reference_values(spec: dict, aspect_measured: float, r_i_measured: float,
                     gravity: float) -> dict:
    """Reference law values, evaluated (never transcribed).

    Both aspect conventions are reported: the card's nominal ``a`` and the
    measured ``a`` of the built packing (audit note 2 of m17534).
    """
    nominal = float(spec["aspect_ratio"])
    out = {}
    for name, aspect, radius in (("nominal", nominal, COLUMN_RADIUS),
                                 ("measured", aspect_measured, r_i_measured)):
        out[name] = {
            "aspect": aspect,
            "runout_ratio": lube_2004_runout_ratio(aspect),
            "emplacement_time": lube_2004_emplacement_time(aspect, radius,
                                                           gravity),
        }
    return out


def error(measured: float, reference: float) -> float:
    if not math.isfinite(measured) or not math.isfinite(reference) or reference == 0.0:
        return float("nan")
    return (measured - reference) / reference


def run_case(spec: dict, progress: bool = False) -> dict:
    """Integrate one pre-registered case (cache-first, deterministic)."""
    cid = spec["case_id"]
    path = case_path(cid)
    if os.path.exists(path):
        with open(path) as fh:
            res = json.load(fh)
        res["cached"] = True
        return res
    case = make_case(spec)
    sig = protocol_signature(spec)
    t0 = time.perf_counter()
    res = simulate_axisymmetric_column(case, return_positions=False,
                                       progress=progress)
    wall = time.perf_counter() - t0
    hist = res["history"]
    time_hist = np.asarray(hist["time"], dtype=float)
    front_hist = np.asarray(hist["front"], dtype=float)
    arrivals = arrival_report(time_hist, front_hist,
                              radius=res["column_radius_measured"],
                              grain_diameter=2.0 * case.radius)
    refs = reference_values(spec, res["aspect_measured"],
                            res["column_radius_measured"], case.gravity)
    out = {
        "case_id": cid,
        "axis": spec["axis"],
        "protocol": sig,
        "protocol_sha256": signature_sha256(sig),
        "n_particles": res["n_particles"],
        "dt": res["dt"],
        "n_steps": res["n_steps"],
        "time_end": res["time_end"],
        "stop_reason": res["stop_reason"],
        "aspect_nominal": float(spec["aspect_ratio"]),
        "aspect_measured": res["aspect_measured"],
        "column_radius_measured": res["column_radius_measured"],
        "column_height_measured": res["column_height_measured"],
        "solid_fraction_initial": res["solid_fraction_initial"],
        "runout_ratio_p995": res["runout_ratio_p995"],
        "ballistic_time": res["ballistic_time"],
        "reference": refs,
        "arrival": arrivals,
        "energy": res["energy"],
        "inertial": res["inertial"],
        "coordination_max": res["coordination_max"],
        "history": {"time": hist["time"], "front": hist["front"],
                    "kinetic": hist["kinetic"], "contacts": hist["contacts"]},
        "wall_clock_s": wall,
    }
    for conv in ("nominal", "measured"):
        out[f"runout_error_{conv}"] = error(res["runout_ratio_p995"],
                                            refs[conv]["runout_ratio"])
        out[f"t_error_{conv}"] = {
            key: error(value, refs[conv]["emplacement_time"])
            for key, value in arrivals["primary"].items()}
    out["within_tolerance_runout"] = (abs(out["runout_error_nominal"]) <= TOLERANCE)
    os.makedirs(CASE_DIR, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh)
    return out


def run_shard(index: int, total: int, progress: bool = False) -> list[str]:
    done = []
    for k, spec in enumerate(CASES):
        if k % total != index:
            continue
        print(f"[shard {index}/{total}] {spec['case_id']} ...", flush=True)
        res = run_case(spec, progress=progress)
        print(f"  N={res['n_particles']} runout={res['runout_ratio_p995']:.3f} "
              f"err={res['runout_error_nominal']:+.3%} wall={res['wall_clock_s']:.0f}s "
              f"stop={res['stop_reason']}", flush=True)
        done.append(spec["case_id"])
    return done


def collect(out_dir: str = OUT_DIR) -> dict:
    """Assemble summary.json + cases.csv from the per-case JSONs."""
    rows, missing = [], []
    for spec in CASES:
        path = case_path(spec["case_id"])
        if not os.path.exists(path):
            missing.append(spec["case_id"])
            continue
        with open(path) as fh:
            rows.append(json.load(fh))
    header = ["case_id", "axis", "aspect_nominal", "aspect_measured",
              "grains_per_radius", "radius", "kn", "n_particles", "dt",
              "stop_reason", "runout_ratio", "runout_ref_nominal",
              "err_runout_nominal", "err_runout_measured"]
    for key in PRIMARY:
        header += [f"t_{key}", f"err_{key}_nominal", f"err_{key}_measured"]
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "cases.csv")
    with open(csv_path, "w") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            vals = [row["case_id"], row["axis"], row["aspect_nominal"],
                    row["aspect_measured"], row["protocol"]["grains_per_radius"],
                    row["protocol"]["radius"], row["protocol"]["kn"],
                    row["n_particles"], row["dt"], row["stop_reason"],
                    row["runout_ratio_p995"],
                    row["reference"]["nominal"]["runout_ratio"],
                    row["runout_error_nominal"], row["runout_error_measured"]]
            for key in PRIMARY:
                vals += [row["arrival"]["primary"][key],
                         row["t_error_nominal"][key], row["t_error_measured"][key]]
            fh.write(",".join(("" if v is None else
                               (f"{v:.6g}" if isinstance(v, float) else str(v)))
                              for v in vals) + "\n")
    summary = {
        "round": ROUND,
        "anchor": "CASE-COLUMN-COLLAPSE",
        "tolerance": TOLERANCE,
        "n_cases_declared": len(CASES),
        "n_cases_present": len(rows),
        "missing": missing,
        "cases_csv": csv_path,
        "cases": [{k: v for k, v in row.items() if k != "history"} for row in rows],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--case")
    ap.add_argument("--shard", help="i/n")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for spec in CASES:
            case = make_case(spec)
            print(f"{spec['case_id']:<18} axis={spec['axis']} "
                  f"a={spec['aspect_ratio']:<4g} gpr={spec['grains_per_radius']:<4g} "
                  f"d={2 * case.radius:.4f}m kn={case.kn:.4g} "
                  f"dt={case.material().timestep:.3e}s "
                  f"gap={case.packing_gap:g} dt_scale={case.dt_scale:g}")
        return 0
    if args.case:
        if args.case not in CASE_BY_ID:
            raise SystemExit(f"unknown case {args.case}")
        run_case(CASE_BY_ID[args.case], progress=args.progress)
        return 0
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        run_shard(i, n, progress=args.progress)
        return 0
    if args.collect:
        summary = collect()
        print(json.dumps({k: summary[k] for k in
                          ("n_cases_declared", "n_cases_present", "missing")},
                         indent=1))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())