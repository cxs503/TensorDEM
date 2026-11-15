"""CASE-DEM-PLANE-SHEAR-MUIDI round 20: full-range, multi-estimator verdict.

Why this file exists
--------------------
Rounds 12-19 left two gaps that no single earlier artefact closed:

1. **Range.**  The ledger acceptance is ``I = 1e-3 ... 0.3``.  Every higher-N
   artefact so far (round 18/19, ``high_n_sweep.py``) sampled only
   ``I in (1e-2, 3e-2, 0.1, 0.3)``; the low-I half of the acceptance range was
   measured only at the *baseline* particle count (N = 294), where rounds 14/16
   showed the noise floor exceeds the 10 % band.
2. **Verdict unit.**  Rounds 13 (provenance) and 19 (abscissa) each fixed one
   axis of the comparison, but the two fixes were never evaluated *together*:
   round 13 scanned the reference envelope on the round-9/10 N = 294 archive
   with the imposed-rate abscissa; round 19 switched to the measured-rate
   abscissa but imported only ``reference_law`` (one triple) and used one mu
   estimator and one phi estimator.  A ledger verdict that mixes an estimator
   choice with a reference-triple choice is not a verdict on the material.

This round therefore runs the *full* ledger range at N ~ 1000+ and reports the
verdict as a **matrix**: {mu estimator} x {abscissa} x {candidate reference
triple}, with the falsification oracle ``best_case_over_envelope`` answering
the only question the ledger can legitimately be failed on -- "no admissible
reading of the cited papers puts this curve inside 10 %".

Protocol parity with round 19 is deliberate (same device, contact law, P/kn,
seed, strain target, box), so the two rounds are comparable point by point and
the four shared I targets double as a reproducibility check.

Usage (shardable; one I target per shard is the intended split)::

    python high_n_coverage.py --shard 0/6
    python high_n_coverage.py --assemble
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tensordem.mu_i_rheology import ServoShearCase, simulate_servo_shear  # noqa: E402
from reference_law import (  # noqa: E402
    I_0, MU_2, MU_REF_STATUS, MU_S, PHI_MAX, PHI_MIN, PHI_REF_STATUS,
    ref_mu, ref_phi,
)
from reference_provenance import (  # noqa: E402
    CANDIDATES, best_case_over_envelope, verdict_table,
)

ROUND = 20
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "high_n_r20"

TOL_PCT = 10.0          # ledger band, stated in the unit it is compared in

# --- device: identical to high_n_sweep.py (round 18/19) --------------------
P = 6.25e4              # Pa, servo-held normal stress
KN = 6.25e4             # N/m
MU_CONTACT = 0.5
MU_ROLLING = 0.05
E = 0.3
R = 0.005
RHO = 2500.0
SEED = 5279
W, D, H = 0.16, 0.08, 0.09      # 32d x 16d x 18d -> N ~ 1000+
SETTLE_STEPS = 3000
SHEAR_STRAIN = 0.5
SAMPLE_STRIDE = 10

# the ledger range, both ends included (round 18/19 started at 1e-2)
I_TARGETS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)

# mu estimators carried side by side (see round-13 Table C for why they differ:
# tau / sigma_zz versus tau / p, where p/sigma_zz ~ 0.77-0.90 in this cell)
MU_ARMS = ("mu_wall", "mu_full", "mu_bulk")
# abscissae: imposed platen rate vs measured bulk shear rate (round 19 D-D)
# "I_abscissa_best" is the round-19 rule: measured bulk rate when the profile
# fit produced one, imposed platen rate otherwise (documented fallback, so an
# arm is never silently dropped because one point lacks a profile).
ABSCISSAE = ("I_measured_rate", "I_imposed_rate", "I_abscissa_best")

PREREG = {
    "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
    "round": ROUND,
    "kind": "full-range (I=1e-3..0.3) higher-N sweep + estimator x abscissa x "
            "reference-triple verdict matrix",
    "device": "tensordem.mu_i_rheology.simulate_servo_shear",
    "box": [W, D, H],
    "n_particles_expected": ">1000",
    "I_targets": list(I_TARGETS),
    "shear_strain": SHEAR_STRAIN,
    "seed": SEED,
    "tolerance_pct": TOL_PCT,
    "hypotheses": {
        "Q1_range": "every ledger-range point is measured at N>1000, so the "
                    "verdict no longer rests on N=294 low-I points",
        "Q2_estimator": "the estimator spread (wall / full-cell / bulk; "
                        "p vs sigma_zz) is smaller than the 10% band, i.e. the "
                        "verdict does not depend on which one is quoted",
        "Q3_abscissa": "the measured-rate abscissa moves mu inside the band "
                       "where the imposed-rate one does not (round-19 claim)",
        "Q4_falsification": "best_case_over_envelope answers whether ANY "
                            "admissible triple can hold the curve inside 10%; "
                            "`provenance_can_rescue=False` means the residual "
                            "is structural, not bibliographic",
        "Q5_honest": "the worst-point error of the best arm is reported "
                     "whether or not it passes",
    },
}


# --------------------------------------------------------------------------
# measurement summary (pure: takes the simulate_servo_shear result dict)
# --------------------------------------------------------------------------
def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs):
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if xs else float("nan")


def _num(v):
    if v is None:
        return float("nan")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _err_pct(measured, reference):
    m, r = _num(measured), _num(reference)
    if math.isnan(m) or math.isnan(r) or r == 0.0:
        return float("nan")
    return 100.0 * (m / r - 1.0)


def summarise(res: dict, I_target: float) -> dict:
    """One steady-state row.  No simulation, no reference fit beyond ``ref_*``."""
    series = res.get("series") or []
    mu_wall_all = [s["mu"] for s in series if s.get("mu") is not None]
    n_tail = max(1, int(0.25 * len(mu_wall_all))) if mu_wall_all else 0
    tail = mu_wall_all[-n_tail:] if n_tail else []
    mu_wall = _mean(tail)
    mu_std_rel_pct = (100.0 * _std(tail) / abs(mu_wall)
                      if tail and mu_wall else float("nan"))

    i_imposed = _num(res.get("inertial_number_imposed_rate"))
    if math.isnan(i_imposed):
        i_imposed = _num(res.get("inertial_number"))
    i_measured = _num(res.get("inertial_number_measured_rate_final"))
    if math.isnan(i_measured):
        i_measured = _num(res.get("inertial_number_measured_rate"))

    abscissa_best = i_measured if not math.isnan(i_measured) else i_imposed
    rec = {
        "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "round": ROUND,
        "I_target": float(I_target),
        "I_imposed_rate": i_imposed,
        "I_measured_rate": i_measured,
        "I_abscissa_best": abscissa_best,
        "mu_wall": mu_wall,
        "mu_full": _num(res.get("full_mu_final")),
        "mu_bulk": _num(res.get("bulk_mu_final")),
        "mu_std_rel_pct": mu_std_rel_pct,
        "mu_n_tail_samples": len(tail),
        "phi_bed_steady": _num(res.get("phi_bed_steady")),
        "phi_extent_final": _num(res.get("phi_extent_final")),
        "phi_container_final": _num(res.get("phi_container_final")),
        "p_over_zz": _num(res.get("full_anisotropy")),
        "pressure_control_error_pct": (
            _err_pct(res.get("normal_stress_measured_pa"),
                     res.get("normal_stress_applied_pa"))
            if res.get("normal_stress_applied_pa") else float("nan")),
        "shear_rate_ratio": _num(res.get("shear_rate_ratio")),
        "shear_strain_achieved": _num(res.get("shear_strain_achieved")),
        "n_particles": _num(res.get("n_particles")),
        "coordination": _num(res.get("coordination_number")),
        "servo_valid": res.get("servo_valid"),
        "dt": _num(res.get("dt")),
        "settle_steps": _num(res.get("settle_steps")),
        "shear_steps": _num(res.get("shear_steps")),
    }
    for arm in MU_ARMS:
        for ab in ABSCISSAE:
            I_ab = rec[ab]
            rec[f"mu_ref__{arm}__{ab}"] = ref_mu(I_ab) if I_ab else float("nan")
            rec[f"err_pct__{arm}__{ab}"] = _err_pct(
                rec[arm], rec[f"mu_ref__{arm}__{ab}"])
    # reference curves at the target (protocol column, not a verdict column)
    rec["mu_ref_at_target"] = ref_mu(I_target)
    rec["phi_ref_at_target"] = ref_phi(I_target)
    rec["err_mu_pct_at_target"] = _err_pct(mu_wall, rec["mu_ref_at_target"])
    rec["err_phi_pct_at_target"] = _err_pct(rec["phi_bed_steady"],
                                            rec["phi_ref_at_target"])
    return rec


def run_point(I: float, *, shear_strain: float = SHEAR_STRAIN) -> dict:
    case = ServoShearCase(inertial_number=I, mu=MU_CONTACT,
                          mu_rolling=MU_ROLLING, restitution=E, kn=KN,
                          normal_stress=P, shear_strain=shear_strain,
                          settle_steps=SETTLE_STEPS,
                          min_shear_steps=1,
                          sample_stride=SAMPLE_STRIDE,
                          packing_gap=0.02, jitter=0.01, seed=SEED,
                          radius=R, density=RHO, box_width=W,
                          box_depth=D, fill_height=H)
    t0 = time.perf_counter()
    res = simulate_servo_shear(case, return_series=True)
    wall = time.perf_counter() - t0
    rec = summarise(res, I)
    rec["wall_clock_s"] = wall
    rec["shear_strain_target"] = shear_strain
    return rec


# --------------------------------------------------------------------------
# verdict assembly
# --------------------------------------------------------------------------
def _rows(runs_dir) -> list[dict]:
    rows = [json.loads(f.read_text()) for f in sorted(Path(runs_dir).glob("point_*.json"))]
    rows.sort(key=lambda r: r["I_target"])
    return rows


def _arm_verdict(rows, arm, abscissa, tol_pct=TOL_PCT):
    """Error table + reference-envelope verdict for one (estimator, abscissa)."""
    pts = [(r[abscissa], r[arm]) for r in rows
           if not math.isnan(_num(r.get(abscissa)))
           and not math.isnan(_num(r.get(arm))) and r[abscissa] > 0]
    I = [p[0] for p in pts]
    mu = [p[1] for p in pts]
    per_triple = verdict_table(I, mu, tol=tol_pct / 100.0) if pts else []
    oracle = (best_case_over_envelope(I, mu, tol=tol_pct / 100.0) if len(pts) >= 2
              else {"provenance_can_rescue": None,
                    "min_max_abs_rel_err": float("nan")})
    errs = [100.0 * (m / ref_mu(i) - 1.0) for i, m in pts]
    return {
        "arm": arm,
        "abscissa": abscissa,
        "n_points": len(pts),
        "I_abscissa": I,
        "mu_measured": mu,
        "err_pct_vs_primary_repo": errs,
        "worst_abs_err_pct": max((abs(e) for e in errs), default=float("nan")),
        "within_band_all_points": bool(errs) and all(abs(e) < tol_pct for e in errs),
        "per_triple": per_triple,
        "envelope_oracle": oracle,
        "tolerance_pct": tol_pct,
    }


def assemble(runs_dir=None) -> dict:
    runs = Path(runs_dir) if runs_dir is not None else RUNS
    rows = _rows(runs)
    if not rows:
        raise SystemExit(f"no point_*.json in {runs}")

    verdicts = [_arm_verdict(rows, arm, ab) for arm in MU_ARMS for ab in ABSCISSAE]
    band = [v for v in verdicts if v["n_points"] == len(rows)]
    best = min(band, key=lambda v: v["worst_abs_err_pct"]) if band else None

    # phi: one estimator spread column + the reference is WITHHELD (provenance)
    phi_cols = ("phi_bed_steady", "phi_extent_final", "phi_container_final")
    phi_rows = []
    for r in rows:
        vals = [_num(r[c]) for c in phi_cols]
        finite = [v for v in vals if not math.isnan(v)]
        phi_rows.append({
            "I_target": r["I_target"],
            "I_measured_rate": r["I_measured_rate"],
            "estimators": dict(zip(phi_cols, vals)),
            "spread_pct": (100.0 * (max(finite) - min(finite)) / _mean(finite)
                           if len(finite) == 3 and _mean(finite) else float("nan")),
            "phi_ref_at_target": r.get("phi_ref_at_target"),
            "err_pct_vs_ref_at_target": r.get("err_phi_pct_at_target"),
        })

    range_complete = (abs(rows[0]["I_target"] - I_TARGETS[0]) < 1e-12
                      and abs(rows[-1]["I_target"] - I_TARGETS[-1]) < 1e-12)
    verdict = {
        "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
        "round": ROUND,
        "reference": {
            "mu": {"constants": {"mu_s": MU_S, "mu_2": MU_2, "I_0": I_0},
                   "status": MU_REF_STATUS},
            "phi": {"constants": {"phi_max": PHI_MAX, "phi_min": PHI_MIN},
                    "status": PHI_REF_STATUS},
            "candidates": [c.name for c in CANDIDATES],
        },
        "tolerance_pct": TOL_PCT,
        "range_covered": [rows[0]["I_target"], rows[-1]["I_target"]],
        "ledger_range": [I_TARGETS[0], I_TARGETS[-1]],
        "range_complete": range_complete,
        "n_particles": [r["n_particles"] for r in rows],
        "min_n_particles": min(_num(r["n_particles"]) for r in rows),
        "servo_valid_all": all(r.get("servo_valid") in (True, None) for r in rows),
        "mu_verdicts": verdicts,
        "mu_best_arm": ({"arm": best["arm"], "abscissa": best["abscissa"],
                         "worst_abs_err_pct": best["worst_abs_err_pct"]}
                        if best else None),
        # a band-pass on a *partial* range is not a full-range pass: the flag
        # is gated on the coverage being complete (round-20 Q1) so a dropped
        # shard cannot masquerade as an acceptance.
        "mu_passes_full_range": bool(
            range_complete and any(v["within_band_all_points"] and
                                   v["n_points"] == len(rows) for v in verdicts)),
        "envelope_can_rescue": bool(
            any(v["envelope_oracle"].get("provenance_can_rescue") for v in verdicts)),
        "phi": {"reference_status": "PROVENANCE-UNVERIFIED -> WITHHELD",
                "rows": phi_rows},
        "rows": rows,
        "prereg": PREREG,
    }
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "result.json").write_text(json.dumps(verdict, indent=1))
    return verdict


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
def _f(v, spec=".4f"):
    v = _num(v)
    return "--" if math.isnan(v) else format(v, spec)


def tables_markdown(verdict: dict) -> str:
    rows = verdict["rows"]
    L = []
    L.append("# Round 20 - full-range higher-N sweep (I = 1e-3 ... 0.3)\n")
    L.append("All numbers generated by `high_n_coverage.py --assemble` from "
             "`runs/high_n_r20/point_*.json`; nothing hand-copied.\n")
    L.append("## Table 1 - measured state per point\n")
    L.append("| I target | N | I imposed | I measured | mu wall | mu full | mu bulk "
             "| mu noise % | q (rate ratio) | p/sigma_zz | servo | t (s) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            _f(r["I_target"], "g"), _f(r["n_particles"], ".0f"),
            _f(r["I_imposed_rate"]), _f(r["I_measured_rate"]),
            _f(r["mu_wall"]), _f(r["mu_full"]), _f(r["mu_bulk"]),
            _f(r["mu_std_rel_pct"], ".1f"), _f(r["shear_rate_ratio"]),
            _f(r["p_over_zz"]), r.get("servo_valid"),
            _f(r.get("wall_clock_s"), ".0f")))

    L.append("\n## Table 2 - mu verdict matrix (error vs PRIMARY_REPO triple)\n")
    L.append("| estimator | abscissa | worst abs err % | all points "
             "&lt; 10 % | oracle min max abs err % | oracle boundary hit | "
             "can rescue? |")
    L.append("|---|---|---|---|---|---|---|")
    for v in verdict["mu_verdicts"]:
        o = v["envelope_oracle"]
        hit = o.get("boundary_hit") or []
        L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            v["arm"], v["abscissa"], _f(v["worst_abs_err_pct"], ".1f"),
            "yes" if v["within_band_all_points"] else "no",
            _f(100.0 * o["min_max_abs_rel_err"], ".1f")
            if not math.isnan(_num(o.get("min_max_abs_rel_err"))) else "--",
            ",".join(hit) if hit else ("none" if hit == [] else "--"),
            o.get("provenance_can_rescue")))

    L.append("\n## Table 3 - per-point error of every (estimator, abscissa) cell\n")
    header = ["I target"] + [f"{v['arm']}/{v['abscissa'][2:6]}"
                             for v in verdict["mu_verdicts"]]
    L.append("| " + " | ".join(header) + " |")
    L.append("|" + "---|" * len(header))
    for k, r in enumerate(rows):
        cells = [_f(r[f"err_pct__{v['arm']}__{v['abscissa']}"], "+.1f")
                 for v in verdict["mu_verdicts"]]
        L.append("| " + " | ".join([_f(r["I_target"], "g")] + cells) + " |")

    L.append("\n## Table 4 - phi estimators and spread\n")
    L.append("| I target | phi bed | phi extent | phi container | spread % | "
             "phi ref (at target, status WITHELD) | err % |")
    L.append("|---|---|---|---|---|---|---|")
    for pr in verdict["phi"]["rows"]:
        e = pr["estimators"]
        L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            _f(pr["I_target"], "g"), _f(e["phi_bed_steady"]),
            _f(e["phi_extent_final"]), _f(e["phi_container_final"]),
            _f(pr["spread_pct"], ".1f"), _f(pr["phi_ref_at_target"]),
            _f(pr["err_pct_vs_ref_at_target"], "+.1f")))
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="i/n, i in [0,n)")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--strain", type=float, default=SHEAR_STRAIN)
    ap.add_argument("--runs-dir", default=None)
    args = ap.parse_args()

    runs = Path(args.runs_dir) if args.runs_dir else RUNS
    runs.mkdir(parents=True, exist_ok=True)

    if args.assemble:
        v = assemble(runs)
        (runs / "tables.md").write_text(tables_markdown(v))
        print(json.dumps({k: v[k] for k in
                          ("range_covered", "min_n_particles", "mu_best_arm",
                           "mu_passes_full_range", "envelope_can_rescue")}, indent=1))
        return

    i, n = (int(x) for x in args.shard.split("/"))
    if not 0 <= i < n:
        raise SystemExit("shard index out of range")
    assigned = [I for k, I in enumerate(I_TARGETS) if k % n == i]
    print(f"shard {i}/{n}: I targets {assigned}", flush=True)
    for I in assigned:
        rec = run_point(I, shear_strain=args.strain)
        rec["shard"] = f"{i}/{n}"
        (runs / f"point_{I:g}.json").write_text(json.dumps(rec, indent=1))
        print(f"I={I:g}: N={rec['n_particles']:.0f} "
              f"mu_wall={rec['mu_wall']:.4f} ({rec['err_pct__mu_wall__I_measured_rate']:+.1f}%) "
              f"mu_full={rec['mu_full']:.4f} phi={rec['phi_bed_steady']:.4f} "
              f"noise={rec['mu_std_rel_pct']:.1f}% ({rec['wall_clock_s']:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()