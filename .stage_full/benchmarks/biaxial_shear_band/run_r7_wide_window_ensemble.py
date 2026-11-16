"""R7 of CASE-BIAXIAL-SHEAR-BAND: a rule and a window declared together.

Where R6 stopped
----------------
R6 (``docs/CASE_BIAXIAL_SHEAR_BAND_R6.md``) delivered the case's PASS
(ensemble 55.00 deg vs the 60 deg anchor = 8.33 % < 15 %, 6/8 specimens
resolved) but its own prediction P-R6 could not be evaluated, and R6 §⑦
diagnosed why: the late window keeps the last 60 % of the post-peak record while
the rule is only ``post_peak_strain = 0.03`` wide, so the largest window span a
perfectly working rule can deliver is ``0.6 * 0.03 = 0.018`` — below the
``0.02`` span that P-R6 demanded.  The prediction was therefore
*arithmetically unsatisfiable*, carried no physical information, and the two
remaining degenerate specimens (s89, s97) were left unexplained.

What R7 does
------------
1. Makes the window width a first-class driver argument
   (``late_window_fraction``, ``tensordem.biaxial_shear_band``), whose default
   (0.6) reproduces the R4/R5/R6 window bit-identically, and reports the
   protocol's span ceiling ``late_window_span_cap = f * w`` in
   ``result["stopping"]["late_window_span_cap_log"]``.
2. Declares rule and window **together, before any R7 number exists**:
   ``post_peak_strain = 0.05`` (R6's recommendation) with
   ``late_window_fraction = 0.8``, so the span ceiling is
   ``0.8 * 0.05 = 0.04`` — twice the 0.02 required for three 0.01-strain
   snapshots.  The runner *refuses to start* if the declared minimum span
   exceeds the ceiling (``_protocol_gate``), so the R6 defect cannot recur
   silently: it is now a startup assertion, not a post-hoc note.
3. Reruns the same eight anchor seeds with the new protocol (paired with R6,
   same lattice, contact law, estimator and resolution rule), predicts that the
   three specimens R5/R6 could not resolve (s89, s97, s103) each acquire a late
   window of at least three snapshots, and reports the ensemble verdict on the
   unchanged PASS rule.

Predeclared protocol (fixed before any R7 number was seen)
----------------------------------------------------------
* Cell and material: identical to R4/R5/R6 — 24 x 48 jittered disc lattice,
  35 % freeze bonds, wall friction 0.55, 4000 consolidation steps, strain rate
  0.05 /s, snapshots every 0.01 strain, p = 100 kPa servo walls, anchor seeds
  53, 67, 79, 83, 89, 97, 101, 103.
* Stopping: ``strain_target = 0.12`` (minimum), ``post_peak_strain = 0.05``,
  ``strain_cap = 0.22`` (= target + 2 rule widths), so a specimen stops at
  ``max(0.12, peak + 0.05)`` and never later than 0.22.
* Late window: last 80 % of the post-peak record (``late_window_fraction =
  0.8``), hence span ceiling 0.04 and, at 0.01 snapshot spacing, at least four
  snapshots for a specimen that stops on the rule.
* Primary estimator / resolution rule / secondary estimators: unchanged from
  R4/R5/R6 (cumulative Radon focus over the late window, ``z > 3`` and
  ``eta2 > 0.2``; ``resolved`` = >= 3 snapshots agreeing within 10 deg).
* PASS rule (unchanged): ``|mean(theta) - 60| / 60 <= 0.15`` **and** at least
  6 of 8 specimens resolved.
* P-R7 (this round's falsifiable prediction): ``anchor_s89``, ``anchor_s97``
  and ``anchor_s103`` each deliver ``>= 3`` late-window snapshots, and the
  ensemble resolves ``>= 7/8``.  If they still do not, the "fixed budget /
  too-narrow window" explanation of their degeneracy is falsified and the next
  round must audit the estimator or the material, not the stopping rule.
* No specimen is excluded, no failure re-labelled, ``psi``-based laws are still
  not used for the verdict (the global dilation angle of a bonded specimen is
  unphysical, R2/R5): only the ``phi``-based Mohr-Coulomb angle is reported
  next to the measured band angle, as a diagnostic.

Outputs ``artifacts/biaxial_shear_band/r7_wide_window_ensemble.json``, the
per-specimen cache ``r7_cache/<name>.json`` and the paired comparison against
R6 (``r6_baseline.json``, identified by sha256 in the output JSON).

Run: ``R4_THREADS=6 python benchmarks/biaxial_shear_band/run_r7_shard.py <i>``
for ``i in 0..7`` and then the assembler
``python benchmarks/biaxial_shear_band/run_r7_wide_window_ensemble.py --resume``
(``--workers N`` runs the whole ensemble in one process pool instead).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_SRC = os.path.join(ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from benchmarks.biaxial_shear_band.run_r4_cumulative_ensemble import (  # noqa: E402
    ANCHOR_BAND_DEG,
    ANCHOR_SEEDS,
    TOLERANCE,
    THETA_REF_DEG,
    _analyse,
    _run_one,
)

ROUND = "R7"
POST_PEAK_STRAIN = 0.05
STRAIN_TARGET = 0.12
STRAIN_CAP = STRAIN_TARGET + 2.0 * POST_PEAK_STRAIN           # 0.22
LATE_WINDOW_FRACTION = 0.8
MIN_LATE_SPAN = 0.02                     # 3 snapshots at 0.01 strain spacing
MIN_LATE_SNAPSHOTS = 3
RESOLVED_BAR = 6
P_R7_MIN_RESOLVED = 7
#: specimens unresolved (or single-increment) in R5 and R6
R5_R6_DEGENERATE = ["anchor_s89", "anchor_s97", "anchor_s103"]

ART = os.path.join(ROOT, "artifacts", "biaxial_shear_band")
CACHE = os.path.join(ART, "r7_cache")
OUT = os.path.join(ART, "r7_wide_window_ensemble.json")
BASELINE = os.path.join(ART, "r6_baseline.json")


# ---------------------------------------------------------------------------
# protocol gate (the R6 §⑦ defect, mechanised)
# ---------------------------------------------------------------------------
def protocol_span_ceiling(late_window_fraction: float = LATE_WINDOW_FRACTION,
                          post_peak_strain: float = POST_PEAK_STRAIN) -> float:
    """Largest late-window span this protocol can deliver (log strain)."""
    return float(late_window_fraction) * float(post_peak_strain)


def _protocol_gate(late_window_fraction: float = LATE_WINDOW_FRACTION,
                   post_peak_strain: float = POST_PEAK_STRAIN,
                   min_span: float = MIN_LATE_SPAN) -> dict:
    """Refuse a protocol whose declared window cannot reach its own minimum span.

    R6 predeclared ``min_span = 0.02`` with ``0.6 * 0.03 = 0.018`` available, so
    its prediction was unsatisfiable by construction.  Here the same arithmetic
    is a startup assertion: a protocol that cannot deliver the span it promises
    never produces numbers.
    """
    ceiling = protocol_span_ceiling(late_window_fraction, post_peak_strain)
    ok = ceiling >= float(min_span)
    if not ok:
        raise SystemExit(
            f"[R7] protocol gate: late window keeps {late_window_fraction:.2f} of the "
            f"post-peak record and the rule is {post_peak_strain:.3f} wide, so the "
            f"span ceiling is {ceiling:.4f} < declared minimum {min_span:.4f}; "
            "the round would be unsatisfiable by construction (R6 defect)")
    return {"late_window_fraction": float(late_window_fraction),
            "post_peak_strain": float(post_peak_strain),
            "span_ceiling_log": ceiling,
            "min_required_span_log": float(min_span),
            "implied_min_snapshots": int(math.floor(ceiling / 0.01 + 1e-9)) + 1,
            "satisfiable": True}


# ---------------------------------------------------------------------------
# ensemble plumbing
# ---------------------------------------------------------------------------
def _configs(quick: bool, only_seed: int | None) -> list[dict]:
    seeds = [only_seed] if only_seed is not None else list(ANCHOR_SEEDS)
    return [{
        "name": f"anchor_s{s}", "axis": "anchor", "seed": int(s),
        "confining_stress": 100.0e3, "quick": bool(quick),
        "post_peak_strain": POST_PEAK_STRAIN, "strain_cap": STRAIN_CAP,
        "late_window_fraction": LATE_WINDOW_FRACTION,
    } for s in seeds]


def _run_cached(config: dict, cache_dir: str, resume: bool) -> dict:
    os.makedirs(cache_dir, exist_ok=True)
    suffix = "_quick" if config.get("quick") else ""
    path = os.path.join(cache_dir, config["name"] + suffix + ".json")
    if resume and os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    result = _run_one(config)
    with open(path, "w") as handle:
        json.dump(result, handle, default=float)
    return result


def _run_cached_safe(config: dict, cache_dir: str, resume: bool) -> dict:
    try:
        return _run_cached(config, cache_dir, resume)
    except Exception as exc:                                  # pragma: no cover
        return {"name": config["name"], "axis": config["axis"],
                "seed": config["seed"],
                "confining_stress": config["confining_stress"],
                "error": repr(exc), "traceback": traceback.format_exc()}


def _sha256(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _baseline() -> dict:
    if not os.path.exists(BASELINE):
        return {}
    with open(BASELINE) as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def _cum(rec, label, key):
    return ((rec or {}).get("cumulative") or {}).get(label, {}).get(key)


def _paired(results: list[dict], base: dict) -> dict:
    """Per-specimen R6 -> R7 comparison (same seed, same estimator, wider rule)."""
    by_name = {r["name"]: r for r in base.get("specimens", [])}
    rows = []
    for rec in sorted(results, key=lambda r: r["seed"]):
        if "error" in rec:
            rows.append({"name": rec["name"], "seed": rec["seed"],
                         "error": rec["error"]})
            continue
        old = by_name.get(rec["name"])
        rows.append({
            "name": rec["name"], "seed": rec["seed"],
            "r6_theta_deg": _cum(old, "late_window", "radon"),
            "r7_theta_deg": _cum(rec, "late_window", "radon"),
            "r6_span": _cum(old, "late_window", "strain_span"),
            "r7_span": _cum(rec, "late_window", "strain_span"),
            "r6_n_late": (old or {}).get("n_late_snapshots"),
            "r7_n_late": rec.get("n_late_snapshots"),
            "r6_resolved": bool(_cum(old, "late_window", "resolved")),
            "r7_resolved": bool(_cum(rec, "late_window", "resolved")),
            "r6_final_strain_log": (old or {}).get("stopping", {}).get(
                "final_axial_strain_log"),
            "r7_final_strain_log": (rec.get("stopping") or {}).get(
                "final_axial_strain_log"),
            "r7_stop_reason": (rec.get("stopping") or {}).get("stop_reason"),
            "phi_deg": rec.get("friction_angle_deg"),
            "psi_global_deg": rec.get("dilation_angle_deg"),
            "theta_mohr_coulomb_deg": (rec.get("theory_angles_deg") or {}).get(
                "mohr_coulomb"),
        })
    flipped = [r for r in rows if r.get("r7_resolved") and not r.get("r6_resolved")]
    lost = [r for r in rows if r.get("r6_resolved") and not r.get("r7_resolved")]
    degenerate = [r for r in rows if r["name"] in R5_R6_DEGENERATE and "error" not in r]
    acquired = [r for r in degenerate if (r.get("r7_n_late") or 0) >= MIN_LATE_SNAPSHOTS]
    n_resolved = int(sum(1 for r in rows
                         if r.get("r7_resolved") and "error" not in r))
    prediction = {
        "specimens": R5_R6_DEGENERATE,
        "min_late_snapshots": MIN_LATE_SNAPSHOTS,
        "per_specimen": [{"name": r["name"], "r7_n_late": r.get("r7_n_late"),
                          "r7_span": r.get("r7_span"),
                          "acquired": bool((r.get("r7_n_late") or 0) >= MIN_LATE_SNAPSHOTS)}
                         for r in degenerate],
        "n_acquired": len(acquired),
        "min_resolved_required": P_R7_MIN_RESOLVED,
        "n_resolved": n_resolved,
        "prediction_met": bool(len(acquired) == len(degenerate)
                               and n_resolved >= P_R7_MIN_RESOLVED),
    }
    return {"per_specimen": rows,
            "n_newly_resolved": len(flipped),
            "newly_resolved": [r["name"] for r in flipped],
            "n_lost_resolution": len(lost),
            "lost_resolution": [r["name"] for r in lost],
            "prediction_P_R7": prediction,
            "baseline_present": bool(base)}


def _stopping_summary(results: list[dict]) -> dict:
    reasons: dict[str, list[str]] = {}
    finals, caps = [], []
    for rec in results:
        if "error" in rec:
            continue
        st = rec.get("stopping") or {}
        reasons.setdefault(str(st.get("stop_reason")), []).append(rec["name"])
        if st.get("final_axial_strain_log") is not None:
            finals.append(float(st["final_axial_strain_log"]))
        if st.get("late_window_span_cap_log") is not None:
            caps.append(float(st["late_window_span_cap_log"]))
    arr = np.asarray(finals, dtype=float) if finals else np.asarray([])
    return {"stop_reasons": {k: {"n": len(v), "specimens": sorted(v)}
                             for k, v in sorted(reasons.items())},
            "final_strain_min": float(arr.min()) if arr.size else None,
            "final_strain_max": float(arr.max()) if arr.size else None,
            "final_strain_mean": float(arr.mean()) if arr.size else None,
            "late_window_span_cap_log": float(caps[0]) if caps else None,
            "late_window_span_cap_consistent": bool(
                not caps or max(caps) - min(caps) < 1e-12)}


def analyse(results: list[dict]) -> dict:
    stage1 = _analyse(results)
    base = _baseline()
    paired = _paired(results, base)
    accuracy = bool(stage1["primary_cumulative_late_window_radon"].get(
        "within_tolerance"))
    resolved = int(stage1["specimens_resolved"])
    return {
        "stage1": stage1,
        "stopping": _stopping_summary(results),
        "protocol": _protocol_gate(),
        "paired_vs_r6": paired,
        "pass_decision": {
            "rule": "|mean-60|/60 <= 0.15 and >= 6/8 resolved (unchanged from R4/R5/R6)",
            "accuracy_half": accuracy,
            "resolution_half": bool(resolved >= RESOLVED_BAR),
            "passed": bool(accuracy and resolved >= RESOLVED_BAR),
            "resolution_target_8_of_8": bool(resolved == len(results)),
            "n_resolved": resolved,
            "n_total": len(results),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--cache", default=CACHE)
    parser.add_argument("--only-seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    gate = _protocol_gate()
    print(f"[R7] protocol gate ok: span ceiling {gate['span_ceiling_log']:.3f} "
          f">= min {gate['min_required_span_log']:.3f} "
          f"(>= {gate['implied_min_snapshots']} snapshots)", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    started = time.time()
    configs = _configs(False, args.only_seed)
    worker = functools.partial(_run_cached_safe, cache_dir=args.cache,
                               resume=args.resume)
    results: list[dict] = []
    if args.workers > 1 and len(configs) > 1:
        with mp.get_context("spawn").Pool(min(args.workers, len(configs))) as pool:
            for result in pool.imap_unordered(worker, configs):
                results.append(result)
                print(f"[done] {result['name']} {result.get('error', '')}", flush=True)
    else:
        for config in configs:
            results.append(worker(config))
            print(f"[done] {results[-1]['name']} {results[-1].get('error', '')}",
                  flush=True)
    results.sort(key=lambda r: r["seed"])

    payload = {
        "case": "CASE-BIAXIAL-SHEAR-BAND",
        "round": ROUND,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "specification": {
            "cell": "plane-strain biaxial, 24 x 48 disc lattice",
            "confining_stress_Pa": 100.0e3,
            "freeze_bond_fraction": 0.35, "wall_friction": 0.55,
            "consolidation_steps": 4000, "strain_rate_per_s": 0.05,
            "strain_target_min": STRAIN_TARGET,
            "post_peak_strain": POST_PEAK_STRAIN,
            "strain_cap": STRAIN_CAP,
            "late_window_fraction": LATE_WINDOW_FRACTION,
            "stopping_rule": "stop at max(strain_target, peak + post_peak_strain), "
                             "capped at strain_cap; peak tracked online on the "
                             "0.01-strain trailing mean of the deviator",
            "anchor_seeds": list(ANCHOR_SEEDS),
            "anchor": ("Desrues & Viggiani 2004 IJNAMG 28:279-321, dense Hostun "
                       "sand plane-strain biaxial, shear band 55-60 deg "
                       "(tolerance anchor = 60 deg)"),
            "tolerance": TOLERANCE,
            "estimator": "cumulative Radon focus over the late post-peak window "
                         "(identical to R4/R5/R6)",
            "r6_baseline": {"path": os.path.relpath(BASELINE, ROOT),
                            "sha256": _sha256(BASELINE)},
        },
        "specimens": results,
        "wall_time_s": time.time() - started,
    }
    payload.update(analyse(results))
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=1, default=float)

    pd = payload["pass_decision"]
    pr = payload["paired_vs_r6"]["prediction_P_R7"]
    print(f"[R7] resolved {pd['n_resolved']}/{pd['n_total']}  "
          f"theta={payload['stage1']['primary_cumulative_late_window_radon'].get('mean_deg')}"
          f"  accuracy={pd['accuracy_half']}  resolution={pd['resolution_half']}"
          f"  passed={pd['passed']}")
    print(f"[R7] P-R7 acquired {pr['n_acquired']}/{len(pr['specimens'])} "
          f"resolved {pr['n_resolved']} (bar {pr['min_resolved_required']}) "
          f"met={pr['prediction_met']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())