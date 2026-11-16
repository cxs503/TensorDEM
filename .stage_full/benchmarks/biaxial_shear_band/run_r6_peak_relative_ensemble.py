"""R6 of CASE-BIAXIAL-SHEAR-BAND: peak-relative stopping rule.

What R5 left open
-----------------
R5 (``docs/CASE_BIAXIAL_SHEAR_BAND_R5.md``) closed the *accuracy* half of the
case (ensemble band inclination 55.20 deg vs the 60 deg anchor = 8.0 % < 15 %)
but failed the *resolution-count* half of the predeclared rule: 5 of 8 anchor
specimens resolved the band (bar: 6 of 8).  The R5 diagnosis is on the record and
is mechanical, not physical: ``strain_target = 0.12`` was a **fixed** strain
budget, and 3 of 8 specimens were still hardening at the cut-off, so their
deviator peak sat at (or one snapshot before) the last snapshot and their
late post-peak window was empty (``strain_span = 0``); two more had a
single-increment window (``strain_span = 0.01``, slip 0.27 spacings) that only
just cleared the focus bar.  R5's recommendation was a *peak-relative* stopping
rule -- "run until ``eps_a >= peak + 0.03``" -- explicitly not done post hoc
because it changes the predeclared protocol.

What R6 does
------------
The rule is now implemented in the driver
(:func:`tensordem.biaxial_shear_band.run_biaxial_shear_band`,
``post_peak_strain`` / ``strain_cap``) and unit-tested
(``tests/test_biaxial_peak_relative_stopping.py``).  R6 reruns **the same eight
anchor specimens** with the rule on, so every specimen is a *paired* before/
after comparison: same seeds, same lattice, same contact law, same estimator.
Nothing else changes -- the protocol edit is the only difference from R4/R5.

Predeclared protocol (fixed before any R6 number was seen)
---------------------------------------------------------
* Cell and material: 24 x 48 jittered disc lattice, 35 % freeze bonds, wall
  friction 0.55, 4000 consolidation steps, strain rate 0.05 /s, snapshots every
  0.01 strain, p = 100 kPa servo walls.  Anchor seeds 53, 67, 79, 83, 89, 97,
  101, 103 -- the R4/R5 anchor set, deliberately *not* new seeds, because the
  question is whether the same specimens resolve under the new rule.
* Stopping: ``strain_target = 0.12`` is a **minimum**, ``post_peak_strain =
  0.03``, ``strain_cap = 0.18`` (= target + 2 rule widths, the driver default).
  A specimen therefore stops at ``max(0.12, peak + 0.03)`` and never later than
  0.18; the achieved ``stop_reason`` per specimen is reported.
* Primary estimator: cumulative Radon focus over the late post-peak window
  (identical operator, window definition and resolution rule as R4/R5:
  ``z > 3`` and ``eta2 > 0.2``).  Secondary estimators (post-peak window,
  structure tensor, one-step incremental) are reported side by side and are
  never used to rescue the verdict.
* PASS rule (unchanged from R4/R5, so the two rounds are comparable):
  ``|mean(theta) - 60| / 60 <= 0.15`` **and** at least 6 of 8 specimens resolved.
* P-R6 (the round's own prediction, falsifiable): the three specimens that had
  no post-peak window in R5 (s89, s97, s103) acquire a late window of at least
  ``0.02`` strain (>= 3 snapshots) under the rule, and the ensemble resolves at
  least 6 of 8 -- otherwise the failure is *not* fully explained by the fixed
  budget and the diagnosis is falsified.
* No specimen is excluded and no failure is re-labelled.  A specimen that still
  has no post-peak window after the rule is reported as such.

Outputs ``artifacts/biaxial_shear_band/r6_peak_relative_ensemble.json``, the
per-specimen cache ``r6_cache/<name>.json`` and the paired comparison against
the R5 baseline (``r5_baseline.json``, the R4 artefact R5 analysed, identified
by sha256 in the output JSON).

Run: ``PYTHONPATH=src python benchmarks/biaxial_shear_band/run_r6_peak_relative_ensemble.py
--workers 8``; one shard: ``python benchmarks/biaxial_shear_band/run_r6_shard.py <i>``.
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

ROUND = "R6"
POST_PEAK_STRAIN = 0.03
STRAIN_TARGET = 0.12
STRAIN_CAP = STRAIN_TARGET + 2.0 * POST_PEAK_STRAIN       # 0.18
RESOLVED_BAR = 6
P_R6_MIN_SPAN = 0.02
#: specimens that had no late post-peak window at all in R4/R5 (strain_span = 0)
R5_DEAD_WINDOW = ["anchor_s89", "anchor_s97", "anchor_s103"]

ART = os.path.join(ROOT, "artifacts", "biaxial_shear_band")
CACHE = os.path.join(ART, "r6_cache")
OUT = os.path.join(ART, "r6_peak_relative_ensemble.json")
BASELINE = os.path.join(ART, "r5_baseline.json")


def _configs(quick: bool, only_seed: int | None) -> list[dict]:
    seeds = [only_seed] if only_seed is not None else list(ANCHOR_SEEDS)
    return [{
        "name": f"anchor_s{s}", "axis": "anchor", "seed": int(s),
        "confining_stress": 100.0e3, "quick": bool(quick),
        "post_peak_strain": POST_PEAK_STRAIN, "strain_cap": STRAIN_CAP,
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
    except Exception as exc:                                 # pragma: no cover - reported
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
    """R5 baseline (the R4 artefact R5 analysed), if it was staged."""
    if not os.path.exists(BASELINE):
        return {}
    with open(BASELINE) as handle:
        return json.load(handle)


def _paired(results: list[dict], base: dict) -> dict:
    """Per-specimen R5 -> R6 comparison (same seed, same estimator)."""
    by_name = {r["name"]: r for r in base.get("specimens", [])}

    def cum(rec, label, key):
        return ((rec or {}).get("cumulative") or {}).get(label, {}).get(key)

    rows = []
    for rec in sorted(results, key=lambda r: r["seed"]):
        if "error" in rec:
            rows.append({"name": rec["name"], "seed": rec["seed"],
                         "error": rec["error"]})
            continue
        old = by_name.get(rec["name"])
        rows.append({
            "name": rec["name"], "seed": rec["seed"],
            "r5_theta_deg": cum(old, "late_window", "radon"),
            "r6_theta_deg": cum(rec, "late_window", "radon"),
            "r5_span": cum(old, "late_window", "strain_span"),
            "r6_span": cum(rec, "late_window", "strain_span"),
            "r5_n_late_snapshots": (old or {}).get("n_late_snapshots"),
            "r6_n_late_snapshots": rec.get("n_late_snapshots"),
            "r5_resolved": bool(cum(old, "late_window", "resolved")),
            "r6_resolved": bool(cum(rec, "late_window", "resolved")),
            "r5_final_strain": (old or {}).get("final_axial_strain"),
            "r6_final_strain_log": (rec.get("stopping") or {}).get("final_axial_strain_log"),
            "r6_stop_reason": (rec.get("stopping") or {}).get("stop_reason"),
        })
    flipped = [r for r in rows if r.get("r6_resolved") and not r.get("r5_resolved")]
    lost = [r for r in rows if r.get("r5_resolved") and not r.get("r6_resolved")]
    dead_now = [r for r in rows if r["name"] in R5_DEAD_WINDOW and "error" not in r]
    prediction = {
        "dead_window_specimens": R5_DEAD_WINDOW,
        "min_span_required": P_R6_MIN_SPAN,
        "acquired_window": [
            {"name": r["name"], "r6_span": r["r6_span"],
             "r6_n_late_snapshots": r["r6_n_late_snapshots"],
             "acquired": bool((r["r6_span"] or 0.0) >= P_R6_MIN_SPAN)}
            for r in dead_now],
        "n_acquired": int(sum(1 for r in dead_now
                              if (r["r6_span"] or 0.0) >= P_R6_MIN_SPAN)),
        "prediction_met": bool(dead_now and all(
            (r["r6_span"] or 0.0) >= P_R6_MIN_SPAN for r in dead_now)),
    }
    return {"per_specimen": rows,
            "n_newly_resolved": len(flipped),
            "newly_resolved": [r["name"] for r in flipped],
            "n_lost_resolution": len(lost),
            "lost_resolution": [r["name"] for r in lost],
            "prediction_P_R6": prediction,
            "baseline_present": bool(base)}


def _stopping_summary(results: list[dict]) -> dict:
    reasons: dict[str, list[str]] = {}
    finals = []
    for rec in results:
        if "error" in rec:
            continue
        st = rec.get("stopping") or {}
        reasons.setdefault(str(st.get("stop_reason")), []).append(rec["name"])
        if st.get("final_axial_strain_log") is not None:
            finals.append(float(st["final_axial_strain_log"]))
    arr = np.asarray(finals, dtype=float) if finals else np.asarray([])
    return {"stop_reasons": {k: {"n": len(v), "specimens": v}
                             for k, v in sorted(reasons.items())},
            "final_strain_min": float(arr.min()) if arr.size else None,
            "final_strain_max": float(arr.max()) if arr.size else None,
            "final_strain_mean": float(arr.mean()) if arr.size else None}


def analyse(results: list[dict]) -> dict:
    stage1 = _analyse(results)
    base = _baseline()
    return {
        "stage1": stage1,
        "stopping": _stopping_summary(results),
        "paired_vs_r5": _paired(results, base),
        "pass_decision": {
            "rule": "|mean-60|/60 <= 0.15 and >= 6/8 resolved (unchanged from R4/R5)",
            "accuracy_half": bool(stage1["primary_cumulative_late_window_radon"]
                                  .get("within_tolerance")),
            "resolution_half": bool(stage1["specimens_resolved"] >= RESOLVED_BAR),
            "passed": bool(stage1.get("passed")),
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
            "stopping_rule": "stop at max(strain_target, peak + post_peak_strain), "
                             "capped at strain_cap; peak tracked online on the "
                             "0.01-strain trailing mean of the deviator",
            "anchor_seeds": list(ANCHOR_SEEDS),
            "anchor": ("Desrues & Viggiani 2004 IJNAMG 28:279-321, dense Hostun "
                       "sand plane-strain biaxial, shear band 55-60 deg "
                       "(tolerance anchor = 60 deg)"),
            "tolerance": TOLERANCE,
            "estimator": "cumulative Radon focus over the late post-peak window "
                         "(identical to R4/R5)",
            "r5_baseline": {"path": os.path.relpath(BASELINE, ROOT),
                            "sha256": _sha256(BASELINE)},
        },
        "specimens": results,
        "wall_time_s": time.time() - started,
    }
    payload.update(analyse(results))
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=1, default=float)

    pd = payload["pass_decision"]
    print(f"[R6] resolved {payload['stage1']['specimens_resolved']}/8  "
          f"theta={payload['stage1']['primary_cumulative_late_window_radon'].get('mean_deg')}"
          f"  accuracy={pd['accuracy_half']}  resolution={pd['resolution_half']}"
          f"  passed={pd['passed']}", flush=True)
    print(f"[R6] P-R6 acquired {payload['paired_vs_r5']['prediction_P_R6']['n_acquired']}/3"
          f"  met={payload['paired_vs_r5']['prediction_P_R6']['prediction_met']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())