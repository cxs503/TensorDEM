"""R9 of CASE-BIAXIAL-SHEAR-BAND: the classical band-angle laws, tested.

Where the case stands
---------------------
R7 (``docs/CASE_BIAXIAL_SHEAR_BAND_R7.md``) delivered the case's published
verdict on the borrowed anchor: ensemble ``theta = 55.00 +- 2.07 deg`` over
eight specimens, 8/8 resolved, 8.33 % from the 60 deg tolerance anchor.  Its
adversarial review (``docs/ADVERSARIAL_REVIEW_BIAXIAL_SHEAR_BAND_R7.md`` §4.2)
left two objections that this round turns into a test:

* "Psi-based laws withheld: global dilation angle of bonded specimen is
  unphysical ... Roscoe/Arthur laws not used for verdict";
* "Anchor sensitivity: 60 deg is the midpoint of 55-60 deg literature band".

Both are the same gap: the case never compared its band angle with a prediction
built from *its own* measured state.  The three laws are already implemented
(``biaxial_shear_band.theory_angles_deg``) and every specimen result carries
``theory`` — but the numbers were never adjudicated, and the dilation angle that
the ``psi`` laws need is saturated in the whole-specimen record.

What R9 does
------------
1. Adjudicates the three laws on the **published R7 specimens** (committed
   evidence, no rerun): ``theta`` from the cumulative Radon late-window
   estimate, ``phi`` from the deviator peak, all extracted from
   ``artifacts/biaxial_shear_band/r7_cache/*.json`` at commit ``a9f3ab91`` with
   per-file sha256 recorded in the output.
2. **Quantifies the saturation** that made the ``psi`` laws look unusable: the
   published ``dilation_angle_deg`` is ``asin(clip(-d eps_v / d eps_a))`` with
   the clip at 0.95, so a specimen whose fit saturates reports exactly
   ``asin(0.95) = 71.81 deg``.  Those specimens are flagged, and the implied
   slope is reported next to the angle so a clamp can never be read as a
   measurement.
3. Supplies a **physical dilation arm** through stress-dilatancy
   (``band_angle_laws.stress_dilatancy_psi_deg``): ``psi`` from the measured
   ``phi`` and a *declared* critical-state friction range ``phi_cv`` — so the
   ``psi``-based laws are tested instead of withheld, and the result is reported
   for every declared ``phi_cv`` (no tuned value).
4. Re-derives the anchor's own material assumption
   (``anchor_implied_angles``): 60 deg is Mohr-Coulomb at ``phi = 30 deg``,
   while these specimens measure ``phi ~ 35.5 deg`` — the anchor is a fixed
   assumption of the source paper, not a prediction for these specimens.
5. Optionally adds a **state-matched** arm from fresh runs of the R7 protocol
   (``--shard i`` then ``--assemble``): ``phi`` rebuilt over the *same* late
   window the band angle is measured in.  Only that arm needs the DEM cell.

Predeclared decision rule (fixed before any R9 number exists)
-------------------------------------------------------------
* Arms: ``measured_psi`` (published ψ, the arm R2-R7 withheld),
  ``stress_dilatancy[phi_cv]`` for ``phi_cv in {26, 28, 30, 32, 34} deg``
  (declared 2D-disc critical-state range), ``peak`` (peak φ and ψ — what R2-R7
  reported next to the angle).
* A law is **tolerance-consistent** for a specimen iff
  ``|theta_meas - theta_law| / theta_law <= 0.15`` (the case's acceptance) and
  **precision-consistent** iff ``<= 0.03`` (the ecosystem's bar).
* A law is **supported by an arm for the ensemble** iff (a) its mean absolute
  relative error ``<= 0.15``, (b) it is the closest law for at least 6 of 8
  specimens, and (c) the mean distance between its prediction and the nearest
  competing law is at least 2 deg (otherwise the comparison does not
  discriminate and the verdict says so).
* ``phi_cv`` is never fitted: the arm is reported per declared value, and an
  arm only counts as support if it clears the bar at **every** declared value.
* No specimen is dropped; unresolved specimens keep their numbers and are
  flagged.

Outputs ``artifacts/biaxial_shear_band/r9_law_test/law_test.json`` plus
``per_specimen.json`` (all numbers program-injected, no hand transcription).

Run (published evidence, no simulation)::

    python benchmarks/biaxial_shear_band/run_r9_law_test.py \
        --from-cache-dir <dir with r7_cache/anchor_s*.json> --assemble

Run (state-matched arm, needs the cell)::

    R4_THREADS=8 python benchmarks/biaxial_shear_band/run_r9_law_test.py --shard i   # i = 0..7
    python benchmarks/biaxial_shear_band/run_r9_law_test.py --assemble
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tensordem.band_angle_laws import (  # noqa: E402
    LAWS,
    adjudicate,
    anchor_implied_angles,
    late_window_band_angle,
    late_window_state,
    stress_dilatancy_psi_deg,
)

ROUND = "R9"
ART = os.path.join(ROOT, "artifacts", "biaxial_shear_band", "r9_law_test")
CACHE = os.path.join(ART, "r9_cache")
OUT = os.path.join(ART, "law_test.json")
PER_SPECIMEN = os.path.join(ART, "per_specimen.json")

TOLERANCE = 0.15
PRECISION = 0.03
#: declared critical-state friction range for the stress-dilatancy arm (deg)
PHI_CV_RANGE = (26.0, 28.0, 30.0, 32.0, 34.0)
SUPPORT_MIN_FRACTION = 6
MIN_DISCRIMINATION_DEG = 2.0
ANCHOR_DEG = 60.0
#: clip of ``biaxial_shear_band``'s dilation fit: |sin psi| <= 0.95
DILATION_CLIP = 0.95


def _saturation(psi_deg: float) -> dict:
    """Is a reported dilation angle actually the clip of the fit?"""
    sin_psi = abs(math.sin(math.radians(float(psi_deg))))
    return {"sin_psi": float(sin_psi),
            "saturated": bool(sin_psi >= DILATION_CLIP - 1e-6),
            "implied_d_eps_v_d_eps_a": float(-sin_psi)}


def _sd_arm(theta: float, phi: float) -> dict:
    return {str(phi_cv): {"psi_deg": stress_dilatancy_psi_deg(phi, phi_cv),
                          **adjudicate(theta, phi,
                                       stress_dilatancy_psi_deg(phi, phi_cv),
                                       tolerance=TOLERANCE, precision=PRECISION)}
            for phi_cv in PHI_CV_RANGE}


def _analyse_specimen(result: dict) -> dict:
    """Per-specimen law comparison from a full result or a pruned R7 cache."""
    out: dict = {"name": result.get("name"), "seed": result.get("seed"),
                 "error": result.get("error")}
    if out["error"]:
        return out

    path = result.get("stress_path")
    band = result.get("band") or {}
    cumulative = ((result.get("cumulative") or {}).get("late_window") or {})
    stopped = result.get("stopping") or {}

    if path and band.get("history"):
        # full result of a fresh run: state matched to the band window
        from benchmarks.biaxial_shear_band.run_r7_wide_window_ensemble import (
            LATE_WINDOW_FRACTION,
        )
        state = late_window_state(path, late_window_fraction=LATE_WINDOW_FRACTION)
        peak_strain = float((result.get("peak") or {}).get("axial_strain", float("nan")))
        final_strain = float((result.get("loading") or {})
                             .get("final_axial_strain", float("nan")))
        tail_start = peak_strain + (1.0 - LATE_WINDOW_FRACTION) * max(
            final_strain - peak_strain, 0.0)
        angles = late_window_band_angle(band.get("history") or [],
                                        tail_start_strain=tail_start)
        published = float(band.get("angle_mean_deg", float("nan")))
        purity = (abs(angles["incremental_mean_deg"] - published)
                  if math.isfinite(published)
                  and math.isfinite(angles["incremental_mean_deg"]) else float("nan"))
        theta = (angles["cumulative_mean_deg"]
                 if math.isfinite(angles["cumulative_mean_deg"])
                 else angles["incremental_mean_deg"])
        phi = state["phi_deg"]
        psi_peak = float((result.get("peak") or {})
                         .get("dilation_angle_deg", state["psi_deg"]))
        source = "late_window_from_stress_path"
        resolved = angles["resolved"]
        n_late = angles["n_snapshots"]
        theta_std = angles["incremental_std_deg"]
    elif "friction_angle_deg" in result:
        # pruned published cache: only the peak state and the estimator output exist
        theta = float(cumulative.get("radon", float("nan")))
        phi = float(result["friction_angle_deg"])
        psi_peak = float(result.get("dilation_angle_deg", float("nan")))
        state = {"phi_deg": phi, "phi_peak_deg": phi, "psi_deg": psi_peak,
                 "source": "pruned_r7_cache (peak state only)"}
        source = "peak_from_pruned_cache"
        resolved = bool(cumulative.get("resolved", False))
        n_late = int(stopped.get("late_window_span_cap_log") is not None) * 0
        n_late = None
        theta_std = float("nan")
        purity = float("nan")
    else:
        out["error"] = "no stress_path and no friction_angle_deg: unusable input"
        return out

    out.update({
        "state_source": source,
        "theta_deg": float(theta),
        "theta_std_deg": theta_std,
        "n_late_snapshots": n_late,
        "resolved": resolved,
        "purity_vs_published_deg": float(purity),
        "state_primary": state,
        "state_peak": {"phi_deg": float(state["phi_peak_deg"]),
                       "psi_deg": float(psi_peak)},
        "dilation_measured": _saturation(state["psi_deg"]),
        "primary": adjudicate(theta, phi, state["psi_deg"],
                              tolerance=TOLERANCE, precision=PRECISION),
        "peak_arm": adjudicate(theta, float(state["phi_peak_deg"]), psi_peak,
                               tolerance=TOLERANCE, precision=PRECISION),
        "stress_dilatancy_arm": _sd_arm(theta, phi),
        "stop_reason": stopped.get("stop_reason"),
        "inertial_number": result.get("inertial_number"),
        "broken_bonds": result.get("broken_bonds"),
        "theory_angles_deg": result.get("theory_angles_deg"),
    })
    return out


def _stats(values) -> dict:
    arr = np.asarray([v for v in values if v is not None and math.isfinite(float(v))],
                     dtype=float)
    if arr.size == 0:
        return {"n": 0}
    sem = float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else float("nan")
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "sem": sem,
            "ci95": [float(arr.mean() - 1.96 * sem), float(arr.mean() + 1.96 * sem)]
            if arr.size > 1 else [float(arr.mean()), float(arr.mean())],
            "min": float(arr.min()), "max": float(arr.max())}


def _arm_block(rows: list[dict], arm) -> dict:
    """Per-law statistics for one dilation arm.

    ``arm(row)`` returns ``(phi_deg, psi_deg, adjudication)``.
    """
    block: dict = {"theta_deg": _stats([r["theta_deg"] for r in rows])}
    phis, psis = [], []
    for r in rows:
        phi, psi, _ = arm(r)
        phis.append(phi)
        psis.append(psi)
    block["phi_deg"] = _stats(phis)
    block["psi_deg"] = _stats(psis)
    for law in LAWS:
        adjudications = [arm(r)[2] for r in rows]
        pred = _stats([a["errors"][law]["predicted_deg"] for a in adjudications])
        err = _stats([a["errors"][law]["abs_relative_error"] for a in adjudications])
        block[law] = {
            "predicted_deg": pred,
            "abs_relative_error": err,
            "mean_prediction_vs_mean_theta_relative": (
                (block["theta_deg"]["mean"] - pred["mean"]) / pred["mean"]
                if pred.get("n") and block["theta_deg"].get("n") else float("nan")),
            "closest_count": int(sum(1 for a in adjudications
                                     if a["closest_law"] == law)),
            "tolerance_consistent_count": int(sum(
                1 for a in adjudications
                if a["errors"][law]["abs_relative_error"] <= TOLERANCE)),
            "precision_consistent_count": int(sum(
                1 for a in adjudications
                if a["errors"][law]["abs_relative_error"] <= PRECISION)),
        }
    block["discrimination_deg"] = _stats([arm(r)[2]["discrimination_deg"] for r in rows])
    return block


def _arm_verdict(block: dict, n_specimens: int) -> dict:
    verdict: dict = {}
    for law in LAWS:
        info = block[law]
        cond_a = abs(info["mean_prediction_vs_mean_theta_relative"]) <= TOLERANCE
        cond_b = info["closest_count"] >= min(SUPPORT_MIN_FRACTION, n_specimens)
        cond_c = (block["discrimination_deg"].get("mean", 0.0)
                  >= MIN_DISCRIMINATION_DEG)
        verdict[law] = {
            "mean_within_tolerance": bool(cond_a),
            "closest_for_min_fraction": bool(cond_b),
            "discriminating": bool(cond_c),
            "supported": bool(cond_a and cond_b and cond_c),
            "mean_abs_relative_error": info["abs_relative_error"].get("mean"),
            "mean_abs_relative_error_under_precision": bool(
                (info["abs_relative_error"].get("mean") or 1e9) <= PRECISION),
        }
    return verdict


def _ensemble(rows: list[dict]) -> dict:
    good = [r for r in rows
            if not r.get("error") and math.isfinite(float(r.get("theta_deg", "nan")))]
    if not good:
        return {"n": 0}
    resolved = [r for r in good if r.get("resolved")]

    arms = {
        "measured_psi": _arm_block(
            good, lambda r: (r["state_primary"]["phi_deg"],
                             r["state_primary"]["psi_deg"], r["primary"])),
        "peak": _arm_block(
            good, lambda r: (r["state_peak"]["phi_deg"],
                             r["state_peak"]["psi_deg"], r["peak_arm"])),
        # isolates the *coefficient* of the law from the dilation input: with
        # psi = 0 the three laws collapse to 45 + phi/2, 45, 45 + phi/4
        "zero_psi": _arm_block(
            good, lambda r: (r["state_primary"]["phi_deg"], 0.0,
                             adjudicate(r["theta_deg"],
                                        r["state_primary"]["phi_deg"], 0.0,
                                        tolerance=TOLERANCE, precision=PRECISION))),
    }
    for phi_cv in PHI_CV_RANGE:
        arms[f"stress_dilatancy[{phi_cv:g}]"] = _arm_block(
            good, lambda r, cv=phi_cv: (
                r["state_primary"]["phi_deg"],
                r["stress_dilatancy_arm"][str(cv)]["psi_deg"],
                r["stress_dilatancy_arm"][str(cv)]))

    verdicts = {name: _arm_verdict(block, len(good)) for name, block in arms.items()}
    supported_by_arm = {name: [law for law in LAWS if verdicts[name][law]["supported"]]
                        for name in arms}
    #: a law is supported by the stress-dilatancy route only if it clears the bar
    #: at *every* declared phi_cv (phi_cv is never fitted)
    sd_names = [n for n in arms if n.startswith("stress_dilatancy[")]
    sd_law_ok = {law: all(law in supported_by_arm[n] for n in sd_names) for law in LAWS}

    return {
        "n": len(good),
        "n_resolved": len(resolved),
        "anchor_deg": ANCHOR_DEG,
        "theta_deg": arms["measured_psi"]["theta_deg"],
        "anchor_relative_error": ((arms["measured_psi"]["theta_deg"]["mean"] - ANCHOR_DEG)
                                  / ANCHOR_DEG),
        "anchor_within_tolerance": bool(
            abs((arms["measured_psi"]["theta_deg"]["mean"] - ANCHOR_DEG)
                / ANCHOR_DEG) <= TOLERANCE),
        "state_sources": sorted({r.get("state_source") for r in good}),
        "dilation_saturation": {
            "n_saturated": int(sum(1 for r in good if r["dilation_measured"]["saturated"])),
            "n": len(good),
            "clip_deg": math.degrees(math.asin(DILATION_CLIP)),
            "psi_deg": _stats([r["state_primary"]["psi_deg"] for r in good]),
            "implied_slope": _stats([r["dilation_measured"]["implied_d_eps_v_d_eps_a"]
                                     for r in good])},
        "arms": arms,
        "verdicts": verdicts,
        "supported_by_arm": supported_by_arm,
        "stress_dilatancy_supported_at_every_phi_cv": sd_law_ok,
        "resolved_subset": {
            "n": len(resolved),
            "measured_psi_theta_deg": _stats([r["theta_deg"] for r in resolved]),
            "stress_dilatancy_arthur": {
                str(cv): _stats([r["stress_dilatancy_arm"][str(cv)]["errors"]["arthur"]
                                 ["abs_relative_error"] for r in resolved])
                for cv in PHI_CV_RANGE},
        } if resolved else {"n": 0},
    }


def _sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _assemble(rows: list[dict], provenance: dict) -> dict:
    purity = [r["purity_vs_published_deg"] for r in rows
              if math.isfinite(r.get("purity_vs_published_deg", float("nan")))]
    return {
        "round": ROUND,
        "case": "CASE-BIAXIAL-SHEAR-BAND",
        "provenance": provenance,
        "protocol": {
            "phi_cv_range_deg": list(PHI_CV_RANGE),
            "tolerance": TOLERANCE,
            "precision": PRECISION,
            "support_min_fraction": SUPPORT_MIN_FRACTION,
            "min_discrimination_deg": MIN_DISCRIMINATION_DEG,
            "dilation_clip": DILATION_CLIP,
            "laws": list(LAWS),
        },
        "anchor": anchor_implied_angles(ANCHOR_DEG),
        "purity": {
            "max_abs_rebuild_minus_published_deg": (float(max(abs(v) for v in purity))
                                                     if purity else float("nan")),
            "n": len(purity),
        },
        "ensemble": _ensemble(rows),
        "specimens": rows,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _simulate_shard(index: int) -> int:
    from benchmarks.biaxial_shear_band.run_r7_wide_window_ensemble import (
        _configs,
        _protocol_gate,
        _run_cached,
    )

    _protocol_gate()
    os.makedirs(CACHE, exist_ok=True)
    config = _configs(False, None)[index]
    started = time.time()
    result = _run_cached(config, CACHE, True)
    print(f"[r9 shard {index}] {config['name']} "
          f"stop={(result.get('stopping') or {}).get('stop_reason')} "
          f"theta={((result.get('band') or {}).get('angle_mean_deg'))} "
          f"phi_peak={((result.get('peak') or {}).get('friction_angle_deg'))} "
          f"wall={time.time() - started:.0f}s", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=None, help="specimen index 0..7")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--from-cache-dir", default=None,
                    help="directory holding anchor_s*.json (e.g. the committed "
                         "r7_cache); analysed as published evidence")
    ap.add_argument("--provenance", default=None,
                    help="label recorded in the output, e.g. 'commit a9f3ab91'")
    ap.add_argument("--out", default=None, help="override output JSON path")
    args = ap.parse_args(argv)

    if args.shard is not None:
        return _simulate_shard(args.shard)
    if not args.assemble:
        ap.error("one of --shard/--assemble is required")

    rows: list[dict] = []
    provenance: dict = {"source": args.provenance or "fresh r9 runs"}
    if args.from_cache_dir:
        files = sorted(glob.glob(os.path.join(args.from_cache_dir, "anchor_s*.json")))
        if not files:
            ap.error(f"no anchor_s*.json under {args.from_cache_dir}")
        for path in files:
            with open(path) as handle:
                rows.append(_analyse_specimen(json.load(handle)))
        provenance["files"] = [{"path": os.path.relpath(p, ROOT),
                                "sha256": _sha256(p)} for p in files]
    else:
        from benchmarks.biaxial_shear_band.run_r7_wide_window_ensemble import (
            _configs,
        )
        for config in _configs(False, None):
            path = os.path.join(CACHE, config["name"] + ".json")
            if not os.path.exists(path):
                rows.append({"name": config["name"], "seed": config["seed"],
                             "error": f"missing cache {path}"})
                continue
            with open(path) as handle:
                rows.append(_analyse_specimen(json.load(handle)))
        provenance["files"] = [
            {"path": os.path.relpath(p, ROOT), "sha256": _sha256(p)}
            for p in sorted(glob.glob(os.path.join(CACHE, "anchor_s*.json")))]

    payload = _assemble(rows, provenance)
    out_path = args.out or OUT
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=float)
    per = os.path.join(os.path.dirname(out_path), "per_specimen.json")
    with open(per, "w") as handle:
        json.dump({"round": ROUND, "specimens": rows}, handle, indent=2,
                  sort_keys=True, default=float)

    ens = payload["ensemble"]
    print(f"[r9 assemble] n={ens.get('n')} resolved={ens.get('n_resolved')} "
          f"theta={ens.get('theta_deg', {}).get('mean'):.2f} deg "
          f"anchor_err={ens.get('anchor_relative_error'):.4f}")
    print(f"  dilation saturated {ens['dilation_saturation']['n_saturated']}"
          f"/{ens['dilation_saturation']['n']}, clip={ens['dilation_saturation']['clip_deg']:.2f} deg")
    for name, block in ens["arms"].items():
        verdict = ens["verdicts"][name]
        cells = []
        for law in LAWS:
            cells.append(f"{law}: pred={block[law]['predicted_deg'].get('mean', float('nan')):.2f} "
                         f"|err|={block[law]['abs_relative_error'].get('mean', float('nan')):.4f} "
                         f"closest={block[law]['closest_count']} "
                         f"{'OK' if verdict[law]['supported'] else '-'}")
        print(f"  [{name}] " + " | ".join(cells))
    print(f"  supported_by_arm={ens['supported_by_arm']}")
    print(f"  sd_supported_every_phi_cv={ens['stress_dilatancy_supported_at_every_phi_cv']}")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())