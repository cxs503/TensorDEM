"""R8 of CASE-BIAXIAL-SHEAR-BAND: independent measurement of the same specimens.

Where the case stands
---------------------
R7 (``docs/CASE_BIAXIAL_SHEAR_BAND_R7.md``) delivered the case's PASS on the
published estimator: ensemble ``theta = 55.00 +- 2.07 deg`` over eight specimens,
8/8 resolved, 8.33 % from the 60 deg tolerance anchor.  The R7 adversarial
review (``docs/ADVERSARIAL_REVIEW_BIAXIAL_SHEAR_BAND_R7.md`` §4.3) left three
checks unrun; the one this round discharges is item 2:

    "Independent estimator reproduction: did not re-implement the Radon focus
     from scratch; trusted the validated ``radon_band_angle``."

so an operator-specific bias could not be separated from the physical band
angle by the published evidence.

What R8 does
------------
1. Adds two operators with different statistics
   (``tensordem.biaxial_band_independent``): a matched-filter fit of an explicit
   Gaussian ridge model *with a conjugate term*, and a spectral-inertia estimate
   of the ``gamma`` field.  Neither maximises the Radon statistic and neither
   bins particle projections.
2. Quantifies each operator's bias against known inclinations on the *specimen
   geometry class* (jittered triangular lattice, 24 x 48) across band widths,
   particle noise, and X-shaped (conjugate) localisation -- the regime post-peak
   biaxial specimens actually enter.  This envelope is written to
   ``estimator_envelope.json`` and decides which operator may be applied.
3. Reruns the **R7 protocol unchanged** on the same eight anchor seeds
   (``post_peak_strain = 0.05``, ``late_window_fraction = 0.8``, ``strain_target
   = 0.12``, ``strain_cap = 0.22``, 24 x 48 lattice, 35 % freeze bonds, wall
   friction 0.55, 4000 consolidation steps, ``p = 100 kPa``) with
   ``record_snapshots=True``, measures every late-window field with all three
   operators, and reports the per-specimen and ensemble comparison.
4. Measures the dilation angle **inside the band** (band frame from the fitted
   model angle, half-width from the fitted ``sigma``), which is the quantity R7
   withheld as "unphysical" because it had only the *global* specimen dilation.
   That makes the Roscoe (``45 + psi/2``) and Arthur (``45 + (phi+psi)/4``) laws
   testable against the measured angle instead of merely quoted.

Outputs ``artifacts/biaxial_shear_band/r8_independent/``:
``estimator_envelope.json``, ``cache/<name>.json`` (one per specimen),
``late_window_positions/<name>.npz`` (raw positions the numbers come from) and
``r8_independent_ensemble.json``.

Run: ``R4_THREADS=4 python benchmarks/biaxial_shear_band/run_r8_independent_estimator.py
--shard <i>`` for ``i in 0..7`` then ``--assemble``.
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ART = os.path.join(ROOT, "artifacts", "biaxial_shear_band", "r8_independent")
CACHE = os.path.join(ART, "cache")
POSITIONS = os.path.join(ART, "late_window_positions")
ENVELOPE = os.path.join(ART, "estimator_envelope.json")
OUT = os.path.join(ART, "r8_independent_ensemble.json")

#: the eight R7 anchor seeds, unchanged
SEEDS = [53, 67, 79, 83, 89, 97, 101, 103]
#: R7 protocol, unchanged (see docs/CASE_BIAXIAL_SHEAR_BAND_R7.md §①)
POST_PEAK_STRAIN = 0.05
LATE_WINDOW_FRACTION = 0.8
STRAIN_TARGET = 0.12
STRAIN_CAP = STRAIN_TARGET + 2.0 * POST_PEAK_STRAIN
CONFINING_STRESS = 100.0e3
COLUMNS, ROWS = 24, 48
CONSOLIDATION_STEPS = 4000
BOND_FRACTION = 0.35
WALL_FRICTION = 0.55
#: published anchor bookkeeping (R4..R7 constant, kept for a like-for-like delta)
THETA_REF_DEG = 60.0
LITERATURE_BAND_DEG = (55.0, 60.0)
TOLERANCE = 0.15


# ---------------------------------------------------------------------------
# specimen
# ---------------------------------------------------------------------------
def _run_specimen(seed: int, *, quick: bool = False) -> tuple[dict, list[dict]]:
    """Run the R7 protocol for one seed and return (result, late-window snapshots)."""
    import torch
    torch.set_num_threads(int(os.environ.get("R4_THREADS", "1")))
    from tensordem.biaxial_shear_band import run_biaxial_shear_band

    result = run_biaxial_shear_band(
        confining_stress=CONFINING_STRESS,
        columns=16 if quick else COLUMNS,
        rows=32 if quick else ROWS,
        strain_rate=0.05,
        strain_target=0.08 if quick else STRAIN_TARGET,
        post_peak_strain=POST_PEAK_STRAIN,
        strain_cap=STRAIN_CAP,
        late_window_fraction=LATE_WINDOW_FRACTION,
        consolidation_steps=600 if quick else CONSOLIDATION_STEPS,
        snapshot_strain=0.01, servo_gain=3.0e-4, servo_velocity_limit=2.0,
        wall_friction=WALL_FRICTION,
        material_overrides={"freeze_bond_fraction": BOND_FRACTION, "seed": int(seed)},
        record_snapshots=True)
    snapshots = result["snapshots"]
    peak_strain = float(result["peak"]["axial_strain"])
    final_strain = float(snapshots[-1]["strain"])
    tail_start = peak_strain + (1.0 - LATE_WINDOW_FRACTION) * max(
        final_strain - peak_strain, 0.0)
    late = [s for s in snapshots if s["strain"] >= tail_start - 1e-12]
    if not late:                                   # defensive: never silently empty
        late = [snapshots[-1]]
    return result, late


def _dilation_in_band(late: list[dict], *, angle_deg: float, half_width: float
                      ) -> dict:
    """Band-local and background dilation angle from the late-window snapshots."""
    from tensordem.biaxial_band_diagnostics import band_local_dilation
    try:
        return band_local_dilation(late, angle_deg=float(angle_deg),
                                   half_width=float(half_width))
    except Exception as exc:                       # reported, never fabricated
        return {"error": repr(exc)}


def measure_specimen(seed: int, *, quick: bool = False) -> dict:
    """Full R8 record for one specimen (JSON-serialisable)."""
    from tensordem.biaxial_band_independent import measure_band_independent
    from tensordem.biaxial_shear_band import fold_angle_deg, theory_angles_deg

    started = time.time()
    result, late = _run_specimen(seed, quick=quick)
    name = f"anchor_s{seed}"
    window = late[0]["window"] or late[-1]["window"]

    record: dict = {
        "name": name, "seed": int(seed), "quick": bool(quick),
        "protocol": {
            "post_peak_strain": POST_PEAK_STRAIN,
            "late_window_fraction": LATE_WINDOW_FRACTION,
            "strain_target": STRAIN_TARGET, "strain_cap": STRAIN_CAP,
            "confining_stress": CONFINING_STRESS,
            "lattice": [COLUMNS, ROWS], "consolidation_steps": CONSOLIDATION_STEPS,
            "freeze_bond_fraction": BOND_FRACTION, "wall_friction": WALL_FRICTION,
        },
        "stopping": result["stopping"],
        "peak": {k: result["peak"][k] for k in
                 ("axial_strain", "friction_angle_deg", "dilation_angle_deg",
                  "sin_phi", "post_peak_d_eps_v_d_eps_a")},
        "published_band": {k: result["band"][k] for k in
                           ("angle_deg", "angle_mean_deg", "angle_std_deg",
                            "resolved", "late_snapshots")},
        "inertial_number": result["loading"]["inertial_number"],
        "particles": int(result["specimen"]["particles"]),
        "bonds": int(result["specimen"]["bonds"]),
    }

    # ---- cumulative measurement over the late window (R7's primary route) --
    try:
        cum = measure_band_independent(late[0]["positions"], late[-1]["positions"],
                                       bounds=window)
    except Exception as exc:                       # reported, never fabricated
        cum = {"error": repr(exc)}
    record["cumulative_late_window"] = cum

    # ---- one-step measurements, per consecutive late-window pair ----------
    steps = []
    for ref, now in zip(late[:-1], late[1:]):
        entry = {"strain_from": float(ref["strain"]), "strain_to": float(now["strain"])}
        try:
            m = measure_band_independent(ref["positions"], now["positions"],
                                         bounds=ref["window"] or now["window"])
            entry.update({"folded_angles_deg": m["folded_angles_deg"],
                          "model_r2": m.get("model", {}).get("r2"),
                          "model_sigma": m.get("model", {}).get("sigma"),
                          "model_conjugate_ratio": m.get("model", {}).get(
                              "conjugate_ratio"),
                          "radon_z": m.get("radon", {}).get("z_score"),
                          "radon_eta2": m.get("radon", {}).get("eta2"),
                          "spread_deg": m.get("spread_deg")})
        except Exception as exc:
            entry["error"] = repr(exc)
        steps.append(entry)
    record["one_step"] = steps

    # ---- band-local dilation, band frame from the fitted model ------------
    model = (cum or {}).get("model", {}) if isinstance(cum, dict) else {}
    angle = model.get("folded_angle_deg")
    sigma = model.get("sigma")
    if angle is not None and sigma is not None:
        dil = _dilation_in_band(late, angle_deg=float(angle), half_width=float(sigma))
        record["band_dilation"] = dil
        psi_band = dil.get("psi_band_deg")
        if psi_band is not None and np.isfinite(psi_band):
            phi = float(result["peak"]["friction_angle_deg"])
            record["theory_with_band_dilation"] = {
                "phi_deg": phi, "psi_band_deg": float(psi_band),
                "laws": theory_angles_deg(phi, float(psi_band)),
            }
    record["elapsed_s"] = float(time.time() - started)

    # ---- raw evidence: the positions the numbers came from ----------------
    os.makedirs(POSITIONS, exist_ok=True)
    np.savez_compressed(
        os.path.join(POSITIONS, f"{name}.npz"),
        strains=np.asarray([s["strain"] for s in late], dtype=np.float64),
        windows=np.asarray([s["window"] for s in late], dtype=np.float64),
        positions=np.asarray([s["positions"] for s in late], dtype=np.float32))
    return record


# ---------------------------------------------------------------------------
# assembly / statistics
# ---------------------------------------------------------------------------
def _operator_angle(record: dict, operator: str) -> float | None:
    cum = record.get("cumulative_late_window") or {}
    if "error" in cum:
        return None
    value = (cum.get("folded_angles_deg") or {}).get(operator)
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _stats(values: list[float]) -> dict:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                     dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean_deg": None, "std_deg": None, "min_deg": None,
                "max_deg": None, "ci95_deg": None}
    sem = float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {"n": int(arr.size), "mean_deg": float(arr.mean()),
            "std_deg": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min_deg": float(arr.min()), "max_deg": float(arr.max()),
            "ci95_deg": [float(arr.mean() - 1.96 * sem),
                         float(arr.mean() + 1.96 * sem)]}


def assemble(records: list[dict]) -> dict:
    """Ensemble comparison of the three operators on the same specimens."""
    ops = ("radon", "model", "spectral")
    per_op = {op: [ _operator_angle(r, op) for r in records] for op in ops}
    delta = []
    for r in records:
        a, b = _operator_angle(r, "radon"), _operator_angle(r, "model")
        delta.append(None if (a is None or b is None) else float(b - a))

    ensemble = {f"{op}_late_window": _stats(per_op[op]) for op in ops}
    ensemble["model_minus_radon"] = _stats(delta)

    theory = []
    for r in records:
        band = _operator_angle(r, "model")
        dil = r.get("band_dilation") or {}
        entry = {
            "name": r["name"], "theta_model_deg": band,
            "theta_radon_deg": _operator_angle(r, "radon"),
            "phi_deg": r.get("peak", {}).get("friction_angle_deg"),
            "psi_global_deg": r.get("peak", {}).get("dilation_angle_deg"),
            "psi_specimen_deg": dil.get("psi_specimen_deg"),
            "psi_background_deg": dil.get("psi_background_deg"),
            "band_half_width": dil.get("half_width"),
            "psi_band_deg": dil.get("psi_band_deg"),
            "model_r2": (r.get("cumulative_late_window") or {}).get("model", {}).get("r2"),
            "model_conjugate_ratio": (r.get("cumulative_late_window") or {})
            .get("model", {}).get("conjugate_ratio"),
        }
        laws = (r.get("theory_with_band_dilation") or {}).get("laws")
        if laws and band is not None:
            entry["laws_deg"] = laws
            entry["relative_error_vs_laws"] = {
                k: (float(band) - v) / v for k, v in laws.items()}
        theory.append(entry)

    def _mean_of(key: str) -> float | None:
        vals = [t[key] for t in theory if t.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    theta_model = ensemble["model_late_window"]["mean_deg"]
    report = {
        "round": "R8",
        "case": "CASE-BIAXIAL-SHEAR-BAND",
        "question": ("does the published Radon band angle survive an independent "
                     "measurement operator on the identical specimens?"),
        "protocol_seeds": [r["seed"] for r in records],
        "operators": {
            "radon": "published (tensordem.biaxial_shear_band.radon_band_angle)",
            "model": "R8 matched-filter Gaussian ridge fit, conjugate aware",
            "spectral": "R8 spectral inertia of the gamma field (single-band only)",
        },
        "ensemble": ensemble,
        "per_specimen": theory,
        "ensemble_phi_deg": _mean_of("phi_deg"),
        "ensemble_psi_band_deg": _mean_of("psi_band_deg"),
        "ensemble_psi_global_deg": _mean_of("psi_global_deg"),
        "anchor": {
            "tolerance_anchor_deg": THETA_REF_DEG,
            "literature_band_deg": list(LITERATURE_BAND_DEG),
            "literature_band_midpoint_deg": float(np.mean(LITERATURE_BAND_DEG)),
            "tolerance": TOLERANCE,
            "relative_error_vs_anchor": (
                abs(theta_model - THETA_REF_DEG) / THETA_REF_DEG
                if theta_model is not None else None),
            "relative_error_vs_band_midpoint": (
                abs(theta_model - float(np.mean(LITERATURE_BAND_DEG)))
                / float(np.mean(LITERATURE_BAND_DEG))
                if theta_model is not None else None),
            "within_tolerance_vs_anchor": (
                bool(abs(theta_model - THETA_REF_DEG) / THETA_REF_DEG <= TOLERANCE)
                if theta_model is not None else None),
            "inside_literature_band": (
                bool(LITERATURE_BAND_DEG[0] <= theta_model <= LITERATURE_BAND_DEG[1])
                if theta_model is not None else None),
        },
        "changed_verdict": None,
        "note": ("measurement study on the R7 protocol; no protocol, material or "
                 "resampling change, so any difference is attributable to the "
                 "measurement operator"),
    }
    radon_mean = ensemble["radon_late_window"]["mean_deg"]
    if theta_model is not None and radon_mean is not None:
        report["changed_verdict"] = bool(
            report["anchor"]["within_tolerance_vs_anchor"] !=
            (abs(radon_mean - THETA_REF_DEG) / THETA_REF_DEG <= TOLERANCE))
    return report


def _sha256(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_envelope() -> dict:
    from tensordem.biaxial_band_independent import recover_angle_sweep
    started = time.time()
    sweep = recover_angle_sweep()
    envelope = {
        "round": "R8",
        "purpose": ("operator bias against known inclinations on the specimen "
                    "geometry class; decides which operator may be applied to a "
                    "specimen field and in which regime"),
        "sweep": sweep,
        "elapsed_s": float(time.time() - started),
    }
    os.makedirs(ART, exist_ok=True)
    with open(ENVELOPE, "w", encoding="utf-8") as handle:
        json.dump(envelope, handle, indent=1, sort_keys=True)
    for case in sweep["cases"]:
        summary = {op: (None if v["mae_deg"] is None else round(v["mae_deg"], 2))
                   for op, v in case["summary"].items()}
        print(f"[envelope] half_width={case['half_width']} "
              f"conjugate={case['conjugate_fraction']} MAE={summary}", flush=True)
    return envelope


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=None,
                        help=f"specimen index 0..{len(SEEDS) - 1}")
    parser.add_argument("--quick", action="store_true",
                        help="16x32 lattice, 600 consolidation steps (smoke only)")
    parser.add_argument("--envelope", action="store_true",
                        help="write estimator_envelope.json")
    parser.add_argument("--assemble", action="store_true")
    args = parser.parse_args(argv)
    os.makedirs(CACHE, exist_ok=True)

    if args.envelope:
        run_envelope()
        return 0

    if args.shard is not None:
        seed = SEEDS[int(args.shard)]
        record = measure_specimen(seed, quick=bool(args.quick))
        path = os.path.join(CACHE, f"anchor_s{seed}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=1, sort_keys=True)
        cum = record.get("cumulative_late_window") or {}
        print(f"[shard {args.shard}] s{seed} "
              f"theta={ (cum.get('folded_angles_deg') or {}) } "
              f"r2={(cum.get('model') or {}).get('r2')} "
              f"conj={(cum.get('model') or {}).get('conjugate_ratio')} "
              f"wall={record['elapsed_s']:.0f}s", flush=True)
        return 0

    if args.assemble:
        records = []
        missing = []
        for seed in SEEDS:
            path = os.path.join(CACHE, f"anchor_s{seed}.json")
            if not os.path.exists(path):
                missing.append(seed)
                continue
            with open(path, encoding="utf-8") as handle:
                records.append(json.load(handle))
        report = assemble(records)
        report["missing_seeds"] = missing
        report["artifacts"] = {"envelope_sha256": _sha256(ENVELOPE),
                               "cache": [os.path.relpath(
                                   os.path.join(CACHE, f"anchor_s{s}.json"), ROOT)
                                   for s in SEEDS
                                   if os.path.exists(
                                       os.path.join(CACHE, f"anchor_s{s}.json"))]}
        with open(OUT, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, sort_keys=True)
        print(json.dumps({"ensemble": report["ensemble"],
                          "anchor": report["anchor"],
                          "missing": missing}, indent=1))
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())