"""R8 of CASE-BIAXIAL-SHEAR-BAND: which operator is the band angle?

Where R7 stopped
----------------
R7 (``docs/CASE_BIAXIAL_SHEAR_BAND_R7.md``) closed the case's *protocol* debt
(rule and late window declared together, 8/8 specimens resolved) and passed the
acceptance rule ``|mean - 60| / 60 = 8.33 % < 15 %`` with the primary
estimator, cumulative Radon focus.  It also published three secondary numbers
on the same late-window fields -- peak-relative Radon 56.25 deg,
single-increment Radon 53.91 deg, **structure tensor 67.75 deg** -- i.e. a
12.75 deg spread, 23 % of the measured value, the same order as the 15 %
tolerance the anchor is judged with.  Nothing in R4-R7 decided which of those
numbers is the band angle; the verdict therefore rests on the choice of
operator.

R7 left two further items explicitly open:

* the operator validation in the repo is done on **clean** synthetic lattices
  (:func:`tensordem.biaxial_xval.cross_validate_synthetic`,
  :func:`tensordem.biaxial_band_diagnostics.estimator_bias_sweep`), so it says
  nothing about how these operators behave on a *real* late-window field with a
  diffuse background and wall-nucleated bands;
* the psi-based laws (Roscoe 1970, Arthur et al. 1977) stayed **WITHHELD**
  because the specimen-global dilation angle of a bonded specimen is
  unphysical -- but the *in-band* dilation angle, which is what those laws
  actually use, was never measured for the R7 fields even though
  :func:`tensordem.biaxial_band_diagnostics.band_local_dilation` can measure it.

What R8 does
------------
1. Reruns three anchor specimens (seeds 53, 67, 89 -- the two R7 extremes and
   the midpoint) with the **unchanged R7 protocol** and reproduces the primary
   measurement, so the fields under audit are the delivered ones.
2. Measures all three orientation operators on **one shared** field per
   specimen (:func:`tensordem.biaxial_operator_audit.operator_angles`): the only
   difference is the operator, not the pair, the window or the snapshot.
3. Adjudicates them with a **planted-band audit on the real field**
   (:func:`tensordem.biaxial_operator_audit.planted_field_audit`): a known-angle
   ridge is superposed on each specimen's measured shear-strain field at
   contamination ``c = 1.0 / 0.5 / 0.25`` and every operator is asked to
   recover the known angle, so each reading can be bias-corrected instead of
   believed a priori.
4. Measures the **in-band** dilation angle of the R7 late window and evaluates
   the three classical laws with in-run ``phi`` and in-band ``psi``
   (:func:`tensordem.biaxial_operator_audit.law_test`), turning R7's WITHHELD
   item into a decided one.
5. Reports an operator-robustness verdict for the anchor: does the acceptance
   decision survive *every* operator, and what is the honest uncertainty of the
   band angle?

Predeclared protocol (fixed before any R8 number was seen)
----------------------------------------------------------
* DEM: identical to R7 -- 24 x 48 jittered disc lattice, 35 % freeze bonds,
  wall friction 0.55, 4000 consolidation steps, strain rate 0.05 /s, snapshots
  every 0.01 strain, p = 100 kPa servo walls, ``strain_target = 0.12``,
  ``post_peak_strain = 0.05``, ``strain_cap = 0.22``,
  ``late_window_fraction = 0.8``.  Seeds 53, 67, 89.
* Field under audit: the cumulative late window, ``late[0] -> late[-1]``,
  restricted to the snapshot's own measurement window (boundary layer
  excluded), ``local_shear_strain(k = 8)``.
* Operators: Radon focus (production), structure tensor (``biaxial_xval``,
  Desrues & Viggiani route), weighted second moment (``regionprops`` route).
* Planted audit: angles 30..75 deg in 5 deg steps, contamination
  1.0 / 0.5 / 0.25, ridge half-width 2.5 x the specimen's mean particle
  spacing.
* **P-R8 (falsifiable, this round):** every operator, after bias correction at
  ``c = 1.0``, lands within the 15 % acceptance band of the 60 deg anchor.  If
  an operator still misses the band after correction, the case's PASS is
  operator-dependent and the anchor is not closed, whatever the primary
  estimator says.
* Secondary prediction **P-R8b**: the in-band dilation angle of the R7 late
  window predicts the measured inclination through Roscoe ``45 + psi/2`` within
  15 %; if it does not, the psi-based laws stay withheld and are *reported as
  refuted for this material state*, not silently dropped.

Outputs ``artifacts/biaxial_shear_band/r8_operator_adjudication.json`` plus
figures; the text report is rendered from that JSON by
``benchmarks/biaxial_shear_band/build_r8_report.py``.

Run (local short-test口径, 7 threads, one specimen per invocation):
``R4_THREADS=7 python benchmarks/biaxial_shear_band/run_r8_operator_adjudication.py --only-seed 53``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from benchmarks.biaxial_shear_band.run_r4_cumulative_ensemble import (  # noqa: E402
    BOND_FRACTION,
    CONSOLIDATION_STEPS,
    WALL_FRICTION,
    _late_snapshots,
)
from tensordem.biaxial_operator_audit import (  # noqa: E402
    ANCHOR_DEG,
    OPERATORS,
    TOLERANCE,
    bias_corrected_angle,
    law_test,
    lattice_spacing,
    operator_angles,
    planted_field_audit,
    signed_angle_diff_deg,
)

ROUND = "R8"
SEEDS = (53, 67, 89)
POST_PEAK_STRAIN = 0.05
STRAIN_TARGET = 0.12
STRAIN_CAP = STRAIN_TARGET + 2.0 * POST_PEAK_STRAIN
LATE_WINDOW_FRACTION = 0.8
CONTAMINATIONS = (1.0, 0.5, 0.25)
AUDIT_ANGLES = tuple(range(30, 76, 5))
#: Radon readings published for the same eight seeds by R7
#: (``docs/CASE_BIAXIAL_SHEAR_BAND_R7.md`` section ③ paired table).  Carried here
#: only so the reproduction check is a table column instead of prose; the R8
#: numbers themselves are recomputed from fresh DEM runs.
R7_REFERENCE_DEG = {53: 55.0, 67: 57.0, 79: 57.0, 83: 55.0,
                    89: 51.0, 97: 54.0, 101: 57.0, 103: 54.0}

#: a late window needs this many snapshot pairs before an in-band dilation
#: angle (and hence any psi-based law) is reported at all
MIN_DILATION_PAIRS = 3

ART = os.path.join(ROOT, "artifacts", "biaxial_shear_band")
OUT = os.path.join(ART, "r8_operator_adjudication.json")


# ---------------------------------------------------------------------------
# one specimen
# ---------------------------------------------------------------------------
def _run_specimen(seed: int, *, quick: bool = False) -> dict:
    """Run one R7-protocol specimen and audit every operator on its field."""
    import torch

    torch.set_num_threads(int(os.environ.get("R4_THREADS", "1")))
    from tensordem.biaxial_shear_band import run_biaxial_shear_band
    from tensordem.biaxial_band_diagnostics import band_local_dilation
    from tensordem.biaxial_shear_band import local_shear_strain

    started = time.time()
    result = run_biaxial_shear_band(
        confining_stress=100.0e3,
        columns=16 if quick else 24, rows=32 if quick else 48,
        strain_rate=0.05,
        strain_target=0.08 if quick else STRAIN_TARGET,
        post_peak_strain=POST_PEAK_STRAIN, strain_cap=STRAIN_CAP,
        late_window_fraction=LATE_WINDOW_FRACTION,
        consolidation_steps=600 if quick else CONSOLIDATION_STEPS,
        snapshot_strain=0.01, servo_gain=3.0e-4, servo_velocity_limit=2.0,
        wall_friction=WALL_FRICTION,
        material_overrides={"freeze_bond_fraction": BOND_FRACTION, "seed": int(seed)},
        record_snapshots=True,
    )
    wall = time.time() - started
    snapshots = result["snapshots"]
    peak_strain = float(result["peak"]["axial_strain"])
    late = _late_snapshots(snapshots, peak_strain, LATE_WINDOW_FRACTION)

    record: dict = {
        "seed": int(seed), "name": f"anchor_s{seed}", "round": ROUND,
        "quick": bool(quick),
        "dem": {
            "wall_seconds": float(wall),
            "stop_reason": (result.get("stopping") or {}).get("stop_reason"),
            "final_axial_strain_log": (result.get("stopping") or {}).get(
                "final_axial_strain_log"),
            "peak_axial_strain": peak_strain,
            "friction_angle_deg": float(result["peak"]["friction_angle_deg"]),
            "n_snapshots": int(len(snapshots)),
            "n_late_snapshots": int(len(late)),
            "late_strain_span": (float(late[-1]["strain"] - late[0]["strain"])
                                 if len(late) > 1 else 0.0),
            "inertial_number": result.get("inertial_number"),
        },
    }
    if len(late) < 2:
        record["error"] = "late window has fewer than two snapshots"
        return record

    ref, now = late[0], late[-1]
    bounds = ref.get("window") or now.get("window")
    x_ref_all = np.asarray(ref["positions"], dtype=np.float64)
    x_now_all = np.asarray(now["positions"], dtype=np.float64)
    if bounds is not None:
        xmin, xmax, ymin, ymax = (float(v) for v in bounds)
        inside = ((x_ref_all[:, 0] >= xmin) & (x_ref_all[:, 0] <= xmax)
                  & (x_ref_all[:, 1] >= ymin) & (x_ref_all[:, 1] <= ymax))
    else:                                                      # pragma: no cover
        inside = np.ones(x_ref_all.shape[0], dtype=bool)
    pos = x_ref_all[inside]
    gamma_real = local_shear_strain(pos, x_now_all[inside], k=8)

    # ---- (1) the three operators on one shared field --------------------
    shared = operator_angles(pos, gamma_real)
    record["shared_field"] = shared
    record["angles_deg"] = {
        name: (shared.get(name) or {}).get("angle_deg") for name in OPERATORS}
    record["operator_spread_deg"] = (
        float(max(v for v in record["angles_deg"].values() if v is not None)
              - min(v for v in record["angles_deg"].values() if v is not None))
        if all(record["angles_deg"].get(n) is not None for n in OPERATORS)
        else None)

    # ---- (2) single-increment Radon (R7's third secondary) --------------
    step_angles = []
    for a, b in zip(late[:-1], late[1:]):
        window = a.get("window") or b.get("window")
        try:
            from tensordem.biaxial_shear_band import measure_band
            step_angles.append(float(measure_band(
                np.asarray(a["positions"]), np.asarray(b["positions"]),
                bounds=window)["folded_angle_deg"]))
        except Exception:                                  # pragma: no cover
            pass
    record["single_increment_radon_deg"] = (
        float(np.mean(step_angles)) if step_angles else None)
    record["single_increment_samples"] = len(step_angles)

    # ---- (3) planted-band audit on the real field -----------------------
    audit = planted_field_audit(pos, gamma_real,
                                angles_deg=AUDIT_ANGLES,
                                contaminations=CONTAMINATIONS)
    record["planted_audit"] = audit

    # ---- (4) bias-corrected readings ------------------------------------
    corrected = {}
    for name in OPERATORS:
        measured = record["angles_deg"].get(name)
        if measured is None:
            continue
        corrected[name] = {}
        for c in CONTAMINATIONS:
            try:
                corrected[name][f"{c:.2f}"] = bias_corrected_angle(
                    float(measured), audit, operator=name, contamination=c)
            except Exception as exc:                       # pragma: no cover
                corrected[name][f"{c:.2f}"] = {"error": repr(exc)}
    record["bias_corrected"] = corrected

    # ---- (5) in-band dilation and the classical laws --------------------
    spacing = lattice_spacing(pos)
    half_width = 2.5 * spacing
    radon_angle = record["angles_deg"].get("radon")
    dilation = None
    if radon_angle is not None:
        try:
            dilation = band_local_dilation(late, angle_deg=float(radon_angle),
                                           half_width=half_width)
        except Exception as exc:                           # pragma: no cover
            dilation = {"error": repr(exc)}
    record["band_dilation"] = dilation
    # A late window of three snapshots gives only two snapshot pairs, and a
    # two-point regression fits *any* line exactly (r2 = 1 by construction), so
    # psi_band is only accepted with at least MIN_DILATION_PAIRS pairs.  Below
    # that the law test is WITHHELD for this specimen rather than reported as a
    # number the data cannot support.
    pairs = int((dilation or {}).get("pairs") or 0)
    record["psi_reliable"] = bool(pairs >= MIN_DILATION_PAIRS)
    if (dilation and dilation.get("psi_band_deg") is not None
            and radon_angle is not None and record["psi_reliable"]):
        record["law_test"] = law_test(
            float(radon_angle),
            phi_deg=float(result["peak"]["friction_angle_deg"]),
            psi_band_deg=float(dilation["psi_band_deg"]))
    elif radon_angle is not None:
        record["law_test_withheld"] = (
            f"only {pairs} snapshot pairs in the late window "
            f"(< {MIN_DILATION_PAIRS}); a two-point regression is degenerate")
    return record


# ---------------------------------------------------------------------------
# ensemble
# ---------------------------------------------------------------------------
def _safe(seed: int, quick: bool) -> dict:
    try:
        return _run_specimen(seed, quick=quick)
    except Exception as exc:                               # pragma: no cover
        return {"seed": int(seed), "name": f"anchor_s{seed}",
                "error": repr(exc), "traceback": traceback.format_exc()}


def _ensemble_stats(values) -> dict:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                     dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    sem = float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else math.nan
    return {"n": int(arr.size), "mean_deg": float(arr.mean()),
            "std_deg": float(arr.std(ddof=1)) if arr.size > 1 else math.nan,
            "sem_deg": sem,
            "min_deg": float(arr.min()), "max_deg": float(arr.max()),
            "ci95_deg": [float(arr.mean() - 1.96 * sem),
                         float(arr.mean() + 1.96 * sem)]
            if arr.size > 1 else None}


def analyse(records: list[dict]) -> dict:
    ok = [r for r in records if "error" not in r]
    out: dict = {"n_specimens": len(records), "n_ok": len(ok)}

    # per-operator ensemble of the *raw* shared-field reading
    out["raw"] = {name: _ensemble_stats([r["angles_deg"].get(name) for r in ok])
                  for name in OPERATORS}
    # bias-corrected ensemble (c = 1.0 intrinsic and c = 0.5 contaminated)
    out["corrected"] = {}
    for name in OPERATORS:
        out["corrected"][name] = {}
        for c in CONTAMINATIONS:
            key = f"{c:.2f}"
            vals = [(r["bias_corrected"].get(name) or {}).get(key, {}).get(
                "corrected_deg") for r in ok]
            out["corrected"][name][key] = _ensemble_stats(vals)

    # operator-robustness verdict on the anchor
    verdict: dict = {"anchor_deg": ANCHOR_DEG, "tolerance": TOLERANCE,
                     "per_operator": {}}
    for name in OPERATORS:
        raw = out["raw"][name]
        cor = (out["corrected"][name].get("1.00") or {})
        entry: dict = {}
        for label, stats in (("raw", raw), ("corrected_c1", cor)):
            mean = stats.get("mean_deg")
            if mean is None:
                entry[label] = {"error": "no data"}
                continue
            err = abs(float(mean) - ANCHOR_DEG) / ANCHOR_DEG
            entry[label] = {"mean_deg": float(mean),
                            "relative_error": float(err),
                            "within_tolerance": bool(err <= TOLERANCE)}
        verdict["per_operator"][name] = entry
    verdict["all_operators_within_tolerance_raw"] = bool(all(
        (verdict["per_operator"][n].get("raw") or {}).get("within_tolerance")
        for n in OPERATORS))
    verdict["all_operators_within_tolerance_corrected"] = bool(all(
        (verdict["per_operator"][n].get("corrected_c1") or {}).get(
            "within_tolerance") for n in OPERATORS))
    # P-R8: every operator, bias-corrected at c = 1, inside the band
    verdict["P_R8"] = {"prediction": ("every operator, bias-corrected at "
                                      "contamination 1.0, lands within 15 % of "
                                      "the 60 deg anchor"),
                       "holds": verdict["all_operators_within_tolerance_corrected"]}
    out["verdict"] = verdict

    # operator spread (raw, per specimen) -- the honest measurement uncertainty
    spreads = [r["operator_spread_deg"] for r in ok
               if r.get("operator_spread_deg") is not None]
    out["operator_spread"] = _ensemble_stats(spreads)

    # R7 secondaries reproduced on the same runs
    out["secondaries"] = {
        "single_increment_radon": _ensemble_stats(
            [r.get("single_increment_radon_deg") for r in ok]),
        "structure_tensor": out["raw"]["structure_tensor"],
        "second_moment": out["raw"]["second_moment"],
    }

    # psi-based laws
    laws = [r["law_test"] for r in ok if r.get("law_test")]
    out["laws"] = {
        "n": len(laws),
        "psi_band_deg": _ensemble_stats([l["psi_band_deg"] for l in laws]),
        "phi_deg": _ensemble_stats([l["phi_deg"] for l in laws]),
        "predictions": {name: _ensemble_stats(
            [l["predictions_deg"][name] for l in laws])
            for name in ("mohr_coulomb", "roscoe", "arthur")},
        "relative_error": {name: _ensemble_stats(
            [l["relative_error"][name] for l in laws])
            for name in ("mohr_coulomb", "roscoe", "arthur")},
        "withheld_note": ("psi-based laws are decided on the *in-band* dilation "
                          "angle (band_local_dilation), not the specimen-global "
                          "one that R2/R5 found unphysical"),
    }
    roscoe_err = out["laws"]["relative_error"]["roscoe"].get("mean_deg")
    out["laws"]["P_R8b"] = {
        "prediction": "Roscoe 45 + psi_band/2 reproduces the measured angle within 15 %",
        "mean_relative_error": roscoe_err,
        "holds": bool(roscoe_err is not None and roscoe_err <= TOLERANCE),
    }
    out["protocol"] = {
        "round": ROUND, "seeds": list(SEEDS),
        "post_peak_strain": POST_PEAK_STRAIN, "strain_cap": STRAIN_CAP,
        "late_window_fraction": LATE_WINDOW_FRACTION,
        "audit_angles_deg": list(AUDIT_ANGLES),
        "contaminations": list(CONTAMINATIONS),
        "operators": list(OPERATORS),
        "anchor_deg": ANCHOR_DEG, "tolerance": TOLERANCE,
        "field": "cumulative late window late[0] -> late[-1], measurement window, k=8",
    }
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figures(records: list[dict], summary: dict, outdir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    paths = []
    ok = [r for r in records if "error" not in r]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = [n for n in OPERATORS]
    width = 0.8 / max(len(ok), 1)
    colours = ["#1f4e79", "#c0504d", "#4f81bd"]
    for i, rec in enumerate(ok):
        xs = np.arange(len(labels)) + i * width - 0.4
        ys = [rec["angles_deg"].get(n) or math.nan for n in labels]
        ax.bar(xs, ys, width=width, label=f"s{rec['seed']}", color=colours[i % 3],
               alpha=0.85)
    ax.axhline(ANCHOR_DEG, color="k", ls="--", lw=1.2, label="anchor 60 deg")
    ax.axhspan(ANCHOR_DEG * (1 - TOLERANCE), ANCHOR_DEG * (1 + TOLERANCE),
               color="grey", alpha=0.15, label="+/-15 % band")
    ax.axhspan(55.0, 60.0, color="tab:green", alpha=0.12,
               label="literature 55-60 deg")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(["Radon\n(production)", "structure tensor\n(D&V route)",
                        "second moment\n(regionprops)"], fontsize=9)
    ax.set_ylabel("band inclination from sigma3 (deg)")
    ax.set_ylim(0, 80)
    ax.set_title("R8: same field, three operators")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p1 = os.path.join(outdir, "r8_operator_comparison.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    paths.append(p1)

    # planted-bias curves
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    for ax, c in zip(axes, (1.0, 0.5)):
        key = f"{c:.2f}"
        for i, name in enumerate(OPERATORS):
            xs, ys = [], []
            for rec in ok:
                for row in rec["planted_audit"]["rows"]:
                    if abs(row["contamination"] - c) > 1e-9:
                        continue
                    err = row.get(name + "_error")
                    if err is None:
                        continue
                    xs.append(row["angle_true_deg"])
                    ys.append(err)
            if not xs:
                continue
            xs = np.asarray(xs)
            ys = np.asarray(ys)
            order = np.argsort(xs)
            # per-angle mean across specimens
            uniq = np.unique(xs[order])
            mean = np.array([ys[xs == u].mean() for u in uniq])
            ax.plot(uniq, mean, marker="o", ms=3, color=colours[i % 3], label=name)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xlabel("injected band angle (deg)")
        ax.set_title(f"planted-band bias, contamination c={c:.2f}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("measured - true (deg), mean over specimens")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    p2 = os.path.join(outdir, "r8_planted_bias.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    paths.append(p2)
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-seed", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    seeds = [args.only_seed] if args.only_seed is not None else list(SEEDS)
    records: list[dict] = []
    if args.resume and os.path.exists(args.out):
        with open(args.out) as handle:
            records = json.load(handle).get("specimens", [])
    done = {r["seed"] for r in records}
    for seed in seeds:
        if args.resume and seed in done:
            continue
        print(f"[R8] running seed {seed} ...", flush=True)
        rec = _safe(seed, args.quick)
        records = [r for r in records if r["seed"] != seed] + [rec]
        records.sort(key=lambda r: r["seed"])
        if "error" in rec:
            print(f"[R8] seed {seed} FAILED: {rec['error']}", flush=True)
            continue
        angles = {k: round(v, 1) if v is not None else None
                  for k, v in rec["angles_deg"].items()}
        print(f"[R8] seed {seed}: stop={rec['dem']['stop_reason']} "
              f"phi={rec['dem']['friction_angle_deg']:.2f} deg "
              f"late_n={rec['dem']['n_late_snapshots']} angles={angles} "
              f"psi_band={((rec.get('band_dilation') or {}).get('psi_band_deg'))} "
              f"wall={rec['dem']['wall_seconds']:.0f}s", flush=True)
        payload = {"round": ROUND, "specimens": records,
                   "summary": analyse(records)}
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=1, default=float)

    for record in records:
        record["r7_reference_deg"] = R7_REFERENCE_DEG.get(int(record["seed"]))
    summary = analyse(records)
    fig_paths: list[str] = []
    if not args.no_figures and records:
        fig_paths = figures(records, summary, ART)
    payload = {"round": ROUND, "specimens": records, "summary": summary,
               "figures": [os.path.relpath(p, ROOT) for p in fig_paths]}
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=1, default=float)
    print(json.dumps({"raw_means": {k: v.get("mean_deg") for k, v in
                                    summary["raw"].items()},
                      "corrected_means": {k: v.get("1.00", {}).get("mean_deg")
                                          for k, v in summary["corrected"].items()},
                      "verdict": summary["verdict"],
                      "laws": summary["laws"].get("relative_error"),
                      "P_R8b": summary["laws"].get("P_R8b")},
                     indent=1, default=float))
    print(f"[R8] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())