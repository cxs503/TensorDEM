"""R4: cumulative-window measurement of the biaxial shear band (predeclared).

Case ``CASE-BIAXIAL-SHEAR-BAND`` / ``CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE``.
Anchor: dense-sand plane-strain biaxial shear-band inclination 55-60 deg
(midpoint ``theta_ref = 60 deg``), Desrues & Viggiani (2004), *Int. J. Numer.
Anal. Methods Geomech.* 28:279-321; tolerance 15 % relative.

Why this round exists (the R3 hand-over, verbatim in its section 6)
------------------------------------------------------------------
R3 passed the 15 % rule with an ensemble of five specimens, but three holes
were left open and are *measurement*, not physics, questions:

* R3 gap 1 -- the **cumulative-window** measurement (first late snapshot ->
  last: same operator, ~6x larger tangential displacement at the same
  positional noise) was implemented but **never measured**.  R3's own noise
  audit says the one-step increment sits near the SNR cliff (catastrophic
  mode switching at >= 0.5 % displacement noise).
* R3 gap 3 -- at p = 200 kPa the one-step band does not resolve and the reading
  jumps ~+20 deg; whether that is an SNR failure or a physical trend was not
  decided.
* R3 statistics-of-one -- single specimens scatter by ~18 deg (55.75 deg at
  p = 100 kPa in R2 vs 72 deg at p = 200 kPa), so the *per-specimen*
  measurement uncertainty was never quantified; only the cross-specimen std.

Predeclared design (fixed before any R4 number was seen)
-------------------------------------------------------
* Specimens: 24 x 48 jittered disc lattice, 35 % freeze bonds, wall friction
  0.55, 4000 consolidation steps, strain rate 0.05 /s to 0.12, snapshots every
  0.01 strain.  **Eight anchor specimens at p = 100 kPa with seeds that are
  disjoint from the R3 ensemble** (R3 used 20261031, 11, 23, 37, 41), i.e. an
  out-of-sample check of the R3 mean, not a re-use of its specimens.
* Primary estimator (declared now): **cumulative Radon focus over the late
  post-peak window** -- ``measure_band`` applied to (first late snapshot, last
  late snapshot).  Rationale: the operator was already shown unbiased (0.00 deg
  bias on this lattice, R2/R3 audits); what limits the case is SNR, and the
  cumulative window multiplies the signal at unchanged positional noise.
* Secondary estimators, reported side by side and never used to rescue the
  verdict: one-step incremental Radon mean (R3 production route), cumulative
  structure tensor (independent operator, experimental route of D&V 2004),
  cumulative post-peak window (width sensitivity).
* Per-specimen measurement uncertainty ``u`` (new in R4): particle bootstrap
  (30 resamples of 80 % of the particles inside the measurement window, fixed
  RNG seed) of the primary estimator; for the one-step route the across-pair
  std is used, as in R3.  Paired per specimen, so the comparison carries no
  cross-specimen variance.
* Sensitivity: three p = 200 kPa specimens (same seeds subset) measured with
  the same primary estimator -- decides R3's gap 3 (SNR vs physics).

Predeclared pass rules and predictions
--------------------------------------
* PASS rule (primary): ``|mean(theta_cumulative) - 60| / 60 <= 0.15`` **and**
  at least 6 of 8 specimens resolved (cumulative ``z > 3`` and
  ``eta2 > 0.2``, the production resolution rule of
  ``run_benchmark.py``).
* P1 (SNR, paired): the cumulative measurement uncertainty is smaller than the
  one-step uncertainty in at least 6 of 8 specimens
  (one-sided Wilcoxon signed-rank at alpha = 0.05 is reported as well).
* P2 (gap 3): with the cumulative estimator at least 2 of 3 p = 200 kPa
  specimens resolve; if they do, the p = 200 kPa inclination is reported as a
  physical datum, if they do not, the case keeps the p = 100 kPa restriction.
* No specimen is excluded.  Failures are reported as failures.

Outputs ``artifacts/biaxial_shear_band/r4_cumulative_ensemble.json``,
per-specimen cache ``artifacts/biaxial_shear_band/r4_cache/<name>.json`` and
``r4_cumulative_ensemble.png``.

Run: ``PYTHONPATH=src python benchmarks/biaxial_shear_band/run_r4_cumulative_ensemble.py --workers 8``
Add ``--only-seed N`` / ``--quick`` for smoke runs.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np

THETA_REF_DEG = 60.0
TOLERANCE = 0.15
BOND_FRACTION = 0.35
WALL_FRICTION = 0.55
CONSOLIDATION_STEPS = 4000
SPACING_M = 1.5                     # 2 x particle radius
Z_RESOLVED = 3.0
ETA2_RESOLVED = 0.2
BOOTSTRAP_RESAMPLES = 30
BOOTSTRAP_FRACTION = 0.8

#: anchors: seeds disjoint from the R3 ensemble (20261031, 11, 23, 37, 41)
ANCHOR_SEEDS = [53, 67, 79, 83, 89, 97, 101, 103]
#: confinement sensitivity, same estimator, p = 200 kPa
P200_SEEDS = [53, 67, 79]


def _configs(quick: bool, only_seed: int | None) -> list[dict]:
    seeds = [only_seed] if only_seed is not None else list(ANCHOR_SEEDS)
    configs = [{"name": f"anchor_s{s}", "axis": "anchor", "seed": int(s),
                "confining_stress": 100.0e3} for s in seeds]
    if not quick and only_seed is None:
        configs += [{"name": f"p200_s{s}", "axis": "confining_pressure",
                     "seed": int(s), "confining_stress": 200.0e3}
                    for s in P200_SEEDS]
    return configs


def _late_snapshots(snapshots: list[dict], peak_strain: float) -> list[dict]:
    final_strain = float(snapshots[-1]["strain"])
    tail_start = peak_strain + 0.4 * max(final_strain - peak_strain, 0.0)
    return [s for s in snapshots if s["strain"] >= tail_start]


def _peak_snapshot(snapshots: list[dict], peak_strain: float) -> dict:
    return min(snapshots, key=lambda s: abs(s["strain"] - peak_strain))


def _measure_pair(x_ref: np.ndarray, x_now: np.ndarray, bounds, want):
    from tensordem.biaxial_shear_band import measure_band
    from tensordem.shear_band_structure_tensor import measure_band_structure_tensor

    out: dict[str, float] = {}
    if "radon" in want:
        rad = measure_band(x_ref, x_now, bounds=bounds)
        out["radon"] = float(rad["folded_angle_deg"])
        out["radon_z"] = float(rad["z_score"])
        out["radon_eta2"] = float(rad["eta2"])
    if "tensor" in want:
        st = measure_band_structure_tensor(x_ref, x_now, bounds=bounds)
        out["tensor"] = float(st["folded_angle_deg"])
        out["tensor_coherence"] = float(st["coherence"])
    return out


def _bootstrap_angle(x_ref: np.ndarray, x_now: np.ndarray, bounds, *, seed: int,
                     operator: str, fraction: float = BOOTSTRAP_FRACTION,
                     resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """Particle-bootstrap uncertainty of one band-angle measurement.

    Resampling *particles* (not snapshots) quantifies how much the measured
    inclination depends on which grains happen to be inside the window, i.e. the
    spatial-sampling part of the measurement error at a fixed displacement
    field.  Identical RNG stream for both operators/slips so comparisons are
    paired.
    """
    from tensordem.biaxial_shear_band import measure_band
    from tensordem.shear_band_structure_tensor import measure_band_structure_tensor

    xmin, xmax, ymin, ymax = bounds
    inside = ((x_ref[:, 0] >= xmin) & (x_ref[:, 0] <= xmax)
              & (x_ref[:, 1] >= ymin) & (x_ref[:, 1] <= ymax))
    idx = np.flatnonzero(inside)
    if idx.size < 40:
        return {"n_resamples": 0, "angles_deg": []}
    rng = np.random.default_rng(seed)
    take = max(int(round(fraction * idx.size)), 30)
    angles: list[float] = []
    for _ in range(int(resamples)):
        sub = rng.choice(idx, size=take, replace=False)
        try:
            if operator == "radon":
                angles.append(float(measure_band(x_ref[sub], x_now[sub])
                                    ["folded_angle_deg"]))
            else:
                angles.append(float(measure_band_structure_tensor(x_ref[sub], x_now[sub])
                                    ["folded_angle_deg"]))
        except Exception:                                # pragma: no cover - reported
            angles.append(math.nan)
    finite = np.asarray([a for a in angles if np.isfinite(a)])
    return {"n_resamples": int(finite.size),
            "mean_deg": float(finite.mean()) if finite.size else math.nan,
            "std_deg": float(finite.std(ddof=1)) if finite.size > 1 else math.nan,
            "angles_deg": [float(a) for a in angles]}


def _run_one(config: dict) -> dict:
    import torch
    torch.set_num_threads(int(os.environ.get("R4_THREADS", "1")))
    from tensordem.biaxial_shear_band import run_biaxial_shear_band

    started = time.time()
    quick = bool(config.get("quick"))
    result = run_biaxial_shear_band(
        confining_stress=config["confining_stress"],
        columns=16 if quick else 24, rows=32 if quick else 48,
        strain_rate=0.05, strain_target=0.08 if quick else 0.12,
        consolidation_steps=600 if quick else CONSOLIDATION_STEPS,
        snapshot_strain=0.01, servo_gain=3.0e-4, servo_velocity_limit=2.0,
        wall_friction=WALL_FRICTION,
        material_overrides={"freeze_bond_fraction": BOND_FRACTION,
                            "seed": int(config["seed"])},
        record_snapshots=True)
    snapshots = result["snapshots"]
    peak_strain = float(result["peak"]["axial_strain"])
    late = _late_snapshots(snapshots, peak_strain)
    pairs = list(zip(late[:-1], late[1:]))

    # ---- one-step (R3 production) route --------------------------------
    inc_radon, inc_tensor, inc_z, inc_eta2 = [], [], [], []
    slips = []
    for ref, now in pairs:
        window = ref["window"] or now["window"]
        try:
            m = _measure_pair(ref["positions"], now["positions"], window,
                              ("radon", "tensor"))
        except Exception as exc:                          # pragma: no cover - reported
            m = {"radon": math.nan, "tensor": math.nan, "radon_z": 0.0,
                 "radon_eta2": 0.0, "tensor_coherence": 0.0,
                 "error": repr(exc)}
        inc_radon.append(m.get("radon", math.nan))
        inc_tensor.append(m.get("tensor", math.nan))
        inc_z.append(m.get("radon_z", 0.0))
        inc_eta2.append(m.get("radon_eta2", 0.0))
        slips.append(float(np.mean(np.linalg.norm(
            now["positions"] - ref["positions"], axis=1))))
    inc_radon_arr = np.asarray([v for v in inc_radon if np.isfinite(v)])

    # ---- cumulative route (primary) ------------------------------------
    cumulative: dict[str, dict] = {}
    if late:
        windows = {
            "late_window": (late[0], late[-1]),
            "post_peak": (_peak_snapshot(snapshots, peak_strain), late[-1]),
        }
        for label, (ref, now) in windows.items():
            window = ref["window"] or now["window"]
            try:
                m = _measure_pair(ref["positions"], now["positions"], window,
                                  ("radon", "tensor"))
            except Exception as exc:                      # pragma: no cover - reported
                m = {"error": repr(exc)}
            slip = float(np.mean(np.linalg.norm(now["positions"] - ref["positions"],
                                                axis=1)))
            entry = {
                "strain_span": float(now["strain"] - ref["strain"]),
                "slip_m": slip,
                "slip_over_spacing": slip / SPACING_M,
                "snr_gain_vs_one_step": (slip / float(np.mean(slips))
                                        if slips and np.mean(slips) > 0 else math.nan),
                **{k: float(v) for k, v in m.items() if k != "error"},
            }
            if "error" in m:
                entry["error"] = m["error"]
            entry["resolved"] = bool(
                float(entry.get("radon_z", 0.0)) > Z_RESOLVED
                and float(entry.get("radon_eta2", 0.0)) > ETA2_RESOLVED)
            if label == "late_window" and "radon" in m:
                for op in ("radon", "tensor"):
                    entry[f"bootstrap_{op}"] = _bootstrap_angle(
                        ref["positions"], now["positions"], window,
                        seed=int(config["seed"]) + 7919, operator=op)
            cumulative[label] = entry

    # field for the figure: cumulative gamma over the late window
    field = None
    if len(late) >= 2:
        from tensordem.biaxial_shear_band import local_shear_strain
        ref, now = late[0]["positions"], late[-1]["positions"]
        xmin, xmax, ymin, ymax = late[-1]["window"]
        inside = ((ref[:, 0] >= xmin) & (ref[:, 0] <= xmax)
                  & (ref[:, 1] >= ymin) & (ref[:, 1] <= ymax))
        gamma = local_shear_strain(ref[inside], now[inside], k=8)
        field = {"x": ref[inside, 0].round(4).tolist(),
                 "y": ref[inside, 1].round(4).tolist(),
                 "gamma": np.asarray(gamma).round(5).tolist(),
                 "bounds": [float(xmin), float(xmax), float(ymin), float(ymax)]}

    inc_rep = float(inc_radon_arr.mean()) if inc_radon_arr.size else math.nan
    inc_std = float(inc_radon_arr.std(ddof=1)) if inc_radon_arr.size > 1 else math.nan
    inc_resolved = bool(
        inc_radon_arr.size >= 3
        and float(np.mean(inc_z)) > Z_RESOLVED
        and float(np.mean(inc_eta2)) > ETA2_RESOLVED
        and int(np.sum(np.abs(inc_radon_arr - np.median(inc_radon_arr)) <= 10.0)) >= 3)

    return {
        "name": config["name"], "axis": config["axis"], "seed": int(config["seed"]),
        "confining_stress": float(config["confining_stress"]),
        "columns": 16 if quick else 24, "rows": 32 if quick else 48,
        "one_step": {
            "theta_deg": inc_rep, "theta_std_deg": inc_std,
            "n_pairs": int(inc_radon_arr.size),
            "tensor_deg": float(np.nanmean(inc_tensor)) if inc_tensor else math.nan,
            "z_mean": float(np.mean(inc_z)) if inc_z else 0.0,
            "eta2_mean": float(np.mean(inc_eta2)) if inc_eta2 else 0.0,
            "slip_over_spacing": (float(np.mean(slips)) / SPACING_M) if slips else math.nan,
            "resolved": inc_resolved,
        },
        "cumulative": cumulative,
        "friction_angle_deg": float(result["peak"]["friction_angle_deg"]),
        "dilation_angle_deg": float(result["peak"]["dilation_angle_deg"]),
        "theory_angles_deg": result.get("theory", {}),
        "inertial_number": result.get("loading", {}).get("inertial_number"),
        "wall_time_s": time.time() - started,
        "gamma_field": field,
    }


def _ensemble_stats(values: list[float]) -> dict:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                     dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if arr.size > 1:
        from scipy import stats
        half = float(stats.t.ppf(0.975, arr.size - 1)) * std / math.sqrt(arr.size)
    else:
        half = math.nan
    return {"n": int(arr.size), "mean_deg": mean, "std_deg": std,
            "ci95_half_width_deg": half,
            "ci95_deg": [mean - half, mean + half] if np.isfinite(half) else None,
            "relative_error": abs(mean - THETA_REF_DEG) / THETA_REF_DEG,
            "within_tolerance": bool(abs(mean - THETA_REF_DEG) / THETA_REF_DEG
                                     <= TOLERANCE)}


def _analyse(results: list[dict]) -> dict:
    anchor = [r for r in results if r["axis"] == "anchor" and "error" not in r]
    sens = [r for r in results if r["axis"] != "anchor" and "error" not in r]

    def cum(r, label, key):
        return (r.get("cumulative") or {}).get(label, {}).get(key)

    primary = _ensemble_stats([cum(r, "late_window", "radon") for r in anchor])
    resolved = sum(1 for r in anchor if cum(r, "late_window", "resolved"))
    post_peak = _ensemble_stats([cum(r, "post_peak", "radon") for r in anchor])
    tensor = [cum(r, "late_window", "tensor") for r in anchor]
    stage1 = {
        "theta_ref_deg": THETA_REF_DEG,
        "tolerance": TOLERANCE,
        "acceptance_band_deg": [THETA_REF_DEG * (1 - TOLERANCE),
                                THETA_REF_DEG * (1 + TOLERANCE)],
        "primary_cumulative_late_window_radon": primary,
        "secondary_cumulative_post_peak_radon": post_peak,
        "secondary_cumulative_late_window_tensor": _ensemble_stats(tensor),
        "one_step_incremental_radon": _ensemble_stats(
            [r["one_step"]["theta_deg"] for r in anchor]),
        "specimens_resolved": int(resolved),
        "specimens_total": int(len(anchor)),
        "pass_rule": ("|mean-60|/60 <= 0.15 (cumulative late-window Radon) and "
                      ">= 6/8 specimens resolved"),
        "passed": bool(primary.get("within_tolerance") and resolved >= 6),
        "theory_angles_deg": {
            "measured_phi_mean": float(np.nanmean([r["friction_angle_deg"]
                                                   for r in anchor])) if anchor else math.nan,
            "measured_psi_mean": float(np.nanmean([r["dilation_angle_deg"]
                                                   for r in anchor])) if anchor else math.nan,
            "cell_example_deg": (anchor[0].get("theory_angles_deg") if anchor else {}),
        },
    }

    # P1: paired measurement uncertainty, cumulative vs one-step
    paired = []
    for r in anchor:
        u_inc = r["one_step"]["theta_std_deg"]
        u_cum = cum(r, "late_window", "bootstrap_radon") or {}
        u_cum = u_cum.get("std_deg") if isinstance(u_cum, dict) else None
        paired.append({"name": r["name"], "u_one_step_deg": u_inc,
                       "u_cumulative_deg": u_cum,
                       "snr_gain": cum(r, "late_window", "snr_gain_vs_one_step")})
    finite = [(p["u_one_step_deg"], p["u_cumulative_deg"]) for p in paired
              if p["u_one_step_deg"] is not None and p["u_cumulative_deg"] is not None
              and np.isfinite(p["u_one_step_deg"]) and np.isfinite(p["u_cumulative_deg"])]
    improved = sum(1 for a, b in finite if b < a)
    p1 = {"per_specimen": paired, "n_pairs": len(finite),
          "n_improved": int(improved),
          "fraction_improved": (improved / len(finite)) if finite else math.nan,
          "p1_holds": bool(len(finite) >= 6 and improved >= 6),
          "wilcoxon_p": None}
    if len(finite) >= 6:
        try:
            from scipy import stats as _stats
            p1["wilcoxon_p"] = float(_stats.wilcoxon(
                [a for a, _ in finite], [b for _, b in finite],
                alternative="greater").pvalue)
        except Exception:                                # pragma: no cover - reported
            p1["wilcoxon_p"] = None
    stage1["p1_snr_paired"] = p1

    # P2: confinement at p = 200 kPa with the cumulative estimator
    p2 = {"specimens": [
        {"name": r["name"], "p_kPa": r["confining_stress"] / 1e3,
         "theta_cumulative_radon_deg": cum(r, "late_window", "radon"),
         "theta_one_step_radon_deg": r["one_step"]["theta_deg"],
         "resolved_cumulative": cum(r, "late_window", "resolved"),
         "resolved_one_step": r["one_step"]["resolved"],
         "snr_gain": cum(r, "late_window", "snr_gain_vs_one_step"),
         "relative_error": (abs(cum(r, "late_window", "radon") - THETA_REF_DEG)
                            / THETA_REF_DEG
                            if cum(r, "late_window", "radon") is not None else None)}
        for r in sens]}
    p2["n_resolved_cumulative"] = int(sum(
        1 for s in p2["specimens"] if s["resolved_cumulative"]))
    p2["n_resolved_one_step"] = int(sum(
        1 for s in p2["specimens"] if s["resolved_one_step"]))
    p2["p2_holds"] = bool(p2["n_resolved_cumulative"] >= 2)
    p2["ensemble_cumulative"] = _ensemble_stats(
        [s["theta_cumulative_radon_deg"] for s in p2["specimens"]])
    stage1["p2_confinement"] = p2
    return stage1


def _figure(results: list[dict], verdict: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    anchor = [r for r in results if r["axis"] == "anchor" and "error" not in r]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    ax = axes[0][0]
    x = np.arange(len(anchor))
    _nan = lambda v: (float(v) if v is not None else math.nan)
    cum = [_nan((r.get("cumulative") or {}).get("late_window", {}).get("radon"))
           for r in anchor]
    inc = [_nan(r["one_step"]["theta_deg"]) for r in anchor]
    ax.bar(x - 0.2, cum, 0.4, label="cumulative late window (primary)")
    ax.bar(x + 0.2, inc, 0.4, label="one-step incremental (R3)")
    ax.axhline(THETA_REF_DEG, color="k", lw=1.2, label="anchor 60°")
    ax.axhspan(THETA_REF_DEG * (1 - TOLERANCE), THETA_REF_DEG * (1 + TOLERANCE),
               color="k", alpha=0.10, label="±15 %")
    mean_primary = verdict["primary_cumulative_late_window_radon"].get("mean_deg")
    if mean_primary is not None and np.isfinite(mean_primary):
        ax.axhline(mean_primary, color="C0", ls="--", lw=1.2,
                   label=f"ensemble mean {mean_primary:.2f}°")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r["seed"]) for r in anchor], rotation=30)
    ax.set_xlabel("RidgeMaterial seed (out-of-sample vs R3)")
    ax.set_ylabel("band inclination from σ₃ (deg)")
    ax.set_title("(a) per-specimen inclination, two measurement windows")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    paired = verdict["p1_snr_paired"]["per_specimen"]
    lab = [p["name"].replace("anchor_s", "s") for p in paired]
    ax.bar(np.arange(len(paired)) - 0.2,
           [_nan(p["u_one_step_deg"]) for p in paired], 0.4, label="one-step σ")
    ax.bar(np.arange(len(paired)) + 0.2,
           [_nan(p["u_cumulative_deg"]) for p in paired], 0.4,
           label="cumulative bootstrap σ")
    ax.set_xticks(np.arange(len(paired)))
    ax.set_xticklabels(lab, rotation=30)
    ax.set_ylabel("measurement uncertainty (deg)")
    ax.set_title(f"(b) paired uncertainty, improved {verdict['p1_snr_paired']['n_improved']}"
                 f"/{verdict['p1_snr_paired']['n_pairs']}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    best = max(anchor, key=lambda r: (r.get("cumulative") or {})
               .get("late_window", {}).get("radon_z", -math.inf))
    field = best.get("gamma_field")
    if field:
        sc = ax.scatter(field["x"], field["y"], c=field["gamma"], s=6, cmap="inferno")
        fig.colorbar(sc, ax=ax, label="cumulative local γ (late window)")
        theta = math.radians(float((best.get("cumulative") or {})
                                   .get("late_window", {}).get("radon", math.nan)))
        xs = np.asarray(field["x"])
        ys = np.asarray(field["y"])
        xc, yc = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
        span = max(ys.max() - ys.min(), xs.max() - xs.min())
        ax.plot([xc - span, xc + span],
                [yc - span * math.sin(theta), yc + span * math.sin(theta)],
                "c--", lw=1.4, label=f"measured band {math.degrees(theta):.1f}°")
        ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.set_title(f"(c) cumulative γ field, seed {best['seed']}")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    ax = axes[1][1]
    sens = verdict["p2_confinement"]["specimens"]
    labels = [f"p={s['p_kPa']:.0f}\nseed {s['name'].split('_s')[-1]}" for s in sens]
    xx = np.arange(len(sens))
    if not sens:
        ax.axis("off")
        ax.set_title("(d) p = 200 kPa: not run in this configuration")
        fig.suptitle("R4 — cumulative-window shear-band measurement, 24×48 biaxial cell")
        fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
        return
    ax.bar(xx - 0.2, [_nan(s["theta_cumulative_radon_deg"]) for s in sens], 0.4,
           label="cumulative (primary)")
    ax.bar(xx + 0.2, [_nan(s["theta_one_step_radon_deg"]) for s in sens], 0.4,
           label="one-step (R3)")
    ax.axhline(THETA_REF_DEG, color="k", lw=1.2)
    ax.axhspan(THETA_REF_DEG * (1 - TOLERANCE), THETA_REF_DEG * (1 + TOLERANCE),
               color="k", alpha=0.10)
    ax.set_xticks(xx)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("band inclination (deg)")
    ax.set_title(f"(d) p = 200 kPa: resolved {verdict['p2_confinement']['n_resolved_cumulative']}"
                 f"/{len(sens)} (cumulative) vs "
                 f"{verdict['p2_confinement']['n_resolved_one_step']}/{len(sens)} (one-step)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("R4 — cumulative-window shear-band measurement, 24×48 biaxial cell")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default="artifacts/biaxial_shear_band/r4_cumulative_ensemble.json")
    parser.add_argument("--figure", default="artifacts/biaxial_shear_band/r4_cumulative_ensemble.png")
    parser.add_argument("--cache", default="artifacts/biaxial_shear_band/r4_cache")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only-seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="reuse per-specimen cache JSON when present")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    if args.quick:
        global BOOTSTRAP_RESAMPLES
        BOOTSTRAP_RESAMPLES = 3

    started = time.time()
    configs = _configs(args.quick, args.only_seed)
    results: list[dict] = []

    def _cached(config):
        path = os.path.join(args.cache, config["name"] + ".json")
        if args.resume and os.path.exists(path):
            with open(path) as handle:
                return json.load(handle)
        return None

    def _one(config):
        config = {**config, "quick": bool(args.quick)}
        cached = _cached(config)
        if cached is not None:
            return cached
        result = _run_one(config)
        with open(os.path.join(args.cache, config["name"] + ".json"), "w") as handle:
            json.dump(result, handle, default=float)
        return result

    if args.workers > 1 and len(configs) > 1:
        with mp.get_context("spawn").Pool(min(args.workers, len(configs))) as pool:
            for result in pool.imap_unordered(_one, configs):
                results.append(result)
                print(f"[done] {result['name']} {result.get('error', '')}", flush=True)
    else:
        for config in configs:
            try:
                results.append(_one(config))
            except Exception as exc:                     # pragma: no cover - reported
                results.append({"name": config["name"], "axis": config["axis"],
                                "seed": config["seed"],
                                "confining_stress": config["confining_stress"],
                                "error": repr(exc),
                                "traceback": traceback.format_exc()})
            print(f"[done] {results[-1]['name']} {results[-1].get('error', '')}", flush=True)

    results.sort(key=lambda r: (r["axis"], r["name"]))
    verdict = _analyse(results)
    payload = {
        "case": "CASE-BIAXIAL-SHEAR-BAND",
        "round": "R4",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "specification": {
            "cell": "plane-strain biaxial, 24 x 48 disc lattice",
            "confining_stress_Pa": [100.0e3, 200.0e3],
            "wall_friction": WALL_FRICTION,
            "consolidation_steps": CONSOLIDATION_STEPS,
            "freeze_bond_fraction": BOND_FRACTION,
            "strain_rate_per_s": 0.05, "strain_target": 0.12,
            "anchor_seeds": ANCHOR_SEEDS, "p200_seeds": P200_SEEDS,
            "anchor": "Desrues & Viggiani 2004 IJNAMG 28:279-321, dense Hostun "
                      "sand plane-strain biaxial, shear band 55-60 deg (midpoint 60)",
            "tolerance": TOLERANCE,
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES,
                          "fraction": BOOTSTRAP_FRACTION},
        },
        "verdict": verdict,
        "specimens": results,
        "wall_time_s": time.time() - started,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    try:
        _figure(results, verdict, args.figure)
    except Exception as exc:                             # pragma: no cover - reported
        print(f"[warn] figure failed: {exc!r}", flush=True)
    print(json.dumps({k: verdict[k] for k in
                      ("primary_cumulative_late_window_radon", "one_step_incremental_radon",
                       "specimens_resolved", "passed")}, indent=2, default=float))
    print("p1:", json.dumps({k: verdict["p1_snr_paired"][k] for k in
                             ("n_improved", "n_pairs", "p1_holds", "wilcoxon_p")},
                            default=float))
    print("p2:", json.dumps({k: verdict["p2_confinement"][k] for k in
                             ("n_resolved_cumulative", "n_resolved_one_step", "p2_holds")},
                            default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())