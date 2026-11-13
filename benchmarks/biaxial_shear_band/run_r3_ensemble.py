"""R3 ensemble reproduction of CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE.

Mission context
---------------
R1 delivered the measurement operator and a single-specimen reading, which was
26 % below the literature anchor; R2 found and fixed an *estimator* defect
(per-axis rescaling in the Radon projection, up to 18 deg bias on this elongated
lattice) and one run then landed at 55.75 deg (7.1 %), while a second
confining pressure landed at 72 deg (20 %).  The remaining problem is therefore
**not** the mean bias of the operator on synthetic data (0.00 deg on the case
lattice) but the run-to-run scatter of a single chaotic DEM realisation.

Design (predeclared before inspecting the R3 numbers)
-----------------------------------------------------
* Decision: does the production plane-strain biaxial cell (24 x 48, p = 100 kPa,
  wall friction 0.55, 4000 consolidation steps) reproduce the dense-sand
  shear-band inclination anchor ``theta_ref = 60 deg``
  (Desrues & Viggiani 2004, 55-60 deg, midpoint 60) within the owner 15 %
  tolerance?
* Alternatives: (a) single-specimen production reading (R1/R2 practice) --
  rejected as the *unit of analysis* because the DEM realisation is chaotic;
  (b) ensemble of independent specimens (different ``RidgeMaterial.seed`` ->
  different lattice jitter and different freeze-bond subset), the material's
  characteristic inclination being the ensemble mean; (c) an independent second
  operator (structure tensor on the gridded shear-strain field, the
  experimental route of Desrues & Viggiani) applied to the **same** snapshots,
  so operator dependence is measured rather than assumed.
* Hypothesis: the ensemble mean over ``N = 5`` independent specimens lies within
  15 % of 60 deg (51-69 deg) with a 95 % CI half-width below the tolerance band.
* Metrics: per-specimen late-window inclination (mean over the resolved tail
  snapshot pairs), ensemble mean, sample std, 95 % CI, per-specimen resolved
  flag (>= 3 tail snapshots within 10 deg), Radon ``eta2`` and ``z``,
  structure-tensor ``coherence``, servo pressure offset, wall time.
* Predeclared pass rule: ``|mean - 60| / 60 <= 0.15`` **and** at least 4 of 5
  specimens resolved.  No exclusion of specimens; failures are reported.
* Sensitivity (two axes, both required by the mission): confining pressure
  (100 -> 200 kPa, 2 specimens) and specimen resolution (24 x 48 -> 32 x 64,
  1 specimen).  These are *not* part of the pass/fail rule, they bound the
  provenance of the number.
* Environment: production module ``tensordem.biaxial_shear_band`` (R2 estimator,
  ``binning="equal_count"``), material ``RidgeMaterial`` with
  ``freeze_bond_fraction = 0.35``, one torch thread per worker.

Outputs ``artifacts/biaxial_shear_band/r3_ensemble.json`` and
``artifacts/biaxial_shear_band/r3_ensemble.png``.

Run: ``PYTHONPATH=src python benchmarks/biaxial_shear_band/run_r3_ensemble.py --workers 5``
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
CONSOLIDATION_STEPS = 4000          # anchor specification
BASE_SEED = 20261031                # RidgeMaterial default

#: independent specimens of the anchor cell (seed -> jitter + bond subset)
ANCHOR_SEEDS = [BASE_SEED, 11, 23, 37, 41]
#: sensitivity: confinement (p = 200 kPa) and resolution (32 x 64)
SENSITIVITY = [
    {"name": "sens_p200_s11", "columns": 24, "rows": 48,
     "confining_stress": 200.0e3, "seed": 11, "axis": "confining_pressure"},
    {"name": "sens_p200_s23", "columns": 24, "rows": 48,
     "confining_stress": 200.0e3, "seed": 23, "axis": "confining_pressure"},
    {"name": "sens_r32x64_s31", "columns": 32, "rows": 64,
     "confining_stress": 100.0e3, "seed": BASE_SEED, "axis": "resolution"},
]


def _config_list() -> list[dict]:
    anchor = [{"name": f"anchor_s{seed}", "columns": 24, "rows": 48,
               "confining_stress": 100.0e3, "seed": seed, "axis": "anchor",
               "ensemble": True}
              for seed in ANCHOR_SEEDS]
    for spec in SENSITIVITY:
        anchor.append({**spec, "ensemble": False})
    return anchor


def _late_pairs(snapshots: list[dict], peak_strain: float) -> list[tuple[dict, dict]]:
    final_strain = float(snapshots[-1]["strain"])
    tail_start = peak_strain + 0.4 * max(final_strain - peak_strain, 0.0)
    late = [s for s in snapshots if s["strain"] >= tail_start]
    return list(zip(late[:-1], late[1:]))


def _run_one(config: dict) -> dict:
    import torch
    torch.set_num_threads(int(os.environ.get("R3_THREADS", "1")))
    from tensordem.biaxial_shear_band import local_shear_strain, measure_band, \
        run_biaxial_shear_band
    from tensordem.shear_band_structure_tensor import measure_band_structure_tensor

    started = time.time()
    result = run_biaxial_shear_band(
        confining_stress=config["confining_stress"],
        columns=config["columns"], rows=config["rows"],
        strain_rate=0.05, strain_target=0.12,
        consolidation_steps=CONSOLIDATION_STEPS, snapshot_strain=0.01,
        servo_gain=3.0e-4, servo_velocity_limit=2.0,
        wall_friction=WALL_FRICTION,
        material_overrides={"freeze_bond_fraction": BOND_FRACTION,
                            "seed": int(config["seed"])},
        record_snapshots=True)
    snapshots = result["snapshots"]
    peak_strain = float(result["peak"]["axial_strain"])
    pairs = _late_pairs(snapshots, peak_strain)

    radon, tensor, z_scores, eta2, coherence, aspects = [], [], [], [], [], []
    increments = []
    for ref, now in pairs:
        window = ref["window"] or now["window"]
        rad = measure_band(ref["positions"], now["positions"], bounds=window)
        st = measure_band_structure_tensor(ref["positions"], now["positions"],
                                          bounds=window)
        radon.append(float(rad["folded_angle_deg"]))
        tensor.append(float(st["folded_angle_deg"]))
        z_scores.append(float(rad["z_score"]))
        eta2.append(float(rad["eta2"]))
        coherence.append(float(st["coherence"]))
        pos = ref["positions"]
        xmin, xmax, ymin, ymax = window
        inside = ((pos[:, 0] >= xmin) & (pos[:, 0] <= xmax)
                  & (pos[:, 1] >= ymin) & (pos[:, 1] <= ymax))
        aspects.append(float(np.ptp(pos[inside, 0]) / np.ptp(pos[inside, 1])))
        increments.append(float(np.mean(np.linalg.norm(
            now["positions"] - ref["positions"], axis=1))))
    # cumulative (one-step) measurement over the late window and over the whole
    # post-peak record: same operator, 5-6x larger tangential displacement, so a
    # much better signal-to-noise ratio (see run_r3_estimator_audit.py)
    cumulative = {}
    for label, start in (("late_window", pairs[0][0] if pairs else None),
                         ("post_peak", [s for s in snapshots
                                        if abs(s["strain"] - peak_strain) < 1e-9])):
        if start is None or not pairs:
            continue
        ref = start if isinstance(start, dict) else start[0]
        now = pairs[-1][1]
        window = ref["window"] or now["window"]
        cum_rad = measure_band(ref["positions"], now["positions"], bounds=window)
        cum_st = measure_band_structure_tensor(ref["positions"], now["positions"],
                                               bounds=window)
        cumulative[label] = {
            "strain_span": float(now["strain"] - ref["strain"]),
            "radon_deg": float(cum_rad["folded_angle_deg"]),
            "radon_eta2": float(cum_rad["eta2"]),
            "radon_z": float(cum_rad["z_score"]),
            "tensor_deg": float(cum_st["folded_angle_deg"]),
            "tensor_coherence": float(cum_st["coherence"]),
            "slip_over_spacing": float(np.mean(np.linalg.norm(
                now["positions"] - ref["positions"], axis=1)) / 1.5),
        }

    theta_prod = float(result["band"]["angle_deg"])
    theta_prod_std = float(result["band"]["angle_std_deg"])
    resolved_prod = bool(result["band"]["resolved"])

    # figure field: cumulative strain over the late window, last snapshot
    field = None
    if pairs:
        ref = pairs[0][0]["positions"]
        now = pairs[-1][1]["positions"]
        window = pairs[-1][1]["window"]
        xmin, xmax, ymin, ymax = window
        inside = ((ref[:, 0] >= xmin) & (ref[:, 0] <= xmax)
                  & (ref[:, 1] >= ymin) & (ref[:, 1] <= ymax))
        gamma = local_shear_strain(ref[inside], now[inside], k=8)
        field = {"x": ref[inside, 0].tolist(), "y": ref[inside, 1].tolist(),
                 "gamma": gamma.tolist()}

    npz_path = config.get("npz")
    if npz_path:
        np.savez_compressed(
            npz_path,
            strains=np.array([s["strain"] for s in snapshots]),
            positions=np.array([s["positions"] for s in snapshots]),
            peak_strain=peak_strain)
    sigma_h = result["stress_path"]["sigma_h"]
    return {
        "name": config["name"], "axis": config["axis"],
        "ensemble_member": bool(config.get("ensemble", False)),
        "seed": int(config["seed"]),
        "confining_stress": float(config["confining_stress"]),
        "columns": int(config["columns"]), "rows": int(config["rows"]),
        "particles": int(result["specimen"]["particles"]),
        "bonds": int(result["specimen"]["bonds"]),
        "inertial_number": float(result["loading"]["inertial_number"]),
        "susceptibility_index": float(result["loading"]["steps"]),
        "peak_axial_strain": peak_strain,
        "final_axial_strain": float(result["loading"]["final_axial_strain"]),
        "theta_production_deg": theta_prod,
        "theta_production_std_deg": theta_prod_std,
        "resolved_production": resolved_prod,
        "late_pairs": len(pairs),
        "theta_radon_mean_deg": float(np.mean(radon)) if radon else math.nan,
        "theta_radon_std_deg": float(np.std(radon)) if radon else math.nan,
        "theta_tensor_mean_deg": float(np.mean(tensor)) if tensor else math.nan,
        "theta_tensor_std_deg": float(np.std(tensor)) if tensor else math.nan,
        "radon_eta2_mean": float(np.mean(eta2)) if eta2 else math.nan,
        "radon_z_mean": float(np.mean(z_scores)) if z_scores else math.nan,
        "tensor_coherence_mean": float(np.mean(coherence)) if coherence else math.nan,
        "window_aspect_mean": float(np.mean(aspects)) if aspects else math.nan,
        "increment_displacement_over_spacing": float(np.mean(increments) / 1.5)
        if increments else math.nan,
        "cumulative": cumulative,
        "phi_deg": float(result["peak"]["friction_angle_deg"]),
        "psi_deg": float(result["peak"]["dilation_angle_deg"]),
        "broken_bonds": int(result["broken_bonds"]),
        "servo_relative_offset": float(np.mean(sigma_h) / config["confining_stress"] - 1.0),
        "wall_time_s": time.time() - started,
        "gamma_field": field,
    }


def _ensemble_stats(values: list[float]) -> dict:
    """mean, sample std, 95 % CI half-width (t, N-1 dof) and relative error."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if arr.size > 1:
        from scipy import stats
        tcrit = float(stats.t.ppf(0.975, arr.size - 1))
    else:
        tcrit = math.nan
    half = tcrit * std / math.sqrt(arr.size) if arr.size > 1 else math.nan
    return {"n": int(arr.size), "mean_deg": mean, "std_deg": std,
            "ci95_half_width_deg": half,
            "ci95_deg": [mean - half, mean + half] if np.isfinite(half) else None,
            "relative_error": abs(mean - THETA_REF_DEG) / THETA_REF_DEG,
            "within_tolerance": bool(abs(mean - THETA_REF_DEG) / THETA_REF_DEG <= TOLERANCE)}


def _analyse(results: list[dict]) -> dict:
    anchor = [r for r in results if r.get("ensemble_member") and "error" not in r]
    radon = [r["theta_radon_mean_deg"] for r in anchor]
    tensor = [r["theta_tensor_mean_deg"] for r in anchor]
    prod = [r["theta_production_deg"] for r in anchor]
    resolved = sum(1 for r in anchor if r["resolved_production"])
    stats_radon = _ensemble_stats(radon)
    stats_tensor = _ensemble_stats(tensor)
    pair_diff = [abs(a - b) for a, b in zip(radon, tensor)]
    verdict = {
        "theta_ref_deg": THETA_REF_DEG,
        "tolerance": TOLERANCE,
        "acceptance_band_deg": [THETA_REF_DEG * (1 - TOLERANCE), THETA_REF_DEG * (1 + TOLERANCE)],
        "ensemble_radon": stats_radon,
        "ensemble_tensor": stats_tensor,
        "ensemble_production_headline": _ensemble_stats(prod),
        "specimens_resolved": resolved,
        "specimens_total": len(anchor),
        "operator_mean_abs_difference_deg": float(np.mean(pair_diff)) if pair_diff else math.nan,
        "operator_max_abs_difference_deg": float(np.max(pair_diff)) if pair_diff else math.nan,
        "pass_rule": "|mean-60|/60 <= 0.15 and >= 4/5 specimens resolved",
        "passed_radon": bool(stats_radon.get("within_tolerance")
                             and resolved >= 4),
        "passed_tensor": bool(stats_tensor.get("within_tolerance")
                              and resolved >= 4),
    }
    for label in ("late_window", "post_peak"):
        key_r = [r.get("cumulative", {}).get(label, {}).get("radon_deg", math.nan)
                 for r in anchor]
        key_t = [r.get("cumulative", {}).get(label, {}).get("tensor_deg", math.nan)
                 for r in anchor]
        verdict[f"cumulative_{label}_radon"] = _ensemble_stats(key_r)
        verdict[f"cumulative_{label}_tensor"] = _ensemble_stats(key_t)
    sens = [r for r in results if not r.get("ensemble_member") and "error" not in r]
    verdict["sensitivity"] = [
        {"name": r["name"], "axis": r["axis"],
         "confining_stress": r["confining_stress"],
         "columns": r["columns"], "rows": r["rows"],
         "theta_radon_deg": r["theta_radon_mean_deg"],
         "theta_tensor_deg": r["theta_tensor_mean_deg"],
         "cumulative_late_window_radon_deg":
             (r.get("cumulative") or {}).get("late_window", {}).get("radon_deg"),
         "resolved": r["resolved_production"],
         "relative_error_radon": abs(r["theta_radon_mean_deg"] - THETA_REF_DEG) / THETA_REF_DEG}
        for r in sens]
    return verdict


def _figure(results: list[dict], verdict: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    anchor = [r for r in results if r.get("ensemble_member") and "error" not in r]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    labels = [r["name"].replace("anchor_", "") for r in anchor]
    x = np.arange(len(anchor))
    axes[0].bar(x - 0.2, [r["theta_radon_mean_deg"] for r in anchor], 0.4,
                yerr=[r["theta_radon_std_deg"] for r in anchor],
                capsize=3, label="Radon focus (production)")
    axes[0].bar(x + 0.2, [r["theta_tensor_mean_deg"] for r in anchor], 0.4,
                yerr=[r["theta_tensor_std_deg"] for r in anchor],
                capsize=3, label="structure tensor")
    axes[0].axhline(THETA_REF_DEG, color="k", ls="-", lw=1.2, label="anchor 60°")
    axes[0].axhspan(THETA_REF_DEG * (1 - TOLERANCE), THETA_REF_DEG * (1 + TOLERANCE),
                    color="k", alpha=0.10, label="±15 %")
    stats = verdict["ensemble_radon"]
    if np.isfinite(stats.get("ci95_half_width_deg", math.nan)):
        axes[0].axhline(stats["mean_deg"], color="C0", ls="--", lw=1.2,
                        label=f"ensemble mean {stats['mean_deg']:.2f}°")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30)
    axes[0].set_ylabel("band inclination from σ₃ (deg)")
    axes[0].set_title("(a) ensemble of independent specimens")
    axes[0].legend(fontsize=7, loc="lower right")
    axes[0].grid(alpha=0.3)

    axes[1].scatter([r["theta_radon_mean_deg"] for r in anchor],
                    [r["theta_tensor_mean_deg"] for r in anchor], s=45)
    lo, hi = 40, 80
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[1].axvspan(THETA_REF_DEG * (1 - TOLERANCE), THETA_REF_DEG * (1 + TOLERANCE),
                    color="green", alpha=0.12)
    axes[1].axhspan(THETA_REF_DEG * (1 - TOLERANCE), THETA_REF_DEG * (1 + TOLERANCE),
                    color="green", alpha=0.12)
    axes[1].set_xlim(lo, hi); axes[1].set_ylim(lo, hi)
    axes[1].set_xlabel("Radon focus (deg)"); axes[1].set_ylabel("structure tensor (deg)")
    axes[1].set_title("(b) operator agreement on identical snapshots")
    axes[1].grid(alpha=0.3)

    best = max(anchor, key=lambda r: r.get("radon_z_mean", -math.inf))
    if best.get("gamma_field"):
        f = best["gamma_field"]
        scatter = axes[2].scatter(f["x"], f["y"], c=f["gamma"], s=6, cmap="inferno")
        fig.colorbar(scatter, ax=axes[2], label="local max shear strain γ")
        theta = math.radians(best["theta_radon_mean_deg"])
        xc = 0.5 * (min(f["x"]) + max(f["x"]))
        yy = np.linspace(min(f["y"]), max(f["y"]), 10)
        axes[2].plot(xc + (yy - yy.mean()) / math.tan(theta) * 0 - (yy - yy.mean()) / math.tan(theta),
                     yy, "c-", lw=1.2, label=f"band line {best['theta_radon_mean_deg']:.1f}°")
        axes[2].legend(fontsize=7)
        axes[2].set_aspect("equal")
        axes[2].set_title(f"(c) late-window γ field — {best['name']}")
        axes[2].set_xlabel("x (m)"); axes[2].set_ylabel("y (m)")

    fig.suptitle("CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE — R3 ensemble reproduction "
                 "(24×48, p=100 kPa, μ=0.55, 4000 consolidation steps)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/biaxial_shear_band/r3_ensemble.json")
    parser.add_argument("--figure", default="artifacts/biaxial_shear_band/r3_ensemble.png")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--cache-dir", default="artifacts/biaxial_shear_band/r3_cache")
    parser.add_argument("--only", default=None,
                        help="comma separated specimen names to (re)run")
    args = parser.parse_args(argv)

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if here not in sys.path:
        sys.path.insert(0, os.path.join(here, "src"))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    cache = os.path.abspath(args.cache_dir)
    os.makedirs(cache, exist_ok=True)
    configs = _config_list()
    if args.only:
        wanted = {v.strip() for v in args.only.split(",") if v.strip()}
        configs = [c for c in configs if c["name"] in wanted]
    pending = []
    results: list[dict] = []
    for config in configs:
        path = os.path.join(cache, config["name"] + ".json")
        if os.path.exists(path):
            with open(path) as handle:
                results.append(json.load(handle))
        else:
            pending.append({**config,
                            "npz": os.path.join(cache, config["name"] + "_snapshots.npz")})
    print(f"[cache] {len(results)} cached, {len(pending)} pending", flush=True)
    started = time.time()
    if args.workers > 1 and pending:
        with mp.get_context("fork").Pool(min(args.workers, len(pending))) as pool:
            for result in pool.imap_unordered(_run_one, pending):
                results.append(result)
                with open(os.path.join(cache, result["name"] + ".json"), "w") as handle:
                    json.dump(result, handle, default=float)
                print(f"[done] {result['name']} radon={result['theta_radon_mean_deg']:.2f} "
                      f"tensor={result['theta_tensor_mean_deg']:.2f} "
                      f"resolved={result['resolved_production']} "
                      f"({result['wall_time_s']:.0f}s)", flush=True)
    else:
        for config in pending:
            try:
                result = _run_one(config)
                results.append(result)
                with open(os.path.join(cache, result["name"] + ".json"), "w") as handle:
                    json.dump(result, handle, default=float)
            except Exception as exc:                     # pragma: no cover - reported
                results.append({"name": config["name"], "error": repr(exc),
                                "traceback": traceback.format_exc(),
                                "ensemble_member": bool(config.get("ensemble")),
                                "axis": config["axis"]})
            print(f"[done] {results[-1]['name']} {results[-1].get('error', '')}", flush=True)

    results.sort(key=lambda r: (r["axis"], r["name"]))
    verdict = _analyse(results)
    payload = {
        "case": "CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE",
        "round": "R3",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "specification": {
            "cell": "plane-strain biaxial, 24 x 48 disc lattice (32 x 64 sensitivity)",
            "confining_stress_Pa": 100.0e3,
            "wall_friction": WALL_FRICTION,
            "consolidation_steps": CONSOLIDATION_STEPS,
            "freeze_bond_fraction": BOND_FRACTION,
            "strain_rate_per_s": 0.05,
            "strain_target": 0.12,
            "servo_gain": 3.0e-4,
            "seeds": ANCHOR_SEEDS,
            "anchor": "Desrues & Viggiani 2004 IJNAMG 28:279-321 dense Hostun sand "
                      "shear band 55-60 deg (midpoint 60)",
            "tolerance": TOLERANCE,
        },
        "verdict": verdict,
        "specimens": results,
        "ensemble_wall_time_s": time.time() - started,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    _figure(results, verdict, args.figure)
    print(json.dumps({k: v for k, v in verdict.items() if k != "sensitivity"}, indent=2,
                     default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())