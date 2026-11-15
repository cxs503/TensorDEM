"""Archived-history pass of the m21945 sampling audit (no DEM needed).

The pre-registered dense runs (``sample_stride = 1``, ``cj3791``) are the primary
evidence.  While they queue, the *same* estimator machinery is applied to the
already-archived trajectories of ``column_collapse_deep_refinement_m20873``
(``sample_stride = 5``), which answers two of the three pre-registered questions
on real data today:

* **H2** -- the definition-induced spread of ``t`` inside one trajectory
  (13 pre-registered definitions) against the 15 % card tolerance;
* **H1-partial** -- the sampling bias when that history is decimated *further*
  (5 -> 10 -> 20 -> 40 integration steps), i.e. exactly the ladder a coarser
  storage protocol would have produced.  The dense-vs-archived comparison
  (stride 1 vs 5) still needs the new runs;
* **H4** -- the analytic one-sample invariant on real (monotone) front histories.

Outputs ``docs/artifacts/column_arrival_sampling_m21945/archive_*.{json,csv,png}``.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensordem.granular_column.arrival_sampling import (  # noqa: E402
    PRIMARY_KEY, adjudicate, arrival_grid, definition_band, definition_keys,
    emplacement_errors, sampling_bound_violations, stride_scan)

OUT = os.path.join(_ROOT, "docs", "artifacts", "column_arrival_sampling_m21945")
ARCHIVE = os.path.join(_ROOT, "docs", "artifacts",
                       "column_collapse_deep_refinement_m20873", "cases")
TOL = 0.15
#: Decimation factors applied *on top of* the archived stride 5.
FACTORS = (1, 2, 4, 8)


def load() -> list[dict]:
    cases = []
    for path in sorted(glob.glob(os.path.join(ARCHIVE, "*.json"))):
        with open(path) as fh:
            case = json.load(fh)
        hist = case.get("history")
        if not hist or "front" not in hist:
            continue
        cases.append({"path": path, "case": case,
                      "time": np.asarray(hist["time"], dtype=float),
                      "front": np.asarray(hist["front"], dtype=float)})
    return cases


def native_stride(case: dict, time: np.ndarray) -> int:
    sig = case.get("protocol", {}).get("sample_stride")
    if sig:
        return int(sig)
    dt = float(case["dt"])
    return int(round(float(np.median(np.diff(time))) / dt))


def analyze() -> dict:
    rows, per_case = [], {}
    for item in load():
        case, t, f = item["case"], item["time"], item["front"]
        cid = case["case_id"]
        radius = float(case["column_radius_measured"])
        diameter = 2.0 * float(case["protocol"]["radius"])
        stride0 = native_stride(case, t)
        ref_t = float(case["reference"]["measured"]["emplacement_time"])
        grid = arrival_grid(t, f, radius=radius, grain_diameter=diameter, stride=1)
        band = definition_band(grid)
        adj = adjudicate(emplacement_errors(grid, ref_t), tolerance=TOL)
        scan = stride_scan(t, f, radius=radius, grain_diameter=diameter,
                           strides=FACTORS)
        violations = sampling_bound_violations(scan)
        entry = scan["definitions"][PRIMARY_KEY]
        per_case[cid] = {
            "artifact": os.path.relpath(item["path"], _ROOT),
            "aspect_nominal": float(case["aspect_nominal"]),
            "aspect_measured": float(case["aspect_measured"]),
            "grains_per_radius": float(case["protocol"]["grains_per_radius"]),
            "native_stride": stride0,
            "native_sample_dt": float(np.median(np.diff(t))),
            "n_samples": int(t.size),
            "reference_time": ref_t,
            "band": band,
            "adjudication": adj,
            "sampling": {
                "steps_per_sample": {str(k * stride0): entry["times"][str(k)]
                                     for k in FACTORS},
                "bias_rel": {str(k * stride0): entry["bias_rel"][str(k)]
                             for k in FACTORS},
                "spread_rel": float(entry["spread_rel"]),
                "spread_s": float(entry["spread_s"]),
            },
            "bound_violations": violations,
            "definition_times": {k: grid[k] for k in definition_keys()},
        }
        for key in definition_keys():
            value = grid[key]
            rows.append({"case_id": cid, "definition": key, "time_s": value,
                         "error": ((value - ref_t) / ref_t)
                         if math.isfinite(value) else math.nan})
    return {"cases": per_case, "rows": rows}


def figures(result: dict) -> None:
    cases = result["cases"]
    keys = definition_keys()
    ordered = sorted(cases, key=lambda c: (cases[c]["aspect_nominal"],
                                          cases[c]["grains_per_radius"]))
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    width = 0.8 / max(1, len(ordered))
    for j, cid in enumerate(ordered):
        c = cases[cid]
        ref_t = c["reference_time"]
        rel = [(c["definition_times"][k] - ref_t) / ref_t * 100.0
               if math.isfinite(c["definition_times"][k]) else np.nan for k in keys]
        x = np.arange(len(keys)) + (j - (len(ordered) - 1) / 2) * width
        ax.bar(x, rel, width=width * 0.95,
               label=f"{cid} (a={c['aspect_nominal']:g}, gpr={c['grains_per_radius']:g})")
    ax.axhspan(-100 * TOL, 100 * TOL, color="0.88", zorder=0,
               label="+/-15 %% card band")
    ax.axhline(3.0, color="C3", ls="--", lw=1.0, label="+3 %")
    ax.axhline(-3.0, color="C3", ls="--", lw=1.0)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=90, fontsize=7)
    ax.set_ylabel("(t - t_ref)/t_ref [%]")
    ax.set_title("H2: arrival-time error by definition, archived trajectories "
                 "(stride 5); ref = 3 sqrt(h_i/g)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_archive_definitions.png"), dpi=150)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for cid in ordered:
        c = cases[cid]
        steps = sorted(int(s) for s in c["sampling"]["bias_rel"])
        a1.plot(steps, [100 * c["sampling"]["bias_rel"][str(s)] for s in steps],
                "o-", label=cid)
        a2.plot(steps, [100 * abs(c["sampling"]["bias_rel"][str(s)]) for s in steps],
                "o-", label=cid)
    a1.set_xscale("log")
    a1.set_xlabel("integration steps per stored sample")
    a1.set_ylabel("t90 bias vs native stride [%]")
    a1.set_title("Further decimation of the archived histories")
    a2.axhline(3.0, color="C3", ls="--", lw=1.0, label="3 % (H1 threshold)")
    a2.set_xscale("log")
    a2.set_xlabel("integration steps per stored sample")
    a2.set_ylabel("|bias| [% of t90]")
    a2.set_title("Absolute sampling effect")
    a1.legend(fontsize=7)
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_archive_sampling.png"), dpi=150)
    plt.close(fig)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    result = analyze()
    if not result["cases"]:
        print("no archived histories found")
        return 1
    with open(os.path.join(OUT, "archive_summary.json"), "w") as fh:
        json.dump(result["cases"], fh, indent=2)
    with open(os.path.join(OUT, "archive_definitions.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result["rows"][0].keys()))
        writer.writeheader()
        writer.writerows(result["rows"])
    figures(result)
    brief = {cid: {"native_stride": c["native_stride"],
                   "band_rel": c["band"]["spread_rel"],
                   "band_min": c["band"]["min_definition"],
                   "band_max": c["band"]["max_definition"],
                   "definition_dominated": c["band"]["definition_dominated"],
                   "n_in_band": c["adjudication"]["n_in_band"],
                   "n_flips": c["adjudication"]["n_flips"],
                   "primary_error": c["adjudication"]["primary_error"],
                   "best": c["adjudication"]["best"],
                   "best_error": c["adjudication"]["best_error"],
                   "worst_error": c["adjudication"]["worst_error"],
                   "within_3pct": c["adjudication"]["any_within_three_percent"],
                   "max_sampling_bias_rel": max(
                       (abs(v) for v in c["sampling"]["bias_rel"].values()),
                       default=0.0),
                   "sampling_bias_rel": c["sampling"]["bias_rel"],
                   "bound_violations": len(c["bound_violations"])}
             for cid, c in result["cases"].items()}
    print(json.dumps(brief, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())