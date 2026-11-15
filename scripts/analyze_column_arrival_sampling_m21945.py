"""Reduce + plot the m21945 arrival-sampling round.

Reads the per-case JSONs written by
``benchmarks/case_column_collapse_arrival_sampling_m21945.py`` (dense history,
``sample_stride = 1``) and produces

* ``summary.json`` / ``cases.csv`` (via the benchmark's own ``collect()``),
* ``definitions.csv`` -- every case x definition x stride arrival time + error,
* ``fig_arrival_definitions.png`` -- definition-induced spread of ``t`` vs the
  Lube (2004) reference, with the 15 % card band and the 3 % "good" line,
* ``fig_sampling_bias.png`` -- sampling-stride bias of ``t90`` against the
  analytic one-sample budget ``stride * dt``,
* ``fig_shallow_resolution.png`` -- the ``a = 0.5`` resolution ladder at stride 1
  vs stride 20 (H3: is the non-monotone scatter a sampling artefact?).

All numbers come from the stored JSONs; nothing is transcribed by hand.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import case_column_collapse_arrival_sampling_m21945 as BM  # noqa: E402
from tensordem.granular_column.arrival_sampling import (  # noqa: E402
    PRIMARY_KEY, adjudicate, arrival_grid, definition_keys, emplacement_errors)

OUT = os.path.join(_ROOT, BM.OUT_DIR)
TOL = BM.TOLERANCE


def load_cases() -> dict:
    cases = {}
    for spec in BM.NEW_CASES:
        path = os.path.join(_ROOT, BM.case_path(spec["case_id"]))
        if os.path.exists(path):
            with open(path) as fh:
                cases[spec["case_id"]] = json.load(fh)
    return cases


def definitions_table(cases: dict) -> list[dict]:
    rows = []
    for cid, case in cases.items():
        radius = float(case["column_radius_measured"])
        diameter = 2.0 * float(case["protocol"]["radius"])
        ref_t = float(case["reference"]["measured"]["emplacement_time"])
        scan = case["stride_scan"]
        for key, entry in scan["definitions"].items():
            for stride_str, value in entry["times"].items():
                rows.append({
                    "case_id": cid,
                    "definition": key,
                    "stride": int(stride_str),
                    "time_s": value,
                    "error": ((value - ref_t) / ref_t) if math.isfinite(value) else math.nan,
                })
        # adjudication of every definition at the dense stride
        grid = arrival_grid(np.asarray(case["history"]["time"]),
                            np.asarray(case["history"]["front"]),
                            radius=radius, grain_diameter=diameter, stride=1)
        case["_grid_dense"] = grid
        case["_adj"] = adjudicate(emplacement_errors(grid, ref_t), tolerance=TOL)
    return rows


def fig_definitions(cases: dict, path: str) -> None:
    ordered = [c["case_id"] for c in BM.NEW_CASES if c["case_id"] in cases]
    keys = definition_keys()
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    width = 0.8 / max(1, len(ordered))
    for j, cid in enumerate(ordered):
        case = cases[cid]
        errs = [case["_grid_dense"][k] for k in keys]
        ref_t = float(case["reference"]["measured"]["emplacement_time"])
        rel = [(e - ref_t) / ref_t * 100.0 if math.isfinite(e) else np.nan
               for e in errs]
        x = np.arange(len(keys)) + (j - (len(ordered) - 1) / 2) * width
        ax.bar(x, rel, width=width * 0.95, label=cid)
    ax.axhspan(-100 * TOL, 100 * TOL, color="0.85", zorder=0,
               label=f"+/-{100 * TOL:.0f} % card band")
    ax.axhline(3.0, color="C3", lw=1.0, ls="--", label="+3 %")
    ax.axhline(-3.0, color="C3", lw=1.0, ls="--")
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=90, fontsize=7)
    ax.set_ylabel("t - t_ref / t_ref  [%]")
    ax.set_title("Arrival-time error by pre-registered definition "
                 f"(dense stride = 1); ref = 3 sqrt(h_i/g), Lube 2004")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_sampling(cases: dict, path: str) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for cid, case in cases.items():
        entry = case["stride_scan"]["definitions"][PRIMARY_KEY]
        strides = sorted(int(s) for s in entry["times"])
        bias = [100.0 * entry["bias_rel"][str(s)] for s in strides]
        dt = float(case["stride_scan"]["dense_sample_dt"])
        a1.plot(strides, bias, "o-", label=cid)
        a1.plot(strides, [100.0 * s * dt / entry["reference_time"] for s in strides],
                "k:", lw=0.8)
        budget_frac = [abs(entry["times"][str(s)] - entry["reference_time"]) / entry["reference_time"]
                       for s in strides]
        a2.plot(strides, [100.0 * b for b in budget_frac], "o-", label=cid)
    a1.set_xscale("log")
    a1.set_xlabel("history sampling stride [integration steps]")
    a1.set_ylabel("t90 bias vs stride 1 [% of t90]")
    a1.set_title("Sampling-stride bias of t90 (dotted: one-sample budget k*dt)")
    a1.legend(fontsize=7)
    a2.axhline(3.0, color="C3", ls="--", lw=1.0, label="3 %")
    a2.set_xscale("log")
    a2.set_xlabel("history sampling stride [integration steps]")
    a2.set_ylabel("|t90(k) - t90(1)| / t90(1)  [%]")
    a2.set_title("Absolute sampling effect (H1 threshold = 3 %)")
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_shallow(cases: dict, path: str) -> None:
    shallow = sorted((c for c in cases.values()
                      if abs(float(c["aspect_nominal"]) - 0.5) < 1e-9),
                     key=lambda c: float(c["protocol"]["grains_per_radius"]))
    if not shallow:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    gpr = [float(c["protocol"]["grains_per_radius"]) for c in shallow]
    ref = [float(c["reference"]["measured"]["emplacement_time"]) for c in shallow]
    e1 = [100.0 * (c["stride_scan"]["definitions"][PRIMARY_KEY]["times"]["1"] - r) / r
          for c, r in zip(shallow, ref)]
    e20 = [100.0 * (c["stride_scan"]["definitions"][PRIMARY_KEY]["times"]["20"] - r) / r
           for c, r in zip(shallow, ref)]
    ax.plot(gpr, e1, "o-", label="stride = 1 (dense)")
    ax.plot(gpr, e20, "s--", label="stride = 20 (archived protocol)")
    ax.axhspan(-100 * TOL, 100 * TOL, color="0.9", zorder=0)
    ax.axhline(3.0, color="C3", ls="--", lw=1.0)
    ax.axhline(-3.0, color="C3", ls="--", lw=1.0)
    ax.set_xlabel("resolution [grains per column radius]")
    ax.set_ylabel("t90 error vs 3 sqrt(h_i/g)  [%]")
    ax.set_title("a = 0.5 emplacement-time error vs resolution (H3)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    cases = load_cases()
    if not cases:
        print("no cases stored yet; run the benchmark first")
        return 1
    rows = definitions_table(cases)
    summary = BM.collect()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "definitions.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    adj = {cid: case["_adj"] for cid, case in cases.items()}
    with open(os.path.join(OUT, "adjudication.json"), "w") as fh:
        json.dump(adj, fh, indent=2)
    fig_definitions(cases, os.path.join(OUT, "fig_arrival_definitions.png"))
    fig_sampling(cases, os.path.join(OUT, "fig_sampling_bias.png"))
    fig_shallow(cases, os.path.join(OUT, "fig_shallow_resolution.png"))
    print(json.dumps({"verdict": summary["verdict"],
                      "H1": summary["H1_sampling_artefact"]["verdict"],
                      "H2": summary["H2_definition_dominated"]["verdict"],
                      "H3": summary["H3_shallow_non_monotone"],
                      "bound_violations": summary["total_bound_violations"],
                      "adjudication": {k: {"n_in_band": v["n_in_band"],
                                           "n_flips": v["n_flips"],
                                           "best": v["best"],
                                           "best_error": v["best_error"],
                                           "worst_error": v["worst_error"],
                                           "within_3pct": v["any_within_three_percent"]}
                                       for k, v in adj.items()}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())