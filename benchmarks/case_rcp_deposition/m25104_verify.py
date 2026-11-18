#!/usr/bin/env python3
"""m25104 — Fresh independent verification of CASE-DEM-RCP-PACKING-FRACTION.

Re-runs the acceptance metric (steady-state porosity of a dropped packing,
N=10000, frictionless / low-friction) with a *new* seed set (4,5,6) that is
disjoint from the archived production seeds (1,2,3), then pools all six seeds
per friction level and re-certifies the anchor acceptance band.

Reference
---------
Scott, G. D. (1960), "Packing of Spheres", *Nature* **188**, 908.
Bernal, J. D. & Mason, J. (1960), "Packing of Spheres", *Nature* **188**, 910.
DOI: 10.1038/188910a0  ->  phi_ref = 0.6366 (RCP, frictionless).

Acceptance (feature_ledger CASE-DEM-RCP-PACKING-FRACTION)
---------------------------------------------------------
* volume fraction phi in [0.62, 0.66]
* metric: steady-state porosity of a dropped packing, 10000 spheres,
  frictionless (mu=0) and low-friction (mu=0.1) two settings.

This script is pure analysis: it reads the archived production runs
(``runs/deposition_n10000_mu{mu}_kn4000_e0.1_seed{1,2,3}.json``) and the fresh
m25104 runs (``runs/m25104_rcp_verify/runs/deposition_n10000_mu{mu}_kn4000_
e0.1_seed{4,5,6}.json``), pools them, and writes a single evidence JSON that
the report generator injects numbers from (zero hand-copy).

Usage
-----
::

    PYTHONPATH=src python benchmarks/case_rcp_deposition/m25104_verify.py \
        --out runs/m25104_rcp_verify/verify_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# Bernal & Mason (1960) Nature 188:910, RCP volume fraction (frictionless).
PHI_REF = 0.6366
# feature_ledger acceptance band.
BAND = (0.62, 0.66)
# production settings (identical to the archived runs).
N = 10000
KN = 4000.0
E = 0.1
MU_LEVELS = (0.0, 0.1)
ARCHIVED_SEEDS = (1, 2, 3)
FRESH_SEEDS = (4, 5, 6)


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def _tag(mu: float, seed: int) -> str:
    return (f"deposition_n{N}_mu{mu:g}_kn{KN:g}_e{E:g}_seed{seed}")


def collect() -> dict:
    """Collect archived + fresh runs for both friction levels."""
    out = {}
    for mu in MU_LEVELS:
        rows = []
        for seed in ARCHIVED_SEEDS:
            p = HERE / "runs" / (_tag(mu, seed) + ".json")
            d = _load(p)
            if d is None:
                continue
            m = d["metrics"]
            rows.append({
                "seed": seed,
                "origin": "archived",
                "phi_bulk": m["phi_bulk"],
                "phi_corr": m["phi_bulk_overlap_corrected"],
                "Z": m["coordination_bulk"],
                "max_ov_d": m["max_pair_overlap_over_d"],
                "mean_ov_d": m["mean_overlap_over_d"],
                "settled": d["settled"],
                "steps": d["steps"],
                "wall_s": d.get("wall_clock_total"),
            })
        for seed in FRESH_SEEDS:
            p = (ROOT / "runs" / "m25104_rcp_verify" / "runs"
                 / (_tag(mu, seed) + ".json"))
            d = _load(p)
            if d is None:
                continue
            m = d["metrics"]
            rows.append({
                "seed": seed,
                "origin": "fresh_m25104",
                "phi_bulk": m["phi_bulk"],
                "phi_corr": m["phi_bulk_overlap_corrected"],
                "Z": m["coordination_bulk"],
                "max_ov_d": m["max_pair_overlap_over_d"],
                "mean_ov_d": m["mean_overlap_over_d"],
                "settled": d["settled"],
                "steps": d["steps"],
                "wall_s": d.get("wall_clock_total"),
            })
        out[mu] = rows
    return out


def _stats(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
    }


def summarize(collected: dict) -> dict:
    summary = {"phi_ref": PHI_REF, "band": list(BAND), "levels": {}}
    for mu, rows in collected.items():
        fresh = [r for r in rows if r["origin"] == "fresh_m25104"]
        pooled = rows
        s_pooled = _stats([r["phi_corr"] for r in pooled])
        s_fresh = _stats([r["phi_corr"] for r in fresh]) if fresh else None
        s_arch = _stats([r["phi_corr"] for r in rows
                         if r["origin"] == "archived"])
        mean = s_pooled["mean"]
        in_band = BAND[0] <= mean <= BAND[1]
        rel = (mean - PHI_REF) / PHI_REF
        summary["levels"][f"mu={mu:g}"] = {
            "n_seeds": s_pooled["n"],
            "n_fresh": s_fresh["n"] if s_fresh else 0,
            "phi_pooled": s_pooled,
            "phi_fresh_only": s_fresh,
            "phi_archived_only": s_arch,
            "Z_mean": float(np.mean([r["Z"] for r in pooled])),
            "max_ov_d_mean": float(np.mean([r["max_ov_d"] for r in pooled])),
            "all_settled": all(r["settled"] for r in pooled),
            "in_band": bool(in_band),
            "rel_err_vs_ref": float(rel),
            "abs_err_vs_ref": float(abs(mean - PHI_REF)),
            "rows": rows,
        }
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/m25104_rcp_verify/verify_summary.json")
    args = ap.parse_args(argv)

    collected = collect()
    summary = summarize(collected)

    # gate verdicts
    gates = {}
    for mu in MU_LEVELS:
        lv = summary["levels"][f"mu={mu:g}"]
        gates[f"mu={mu:g}"] = {
            "phi_in_band": lv["in_band"],
            "abs_err_lt_0.02": lv["abs_err_vs_ref"] < 0.02,
            "rel_err_lt_3pct": abs(lv["rel_err_vs_ref"]) < 0.03,
            "all_settled": lv["all_settled"],
            "n_fresh_seeds": lv["n_fresh"],
        }
    summary["gates"] = gates
    summary["all_gates_pass"] = all(
        g["phi_in_band"] and g["abs_err_lt_0.02"] and g["rel_err_lt_3pct"]
        and g["all_settled"] and g["n_fresh_seeds"] >= 3
        for g in gates.values())

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {outp}")
    for mu in MU_LEVELS:
        lv = summary["levels"][f"mu={mu:g}"]
        print(f"  mu={mu:g}: phi_pooled={lv['phi_pooled']['mean']:.5f} "
              f"±{lv['phi_pooled']['std']:.5f} (n={lv['phi_pooled']['n']}, "
              f"fresh={lv['n_fresh']}) in_band={lv['in_band']} "
              f"rel_err={lv['rel_err_vs_ref']*100:+.2f}%")
    print(f"  ALL_GATES_PASS={summary['all_gates_pass']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
