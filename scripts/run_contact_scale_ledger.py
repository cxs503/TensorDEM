#!/usr/bin/env python
"""Contact-parameter dimension ledger + geometric-scale invariance campaign.

Runs the M1 ledger scan, the M2/M3 invariance experiment (repose / direct shear
/ blade resistance), the M4 CI gate manifest and the figures.

Examples
--------
    # ledger only (fast, no simulation)
    python scripts/run_contact_scale_ledger.py --ledger-only

    # one shard of the simulation campaign
    python scripts/run_contact_scale_ledger.py --shard 0 --shards 8

    # merge the shards, audit the gate and draw the figures
    python scripts/run_contact_scale_ledger.py --merge

Artifacts land in ``artifacts/contact_scale_ledger/``:

    ledger.json          M1 dimension ledger of the whole fleet
    shards/shard_XX.json raw probe records (one file per shard)
    invariance.json      M2/M3 analysis (deviations, verdicts, gate)
    gate.json            M4 similarity-manifest audit (red/green)
    figures/*.png        deviation bars, compliance sweep, metric spread
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.contact_param_ledger import (  # noqa: E402
    FROUDE, audit_manifest, exponent_table, scan_fleet, write_ledger,
)
from tensordem.contact_scale_invariance import (  # noqa: E402
    METRIC_KEYS, PROBES, GranularCase, analyse, scaled_variants,
)

OUT = REPO / "artifacts" / "contact_scale_ledger"
SHARDS = OUT / "shards"

#: Pre-declared reference stiffness [N/m] (see docs/contact_scale_ledger_method.md).
REFERENCE_KN = 600.0


def _task_list(kn_sweep: Sequence[float], lam: float,
               probes: Sequence[str]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for kn in kn_sweep:
        base = replace(GranularCase(), kn=float(kn))
        for probe in probes:
            for variant in scaled_variants(base, lam):
                tasks.append({
                    "id": f"{probe}|kn{kn:g}|{variant.name}",
                    "probe": probe,
                    "kn": float(kn),
                    "variant": variant.name,
                    "lam": variant.lam,
                    "kind": variant.kind,
                    "expected": variant.expected,
                    "note": variant.note,
                    "case": variant.case,
                })
    return tasks


def run_shard(shard: int, n_shards: int, tasks: Sequence[dict[str, Any]]
              ) -> dict[str, Any]:
    SHARDS.mkdir(parents=True, exist_ok=True)
    mine = [t for i, t in enumerate(tasks) if i % n_shards == shard]
    records: list[dict[str, Any]] = []
    for task in mine:
        probe = PROBES[task["probe"]]
        metric = METRIC_KEYS[task["probe"]]
        result = probe(task["case"])
        records.append({
            "probe": task["probe"],
            "kn_level": task["kn"],
            "metric": metric,
            "variant": task["variant"],
            "lam": task["lam"],
            "kind": task["kind"],
            "expected": task["expected"],
            "note": task["note"],
            "value": float(result[metric]),
            "raw": {k: v for k, v in result.items()
                    if isinstance(v, (int, float, str))},
            "case": task["case"].as_dict(),
        })
    payload = {"shard": shard, "n_shards": n_shards,
               "n_tasks": len(mine), "records": records}
    (SHARDS / f"shard_{shard:02d}.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def _load_shards() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(SHARDS.glob("shard_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(payload["records"])
    return records


def gate_manifest(base: GranularCase, lam: float) -> dict[str, Any]:
    """M4 manifest: the similarity law as data, audited dimensionally."""
    variants = scaled_variants(base, lam)
    base_params = base.as_parameters()
    out_variants: dict[str, Any] = {}
    for variant in variants[1:]:
        params = variant.case.as_parameters()
        out_variants[variant.name] = {
            # controls *and* the two literature stiffness laws are deliberately
            # non-similarity-preserving, so the dimensional gate must flag them
            "expected": ("violation"
                         if variant.kind in ("control", "diagnostic") else "OK"),
            "params": params,
        }
    return {
        "schema": "tensordem.contact_scale_gate/1",
        "similarity_mode": FROUDE.describe(),
        "tolerance": {"invariance": 0.02, "negative_control": 0.10},
        "cases": [{"name": "granular_reference", "lam": lam,
                   "base": base_params, "variants": out_variants}],
    }


def _figures(analysis: dict[str, Any], out: Path) -> list[dict[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    figs: list[dict[str, str]] = []
    kn_keys = sorted(analysis["by_stiffness"], key=float)
    probes = sorted(analysis["by_stiffness"][kn_keys[0]])
    ref = analysis["reference_kn"]

    # (1) deviation bars per probe and variant
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    entry = analysis["by_stiffness"][str(ref)]
    names: list[str] = []
    for probe in probes:
        variants = entry[probe]["variants"]
        names = [n for n in variants if n != "lam1_reference"]
        break
    width = 0.8 / max(len(probes), 1)
    for k, probe in enumerate(probes):
        variants = entry[probe]["variants"]
        devs = [100.0 * variants[n]["deviation"] for n in names]
        pos = np.arange(len(names)) + (k - (len(probes) - 1) / 2) * width
        ax.bar(pos, devs, width=width, label=probe)
    ax.set_yscale("log")
    ax.axhline(2.0, color="tab:green", ls="--", lw=1.0,
               label="invariance tolerance 2%")
    ax.axhline(10.0, color="tab:red", ls=":", lw=1.0,
               label="negative-control threshold 10%")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("deviation from the correctly scaled case  [%]")
    ax.set_title(f"Scale-invariance deviations at lambda={entry[probes[0]]['lam']:g}"
                 f" (kn = {ref:g} N/m)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "deviation_bars.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"name": "deviation_bars", "path": str(path),
                 "caption": "Macroscopic deviation of every scale variant "
                            "from the correctly transformed case."})

    # (2) compliance sweep: does the stiffness control survive?
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for probe in probes:
        xs, ys = [], []
        for kn_key in analysis["sweep_kn"]:
            key = str(float(kn_key))
            if key not in analysis["by_stiffness"]:
                continue
            variants = analysis["by_stiffness"][key][probe]["variants"]
            knob = variants.get("ctl_kn_constant")
            if knob is None:
                continue
            xs.append(float(kn_key))
            ys.append(100.0 * knob["deviation"])
        if xs:
            ax.plot(xs, ys, "o-", label=f"{probe}: kn held constant")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(10.0, color="tab:red", ls=":", lw=1.0,
               label="negative-control threshold 10%")
    ax.set_xlabel("reference stiffness kn  [N/m]  (soft -> stiff)")
    ax.set_ylabel("deviation of the 'kn constant' control  [%]")
    ax.set_title("Where the stiffness control has discriminating power")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "compliance_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"name": "compliance_sweep", "path": str(path),
                 "caption": "Deviation of the 'stiffness not scaled' control "
                            "as a function of the reference compliance."})

    # (3) the probes' dimensionless metrics across variants
    fig, axes = plt.subplots(1, len(probes), figsize=(11.0, 3.6))
    if len(probes) == 1:
        axes = [axes]
    for ax, probe in zip(axes, probes):
        variants = entry[probe]["variants"]
        labels = list(variants)
        values = [variants[n]["value"] for n in labels]
        base_v = values[labels.index("lam1_reference")] if \
            "lam1_reference" in labels else values[0]
        ax.bar(range(len(labels)), [v / base_v for v in values],
               color=["tab:gray"] + ["tab:blue"] * (len(labels) - 1))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_title(f"{probe}: {entry[probe]['metric']}", fontsize=9)
        ax.set_ylabel("metric / reference")
    fig.suptitle("Macroscopic probes across scale variants", fontsize=10)
    fig.tight_layout()
    path = out / "metric_spread.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"name": "metric_spread", "path": str(path),
                 "caption": "Dimensionless probe values (normalised by the "
                            "unscaled reference) for every variant."})
    return figs


def merge(lam: float, kn_sweep: Sequence[float], probes: Sequence[str]
          ) -> dict[str, Any]:
    records = _load_shards()
    by_kn: dict[str, dict[str, Any]] = {}
    for kn in kn_sweep:
        subset = [r for r in records if abs(r["kn_level"] - float(kn)) < 1e-9]
        if not subset:
            continue
        per_probe: dict[str, Any] = {}
        for probe in probes:
            sub = [r for r in subset if r["probe"] == probe]
            if not sub:
                continue
            analysis = analyse(sub)
            per_probe[probe] = {
                "metric": METRIC_KEYS[probe],
                "lam": sub[0]["lam"],
                "variants": analysis["probes"][probe]["variants"],
                "gate": analysis["gate"],
                "raw": {r["variant"]: r["raw"] for r in sub},
            }
        by_kn[str(float(kn))] = per_probe

    # Pre-declared reference stiffness (see docs/preregistration): the level
    # whose worst-case contact overlap stays below the 10 % validity envelope
    # while being compliant enough for a stiffness mistake to be observable.
    reference_kn = REFERENCE_KN if str(float(REFERENCE_KN)) in by_kn else float(
        kn_sweep[0])
    invariance = {
        "schema": "tensordem.contact_scale_invariance/1",
        "lam": lam,
        "similarity_mode": FROUDE.describe(),
        "thresholds": {"invariance": 0.02, "negative_control": 0.10},
        "reference_kn": reference_kn,
        "sweep_kn": [float(k) for k in kn_sweep if str(float(k)) in by_kn],
        "by_stiffness": by_kn,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "invariance.json").write_text(json.dumps(invariance, indent=1),
                                         encoding="utf-8")

    base = replace(GranularCase(), kn=float(reference_kn))
    manifest = gate_manifest(base, lam)
    (OUT / "gate_manifest.json").write_text(json.dumps(manifest, indent=1),
                                            encoding="utf-8")
    gate = audit_manifest(manifest)
    (OUT / "gate.json").write_text(json.dumps(gate, indent=1),
                                   encoding="utf-8")
    figs = _figures(invariance, OUT / "figures")
    summary = _summary(invariance, gate, figs)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1),
                                      encoding="utf-8")
    return summary


def _summary(invariance: dict[str, Any], gate: dict[str, Any],
             figs: list[dict[str, str]]) -> dict[str, Any]:
    ref = invariance["reference_kn"]
    entry = invariance["by_stiffness"][str(float(ref))]
    probes: dict[str, Any] = {}
    all_inv_ok = True
    all_ctl_ok = True
    for probe, data in entry.items():
        variants = data["variants"]
        inv = {n: v["deviation"] for n, v in variants.items()
               if v["verdict"] in ("PASS", "FAIL")}
        ctl = {n: v["deviation"] for n, v in variants.items()
               if v["verdict"] in ("CAUGHT", "MISSED")}
        inv_ok = all(v["verdict"] == "PASS" for v in variants.values()
                     if v["verdict"] in ("PASS", "FAIL"))
        ctl_ok = all(v["verdict"] == "CAUGHT" for v in variants.values()
                     if v["verdict"] in ("CAUGHT", "MISSED"))
        all_inv_ok &= inv_ok
        all_ctl_ok &= ctl_ok
        probes[probe] = {
            "metric": data["metric"],
            "value_reference": variants["lam1_reference"]["value"],
            "value_similar": variants["lamN_similar"]["value"],
            "invariance_deviations": inv,
            "control_deviations": ctl,
            "invariance_pass": inv_ok,
            "controls_caught": ctl_ok,
            "max_invariance_deviation": max(inv.values()) if inv else None,
            "min_control_deviation": min(ctl.values()) if ctl else None,
        }
    # cross-probe roll-up: a mistake must be caught by *some* observable, not
    # by every observable (rolling resistance is irrelevant to a normal bounce,
    # stiffness is irrelevant to restitution) - that is the pre-registered rule.
    controls_rollup: dict[str, Any] = {}
    for probe, data in probes.items():
        for name, dev in data["control_deviations"].items():
            row = controls_rollup.setdefault(name, {"max_deviation": dev,
                                                    "caught_by": [probe]})
            if dev > row["max_deviation"]:
                row["max_deviation"] = dev
            if dev > 0.10 and probe not in row["caught_by"]:
                row["caught_by"].append(probe)
    for name, row in controls_rollup.items():
        row["caught_by"] = [p_ for p_ in row["caught_by"]
                            if probes[p_]["control_deviations"].get(name, 0.0) > 0.10]
        row["caught"] = row["max_deviation"] > 0.10
    all_ctl_ok = all(row["caught"] for row in controls_rollup.values())
    sweep: dict[str, Any] = {}
    for kn in invariance["sweep_kn"]:
        row: dict[str, Any] = {}
        for probe, data in invariance["by_stiffness"][str(float(kn))].items():
            knob = data["variants"].get("ctl_kn_constant")
            inv = data["variants"].get("lamN_similar")
            row[probe] = {
                "kn_deviation": knob["deviation"] if knob else None,
                "invariance_deviation": inv["deviation"] if inv else None,
            }
        sweep[str(kn)] = row
    return {
        "schema": "tensordem.contact_scale_ledger_summary/1",
        "criteria": {
            "M2_invariance_max_deviation": 0.02,
            "M3_negative_control_min_deviation": 0.10,
            "M4_gate_must_flag_all_controls": True,
        },
        "reference_kn": ref,
        "lam": invariance["lam"],
        "M2_pass": all_inv_ok,
        "M3_pass": all_ctl_ok,
        "M4_gate_status": gate["status"],
        "M4_violations_flagged": gate["n_violations"],
        "probes": probes,
        "controls_rollup": controls_rollup,
        "compliance_sweep": sweep,
        "figures": [f["path"] for f in figs],
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--kn-sweep", default="150,600,2400")
    ap.add_argument("--probes", default="repose,shear,drag")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--ledger-only", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--list-tasks", action="store_true")
    args = ap.parse_args(argv)

    kn_sweep = [float(x) for x in args.kn_sweep.split(",") if x.strip()]
    probes = [p for p in args.probes.split(",") if p.strip()]
    OUT.mkdir(parents=True, exist_ok=True)

    if args.ledger_only:
        payload = write_ledger(REPO, OUT / "ledger.json")
        print(json.dumps(payload["summary"], indent=2))
        return 0

    if args.merge:
        summary = merge(args.lam, kn_sweep, probes)
        print(json.dumps(summary, indent=2))
        return 0

    tasks = _task_list(kn_sweep, args.lam, probes)
    if args.list_tasks:
        print(json.dumps([t["id"] for t in tasks], indent=1))
        return 0
    shard = 0 if args.shard is None else args.shard
    payload = run_shard(shard, args.shards, tasks)
    print(json.dumps({"shard": shard, "n_tasks": payload["n_tasks"],
                      "ids": [r["probe"] + "|" + r["variant"]
                              for r in payload["records"]]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())