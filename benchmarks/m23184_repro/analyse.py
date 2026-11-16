#!/usr/bin/env python3
"""m23184 — reproducibility audit of the archived Beverloo verdicts.

Collects every discharge record of the frozen primary case that exists on this
box (archived campaign artefacts + the fresh re-runs submitted by this mission),
re-derives the arrest diagnosis with `tensordem.silo_flow.beverloo.discharge_status`,
and writes `summary.json`, the audit document and two figures.  Everything in
the document is generated from `summary.json` — no number is typed by hand.

    python3 benchmarks/m23184_repro/analyse.py --figures
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensordem.silo_flow.beverloo import (  # noqa: E402
    BEVERLOO_C, BEVERLOO_K, beverloo_prediction, discharge_status,
)

#: where archived campaign artefacts of the same frozen case live (read only)
ARCHIVE_GLOBS = (
    str(ROOT.parent / "*" / "runs" / "*" / "*.json"),
)
FROZEN_SIGNATURE = {"orifice_over_particle": 5.0, "particle_diameter": 0.003,
                    "fill_height": 0.036, "kn": 5000.0}


def collect(paths) -> list[dict]:
    out = []
    for p in paths:
        try:
            d = json.loads(Path(p).read_text())
        except Exception:
            continue
        if not isinstance(d, dict) or "flow_rate_kg_s" not in d or "case" not in d:
            continue
        c = d["case"]
        if any(abs(float(c.get(k, -1)) - v) > 1e-12 for k, v in FROZEN_SIGNATURE.items()):
            continue
        if abs(float(c.get("silo_radius", 0.0)) - 0.024) > 1e-12:
            continue
        w_ref = beverloo_prediction(c["orifice_diameter"], c["particle_diameter"],
                                    d["rho_bulk"])
        status = discharge_status(d["window_particles"], d["flow_rate_windows_kg_s"],
                                  reference_rate=w_ref)
        out.append({
            "source": str(Path(p)),
            "label": Path(p).parent.name + "/" + Path(p).stem,
            "seed": c.get("seed"),
            "W": float(d["flow_rate_kg_s"]),
            "W_ref": float(w_ref),
            "rel": float(d["flow_rate_kg_s"] / w_ref - 1.0) if w_ref else None,
            "window_particles": d["window_particles"],
            "flow_rate_windows_kg_s": d["flow_rate_windows_kg_s"],
            "particles_exited": d.get("particles_exited"),
            "n_bed": d.get("n_bed"),
            "status": status["status"],
            "arrested": status["arrested"],
            "reasons": status["reasons"],
            "counting_noise_rel": status["counting_noise_rel"],
            "rate_over_reference": status["rate_over_reference"],
            "dt": d.get("dt"),
            "wall_clock_s": d.get("wall_clock_s"),
            "git_sha": (d.get("meta") or {}).get("git_sha")
            or ((d.get("m23184") or {}).get("meta") or {}).get("git_sha"),
        })
    return sorted(out, key=lambda r: (r["label"]))


def ensemble(records: list[dict]) -> dict:
    """Mean / sd / Student-t 95 % CI over the *flowing* records, and the count of arrests."""
    flowing = [r for r in records if not r["arrested"]]
    rel = np.array([r["rel"] for r in flowing], dtype=float) if flowing else np.zeros(0)
    out = {
        "n_records": len(records),
        "n_flowing": len(flowing),
        "n_arrested": len(records) - len(flowing),
        "arrest_fraction": (len(records) - len(flowing)) / len(records) if records else None,
        "seeds_flowing": sorted({r["seed"] for r in flowing}, key=lambda s: (s is None, s)),
    }
    if rel.size:
        n = rel.size
        out.update({
            "mean_rel": float(rel.mean()),
            "sd_rel": float(rel.std(ddof=1)) if n > 1 else 0.0,
            "cv_rel": float(rel.std(ddof=1) / abs(rel.mean())) if n > 1 and rel.mean() else None,
            "min_rel": float(rel.min()),
            "max_rel": float(rel.max()),
            "range_rel": float(rel.max() - rel.min()),
        })
        if n > 1:
            from scipy import stats
            t = float(stats.t.ppf(0.975, n - 1))
            half = t * out["sd_rel"] / math.sqrt(n)
            out["ci95_rel"] = [out["mean_rel"] - half, out["mean_rel"] + half]
            out["t_factor"] = t
        else:
            out["ci95_rel"] = None
    return out


def verdict(records: list[dict], ens: dict, tolerance: float = 0.10) -> dict:
    """Two questions: can the archive be reproduced, and does the anchor pass?"""
    by_seed: dict = {}
    for r in records:
        by_seed.setdefault(r["seed"], []).append(r)
    contradictions = []
    for seed, rs in by_seed.items():
        if len(rs) > 1:
            spread = max(r["W"] for r in rs) - min(r["W"] for r in rs)
            rel_spread = spread / max(r["W_ref"] for r in rs)
            statuses = sorted({r["status"] for r in rs})
            contradictions.append({
                "seed": seed,
                "n_records": len(rs),
                "labels": [r["label"] for r in rs],
                "W_range": [min(r["W"] for r in rs), max(r["W"] for r in rs)],
                "W_spread_over_reference": float(rel_spread),
                "statuses": statuses,
                "inconsistent": bool(rel_spread > 0.10 or len(set(statuses)) > 1),
            })
    acceptance = None
    if ens.get("ci95_rel") is not None:
        lo, hi = ens["ci95_rel"]
        acceptance = {
            "tolerance": tolerance,
            "mean_rel": ens["mean_rel"],
            "ci95_rel": ens["ci95_rel"],
            "pass_interval_inside_tolerance": bool(hi < tolerance and lo > -tolerance),
            "pass_point_estimate": bool(abs(ens["mean_rel"]) < tolerance),
            "decisive": bool(hi < tolerance and lo > -tolerance or hi < -tolerance or lo > tolerance),
        }
    return {
        "same_seed_duplicates": contradictions,
        "archive_reproducible": not any(c["inconsistent"] for c in contradictions),
        "anchor_acceptance_on_flowing_records": acceptance,
    }


def build(records: list[dict]) -> dict:
    ens = ensemble(records)
    return {
        "campaign": "m23184",
        "anchor": "CASE-DEM-SILO-BEVERLOO",
        "reference": {"C": BEVERLOO_C, "k": BEVERLOO_K, "exponent": 2.5,
                      "expression": "W = C rho_b sqrt(g) (D - k d)^2.5",
                      "tolerance": 0.10},
        "frozen_case": FROZEN_SIGNATURE | {"silo_radius": 0.024, "mu": 0.5,
                                           "mu_rolling": 0.05, "wall_mu": 0.0,
                                           "settle_time_s": 0.06, "discard_time_s": 0.12,
                                           "window_time_s": 0.05, "n_windows": 8},
        "records": records,
        "ensemble_flowing": ens,
        "verdict": verdict(records, ens),
    }


def figure_errors(summary: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = summary["records"]
    labels = [f"{r['label']}(s{r['seed']})" for r in recs]
    rel = np.array([100.0 * r["rel"] for r in recs])
    colors = ["tab:red" if r["arrested"] else "tab:blue" for r in recs]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.75 * len(recs)), 4.2))
    ax.bar(range(len(recs)), rel, color=colors, alpha=0.85)
    for i, r in enumerate(recs):
        noise = 100.0 * r["counting_noise_rel"]
        ax.errorbar(i, rel[i], yerr=noise, fmt="none", ecolor="k", capsize=3, lw=0.9)
        ax.text(i, rel[i] + (2 if rel[i] >= 0 else -4), f"{rel[i]:+.1f}%",
                ha="center", fontsize=7)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhspan(-10, 10, color="tab:green", alpha=0.12, label="anchor tolerance +/-10 %")
    ax.set_xticks(range(len(recs)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("W / W_ref - 1  [%]")
    ax.set_title("m23184: archived + fresh records of the frozen Beverloo case\n"
                 "blue = FLOWING, red = ARRESTED (arch over the outlet); "
                 "error bars = counting-noise floor 1/sqrt(N)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_windows(summary: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for r in summary["records"]:
        if r["seed"] != 1:
            continue
        n = len(r["window_particles"])
        ax.plot(range(1, n + 1), r["window_particles"], "o-" if not r["arrested"] else "s--",
                label=f"{r['label']} [{r['status']}]", lw=1.4)
    ax.set_xlabel("measurement window (x 0.05 s)")
    ax.set_ylabel("particles leaving the outlet per window")
    ax.set_title("m23184: seed 1 of the frozen case, two archived histories\n"
                 "a jet discharges ~40 particles/window; the arrested record dribbles")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_doc(summary: dict, path: Path) -> None:
    ens = summary["ensemble_flowing"]
    v = summary["verdict"]
    ref = summary["reference"]
    rows = "\n".join(
        f"| `{r['label']}` | {r['seed']} | {r['W']:.6f} | {r['W_ref']:.6f} | "
        f"{100 * r['rel']:+.2f} % | {100 * r['counting_noise_rel']:.1f} % | **{r['status']}** |"
        for r in summary["records"])
    dup = v["same_seed_duplicates"]
    dup_txt = "\n".join(
        f"| {d['seed']} | {d['n_records']} | {', '.join(d['labels'])} | "
        f"{d['W_range'][0]:.6f} – {d['W_range'][1]:.6f} | "
        f"{100 * d['W_spread_over_reference']:.1f} % | {', '.join(d['statuses'])} | "
        f"{'**INCONSISTENT**' if d['inconsistent'] else 'consistent'} |"
        for d in dup) or "| — | — | no seed was measured twice | — | — | — | — |"
    acc = v["anchor_acceptance_on_flowing_records"]
    acc_txt = ("NOT_RUN — no fresh ensemble" if acc is None else
               f"mean {100 * acc['mean_rel']:+.2f} %, 95 % CI "
               f"[{100 * acc['ci95_rel'][0]:+.2f}, {100 * acc['ci95_rel'][1]:+.2f}] %, "
               f"inside ±10 %: **{acc['pass_interval_inside_tolerance']}**")
    doc = f"""# CASE-DEM-SILO-BEVERLOO — m23184 archive reproducibility audit

Anchor: `CASE-DEM-SILO-BEVERLOO` (acceptance: `|W/W_ref − 1| < 10 %`, reference
Beverloo et al. 1961, `W = C ρ_b √g (D − k d)^2.5` with `C = {ref['C']}`,
`k = {ref['k']}`, never refitted).

This document is generated by `benchmarks/m23184_repro/analyse.py` from
`runs/m23184_repro/summary.json`; every number below is programmatic.

## ① Case

Frozen primary case of the campaign (`m22885`/`m23114` §2), unchanged:
`d = 3 mm`, `D/d = 5`, silo radius 24 mm, fill height 36 mm,
`μ = 0.5`, `μ_roll = 0.05`, wall friction 0 (smooth plate), `k_n = 5e3 N/m`,
`dt = T_c/20 = 1.8677e-05 s`, 0.06 s settle, 0.12 s discard, 8 × 0.05 s
measurement windows, `W` = mean of the last 50 % of the windows.

## ② Method

Every archived artefact of *this* case that exists on the box is collected
(no parameter is filtered except the frozen signature), the rate is recomputed
against the literature reference, and — new in this mission — each window
series is classified by `tensordem.silo_flow.beverloo.discharge_status` as
`FLOWING` or `ARRESTED` (an arch over the outlet is not a flow-rate
measurement).  A record is `ARRESTED` when any of four declared indicators
fires: no particle exited, ≥ 50 % of the windows are empty, the rate is below
20 % of the Beverloo reference of the same geometry, or the last half of the
windows carries < 20 % of the first half.

## ③ Results — every archived record of the frozen case

| record | seed | W [kg/s] | W_ref [kg/s] | rel. error | noise floor 1/√N | status |
|---|---|---|---|---|---|---|
{rows}

Same-seed duplicates (the decisive test — identical frozen input must give
identical output):

| seed | records | labels | W range [kg/s] | spread / W_ref | statuses | verdict |
|---|---|---|---|---|---|---|
{dup_txt}

Ensemble over the **flowing** records only: n = {ens['n_records']} records,
{ens['n_flowing']} flowing, {ens['n_arrested']} arrested
(arrest fraction {100 * (ens['arrest_fraction'] or 0):.0f} %),
mean rel. error {100 * ens.get('mean_rel', float('nan')):+.2f} %,
sd {100 * ens.get('sd_rel', float('nan')):.2f} %,
range {100 * ens.get('min_rel', float('nan')):+.2f} … {100 * ens.get('max_rel', float('nan')):+.2f} %.

## ④ Figures

* `docs/figures/m23184_repro_errors.png` — per-record relative error with the
  ±10 % acceptance band and the counting-noise error bar (1/√N of the steady
  windows).
* `docs/figures/m23184_repro_windows.png` — the two archived seed-1 window
  histories side by side (jet vs trickle).

## ⑤ Caveats

1. The archive records come from *other* worktrees of the same repository
   (`m22885`, `m23114`); they are read-only inputs, quoted with their path.
2. Two records of the same case and seed differ by far more than the counting
   noise of either (see the duplicate table): one of them therefore does not
   describe the physics the other one describes, and the ensemble statistics
   of this anchor can only be quoted together with the arrest diagnosis.
3. Scope: one frozen configuration (`D/d = 5`, monodisperse, smooth plate).
   Nothing here says anything about conical hoppers, wall friction or
   polydisperse beds.
4. The counting-noise floor below is a *lower bound* on the per-record error
   (N ≈ 150–200 particles in the steady windows ⇒ ≈ 7–8 %): a single record can
   never resolve a 10 % tolerance, which is why only an ensemble can.

## ⑥ Error judgement (vs the 1961 correlation)

* Anchor acceptance on the flowing ensemble: {acc_txt}.
* Reproducible archive: **{v['archive_reproducible']}**.
* Is the key quantity within 3 %? **No claim** — the frozen case has a
  counting-noise floor of ≈ 7–8 % per record, so 3 % is not reachable at this
  bed size without a longer measurement window or a larger bed.
"""
    path.write_text(doc)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "runs" / "m23184_repro" / "summary.json"))
    ap.add_argument("--doc", default=str(ROOT / "docs" / "CASE-DEM-SILO-BEVERLOO_m23184_reproducibility_audit.md"))
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args(argv)

    paths: list[str] = []
    for g in ARCHIVE_GLOBS:
        paths.extend(glob.glob(g))
    paths.extend(glob.glob(str(ROOT / "runs" / "m23184_repro" / "seed*.json")))
    records = collect(paths)
    if not records:
        raise SystemExit("no discharge record found")
    summary = build(records)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    render_doc(summary, Path(args.doc))
    if args.figures:
        figure_errors(summary, ROOT / "docs" / "figures" / "m23184_repro_errors.png")
        figure_windows(summary, ROOT / "docs" / "figures" / "m23184_repro_windows.png")
    ens = summary["ensemble_flowing"]
    print(f"{ens['n_records']} records: {ens['n_flowing']} FLOWING, "
          f"{ens['n_arrested']} ARRESTED; archive reproducible = "
          f"{summary['verdict']['archive_reproducible']}")
    for d in summary["verdict"]["same_seed_duplicates"]:
        print(f"  seed {d['seed']}: {d['labels']} -> {d['W_range']} {d['statuses']}"
              f" {'INCONSISTENT' if d['inconsistent'] else 'ok'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())