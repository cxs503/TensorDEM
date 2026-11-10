#!/usr/bin/env python3
"""Post-processing of a size-segregation campaign: criteria, figures, markdown.

Every number written by this script is read from the per-case JSON records that
the campaign produced; nothing is typed by hand and no synthetic data path
exists.  The output is

* ``<out-summary>``        machine-readable verdict (per-case metrics + criteria),
* ``<fig-dir>/*.png``      the figures cited by the report,
* ``<markdown>``           a draft section list for the paper-level report.

Usage::

    PYTHONPATH=src python scripts/segregation_campaign_report.py \
        --run-dir runs/segregation_v2/out \
        --fig-dir artifacts/segregation_v2/figures \
        --out-summary artifacts/segregation_v2/summary.json \
        --markdown artifacts/segregation_v2/report_tables.md \
        --index S_fine --gamma-family 2.0:0.3:20.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensordem.segregation.analysis import (  # noqa: E402
    GAMMA_C_LITERATURE,
    P_LITERATURE,
    RATIOS,
    evaluate_criteria,
    load_records,
)
from tensordem.segregation.bed2d import BidisperseBed  # noqa: E402,F401  (documentation)
from tensordem.segregation.transfer import HoldGeometry, distribution_from_record, gm_mapping  # noqa: E402

GEO = HoldGeometry(length_m=20.0, breadth_m=12.0, depth_m=6.0, n_longitudinal=8)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mean(values) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def _match(value, target) -> bool:
    """Filter predicate: exact for strings, tolerance-based for numbers."""
    if isinstance(target, str):
        return value == target
    try:
        return abs(float(value) - float(target)) < 1e-9
    except (TypeError, ValueError):
        return False


def _family(records: dict[str, dict], **filters) -> list[dict]:
    out = []
    for rec in records.values():
        spec = rec["spec"]
        if all(_match(spec.get(k), v) for k, v in filters.items()):
            out.append(rec)
    return sorted(out, key=lambda r: (r["spec"]["ratio"], r["spec"]["gamma"],
                                      r["spec"]["seed"]))


def _err(rec: dict, key: str) -> float:
    return float(rec["metrics"].get("segregation_blocks", {}).get(f"{key}_stderr", 0.0))


# --------------------------------------------------------------------- figures
def fig_ratio(records, criteria, index, path):
    fam = _family_ratio(records)
    if not fam:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ratios, means, errs, seeds = [], [], [], []
    for ratio in sorted(fam):
        recs = fam[ratio]
        vals = [r["metrics"]["segregation"][index] for r in recs]
        ratios.append(ratio)
        means.append(float(np.mean(vals)))
        errs.append(float(np.mean([_err(r, index) for r in recs])) if recs else 0.0)
        seeds.append(vals)
    ax.errorbar(ratios, means, yerr=errs, marker="o", lw=1.6, capsize=3,
                label=f"mean {index} (block stderr)")
    for ratio, vals in zip(ratios, seeds):
        ax.plot([ratio] * len(vals), vals, "k.", ms=4, alpha=0.6)
    ax.axhline(0.02, color="tab:red", ls="--", lw=0.9, label="±0.02 control band")
    ax.axhline(-0.02, color="tab:red", ls="--", lw=0.9)
    ax.axvspan(0.9, 1.1, color="gray", alpha=0.15)
    ax.set_xlabel("$d_l/d_s$")
    ax.set_ylabel(f"$S$ ({index})")
    ax.set_title("Segregation index versus diameter ratio (common drive)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"name": os.path.basename(path), "caption":
            f"Measured {index} versus diameter ratio at the common drive level; "
            "black dots are individual seeds, the grey band is the equal-size "
            "negative control."}


def _family_ratio(records, *, fraction=0.3):
    fam: dict[float, list[dict]] = {}
    gammas = {}
    for rec in records.values():
        spec = rec["spec"]
        if spec["drive"] != "vib":
            continue
        if abs(spec["large_fraction"] - fraction) > 1e-9:
            continue
        fam.setdefault(round(spec["ratio"], 6), []).append(rec)
        gammas.setdefault(round(spec["ratio"], 6), set()).add(round(spec["gamma"], 6))
    if not fam:
        return {}
    common = set.intersection(*gammas.values())
    if not common:
        return {}
    gam = max(common)
    return {ratio: [r for r in recs if abs(r["spec"]["gamma"] - gam) < 1e-9]
            for ratio, recs in fam.items()}


def fig_gamma(records, criteria, index, path, family):
    ratio, fraction, freq = family
    recs = _family(records, ratio=ratio, large_fraction=fraction, frequency=freq,
                   drive="vib")
    if len(recs) < 3:
        recs = _family(records, ratio=ratio, large_fraction=fraction, drive="vib")
    if len(recs) < 3:
        return None
    gammas = sorted({r["spec"]["gamma"] for r in recs})
    vals, errs, signed = [], [], []
    for gam in gammas:
        sel = [r for r in recs if abs(r["spec"]["gamma"] - gam) < 1e-9]
        s = [r["metrics"]["segregation"][index] for r in sel]
        vals.append(float(np.mean(np.abs(s))))
        signed.append(float(np.mean(s)))
        errs.append(float(np.mean([_err(r, index) for r in sel])))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.errorbar(gammas, vals, yerr=errs, marker="s", lw=1.6, capsize=3,
                label=f"|{index}| (mean of seeds)")
    ax.plot(gammas, signed, "o-", ms=4, lw=1.0, alpha=0.7, label=f"signed {index}")
    d = criteria.get("d_gamma_threshold", {})
    fit = d.get("fit", {})
    if np.isfinite(fit.get("gamma_c", float("nan"))) and np.isfinite(fit.get("s_sat", float("nan"))):
        gg = np.linspace(min(gammas), max(gammas), 200)
        model = fit["s_sat"] * np.maximum(0.0, 1.0 - np.exp(-(gg - fit["gamma_c"]) / fit["gamma_0"]))
        ax.plot(gg, model, "k--", lw=1.0,
                label=f"fit: $\\Gamma_c$={fit['gamma_c']:.2f}, $S_{{sat}}$={fit['s_sat']:.3f}")
        ax.axvline(fit["gamma_c"], color="k", ls=":", lw=0.8)
    ax.axvline(GAMMA_C_LITERATURE, color="tab:red", ls="--", lw=0.9,
               label=f"literature $\\Gamma_c\\approx${GAMMA_C_LITERATURE}")
    ax.set_xlabel(r"$\Gamma = A\omega^2/g$")
    ax.set_ylabel(f"segregation index {index}")
    ax.set_title("Drive threshold and saturation (ratio %.2f)" % ratio)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"name": os.path.basename(path), "caption":
            f"Segregation index versus vibration drive Γ at $d_l/d_s$={ratio}: "
            "threshold fit and the literature onset Γ_c≈1.2."}


def fig_rise(records, criteria, path):
    b = criteria.get("b_ratio_scaling", {})
    by_ratio = b.get("by_ratio", {})
    if len(by_ratio) < 3:
        return None
    ratios = sorted(by_ratio)
    xs = np.array([r - 1.0 for r in ratios])
    ys = np.array([abs(by_ratio[r]["v_up_mean_m_s"]) for r in ratios])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.loglog(xs, ys, "o", ms=6, label="measured $|v_{up}|$")
    fit = b.get("power_law", {})
    if np.isfinite(fit.get("p", float("nan"))):
        gg = np.linspace(xs.min(), xs.max(), 50)
        ax.loglog(gg, fit["c"] * gg ** fit["p"], "k--", lw=1.0,
                  label=f"fit p={fit['p']:.2f} (lit. p={P_LITERATURE})")
    ax.set_xlabel("$d_l/d_s - 1$")
    ax.set_ylabel("$|v_{up}|$ (m/s)")
    ax.set_title("Rise-velocity scaling with the diameter-ratio excess")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"name": os.path.basename(path), "caption":
            "Large-species rise speed versus $(d_l/d_s-1)$ in log-log axes with "
            "the fitted exponent p."}


def fig_snapshots(records, path, index):
    vib = {r["spec"]["ratio"]: r for r in records.values()
           if r["spec"]["drive"] == "vib" and "snapshot" in r}
    if len(vib) < 2:
        return None
    picks = [min(vib), max(vib)]
    order = {0.0: "a", 1.0: "b", 2.0: "c", 3.0: "d"}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
    for k, (ratio, rec) in enumerate(zip(picks, [vib[p] for p in picks])):
        snap = rec["snapshot"]
        pos = np.asarray(snap["pos"], dtype=np.float64)
        rad = np.asarray(snap["radii"], dtype=np.float64)
        lg = np.asarray(snap["is_large"], dtype=bool)
        ax = axes[k]
        for mask, colour, label in ((~lg, "tab:orange", "small"), (lg, "tab:blue", "large")):
            ax.scatter(pos[mask, 0], pos[mask, 1], s=(rad[mask] * 900) ** 2,
                       c=colour, alpha=0.65, lw=0, label=label)
        ax.set_xlim(0, snap["width_m"])
        ax.set_ylim(0, snap["settled_bed_height_m"] * 2.2)
        ax.set_aspect("equal")
        ax.set_title(f"$d_l/d_s$={ratio:.1f}, S({index})={rec['metrics']['segregation'][index]:+.3f}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        if k == 0:
            ax.legend(fontsize=8, markerscale=0.25)
    fig.suptitle("Bed state at the end of the vibrated run (species map)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"name": os.path.basename(path), "caption":
            "Particle map at the end of the run for the smallest and largest "
            "diameter ratio: the segregation is visible as a colour gradient in "
            "the vertical direction."}


def fig_profiles(records, path, index):
    recs = [r for r in records.values() if r["spec"]["drive"] == "vib"]
    if not recs:
        return None
    ratios = sorted({r["spec"]["ratio"] for r in recs})
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for ratio in ratios:
        sel = [r for r in recs if abs(r["spec"]["ratio"] - ratio) < 1e-9]
        if not sel:
            continue
        z = np.asarray(sel[0]["metrics"]["profile_z_centres"], dtype=np.float64)
        f = np.asarray(sel[0]["metrics"]["profile_large_fraction"], dtype=np.float64)
        h = sel[0]["metrics"]["segregation"]["h_bed_m"]
        ax.plot(f, z / h, lw=1.3, label=f"$d_l/d_s$={ratio:.1f}")
    ax.axvline(0.3, color="k", ls=":", lw=0.8, label="nominal large fraction")
    ax.set_xlabel("large-species fraction of the solid area, $\\phi_l(z)$")
    ax.set_ylabel("$z/h_{bed}$")
    ax.set_title("Vertical composition profile (last-sample average)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"name": os.path.basename(path), "caption":
            "Vertical large-species fraction profile for the diameter-ratio "
            "family: the enriched layer and its thickness are the morphology "
            "input of the stability transfer."}


def fig_transfer(criteria, path):
    cases = criteria.get("e_transfer", {}).get("cases", {})
    rows = [(k, v["gm"]) for k, v in cases.items() if "gm" in v]
    if not rows:
        return None
    names = [r[0] for r in rows]
    dvcg = [abs(r[1].get("dVcg_m", float("nan"))) for r in rows]
    dtcg = [abs(r[1].get("dTcg_m", float("nan"))) for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.bar(x - 0.2, dvcg, width=0.38, label="|ΔVCG| (m)")
    ax.bar(x + 0.2, dtcg, width=0.38, label="|ΔTCG| (m)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("offset (m)")
    ax.set_title("Ship-side transfer: segregated cargo versus uniform reference")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"name": os.path.basename(path), "caption":
            "Closed-form mapping of the segregated cargo distribution to ΔVCG and "
            "ΔTCG per case."}


# ------------------------------------------------------------------- markdown
def build_markdown(records, criteria, figures, index, family) -> str:
    lines: list[str] = []
    A = lines.append
    A("# Segregation campaign — programmatic tables\n")
    A(f"Records: {len(records)} cases. Index used for the criteria: `{index}`.\n")
    A("## Per-case metrics\n")
    A("| case | drive | $d_l/d_s$ | $\\Gamma$ | $f$ (Hz) | $\\phi_l$ | S_fine | S_bn | "
      "$v_{up}$ (m/s) | mass cons. | blocks |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for name in sorted(records):
        rec = records[name]
        sp = rec["spec"]
        seg = rec["metrics"]["segregation"]
        A("| {n} | {d} | {r:.2f} | {g:.2f} | {f:.0f} | {p:.2f} | {sf:+.4f} | {sb:+.4f} | "
          "{v:+.3e} | {m:.1e} | {nb} |".format(
              n=name, d=sp["drive"], r=sp["ratio"], g=sp["gamma"], f=sp["frequency"],
              p=sp["large_fraction"], sf=seg["S_fine"], sb=seg["S_bn"],
              v=rec["metrics"]["rise_large"]["v_up_m_s"],
              m=rec["metrics"]["mass_conservation_rel"],
              nb=rec["metrics"].get("segregation_blocks", {}).get("n_blocks", 0)))
    A("\n## Criteria\n")
    A("| criterion | measured | limit | verdict |")
    A("|---|---|---|---|")
    a = criteria.get("a_negative_control", {})
    A(f"| (a) equal-size control max\\|S\\| | {a.get('max_abs_S_fine', float('nan')):.4f} | "
      f"≤0.02 | {'PASS' if a.get('pass') else 'FAIL'} |")
    b = criteria.get("b_ratio_scaling", {})
    p = b.get("power_law", {}).get("p", float("nan"))
    A(f"| (b) monotone in $d_l/d_s$ / exponent p | {b.get('strictly_monotone')} / "
      f"{p:.3f} | strict / 1.0±30% | {'PASS' if b.get('pass') else 'FAIL'} |")
    c = criteria.get("c_mass_conservation", {})
    A(f"| (c) mass conservation | {c.get('max_mass_conservation_rel', float('nan')):.2e} | "
      f"≤1e-9 | {'PASS' if c.get('pass') else 'FAIL'} |")
    d = criteria.get("d_gamma_threshold", {})
    fit = d.get("fit", {})
    A(f"| (d) threshold $\\Gamma_c$ | {fit.get('gamma_c', float('nan')):.3f} | 1.2±30% | "
      f"{'PASS' if d.get('pass') else 'FAIL'} |")
    A(f"| (d') saturation $S_{{sat}}$ | {fit.get('s_sat', float('nan')):.3f} | reported | — |")
    e = criteria.get("e_transfer", {})
    A(f"| (e) transfer vs shard sums | {e.get('worst_shard_rel', float('nan')):.2e} | "
      f"≤1e-6 | {'PASS' if e.get('pass') else 'FAIL'} |")
    f = criteria.get("f_seed_reproducibility", {})
    A(f"| (f) seed CV of S | {f.get('max_cv', float('nan')):.3f} | ≤0.15 | "
      f"{'PASS' if f.get('pass') else 'FAIL'} |")
    A("\n## Verdict\n")
    verdict = criteria.get("verdict", {})
    A(f"all_pass = **{verdict.get('all_pass')}**  "
      f"({sum(1 for v in verdict.get('criteria', {}).values() if v)}/"
      f"{len(verdict.get('criteria', {}))} criteria)\n")
    for fig in figures:
        if fig:
            A(f"![{fig['name']}]({fig['name']})")
            A("")
            A(f"*{fig['caption']}*")
            A("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--markdown", required=True)
    ap.add_argument("--index", default="S_fine", choices=("S_fine", "S_bn"))
    ap.add_argument("--gamma-family", default="2.0:0.3:20.0",
                    help="ratio:large_fraction:frequency used for the threshold scan")
    args = ap.parse_args(argv)

    ratio, fraction, freq = (float(v) for v in args.gamma_family.split(":"))
    records = load_records(args.run_dir)
    if not records:
        raise SystemExit(f"no records in {args.run_dir}")
    criteria = evaluate_criteria(records, index=args.index, gamma_ratio=ratio,
                                 gamma_fraction=fraction, gamma_frequency=freq)
    os.makedirs(args.fig_dir, exist_ok=True)
    figs = [
        fig_ratio(records, criteria, args.index, os.path.join(args.fig_dir, "S_vs_ratio.png")),
        fig_gamma(records, criteria, args.index, os.path.join(args.fig_dir, "S_vs_gamma.png"),
                  (ratio, fraction, freq)),
        fig_rise(records, criteria, os.path.join(args.fig_dir, "rise_scaling.png")),
        fig_snapshots(records, os.path.join(args.fig_dir, "bed_snapshots.png"), args.index),
        fig_profiles(records, os.path.join(args.fig_dir, "composition_profiles.png"), args.index),
        fig_transfer(criteria, os.path.join(args.fig_dir, "stability_transfer.png")),
    ]
    figs = [f for f in figs if f]
    summary: dict[str, Any] = {
        "run_dir": os.path.abspath(args.run_dir),
        "index": args.index,
        "gamma_family": {"ratio": ratio, "large_fraction": fraction, "frequency_hz": freq},
        "n_cases": len(records),
        "criteria": criteria,
        "figures": figs,
        "provenance": {
            name: {"path": os.path.relpath(rec["_path"]),
                   "sha256": _sha256(rec["_path"]),
                   "digest": rec.get("audit", {}).get("digest")}
            for name, rec in sorted(records.items())
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_summary)), exist_ok=True)
    with open(args.out_summary, "w") as fh:
        json.dump(summary, fh, indent=1)
    md = build_markdown(records, criteria, figs, args.index, (ratio, fraction, freq))
    os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
    with open(args.markdown, "w") as fh:
        fh.write(md)
    verdict = summary["criteria"]["verdict"]
    print(json.dumps({"n_cases": len(records), "verdict": verdict,
                      "figures": [f["name"] for f in figs]}, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())