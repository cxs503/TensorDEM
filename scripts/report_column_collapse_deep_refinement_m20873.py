"""Figures + machine-generated markdown table for the m20873 deep refinement.

Everything is read from ``docs/artifacts/column_collapse_deep_refinement_m20873/
verdict.json`` (written by ``scripts/analyze_column_collapse_deep_refinement_
m20873.py``); no number is typed by hand.

Figures
-------
1. ``fig_refinement_ladders.png`` — runout-law error vs resolution for ``a=1``
   (8/12/16) and ``a=3`` (8/12), with the ±15 % acceptance band and the archived
   ``a=2`` deep-branch points.
2. ``fig_anchor_coverage.png`` — the whole anchor coverage: runout error vs
   aspect ratio at every resolution the anchor has ever run (m20337 archive +
   this mission), band shown.
3. ``fig_time_vs_resolution.png`` — emplacement-time error vs resolution on the
   same ladders: the shrinkage test for the second Lube prediction.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "artifacts",
                       "column_collapse_deep_refinement_m20873")
M20337_CSV = os.path.join(ROOT, "docs", "artifacts",
                          "column_collapse_resolution_ladder_m20337",
                          "cases_resolution_ladder.csv")
BAND = 0.15


def load_verdict() -> dict:
    with open(os.path.join(OUT_DIR, "verdict.json")) as fh:
        return json.load(fh)


def archived_rows() -> list[dict]:
    import csv
    rows = []
    with open(M20337_CSV) as fh:
        for r in csv.DictReader(fh):
            rows.append({"case_id": r["case_id"],
                         "a_nom": float(r["aspect_ratio"]),
                         "a_meas": float(r["aspect_measured"]),
                         "gpr": float(r["grains_per_radius"]),
                         "err_runout": float(r["runout_err"]),
                         "n_particles": int(float(r["n_particles"]))})
    return rows


def fig_ladders(payload: dict, path: str) -> str:
    rows = payload["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhspan(-100 * BAND, 100 * BAND, color="0.85", zorder=0,
               label="acceptance band ±15 %")
    ax.axhline(0.0, color="0.4", lw=1)
    styles = {"R_a1": ("a = 1  (linear branch)", "o", "tab:blue"),
              "R_a3": ("a = 3  (square-root branch)", "s", "tab:red")}
    for key, (label, marker, colour) in styles.items():
        sel = [r for r in rows if r["case_id"].startswith(key + "_g")]
        sel.sort(key=lambda r: r["grains_per_radius"])
        ax.plot([r["grains_per_radius"] for r in sel],
                [100 * r["err_runout"] for r in sel],
                marker=marker, color=colour, lw=1.6, label=label)
        for r in sel:
            ax.annotate(f"N={r['n_particles']}",
                        (r["grains_per_radius"], 100 * r["err_runout"]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=7, ha="center")
    deep = [r for r in rows if r["aspect_nominal"] == 2.0]
    if deep:
        ax.plot([r["grains_per_radius"] for r in deep],
                [100 * r["err_runout"] for r in deep],
                marker="^", ls="--", color="tab:green",
                label="a = 2  (deep, archived m20337)")
    ax.set_xlabel("grains per column radius  (grid refinement, $r_i$ and $\\delta/d$ frozen)")
    ax.set_ylabel("runout error $r_\\infty/r_i$ vs Lube (2004)  [%]")
    ax.set_title("CASE-COLUMN-COLLAPSE · refinement ladders at $a=1$, $a=3$ (m20873)")
    ax.set_xticks(sorted({r["grains_per_radius"] for r in rows}))
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_coverage(payload: dict, path: str) -> str:
    arch = archived_rows()
    fresh = [r for r in payload["rows"] if r["source"].startswith("this mission")]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.axhspan(-100 * BAND, 100 * BAND, color="0.85", zorder=0,
               label="acceptance band ±15 %")
    ax.axhline(0.0, color="0.4", lw=1)
    ax.scatter([r["a_meas"] for r in arch], [100 * r["err_runout"] for r in arch],
               marker="x", s=46, color="tab:orange", label="archived m20337 ladder")
    ax.scatter([r["aspect_measured"] for r in fresh],
               [100 * r["err_runout"] for r in fresh],
               marker="o", s=52, color="tab:blue", zorder=3,
               label="this mission (a=1: 8/12/16, a=3: 8/12)")
    for r in fresh:
        ax.annotate(f"a={r['aspect_nominal']:g}, gpr={r['grains_per_radius']:g}",
                    (r["aspect_measured"], 100 * r["err_runout"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=7)
    ax.axvline(1.7, ls=":", color="0.3", lw=1)
    ax.text(1.72, 13.4, "Lube regime break $a=1.7$", rotation=90, fontsize=8,
            color="0.3", va="top")
    ax.set_xlabel("measured aspect ratio $a_{meas}=h_i/r_i$")
    ax.set_ylabel("runout error vs Lube (2004)  [%]")
    ax.set_title("Anchor coverage of the runout law across aspect ratio and resolution")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_time(payload: dict, path: str) -> str:
    rows = payload["rows"]
    arch = archived_rows()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhspan(-100 * BAND, 100 * BAND, color="0.85", zorder=0,
               label="acceptance band ±15 %")
    ax.axhline(0.0, color="0.4", lw=1)
    for key, marker, colour in (("R_a1", "o", "tab:blue"), ("R_a3", "s", "tab:red")):
        sel = sorted([r for r in rows if r["case_id"].startswith(key + "_g")],
                     key=lambda r: r["grains_per_radius"])
        ax.plot([r["grains_per_radius"] for r in sel],
                [100 * r["err_t95"] for r in sel],
                marker=marker, color=colour, lw=1.6,
                label=f"$a={sel[0]['aspect_nominal']:g}$ (this mission)")
    ash = sorted([r for r in arch if r["a_nom"] in (0.5, 1.5, 2.0)],
                 key=lambda r: r["a_nom"])
    # archived t95 errors are not in the figure's CSV columns read above; the
    # verdict payload carries them for the a=2 deep points only.
    del ash
    # NOTE: the archived a=2 ladder (m20337) reports the *0.95* arrival
    # definition while m18261/m20873 freeze fraction 0.90 as primary, so those
    # points are NOT plotted here (mixing definitions would fake a trend).
    ax.text(0.02, 0.97, "arrival definition: front reaches 90 % of final runout",
            transform=ax.transAxes, fontsize=8, va="top", color="0.35")
    ax.set_xlabel("grains per column radius")
    ax.set_ylabel("$t_{90}$ error vs $3\\sqrt{h_i/g}$  [%]")
    ax.set_title("Emplacement time vs resolution: the second Lube prediction")
    ax.set_xticks(sorted({r["grains_per_radius"] for r in rows}))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def markdown_table(payload: dict) -> str:
    lines = ["| case | a_nom | a_meas | gpr | N | $r_\\infty/r_i$ | ref(a_meas) | "
             "err runout | $t_{95}$ [s] | ref $t$ [s] | err $t_{95}$ | stop |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in payload["rows"]:
        lines.append(
            f"| `{r['case_id']}` | {r['aspect_nominal']:g} | {r['aspect_measured']:.4f} | "
            f"{r['grains_per_radius']:g} | {r['n_particles']} | {r['runout_ratio']:.4f} | "
            f"{r['ref_runout_ratio']:.4f} | {100 * r['err_runout']:+.2f} % | "
            f"{r['t95']:.4f} | {r['ref_time']:.4f} | {100 * r['err_t95']:+.2f} % | "
            f"{r['stop_reason']} |")
    return "\n".join(lines)


def markdown_verdicts(payload: dict) -> str:
    lines = ["| criterion | result | measured | threshold |", "|---|---|---|---|"]
    for key, val in payload["verdicts"].items():
        if key == "P7_INTEGRITY":
            lines.append(f"| `{key}` | **{val['result']}** | stop_reason all "
                         f"`at_rest`; overlap clause `{val['overlap_clause']}` | — |")
        elif key == "insufficient":
            lines.append(f"| insufficient | — | {val} | — |")
        else:
            shown = val.get("value", val.get("value_pp", val.get("worst")))
            thr = val.get("threshold", val.get("threshold_pp"))
            unit = "%" if "value" in val or "worst" in val else " pp"
            lines.append(f"| `{key}` | **{val['result']}** | {shown * (100 if unit == '%' else 1):.2f}{unit} "
                         f"| {thr * (100 if unit == '%' else 1):g}{unit} |")
    return "\n".join(lines)


def main() -> int:
    payload = load_verdict()
    figs = [fig_ladders(payload, os.path.join(OUT_DIR, "fig_refinement_ladders.png")),
            fig_coverage(payload, os.path.join(OUT_DIR, "fig_anchor_coverage.png")),
            fig_time(payload, os.path.join(OUT_DIR, "fig_time_vs_resolution.png"))]
    md = ("## Table — five-case ladder (program-generated from verdict.json)\n\n"
          + markdown_table(payload) + "\n\n## Pre-registered verdicts\n\n"
          + markdown_verdicts(payload) + "\n")
    with open(os.path.join(OUT_DIR, "tables_m20873.md"), "w") as fh:
        fh.write(md)
    print(md)
    for f in figs:
        print("figure:", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())