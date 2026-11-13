"""Aggregate the CASE-PENETRATION sweep records into one table + figures.

Reads every ``study.json`` under ``--root`` (written by
``scripts/penetration_sweep.py``), writes ``sweep_summary.json`` and two
publication figures:

``fig_floor_clearance.png``  alpha vs floor clearance (the semi-infinite limit)
``fig_alpha_window.png``     the window-resolved alpha ladder per case

The numbers in the JSON are read from the records only -- nothing is typed by
hand into the report (zero transcription).

Usage::

    PYTHONPATH=src python scripts/penetration_sweep_report.py \
        --root artifacts/case_penetration --out artifacts/case_penetration
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

REFERENCE_ALPHA = 35.0
REFERENCE_ALPHA_ERR = 5.0


def load(root: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "**", "study.json"),
                                 recursive=True)):
        with open(path) as fh:
            row = json.load(fh)
        row["path"] = os.path.relpath(path, root)
        rows.append(row)
    return rows


def table(rows: list[dict]) -> dict:
    cols = ("label", "rd", "penetration_speed", "speed_ratio", "n_grains",
            "floor_clearance_over_r", "alpha_shape", "alpha_free", "mean_force_N",
            "reference_mean_force_N", "force_variation_coefficient",
            "kinetic_energy_max_J", "status", "wall_clock_s")
    out = {}
    for key in cols:
        out[key] = [row.get(key) for row in rows]
    return out


def markdown_table(rows: list[dict]) -> str:
    """Markdown of the measured cases (numbers read from the records only)."""
    head = ("| case | R/d | v [m/s] | v/sqrt(gR) | N | floor [R] | alpha (shape) "
            "| alpha (free) | F mean [N] | F ref [N] | status |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for r in sorted(rows, key=lambda r: (r["rd"], r["floor_clearance_over_r"],
                                         r["penetration_speed"])):
        lines.append(
            f"| {r['label']} | {r['rd']:.1f} | {r['penetration_speed']:.4g} "
            f"| {r['speed_ratio']:.3f} | {r['n_grains']} "
            f"| {r['floor_clearance_over_r']:.2f} | {r['alpha_shape']:.1f} "
            f"| {r['alpha_free']:.1f} | {r.get('mean_force_N', float('nan')):.3f} "
            f"| {r.get('reference_mean_force_N', float('nan')):.3f} | {r['status']} |")
    return head + "\n".join(lines) + "\n"


def figures(rows: list[dict], out_dir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    floored = sorted([r for r in rows if r.get("floor_clearance_over_r")],
                     key=lambda r: r["floor_clearance_over_r"])
    if floored:
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        x = np.array([r["floor_clearance_over_r"] for r in floored])
        y = np.array([r["alpha_shape"] for r in floored])
        ax.plot(x, y, "o-", label=r"DEM, R/d = %g" % floored[0]["rd"])
        ax.axhspan(REFERENCE_ALPHA - REFERENCE_ALPHA_ERR,
                   REFERENCE_ALPHA + REFERENCE_ALPHA_ERR, color="0.8",
                   label=r"Brzinski 2013: $\alpha$ = 35 $\pm$ 5")
        ax.axhline(REFERENCE_ALPHA, color="0.35", lw=1.0, ls="--")
        for r in floored:
            ax.annotate(f"{r['alpha_shape']:.0f}\n(N={r['n_grains']})",
                        (r["floor_clearance_over_r"], r["alpha_shape"]),
                        textcoords="offset points", xytext=(4, 6), fontsize=7)
        ax.set_xlabel("floor clearance below deepest intruder point  [R]")
        ax.set_ylabel(r"fitted $\alpha$")
        ax.set_yscale("log")
        ax.set_title("CASE-PENETRATION: rigid floor below the intruder")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, "fig_floor_clearance.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    laddered = [r for r in rows if r.get("alpha_window")]
    if laddered:
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        for r in sorted(laddered, key=lambda r: r["label"]):
            zw = np.array([w["z_to_over_r"] for w in r["alpha_window"]])
            aw = np.array([w["alpha"] for w in r["alpha_window"]])
            ax.plot(zw, aw, "o-", ms=3,
                    label=f"{r['label']} (clear {r['floor_clearance_over_r']:.1f} R)")
        ax.axhspan(REFERENCE_ALPHA - REFERENCE_ALPHA_ERR,
                   REFERENCE_ALPHA + REFERENCE_ALPHA_ERR, color="0.8", label="reference")
        ax.set_xlabel("upper fit limit  z/R")
        ax.set_ylabel(r"window-resolved $\alpha$")
        ax.set_yscale("log")
        ax.set_title("additive contamination: alpha grows over a shallow window")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, "fig_alpha_window.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="artifacts/case_penetration")
    ap.add_argument("--out", default="artifacts/case_penetration")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    rows = load(args.root)
    if not rows:
        raise SystemExit(f"no study.json under {args.root}")
    summary = dict(reference_alpha=REFERENCE_ALPHA,
                   reference_alpha_band=[REFERENCE_ALPHA - REFERENCE_ALPHA_ERR,
                                         REFERENCE_ALPHA + REFERENCE_ALPHA_ERR],
                   cases=table(rows),
                   records=[r["path"] for r in rows])
    summary["figures"] = figures(rows, args.out)
    summary["markdown"] = markdown_table(rows)
    with open(os.path.join(args.out, "sweep_table.md"), "w") as fh:
        fh.write(summary["markdown"])
    path = os.path.join(args.out, "sweep_summary.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: summary["cases"][k] for k in (
        "label", "rd", "penetration_speed", "floor_clearance_over_r",
        "alpha_shape", "alpha_free", "status")}, indent=2))
    print(json.dumps({"summary": path, "figures": summary["figures"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())