"""Figures + machine-readable summary for the m21164 energy-closure audit.

Reads the archived runs written by
``scripts/audit_column_energy_closure_m21164.py --json`` and the pre-repair
numbers measured in this session before the ledger repair, and renders:

* ``fig_closure_channels.png`` -- canonical one-grain closure per channel, with the
  pre-repair imbalance reproduced by *dropping* the wall rolling channel;
* ``fig_dt_contraction.png`` -- remaining residual vs timestep (pre-registered H2
  band 0.35..0.65 per halving);
* ``fig_ledger_waterfall.png`` -- released potential split into the booked channels
  for ``a = 1`` and ``a = 3``.

Run::

    PYTHONPATH=src python scripts/report_column_energy_closure_m21164.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ART = os.path.join(os.path.dirname(__file__), "..", "docs", "artifacts",
                   "column_energy_closure_m21164")

#: Pre-repair residuals measured in this session on the un-repaired tree with the
#: instrumented integrator (same protocols, same seeds).  Provenance: session log
#: recorded in docs/CASE_COLUMN_COLLAPSE_ENERGY_CLOSURE_m21164.md §3.1; they also
#: reproduce the independent archived readings of m20931 (10.86-12.36 % at
#: gpr 12-16), which is the cross-check that these numbers are not eyeballed.
PRE_REPAIR = {
    "C4_col_a3_g4": {"residual": 0.11802, "unbooked_wall_rolling": 0.05758},
    "C3_col_a1_g6": {"residual": 0.04103, "unbooked_wall_rolling": 0.02299},
    "canonical_pure_roll_E0": 0.2289,     # single grain, pure rolling
    "canonical_pure_slip_E0": 0.0910,     # single grain, pure sliding
}


def _fig_close_ratio():
    runs = json.load(open(os.path.join(ART, "closure_runs.json")))
    labels, post, wall = [], [], []
    for key, nice in (("C3_col_a1_g6", "a=1, gpr 6"), ("C4_col_a3_g4", "a=3, gpr 4")):
        labels.append(nice)
        post.append(100.0 * runs[key]["R"])
        wall.append(100.0 * runs[key]["wall_roll"])
    pre = [100.0 * PRE_REPAIR["C3_col_a1_g6"]["residual"],
           100.0 * PRE_REPAIR["C4_col_a3_g4"]["residual"]]

    fig, axp = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    x = np.arange(len(labels))
    w = 0.26
    axp.bar(x - w, pre, w, label="pre-repair ledger (residual)", color="#c0392b")
    axp.bar(x, wall, w, label="wall rolling resistance (never booked)", color="#e67e22")
    axp.bar(x + w, post, w, label="repaired ledger (residual)", color="#27ae60")
    for xi, v in zip(x - w, pre):
        axp.text(xi, v + 0.2, f"{v:.2f}%", ha="center", fontsize=8)
    for xi, v in zip(x, wall):
        axp.text(xi, v + 0.2, f"{v:.2f}%", ha="center", fontsize=8)
    for xi, v in zip(x + w, post):
        axp.text(xi, v + 0.2, f"{v:.2f}%", ha="center", fontsize=8)
    axp.axhline(0, color="k", lw=0.8)
    axp.set_xticks(x, labels)
    axp.set_ylabel("energy imbalance  [% of initial potential]")
    axp.set_title("Column-collapse energy ledger: the missing wall rolling channel")
    axp.legend(fontsize=8, loc="upper left")
    axp.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(ART, "fig_closure_channels.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def _fig_dt():
    runs = json.load(open(os.path.join(ART, "closure_runs.json")))
    keys = ["C4_col_a3_g4", "C5_col_a3_g4_dt2", "C6_col_a3_g4_dt4"]
    dt = np.array([runs[k]["dt"] for k in keys])
    R = np.array([100.0 * runs[k]["R"] for k in keys])
    fig, axp = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    axp.loglog(dt, R, "o-", label="a=3, gpr 4 (repaired ledger)")
    axp.loglog(dt, R[0] * dt / dt[0], "k--", label="slope 1 (O(dt))")
    prev = None
    for k, r, d in zip(keys, R, dt):
        ratio = None if prev is None else r / prev
        prev = r
        axp.annotate(f"{r:.2f}%" + ("" if ratio is None else f"  (x{ratio:.2f})"),
                     (d, r), textcoords="offset points", xytext=(6, 6), fontsize=8)
    axp.set_xlabel("timestep dt [s]")
    axp.set_ylabel("residual  [% of initial potential]")
    axp.set_title("Residual left after the repair contracts with dt (H2 band 0.35-0.65)")
    axp.legend(fontsize=8)
    axp.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(ART, "fig_dt_contraction.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def _fig_waterfall():
    runs = json.load(open(os.path.join(ART, "closure_runs.json")))
    fig, axp = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    labels = [("a=1, gpr 6", "C3_col_a1_g6"), ("a=3, gpr 4", "C4_col_a3_g4")]
    x = np.arange(len(labels))
    diss = [100.0 * runs[k]["diss"] for _, k in labels]
    wall = [100.0 * runs[k]["wall_roll"] for _, k in labels]
    rel = [100.0 * runs[k]["rel"] for _, k in labels]
    resid = [100.0 * runs[k]["R"] for _, k in labels]
    axp.bar(x - 0.2, diss, 0.4, label="physical dissipation booked", color="#2980b9")
    axp.bar(x, wall, 0.4, label="  of which wall rolling (was missing)", color="#e67e22")
    axp.bar(x + 0.2, rel, 0.4, label="tension-cutoff release loss", color="#8e44ad")
    for xi, v, r in zip(x + 0.2, rel, resid):
        axp.annotate(f"residual {r:.2f}%", (xi, v + 0.5), fontsize=8, ha="center")
    axp.set_xticks(x, [n for n, _ in labels])
    axp.set_ylabel("share of released potential  [%]")
    axp.set_title("Where the released potential goes (repaired ledger)")
    axp.legend(fontsize=8)
    axp.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(ART, "fig_ledger_waterfall.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    os.makedirs(ART, exist_ok=True)
    made = [_fig_close_ratio(), _fig_dt(), _fig_waterfall()]
    runs = json.load(open(os.path.join(ART, "closure_runs.json")))
    r = [runs[k]["R"] for k in ("C4_col_a3_g4", "C5_col_a3_g4_dt2", "C6_col_a3_g4_dt4")]
    summary = {
        "anchor": "CASE-COLUMN-COLLAPSE",
        "pre_repair_residual": {k: v["residual"] for k, v in PRE_REPAIR.items()
                                if isinstance(v, dict)},
        "post_repair_residual": {k: runs[k]["R"] for k in runs},
        "dt_contraction_ratios": [r[1] / r[0], r[2] / r[1]],
        "prereg_band": [0.35, 0.65],
        "h2_verdict": ("pass" if all(0.35 <= q <= 0.65
                                     for q in (r[1] / r[0], r[2] / r[1])) else "fail"),
        "figures": [os.path.basename(p) for p in made],
    }
    with open(os.path.join(ART, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())