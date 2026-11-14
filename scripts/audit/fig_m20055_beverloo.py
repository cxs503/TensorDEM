#!/usr/bin/env python3
"""Figures for the m20055 (CASE-DEM-SILO-BEVERLOO) board audit.

Reads ``docs/board_audit/m20232/audit_m20055_beverloo.json`` (written by
``scripts/audit/audit_m20055_beverloo.py``) only; no number is typed in.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "docs" / "board_audit" / "m20232"
DATA = json.loads((OUT / "audit_m20055_beverloo.json").read_text())


def fig_rate():
    bins = DATA["rate_bins"]
    t = np.array([b[0] for b in bins])
    w = np.array([b[1] for b in bins])
    W_ref = DATA["beverloo_ref"]["W_ref_kg_s"]
    W_filed = DATA["filed"]["W_got_kg_s"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(t, w, "o-", color="C0", ms=4, lw=1.2, label="DEM discharge rate, 0.25 s bins")
    ax.axhline(W_ref, color="C3", lw=1.8, ls="-",
               label=f"Beverloo 1961 (C=0.58, k=1.4): {W_ref:.3f} kg/s")
    ax.axhline(W_filed, color="C0", lw=1.0, ls="--",
               label=f"filed steady W (mass quantile 0.2-0.8): {W_filed:.3f} kg/s")
    ax.axhline(np.mean(w), color="0.4", lw=1.0, ls=":",
               label=f"bin mean: {np.mean(w):.3f} kg/s")
    ax.set_xlabel("t [s]  (t = 0 is the instant the plug is removed)")
    ax.set_ylabel("W [kg/s]")
    ax.set_title("m20055 CASE-DEM-SILO-BEVERLOO, D/d = 4: the only filed case\n"
                 "the flow fluctuates +/-12 % bin-to-bin; no clean plateau")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = OUT / "fig_m20055_rate_vs_time.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_estimators():
    est = DATA["estimators"]
    names = list(est)
    rel = [est[k]["rel_pct"] for k in names]
    colors = ["C2" if abs(r) < 10.0 else ("C1" if abs(r) < 15.0 else "C3") for r in rel]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    y = np.arange(len(names))
    ax.barh(y, rel, color=colors)
    ax.axvline(0, color="0.3", lw=0.8)
    for v, lab in ((10.0, "+/-10 % case tolerance"), (15.0, "15 % acceptability bound")):
        ax.axvline(v, color="0.5", lw=0.8, ls="--")
        ax.axvline(-v, color="0.5", lw=0.8, ls="--")
        ax.text(v, len(names) - 0.4, lab, fontsize=7, rotation=90, va="top", ha="right")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("relative error of W vs Beverloo 1961  [%]")
    ax.set_title("Same raw discharge record, different rate estimators\n"
                 "only the legacy fixed-window mean (with the plug-phase zero bin) sits inside tolerance")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    p = OUT / "fig_m20055_estimators.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


if __name__ == "__main__":
    print(fig_rate())
    print(fig_estimators())