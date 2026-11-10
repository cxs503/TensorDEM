#!/usr/bin/env python
"""Figures for the P155-N09 report (data come from the merged campaign JSON)."""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: str) -> dict:
    return json.loads(pathlib.Path(path).read_text())


def panel_profiles(ax, data: dict) -> None:
    for row in data["summary"]["shear"]:
        ratio = row["key"]["size_ratio"]
        if row["n_seeds"] < 1:
            continue
        S = row["S_mean"]
        ax.plot([ratio] * 2, [0, S], alpha=0.0)     # placeholder for legend order
    for name in ("shear", "vibration"):
        rows = [r for r in data["summary"][name] if r["key"]["size_ratio"] > 1.0]
        xs = [r["key"]["size_ratio"] for r in rows]
        ys = [r["S_mean"] for r in rows]
        es = [r["S_std"] for r in rows]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                    label="shear" if name == "shear" else "vibration")
    ax.set_xlabel("size ratio $d_l/d_s$")
    ax.set_ylabel("segregation degree $S$")
    ax.set_title("(a) segregation vs size ratio")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=8)


def panel_scaling(ax, data: dict) -> None:
    for name, color in (("shear", "C0"), ("vibration", "C1")):
        fits = data["scaling"].get(name, {})
        for key, fit in fits.items():
            if "x" not in fit:
                continue
            x = np.asarray(fit["x"]) + 1.0
            pred = np.asarray(fit["predicted"])
            ax.plot(x, pred, "-", color=color, lw=1.0,
                    label=f"{name} $\\alpha$={fit['alpha']:.2f} ($R^2$={fit['R2']:.2f})")
            ax.plot(x, fit["y"], "o", color=color, ms=4)
    ax.set_xlabel("size ratio $d_l/d_s$")
    ax.set_ylabel("$S$")
    ax.set_title("(b) scaling $S=A(d_l/d_s-1)^\\alpha$")
    ax.legend(fontsize=7)


def panel_phi(ax, data: dict) -> None:
    for name, color in (("shear", "C0"), ("vibration", "C1")):
        pts = data["phi_dependence"].get(name, {}).get("points", [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "o-", color=color, label=name)
    ax.set_xlabel("measured solid fraction $\\phi$")
    ax.set_ylabel("$S$")
    ax.set_title("(c) $S$ vs $\\phi$ (jamming suppression)")
    ax.legend(fontsize=8)


def panel_permeability(ax, data: dict) -> None:
    perm = data.get("permeability") or {}
    rows = perm.get("rows", [])
    if not rows:
        ax.text(0.5, 0.5, "no permeability data", ha="center")
        return
    names = [r["profile"] for r in rows]
    ratios = [r["t50_ratio_to_uniform"] for r in rows]
    ax.bar(range(len(names)), ratios, color="C2")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, fontsize=7)
    ax.set_ylabel("$t_{50}$ / $t_{50}$(uniform)")
    ax.set_title("(d) drainage time vs stratification")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    data = load(args.report)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    panel_profiles(axes[0, 0], data)
    panel_scaling(axes[0, 1], data)
    panel_phi(axes[1, 0], data)
    panel_permeability(axes[1, 1], data)
    fig.suptitle("P155-N09 bidisperse granular segregation verification")
    fig.tight_layout()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
