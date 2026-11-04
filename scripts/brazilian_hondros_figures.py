"""Figures for ``docs/dem3d_brazilian_hondros_refinement.md``.

Reads the scan artifacts (never hard-codes a number) and draws three panels:

``fig1_window_sweep``   -- centre tensile stress against averaging volume, with
                           the +-15% tolerance band; shows the verdict flipping;
``fig2_error_vs_window``-- signed error with its sampling-noise bars;
``fig3_refinement``     -- error against lattice resolution, against the Born
                           anisotropy that refinement cannot change.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "dem3d_brazilian_refinement"
TOL = 0.15
COLORS = {15: "#1f77b4", 20: "#d62728", 25: "#2ca02c", 30: "#9467bd"}


def _load(name: str) -> dict:
    return json.loads((ART / name).read_text(encoding="utf-8"))


def fig_window_sweep(est: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for case in est["cases"]:
        n = case["resolution_across_diameter"]
        pts = [(w["window_ratio"], w["sigma_xx_Pa"]) for w in case["windows"]
               if w.get("sigma_xx_Pa") is not None]
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, "o-", ms=4, lw=1.5, color=COLORS.get(n), label=f"N = {n}")
        split = case["split_stress_Pa"]
        ax.axhline(split, ls=":", lw=1.0, color=COLORS.get(n), alpha=0.7)
        ax.fill_between([min(x), max(x)], split * (1 - TOL), split * (1 + TOL),
                        color=COLORS.get(n), alpha=0.06)
    first = est["cases"][0]
    ax.axhspan(first["split_stress_Pa"] * (1 - TOL), first["split_stress_Pa"] * (1 + TOL),
               color="grey", alpha=0.18, zorder=0,
               label="Hondros $2P/(\\pi D t)$ $\\pm15\\%$ (N=15 scale)")
    ax.set_xlabel("averaging half-width  $w/D$")
    ax.set_ylabel("centre $\\sigma_{xx}$  [Pa]")
    ax.set_title("Centre tension depends on the averaging volume")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    out = ART / "fig1_window_sweep.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


def fig_error_vs_window(est: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhspan(-TOL, TOL, color="green", alpha=0.12, label="$\\pm15\\%$ tolerance")
    ax.axhline(0.0, color="black", lw=0.8, ls=":")
    for case in est["cases"]:
        n = case["resolution_across_diameter"]
        pts = [(w["window_ratio"], 100 * w["tension_rel_error"],
                100 * (w["sem_over_abs_mean"] or 0.0))
               for w in case["windows"] if w.get("tension_rel_error") is not None]
        if not pts:
            continue
        x, y, e = zip(*pts)
        ax.errorbar(x, y, yerr=e, marker="o", ms=4, lw=1.5, capsize=2.5,
                    color=COLORS.get(n), label=f"N = {n}")
    ax.set_xlabel("averaging half-width  $w/D$")
    ax.set_ylabel("centre tension error  [%]")
    ax.set_title("Same physics, opposite verdicts: the tolerance is straddled")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out = ART / "fig2_error_vs_window.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


def fig_refinement(ref: dict) -> Path:
    rows = ref["rows"]
    n = [r["N"] for r in rows]
    err = [100 * r["centre_tension_rel_error"] for r in rows]
    zener = [r["zener_anisotropy"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhspan(-TOL, TOL, color="green", alpha=0.12, label="$\\pm15\\%$ tolerance")
    ax.axhline(0.0, color="black", lw=0.8, ls=":")
    ax.plot(n, err, "o-", color="#1f77b4", ms=6, lw=1.6, label="centre tension error")
    for x, y in zip(n, err):
        ax.annotate(f"{y:+.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("grains across diameter  $N$  (grain radius $= 0.15/N$ m)")
    ax.set_ylabel("centre tension error  [%]")
    ax.set_title("Refinement of the lattice does not resolve the error")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(n, zener, "s--", color="#d62728", ms=5, lw=1.2)
    ax2.axhline(1.0, color="#d62728", lw=0.8, ls=":", alpha=0.7)
    ax2.set_ylabel("Zener anisotropy $A$ of the bond lattice", color="#d62728")
    ax2.set_ylim(0.0, 3.0)
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.annotate("$A = 1$ (isotropic)", (n[0], 1.0), textcoords="offset points",
                 xytext=(4, -12), fontsize=7, color="#d62728")
    ax.legend(fontsize=8, loc="lower left")
    out = ART / "fig3_refinement.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


def main() -> None:
    est = _load("brazilian_estimator.json")
    ref = _load("brazilian_refinement.json")
    for path in (fig_window_sweep(est), fig_error_vs_window(est), fig_refinement(ref)):
        print(f"{path}  {path.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()