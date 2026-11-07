"""Render the report figures for the superquadric contact maturity node.

Reads ``artifacts/dem/shape_error_table.json`` (no numbers are retyped) and
writes two PNGs under ``docs/figures/``:

* ``superquadric_contact_maturity.png`` -- clump surface/normal error versus
  sphere count, contact-cost ratio versus batch size, fixed-point convergence.
* ``superquadric_macro_anchors.png`` -- measured repose anchor with the 22 deg
  target, drum/shear cross-checks, and the API-gate verdict map.

Usage: python scripts/render_superquadric_report_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ARTIFACT = "artifacts/dem/shape_error_table.json"
OUT_DIR = "docs/figures"


def _prime(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, linewidth=0.5)


def contact_figure(data, path):
    rows = data["geometry"]["rows"]
    shapes = sorted({row["shape"] for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    for shape in shapes:
        sub = sorted([r for r in rows if r["shape"] == shape],
                     key=lambda r: r["n_clump_spheres"])
        n = [r["n_clump_spheres"] for r in sub]
        axes[0].plot(n, [r["max_surface_distance_rel"] for r in sub], "o-",
                     ms=3, lw=1.0,
                     label=f"{shape} (ar={sub[0]['aspect_ratio']:g}, n={sub[0]['shape_exponent']:g})")
        axes[1].plot(n, [r["mean_normal_deviation_deg"] for r in sub], "o-", ms=3, lw=1.0)
    axes[0].set_xscale("log", base=2)
    _prime(axes[0], "clump vs true shape: surface error", "spheres in clump",
           r"max $|$SDF$|$ / $R_{eq}$")
    axes[0].legend(fontsize=5.5, loc="upper right")
    axes[1].set_xscale("log", base=2)
    _prime(axes[1], "contact-normal deviation (force chains)", "spheres in clump",
           "mean normal deviation [deg]")
    axes[1].axhline(5.0, color="crimson", ls="--", lw=1.0)
    axes[1].text(1.1, 6.0, "5 deg gate", color="crimson", fontsize=6)

    curve = data["single_contact"]["cost"]["batch_size_curve_n8"]
    batches = sorted(int(k) for k in curve)
    axes[2].plot(batches, [curve[str(b)]["ratio"] for b in batches], "s-", ms=4, lw=1.2,
                 color="darkgreen")
    axes[2].axhline(3.0, color="crimson", ls="--", lw=1.0)
    axes[2].text(batches[0], 3.1, "3x criterion", color="crimson", fontsize=6)
    axes[2].set_xscale("log", base=2)
    _prime(axes[2], "cost: true shape vs 8-sphere clump", "pair batch",
           "cost ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def macro_figure(data, path):
    macro = data["macro"]
    entries = data["disable_ratio_table"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))

    anchor = macro["anchor"]
    mus = [row["mu"] for row in anchor]
    thetas = [row["theta_r_deg"] for row in anchor]
    axes[0].plot(mus, thetas, "o-", color="navy", ms=5, lw=1.2, label="3D column collapse")
    axes[0].axhline(22.0, color="crimson", ls="--", lw=1.0)
    axes[0].fill_between([min(mus) - 0.05, max(mus) + 0.05], 20.0, 24.0,
                         color="crimson", alpha=0.08)
    if macro.get("drum"):
        mu_drum = [row["mu"] for row in macro["drum"]]
        th_drum = [row["theta_deg"] for row in macro["drum"]]
        axes[0].plot(mu_drum, th_drum, "^", color="darkorange", ms=5,
                     label="2D drum (cross-check)")
    axes[0].text(min(mus), 22.6, "22 +/- 2 deg target", color="crimson", fontsize=6)
    _prime(axes[0], "measured repose anchor", "particle friction mu",
           "repose angle [deg]")
    axes[0].legend(fontsize=5.5, loc="lower right")

    shear = macro["shear_box"]
    if shear:
        mu_in = [row["mu_input"] for row in shear]
        mu_i = [row["mu_apparent"] for row in shear]
        lim = [0.0, max(mu_in) * 1.15]
        axes[1].plot(lim, lim, "k--", lw=0.9, label="mu_i = mu")
        axes[1].plot(mu_in, mu_i, "o", color="seagreen", ms=5, label="2D shear cell")
        axes[1].set_xlim(lim)
        axes[1].set_ylim(lim)
    _prime(axes[1], "internal vs external friction", "input mu",
           "mobilised mu_i = tau/sigma")
    axes[1].legend(fontsize=5.5, loc="upper left")

    ars = sorted({e["aspect_ratio"] for e in entries})
    exps = sorted({e["shape_exponent"] for e in entries})
    counts = sorted({e["n_clump_spheres"] for e in entries})
    nmax = counts[-1]
    grid = np.full((len(exps), len(ars)), np.nan)
    for entry in entries:
        if entry["n_clump_spheres"] != nmax:
            continue
        i = exps.index(entry["shape_exponent"])
        j = ars.index(entry["aspect_ratio"])
        grid[i, j] = entry["normal_deviation_deg"] if not entry["use_clump"] else 0.5
    im = axes[2].imshow(grid, origin="lower", aspect="auto", cmap="RdYlGn_r",
                        vmin=0.0, vmax=60.0)
    axes[2].set_xticks(range(len(ars)))
    axes[2].set_xticklabels([f"{a:g}" for a in ars], fontsize=7)
    axes[2].set_yticks(range(len(exps)))
    axes[2].set_yticklabels([f"{e:g}" for e in exps], fontsize=7)
    _prime(axes[2], f"gate verdict, {nmax} spheres", "aspect ratio",
           "shape exponent n1=n2")
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("max normal dev. [deg]", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    with open(ARTIFACT) as fh:
        data = json.load(fh)
    os.makedirs(OUT_DIR, exist_ok=True)
    contact_figure(data, os.path.join(OUT_DIR, "superquadric_contact_maturity.png"))
    macro_figure(data, os.path.join(OUT_DIR, "superquadric_macro_anchors.png"))
    for name in ("superquadric_contact_maturity.png", "superquadric_macro_anchors.png"):
        p = os.path.join(OUT_DIR, name)
        print(p, os.path.getsize(p) // 1024, "kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())