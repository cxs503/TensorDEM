"""Figures + markdown table generator for IN32-D01 (drum flow regime map).

Reads ``artifacts/drum_flow/phase_map.json`` and writes PNG figures plus a
markdown snippet with every number injected programmatically (no hand copying).

    python benchmarks/drum_flow_figures.py --map artifacts/drum_flow/phase_map.json \
        --outdir artifacts/drum_flow
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                       # noqa: E402

REGIME_COLORS = {
    "slipping": "#4c72b0",
    "rolling": "#55a868",
    "cataracting": "#dd8452",
    "centrifuging": "#c44e52",
}
REGIME_ORDER = ["slipping", "rolling", "cataracting", "centrifuging"]


def _grid(cells, key):
    fills = sorted({c["fill"] for c in cells})
    frs = sorted({c["fr"] for c in cells})
    z = np.full((len(fills), len(frs)), np.nan)
    for c in cells:
        i = fills.index(c["fill"])
        j = frs.index(c["fr"])
        val = c[key]
        z[i, j] = REGIME_ORDER.index(val) if isinstance(val, str) else val
    return np.array(fills), np.array(frs), z


def fig_phase_map(cells, outdir: Path):
    fills, frs, z = _grid(cells, "regime_kinematic")
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    cmap = matplotlib.colors.ListedColormap([REGIME_COLORS[r] for r in REGIME_ORDER])
    ax.imshow(z, origin="lower", aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5,
              extent=[frs.min(), frs.max(), fills.min(), fills.max()])
    for fr_edge in (0.3, 0.7, 1.0):
        ax.axvline(fr_edge, color="k", lw=1.0, ls="--")
    handles = [matplotlib.patches.Patch(color=REGIME_COLORS[r], label=r) for r in REGIME_ORDER]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9,
              title="measured regime")
    ax.set_xlabel("Froude number  Fr = $\\omega^2R/g$")
    ax.set_ylabel("fill fraction $\\varphi$")
    ax.set_title("Rotating-drum phase map from DEM kinematics\n(dashed: prescribed Fr band edges 0.3 / 0.7 / 1.0)")
    fig.tight_layout()
    path = outdir / "phase_map.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_agreement(cells, outdir: Path):
    """Fr-band label vs measured label, one marker per cell."""
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    seen = set()
    for c in cells:
        agree = c["regime_froude"] == c["regime_kinematic"]
        col = "#55a868" if agree else "#c44e52"
        label = "agree" if agree else "disagree"
        ax.scatter(c["fr"], c["observables"]["airborne_fraction"], s=28, color=col,
                   marker="o" if agree else "x", alpha=0.85,
                   label=label if label not in seen else None)
        seen.add(label)
    ax.axhline(0.02, color="k", ls=":", lw=1.0)
    ax.text(0.02, 0.024, "cataracting threshold 2 %", fontsize=7)
    ax.set_xlabel("Froude number")
    ax.set_ylabel("airborne fraction (no contact)")
    ax.set_title("Measured flight fraction vs Fr: agreement of the Fr-band classifier")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "fr_agreement.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_ledger(cells, outdir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    fills = sorted({c["fill"] for c in cells})
    cmap = plt.get_cmap("viridis")
    for k, f in enumerate(fills):
        sub = [c for c in cells if c["fill"] == f]
        sub.sort(key=lambda c: c["fr"])
        fr = [c["fr"] for c in sub]
        cl = [100.0 * c["ledger"]["closure_rel"] for c in sub]
        ax1.plot(fr, cl, "o-", ms=3.5, lw=1.0, color=cmap(k / max(len(fills) - 1, 1)),
                 label=f"$\\varphi$={f}")
    ax1.axhline(2.0, color="r", ls="--", lw=1.0)
    ax1.text(0.12, 2.15, "acceptance 2 %", color="r", fontsize=7)
    ax1.set_xlabel("Fr"); ax1.set_ylabel(r"$|P_{in}-\Phi_{diss}|/P_{in}$  [%]")
    ax1.set_title("Power-ledger closure")
    ax1.legend(fontsize=7)
    for k, f in enumerate(fills):
        sub = sorted([c for c in cells if c["fill"] == f], key=lambda c: c["fr"])
        fr = [c["fr"] for c in sub]
        tq = [c["ledger"]["tau_drum_nm"] for c in sub]
        ax2.plot(fr, tq, "s-", ms=3.0, lw=1.0, color=cmap(k / max(len(fills) - 1, 1)),
                 label=f"$\\varphi$={f}")
    ax2.set_xlabel("Fr"); ax2.set_ylabel(r"mean shell torque $\tau$  [N m]")
    ax2.set_title("Shell torque (drive)")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    path = outdir / "power_ledger.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_free_surface(cells, outdir: Path):
    """Rolling-regime free surface: fitted parabola vs the centrifugal analytic one."""
    rolling = [c for c in cells if c.get("surface", {}).get("fit")
               and c["regime_kinematic"] == "rolling"]
    rolling.sort(key=lambda c: c["fr"])
    if not rolling:
        rolling = [c for c in cells if c.get("surface", {}).get("fit")]
    pick = rolling[:4]
    n = len(pick)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, c in zip(axes, pick):
        fit = c["surface"]["fit"]
        x = np.linspace(*fit["x_range"], 60)
        y = fit["a"] + fit["b"] * x + fit["c"] * x ** 2
        c_an = c["surface"]["analytic"]["c_analytic"]
        y_an = fit["a"] + fit["b"] * x + c_an * x ** 2
        ax.plot(x, y, "C0-", lw=1.6, label=f"fit  $c$={fit['c']:.2f} 1/m")
        ax.plot(x, y_an, "C3--", lw=1.2, label=f"analytic $c$={c_an:.2f} 1/m")
        r2_all = c["surface"].get("r2_all_particles", fit["r2"])
        ax.set_title(f"Fr={c['fr']:.2f}, $\\varphi$={c['fill']:.2f}\n"
                     f"$R^2$(all)={r2_all:.4f}", fontsize=8)
        ax.set_xlabel("x [m]"); ax.set_ylabel("free surface y [m]")
        ax.legend(fontsize=6)
    fig.suptitle("Free-surface parabola: DEM fit vs centrifugal analytic curvature", fontsize=9)
    fig.tight_layout()
    path = outdir / "free_surface.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def markdown_table(cells, summary) -> str:
    lines = ["| Fr | regime (Fr band) | regime (measured) | wall ratio | flight frac | r_p05/R | "
             "$P_{in}$ [W] | $\\Phi_{diss}$ [W] | closure | $R^2$ |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(cells, key=lambda c: (c["fr"], c["fill"])):
        fit = c.get("surface", {}).get("fit", {})
        lines.append("| {fr:.2f} | {rf} | {rk} | {w:.2f} | {ab:.3f} | {p05:.2f} | "
                     "{pin:.3f} | {phi:.3f} | {cl:.3f} | {r2} |".format(
                         fr=c["fr"], rf=c["regime_froude"], rk=c["regime_kinematic"],
                         w=c["observables"]["wall_velocity_ratio"],
                         ab=c["observables"]["airborne_fraction"],
                         p05=c["observables"]["r_p05_over_R"],
                         pin=c["ledger"]["p_in_w"], phi=c["ledger"]["phi_diss_w"],
                         cl=c["ledger"]["closure_rel"],
                         r2=("%.4f" % c["surface"].get("r2_all_particles", fit["r2"]))
                         if fit else "n/a"))
    head = (f"cells={summary['n_cells']}  agreement={summary['fr_x_fill_agreement']:.3f}  "
            f"closure_median={summary['closure_median']:.4f}  "
            f"closure<2%={summary['closure_within_2pct_fraction']:.3f}  "
            f"R2_min={summary['surface_r2_min']:.4f}")
    return head + "\n\n" + "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="artifacts/drum_flow/phase_map.json")
    ap.add_argument("--outdir", default="artifacts/drum_flow")
    args = ap.parse_args(argv)
    payload = json.loads(Path(args.map).read_text())
    cells, summary = payload["cells"], payload["summary"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figs = [fig_phase_map(cells, outdir), fig_agreement(cells, outdir),
            fig_ledger(cells, outdir), fig_free_surface(cells, outdir)]
    (outdir / "tables.md").write_text(markdown_table(cells, summary))
    print(json.dumps({"figures": [str(f) for f in figs], "summary": summary}, indent=1))


if __name__ == "__main__":
    main()