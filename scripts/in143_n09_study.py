"""IN143-N09 production run: packed-bed drag closure + Richardson-Zaki scaling.

Writes ``artifacts/IN143-N09/results.json`` and the report figures.

    python scripts/in143_n09_study.py --out-dir artifacts/IN143-N09
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem.bedcoupling.study import run_acceptance_study, write_results  # noqa: E402


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_packing(res: dict, path: Path) -> Path:
    """Local-voidage field map, histogram and probe-radius sensitivity."""
    plt = _plt()
    sec = res["sections"]["packing"]
    eps = np.array(sec["local_voidage"])
    stats = sec["porosity_statistics"]
    packing = sec["packing"]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    ax = axes[0]
    slab = sec["slab"]
    xs = np.array([p["x"] for p in slab]) / 1e-3
    ys = np.array([p["y"] for p in slab]) / 1e-3
    es = np.array([p["eps"] for p in slab])
    sc = ax.scatter(xs, ys, c=es, s=26, cmap="viridis", vmin=es.min(), vmax=es.max())
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("$\\varepsilon_i$ (probe $R=3d$)")
    lim = packing["box_length"] / 1e-3
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x [d]")
    ax.set_ylabel("y [d]")
    ax.set_title("one-diameter slab of the random packing")

    ax = axes[1]
    ax.hist(eps, bins=40, color="#3b6ea5", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(packing["porosity"], color="k", ls="--",
               label=f"volume balance {packing['porosity']:.4f}")
    ax.axvline(stats["mean"], color="crimson", ls="-", label=f"probe mean {stats['mean']:.4f}")
    ax.axvline(stats["mean"] - stats["std"], color="crimson", ls=":", lw=1.0)
    ax.axvline(stats["mean"] + stats["std"], color="crimson", ls=":", lw=1.0,
               label=f"$\\pm\\sigma$ = {stats['std']:.4f}")
    ax.set_xlabel("local voidage $\\varepsilon_i$")
    ax.set_ylabel("count")
    ax.set_title(f"$\\varepsilon$ = {packing['porosity']:.3f}, "
                 f"$N$ = {packing['n_particles']}, max overlap "
                 f"{1000 * packing['max_overlap_over_diameter']:.2f}$\\times10^{{-3}}d$")
    ax.legend(fontsize=8)

    ax = axes[2]
    sens = sec["probe_sensitivity"]
    r = [s["probe_radius_over_d"] for s in sens]
    ax.plot(r, [s["mean"] for s in sens], "o-", label="mean")
    ax.plot(r, [s["std"] for s in sens], "s-", label="std")
    ax.axhline(0.02, color="crimson", ls="--", label="acceptance std bound 0.02")
    ax.axhline(0.38, color="k", ls=":", lw=1.0, label="target 0.38")
    ax.set_xlabel("probe radius $R/d$")
    ax.set_ylabel("voidage statistic")
    ax.set_title("probe-radius sensitivity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_ergun(res: dict, path: Path) -> Path:
    plt = _plt()
    sec = res["sections"]["ergun"]
    rows = sec["rows"]
    u = np.array([r["u_superficial"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    ax = axes[0]
    ax.loglog(u, [r["ergun_dpdz"] for r in rows], "k-", lw=2.0, label="Ergun (1952)")
    ax.loglog(u, [r["measured_local_voidage_dpdz"] for r in rows], "o", ms=6,
              label=r"assembly, probe voidage $\varepsilon_i$")
    ax.loglog(u, [r["measured_mean_voidage_dpdz"] for r in rows], "x", ms=6,
              label=r"assembly, mean voidage $\bar\varepsilon$")
    ax.loglog(u, [r["wen_yu_branch_dpdz"] for r in rows], "s--", ms=5, alpha=0.7,
              label="Wen--Yu branch (negative control)")
    ax.set_xlabel("superficial velocity $U$ [m/s]")
    ax.set_ylabel("$\\Delta p/L$ [Pa/m]")
    ax.set_title("fixed bed: $\\varepsilon$ = %.3f, $d$ = 1 mm" % res["sections"]["packing"]["packing"]["porosity"])
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhspan(-0.15, 0.15, color="tab:green", alpha=0.12, label="$\\pm$15 % acceptance band")
    ax.semilogx(u, [100 * r["rel_error_local"] for r in rows], "o-", label="probe voidage")
    ax.semilogx(u, [100 * r["rel_error_mean"] for r in rows], "x--", label="mean voidage")
    ax.semilogx(u, [100 * r["rel_error_wen_yu_branch"] for r in rows], "s-", alpha=0.7,
                label="Wen--Yu branch")
    for u_mark, lab in ((2e-3, "laminar"), (5e-2, "transition")):
        ax.axvline(u_mark, color="tab:red", ls=":", lw=1.0)
        ax.text(u_mark, 8, lab, rotation=90, fontsize=8, color="tab:red", ha="right")
    ax.set_xlabel("superficial velocity $U$ [m/s]")
    ax.set_ylabel("relative error vs Ergun [%]")
    ax.set_title("closure error: the test discriminates branches")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_rz(res: dict, path: Path) -> Path:
    plt = _plt()
    sec = res["sections"]["rz_settling"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    ax = axes[0]
    for model, style in (("mixture", "o-"), ("hydrostatic", "s--")):
        runs = sec["variants"][model]["runs"]
        eps = np.array([r["porosity"] for r in runs])
        up = np.array([r["u_particle"] for r in runs])
        fit = sec["variants"][model]["fit"]
        ax.loglog(eps, up, style, ms=6, alpha=0.85,
                  label=f"DEM, {model} buoyancy ($n$={fit['exponent']:.3f})")
        grid = np.linspace(eps.min(), eps.max(), 50)
        ax.loglog(grid, fit["prefactor"] * grid ** fit["exponent"], "-", lw=1.0, alpha=0.6)
    eps = np.array([r["porosity"] for r in sec["variants"]["mixture"]["runs"]])
    n_rz = sec["n_rz"]
    ax.loglog(eps, sec["u_terminal"] * eps**n_rz, "k:", lw=1.6,
              label=f"Richardson--Zaki ($n$={n_rz:.3f}, $u_t$ from Schiller--Naumann)")
    ax.set_xlabel("voidage $\\varepsilon$")
    ax.set_ylabel("settling velocity $u_p$ [m/s]")
    ax.set_title("batch settling: $u_p = u_t\\varepsilon^{n}$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[1]
    rows = sec["re_sweep"]
    re = np.array([r["re_terminal"] for r in rows])
    for model, style in (("mixture", "o-"), ("hydrostatic", "s--")):
        ax.semilogx(re, [r[f"dev_{model}"] for r in rows], style, label=f"{model} buoyancy")
    ax.axhspan(-0.3, 0.3, color="tab:green", alpha=0.12, label="$\\pm$0.3 acceptance band")
    ax.axhline(0.0, color="k", lw=0.8)
    for r in rows:
        ax.annotate(r["label"], (r["re_terminal"], r["dev_mixture"]), fontsize=7,
                    textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("terminal Reynolds number $Re_t$")
    ax.set_ylabel("$n_{sim} - n_{RZ}$")
    ax.set_title("where the Gidaspow blend tracks Richardson--Zaki")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_timestep(res: dict, path: Path) -> Path:
    plt = _plt()
    sec = res["sections"]["timestep"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    ax = axes[0]
    tau = sec["dt"] / sec["dt_over_tau"] if sec["dt_over_tau"] else 1.0
    for key, label, style in (("history_dt", "dt", "o-"), ("history_dt_half", "dt/2", "s--")):
        t = np.array(sec[key]["time"])
        dp = np.array(sec[key]["dpdz"])
        ax.plot(t / tau, dp, style, ms=2.5, lw=1.0, label=label)
    ax.axvline(sec["t_star"] / tau, color="tab:red", ls=":", lw=1.0, label="comparison time $2\\tau$")
    ax.set_xlabel("time / drag relaxation time $\\tau$")
    ax.set_ylabel("drag pressure drop $\\Delta p/L$ [Pa/m]")
    ax.set_title("time-step sensitivity of the coupled pressure drop")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = ["steady\n$\\Delta p/L$", "transient\nt = 2$\\tau$"]
    vals = [100 * sec["rel_change"], 100 * sec["transient_rel_change"]]
    bars = ax.bar(labels, vals, color=["#3b6ea5", "#d08c3a"])
    ax.axhline(3.0, color="crimson", ls="--", label="acceptance 3 %")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2e} %", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("|change| on halving $\\Delta t$ [%]")
    ax.set_title("criterion 6: $\\Delta t \\to \\Delta t/2$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="artifacts/IN143-N09")
    parser.add_argument("--n-particles", type=int, default=1500)
    parser.add_argument("--n-steps", type=int, default=2000)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    res = run_acceptance_study(n_particles=args.n_particles, n_steps=args.n_steps)
    write_results(res, out / "results.json")
    figures = [
        fig_packing(res, out / "fig1_packing.png"),
        fig_ergun(res, out / "fig2_ergun.png"),
        fig_rz(res, out / "fig3_richardson_zaki.png"),
        fig_timestep(res, out / "fig4_timestep.png"),
    ]
    print(json.dumps({k: {"value": v["value"], "threshold": v["threshold"], "passed": v["passed"]}
                      for k, v in res["criteria"].items()}, indent=2))
    print("all passed:", res["all_criteria_passed"])
    print("figures:", [str(p) for p in figures])


if __name__ == "__main__":
    main()
