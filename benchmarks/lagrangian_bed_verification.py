"""Reproduce the IN145-N05 acceptance study (Lagrangian bed-fluid coupling).

Usage::

    python benchmarks/lagrangian_bed_verification.py            # full study (~5 min)
    python benchmarks/lagrangian_bed_verification.py --quick    # reduced sweep

Outputs (written under ``artifacts/IN145-N05/``):
``results.json``, ``summary.md`` and ``fig1..fig5*.png``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem.lagrangian_bed import study as st  # noqa: E402


def make_figures(results: dict, outdir: Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    figs = []
    sections = results["sections"]

    # ---- fig 1: packing ---------------------------------------------------
    packing = sections.get("packing")
    if packing:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
        stats = packing["local_statistics"]
        key = sorted(stats)[0]
        eps_i = np.asarray(stats[key]["mean"]) + 0.0
        # rebuild the local-porosity distribution from stored statistics
        ax = axes[0]
        ax.axvspan(0.36, 0.40, color="tab:green", alpha=0.12, label="target 0.36-0.40")
        ax.axvline(packing["porosity"]["porosity"], color="k", ls="--", lw=1.2,
                   label=f"bed porosity {packing['porosity']['porosity']:.4f}")
        ax.set_xlabel("porosity [-]")
        ax.set_ylabel("count")
        ax.set_title(f"local porosity (probe {key})  std={stats[key]['std']:.4f}")
        ax.legend(fontsize=7)
        ax = axes[1]
        hist = results["_figure_payload"].get("local_porosity_hist")
        if hist:
            ax.hist(np.asarray(hist["values"]), bins=30, color="tab:blue", alpha=0.75)
            ax.axvline(hist["mean"], color="k", ls="--", lw=1.0)
        ax.set_xlabel("local porosity [-]")
        ax.set_ylabel("count")
        ax.set_title("distribution over bed interior")
        fig.suptitle("IN145-N05 packing: gravity-deposited polydisperse bed", fontsize=10)
        fig.tight_layout()
        path = outdir / "fig1_packing.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figs.append({"name": "fig1_packing.png", "path": path,
                     "caption": "Packed-bed porosity: bed average vs target band and the "
                                "local (probe) porosity distribution inside the bed."})

    # ---- fig 2: Ergun -----------------------------------------------------
    ergun = sections.get("ergun")
    if ergun:
        rows = ergun["rows"]
        re = np.array([r["re_p"] for r in rows])
        meas = np.array([r["dp_over_L_measured"] for r in rows])
        ref = np.array([r["dp_over_L_ergun"] for r in rows])
        order = np.argsort(re)
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.loglog(re[order], meas[order], "o-", label="Lagrangian DEM (measured)")
        ax.loglog(re[order], ref[order], "s--", label=r"Ergun at $\bar\epsilon$")
        ax.fill_between(re[order], ref[order] * 0.85, ref[order] * 1.15, color="tab:orange",
                        alpha=0.15, label=r"$\pm 15\%$")
        ax.set_xlabel(r"$Re_p = \rho_f U d/\mu$")
        ax.set_ylabel(r"$\Delta p / L$  [Pa/m] (hydrostatic removed)")
        worst = max(abs(r["rel_error"]) for r in rows)
        ax.set_title(f"Ergun pressure drop, worst |err| = {100*worst:.1f}%")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = outdir / "fig2_ergun.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figs.append({"name": "fig2_ergun.png", "path": path,
                     "caption": "Pressure drop per unit bed length vs particle Reynolds number: "
                                "Lagrangian point-particle closure on the deposited bed vs the "
                                "Ergun correlation evaluated at the mean voidage."})

    # ---- fig 3: RZ settling curves ---------------------------------------
    rz = sections.get("rz_settling")
    if rz:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
        ax = axes[0]
        colours = plt.cm.viridis(np.linspace(0.0, 0.85, len(rz["rz_hindrance"]["sweep"])))
        for c, s in zip(colours, rz["rz_hindrance"]["sweep"]):
            eps = np.array([p["eps_measured"] for p in s["points"]])
            u = np.array([p["u_over_ut"] for p in s["points"]])
            ax.semilogy(eps, u, "o-", color=c, label=f"Re$_t$={s['re_terminal']:.1f}")
            grid = np.linspace(eps.min(), eps.max(), 100)
            ax.semilogy(grid, grid ** (s["n_correlation"] - 1.0), ":", color=c, lw=1.0)
        ax.set_xlabel(r"porosity $\epsilon$")
        ax.set_ylabel(r"$u_s/u_t$")
        ax.set_title("Richardson-Zaki hindered settling (points: DEM, dotted: correlation)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
        ax = axes[1]
        sz = rz["rz_hindrance"]["sweep"]
        re = [s["re_terminal"] for s in sz]
        ax.semilogx(re, [s["n_correlation"] for s in sz], "k--", label="RZ correlation")
        ax.semilogx(re, [s["n_fit"] for s in sz], "o-", color="tab:blue", label="DEM, RZ hindrance")
        cls = rz["classical_gidaspow"]["sweep"]
        ax.semilogx([s["re_terminal"] for s in cls], [s["n_fit"] for s in cls], "s-",
                    color="tab:red", label="DEM, classical Gidaspow")
        ax.set_xlabel(r"$Re_t$")
        ax.set_ylabel("voidage exponent $n$")
        ax.set_title("settling exponent vs Reynolds number")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = outdir / "fig3_rz.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figs.append({"name": "fig3_rz.png", "path": path,
                     "caption": "Group settling: normalised slip velocity vs voidage for four "
                                "terminal Reynolds numbers (left) and the fitted Richardson-Zaki "
                                "exponent vs the correlation (right), for the RZ-consistent and "
                                "the classical Gidaspow closure."})

    # ---- fig 4: static + ledger ------------------------------------------
    static = sections.get("static_and_ledger")
    if static:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        labels = ["grain side\n(divided volume)", "fluid side\n(cell integral)"]
        values = [static["particle_side_drag_per_area"], static["fluid_side_drag_per_area"]]
        ax.bar(labels, values, color=["tab:blue", "tab:orange"], width=0.55)
        for x, v in zip(range(2), values):
            ax.annotate(f"{v:.4g}", (x, v), ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(r"drag force per unit bed area  [Pa]")
        ax.set_title(
            "momentum ledger at U = %.3f m/s\n"
            "ledger residual = %.1e (bound 2 %%), static single-grain residual = %.1e"
            % (ergun["rows"][0]["u_superficial"] if ergun else 0.0,
               static["criterion"]["ledger_residual"],
               static["criterion"]["static_residual_over_weight"]))
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = outdir / "fig4_ledger.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figs.append({"name": "fig4_ledger.png", "path": path,
                     "caption": "Action-reaction momentum ledger: drag summed over the grains "
                                "(divided-volume distribution) against the independent fluid-cell "
                                "integral, with the static-bed residual."})

    # ---- fig 5: time step / GCI ------------------------------------------
    ts = sections.get("timestep")
    if ts:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for hist, label, colour in zip(
            (ts["history_dt"], ts["history_dt_half"], ts["history_dt_quarter"]),
            ("dt", "dt/2", "dt/4"),
            ("tab:blue", "tab:orange", "tab:green"),
        ):
            ax.plot(np.asarray(hist["time"]) * 1e3, np.asarray(hist["dp_drag"]) / 1e3, label=label,
                    color=colour, lw=1.2)
        ax.set_xlabel("t [ms]")
        ax.set_ylabel(r"$\Delta p$ [kPa] (fluid-at-rest reconstruction)")
        ax.set_title(
            "batch settling: dt convergence (rel. change %.2f%%, GCI U = %.2f%%)"
            % (100 * ts["rel_change_dt_halving"], 100 * (ts["u_value"] or float("nan"))))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = outdir / "fig5_timestep.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figs.append({"name": "fig5_timestep.png", "path": path,
                     "caption": "Batch-settling pressure drop for three time-step levels with the "
                                "GCI (Roache) uncertainty certificate."})
    return figs


def summary_markdown(results: dict, figs: list[dict]) -> str:
    lines = [
        "# IN145-N05 - Lagrangian bed-fluid coupling: acceptance summary",
        "",
        f"Module: `{results['module']}`  |  all machine criteria passed: **{results['all_criteria_passed']}**",
        "",
        "| criterion | value | threshold | passed |",
        "|---|---|---|---|",
    ]
    for key, c in results["criteria"].items():
        lines.append(f"| {key} | {c['value']:.3e} | {c['threshold']:.3e} | {c['passed']} |")
    lines += ["", "## sections", ""]
    for name, sec in results["sections"].items():
        lines.append(f"### {name}")
        if isinstance(sec, dict) and "criterion" in sec:
            lines.append("```json")
            lines.append(json.dumps(sec["criterion"], indent=2, default=st.json_default))
            lines.append("```")
        else:
            lines.append("```json")
            lines.append(json.dumps({k: (v.get("criterion") if isinstance(v, dict) else str(v))
                                     for k, v in sec.items()}, indent=2, default=st.json_default))
            lines.append("```")
        lines.append("")
    if figs:
        lines.append("## figures")
        for f in figs:
            lines.append(f"- `{f['name']}` - {f['caption']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="reduced sweep for a fast check")
    parser.add_argument("--outdir", default="artifacts/IN145-N05")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    opts = st.StudyOptions()
    if args.quick:
        opts.n_particles = 150
        opts.deposit_steps = 700
        opts.ergun_transition = (0.02, 0.10, 0.20)
        opts.rz_re_targets = (1.0, 60.0)
        opts.rz_porosities = (0.5, 0.7, 0.9)
        opts.group_particles = 120
        opts.group_steps = 300

    t0 = time.time()
    results = st.run_acceptance_study(opts)
    results["runtime_s"] = time.time() - t0
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # figure payload: local porosity samples (kept out of results.json)
    payload = {}
    packing = results["sections"].get("packing")
    if packing:
        stats = packing["local_statistics"]
        key = f"{opts.criterion_probe_factor:.1f}d"
        payload["local_porosity_hist"] = {
            "values": packing.get("local_porosity_values", []),
            "mean": stats[key]["mean"],
        }
    results["_figure_payload"] = payload
    figs = []
    if not args.no_figures:
        try:
            figs = make_figures(results, outdir)
        except Exception as exc:  # a plotting failure must not lose the results
            results["figure_error"] = repr(exc)
            print("figure generation failed:", repr(exc))
    results.pop("_figure_payload", None)
    results["figures"] = [{"name": f["name"], "caption": f["caption"]} for f in figs]

    (outdir / "results.json").write_text(json.dumps(results, indent=2, default=st.json_default))
    (outdir / "summary.md").write_text(summary_markdown(results, figs))
    print(json.dumps({k: v["passed"] for k, v in results["criteria"].items()}, indent=2))
    print("all criteria passed:", results["all_criteria_passed"], "runtime %.1f s" % results["runtime_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())