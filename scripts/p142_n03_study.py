"""P142W3-N03 production run: fluidisation onset, three-force ledger, convergence.

    python scripts/p142_n03_study.py --out-dir artifacts/P142W3-N03

Writes ``results.json`` plus three figures:

``fig1_pressure_drop_vs_velocity.png``   dp/dz vs U, closure vs Ergun, U_mf markers
``fig2_settling_vs_porosity.png``        u_p(eps) vs the Richardson-Zaki law
``fig3_force_ledger.png``                three-force ledger time history + residual

Everything unscripted (the numbers in the report) comes from ``results.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem.bedcoupling.study_p142 import run_acceptance_study, write_results  # noqa: E402


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_pressure_drop(res: dict, path: Path) -> Path:
    plt = _plt()
    sec = res["sections"]["ergun"]
    rows = sec["rows"]
    re = np.array([r["re_superficial"] for r in rows])
    u = np.array([r["u_superficial"] for r in rows])
    dp_ergun = np.array([r["ergun_dpdz"] for r in rows])
    dp_visc = np.array([r["ergun_viscous"] for r in rows])
    dp_inert = np.array([r["ergun_inertial"] for r in rows])
    dp_meas = np.array([r["measured_local_voidage_dpdz"] for r in rows])
    dp_wy = np.array([r["wen_yu_branch_dpdz"] for r in rows])
    err = np.array([r["rel_error_local"] for r in rows])
    onset = res["sections"]["fluidization"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    ax = axes[0]
    ax.loglog(u, dp_ergun, "k-", lw=2.2, label="Ergun (1952), uniform $\\varepsilon$")
    ax.loglog(u, dp_meas, "o", ms=7, mfc="none", mew=1.6, color="tab:red", label="closure, local $\\varepsilon$")
    ax.loglog(u, dp_visc, "--", lw=1.2, color="tab:blue", label="Ergun viscous part")
    ax.loglog(u, dp_inert, ":", lw=1.4, color="tab:green", label="Ergun inertial part")
    ax.loglog(u, dp_wy, "-.", lw=1.2, color="tab:gray", label="Wen-Yu branch only (N3)")
    weight_grad = onset["ergun"]["weight_gradient"]
    ax.axhline(weight_grad, color="tab:purple", lw=1.0, ls="--")
    ax.annotate("bed weight gradient $W/(AL)$", (u[0], weight_grad), fontsize=8,
                xytext=(3, 4), textcoords="offset points", color="tab:purple")
    for label, key, color in (
        ("$U_{mf}$ closure", "simulated_local", "tab:red"),
        ("$U_{mf}$ Wen-Yu", "wen_yu", "tab:orange"),
    ):
        u_mf = onset[key]["u_mf"]
        ax.axvline(u_mf, color=color, lw=1.2, ls=(0, (4, 3)))
        ax.annotate(f"{label}\n{u_mf * 1e3:.3f} mm/s", (u_mf, dp_ergun[0] * 2), rotation=90,
                    fontsize=8, color=color, va="bottom", ha="right")
    ax.set_xlabel("superficial velocity $U$ [m/s]")
    ax.set_ylabel("$-dp/dz$ [Pa/m]")
    ax.set_title(f"fixed bed, $\\varepsilon$ = {sec['porosity']:.3f}, $N$ = {sec['n_particles']}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[1]
    ax.semilogx(re, 100.0 * err, "o-", color="tab:red", label="closure vs Ergun")
    ax.axhspan(-10.0, 10.0, color="tab:green", alpha=0.12, label="±10 % acceptance band")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("$Re_s = \\rho_f U d/\\mu$")
    ax.set_ylabel("relative error [%]")
    ax.set_title(f"max |error| = {100 * sec['worst_abs_error_local_voidage']:.2f} %")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_settling(res: dict, path: Path) -> Path:
    plt = _plt()
    sec = res["sections"]["rz_curve"]
    rows = sec["rows"]
    mixture = [r for r in rows if r["buoyancy_model"] == "mixture"]
    hydro = [r for r in rows if r["buoyancy_model"] == "hydrostatic"]
    eps_m = np.array([r["porosity"] for r in mixture])
    u_m = np.array([r["u_particle_simulated"] for r in mixture])
    eps_h = np.array([r["porosity"] for r in hydro])
    u_h = np.array([r["u_particle_simulated"] for r in hydro])
    u_t = sec["u_terminal"]
    n_rz = sec["n_rz"]
    eps_line = np.linspace(0.78, 0.97, 200)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    ax = axes[0]
    ax.semilogy(eps_line, u_t * eps_line**n_rz, "k-", lw=2.0,
                label=f"Richardson-Zaki $u_t\\varepsilon^n$, $n$ = {n_rz:.3f}")
    ax.semilogy(eps_m, u_m, "o", ms=7, color="tab:red", label="DEM settling, mixture buoyancy")
    ax.semilogy(eps_h, u_h, "s", ms=6, mfc="none", color="tab:blue",
                label="DEM settling, hydrostatic buoyancy (N1)")
    ax.set_xlabel("porosity $\\varepsilon$")
    ax.set_ylabel("settling speed $u_p$ [m/s]")
    ax.set_title(f"$u_t$ = {u_t * 1e3:.3f} mm/s, $Re_t$ = {sec['re_terminal']:.3f}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    width = 0.35
    x = np.arange(len(eps_m))
    ax.bar(x - width / 2, 100 * np.array([r["rel_error"] for r in mixture]), width,
           color="tab:red", label="mixture buoyancy")
    ax.bar(x + width / 2, 100 * np.array([r["rel_error"] for r in hydro]), width,
           color="tab:blue", alpha=0.7, label="hydrostatic buoyancy (N1)")
    ax.axhspan(-10.0, 10.0, color="tab:green", alpha=0.12, label="±10 % acceptance band")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e:.2f}" for e in eps_m])
    ax.set_xlabel("porosity $\\varepsilon$")
    ax.set_ylabel("error vs RZ [%]")
    ax.set_title("mixture passes, hydrostatic does not: the test discriminates")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_ledger(res: dict, path: Path) -> Path:
    plt = _plt()
    sec = res["sections"]["ledger"]
    series = sec["series"]
    t = np.array(series["time_over_tau"])
    w = series["weight_total"]
    f_drag = np.array(series["f_drag"]) / w
    f_pres = np.array(series["f_pressure"]) / w
    f_grav = np.array(series["f_gravity"]) / w
    f_tot = np.array(series["f_total"]) / w
    mom_rate = np.array(series["momentum_rate"]) / w
    vz = np.array(series["mean_vz"])

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    ax = axes[0]
    ax.plot(t, f_grav, color="tab:green", lw=1.4, label="$F_{gravity}$")
    ax.plot(t, f_pres, color="tab:blue", lw=1.4, label="$F_{pressure}$")
    ax.plot(t, f_drag, color="tab:red", lw=1.4, label="$F_{drag}$")
    ax.plot(t, f_tot, "k-", lw=2.0, label="$\\sum F$ (ledger)")
    ax.plot(t, mom_rate, "k.", ms=3, alpha=0.5, label="$d(mv)/dt$")
    ax.axhline(0.0, color="k", lw=0.6)
    steady_start = 0.75 * sec["case"]["n_steps"] * sec["case"]["dt_over_tau"]
    ax.axvspan(steady_start, t[-1], color="tab:orange", alpha=0.10, label="steady window")
    ax.set_xlabel("time $t/\\tau$")
    ax.set_ylabel("force / bed weight $W$  [-]")
    ax.set_title(f"three-force ledger, $N$ = {sec['case']['n_particles']}, "
                 f"$\\varepsilon$ = {sec['case']['porosity']:.2f}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    residual = np.abs(f_tot - mom_rate)
    ax.semilogy(t, np.maximum(residual, 1e-18), color="tab:purple", lw=1.0,
                label="$|\\sum F - d(mv)/dt| / W$ per step")
    ax.axhline(0.01, color="tab:red", ls="--", lw=1.2, label="1 % acceptance")
    ax.axhline(sec["negative_controls"]["dropped_drag_bookkeeping"], color="tab:gray", ls=":",
               lw=1.2, label=f"N2 dropped-drag residual = "
                            f"{sec['negative_controls']['dropped_drag_bookkeeping']:.1f}")
    ax.set_xlabel("time $t/\\tau$")
    ax.set_ylabel("residual / bed weight [-]")
    ax.set_ylim(1e-18, 1e3)
    ax.set_title(f"bookkeeping {sec['bookkeeping_residual']:.1e}, "
                 f"equilibrium {sec['equilibrium_residual']:.1e}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="center right")

    inset = ax.inset_axes([0.12, 0.12, 0.30, 0.30])
    inset.plot(t, 1e3 * vz, color="tab:blue", lw=0.9)
    inset.set_xlabel("$t/\\tau$", fontsize=7)
    inset.set_ylabel("$\\langle v_z\\rangle$ [mm/s]", fontsize=7)
    inset.tick_params(labelsize=7)
    inset.set_title("particle velocity", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def stamp_artifacts(res: dict, out: Path) -> dict:
    """Attach the artifact paths to the criterion numbers so results.json is self-describing."""
    res["artifacts"] = {
        "results_json": str(out / "results.json"),
        "figures": {
            "C1_ergun": str(out / "fig1_pressure_drop_vs_velocity.png"),
            "C4_fluidization": str(out / "fig1_pressure_drop_vs_velocity.png"),
            "C2a_stokes_terminal": str(out / "fig2_settling_vs_porosity.png"),
            "C2b_transition_terminal": str(out / "fig2_settling_vs_porosity.png"),
            "C2c_rz_curve": str(out / "fig2_settling_vs_porosity.png"),
            "C3_ledger": str(out / "fig3_force_ledger.png"),
            "C5_convergence": str(out / "results.json"),
        },
        "report_doc": "docs/P142W3-N03_fluidised_bed_coupling.md",
        "prereg": "prereg/P142W3-N03_fluidization_prereg.md",
        "reproduce": "python scripts/p142_n03_study.py --out-dir " + str(out),
        "tests": [
            "python -m pytest tests/test_bedcoupling_fluidization.py -q",
            "python -m pytest tests/test_bedcoupling_p142_acceptance.py -q",
        ],
    }
    res["artifacts"]["criteria_source"] = "sections/*/criterion in this file; no hand-copied numbers"
    return res


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="artifacts/P142W3-N03")
    parser.add_argument("--quick", action="store_true", help="reduced steps, for CI smoke")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    steps = dict(terminal_steps=800, rz_steps=400, ledger_steps=400) if args.quick else {}
    res = run_acceptance_study(**steps)
    stamp_artifacts(res, out)
    write_results(res, out / "results.json")
    fig_pressure_drop(res, out / "fig1_pressure_drop_vs_velocity.png")
    fig_settling(res, out / "fig2_settling_vs_porosity.png")
    fig_ledger(res, out / "fig3_force_ledger.png")
    summary = {k: {"passed": v["passed"], "value": v["value"], "threshold": v["threshold"]}
               for k, v in res["criteria"].items()}
    summary["all_criteria_passed"] = res["all_criteria_passed"]
    print(json.dumps(summary, indent=2))
    return 0 if res["all_criteria_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())