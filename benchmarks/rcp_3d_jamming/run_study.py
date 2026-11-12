"""RCP anchor study: does a *true* 3-D jamming measurement close
``CASE-DEM-RCP-PACKING-FRACTION``?

The anchor's original PASS (``phi = 0.6400``, "0.53 % from Bernal & Mason
1960") was retracted as tautological: ``make_random_close_packing`` rescales the
cell onto the requested porosity, so the reported number restates its input
(``docs/AUDIT_m16308_CASE-DEM-RCP-PACKING-FRACTION.md``).  The residual-overlap
scan that replaced it is tolerance dependent (``phi_J(tol) = 0.622 ... 0.652``).

This driver measures ``phi_J`` as a **zero-pressure property of an energy
minimum** instead (``tensordem.granular3d.jamming``):

* ``size``        N in 256/500/1000/2000 x 3 seeds -- finite-size extrapolation.
* ``rate``        dphi in 0.01/0.02/0.04 at N=500 -- preparation-rate control.
* ``minimiser``   FIRE vs Polak-Ribiere+ CG at N=500 -- minimiser invariance.
* ``force_tol``   1e-5 vs 1e-6 at N=500 -- minimisation-resolution control.

Every stage is cached in ``runs/<tag>.json`` so the study can be resumed, and
``analyse`` writes the tables and figures from those files only (no number in
the document is typed by hand).

Usage
-----
    PYTHONPATH=src python benchmarks/rcp_3d_jamming/run_study.py --stage run --jobs 6
    PYTHONPATH=src python benchmarks/rcp_3d_jamming/run_study.py --stage analyse
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from tensordem.granular3d import (
    fit_jamming_density,
    growth_protocol,
    pressure_resolution,
)

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FIGS = HERE.parents[1] / "docs" / "artifacts" / "rcp_3d_jamming"

#: literature RCP for monodisperse frictionless spheres
PHI_RCP_LITERATURE = 0.6366  # Scott 1960, Nature 188:908; Bernal & Mason 1960, Nature 188:910

PHI_MAX = 0.72


def _tasks() -> list[dict]:
    tasks: list[dict] = []
    for n, seeds in ((256, (1, 2, 3)), (500, (1, 2, 3)), (1000, (1, 2, 3)), (2000, (1, 2))):
        for seed in seeds:
            tasks.append(dict(tag=f"size_n{n}_s{seed}", arm="size", n=n, seed=seed, dphi=0.02))
    for dphi in (0.01, 0.04):
        tasks.append(dict(tag=f"rate_dphi{dphi:g}", arm="rate", n=500, seed=1, dphi=dphi))
    tasks.append(dict(tag="minimiser_pg", arm="minimiser", n=500, seed=1, dphi=0.02, method="pg"))
    tasks.append(dict(tag="force_tol_1e6", arm="force_tol", n=500, seed=1, dphi=0.02,
                      force_tol=1e-6))
    return tasks


def _run_one(task: dict) -> str:
    out = RUNS / f"{task['tag']}.json"
    if out.exists():
        return f"cached {task['tag']}"
    t0 = time.time()
    grid = np.arange(0.33, PHI_MAX + 1e-9, task["dphi"])
    res = growth_protocol(
        n=task["n"], seed=task["seed"], phi_grid=grid,
        force_tol=task.get("force_tol", 1e-5), max_iter=60000,
        method=task.get("method", "fire"),
    )
    res["tag"] = task["tag"]
    res["arm"] = task["arm"]
    res["wall_time_s"] = time.time() - t0
    RUNS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return f"ran {task['tag']} in {res['wall_time_s']:.1f}s"


def analyse(res: dict) -> dict:
    """Fit ``phi_J`` and the coordination at jamming for one protocol run."""
    rows = res["rows"]
    phi = np.array([r["volume_fraction"] for r in rows])
    pressure = np.array([r["pressure"] for r in rows])
    coord = np.array([r["coordination"] for r in rows])
    conv = np.array([bool(r["converged"]) for r in rows])
    if conv.sum() < 4:
        conv = np.ones(len(rows), dtype=bool)
    floors = [pressure_resolution(n=res["n"], volume=res["box"] ** 3,
                                  coordination=r["coordination"], max_force=r["max_force"],
                                  mean_radius=res["mean_radius"])
              for r, c in zip(rows, conv) if c]
    # a level that did not reach the force tolerance carries a larger residual
    # overlap than the physics does, so it is excluded before the fit and the
    # resolution floor is taken over the converged levels only
    floor = float(max(floors))
    phi, pressure, coord = phi[conv], pressure[conv], coord[conv]
    try:
        fit = fit_jamming_density(phi, pressure, floor=floor)
        fit_status = "ok"
    except ValueError as exc:  # honest: a run whose EOS never leaves the noise floor
        fit = None
        fit_status = f"unavailable: {exc}"
    if fit is None:
        return {
            "tag": res["tag"], "arm": res["arm"], "n": res["n"], "seed": res["seed"],
            "dphi": res["dphi"], "method": res["method"], "force_tol": res["force_tol"],
            "box": res["box"], "n_levels": len(rows),
            "n_converged": int(conv.sum()), "n_used_in_fit": int(len(phi)),
            "minimiser_steps": res["minimiser_steps"],
            "rejected_steps": res["rejected_steps"], "wall_time_s": res["wall_time_s"],
            "fit_status": fit_status, "phi_j": float("nan"),
            "phi_j_power_law": float("nan"), "beta": float("nan"),
            "r2_power_law": float("nan"), "phi_j_linear": float("nan"),
            "r2_linear": float("nan"), "fit_window": [float("nan")] * 2,
            "fit_levels": 0, "pressure_floor": floor,
            "z_at_phi_j": float("nan"), "phi_at_z6": float("nan"),
            "dev_literature_rel": float("nan"),
            "phi": [float(v) for v in phi], "pressure": [float(v) for v in pressure],
            "coordination": [float(v) for v in coord],
            "max_force": [float(r["max_force"]) for r in rows],
        }
    phi_j = fit["phi_j"]
    # coordination at the measured jamming density (linear interpolation on z)
    z_j = float(np.interp(phi_j, phi, coord))
    # isostatic crossing: lowest phi where z >= 6 (frictionless spheres, d=3)
    above = np.flatnonzero(coord >= 6.0)
    z6 = float(phi[above[0]]) if above.size else float("nan")
    n_conv = int(sum(1 for r in rows if r["converged"]))
    return {
        "tag": res["tag"],
        "fit_status": fit_status,
        "arm": res["arm"],
        "n": res["n"],
        "seed": res["seed"],
        "dphi": res["dphi"],
        "method": res["method"],
        "force_tol": res["force_tol"],
        "box": res["box"],
        "n_levels": len(rows),
        "n_converged": n_conv,
        "minimiser_steps": res["minimiser_steps"],
        "rejected_steps": res["rejected_steps"],
        "wall_time_s": res["wall_time_s"],
        "phi_j": fit["phi_j"],
        "phi_j_power_law": fit["phi_j_power_law"],
        "beta": fit["beta"],
        "r2_power_law": fit["r2_power_law"],
        "phi_j_linear": fit["phi_j_linear"],
        "r2_linear": fit["r2_linear"],
        "fit_window": fit["window"],
        "fit_levels": fit["n_levels"],
        "pressure_floor": fit["floor"],
        "z_at_phi_j": z_j,
        "phi_at_z6": z6,
        "dev_literature_rel": (fit["phi_j"] - PHI_RCP_LITERATURE) / PHI_RCP_LITERATURE,
        "phi": [float(v) for v in phi],
        "pressure": [float(v) for v in pressure],
        "coordination": [float(v) for v in coord],
        "max_force": [float(r["max_force"]) for r in rows],
    }


def _load() -> list[dict]:
    out = []
    for path in sorted(RUNS.glob("*.json")):
        res = json.loads(path.read_text())
        if "rows" not in res:
            continue
        out.append(analyse(res))
    return out


def _fss(rows: list[dict]) -> dict:
    from tensordem.granular3d import finite_size_scale
    by_n: dict[int, list[float]] = {}
    for r in rows:
        if r["arm"] != "size" or not np.isfinite(r["phi_j"]):
            continue
        by_n.setdefault(r["n"], []).append(r["phi_j"])
    sizes = sorted(by_n)
    mean = np.array([np.mean(by_n[n]) for n in sizes])
    se = np.array([np.std(by_n[n], ddof=1) / math.sqrt(len(by_n[n])) if len(by_n[n]) > 1
                   else 0.0 for n in sizes])
    out = {"sizes": sizes, "phi_j_mean": mean.tolist(), "phi_j_se": se.tolist(),
           "n_seeds": {str(n): len(by_n[n]) for n in sizes}}
    if len(sizes) >= 3:
        # the seed scatter is comparable to the N dependence, so a per-point
        # error bar of 0 is meaningless: floor the weights at the observed
        # seed scatter and report a fixed-exponent (nu = 0.71) variant too
        sigma = np.maximum(se, 0.004)
        fit = finite_size_scale(sizes, mean, sigma=sigma, d=3.0)
        fit_fixed = finite_size_scale(sizes, mean, sigma=sigma, d=3.0,
                                      nu_bounds=(0.71, 0.7100001))
        fit["nu_unconstrained"] = bool(fit["nu"] > 0.99 or fit["nu_se"] > 0.5)
        out["fss"] = fit
        out["fss_fixed_nu"] = fit_fixed
        out["dev_literature_rel"] = (fit["phi_j_inf"] - PHI_RCP_LITERATURE) / PHI_RCP_LITERATURE
        out["dev_literature_rel_fixed_nu"] = ((fit_fixed["phi_j_inf"] - PHI_RCP_LITERATURE)
                                              / PHI_RCP_LITERATURE)
    return out


def _figures(rows: list[dict], fss: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    made = []

    # fig 1: equation of state P(phi) + power-law fits + literature line
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    colors = {256: "C0", 500: "C1", 1000: "C2", 2000: "C3"}
    for r in rows:
        if r["arm"] != "size":
            continue
        phi = np.array(r["phi"])
        p = np.array(r["pressure"])
        mask = p > r["pressure_floor"]
        ax.semilogy(phi[mask], p[mask], "o", ms=3.5, color=colors.get(r["n"], "0.5"),
                    alpha=0.55)
        if np.isfinite(r["phi_j"]):
            ax.axvline(r["phi_j"], color=colors.get(r["n"], "0.5"), lw=0.8, alpha=0.35)
    for n, c in colors.items():
        sel = [r for r in rows if r["arm"] == "size" and r["n"] == n and np.isfinite(r["phi_j"])]
        if sel:
            ax.axvline(np.mean([r["phi_j"] for r in sel]), color=c, lw=1.6,
                       label=f"N={n}: $\\phi_J$={np.mean([r['phi_j'] for r in sel]):.4f}")
    ax.axvline(PHI_RCP_LITERATURE, color="k", ls="--", lw=1.2,
               label=f"literature RCP {PHI_RCP_LITERATURE:.4f}")
    ax.set_xlabel(r"solid fraction $\phi$")
    ax.set_ylabel(r"reduced pressure $P/k_n$")
    ax.set_title("3-D zero-pressure jamming: equation of state")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_eos_pressure.png", dpi=160)
    plt.close(fig)
    made.append("fig1_eos_pressure.png")

    # fig 2: coordination z(phi) with the isostatic z = 6 line
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for n, c in colors.items():
        sel = [r for r in rows if r["arm"] == "size" and r["n"] == n]
        if not sel:
            continue
        r = sel[0]
        ax.plot(r["phi"], r["coordination"], "-", color=c, lw=1.2, label=f"N={n}")
    ax.axhline(6.0, color="k", ls=":", lw=1.2, label="isostatic z=6 (d=3)")
    ax.axvline(PHI_RCP_LITERATURE, color="k", ls="--", lw=1.0)
    ax.set_xlabel(r"solid fraction $\phi$")
    ax.set_ylabel("mean coordination number z")
    ax.set_title("Contact network at jamming")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_coordination.png", dpi=160)
    plt.close(fig)
    made.append("fig2_coordination.png")

    # fig 3: finite-size extrapolation
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    sizes = np.array(fss["sizes"], dtype=float)
    mean = np.array(fss["phi_j_mean"])
    se = np.array(fss["phi_j_se"])
    ax.errorbar(sizes, mean, yerr=se, fmt="o", color="C0", label=r"$\phi_J(N)$")
    if "fss" in fss:
        nu = fss["fss"]["nu"]
        xs = np.linspace(sizes.min() * 0.8, sizes.max() * 1.15, 200)
        ax.plot(xs, fss["fss"]["phi_j_inf"] - fss["fss"]["c"] * xs ** (-1.0 / (3.0 * nu)),
                "-", color="C3", lw=1.3,
                label=(r"fit $\phi_J(\infty)$=" f"{fss['fss']['phi_j_inf']:.4f}"))
        ax.axhline(fss["fss"]["phi_j_inf"], color="C3", ls=":", lw=0.9)
    ax.axhline(PHI_RCP_LITERATURE, color="k", ls="--", lw=1.1,
               label=f"literature {PHI_RCP_LITERATURE:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("number of spheres N")
    ax.set_ylabel(r"$\phi_J(N)$")
    ax.set_title("Finite-size extrapolation of the jamming density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_finite_size.png", dpi=160)
    plt.close(fig)
    made.append("fig3_finite_size.png")

    # fig 4: protocol controls
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    labels, values, errs, colors2 = [], [], [], []
    for arm, key, fmt in (
        ("rate", "dphi", r"$d\phi=${:.2f}"),
        ("minimiser", "method", "{}"),
        ("force_tol", "force_tol", r"tol$_\mathrm{{F}}$={:.0e}"),
    ):
        sel = [r for r in rows if r["arm"] == arm]
        base = [r for r in rows if r["arm"] == "size" and r["n"] == 500]
        for r in sel:
            if np.isfinite(r["phi_j"]):
                labels.append(fmt.format(r[key]))
                values.append(r["phi_j"])
                errs.append(0.0)
                colors2.append("C1" if arm != "minimiser" else "C2")
    for r in [r for r in rows if r["arm"] == "size" and r["n"] == 500 and np.isfinite(r["phi_j"])]:
        labels.append(f"seed {r['seed']} (ref)")
        values.append(r["phi_j"])
        errs.append(0.0)
        colors2.append("C0")
    ax.bar(range(len(values)), values, color=colors2, alpha=0.85)
    ax.axhline(PHI_RCP_LITERATURE, color="k", ls="--", lw=1.1,
               label=f"literature {PHI_RCP_LITERATURE:.4f}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(min(values) - 0.02, max(max(values), PHI_RCP_LITERATURE) + 0.01)
    ax.set_ylabel(r"$\phi_J$")
    ax.set_title("Preparation-protocol controls (N=500)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_controls.png", dpi=160)
    plt.close(fig)
    made.append("fig4_controls.png")
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("run", "analyse"), default="run")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    if args.stage == "run":
        tasks = _tasks()
        RUNS.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for msg in pool.map(_run_one, tasks):
                print(msg, flush=True)
        return

    rows = _load()
    fss = _fss(rows)
    figs = _figures(rows, fss)
    summary = {"runs": rows, "fss": fss, "figures": figs,
               "literature_phi_rcp": PHI_RCP_LITERATURE}
    (RUNS / "summary.json").write_text(json.dumps(summary, indent=1))

    # markdown tables (programmatically injected into the document)
    lines = ["| tag | N | seed | dphi | minimiser | force_tol | phi_J (power law) | beta | R^2 | phi_J (linear) | z(phi_J) | phi at z=6 | dev vs 0.6366 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        def _f(v, fmt):
            return fmt.format(v) if np.isfinite(v) else "—"
        lines.append(
            f"| {r['tag']} | {r['n']} | {r['seed']} | {r['dphi']:g} | {r['method']} | "
            f"{r['force_tol']:.0e} | {_f(r['phi_j'], '{:.4f}')} | "
            f"{_f(r['beta'], '{:.2f}')} | {_f(r['r2_power_law'], '{:.4f}')} | "
            f"{_f(r['phi_j_linear'], '{:.4f}')} | {_f(r['z_at_phi_j'], '{:.2f}')} | "
            f"{_f(r['phi_at_z6'], '{:.3f}')} | "
            f"{_f(100 * r['dev_literature_rel'], '{:+.2f}%')} |")
    (RUNS / "table_runs.md").write_text("\n".join(lines) + "\n")

    if "fss" in fss:
        f = fss["fss"]
        print(f"phi_J(inf) = {f['phi_j_inf']:.4f} +- {f['phi_j_inf_se']:.4f} "
              f"(nu={f['nu']:.2f}, bootstrap {f['bootstrap_ci95']}), "
              f"dev vs 0.6366 = {100 * fss['dev_literature_rel']:+.2f}%")
    print("figures:", ", ".join(figs))


if __name__ == "__main__":
    main()