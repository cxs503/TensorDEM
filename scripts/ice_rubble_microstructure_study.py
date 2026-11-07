"""Ice-rubble microstructure study driver (TensorDEM m5074).

Runs the DEM cases of :mod:`tensordem.ice_rubble_microstructure` and writes the
machine-readable artifacts + figures of ``docs/ice_rubble_microstructure.md``.

Usage (from the repository root)::

    # run one or more cases (each writes artifacts/ice_rubble_microstructure/cases/<name>.json)
    PYTHONPATH=src python scripts/ice_rubble_microstructure_study.py --cases mu_1.00,shape_k2.00

    # merge every available case into study.json + figures
    PYTHONPATH=src python scripts/ice_rubble_microstructure_study.py --analyse

The --fetch flag skips --cases and just prints the list of case names.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

import tensordem.ice_rubble_microstructure as rubble

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "ice_rubble_microstructure"
CASES = OUT / "cases"
FIGS = OUT / "figures"


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=1, default=_json_default)


def run_cases(cfg, names: list[str]) -> None:
    for case in rubble.study_cases(cfg):
        if case["name"] not in names:
            continue
        target = CASES / f"{case['name']}.json"
        if target.exists():
            print(f"[skip] {case['name']} (already present)", flush=True)
            continue
        t0 = time.time()
        record = rubble.run_study_case(cfg, case)
        record["wall_seconds"] = time.time() - t0
        record["cfg"] = json.loads(json.dumps(cfg.__dict__, default=_json_default))
        write_json(target, record)
        micro = record["microstructure"]
        print(
            "[done] {name:18s} n={n:.3f} phicell={phi:.3f} Z={z:.2f} NC={nc:d} "
            "F={f:.0f}N steps={st:d} {sec:.0f}s".format(
                name=case["name"],
                n=micro["porosity_voronoi"],
                phi=micro["packing_fraction_cell"],
                z=micro["coordination_number"],
                nc=micro["n_contacts"],
                f=record["drag"].get("mean_force_tail", float("nan")),
                st=record["steps_total"],
                sec=record["wall_seconds"],
            ),
            flush=True,
        )


def load_cases(cfg) -> dict[str, dict]:
    runs = {}
    for case in rubble.study_cases(cfg):
        path = CASES / f"{case['name']}.json"
        if path.exists():
            with path.open() as handle:
                runs[case["name"]] = json.load(handle)
    return runs


def stripped(runs: dict[str, dict], cfg) -> dict[str, dict]:
    """Drop the bulk arrays kept for plotting before storing the merged study."""
    out = {}
    for name, run in runs.items():
        copy = json.loads(json.dumps(run, default=_json_default))
        micro = copy.get("microstructure", {})
        micro.pop("force_chain", None)
        micro.pop("local_porosity_histogram", None)
        copy.pop("diameters", None)
        copy.pop("positions_sample", None)
        micro.pop("local_porosity_sample", None)
        out[name] = copy
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def make_figures(cfg, runs: dict[str, dict], study: dict) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                         "figure.dpi": 100, "savefig.dpi": 92})
    FIGS.mkdir(parents=True, exist_ok=True)
    figures = []

    def save(fig, name: str, caption: str) -> None:
        path = FIGS / name
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        figures.append({"name": name, "caption": caption, "path": str(path.relative_to(ROOT))})

    primary = runs[f"mu_{cfg.primary_friction:.2f}"]
    mu_runs = [runs[n] for n in sorted(runs) if n.startswith("mu_")]
    shape_runs = [runs[n] for n in sorted(runs) if n.startswith("shape_")]

    # 1 -- porosity / coordination friction map
    mus = [r["friction_coefficient"] for r in mu_runs]
    ns = [r["microstructure"]["porosity_voronoi"] for r in mu_runs]
    zs = [r["microstructure"]["coordination_number"] for r in mu_runs]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax[0].axhspan(0.32 - 0.05, 0.32 + 0.05, color="tab:green", alpha=0.18,
                  label="Timco & O'Brien 0.32$\\pm$0.05")
    ax[0].plot(mus, ns, "o-", color="tab:blue", label="DEM (Voronoi)")
    ax[0].set_xlabel("inter-floe friction $\\mu$")
    ax[0].set_ylabel("porosity $n$")
    ax[0].set_title("dem porosity vs $\mu$ ($\\sigma_c$ = %.0f kPa)" % (cfg.compact_stress / 1e3))
    ax[0].legend(loc="upper left")
    ax[1].axhspan(3.5, 4.5, color="tab:orange", alpha=0.18, label="reference $Z=4.0\\pm0.5$")
    ax[1].plot(mus, zs, "s-", color="tab:red", label="$Z$ (all contacts)")
    ax[1].plot(
        mus,
        [r["microstructure"]["coordination_number_all"] for r in mu_runs],
        "^--",
        color="tab:purple",
        label="$Z$ (whole packing)",
    )
    ax[1].set_xlabel("inter-floe friction $\\mu$")
    ax[1].set_ylabel("coordination number $Z$")
    ax[1].legend(loc="upper right", fontsize=6)
    save(fig, "fig1_porosity_coordination_map.png",
         "Porosity and coordination number of the compacted rubble versus inter-floe friction; "
         "bands are the literature references.")

    # 2 -- force-chain PDF
    fc = primary["microstructure"]["force_chain"]
    normalised = np.asarray(fc["normalised"], dtype=float)
    ref = rubble.HOPKINS_TVEITEN_2002
    plateau = rubble.HOPKINS_TVEITEN_2002_PLATEAU
    edges = np.linspace(0, 4, 41)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax[0].hist(normalised, bins=edges, density=True, color="tab:blue", alpha=0.55,
               label="DEM (N=%d contacts)" % normalised.size)
    grid = np.linspace(0.01, 4, 400)
    ax[0].plot(grid, ref.pdf(grid), "k-", lw=1.2, label="exp(-f) reference")
    ax[0].plot(grid, plateau.pdf(grid), "k--", lw=1.0, label="plateau+tail variant")
    ax[0].set_xlabel("$f = F/\\langle F\\rangle$")
    ax[0].set_ylabel("$P(f)$")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=6)
    ax[0].set_title("contact-force PDF (KS p = %.2f at matched size)"
                    % study["criteria"]["b_force_chain_pdf"]["p_value"])
    cum = np.sort(normalised)
    ax[1].semilogy(cum, 1.0 - np.arange(cum.size) / cum.size, "b-", label="DEM survival")
    ax[1].semilogy(grid, ref.survival(grid), "k--", label="exp(-f)")
    ax[1].set_xlabel("$f$")
    ax[1].set_ylabel("$P(>f)$")
    ax[1].legend(fontsize=6)
    ax[1].set_title("survival: strong-force tail")
    save(fig, "fig2_force_chain_pdf.png",
         "Normalised contact-force PDF and survival function of the DEM packing against the "
         "literature force-chain shape (exponential q-model; digitised data withheld).")

    # 3 -- resistance feedback
    pts = study["criteria"]["d_resistance_feedback"]["points"]
    x = np.array([p["solid_fraction"] for p in pts])
    y = np.array([p["resistance"] for p in pts])
    fit = study["criteria"]["d_resistance_feedback"]
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.plot(x, y, "o", color="tab:red", label="DEM platen resistance")
    xx = np.linspace(x.min() * 0.97, x.max() * 1.03, 50)
    ax.plot(xx, fit["slope"] * xx + fit["intercept"], "k-",
            label="$R^2$ = %.3f" % fit["r_squared"])
    ax.set_xlabel("solid fraction $1-n$")
    ax.set_ylabel("platen resistance $F$ [N]")
    ax.set_title("$F \\propto (1-n)$ across the size-distribution family")
    ax.legend(fontsize=7)
    save(fig, "fig3_resistance_feedback.png",
         "Macroscopic platen resistance versus solid fraction for five floe-size distributions at "
         "fixed friction and consolidation stress; the line is the least-squares fit.")

    # 4 -- local porosity cloud (Voronoi cell porosity in a vertical slice)
    pos = np.asarray(primary["microstructure"].get("positions_sample", []), dtype=float)
    loc = np.asarray(primary["microstructure"].get("local_porosity_sample", []), dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    if pos.size and loc.size == pos.shape[0]:
        sc = ax.scatter(pos[:, 0], pos[:, 2], c=loc, s=6, cmap="viridis")
        fig.colorbar(sc, ax=ax, label="local Voronoi porosity $n_i$")
    else:
        ax.text(0.5, 0.5, "local porosity sample unavailable", ha="center")
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$z$ [m]")
    ax.set_title("local porosity cloud (Voronoi cells), primary case")
    save(fig, "fig4_local_porosity_cloud.png",
         "Spatial distribution of the Voronoi-cell porosity in the compacted packing (each marker "
         "is one floe, colour = porosity of its Voronoi cell).")

    # 5 -- coordination spectrum
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for run in mu_runs:
        spectrum = run["microstructure"]["coordination_spectrum"]
        ax.plot([s["force_fraction"] for s in spectrum], [s["Z"] for s in spectrum],
                "o-", ms=3, label="$\\mu$ = %.1f" % run["friction_coefficient"])
    ax.axhspan(3.5, 4.5, color="tab:orange", alpha=0.15)
    ax.set_xlabel("contact-force cutoff $F_{cut}/\\langle F\\rangle$")
    ax.set_ylabel("$Z$")
    ax.set_title("coordination number vs force cutoff")
    ax.legend(fontsize=6)
    save(fig, "fig5_coordination_spectrum.png",
         "Coordination number as a function of the contact-force cutoff (marginal-contact "
         "sensitivity) for the friction family; shaded band is the ice-rubble reference.")

    # 6 -- floe size distribution + validation
    diameters = np.asarray(primary.get("diameters", []), dtype=float)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    dist = rubble.RosinRammlerDistribution(d_min=cfg.d_min, d_max=cfg.d_max,
                                           d_char=cfg.d_char, exponent=cfg.exponent)
    if diameters.size:
        ax[0].hist(diameters, bins=30, density=True, alpha=0.6, color="tab:cyan",
                   label="DEM floes (N=%d)" % diameters.size)
        grid_d = np.linspace(cfg.d_min, cfg.d_max, 300)
        ax[0].plot(grid_d, dist.pdf(grid_d), "k-", label="truncated Rosin-Rammler")
    ax[0].set_xlabel("floe diameter $d$ [m]")
    ax[0].set_ylabel("pdf")
    ax[0].legend(fontsize=6)
    ax[0].set_title("floe size distribution")
    if diameters.size:
        ax[1].plot(np.sort(diameters), np.arange(1, diameters.size + 1) / diameters.size,
                   "b-", label="DEM empirical CDF")
        ax[1].plot(grid_d, dist.cdf(grid_d), "k--", label="Rosin-Rammler CDF")
    ax[1].set_xlabel("$d$ [m]")
    ax[1].set_ylabel("CDF")
    ax[1].legend(fontsize=6)
    ax[1].set_title("K-S D = %.3f" % study["criteria"]["f_floe_field"]["ks_vs_rosin_rammler"])
    save(fig, "fig6_floe_size_distribution.png",
         "Generated floe-size field against the truncated Rosin-Rammler law (histogram, CDF and "
         "Kolmogorov-Smirnov distance).")
    return figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=None, help="comma separated case names")
    parser.add_argument("--analyse", action="store_true")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    cfg = rubble.MicrostructureStudyConfig()
    if args.list:
        for case in rubble.study_cases(cfg):
            print(case["name"])
        return
    if args.fetch:
        print(json.dumps([c["name"] for c in rubble.study_cases(cfg)]))
        return
    if args.cases:
        names = [n.strip() for n in args.cases.split(",") if n.strip()]
        run_cases(cfg, names)
    if args.analyse:
        runs = load_cases(cfg)
        missing = [
            c["name"] for c in rubble.study_cases(cfg) if c["name"] not in runs
        ]
        study = rubble.analyse_runs(cfg, runs)
        study["available_cases"] = sorted(runs)
        study["missing_cases"] = missing
        study["case_wall_seconds"] = {n: runs[n].get("wall_seconds") for n in runs}
        figures = make_figures(cfg, runs, study)
        study["figures"] = [
            {"name": f["name"], "caption": f["caption"], "path": f["path"]} for f in figures
        ]
        write_json(OUT / "study.json", study)
        # the run arrays are kept separately (they are large)
        write_json(OUT / "runs_summary.json", stripped(runs, cfg))
        print(json.dumps(study["summary"], indent=1))
        print("missing:", missing)


if __name__ == "__main__":
    main()