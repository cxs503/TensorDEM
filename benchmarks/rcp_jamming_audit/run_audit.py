"""Audit of CASE-DEM-RCP-PACKING-FRACTION (m16179) — is the pass real?

The merged case claims phi = 0.6400, "within 0.53 % of Bernal & Mason (1960)
0.6366", and "independent of particle count".  This script tests those three
claims independently:

``tautology``
    Re-run ``make_random_close_packing`` at several target porosities and show
    that the returned solid fraction is the *input* (phi = 1 - eps) to machine
    precision, for several N and seeds.  Any gate on the returned phi is then
    satisfied by construction, so the "0.53 % agreement" measures eps = 0.36,
    not the packing.

``scan``
    Impose solid fractions on a grid and measure the residual overlap, which is
    *not* prescribed by the algorithm.  The grid is cached incrementally so the
    scan can be resumed.

``analyse``
    Tolerance-interpolated jamming solid fraction phi_J(tol) from the scan,
    tolerance spread, comparison with RCP 0.6366, and figures.

Usage
-----
::

    PYTHONPATH=src python3 benchmarks/rcp_jamming_audit/run_audit.py --stage tautology
    PYTHONPATH=src python3 benchmarks/rcp_jamming_audit/run_audit.py --stage scan --n 1000
    PYTHONPATH=src python3 benchmarks/rcp_jamming_audit/run_audit.py --stage analyse
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import numpy as np

from tensordem.bedcoupling.jamming import jamming_curve, overlap_scan
from tensordem.bedcoupling.packing import make_random_close_packing

RCP_REF_PHI = 0.6366  # Scott 1960 Nature 188:908; Bernal & Mason 1960 Nature 188:910
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def stage_tautology(n_list=(400, 1000), seeds=(42, 7)) -> dict:
    rows = []
    for n in n_list:
        for seed in seeds:
            for eps in (0.30, 0.36, 0.45):
                bed = make_random_close_packing(
                    n, 1e-3, eps, seed=seed, final_relax_iterations=50, overlap_tolerance=1e-3
                )
                rows.append(
                    {
                        "n_particles": n,
                        "seed": seed,
                        "target_porosity": eps,
                        "target_phi": 1.0 - eps,
                        "achieved_phi": bed.solid_fraction,
                        "abs_deviation": abs(bed.solid_fraction - (1.0 - eps)),
                    }
                )
                print(
                    f"  N={n} seed={seed} eps={eps:.2f} -> phi={bed.solid_fraction:.12f} "
                    f"|phi-(1-eps)|={rows[-1]['abs_deviation']:.2e}",
                    flush=True,
                )
    worst = max(r["abs_deviation"] for r in rows)
    result = {
        "rows": rows,
        "max_abs_deviation": worst,
        "tautological": worst < 1e-9,
        "note": (
            "phi is imposed as 1 - target_porosity; a tolerance gate on the returned "
            "phi cannot fail and does not measure the packing."
        ),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "tautology.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[tautology] max |phi-(1-eps)| = {worst:.2e} tautological={result['tautological']}")
    return result


def stage_scan(n: int, phi_grid, relax: int) -> dict:
    cache = os.path.join(OUT, f"scan_n{n}.json")
    os.makedirs(OUT, exist_ok=True)
    data = {"n_particles": n, "final_relax_iterations": relax, "points": []}
    if os.path.exists(cache):
        with open(cache) as fh:
            data = json.load(fh)
    done = {round(p["solid_fraction"], 6) for p in data["points"]}
    for phi in phi_grid:
        if round(phi, 6) in done:
            continue
        pts = overlap_scan(n, 1e-3, (phi,), seed=42, final_relax_iterations=relax)
        data["points"].extend(p.as_dict() for p in pts)
        with open(cache, "w") as fh:
            json.dump(data, fh, indent=2)
        p = data["points"][-1]
        print(
            f"  N={n} phi={p['solid_fraction']:.4f} max/d={p['max_overlap_over_diameter']:.3e} "
            f"mean/d={p['mean_overlap_over_diameter']:.3e} contacts={p['n_contacts']} "
            f"({p['wall_time_s']:.1f}s)",
            flush=True,
        )
    print(f"[scan] {len(data['points'])} points cached at {cache}")
    return data


def _load_scan(n: int):
    from tensordem.bedcoupling.jamming import ScanPoint

    with open(os.path.join(OUT, f"scan_n{n}.json")) as fh:
        data = json.load(fh)
    return data, [ScanPoint(**p) for p in sorted(data["points"], key=lambda q: q["solid_fraction"])]


def stage_analyse(n_list=(1000,)) -> dict:
    tolerances = (1e-4, 1e-3, 1e-2, 5e-2)
    scans = {}
    curves = {}
    for n in n_list:
        data, pts = _load_scan(n)
        if not pts:
            continue
        scans[str(n)] = data
        curves[str(n)] = jamming_curve(pts, tolerances=tolerances)
        print(f"[analyse] N={n}")
        for tol in tolerances:
            c = curves[str(n)][f"{tol:.0e}"]
            print(
                f"    tol={tol:.0e}: phi_J={c['phi_j']:.4f} status={c['status']} "
                f"bracket={[round(x, 4) for x in c['bracket']]}"
            )
        print(f"    tolerance spread = {curves[str(n)]['spread']:.4f}")

    # the strictest tolerance is the honest "overlap-free" jamming estimate
    primary = {}
    for key, cur in curves.items():
        strict = cur[f"{1e-3:.0e}"]
        primary[key] = {
            "phi_j_strict_tol_1e-3": strict["phi_j"],
            "bracket": strict["bracket"],
            "status": strict["status"],
            "tolerance_spread": cur["spread"],
            "deviation_vs_rcp_0.6366": strict["phi_j"] - RCP_REF_PHI,
        }
    result = {
        "reference": {"phi_rcp": RCP_REF_PHI, "source": "Scott 1960; Bernal & Mason 1960"},
        "scans": {k: {"n_particles": v["n_particles"], "points": v["points"]} for k, v in scans.items()},
        "jamming_curves": curves,
        "primary": primary,
    }
    with open(os.path.join(OUT, "audit.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    _figures(result)
    print(f"[analyse] wrote {os.path.join(OUT, 'audit.json')}")
    return result


def _figures(result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = os.path.join(OUT, "figures")
    os.makedirs(figs, exist_ok=True)

    # 1) residual overlap vs imposed solid fraction, with tolerance bands
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for n, scan in result["scans"].items():
        phi = [p["solid_fraction"] for p in scan["points"]]
        mx = np.array([p["max_overlap_over_diameter"] for p in scan["points"]])
        ax.semilogy(phi, np.maximum(mx, 1e-12), "o-", label=f"max overlap, N={n}")
    for tol, style in ((1e-3, ":"), (1e-2, "--")):
        ax.axhline(tol, color="grey", ls=style, lw=1)
        ax.text(0.6005, tol * 1.3, f"tol={tol:.0e}", fontsize=7, color="grey")
    ax.axvline(RCP_REF_PHI, color="crimson", lw=1.2, label="RCP 0.6366 (Scott 1960)")
    ax.set_xlabel("imposed solid fraction $\\phi$")
    ax.set_ylabel("max residual overlap / $d$")
    ax.set_title("Residual overlap is measurable; the solid fraction is imposed")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "overlap_vs_phi.png"), dpi=140)
    plt.close(fig)

    # 2) phi_J as a function of the overlap tolerance
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    tols = [1e-4, 1e-3, 1e-2, 5e-2]
    for n, cur in result["jamming_curves"].items():
        ys = [cur[f"{t:.0e}"]["phi_j"] for t in tols]
        ax.semilogx(tols, ys, "s-", label=f"$\\phi_J$(tol), N={n}")
    ax.axhline(RCP_REF_PHI, color="crimson", lw=1.2, label="RCP 0.6366")
    ax.set_xlabel("overlap tolerance (max overlap / $d$)")
    ax.set_ylabel("jamming solid fraction $\\phi_J$")
    ax.set_title("The jamming estimate depends on the tolerance — report the curve")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "phi_j_vs_tolerance.png"), dpi=140)
    plt.close(fig)

    # 3) the tautology: achieved phi vs target phi
    taut_path = os.path.join(OUT, "tautology.json")
    if os.path.exists(taut_path):
        with open(taut_path) as fh:
            taut = json.load(fh)
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        for row in taut["rows"]:
            ax.plot(row["target_phi"], row["achieved_phi"], "o")
        lim = [0.5, 0.75]
        ax.plot(lim, lim, "k--", lw=1, label="identity $\\phi = 1-\\epsilon$")
        ax.set_xlabel("target solid fraction $1-\\epsilon$ (benchmark input)")
        ax.set_ylabel("achieved solid fraction (reported as a result)")
        ax.set_title("Tautology control: the 'result' is the input")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(figs, "tautology_control.png"), dpi=140)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="analyse", choices=["tautology", "scan", "analyse"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-list", type=int, nargs="+", default=[1000])
    ap.add_argument("--relax", type=int, default=6000)
    ap.add_argument("--phi-grid", type=float, nargs="+", default=None)
    args = ap.parse_args()

    if args.stage == "tautology":
        stage_tautology()
    elif args.stage == "scan":
        grid = args.phi_grid or [round(0.60 + 0.005 * i, 3) for i in range(13)]
        stage_scan(args.n, grid, args.relax)
    else:
        stage_analyse(n_list=args.n_list)


if __name__ == "__main__":
    main()