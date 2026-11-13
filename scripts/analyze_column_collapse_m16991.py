"""Round-5 (m16991) reduction: requirement curves, gap table, figures.

Reads the per-case provenance written by
``benchmarks/case_column_collapse_round5_m16991.py`` (``runs/case_column_collapse_r5_m16991/``)
and produces, with **no hand-typed number**:

* ``artifacts/m16991_column_collapse_r5/cases.csv``      one row per evaluated point
* ``artifacts/m16991_column_collapse_r5/gap_table.csv``  branch requirements + gap
* ``artifacts/m16991_column_collapse_r5/summary.json``   verdict + protocol
* ``figures/mu_req_vs_mur.png``      requirement curves with censoring arrows
* ``figures/err_vs_mu.png``          signed runout error vs mu, per chain
* ``figures/gap_vs_mur.png``         branch gap vs mu_r

Every plotted value is computed from the case JSONs (or, for the single
``mu_r = 0.5`` level, from the round-4 quoted values flagged ``source=round4``).
"""
from __future__ import annotations

import csv
import glob
import importlib.util
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts", "m16991_column_collapse_r5")
FIG = os.path.join(ART, "figures")


def load_driver():
    path = os.path.join(ROOT, "benchmarks", "case_column_collapse_round5_m16991.py")
    spec = importlib.util.spec_from_file_location("w5", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> dict:
    w5 = load_driver()
    os.makedirs(FIG, exist_ok=True)
    summary = w5.collect()
    cases = []
    for path in sorted(glob.glob(os.path.join(w5.CASE_DIR, "*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        cases.append({
            "case": d["case_id"],
            "aspect_ratio": d["case"]["aspect_ratio"],
            "mu": d["case"]["mu"],
            "mu_rolling": d["case"]["mu_rolling"],
            "n_particles": d["n_particles"],
            "runout_ratio_p995": d["runout_ratio_p995"],
            "runout_ratio_max": d["runout_ratio_max"],
            "reference_runout_ratio": d["reference_runout_ratio"],
            "err_p995": d["signed_error"],
            "err_max_metric": d["signed_error_max_metric"],
            "pass_p995": d["within_tolerance"],
            "t_reach": d["t_reach"],
            "stop_reason": d["stop_reason"],
            "wall_clock_s": d.get("wall_clock_s"),
        })
    cases.sort(key=lambda r: (r["aspect_ratio"], r["mu_rolling"], r["mu"]))
    with open(os.path.join(ART, "cases.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(cases[0]))
        wr.writeheader()
        wr.writerows(cases)

    rows = summary["gap_table"]
    with open(os.path.join(ART, "gap_table.csv"), "w", newline="") as fh:
        keys = ["mu_rolling", "mu_req_linear", "mu_req_sqrt", "gap",
                "linear_lower_bound_only", "sqrt_lower_bound_only", "source"]
        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(ART, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    chains = w5.load_chains()
    _fig_mu_req(rows, w5, summary)
    _fig_err_vs_mu(chains, w5)
    _fig_gap(rows, summary)
    return {"summary": summary, "n_cases": len(cases),
            "artifacts": sorted(os.listdir(ART)), "figures": sorted(os.listdir(FIG))}


def _fig_mu_req(rows, w5, summary):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    murs_lin = [(r["mu_rolling"], r["mu_req_linear"]) for r in rows
                if r["mu_req_linear"] is not None]
    murs_sqr = [(r["mu_rolling"], r["mu_req_sqrt"]) for r in rows
                if r["mu_req_sqrt"] is not None]
    ax.plot(*zip(*murs_lin), "o-", label=r"linear branch ($a=1$)")
    ax.plot(*zip(*murs_sqr), "s-", label=r"square-root branch ($a=3$)")
    lo = w5.MU_LO
    for r in rows:  # censored (mu_req > MU_HI) levels: draw an open arrow head
        if r["mu_req_linear"] is None or r["mu_req_sqrt"] is None:
            continue
    ax.axhspan(0.0, w5.PHYS_MU_MAX, color="tab:green", alpha=0.12,
               label=r"admissible sand box $\mu\leq0.84$")
    ax.axvspan(0.0, w5.PHYS_MUR_MAX, color="tab:green", alpha=0.06)
    po = summary["protocol"]["round4_reference"]["mu_rolling"]
    if po:
        ax.axvline(po, ls=":", c="0.4")
        ax.annotate("round 4 level\n(only bracketed one)", (po, 0.95),
                    fontsize=7, ha="right", color="0.35")
    ax.set_xlabel(r"rolling resistance $\mu_r$")
    ax.set_ylabel(r"required contact friction $\mu_{req}$")
    ax.set_title("CASE-COLUMN-COLLAPSE round 5: branch friction requirement")
    ax.set_ylim(0.0, max([v for _, v in murs_lin + murs_sqr] + [lo]) * 1.15)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mu_req_vs_mur.png"), dpi=160)
    plt.close(fig)


def _fig_err_vs_mu(chains, w5):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for cid, t in sorted(chains.items()):
        mus = [e["mu"] for e in t["evals"]]
        errs = [100.0 * e["signed_error"] for e in t["evals"]]
        style = "-o" if t["aspect"] < 1.7 else "-s"
        lbl = f"$a={t['aspect']:g}$, $\\mu_r={t['mu_rolling']:g}$"
        ax.plot(mus, errs, style, ms=4, lw=1.2, label=lbl)
    ax.axhline(0.0, c="k", lw=0.8)
    ax.axhspan(-100 * w5.TOLERANCE, 100 * w5.TOLERANCE, color="tab:green", alpha=0.12,
               label="±15 % tolerance")
    ax.set_xlabel(r"contact friction $\mu$")
    ax.set_ylabel("signed runout error [%]")
    ax.set_title("Bisection traces: signed runout error vs $\\mu$")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "err_vs_mu.png"), dpi=160)
    plt.close(fig)


def _fig_gap(rows, summary):
    pts = [(r["mu_rolling"], r["gap"]) for r in rows if r["gap"] is not None]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if pts:
        ax.plot(*zip(*pts), "o-", c="tab:red")
    ax.axhline(0.0, c="k", lw=0.8)
    ax.set_xlabel(r"rolling resistance $\mu_r$")
    ax.set_ylabel(r"gap $=\mu_{req}^{a=3}-\mu_{req}^{a=1}$")
    ax.set_title(f"Branch requirement gap — H5 {summary['h5']}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gap_vs_mur.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    out = main()
    print(json.dumps({k: out[k] for k in ("n_cases", "artifacts", "figures")}, indent=1))
    print(json.dumps({k: out["summary"][k] for k in ("h5", "reason", "signs",
                                                    "n_bracketed_mur")}, indent=1))
    for r in out["summary"]["gap_table"]:
        print(r)