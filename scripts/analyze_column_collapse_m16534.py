"""CASE-COLUMN-COLLAPSE round 3 (m16534) — reduce the mu x a curve grid.

Reads only the per-case JSON written by
``benchmarks/case_column_collapse_lube2004.py::run_case`` (fetch + parameters +
front time series + energy ledger = provenance) and produces

* ``artifacts/case_column_collapse_m16534/sweep_table.csv``  one row per case
* ``artifacts/case_column_collapse_m16534/cases_slim.json``  same numbers, no
  position arrays (so the evidence is small enough to live in git)
* ``artifacts/case_column_collapse_m16534/curve_summary.json``  per-mu
  curve-level statistics + the pre-registered decision
* ``runout_vs_a.png``, ``err_vs_a.png``, ``time_metric.png``

No reference number is typed here: the Lube (2004) targets are evaluated from
:func:`tensordem.granular_column.axisym.lube_2004_runout_ratio` /
``lube_2004_emplacement_time``.

Usage::

    python3 scripts/analyze_column_collapse_m16534.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.granular_column.axisym import (LUBE2004_DOI,  # noqa: E402
                                              LUBE2004_SOURCE,
                                              lube_2004_emplacement_time,
                                              lube_2004_runout_ratio)

CASE_DIR = os.path.join(ROOT, "runs", "case_column_collapse_lube2004", "cases")
OUT = os.path.join(ROOT, "artifacts", "case_column_collapse_m16534")
TOL = 0.15           # pre-registered tolerance, unchanged since round 1
A_REF_SPLIT = 1.7    # branch switch of the reference law


# --------------------------------------------------------------------------
def load_cases(pattern: str = "sweep_mu*_mur0.4.json") -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(CASE_DIR, pattern))):
        with open(path) as fh:
            r = json.load(fh)
        c = r["case"]
        a = c["aspect_ratio"]
        tfrac = r.get("t_reach_by_fraction", {})
        rows.append({
            "case": c["label"],
            "aspect": a,
            "mu": c["mu"],
            "mu_rolling": c["mu_rolling"],
            "n_particles": r["n_particles"],
            "dt": r["dt"],
            "n_steps": r["n_steps"],
            "grains_per_radius": c["grains_per_radius"],
            "rest_coord": c.get("restitution"),
            "runout_ratio_p995": r["runout_ratio_p995"],
            "runout_ratio_max": r["runout_ratio_max"],
            "ref_runout_ratio": r["reference_runout_ratio"],
            "err_runout": r["runout_ratio_p995"] / r["reference_runout_ratio"] - 1.0,
            "err_runout_max_metric": r["runout_ratio_max"] / r["reference_runout_ratio"] - 1.0,
            "t99": r["t_reach"],
            "t95": tfrac.get("0.95"),
            "t90": tfrac.get("0.9"),
            "ref_t": r["reference_emplacement_time"],
            "err_t99": r["t_reach"] / r["reference_emplacement_time"] - 1.0,
            "err_t95": (tfrac.get("0.95") / r["reference_emplacement_time"] - 1.0)
            if tfrac.get("0.95") is not None else None,
            "err_t90": (tfrac.get("0.9") / r["reference_emplacement_time"] - 1.0)
            if tfrac.get("0.9") is not None else None,
            "stop_reason": r["stop_reason"],
            "solid_fraction_initial": r["solid_fraction_initial"],
            "energy_residual": r["energy"]["residual_fraction_of_potential"],
            "dissipated_physical_frac": r["energy"]["dissipated_physical_only"]
            / r["energy"]["potential_released"],
            "wall_clock_s": r.get("wall_clock_s"),
        })
    if not rows:
        raise SystemExit(f"no case JSON under {CASE_DIR}")
    return rows


def curve_stats(rows: list[dict], field_err: str) -> dict:
    errs = np.array([r[field_err] for r in rows], dtype=float)
    return {
        "n": int(errs.size),
        "max_abs_err": float(np.max(np.abs(errs))),
        "rms_err": float(np.sqrt(np.mean(errs ** 2))),
        "mean_err": float(np.mean(errs)),
        "n_pass": int(np.sum(np.abs(errs) < TOL)),
        "all_pass": bool(np.all(np.abs(errs) < TOL)),
    }


def required_mu(rows_at_aspect: list[dict]) -> dict:
    """Invert the mu -> runout family at one aspect ratio.

    Returns where the DEM runout crosses the reference value, by linear
    interpolation between the sampled ``mu`` levels.  A crossing outside the
    sampled range is reported as ``extrapolated`` — the number is then a
    lower bound on the required friction, not a measurement.
    """
    rs = sorted(rows_at_aspect, key=lambda r: r["mu"])
    mus = np.array([r["mu"] for r in rs], dtype=float)
    out = np.array([r["runout_ratio_p995"] for r in rs], dtype=float)
    ref = rs[0]["ref_runout_ratio"]
    d = out - ref
    crossing = None
    extrapolated = False
    for k in range(len(mus) - 1):
        if d[k] == 0.0:
            crossing = float(mus[k])
            break
        if d[k] * d[k + 1] < 0.0:
            w = -d[k] / (d[k + 1] - d[k])
            crossing = float(mus[k] + w * (mus[k + 1] - mus[k]))
            break
    if crossing is None and d.size:
        # no sign change: the reference is not bracketed by the sampled mu
        extrapolated = True
        k = int(np.argmin(np.abs(d)))
        if 0 < k < len(mus) - 1:
            k0, k1 = k - 1, k + 1
        else:
            k0, k1 = (0, 1) if k == 0 else (len(mus) - 2, len(mus) - 1)
        slope = (out[k1] - out[k0]) / (mus[k1] - mus[k0])
        crossing = float(mus[k] - d[k] / slope) if slope != 0 else float("nan")
    return {"aspect": rs[0]["aspect"], "ref_runout_ratio": ref,
            "sampled_mu": mus.tolist(), "runout": out.tolist(),
            "required_mu": crossing, "extrapolated": extrapolated,
            "mu_range": [float(mus.min()), float(mus.max())]}


def tilt_fit(rows: list[dict], field_err: str = "err_runout") -> dict:
    """Least-squares err = c0 + c1*a for one mu: c1 is the structural tilt."""
    a = np.array([r["aspect"] for r in rows], dtype=float)
    e = np.array([r[field_err] for r in rows], dtype=float)
    c1, c0 = np.polyfit(a, e, 1)
    resid = e - (c0 + c1 * a)
    return {"intercept": float(c0), "slope_per_aspect": float(c1),
            "rms_residual": float(np.sqrt(np.mean(resid ** 2))),
            "residual_span": float(resid.max() - resid.min())}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    rows = load_cases()

    mus = sorted({r["mu"] for r in rows})
    aspects = sorted({r["aspect"] for r in rows})
    by_mu = {mu: sorted([r for r in rows if r["mu"] == mu],
                        key=lambda r: r["aspect"]) for mu in mus}

    # ---- per-mu curve-level statistics (pre-registered decision rule) -----
    summary = {
        "case": "CASE-COLUMN-COLLAPSE",
        "round": "3 (m16534)",
        "question": ("does ONE contact friction mu reproduce the Lube (2004) "
                     "runout law at every aspect ratio (<15% each) at the "
                     "round-1 rolling resistance mu_r = 0.4?"),
        "reference": LUBE2004_SOURCE,
        "doi": LUBE2004_DOI,
        "tolerance": TOL,
        "mus": mus,
        "aspects": aspects,
        "mu_rolling": sorted({r["mu_rolling"] for r in rows}),
        "per_mu": {},
        "grid_complete": None,
    }
    for mu, rs in by_mu.items():
        summary["per_mu"][f"{mu:g}"] = {
            "curve": curve_stats(rs, "err_runout"),
            "curve_max_metric": curve_stats(rs, "err_runout_max_metric"),
            "time_t99": curve_stats(rs, "err_t99"),
            "time_t95": curve_stats(rs, "err_t95"),
            "time_t90": curve_stats(rs, "err_t90"),
        }
    grid = {(r["mu"], r["aspect"]) for r in rows}
    missing = [(mu, a) for mu in mus for a in aspects if (mu, a) not in grid]
    summary["grid_complete"] = not missing
    summary["grid_missing"] = [f"mu={mu:g},a={a:g}" for mu, a in missing]

    # single best mu by max-abs runout error ("worst-case" criterion)
    ranked = sorted(summary["per_mu"].items(),
                    key=lambda kv: kv[1]["curve"]["max_abs_err"])
    summary["ranking_by_worst_case"] = [k for k, _ in ranked]
    summary["best_single_mu"] = ranked[0][0]
    summary["best_single_mu_stats"] = ranked[0][1]["curve"]
    summary["hypothesis_one_mu_fits_whole_curve"] = bool(
        ranked[0][1]["curve"]["all_pass"])

    # structural tilt per mu: a constant-offset misfit (slope ~ 0) would mean
    # one mu is nearly right; a significant slope means no constant mu can fit
    # the whole curve no matter how it is tuned.
    for mu, rs in by_mu.items():
        summary["per_mu"][f"{mu:g}"]["tilt"] = tilt_fit(rs)

    # regime split: the reference law changes branch at a = 1.7, and the
    # required friction (below) jumps there, so the same statistics are
    # reported separately inside each branch of the empirical law.
    regimes = {"linear_branch": [a for a in aspects if a < A_REF_SPLIT],
               "sqrt_branch": [a for a in aspects if a >= A_REF_SPLIT]}
    summary["regimes"] = {k: {"aspects": v} for k, v in regimes.items()}
    for mu, rs in by_mu.items():
        for name, aset in regimes.items():
            sub = [r for r in rs if r["aspect"] in aset]
            if sub:
                summary["regimes"][name][f"mu={mu:g}"] = curve_stats(sub, "err_runout")
    for name, aset in regimes.items():
        cand = {f"{mu:g}": curve_stats([r for r in rs if r["aspect"] in aset],
                                       "err_runout")
                for mu, rs in by_mu.items()}
        best = min(cand.items(), key=lambda kv: kv[1]["max_abs_err"])
        summary["regimes"][name]["best_mu"] = best[0]
        summary["regimes"][name]["best_stats"] = best[1]

    # stopping-criterion sensitivity: is the emplacement-time FAIL (round 1/2:
    # +25..+44 %) a property of the contact law or of "t = end of the creep
    # tail"?  Same runs, three front-fraction definitions.
    tm = {}
    for field in ("err_t99", "err_t95", "err_t90"):
        vals = np.array([r[field] for r in rows if r[field] is not None], dtype=float)
        tm[field.replace("err_", "")] = {"n": int(vals.size),
                         "median_abs_err": float(np.median(np.abs(vals))),
                         "mean_err": float(np.mean(vals)),
                         "max_abs_err": float(np.max(np.abs(vals))),
                         "n_pass": int(np.sum(np.abs(vals) < TOL))}
    summary["time_metric_overview"] = tm

    # per-aspect friction demanded by the reference law (family inversion)
    summary["required_mu"] = [required_mu([r for r in rows if r["aspect"] == a])
                              for a in aspects]

    with open(os.path.join(OUT, "curve_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    with open(os.path.join(OUT, "cases_slim.json"), "w") as fh:
        json.dump(rows, fh, indent=1)

    keys = list(rows[0])
    with open(os.path.join(OUT, "sweep_table.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    # ---- figures ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis")
    colors = {mu: cmap(i / max(1, len(mus) - 1)) for i, mu in enumerate(mus)}

    # (1) runout ratio vs aspect ratio, reference law + DEM points per mu
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    aa_ref = np.linspace(0.2, 3.2, 400)
    ax[0].plot(aa_ref, [lube_2004_runout_ratio(a) for a in aa_ref], "k-", lw=2,
               label="Lube et al. (2004) law")
    ax[0].axvline(A_REF_SPLIT, color="k", ls=":", lw=1)
    for mu, rs in by_mu.items():
        ax[0].plot([r["aspect"] for r in rs], [r["runout_ratio_p995"] for r in rs],
                   "o-", color=colors[mu], ms=6, label=f"DEM, mu={mu:g}")
    ax[0].set_xlabel("aspect ratio a = h_i / r_i")
    ax[0].set_ylabel("runout ratio $r_\\infty / r_i$")
    ax[0].set_title("axisymmetric column collapse (mu_r = 0.4)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    for mu, rs in by_mu.items():
        ax[1].plot([r["aspect"] for r in rs], [100 * r["err_runout"] for r in rs],
                   "o-", color=colors[mu], ms=6, label=f"mu={mu:g}")
    ax[1].axhspan(-100 * TOL, 100 * TOL, color="green", alpha=0.12,
                  label=f"+/-{100 * TOL:.0f}% band")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("aspect ratio a")
    ax[1].set_ylabel("runout relative error [%]")
    ax[1].set_title("runout error vs aspect ratio")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "runout_vs_a.png"), dpi=140)
    plt.close(fig)

    # (2) worst-case error per mu + that mu's error profile
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    worst = [100 * summary["per_mu"][f"{mu:g}"]["curve"]["max_abs_err"] for mu in mus]
    rms = [100 * summary["per_mu"][f"{mu:g}"]["curve"]["rms_err"] for mu in mus]
    xs = np.arange(len(mus))
    ax[0].bar(xs - 0.18, worst, width=0.36, label="max |err| over a")
    ax[0].bar(xs + 0.18, rms, width=0.36, label="rms err over a")
    ax[0].axhline(100 * TOL, color="r", ls="--", label=f"{100 * TOL:.0f}% tolerance")
    ax[0].set_xticks(xs, [f"mu={mu:g}" for mu in mus])
    ax[0].set_ylabel("runout error [%]")
    ax[0].set_title("curve-level error per contact friction")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y")

    best = float(summary["best_single_mu"])
    rb = by_mu[best]
    ax[1].plot([r["aspect"] for r in rb], [100 * r["err_runout"] for r in rb],
               "o-", color="tab:blue", label="runout (p99.5 front)")
    ax[1].plot([r["aspect"] for r in rb], [100 * r["err_t99"] for r in rb],
               "s--", color="tab:red", label="emplacement t99")
    ax[1].axhspan(-100 * TOL, 100 * TOL, color="green", alpha=0.12)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("aspect ratio a")
    ax[1].set_ylabel("relative error [%]")
    ax[1].set_title(f"best single mu = {best:g}: runout vs emplacement time")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "err_vs_a.png"), dpi=140)
    plt.close(fig)

    # (3) emplacement-time metric sensitivity (definitional artifact test)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for mu, rs in by_mu.items():
        for field, ls in (("err_t90", ":"), ("err_t95", "--"), ("err_t99", "-")):
            vals = [r[field] for r in rs]
            if any(v is None for v in vals):
                continue
            ax.plot([r["aspect"] for r in rs], [100 * v for v in vals], ls,
                    color=colors[mu], lw=1.4,
                    label=f"mu={mu:g} {field[4:]}" if field != "err_t99" else f"mu={mu:g} t99")
    ax.axhspan(-100 * TOL, 100 * TOL, color="green", alpha=0.12,
               label=f"+/-{100 * TOL:.0f}% band")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("aspect ratio a")
    ax.set_ylabel("emplacement-time relative error [%]")
    ax.set_title("t90 / t95 / t99 vs reference $3\\sqrt{h_i/g}$")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "time_metric.png"), dpi=140)
    plt.close(fig)

    # (4) friction demanded by the reference law, per aspect ratio
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    rq = summary["required_mu"]
    ax.plot([q["aspect"] for q in rq], [q["required_mu"] for q in rq], "o-",
            color="tab:purple", label="mu required by the Lube law")
    ax.fill_between([q["aspect"] for q in rq],
                    [q["mu_range"][0] for q in rq],
                    [q["mu_range"][1] for q in rq],
                    color="tab:purple", alpha=0.10,
                    label="sampled mu range (0.5-0.8)")
    for q in rq:
        if q["extrapolated"]:
            ax.annotate("extrapolated", (q["aspect"], q["required_mu"]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=7, color="tab:red", ha="center")
    ax.set_xlabel("aspect ratio a")
    ax.set_ylabel("contact friction mu that matches the reference runout")
    ax.set_title("no single mu reproduces the whole curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "required_mu.png"), dpi=140)
    plt.close(fig)

    # ---- console table ---------------------------------------------------
    print(f"{'case':>26} {'a':>4} {'mu':>4} {'runout':>7} {'ref':>6} "
          f"{'err%':>7} {'t99':>6} {'ref_t':>6} {'err_t%':>7} {'Eres':>9}")
    for r in sorted(rows, key=lambda r: (r["mu"], r["aspect"])):
        print(f"{r['case']:>26} {r['aspect']:>4.1f} {r['mu']:>4.1f} "
              f"{r['runout_ratio_p995']:>7.3f} {r['ref_runout_ratio']:>6.3f} "
              f"{100 * r['err_runout']:>+7.1f} {r['t99']:>6.3f} {r['ref_t']:>6.3f} "
              f"{100 * r['err_t99']:>+7.1f} {r['energy_residual']:>9.2e}")
    print()
    for mu in mus:
        s = summary["per_mu"][f"{mu:g}"]["curve"]
        print(f"mu={mu:g}: max|err|={100 * s['max_abs_err']:.1f}%  "
              f"rms={100 * s['rms_err']:.1f}%  pass {s['n_pass']}/{s['n']}  "
              f"all_pass={s['all_pass']}")
    print("[emplacement time by stopping criterion, all cells]")
    for k, v in summary["time_metric_overview"].items():
        print(f"  {k}: median|err|={100 * v['median_abs_err']:5.1f}%  "
              f"mean={100 * v['mean_err']:+5.1f}%  max|err|={100 * v['max_abs_err']:5.1f}%  "
              f"pass {v['n_pass']}/{v['n']}")
    for name, data in summary["regimes"].items():
        print(f"[{name}] aspects={data['aspects']}  best mu={data['best_mu']}  "
              f"max|err|={100 * data['best_stats']['max_abs_err']:.1f}%  "
              f"pass {data['best_stats']['n_pass']}/{data['best_stats']['n']}")
        for k, v in data.items():
            if k.startswith("mu="):
                print(f"    {k}: max|err|={100 * v['max_abs_err']:>5.1f}%  "
                      f"pass {v['n_pass']}/{v['n']}")
    for q in summary["required_mu"]:
        print(f"a={q['aspect']:>4.1f}: required mu = {q['required_mu']:.3f}"
              f"{'  (EXTRAPOLATED)' if q['extrapolated'] else ''}"
              f"  ref={q['ref_runout_ratio']:.3f}")
    for mu in mus:
        t = summary["per_mu"][f"{mu:g}"]["tilt"]
        print(f"mu={mu:g} tilt: d(err)/da = {100 * t['slope_per_aspect']:+.1f} %/a, "
              f"intercept {100 * t['intercept']:+.1f}%, resid rms "
              f"{100 * t['rms_residual']:.1f}%")
    print(f"\nbest single mu (worst-case): {summary['best_single_mu']}  "
          f"hypothesis holds: {summary['hypothesis_one_mu_fits_whole_curve']}")
    if missing:
        print("MISSING GRID POINTS:", summary["grid_missing"])
    print("artifacts ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())