#!/usr/bin/env python3
"""Seed-resolved adjudication of the RCP *stiffness* ladder (hard-sphere limit).

Why this script exists
----------------------
The deposition packing fraction is measured on a soft-sphere DEM bed, so every
ladder point carries a compressibility systematic and the reported RCP value is
the ``k_n -> inf`` extrapolation

    phi(k_n) = phi_inf + A k_n^{-p}.

``aggregate.py`` builds that ladder from the **seed-1 chain** (one configuration
family, only ``k_n`` varying) and takes ``p`` from the two outermost deltas
(:func:`tensordem.deposition_packing.stiffness_limit`).  That estimator is
honest about *which* points it used, but it has no error bar: the outermost
delta of the ``mu = 0.1`` ladder was ``6.8e-4`` built from **two single-seed
points**, while the seed scatter of that tier is ``~4e-4``.  A limit quoted to
four digits from a delta at the 1.7 sigma level is not a measurement, it is a
coin flip, and it is the difference between the low-friction tier passing or
failing its acceptance band ``[0.62, 0.66]`` (the extrapolated value ``0.6171``
sits *below* the band).

This module supplies the missing error analysis.  Given seed replicates at
every stiffness level it

* aggregates each level to ``mean +/- sem`` and fits the three-parameter law by
  weighted non-linear least squares (:func:`fit_ladder`);
* propagates the *seed* uncertainty into the intercept by a parametric
  Monte-Carlo (:func:`limit_interval`), which yields a CI for ``phi_inf``
  instead of a point value -- components with ``n = 1`` borrow a pooled ``s``
  and are flagged ``pooled_sd`` so the report cannot mistake them for measured;
* evaluates three ladder definitions side by side -- seed-1 chain (the current
  production estimator), seed-mean chain, and leave-out-the-stiffest-level --
  so the *definition* sensitivity (:func:`definition_spread`) is visible next
  to the statistical one;
* judges the acceptance band by interval, not by point: ``IN_BAND``,
  ``OUT_OF_BAND`` or ``UNDECIDABLE`` (:func:`judge_band`).

Usage::

    PYTHONPATH=src python3 benchmarks/case_rcp_deposition/stiffness_seed_audit.py
    PYTHONPATH=src python3 benchmarks/case_rcp_deposition/stiffness_seed_audit.py --json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNS = os.path.join(HERE, "runs")
FIGDIR = os.path.join(ROOT, "benchmarks", "case_rcp_deposition", "figures")
OUT_JSON = os.path.join(RUNS, "stiffness_seed_audit.json")

N_PRIMARY = 10000
DT_DEFAULT = 1.0 / 20.0
BAND = (0.62, 0.66)          # CASE-DEM-RCP-PACKING-FRACTION acceptance band
PHI_REF_FRICTIONLESS = 0.6366  # Scott 1960 / Bernal & Mason 1960
P_BOUNDS = (0.2, 4.0)
N_MC = 4000
SEED = 20260117

__all__ = [
    "load_runs", "levels", "fit_ladder", "limit_interval", "definition_spread",
    "judge_band", "measured_interval", "audit", "BAND", "P_BOUNDS",
]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_runs(runs_dir: str = RUNS, n: int = N_PRIMARY,
              dt_frac: float = DT_DEFAULT) -> list:
    """Coarse-timestep production runs as plain dicts (deterministic order)."""
    out = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "deposition_*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        spec, m = d["spec"], d["metrics"]
        if int(spec["n_particles"]) != n:
            continue
        if abs(float(spec.get("dt_frac", DT_DEFAULT)) - dt_frac) > 1e-12:
            continue
        phi = m.get("phi_bulk")
        if phi is None or phi != phi:
            continue
        out.append({
            "path": os.path.relpath(path, ROOT),
            "n": int(spec["n_particles"]),
            "mu": float(spec["mu"]),
            "kn": float(spec["kn"]),
            "seed": int(spec["seed"]),
            "phi": float(phi),
        })
    return out


def levels(runs: list, mu: float) -> list:
    """``[{kn, n, mean, sd, sem, pooled_sd, phis}]`` sorted by ascending kn."""
    by_kn: dict[float, list] = {}
    by_seed: dict[float, dict] = {}
    for r in runs:
        if abs(r["mu"] - mu) < 1e-12:
            by_kn.setdefault(r["kn"], []).append(r["phi"])
            by_seed.setdefault(r["kn"], {})[r["seed"]] = r["phi"]
    # pooled sd across the levels that do have replicates (used only for n = 1)
    reps = [np.asarray(v, float) for v in by_kn.values() if len(v) > 1]
    if reps:
        dof = sum(len(v) - 1 for v in reps)
        pooled = math.sqrt(sum(float(((v - v.mean()) ** 2).sum()) for v in reps)
                           / dof) if dof > 0 else 0.0
    else:
        pooled, dof = 0.0, 0
    out = []
    for kn in sorted(by_kn):
        phis = np.asarray(sorted(by_kn[kn]), float)
        n = len(phis)
        mean = float(phis.mean())
        sd = float(phis.std(ddof=1)) if n > 1 else None
        if n > 1:
            sem, err_model, df = sd / math.sqrt(n), "seed", n - 1
        elif pooled > 0.0:
            sem, err_model, df = pooled, "pooled_sd", dof
        else:
            # no replicate anywhere: the ladder carries no noise information
            sem, err_model, df = None, "no_noise", 0
        out.append({
            "kn": float(kn), "n": n, "mean": mean, "sd": sd,
            "sem": (float(sem) if sem is not None else None),
            "df": int(df), "err_model": err_model,
            "pooled_sd": float(pooled), "pooled_dof": int(dof), "phis": [float(x) for x in phis],
            "phi_seed1": (float(by_seed[kn][1]) if 1 in by_seed[kn] else None),
        })
    return out


# ---------------------------------------------------------------------------
# estimator
# ---------------------------------------------------------------------------
def fit_ladder(points: list, p_fixed: float | None = None) -> dict:
    """Weighted fit of ``phi = phi_inf + A k_n^{-p}`` (closed-form in ``A``).

    ``points`` is ``[(kn, phi, sigma), ...]`` with ``sigma > 0``.  For a given
    order ``p`` the model is *linear* in ``(phi_inf, A)``, so the weighted
    least-squares problem reduces to a 1-D minimisation of ``chi2(p)`` -- exact,
    fast, and free of the local minima a 3-parameter Levenberg-Marquardt fit can
    fall into.  ``p`` is bounded to ``P_BOUNDS`` unless ``p_fixed`` is given.
    """
    from scipy.optimize import minimize_scalar

    kn = np.array([pt[0] for pt in points], float)
    val = np.array([pt[1] for pt in points], float)
    sig = np.array([pt[2] for pt in points], float)
    if len(kn) < 2:
        raise ValueError("need at least two stiffness levels")
    if np.any(sig <= 0):
        raise ValueError("sigma must be positive")
    order = np.argsort(kn)
    kn, val, sig = kn[order], val[order], sig[order]

    def solve(p: float):
        w = 1.0 / sig
        X = np.column_stack([np.ones_like(kn), kn ** (-p)]) * w[:, None]
        y = val * w
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = X @ coef - y
        return coef, float((r ** 2).sum())

    if p_fixed is None:
        res = minimize_scalar(lambda p: solve(p)[1], bounds=P_BOUNDS,
                              method="bounded", options={"xatol": 1e-8})
        p_out = float(res.x)
        coef, chi2 = solve(p_out)
    else:
        p_out = float(p_fixed)
        coef, chi2 = solve(p_out)
    phi_inf, amp = float(coef[0]), float(coef[1])
    dof = max(len(kn) - (2 if p_fixed is not None else 3), 1)
    r = (phi_inf + amp * kn ** (-p_out) - val) / sig
    return {
        "phi_inf": phi_inf,
        "A": amp,
        "p": p_out,
        "p_free": p_fixed is None,
        "chi2": chi2,
        "chi2_red": chi2 / dof,
        "rms": float(np.sqrt((r ** 2).mean())),
        "kn": kn.tolist(),
        "phi": val.tolist(),
        "sigma": sig.tolist(),
        "residuals": r.tolist(),
    }


def limit_interval(levels_: list, *, n_mc: int = N_MC, p_fixed: float | None = None,
                   rng: np.random.Generator | None = None) -> dict:
    """Student-t Monte-Carlo interval for ``phi_inf`` from the *seed* errors.

    Each level mean is redrawn as ``mean + sem * T_df`` with ``T_df`` a Student-t
    deviate on the degrees of freedom of the standard error (``n-1`` for a
    replicated level, the pooled dof for a single-seed one).  A normal draw
    would treat ``sem`` as exact; with three seeds per level it is a two-dof
    estimate, and a normal-draw interval then undercovers (measured: 67 %
    instead of 95 % on a synthetic ladder -- see
    ``tests/test_rcp_stiffness_seed_audit.py``).

    A ladder in which *no* level has replicates carries no noise information at
    all; the interval is then reported as absent (``lo = hi = None``) rather
    than implying confidence the data do not support.
    """
    rng = rng or np.random.default_rng(SEED)
    if any(lv["err_model"] == "no_noise" for lv in levels_):
        base = [(lv["kn"], lv["mean"], 1e-12) for lv in levels_]
        center = fit_ladder(base, p_fixed=p_fixed)
        return {"phi_inf": center["phi_inf"], "p": center["p"], "A": center["A"],
                "chi2_red": None, "lo": None, "hi": None, "sd": None,
                "p_lo": None, "p_hi": None, "n_mc": 0,
                "pooled_levels": [lv["kn"] for lv in levels_],
                "noise_available": False}
    base = [(lv["kn"], lv["mean"], max(lv["sem"], 1e-9), max(int(lv["df"]), 1))
            for lv in levels_]
    center = fit_ladder([(k, m, s) for k, m, s, _ in base], p_fixed=p_fixed)
    lims, ps = [], []
    for _ in range(n_mc):
        pts = [(k, float(m + s * rng.standard_t(df)), s) for k, m, s, df in base]
        try:
            f = fit_ladder(pts, p_fixed=p_fixed)
        except Exception:          # a draw that cannot be fitted is dropped
            continue
        lims.append(f["phi_inf"])
        ps.append(f["p"])
    lims = np.array(lims)
    ps = np.array(ps)
    return {
        "phi_inf": center["phi_inf"],
        "p": center["p"],
        "A": center["A"],
        "chi2_red": center["chi2_red"],
        "lo": float(np.percentile(lims, 2.5)) if len(lims) else None,
        "hi": float(np.percentile(lims, 97.5)) if len(lims) else None,
        "sd": float(lims.std(ddof=1)) if len(lims) > 1 else None,
        "p_lo": float(np.percentile(ps, 2.5)) if len(ps) else None,
        "p_hi": float(np.percentile(ps, 97.5)) if len(ps) else None,
        "n_mc": int(len(lims)),
        "pooled_levels": [lv["kn"] for lv in levels_ if lv["err_model"] != "seed"],
        "noise_available": True,
    }


def measured_interval(level: dict, conf: float = 0.95) -> dict:
    """Two-sided t-interval for a *measured* level (no extrapolation).

    The acceptance band of the anchor refers to a poured packing fraction, i.e.
    to a measured number at a declared stiffness, so every level is reported
    with its own interval and its own band verdict.  A single-seed level uses
    the pooled ``s`` (and its dof), which is flagged in ``err_model``.
    """
    from scipy import stats

    n = int(level["n"])
    if n > 1:
        df = n - 1
        sem = float(level["sd"]) / math.sqrt(n)
        centre = float(level["mean"])
        model = "seed"
    elif level.get("sem"):
        df = int(max(level["df"], 1))
        sem = float(level["sem"])
        centre = float(level["mean"])
        model = "pooled_sd"
    else:
        return {"kn": level["kn"], "n": n, "mean": level["mean"],
                "lo": None, "hi": None, "sem": None,
                "err_model": "no_noise", "verdict": "NO_INTERVAL"}
    tq = float(stats.t.ppf(0.5 + conf / 2.0, df))
    lo, hi = centre - tq * sem, centre + tq * sem
    return {"kn": level["kn"], "n": n, "mean": centre, "sem": sem,
            "df": df, "t": tq, "lo": lo, "hi": hi, "err_model": model,
            "verdict": judge_band(lo, hi)}


def judge_band(lo: float | None, hi: float | None,
               band: tuple = BAND) -> str:
    """Interval verdict against the acceptance band (never a point verdict)."""
    if lo is None or hi is None:
        return "NO_INTERVAL"
    if lo >= band[0] and hi <= band[1]:
        return "IN_BAND"
    if hi < band[0] or lo > band[1]:
        return "OUT_OF_BAND"
    return "UNDECIDABLE"


def definition_spread(levels_: list, p_fixed: float | None = None) -> dict:
    """Spread of ``phi_inf`` over ladder *definitions* (not just seed noise).

    * ``seed1_chain``   -- the production estimator (``aggregate.py``): seed 1
      at every level, unweighted, ``p`` from the outermost deltas;
    * ``seedmean_chain`` -- seed means, weighted;
    * ``drop_stiffest``  -- seed-mean chain without the stiffest level, i.e. the
      extrapolation the ladder would give if the most expensive level were
      unavailable;
    * ``p_fixed_1``      -- seed-mean chain with the theoretical linear-spring
      order ``p = 1`` (a soft-sphere overlap is proportional to ``1/k_n``).
    """
    from tensordem.deposition_packing import stiffness_limit

    out: dict = {}
    chain1 = [(lv["kn"], lv["phi_seed1"]) for lv in levels_
              if lv.get("phi_seed1") is not None]
    if len(chain1) >= 2:
        s = stiffness_limit(chain1)
        out["seed1_chain"] = {"phi_inf": s["limit"], "p": s["order_used"],
                             "order_observed": s["order"], "weighted": False}
    if len(levels_) >= 2:
        out["seedmean_chain"] = limit_interval(levels_, p_fixed=p_fixed)
    if len(levels_) >= 3:
        out["drop_stiffest"] = limit_interval(levels_[:-1], p_fixed=p_fixed)
    if len(levels_) >= 2:
        out["p_fixed_1"] = limit_interval(levels_, p_fixed=1.0)
    vals = [v["phi_inf"] for v in out.values()
            if v.get("phi_inf") is not None and v.get("phi_inf") == v.get("phi_inf")]
    out["_spread"] = (max(vals) - min(vals)) if vals else None
    return out


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def audit(runs: list | None = None, *, n_mc: int = N_MC) -> dict:
    runs = load_runs() if runs is None else runs
    rows = {}
    for mu in sorted({r["mu"] for r in runs}):
        lv = levels(runs, mu)
        if len(lv) < 2:
            continue
        rows[f"{mu:g}"] = {
            "mu": float(mu),
            "levels": lv,
            "n_levels": len(lv),
            "fit": limit_interval(lv, n_mc=n_mc),
            "measured": [measured_interval(l) for l in lv],
            "definitions": definition_spread(lv),
        }
        f = rows[f"{mu:g}"]["fit"]
        rows[f"{mu:g}"]["verdict_band"] = judge_band(f["lo"], f["hi"])
        if abs(mu) < 1e-12:
            rows[f"{mu:g}"]["ref"] = {
                "phi_ref": PHI_REF_FRICTIONLESS,
                "source": "Scott 1960 Nature 188:908; Bernal & Mason 1960 Nature 188:910",
                "rel_err": f["phi_inf"] / PHI_REF_FRICTIONLESS - 1.0,
            }
    n_cert = sum(1 for r in rows.values() if r["verdict_band"] in ("IN_BAND",
                                                                   "OUT_OF_BAND"))
    return {
        "n_primary": N_PRIMARY,
        "band": list(BAND),
        "n_mc": n_mc,
        "tiers": rows,
        "n_resolved": n_cert,
        "n_tiers": len(rows),
        "all_resolved": n_cert == len(rows) and bool(rows),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _figures(res: dict, path_prefix: str) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
    files = []

    # (1) ladder with seed error bars and the fitted law
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhspan(*BAND, color="0.86", zorder=0,
               label=f"acceptance band {BAND[0]}-{BAND[1]}")
    ax.axhline(PHI_REF_FRICTIONLESS, color="k", ls=":", lw=1.2,
               label="Scott/Bernal & Mason 0.6366")
    for i, (key, row) in enumerate(sorted(res["tiers"].items())):
        col = f"C{i}"
        kn = np.array([lv["kn"] for lv in row["levels"]])
        mean = np.array([lv["mean"] for lv in row["levels"]])
        sem = np.array([max(lv["sem"], 1e-9) for lv in row["levels"]])
        ns = [lv["n"] for lv in row["levels"]]
        ax.errorbar(kn, mean, yerr=sem, fmt="o", color=col, capsize=3,
                    label=f"mu={key}: seed mean (n={ns})")
        f = row["fit"]
        kk = np.linspace(kn.min() * 0.9, kn.max() * 1.05, 200)
        ax.plot(kk, f["phi_inf"] + f["A"] * kk ** (-f["p"]), color=col, lw=1.2)
        ax.plot([kn.min()], [f["phi_inf"]], "s", color=col, ms=7)
        off = 12 if i == 0 else -18
        ax.annotate(f"phi_inf={f['phi_inf']:.5f} (p={f['p']:.2f})",
                    (kn.min(), f["phi_inf"]), color=col, fontsize=8,
                    textcoords="offset points", xytext=(6, off))
    ax.set_xscale("log")
    ax.set_xlabel("normal stiffness $k_n$ (lattice units)")
    ax.set_ylabel(r"bulk volume fraction $\phi$")
    ax.set_title("Soft-sphere stiffness ladder with seed error bars")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = path_prefix + "stiffness_ladder_seeds.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    files.append(p1)

    # (2) Monte-Carlo distribution of the hard-sphere limit vs the band
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    rng = np.random.default_rng(SEED + 1)
    ax.axvspan(*BAND, color="0.86", zorder=0)
    for i, (key, row) in enumerate(sorted(res["tiers"].items())):
        lv = row["levels"]
        base = [(l["kn"], l["mean"], max(l["sem"], 1e-9)) for l in lv]
        draws = []
        for _ in range(N_MC):
            pts = [(k, float(rng.normal(m, s)), s) for k, m, s in base]
            try:
                draws.append(fit_ladder(pts)["phi_inf"])
            except Exception:
                continue
        ax.hist(draws, bins=60, alpha=0.55, color=f"C{i}",
                label=f"mu={key}: {row['verdict_band']}")
        ax.axvline(row["fit"]["phi_inf"], color=f"C{i}", lw=1.5)
    ax.set_xlabel(r"hard-sphere limit $\phi_\infty$")
    ax.set_ylabel("Monte-Carlo draws")
    ax.set_title("Seed-propagated uncertainty of the $k_n\\to\\infty$ limit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = path_prefix + "stiffness_limit_ci.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    files.append(p2)
    # Store the paths repo-relative: an absolute path bakes this checkout's
    # location into a committed artefact, so a copy of the artefact points at
    # somebody else's working tree (and dies when the checkout moves).
    return [os.path.relpath(f, ROOT) for f in files]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--n-mc", type=int, default=N_MC)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args(argv)

    res = audit(n_mc=args.n_mc)
    if not args.no_figures:
        res["figures"] = _figures(res, os.path.join(FIGDIR, "fig_stiff_"))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    if args.json:
        print(json.dumps(res, indent=1, sort_keys=True))
        return 0
    for key, row in sorted(res["tiers"].items()):
        f = row["fit"]
        print(f"[mu={key}] levels={row['n_levels']} "
              f"phi_inf={f['phi_inf']:.5f} CI=[{f['lo']:.5f},{f['hi']:.5f}] "
              f"p={f['p']:.2f} [{f['p_lo']:.2f},{f['p_hi']:.2f}] "
              f"chi2_red={f['chi2_red']:.2f} -> {row['verdict_band']} "
              f"(spread over definitions {row['definitions']['_spread']:.5f})")
        for m in row["measured"]:
            iv = ("[%.5f,%.5f]" % (m["lo"], m["hi"])) if m["lo"] is not None else "n/a"
            print(f"    kn={m['kn']:>9.0f} n={m['n']} phi={m['mean']:.5f} "
                  f"CI={iv} -> {m['verdict']} ({m['err_model']})")
    print(f"resolved tiers: {res['n_resolved']}/{res['n_tiers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())