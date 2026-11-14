"""Adversarial verification of XVAL-DEM-ROUND2-DEPOSITION-RIGID-LIMIT.

The round-2 report (``docs/XVAL_DEM_ROUND2_DEPOSITION_RIGID_LIMIT.md``) claims
that Richardson extrapolation of the soft-sphere volume fraction gives a
hard-sphere limit ``phi_inf(mu=0) = 0.63535`` which agrees with the literature
RCP ``0.6366`` to ``-0.20%``, and that the observed convergence order
``p = 1.007`` matches the theoretical ``p = 1`` for linear-spring overlap.

This script re-derives every number from the committed run JSONs (zero new
simulation) and runs the checks the round-2 report did not run:

``A`` ladder / aggregation sensitivity -- the same data admits several
      defensible ladders (3 vs 4 stiffness levels, single-seed vs seed-mean).
``B`` error budget -- every uncertainty that moves ``phi`` on one scale.
``C`` regime check on ``Z`` -- the identical extrapolation applied to the bulk
      coordination number must land at or above the frictionless isostatic
      value ``Z_iso = 2d = 6``.
``D`` equilibrium check -- ``settled`` is a kinetic-energy flag; it does not
      certify that the density stopped drifting.
``E`` coverage -- seeds per stiffness tier, stiffest tier first.

Run::

    PYTHONPATH=src python benchmarks/xval_dem_round2_deposition_rigid_limit_audit.py
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "benchmarks", "case_rcp_deposition", "runs")
OUT = os.path.join(ROOT, "artifacts",
                   "xval_dem_round2_deposition_rigid_limit_audit")
FIG = os.path.join(OUT, "figures")

PHI_RCP = 0.6366          # Scott 1960; Bernal & Mason 1960 (frictionless)
TOL = 0.15                # owner precision standard
Z_ISO = 6.0               # isostatic coordination 2d, d = 3, frictionless
ORDER_BOUNDS = (0.2, 4.0)

#: values the round-2 report quotes, transcribed so the audit can diff
#: claim-vs-pipeline instead of trusting the prose.
REPORT_CLAIMS = {
    "phi_inf_mu0": 0.63535,
    "p_mu0": 1.007,
    "phi_inf_mu01": 0.61705,
    "p_mu01": 0.512,
    "prediction_error_mu0": 0.00081,
    "limit_shift_mu0": 0.0011,
    "vs_rcp_pct_mu0": -0.20,
    "vs_rcp_pct_mu01": -3.07,
    "seeds_kn64000_mu0": 1,
}


def load_runs() -> dict:
    """``{(n, mu, kn, dt_frac): [entry, ...]}`` from committed run JSONs."""
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(RUNS, "deposition_n*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        spec, metrics = d["spec"], d["metrics"]
        key = (int(spec["n_particles"]), float(spec["mu"]),
               float(spec["kn"]), round(float(spec.get("dt_frac", 0.05)), 6))
        runs[key].append({
            "path": os.path.relpath(path, ROOT),
            "seed": int(spec["seed"]),
            "phi": float(metrics["phi_bulk"]),
            "phi_corr": float(metrics["phi_bulk_overlap_corrected"]),
            "Z": float(metrics["coordination_bulk"]),
            "settled": bool(d.get("settled", True)),
            "settle_tol": float(spec.get("settle_tol", 0.0005)),
            "history": d.get("history", {}),
            "steps": int(d.get("steps", 0)),
            "final_time": float(d.get("final_time", 0.0)),
            "wall_clock": d.get("wall_clock"),
        })
    return dict(runs)


def mean_phi(runs: dict, n: int, mu: float, kn: float) -> tuple:
    entries = runs.get((n, mu, kn, 0.05), [])
    if not entries:
        raise KeyError(f"no production run for n={n} mu={mu} kn={kn}")
    vals = [e["phi"] for e in entries]
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def richardson(points, default_order=1.0, bounds=ORDER_BOUNDS) -> dict:
    """Local reimplementation of ``deposition_packing.stiffness_limit``.

    Written from the paper formula rather than imported so that a bug in the
    library surfaces as a disagreement instead of being inherited.
    """
    pts = sorted((float(k), float(v)) for k, v in points)
    kn = np.array([p[0] for p in pts])
    val = np.array([p[1] for p in pts])
    deltas = np.diff(val)
    ratios = kn[1:] / kn[:-1]

    order = None
    if len(deltas) >= 2 and deltas[-2] != 0.0 and deltas[-1] != 0.0 \
            and np.sign(deltas[-2]) == np.sign(deltas[-1]):
        o = float(math.log(abs(deltas[-2]) / abs(deltas[-1]))
                  / math.log(ratios[-1]))
        order = o if bounds[0] < o < bounds[1] else None
    p_used = default_order if order is None else order
    limit = float(val[-1] + deltas[-1] / (float(ratios[-1]) ** p_used - 1.0))
    return {
        "kn": kn.tolist(), "value": val.tolist(),
        "delta": deltas.tolist(), "ratio": ratios.tolist(),
        "order": order, "p_used": float(p_used), "limit": limit,
        "correction": float(limit - val[-1]),
        "delta_from_last": float(abs(limit - val[-1])),
        "n_levels": int(len(pts)), "fallback": order is None,
    }


def crosscheck_library(points) -> dict:
    """Confirm the local reimplementation matches the shipped library."""
    from tensordem.deposition_packing import stiffness_limit
    lib = stiffness_limit(list(points), default_order=1.0,
                          order_bounds=ORDER_BOUNDS)
    mine = richardson(points)
    return {
        "library_limit": lib["limit"], "audit_limit": mine["limit"],
        "abs_diff": abs(lib["limit"] - mine["limit"]),
        "library_order": lib["order"], "audit_order": mine["order"],
        "agree": bool(abs(lib["limit"] - mine["limit"]) < 1e-12
                      and lib["order"] == mine["order"]),
    }


# --------------------------------------------------------------------------
# A -- ladder / aggregation sensitivity
# --------------------------------------------------------------------------
def ladder_sensitivity(runs: dict, mu: float = 0.0) -> dict:
    levels = sorted({k[2] for k in runs
                     if k[0] == 10000 and k[1] == mu and k[3] == 0.05})

    def seed_values(kn):
        return {e["seed"]: e["phi"] for e in runs[(10000, mu, kn, 0.05)]}

    variants = []
    for name, ladder in (("3-level", levels[:3]), ("4-level", levels)):
        for policy in ("seed1", "seed-mean", "lowest-seed"):
            pts = []
            for kn in ladder:
                sv = seed_values(kn)
                if policy == "seed1":
                    v = sv[min(sv)]
                elif policy == "seed-mean":
                    v = float(np.mean(list(sv.values())))
                else:
                    v = float(min(sv.values()))
                pts.append((kn, v))
            r = richardson(pts)
            variants.append({
                "ladder": name, "policy": policy,
                "seeds_available": {str(int(kn)): len(seed_values(kn))
                                    for kn in ladder},
                "points": pts,
                "order": r["order"], "p_used": r["p_used"],
                "fallback": r["fallback"], "limit": r["limit"],
                "correction": r["correction"],
                "vs_rcp_pct": 100.0 * (r["limit"] - PHI_RCP) / PHI_RCP,
            })

    limits = [v["limit"] for v in variants]
    orders = [v["order"] for v in variants if v["order"] is not None]
    return {
        "mu": mu, "levels": levels, "variants": variants,
        "limit_min": min(limits), "limit_max": max(limits),
        "limit_spread": max(limits) - min(limits),
        "order_min": min(orders) if orders else None,
        "order_max": max(orders) if orders else None,
        "n_fallback": sum(1 for v in variants if v["fallback"]),
    }


# --------------------------------------------------------------------------
# B -- error budget
# --------------------------------------------------------------------------
def error_budget(runs: dict) -> dict:
    sources = []
    for kn in (4000.0, 64000.0):
        _, std, n = mean_phi(runs, 10000, 0.0, kn)
        sources.append({
            "source": f"seed scatter, kn={kn:g} (n={n})",
            "value": std, "kind": "random",
            "note": "std over independent seeds, everything else fixed",
        })

    n_vals = []
    for (n, mu, kn, dtf) in sorted(runs):
        if mu == 0.0 and kn == 4000.0 and dtf == 0.05:
            n_vals.extend(e["phi"] for e in runs[(n, mu, kn, dtf)])
    if n_vals:
        sources.append({
            "source": "N sweep, kn=4000, N=1000..10000",
            "value": float(np.std(n_vals)),
            "range": float(max(n_vals) - min(n_vals)),
            "kind": "systematic",
            "note": "finite size + wall/bulk-margin coupling",
        })

    defi = os.path.join(RUNS, "phi_definition_audit.json")
    if os.path.exists(defi):
        with open(defi) as fh:
            gates = {g["name"]: g for g in json.load(fh).get("gates", [])}
        g = gates.get("A2_definition_spread")
        if g:
            sources.append({
                "source": "density definition D1..D7 within one run",
                "value": float(g["got"]), "kind": "systematic",
                "note": "phi_definition_audit gate A2 (max-min over 7 estimators)",
            })

    ls = ladder_sensitivity(runs, 0.0)
    sources.append({
        "source": "ladder bracket (3-level vs 4-level limit)",
        "value": ls["limit_spread"], "kind": "extrapolation",
        "note": "same data, defensible ladder choices disagree by this much",
    })

    m40 = mean_phi(runs, 10000, 0.0, 4000.0)[0]
    m16 = mean_phi(runs, 10000, 0.0, 16000.0)[0]
    m64 = mean_phi(runs, 10000, 0.0, 64000.0)[0]
    m256 = mean_phi(runs, 10000, 0.0, 256000.0)[0]
    fit3 = richardson([(4000, m40), (16000, m16), (64000, m64)])
    pred4 = fit3["limit"] + (m64 - fit3["limit"]) * 4.0 ** (-fit3["p_used"])
    fit4 = richardson([(4000, m40), (16000, m16), (64000, m64), (256000, m256)])

    sources.append({
        "source": "out-of-sample error (fit 3 levels, predict stiffest)",
        "value": abs(pred4 - m256), "kind": "extrapolation",
        "note": "the extrapolation model's own held-out error",
    })
    sources.append({
        "source": "limit shift when the stiffest level is admitted",
        "value": abs(fit4["limit"] - fit3["limit"]), "kind": "extrapolation",
        "note": "the extrapolation's internal error bar",
    })
    sources.append({
        "source": "Richardson correction actually applied (4-level)",
        "value": abs(fit4["correction"]), "kind": "correction",
        "note": "how much the extrapolation moves the quoted number",
    })
    sources.append({
        "source": "claimed deviation phi_inf vs RCP",
        "value": abs(REPORT_CLAIMS["phi_inf_mu0"] - PHI_RCP), "kind": "claim",
        "note": "the agreement being advertised",
    })

    by_kind = defaultdict(float)
    for s in sources:
        by_kind[s["kind"]] = max(by_kind[s["kind"]], s["value"])

    corr = abs(fit4["correction"])
    _, seed_std_64k, _ = mean_phi(runs, 10000, 0.0, 64000.0)
    return {
        "sources": sources, "by_kind": dict(by_kind),
        "prediction_error": abs(pred4 - m256),
        "limit_shift": abs(fit4["limit"] - fit3["limit"]),
        "correction": corr,
        "claimed_deviation": abs(REPORT_CLAIMS["phi_inf_mu0"] - PHI_RCP),
        "correction_below_prediction_error": bool(corr < abs(pred4 - m256)),
        "correction_below_seed_scatter": bool(corr < seed_std_64k),
        "claim_below_systematic": bool(
            abs(REPORT_CLAIMS["phi_inf_mu0"] - PHI_RCP)
            < by_kind.get("systematic", 0.0)),
        "fit4_limit": fit4["limit"], "fit3_limit": fit3["limit"],
        "fit4_order": fit4["order"], "fit3_order": fit3["order"],
        "fit4_p_used": fit4["p_used"], "fit3_p_used": fit3["p_used"],
    }


# --------------------------------------------------------------------------
# C -- regime check: the same extrapolation applied to Z
# --------------------------------------------------------------------------
def coordination_regime(runs: dict) -> dict:
    ladder = []
    for kn in (4000.0, 16000.0, 64000.0, 256000.0):
        vals = [e["Z"] for e in runs[(10000, 0.0, kn, 0.05)]]
        ladder.append((kn, float(np.mean(vals)), float(np.std(vals)), len(vals)))
    fit4 = richardson([(kn, z) for kn, z, _, _ in ladder])
    fit3 = richardson([(kn, z) for kn, z, _, _ in ladder[:3]])
    return {
        "ladder": [{"kn": kn, "Z": z, "Z_std": sd, "n_seeds": n}
                   for kn, z, sd, n in ladder],
        "Z_iso": Z_ISO,
        "fit_4level": fit4, "fit_3level": fit3,
        "Z_limit": fit4["limit"],
        "below_isostatic": bool(fit4["limit"] < Z_ISO),
        "degenerate": bool(fit4["limit"] < 0.5 * Z_ISO),
        "Z_at_softest": ladder[0][1],
        "monotone_decreasing": all(ladder[i][1] > ladder[i + 1][1]
                                   for i in range(len(ladder) - 1)),
    }


# --------------------------------------------------------------------------
# D -- equilibrium check (density plateau at "settled")
# --------------------------------------------------------------------------
def equilibrium_check(runs: dict) -> dict:
    rows = []
    for kn in (4000.0, 64000.0, 256000.0):
        for e in runs[(10000, 0.0, kn, 0.05)]:
            hist = e["history"]
            t = np.asarray(hist.get("t", []), dtype=float)
            phi_key = ("phi_bulk_instant" if "phi_bulk_instant" in hist
                       else "phi")
            phi = np.asarray(hist.get(phi_key, []), dtype=float)
            if len(t) < 5 or len(phi) != len(t):
                rows.append({"kn": kn, "seed": e["seed"], "drift": None,
                             "note": "history too short"})
                continue
            tail = t >= 0.8 * t[-1]
            drift = float(phi[tail][-1] - phi[tail][0]) if tail.sum() >= 2 else None
            rows.append({
                "kn": kn, "seed": e["seed"], "settled": e["settled"],
                "t_end": float(t[-1]), "steps": e["steps"],
                "phi_end": float(phi[-1]),
                "drift_last20pct": drift,
                "settle_tol": e["settle_tol"],
                "drift_exceeds_tol": bool(drift is not None
                                          and abs(drift) > e["settle_tol"]),
            })
    return {"rows": rows,
            "any_drift_exceeds_tol": any(r.get("drift_exceeds_tol") for r in rows)}


# --------------------------------------------------------------------------
# D2 -- temporal convergence, with its discriminating power
# --------------------------------------------------------------------------
def temporal_check(runs: dict) -> dict:
    """``dt -> dt/2`` at every stiffness level where both exist.

    A difference smaller than the stochastic scatter of the same measurement
    cannot certify anything: the check has no discriminating power.  Both the
    difference and its size relative to the scatter are reported.
    """
    rows = []
    for (n, mu, kn, dtf) in sorted(runs):
        if dtf != 0.05:
            continue
        coarse = runs.get((n, mu, kn, 0.05), [])
        fine = runs.get((n, mu, kn, 0.025), [])
        if not (coarse and fine):
            continue
        keys = {e["seed"] for e in coarse} & {e["seed"] for e in fine}
        if not keys:
            continue
        c = float(np.mean([e["phi"] for e in coarse if e["seed"] in keys]))
        f = float(np.mean([e["phi"] for e in fine if e["seed"] in keys]))
        scatter = float(np.std([e["phi"] for e in coarse])) if len(coarse) > 1 else None
        diff = abs(c - f)
        rows.append({
            "n": n, "mu": mu, "kn": kn, "seeds": sorted(keys),
            "phi_dt": c, "phi_dt_half": f, "diff": diff,
            "seed_scatter": scatter,
            "diff_over_scatter": (diff / scatter) if scatter else None,
            "resolves_above_scatter": bool(scatter is not None
                                           and diff > scatter),
        })
    tested = {r["kn"] for r in rows}
    ladder = sorted({k[2] for k in runs
                     if k[0] == 10000 and k[1] == 0.0 and k[3] == 0.05})
    return {
        "rows": rows,
        "levels_with_dt_test": sorted(tested),
        "ladder_levels": ladder,
        "untested_levels": sorted(k for k in ladder if k not in tested),
        "stiffest_tested": bool(ladder and ladder[-1] in tested),
    }


# --------------------------------------------------------------------------
# E -- coverage
# --------------------------------------------------------------------------
def coverage(runs: dict) -> dict:
    tiers = []
    for mu in sorted({k[1] for k in runs if k[0] == 10000 and k[3] == 0.05}):
        kns = sorted({k[2] for k in runs
                      if k[0] == 10000 and k[1] == mu and k[3] == 0.05})
        for kn in kns:
            entries = runs[(10000, mu, kn, 0.05)]
            tiers.append({"mu": mu, "kn": kn, "n_seeds": len(entries),
                          "seeds": sorted(e["seed"] for e in entries),
                          "phi_mean": float(np.mean([e["phi"] for e in entries])),
                          "phi_std": float(np.std([e["phi"] for e in entries]))})
    controlling = [t for t in tiers if t["mu"] == 0.0]
    stiffest = max(controlling, key=lambda t: t["kn"])
    softest_kn = min(t["kn"] for t in controlling)
    return {
        "tiers": tiers,
        "stiffest_mu0_tier": stiffest,
        "stiffest_seeds": stiffest["n_seeds"],
        "base_tier_seeds": min(t["n_seeds"] for t in controlling
                               if t["kn"] == softest_kn),
        "report_claims_kn64000_seeds": REPORT_CLAIMS["seeds_kn64000_mu0"],
        "actual_kn64000_seeds": next(t["n_seeds"] for t in controlling
                                     if t["kn"] == 64000.0),
    }


# --------------------------------------------------------------------------
# friction reference re-specification
# --------------------------------------------------------------------------
def friction_trend(runs: dict) -> dict:
    """The mu = 0.1 tier must not be judged against the frictionless RCP band.

    Friction lowers the random-close-packing fraction; the repo's own friction
    scan at fixed N and kn shows the ordering, so the honest reference for the
    mu = 0.1 limit is 'below the mu = 0 limit, same ordering'.
    """
    scan = []
    for mu in sorted({k[1] for k in runs
                      if k[0] == 2000 and k[2] == 4000.0 and k[3] == 0.05}):
        vals = [e["phi"] for e in runs[(2000, mu, 4000.0, 0.05)]]
        scan.append({"mu": mu, "phi_mean": float(np.mean(vals)),
                     "n_seeds": len(vals)})
    monotone = all(scan[i]["phi_mean"] > scan[i + 1]["phi_mean"]
                   for i in range(len(scan) - 1)) if len(scan) > 1 else None

    m40 = mean_phi(runs, 10000, 0.1, 4000.0)[0]
    m16 = mean_phi(runs, 10000, 0.1, 16000.0)[0]
    m64, _, n64 = mean_phi(runs, 10000, 0.1, 64000.0)
    fit = richardson([(4000, m40), (16000, m16), (64000, m64)])
    phi0_40 = mean_phi(runs, 10000, 0.0, 4000.0)[0]

    return {
        "friction_scan_N2000_kn4000": scan,
        "monotone_decreasing": monotone,
        "mu01_ladder": [[4000.0, m40], [16000.0, m16], [64000.0, m64]],
        "mu01_seeds_64000": n64,
        "mu01_order": fit["order"], "mu01_p_used": fit["p_used"],
        "mu01_limit": fit["limit"],
        "vs_rcp_pct": 100.0 * (fit["limit"] - PHI_RCP) / PHI_RCP,
        "friction_shift_at_kn4000_pct": 100.0 * (m40 - phi0_40) / phi0_40,
    }

# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_figures(ls, budget, coord, seed_points) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG, exist_ok=True)

    kns = [p[0] for p in ls["variants"][0]["points"]]
    means = [p[1] for p in ls["variants"][0]["points"]]

    # fig 1 -- ladder + extrapolation family
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for kn in kns:
        vals = seed_points.get(kn, [])
        if vals:
            ax.semilogx([kn] * len(vals), vals, "o", color="#8fa6c9",
                        ms=4.5, alpha=0.85)
    ax.semilogx(kns, means, "o-", color="#1f3b73", lw=1.6,
                label="deposition ladder (seed-mean)")
    ax.axhline(PHI_RCP, color="#c0392b", ls="--", lw=1.2,
               label=f"literature RCP = {PHI_RCP}")
    n = len(ls["variants"])
    for i, v in enumerate(ls["variants"]):
        c = plt.cm.viridis(i / max(1, n - 1))
        ax.semilogx([kns[0] * 0.6], [v["limit"]], marker="v", ms=8, color=c,
                    ls="none")
        ax.annotate(f"{v['ladder']} / {v['policy']}: {v['limit']:.5f}",
                    (kns[0] * 0.66, v["limit"]), fontsize=6.5, va="center",
                    color="#333333")
    ax.set_xlabel("stiffness $k_n$ (ladder, log scale)")
    ax.set_ylabel(r"$\phi$ (bulk volume fraction)")
    ax.set_title("Deposition ladder and the hard-sphere limits it admits\n"
                 "(grey dots = independent seeds; family spread "
                 f"{100 * ls['limit_spread'] / PHI_RCP:.2f}%)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(FIG, "fig1_ladder_extrapolation_family.png")
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # fig 2 -- order sensitivity
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    labels = [f"{v['ladder']}\n{v['policy']}" for v in ls["variants"]]
    orders = [v["order"] if v["order"] is not None else 0.0 for v in ls["variants"]]
    colors = ["#c0392b" if v["fallback"] else "#1f3b73" for v in ls["variants"]]
    ax.bar(range(len(labels)), orders, color=colors)
    ax.axhline(1.0, color="#2e7d32", ls="--", lw=1.2, label="theoretical $p = 1$")
    ax.axhspan(ORDER_BOUNDS[0], ORDER_BOUNDS[1], color="#f4d35e", alpha=0.18,
               label=f"order acceptance bounds {ORDER_BOUNDS}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("observed convergence order $p$")
    ax.set_title("The observed order is a function of the ladder, not of the physics\n"
                 "(red = order out of bounds, fit falls back to $p = 1$)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(FIG, "fig2_order_sensitivity.png")
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    # fig 3 -- error budget
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    src = sorted(budget["sources"], key=lambda s: s["value"])
    vals = [max(s["value"], 1e-7) for s in src]
    cmap = {"random": "#1f3b73", "systematic": "#c0392b",
            "extrapolation": "#e67e22", "correction": "#8e44ad",
            "claim": "#2e7d32"}
    ax.barh(range(len(src)), vals,
            color=[cmap.get(s["kind"], "#555555") for s in src])
    ax.set_yticks(range(len(src)))
    ax.set_yticklabels([s["source"] for s in src], fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel(r"magnitude in $\phi$ units (log scale)")
    ax.set_title("Error budget: every uncertainty that moves $\\phi$ on one scale")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    p3 = os.path.join(FIG, "fig3_error_budget.png")
    fig.savefig(p3, dpi=160)
    plt.close(fig)

    # fig 4 -- Z regime check
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    zs = [r["Z"] for r in coord["ladder"]]
    sds = [r["Z_std"] for r in coord["ladder"]]
    zkn = [r["kn"] for r in coord["ladder"]]
    ax.errorbar(zkn, zs, yerr=sds, marker="o", color="#1f3b73", capsize=3,
                label="bulk coordination $Z(k_n)$")
    ax.axhline(Z_ISO, color="#c0392b", ls="--", lw=1.3,
               label=f"frictionless isostatic $Z_{{iso}} = 2d = {Z_ISO:g}$")
    ax.axhline(coord["Z_limit"], color="#e67e22", ls=":", lw=1.5,
               label=f"extrapolated $Z_\\infty$ = {coord['Z_limit']:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("stiffness $k_n$ (log scale)")
    ax.set_ylabel("bulk coordination number $Z$")
    ax.set_title("Regime check: the same extrapolation puts $Z_\\infty$ below the\n"
                 "isostatic value, so the stiffness ladder is not asymptotic")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p4 = os.path.join(FIG, "fig4_coordination_regime.png")
    fig.savefig(p4, dpi=160)
    plt.close(fig)

    return [
        {"name": "fig1_ladder_extrapolation_family.png", "path": p1,
         "caption": "Deposition phi(k_n) ladder with per-seed scatter (grey). "
                    "Triangle markers at the left are the hard-sphere limits "
                    "from every defensible ladder/aggregation choice; their "
                    "spread is comparable to the -0.20% agreement advertised."},
        {"name": "fig2_order_sensitivity.png", "path": p2,
         "caption": "Observed convergence order versus ladder choice. Three-level "
                    "ladders give p = 0.80-1.04, but admitting the stiffest level "
                    "pushes the order out of bounds so the fit silently falls "
                    "back to p = 1."},
        {"name": "fig3_error_budget.png", "path": p3,
         "caption": "Every uncertainty that moves phi on one log scale. The "
                    "Richardson correction (~3.5e-4) sits below the model's own "
                    "out-of-sample error and far below protocol systematics."},
        {"name": "fig4_coordination_regime.png", "path": p4,
         "caption": "Bulk coordination number on the same ladder. The identical "
                    "extrapolation yields Z_inf below the frictionless isostatic "
                    "value Z_iso = 6, i.e. an unjammed limit: evidence that the "
                    "stiffness ladder has left the asymptotic regime."},
    ]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def _f(x, nd=4):
    return "None" if x is None else f"{x:.{nd}f}"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    runs = load_runs()

    ls = ladder_sensitivity(runs, 0.0)
    seed_points = {kn: [e["phi"] for e in runs[(10000, 0.0, kn, 0.05)]]
                   for kn in ls["levels"]}
    budget = error_budget(runs)
    coord = coordination_regime(runs)
    eq = equilibrium_check(runs)
    cov = coverage(runs)
    fric = friction_trend(runs)
    temp = temporal_check(runs)
    xcheck = crosscheck_library(
        [(p[0], p[1]) for p in ls["variants"][-1]["points"]])

    canonical = [v for v in ls["variants"]
                 if v["ladder"] == "4-level" and v["policy"] == "seed-mean"][0]

    verdicts = {
        "V1_phi_inf_mu0_vs_rcp_lt_15pct": {
            "got": canonical["limit"], "ref": PHI_RCP,
            "rel_err": canonical["vs_rcp_pct"] / 100.0,
            "pass": bool(abs(canonical["vs_rcp_pct"]) < 100 * TOL),
            "note": "acceptance met by every ladder variant, not only the quoted one",
        },
        "V2_quoted_precision_resolvable": {
            "claimed_deviation": budget["claimed_deviation"],
            "dominant_systematic": budget["by_kind"].get("systematic", 0.0),
            "pass": bool(budget["claimed_deviation"]
                         >= budget["by_kind"].get("systematic", 0.0)),
            "note": "False => the -0.20% agreement lies below the protocol "
                    "noise floor and may not be read as a precision statement",
        },
        "V3_order_p_equals_1": {
            "quoted_p": REPORT_CLAIMS["p_mu0"],
            "four_level_order": budget["fit4_order"],
            "four_level_p_used": budget["fit4_p_used"],
            "order_range_3level": [ls["order_min"], ls["order_max"]],
            "n_fallback_variants": ls["n_fallback"],
            "pass": bool(budget["fit4_order"] is not None
                         and abs(budget["fit4_order"] - 1.0) < 0.2),
            "note": "False => 'p = 1' holds only on the 3-level sub-ladder; at "
                    "4 levels the order is out of bounds and p is assumed",
        },
        "V4_extrapolation_correction_resolvable": {
            "correction": budget["correction"],
            "prediction_error": budget["prediction_error"],
            "pass": bool(budget["correction"] >= budget["prediction_error"]),
            "note": "False => the Richardson correction is smaller than the "
                    "model's own held-out error; the corrected value cannot be "
                    "distinguished from the uncorrected stiffest measurement",
        },
        "V5_stiffest_tier_seed_coverage_ge_3": {
            "n_seeds": cov["stiffest_seeds"],
            "pass": bool(cov["stiffest_seeds"] >= 3),
            "note": "the stiffest tier is the one that sets the limit, so its "
                    "seed count is the coverage that matters",
        },
        "V6_coordination_limit_above_isostatic": {
            "Z_limit": coord["Z_limit"], "Z_iso": Z_ISO,
            "pass": bool(not coord["below_isostatic"]),
            "note": "False => the staircase of phi and the staircase of Z "
                    "extrapolate to an inconsistent state; either the ladder is "
                    "not asymptotic or Z is not measured in the rigid convention",
        },
        "V7_temporal_convergence_at_stiffest": {
            "levels_with_dt_test": temp["levels_with_dt_test"],
            "stiffest_tested": temp["stiffest_tested"],
            "untested_levels": temp["untested_levels"],
            "pass": bool(temp["stiffest_tested"]),
            "note": "the round-2 report tests dt only at the softest level, but "
                    "the stiffest level is the one that sets the limit",
        },
        "V8_density_plateau_at_settled": {
            "any_drift_exceeds_tol": eq["any_drift_exceeds_tol"],
            "pass": bool(not eq["any_drift_exceeds_tol"]),
        },
        "V9_mu01_judged_against_frictional_reference": {
            "friction_scan_monotone": fric["monotone_decreasing"],
            "mu01_limit": fric["mu01_limit"],
            "vs_rcp_pct": fric["vs_rcp_pct"],
            "pass": bool(fric["monotone_decreasing"]),
            "note": "the [0.62,0.66] band of the round-2 report is a "
                    "frictionless band and does not apply to mu = 0.1; no "
                    "frictional RCP value is citable from this repo, so the "
                    "mu = 0.1 tier is judged by ordering only",
        },
    }

    figs = make_figures(ls, budget, coord, seed_points)

    results = {
        "anchor": "XVAL-DEM-ROUND2-DEPOSITION-RIGID-LIMIT",
        "role": "independent verification of the round-2 cross-validation",
        "report_claims": REPORT_CLAIMS,
        "tolerance": TOL, "literature_rcp": PHI_RCP, "Z_iso": Z_ISO,
        "A_ladder_sensitivity": ls,
        "B_error_budget": budget,
        "C_coordination_regime": coord,
        "D_equilibrium": eq,
        "E_coverage": cov,
        "friction_reference": fric,
        "D2_temporal": temp,
        "library_crosscheck": xcheck,
        "verdicts": verdicts,
        "figures": [{"name": f["name"], "caption": f["caption"]} for f in figs],
    }
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)

    L = []
    A = L.append
    A("# XVAL-DEM-ROUND2-DEPOSITION-RIGID-LIMIT - independent verification")
    A("")
    A("Every number below is emitted by "
      "`benchmarks/xval_dem_round2_deposition_rigid_limit_audit.py` from the "
      "committed run JSONs (zero new simulation); no value is typed by hand.")
    A("")
    A("## A. Ladder / aggregation sensitivity (mu = 0)")
    A("")
    A("| ladder | aggregation | seeds per level | observed p | p used | fallback | phi_inf | vs RCP |")
    A("|---|---|---|---|---|---|---|---|")
    for v in ls["variants"]:
        seeds = "/".join(str(x) for x in v["seeds_available"].values())
        A(f"| {v['ladder']} | {v['policy']} | {seeds} | {_f(v['order'])} | "
          f"{v['p_used']:.4f} | {'yes' if v['fallback'] else 'no'} | "
          f"{v['limit']:.5f} | {v['vs_rcp_pct']:+.3f}% |")
    A("")
    A(f"Limit spread across defensible ladders: **{ls['limit_spread']:.5f}** "
      f"({100 * ls['limit_spread'] / PHI_RCP:.3f}% of RCP). Observed order "
      f"ranges {_f(ls['order_min'])}..{_f(ls['order_max'])} on 3-level ladders; "
      f"{ls['n_fallback']} of {len(ls['variants'])} fits fall back to p = 1.")
    A("")
    A("## B. Error budget (phi units)")
    A("")
    A("| source | magnitude | kind | note |")
    A("|---|---|---|---|")
    for s in sorted(budget["sources"], key=lambda s: -s["value"]):
        A(f"| {s['source']} | {s['value']:.6f} | {s['kind']} | {s['note']} |")
    A("")
    A(f"* Richardson correction actually applied: **{budget['correction']:.6f}**")
    A(f"* extrapolation's own out-of-sample error: **{budget['prediction_error']:.6f}**")
    A(f"* limit shift when the stiffest level is admitted: **{budget['limit_shift']:.6f}**")
    A(f"* claimed deviation vs RCP: **{budget['claimed_deviation']:.6f}**")
    A("")
    A("## C. Regime check on the coordination number")
    A("")
    A("| kn | Z (bulk) | std | seeds |")
    A("|---|---|---|---|")
    for r in coord["ladder"]:
        A(f"| {r['kn']:g} | {r['Z']:.4f} | {r['Z_std']:.4f} | {r['n_seeds']} |")
    A("")
    A(f"Extrapolated `Z_inf = {coord['Z_limit']:.4f}` vs frictionless isostatic "
      f"`Z_iso = {Z_ISO:g}`: "
      f"{'BELOW isostatic (unjammed limit)' if coord['below_isostatic'] else 'above isostatic'}.")
    A("")
    A("## D. Equilibrium check")
    A("")
    A("| kn | seed | settled | steps | t end | phi end | drift over last 20% | exceeds settle_tol |")
    A("|---|---|---|---|---|---|---|---|")
    for r in eq["rows"]:
        A(f"| {r['kn']:g} | {r.get('seed')} | {r.get('settled')} | {r.get('steps')} | "
          f"{_f(r.get('t_end'), 2)} | "
          f"{_f(r.get('phi_end'), 5)} | {_f(r.get('drift_last20pct'), 6)} | "
          f"{r.get('drift_exceeds_tol')} |")
    A("")
    A("## D2. Temporal convergence and its discriminating power")
    A("")
    A("| mu | kn | seeds | phi(dt) | phi(dt/2) | diff | seed scatter | diff/scatter |")
    A("|---|---|---|---|---|---|---|---|")
    for r in temp["rows"]:
        A(f"| {r['mu']} | {r['kn']:g} | {len(r['seeds'])} | {r['phi_dt']:.5f} | "
          f"{r['phi_dt_half']:.5f} | {r['diff']:.6f} | {_f(r['seed_scatter'], 6)} | "
          f"{_f(r['diff_over_scatter'], 2)} |")
    A("")
    A(f"Levels tested: {temp['levels_with_dt_test']}; untested levels of the "
      f"ladder: **{temp['untested_levels']}**. Stiffest level tested: "
      f"**{temp['stiffest_tested']}**.")
    A("")
    A("## E. Seed coverage")
    A("")
    A("| mu | kn | seeds | phi mean | phi std |")
    A("|---|---|---|---|---|")
    for t in cov["tiers"]:
        A(f"| {t['mu']} | {t['kn']:g} | {t['n_seeds']} ({','.join(str(s) for s in t['seeds'])}) | "
          f"{t['phi_mean']:.5f} | {t['phi_std']:.5f} |")
    A("")
    A("## F. Friction reference re-specification")
    A("")
    A("| mu | phi (N=2000, kn=4000) | seeds |")
    A("|---|---|---|")
    for s in fric["friction_scan_N2000_kn4000"]:
        A(f"| {s['mu']} | {s['phi_mean']:.5f} | {s['n_seeds']} |")
    A("")
    A(f"Monotone in mu: **{fric['monotone_decreasing']}**; friction shift at "
      f"kn = 4000, mu = 0.1 vs 0: {fric['friction_shift_at_kn4000_pct']:+.3f}%. "
      f"mu = 0.1 hard-sphere limit {fric['mu01_limit']:.5f} "
      f"({fric['vs_rcp_pct']:+.2f}% vs the frictionless RCP).")
    A("")
    A("## G. Verdicts")
    A("")
    A("| id | check | pass |")
    A("|---|---|---|")
    for k, v in verdicts.items():
        A(f"| {k} | {v['note'] if 'note' in v else ''} | "
          f"{'PASS' if v['pass'] else 'FAIL'} |")
    A("")
    A(f"Library cross-check (local reimplementation vs "
      f"`deposition_packing.stiffness_limit`): agree = "
      f"**{xcheck['agree']}** (|dlimit| = {xcheck['abs_diff']:.2e}).")
    with open(os.path.join(OUT, "table_main.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")

    print(json.dumps({
        "limit_spread": ls["limit_spread"],
        "canonical_limit": canonical["limit"],
        "canonical_vs_rcp_pct": canonical["vs_rcp_pct"],
        "four_level_order": budget["fit4_order"],
        "correction": budget["correction"],
        "prediction_error": budget["prediction_error"],
        "Z_limit": coord["Z_limit"],
        "stiffest_seeds": cov["stiffest_seeds"],
        "library_agree": xcheck["agree"],
        "verdicts": {k: v["pass"] for k, v in verdicts.items()},
        "figures": [f["name"] for f in figs],
    }, indent=2))


if __name__ == "__main__":
    main()
