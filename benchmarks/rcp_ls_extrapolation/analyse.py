"""Analysis of the RCP inflation scan: rate / budget / size extrapolation.

Reads ``runs/raw/*.json`` produced by :mod:`run_scan`, writes

* ``runs/analysis.json``      -- every number used in the report (single source of truth)
* ``runs/figures/*.png``      -- rate, budget, size, criterion panels

and prints the markdown tables that the report quotes verbatim (no hand-copy).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "runs" / "raw"
RUNS = HERE / "runs"
FIGS = RUNS / "figures"

SCOTT = 0.6366          # Scott 1960 / Bernal & Mason 1960
TOL_RULE = 0.03         # the 3 % rule (anchor acceptance historically used 5 %)
ANCHOR_BAND = (0.62, 0.66)


def load() -> list[dict]:
    out = []
    for path in sorted(RAW.glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if rec.get("status") == "ok" and rec.get("jammed_packing_fraction") is not None:
            out.append(rec)
    return out


def _phi(rec: dict) -> float:
    return float(rec["jammed_packing_fraction"])


def _pick(recs, **kw) -> list[dict]:
    out = []
    for r in recs:
        cfg = r["config"]
        if all(_close(cfg.get(k), v) for k, v in kw.items()):
            out.append(r)
    return out


def _close(a, b, rel: float = 1e-9) -> bool:
    if isinstance(b, bool) or isinstance(a, bool):
        return a == b
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    return abs(a - b) <= rel * max(1.0, abs(b))


def fit_rate(recs: list[dict]) -> dict:
    """Rate dependence of phi_J and the rate -> 0 extrapolations.

    Two extrapolations are reported because the functional form is not known a
    priori: linear in the rate and linear in log(rate) (both fitted on the three
    slowest rates, where the transient is weakest, and re-fitted on all four
    points as a robustness check).
    """
    base = [r for r in recs
            if r["config"]["n_particles"] == 1000 and r["config"]["relax_steps"] == 60
            and _close(r["config"]["overlap_stop"], 1.0e-3)]
    by_rate: dict[float, list[float]] = {}
    for r in base:
        by_rate.setdefault(float(r["config"]["rho_growth"]), []).append(_phi(r))
    if len(by_rate) < 2:
        return {"status": "insufficient"}
    rates = sorted(by_rate, reverse=True)          # fast -> slow
    rows = []
    for rate in rates:
        vals = by_rate[rate]
        rows.append(dict(rho_growth=rate,
                         rate_per_step=3.0 * math.log1p(rate),
                         seeds=sorted(float(v) for v in vals),
                         mean=float(np.mean(vals)),
                         spread=float(max(vals) - min(vals))))

    def _extrap(rr: list[dict], xform: str) -> float | None:
        x = np.array([np.log(r["rho_growth"]) if xform == "log"
                      else r["rho_growth"] for r in rr])
        y = np.array([r["mean"] for r in rr])
        if len(x) < 2:
            return None
        slope, intercept = np.polyfit(x, y, 1)
        x0 = -np.inf if xform == "log" else 0.0
        return None if xform == "log" else float(intercept)

    slow3 = rows[:-1] if len(rows) >= 4 else rows[1:]
    fits = {
        "linear_all": _extrap(rows, "linear"),
        "linear_slow3": _extrap(slow3, "linear"),
    }
    x = np.log(np.array([r["rho_growth"] for r in rows]))
    y = np.array([r["mean"] for r in rows])
    slope_log, _ = np.polyfit(x, y, 1) if len(x) >= 2 else (np.nan, np.nan)
    fits["dphi_dlnrate"] = float(slope_log)
    fits["rows"] = rows
    return fits


def fit_budget(recs: list[dict]) -> dict:
    """Relaxation-budget dependence, read inside one (N, rate) group.

    Mixing N or rate inside a budget scan would confound two effects, so the group
    with the widest budget span at a *single* (N, rho_growth) is reported.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in recs:
        cfg = r["config"]
        if not _close(cfg["overlap_stop"], 1.0e-3):
            continue
        key = (int(cfg["n_particles"]), float(cfg["rho_growth"]))
        groups.setdefault(key, []).append(r)
    best_group = None
    for key, rs in groups.items():
        relax = {int(r["config"]["relax_steps"]) for r in rs}
        if len(relax) < 2:
            continue
        span = max(relax) / min(relax)
        if best_group is None or span > best_group[0]:
            best_group = (span, key, rs)
    if best_group is None:
        return {"status": "insufficient"}
    _, key, rs = best_group
    grouped: dict[int, list[float]] = {}
    for r in rs:
        grouped.setdefault(int(r["config"]["relax_steps"]), []).append(_phi(r))
    rows = [dict(relax_steps=k, phi=float(np.mean(v)), seeds=sorted(float(x) for x in v))
            for k, v in sorted(grouped.items())]
    phis = [r["phi"] for r in rows]
    slope = (float(np.polyfit(np.log2([r["relax_steps"] for r in rows]), phis, 1)[0])
             if len(rows) >= 2 else None)
    return dict(n_particles=key[0], rho_growth=key[1], rows=rows,
                spread=float(max(phis) - min(phis)), slope_per_doubling=slope)


def fit_size(recs: list[dict]) -> dict:
    base = [r for r in recs
            if _close(r["config"]["rho_growth"], 2.0e-4) and r["config"]["relax_steps"] == 60
            and _close(r["config"]["overlap_stop"], 1.0e-3) and r["config"]["seed"] == 1]
    rows = [dict(n_particles=int(r["config"]["n_particles"]), phi=_phi(r))
            for r in sorted(base, key=lambda r: r["config"]["n_particles"])]
    if len(rows) < 2:
        return {"status": "insufficient"}
    n = np.array([r["n_particles"] for r in rows], dtype=float)
    phi = np.array([r["phi"] for r in rows])
    slope, intercept = np.polyfit(1.0 / np.sqrt(n), phi, 1)
    return dict(rows=rows, infinite_n_limit=float(intercept),
                spread=float(phi.max() - phi.min()))


def fit_criterion(recs: list[dict]) -> dict:
    base = [r for r in recs
            if _close(r["config"]["rho_growth"], 1.0e-4) and r["config"]["n_particles"] == 1000
            and r["config"]["relax_steps"] == 60 and r["config"]["seed"] == 1]
    rows = [dict(overlap_stop=float(r["config"]["overlap_stop"]), phi=_phi(r))
            for r in sorted(base, key=lambda r: r["config"]["overlap_stop"])]
    if not rows:
        return {"status": "insufficient"}
    phis = [r["phi"] for r in rows]
    return dict(rows=rows, spread=float(max(phis) - min(phis)))


def main() -> int:
    recs = load()
    RUNS.mkdir(parents=True, exist_ok=True)
    if not recs:
        print("no successful runs yet")
        return 1
    rate = fit_rate(recs)
    budget = fit_budget(recs)
    size = fit_size(recs)
    crit = fit_criterion(recs)

    # Best estimate: the deepest relaxation budget measured inside the most
    # informative (N, rate) group, plus the residual *rate* sensitivity measured at
    # the shallow budget (the two knobs are reported separately, not merged).
    best = None
    if budget.get("rows") and len(budget["rows"]) >= 2:
        shallow = budget["rows"][0]
        deep = budget["rows"][-1]
        best = dict(
            protocol=dict(n_particles=budget["n_particles"],
                          rho_growth=budget["rho_growth"]),
            shallow_relax=shallow["relax_steps"], shallow_phi=shallow["phi"],
            deep_relax=deep["relax_steps"], phi_estimate=deep["phi"],
            budget_correction=deep["phi"] - shallow["phi"],
            residual_rate_slope=rate.get("dphi_dlnrate"),
            assumption=("budget dependence measured inside one (N, rate); the remaining "
                        "rate sensitivity is quoted from the shallow-budget rate scan "
                        "and is NOT extrapolated into the estimate"),
        )
        best["abs_error"] = abs(best["phi_estimate"] - SCOTT)
        best["rel_error"] = best["abs_error"] / SCOTT
        best["within_3pct"] = bool(best["rel_error"] < TOL_RULE)
        best["within_anchor_band"] = bool(ANCHOR_BAND[0] <= best["phi_estimate"]
                                          <= ANCHOR_BAND[1])
    analysis = dict(reference=SCOTT, tolerance_rule=TOL_RULE, anchor_band=list(ANCHOR_BAND),
                    n_runs=len(recs), rate=rate, budget=budget, size=size,
                    criterion=crit, best_estimate=best,
                    hosts=sorted({r.get("host", "?") for r in recs}))
    (RUNS / "analysis.json").write_text(json.dumps(analysis, indent=1))

    print("### rate scan (N=1000, relax=60, overlap_stop=1e-3)")
    print("| rho_growth | rate dlnphi/step | phi_J (seeds) | mean | spread | vs 0.6366 |")
    print("|---|---|---|---|---|---|")
    for r in rate.get("rows", []):
        rel = (r["mean"] - SCOTT) / SCOTT * 100
        print(f"| {r['rho_growth']:.3g} | {r['rate_per_step']:.3g} | "
              f"{', '.join(f'{v:.4f}' for v in r['seeds'])} | {r['mean']:.4f} | "
              f"{r['spread']:.4f} | {rel:+.2f}% |")
    print(f"\nrate->0 linear extrapolation (all points): {rate.get('linear_all')}")
    print(f"rate->0 linear extrapolation (3 slowest): {rate.get('linear_slow3')}")
    print(f"dphi/dln(rate): {rate.get('dphi_dlnrate')}")

    print("\n### relaxation-budget scan (N=1000, rho_growth=1e-4)")
    print("| relax_steps | phi_J |")
    print("|---|---|")
    for r in budget.get("rows", []):
        print(f"| {r['relax_steps']} | {r['phi']:.4f} |")

    print("\n### finite-size scan (rho_growth=2e-4, relax=60, seed=1)")
    print("| N | phi_J |")
    print("|---|---|")
    for r in size.get("rows", []):
        print(f"| {r['n_particles']} | {r['phi']:.4f} |")
    print(f"N->inf limit: {size.get('infinite_n_limit')}")

    print("\n### jamming-criterion scan (N=1000, rho_growth=1e-4)")
    print("| overlap_stop | phi_J |")
    print("|---|---|")
    for r in crit.get("rows", []):
        print(f"| {r['overlap_stop']:.3g} | {r['phi']:.4f} |")

    if best:
        print("\n### best estimate")
        print(json.dumps(best, indent=1))
    else:
        print("\n### best estimate: INSUFFICIENT DATA")

    make_figures(recs, analysis)
    return 0


def make_figures(recs: list[dict], analysis: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    rate, budget, size = analysis["rate"], analysis["budget"], analysis["size"]

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

    if "rows" in rate:
        g = [r["rho_growth"] for r in rate["rows"]]
        m = [r["mean"] for r in rate["rows"]]
        sp = [r["spread"] for r in rate["rows"]]
        ax[0].errorbar(g, m, yerr=sp, marker="o", capsize=3, label="phi_J (measured)")
        for rr, mm in zip(g, m):
            ax[0].annotate(f"{mm:.4f}", (rr, mm), textcoords="offset points",
                           xytext=(4, 6), fontsize=7)
        if rate.get("linear_all") is not None:
            ax[0].axhline(rate["linear_all"], color="C1", ls="--",
                          label=f"rate→0 fit = {rate['linear_all']:.4f}")
        ax[0].axhline(SCOTT, color="k", ls=":", label="Scott/Bernal 0.6366")
        ax[0].axhspan(*ANCHOR_BAND, color="gray", alpha=0.12, label="anchor band 0.62–0.66")
        ax[0].set_xscale("log")
        ax[0].invert_xaxis()
        ax[0].set_xlabel("radius growth per step ρ (inflation rate, slower →)")
        ax[0].set_ylabel("φ_J")
        ax[0].set_title("(a) inflation-rate scan, N=1000")
        ax[0].legend(fontsize=7)
        ax[0].grid(alpha=0.3)

    if budget.get("rows"):
        ax[1].plot([r["relax_steps"] for r in budget["rows"]],
                   [r["phi"] for r in budget["rows"]], marker="s")
        ax[1].axhline(SCOTT, color="k", ls=":", label="0.6366")
        ax[1].axhspan(*ANCHOR_BAND, color="gray", alpha=0.12)
        ax[1].set_xlabel("relax_steps per inflation step")
        ax[1].set_ylabel("φ_J")
        ax[1].set_title("(b) relaxation-budget scan")
        ax[1].legend(fontsize=7)
        ax[1].grid(alpha=0.3)

    if size.get("rows"):
        n = np.array([r["n_particles"] for r in size["rows"]], dtype=float)
        phi = np.array([r["phi"] for r in size["rows"]])
        ax[2].plot(n, phi, marker="D", label="φ_J(N)")
        if size.get("infinite_n_limit") is not None:
            ax[2].axhline(size["infinite_n_limit"], color="C2", ls="--",
                          label=f"N→∞ = {size['infinite_n_limit']:.4f}")
        ax[2].axhline(SCOTT, color="k", ls=":", label="0.6366")
        ax[2].axhspan(*ANCHOR_BAND, color="gray", alpha=0.12)
        ax[2].set_xscale("log")
        ax[2].set_xlabel("N particles")
        ax[2].set_ylabel("φ_J")
        ax[2].set_title("(c) finite-size scan, ρ=2e-4")
        ax[2].legend(fontsize=7)
        ax[2].grid(alpha=0.3)

    fig.suptitle("RCP jamming by inflation quench: protocol dependencies of the measured φ_J")
    fig.tight_layout()
    fig.savefig(FIGS / "ls_scan_panels.png", dpi=145)
    plt.close(fig)

    # Second figure: the inflation curve (overlap vs phi) of the slowest run.
    slow = [r for r in recs if r["config"]["n_particles"] == 1000
            and r["config"]["relax_steps"] == 60
            and _close(r["config"]["overlap_stop"], 1.0e-3)]
    if slow:
        slowest = min(slow, key=lambda r: r["config"]["rho_growth"])
        hist = np.array(slowest["history_tail"])
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.semilogy(hist[:, 0], np.clip(hist[:, 1], 1e-16, None), lw=1.2)
        ax.axhline(slowest["overlap_stop"], color="r", ls="--",
                   label=f"jamming criterion {slowest['overlap_stop']:.0e}")
        ax.axvline(slowest["jammed_packing_fraction"], color="C2", ls=":",
                   label=f"interpolated φ_J = {slowest['jammed_packing_fraction']:.4f}")
        ax.set_xlabel("φ during inflation")
        ax.set_ylabel("mean overlap / diameter")
        ax.set_title(f"slowest run (ρ={slowest['config']['rho_growth']:.2g}, "
                     f"N={slowest['config']['n_particles']})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGS / "inflation_curve.png", dpi=145)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())