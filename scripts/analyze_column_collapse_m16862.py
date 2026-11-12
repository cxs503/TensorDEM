"""CASE-COLUMN-COLLAPSE round 4 (m16862) — joint ``(mu, mu_r)`` identifiability.

Round 3 (m16534) closed with a *conditional* structural statement: at the
round-1 rolling resistance ``mu_r = 0.4`` the linear branch wants
``mu = 0.53-0.57`` and the square-root branch wants ``mu = 0.80-0.84``, so one
contact friction cannot cover the Lube (2004) runout curve.  ``mu_r`` itself
was fitted once (round 1, at ``a = 0.5``) and never varied again.  Round 4 asks
the joint question that round 3 listed as its first follow-up:

    H4 (pre-registered in ``benchmarks/case_column_collapse_lube2004.py``):
    does ONE constant pair ``(mu, mu_r)`` reproduce the reference runout ratio
    within 15 % at every aspect ratio ``a in {0.5, 1, 1.5, 2, 3}``?

This script is the reduction of the probe grid.  It reads only the per-case
JSON written by ``run_case`` (particle counts, parameters, front time series,
energy ledger — i.e. provenance travels with the numbers) and

* interpolates, per ``(branch, mu_r)``, the *required* contact friction
  ``mu_req`` as the zero crossing of the signed runout error in ``mu``;
* reports ``gap(mu_r) = mu_req(sqrt) - mu_req(linear)`` and tests the
  pre-registered refutation rule (no gap sign change on the ``mu_r`` grid =>
  H4 refuted);
* if a crossing exists, evaluates the pre-registered pair on all five aspects
  (``wave4_valid_*`` rows; the ``a in {0.5, 1.5, 2.0}`` rows are out of sample
  for the fit) and lists the per-aspect error against the 15 % / 3 % bars.

No reference number is typed here: Lube (2004) targets are evaluated from
:func:`tensordem.granular_column.axisym.lube_2004_runout_ratio` /
``lube_2004_emplacement_time``.

Usage::

    python3 scripts/analyze_column_collapse_m16862.py
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
OUT = os.path.join(ROOT, "artifacts", "case_column_collapse_m16862")
TOL = 0.15            # pre-registered tolerance, unchanged since round 1
TIGHT = 0.03          # the platform's "acceptable" bar (< 3 %)
A_REF_SPLIT = 1.7     # branch switch of the reference law
PROBE_ASPECTS = (1.0, 3.0)
WAVE4_MU = (0.5, 0.6, 0.7, 0.8, 0.9)
WAVE4_MUR = (0.1, 0.2, 0.3, 0.4, 0.5)


# --------------------------------------------------------------------------
def _branch(a: float) -> str:
    return "linear" if a < A_REF_SPLIT else "sqrt"


def load(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(CASE_DIR, pattern))):
        with open(path) as fh:
            d = json.load(fh)
        c = d["case"]
        rows.append({
            "case": c["label"],
            "phase": d["phase"],
            "aspect": c["aspect_ratio"],
            "branch": _branch(c["aspect_ratio"]),
            "mu": c["mu"],
            "mu_rolling": c["mu_rolling"],
            "n_particles": d["n_particles"],
            "runout_ratio": d["runout_ratio_p995"],
            "runout_ratio_max": d["runout_ratio_max"],
            "ref_runout_ratio": lube_2004_runout_ratio(c["aspect_ratio"]),
            "err": d["criteria"]["runout_ratio"]["relative_error"],
            "pass15": d["criteria"]["runout_ratio"]["pass"],
            "t_reach": d["t_reach"],
            "ref_t_reach": lube_2004_emplacement_time(
                c["aspect_ratio"], c["column_radius"], c["gravity"]),
            "err_t": d["criteria"]["emplacement_time"]["relative_error"],
            "pass15_t": d["criteria"]["emplacement_time"]["pass"],
            "stop_reason": d["stop_reason"],
            "energy_residual": d["energy"]["residual_fraction_of_potential"],
            "wall_clock_s": d.get("wall_clock_s"),
        })
    return rows


# --------------------------------------------------------------------------
def zero_crossing(mus: np.ndarray, errs: np.ndarray) -> float | None:
    """Required friction: linear interpolation of the signed error through 0.

    Only *adjacent* grid pairs are used, so a non-monotone error curve returns
    the first bracketing interval rather than a spurious extrapolation.  If the
    error never changes sign on the grid the requirement is outside the grid
    box and ``None`` is returned (reported as ``not bracketed``).
    """
    order = np.argsort(mus)
    mus, errs = mus[order], errs[order]
    for k in range(len(mus) - 1):
        e0, e1 = errs[k], errs[k + 1]
        if e0 == 0.0:
            return float(mus[k])
        if e0 * e1 < 0.0:
            return float(mus[k] + (mus[k + 1] - mus[k]) * (-e0) / (e1 - e0))
    if errs[-1] == 0.0:
        return float(mus[-1])
    return None


def probe_table(rows: list[dict]) -> list[dict]:
    out = []
    for a in PROBE_ASPECTS:
        for mur in WAVE4_MUR:
            sel = [r for r in rows
                   if abs(r["aspect"] - a) < 1e-9 and abs(r["mu_rolling"] - mur) < 1e-9]
            if not sel:
                continue
            mus = np.array([r["mu"] for r in sel])
            errs = np.array([r["err"] for r in sel])
            out.append({
                "aspect": a,
                "branch": _branch(a),
                "mu_rolling": mur,
                "mu_grid": [float(x) for x in mus],
                "err_grid": [float(x) for x in errs],
                "best_mu": float(mus[int(np.argmin(np.abs(errs)))]),
                "best_abs_err": float(np.min(np.abs(errs))),
                "mu_required": zero_crossing(mus, errs),
                "n_particles": sel[0]["n_particles"],
                "stop_reasons": sorted({r["stop_reason"] for r in sel}),
            })
    return out


def gap_table(probes: list[dict]) -> list[dict]:
    out = []
    for mur in WAVE4_MUR:
        lin = next((p for p in probes
                    if p["branch"] == "linear" and abs(p["mu_rolling"] - mur) < 1e-9), None)
        sq = next((p for p in probes
                   if p["branch"] == "sqrt" and abs(p["mu_rolling"] - mur) < 1e-9), None)
        if lin is None or sq is None:
            continue
        req_l, req_s = lin["mu_required"], sq["mu_required"]
        out.append({
            "mu_rolling": mur,
            "mu_req_linear": req_l,
            "mu_req_sqrt": req_s,
            "gap": None if (req_l is None or req_s is None) else req_s - req_l,
            "both_bracketed": req_l is not None and req_s is not None,
        })
    return out


def h4_decision(gaps: list[dict]) -> dict:
    """Pre-registered rule: H4 needs ``gap`` to change sign inside the grid."""
    usable = [g for g in gaps if g["gap"] is not None]
    detail = {
        "rule": "gap(mu_r) = mu_req(sqrt) - mu_req(linear) must cross zero on "
                "the WAVE4_MUR grid; otherwise H4 is refuted",
        "grid": list(WAVE4_MUR),
        "usable_points": len(usable),
    }
    if len(usable) < 2:
        detail.update(h4_supported=False, reason="fewer than two bracketed mu_r points")
        return detail
    mur = [g["mu_rolling"] for g in usable]
    gap = [g["gap"] for g in usable]
    for k in range(len(usable) - 1):
        if gap[k] == 0.0:
            detail.update(h4_supported=True, crossing_mur=float(mur[k]),
                          crossing_mu=float(usable[k]["mu_req_linear"]),
                          reason="gap vanished on a grid point")
            return detail
        if gap[k] * gap[k + 1] < 0.0:
            t = (-gap[k]) / (gap[k + 1] - gap[k])
            mur_star = mur[k] + t * (mur[k + 1] - mur[k])
            mu_star = usable[k]["mu_req_linear"] + t * (
                usable[k + 1]["mu_req_linear"] - usable[k]["mu_req_linear"])
            detail.update(h4_supported=True,
                          crossing_mur=float(mur_star), crossing_mu=float(mu_star),
                          bracket=[mur[k], mur[k + 1]],
                          reason="gap changes sign between two mu_r grid points")
            return detail
    k = int(np.argmin(np.abs(gap)))
    detail.update(h4_supported=False,
                  min_abs_gap=float(abs(gap[k])), min_abs_gap_mur=float(mur[k]),
                  reason="gap keeps one sign over the whole mu_r grid: the two "
                         "branch requirements move together with mu_r, so no "
                         "single (mu, mu_r) can satisfy both")
    return detail


# --------------------------------------------------------------------------
def validate(rows: list[dict]) -> dict:
    vals = [r for r in rows if r["phase"] == "wave4_validation"]
    if not vals:
        return {"ran": False}
    pairs = {(round(r["mu"], 6), round(r["mu_rolling"], 6)) for r in vals}
    pairs = sorted(pairs)
    out = {"ran": True, "pairs": pairs, "per_pair": []}
    for mu, mur in pairs:
        sel = sorted([r for r in vals
                      if abs(r["mu"] - mu) < 1e-9 and abs(r["mu_rolling"] - mur) < 1e-9],
                     key=lambda r: r["aspect"])
        worst = max(abs(r["err"]) for r in sel)
        worst_t = max((abs(r["err_t"]) for r in sel
                       if r["err_t"] == r["err_t"]), default=None)
        in_sample = [r for r in sel if abs(r["aspect"] - 1.0) < 1e-9
                     or abs(r["aspect"] - 3.0) < 1e-9]
        out_of_sample = [r for r in sel if r not in in_sample]
        out["per_pair"].append({
            "mu": mu, "mu_rolling": mur,
            "n_aspects": len(sel),
            "pass15_runout": sum(1 for r in sel if r["pass15"]),
            "pass15_time": sum(1 for r in sel if r["pass15_t"]),
            "worst_abs_err_runout": float(worst),
            "worst_abs_err_time": None if worst_t is None else float(worst_t),
            "worst_abs_err_runout_out_of_sample": float(
                max(abs(r["err"]) for r in out_of_sample)) if out_of_sample else None,
            "per_aspect": [
                {"aspect": r["aspect"], "branch": r["branch"],
                 "runout_ratio": r["runout_ratio"], "ref": r["ref_runout_ratio"],
                 "err": r["err"], "pass15": r["pass15"],
                 "err_under_3pct": abs(r["err"]) < TIGHT,
                 "t_reach": r["t_reach"], "ref_t": r["ref_t_reach"],
                 "err_t": r["err_t"], "pass15_t": r["pass15_t"],
                 "energy_residual": r["energy_residual"],
                 "stop_reason": r["stop_reason"]}
                for r in sel],
        })
    return out



# --------------------------------------------------------------------------
def sensitivity(probes: list[dict]) -> dict:
    """``d mu_req / d mu_r`` per branch, plus what the grid box did **not** see.

    Points whose required friction is outside ``WAVE4_MU`` are reported as
    *censored* (only a bound is known), which is how the analysis stays honest
    about a requirement that escaped the sampled box.
    """
    lo, hi = min(WAVE4_MU), max(WAVE4_MU)
    out = {"mu_box": [lo, hi], "branches": {}}
    for br in ("linear", "sqrt"):
        pts = [(p["mu_rolling"], p["mu_required"]) for p in probes if p["branch"] == br]
        brack = sorted([(m, r) for m, r in pts if r is not None])
        censored = sorted([m for m, r in pts if r is None])
        entry = {"bracketed": brack, "censored_mu_rolling": censored}
        if len(brack) >= 2:
            (m0, r0), (m1, r1) = brack[0], brack[-1]
            entry["slope_dmu_req_dmur"] = float((r1 - r0) / (m1 - m0))
        out["branches"][br] = entry
    lin = out["branches"]["linear"]
    sq = out["branches"]["sqrt"]
    # gap at the only mu_r where both branches were bracketed
    both = [g for g in gap_table(probes) if g["gap"] is not None]
    out["bracketed_gap_points"] = both
    # extrapolated crossing, explicitly labelled unverified
    if both and "slope_dmu_req_dmur" in lin and "slope_dmu_req_dmur" in sq:
        g0 = both[0]
        dg = sq["slope_dmu_req_dmur"] - lin["slope_dmu_req_dmur"]
        if dg != 0.0:
            mur_star = g0["mu_rolling"] - g0["gap"] / dg
            out["extrapolated_crossing"] = {
                "mu_rolling": float(mur_star),
                "mu": float(g0["mu_req_linear"]
                           + lin["slope_dmu_req_dmur"] * (mur_star - g0["mu_rolling"])),
                "verified": False,
                "note": "linear extrapolation of the two branch requirements; "
                        "outside the sampled mu box, therefore NOT a verdict",
            }
    return out


# --------------------------------------------------------------------------
def write_csv(path: str, rows: list[dict], keys: list[str]) -> None:
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in keys})


def plot(probes: list[dict], gaps: list[dict], decision: dict, val: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    # 1) signed error vs mu, per branch, coloured by mu_r
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, a in zip(axes, PROBE_ASPECTS):
        for k, p in enumerate([p for p in probes if abs(p["aspect"] - a) < 1e-9]):
            col = cmap(k / max(1, len(WAVE4_MUR) - 1))
            ax.plot(p["mu_grid"], np.array(p["err_grid"]) * 100, "o-", color=col,
                    label=fr"$\mu_r$ = {p['mu_rolling']:g}")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.axhspan(-15, 15, color="tab:green", alpha=0.08)
        ax.set_title(f"a = {a:g} ({_branch(a)} branch)")
        ax.set_xlabel(r"contact friction $\mu$")
        ax.set_ylabel("runout error vs Lube (2004) [%]")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8, title="$\\mu_r$")
    fig.suptitle("CASE-COLUMN-COLLAPSE round 4: signed runout error on the joint grid")
    fig.tight_layout()
    path = os.path.join(OUT, "err_vs_mu_by_mur.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append(path)

    # 2) required mu vs mu_r (both branches) + gap
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    for br, style in (("linear", "o-"), ("sqrt", "s--")):
        pts = [(p["mu_rolling"], p["mu_required"]) for p in probes
               if p["branch"] == br and p["mu_required"] is not None]
        if pts:
            mur, req = zip(*pts)
            ax.plot(mur, req, style, label=f"{br} branch (a = {1.0 if br=='linear' else 3.0})")
    if decision.get("h4_supported"):
        ax.plot([decision["crossing_mur"]], [decision["crossing_mu"]], "*", ms=16,
                color="tab:red", label="pre-registered crossing")
    ax.set_xlabel(r"rolling resistance $\mu_r$")
    ax.set_ylabel(r"required contact friction $\mu_{req}$")
    ax.set_title("branch requirements vs $\\mu_r$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    us = [g for g in gaps if g["gap"] is not None]
    if us:
        ax.plot([g["mu_rolling"] for g in us], [g["gap"] for g in us], "o-")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"rolling resistance $\mu_r$")
    ax.set_ylabel(r"$\mu_{req}^{sqrt} - \mu_{req}^{lin}$")
    ax.set_title("gap: zero crossing = single $(\\mu,\\mu_r)$ exists")
    ax.grid(alpha=0.3)
    fig.suptitle("joint identifiability of contact vs rolling friction")
    fig.tight_layout()
    path = os.path.join(OUT, "mu_required_and_gap.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append(path)

    # 3) validation curve at the pre-registered pair
    if val.get("ran") and val["per_pair"]:
        for pp in val["per_pair"]:
            per = pp["per_aspect"]
            a_vals = [p["aspect"] for p in per]
            ref = [p["ref"] for p in per]
            sim = [p["runout_ratio"] for p in per]
            err = np.array([p["err"] for p in per]) * 100
            fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
            ax = axes[0]
            aa = np.linspace(0.3, 3.2, 200)
            ax.plot(aa, [lube_2004_runout_ratio(x) for x in aa], "-", color="k",
                    label="Lube (2004) reference")
            ax.plot(a_vals, sim, "o", ms=7, color="tab:red",
                    label=f"DEM ($\\mu$ = {pp['mu']:g}, $\\mu_r$ = {pp['mu_rolling']:g})")
            ax.axvline(A_REF_SPLIT, color="tab:blue", ls=":", lw=1.0,
                       label="reference branch switch $a$ = 1.7")
            ax.set_xlabel("aspect ratio a")
            ax.set_ylabel(r"$r_\infty/r_i$")
            ax.set_title("runout law")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            ax = axes[1]
            cols = ["tab:green" if abs(e) < TOL * 100 else "tab:red" for e in err]
            ax.bar([f"{x:g}" for x in a_vals], err, color=cols)
            ax.axhline(15, color="tab:green", ls="--", lw=1.0, label="±15 % bar")
            ax.axhline(-15, color="tab:green", ls="--", lw=1.0)
            ax.axhline(3, color="tab:orange", ls=":", lw=1.0, label="±3 % bar")
            ax.axhline(-3, color="tab:orange", ls=":", lw=1.0)
            ax.set_xlabel("aspect ratio a")
            ax.set_ylabel("runout error [%]")
            ax.set_title("per-aspect error (green = inside 15 %)")
            ax.grid(alpha=0.3, axis="y")
            ax.legend(fontsize=8)
            fig.suptitle(f"pre-registered joint fit: $\\mu$ = {pp['mu']:g}, "
                         f"$\\mu_r$ = {pp['mu_rolling']:g}")
            fig.tight_layout()
            path = os.path.join(OUT, f"validation_mu{pp['mu']:g}_mur{pp['mu_rolling']:g}.png")
            fig.savefig(path, dpi=140)
            plt.close(fig)
            figs.append(path)
    return figs


# --------------------------------------------------------------------------
def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    probe_rows = load("probe_mu*_mur*.json")
    val_rows = load("wave4_valid_*.json")
    if not probe_rows:
        print("no probe cases found in", CASE_DIR, file=sys.stderr)
        return 2

    probes = probe_table(probe_rows)
    gaps = gap_table(probes)
    decision = h4_decision(gaps)
    val = validate(val_rows)

    keys = ["case", "phase", "aspect", "branch", "mu", "mu_rolling", "n_particles",
            "runout_ratio", "ref_runout_ratio", "err", "pass15", "t_reach",
            "ref_t_reach", "err_t", "pass15_t", "stop_reason", "energy_residual",
            "wall_clock_s"]
    write_csv(os.path.join(OUT, "probe_table.csv"), probe_rows, keys)
    write_csv(os.path.join(OUT, "mu_required_table.csv"), gaps,
              ["mu_rolling", "mu_req_linear", "mu_req_sqrt", "gap", "both_bracketed"])

    summary = {
        "case": "CASE-COLUMN-COLLAPSE",
        "round": 4,
        "reference": LUBE2004_SOURCE,
        "doi": LUBE2004_DOI,
        "tolerance": TOL,
        "tight_tolerance": TIGHT,
        "grid": {"mu": list(WAVE4_MU), "mu_rolling": list(WAVE4_MUR),
                 "probe_aspects": list(PROBE_ASPECTS)},
        "n_probe_cases": len(probe_rows),
        "total_wall_clock_s": float(sum(r["wall_clock_s"] or 0.0 for r in probe_rows)),
        "probes": probes,
        "gaps": gaps,
        "h4": decision,
        "sensitivity": sensitivity(probes),
        "validation": val,
    }
    with open(os.path.join(OUT, "curve_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    # slim copy without the per-case provenance arrays (those stay in runs/)
    with open(os.path.join(OUT, "probe_cases_slim.json"), "w") as fh:
        json.dump(probe_rows + val_rows, fh, indent=1)

    figs = plot(probes, gaps, decision, val)
    summary["figures"] = [os.path.relpath(f, ROOT) for f in figs]

    print(json.dumps({k: summary[k] for k in
                      ("n_probe_cases", "total_wall_clock_s", "h4",
                       "sensitivity")}, indent=1))
    for g in gaps:
        gl = "None" if g["mu_req_linear"] is None else f"{g['mu_req_linear']:.3f}"
        gs = "None" if g["mu_req_sqrt"] is None else f"{g['mu_req_sqrt']:.3f}"
        gg = "None" if g["gap"] is None else f"{g['gap']:+.3f}"
        print(f"mu_r={g['mu_rolling']:.2f}  mu_req_lin={gl}  mu_req_sqrt={gs}  gap={gg}")
    if val.get("ran"):
        for pp in val["per_pair"]:
            print(f"VALID mu={pp['mu']:g} mur={pp['mu_rolling']:g}: "
                  f"runout {pp['pass15_runout']}/{pp['n_aspects']} within 15 %, "
                  f"worst |err| = {pp['worst_abs_err_runout']*100:.2f} % "
                  f"(out-of-sample worst = "
                  f"{(pp['worst_abs_err_runout_out_of_sample'] or 0)*100:.2f} %), "
                  f"time {pp['pass15_time']}/{pp['n_aspects']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())