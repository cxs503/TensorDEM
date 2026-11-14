#!/usr/bin/env python3
"""Analyse CASE-DEM-SILO-BEVERLOO runs: tables, sensitivity, figures.

Every number in the generated markdown/figure is read from
``runs/m20055_beverloo/*.json`` (written by ``run_case.py``) and from
``benchmarks/m20055_beverloo/ref_beverloo_1961.json``.  Nothing is typed in.

Usage
-----
    PYTHONPATH=src python benchmarks/m20055_beverloo/analyse.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

from tensordem.silo_discharge import steady_rate    # noqa: E402

RUNS = ROOT / "runs" / "m20055_beverloo"
FIGS = ROOT / "docs" / "figures"
REF = json.loads((ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json").read_text())
C_LIT = float(REF["coefficients"]["C"])
K_LIT = float(REF["coefficients"]["k"])


def load_runs() -> dict:
    out = {}
    for p in sorted(RUNS.glob("*.json")):
        if p.name in ("summary.json",):
            continue
        d = json.loads(p.read_text())
        if "telemetry" in d and "verdict" in d:
            out[d["case_id"]] = d
    return out


def sensitivity(run: dict) -> dict:
    """Steady rate under alternative quantile bands (same raw event record)."""
    tele = run["telemetry"]
    t = tele["discharge_time_s"]
    n = tele["discharge_cumulative_n"]
    d = tele["particle_diameter"]
    mass = tele["bulk_density_bed_kg_m3"] * 0.0  # placeholder, replaced below
    # particle mass from the run record (rho_p and d are both recorded)
    rho_p = run["dimensionless"].get("rho_p", None)
    if rho_p is None:
        # rho_p = rho_b / (rho_b/rho_p)
        rho_p = tele["bulk_density_bed_kg_m3"] / run["dimensionless"]["rho_b_over_rho_p"]
    mass = rho_p * 4.0 / 3.0 * math.pi * (0.5 * d) ** 3
    bands = {"0.20-0.80 (predeclared)": (0.20, 0.80),
             "0.10-0.60 (early)": (0.10, 0.60),
             "0.35-0.65 (tight)": (0.35, 0.65)}
    res = {}
    for label, q in bands.items():
        r = steady_rate(t, n, mass, 0.5, 5, q)
        res[label] = {"W": r["steady_flow_rate_kg_s"], "cv": r["steady_rate_cv"],
                      "span": r["steady_interval_s"]}
    return res


def power_law_slope(runs: dict) -> dict:
    """log-log slope of W vs (D - k d): the Beverloo law's 5/2 exponent."""
    keys = [k for k in ("D4", "D5", "D6") if k in runs]
    if len(keys) < 3:
        return {}
    x = np.array([runs[k]["verdict"]["Deff_m"] for k in keys])
    y = np.array([runs[k]["verdict"]["W_got_kg_s"] for k in keys])
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    return {"slope": float(slope), "target": 2.5,
            "rel_err": float((slope - 2.5) / 2.5), "points": keys}


def fig_flow_law(runs: dict, out: Path) -> None:
    keys = [k for k in ("D4", "D5", "D6") if k in runs]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    x = np.array([runs[k]["dimensionless"]["D_over_d_minus_k"] for k in keys])
    y = np.array([runs[k]["verdict"]["W_got_kg_s"] for k in keys])
    e = np.array([runs[k]["telemetry"]["steady_rate_cv"] for k in keys]) * y
    ax.errorbar(x, y, yerr=e, fmt="o", color="C0", capsize=3,
                label="TensorDEM (steady window)")
    xx = np.linspace(x.min() * 0.8, x.max() * 1.2, 50)
    anchor = float(np.exp(np.mean(np.log(y) - 2.5 * np.log(x))))
    ax.plot(xx, anchor * xx ** 2.5, "--", color="C1",
            label="Beverloo exponent 5/2 (anchored)")
    C_dem = runs[keys[-1]]["dimensionless"]["C_recovered"]
    rho_b = runs[keys[-1]]["verdict"]["rho_b_kg_m3"]
    g = runs[keys[-1]]["telemetry"]["gravity"]
    d = runs[keys[-1]]["telemetry"]["particle_diameter"]
    D = runs[keys[-1]]["telemetry"]["orifice_diameter"]
    ax.plot(xx, C_LIT * rho_b * math.sqrt(g) * (xx * d) ** 2.5, "-", color="C3",
            label=f"Beverloo C={C_LIT}, k={K_LIT} (ref, same rho_b)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$(D-kd)/d$"); ax.set_ylabel("W  [kg/s]")
    ax.set_title("Silo discharge: DEM vs Beverloo (1961)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_errors(runs: dict, out: Path) -> None:
    keys = [k for k in ("D4", "D5", "D6", "D5_mu15", "D5_mu45", "D5_dthalf") if k in runs]
    if not keys:
        return
    err = [100.0 * runs[k]["verdict"]["rel_err_W"] for k in keys]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    colors = ["C0" if abs(e) < 10.0 else "C3" for e in err]
    ax.bar(keys, err, color=colors)
    ax.axhspan(-10, 10, color="C2", alpha=0.12, label="±10 % acceptance band")
    ax.axhline(0.0, color="k", lw=0.8)
    for i, e in enumerate(err):
        ax.text(i, e + (1.5 if e >= 0 else -3.0), f"{e:+.1f}%", ha="center", fontsize=8)
    ax.set_ylabel("rel. error vs Beverloo  [%]")
    ax.set_title("CASE-DEM-SILO-BEVERLOO: W error per case")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_discharge_curve(run: dict, out: Path) -> None:
    tele = run["telemetry"]
    t = np.asarray(tele["discharge_time_s"])
    n = np.asarray(tele["discharge_cumulative_n"], dtype=float)
    lo, hi = tele["steady_interval_s"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(t, n, color="C0", label="cumulative discharged particles")
    ax.axvspan(lo, hi, color="C2", alpha=0.15,
               label=f"steady window {lo:.2f}–{hi:.2f} s")
    slope = np.polyfit([lo, hi], np.interp([lo, hi], t, n), 1)[0]
    ax.plot([lo, hi], np.interp([lo, hi], t, n), "k--", lw=1.2,
            label=f"steady rate {slope:.0f} part./s")
    ax.set_xlabel("t  [s]"); ax.set_ylabel("N discharged")
    ax.set_title(f"{run['case_id']}: batch discharge curve (settled bed)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main() -> int:
    runs = load_runs()
    if not runs:
        print("no runs found")
        return 1
    FIGS.mkdir(parents=True, exist_ok=True)
    (RUNS / "figures").mkdir(parents=True, exist_ok=True)

    rows = []
    for cid, run in runs.items():
        tele, dim, ver = run["telemetry"], run["dimensionless"], run["verdict"]
        rows.append({
            "case": cid,
            "D_over_d": dim["D_over_d"],
            "mu_p": dim["mu_p"],
            "n_bed": dim["n_particles_bed"],
            "H_over_D": dim["bed_height_over_D"],
            "silo_D_over_D": dim["silo_D_over_D"],
            "rho_b": ver["rho_b_kg_m3"],
            "phi": dim["packing_fraction_bed"],
            "n_disch_pct": 100.0 * tele["n_discharged_total"] / tele["n_initial"],
            "W": ver["W_got_kg_s"],
            "W_ref": ver["W_ref_kg_s"],
            "rel_pct": 100.0 * ver["rel_err_W"],
            "C_recovered": dim["C_recovered"],
            "C_rel_pct": 100.0 * ver["rel_err_C"],
            "cv": tele["steady_rate_cv"],
            "pass10": ver["pass_10pct"],
            "wall_s": run["wall_clock_s"],
            "dt_over_T": dim["dt_over_contact_period"],
        })
    order = {k: i for i, k in enumerate(
        ["D4", "D5", "D6", "D5_mu15", "D5_mu45", "D5_dthalf"])}
    rows.sort(key=lambda r: order.get(r["case"], 99))

    slope = power_law_slope(runs)
    sens = {cid: sensitivity(run) for cid, run in runs.items()}
    primary = runs.get("D5")
    figs = {}
    if primary is not None:
        p = FIGS / "m20055_beverloo_discharge_curve.png"
        fig_discharge_curve(primary, p); figs["discharge_curve"] = str(p.relative_to(ROOT))
    p = FIGS / "m20055_beverloo_flow_law.png"; fig_flow_law(runs, p)
    figs["flow_law"] = str(p.relative_to(ROOT))
    p = FIGS / "m20055_beverloo_errors.png"; fig_errors(runs, p)
    figs["errors"] = str(p.relative_to(ROOT))

    summary = {
        "case": "CASE-DEM-SILO-BEVERLOO",
        "reference": REF["source"],
        "reference_coefficients": REF["coefficients"],
        "rows": rows,
        "power_law": slope,
        "quantile_sensitivity": sens,
        "figures": figs,
        "n_runs": len(rows),
        "all_pass_10pct": all(r["pass10"] for r in rows),
    }
    (RUNS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---- markdown tables (program-generated, no hand-copied digits) -----
    lines = []
    lines.append("| case | D/d | mu_p | n_bed | H/D | silo/D | rho_b [kg/m3] | phi | discharged | W [kg/s] | W_ref [kg/s] | err | C_rec | CV | <10% |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['D_over_d']:.2f} | {r['mu_p']:.2f} | {r['n_bed']} | "
            f"{r['H_over_D']:.2f} | {r['silo_D_over_D']:.2f} | {r['rho_b']:.0f} | "
            f"{r['phi']:.3f} | {r['n_disch_pct']:.0f}% | {r['W']:.4f} | {r['W_ref']:.4f} | "
            f"{r['rel_pct']:+.1f}% | {r['C_recovered']:.3f} | {r['cv']:.3f} | "
            f"{'PASS' if r['pass10'] else 'FAIL'} |")
    if slope:
        lines.append("")
        lines.append(f"power-law slope of W vs (D-kd): {slope['slope']:.3f} "
                     f"(law 2.5, rel {100*slope['rel_err']:+.1f}%)")
    lines.append("")
    lines.append("| case | quantile band | W [kg/s] | CV |")
    lines.append("|---|---|---|---|")
    for cid, s in sens.items():
        for label, v in s.items():
            lines.append(f"| {cid} | {label} | {v['W']:.4f} | {v['cv']:.3f} |")
    tbl = RUNS / "report_tables.md"
    tbl.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsummary -> {RUNS/'summary.json'}\ntables  -> {tbl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())