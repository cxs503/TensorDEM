#!/usr/bin/env python3
"""Aggregate the m20934 kn-sweep results and render the report figures.

Reads ``runs/m20934_beverloo_kn/*.json`` (one per arm), re-derives the
Beverloo comparison from the machine-readable reference
(``benchmarks/m20055_beverloo/ref_beverloo_1961.json``), writes
``summary.json`` + ``report_tables.md`` and three figures:

* ``fig_w_vs_dd.png``      -- W vs D/d per kn level vs the Beverloo curve
* ``fig_err_vs_kn.png``     -- relative error vs kn (log x)
* ``fig_cumulative_D5.png`` -- cumulative discharge count, D5 kn=1e5 vs 1e6

Usage
-----
    PYTHONPATH=src python benchmarks/m20934_beverloo_kn/analyse.py \
        [--in-dir runs/m20934_beverloo_kn] [--out-dir runs/m20934_beverloo_kn]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"

KN_COLORS = {1.0e5: "tab:blue", 3.0e5: "tab:orange", 1.0e6: "tab:red"}
KN_MARKERS = {1.0e5: "o", 3.0e5: "s", 1.0e6: "^"}


def beverloo_W(C: float, k: float, D: float, d: float, rho_b: float, g: float) -> float:
    Deff = D - k * d
    if Deff <= 0:
        return float("nan")
    return C * rho_b * math.sqrt(g) * Deff ** 2.5


def load_arms(in_dir: Path) -> list[dict]:
    arms = []
    for p in sorted(in_dir.glob("*.json")):
        if p.name in ("summary.json",):
            continue
        rec = json.loads(p.read_text())
        if "W_got_kg_s" not in rec:
            continue
        arms.append(rec)
    return arms


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(ROOT / "runs" / "m20934_beverloo_kn"))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = json.loads(REF_PATH.read_text())
    C, k = float(ref["coefficients"]["C"]), float(ref["coefficients"]["k"])
    arms = load_arms(in_dir)
    if not arms:
        raise SystemExit(f"no arm results in {in_dir}")

    rows = []
    for rec in arms:
        D = rec["orifice_diameter"]
        d = rec["particle_diameter"]
        rho_b = rec["rho_b_kg_m3"]
        g = rec["gravity"]
        W_ref = beverloo_W(C, k, D, d, rho_b, g)
        W_got = rec["W_got_kg_s"]
        rel = (W_got - W_ref) / W_ref
        Deff = D - k * d
        C_got = W_got / (rho_b * math.sqrt(g) * Deff ** 2.5) if Deff > 0 else float("nan")
        rows.append({
            "tag": rec["tag"],
            "D_over_d": rec["D_over_d"],
            "kn_N_m": rec["kn_N_m"],
            "dt_s": rec["dt_s"],
            "n_initial": rec["n_initial"],
            "n_bed": rec["n_bed"],
            "rho_b_kg_m3": rho_b,
            "W_got_kg_s": W_got,
            "W_ref_kg_s": W_ref,
            "rel_err": rel,
            "rel_err_pct": rel * 100,
            "C_recovered": C_got,
            "steady_rate_cv": rec["steady_rate_cv"],
            "n_discharged_total": rec["n_discharged_total"],
            "wall_clock_s": rec["wall_clock_s"],
            "pass_10pct": abs(rel) < 0.10,
            "pass_15pct": abs(rel) < 0.15,
        })
    rows.sort(key=lambda r: (r["D_over_d"], r["kn_N_m"], r["dt_s"]))

    summary = {
        "mission_id": 20934,
        "anchor": "CASE-DEM-SILO-BEVERLOO",
        "reference": {
            "source": ref["source"],
            "correlation": "W = C * rho_b * sqrt(g) * (D - k*d)**(5/2)",
            "C": C,
            "k": k,
            "C_range": ref["coefficients"]["C_range_spherical"],
            "k_range": ref["coefficients"]["k_range_spherical"],
        },
        "arms": rows,
        "n_arms": len(rows),
        "n_pass_10pct": sum(r["pass_10pct"] for r in rows),
        "n_pass_15pct": sum(r["pass_15pct"] for r in rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # markdown table (programmatic, zero hand-copy)
    lines = [
        "| arm | D/d | kn [N/m] | dt [s] | n_bed | ρ_b [kg/m³] | W_got [kg/s] | W_ref [kg/s] | rel err | C_rec | CV | <10% |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['tag']} | {r['D_over_d']:.1f} | {r['kn_N_m']:.1e} | {r['dt_s']:.2e} "
            f"| {r['n_bed']} | {r['rho_b_kg_m3']:.0f} | {r['W_got_kg_s']:.4f} "
            f"| {r['W_ref_kg_s']:.4f} | {r['rel_err_pct']:+.1f}% | {r['C_recovered']:.3f} "
            f"| {r['steady_rate_cv']:.3f} | {'PASS' if r['pass_10pct'] else 'FAIL'} |")
    (out_dir / "report_tables.md").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ figs
    # fig 1: W vs D/d per kn vs Beverloo curve
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    Ds = np.linspace(0.075, 0.135, 60)
    seen = set()
    for rec in arms:
        if "dthalf" in rec["tag"]:
            continue
        kn = rec["kn_N_m"]
        rho_b = rec["rho_b_kg_m3"]
        d = rec["particle_diameter"]
        g = rec["gravity"]
        Wb = beverloo_W(C, k, Ds, d, rho_b, g)
        if kn not in seen:
            ax.plot(Ds / d, Wb, "--", color=KN_COLORS[kn], alpha=0.35,
                    label=f"Beverloo (kn={kn:.0e})")
            ax.plot(rec["D_over_d"], rec["W_got_kg_s"], KN_MARKERS[kn],
                   color=KN_COLORS[kn], ms=9, label=f"DEM kn={kn:.0e}")
            seen.add(kn)
        else:
            ax.plot(rec["D_over_d"], rec["W_got_kg_s"], KN_MARKERS[kn],
                    color=KN_COLORS[kn], ms=9)
    ax.set_xlabel("D/d [-]")
    ax.set_ylabel("steady mass flow rate W [kg/s]")
    ax.set_title("Flat-bottom silo discharge vs Beverloo (1961)\nm20934 kn-sweep, μ=0.30, μ_w=0")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_w_vs_dd.png", dpi=150)
    plt.close(fig)

    # fig 2: relative error vs kn
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for rec in arms:
        if "dthalf" in rec["tag"]:
            continue
        kn = rec["kn_N_m"]
        ax.plot(kn, rec["rel_err_pct"], KN_MARKERS[kn], color=KN_COLORS[kn], ms=10,
                label=f"D/d={rec['D_over_d']:.0f}")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(10, color="r", ls=":", lw=1)
    ax.axhline(-10, color="r", ls=":", lw=1)
    ax.text(1.05e5, 10.5, "±10% acceptance", color="r", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("kn [N/m]")
    ax.set_ylabel("relative error vs Beverloo [%]")
    ax.set_title("Contact-stiffness sensitivity (m20934)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_err_vs_kn.png", dpi=150)
    plt.close(fig)

    # fig 3: cumulative discharge, D5 kn=1e5 vs 1e6
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for tag, col in (("D5_kn1e5", "tab:blue"), ("D5_kn1e6", "tab:red")):
        rec = next((a for a in arms if a["tag"] == tag), None)
        if rec is None:
            continue
        t = np.asarray(rec["discharge_time_s"])
        n = np.asarray(rec["discharge_cumulative_n"])
        if t.size:
            ax.plot(t, n, color=col, lw=1.5, label=tag)
    ax.set_xlabel("time since orifice opening [s]")
    ax.set_ylabel("cumulative discharged particles [-]")
    ax.set_title("D5 (D/d=5): discharge kinetics, kn=1e5 vs kn=1e6")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_cumulative_D5.png", dpi=150)
    plt.close(fig)

    print(f"summary: {len(rows)} arms, {summary['n_pass_10pct']} pass <10%, "
          f"{summary['n_pass_15pct']} pass <15%")
    for r in rows:
        print(f"  {r['tag']:<16s} rel={r['rel_err_pct']:+6.1f}%  "
              f"{'PASS' if r['pass_10pct'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
