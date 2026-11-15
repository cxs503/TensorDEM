#!/usr/bin/env python3
"""m21421: programmatic verdict for the wall-proximity scan of
CASE-DEM-SILO-BEVERLOO.

Reads ``runs/m21421_beverloo_wall/*.json`` (raw cluster telemetry) and
``benchmarks/m20055_beverloo/ref_beverloo_1961.json``; produces
``summary.json``, ``report_tables.md`` and the figures under
``docs/figures/``.  Every number in the tables is computed here — nothing is
typed by hand.

Usage:
    PYTHONPATH=src python benchmarks/m21421_beverloo_wall/analyse.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs" / "m21421_beverloo_wall"
REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"
FIG_DIR = ROOT / "docs" / "figures"
ORDER = ["S26", "S35", "S40", "F26"]


def load(cid: str) -> dict:
    p = RUN_DIR / f"{cid}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def verdict(rec: dict, ref: dict) -> dict:
    v = dict(rec["verdict"])
    C = float(ref["coefficients"]["C"])
    return {
        "case_id": rec["case_id"],
        "silo_D_over_D": v["silo_D_over_D"],
        "in_domain": v["silo_D_over_D"] >= float(ref["validation_domain"]["silo_D_over_D_min"]),
        "n_bed": rec["dimensionless"]["n_particles_bed"],
        "rho_b": v["rho_b_kg_m3"],
        "W": v["W_got_kg_s"],
        "W_ref": v["W_ref_kg_s"],
        "rel": v["rel_err_W"],
        "C_rec": v["C_recovered"],
        "C_ref": C,
        "CV": rec["telemetry"]["steady_rate_cv"],
        "W_flux": v["W_flux_kg_s"],
        "flux_gap": v["flux_vs_count_rel"],
        "flux_ok": v["flux_consistent_3pct"],
        "pass10": v["pass_10pct"],
        "discharged_fraction": v["discharged_fraction"],
        "protocol_ok": v["protocol_ok"],
        "wall_clock_s": rec["wall_clock_s"],
        "host": rec.get("host"),
    }


def main() -> int:
    ref = json.loads(REF_PATH.read_text())
    rows = [verdict(load(cid), ref) for cid in ORDER if load(cid)]
    if not rows:
        print("no run artefacts in", RUN_DIR)
        return 1
    by = {r["case_id"]: r for r in rows}

    # ---- pre-registered decisions -----------------------------------------
    decisions = {}
    scan = [r for r in rows if r["case_id"] in ("S26", "S35", "S40")]
    scan.sort(key=lambda r: r["silo_D_over_D"])
    if len(scan) >= 2:
        rels = [abs(r["rel"]) for r in scan]
        decisions["H1_monotone_decreasing_error"] = bool(
            all(b <= a + 1e-12 for a, b in zip(rels, rels[1:])))
    s40 = by.get("S40")
    if s40:
        decisions["H1_pass_at_siloD_over_D_4.0"] = bool(abs(s40["rel"]) < 0.10)
        decisions["H1_strong_C_in_0.55_0.60"] = bool(0.55 <= s40["C_rec"] <= 0.60)
        decisions["H1_C_in_0.522_0.638"] = bool(0.522 <= s40["C_rec"] <= 0.638)
    if "S26" in by and "F26" in by:
        d = abs(by["F26"]["W"] - by["S26"]["W"]) / by["S26"]["W"]
        decisions["H2_protocol_not_artifact"] = bool(d < 0.05)
        decisions["H2_W_F26_vs_S26_rel"] = d
    if rows:
        decisions["flux_consistency_all_cases"] = bool(all(r["flux_ok"] for r in rows))
    primary = by.get("S40") or (scan[-1] if scan else None)
    decisions["ledger_acceptance_met"] = bool(
        primary is not None and abs(primary["rel"]) < 0.10
        and decisions.get("H2_protocol_not_artifact", True)
        and decisions.get("flux_consistency_all_cases", False))

    summary = {
        "anchor": "CASE-DEM-SILO-BEVERLOO",
        "mission_id": 21421,
        "ref": ref["source"],
        "coefficients": ref["coefficients"],
        "rows": rows,
        "decisions": decisions,
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---- markdown tables ---------------------------------------------------
    md = ["| case | silo_D/D | n_bed | rho_b [kg/m3] | W_DEM [kg/s] | W_ref [kg/s] | err | C_rec | CV | W_flux [kg/s] | flux gap | <10% |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["case_id"] != "S26", r["silo_D_over_D"])):
        md.append(
            f"| {r['case_id']} | {r['silo_D_over_D']:.2f} | {r['n_bed']} | "
            f"{r['rho_b']:.1f} | {r['W']:.4f} | {r['W_ref']:.4f} | "
            f"{100*r['rel']:+.1f}% | {r['C_rec']:.3f} | {r['CV']:.3f} | "
            f"{r['W_flux']:.4f} | {100*r['flux_gap']:+.2f}% | "
            f"{'yes' if r['pass10'] else 'NO'} |")
    (RUN_DIR / "report_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print()
    for k, v in decisions.items():
        print(f"  {k} = {v}")

    # ---- figures -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                 # pragma: no cover
        print("matplotlib unavailable:", exc)
        return 0
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # fig 1: recovered Beverloo prefactor and error vs silo_D/D
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for r in scan:
        ax[0].errorbar(r["silo_D_over_D"], r["C_rec"], yerr=0, fmt="o",
                       color="#1f77b4", ms=7)
        ax[0].annotate(r["case_id"], (r["silo_D_over_D"], r["C_rec"]),
                       textcoords="offset points", xytext=(6, -10), fontsize=9)
    ax[0].axhspan(ref["coefficients"]["C_range_spherical"][0],
                  ref["coefficients"]["C_range_spherical"][1],
                  color="#2ca02c", alpha=0.18,
                  label=f"Beverloo C band {ref['coefficients']['C_range_spherical']}")
    ax[0].axhline(ref["coefficients"]["C"], color="#2ca02c", ls="--", lw=1.2,
                  label=f"C = {ref['coefficients']['C']}")
    ax[0].axvline(ref["validation_domain"]["silo_D_over_D_min"], color="0.5",
                  ls=":", lw=1.2, label="applicability bound silo_D/D = 2.5")
    ax[0].set_xlabel("silo_D / D  [-]")
    ax[0].set_ylabel("C_recovered = W / (rho_b sqrt(g) (D-kd)^2.5)")
    ax[0].set_title("Recovered Beverloo prefactor vs wall proximity")
    ax[0].legend(fontsize=7, loc="best")
    ax[0].grid(alpha=0.25)

    for r in scan:
        ax[1].errorbar(r["silo_D_over_D"], 100 * r["rel"], fmt="o-", color="#d62728",
                       ms=7)
        ax[1].annotate(r["case_id"], (r["silo_D_over_D"], 100 * r["rel"]),
                       textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax[1].axhspan(-10, 10, color="#2ca02c", alpha=0.15, label="+/-10 % tolerance")
    ax[1].axhline(0, color="0.4", lw=0.8)
    ax[1].axvline(ref["validation_domain"]["silo_D_over_D_min"], color="0.5",
                  ls=":", lw=1.2)
    ax[1].set_xlabel("silo_D / D  [-]")
    ax[1].set_ylabel("(W_DEM - W_ref)/W_ref  [%]")
    ax[1].set_title("Beverloo error vs wall proximity")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "m21421_beverloo_wall_scan.png", dpi=150)
    plt.close(fig)

    # fig 2: cumulative discharge curves + flux/count parity
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for r in rows:
        rec = load(r["case_id"])
        tele = rec["telemetry"]
        t = tele["discharge_time_s"]
        n = tele["discharge_cumulative_n"]
        frac = [x / tele["n_initial"] for x in n]
        ax[0].plot(t, frac, lw=1.4, label=f"{r['case_id']} (silo_D/D={r['silo_D_over_D']:.1f})")
    ax[0].set_xlabel("t [s]")
    ax[0].set_ylabel("discharged fraction of the bed")
    ax[0].set_title("Cumulative discharge (raw telemetry)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.25)

    lo = min(r["W"] for r in rows) * 0.9
    hi = max(r["W"] for r in rows) * 1.1
    ax[1].plot([lo, hi], [lo, hi], color="0.5", ls="--", lw=1.0, label="parity")
    for r in rows:
        ax[1].errorbar(r["W"], r["W_flux"], fmt="s", color="#1f77b4", ms=7)
        ax[1].annotate(r["case_id"], (r["W"], r["W_flux"]),
                       textcoords="offset points", xytext=(6, -10), fontsize=9)
    ax[1].set_xlabel("W_count (particles ended below plate)  [kg/s]")
    ax[1].set_ylabel("W_flux (aperture-plane crossings)  [kg/s]")
    ax[1].set_title("Two independent rate estimators")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "m21421_beverloo_estimators.png", dpi=150)
    plt.close(fig)
    print("\nfigures ->", FIG_DIR / "m21421_beverloo_wall_scan.png",
          FIG_DIR / "m21421_beverloo_estimators.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())