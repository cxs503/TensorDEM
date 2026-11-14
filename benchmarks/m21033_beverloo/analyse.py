#!/usr/bin/env python3
"""CASE-DEM-SILO-BEVERLOO (m21033): analyse the campaign results.

Reads ``runs/m21033_beverloo/<case>.json`` (produced by ``run_case.py``),
recomputes the verdict table from the raw telemetry (zero hand-typed numbers),
renders the two report figures and writes ``summary.json`` +
``report_tables.md``.

Usage
-----
    PYTHONPATH=src python benchmarks/m21033_beverloo/analyse.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"
RUN_DIR = ROOT / "runs" / "m21033_beverloo"
FIG_DIR = ROOT / "docs" / "figures"

ORDER = ["D5", "D4", "D6", "D5_mu30", "D5_kn1e5", "D5_dthalf"]
LAW_CASES = ["D4", "D5", "D6"]


def load(case_id: str) -> dict:
    p = RUN_DIR / f"{case_id}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def particle_mass_kg(rec: dict) -> float:
    """Per-particle mass from the run's own bookkeeping (no hand-typed value)."""
    tele = rec["telemetry"]
    if tele.get("n_discharged_total", 0) > 0 and tele.get("discharged_mass_kg", 0) > 0:
        return tele["discharged_mass_kg"] / tele["n_discharged_total"]
    d = tele["particle_diameter_m"] if "particle_diameter_m" in tele \
        else tele["particle_diameter"]
    return 2500.0 * (4.0 / 3.0) * math.pi * (0.5 * d) ** 3


def main() -> int:
    ref = json.loads(REF_PATH.read_text())
    C, k = float(ref["coefficients"]["C"]), float(ref["coefficients"]["k"])

    rows = []
    for cid in ORDER:
        rec = load(cid)
        if not rec:
            continue
        v = rec["verdict"]
        tele = rec["telemetry"]
        rows.append({
            "case": cid,
            "D_over_d": rec["dimensionless"]["D_over_d"],
            "mu_p": rec["dimensionless"]["mu_p"],
            "kn_N_m": rec["dimensionless"]["kn_N_m"],
            "n_bed": rec["dimensionless"]["n_particles_bed"],
            "rho_b_kg_m3": v["rho_b_kg_m3"],
            "W_got_kg_s": v["W_got_kg_s"],
            "W_ref_kg_s": v["W_ref_kg_s"],
            "rel_err_pct": 100.0 * v["rel_err_W"],
            "C_recovered": rec["dimensionless"]["C_recovered"],
            "CV": tele["steady_rate_cv"],
            "discharged_frac": (tele["n_discharged_total"] / tele["n_initial"]
                                if tele["n_initial"] else float("nan")),
            "pass_10pct": v["pass_10pct"],
            "wall_clock_s": rec["wall_clock_s"],
        })

    # ---- summary -----------------------------------------------------------
    primary = next((r for r in rows if r["case"] == "D5"), None)
    law = [r for r in rows if r["case"] in LAW_CASES]
    summary = {
        "mission_id": 21033,
        "anchor": "CASE-DEM-SILO-BEVERLOO",
        "reference": ref["source"],
        "C": C,
        "k": k,
        "tolerance_pct": 10.0,
        "primary_case": "D5",
        "primary_rel_err_pct": (primary["rel_err_pct"] if primary else None),
        "primary_pass_10pct": (primary["pass_10pct"] if primary else None),
        "law_cases": law,
        "all_rows": rows,
        "verdict": ("PASS" if (primary and primary["pass_10pct"]) else "FAIL"),
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- markdown tables ---------------------------------------------------
    lines = ["| case | D/d | μ_p | kn [N/m] | n_bed | ρ_b [kg/m³] | W_dem [kg/s] | "
             "W_ref [kg/s] | 误差 | C_rec | CV | 卸料占比 | <10% |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['D_over_d']:.2f} | {r['mu_p']:.2f} | "
            f"{r['kn_N_m']:.0e} | {r['n_bed']} | {r['rho_b_kg_m3']:.0f} | "
            f"{r['W_got_kg_s']:.4f} | {r['W_ref_kg_s']:.4f} | "
            f"{r['rel_err_pct']:+.1f}% | {r['C_recovered']:.3f} | "
            f"{r['CV']:.3f} | {100*r['discharged_frac']:.0f}% | "
            f"{'✓' if r['pass_10pct'] else '✗'} |")
    (RUN_DIR / "report_tables.md").write_text("\n".join(lines) + "\n")

    # ---- figures -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG_DIR.mkdir(parents=True, exist_ok=True)

        # fig 1: measured vs Beverloo per law case
        xs = np.arange(len(law))
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
        ax.bar(xs - 0.18, [r["W_ref_kg_s"] for r in law], width=0.36,
               label="Beverloo (1961) ref", color="#4878a8")
        ax.bar(xs + 0.18, [r["W_got_kg_s"] for r in law], width=0.36,
               label="TensorDEM silo DEM", color="#d65f5f")
        for i, r in enumerate(law):
            ax.text(i + 0.18, r["W_got_kg_s"], f"{r['rel_err_pct']:+.1f}%",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"D/d={r['D_over_d']:.0f}" for r in law])
        ax.set_ylabel("steady mass flow rate W [kg/s]")
        ax.set_title("Flat-bottom silo discharge vs Beverloo (1961)\n"
                     "C=0.58, k=1.4, settled+plug protocol, kn=1e6 N/m")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "m21033_beverloo_flow_law.png")
        plt.close(fig)

        # fig 2: cumulative discharged mass (steadiness) for the law cases
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
        for r in law:
            rec = load(r["case"])
            t = np.asarray(rec["telemetry"]["discharge_time_s"])
            n = np.asarray(rec["telemetry"]["discharge_cumulative_n"])
            m = particle_mass_kg(rec)
            ax.plot(t, n * m, label=r["case"])
        ax.set_xlabel("time since orifice opening [s]")
        ax.set_ylabel("cumulative discharged mass [kg]")
        ax.set_title("Discharge stream (steady middle 60% used for W)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "m21033_beverloo_discharge_stream.png")
        plt.close(fig)
        print("figures written to", FIG_DIR)
    except Exception as exc:  # pragma: no cover
        print(f"figure generation skipped: {exc}")

    print("\n".join(lines))
    if summary["primary_rel_err_pct"] is not None:
        print(f"\nVERDICT: {summary['verdict']} "
              f"(primary D5 rel err = {summary['primary_rel_err_pct']:+.1f}%)")
    else:
        print(f"\nVERDICT: {summary['verdict']} (no primary result)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
