#!/usr/bin/env python
"""Merge CASE-ROTATING-DRUM round-2 case files, test the frozen predictions,
and generate the figures/table of the report (numbers are read from the JSON
artifacts, never typed by hand)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "artifacts/case_rotating_drum_r2/cases"
OUT = ROOT / "artifacts/case_rotating_drum_r2"
FIG = OUT / "figures"


def load_cases():
    recs = []
    for f in sorted(CASES.glob("case_*.json")):
        recs.append(json.loads(f.read_text()))
    return recs


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[m]))
    ry = np.argsort(np.argsort(y[m]))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = math.sqrt(float(np.sum(rx ** 2) * np.sum(ry ** 2)))
    return float(np.sum(rx * ry) / den) if den > 0 else float("nan")


def main():
    recs = load_cases()
    for r in recs:                      # normalise the frozen output key
        r["theta_err"] = r["theta_rel_err"]
    main = sorted([r for r in recs if r["case"].startswith("main_")],
                  key=lambda r: r["fr"])
    sens = [r for r in recs if not r["case"].startswith("main_")]

    # --- prediction P4: release angle -------------------------------------
    rows = []
    for r in main:
        rows.append({
            "case": r["case"], "fr": r["fr"], "omega": r["omega"],
            "N": r["n_total"], "dt": r["dt"], "revs": r["revs"],
            "theta_ref": r["theta_ref_deg"], "theta_got": r["theta_got_deg"],
            "theta_err": r["theta_rel_err"],
            "theta_pass_20pct": bool(r["theta_err"] < 0.20),
            "wall_velocity_ratio": r["observables"]["wall_velocity_ratio"],
            "upper_flight_fraction": r["observables"]["upper_flight_fraction"],
            "r_p05_over_R": r["observables"]["r_p05_over_R"],
            "bed_omega_over_wall": r["observables"]["bed_omega_over_wall"],
            "measured_regime": r["measured_regime"],
            "literature_regime": r["literature_regime"],
            "resting_ok": r["resting_ok"],
            "settle_rms_speed": r["settle_rms_speed"],
            "S_steady": r["S_steady"], "excess_z": r["excess_z"],
            "mode": r["mode"],
            "S_rev5": r["S_at_rev5"], "S_rev10": r["S_at_rev10"],
            "S_rev20": r["S_at_rev20"],
        })

    # --- P1..P3 trends ----------------------------------------------------
    fr = [r["fr"] for r in rows]
    p1 = spearman(fr, [r["wall_velocity_ratio"] for r in rows])
    p2 = spearman(fr, [r["upper_flight_fraction"] for r in rows])
    p3 = spearman(fr, [r["r_p05_over_R"] for r in rows])

    # --- centrifuging onset (first Fr with r_p05 >= 0.75) ------------------
    onset = None
    for r in sorted(main, key=lambda x: x["fr"]):
        if r["observables"]["r_p05_over_R"] >= 0.75:
            onset = r["fr"]
            break
    frc_err = abs(onset - 1.0) / 1.0 if onset is not None else float("nan")

    # --- regime label agreement (qualitative, literature order of magnitude)
    regime_pairs = [(r["measured_regime"], r["literature_regime"]) for r in main]

    def family(label):
        return {"slipping": "rolling", "rolling": "rolling", "cascading": "rolling",
                "cataracting": "cataracting", "centrifuging": "centrifuging"}[label]

    agree = sum(family(a) == family(b) for a, b in regime_pairs)
    theta_pass = sum(r["theta_pass_20pct"] for r in rows)

    # --- sensitivity: S(rev) plateau --------------------------------------
    plat = {}
    by = {r["case"]: r for r in sens}
    if "rev20_adhoc" in by:
        plat = {"S_rev5": by["rev20_adhoc"]["S_at_rev5"],
                "S_rev10": by["rev20_adhoc"]["S_at_rev10"],
                "S_rev20": by["rev20_adhoc"]["S_at_rev20"]}
    r20 = next((r for r in sens if r["case"] == "rev20"), None)
    prof10 = next((r for r in main if abs(r["fr"] - 0.1) < 1e-9), None)
    if r20 is not None:
        plat = {"S_rev5": r20["S_at_rev5"], "S_rev10": r20["S_at_rev10"],
                "S_rev20": r20["S_at_rev20"],
                "S_steady_10rev_case": prof10["S_steady"] if prof10 else None}

    summary = {
        "n_main_cases": len(main),
        "theta_pass_20pct": f"{theta_pass}/{len(rows)}",
        "theta_max_err": max((r["theta_err"] for r in rows
                              if math.isfinite(r["theta_err"])), default=None),
        "centrifuging_onset_measured": onset,
        "centrifuging_onset_rel_err": frc_err,
        "P1_spearman_wall_ratio_vs_fr": p1,
        "P2_spearman_upper_flight_vs_fr": p2,
        "P3_spearman_r_p05_vs_fr": p3,
        "regime_family_agreement": f"{agree}/{len(main)}",
        "regime_pairs": regime_pairs,
        "all_cases_resting_ok": all(r["resting_ok"] for r in recs),
        "settle_rms_max": max(r["settle_rms_speed"] for r in recs),
        "plateau": plat,
        "sensitivity": [
            {"case": r["case"], "fr": r["fr"], "revs": r["revs"], "dt": r["dt"],
             "n_total": r["n_total"],
             "theta_rel_err": r.get("theta_rel_err"),
             "r_p05_over_R": r["observables"]["r_p05_over_R"],
             "S_steady": r["S_steady"], "mode": r["mode"]} for r in sens],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    (OUT / "merged.json").write_text(json.dumps(
        {"main": [{k: r[k] for k in ("case", "fr", "revs", "dt", "n_total",
                                     "theta_ref_deg", "theta_got_deg",
                                     "theta_rel_err", "observables", "measured_regime",
                                     "literature_regime", "resting_ok",
                                     "settle_rms_speed", "S_steady", "excess_z",
                                     "mode", "series")} for r in main],
         "sensitivity": summary["sensitivity"]}, indent=1))

    # --- table ------------------------------------------------------------
    lines = ["| Fr | N | theta_ref [deg] | theta_got [deg] | err | <20% | r_p05/R | wall ratio | upper flight | regime (measured / literature) | resting_ok | S | mode |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['fr']:g} | {r['N']} | {r['theta_ref']:.1f} | {r['theta_got']:.1f} | "
            f"{r['theta_err']:.1%} | {'PASS' if r['theta_pass_20pct'] else 'FAIL'} | "
            f"{r['r_p05_over_R']:.2f} | {r['wall_velocity_ratio']:.2f} | "
            f"{r['upper_flight_fraction']:.3f} | {r['measured_regime']} / {r['literature_regime']} | "
            f"{r['resting_ok']} | {r['S_steady']:.3f} | {r['mode']} |")
    (OUT / "table_main.md").write_text("\n".join(lines) + "\n")

    # --- figures ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG.mkdir(parents=True, exist_ok=True)
        f = np.array(fr)
        th_ref = np.array([r["theta_ref"] for r in rows])
        th_got = np.array([r["theta_got"] for r in rows])
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ff = np.linspace(0.01, 1.4, 200)
        ax.plot(ff, np.degrees(np.arccos(np.clip(-ff, -1, 1))), "k-",
                label=r"ref $\theta_d=\arccos(-Fr)$")
        ax.fill_between(ff, np.degrees(np.arccos(np.clip(-ff, -1, 1))) * 0.8,
                        np.degrees(np.arccos(np.clip(-ff, -1, 1))) * 1.2,
                        color="0.8", alpha=0.5, label="+/-20% band")
        ax.plot(f, th_got, "o", ms=6, color="C3", label="DEM wall-contact arc")
        ax.set_xlabel("Froude number Fr = $\\omega^2 R/g$")
        ax.set_ylabel("release angle from 6 o'clock [deg]")
        ax.set_title("CASE-ROTATING-DRUM: release angle vs Froude number")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(FIG / "fig1_release_angle.png", dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
        for ax, key, lab in zip(axes,
                                ("wall_velocity_ratio", "upper_flight_fraction",
                                 "r_p05_over_R"),
                                ("wall-layer speed / (omega R)",
                                 "ballistic fraction above centre",
                                 "5th percentile of r/R")):
            ax.plot(f, [r[key] for r in rows], "o-")
            ax.axvline(1.0, color="0.6", ls="--", label="Fr=1")
            ax.set_xlabel("Fr"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
        axes[0].legend()
        fig.suptitle("measured kinematic observables (independent of Fr in the classifier)")
        fig.tight_layout(); fig.savefig(FIG / "fig2_observables.png", dpi=140)
        plt.close(fig)

        if r20 is not None:
            fig, ax = plt.subplots(figsize=(6.2, 4.0))
            s = r20["series"]
            ax.plot(s["rev"], s["segregation"], "-", label="S(rev), Fr=0.1, 20 rev")
            if prof10 is not None:
                ax.plot(prof10["series"]["rev"], prof10["series"]["segregation"],
                        "--", label="S(rev), Fr=0.1, 10 rev")
            ax.axhline(0.0, color="0.5", lw=0.8)
            ax.set_xlabel("revolutions"); ax.set_ylabel("segregation S")
            ax.set_title("segregation metric vs revolutions (plateau check)")
            ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(FIG / "fig3_S_rev.png", dpi=140)
            plt.close(fig)
    except Exception as exc:                     # pragma: no cover
        (OUT / "figure_error.txt").write_text(repr(exc))

    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()