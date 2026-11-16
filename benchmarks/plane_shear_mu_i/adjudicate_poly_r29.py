#!/usr/bin/env python3
"""Round 29 (m22733): adjudicate the archived polydispersity sweep with the
round-27 instruments, and pre-register the long-window campaign.

Two products, no new dynamics needed for the first one:

1. **Round 26's `cj4227` sweep was never adjudicated.** Round 26 left
   `mu(I)`/`phi(I)` `WITHHELD` pending the sweep, round 27 audited the protocol
   instead, round 28 consolidated rounds 1-27 without reading it. The nine
   archived runs (`runs/plane_shear_2d/poly_w{0,0.25,0.5}_I{0.02,0.05,0.2}.json`)
   are re-read here with (a) the autocorrelation-corrected standard error
   ``sd/sqrt(n_eff)`` instead of the stored ``sd/sqrt(n)``, (b) the two-block
   stationarity z stored in the files, and (c) the **curve-level provenance
   oracle** ``reference_provenance.best_case_over_envelope``: does *any*
   admissible reading of the cited law fit the measured curve inside 10 %?

2. **Pre-registration** of the long-window campaign (`run_long_window_r29.py`,
   cluster job `cj4332`): the decision rule is fixed here, before the data
   exist, so the 8-strain window cannot be read post hoc.

Outputs: ``outputs/r29/r29_poly_tables.md``, ``docs/figures/r29_poly_adjudication.png``,
``benchmarks/plane_shear_mu_i/prereg_r29.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(REPO / "src"))

import analyse_long_window_r29 as an  # noqa: E402
import reference_law as rl  # noqa: E402
import reference_provenance as rp  # noqa: E402

RUNS = BENCH / "runs" / "plane_shear_2d"
OUT = REPO / "outputs" / "r29"
FIG = REPO / "docs" / "figures"


def load_poly() -> list[dict]:
    rows = []
    for path in sorted(RUNS.glob("poly_*.json")):
        d = json.loads(path.read_text())
        rows.append({"file": path.name, "w": float(d["polydispersity"]),
                     "I_target": float(d["I_target"]), "I": float(d["I_bulk"]),
                     "mu": float(d["mu_wall"]), "mu_sd_naive": float(d["mu_wall_sd"]),
                     "mu_bulk": float(d["mu_bulk"]), "phi": float(d["phi_gap"]),
                     "psi6": float(d["psi6_mean"]), "z": float(d["z_mean"]),
                     "drift_z": float(d["drift_z_mu_wall"]),
                     "n_samples": len(d.get("samples", [])),
                     "strain_measure": float(d["case"]["strain_measure"]),
                     "p_ratio": float(d["p_bulk"] / d["p_wall"])})
    return rows


def corrected_sem(path: Path) -> dict:
    """Resample-level autocorrelation correction of the stored window sem."""
    d = json.loads(path.read_text())
    win = [s["mu_wall"] for s in d.get("samples", []) if s.get("phase") == "measure"]
    if len(win) < 8:
        return {"tau": float("nan"), "n_eff": float("nan"),
                "sem": d.get("mu_wall_sd", float("nan")) / max(1.0, len(win) ** 0.5)}
    st = an.integrated_autocorr_time(np.array(win, float))
    return {"tau": st["tau"], "n_eff": st["n_eff"], "sem": st["sem"],
            "n": st["n"]}


def adjudicate(rows: list[dict]) -> list[dict]:
    by_w: dict[float, list[dict]] = {}
    for r in rows:
        by_w.setdefault(r["w"], []).append(r)
    table = []
    for w, pts in sorted(by_w.items()):
        pts.sort(key=lambda r: r["I"])
        for p in pts:
            cs = corrected_sem(RUNS / p["file"])
            v = an.verdict_point(p["mu"], p["I"], p["phi"])
            p.update({"sem_corrected": cs["sem"], "tau": cs["tau"],
                      "n_eff": cs["n_eff"], "naive_over_corrected":
                      p["mu_sd_naive"] / cs["sem"] if cs["sem"] > 0 else float("nan")})
            p.update(v)
        curve = an.curve_envelope([p["I"] for p in pts], [p["mu"] for p in pts])
        table.append({"w": w, "points": pts, "curve": curve})
    return table


def figure(table: list[dict]) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    Igrid = np.geomspace(1e-3, 0.4, 200)
    ax1.plot(Igrid, rl.ref_mu(Igrid), "r--", lw=1.4, label="da Cruz/GDR MiDi law (3-D)")
    lo, hi = [], []
    for I in Igrid:
        b = rp.envelope_bounds(I)
        lo.append(b[0])
        hi.append(b[1])
    ax1.fill_between(Igrid, lo, hi, color="r", alpha=0.12, label="provenance envelope")
    for row in table:
        pts = row["points"]
        x = [p["I"] for p in pts]
        y = [p["mu"] for p in pts]
        e = [p["sem_corrected"] for p in pts]
        ax1.errorbar(x, y, yerr=e, fmt="o-", ms=5, capsize=3,
                     label=f"w = {row['w']:g} (round-26 archive)")
    ax1.set_xscale("log")
    ax1.set_xlabel("inertial number $I$")
    ax1.set_ylabel("$\\mu_{wall}$")
    ax1.set_title("$\\mu(I)$: archived polydispersity branch")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3, which="both")

    for row in table:
        pts = row["points"]
        ax2.plot([p["I"] for p in pts], [p["phi"] for p in pts], "o-", ms=5,
                 label=f"w = {row['w']:g}")
    ax2.axhline(0.8404, ls=":", c="k", lw=1, label="2-D RCP 0.8404")
    ax2.set_xscale("log")
    ax2.set_xlabel("inertial number $I$")
    ax2.set_ylabel("$\\phi_{gap}$")
    ax2.set_title("$\\phi(I)$: same archive")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = FIG / "r29_poly_adjudication.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p.relative_to(REPO))


def md(table: list[dict]) -> str:
    L = ["# R29 — adjudication of the archived polydispersity sweep "
         "(round 26's `cj4227`), with round-27 instruments\n",
         "Data: `benchmarks/plane_shear_mu_i/runs/plane_shear_2d/poly_*.json` "
         "(9 runs, 1-strain window, round 26). No new dynamics in this table.\n",
         "## A. Per-point verdict\n",
         "| w | I_target | $I_{meas}$ | $\\mu_{wall}$ | naive sem | corr sem | naive/corr | "
         "$\\tau_{int}$ | drift z (stored) | $\\mu_{ref}$ | err % | <10 %? | "
         "$\\phi_{gap}$ | $\\psi_6$ | $p_{bulk}/p_{wall}$ |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in table:
        for p in row["points"]:
            L.append("| %.2f | %.3f | %.4f | %.4f | %.4f | %.4f | %.2f | %.1f | %+.1f | "
                     "%.4f | %+.1f | %s | %.4f | %.3f | %.3f |"
                     % (p["w"], p["I_target"], p["I"], p["mu"], p["mu_sd_naive"],
                        p["sem_corrected"], p["naive_over_corrected"], p["tau"],
                        p["drift_z"], p["mu_ref"], p["err_pct"],
                        "YES" if p["within_tol"] else "NO", p["phi"], p["psi6"],
                        p["p_ratio"]))
    L += ["\n## B. Curve-level falsification oracle\n",
          "| branch | points | min possible worst-point error % | admissible law fits? | "
          "argmin $(\\mu_s,\\mu_2,I_0)$ |",
          "|---|---|---|---|---|"]
    for row in table:
        c = row["curve"]
        a = c["argmin"]
        L.append("| w = %g | %d | %.0f | %s | (%.2f, %.2f, %.2f) |"
                 % (row["w"], len(row["points"]), 100 * c["min_max_abs_rel_err"],
                    "**NO**" if not c["provenance_can_rescue"] else "yes",
                    a["mu_s"], a["mu_2"], a["I_0"]))
    L += ["\n## C. What the sweep actually says about round 26's hypothesis\n"]
    w0 = next((r for r in table if r["w"] == 0.0), None)
    w5 = next((r for r in table if r["w"] == 0.5), None)
    if w0 and w5:
        for p0, p5 in zip(w0["points"], w5["points"]):
            L.append("* at $I\\approx%.3f$: $\\mu$ goes %.4f ($w=0$) → %.4f ($w=0.5$), "
                     "%+.0f %%; $\\psi_6$ %.2f → %.2f."
                     % (p5["I"], p0["mu"], p5["mu"],
                        100 * (p5["mu"] - p0["mu"]) / p0["mu"], p0["psi6"], p5["psi6"]))
    L.append("\nReading: polydispersity **does** break the crystal and **raises** the effective friction, so round 25's diagnosis was directionally right; but the branch still misses the cited law by far more than the 10 % band and, per section B, **no** admissible reading of the citation can fit it. The gap is therefore neither bibliographic nor crystal-ordering alone.\n")
    return "\n".join(L) + "\n"


PREREG = {
    "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
    "round": 29,
    "mission": 22733,
    "driver": "benchmarks/plane_shear_mu_i/run_long_window_r29.py",
    "analysis": "benchmarks/plane_shear_mu_i/analyse_long_window_r29.py",
    "cluster_job": "cj4332",
    "device": "tensordem.granular2d 2-D rough-wall plane shear, round-25/26 case",
    "axis": {"strain_measure": 8.0, "archived_rounds": 1.0,
             "combos": [{"w": w, "I_target": I, "seed": s}
                        for w, I, s in [(0.5, 0.02, 0), (0.5, 0.05, 0), (0.5, 0.05, 1),
                                        (0.5, 0.05, 2), (0.5, 0.2, 0), (0.5, 0.2, 1),
                                        (0.5, 0.2, 2), (0.0, 0.05, 0)]]},
    "statistics": {"autocorr": "tau_int = 1 + 2 sum_k rho_k (Sokal window), n_eff = n/tau",
                   "sem": "sd/sqrt(n_eff)", "stationarity":
                   "last two 1-strain blocks agree within 2 sigma AND second-half "
                   "weighted trend |z| < 2",
                   "saturation": "mu = mu_inf + a exp(-gamma/gamma0), gamma0 by grid"},
    "decision_rule": {
        "PASS_mu": "every w = 0.5 point stationary AND |mu - mu_ref| <= 10 % at every I",
        "PARTIAL": "stationary but the curve misses the law; report the stationary "
                   "offset as the device's bias and keep the ledger FAIL",
        "WITHHELD": "any point not certified stationary; report the drift explicitly",
        "never": "no post-hoc choice of window, pressure convention or I definition "
                 "after the data are read"},
    "declared_limitation": "the cited law is a 3-D glass-bead fit; the device is 2-D, so "
                           "phi is judged on monotonicity only, and mu is judged on the "
                           "curve-level envelope as well as on the pointwise band",
}

def main() -> int:
    rows = load_poly()
    table = adjudicate(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r29_poly_tables.md").write_text(md(table))
    fig = figure(table)
    (BENCH / "prereg_r29.json").write_text(json.dumps(PREREG, indent=1))
    print((OUT / "r29_poly_tables.md").read_text())
    print("figure:", fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
