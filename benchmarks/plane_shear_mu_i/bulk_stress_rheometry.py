"""CASE-DEM-PLANE-SHEAR-MUIDI round 6: bulk (Love-Weber) vs wall rheometry.

Round 5 (`docs/CASE_DEM_PLANE_SHEAR_MUIDI_R5_CONSTANT_STRESS_AUDIT.md`) showed
that the reported ``mu`` of the servo shear cell is a **wall traction**
(``tau_wall / p_wall`` on the platen) and recommended replacing it by a bulk
stress measurement before the mu(I) point can be judged at all.  This script
does exactly that on the same device and the same points:

* pre-registered questions (written to ``prereg_m16758.json`` before the run)
  - P1 (estimator validity): the slab Love-Weber pressure balances the applied
    stress, ``|p_bulk/P - 1| <= 0.30`` in the tail window;
  - P2 (wall vs bulk): ``mu_bulk`` sits within 15 % of ``mu_wall`` (i.e. the wall
    traction is not the reason the R5 gate-on points read 20-42 % low);
  - P3 (strain localisation): the measured bulk shear rate is the imposed one,
    ``|gamma_dot_meas/gamma_dot - 1| <= 0.15`` within the bulk slab.

Deliverables: ``runs/bulk_rheometry/result.json``, ``tables.md``,
``fig_bulk_vs_wall.png``.

Usage: ``python benchmarks/plane_shear_mu_i/bulk_stress_rheometry.py``
"""
from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "bulk_rheometry"
RUNS.mkdir(parents=True, exist_ok=True)

# --- reference (da Cruz 2005 / GDR MiDi 2004), identical to rounds 1-5 ------
MU_S, MU_2, I_0 = 0.36, 0.84, 0.30
# --- device parameters of the R5 baseline arm (unchanged) ------------------
KN = 6.25e4
MU_CONTACT = 0.5
MU_ROLLING = 0.05
E = 0.3
P_APPLIED = 6.25e4
W, D, H = 0.12, 0.05, 0.06
SEED = 5279
I_TARGETS = (1e-2, 3e-2, 1e-1, 3e-1)
SETTLE_STEPS = 2500
MIN_SHEAR_STEPS = 2500
SHEAR_STRAIN = 0.2

PREREG = {
    "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
    "round": 6,
    "mission": 16758,
    "device": "tensordem.mu_i_rheology.simulate_servo_shear (bulk_stress=True)",
    "points": {"arm": "baseline", "I_target": list(I_TARGETS), "kn": KN,
               "normal_stress_pa": P_APPLIED, "seed": SEED},
    "hypotheses": {
        "P1_estimator_validity": "|p_bulk/P_applied - 1| <= 0.30 (tail window)",
        "P2_wall_vs_bulk": "|mu_bulk/mu_wall - 1| <= 0.15",
        "P3_no_localisation": "|gamma_dot_measured/gamma_dot_imposed - 1| <= 0.15",
    },
    "falsification": "any |mu_bulk - mu_ref| <= 10 % would overturn the R5 verdict "
                     "that the mu(I) point is WITHHELD at I >= 0.1",
}


def mu_ref(I: float) -> float:
    return MU_S + (MU_2 - MU_S) / (I_0 / I + 1.0)


def run_point(I: float) -> dict:
    from tensordem.mu_i_rheology import ServoShearCase, simulate_servo_shear

    case = ServoShearCase(
        radius=0.005, density=2500.0, mu=MU_CONTACT, mu_rolling=MU_ROLLING,
        restitution=E, kn=KN, kt_ratio=0.5, box_width=W, box_depth=D,
        fill_height=H, normal_stress=P_APPLIED, inertial_number=I, seed=SEED,
        settle_steps=SETTLE_STEPS, min_shear_steps=MIN_SHEAR_STEPS,
        shear_strain=SHEAR_STRAIN, bulk_stress=True)
    res = simulate_servo_shear(case)
    row = {
        "I_target": I,
        "I_measured": res["inertial_number"],
        "mu_wall": res["mu_effective"],
        "mu_wall_final": res["mu_final"],
        "p_wall": res["normal_stress_measured_pa"],
        "tau_wall": res["shear_stress_pa"],
        "mu_bulk": res["bulk_mu"],
        "mu_bulk_final": res["bulk_mu_final"],
        "p_bulk": res["bulk_pressure_pa"],
        "tau_bulk": res["bulk_shear_pa"],
        "p_bulk_final": res["bulk_pressure_final_pa"],
        "bulk_kinetic_p": res["bulk_kinetic_pressure_pa"],
        "sigma_zz_bulk": res["bulk_sigma_zz_final"],
        "shear_rate_imposed": res["shear_rate_imposed"],
        "shear_rate_measured": res["shear_rate_measured"],
        "profile_r2": res["profile_r2"],
        "I_bulk": res["inertial_number_bulk"],
        "phi_bed_final": res["phi_bed_final"],
        "coordination": res["coordination_number"],
        "n_particles": res["n_particles"],
        "pressure_control_error_final": res["pressure_control_error_final"],
        "servo_valid": res["servo_valid"],
        "recover_steps": res["recover_steps"],
        "wall_clock_s": res["wall_clock_s"],
        "mu_ref": mu_ref(I),
    }
    row["mu_wall_err_pct"] = 100.0 * (row["mu_wall"] / row["mu_ref"] - 1.0)
    row["mu_bulk_err_pct"] = 100.0 * (row["mu_bulk"] / row["mu_ref"] - 1.0)
    row["mu_bulk_err_final_pct"] = 100.0 * (row["mu_bulk_final"] / row["mu_ref"] - 1.0)
    row["p_balance_rel"] = row["p_bulk"] / P_APPLIED - 1.0
    row["mu_bulk_over_wall"] = (row["mu_bulk"] / row["mu_wall"]
                                if row["mu_wall"] else float("nan"))
    row["rate_ratio"] = (row["shear_rate_measured"] / row["shear_rate_imposed"]
                         if row["shear_rate_imposed"] else float("nan"))
    return row


def write_tables(rows: list[dict]) -> str:
    lines = ["# R6 bulk versus wall rheometry — programmatic tables", "",
             "> generated by `benchmarks/plane_shear_mu_i/bulk_stress_rheometry.py`; "
             "no number here is hand-copied.", "",
             "## (1) same point, two stress estimators (tail 60 % window)", "",
             "| I_target | I_meas | mu ref | mu wall | mu bulk | err wall % | err bulk % | p wall [Pa] | p bulk [Pa] | p balance % |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['I_target']:.3g} | {r['I_meas']:.4g} | {r['mu_ref']:.4f} | "
            f"{r['mu_wall']:.4f} | {r['mu_bulk']:.4f} | {r['mu_wall_err_pct']:+.1f} | "
            f"{r['mu_bulk_err_pct']:+.1f} | {r['p_wall']:.0f} | {r['p_bulk']:.0f} | "
            f"{100 * r['p_balance_rel']:+.1f} |")
    lines += ["", "## (2) strain localisation and I abscissa", "",
              "| I_target | gamma_dot imposed | gamma_dot measured | ratio | profile R2 | I_meas (wall p) | I bulk (bulk p, measured rate) |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['I_target']:.3g} | {r['shear_rate_imposed']:.3f} | "
            f"{r['shear_rate_measured']:.3f} | {r['rate_ratio']:.3f} | "
            f"{r['profile_r2']:.2f} | {r['I_measured']:.4g} | {r['I_bulk']:.4g} |")
    lines += ["", "## (3) pre-registered verdicts", "",
              "| point | P1 pressure balance | P2 bulk/wall mu | P3 rate ratio | servo valid | recover steps |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        p1 = "pass" if abs(r["p_balance_rel"]) <= 0.30 else "FAIL"
        p2 = "pass" if abs(r["mu_bulk_over_wall"] - 1.0) <= 0.15 else "FAIL"
        p3 = "pass" if abs(r["rate_ratio"] - 1.0) <= 0.15 else "FAIL"
        lines.append(f"| I={r['I_target']:.3g} | {p1} | {p2} | {p3} | "
                     f"{r['servo_valid']} | {r['recover_steps']} |")
    lines += ["", "## (4) error against the da Cruz (2005) mu(I) law (10 % band)", "",
              "| I_target | mu ref | mu wall err % | mu bulk err % | mu bulk (final window) err % | verdict |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        band = min(abs(r["mu_wall_err_pct"]), abs(r["mu_bulk_err_pct"]))
        lines.append(f"| {r['I_target']:.3g} | {r['mu_ref']:.4f} | "
                     f"{r['mu_wall_err_pct']:+.1f} | {r['mu_bulk_err_pct']:+.1f} | "
                     f"{r['mu_bulk_err_final_pct']:+.1f} | "
                     f"{'PASS' if band <= 10.0 else 'FAIL'} |")
    text = "\n".join(lines) + "\n"
    (RUNS / "tables.md").write_text(text)
    return text


def make_figure(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    I = np.array([r["I_target"] for r in rows])
    grid = np.logspace(-3, math.log10(0.5), 200)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax = axes[0]
    ax.semilogx(grid, [mu_ref(x) for x in grid], "k-", label="da Cruz 2005")
    ax.semilogx(I, [r["mu_wall"] for r in rows], "o--", label="wall traction")
    ax.semilogx(I, [r["mu_bulk"] for r in rows], "s-", label="bulk Love-Weber")
    ax.set_xlabel("I"); ax.set_ylabel(r"$\mu$"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("mu(I): wall vs bulk")
    ax = axes[1]
    ax.axhline(0.0, color="k", lw=.8)
    ax.semilogx(I, [100 * r["p_balance_rel"] for r in rows], "o-")
    ax.set_xlabel("I_target"); ax.set_ylabel(r"$p_{bulk}/P_{applied}-1$ [%]")
    ax.grid(alpha=.3); ax.set_title("slab pressure balance")
    ax = axes[2]
    ax.axhline(1.0, color="k", lw=.8)
    ax.semilogx(I, [r["rate_ratio"] for r in rows], "s-")
    ax.set_xlabel("I_target"); ax.set_ylabel(r"$\dot\gamma_{meas}/\dot\gamma_{imp}$")
    ax.grid(alpha=.3); ax.set_title("strain rate seen by the slab")
    fig.tight_layout()
    fig.savefig(RUNS / "fig_bulk_vs_wall.png", dpi=140)
    plt.close(fig)


def main() -> int:
    (RUNS / "prereg.json").write_text(json.dumps(PREREG, indent=2))
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(4, len(I_TARGETS))) as pool:
        for row in pool.map(run_point, I_TARGETS):
            rows.append(row)
            print(f"I={row['I_target']:.3g} mu_wall={row['mu_wall']:.4f} "
                  f"mu_bulk={row['mu_bulk']:.4f} p_bal={100*row['p_balance_rel']:+.1f}%",
                  flush=True)
    rows.sort(key=lambda r: r["I_target"])
    (RUNS / "result.json").write_text(json.dumps(
        {"case": "CASE-DEM-PLANE-SHEAR-MUIDI", "round": 6,
         "prereg": PREREG, "rows": rows}, indent=2))
    write_tables(rows)
    make_figure(rows)
    print("written:", RUNS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())