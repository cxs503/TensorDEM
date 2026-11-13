"""CASE-DEM-PLANE-SHEAR-MUIDI round 8: momentum-balance-validated bulk stress.

Round 7 left two defects open and named the first as the next round's target:

* D1 (estimator validity): the thin-slab Love-Weber sum counts only contacts
  whose *both* endpoints lie in the slab, and on a 2-4 diameter slab it drops
  41-49 % of the load-bearing chains.  ``p_bulk/P_applied - 1`` was -17 ... -51 %,
  which could not be separated from a genuine stress deficit.
* D2 (verdict basis): ``mu`` was quoted from wall traction and from that
  truncated slab average, i.e. from two quantities with no independent check.

This round adds the missing check and re-measures the same four points:

1. ``full_cell_virial_stress`` -- the domain average over *all* particle-particle
   contacts, normalised by the centre-accessible volume ``A (h - 2r)``.  For a
   packing in static equilibrium at zero gravity the virial identity gives
   ``sigma_zz = -P_applied`` **exactly**, so ``sigma_zz/P = -1`` is an analytic
   reference that involves no literature constant.  It is evaluated on every
   sample; the final-window value is the estimator's calibration.
2. ``bulk_virial_stress(rule="midpoint")`` -- the R7-recommended midpoint rule for
   the thin slab, run on the same samples so the two rules are compared on
   identical states.
3. ``mu`` is reported against ``p = -tr(sigma)/3`` and against ``-sigma_zz``
   separately (the da Cruz construction is plane shear with the transverse
   normal stress; the two differ by the normal-stress anisotropy, so the choice
   must be stated rather than implied).

Pre-registered verdicts (``prereg.json``, written before the sweep):

* P1 (estimator calibration): ``|sigma_zz^full/P + 1| <= 0.05`` in the final
  window at every point.
* P2 (truncation): the midpoint rule moves ``mu_slab`` toward ``mu_full`` and
  the legay both-endpoints rule is the low outlier at every point.
* P3 (verdict): ``|mu_full/mu_ref - 1| <= 10 %`` at every point would pass the
  ledger acceptance; the honest expectation from round 6 is FAIL at I >= 0.1.

Usage::

    python benchmarks/plane_shear_mu_i/full_cell_rheometry.py            # all 4 points
    python benchmarks/plane_shear_mu_i/full_cell_rheometry.py --point 0.1 # one point
    python benchmarks/plane_shear_mu_i/full_cell_rheometry.py --assemble  # tables/fig only
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from reference_law import ref_mu as _ref_mu, ref_phi as _ref_phi, reference  # noqa: E402

RUNS = HERE / "runs" / "full_cell"
RUNS.mkdir(parents=True, exist_ok=True)

# identical device parameters to round 6 (no contact law / protocol change)
MU_S, MU_2, I_0 = 0.36, 0.84, 0.30
KN = 6.25e4
MU_CONTACT = 0.5
MU_ROLLING = 0.05
E = 0.3
P_APPLIED = 6.25e4
W, D, H = 0.12, 0.05, 0.06
RADIUS = 0.005
SEED = 5279
I_TARGETS = (1e-2, 3e-2, 1e-1, 3e-1)
SETTLE_STEPS = 2500
MIN_SHEAR_STEPS = 2500
SHEAR_STRAIN = 0.2
TOL_P1 = 0.05

PREREG = {
    "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
    "round": 8,
    "device": "tensordem.mu_i_rheology.simulate_servo_shear (bulk_stress=True)",
    "points": {"arm": "baseline", "I_target": list(I_TARGETS), "kn": KN,
               "normal_stress_pa": P_APPLIED, "seed": SEED},
    "reference": reference(),
    "hypotheses": {
        "P1_estimator_calibration": "|sigma_zz_full / P_applied + 1| <= 0.05 "
                                    "(final 20 % window, every point)",
        "P2_truncation": "mu_slab(both) < mu_slab(midpoint) < mu_full at every point",
        "P3_ledger_verdict": "|mu_full/mu_ref - 1| <= 10 % at every point",
    },
    "analytic_anchor": "virial identity: sum_pairs f (x) (x_j - x_i) = -sum_i F_ext (x) x_i "
                       "=> sigma_zz * A (h - 2r) = -P A (h - 2r)",
}


def mu_ref(I: float) -> float:
    return _ref_mu(I)


def run_point(I: float) -> dict:
    from tensordem.mu_i_rheology import ServoShearCase, simulate_servo_shear

    case = ServoShearCase(
        radius=RADIUS, density=2500.0, mu=MU_CONTACT, mu_rolling=MU_ROLLING,
        restitution=E, kn=KN, kt_ratio=0.5, box_width=W, box_depth=D,
        fill_height=H, normal_stress=P_APPLIED, inertial_number=I, seed=SEED,
        settle_steps=SETTLE_STEPS, min_shear_steps=MIN_SHEAR_STEPS,
        shear_strain=SHEAR_STRAIN, bulk_stress=True)
    res = simulate_servo_shear(case)
    row = {
        "I_target": I,
        "I_measured": res["inertial_number"],
        "I_bulk": res["inertial_number_bulk"],
        "mu_wall": res["mu_effective"],
        "mu_wall_final": res["mu_final"],
        "mu_slab_both": res["bulk_mu"],
        "mu_slab_both_final": res["bulk_mu_final"],
        "mu_slab_mid": res["bulk_mid_mu"],
        "mu_slab_mid_final": res["bulk_mid_mu_final"],
        "mu_full": res["full_mu"],
        "mu_full_final": res["full_mu_final"],
        "mu_full_zz": res["full_mu_zz"],
        "mu_full_zz_final": res["full_mu_zz_final"],
        "anisotropy_p_over_zz": res["full_anisotropy"],
        "p_slab_both": res["bulk_pressure_pa"],
        "p_slab_mid": res["bulk_mid_pressure_pa"],
        "p_full": res["full_pressure_pa"],
        "sigma_zz_full_final": res["full_sigma_zz_final"],
        "momentum_balance_tail": res["full_momentum_balance"],
        "momentum_balance_final": res["full_momentum_balance_final"],
        "kinetic_p_full": res["full_kinetic_pressure_pa"],
        "n_contacts_full": res["full_n_contacts"],
        "straddle_fraction": res["bulk_straddle_fraction"],
        "n_slab_both": None,
        # solid fraction (unchanged estimators, reported for completeness)
        "phi_slab": res["solid_fraction"],
        "phi_bed_steady": res["phi_bed_steady"],
        "dilatancy": res["dilatancy"],
        "coordination": res["coordination_number"],
        "n_particles": res["n_particles"],
        "shear_rate_imposed": res["shear_rate_imposed"],
        "shear_rate_measured": res["shear_rate_measured"],
        "profile_r2": res["profile_r2"],
        "pressure_control_error_final": res["pressure_control_error_final"],
        "servo_valid": res["servo_valid"],
        "recover_steps": res["recover_steps"],
        "wall_clock_s": res["wall_clock_s"],
        "mu_ref": mu_ref(I),
        "phi_ref": _ref_phi(I),
    }
    row["err_wall_pct"] = 100.0 * (row["mu_wall"] / row["mu_ref"] - 1.0)
    row["err_slab_both_pct"] = 100.0 * (row["mu_slab_both"] / row["mu_ref"] - 1.0)
    row["err_slab_mid_pct"] = 100.0 * (row["mu_slab_mid"] / row["mu_ref"] - 1.0)
    row["err_full_pct"] = 100.0 * (row["mu_full"] / row["mu_ref"] - 1.0)
    row["err_full_zz_pct"] = 100.0 * (row["mu_full_zz"] / row["mu_ref"] - 1.0)
    row["p_balance_slab_both"] = row["p_slab_both"] / P_APPLIED - 1.0
    row["p_balance_slab_mid"] = row["p_slab_mid"] / P_APPLIED - 1.0
    row["mu_slab_mid_over_full"] = row["mu_slab_mid"] / row["mu_full"]
    row["mu_slab_both_over_full"] = row["mu_slab_both"] / row["mu_full"]
    row["rate_ratio"] = (row["shear_rate_measured"] / row["shear_rate_imposed"]
                         if row["shear_rate_imposed"] else float("nan"))
    return row


def write_tables(rows: list[dict]) -> str:
    lines = ["# R8 momentum-balance-calibrated bulk rheometry — programmatic tables", "",
             "> generated by `benchmarks/plane_shear_mu_i/full_cell_rheometry.py`; "
             "every number is written by code, none is hand-copied.", "",
             "## (1) estimator calibration: does the virial reproduce the applied load?", "",
             "| I_target | sigma_zz full [Pa] | sigma_zz/(-P) final | sigma_zz/(-P) tail | P1 |",
             "|---|---|---|---|---|"]
    for r in rows:
        ok = "pass" if abs(1.0 - r["momentum_balance_final"]) <= TOL_P1 else "FAIL"
        lines.append(f"| {r['I_target']:.3g} | {r['sigma_zz_full_final']:.0f} | "
                     f"{-P_APPLIED:.0f} | {r['momentum_balance_final']:.4f} | "
                     f"{r['momentum_balance_tail']:.4f} | {ok} |")
    lines += ["", "## (2) the three estimators on identical samples (60 % tail)", "",
              "| I_target | p slab both [Pa] | p slab mid [Pa] | p full [Pa] | p bal both % | p bal mid % |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['I_target']:.3g} | {r['p_slab_both']:.0f} | "
                     f"{r['p_slab_mid']:.0f} | {r['p_full']:.0f} | "
                     f"{100 * r['p_balance_slab_both']:+.1f} | "
                     f"{100 * r['p_balance_slab_mid']:+.1f} |")
    lines += ["", "## (3) mu against the da Cruz 2005 law (10 % ledger band)", "",
              "`mu_full` normalises by the mean pressure p = -tr(sigma)/3; `mu_full_zz` by "
              "the normal stress in the shear plane, |sigma_xz|/|sigma_zz| -- the "
              "3-D analogue of the 2-D sigma_xy/sigma_yy of the reference.", "",
              "| I_target | mu ref | mu wall | mu slab both | mu slab mid | mu full | mu full (sigma_zz) | err wall % | err both % | err mid % | err full % | err full zz % |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['I_target']:.3g} | {r['mu_ref']:.4f} | {r['mu_wall']:.4f} | "
                     f"{r['mu_slab_both']:.4f} | {r['mu_slab_mid']:.4f} | "
                     f"{r['mu_full']:.4f} | {r['mu_full_zz']:.4f} | "
                     f"{r['err_wall_pct']:+.1f} | "
                     f"{r['err_slab_both_pct']:+.1f} | {r['err_slab_mid_pct']:+.1f} | "
                     f"{r['err_full_pct']:+.1f} | {r['err_full_zz_pct']:+.1f} |")
    lines += ["", "## (4) truncation ordering P2 and strain state", "",
              "| I_target | mu both/mu full | mu mid/mu full | straddle fraction | rate ratio | profile R2 | servo valid | wall clock [s] |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p2 = "pass" if (r["mu_slab_both_over_full"] < r["mu_slab_mid_over_full"] < 1.0) else "FAIL"
        lines.append(f"| {r['I_target']:.3g} | {r['mu_slab_both_over_full']:.3f} | "
                     f"{r['mu_slab_mid_over_full']:.3f} | "
                     f"{100 * (r['straddle_fraction'] or 0.0):.1f} % | "
                     f"{r['rate_ratio']:.2f} | {r['profile_r2']:.2f} | "
                     f"{r['servo_valid']} | {r['wall_clock_s']:.0f} |")
    lines += ["", "## (5) solid fraction (estimators unchanged from R7)", "",
              "| I_target | phi slab | phi bed | phi ref | coordination | dilatancy % |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['I_target']:.3g} | {r['phi_slab']:.4f} | "
                     f"{r['phi_bed_steady']:.4f} | {r['phi_ref']:.4f} | "
                     f"{r['coordination']:.2f} | {100 * r['dilatancy']:+.2f} |")
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
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.1))
    ax = axes[0]
    ax.semilogx(grid, [_ref_mu(x) for x in grid], "k-", label="da Cruz 2005 law")
    ax.semilogx(I, [r["mu_wall"] for r in rows], "o--", label="wall traction")
    ax.semilogx(I, [r["mu_slab_both"] for r in rows], "v:", label="slab, both-endpoints (R6)")
    ax.semilogx(I, [r["mu_slab_mid"] for r in rows], "^-.", label="slab, midpoint (R8)")
    ax.semilogx(I, [r["mu_full"] for r in rows], "s-", label="full cell, p=-tr/3 (R8)")
    ax.semilogx(I, [r["mu_full_zz"] for r in rows], "D:", label=r"full cell, $\sigma_{zz}$ (R8)")
    ax.set_xlabel("I  [-]"); ax.set_ylabel(r"$\mu$"); ax.legend(fontsize=7)
    ax.grid(alpha=.3); ax.set_title(r"$\mu(I)$: three estimators vs the law")
    ax = axes[1]
    ax.axhline(1.0, color="k", lw=1.0, label=r"exact: $\sigma_{zz}=-P$")
    ax.axhspan(0.95, 1.05, color="g", alpha=.15, label="+/-5 % (P1)")
    ax.semilogx(I, [r["momentum_balance_final"] for r in rows], "s-",
                label="final 20 % window")
    ax.semilogx(I, [r["momentum_balance_tail"] for r in rows], "o--",
                label="60 % tail")
    ax.set_xlabel("I_target"); ax.set_ylabel(r"$\sigma_{zz}^{full}/P_{applied}$")
    ax.grid(alpha=.3); ax.set_title("momentum balance (analytic anchor)")
    ax.legend(fontsize=7)
    ax = axes[2]
    width = 0.2
    for k, key in enumerate(["err_wall_pct", "err_slab_both_pct",
                             "err_slab_mid_pct", "err_full_pct"]):
        ax.bar(np.arange(len(I)) + (k - 1.5) * width,
               [r[key] for r in rows], width, label=key.replace("err_", "").replace("_pct", ""))
    ax.axhspan(-10, 10, color="g", alpha=.15, label="+/-10 % ledger band")
    ax.set_xticks(np.arange(len(I)))
    ax.set_xticklabels([f"{x:.3g}" for x in I])
    ax.set_xlabel("I_target"); ax.set_ylabel("error vs law [%]")
    ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")
    ax.set_title("mu error by estimator")
    fig.tight_layout()
    fig.savefig(RUNS / "fig_full_cell.png", dpi=140)
    plt.close(fig)


def _point_file(I: float) -> Path:
    return RUNS / f"point_{I:.3g}.json"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    (RUNS / "prereg.json").write_text(json.dumps(PREREG, indent=2))
    if "--point" in argv:
        I = float(argv[argv.index("--point") + 1])
        row = run_point(I)
        _point_file(I).write_text(json.dumps(row, indent=2))
        print(f"I={I:.3g} mu_full={row['mu_full']:.4f} mu_ref={row['mu_ref']:.4f} "
              f"err={row['err_full_pct']:+.1f}% balance="
              f"{row['momentum_balance_final']:.4f} ({row['wall_clock_s']:.0f}s)",
              flush=True)
        return 0
    rows = []
    for I in I_TARGETS:
        f = _point_file(I)
        if not f.exists():
            row = run_point(I)
            f.write_text(json.dumps(row, indent=2))
            print(f"I={I:.3g} done ({row['wall_clock_s']:.0f}s)", flush=True)
        rows.append(json.loads(f.read_text()))
    (RUNS / "result.json").write_text(json.dumps(
        {"prereg": PREREG, "rows": rows}, indent=2))
    write_tables(rows)
    make_figure(rows)
    print("wrote", RUNS / "result.json", RUNS / "tables.md", RUNS / "fig_full_cell.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())