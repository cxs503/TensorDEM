"""CASE-DEM-PLANE-SHEAR-MUIDI round 7: run the round-6 pre-registration and
test the two explanations it left open (layer count, servo bandwidth).

Background
----------
Round 5 (``docs/CASE_DEM_PLANE_SHEAR_MUIDI_R5_CONSTANT_STRESS_AUDIT.md``) showed
that the shipped servo-cell ``mu`` is a wall traction and that the platen stress
drifts badly at high ``I``; with a constant-stress gate the same points read
20-42 % low.  Round 6 (m16858, ``bulk_stress_rheometry.py``) added the bulk
Love-Weber slab estimator and wrote its pre-registration
(``runs/bulk_rheometry/prereg.json``: P1 estimator validity, P2 wall vs bulk,
P3 no strain localisation) -- **but the run never executed**: the only artifact
in the tree is ``prereg.json``.  So the R5 verdict ``WITHHELD at I >= 0.1`` has
never been adjudicated by the bulk estimator it asks for.

This script

1. executes the round-6 pre-registration verbatim (baseline arm, same
   parameters, same I points) and evaluates P1/P2/P3 + its falsification line;
2. adds two pre-registered *explanations* for the residual high-I deficit that
   round 5 left untested:
   * ``B_layers`` -- layer count: the baseline bed is 5d x 6 layers.  R5 asked
     for "N >> 294, >= 10 layers".  The thick arm (8d x 9 layers, N = 924, the
     same geometry round 5 used) tests whether ``mu`` at I = 0.1 / 0.3 is
     layer-converged (H4);
   * ``C_servo``  -- servo bandwidth: the round-5 gate holds ``P`` only *after*
     the strain budget is spent (up to 20000 extra steps, strain up to ~98).
     A faster servo (stride 2, relax 0.5) asks whether ``P`` can be held
     *during* the shear phase, which would remove the long-shearing state
     change altogether (H5).
3. publishes programmatic tables + a figure, and states the verdict against the
   da Cruz et al. (2005) mu(I) law in the case's 10 % band.

Usage::

    python benchmarks/plane_shear_mu_i/round7_protocol.py            # all arms
    python benchmarks/plane_shear_mu_i/round7_protocol.py --arm A    # one arm
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs" / "round7"

# --- reference law (da Cruz 2005 / GDR MiDi 2004), identical to rounds 1-6 ---
MU_S, MU_2, I_0 = 0.36, 0.84, 0.30
# --- device parameters, identical to round 6 ---------------------------------
KN = 6.25e4
MU_CONTACT = 0.5
MU_ROLLING = 0.05
E = 0.3
P_APPLIED = 6.25e4
SEED = 5279
SETTLE_STEPS = 2500
MIN_SHEAR_STEPS = 2500
SHEAR_STRAIN = 0.2
RECOVER_CAP = 20000   # constant-stress gate cap, round-5 value
RADIUS = 0.005        # module-level so tests can shrink the specimen

# geometry arms (W x D x H in m); ``baseline`` = round 6, ``thick`` = round 5
GEOM = {
    "baseline": dict(box_width=0.12, box_depth=0.05, fill_height=0.06),
    "thick": dict(box_width=0.12, box_depth=0.08, fill_height=0.09),
}

PREREG = {
    "case": "CASE-DEM-PLANE-SHEAR-MUIDI",
    "round": 7,
    "device": "tensordem.mu_i_rheology.simulate_servo_shear (bulk_stress=True, "
              "pressure gate on)",
    "inherits": "round 6 pre-registration (runs/bulk_rheometry/prereg.json)",
    "common": dict(kn=KN, normal_stress_pa=P_APPLIED, mu_contact=MU_CONTACT,
                   mu_rolling=MU_ROLLING, restitution=E, seed=SEED,
                   settle_steps=SETTLE_STEPS, min_shear_steps=MIN_SHEAR_STEPS,
                   shear_strain=SHEAR_STRAIN, max_recover_steps=RECOVER_CAP,
                   radius=RADIUS, geometries=GEOM),
    "arms": {
        "A_baseline": dict(geom="baseline", I=(1e-2, 3e-2, 1e-1, 3e-1),
                           servo=None),
        "B_layers": dict(geom="thick", I=(1e-1, 3e-1), servo=None),
        "C_servo": dict(geom="baseline", I=(3e-1,), servo=dict(servo_stride=2,
                                                               servo_relax=0.5)),
    },
    "hypotheses": {
        "H1_pressure_balance": "|p_bulk/P_applied - 1| <= 0.30 on every A point "
                               "(round-6 P1, the bulk estimator is a stress)",
        "H2_wall_vs_bulk": "|mu_bulk/mu_wall - 1| <= 0.15 on every A point "
                           "(round-6 P2)",
        "H3_no_localisation": "|gamma_dot_meas/gamma_dot_imp - 1| <= 0.15 on "
                              "every A point (round-6 P3)",
        "H4_layer_convergence": "at I = 0.3 and I = 0.1, |mu_bulk(thick) - "
                                "mu_bulk(baseline)| <= 0.10*mu_ref -- if false "
                                "the 10 % band test is a finite-size test, not "
                                "a rheology test",
        "H5_servo_bandwidth": "at I = 0.3 the fast servo (stride 2, relax 0.5) "
                              "cuts recover_steps by >= 5x AND ends with "
                              "|p_wall/P - 1| <= 0.05",
    },
    "falsification": "any A point with |mu_bulk - mu_ref| <= 10 % overturns the "
                     "R5 verdict 'WITHHELD at I >= 0.1'",
}


def mu_ref(I: float) -> float:
    return MU_S + (MU_2 - MU_S) / (I_0 / I + 1.0)


def _f(d: dict, key: str, default=float("nan")):
    v = d.get(key, default)
    return default if v is None else v


def run_point(spec: tuple[str, str, float]) -> dict:
    """One (arm, geometry, I) point of the servo shear cell."""
    from tensordem.mu_i_rheology import ServoShearCase, simulate_servo_shear

    arm, geom, I = spec
    servo = PREREG["arms"][arm]["servo"] or {}
    t0 = time.time()
    case = ServoShearCase(
        radius=RADIUS, density=2500.0, mu=MU_CONTACT, mu_rolling=MU_ROLLING,
        restitution=E, kn=KN, kt_ratio=0.5, normal_stress=P_APPLIED,
        inertial_number=I, seed=SEED, settle_steps=SETTLE_STEPS,
        min_shear_steps=MIN_SHEAR_STEPS, shear_strain=SHEAR_STRAIN,
        bulk_stress=True, max_recover_steps=RECOVER_CAP, **GEOM[geom], **servo)
    res = simulate_servo_shear(case)
    ref = mu_ref(I)
    row = {
        "arm": arm, "geom": geom, "I_target": I, "mu_ref": ref,
        "n_particles": _f(res, "n_particles"),
        "I_measured": _f(res, "inertial_number"),
        "I_bulk": _f(res, "inertial_number_bulk"),
        "I_bulk_final": _f(res, "inertial_number_bulk_final"),
        "mu_wall": _f(res, "mu_effective"),
        "mu_wall_final": _f(res, "mu_final"),
        "mu_bulk": _f(res, "bulk_mu"),
        "mu_bulk_final": _f(res, "bulk_mu_final"),
        "p_wall": _f(res, "normal_stress_measured_pa"),
        "p_bulk": _f(res, "bulk_pressure_pa"),
        "p_bulk_final": _f(res, "bulk_pressure_final_pa"),
        "tau_wall": _f(res, "shear_stress_pa"),
        "bulk_kinetic_pressure_pa": _f(res, "bulk_kinetic_pressure_pa"),
        "shear_rate_imposed": _f(res, "shear_rate_imposed"),
        "shear_rate_measured": _f(res, "shear_rate_measured"),
        "profile_r2": _f(res, "profile_r2"),
        "phi_bed_settled": _f(res, "phi_bed_settled"),
        "phi_bed_steady": _f(res, "phi_bed_steady"),
        "phi_bed_final": _f(res, "phi_bed_final"),
        "coordination": _f(res, "coordination_number"),
        "dilatancy": _f(res, "dilatancy"),
        "pressure_control_error": _f(res, "pressure_control_error"),
        "pressure_control_error_final": _f(res, "pressure_control_error_final"),
        "servo_valid": bool(res.get("servo_valid", False)),
        "recover_steps": _f(res, "recover_steps", 0),
        "shear_steps": _f(res, "shear_steps"),
        "shear_steps_executed": _f(res, "shear_steps_executed"),
        "shear_strain_achieved": _f(res, "shear_strain_achieved"),
        "wall_clock_s": time.time() - t0,
    }
    row["mu_wall_err_pct"] = 100.0 * (row["mu_wall"] / ref - 1.0)
    row["mu_bulk_err_pct"] = 100.0 * (row["mu_bulk"] / ref - 1.0)
    row["mu_bulk_final_err_pct"] = 100.0 * (row["mu_bulk_final"] / ref - 1.0)
    row["p_balance_rel"] = row["p_bulk"] / P_APPLIED - 1.0
    row["mu_bulk_over_wall"] = (row["mu_bulk"] / row["mu_wall"]
                                if row["mu_wall"] else float("nan"))
    row["rate_ratio"] = (row["shear_rate_measured"] / row["shear_rate_imposed"]
                         if row["shear_rate_imposed"] else float("nan"))
    row["pressure_err_final"] = (row["p_wall"] / P_APPLIED - 1.0)
    return row


# ----------------------------------------------------------------- verdicts
def adjudicate(rows: list[dict]) -> dict:
    A = [r for r in rows if r["arm"] == "A_baseline"]
    B = {r["I_target"]: r for r in rows if r["arm"] == "B_layers"}
    C = {r["I_target"]: r for r in rows if r["arm"] == "C_servo"}
    A_by_I = {r["I_target"]: r for r in A}

    def _all(pred, rs):
        return bool(rs) and all(pred(r) for r in rs)

    out = {}
    out["H1_pressure_balance"] = {
        "criterion": "|p_bulk/P-1| <= 0.30 (all A)",
        "values": {f"I={r['I_target']:.3g}": round(100 * r["p_balance_rel"], 2)
                   for r in A},
        "pass": _all(lambda r: abs(r["p_balance_rel"]) <= 0.30, A),
    }
    out["H2_wall_vs_bulk"] = {
        "criterion": "|mu_bulk/mu_wall-1| <= 0.15 (all A)",
        "values": {f"I={r['I_target']:.3g}": round(r["mu_bulk_over_wall"], 4)
                   for r in A},
        "pass": _all(lambda r: abs(r["mu_bulk_over_wall"] - 1.0) <= 0.15, A),
    }
    out["H3_no_localisation"] = {
        "criterion": "|rate_meas/rate_imp-1| <= 0.15 (all A)",
        "values": {f"I={r['I_target']:.3g}": round(r["rate_ratio"], 4) for r in A},
        "pass": _all(lambda r: abs(r["rate_ratio"] - 1.0) <= 0.15, A),
    }
    conv = {}
    for I, rb in B.items():
        ra = A_by_I.get(I)
        if ra is None:
            continue
        conv[f"I={I:.3g}"] = {
            "mu_bulk_baseline": round(ra["mu_bulk"], 4),
            "mu_bulk_thick": round(rb["mu_bulk"], 4),
            "delta_mu_bulk": round(rb["mu_bulk"] - ra["mu_bulk"], 4),
            "band_10pct_of_ref": round(0.10 * ra["mu_ref"], 4),
            "converged": abs(rb["mu_bulk"] - ra["mu_bulk"]) <= 0.10 * ra["mu_ref"],
        }
    out["H4_layer_convergence"] = {
        "criterion": "|mu_bulk(thick)-mu_bulk(baseline)| <= 0.10 mu_ref",
        "values": conv,
        "pass": bool(conv) and all(v["converged"] for v in conv.values()),
    }
    if C:
        rc = C.get(0.3)
        ra = A_by_I.get(0.3)
        if rc is not None and ra is not None:
            out["H5_servo_bandwidth"] = {
                "criterion": "recover_steps cut >= 5x AND |p/P-1| <= 0.05 at I=0.3",
                "values": {
                    "recover_steps_slow": ra["recover_steps"],
                    "recover_steps_fast": rc["recover_steps"],
                    "pressure_err_final_slow": round(ra["pressure_err_final"], 4),
                    "pressure_err_final_fast": round(rc["pressure_err_final"], 4),
                    "mu_bulk_slow": round(ra["mu_bulk"], 4),
                    "mu_bulk_fast": round(rc["mu_bulk"], 4),
                },
                "pass": bool(
                    rc["recover_steps"] * 5 <= max(ra["recover_steps"], 1)
                    and abs(rc["pressure_err_final"]) <= 0.05),
            }
    # falsification of the R5 verdict
    fals = []
    for r in A:
        if abs(r["mu_bulk_err_pct"]) <= 10.0:
            fals.append({"I": r["I_target"], "mu_bulk_err_pct":
                         round(r["mu_bulk_err_pct"], 2)})
    out["falsification"] = {
        "criterion": "any A point |mu_bulk-mu_ref| <= 10 % overturns R5 WITHHELD",
        "triggered": fals,
        "reading": ("R5 verdict overturned at these points" if fals else
                    "R5 verdict (WITHHELD at I >= 0.1) survives the bulk test"),
    }
    # per-point 10 % band verdict (case acceptance), both estimators
    point_verdicts = []
    for r in rows:
        point_verdicts.append({
            "arm": r["arm"], "I": r["I_target"],
            "mu_wall_err_pct": round(r["mu_wall_err_pct"], 2),
            "mu_bulk_err_pct": round(r["mu_bulk_err_pct"], 2),
            "best_abs_err_pct": round(min(abs(r["mu_wall_err_pct"]),
                                          abs(r["mu_bulk_err_pct"])), 2),
            "in_10pct_band": min(abs(r["mu_wall_err_pct"]),
                                 abs(r["mu_bulk_err_pct"])) <= 10.0,
            "servo_valid": r["servo_valid"],
        })
    out["points"] = point_verdicts
    return out


# ------------------------------------------------------------------ tables
def write_tables(rows: list[dict], verdicts: dict, path: Path) -> str:
    order = {"A_baseline": 0, "B_layers": 1, "C_servo": 2}
    rows = sorted(rows, key=lambda r: (order[r["arm"]], r["I_target"]))
    L = ["# R7 — plane shear mu(I): round-6 pre-registration executed + layer / servo arms", "",
         "> generated by `benchmarks/plane_shear_mu_i/round7_protocol.py`; "
         "every number below is read from `result.json`, none is hand-copied.", "",
         "## (1) bulk versus wall rheometry (tail window), and the reference law", "",
         "| arm | geom | N | I_target | I_meas | I_bulk | mu ref | mu wall | mu bulk | "
         "err wall % | err bulk % | in 10 % band |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        band = min(abs(r["mu_wall_err_pct"]), abs(r["mu_bulk_err_pct"])) <= 10.0
        L.append(f"| {r['arm']} | {r['geom']} | {r['n_particles']:.0f} | "
                 f"{r['I_target']:.3g} | {r['I_measured']:.4g} | "
                 f"{r['I_bulk']:.4g} | {r['mu_ref']:.4f} | {r['mu_wall']:.4f} | "
                 f"{r['mu_bulk']:.4f} | {r['mu_wall_err_pct']:+.1f} | "
                 f"{r['mu_bulk_err_pct']:+.1f} | {'yes' if band else 'no'} |")
    L += ["", "## (2) stress control and estimator validity", "",
          "| arm | I | p wall [Pa] | p bulk [Pa] | p_bulk/P-1 % | mu_bulk/mu_wall | "
          "rate ratio | profile R2 | servo valid | recover steps | strain achieved |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['arm']} | {r['I_target']:.3g} | {r['p_wall']:.0f} | "
                 f"{r['p_bulk']:.0f} | {100 * r['p_balance_rel']:+.1f} | "
                 f"{r['mu_bulk_over_wall']:.3f} | {r['rate_ratio']:.3f} | "
                 f"{r['profile_r2']:.2f} | {r['servo_valid']} | "
                 f"{r['recover_steps']:.0f} | {r['shear_strain_achieved']:.1f} |")
    L += ["", "## (3) solid fraction (dilatancy) — the second half of the mu(I) law", "",
          "| arm | I | phi settled | phi steady | phi final | dilatancy (steady/settled - 1) |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['arm']} | {r['I_target']:.3g} | {r['phi_bed_settled']:.4f} | "
                 f"{r['phi_bed_steady']:.4f} | {r['phi_bed_final']:.4f} | "
                 f"{100 * r['dilatancy']:+.2f} % |")
    L += ["", "## (4) pre-registered verdicts", "",
          "| hypothesis | criterion | measured | verdict |", "|---|---|---|---|"]
    for key in ("H1_pressure_balance", "H2_wall_vs_bulk", "H3_no_localisation",
                "H4_layer_convergence", "H5_servo_bandwidth"):
        v = verdicts.get(key)
        if not v:
            L.append(f"| {key} | {PREREG['hypotheses'].get(key, '')} | not run | n/a |")
            continue
        if key == "H4_layer_convergence":
            meas = "; ".join(f"{k}: {d['mu_bulk_baseline']:.4f} -> "
                             f"{d['mu_bulk_thick']:.4f} (delta {d['delta_mu_bulk']:+.4f})"
                             for k, d in v["values"].items())
        elif key == "H5_servo_bandwidth":
            meas = (f"recover {v['values']['recover_steps_slow']:.0f} -> "
                    f"{v['values']['recover_steps_fast']:.0f}; p err "
                    f"{100 * v['values']['pressure_err_final_slow']:+.1f} % -> "
                    f"{100 * v['values']['pressure_err_final_fast']:+.1f} %")
        else:
            meas = "; ".join(f"{k}: {x:+.3f}" for k, x in v["values"].items())
        L.append(f"| {key} | {v['criterion']} | {meas} | "
                 f"{'PASS' if v['pass'] else 'FAIL'} |")
    f = verdicts["falsification"]
    L += ["", "### falsification of the R5 verdict", "", f"- criterion: {f['criterion']}",
          f"- triggered at: {f['triggered'] if f['triggered'] else 'no point'}",
          f"- reading: **{f['reading']}**", "",
          "## (5) per-point acceptance (case band = 10 % vs da Cruz 2005)", "",
          "| arm | I | err mu wall % | err mu bulk % | best | servo valid |",
          "|---|---|---|---|---|---|"]
    for p in verdicts["points"]:
        L.append(f"| {p['arm']} | {p['I']:.3g} | {p['mu_wall_err_pct']:+.1f} | "
                 f"{p['mu_bulk_err_pct']:+.1f} | "
                 f"{'PASS' if p['in_10pct_band'] else 'FAIL'} | {p['servo_valid']} |")
    text = "\n".join(L) + "\n"
    path.write_text(text)
    return text


def make_figure(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    grid = np.logspace(-3, math.log10(0.5), 240)
    style = {"A_baseline": ("o", "baseline 5dx6L"), "B_layers": ("^", "thick 8dx9L"),
             "C_servo": ("s", "fast servo")}
    for arm, (mark, lab) in style.items():
        rs = sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["I_target"])
        if not rs:
            continue
        I = [r["I_target"] for r in rs]
        axes[0].semilogx(I, [r["mu_bulk"] for r in rs], mark + "-", label=f"{lab} bulk")
        axes[0].semilogx(I, [r["mu_wall"] for r in rs], mark + "--", alpha=.55,
                         label=f"{lab} wall")
        axes[1].semilogx(I, [100 * r["p_balance_rel"] for r in rs], mark + "-",
                         label=lab)
        axes[2].semilogx(I, [100 * r["mu_bulk_err_pct"] for r in rs], mark + "-",
                         label=lab)
    axes[0].semilogx(grid, [mu_ref(x) for x in grid], "k-", lw=1.6,
                     label="da Cruz 2005")
    axes[0].set_xlabel("I"); axes[0].set_ylabel(r"$\mu$"); axes[0].legend(fontsize=7)
    axes[0].set_title(r"$\mu(I)$: bulk vs wall vs reference"); axes[0].grid(alpha=.3)
    axes[1].axhline(0, color="k", lw=.8); axes[1].axhspan(-30, 30, color="0.85", zorder=0)
    axes[1].set_xlabel("I_target"); axes[1].set_ylabel(r"$p_{bulk}/P-1$ [%]")
    axes[1].set_title("slab pressure balance (grey = P1 band)"); axes[1].grid(alpha=.3)
    axes[1].legend(fontsize=7)
    axes[2].axhline(0, color="k", lw=.8)
    axes[2].axhspan(-10, 10, color="0.85", zorder=0)
    axes[2].set_xlabel("I_target"); axes[2].set_ylabel(r"$\mu_{bulk}$ error [%]")
    axes[2].set_title("deviation from reference (grey = 10 % band)")
    axes[2].grid(alpha=.3); axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None, help="A | B | C (default: all)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prereg.json").write_text(json.dumps(PREREG, indent=2, default=str))

    specs: list[tuple[str, str, float]] = []
    for arm, spec in PREREG["arms"].items():
        if args.arm and not arm.startswith(args.arm):
            continue
        for I in spec["I"]:
            specs.append((arm, spec["geom"], float(I)))
    workers = int(os.environ.get("R7_WORKERS", min(len(specs), 7)))
    t0 = time.time()
    rows: list[dict] = []

    def persist(done: list[dict]) -> None:
        """Crash-safe artifacts: every completed point is published at once so a
        long arm (thick bed) can never leave the cheap arms unpublished."""
        done = sorted(done, key=lambda r: (r["arm"], r["I_target"]))
        v = adjudicate(done)
        (out / "result.json").write_text(json.dumps(
            {"case": "CASE-DEM-PLANE-SHEAR-MUIDI", "round": 7, "prereg": PREREG,
             "rows": done, "verdicts": v, "wall_clock_s": time.time() - t0,
             "n_points_requested": len(specs),
             "n_points_completed": len(done)},
            indent=2, default=str))
        write_tables(done, v, out / "tables.md")
        make_figure(done, out / "fig_round7.png")
        return v

    if workers <= 1:
        for spec in specs:
            rows.append(run_point(spec))
            persist(rows)
            print("done", spec, flush=True)
    else:
        from concurrent.futures import as_completed
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_point, spec): spec for spec in specs}
            for fut in as_completed(futures):
                row = fut.result()
                rows.append(row)
                persist(rows)
                print(f"{row['arm']} I={row['I_target']:.3g} "
                      f"mu_wall={row['mu_wall']:.4f} mu_bulk={row['mu_bulk']:.4f} "
                      f"p_bal={100 * row['p_balance_rel']:+.1f}% "
                      f"valid={row['servo_valid']} t={row['wall_clock_s']:.0f}s",
                      flush=True)
    verdicts = persist(rows)
    print(json.dumps({k: v for k, v in verdicts.items() if k != "points"},
                     indent=2, default=str))
    print("written:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())