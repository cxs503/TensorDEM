#!/usr/bin/env python3
"""m16226 board audit: assemble evidence, pre-judge with the platform gate, insert verdict.

Independence notes
------------------
* Every numeric below is copied from the audited artifacts / from commands run by
  this audit, never re-typed from the delivery README.
* `judge()` is imported from the *platform's own* gate (/root/tp-agent/maturity_gate.py)
  so the level is predicted by the same code that will machine-check the row later.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO + "/scripts")
from board_audit_db import dsn  # noqa: E402

MISSION = 16226
SUBJECT = 16147
SCOPE = "m16147 CASE-DEM-MINDLIN-TANGENTIAL"
ANCHOR = "CASE-DEM-MINDLIN-TANGENTIAL"
EV = "docs/board_audit_m16226/evidence"

ARTIFACTS = [f"{EV}/results.json", f"{EV}/curves.csv", f"{EV}/checksums.json",
             f"{EV}/run.log", f"{EV}/repro_full_run.log",
             f"{EV}/figures/fig1_ft_delta_curve.png",
             f"{EV}/figures/fig2_error_vs_delta.png",
             f"{EV}/figures/fig3_dt_and_sensitivity.png",
             f"{EV}/figures/fig4_stick_circle_map.png"]

EVIDENCE = {
    "impl_files": [
        "src/tensordem/mindlin_partial_slip.py",
        "src/tensordem/dem3d.py",
    ],
    "entry": [
        "benchmarks/case_mindlin_tangential/run.py",
        "benchmarks/case_mindlin_tangential/validate.py",
    ],
    "run_artifacts": ARTIFACTS,
    "tests": [
        "tests/test_mindlin_partial_slip.py",
        "auditor re-run 2026-11-12: 10 passed in 12.22 s (closed-form anchors, inverse "
        "round-trip <=1e-12*delta_s, stick radius (1/2)^(1/3), reversal branch slope/area, "
        "cross-implementation agreement <=1e-14*mu*Fn, dem3d integration, legacy invariance)",
    ],
    "integration_callers": [
        "tangential_model",
        "src/tensordem/dem3d.py:1191 platen branch returns mindlin_partial_slip.partial_slip_magnitude",
        "src/tensordem/dem3d.py:1436 particle-particle branch returns mindlin_partial_slip.partial_slip_force",
        "benchmarks/case_mindlin_tangential/run.py:57 imports DEM3DConfig/IceDEM3D from the production solver",
    ],
    "benchmark": {
        "referee_rows": 0,
        "referee_query": ("SELECT case_name,metric,tolerance FROM tensor_benchmark_cases "
                          "WHERE anchor_slug='CASE-DEM-MINDLIN-TANGENTIAL' -> 0 rows "
                          "(table total 29 rows, LIKE '%MINDLIN%' -> 0)"),
        "criterion_in_force": "mission goal '<5%' + pre-registered gates in validate.py "
                              "(5% micro/transition, 1% gross, 2% slope) — no post-hoc tuning",
        "ref_source": "Mindlin & Deresiewicz (1953) J. Appl. Mech. 20:327-344 DOI 10.1115/1.4010702 "
                      "closed form, re-derived in run.py from Hertz; Johnson (1985) Sec.4.3 eq.4.32",
        "ref_vs_got": {
            "B1_micro_slip_max_dev_over_muFn": {"got": 3.7007434154171886e-16,
                                                "ref": "mu*Fn[1-(1-d/ds)^1.5]", "tol": 0.05,
                                                "pass": True},
            "B2_transition_max_dev_over_muFn": {"got": 1.8503717077085943e-16, "tol": 0.05,
                                                "pass": True},
            "B3_gross_slip_plateau_N": {"got": 0.2999999999999999, "ref": 0.3, "tol": 0.01,
                                        "pass": True},
            "B4_initial_slope_k_t_N_per_m": {"got": 461694.6267949187,
                                             "ref": 465959.9669301074, "rel_err": 0.009153876808967376,
                                             "tol": 0.02, "pass": True},
            "B5_fitted_delta_s_m": {"got": 9.657482014275678e-07, "ref": 9.657482014275674e-07,
                                    "rel_err": 4.3853716010147245e-16, "tol": 0.05, "pass": True},
            "B6_legacy_cundall_strack_max_dev": {"got": 0.19244083267712608, "must_exceed": 0.05,
                                                 "pass": True},
        },
        "auditor_recompute_from_curves_csv": {
            "n_samples": 6438,
            "max_micro_dev_over_muFn": 2.243e-13,
            "max_transition_dev": 1.817e-13,
            "plateau_N": 0.3,
            "note": "recomputed from committed curves.csv bytes; gap to 3.7e-16 is 15-digit CSV "
                    "serialisation floor, still 11 orders below the gate",
        },
        "auditor_independent_hertz_chain": {
            "a_m": 2.574429e-04, "k_t_N_per_m": 465959.966930, "delta_s_m": 9.657482e-07,
            "rel_diff_vs_results_json": [2.11e-16, 1.25e-16, 2.19e-16],
            "k_t_over_k_n_identity": 1.235294,
        },
        "sensitivity_axes": "dt x {0.5,1,2}; mu x {0.2,0.3,0.5} -> delta_s ~ mu^1.000000; "
                            "Fn x {0.5,1,2} N -> delta_s ~ Fn^0.666667 (theory 2/3)",
        "reproduced": "auditor clean-copy re-run (/tmp/repro_m16147, no .git): micro 3.700e-16, "
                      "plateau 0.300000 N, k_t rel 9.154e-03, fitted delta_s 4.385e-16, "
                      "legacy 19.244%; validate.py 9/9 ACCEPT",
        "validate_gate": "benchmarks/case_mindlin_tangential/validate.py --out artifacts/mindlin_tangential "
                         "-> 9/9 PASS ACCEPT (exit 0)",
    },
    "repro_cmd": ("PYTHONPATH=src python benchmarks/case_mindlin_tangential/run.py "
                  "--out artifacts/mindlin_tangential && PYTHONPATH=src python "
                  "benchmarks/case_mindlin_tangential/validate.py --out artifacts/mindlin_tangential"),
    "external_anchor": [
        "docs/dem3d_mindlin_partial_slip.md",
        "Mindlin & Deresiewicz (1953) J. Appl. Mech. 20:327-344 DOI 10.1115/1.4010702; "
        "Johnson, Contact Mechanics (1985) Sec.4.3 eq.4.32 (same closed form)",
        "NOT used: original 1953 tables (paywalled, sandbox offline) and any experimental data "
        "-> no independent external anchor, hence level 4 not 5",
    ],
    "level_rationale": ("L4 = real constitutive module (317 lines) + opt-in integration in the "
                        "production DEM3D solver (platen :1191, particle :1436) + complete case with "
                        "6 gated metrics, dt/mu/Fn sensitivity, discrimination against the legacy law, "
                        "and an auditor clean-copy re-run reproducing every headline number. "
                        "Not L5: textbook 1953 closed form (no new physics), no experimental/third-party "
                        "numerical anchor, and the solver wires only the monotonic loading branch."),
    "findings": {
        "F1_medium": "tensor_benchmark_cases: 0 rows for the anchor -> machine referee absent; "
                     "criterion taken from mission goal '<5%' + pre-registered validate.py gates. "
                     "Not back-filled by the auditor (independence).",
        "F2_medium": "README Sec.4 segment sample counts (5766 micro / 672 gross) not traceable to "
                     "artifacts: independent recount from curves.csv gives 5602 / 438 / 398 and "
                     "results.json carries no n_micro/n_gross. Gate verdict unaffected.",
        "F3_low": "solver integration is monotonic-loading only (tang_disp_next = tang_disp, no cone "
                  "projection, no kinematic anchor); reversal_branch/reversal_loop_area exist in the "
                  "module but are not wired into dem3d -> the model dissipates nothing under cyclic "
                  "loading. Declared in README Sec.6; out of scope for this monotonic case.",
        "F4_low": "run.py:256-272 err_rel_muN / results max_*_err are deviations normalised by mu*Fn, "
                  "not relative errors (residual naming from the previous round).",
        "F5_info": "maturity_gate.callers_verified() greps /root/tensor-ecosystem/TensorDEM whose local "
                   "main is 277 commits behind origin/main (ad75394d vs 8133d44e), so newly merged code "
                   "is invisible to it - this caused the previous row (id=425) to reopen with "
                   "'integration_callers(grep no reference)'. Environment, not a delivery defect.",
    },
    "audit_page": "docs/board_audit_m16226/M16226_BOARD_AUDIT.md",
    "audit_page_commit_note": "committed on branch agent/m16226 (this audit)",
    "subject_commit": "00025434 Merge #1351 (m16147) via merge-master",
    "subject_worktree": "/root/tp-agent/work/m16147",
    "disposition": "approve with notes -> L4; rework items F1..F4 do not block",
}

LEVEL = 4


def main():
    # 1. preserve the reproduction log next to the audited bytes (durable evidence)
    dst = os.path.join(REPO, EV, "repro_full_run.log")
    if not os.path.exists(dst) and os.path.exists("/tmp/repro_m16147/full.log"):
        shutil.copyfile("/tmp/repro_m16147/full.log", dst)
        print("copied reproduction log ->", dst)
    missing = [p for p in ARTIFACTS + EVIDENCE["impl_files"] + EVIDENCE["entry"]
               if not os.path.exists(os.path.join(REPO, p))]
    if missing:
        raise SystemExit("missing evidence paths: %s" % missing)

    # 2. predict the gate with the platform's own judge
    sys.path.insert(0, "/root/tp-agent")
    import maturity_gate  # noqa: E402
    g_level, reasons = maturity_gate.judge(LEVEL, EVIDENCE, "TensorDEM", MISSION)
    print(json.dumps({"claimed": LEVEL, "gate_level_predicted": g_level,
                      "reasons": reasons}, indent=1, ensure_ascii=False))

    # 3. insert the verdict (idempotent per mission + scope)
    import psycopg
    with psycopg.connect(dsn()) as c, c.cursor() as cur:
        cur.execute("SELECT id, level FROM tensor_maturity_verdicts WHERE mission_id=%s AND scope=%s",
                    (MISSION, SCOPE))
        row = cur.fetchone()
        if row:
            cur.execute("UPDATE tensor_maturity_verdicts SET level=%s, evidence=%s, repo=%s, "
                        "verdict_by=%s WHERE id=%s",
                        (LEVEL, json.dumps(EVIDENCE), "TensorDEM", "board-audit", row[0]))
            vid = row[0]
            print("updated existing verdict id", vid)
        else:
            cur.execute("INSERT INTO tensor_maturity_verdicts "
                        "(repo, scope, mission_id, level, evidence, verdict_by) "
                        "VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
                        ("TensorDEM", SCOPE, MISSION, LEVEL, json.dumps(EVIDENCE), "board-audit"))
            vid = cur.fetchone()[0]
            print("inserted verdict id", vid)
        c.commit()
    out = {"verdict_id": vid, "mission": MISSION, "scope": SCOPE, "level": LEVEL,
           "gate_level_predicted": g_level, "reasons": reasons}
    with open(os.path.join(REPO, "artifacts", "board_audit_m16226_verdict.json"), "w") as fh:
        os.makedirs(os.path.join(REPO, "artifacts"), exist_ok=True) if False else None
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()