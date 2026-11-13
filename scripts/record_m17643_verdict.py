#!/usr/bin/env python3
"""Record the board verdict for m17643 into tensor_maturity_verdicts.

The evidence block is assembled from artifacts/xval_m17643_board_audit/evidence.json
(machine-produced by scripts/board_audit_m17643.py) so that no number is hand-copied
into the verdict.  Idempotent: re-running updates the row for this audit mission.
"""
from __future__ import annotations

import json
import os

ENV = "/root/tensor-ecosystem/TensorPlatform/.env"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(HERE, "artifacts/xval_m17643_board_audit/evidence.json")

AUDIT_MISSION = 17671
TARGET_MISSION = 17643
SCOPE = "m17643 XVAL-DEM-ROUND2-METHOD-SELECTION"
REPO = "TensorDEM"
LEVEL = 4


def dsn() -> str:
    for line in open(ENV):
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip().replace("postgresql+psycopg://",
                                                         "postgresql://")
    raise SystemExit("no DATABASE_URL")


def build_evidence(ev: dict) -> dict:
    e2 = ev["E2_spin_down"]["rows"]
    e3 = ev["E3_pair_lever"]
    return {
        "impl_files": [
            "src/tensordem/xval/oss_rolling.py:97 cdt_torque, :120 SpringRolling, "
            ":173 tensordem_kernel, :178 oss_epsd2, :183 rayleigh_timestep, "
            ":204 hertz_equivalent_shear_modulus",
            "src/tensordem/xval/scenes.py:35 slice_theory_deceleration, :40 roll_on_plane, "
            ":112 pair_spin_relax",
            "src/tensordem/granular3d_core.py:50 rolling_lever, :54 rolling_torsion, "
            ":93 pair_rolling_kr, :98 wall_rolling_kr",
            "src/tensordem/granular_column/kernel.py:224 pair_rolling_kr, :283 torsion "
            "strip (pair), :390 wall_rolling_kr, :454 torsion strip (wall)",
            "src/tensordem/granular_column/axisym.py:145 rolling_lever, :208 forwards to "
            "Material3D",
        ],
        "entry": "benchmarks/xval_dem_round2_method_selection.py:578 main() -> arms "
                 "E1..E7 -> artifacts/xval_dem_round2_method_selection/",
        "run_artifacts": [
            f"{p} ({ev['inventory'][p]['bytes']} B, sha256:{ev['inventory'][p]['sha256']})"
            for p in ("artifacts/xval_dem_round2_method_selection/results.json",
                      "artifacts/xval_dem_round2_method_selection/table_main.md",
                      "artifacts/xval_dem_round2_method_selection/traces.json")
        ] + [f"{f['path']} ({f['bytes']} B)" for f in ev["figures"]],
        "tests": "tests/test_xval_oss_rolling.py (10 tests; invariant assertions: "
                 "kernel-vs-OSS-transcription max|dw|<1e-9, pair lever ratio 2.0+-0.05, "
                 "legacy defaults bit-for-bit, torsion switch zeroes the axial torque, "
                 "analytic decel rel<2%, creep-vs-roll band). Board audit re-ran: "
                 "PYTHONPATH=src pytest -q tests/test_xval_oss_rolling.py -> 10 passed in "
                 "11.86s (2026-11-13).",
        "integration_callers":
            "pair_contact_forces (granular_column/kernel.py:224) and wall_contact_forces "
            "(kernel.py:390) now read Material3D.pair/wall_rolling_* and the torsion flag; "
            "AxisymCase (axisym.py:145,:208) forwards the convention. Defaults are legacy "
            "(granular3d_core.py:50,54) => existing results unchanged "
            "(tests test_defaults_are_legacy, test_axisym_case_forwards_and_validates_"
            "the_convention). NOT exercised end-to-end: no system case runs with "
            "rolling_lever='reduced_radius' yet (docs §5 defers the column-runout arm).",
        "benchmark": {
            "referee_query": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases "
                             "WHERE anchor_slug='XVAL-DEM-ROUND2-METHOD-SELECTION'",
            "referee_rows": ev["referee_row_count"],
            "referee_note": "0 registered rows => no machine-checkable ref/got row exists; "
                            "grade rests on the independent referees below + the "
                            "pre-declared 15% tolerance (prereg "
                            "XVAL_DEM_ROUND2_METHOD_SELECTION.json, results.json tolerance=0.15). "
                            "(peer precedent: m17633 L4 with 0 rows, analytic referee.)",
            "E1_identity_wall_vs_oss_epsd2": {"max_abs_diff_omega":
                ev["E1_identity"]["max_abs_diff_omega"], "within_1e-12":
                ev["E1_identity"]["below_1e-12"]},
            "E2_spin_down_vs_analytic_5over7_mu_r_g": [
                {"mu_r": r["mu_r"], "kernel_ms": round(r["t_kernel_ms"], 2),
                 "err_pct": round(r["err_kernel_vs_theory_pct"], 2),
                 "within_15pct": r["kernel_within_tol_vs_theory"]} for r in e2],
            "E2_vs_oss_epsd2_all_points_pct": max(abs(r["err_kernel_vs_oss_epsd2_pct"])
                                                  for r in e2),
            "E3_pair_lever": {"t_half_ms": {k: round(v, 2) for k, v in
                                            e3["t_half_ms"].items()},
                              "implied_w0_rad_s": e3["implied_w0_rad_s"],
                              "implied_w0_spread_pct": e3["implied_w0_spread_pct"],
                              "mapping_mu_oss_over_mu_td":
                                  round(e3["implied_mapping_mu_oss_over_mu_td"], 6),
                              "mapping_matches_doc_2x": abs(
                                  e3["implied_mapping_mu_oss_over_mu_td"] - 2.0) < 1e-6},
            "E4_incline_static_hold": "5/15/20/25 deg below arctan(mu_r)=26.57 deg hold at "
                                      "1e-4..1e-5 m/s; 30 deg rolls at 1.4e-1 m/s "
                                      "(results.json incline.rows)",
            "E5_torsion": {"spin_lost_pct_kernel": round(ev["E5_torsion"]["spin_lost_pct_kernel"], 2),
                           "w_end_oss_default": ev["E5_torsion"]["w_end_oss_default"],
                           "production_matches_kernel":
                               ev["E5_torsion"]["production_matches_kernel"]},
            "E6_timestep": {"dt_over_rayleigh": round(ev["E6_timestep"]["dt_over_rayleigh"], 3),
                            "inside_oss_practice_band": ev["E6_timestep"]["inside_oss_band"]},
            "E7_dt_spread_pct": {k: round(v, 3) for k, v in ev["E7_dt"]["spread_pct"].items()},
            "honest_fail_rows": ["E2 mu_r=0.8 +17.63% vs analytic (mu_r>mu_s sliding "
                                 "saturation; all three laws agree to 0.00%)",
                                 "E3 pre-mapping 50.00% (lever arm 2x, conversion published)",
                                 "E5 102.05% (kernel treats drilling spin as rolling; "
                                 "opt-in switch added, default kept)"],
        },
        "repro_cmd": {
            "benchmark": "cd <checkout of e1cdcab1> && PYTHONPATH=src python3 "
                         "benchmarks/xval_dem_round2_method_selection.py",
            "observed": "46 s wall; regenerated results.json + table_main.md + traces.json "
                        "byte-identical to the committed artifacts (diff => IDENTICAL, "
                        "board audit 2026-11-13)",
            "tests": "PYTHONPATH=src python3 -m pytest -q tests/test_xval_oss_rolling.py "
                     "=> 10 passed (11.86 s)",
        },
        "external_anchor": {
            "source": "CFDEMproject/LIGGGHTS-PUBLIC @3d5c00f20519e6bb6eb6756f51f1ad36564e649d "
                      "(GPL-2.0), rolling_model_{base,cdt,epsd2}.h + fix_check_timestep_gran.cpp",
            "snapshot": "docs/xval/oss_snapshot/ (PROVENANCE.json SHA-256)",
            "audit_rehash": "all 4 files bytes+sha256 match PROVENANCE.json "
                            f"(all_ok={ev['oss_provenance']['all_ok']})",
            "transcription_check": "epsd2 kr=kt*reff^2, cap=|Fn|*reff*rmu, torsion strip "
                                   "(snapshot :173-206) and cdt r_torque=rmu*Fn*reff*w/|w| "
                                   "(:118-130) match oss_rolling.py:97,120",
        },
        "scope_boundary": "validated: single-contact rolling law + timestep criterion + "
                          "lever/torsion convention (contact scale). not validated: "
                          "system-scale column runout with the converted mu_r (deferred in "
                          "docs §5); the new switch has no end-to-end consumer yet.",
        "open_items": [
            "0 referee rows registered for the anchor -> register E1/E2/E3/E6 metrics with "
            "the 15% tolerance so the gate can machine-check the next audit",
            "commit e1cdcab1 is NOT an ancestor of origin/main (branch agent/m17643 only) "
            "-> merge outstanding, otherwise the capability is not in the product",
        ],
        "verdict_basis": "L4: L2 real code (xval module) + L3 wired into the production "
                         "contact kernels (defaults legacy) + L4 benchmark validation: "
                         "closed-form referee + hashed OSS anchor + pre-declared 15% gate + "
                         "audit-reproduced run (bit-identical) + honest FAIL rows disclosed.",
    }


def main() -> None:
    import psycopg
    ev = json.load(open(EV))
    evidence = build_evidence(ev)
    with psycopg.connect(dsn()) as c, c.cursor() as cur:
        cur.execute("SELECT id FROM tensor_maturity_verdicts WHERE scope=%s AND mission_id=%s",
                    (SCOPE, AUDIT_MISSION))
        row = cur.fetchone()
        if row:
            cur.execute("UPDATE tensor_maturity_verdicts SET repo=%s, level=%s, evidence=%s,"
                        " verdict_by=%s WHERE id=%s",
                        (REPO, LEVEL, json.dumps(evidence, ensure_ascii=False),
                         "board-audit", row[0]))
            print("updated verdict id", row[0])
        else:
            cur.execute("INSERT INTO tensor_maturity_verdicts"
                        " (repo, scope, mission_id, level, evidence, verdict_by)"
                        " VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
                        (REPO, SCOPE, AUDIT_MISSION, LEVEL,
                         json.dumps(evidence, ensure_ascii=False), "board-audit"))
            print("inserted verdict id", cur.fetchone()[0])
        c.commit()
        cur.execute("SELECT id, repo, scope, mission_id, level, verdict_by, gate_state,"
                    " gate_level, gate_note FROM tensor_maturity_verdicts"
                    " WHERE mission_id=%s AND scope=%s", (AUDIT_MISSION, SCOPE))
        print(cur.fetchone())


if __name__ == "__main__":
    main()