#!/usr/bin/env python3
"""m20066 -- land the board-audit verdict for m20004 (TensorDEM, CASE-DEM-OEDOMETER-STIFF).

Reads the machine evidence produced by ``docs/AUDIT_M20004/verify.py``
(``runs/audit_m20004/evidence.json``) and writes / executes:

  runs/audit_m20004/verdict_payload.json   exact evidence JSON inserted
  runs/audit_m20004/insert_verdict.sql     replayable SQL

Claimed level = 3 (L1 tests pass, L2 real DEM implementation, L3 consumed by the
consolidation chain in trunk).  L4 is *refused*: the case's own 10 % tolerance is
missed by 15.69 %, the board referee table has 0 rows for the anchor, and the case
harness was never committed.  ``gate_state`` is left at its default (pending) so the
maturity gate adjudicates confirmed / reopened.
"""
from __future__ import annotations

import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent          # docs/AUDIT_M20004
ROOT = HERE.parents[1]
OUT = ROOT / "runs" / "audit_m20004"
EV = json.loads((OUT / "evidence.json").read_text())

SNAP = "runs/audit_m20004/audited_snapshot"
P2 = EV["P2_benchmark"]
P3 = EV["P3_repro_cmd"]
P4 = EV["P4_report"]
P5 = EV["P5_claims"]
P1 = EV["P1_run_artifacts"]

evidence = {
    "impl_files": [
        "src/tensordem/consolidation/dem_oedometer.py",
        f"{SNAP}/dem_oedometer.py",
    ],
    "impl_pointers": {
        "src/tensordem/consolidation/dem_oedometer.py": "509 loc real DEM oedometer: HCP+jitter bed in a rigid cell,"
        " two stress increments (5/15 kPa), E_oed = d_sigma/d_eps_v read off the piston travel;"
        " run_oedometer at :412-509; contact law re-used from tensordem.granular3d_core",
        f"{SNAP}/dem_oedometer.py": "frozen byte-identical copy of the trunk src module"
        " (sha256 020fe03b6a623efc90ebfe622aa91ac54ca3f446d7f682affd8dd6ee11bc82f8)",
    },
    "entry": [
        f"{SNAP}/benchmarks_case_dem_oedometer_stiff/run_case.py",
        f"{SNAP}/benchmarks_case_dem_oedometer_stiff/benchmark.py",
    ],
    "entry_pointers": {
        f"{SNAP}/benchmarks_case_dem_oedometer_stiff/run_case.py": ":64 main — primary zero-friction run,"
        " writes dem_result.json + reference.json",
        f"{SNAP}/benchmarks_case_dem_oedometer_stiff/benchmark.py": ":56 run_benchmark — ref-vs-got legs,"
        " :16 TOL_E_OED_PCT=10.0",
    },
    "run_artifacts": [
        f"{SNAP}/artifacts_case_dem_oedometer_stiff/dem_result.json",
        f"{SNAP}/artifacts_case_dem_oedometer_stiff/reference.json",
        f"{SNAP}/artifacts_case_dem_oedometer_stiff/benchmark.json",
        f"{SNAP}/artifacts_case_dem_oedometer_stiff/benchmark.md",
        f"{SNAP}/artifacts_case_dem_oedometer_stiff/figures/stress_strain.png",
        f"{SNAP}/artifacts_case_dem_oedometer_stiff/figures/comparison.png",
        f"{SNAP}/artifacts_case_dem_oedometer_stiff/figures/packing.png",
        "runs/audit_m20004/repro_run/artifacts/case_dem_oedometer_stiff/dem_result.json",
        "runs/audit_m20004/repro_run/artifacts/case_dem_oedometer_stiff/reference.json",
        "runs/audit_m20004/figures/audit_m20004_fig1.png",
    ],
    "run_artifacts_note": {
        "audited_worktree": P1["audited_worktree"],
        "audited_head": P1["audited_head"],
        "n_files_nonempty": P1["n_files_nonempty"],
        "git_tracked_case_files": P1["git_tracked_case_files"],
        "git_added_anywhere": P1["git_added_anywhere"],
        "why_snapshot": "the audited case directories are 100% untracked in m20004 (git status shows 5 '??' dirs,"
                        " `git log --all --diff-filter=A -- benchmarks/case_dem_oedometer_stiff` is empty), so the"
                        " evidence is frozen here with sha256 provenance instead of pointing at an ephemeral tree",
    },
    "tests": [
        "python -m pytest -q tests/test_consolidation_terzaghi.py -> 25 passed in 68.18s (audit re-run, m20066 worktree, trunk code)",
        "tests/test_consolidation_terzaghi.py::test_dem_oedometer_cell_integrates_without_diverging — smoke only"
        " (finite forces, porosity drops); NO E_oed accuracy assertion (finding F9)",
        "tests/test_consolidation_terzaghi.py::test_carman_kozeny_matches_hand_computation, ::test_pore_reynolds_definition",
    ],
    "integration_callers": [
        "run_oedometer",
        "benchmarks/consolidation_terzaghi/run.py:41,104 imports and calls run_oedometer (trunk-tracked)",
        "src/tensordem/consolidation/chain.py:164-168 consumes dem['E_oed'] -> C_v = k_hyd*E_oed/gamma_w (trunk-tracked)",
        "src/tensordem/consolidation/__init__.py:35-40,87 exports run_oedometer (trunk-tracked)",
        "platform side: grep 'oedometer|run_oedometer|E_oed' in ship-performance-platform/backend (*.py) -> 0 hits,"
        " so L3 holds only at the in-repo consolidation-pipeline level, not the :8010 API level",
    ],
    "benchmark": {
        "referee_sql": P2["referee_sql"],
        "referee_row_count": P2["referee_row_count"],
        "referee_note": "0 rows for this anchor: the board's sole judge registered no criterion,"
                        " so the case cannot be judged PASS by the referee (finding F2)",
        "predeclared_tolerance_pct": P2["predeclared_tolerance_pct_from_benchmark_py"],
        "predeclared_in": "benchmarks/case_dem_oedometer_stiff/benchmark.py:16 TOL_E_OED_PCT",
        "ref_E_oed_Pa": P2["ref_micromechanical_E_oed"],
        "got_E_oed_Pa": P2["got_E_oed"],
        "abs_dev_Pa": abs(P2["got_E_oed"] - P2["ref_micromechanical_E_oed"]),
        "rel_dev_pct_recomputed": P2["rel_dev_pct_recomputed"],
        "verdict_recomputed": P2["verdict_recomputed"],
        "verdict_shipped": P2["verdict_shipped_in_benchmark_json"],
        "E_oed_rederived_from_stages_Pa": P2["E_oed_rederived_from_stages"],
        "E_oed_rederive_rel_dev_pct": P2["E_oed_rederive_rel_dev_pct"],
        "tolerance_missed_by_pp": P2["rel_dev_pct_recomputed"] - P2["predeclared_tolerance_pct_from_benchmark_py"],
        "eco_15pct_redline_exceeded": P2["rel_dev_pct_recomputed"] > 15.0,
        "reference_independence": "ref uses the SAME run's measured porosity/coordination"
                                  f" (n={P2['reference_conditioned_on_dem_measured_state']['porosity']:.4f},"
                                  f" Z={P2['reference_conditioned_on_dem_measured_state']['coordination']:.2f})"
                                  " -> state-point closure check, not an independent anchor (F7)",
        "legs": P2["legs_shipped"],
    },
    "repro_cmd": ("cd " + str(ROOT) + "/runs/audit_m20004/repro/case_dem_oedometer_stiff && "
                  "PYTHONPATH=" + str(ROOT) + "/src python3 run_case.py   # 98.3 s, 80 grains; fresh dem_result.json"
                  " compared field-by-field with the audited one: all_fields_identical=True"
                  f" ({', '.join(k for k, v in P3['field_identical'].items() if v)}),"
                  " reference.json identical=True"),
    "repro_result": {
        "all_fields_identical": P3["all_fields_identical"],
        "field_identical": P3["field_identical"],
        "reference_identical": P3["reference_identical"],
        "walL_time_audited_s": P3["wall_time_audited_s"],
        "wall_time_fresh_s": P3["wall_time_fresh_s"],
        "shipped_script_mode": P3["script_mode"],
        "shipped_script_masks_failure": "benchmark.py || true in repro_cmd.sh (F8)",
    },
    "report_check": {
        "report_id": P4["report_id"],
        "six_sections_present": P4["six_sections_present"],
        "n_figures": P4["n_figures"],
        "figure_names": P4["figure_names"],
        "figure_png_base64_present": P4["figure_png_base64_present"],
        "platform_figure_persistence": P4["platform_figure_persistence"],
        "verdict": "six sections complete; 0 figures embedded for report 18245, but the platform persists"
                   " png_base64 for only 10 of 27715 figure rows and also stripped this audit's own report"
                   " 18267 -> root cause is the platform figure pipeline, NOT an m20004-author defect;"
                   " the audited report still renders without images (F6 reframed, non-blocking)",
    },
    "claims_overstated": {
        "report_claims_pct_matching_literature": P5["report_claims_deviation_matching_literature_pct"],
        "actual_dev_pct": P5["actual_deviation_pct"],
        "claim_consistent": P5["claim_consistent_with_actual"],
        "section_4_2_attributes_to_friction": P5["section_4_2_attributes_deviation_to_friction"],
        "config_friction": P5["config_friction"],
        "config_mu_rolling": P5["config_mu_rolling"],
        "note": "report attributes the 15.69% to mu=0.4 / mu_r=0.02 sliding, but the primary run is"
                " frictionless (friction=0.0, mu_rolling=0.0) -> attribution unsupported (F4);"
                " '47% deviation' contradicts the measured 15.69% (F5)",
    },
    "external_anchor_note": "mean-field closure transcribed from Chang/Kuhn/Rothenburg (2004) without DOI/page in"
                            " run_case.py:32-60; audit re-derived the closure from measured n, Z to 0.0% of the"
                            " shipped ref value, i.e. arithmetic verified, citation NOT independently verified (F7)",
    "level_rationale": {
        "L1": "25 pytest passes in the consolidation suite (audit re-run)",
        "L2": "real DEM (509 loc, trunk contact kernel), measured stress-strain, E_oed re-derived from"
              " the two stages to 0.0% of got, reproduced bit-identical",
        "L3": "consumed by trunk: consolidation chain assembles C_v from dem['E_oed'];"
              " platform backend has 0 references",
        "L4": "REFUSED: 15.69% > pre-declared 10% tolerance, referee table 0 rows,"
              " case harness never committed (from-repo repro impossible)",
        "L5": "n/a: no independent experimental anchor",
    },
    "findings_severity": {
        "F1_BLOCKING": "benchmark FAIL: 15.6923% vs pre-declared 10% tolerance (also > ecosystem 15% redline)",
        "F2_BLOCKING": "tensor_benchmark_cases has 0 rows for the anchor -> no registered judge",
        "F3_BLOCKING": "case dirs 100% untracked in the audited worktree; never added by any commit",
        "F4_MAJOR": "report's friction attribution contradicts the frictionless primary config",
        "F5_MAJOR": "report claims '47% deviation matches literature' while measured is 15.69%; no DOI/page",
        "F6_MINOR_reframed": "report 18245 has 0 embedded figures, but so does 27705/27715 of the whole"
                             " table and this audit's own report — platform-side figure persistence,"
                             " not an author defect; kept minor (fallback: point at artifact paths)",
        "F7_MAJOR": "reference conditioned on the same run's measured n/Z -> not an independent anchor",
        "F8_MINOR": "repro_cmd.sh mode 0644 and 'benchmark.py || true' masks the FAIL exit code",
        "F9_MINOR": "no E_oed accuracy assertion in tests (smoke only)",
    },
    "audit_docs": [
        "docs/AUDIT_M20004/README.md",
        "docs/AUDIT_M20004/verify.py",
        "runs/audit_m20004/evidence.json",
    ],
    "disposition": "request changes (F1/F2/F3 blocking for L4); functional substance = L3 confirmed",
}

payload = json.dumps(evidence, ensure_ascii=False, indent=1)
(OUT / "verdict_payload.json").write_text(payload)

sql = ("INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
       "VALUES ('TensorDEM', 'm20004 CASE-DEM-OEDOMETER-STIFF', 20066, 3, $json$"
       + payload + "$json$, 'board-audit') "
       "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, "
       "evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id, gate_state='pending', "
       "gate_level=NULL, gate_note=NULL;\n")
(OUT / "insert_verdict.sql").write_text(sql)

r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-v", "ON_ERROR_STOP=1",
                    "-f", str(OUT / "insert_verdict.sql")], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip())
r.check_returncode()
row = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tA", "-F", " | ",
                      "-c", "SELECT id, repo, scope, mission_id, level, verdict_by, gate_state, "
                            "jsonb_array_length(jsonb_path_query_array(evidence::jsonb, '$.*')) "
                            "FROM tensor_maturity_verdicts WHERE repo='TensorDEM' AND "
                            "scope='m20004 CASE-DEM-OEDOMETER-STIFF'"],
                     capture_output=True, text=True)
print("row:", row.stdout.strip() or row.stderr.strip())
