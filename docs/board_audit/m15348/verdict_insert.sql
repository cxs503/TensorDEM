INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorDEM', 'm15259 CASE-HOPPER-BEVERLOO', 15348, 2, $json$
{
  "impl_files": [
    "benchmarks/beverloo_hopper_m15259.py:100-135 build_case() (D=0.075/d=0.015), :190 simulate() call into verification.granular.hopper_discharge.kernel",
    "src/verification/granular/hopper_discharge/kernel.py:1-40 (pre-existing m7751 vectorised DEM kernel invoked, NOT authored by m15259)"
  ],
  "entry": "PYTHONPATH=src python benchmarks/beverloo_hopper_m15259.py --seed 42 (auditor relaunch = cluster cj1067, still queued at audit close -> reproducibility not_run)",
  "run_artifacts": [
    "docs/board_audit/m15348/refs/m15259_local_med.json (byte-identical copy, sha256 03d755f0a9d2..., of mission commit cda15103:runs/beverloo_m15259/local_med.json)",
    "runs/beverloo_m15259/local_med.json is ABSENT from main HEAD (git cat-file -e HEAD:runs/beverloo_m15259/local_med.json fails) -> the only run artifact was filtered at push"
  ],
  "tests": "none added by m15259 (commit cda15103 touches 0 test files); pre-existing tests/test_hopper_discharge.py:48-60 (C2/C3, m7751) only",
  "integration_callers": "NONE: grep -rn beverloo_hopper_m15259 outside benchmarks/beverloo_hopper_m15259.py = 0 hits; no export/CLI/API; benchmarks/m15259_final_report.py and benchmarks/m15259_report.py absent from main HEAD",
  "benchmark": {
    "ref": {"source": "Beverloo, Leniger & Van de Velde 1961, Chem. Eng. Sci. 15:260-269, DOI 10.1016/0009-2509(61)80059-4", "C": 0.58, "C_range": [0.55, 0.60], "k": 1.4, "k_range": [1.2, 1.6], "W_bev_kg_s": 1.4490165457739104},
    "got": {"W_meas_kg_s": 4.0927272827566155, "rel_err": 1.8244862315017365, "recovered_C": 1.6382020142710068, "window_cv_2_6": 0.232, "mass_residual": 1.747403899362428e-16, "energy_residual": 3.1353314615988412},
    "tolerance": 0.15,
    "status": "FAIL",
    "filed_headline": {"W_meas_kg_s": 1.829, "W_bev_kg_s": 1.8356, "rel_err": -0.004, "provenance": "hard-coded literal in m15259_final_report.py:30 (M6457_CASE_C dict, comment 'from audit document'), copied from docs/beverloo_discharge_measurement_audit.md:82 row C_lowmu_D5 (m6457); no m15259 artifact contains 1.8290", "source_disclaimer": "docs/beverloo_discharge_measurement_audit.md §4: 'that match must not be read as a validated C = 0.58 ... C = 0.955 (+65%) ... stays WITHHELD as a physical claim'"},
    "registry": "tensor_benchmark_cases WHERE anchor_slug='CASE-HOPPER-BEVERLOO' -> 0 rows (15 rows in table) => no machine-owned criterion; mission goal text (<15%) + prereg/m15259_beverloo.md are the only criteria",
    "prereg_criteria": {"abs_rel_err_lt_15pct": false, "window_cv_2_6_lt_20pct": false, "mass_ledger_residual_lt_1e-10": true}
  },
  "repro_cmd": "PYTHONPATH=src python benchmarks/beverloo_hopper_m15259.py --seed 42 -> regenerates D=0.075/d=0.015/r_bin=0.12/fill=0.20, whereas the only committed artifact is D=0.08/d=0.016: geometry mismatch, so the merged repro neither reproduces nor falsifies the filed numbers (auditor run cj1067 queued, and m6457 C_lowmu_D5 replay cj1072 queued, both not_run at audit close)",
  "external_anchor": "Beverloo 1961 Eq.4 p.265: C=0.58 (0.55-0.60), k=1.4 (1.2-1.6); in-repo corroboration docs/hopper_discharge_verification.md:120-121 (m7751 campaign, same 30 deg cone, D/d=4/5/6, d=0.05) C=0.596 (-2.7%), R2=0.9999 -- capability exists in repo but is not demonstrated by m15259, and its artifacts (discharge_D0*.json) are also absent from main",
  "level_reason": "L2 = real input->compute->output code (case builder -> kernel.simulate -> Beverloo comparison -> JSON record; run shows 948 initial / 259 discharged particles, mass ledger 1.7e-16). L3 refused: no integration caller/export/CLI/API, both report generators filtered out of main. L4 refused: the anchor criterion FAILS on the mission's own artifact (+182.4% vs 15%; recovered C=1.638 = 2.8x literature; prereg CV criterion 23.2% > 20% unevaluated in the report), the PASS headline is a hand-transcribed m6457 number whose source doc marks C WITHHELD, repro_cmd is a different geometry than the artifact, no test added, anchor unregistered. Report itself never published (tensor_reports has 0 rows for mission_id=15259, tensor_dev_missions.report_id IS NULL); cited cluster evidence cj1037 is still state=queued (never executed).",
  "audit_artifacts": [
    "docs/board_audit/m15348/AUDIT.md",
    "docs/board_audit/m15348/audit_m15259.py",
    "docs/board_audit/m15348/audit_m15259.json",
    "docs/board_audit/m15348/make_figure.py",
    "docs/board_audit/m15348/m15348_m15259_anchor_check.png",
    "docs/board_audit/m15348/refs/m15259_local_med.json",
    "docs/board_audit/m15348/refs/m15259_report_posted_candidate.json"
  ],
  "repo_head_audited": "TensorDEM 4fa79610 (branch agent/m15348; includes merge of PR #1282 = mission commit cda15103)",
  "gate_expectation": "reopened"
}
$json$::jsonb, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id,
      level = EXCLUDED.level,
      evidence = EXCLUDED.evidence,
      created_at = now(),
      gate_state = 'pending', gate_level = NULL, gate_note = NULL
RETURNING id, repo, scope, level, mission_id;