INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorDEM', 'm17676 XVAL-DEM-SANDPILE-REPOSE-CALIBRATION-ANCHOR', 17745, 3,
$js${
 "impl_files": [
  "docs/audit_m17676/xval_zhou1999.py",
  "docs/audit_m17676/cross_route.py",
  "docs/audit_m17676/sandpile_measure.patch",
  "docs/audit_m17676/test_sandpile_degenerate_measure_m17676.py"
 ],
 "impl_files_source": {
  "originals_uncommitted": [
   "/root/tp-agent/work/m17676/benchmarks/xval_sandpile_zhou1999/xval_zhou1999.py sha256 b2c6656f...",
   "/root/tp-agent/work/m17676/src/tensordem/protocols/sandpile.py sha256 d5c98856... (measure() _safe patch)",
   "/root/tp-agent/work/m17676/tests/test_sandpile_degenerate_measure_m17676.py sha256 e8f6272c..."
  ],
  "snapshot_note": "deliverables were never committed (head_commit=null); byte-identical snapshot committed under docs/audit_m17676/ (SHA256SUMS)"
 },
 "entry": "docs/audit_m17676/xval_zhou1999.py",
 "entry_detail": "argparse CLI: --list / --run <case> / --run-all / --aggregate; main() runs SandpileRun -> measure() -> checks -> verdict.json + doc + figures",
 "run_artifacts": [
  "docs/audit_m17676/artifacts/verdict.json",
  "docs/audit_m17676/artifacts/cases/gentle_anchor.json",
  "docs/audit_m17676/artifacts/cases/ctrl_mu_r_zero.json",
  "docs/audit_m17676/artifacts/cases/ctrl_mu_s_zero.json",
  "docs/audit_m17676/artifacts/cases/sens_N1000.json",
  "docs/audit_m17676/artifacts/cases/sens_N4000.json",
  "docs/audit_m17676/artifacts/cases/sens_settle_0p6.json",
  "docs/audit_m17676/artifacts/cases/sens_dt_half.json",
  "docs/audit_m17676/artifacts/cases/as_delivered.json",
  "docs/audit_m17676/artifacts/figures/fig1_profiles.png",
  "docs/audit_m17676/artifacts/figures/fig2_theta_vs_band.png",
  "docs/audit_m17676/artifacts/figures/fig3_sensitivity.png",
  "docs/audit_m17676/artifacts/cross_route.json"
 ],
 "tests": [
  "tests/test_sandpile_degenerate_measure_m17676.py::test_flat_deposit_keeps_the_angle_and_reports_packing_gap -> RED (TypeError: argument of type 'float' is not iterable @ line 62; the test asserts packing['packing_fraction'] is a dict, but probe_packing_fraction returns a float) -> shipped test can never pass; defect is in the test, not the patch",
  "audit probe (m17745, degenerate N=400 mu_r_zero deposit): run.measure() returns repose_deg=13.3802679906812 and packing_fraction=0.0 (float) with bulk/core/voronoi as dicts -> the shipped _safe() graceful-degradation patch is REAL and effective",
  "control re-runs: ctrl_mu_r_zero theta=4.99e-07 deg, ctrl_mu_s_zero theta=3.12e-17 deg, poured=2000 each (regenerated 08:16, patch landed 08:12:58); run_all.log rc=1 for both controls is PRE-patch (08:11) and stale"
 ],
 "integration_callers": [
  "SandpileRun",
  "scripts/run_dem_macro_anchors.py:85 run = SandpileRun(cfg) (production macro-anchor entry, pre-existing)",
  "src/tensordem/protocols/sandpile.py:1250 SandpileRun.measure() = patch site shared by every caller",
  "benchmarks/xval_sandpile_zhou1999/xval_zhou1999.py:246 run = SandpileRun(cfg)"
 ],
 "benchmark": {
  "case": "gentle_anchor (N=2000, mu_s=0.50, mu_r=0.05, k_n=1500 N/m, pour 500/s, settle 2.0 s)",
  "metric": "angle of repose theta_deg",
  "ref_source": "Zhou et al. 1999 Physica A 269:536-553 sphere band [22,25] deg, parsed at run time from docs/GAP_DEM_CLUMP_ROUND2.md (sha256 47675131...072b, re-verified by auditor); citation line parsed from src/tensordem/protocols/sandpile.py:80-82 with re.S",
  "ref_band_deg": [
   22.0,
   25.0
  ],
  "ref_centre_deg": 23.5,
  "got_deg": 19.0527,
  "rel_err_pct": 18.925,
  "tolerance": "<15% (owner standard / predeclared TOL_REL=0.15)",
  "pass": false,
  "other_checks": {
   "D2_in_band": false,
   "D3_rolling_mechanism": true,
   "D4_mu_s_insensitivity": false,
   "D5_spread_le_3deg": false,
   "D6_fit_quality": true
  },
  "all_checks_pass": false,
  "judge_registered_in_tensor_benchmark_cases": false,
  "judge_query": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-DEM-SANDPILE-REPOSE-CALIBRATION-ANCHOR' -> 0 rows; nearest DEM repose row is id=1 GAP-DEM-CLUMP tolerance +/-20% (would flip this verdict)"
 },
 "repro_cmd": "cd <TensorDEM checkout> && python benchmarks/xval_sandpile_zhou1999/xval_zhou1999.py --aggregate  # exit=0, regenerates verdict.json sha256 d541d7af18aaeea1503e4d0cadbfb4df8ec7c6d0a4ac50857521389a2dc3c4ac (byte-identical to shipped); full rerun: --run-all --jobs 8 (~2.5h CPU); red test: python -m pytest tests/test_sandpile_degenerate_measure_m17676.py -x -q",
 "external_anchor": [
  "docs/audit_m17676/XVAL_SANDPILE_ZHOU1999.md",
  "docs/GAP_DEM_CLUMP_ROUND2.md"
 ],
 "external_anchor_status": "WITHHELD at source level: Zhou-1999 original text/algorithm not obtainable in this run (knowledge engine offline, oss_solutions unreadable); only band-level + mechanism-level comparison is legitimate",
 "claimed_level": 3,
 "level_rationale": "L2 met: real code input->compute->output (xval harness + cross_route + sandpile measure patch, 0 placeholders) with 8 executed case artifacts. L3 met: patch lands in the production protocol src/tensordem/protocols/sandpile.py:1250 measure(), instantiated by scripts/run_dem_macro_anchors.py:85 and the benchmark driver. L4 NOT met: full case + error analysis + reproducible YES, but the benchmark FAILS (18.925% > 15%, D1/D2/D4/D5 fail) and no tensor_benchmark_cases row exists to machine-judge it -> validation not closed. L5 impossible (external anchor withheld). Level = functional substance = 3. Consistent with precedent m15187 (GAP-DEM-CLUMP, repose +/-20% failed) = L3.",
 "findings": [
  "F1 blocking-for-L4: anchor never registered in tensor_benchmark_cases (0 rows) -> mission succeeded via TP_AUTO_APPROVE_DEV_ACCEPTANCE, not via the benchmark judge",
  "F2 high: honest FAIL 18.925% > 15% (D1), plus D2/D4/D5 fail",
  "F3 high: shipped regression test is structurally wrong and always red (TypeError at line 62)",
  "F4 medium: all deliverables uncommitted (head_commit=null; ECO main lacks benchmark dir / doc / _safe patch) -> snapshot committed under docs/audit_m17676/",
  "F5 medium: no tensor_reports row for mission_id=17676 although report_owed=f",
  "F6 low: as_delivered labelled canonical baseline but is a spray (Z=2.08, theta=9.51 deg) - self-disclosed in report",
  "F7 low: route B (lifted cylinder) 8.08 deg vs route A 19.05 deg, r2=0.63; documented node 12.54 deg not reproducible (35.5% off) - self-disclosed, B declared non-anchor",
  "F8 low: verdict.json cases lack the 'valid' flag promised by the script docstring"
 ],
 "verified_positives": [
  "reference values are genuinely parsed at run time, not hand-copied: source sha256 47675131...072b re-verified; regex yields [22,25] and [35,45]; no literal band numbers in the benchmark script",
  "--aggregate reproduces verdict.json byte-identically (sha256 d541d7af...) in an independent /tmp copy",
  "the measure() _safe() patch is real and effective (audit probe: degenerate deposit keeps theta, packing reports 0.0/nan instead of raising)",
  "honest-FAIL accounting is correct: decision.xval_within_15pct=false, report section 8 states 'not <15%' without cosmetics",
  "D6 fit quality of the anchor case: r2=0.9938, stderr=0.4659 deg, Z=3.856"
 ],
 "audit_artifacts": [
  "docs/audit_m17676/AUDIT_m17676.md",
  "docs/audit_m17676/make_figures.py",
  "docs/audit_m17676/SHA256SUMS",
  "docs/audit_m17676/figures/fig1_theta_vs_anchor.png",
  "docs/audit_m17676/figures/fig2_profile_fit.png",
  "docs/audit_m17676/figures/fig3_error_vs_tolerance.png"
 ],
 "checks_not_run": [
  "full 8-case time integration rerun (~2.5h CPU) - audited aggregate determinism + control runs + targeted probe instead",
  "cross-architecture (loong64) rerun",
  "non-spherical clump branch and mu_r sweep (priors are fit-locked by design)",
  "original paper table comparison (source withheld)"
 ],
 "disposition": "request changes (remediation list, no downgrade-reopen: L3 confirmed by mechanism); must-fix order: register benchmark case row -> fix line 62 test -> merge deliverables -> post mission report -> new pre-registered experiment to enter the 15% band"
}$js$::jsonb, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id=EXCLUDED.mission_id, level=EXCLUDED.level, evidence=EXCLUDED.evidence,
      gate_state='pending', gate_level=NULL, gate_note=NULL
RETURNING id, repo, scope, level, gate_state;
